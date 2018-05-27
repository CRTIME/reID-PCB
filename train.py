import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from utils import log
from utils import save_model
from utils import do_cprofile
from config import transform
from data import Market1501
from utils import get_time
from net import Net
from net import MyCrossEntropyLoss

def get_net(args, net):
    return net.module if args.use_gpu else net

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def base_train(args, net, criterion, trainloader, train_sampler,
               optimizer_40, optimizer_60):
    LOG_FREQ = 20
    for epoch in range(args.epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        optimizer = optimizer_40 if (optimizer_60 is None or epoch < 40) else optimizer_60
        epoch_loss = .0
        for i, data in enumerate(trainloader):
            net.train()
            inputs, labels = Variable(data[0]), Variable(data[1])
            if args.use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            if args.distributed and not args.use_gpu:
                average_gradients(net)

            optimizer.step()

            epoch_loss += loss.data[0]
            if i % (LOG_FREQ + 1) == LOG_FREQ:
                log('[%s] [Epoch] %2d [Iter] %3d [Loss] %.10f' %
                    (args.process_name, epoch, i, epoch_loss / (LOG_FREQ + 1)))
                epoch_loss = .0
    return net

def standard_pcb_train(args, net, criterion, trainloader, train_sampler):
    optimizer_40 = optim.SGD([
        { 'params': get_net(args, net).resnet.parameters(), 'lr': 0.01 },
        { 'params': get_net(args, net).convs.parameters() },
        { 'params': get_net(args, net).fcs.parameters() }
    ], lr=0.1, momentum=0.9, weight_decay=0.0005)
    optimizer_60 = optim.SGD([
        { 'params': get_net(args, net).resnet.parameters(), 'lr': 0.001 },
        { 'params': get_net(args, net).convs.parameters() },
        { 'params': get_net(args, net).fcs.parameters() }
    ], lr=0.01, momentum=0.9, weight_decay=0.0005)
    args.process_name = 'standard_pcb_train'
    net = base_train(args, net, criterion, trainloader, train_sampler,
                     optimizer_40, optimizer_60)
    return net

def refined_pcb_train(args, net, criterion, trainloader, train_sampler):
    net = get_net(args, net).convert_to_rpp()
    if args.use_gpu:
        net = net.cpu().cuda()
        if args.distributed:
            net = DistributedDataParallel(net)
        else:
            net = DataParallel(net)
    args.epoch = 10
    optimizer = optim.SGD(get_net(args, net).pool.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=0.0005)
    args.process_name = 'refined_pcb_train'
    net = base_train(args, net, criterion, trainloader, train_sampler,
                     optimizer, None)
    return net

def overall_fine_tune_train(args, net, criterion, trainloader, train_sampler):
    args.epoch = 10
    optimizer = optim.SGD([
        { 'params': get_net(args, net).resnet.parameters(), 'lr': 0.001 },
        { 'params': get_net(args, net).convs.parameters() },
        { 'params': get_net(args, net).fcs.parameters() },
        { 'params': get_net(args, net).pool.parameters() }
    ], lr=0.01, momentum=0.9, weight_decay=0.0005)
    args.process_name = 'overall_fine_tune_train'
    net = base_train(args, net, criterion, trainloader, train_sampler,
                     optimizer, None)
    return net

@do_cprofile('train.prof')
def train(args):

    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.dist_rank
        )

    log('[START] Loading Training Data')
    trainset = Market1501(root=args.dataset, data_type='train',
                          transform=transform, once=args.load_once)
    if args.distributed:
        train_sampler = DistributedSampler(trainset)
    else:
        train_sampler = None
    trainloader = DataLoader(trainset,
                             batch_size=args.batch_size,
                             shuffle=(train_sampler is None),
                             num_workers=args.num_workers,
                             pin_memory=True,
                             sampler=train_sampler)
    log('[ END ] Loading Training Data')

    log('[START] Building Net')
    net = Net(rpp_std=args.rpp_std, conv_std=args.conv_std)
    criterion = MyCrossEntropyLoss()
    if args.use_gpu:
        net = net.cuda()
        criterion = criterion.cuda()
        if args.distributed:
            net = DistributedDataParallel(net)
        else:
            net = DataParallel(net)
    log('[ END ] Building Net')

    log('[START] Training')
    net = standard_pcb_train(args, net, criterion, trainloader, train_sampler)
    if (not args.distributed) or args.dist_rank == 0:
        save_model(net, '%s.checkpoint_pcb' % args.model_file)

    net = refined_pcb_train(args, net, criterion, trainloader, train_sampler)
    if (not args.distributed) or args.dist_rank == 0:
        save_model(net, '%s.checkpoint_rpp' % args.model_file)

    net = overall_fine_tune_train(args, net, criterion,
                                  trainloader, train_sampler)
    if (not args.distributed) or args.dist_rank == 0:
        save_model(net, '%s.checkpoint_fnl' % args.model_file)
    log('[ END ] Training')