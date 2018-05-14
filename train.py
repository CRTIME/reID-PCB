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
    return net.module if args.gpu else net

def base_train(args, net, criterion, trainloader, train_sampler,
               optimizer, lr_scheduler):
    for epoch in range(args.epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        net.train(True)
        lr_scheduler.step()
        epoch_loss = .0
        for i, data in enumerate(trainloader):
            inputs, labels = Variable(data[0]), Variable(data[1])
            if args.gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data[0]
            if i % 21 == 20:
                log('[%s] [Epoch] %2d [Iter] %3d [Loss] %.10f' %
                    (args.process_name, epoch, i, epoch_loss / 21))
                epoch_loss = .0
    return net

def standard_pcb_train(args, net, criterion, trainloader, train_sampler):
    args.process_name = 'standard_pcb_train'
    params = [
        { 'params': get_net(args, net).resnet.parameters(), 'lr': 0.01 },
        { 'params': get_net(args, net).fcs.parameters() }
    ]
    if args.last_conv:
        params += [{ 'params': get_net(args, net).convs.parameters() }]
    optimizer = optim.SGD(params, lr=0.1, momentum=0.9,
                          weight_decay=0.0005, nesterov=True)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer, step_size=30, gamma=0.1)
    net = base_train(args, net, criterion, trainloader, train_sampler,
                     optimizer, exp_lr_scheduler)
    return net

def refined_pcb_train(args, net, criterion, trainloader, train_sampler):
    args.process_name = 'refined_pcb_train'
    args.epoch = 10
    net = get_net(args, net).convert_to_rpp()
    if args.gpu:
        net = net.cpu().cuda()
        if args.distributed:
            net = DistributedDataParallel(net)
        else:
            net = DataParallel(net)
    optimizer = optim.SGD(get_net(args, net).pool.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0005, nesterov=True)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer, step_size=30, gamma=0.1)
    net = base_train(args, net, criterion, trainloader, train_sampler,
                     optimizer, exp_lr_scheduler)
    return net

def overall_fine_tune_train(args, net, criterion, trainloader, train_sampler):
    args.process_name = 'overall_fine_tune_train'
    args.epoch = 10
    params = [
        { 'params': get_net(args, net).resnet.parameters(), 'lr': 0.001 },
        { 'params': get_net(args, net).fcs.parameters() },
        { 'params': get_net(args, net).pool.parameters() }
    ]
    if args.last_conv:
        params += [{ 'params': get_net(args, net).convs.parameters() }]
    optimizer = optim.SGD(params, lr=0.01, momentum=0.9,
                          weight_decay=0.0005, nesterov=True)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer, step_size=30, gamma=0.1)
    net = base_train(args, net, criterion, trainloader, train_sampler,
                     optimizer, exp_lr_scheduler)
    return net

@do_cprofile('train.prof')
def train(args):

    if args.distributed:
        dist.init_process_group(backend='gloo', init_method=args.dist_url,
            world_size=args.world_size, rank=args.dist_rank)

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
    net = Net(last_conv=args.last_conv, normalize=args.normalize,
              rpp_std=args.rpp_std, conv_std=args.conv_std)
    criterion = MyCrossEntropyLoss()
    if args.gpu:
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