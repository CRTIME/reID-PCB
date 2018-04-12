import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from utils import log
from config import transform
from data import Market1501
from utils import get_time
from net import Net
from net import MyCrossEntropyLoss

def get_net(args, net):
    if args.use_gpu:
        return net.module
    return net

def base_train(args, net, criterion, trainloader, train_sampler, optimizer_40, optimizer_60):
    for epoch in range(args.epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        optimizer = optimizer_40
        if epoch >= 40:
            optimizer = optimizer_60

        epoch_loss = .0
        for i, data in enumerate(trainloader):
            net.train()
            inputs, labels, _, _ = data
            if args.use_gpu:
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data[0]
            if i % 20 == 19:
                log('[%s] [Epoch] %2d [Iter] %3d [Loss] %.10f' % (args.process_name, epoch, i, epoch_loss / 20))
                epoch_loss = .0

def standard_pcb_train(args, net, criterion, trainloader, train_sampler):
    optimizer_40 = optim.SGD([
        { 'params': get_net(args, net).resnet.parameters(), 'lr': 0.01 },
        { 'params': get_net(args, net).avgpool.parameters() },
        { 'params': get_net(args, net).conv1.parameters() },
        { 'params': get_net(args, net).fcs.parameters() }
    ], lr=0.1, momentum=0.9, weight_decay=0.0005)
    optimizer_60 = optim.SGD([
        { 'params': get_net(args, net).resnet.parameters(), 'lr': 0.001 },
        { 'params': get_net(args, net).avgpool.parameters() },
        { 'params': get_net(args, net).conv1.parameters() },
        { 'params': get_net(args, net).fcs.parameters() }
    ], lr=0.01, momentum=0.9, weight_decay=0.0005)
    args.process_name = 'standard_pcb_train'
    base_train(args, net, criterion, trainloader, train_sampler, optimizer_40, optimizer_60)

def refined_pcb_train(args, net, criterion, trainloader, train_sampler):
    optimizer_40 = optim.SGD(get_net(args, net).Ws.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    optimizer_60 = optim.SGD(get_net(args, net).Ws.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    args.process_name = 'refined_pcb_train'
    base_train(args, net, criterion, trainloader, train_sampler, optimizer_40, optimizer_60)

def overall_fine_tune_train(args, net, criterion, trainloader, train_sampler):
    optimizer_40 = optim.SGD(get_net(args, net).parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer_60 = optim.SGD(get_net(args, net).parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    args.process_name = 'overall_fine_tune_train'
    base_train(args, net, criterion, trainloader, train_sampler, optimizer_40, optimizer_60)

def train(args):

    if args.distributed:
        dist.init_process_group(backend='gloo', init_method=args.dist_url,
            world_size=args.world_size, rank=args.dist_rank)

    log('[START] Loading Training Data')
    trainset = Market1501(root=args.dataset, data_type='train', transform=transform, once=args.load_once)
    if args.distributed:
        train_sampler = DistributedSampler(trainset)
    else:
        train_sampler = None
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    log('[ END ] Loading Training Data')

    log('[START] Build Net')
    net = Net()
    criterion = MyCrossEntropyLoss(args)
    if args.use_gpu:
        net = net.cuda()
        criterion = criterion.cuda()
        if args.distributed:
            net = DistributedDataParallel(net)
        else:
            net = DataParallel(net)
    log('[ END ] Building Net')

    log('[START] Training')
    standard_pcb_train(args, net, criterion, trainloader, train_sampler)
    refined_pcb_train(args, net, criterion, trainloader, train_sampler)
    overall_fine_tune_train(args, net, criterion, trainloader, train_sampler)
    log('[ END ] Training')

    log('[START] Saving Model')
    torch.save(get_net(args, net).cpu().state_dict(), args.model_file)
    log('[ END ] Saving Model')