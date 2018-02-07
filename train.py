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

def base_train(args, net, criterion, trainloader, train_sampler, optimizer_40, optimizer_60):
    for epoch in range(args.epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        optimizer = optimizer_40
        if epoch >= 40:
            optimizer = optimizer_60
        
        epoch_loss = .0
        for i, data in enumerate(trainloader):
            inputs, labels, _ = data
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
        { 'params': net.module.resnet.parameters(), 'lr': 0.01 },
        { 'params': net.module.avgpool.parameters() },
        { 'params': net.module.conv1.parameters() },
        { 'params': net.module.fcs.parameters() }
    ], lr=0.1)
    optimizer_60 = optim.SGD([
        { 'params': net.module.resnet.parameters(), 'lr': 0.001 },
        { 'params': net.module.avgpool.parameters() },
        { 'params': net.module.conv1.parameters() },
        { 'params': net.module.fcs.parameters() }
    ], lr=0.01)
    args.epoch = 60
    args.process_name = 'standard_pcb_train'
    base_train(args, net, criterion, trainloader, train_sampler, optimizer_40, optimizer_60)

def refined_pcb_train(args, net, criterion, trainloader, train_sampler):
    optimizer_40 = optim.SGD(net.module.Ws.parameters(), lr=0.1)
    optimizer_60 = optim.SGD(net.module.Ws.parameters(), lr=0.01)
    args.epoch = 70    
    args.process_name = 'refined_pcb_train'
    base_train(args, net, criterion, trainloader, train_sampler, optimizer_40, optimizer_60)
    
def overall_fine_tune_train(args, net, criterion, trainloader, train_sampler):
    optimizer_40 = optim.SGD(net.module.parameters(), lr=0.1)
    optimizer_60 = optim.SGD(net.module.parameters(), lr=0.01)
    args.epoch = 70    
    args.process_name = 'overall_fine_tune_train'
    base_train(args, net, criterion, trainloader, train_sampler, optimizer_40, optimizer_60)

def train(args):

    if args.distributed:
        dist.init_process_group(backend='gloo', init_method=args.dist_url, 
            world_size=args.world_size, rank=args.dist_rank)

    log('[START] Loading Training Data')
    trainset = Market1501(root=args.dataset, data_type='train', transform=transform)
    if args.distributed: 
        train_sampler = DistributedSampler(trainset)
    else: 
        train_sampler = None
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    log('[ END ] Loading Training Data')

    log('[START] Build Net')
    net = Net(trainset.train_size)
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
    standard_pcb_train(args, net, criterion, trainloader, train_sampler)
    refined_pcb_train(args, net, criterion, trainloader, train_sampler)
    overall_fine_tune_train(args, net, criterion, trainloader, train_sampler)
    log('[ END ] Training')

    log('[START] Saving Model')
    torch.save(net.module.cpu().state_dict(), args.model_file)
    log('[ END ] Saving Model')