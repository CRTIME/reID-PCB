import os
import torch
import torch.optim as optim
from torch.autograd import Variable

from config import conf
from config import transform
from data import Market1501
from utils import get_time
from net import Net
from net import MyCrossEntropyLoss

def base_train(procedure_name, net, criterion, trainloader, optimizer_40, optimizer_60, gpu=True):
    for epoch in range(60):
        optimizer = optimizer_40
        if epoch >= 40:
            optimizer = optimizer_60
        
        running_loss = .0
        for i, data in enumerate(trainloader):
            inputs, labels, _ = data
            if gpu:
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = net.forward(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]
            if i % 20 == 19:
                print('%s [%s] [Epoch] %2d [Iter] %3d [Loss] %.10f' % (get_time(), procedure_name, epoch, i, running_loss / 20))
                running_loss = .0

def standard_pcb_train(net, criterion, trainloader, gpu=True):
    optimizer_40 = optim.SGD([
        { 'params': net.resnet.parameters(), 'lr': 0.01 },
        { 'params': net.avgpool.parameters() },
        { 'params': net.conv1.parameters() },
        { 'params': net.fcs.parameters() }
    ], lr=0.1)
    optimizer_60 = optim.SGD([
        { 'params': net.resnet.parameters(), 'lr': 0.001 },
        { 'params': net.avgpool.parameters() },
        { 'params': net.conv1.parameters() },
        { 'params': net.fcs.parameters() }
    ], lr=0.01)
    base_train('standard_pcb_train', net, criterion, trainloader, optimizer_40, optimizer_60)

def refined_pcb_train(net, criterion, trainloader, gpu=True):
    optimizer_40 = optim.SGD([
        { 'params': net.Ws.parameters() }
    ], lr=0.1)
    optimizer_60 = optim.SGD([
        { 'params': net.Ws.parameters() }
    ], lr=0.01)
    base_train('refined_pcb_train', net, criterion, trainloader, optimizer_40, optimizer_60)
    
def overall_fine_tune_train(net, criterion, trainloader, gpu=True):
    optimizer_40 = optim.SGD(net.parameters(), lr=0.1)
    optimizer_60 = optim.SGD(net.parameters(), lr=0.01)
    base_train('overall_fine_tune_train', net, criterion, trainloader, optimizer_40, optimizer_60)

def train():
    print('%s [START] Loading Training Data' % get_time())
    torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
    trainset = Market1501(root=os.path.join(torch_home, 'datasets'), data_type='train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64)
    print('%s [ END ] Loading Training Data' % get_time())

    print('%s [START] Build Net' % get_time())
    net = Net(trainset.train_size)
    if conf['use_gpu']:
        net.cuda()
    print('%s [ END ] Building Net' % get_time())

    print('%s [START] Building Criterion' % get_time())
    criterion = MyCrossEntropyLoss()
    if conf['use_gpu']:
        criterion.cuda()
    print('%s [ END ] Building Criterion' % get_time())

    print('%s [START] Training' % get_time())
    standard_pcb_train(net, criterion, trainloader, gpu=conf['use_gpu'])
    refined_pcb_train(net, criterion, trainloader, gpu=conf['use_gpu'])
    overall_fine_tune_train(net, criterion, trainloader, gpu=conf['use_gpu'])
    print('%s [ END ] Training' % get_time())

    print('%s [START] Saving Model' % get_time())
    torch.save(net.cpu().state_dict(), os.path.join(torch_home, 'models', conf['model_name']))
    print('%s [ END ] Saving Model' % get_time())