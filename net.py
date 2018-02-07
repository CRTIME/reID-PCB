import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import resnet50

class Net(nn.Module):
    def __init__(self, out_size=1501):
        super(Net, self).__init__()

        resnet = resnet50(pretrained=True)
        backbone_model = nn.Sequential(*list(resnet.children())[:-2])
        
        self.resnet = backbone_model
        self.resnet = self.resnet
        
        init_val = [torch.zeros(24, 8) for _ in range(6)]
        for i in range(6):
            init_val[i][4*i:4*(i+1),:] = 1.0
        self.Ws = nn.ParameterList([Parameter(init_val[i]) for i in range(6)])

        self.avgpool = nn.AvgPool2d((24, 8))
        self.conv1 = nn.Conv2d(2048, 256, 1)
        self.fcs = nn.ModuleList([nn.Linear(256, out_size) for _ in range(6)])
        
    def forward(self, x):
        x = self.resnet.forward(x)
        xs = [None for _ in range(6)]
        for i in range(6):
            x_i = torch.mul(x, self.Ws[i])
            x_i = self.avgpool(x_i)
            x_i = self.conv1(x_i)
            x_i = x_i.view(-1, 256)
            x_i = self.fcs[i](x_i)
            xs[i] = x_i
        return xs

class FeatureExtractor(Net):
    def __init__(self, state_path, last_conv=True):
        super(FeatureExtractor, self).__init__()
        self.last_conv = last_conv
        self.load_state_dict(torch.load(state_path))

    def forward(self, x):
        x = self.resnet.forward(x)
        xs = [None for _ in range(6)]
        for i in range(6):
            x_i = torch.mul(x, self.Ws[i])
            x_i = self.avgpool(x_i)
            if self.last_conv:
                x_i = self.conv1(x_i)
                x_i = x_i.view(-1, 256)
            else:
                x_i = x_i.view(-1, 2048)
            xs[i] = x_i
        x = torch.cat(xs, 1)
        return x

class MyCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, args):
        super(MyCrossEntropyLoss, self).__init__()
        self.use_gpu = args.use_gpu
    def forward(self, inputs, target):
        ret = Variable(torch.FloatTensor([0])).cuda()        
        if not self.use_gpu:
            ret = Variable(torch.FloatTensor([0]))
        for ipt in inputs:
            ret += F.cross_entropy(ipt, target, self.weight, self.size_average,
                                   self.ignore_index, self.reduce)
        return ret