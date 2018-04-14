import torch
import torchvision
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import resnet50
import torch.utils.model_zoo as model_zoo

import torch.nn.init as init

class Net(nn.Module):
    def __init__(self, out_size=1501):
        super(Net, self).__init__()

        resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        init_val = [torch.zeros(24, 8) for _ in range(6)]
        for i in range(6):
            init_val[i][4*i:4*(i+1),:] = 1.0
        self.Ws = nn.ParameterList([Parameter(init_val[i]) for i in range(6)])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.fcs = nn.ModuleList()
        for i in range(6):
            fc = nn.Linear(256, out_size)
            init.normal(fc.weight, std=0.001)
            init.constant(fc.bias, 0)
            self.fcs.append(fc)

    def forward(self, x):
        x = self.resnet.forward(x)
        xs = [None for _ in range(6)]
        for i in range(6):
            w = F.softmax(self.Ws[i]) * 32
            x_i = torch.mul(x, w)
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
        self.load_state_dict(torch.load(state_path), strict=False)

    def forward(self, x):
        x = self.resnet.forward(x)
        xs = [None for _ in range(6)]
        for i in range(6):
            w = F.softmax(self.Ws[i]) * 32
            x_i = torch.mul(x, w)
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