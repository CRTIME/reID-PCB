import torch
import torchvision
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import resnet50
import torch.utils.model_zoo as model_zoo

import torch.nn.init as init

class PCB(nn.Module):
    def __init__(self):
        super(PCB, self).__init__()
    def forward(self, x):
        y = []
        for i in range(6):
            y_i = x[:, :, i*6:(i+1)*6, :]
            y_i = F.adaptive_avg_pool2d(y_i, (1, 1))
            y.append(y_i)
        return y

class RPP(nn.Module):
    def __init__(self):
        super(RPP, self).__init__()
        # self.W.size(): [2048, 6]
        W = torch.zeros(2048, 6)
        self.W = nn.Parameter(W)

    def forward(self, x):
        """
            input: x.size():  [N, C, H, W]
            output: y.size(): [N, C, 1, 1] x 6
        """
        N, C, H, W = x.size()
        vectors = x.permute(0, 2, 3, 1).view(-1, C)
        masks = F.softmax(torch.mm(vectors, self.W), dim=1).view(N, H, W, 6).permute(3, 0, 1, 2)
        y = []
        for i in range(6):
            # mask.size(): N, H, W
            mask = masks[i, :, :, :]
            x_i = x.permute(1, 0, 2, 3)
            y_i = torch.mul(x_i.view(C, -1), mask.view(-1))
            y_i = y_i.view(C, N, H, W).permute(1, 0, 2, 3)
            y_i = F.adaptive_avg_pool2d(y_i, (1, 1))
            y.append(y_i)
        return y

class Net(nn.Module):
    def __init__(self, out_size=1501):
        super(Net, self).__init__()

        resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        self.pcb = PCB()
        self.rpp = RPP()
        init.normal(self.rpp.W, std=0.001)

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for _ in range(6):
            self.convs.append(nn.Sequential(
                nn.Conv2d(2048, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ))
            fc = nn.Linear(256, out_size)
            init.normal(fc.weight, std=0.001)
            init.constant(fc.bias, 0)
            self.fcs.append(fc)

        self.baseline = False

    def forward(self, x):
        x = self.resnet.forward(x)
        y = self.pcb(x) if self.baseline else self.rpp(x)
        for i, y_i in enumerate(y):
            y_i = self.convs[i](y_i)
            y_i = y_i.view(-1, 256)
            y_i = self.fcs[i](y_i)
        return y

class FeatureExtractor(Net):
    def __init__(self, state_path, last_conv=True):
        super(FeatureExtractor, self).__init__()
        self.last_conv = last_conv
        self.load_state_dict(torch.load(state_path), strict=False)

    def forward(self, x):
        x = self.resnet.forward(x)
        y = self.rpp(x)
        for i, y_i in enumerate(y):
            if self.last_conv:
                y_i = self.convs[i](y_i)
                y_i = y_i.view(-1, 256)
            else:
                y_i = y_i.view(-1, 2048)
        y = torch.cat(y, 1)
        return y

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