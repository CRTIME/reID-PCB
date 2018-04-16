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
    """Part-based Convolutional Baseline (PCB) Layer

    Average divide feature map into p (p = 6) parts.
    """
    def __init__(self, p=6):
        """
        Args:
            p: The numer of parts.
        """
        super(PCB, self).__init__()
        self.p = p

    def forward(self, x):
        assert x.size()[2] % self.p == 0
        h = int(x.size()[2] / self.p)
        y = []
        for i in range(self.p):
            y_i = x[:, :, i*h:(i+1)*h, :]
            y_i = F.adaptive_avg_pool2d(y_i, (1, 1))
            y.append(y_i)
        return y

class RPP(nn.Module):
    """Refined Part Pooling (RPP) Layer

    Relocating outliers by calculating the probability of each column vector.

    Attributes:
        W: The trainable weight matrix of the part classifier.
           Its size is [C, p], where C is the length of column vector,
           and p is the number of parts.
    """
    def __init__(self, vector_length=2048, p=6):
        """
        Args:
            vector_length: The length of a column vector (or the number of channels).
            p: The number of parts.
        """
        super(RPP, self).__init__()
        self.vector_length = vector_length
        self.p = p
        W = torch.zeros(vector_length, p)
        self.W = nn.Parameter(W)

    def forward(self, x):
        """
        Args:
            x: The feature tensors, whose size is [N, C, H, W]

        Returns:
            y: Feature vectors. [N, C, 1, 1] x p
        """
        N, C, H, W = x.size()
        vectors = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        masks = F.softmax(torch.mm(vectors, self.W), dim=1).view(N, H, W, 6).permute(3, 0, 1, 2).contiguous()
        y = []
        for i in range(6):
            # mask.size(): N, H, W
            mask = masks[i, :, :, :]
            x_i = x.permute(1, 0, 2, 3).contiguous()
            y_i = torch.mul(x_i.view(C, -1), mask.view(-1))
            y_i = y_i.view(C, N, H, W).permute(1, 0, 2, 3).contiguous()
            y_i = F.adaptive_avg_pool2d(y_i, (1, 1))
            y.append(y_i)
        return y

class Net(nn.Module):
    """Part-based Model

    Attributes:
        p: The number of parts.
        resnet: ResNet-50, the backbone network.
        pcb: Part-based convolutional baseline layer.
        rpp: Refined part pooling layer.
        convs: Some 1x1 convolution layers to reduces the dimension of column vector.
        fcs: Some full-connected layers to classification.
    """
    def __init__(self, out_size=1501, p=6):
        """
        Args:
            out_size: The number of training labels.
        """
        super(Net, self).__init__()
        self.p = p

        resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        self.pcb = PCB(p=p)
        self.rpp = RPP(vector_length=2048, p=p)
        init.normal(self.rpp.W, std=0.001)

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for _ in range(p):
            self.convs.append(nn.Sequential(
                nn.Conv2d(2048, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ))
            fc = nn.Linear(256, out_size)
            init.normal(fc.weight, std=0.001)
            init.constant(fc.bias, 0)
            self.fcs.append(fc)

        self.baseline = True

    def forward(self, x):
        """
        Args:
            x: Image tensors, whose size is [N, 3, 768, 256],
                where N is the batch size.

        Returns:
            y: Feature vectors. [N, C, 1, 1] x p
        """
        x = self.resnet.forward(x)
        y = self.pcb(x) if self.baseline else self.rpp(x)
        for i in range(self.p):
            y[i] = self.convs[i](y[i])
            y[i] = y[i].view(-1, 256)
            y[i] = self.fcs[i](y[i])
        return y

    def set_stage(self, stage):
        """Setting training stage.

        Args:
            stage: 1 if using PCB, otherwise using RPP.
        """
        self.baseline = stage == 1

class FeatureExtractor(Net):
    """Feature extractor
    """
    def __init__(self, state_path, last_conv=True):
        """
        Args:
            state_path: Path to the state dict file.
            last_conv: Whether contains the last convolution layer.
        """
        super(FeatureExtractor, self).__init__()
        self.last_conv = last_conv
        self.load_state_dict(torch.load(state_path), strict=False)

    def forward(self, x):
        """
        Args:
            x: Image tensors, whose size is [N, 3, 768, 256],
                where N is the batch size.

        Returns:
            y: Feature vector, whose size is p x 2048 if not containing
                the last convolution layer, otherwise p x 256,
                where p is the number of parts.
        """
        x = self.resnet.forward(x)
        y = self.rpp(x)
        for i in range(6):
            if self.last_conv:
                y[i] = self.convs[i](y[i])
                y[i] = y[i].view(-1, 256)
            else:
                y[i] = y[i].view(-1, 2048)
        y = torch.cat(y, 1)
        return y

class MyCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for multiple output.
    """
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