import torch
import torchvision
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from resnet import resnet50
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
        assert x.size(2) % self.p == 0
        h = int(x.size(2) / self.p)
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

        self.classifier = nn.Linear(vector_length, p)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        """
        Args:
            x: The feature tensors, whose size is [N, C, H, W]

        Returns:
            y: Feature vectors. [N, C, 1, 1] x p
        """
        N, C, H, W = x.size()
        vectors = x.permute(0, 2, 3, 1)
        prob = self.softmax(self.classifier(vectors))
        masks = prob.permute(3, 0, 1, 2).contiguous()
        imgs = x.permute(1, 0, 2, 3).contiguous().view(C, -1)
        y = []
        for mask in masks:
            # mask.size(): [N, H, W]
            # imgs.size(): [C, N x H x W]
            y_i = torch.mul(imgs, mask.view(-1))
            y_i = y_i.view(C, N, H, W).permute(1, 0, 2, 3).contiguous()
            y_i = F.adaptive_avg_pool2d(y_i, (1, 1))
            y.append(y_i)
        return y

class Net(nn.Module):
    """Part-based Model

    Attributes:
        p: The number of parts.
        resnet: ResNet-50, the backbone network.
        pool: Part-based convolutional baseline (PCB) layer or
              Refined part pooling (RPP) layer.
        convs: Some 1x1 convolution layers to reduces the dimension of
               column vector.
        fcs: Some full-connected layers to classification.
    """
    def __init__(self, out_size=1501, p=6, last_conv=True,
                 normalize=True, conv_std=0.001, rpp_std=0.01):
        """
        Args:
            out_size: The number of training labels.
            p: The number of parts.
            last_conv: Whether contains the last convolution layer.
            conv_std: The standard deviation of initialization of conv layer.
            rpp_std: The standard deviation of initialization of rpp layer.
            normalize: Whether normalize feature vector.
        """
        super(Net, self).__init__()
        self.p = p
        self.last_conv = last_conv
        self.conv_std = conv_std
        self.rpp_std = rpp_std
        self.normalize = normalize

        self.resnet = resnet50(pretrained=True,
                          last_conv_stride=1,
                          last_conv_dilation=1)

        self.pool = PCB(p=p)

        self.fcs = nn.ModuleList()
        if self.last_conv:
            self.convs = nn.ModuleList()
            for _ in range(p):
                self.convs.append(nn.Sequential(
                    nn.Conv2d(2048, 256, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                ))
                fc = nn.Linear(256, out_size)
                init.normal(fc.weight, std=conv_std)
                init.constant(fc.bias, 0)
                self.fcs.append(fc)
        else:
            for _ in range(p):
                fc = nn.Linear(2048, out_size)
                init.normal(fc.weight, std=conv_std)
                init.constant(fc.bias, 0)
                self.fcs.append(fc)

    def forward(self, x):
        """
        Args:
            x: Image tensors, whose size is [N, 3, 768, 256],
                where N is the batch size.

        Returns:
            y: Feature vectors. [N, C, 1, 1] x p
        """
        x = self.resnet.forward(x)
        y = self.pool(x)
        for i in range(self.p):
            if self.last_conv:
                y[i] = self.convs[i](y[i])
                y[i] = y[i].view(-1, 256)
            else:
                y[i] = y[i].view(-1, 2048)
            if self.normalize:
                y[i] = F.normalize(y[i]) * 10
            y[i] = self.fcs[i](y[i])
        return y

    def convert_to_rpp(self):
        self.pool = RPP(vector_length=2048, p=self.p)
        init.normal(self.pool.classifier.weight, std=self.rpp_std)
        init.constant(self.pool.classifier.bias, 0)
        return self

class FeatureExtractor(Net):
    """Feature extractor
    """
    def __init__(self, state_path, last_conv=True, normalize=True,
                 model_type='pcb'):
        """
        Args:
            state_path: Path to the state dict file.
            last_conv: Whether contains the last convolution layer.
            model_type: PCB or RPP.
            normalize: Whether normalize feature vector.
        """
        super(FeatureExtractor, self).__init__(last_conv=last_conv, normalize=normalize)
        self.last_conv = last_conv
        self.normalize = normalize
        if not model_type == 'pcb':
            self.convert_to_rpp()
        self.load_state_dict(torch.load(state_path), strict=True)

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
        y = self.pool(x)
        for i in range(len(y)):
            if self.last_conv:
                y[i] = self.convs[i](y[i])
                y[i] = y[i].view(-1, 256)
            else:
                y[i] = y[i].view(-1, 2048)
            if self.normalize:
                y[i] = F.normalize(y[i]) * 10
        y = torch.cat(y, 1)
        # y = F.normalize(torch.cat(y, 1))
        return y

class MyCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for multiple output.
    """
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()

    def forward(self, inputs, target):
        return torch.sum(torch.cat(
            [F.cross_entropy(ipt, target,
            self.weight, self.size_average,
            self.ignore_index, self.reduce)
            for ipt in inputs]))
