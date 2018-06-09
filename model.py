import torch
import torch.nn as nn
import torchvision
import pdb
from torch.nn import init
from functools import reduce

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.features = []
        self.features.append(nn.Sequential(           # 前馈神经网络
            # conv1
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(1, stride=1, ceil_mode=True),  # 1/8
            # conv4
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(1, stride=1, ceil_mode=True),  # 1/16
            # conv5 features
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(),
        ))
        self.features = nn.ModuleList(self.features)
        vgg16 = torchvision.models.vgg16(pretrained=True)
        L_vgg16 = list(vgg16.features)
        L_self = reduce(lambda x, y: list(x)+list(y), self.features)
        # 定义个函数
        L_self[0].weight.data[:, :-1] = L_vgg16[0].weight.data
        for l1, l2 in zip(L_vgg16[1:], L_self[1:]):
            if (isinstance(l1, nn.Conv2d) and
                    isinstance(l2, nn.Conv2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            if (isinstance(l1, nn.BatchNorm2d) and
                    isinstance(l2, nn.BatchNorm2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

    def cuda(self):
        for f in self.features:
            f.cuda()

    def forward(self, x):
        feats = []
        for f in self.features:
            x = f(x)
            feats.append(x)
        return feats  # the biggest the first


class Feature_FCN(nn.Module):
    def __init__(self):
        super(Feature_FCN, self).__init__()
        self.features = []
        self.features.append(nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(1, stride=1, ceil_mode=True),  # 1/8
            # conv4
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
        ))
        self.features.append(nn.Sequential(
            nn.MaxPool2d(1, stride=1, ceil_mode=True),  # 1/16
            # conv5 features
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(),
        ))
        self.features = nn.ModuleList(self.features)
        vgg16 = torchvision.models.vgg16(pretrained=True)
        L_vgg16 = list(vgg16.features)
        L_self = reduce(lambda x,y: list(x)+list(y), self.features)
        for l1, l2 in zip(L_vgg16, L_self):
            if (isinstance(l1, nn.Conv2d) and
                    isinstance(l2, nn.Conv2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            if (isinstance(l1, nn.BatchNorm2d) and
                    isinstance(l2, nn.BatchNorm2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

    def cuda(self):
        for f in self.features:
            f.cuda()

    def forward(self, x):
        feats = []
        for f in self.features:
            x = f(x)
            feats.append(x)
        return feats  # the biggest the first


class Deconv(nn.Module):
    def __init__(self):
        super(Deconv, self).__init__()
        _reduce_dimension = [
            nn.Conv2d(512, 32, 1),
            nn.Conv2d(512, 32, 1),
            nn.Conv2d(256, 32, 1)
        ]
        _prediction = [
            nn.Conv2d(32, 1, 1),
            nn.Conv2d(33, 1, 1),
            nn.Conv2d(33, 1, 1),
        ]
        self.reduce_dimension = nn.ModuleList(_reduce_dimension)
        self.prediction = nn.ModuleList(_prediction)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, feats):
        for i in range(len(feats)):
            feats[i] = self.reduce_dimension[i](feats[i])
        y = self.prediction[0](feats[0])
        for i  in range(1, len(feats)):
            y = self.prediction[i](
                torch.cat((feats[i], y), 1)
            )
        return y
