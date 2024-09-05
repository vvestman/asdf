# 2024 Ville Vestman



import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CnnLayer2d(nn.Module):
    def __init__(self, D_in: int, D_out: int, filter_reach: int, dilation:int):
        super().__init__()
        self.bn = nn.BatchNorm2d(D_out)
        self.activation = F.selu
        self.cnn = torch.nn.Conv2d(D_in, D_out, (1, filter_reach*2+1), padding=(0, filter_reach), dilation=(1, dilation))

    def forward(self, x):
        x = self.cnn(x)
        return self.activation(self.bn(x))


class MeanMaxMinStdPoolingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        m = torch.mean(x, dim=3)
        maximum, _ = torch.max(x, dim=3)
        minimum, _ = torch.min(x, dim=3)
        sigma = torch.sqrt(torch.clamp(torch.mean(x ** 2, dim=3) - m ** 2, min=1e-6))
        return torch.cat((m, maximum, minimum, sigma), 1)




class CnnLayer(nn.Module):
    def __init__(self, D_in: int, D_out: int, filter_reach: int, dilation:int):
        super().__init__()
        self.bn = nn.BatchNorm1d(D_out)
        self.activation = F.selu
        self.cnn = torch.nn.Conv1d(D_in, D_out, filter_reach*2+1, padding=filter_reach, dilation=dilation)

    def forward(self, x):
        x = self.cnn(x)
        return self.activation(self.bn(x))


