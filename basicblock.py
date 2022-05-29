import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d((0, 0, 1, 1)),
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=(0, 1), padding_mode='circular', bias=False)
    )

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
