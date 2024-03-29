import collections
import math
import pdb

from itertools import repeat

import torch
import torch.nn.functional as F

from torch import nn


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def conv2d_local(input, weight, bias=None, padding=0, stride=1, dilation=1):
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))

    outH, outW, outC, inC, kH, kW = weight.size()
    kernel_size = (kH, kW)

    # N x [inC * kH * kW] x [outH * outW]
    cols = F.unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
    cols = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)

    w = weight.view(outH * outW, outC, inC * kH * kW).permute(0, 2, 1)
    out = torch.einsum("bpoi, pif-> bpof", cols, w)
    # out = torch.matmul(cols, w)
    out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)

    if bias is not None:
        out = out + bias.expand_as(out)

    return out


class Conv2dLocal(nn.Module):

    def __init__(self, in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2dLocal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.weight = nn.parameter.Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input):
        return conv2d_local(
            input, self.weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation)


class DenseBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate=0.0):
        super(DenseBasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(x))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, drop_rate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
