import numpy as np

import torch
import torch.nn as nn


N_FRAMES = 3


class Zeros:

    def predict(self, sample, frame):
        # TODO Are all the images the same size?
        _, H, W, C = sample.data.shape
        return np.zeros((N_FRAMES, H, W, C))


class Naive:

    def predict(self, sample, frame):
        data = sample.data[frame - 1].numpy()
        data = data[np.newaxis]
        return np.repeat(data, N_FRAMES, axis=0)


def conv3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


class AdiNet(nn.Module):

    def __init__(self):
        super(AdiNet, self).__init__()
        self.conv1 = conv3(12 * 3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(16, 16)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = conv3(16, 3 * 3)

    def forward(self, x):
        x = x / 255
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu(x)
        return 255 * x
