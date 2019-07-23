import pdb

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


class TemporalRegression(nn.Module):

    def __init__(self):
        super(TemporalRegression, self).__init__()
        kwargs = dict(kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(12, 16, **kwargs)
        self.conv2 = nn.Conv2d(16, 16, **kwargs)
        self.conv3 = nn.Conv2d(16, 1, **kwargs)

    def forward(self, x):
        x = x / 255 - 0.5
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        return 255 * x
