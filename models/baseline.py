import pdb

import numpy as np

import torch
import torch.nn as nn

from src.dataset import Traffic4CastSample


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

    def __init__(self, channel: str, history: int):
        super(TemporalRegression, self).__init__()
        self.channel = channel
        self.history = history
        kwargs = dict(kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(history, 16, **kwargs)
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

    def predict(self, sample, frame):
        ch = Traffic4CastSample.channel_to_index[self.channel.capitalize()]
        axes = 0, 3, 1, 2
        x = sample.data.permute(axes)
        x = x[:, ch]
        x = x[frame - self.history - 1: frame - 1]
        x = x.unsqueeze(0).float()
        with torch.no_grad():
            preds = []
            for i in range(N_FRAMES):
                pred = self.forward(x)
                preds.append(pred)
                x = torch.cat((x[:, 1:], pred), dim=1)
        preds = torch.cat(preds, 1).permute([1, 2, 3, 0])
        _, H, W, _ = preds.shape
        res = torch.zeros(N_FRAMES, H, W, 3)
        res[:, :, :, ch] = preds.squeeze()
        return res
