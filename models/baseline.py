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
    """Temporal model for a single single channel. Predicts the next value by
    considering a fixed history at the same coordinate. """

    def __init__(self, channel: str, history: int):
        super(TemporalRegression, self).__init__()
        self.channel = channel
        self.history = history
        kwargs = dict(kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(history, 16, **kwargs)
        self.conv2 = nn.Conv2d(16, 16, **kwargs)
        self.conv3 = nn.Conv2d(16, 1, **kwargs)

    def forward(self, x):
        # Brute standardization
        x = x / 255 - 0.5
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        return 255 * x  # Rescale result to predict in [0, 255]

    def predict(self, sample, frame):
        N_CHANNELS = 3
        AXES1 = 0, 3, 1, 2
        AXES2 = 1, 2, 3, 0

        ch = Traffic4CastSample.channel_to_index[self.channel.capitalize()]

        x = sample.data.permute(AXES1)  # T × C × H × W
        x = x[frame - self.history - 1:frame - 1, ch]
        x = x.unsqueeze(0).float()  # 1 × T × H × W

        def update_history(xs, x):
            return torch.cat((xs[:, 1:], x), 1)

        with torch.no_grad():
            preds = []
            for i in range(N_FRAMES):
                pred = self.forward(x)
                preds.append(pred)
                x = update_history(x, pred)

        preds = torch.cat(preds, 1)  # 1 × T × H × W
        preds = preds.permute(AXES2)  # T × H × W × 1

        # Predict zeros for the rest of the channels
        res = torch.zeros(N_FRAMES, preds.shape[1], preds.shape[2], N_CHANNELS)
        res[:, :, :, ch] = preds.squeeze()

        return res
