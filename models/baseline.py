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

