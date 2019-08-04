import collections
import pdb

import numpy as np


class Zeros:

    def __init__(self):
        self.num_channels = 3

    def predict(self, frames, sample):
        _, C, H, W = sample.data.shape
        return {f: np.zeros((C, H, W)) for f in frames}


class Naive:

    def __init__(self):
        self.num_channels = 3

    def predict(self, frames, sample):
        predictions = collections.OrderedDict.fromkeys(frames)
        for f in range(len(frames) // 3):
            predictions[frames[f * 3 + 0]] = sample.data[frames[f * 3] - 1]
            predictions[frames[f * 3 + 1]] = sample.data[frames[f * 3] - 1]
            predictions[frames[f * 3 + 2]] = sample.data[frames[f * 3] - 1]

        return predictions
