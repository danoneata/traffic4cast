from typing import List, Union
import numpy as np

import torch
import torch.nn as torch_nn

import src


class Temporal(torch_nn.Module):
    """ Temporal model for a single single channel. Predicts the next value by
    considering a fixed history at the same coordinate. """

    def __init__(self, past: int, future: int, channels: List[str],
                 module: Union[torch_nn.Module, torch_nn.Sequential]):
        super(Temporal, self).__init__()
        self.past = past
        self.future = future
        self.num_channels = len(channels)
        self.channels = channels
        self.add_module('module', module)

    def forward(self, input):
        return self.module(input)

    def predict(self, frames: List[int],
                sample: src.dataset.Traffic4CastSample):
        predictions = {f: None for f in frames}
        for frame, (slice, valid) in zip(
                frames, sample.temporal_slices(self.past, frames, True)):

            def fill_with_predicted():
                for f in range(valid.shape[0]):
                    if not valid[f]:
                        replaced = False
                        for pred_f in range(f, -self.future, f - self.future):
                            if predictions[pred_f] is not None:
                                i = (f - pred_f) * self.num_channels
                                j = i + self.num_channels
                                slice[f] = predictions[pred_f][i:j]
                                break
                        if not replaced:
                            raise Exeception(f"Invalid frames for slice {f}.")

            if not valid.all():
                fill_with_predicted()

            predictions[frame] = self(slice.unsqueeze()).squeeze(0)

        return predictions

    def ignite_batch(self, batch, device, non_blocking):
        batch = batch.to(device)
        batch = batch.reshape(
            (batch.shape[0], -1, batch.shape[3], batch.shape[4]))
        result = (batch[:, :self.past * self.num_channels],
                  batch[:, self.past * self.num_channels:])
        return result

    def ignite_random(self, loader, num_minibatches, minibatch_size):
        for batch in loader:
            for sample in batch:
                for minibatch in sample.random_temporal_batches(
                        num_minibatches, minibatch_size,
                        self.past + self.future):
                    yield minibatch

    def ignite_all(self, loader, minibatch_size):
        for batch in loader:
            for sample in batch:
                num_frames = sample.data.shape[0]
                for minibatch in sample.selected_temporal_batches(
                        minibatch_size, self.past + self.future,
                        range(self.past + self.future,
                              num_frames - self.past - self.future)):
                    yield minibatch


class TemporalRegression(torch_nn.Module):
    """ Simple 1 channel model. """

    def __init__(self, history: int):
        super(TemporalRegression, self).__init__()
        self.history = history
        kwargs = dict(kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = torch_nn.Conv2d(history, 16, **kwargs)
        self.conv2 = torch_nn.Conv2d(16, 16, **kwargs)
        self.conv3 = torch_nn.Conv2d(16, 1, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)
        return x
