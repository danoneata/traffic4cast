from typing import List, Union, Dict
import collections

import numpy as np
import torch
import torch.nn as torch_nn

import src.dataset


class Temporal(torch_nn.Module):
    """ Temporal model for a single single channel. Predicts the next value by
    considering a fixed history at the same coordinate. """

    def __init__(self, past: int, future: int, num_channels: int,
                 module: Union[torch_nn.Module, torch_nn.Sequential]):
        super(Temporal, self).__init__()
        self.past = past
        self.future = future
        self.num_channels = num_channels
        self.add_module('module', module)

    def forward(self, input):
        return self.module(input)

    def predict(self, frames: List[int], sample: src.dataset.Traffic4CastSample
               ) -> Dict[int, torch.tensor]:
        """ Predict requested frames.

            Predicts the requested frames for the given sample. If the frames
            required to predict the current frames are invalid they will be
            replaced with predicted frames.

            Args:
                frames: Frames for which to generate predictions.
                sample: Sample for which to generate predictions.

            Return:
                Dictionary map of frames to predictions tensors.

            Reaise:
                Exception: When the any frames necessary to predict the current
                    frame are invaild and they cannot be replaced with already
                    predicted frames.
        """
        predictions = collections.OrderedDict.fromkeys(frames)
        self.eval()
        with torch.no_grad():
            for frame, (slice, valid) in zip(
                    frames, sample.temporal_slices(self.past, frames, True)):

                if predictions[frame] is not None:
                    continue

                if not valid.all():
                    for f in range(valid.shape[0]):
                        if not valid[f]:
                            if predictions[f] is not None:
                                slice[f] = predictions[f]
                            else:
                                raise Exeception(
                                    f"Invalid frames for slice {f}.")

                pred_slice = self(
                    slice.reshape(-1, slice.shape[2],
                                  slice.shape[3]).unsqueeze(0)).squeeze(0)
                for pred_frame in range(pred_slice.shape[0] //
                                        self.num_channels):
                    predictions[frame +
                                pred_frame] = pred_slice[pred_frame *
                                                         self.num_channels:
                                                         (pred_frame + 1) *
                                                         self.num_channels]

                for key in predictions.keys():
                    if key not in frames:
                        del predictions[key]

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
    """ Simple 1 frame 1 channel model. """

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
