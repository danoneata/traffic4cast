from typing import List, Union, Dict

import collections
import enum
import pdb

import numpy as np
import torch
import torch.nn as torch_nn

import src.dataset


class ReplacementMode(enum.Enum):
    """ Enumeration of frame replacement modes for prediction.

        ALL_FROM_PREDICTED - Given a sequence of F frames, denoted by the
            frame indice [f0, f1,...,fi,...,fF-1] into the given sample stream
            and a list of P, predicted, frames denoted by frame indices
            [p0, p1,...,pk,...,pP-1], replace every frame, fi, of the sequence F
            with predicted frames from the P list if fi in [p0, p1,..., pP-1].

        INVALID_FROM_PREDICTED - Given a sequence of F frames, denoted by the
            frame indice [f0, f1,...,fi,...,fF-1] into the given sample stream,
            a list of booleans denoting the validity of each frame in the
            sequence [vf0, vf1,...,vfi,...,vfF-1] and a list of P, predicted,
            frames denoted by frame indices [p0, p1,...,pk,...,pP-1], replace
            every frame, fi, of the sequence F with predicted frames from the P
            list if vfi is False and fi in [p0, p1,..., pP-1].
    """
    ALL_FROM_PREDICTED = 0
    INVALID_FROM_PREDICTED = 1


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

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, input):
        return self.module(input)

    def predict(self,
                frames: List[int],
                sample: src.dataset.Traffic4CastSample,
                mode: ReplacementMode = ReplacementMode.ALL_FROM_PREDICTED
               ) -> Dict[int, torch.tensor]:
        """ Predict requested frames.

            Predicts the requested frames for the given sample. For each frame
            to predict the input frames are replaced with predicted frames
            based on the selected replacement mode. The default replacement
            mode is ReplacementMode.ALL_FROM_PREDICTED

            Args:
                frames: Frames for which to generate predictions.
                sample: Sample for which to generate predictions.
                mode: Replacement mode.

            Return:
                Dictionary map of frames indices to predicted frames(tensors).

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

                # Don't predict again if already predicted.
                if predictions[frame] is not None:
                    continue

                if mode == ReplacementMode.ALL_FROM_PREDICTED:
                    for v, f in enumerate(range(frame - self.past, frame)):
                        if (f in predictions.keys() and
                                predictions[f] is not None):
                            slice[v] = predictions[f]
                            valid[v] = 1
                    if not valid.all():
                        raise Exception(f"Invalid frames for slice {f}.")
                elif mode == ReplacementMode.INVALID_FROM_PREDICTED:
                    if not valid.all():
                        for v, f in enumerate(range(frame - self.past, frame)):
                            if not valid[v]:
                                if (f in predictions.keys() and
                                        predictions[f] is not None):
                                    slice[v] = predictions[f]
                                else:
                                    raise Exception(
                                        f"Invalid frames for slice {f}.")
                else:
                    raise Exeception(f"Invalid frame repacement mode.")

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

            # The model might predict other frames then the requested ones.
            # Remove from the result the frames that were not requested.
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

    def ignite_random(self, loader, num_minibatches, minibatch_size,
                      epoch_fraction):
        num_batches = int(len(loader) * epoch_fraction)
        for batch in loader:
            for sample in batch:
                for minibatch in sample.random_temporal_batches(
                        num_minibatches, minibatch_size,
                        self.past + self.future):
                    yield minibatch
            num_batches -= 1
            if num_batches <= 0:
                return

    def ignite_all(self, loader, minibatch_size):
        for batch in loader:
            for sample in batch:
                num_frames = sample.data.shape[0]
                for minibatch in sample.selected_temporal_batches(
                        minibatch_size, self.past + self.future,
                        range(self.past + self.future,
                              num_frames - self.past - self.future)):
                    yield minibatch


class TemporalDate(torch_nn.Module):
    """ Temporal model for a single single channel. Predicts the next value by
    considering a fixed history at the same coordinate. """

    def __init__(self, past: int, future: int, num_channels: int,
                 module: Union[torch_nn.Module, torch_nn.Sequential]):
        super(TemporalDate, self).__init__()
        self.past = past
        self.future = future
        self.num_channels = num_channels
        self.add_module('module', module)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, input):
        return self.module(input)

    def predict(self,
                frames: List[int],
                sample: src.dataset.Traffic4CastSample,
                mode: ReplacementMode = ReplacementMode.ALL_FROM_PREDICTED
               ) -> Dict[int, torch.tensor]:
        """ Predict requested frames.

            Predicts the requested frames for the given sample. For each frame
            to predict the input frames are replaced with predicted frames
            based on the selected replacement mode. The default replacement
            mode is ReplacementMode.ALL_FROM_PREDICTED

            Args:
                frames: Frames for which to generate predictions.
                sample: Sample for which to generate predictions.
                mode: Replacement mode.

            Return:
                Dictionary map of frames indices to predicted frames(tensors).

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

                # Don't predict again if already predicted.
                if predictions[frame] is not None:
                    continue

                if mode == ReplacementMode.ALL_FROM_PREDICTED:
                    for v, f in enumerate(range(frame - self.past, frame)):
                        if (f in predictions.keys() and
                                predictions[f] is not None):
                            slice[v] = predictions[f]
                            valid[v] = 1
                    if not valid.all():
                        raise Exception(f"Invalid frames for slice {f}.")
                elif mode == ReplacementMode.INVALID_FROM_PREDICTED:
                    if not valid.all():
                        for v, f in enumerate(range(frame - self.past, frame)):
                            if not valid[v]:
                                if (f in predictions.keys() and
                                        predictions[f] is not None):
                                    slice[v] = predictions[f]
                                else:
                                    raise Exception(
                                        f"Invalid frames for slice {f}.")
                else:
                    raise Exeception(f"Invalid frame repacement mode.")

                inp = (
                    slice.reshape(-1, slice.shape[2], slice.shape[3]).unsqueeze(0),
                    sample.date,
                    [frame],
                )
                pred_slice = self(inp).squeeze(0)
                for pred_frame in range(pred_slice.shape[0] //
                                        self.num_channels):
                    s = pred_frame * self.num_channels
                    e = (pred_frame + 1) * self.num_channels
                    predictions[frame + pred_frame] = pred_slice[s: e]

            # The model might predict other frames then the requested ones.
            # Remove from the result the frames that were not requested.
            for key in predictions.keys():
                if key not in frames:
                    del predictions[key]

        return predictions

    def ignite_batch(self, batch, device, non_blocking):
        batch1, date, frames = batch
        batch1 = batch1.to(device)
        batch1 = batch1.reshape(
            batch1.shape[0],
            -1,
            batch1.shape[3],
            batch1.shape[4],
        )
        inp = batch1[:, :self.past * self.num_channels], date, frames
        tgt = batch1[:, self.past * self.num_channels:]
        return inp, tgt

    def ignite_random(self, loader, num_minibatches, minibatch_size,
                      epoch_fraction):
        num_batches = int(len(loader) * epoch_fraction)
        for batch in loader:
            for sample in batch:
                for minibatch, frames in sample.random_temporal_batches(
                        num_minibatches,
                        minibatch_size,
                        self.past + self.future):
                    yield minibatch, sample.date, frames
            num_batches -= 1
            if num_batches <= 0:
                return

    def ignite_all(self, loader, minibatch_size):
        for batch in loader:
            for sample in batch:
                num_frames = sample.data.shape[0]
                for minibatch, frames in sample.selected_temporal_batches(
                        minibatch_size, self.past + self.future,
                        range(self.past + self.future, num_frames - self.past - self.future)):
                    yield minibatch, sample.date, frames


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


class SeasonalTemporalRegression(torch_nn.Module):
    def __init__(self, history: int):
        super(SeasonalTemporalRegression, self).__init__()
        self.history = history
        kwargs = dict(kernel_size=1, stride=1, padding=0, bias=True)
        kwargs3 = dict(kernel_size=3, stride=1, padding=1, bias=True)
        self.temp_regr = torch_nn.Sequential(
            torch_nn.Conv2d(history, 16, **kwargs),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, 16, **kwargs),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, 1, **kwargs)
        )
        self.nn2 = torch_nn.Sequential(
            torch_nn.Conv2d(history, 16, **kwargs3),
            torch_nn.BatchNorm2d(16),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, 16, **kwargs3),
            torch_nn.BatchNorm2d(16),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, 1, **kwargs3)
        )
        self.bias_loc = torch_nn.Parameter(torch.zeros(24, 1, 495, 436))
        self.bias_day = torch_nn.Parameter(torch.zeros(7))
        # self.bias_hour = torch_nn.Parameter(torch.zeros(24, 1, 1, 1))

    def forward(self, x_date_frames):
        x, date, frames = x_date_frames
        t = self.temp_regr(x)
        d = date.weekday()
        h = [int(f / 12) for f in frames]
        z = torch.tanh(t) + self.bias_loc[h] + self.bias_day[d]
        m = torch.sigmoid(self.nn2(x))
        return z * m, m


class SeasonalTemporalRegressionHeading(torch_nn.Module):
    def __init__(self, history: int):
        super(SeasonalTemporalRegressionHeading, self).__init__()
        self.history = history
        kwargs = dict(kernel_size=1, stride=1, padding=0, bias=True)
        self.temp_regr = torch_nn.Sequential(
            torch_nn.Conv2d(history, 16, **kwargs),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, 16, **kwargs),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, 5, **kwargs),
        )
        self.bias_loc = torch_nn.Parameter(torch.zeros(1, 5, 495, 436))
        self.bias_day = torch_nn.Parameter(torch.zeros(7, 5, 1, 1))
        self.bias_hour = torch_nn.Parameter(torch.zeros(24, 5, 1, 1))
        self.v = [0, 1, 85, 170, 255]
        self.v = torch.tensor(self.v).float().to('cuda').view(1, 5, 1, 1)
        self.v = self.v / 255

    def forward(self, x_date_frames):
        x, date, frames = x_date_frames
        x = self.temp_regr(x)
        d = date.weekday()
        h = [int(f / 12) for f in frames]
        x = x + self.bias_loc + self.bias_day[d].view(1, 5, 1, 1) + self.bias_hour[h]
        x = torch.softmax(x, dim=1)
        x = (x * self.v).sum(dim=1, keepdim=True)
        return x
