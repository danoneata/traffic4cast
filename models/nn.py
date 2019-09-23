from typing import List, Union, Dict

import collections
import enum
import pdb

import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as F

import src.dataset
import constants


def ignite_selected(loader, slice_size=15, epoch_fraction=None, to_return_temporal_info=True):
    num_batches = epoch_fraction and int(len(loader) * epoch_fraction)
    for i, batch in enumerate(loader):
        for sample in batch:
            selected_frames = [f + 3 for f in constants.START_FRAMES[sample.city]]
            minibatch_size = len(selected_frames)
            num_frames = sample.data.shape[0]
            for minibatch, frames in sample.selected_temporal_batches(
                    minibatch_size,
                    slice_size,
                    selected_frames,
                ):
                assert frames == selected_frames
                if to_return_temporal_info:
                    yield minibatch, sample.date, frames
                else:
                    yield minibatch
        if epoch_fraction and i >= num_batches:
            return


def plot_pred(data):
    """Small utility to generate plots of predictions for visual inspection."""
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(len(data), 3)
    for r, datum in enumerate(data):
        datum = datum.detach().cpu().numpy().squeeze()
        for c in range(3):
            axes[r, c].imshow(datum[c])
    plt.show()


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


def double_conv(in_channels, out_channels):
    return torch_nn.Sequential(
        torch_nn.Conv2d(in_channels, out_channels, 3, padding=1),
        torch_nn.ReLU(inplace=True),
        torch_nn.Conv2d(out_channels, out_channels, 3, padding=1),
        torch_nn.ReLU(inplace=True)
    )


class UNet(torch_nn.Module):

    def __init__(self, in_channels, out_channels, use_biases):
        super().__init__()
        C = 4

        self.dconv_down1 = double_conv(in_channels, 1 * C)
        self.dconv_down2 = double_conv(1 * C, 2 * C)
        self.dconv_down3 = double_conv(2 * C, 4 * C)
        self.dconv_down4 = double_conv(4 * C, 8 * C)

        self.maxpool = torch_nn.MaxPool2d(2)
        self.upsample = torch_nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(8 * C + 4 * C, 4 * C)
        self.dconv_up2 = double_conv(4 * C + 2 * C, 2 * C)
        self.dconv_up1 = double_conv(2 * C + 1 * C, 1 * C)

        self.conv_last = torch_nn.Conv2d(C, out_channels, kernel_size=1)
        self.use_biases = use_biases

        if self.use_biases:
            self.bias_loc = torch_nn.Parameter(torch.zeros(24, 1, 512, 512))
            self.bias_day = torch_nn.Parameter(torch.zeros(7))

    def forward(self, x, day=None, hours=None):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        if self.use_biases:
            x = x + self.bias_loc[hours] + self.bias_day[day]

        out = self.conv_last(x)

        return out


class SeasonalTemporalRegression(torch_nn.Module):
    def __init__(self, history: int, future: int):
        super(SeasonalTemporalRegression, self).__init__()
        self.history = history
        kwargs = dict(kernel_size=1, stride=1, padding=0, bias=True)
        self.temp_regr = torch_nn.Sequential(
            torch_nn.Conv2d(history, 16, **kwargs),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, 16, **kwargs),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, future, **kwargs)
        )
        self.unet = UNet(history, future, use_biases=True)
        self.bias_loc = torch_nn.Parameter(torch.zeros(24, 1, 495, 436))
        self.bias_day = torch_nn.Parameter(torch.zeros(7))

    def forward(self, x_date_frames):
        x, date, frames = x_date_frames
        B, T, H, W = x.shape
        weekday = date.weekday()
        hours = [int(f / 12) for f in frames]
        t = self.temp_regr(x)
        y = torch.tanh(t) + self.bias_loc[hours] + self.bias_day[weekday]
        x_padded = F.pad(x, (0, 512 - W, 0, 512 - H), "constant", 0)
        m_padded = self.unet(x_padded, day=weekday, hours=hours)
        mask = m_padded[:, :, :H, :W]
        mask = torch.sigmoid(mask)
        out = mask * y
        return out, mask, y


class SeasonalTemporalRegressionHeading(torch_nn.Module):
    def __init__(self, history: int, future: int):
        super(SeasonalTemporalRegressionHeading, self).__init__()
        self.history = history
        kwargs = dict(kernel_size=1, stride=1, padding=0, bias=True)
        self.temp_regr = torch_nn.Sequential(
            torch_nn.Conv2d(history, 16, **kwargs),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, 16, **kwargs),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, future * 5, **kwargs),
        )
        self.future = future
        self.directions = [0, 1, 85, 170, 255]
        self.n_directions = len(self.directions)
        self.directions = torch.tensor(self.directions).float().to('cuda').view(1, 5, 1, 1)
        self.directions = self.directions / 255
        self.bias_loc = torch_nn.Parameter(torch.zeros(24, 1, self.n_directions, 495, 436))
        self.bias_day = torch_nn.Parameter(torch.zeros(7, 1, self.n_directions, 1, 1))

    def forward(self, x_date_frames):
        x, date, frames = x_date_frames
        B, _, H, W = x.shape
        t = self.temp_regr(x)
        t = t.view(B, self.future, self.n_directions, H, W)
        weekday = date.weekday()
        hours = [int(f / 12) for f in frames]
        y = t + self.bias_loc[hours] + self.bias_day[weekday].view(1, 1, self.n_directions, 1, 1)
        y = torch.softmax(y, dim=2)
        out = (y * self.directions).sum(dim=2)
        return out


class Calba(torch_nn.Module):
    def __init__(self, history: int, future: int, n_layers=3, n_channels=16):
        super(Calba, self).__init__()
        self.history = history
        kwargs = dict(kernel_size=1, stride=1, padding=0, bias=True)

        self.temp_regr = []
        in_channels = history
        for _ in range(n_layers - 1):
            self.temp_regr.append(torch_nn.Conv2d(in_channels, n_channels, **kwargs))
            self.temp_regr.append(torch_nn.ReLU())
            in_channels = n_channels
        self.temp_regr.append(torch_nn.Conv2d(in_channels, future * 5, **kwargs))
        self.temp_regr = torch_nn.Sequential(*self.temp_regr)

        self.future = future
        self.directions = [0, 1, 85, 170, 255]
        self.n_directions = len(self.directions)
        self.directions = torch.tensor(self.directions).float().view(1, 5, 1, 1)
        self.directions = self.directions / 255
        self.bias_loc = torch_nn.Parameter(torch.zeros(24, 1, self.n_directions, 495, 436))
        self.bias_day = torch_nn.Parameter(torch.zeros(7, 1, self.n_directions, 1, 1))

    def forward(self, x_date_frames):
        x, date, frames = x_date_frames
        B, _, H, W = x.shape
        t = self.temp_regr(x)
        t = t.view(B, self.future, self.n_directions, H, W)
        weekday = date.weekday()
        hours = [int(f / 12) for f in frames]
        y = t + self.bias_loc[hours] + self.bias_day[weekday].view(1, 1, self.n_directions, 1, 1)
        y = torch.softmax(y, dim=2)
        d = self.directions.to(x.device)
        out = (y * d).sum(dim=2)
        return out
