from typing import List, Union, Dict

import collections
import enum
import pdb

import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

import src.dataset


# TODO Move to constants file
CHANNELS = ["volume", "speed", "heading"]
HEADING_VALUES = [0, 1, 85, 170, 255]


def plot_pred(data):
    """Small utility to generate plots of predictions for visual inspection."""
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

    def __init__(self, in_channels, out_channels, type_biases=None):
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

        self.bn = torch_nn.BatchNorm2d(C)
        self.conv_last = torch_nn.Conv2d(C, out_channels, kernel_size=1)
        self.type_biases = type_biases

        if self.type_biases == "L":
            self.bias_loc = torch_nn.Parameter(torch.zeros(1, 1, 512, 512))

        if self.type_biases == "LxH+W":
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

        if self.type_biases == "L":
            x = x + self.bias_loc

        if self.type_biases == "LxH+W":
            x = x + self.bias_loc[hours] + self.bias_day[day]

        x = self.bn(x)

        return self.conv_last(x)


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
        self.directions = HEADING_VALUES
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


def get_temporal_regressor(history, future):
    kwargs = dict(kernel_size=1, stride=1, padding=0, bias=True)
    return torch_nn.Sequential(
        torch_nn.Conv2d(history, 16, **kwargs),
        torch_nn.ReLU(),
        torch_nn.Conv2d(16, 16, **kwargs),
        torch_nn.ReLU(),
        torch_nn.Conv2d(16, future, **kwargs)
    )


H = 495
W = 436


def pad(x):
    return F.pad(x, (0, 512 - W, 0, 512 - H), "constant", 0)


def unpad(x):
    return x[:, :, :H, :W]


def map_heading_to_consecutive(d):
    o = torch.zeros(d.shape, dtype=torch.uint8)
    o = o.to(d.device)
    for i, v in enumerate(HEADING_VALUES):
        o.masked_fill_(d == v / 255, i)
    return o


class Conv2dLocation(torch_nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Conv2dLocation, self).__init__()
        self.w = torch_nn.Parameter(torch.zeros(1, outplanes, inplanes, 495, 436))
        self.b = torch_nn.Parameter(torch.zeros(1, outplanes,        1, 495, 436))

    def forward(self, x):
        # x.shape = B x      C x H x W
        # w.shape = 1 x C' x C x H x W
        o = (self.w * x.unsqueeze(1)).sum(dim=2, keepdim=True) + self.b
        return o.squeeze(2)


class MaskPredictor(torch_nn.Module):
    def __init__(self, future):
        super(MaskPredictor, self).__init__()
        N_CHANNELS = 3
        E = 2
        self.embed = torch_nn.Embedding(num_embeddings=5, embedding_dim=E)
        self.unet = UNet(E + 1, 2, type_biases="L")
        self.loc_net = Conv2dLocation(3, 2)
        self.tr = get_temporal_regressor(2, 1)
        self.b = torch_nn.Parameter(torch.zeros(1, 1, 495, 436))
        # self.tconv1 = get_temporal_regressor(12, 8)
        # self.tconv2 = get_temporal_regressor(12, 8)
        # self.tconv3 = get_temporal_regressor(8, 8)
        # self.tconv4 = get_temporal_regressor(8, 3)

    def forward(self, x):
        # x.shape → B, T, C, H, W
        B, _, _, H, W = x.shape
        h = x[:, -1, 2]
        h = map_heading_to_consecutive(h)
        h = self.embed(h.long())
        h = h.permute(0, 3, 1, 2)
        f = torch.cat([x[:, -1, 1:2], h], dim=1)  # speed + heading
        f = self.loc_net(f)
        # g = pad(g)
        # g = self.unet(g)
        # f = unpad(g)
        f = f.permute(0, 2, 3, 1)
        # f = 0.5 * torch.tanh(f)
        # g = torch.tanh(g)
        g = torch.meshgrid([torch.arange(-1, 1, step=2 / W), torch.arange(-1, 1, step=2 / H)])
        g = torch.stack((g[0].t(), g[1].t()), dim=-1).cuda().repeat(B, 1, 1, 1)
        g = g + 0.1 * torch.tanh(f)
        y = self.tr(x[:, -1, :2])
        # y = (x[:, -1, :1] > 0).float()
        out = F.grid_sample(y, g)
        # print(f.min().detach().cpu(), f.max().detach().cpu())
        # print(g.min().detach().cpu(), g.max().detach().cpu())
        # print(self.b.min().detach().cpu(), self.b.max().detach().cpu())
        # print(out.min().detach().cpu(), out.max().detach().cpu())
        out = torch.sigmoid(out + self.b)
        # fig, axes = plt.subplots(nrows=2)
        # axes[0].imshow(y[-1].detach().cpu().numpy().squeeze())
        # axes[1].imshow(out[-1].detach().cpu().numpy().squeeze())
        # plt.savefig('/tmp/ooo.png')
        return out


class Lygia(torch_nn.Module):
    def __init__(self, history: int, future: int):
        super(Lygia, self).__init__()
        self.history = history
        self.future = future
        self.n_channels = 3
        self.mask_predictor = MaskPredictor(future)
        self.temporal_regressors = torch_nn.ModuleList([
            get_temporal_regressor(history, future)
            for channel in CHANNELS
        ])
        self.b = torch_nn.Parameter(torch.zeros(1, 1, 3, 495, 436))

    def forward(self, data):
        x, _, _ = data
        B, _, H, W = x.shape
        # B, T, C, H, W
        x = x.view(B, self.history, self.n_channels, H, W)
        mask = self.mask_predictor(x)
        mask = mask.unsqueeze(2)
        y = [
            self.temporal_regressors[i](x[:, :, i]).unsqueeze(2)
            for i in range(self.n_channels)
        ]
        y = torch.cat(y, dim=2)
        y = torch.sigmoid(y + self.b)
        out = mask * y
        return out, mask, y
