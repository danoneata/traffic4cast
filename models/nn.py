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

from models.layers import (
    Conv2dLocal,
    DenseBasicBlock,
    DenseBlock,
)


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
                    [frame + 3],
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


class Petronius(torch_nn.Module):
    def __init__(self, history):
        super(Petronius, self).__init__()
        kwargs = dict(kernel_size=1, stride=1, padding=0, bias=True)
        self.bias = torch_nn.Parameter(torch.zeros(5, 3, 495, 436))
        self.bias_day = torch_nn.Parameter(torch.zeros(7))
        self.history = history
        self.temp_regr = torch_nn.Sequential(
            torch_nn.Conv2d(history, 16, **kwargs),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, 16, **kwargs),
            torch_nn.ReLU(),
            torch_nn.Conv2d(16, 3, **kwargs),
            torch_nn.Tanh(),
        )

    def forward(self, data):
        x, date, _ = data
        weekday = date.weekday()
        B, _, H, W = x.shape
        out = self.temp_regr(x) + self.bias + self.bias_day[weekday]
        return out


class Filter1x1(torch_nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Filter1x1, self).__init__()
        self.w = torch_nn.Parameter(torch.randn(1, outplanes, inplanes, 495, 436))
        self.b = torch_nn.Parameter(torch.zeros(1, outplanes,        1, 495, 436))

    def forward(self, x):
        # x.shape = B x      C x H x W
        # w.shape = 1 x C' x C x H x W
        o = (self.w * x.unsqueeze(1)).sum(dim=2, keepdim=True) + self.b
        return o.squeeze(2)


class GetLastChannels(torch_nn.Module):
    def __init__(self, t):
        super(GetLastChannels, self).__init__()
        self.t = t

    def forward(self, x):
        return x[:, -self.t:]


def build_uniform_network(
        get_block,
        get_activ,
        history,
        future,
        n_layers,
        n_channels,
        out_activ=lambda: torch_nn.Tanh(),
    ):
    network = []
    in_channels = history
    network.append(GetLastChannels(history))
    for _ in range(n_layers - 1):
        network.append(get_block(in_channels, n_channels))
        network.append(get_activ())
        in_channels = n_channels
    network.append(get_block(in_channels, future))
    if out_activ is not None:
        network.append(out_activ())  # FIXME Should we parmeterize the output activation?
    return torch_nn.Sequential(*network)


class PetroniusParam(torch_nn.Module):
    """Parameterized version of Petronius used for hyper-parameter tuning."""
    N_BATCH = 5
    FUTURE = 3
    HEIGHT = 495
    WIDTH = 436

    def __init__(self, temp_reg_params, filt_1x1_params, biases_type):
        super(PetroniusParam, self).__init__()
        kernel_size = temp_reg_params.pop("kernel_size")
        padding = (kernel_size - 1) // 2
        get_block_conv = lambda i, o: torch_nn.Conv2d(i, o, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
        get_block_f1x1 = lambda i, o: Filter1x1(i, o)
        get_activ = lambda: torch_nn.ReLU()
        self.temp_reg = build_uniform_network(get_block_conv, get_activ, future=self.FUTURE, **temp_reg_params)
        self.filt_1x1 = build_uniform_network(get_block_f1x1, get_activ, future=self.FUTURE, **filt_1x1_params) if filt_1x1_params["n_layers"] else None
        # Biases
        self.bias_loctime = torch_nn.ParameterList(self._get_bias_loctime(biases_type["loctime"]))
        self.bias_weekday = torch_nn.Parameter(torch.zeros(7)) if biases_type["weekday"] else None
        self.bias_month = torch_nn.Parameter(torch.zeros(12)) if biases_type["month"] else None

    def forward(self, data):
        x, date, _ = data
        B, _, H, W = x.shape
        out = self.temp_reg(x)
        if self.filt_1x1 is not None:
            out = out + self.filt_1x1(x)
        # Add the biases
        for b in self.bias_loctime:
            out = out + b
        if self.bias_weekday is not None:
            out = out + self.bias_weekday[date.weekday()]
        if self.bias_month is not None:
            out = out + self.bias_month[date.month - 1]
        return out

    def _get_bias_loctime(self, type1):
        get_param = lambda *shape: torch_nn.Parameter(torch.zeros(*shape))
        if type1 == "L":
            return [get_param(1, 1, self.HEIGHT, self.WIDTH)]
        elif type1 == "T":
            return [get_param(self.N_BATCH, 1, 1, 1)]
        elif type1 == "LxT":
            return [get_param(self.N_BATCH, 1, self.HEIGHT, self.WIDTH)]
        elif type1 == "L+T":
            return [
                get_param(1, 1, self.HEIGHT, self.WIDTH),
                get_param(self.N_BATCH, self.FUTURE, 1, 1),
            ]
        else:
            assert False, "Unknown type of bias"


class PetroniusHeading(torch_nn.Module):
    """Version of Petronius for the heading channel. This version allows for
    input and output embeddings.

    """
    N_BATCH = 5
    FUTURE = 3
    HEIGHT = 495
    WIDTH = 436
    BIASES_TYPE = {
        "loctime": "L+T",
        "weekday": True,
        "month": True,
    }
    TEMP_REG_PARAMS = {
        "history": 12,
        "kernel_size": 1,
        "n_channels": 16,
        "n_layers": 6,
    }
    EMB_DIM_IN = 2
    DIRECTIONS = torch.tensor([0, 1, 85, 170, 255]).float() / 255
    N_DIRECTIONS = 5

    def __init__(self, embeddings_type):
        super(PetroniusHeading, self).__init__()
        # Prepare parameters
        temp_reg_params = self.TEMP_REG_PARAMS
        biases_type = self.BIASES_TYPE
        kernel_size = temp_reg_params.pop("kernel_size")
        history = temp_reg_params.pop("history")
        if embeddings_type["in"]:
            history = self.EMB_DIM_IN * history
        if embeddings_type["out"]:
            future = self.N_DIRECTIONS * self.FUTURE
        padding = (kernel_size - 1) // 2
        get_block_conv = lambda i, o: torch_nn.Conv2d(
            i,
            o,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )
        get_activ = lambda: torch_nn.ReLU()
        self.temp_reg = build_uniform_network(
            get_block_conv,
            get_activ,
            future=future,
            history=history,
            **temp_reg_params,
        )
        # Embeddings
        self.emb1 = (
            torch.nn.Embedding(self.N_DIRECTIONS, self.EMB_DIM_IN)
            if embeddings_type["in"]
            else None
        )
        self.emb2 = embeddings_type["out"]
        # Biases
        self.bias_loctime = torch_nn.ParameterList(self._get_bias_loctime(biases_type["loctime"]))
        self.bias_weekday = torch_nn.Parameter(torch.zeros(7)) if biases_type["weekday"] else None
        self.bias_month = torch_nn.Parameter(torch.zeros(12)) if biases_type["month"] else None

    def forward(self, data):
        x, date, _ = data
        B, _, H, W = x.shape
        self.DIRECTIONS = self.DIRECTIONS.to(x.device)
        # Input embedding
        if self.emb1 is not None:
            x = self._embed_in(x)
        out = self.temp_reg(x)
        if self.emb2 is not None:
            out = out.view(B, self.FUTURE, self.N_DIRECTIONS, H, W)
        # Add the biases
        for b in self.bias_loctime:
            out = out + b
        if self.bias_weekday is not None:
            out = out + self.bias_weekday[date.weekday()]
        if self.bias_month is not None:
            out = out + self.bias_month[date.month - 1]
        # Output embedding
        if self.emb2 is not None:
            out = torch.softmax(out, dim=2)
            out = (out * self.DIRECTIONS.view(1, 1, self.N_DIRECTIONS, 1, 1)).sum(dim=2)
        return out

    def _embed_in(self, x):
        y = torch.zeros(x.shape).long().to(x.device)
        for i, v in enumerate(self.DIRECTIONS):
            y[x == v] = i
        y = self.emb1(y)
        y = y.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = y.shape
        y = y.reshape(B, T * C, H, W)
        return y

    def _get_shape(self, n_batch, n_channels, height, width):
        if self.emb2:
            return n_batch, n_channels, self.N_DIRECTIONS, height, width
        else:
            return n_batch, n_channels, height, width

    def _get_bias_loctime(self, type1):
        get_param = lambda *shape: torch_nn.Parameter(torch.zeros(self._get_shape(*shape)))
        if type1 == "L":
            return [get_param(1, 1, self.HEIGHT, self.WIDTH)]
        elif type1 == "T":
            return [get_param(self.N_BATCH, self.FUTURE, 1, 1)]
        elif type1 == "LxT":
            return [get_param(self.N_BATCH, self.FUTURE, self.HEIGHT, self.WIDTH)]
        elif type1 == "L+T":
            return [
                get_param(1, 1, self.HEIGHT, self.WIDTH),
                get_param(self.N_BATCH, self.FUTURE, 1, 1),
            ]
        else:
            assert False, "Unknown type of bias"


class Marcus(torch_nn.Module):

    FUTURE = 3
    HISTORY = 12
    N_LAYERS = 3
    N_BATCHES = 5
    N_CHANNELS = 16
    HEIGHT = 495
    WIDTH = 436

    DIRECTIONS = torch.tensor([0, 1, 85, 170, 255]).float() / 255
    N_DIRECTIONS = 5
    EMB_DIM_IN = 2

    def __init__(self, local_filt_params, embed_in):
        super(Marcus, self).__init__()
        get_block_conv = lambda i, o: torch_nn.Conv2d(
            i,
            o,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        get_activ = lambda: torch_nn.ReLU()
        if embed_in:
            history = self.EMB_DIM_IN * self.HISTORY
        else:
            history = self.HISTORY
        self.temp_reg = build_uniform_network(
            get_block_conv,
            get_activ,
            history=history,
            future=self.N_DIRECTIONS * self.FUTURE,
            n_layers=self.N_LAYERS,
            n_channels=self.N_CHANNELS,
            out_activ=None,
        )
        self.local_filt = (
            torch_nn.Sequential(
                GetLastChannels(local_filt_params["history"]),
                get_block_conv(local_filt_params["history"], local_filt_params["n_channels"]),
                torch_nn.ReLU(),
                Conv2dLocal(self.HEIGHT, self.WIDTH, local_filt_params["n_channels"], local_filt_params["n_channels"], 9, padding=4),
                torch_nn.ReLU(),
                get_block_conv(local_filt_params["n_channels"], self.N_DIRECTIONS * self.FUTURE),
            )
            if local_filt_params["n_layers"] > 0
            else None
        )
        # Embeddings
        self.emb1 = (
            torch.nn.Embedding(self.N_DIRECTIONS, self.EMB_DIM_IN)
            if embed_in
            else None
        )
        self.bias = torch_nn.Parameter(torch.zeros(
            self.N_BATCHES,
            1, # self.FUTURE,
            self.N_DIRECTIONS,
            self.HEIGHT,
            self.WIDTH,
        ))
        self.bias_weekday = torch_nn.Parameter(torch.zeros(
            7,
            1,
            1,
            self.N_DIRECTIONS,
            1,
            1,
        ))

    def forward(self, data):
        x, date, _ = data
        B, _, H, W = x.shape
        d = self.DIRECTIONS.to(x.device)
        # Input embedding
        if self.emb1 is not None:
            out = self._embed_in(x)
        else:
            out = x
        out = self.temp_reg(out)
        if self.local_filt is not None:
            out = out + self.local_filt(x)
        out = out.view(B, self.FUTURE, self.N_DIRECTIONS, H, W)
        out = out + self.bias
        out = out + self.bias_weekday[date.weekday()]
        # out = torch.sigmoid(out)
        out = torch.softmax(out, dim=2)
        out = (out * d.view(1, 1, self.N_DIRECTIONS, 1, 1)).sum(dim=2)
        return out

    def _embed_in(self, x):
        d = self.DIRECTIONS.to(x.device)
        y = torch.zeros(x.shape).long().to(x.device)
        for i, v in enumerate(d):
            y[x == v] = i
        y = self.emb1(y)
        y = y.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = y.shape
        y = y.reshape(B, T * C, H, W)
        return y


class CalinaHeading(torch_nn.Module):

    FUTURE = 3
    HISTORY = 12
    N_BATCHES = 5
    N_CHANNELS = 16
    N_LAYERS = 3
    HEIGHT = 495
    WIDTH = 436

    DIRECTIONS = torch.tensor([0, 1, 85, 170, 255]).float() / 255
    N_DIRECTIONS = 5

    def __init__(self, temp_reg_params, biases_type):
        super(CalinaHeading, self).__init__()
        kernel_size = temp_reg_params["kernel_size"]
        padding = (kernel_size - 1) // 2
        get_block_conv = lambda i, o: torch_nn.Conv2d(
            i,
            o,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )
        self.temp_reg = build_uniform_network(
            get_block_conv,
            lambda: self._get_activ(temp_reg_params["activation"]),
            history=self.HISTORY,
            future=self.N_DIRECTIONS * self.FUTURE,
            n_layers=temp_reg_params["n_layers"],
            n_channels=temp_reg_params["n_channels"],
            out_activ=None,
        )
        # Biases
        self.bias_location = torch_nn.ParameterList(self._get_bias_location(biases_type["location"]))
        self.bias_weekday = self._get_bias_weekday(biases_type["weekday"])
        self.bias_month = self._get_bias_month(biases_type["month"])

    def forward(self, data):
        x, date, _ = data
        B, _, H, W = x.shape
        dirs = self.DIRECTIONS.to(x.device)
        dirs = dirs.view(1, 1, self.N_DIRECTIONS, 1, 1)
        out = x
        out = self.temp_reg(out)
        out = out.view(B, self.FUTURE, self.N_DIRECTIONS, H, W)
        # Add the biases
        for b in self.bias_location:
            out = out + b
        if self.bias_weekday is not None:
            out = out + self.bias_weekday[date.weekday()]
        if self.bias_month is not None:
            out = out + self.bias_month[date.month - 1]
        out = torch.softmax(out, dim=2)
        out = torch.sum(out * dirs, dim=2)
        return out

    def _get_activ(self, type1):
        if type1 == "ReLU":
            return torch_nn.ReLU()
        elif type1 == "ELU":
            return torch_nn.ELU()
        elif type1 == "LeakyReLU":
            return torch_nn.LeakyReLU()
        elif type1 == "SELU":
            return torch_nn.SELU()
        else:
            assert False, "Unknown type of activation"

    def _get_bias_location(self, type1):
        get_param = lambda *shape: torch_nn.Parameter(torch.zeros(*shape))
        if type1 == "LxT":
            return [get_param(self.N_BATCHES, 1, self.N_DIRECTIONS, self.HEIGHT, self.WIDTH)]
        elif type1 == "L+T":
            return [
                get_param(1, 1, self.N_DIRECTIONS, self.HEIGHT, self.WIDTH),
                get_param(self.N_BATCHES, 1, self.N_DIRECTIONS, 1, 1),
            ]
        else:
            assert False, "Unknown type of bias"

    def _get_bias_weekday(self, type1):
        if type1 == "W":
            return torch_nn.Parameter(torch.zeros(7, 1, self.N_DIRECTIONS, 1, 1))
        elif type1 == "WxT":
            return torch_nn.Parameter(torch.zeros(7, self.N_BATCHES, 1, self.N_DIRECTIONS, 1, 1))
        elif type1 == "":
            return None
        else:
            assert False, "Unknown type of bias"

    def _get_bias_month(self, type1):
        if type1 == "M":
            return torch_nn.Parameter(torch.zeros(12, 1, self.N_DIRECTIONS, 1, 1))
        elif type1 == "MxT":
            return torch_nn.Parameter(torch.zeros(12, self.N_BATCHES, 1, self.N_DIRECTIONS, 1, 1))
        elif type1 == "":
            return None
        else:
            assert False, "Unknown type of bias"


class Vicinius(torch_nn.Module):

    FUTURE = 3
    HISTORY = 12
    N_LAYERS = 3
    N_BATCHES = 5
    N_CHANNELS = 16
    HEIGHT = 495
    WIDTH = 436

    DIRECTIONS = torch.tensor([0, 1, 85, 170, 255]).float() / 255
    N_DIRECTIONS = 5

    def __init__(self):
        super(Vicinius, self).__init__()
        self.conv1 = torch_nn.Conv2d(12, 8, kernel_size=1, stride=1, padding=0, bias=True)
        self.dense = DenseBlock(3, 8, 4, DenseBasicBlock)
        self.relu1 = torch_nn.ReLU(inplace=True)
        self.conv2 = torch_nn.Conv2d(20, 15, kernel_size=1, stride=1, padding=0, bias=True)
        self.bias = torch_nn.Parameter(torch.zeros(
            self.N_BATCHES,
            1, # self.FUTURE,
            self.N_DIRECTIONS,
            self.HEIGHT,
            self.WIDTH,
        ))
        self.bias_weekday = torch_nn.Parameter(torch.zeros(
            7,
            1,
            1,
            self.N_DIRECTIONS,
            1,
            1,
        ))

    def forward(self, data):
        x, date, _ = data
        B, _, H, W = x.shape
        d = self.DIRECTIONS.to(x.device)
        x = self.conv1(x)
        x = self.dense(x)
        x = self.conv2(self.relu1(x))
        x = x.view(B, self.FUTURE, self.N_DIRECTIONS, H, W)
        x = x + self.bias + self.bias_weekday[date.weekday()]
        x = torch.softmax(x, dim=2)
        x = (x * d.view(1, 1, self.N_DIRECTIONS, 1, 1)).sum(dim=2)
        return x


class Nero(torch_nn.Module):

    FUTURE = 3
    HISTORY = 12
    N_LAYERS = 3
    N_BATCHES = 5
    N_CHANNELS = 3
    HEIGHT = 495
    WIDTH = 436

    def __init__(self):
        super(Nero, self).__init__()
        self.filter_size = 3
        self.filter_history = 3
        self.filter_future = 1
        get_block_conv = lambda i, o: torch_nn.Conv2d(
            i,
            o,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        get_activ = lambda: torch_nn.ELU()
        self.temp_reg = build_uniform_network(
            get_block_conv,
            get_activ,
            history=self.HISTORY * self.N_CHANNELS,
            future=self.FUTURE * self.N_CHANNELS,
            n_layers=5,
            n_channels=16,
            out_activ=None,
        )
        # self.filter_pred_1 = build_uniform_network(
        #     get_block_conv,
        #     get_activ,
        #     history=self.filter_history * self.N_CHANNELS,
        #     future=self.filter_size * self.filter_size * self.filter_history * self.N_CHANNELS * 2,
        #     n_layers=5,
        #     n_channels=16,
        #     out_activ=None,
        # )
        # self.filter_pred_2 = build_uniform_network(
        #     get_block_conv,
        #     get_activ,
        #     history=self.filter_history * self.N_CHANNELS,
        #     future=self.filter_size * self.filter_size * 2 * self.N_CHANNELS * self.filter_future,
        #     n_layers=5,
        #     n_channels=16,
        #     out_activ=None,
        # )
        self.bias_location = torch_nn.Parameter(torch.zeros(
            self.N_BATCHES,
            1,
            self.N_CHANNELS,
            self.HEIGHT,
            self.WIDTH,
        ))
        self.bias_weekday = torch_nn.Parameter(torch.zeros(
            7,
            1,
            1,
            self.N_CHANNELS,
            1,
            1,
        ))

    def forward(self, data):
        x, date, _ = data
        # y = self._select_last(x, self.filter_history)
        # w1 = self.filter_pred_1(y)  # Predict local filter weights.
        # w2 = self.filter_pred_2(y)  # Predict local filter weights.
        # f = self._apply_filter(y, w1, self.filter_history * self.N_CHANNELS, 2)
        # f = torch.relu(f)
        # f = self._apply_filter(f, w2, 2, self.filter_future * self.N_CHANNELS)
        # f = f.view(self.N_BATCHES, self.filter_future, self.N_CHANNELS, self.HEIGHT, self.WIDTH)
        t = self.temp_reg(x).view(self.N_BATCHES, self.FUTURE, self.N_CHANNELS, self.HEIGHT, self.WIDTH)
        out = t + self.bias_location + self.bias_weekday[date.weekday()]
        # out = torch.sigmoid(out)
        out = out.view(self.N_BATCHES, self.FUTURE * self.N_CHANNELS, self.HEIGHT, self.WIDTH)
        return out

    def _select_last(self, x, t):
        # Selects last `t` frames.
        y = x.view(self.N_BATCHES, self.HISTORY, self.N_CHANNELS, self.HEIGHT, self.WIDTH)
        y = y[:, -t:]
        y = y.view(self.N_BATCHES, self.filter_history * self.N_CHANNELS, self.HEIGHT, self.WIDTH)
        return y

    def _apply_filter(self, y, w, i, o):
        # Apply filter weights to data.
        y = F.unfold(y, self.filter_size, padding=1)
        w = w.view(self.N_BATCHES, self.filter_size * self.filter_size * i, o, self.HEIGHT * self.WIDTH)
        f = torch.einsum("bfs, bfos -> bos", y, w)
        f = f.view(self.N_BATCHES, o, self.HEIGHT, self.WIDTH)
        return f
