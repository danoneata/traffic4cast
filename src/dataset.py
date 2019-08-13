from typing import List, Callable, Union, Tuple
import datetime
import inspect
import math
import os
import random

import numpy as np
import h5py
import torch
import torch.utils.data


SEED = 1337
random.seed(SEED)


def path_to_date(path: str) -> datetime.datetime:
    return datetime.datetime.strptime(
        os.path.basename(path).split('_')[0], '%Y%m%d')


class Traffic4CastSample(object):
    """ Traffic4cast data wrapper.

        Attributes:
            path (str): Path to the .hdf5 data file.
            data (torch.tensor): 4D torch.tensor for the sample data.
            city (str): City where data sample was collected.
            date (datetime): Date when the data sample was collected.
            layout (str): Current axes meaning.
            channel_layout (str): Current channels meaning.
            valid (torch.tensor): Boolean mask, over the frames, denoting the
                validity of the frame.(In the test set the frames that are to
                be predicted are set to zero, hence they are not valid to use
                for predicting other frames.)
    """
    time_step_delta = datetime.timedelta(minutes=5)
    index_to_channel = {0: "Volume", 1: "Speed", 2: "Heading"}
    channel_to_index = {v: k for k, v in index_to_channel.items()}

    def __init__(self, path: str, city: str):
        """ Initializes the Traffic4CastSample data sample

            Args:
                path: Path to the .hdf5 data file.
                city: Name of they city where the data sample was collected.
        """

        self.path = path
        self.data = None
        self.city = city
        self.date = path_to_date(path)
        self.layout = None
        self.channel_layout = None
        self.valid = None

    def predicted_path(self, root):
        filename, _ = os.path.splitext(os.path.basename(self.path))
        return os.path.join(root, filename + "_predicted.npy")

    def load(self):
        """ Load the data sample in the .hdf5 file """

        self.data = torch.from_numpy(
            np.array(h5py.File(self.path, 'r')['array']))
        self.layout = 'THWC'
        self.channel_layout = "VSH"
        self.data = self.data.to(torch.uint8)
        self.valid = self.data.view(self.data.shape[0], -1).any(1)

    def temporal_slices(
            self, size: int, frames: List[int], valid: bool
    ) -> Union[Tuple[torch.tensor, torch.tensor], torch.tensor]:
        """ Temporal slice generator.

            Generates termporal slices of the data stream. For each frame_i
            in the frames list a slice along the time dimension is generated.
            The frames in the slice are [frame_i - size, frame_i).
            The returned tensors are views into the original data, i.e. they
            share underlying storage.

            Args:
                size: slice size
                frames: frame indices for which to generate slices
                valid: whether or not to return the list of valid frames in the
                       slice

            Yields:
                torch.tensor temporal slice.
        """

        time_axis = self.layout.find('T')
        for frame in frames:
            if valid:
                yield (self.data.narrow(time_axis, frame - size, size),
                       self.valid[frame - size:frame].clone())
            else:
                yield self.data.narrow(time_axis, frame - size, size)

    def selected_temporal_batches(self, batch_size: int, slice_size: int,
                                  frames: List[int]) -> torch.tensor:
        """ Temporal slice batch generator.

            Generates batches of termporal slices, of size slice_size, of the
            data stream.
            The first dimension of the output tensor is the "batch" dimension.
            The other dimensions are dependent on the current layout of the
            data. E.g: current data layout = [T, C, H, W] the shape of the
            result is [N, slice_size, C, H, W]. Where:
            N = batch_size or
            N = len(frames) - floor(len(frames) / batch_size) * batch_size for
            the last batch.

            Args:
                size: slice size
                frames: frame indices for which to generate slices
                valid: whether or not to return the list of valid frames in the
                       slice

            Yields:
                torch.tensor batch of temporal slices.
        """

        num_batches = (len(frames) + batch_size - 1) // batch_size
        for batch in range(num_batches):
            if batch < num_batches - 1:
                batch_frames = frames[batch * batch_size:(batch + 1) *
                                      batch_size]
            else:
                batch_frames = frames[batch * batch_size:]
            yield torch.stack(
                list(self.temporal_slices(slice_size, batch_frames,
                                          valid=False)))

    def random_temporal_batches(self, num_batches: int, batch_size: int,
                                slice_size: int) -> torch.tensor:
        """ Random temporal slice batch generator.

            Generates num_batches batches of termporal slices, of size
            slice_size, of the data stream.
            The first dimension of the output tensor is the "batch" dimension.
            The other dimensions are dependent on the current layout of the
            data. E.g: current data layout = [T, C, H, W], the shape of the
            result is [batch_size, slice_size, C, H, W].

            Args:
                size: slice size
                frames: frame indices for which to generate slices
                valid: whether or not to return the list of valid frames in the
                       slice

            Yields:
                torch.tensor batch of temporal slices.
        """

        num_frames = self.data.shape[self.layout.find('T')]
        for batch in range(num_batches):
            frames = random.sample(range(slice_size, num_frames - slice_size),
                                   batch_size)
            yield torch.stack(
                list(self.temporal_slices(slice_size, frames, valid=False)))

    def sliding_window_generator(self, width: int, stride: int,
                                 batch_size: int):
        """ Sliding window generator.

            Slice a [T, Hin, Win, Cin] tensor across the T dimension into
            slices of size 'width' and step equal to 'stride'. For each slice
            the T and C dimensions are concatenated along the T dimension.
            The resulting 3D tensors of shape [width * Cin, Hin, Win] are
            batched in batches of size batch_size.
            The generators yields a tensor of shape [N, Cout, Hout, Wout] where:
            N = batch_size or last_batch_size
            Cout = width * Cin
            Hout = Hin
            Wout = Win
            The last batch can have a smaller size since there are not enough
            sliding window sequences to fill it whole.

            Number of tensors generated is:
                ceil((floor((T - width) / stride) + 1) / batch_size)

            Args:
                width: size of the sliding window
                stride: stride of the sliding window
                batch_size: size of the batch.

            Yeilds:
                4D tensor with shape [N, Cout, Hout, Wout]
        """

        num_slices = math.floor((self.data.shape[0] - width) / stride) + 1
        num_batches = math.ceil(num_slices / batch_size)
        last_batch_size = num_slices - (num_slices // batch_size) * batch_size
        if last_batch_size == 0:
            last_batch_size = batch_size

        for batch_i in range(num_batches):
            if batch_i < (num_batches - 1):
                batch_i_size = batch_size
            else:
                batch_i_size = last_batch_size

            batch_shape = (batch_i_size, width * self.data.shape[3],
                           self.data.shape[1], self.data.shape[2])

            batch = torch.empty(batch_shape, dtype=self.data.dtype)
            for slice_i in range(batch.shape[0]):
                start = (batch_i * batch_size + slice_i) * stride
                stop = start + width
                axes = 0, 3, 1, 2
                batch[slice_i] = self.data[start:stop].permute(axes).flatten(
                    0, 1)
            yield batch

    def permute(self, layout: str):
        """ Change data layout. """

        self.data = self.data.permute([self.layout.find(ax) for ax in layout])
        self.layout = layout

    def select_channels(self, channels: List[str]):
        channels = set(channels)
        if not all([c[0].upper() in self.channel_layout for c in channels]):
            raise ValueError(
                f"Invalid channel to select. Data channels = {self.channel_layout}"
            )
        if len(channels) > 3:
            raise ValueError(f"Max 3 channels got {len(channels)} to select")
        elif len(channels) == 3:
            return
        else:
            keep = torch.tensor(
                [self.channel_layout.find(c[0].upper()) for c in channels],
                dtype=torch.long)
            c_axis = self.layout.find('C')
            self.data = torch.index_select(self.data, c_axis, keep)
            self.channel_layout = "".join([c[0].upper() for c in channels])

    class Transforms(object):

        class Permute(object):

            def __init__(self, to_layout: str):
                self.layout = to_layout

            def __call__(self, sample):
                sample.permute(self.layout)

        class SelectChannels(object):

            def __init__(self, channels: List[str]):
                self.channels = channels

            def __call__(self, sample):
                sample.select_channels(self.channels)


class Traffic4CastDataset(torch.utils.data.Dataset):
    """ Implementation of the pytorch Dataset. """

    def __init__(self,
                 root: str,
                 phase: str,
                 cities: List[str] = None,
                 transform: List[Callable] = None):
        """ Initializes the Traffic4CastDataset.

        Args:
            root (string): Path to the root data directory.
            phase (string): One of: 'training', 'test', 'validation'. Used to
                select between the data splits.
            cities ([string]): Name of the cities to use. Defaults to:
                ["Berlin", "Istanbul", "Moscow"].
            transform ([transforms]): Optional transforms to be applied on a
                sample.
        """

        self.transforms = [] if transform is None else transform

        if cities is None:
            cities = ["Berlin", "Istanbul", "Moscow"]
        self.files = {
            city: [
                f"{root}/{city}/{city}_{phase}/{file}"
                for file in sorted(os.listdir(f"{root}/{city}/{city}_{phase}/"))
            ] for city in cities
        }

        self.size = sum([len(files) for city, files in self.files.items()])

    def __len__(self):
        """ Gets length of the dataset. """
        return self.size

    def __getitem__(self, idx: int):
        for city, files in self.files.items():
            if idx >= len(files):
                idx = idx - len(files)
            else:
                stream = Traffic4CastSample(files[idx], city)
                stream.load()
                break

        for transform in self.transforms:
            if type(transform) in [
                    cls
                    for cls in Traffic4CastSample.Transforms.__dict__.values()
                    if inspect.isclass(cls)
            ]:
                transform(stream)
            else:
                stream.data = transform(stream.data)

        return stream

    @classmethod
    def collate_list(cls, samples_list: List[Traffic4CastSample]
                    ) -> List[Traffic4CastSample]:
        """ Collates a list of Traffic4CastSample.

            Args:
                samples_list: List of Traffic4CastSample samples.

            Return:
                List[Traffic4CastSample]
        """

        return samples_list
