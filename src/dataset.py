from typing import List, Callable
import datetime
import os
import math

import numpy as np
import h5py
import torch
import torch.utils.data


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
    """
    time_step_delta = datetime.timedelta(minutes=5)
    channel_label = {0: "Volume", 1: "Speed", 2: "Heading"}

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

    def load(self):
        """ Load the data sample in the .hdf5 file """

        self.data = torch.from_numpy(
            np.array(h5py.File(self.path, 'r')['array']))

    def sliding_window_generator(self, width: int, stride: int,
                                 batch_size: int):
        """ Sliding window generator.

            Slice a [T, Hin, Win, Cin] tensor across the T dimension into
            slices of size 'width' and step equal to 'stride'. For each slice
            the T and C dimensions are concatenated along the T dimension.
            The resulting 3D tensors of shape [width * Cin, Hin, Win] are
            batched in batches of size batch_size.
            The generators yields a tensor of shape [N, Cout, Hout, Wout] where:
            N = batch_size
            Cout = width * Cin
            Hout = Hin
            Wout = Win

            Number of tensors generated is:
                floor((floor((T - width) / stride) + 1) / batch_size)

            Args:
                width: size of the sliding window
                stride: stride of the sliding window
                batch_size: size of the batch.

            Yeilds:
                4D tensor with shape [N, Cout, Hout, Wout]
        """

        num_batches = (math.floor(
            (self.data.shape[0] - width) / stride) + 1) // batch_size
        batch_shape = (batch_size, width * self.data.shape[3],
                       self.data.shape[1], self.data.shape[2])
        for batch_i in range(num_batches):
            batch = torch.empty(batch_shape, dtype=self.data.dtype)
            for slice_i in range(batch_size):
                start = (batch_i * batch_size + slice_i) * stride
                stop = start + width
                axes = 0, 3, 1, 2
                batch[slice_i] = self.data[start:stop].permute(axes).flatten(
                    0, 1)
            yield batch


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
