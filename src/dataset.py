from typing import List, Callable
import os
import datetime

import numpy as np
import h5py
import torch
import torch.utils.data


class Traffic4CastSmaple(object):
    """ Traffic4cast data wrapper.

        Attributes:
            path (str): Path to the .hdf5 data file.
            data (torch.tensor): 4D torch.tensor for the sample data.
            city (str): City where data sample was collected.
            date (datetime): Date when the data sample was collected.
    """

    def __init__(self, city: str, path: str):
        """ Initializes the Traffic4CastSmaple data sample

            Args:
                city: Name of they city where the data sample was collected.
                path: Path to the .hdf5 data file.
        """

        self.path = path
        self.data = None
        self.city = city
        self.date = datetime.datetime.strptime(
            os.path.basename(path).split('_')[0], '%Y%m%d')

    def load(self):
        """ Load the data sample in the .hdf5 file """

        self.data = torch.from_numpy(
            np.array(h5py.File(self.path, 'r')['array']))


class Traffic4CastDataset(torch.utils.data.Dataset):
    """ Implementation of the pytorch Dataset. """

    def __init__(self,
                 root_dir: str,
                 phase: str,
                 cities: List[str] = ["Berlin", "Istanbul", "Moscow"],
                 transform: List[Callable] = []):
        """ Initializes the Traffic4CastDataset.

        Args:
            root_dir (string): Path to the root data directory.
            phase (string): One of: 'training', 'test', 'validation'. Used to
                select between the data splits.
            cities ([string]): Name of the cities to use. Defaults to:
                ["Berlin", "Istanbul", "Moscow"].
            transform ([transforms]): Optional transforms to be applied on a
                sample.
        """

        self.transforms = transform
        self.files = {
            city: [
                f"{root_dir}/{city}/{city}_{phase}/{file}"
                for file in sorted(
                    os.listdir(f"{root_dir}/{city}/{city}_{phase}/"))
            ]
            for city in cities
        }

        self.len = sum([len(files) for city, files in self.files.items()])

    def __len__(self):
        """ Gets length of the dataset. """
        return self.len

    def __getitem__(self, idx: int):
        for city, files in self.files.items():
            if idx >= len(files):
                idx = idx - len(files)
            else:
                stream = Traffic4CastSmaple(city, files[idx])
                stream.load()
                break

        for transform in self.transforms:
            stream.data = transform(stream.data)

        return stream

    @classmethod
    def collate_list(cls, samples_list: List[Traffic4CastSmaple]
                    ) -> List[Traffic4CastSmaple]:
        """ Collates a list of Traffic4CastSmaple.

            Args:
                samples_list: List of Traffic4CastSmaple samples.

            Return:
                List[Traffic4CastSmaple]
        """

        return samples_list
