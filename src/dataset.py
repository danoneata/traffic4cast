import os
import operator
import functools

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class Traffic4CastDataset(Dataset):
    def __init__(self, root_dir, phase, cities=["Berlin", "Istanbul", "Moscow"],
            transform=[]):
        """
        Args:
            root_dir (string): Path to the root data directory.
            phase (string): One of: 'training', 'test', 'validation'. Used to
                select between the data splits.
            cities ([string]): Name of the cities to use. Defaults to:
                ["Berlin", "Istanbul", "Moscow"].
            transform ([transforms]): Optional transforms to be applied on a
                sample.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.cities = cities
        self.files = [f"{root_dir}/{city}/{city}_{phase}/{file}"
                for city in self.cities
                for file in os.listdir(f"{root_dir}/{city}/{city}_{phase}/")]
        self.len = len(self.files)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = torch.from_numpy(np.array(h5py.File(self.files[idx], 'r')['array']))
        for transform in self.transform:
            img = transform(img)

        return img

