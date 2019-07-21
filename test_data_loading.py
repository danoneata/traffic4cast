#!/usr/bin/env python3
""" Example usage of the Traffic4CastDataset.

This module demonstrates how the Traffic4CastDataset can be used in conjunction
with a pytroch DataLoader instance.

In the example we will use a simple pytorch NN model, SimpleNet, that. Given
a batch of shape [batch_size, 12 frames * 3 channels, width, height] will
predict the next 3 frames and return a tensor of shape
[batch_size, 3 frames * 3 channels, width, and height].
The output is used to be compared with the ground truth.

"""

import argparse
import datetime
import sys

import torch
import torch.nn as nn
import torch.nn.functional as nn_func

import src.dataset
import src.visualization

import pdb


class SimpleNet(nn.Module):

    def __init__(self, input_channels: int):
        super(SimpleNet, self).__init__()

        conv_size = 5
        output_channels = 3 * 3  # 3 frames * (volume, speed and heading)
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              conv_size,
                              padding=2)

    def forward(self, tensor):
        return nn_func.relu(self.conv(tensor))


def main():
    # rescale the input values to [0, 1] interval and set the tensor type to
    # torch.float
    transforms = [lambda input: input.float() / 255]
    dataset = src.dataset.Traffic4CastDataset("./data",
                                              "training",
                                              transform=transforms)

    # In the loader creation look out for:
    # batch_size = number of sample files to be read for 1 batch. Files are read
    #   from the disk which is slow process. A bigger batch size might be useful
    #   for application speed up.
    # num_workers = number of threads used for file reading and aplication of
    #   transforms.
    # collate_fn = function used for collating batch_size number of
    #   Traffic4CastSample objects into a batch.
    #   src.dataset.Traffic4CastDataset.collate_list returns a list of
    #   batch_size length.
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=src.dataset.Traffic4CastDataset.collate_list)

    window_width = 15
    window_stride = 15
    mini_batch_size = 16
    simple_net = SimpleNet((window_width - 3) * 3)

    for batch in loader:
        for sample in batch:
            for minibatch in sample.sliding_window_generator(
                    window_width, window_stride, mini_batch_size):

                output = simple_net(minibatch[:, 0:(window_width - 3) *
                                              3, :, :])
                mse = torch.mean(
                    ((minibatch[:, ((window_width - 3) * 3):, :, :] -
                      output)**2).flatten(1, 3))
                print(f"MSE = {mse}")


if __name__ == "__main__":
    main()
