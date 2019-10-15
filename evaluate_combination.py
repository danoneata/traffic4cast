import argparse
import copy
import h5py
import json
import os
import pdb
import random
import sys
import time

from itertools import product

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn

from torch.nn.functional import pairwise_distance
from torch.utils.data import (
    DataLoader,
    Subset,
)

from tabulate import tabulate

from models import MODELS
from models.nn import ignite_selected

import src.dataset

from utils import cache

import submission_write

from constants import *

from train import filter_dict

from evaluate import (
    ROOT,
    get_prediction_folder,
)

from evaluate3 import to_uint8

from notebook_utils import load_data as load_data_full


L = float(os.environ.get("LAMBDA", 0.8))
K = int(os.environ.get("K", 3))

CHANNELS = ["Volume", "Speed", "Heading"]
MODEL_KNN = "knn-{}".format(K)
MODEL_NAME = "nero+knn"
DEVICE = "cuda:0"
P = 12  # No. of frames from the past
F = 3  # No. of future frames
C = 3
H = 495
W = 436


SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)


def load_h5(path):
    result = np.array(h5py.File(path, 'r')['array'])
    result = result.astype(np.uint8)
    result = result.astype(np.float32) / 255.0
    result = torch.from_numpy(result)
    result = result.permute(0, 1, 4, 2, 3)
    result = result.to(DEVICE)
    result = result.view(5, F * C, H, W)
    return result


def model(batch, get_path):
    _, date, _ = batch
    data1 = load_h5(get_path("nero", date))
    data2 = load_h5(get_path(MODEL_KNN, date))
    LL = torch.ones(5, F, C)
    LL[:, 0] = L
    LL = LL.view(5, F * C, 1, 1).to(DEVICE)
    # LL = L
    return LL * data1 + (1 - LL) * data2


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Nero + kNN model combination")
    parser.add_argument("-s",
                        "--split",
                        default="validation",
                        choices={"validation", "test"},
                        help="data split (for 'test' it only predicts)")
    parser.add_argument("-c",
                        "--city",
                        required=True,
                        choices=CITIES,
                        help="which city to evaluate")
    parser.add_argument("--overwrite",
                        default=False,
                        action="store_true",
                        help="overwrite existing predictions if they exist")
    parser.add_argument("-v",
                        "--verbose",
                        action="count",
                        help="verbosity level")
    args = parser.parse_args()

    if args.verbose:
        print(args)

    transforms = [
        lambda x: x.float(),
        lambda x: x / 255,
        src.dataset.Traffic4CastSample.Transforms.Permute("TCHW"),
        src.dataset.Traffic4CastSample.Transforms.SelectChannels(CHANNELS),
    ]
    dataset = src.dataset.Traffic4CastDataset(ROOT, args.split, [args.city], transforms)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=src.dataset.Traffic4CastDataset.collate_list)

    to_str = lambda v: f"{v:10.7f}"

    losses = []
    device = "cuda"
    non_blocking = False
    output_transform = lambda x, y, y_pred: (y_pred, y,)
    slice_size = P + F

    dirname = get_prediction_folder(args.split, MODEL_NAME, args.city)
    os.makedirs(dirname, exist_ok=True)

    loss = nn.MSELoss()

    def prepare_batch(batch, device, non_blocking):
        batch1, date, frames = batch
        batch1 = batch1.to(device)
        batch1 = batch1.reshape(batch1.shape[0], -1, batch1.shape[3], batch1.shape[4])
        inp = batch1[:, :P * C], date, frames
        tgt = batch1[:, P * C:]
        return inp, tgt

    def _inference(batch):
        with torch.no_grad():
            x, y = prepare_batch(batch, device=DEVICE, non_blocking=non_blocking)
            y_pred = model(x, get_path1)
            return output_transform(x, y, y_pred)

    def get_path(date):
        return os.path.join(dirname, src.dataset.date_to_path(date))

    def get_path1(model_name, date):
        str_channels = "_".join(CHANNELS)
        if model_name == "nero" and args.city == "Berlin":
            model_name = "nero-8-32_" + str_channels + "_" + args.city
        elif model_name == "nero" and args.city == "Moscow":
            model_name = "nero-8-64_" + str_channels + "_" + args.city
        elif model_name == "nero" and args.city == "Istanbul":
            model_name = "nero-4-64_" + str_channels + "_" + args.city
        dirname = get_prediction_folder(args.split, model_name, args.city)
        return os.path.join(dirname, src.dataset.date_to_path(date))

    for batch in ignite_selected(loader, slice_size=slice_size):
        output = _inference(batch)
        curr_loss = loss(output[0], output[1]).item()
        losses.append(curr_loss)
        if not os.path.exists(get_path(batch[1])) or args.overwrite:
            path = get_path(batch[1])
            data = to_uint8(output[0])
            data = data.view(5, 3, 3, 495, 436)
            data = data.permute(0, 1, 3, 4, 2)  # Move channels at the end
            data = data.cpu().numpy().astype(np.uint8)
            submission_write.write_data(data, path)
        if args.verbose:
            diff = ((output[0] - output[1]).view(5, 3, 3, H, W) ** 2)
            print("T", diff[:, 0].mean(dim=(1, 2, 3)))
            print("F", diff.mean(dim=(0, 2, 3, 4)))
            print(to_str(curr_loss))

        # sys.exit()

    print(to_str(np.mean(losses)))


if __name__ == "__main__":
    main()
