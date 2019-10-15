import argparse
import copy
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


CHANNELS = ["Volume", "Speed", "Heading"]
DEVICE = "cuda:0"
P = 12  # No. of frames from the past
F = 3  # No. of future frames
C = 3
H = 495
W = 436

DELTA = int(os.environ.get("DELTA", 15))
STRIDE = int(os.environ.get("STRIDE", 15))

K = int(os.environ.get("K", 7))
MODEL_NAME = f"knn-{K}-{DELTA}-{STRIDE}"

SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)


def model_knn(batch, tr_data, delta=DELTA, stride=STRIDE, verbose=False):

    start_time = time.time()

    xs, date, _ = batch
    N, PxC, H1, W1 = xs.shape
    xs = xs.view(N, P, C, H, W)

    # Some checks
    assert PxC == P * C
    assert H1 == H
    assert W1 == W

    pred = torch.zeros(N, F, C, H, W).to(DEVICE)
    counts = torch.zeros(N, F, C, H, W).to(DEVICE)

    def get_slice(i, max_val):
        return slice(max(0, i - delta), min(i + delta + 1, max_val))

    def crop(x, loc):
        i, j = loc
        slice_i = get_slice(i, H)
        slice_j = get_slice(j, W)
        return x[:, slice_i, slice_j]

    if verbose:
        print(date)

    for n, x in enumerate(xs):

        if verbose:
            print(n, time.time() - start_time)
            start_time = time.time()

        for i, j in product(range(0, H + stride, stride), range(0, W + stride, stride)):

            loc = (i, j)

            # Query
            q = crop(x[-1], loc)
            q = q.flatten()
            q = q.unsqueeze(0)

            # Data
            c = crop(tr_data, loc)
            c = torch.from_numpy(c).float() / 255.0
            c = c.permute(0, 3, 1, 2).to(DEVICE)
            d = c.view(-1, q.shape[1])
            dist = pairwise_distance(d, q)
            idxs = dist.argsort()  # TODO More than one neighbor?
            idxs = idxs[:K]

            idxs = idxs + 1
            idxs[idxs >= len(tr_data)] = len(tr_data) - 1

            # Selected values
            v = c[idxs]
            v = v.mean(dim=0, keepdim=True)
            v = v.repeat(F, 1, 1, 1)

            # Handle special cases, when `idx` is at the end of the training data
            # if len(v) == 0:
            #     v = c[idx: idx + 1].repeat(F, 1, 1, 1)
            # elif len(v) < 3:
            #     v = c[idx + 1: idx + 2].repeat(F, 1, 1, 1)

            slice_i = get_slice(i, H)
            slice_j = get_slice(j, W)

            # Update predictions
            pred[n, :, :, slice_i, slice_j] += v
            counts[n, :, :, slice_i, slice_j] += torch.ones(v.shape).to(DEVICE)

    pred = pred / counts
    pred = pred.view(N, F * C, H, W)
    return pred


def main():
    parser = argparse.ArgumentParser(description="Evaluate the kNN model")
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

    to_str = lambda v: f"{v:9.7f}"

    losses = []
    device = "cuda"
    non_blocking = False
    output_transform = lambda x, y, y_pred: (y_pred, y,)
    slice_size = P + F

    dirname = get_prediction_folder(args.split, MODEL_NAME, args.city)
    os.makedirs(dirname, exist_ok=True)

    tr_dataset = src.dataset.Traffic4CastDataset(ROOT, "training", cities=[args.city])
    tr_dataset = Subset(tr_dataset, indices=sorted(random.sample(range(len(tr_dataset)), 35)))
    tr_data, _ = load_data_full(tr_dataset)
    D, T, _, _, _ = tr_data.shape
    tr_data = tr_data.reshape(D * T, H, W, C)

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
            y_pred = model_knn(x, tr_data=tr_data, verbose=args.verbose)
            return output_transform(x, y, y_pred)

    def get_path(date):
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

    print(to_str(np.mean(losses)))


if __name__ == "__main__":
    main()
