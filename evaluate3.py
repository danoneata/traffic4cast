import argparse
import copy
import os
import pdb
import sys

import numpy as np

import torch
import torch.nn as nn

from tabulate import tabulate

from models import MODELS

import src.dataset

from utils import cache

import ignite
import ignite.engine as engine

from models.nn import ignite_selected

from constants import *


def get_prediction_folder(split, model_name, city):
    return os.path.join("output", "predictions", split, model_name, city)


def round_torch(t):
    t = torch.min(t, torch.ones(t.shape).to(t.device))
    t = torch.max(t, torch.zeros(t.shape).to(t.device))
    t = t * 255
    t = t.type(torch.uint8)
    t = t.type(torch.float) / 255.0
    return t


def main():
    parser = argparse.ArgumentParser(description="Evaluate a given model")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        required=True,
                        choices=MODELS,
                        help="which model to use")
    parser.add_argument("-p",
                        "--model-path",
                        type=str,
                        help="path to the saved model")
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
    parser.add_argument("--channels",
                        nargs='+',
                        help="List of channels to predict")
    parser.add_argument("-v",
                        "--verbose",
                        action="count",
                        help="verbosity level")
    args = parser.parse_args()
    args.channels.sort(
        key=lambda x: src.dataset.Traffic4CastSample.channel_to_index[x])

    if args.verbose:
        print(args)

    Model = MODELS[args.model]
    model = Model()

    loss = nn.MSELoss()

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

    transforms = [
        lambda x: x.float(),
        lambda x: x / 255,
        src.dataset.Traffic4CastSample.Transforms.Permute("TCHW"),
        src.dataset.Traffic4CastSample.Transforms.SelectChannels(args.channels),
    ]
    dataset = src.dataset.Traffic4CastDataset(ROOT, args.split, [args.city], transforms)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=src.dataset.Traffic4CastDataset.collate_list)

    # Cache predictions to a specified path
    # to_overwrite = args.overwrite
    # cached_predict = lambda path, *args: cache(predict, path, to_overwrite, *args)

    if args.model_path:
        model_name, _ = os.path.splitext(os.path.basename(args.model_path))
    else:
        model_name = args.model

    # dirname = get_prediction_folder(args.split, model_name, args.city)
    # os.makedirs(dirname, exist_ok=True)

    to_str = lambda v: f"{v:7.5f}"

    losses = []
    device = "cuda"
    prepare_batch = model.ignite_batch
    non_blocking = False
    output_transform = lambda x, y, y_pred: (y_pred, y,)
    slice_size = model.past + model.future

    if device:
        model.to(device)

    def _inference(batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    for batch in ignite_selected(loader, slice_size=slice_size):
        output = _inference(batch)
        curr_loss = loss(round_torch(output[0]), output[1]).item()
        losses.append(curr_loss)
        print(curr_loss)

    print(to_str(np.mean(losses)))


if __name__ == "__main__":
    main()
