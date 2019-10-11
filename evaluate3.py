import argparse
import copy
import json
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

import submission_write

from constants import *

from train import filter_dict

from evaluate import get_prediction_folder


def to_uint8(t):
    t = torch.min(t, torch.ones(t.shape).to(t.device))
    t = torch.max(t, torch.zeros(t.shape).to(t.device))
    t = t * 255
    t = t.type(torch.uint8)
    return t


def round_torch(t):
    t = to_uint8(t)
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
    parser.add_argument(
        "--hyper-params",
        required=False,
        help=(
            "path to JSON file containing hyper-parameter configuration "
            "(over-writes other hyper-parameters passed through the "
            "command line)."),
    )
    parser.add_argument("-v",
                        "--verbose",
                        action="count",
                        help="verbosity level")
    args = parser.parse_args()
    args.channels.sort(
        key=lambda x: src.dataset.Traffic4CastSample.channel_to_index[x])

    if args.verbose:
        print(args)

    if args.hyper_params and os.path.exists(args.hyper_params):
        with open(args.hyper_params, "r") as f:
            hyper_params = json.load(f)
    else:
        hyper_params = {}

    Model = MODELS[args.model]
    model = Model(**filter_dict(hyper_params, "model"))

    loss = nn.MSELoss()

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location="cuda:0"))
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

    if args.model_path:
        model_name, _ = os.path.splitext(os.path.basename(args.model_path))
    else:
        model_name = args.model

    to_str = lambda v: f"{v:9.7f}"

    losses = []
    device = "cuda"
    prepare_batch = model.ignite_batch
    non_blocking = False
    output_transform = lambda x, y, y_pred: (y_pred, y,)
    slice_size = model.past + model.future

    dirname = get_prediction_folder(args.split, model_name, args.city)
    os.makedirs(dirname, exist_ok=True)

    if device:
        model.to(device)

    def _inference(batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    def get_path(date):
        return os.path.join(dirname, src.dataset.date_to_path(date))

    for batch in ignite_selected(loader, slice_size=slice_size):
        output = _inference(batch)
        curr_loss = loss(round_torch(output[0]), output[1]).item()
        losses.append(curr_loss)
        if not os.path.exists(get_path(batch[1])) or args.overwrite:
            path = get_path(batch[1])
            data = to_uint8(output[0])
            data = data.view(5, 3, 3, 495, 436)
            data = data.permute(0, 1, 3, 4, 2)
            data = data.cpu().numpy().astype(np.uint8)
            submission_write.write_data(data, path)
        if args.verbose:
            print(to_str(curr_loss))
        # sys.exit()

    print(to_str(np.mean(losses)))


if __name__ == "__main__":
    main()
