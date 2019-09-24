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

    def predict(sample, channel_transforms):
        predictions = np.zeros(EVALUATION_SHAPE)
        for transform in channel_transforms:
            s = copy.deepcopy(sample)
            transform(s)
            for f, p in model.predict(SUBMISSION_FRAMES[args.city], s).items():
                for c_i, c in enumerate(transform.channels):
                    predictions[
                        SUBMISSION_FRAMES[args.city].index(f),
                        :,
                        :,
                        src.dataset.Traffic4CastSample.channel_to_index[c]
                    ] = p[c_i]

        predictions = predictions * 255.0
        predictions = predictions.reshape(SUBMISSION_SHAPE)
        return predictions

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

    evaluator = engine.create_supervised_evaluator(
        model,
        metrics={'loss': ignite.metrics.Loss(loss)},
        device="cuda",
        prepare_batch=model.ignite_batch,
    )

    slice_size = model.past + model.future
    evaluator.run(ignite_selected(loader, slice_size=slice_size))
    print(to_str(evaluator.state.metrics["loss"]))


if __name__ == "__main__":
    main()
