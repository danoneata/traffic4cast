import argparse
import copy
import os
import pdb
import sys

import numpy as np

import torch

from tabulate import tabulate

from models import MODELS

import src.dataset

from utils import cache

ROOT = os.environ.get("ROOT", "data")
CHANNELS = ["volume", "speed", "heading"]
CITIES = ["Berlin", "Istanbul", "Moscow"]

BERLIN_START_FRAMES = [30, 69, 126, 186, 234]
ISTANBUL_START_FRAMES = [57, 114, 174, 222, 258]
MOSCOW_START_FRAMES = ISTANBUL_START_FRAMES

N_FRAMES = 3  # Predict this many frames into the future
SUBMISSION_FRAMES = {
    city: [s + i for s in start_frames for i in range(N_FRAMES)]
    for start_frames, city in
    zip([BERLIN_START_FRAMES, ISTANBUL_START_FRAMES, MOSCOW_START_FRAMES],
        CITIES)
}

SUBMISSION_SHAPE = (5 * N_FRAMES, 495, 436, 3)


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
                        default="Volume,Speed,Heading",
                        help="List of channels to predict")
    parser.add_argument("-v",
                        "--verbose",
                        action="count",
                        help="verbosity level")
    args = parser.parse_args()
    args.channels = args.channels.split(',')
    args.channels.sort(
        key=lambda x: src.dataset.Traffic4CastSample.channel_to_index[x])

    if args.verbose:
        print(args)

    Model = MODELS[args.model]
    model = Model()

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

    if model.num_channels == len(args.channels):
        if (model.num_channels != 3):
            print(f"WARNING: Model predicts {model.num_channels} and "
                  f"{args.channels} were selected. Unselected channels will be "
                  "predicted as 0.")
        selected_channels_transforms = [
            src.dataset.Traffic4CastSample.Transforms.SelectChannels(
                args.channels)
        ]
    elif model.num_channels == 1:
        print(f"WARNING: Model predicts {model.num_channels} channel but "
              f"channels {args.channels} were selected. Iteration mode enabled."
              "Unselected channels will be predicted as 0.")
        selected_channels_transforms = [
            src.dataset.Traffic4CastSample.Transforms.SelectChannels([c])
            for c in args.channels
        ]
    else:
        print(f"ERROR: Model to channels missmatch. Model can predict "
              f"{model.num_channels} channels. {len(args.channels)} were "
              "selected.")
        sys.exit(1)

    transforms = [
        lambda x: x.float(),
        lambda x: x / 255,
        src.dataset.Traffic4CastSample.Transforms.Permute("TCHW"),
    ]
    dataset = src.dataset.Traffic4CastDataset(ROOT, args.split, [args.city],
                                              transforms)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=src.dataset.Traffic4CastDataset.collate_list)

    def predict(sample, channel_transforms):
        predictions = np.zeros(SUBMISSION_SHAPE)
        for transform in channel_transforms:
            s = copy.deepcopy(sample)
            transform(s)
            for f, p in model.predict(SUBMISSION_FRAMES[args.city], s).items():
                for c_i, c in enumerate(transform.channels):
                    predictions[SUBMISSION_FRAMES[args.city].
                                index(f), :, :, src.dataset.Traffic4CastSample.
                                channel_to_index[c]] = p[c_i]

        predictions = predictions * 255.0
        return predictions

    # Cache predictions to a specified path
    to_overwrite = args.overwrite
    cached_predict = lambda path, *args: cache(predict, path, to_overwrite,
                                               *args)

    dirname = os.path.join("output", "predictions", args.split, args.model,
                           args.city)
    os.makedirs(dirname, exist_ok=True)

    to_str = lambda v: f"{v:7.5f}"

    errors = []
    for sample in loader:
        sample = sample[0]
        predictions = cached_predict(sample.predicted_path(dirname), sample,
                                     selected_channels_transforms) / 255.0
        if args.split == "validation":
            sample.permute('THWC')
            gt = sample.data.index_select(
                0, torch.tensor(SUBMISSION_FRAMES[args.city],
                                dtype=torch.long)).numpy()
            mse = np.mean((gt - predictions)**2, axis=(0, 1, 2))
            errors.append(mse)
            if args.verbose:
                print(sample.date, "|", " | ".join(to_str(e) for e in mse))
        elif args.split == "test":
            if args.verbose:
                print(sample.date)

    if args.split == "validation":
        errors = np.vstack(errors)
        table = [[args.model] +
                 [to_str(v) for v in errors.mean(axis=0).tolist()] +
                 [to_str(errors.mean())]]
        headers = ["model"] + CHANNELS + ["mean"]
        print(tabulate(table, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
