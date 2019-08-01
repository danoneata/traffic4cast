import argparse
import os
import pdb

import numpy as np

import torch

from tabulate import tabulate

from models import MODELS

from src.dataset import Traffic4CastDataset

from utils import cache

ROOT = os.environ.get("ROOT", "data")
CHANNELS = ["volume", "speed", "heading"]
CITIES = ["Berlin", "Istanbul", "Moscow"]

START_FRAMES = [30, 69, 126, 186, 234]
N_FRAMES = 3  # Predict this many frames into the future
SUBMISSION_FRAMES = [s + i for s in START_FRAMES for i in range(N_FRAMES)]


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
    parser.add_argument("-v",
                        "--verbose",
                        action="count",
                        help="verbosity level")
    args = parser.parse_args()

    if args.verbose:
        print(args)

    Model = MODELS[args.model]
    model = Model()

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

    dataset = Traffic4CastDataset(ROOT, args.split, cities=[args.city])
    nr_days = len(dataset)

    def predict(sample):
        frame_preds = [model.predict(sample, frame) for frame in START_FRAMES]
        return np.stack(frame_preds)

    # Cache predictions to a specified path
    to_overwrite = args.overwrite
    cached_predict = lambda path, *args: cache(predict, path, to_overwrite,
                                               *args)

    dirname = os.path.join("output", "predictions", args.split, args.model,
                           args.city)
    os.makedirs(dirname, exist_ok=True)

    def get_path_pr(date):
        filename = date.strftime('%Y-%m-%d') + ".npy"
        return os.path.join(dirname, filename)

    to_str = lambda v: f"{v:7.5f}"

    if args.split == "validation":

        errors = []

        for i in range(nr_days):
            sample = dataset[i]
            data = sample.data.numpy()

            gt = np.stack([data[s:s + N_FRAMES] for s in START_FRAMES]) / 255.0
            pr = cached_predict(get_path_pr(sample.date), sample) / 255.0

            mse = np.mean((gt - pr)**2, axis=(0, 1, 2, 3))
            errors.append(mse)

            if args.verbose:
                print(sample.date, "|", " | ".join(to_str(e) for e in mse))

        errors = np.vstack(errors)
        table = [[args.model] +
                 [to_str(v) for v in errors.mean(axis=0).tolist()] +
                 [to_str(errors.mean())]]
        headers = ["model"] + CHANNELS + ["mean"]
        print(tabulate(table, headers=headers, tablefmt="github"))

    elif args.split == "test":
        for i in range(nr_days):
            sample = dataset[i]
            cached_predict(get_path_pr(sample.date), sample)
            if args.verbose:
                print(sample.date)


if __name__ == "__main__":
    main()
