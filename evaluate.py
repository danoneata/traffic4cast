import argparse
import os
import pdb

import numpy as np

from tabulate import tabulate

from models import MODELS

from src.dataset import Traffic4CastDataset

from utils import cache

ROOT = os.environ.get("ROOT", "data")
CHANNELS = ["volume", "speed", "heading"]
CITIES = ["Berlin", "Istanbul", "Moscow"]

START_FRAMES = [30, 69, 126, 186, 234]
N_FRAMES = 3  # Predict this many frames into the future
EVALUATION_FRAMES = [s + i for s in START_FRAMES for i in range(N_FRAMES)]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a given model")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        required=True,
                        choices=MODELS,
                        help="which model to use")
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
    # TODO Load model

    dataset = Traffic4CastDataset(ROOT, args.split, cities=[args.city])
    nr_days = len(dataset)

    def predict(sample):
        frame_preds = [model.predict(sample, frame) for frame in START_FRAMES]
        return np.concatenate(frame_preds, axis=0)

    # Cache predictions to a specified path
    cached_predict = lambda path, *args: cache(predict, path, *args)

    def get_path_pr(date):
        dirname = os.path.join("output", "predictions", args.split, args.model, args.city)
        filename = date.strftime('%Y-%m-%d') + ".npy"
        os.makedirs(dirname, exist_ok=True)
        return os.path.join(dirname, filename)

    if args.split == "validation":

        errors = []

        for i in range(nr_days):
            sample = dataset[i]
            data = sample.data.numpy().astype('float')

            gt = data[EVALUATION_FRAMES]
            pr = cached_predict(get_path_pr(sample.date), sample)

            sq_err = np.mean((gt - pr)**2, axis=(0, 1, 2))
            errors.append(sq_err)

            if args.verbose:
                print(sample.date, "|", " | ".join(f"{e:7.3f}" for e in sq_err))

        table = [np.vstack(errors).mean(axis=0).tolist()]
        print(tabulate(table, headers=CHANNELS, tablefmt="github"))

    elif args.split == "test":
        for i in range(nr_days):
            sample = dataset[i]
            cached_predict(get_path_pr(sample.date), sample)
            if args.verbose:
                print(sample.date)


if __name__ == "__main__":
    main()
