import argparse
import os
import pdb

from itertools import product

import numpy as np

from tabulate import tabulate

from models import MODELS

from src.dataset import (
    Traffic4CastDataset,
    path_to_date,
)

from utils import (
    cache,
    day_frame_to_date,
)

ROOT = os.environ.get("ROOT", "data")
CHANNELS = ["volume", "speed", "heading"]
CITIES = ["Berlin", "Istanbul", "Moscow"]

START_FRAMES = [30, 69, 126, 186, 234]
NR_FRAMES = 3  # Predict this many frames into the future


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
    model = Model(city=args.city)
    # TODO Load model

    dataset = Traffic4CastDataset(ROOT, args.split, cities=[args.city])
    nr_days = len(dataset)

    # Cache predictions to a specified path
    cached_predict = lambda path, *args: cache(model.predict, path, *args)

    def get_path_pr(date):
        dirname = os.path.join("output", "predictions", args.model, args.city,
                               args.split)
        filename = date.strftime('%Y-%m-%d_%H-%M') + f"_n-frames-{NR_FRAMES}.npy"
        os.makedirs(dirname, exist_ok=True)
        return os.path.join(dirname, filename)

    if args.split == "validation":

        errors = []

        for i, frame in product(range(nr_days), START_FRAMES):

            sample = dataset[i]
            date = day_frame_to_date(sample.date, frame)

            gt = dataset[i].data[frame: frame + NR_FRAMES].numpy()
            pr = cached_predict(get_path_pr(date), date, NR_FRAMES)

            sq_err = np.mean((gt - pr)**2, axis=(0, 1, 2))
            errors.append(sq_err)

            if args.verbose:
                print(date, "| 3 frames |",
                      " | ".join(f"{e:7.2f}" for e in sq_err))

        table = [np.vstack(errors).mean(axis=0).tolist()]
        print(tabulate(table, headers=CHANNELS, tablefmt="github"))

    elif args.split == "testing":
        for path, frame in product(dataset.files, START_FRAMES):
            date = day_frame_to_date(path_to_date(path), frame)
            cached_predict(get_path_pr, model, date)


if __name__ == "__main__":
    main()
