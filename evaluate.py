import argparse
import os
import pdb

from datetime import datetime, timedelta

from functools import partial

import h5py

import numpy as np

from typing import Callable, List

from tabulate import tabulate

from models import MODELS

from utils import (
    cache,
    date_to_day_frame,
    day_frame_to_date,
)

ROOT = os.environ.get("ROOT", "data")
CHANNELS = ["volume", "speed", "heading"]
CITIES = ["Berlin", "Istanbul", "Moscow"]

START_FRAMES = [30, 69, 126, 186, 234]
FRAMES_TO_PREDICT = [s + d for s in START_FRAMES for d in range(4)]


def get_path(root, city, phase, date):
    filename = date.strftime('%Y%m%d') + "_100m_bins.h5"
    return f"{root}/{city}/{city}_{phase}/{filename}"


def load_groundtruth(get_path: Callable[[datetime], str],
                     date: datetime,
                     n_frames: int = 1) -> np.ndarray:
    """Size of return (F, H, W, C)"""
    day, frame = date_to_day_frame(date)
    data = np.array(h5py.File(get_path(date), 'r')['array'])
    return data[frame:frame + n_frames]


def path_to_date(path: str) -> datetime:
    return datetime.strptime(os.path.basename(path).split('_')[0], '%Y%m%d')


def get_days(root: str, city: str, phase: str) -> List[datetime]:
    files = sorted(os.listdir(f"{root}/{city}/{city}_{phase}/"))
    dates = [path_to_date(f) for f in files]
    return dates


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

    days = get_days(ROOT, args.city, args.split)
    dates = [
        day_frame_to_date(day, frame) for day in days for frame in START_FRAMES
    ]

    Model = MODELS[args.model]
    model = Model(city=args.city)
    # TODO Load model

    # Cache predictions to a specified path
    cached_predict = lambda path, *args: cache(path)(model.predict)(*args)

    get_path_gt = partial(get_path, ROOT, args.city, args.split)
    def get_path_pr(date):
        dirname = os.path.join("output", "predictions", args.model, args.city, args.split)
        filename = date.strftime('%Y-%m-%d_%H-%M') + "_n-frames-3.npy"
        os.makedirs(dirname, exist_ok=True)
        return os.path.join(dirname, filename)

    if args.split == "validation":
        errors = []
        for date in dates:

            gt = load_groundtruth(get_path_gt, date, n_frames=3)
            pr = cached_predict(get_path_pr(date), date, 3)

            sq_err = np.mean((gt - pr)**2, axis=(0, 1, 2))
            errors.append(sq_err)

            if args.verbose:
                print(date, "| 3 frames |", " | ".join(f"{e:7.2f}" for e in sq_err))

        table = [np.vstack(errors).mean(axis=0).tolist()]
        print(tabulate(table, headers=CHANNELS, tablefmt="github"))

    elif args.split == "testing":
        for date in dates:
            cached_predict(get_path_pred, model, date)


if __name__ == "__main__":
    main()
