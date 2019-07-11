import argparse
import os
import pdb

from datetime import datetime, timedelta

from functools import partial

import h5py

import numpy as np

from typing import Callable, List

from tabulate import tabulate

from utils import (
    cache,
    date_to_day_frame,
    day_frame_to_date,
)

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


def predict(model, date, n_frames=1):
    H, W, C = 495, 436, 3
    return np.random.randn(n_frames, H, W, C)


def path_to_date(path):
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

    days = get_days("data", args.city, args.split)
    dates = [
        day_frame_to_date(day, frame) for day in days for frame in START_FRAMES
    ]

    # model = load_model(args.model, city=args.city)
    model = None
    cached_predict = lambda path, *args: cache(path)(predict)(*args)

    if args.split == "validation":
        get_path1 = partial(get_path, "data", args.city, args.split)
        # get_path2 = partial(get_path, "pred", args.city, args.split)
        errors = []
        for date in dates:
            groundtruth = load_groundtruth(get_path1, date, n_frames=3)
            prediction = cached_predict("/tmp/o.npy", model, date)
            squared_error = np.mean((groundtruth - prediction)**2,
                                    axis=(0, 1, 2))
            errors.append(squared_error)
            if args.verbose:
                print(date, "| 3 frames |", " | ".join(f"{e:7.2f}" for e in sq_err))
            break

        table = [np.vstack(errors).mean(axis=0).tolist()]
        print(tabulate(table, headers=CHANNELS, tablefmt="github"))

    elif args.split == "testing":
        for date in dates:
            cached_predict(path, model, date)


if __name__ == "__main__":
    main()
