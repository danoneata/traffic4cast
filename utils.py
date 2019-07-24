import os

from datetime import datetime

import h5py
import numpy as np
import submission_write

from src.dataset import Traffic4CastSample


def cache(func, path, to_overwrite, *args, **kwargs):
    if os.path.exists(path) and not to_overwrite:
        result = np.array(h5py.File(path, 'r')['array'])
        result = result.astype(np.uint8)
        return result
    else:
        result = func(*args, **kwargs)
        results = result.astype(np.uint8)
        submission_write.write_data(result, path)
        return result


def day_frame_to_date(day: datetime, frame: int) -> datetime:
    # This function is the inverse of `date_to_day_frame`
    return day + frame * Traffic4CastSample.time_step_delta


def date_to_day_frame(date: datetime) -> (datetime, int):
    # This function is the inverse of `day_frame_to_date`
    day = date.replace(hour=0, minute=0)
    frame = (date - day).seconds // 300
    return day, frame
