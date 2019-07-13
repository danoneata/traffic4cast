from datetime import datetime, timedelta

import numpy as np

FRAME_DURATION = 5  # minutes


def cache(func, path, *args, **kwargs):
    try:
        return np.load(path)
    except FileNotFoundError:
        result = func(*args, **kwargs)
        np.save(path, result)
        return result


def day_frame_to_date(day: datetime, frame: int) -> datetime:
    # This function is the inverse of `date_to_day_frame`
    return day + timedelta(minutes=frame * FRAME_DURATION)


def date_to_day_frame(date: datetime) -> (datetime, int):
    # This function is the inverse of `day_frame_to_date`
    day = date.replace(hour=0, minute=0)
    frame = (date - day).seconds // 300
    return day, frame
