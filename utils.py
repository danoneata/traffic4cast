from datetime import datetime, timedelta

import numpy as np

FRAME_DURATION = 5  # minutes


def cache(path):

    def wrapped(func):

        def f(*args):
            try:
                return np.load(path)
            except FileNotFoundError:
                result = func(*args)
                np.save(path, result)
                return result

        return f

    return wrapped


def day_frame_to_date(day: datetime, frame: int) -> datetime:
    # day_frame_to_date . date_to_day_frame = id
    # date_to_day_frame . day_frame_to_date = id
    return day + timedelta(minutes=frame * FRAME_DURATION)


def date_to_day_frame(date: datetime) -> (datetime, int):
    # day_frame_to_date . date_to_day_frame = id
    # date_to_day_frame . day_frame_to_date = id
    day = date.replace(hour=0, minute=0)
    frame = (date - day).seconds // 300
    return day, frame
