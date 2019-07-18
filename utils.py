import os

from datetime import datetime

import h5py
import numpy as np

from src.dataset import Traffic4CastSample


def write_data(data, filename):
    """Write data in gzipped h5 format.

    This function was taken almost verbatim from the official competition repo:
    https://github.com/iarai/NeurIPS2019-traffic4cast/blob/master/utils/create_submissiontest_like.py#L34-L40
    """
    f = h5py.File(filename, 'w', libver='latest')
    f.create_dataset('array',
                     shape=data.shape,
                     data=data,
                     compression='gzip',
                     compression_opts=9)
    f.close()


def cache(func, path, to_overwrite, *args, **kwargs):
    if os.path.exists(path) and not to_overwrite:
        result = np.array(h5py.File(path, 'r')['array'])
        result = result.astype(np.uint8)
        return result
    else:
        result = func(*args, **kwargs)
        results = result.astype(np.uint8)
        write_data(result, path)
        return result


def day_frame_to_date(day: datetime, frame: int) -> datetime:
    # This function is the inverse of `date_to_day_frame`
    return day + frame * Traffic4CastSample.time_step_delta


def date_to_day_frame(date: datetime) -> (datetime, int):
    # This function is the inverse of `day_frame_to_date`
    day = date.replace(hour=0, minute=0)
    frame = (date - day).seconds // 300
    return day, frame


def sliding_window(data, size, stepsize=1, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal

    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.

    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.

    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.

    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])

    See Also
    --------
    pieces : Calculate number of pieces available by sliding

    """
    if axis >= data.ndim:
        raise ValueError("Axis value out of range")

    if stepsize < 1:
        raise ValueError("Stepsize may not be zero or negative")

    if size > data.shape[axis]:
        raise ValueError("Sliding window size may not exceed size of selected axis")

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1)
    shape[axis] = shape[axis].astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    if copy:
        return strided.copy()
    else:
        return strided
