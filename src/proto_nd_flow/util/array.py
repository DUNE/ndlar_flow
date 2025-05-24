import numpy as np


def fill_with_last(arr: np.ndarray, marker=0) -> np.ndarray:
    '''
        Given an array ARR, copy it and replace all elements equal to MARKER
        (default: 0) with the last value that was not equal to MARKER. E.g.,
        [3, 0, 0, 5, 0, 8, 0, 0, 0] => [3, 3, 3, 5, 5, 8, 8, 8, 8].
        TODO: Replace with a faster implementation. (This one is based on
        the code that calculates unix_ts in raw_event_generator.py.)
    '''
    groups = np.split(arr, np.argwhere(arr != marker).ravel())
    # SLOW:
    groups = [np.full(len(group), group[0])
              for group in groups if len(group)]
    return np.concatenate(groups, axis=0)


def fill_with_next(arr: np.ndarray, marker=0) -> np.ndarray:
    '''
        Given an array ARR, copy it and replace all elements equal to MARKER
        (default: 0) with the next value that is not equal to MARKER. E.g.,
        [0, 3, 0, 0, 5, 0, 8] => [3, 3, 5, 5, 5, 8, 8].
        TODO: Replace with a faster implementation.
    '''
    a = arr[::-1]
    a = fill_with_last(a, marker=marker)
    return a[::-1]