import numpy as np


def fill_with_last(arr: np.ndarray) -> np.ndarray:
    '''
        Given an array ARR, copy it and replace all zeros with the last
        nonzero value. E.g.,
        [3, 0, 0, 5, 0, 8, 0, 0, 0] => [3, 3, 3, 5, 5, 8, 8, 8, 8].
        TODO: Replace with a faster implementation. (This one is based on
        the code that calculates unix_ts in raw_event_generator.py.)
    '''
    groups = np.split(arr, np.argwhere(arr).ravel())
    # SLOW:
    groups = [np.full(len(group), group[0])
              for group in groups if len(group)]
    return np.concatenate(groups, axis=0)
