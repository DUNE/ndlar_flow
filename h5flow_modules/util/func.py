import numpy as np
import numpy.ma as ma


def mode(arr):
    '''
        Finds the most common element along the last dimension

        :param arr: array ``shape: (..., N)``

        :returns: array ``shape: (..., 1)``
    '''
    orig_shape = arr.shape
    unique_val = np.unique(arr.ravel())
    arr = arr.reshape(-1, orig_shape[-1])

    count = np.zeros((len(arr), len(unique_val)))
    for j in range(len(unique_val)):
        count[..., j] = np.sum(arr == unique_val[j], axis=-1)

    mode = np.take_along_axis(unique_val, np.argmax(count, axis=-1), axis=-1)

    new_shape = orig_shape[:-1] + (1,)
    if isinstance(arr, ma.MaskedArray):
        return ma.array(mode.reshape(new_shape), mask=np.all(arr.mask, axis=-1).reshape(new_shape))
    return mode.reshape(new_shape)


def condense_array(arr, mask):
    '''
        Densify a masked array on last axis, throwing out invalid values
        (up to the size needed to keep the array regular). E.g.::

            mask = [[False, True, True],
                    [False, False, True],
                    [True, False, True]]

        will condense a 3x3 array to shape: ``(3, 2)`` and produce a final
        mask of::

            new_mask = [[False, True],
                        [False, False],
                        [False, True]]

        Note that this operation does not have an inverse.

        :param arr: array ``shape: (..., N, M)``

        :param mask: boolean array ``shape: (..., N, M)``, ``True == invalid``

        :returns: array ``shape: (..., N, m)``

    '''
    axis = -1
    n_valid = np.expand_dims(np.count_nonzero(~mask, axis=axis), axis=axis)

    new_shape = list(arr.shape)
    new_shape[axis] = n_valid.max()
    condensed = np.empty(new_shape, dtype=arr.dtype)
    idx = np.indices(condensed.shape)[-1]
    np.place(condensed, idx < n_valid, arr[~mask])

    return ma.array(condensed, mask=idx >= n_valid)
