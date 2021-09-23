import pytest
import numpy as np
import numpy.ma as ma

from module0_flow.util.func import condense_array, mode


def test_condense_array():
    a = np.array([
        [(1, False), (2, False), (3, True)],
        [(1, False), (2, True), (3, True)],
        [(1, True), (2, True), (3, False)],
    ], dtype=np.dtype([('val', 'i4'), ('mask', bool)]))

    b = condense_array(a['val'], a['mask'])
    assert b.shape == (3, 2)
    assert len(b.compressed()) == len(a['val'][~a['mask']])
    assert np.all(a['val'][~a['mask']] == b.compressed())
    assert np.all(b[2, 0] == 3)


def test_mode_array():
    a = np.array([
        [0, 1, 2],
        [0, 0, 1],
        [2, 2, 0]
    ])
    assert np.all(mode(a) == np.array([[0], [0], [2]]))


def test_mode_masked_array():
    a = np.array([
        [0, 1, 2, 2],
        [0, 0, 1, 1],
        [2, 2, 0, 2],
        [0, 1, 2, 3]
    ])
    mask = np.array([
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 1]
    ], dtype=bool)
    result = mode(ma.array(a, mask=mask))
    assert np.all(result == np.array([[1], [1], [2], [0]]))
    assert np.all(result.mask == np.array([[False], [False], [False], [True]]))
