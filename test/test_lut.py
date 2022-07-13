import pytest
import numpy as np

from module0_flow.util.lut import LUT


@pytest.fixture(params=[
    ((np.arange(10), np.arange(10) * 10), 'i8', 0, (2,)),
    ((np.arange(10), ), 'i8', 0, (2,)),
    ((np.arange(10), ), 'i8', 0, (2, 3,)),
    ((np.arange(10), ), 'i8', 0, None),
    ((np.arange(10), ), 'i8', None, None),
    ((np.arange(10), ), 'i8', None, (2, 3,)),
    ((np.arange(10), np.arange(10) * 10), 'i8', None, (2, 3,)),
    ((np.arange(10), ), np.dtype([('f0', 'i8'), ('f1', 'f8')]), None, None),
    ((np.arange(10), ), np.dtype([('f0', 'i8'), ('f1', 'f8')]), None, (2, 3,)),
    ((np.arange(10), np.arange(10) * 10), np.dtype([('f0', 'i8'), ('f1', 'f8')]), None, (2, 3,)),
])
def lut(request):
    param = request.param
    keys = param[0]
    dtype = param[1]
    default = param[2]
    shape = param[3]

    lut = LUT(dtype, *[(min(k), max(k)) for k in keys], default=default,
              shape=shape)

    return lut, param


def test_lut_init(lut):
    assert lut[0].dtype == lut[1][1]
    assert np.all(lut[0].min_max_keys == np.array([(min(k), max(k)) for k in lut[1][0]]))
    assert np.all(lut[0].lengths == np.array([max(k) - min(k) + 1 for k in lut[1][0]]))
    assert lut[0].max_hash == lut[0].hash(*[max(k) for k in lut[1][0]])
    assert lut[1][2] is None or np.all(lut[0].default == lut[1][2])

    assert lut[0]._data.shape[0] == lut[0].max_hash + 1
    assert lut[1][3] is None or lut[0]._data.shape[1:] == lut[1][3]


def test_lut_array(lut):
    assert lut[0] == LUT.from_array(*lut[0].to_array())


def test_lut_getsetitem(lut):
    keys = lut[1][0]

    lut[0][keys] = 10
    assert np.all(lut[0][keys] == np.array(10).astype(lut[0].dtype))

    lut[0][[k[0] for k in keys]] = 9
    assert np.all(lut[0][[k[0] for k in keys]] == np.array(9).astype(lut[0].dtype))
    assert np.all(lut[0][[k[1:] for k in keys]] == np.array(10).astype(lut[0].dtype))


def test_lut_hash(lut):
    keys = lut[1][0]
    assert not np.any(lut[0].hash(*keys) <= 0)
    assert not np.any(lut[0].hash(*keys) > lut[0].max_hash)


def test_lut_get_keys(lut):
    keys = lut[1][0]
    shape = lut[1][3]
    lut[0][keys] = 10
    if shape is not None:
        assert np.all(lut[0][lut[0].keys()][..., 0] == lut[0].compress(sel=(0,)))
    else:
        assert np.all(lut[0][lut[0].keys()] == lut[0].compress())


def test_lut_default(lut):
    keys = lut[1][0]

    lut[0].default = 10
    assert np.all(lut[0][[k[0] for k in keys]] == np.array(10).astype(lut[0].dtype))

    lut[0][[k[0] for k in keys]] = 9
    assert np.all(lut[0][[k[0] for k in keys]] == np.array(9).astype(lut[0].dtype))

    lut[0].clear(*[k[0] for k in keys])
    assert np.all(lut[0][[k[0] for k in keys]] == np.array(10).astype(lut[0].dtype))
