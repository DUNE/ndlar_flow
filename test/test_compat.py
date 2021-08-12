import pytest

from module0_flow.util.compat import assert_compat_version


def test_assert_compat_version_valid():
    assert_compat_version('1.2.3', '1.2.3')
    assert_compat_version('1.2.3', '1.2.2')
    assert_compat_version('1.2.3', '1.2.4')
    assert_compat_version('1.3.0', '1.2.0')


def test_assert_compat_version_invalid():
    with pytest.raises(AssertionError):
        assert_compat_version('1.2.0', '1.3.0')

    with pytest.raises(AssertionError):
        assert_compat_version('1.2.0', '4.3.0')

    with pytest.raises(AssertionError):
        assert_compat_version('4.2.0', '1.3.0')
