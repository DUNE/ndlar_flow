import pytest

from h5flow.data import H5FlowDataManager
from module0_flow.resources.geometry import Geometry


@pytest.fixture
def geo(tmp_h5_file, geometry_file, light_geometry_file):
    dm = H5FlowDataManager(tmp_h5_file)

    geo = Geometry(data_manager=dm, classname='Geometry',
                   crs_geometry_file=geometry_file,
                   lrs_geometry_file=light_geometry_file)

    return geo


def test_geometry(tmp_h5_file, geo):
    # initialize and clean up
    geo.init(pytest.example_source_name)
    geo.finish(pytest.example_source_name)
    geo.data_manager.close_file()

    # re-open file and re-initialize
    dm = H5FlowDataManager(tmp_h5_file)
    geo.data_manager = dm
    geo.init(pytest.example_source_name)
    geo.finish(pytest.example_source_name)
