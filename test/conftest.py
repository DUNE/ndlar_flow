import pytest
import h5py
import subprocess
import os
import shutil

import h5flow


def pytest_configure():
    pytest.example_source_name = 'example/source'


@pytest.fixture
def tmp_h5_file():
    yield 'test.h5'
    if os.path.exists('test.h5'):
        os.remove('test.h5')


def maybe_fetch_from_url(pytestconfig, tmp_path_factory, url):
    # checks to see if the file already exists in cache, if so, copy from there
    # otherwise download from a url
    src = pytestconfig.cache.get(f'cached_{url}', None)
    file = os.path.basename(url)
    try:
        dest = tmp_path_factory.mktemp('module0_flow', numbered=False)
    except FileExistsError:
        dest = os.path.join(tmp_path_factory.getbasetemp(), 'module0_flow')

    if not os.path.exists(os.path.join(dest, file)):
        print(f'Saving to cache {dest}...')
        if src is None or not os.path.exists(os.path.join(src, file)):
            # copy from url
            print(f'{url} -> {file}')
            subprocess.run(['curl', '-f', '-O', url], check=True)
            os.replace(file, os.path.join(dest, file))
        else:
            # copy from cache
            print(f'{src}/{file} -> {file}')
            shutil.copy(os.path.join(src, file), os.path.join(dest, file))

        pytestconfig.cache.set(f'cached_{url}', str(dest))

    # copy file to cwd
    if os.path.exists(file):
        os.remove(file)
    shutil.copy(os.path.join(dest, file), './')

    yield file

    # cleanup
    if os.path.exists(file):
        os.remove(file)


@pytest.fixture(params=[
    'https://portal.nersc.gov/project/dune/data/Module0/TPC1+2/dataRuns/packetData/datalog_2021_04_04_00_41_40_CEST.h5',
    'https://portal.nersc.gov/project/dune/data/Module0/simulation/larndsim/datalog.edep.all.h5'
])
def charge_source_file(pytestconfig, tmp_path_factory, request):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory, request.param))


@pytest.fixture(params=[
    'https://portal.nersc.gov/project/dune/data/Module0/LRS/Converted/rwf_20210404_004206.data.root'
])
def light_source_file(pytestconfig, tmp_path_factory, request):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory, request.param))


@pytest.fixture(params=[
    'https://portal.nersc.gov/project/dune/data/Module0/multi_tile_layout-2.2.16.yaml'
])
def geometry_file(pytestconfig, tmp_path_factory, request):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory, request.param))


@pytest.fixture
def larpix_config_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     'https://portal.nersc.gov/project/dune/data/Module0/TPC1+2/configFiles/evd_config_21-03-31_12-36-13.json'
                                     ))


@pytest.fixture
def larpix_pedestal_config_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     'https://portal.nersc.gov/project/dune/data/Module0/TPC1+2/configFiles/datalog_2021_04_02_19_00_46_CESTevd_ped.json'
                                     ))


@pytest.fixture
def runlist_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     'https://portal.nersc.gov/project/dune/data/Module0/runlist.txt'
                                     ))


@pytest.fixture
def light_noise_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     'https://portal.nersc.gov/project/dune/data/Module0-Run2/LRS/LED/rwf_20210624_094156.fwvfm.noise_power.npz'
                                     ))


@pytest.fixture
def light_signal_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     'https://portal.nersc.gov/project/dune/data/Module0/merged/prod2/light_noise_filtered/wvfm_deconv_signal_power.npz'
                                     ))


@pytest.fixture
def light_impulse_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     'https://portal.nersc.gov/project/dune/data/Module0/merged/prod2/light_noise_filtered/wvfm_deconv_signal_impulse.fit.npz'
                                     ))
