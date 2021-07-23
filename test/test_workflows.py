import pytest
import h5py
import subprocess
import os
import shutil

import h5flow

charge_source_file    = 'datalog_2021_04_04_00_41_40_CEST.h5'
charge_source_file_mc = 'datalog.edep.14110.h5'
light_source_file     = 'rwf_20210404_004206.data.root'
geometry_file       = 'multi_tile_layout-2.2.16.yaml'
larpix_config_file  = 'evd_config_21-03-31_12-36-13.json'
larpix_pedestal_config_file = 'datalog_2021_04_02_19_00_46_CESTevd_ped.json'
runlist_file        = 'runlist.txt'
light_noise_file    = 'rwf_20210624_094156.fwvfm.noise_power.npz'
light_signal_file   = 'wvfm_deconv_signal_power.npz'
light_impulse_file  = 'wvfm_deconv_signal_impulse.fit.npz'

data_files = [
    charge_source_file,
    charge_source_file_mc,
    light_source_file,
    geometry_file,
    larpix_config_file,
    larpix_pedestal_config_file,
    runlist_file,
    light_noise_file,
    light_signal_file,
    light_impulse_file,
    ]

output_filename = 'test.h5'

@pytest.fixture
def data_directory(pytestconfig, tmp_path_factory):
    src = pytestconfig.cache.get('data_directory', None)
    dest = tmp_path_factory.mktemp('module0_flow', numbered=False)

    print(f'Saving data to cache {dest}...')
    pytestconfig.cache.set('data_directory', str(dest))

    urls = (
        f'https://portal.nersc.gov/project/dune/data/Module0/TPC1+2/dataRuns/packetData/{charge_source_file}',
        f'https://portal.nersc.gov/project/dune/data/Module0/simulation/larndsim/{charge_source_file_mc}',
        f'https://portal.nersc.gov/project/dune/data/Module0/LRS/Converted/{light_source_file}',
        f'https://portal.nersc.gov/project/dune/data/Module0/{geometry_file}',
        f'https://portal.nersc.gov/project/dune/data/Module0/TPC1+2/configFiles/{larpix_config_file}',
        f'https://portal.nersc.gov/project/dune/data/Module0/TPC1+2/configFiles/{larpix_pedestal_config_file}',
        f'https://portal.nersc.gov/project/dune/data/Module0/{runlist_file}',
        f'https://portal.nersc.gov/project/dune/data/Module0-Run2/LRS/LED/{light_noise_file}',
        f'https://portal.nersc.gov/project/dune/data/Module0/merged/prod2/light_noise_filtered/{light_signal_file}',
        f'https://portal.nersc.gov/project/dune/data/Module0/merged/prod2/light_noise_filtered/{light_impulse_file}'
        )

    for file,url in zip(data_files,urls):
        if not os.path.exists(os.path.join(dest, file)):
            if src is None or not os.path.exists(os.path.join(src, file)):
                # copy from nersc portal
                print(f'{url} -> {file}')
                subprocess.run(['curl','-f','-O',url], check=True)
                os.replace(file, os.path.join(dest, file))

            else:
                # copy from cache
                print(f'{src}/{file} -> {file}')
                shutil.copy(os.path.join(src, file), os.path.join(dest, file))

    return dirname

@pytest.fixture
def fresh_data_files(data_directory):
    for file in data_files:
        if os.path.exists(file):
            os.remove(file)
        shutil.copy(os.path.join(data_directory, file), './')

    if os.path.exists(output_filename):
        os.remove(output_filename)

@pytest.mark.parametrize('source_file,start,npackets',
    [(charge_source_file, charge_source_file_mc), (5273174, 0), (1000, 1000)])
@pytest.fixture
def charge_event_built_file(fresh_data_files, source_file, start, npackets):
    print('Charge event building...')
    h5flow.run('h5flow_yamls/charge/charge_event_building.yaml',
        output_filename,
        source_file,
        verbose=2,
        start_position=start,
        end_position=start+npackets)

    return output_filename

def test_charge_event_building(charge_event_built_file):
    f = h5py.File(charge_event_built_file,'r')

    required_datasets = (
        'charge/raw_events/data',
        'charge/packets/data',
        )

    assert all([d in f for d in required_datasets])
    assert all([len(f[d]) for d in required_datasets])

@pytest.fixture
def charge_reco_file(charge_event_built_file):
    print('Charge event reconstruction...')
    h5flow.run('h5flow_yamls/charge/charge_event_reconstruction.yaml',
        output_filename,
        charge_event_built_file,
        verbose=2)

    return output_filename

def test_charge_reco(charge_reco_file):
    f = h5py.File(charge_reco_file,'r')

    required_datasets = (
        'charge/hits/data',
        'charge/ext_trigs/data',
        'charge/events/data'
        )

    assert all([d in f for d in required_datasets])
    assert all([len(f[d]) for d in required_datasets])

@pytest.fixture
def light_event_built_file(fresh_data_files):
    print('Light event building...')
    h5flow.run('h5flow_yamls/light/light_event_building.yaml',
        output_filename,
        light_source_file,
        verbose=2,
        start_position=153840,
        end_position=153840+10000)

    return output_filename

def test_light_event_building(light_event_built_file):
    f = h5py.File(light_event_built_file,'r')

    required_datasets = (
        'light/events/data',
        'light/wvfm/data',
        )

    assert all([d in f for d in required_datasets])
    assert all([len(f[d]) for d in required_datasets])

@pytest.fixture
def light_reco_file(light_event_built_file):
    print('Light event reconstruction...')
    h5flow.run('h5flow_yamls/light/light_event_reconstruction.yaml',
        output_filename,
        light_event_built_file,
        verbose=2)

    return output_filename

def test_light_reco(light_reco_file):
    f = h5py.File(light_reco_file,'r')

    required_datasets = (
        'light/hits/data',
        'light/t_ns/data',
        )

    assert all([d in f for d in required_datasets])
    assert all([len(f[d]) for d in required_datasets])

@pytest.fixture
def charge_assoc_file(charge_reco_file, light_reco_file):
    print('Charge/light association...')
    h5flow.run('h5flow_yamls/charge/charge_light_assoc.yaml',
        output_filename,
        charge_reco_file,
        verbose=2)

    return output_filename

def test_chain(charge_assoc_file):
    f = h5py.File(charge_assoc_file,'r')

    required_datasets = (
        # charge datasets
        'charge/raw_events/data',
        'charge/packets/data',
        'charge/hits/data',
        'charge/ext_trigs/data',
        'charge/events/data',

        # light datasets
        'light/events/data',
        'light/hits/data',
        'light/t_ns/data'
        )

    assert all([d in f for d in required_datasets])
    assert all([len(f[d]) for d in required_datasets])
