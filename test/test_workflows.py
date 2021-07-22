import pytest
import h5py
import subprocess
import os

import h5flow

@pytest.fixture
def clean():
    temp_files = ('test.h5',)

    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)

@pytest.fixture
def charge_source_file(clean):
    filename = 'datalog_2021_04_04_00_41_40_CEST.h5'

    if not os.path.exists(filename):
        url = f'https://portal.nersc.gov/project/dune/data/Module0/TPC1+2/dataRuns/packetData/{filename}'
        subprocess.run(['curl','-O',url], check=True)

    return filename

@pytest.fixture
def light_source_file(clean):
    filename = 'rwf_20210404_004206.data.root'

    if not os.path.exists(filename):
        url = f'https://portal.nersc.gov/project/dune/data/Module0/LRS/Converted/{filename}'
        subprocess.run(['curl','-O',url], check=True)

    return filename

@pytest.fixture
def aux_files(clean):
    filename = 'multi_tile_layout-2.2.16.yaml'
    if not os.path.exists(filename):
        url = f'https://portal.nersc.gov/project/dune/data/Module0/{filename}'
        subprocess.run(['curl','-O',url], check=True)

    filename = 'evd_config_21-03-31_12-36-13.json'
    if not os.path.exists(filename):
        url = f'https://portal.nersc.gov/project/dune/data/Module0/TPC1+2/configFiles/{filename}'
        subprocess.run(['curl','-O',url], check=True)

    filename = 'datalog_2021_04_02_19_00_46_CESTevd_ped.json'
    if not os.path.exists(filename):
        url = f'https://portal.nersc.gov/project/dune/data/Module0/TPC1+2/configFiles/{filename}'
        subprocess.run(['curl','-O',url], check=True)

    filename = 'runlist.txt'
    if not os.path.exists(filename):
        url = f'https://portal.nersc.gov/project/dune/data/Module0/{filename}'
        subprocess.run(['curl','-O',url], check=True)

    filename = 'rwf_20210624_094156.fwvfm.noise_power.npz'
    if not os.path.exists(filename):
        url = f'https://portal.nersc.gov/project/dune/data/Module0-Run2/LRS/LED/{filename}'
        subprocess.run(['curl','-O',url], check=True)

    filename = 'wvfm_deconv_signal_power.npz'
    if not os.path.exists(filename):
        url = f'https://portal.nersc.gov/project/dune/data/Module0/merged/prod2/light_noise_filtered/{filename}'
        subprocess.run(['curl','-O',url], check=True)

    filename = 'wvfm_deconv_signal_impulse.fit.npz'
    if not os.path.exists(filename):
        url = f'https://portal.nersc.gov/project/dune/data/Module0/merged/prod2/light_noise_filtered/{filename}'
        subprocess.run(['curl','-O',url], check=True)


@pytest.fixture
def charge_event_built_file(charge_source_file, aux_files):
    output_filename = 'test.h5'
    h5flow.run('h5flow_yamls/charge/charge_event_building.yaml',
        output_filename,
        charge_source_file,
        verbose=2,
        start_position=5273174,
        end_position=5273174+10000)

    return output_filename

def test_charge_event_building(charge_event_built_file):
    f = h5py.File(charge_event_built_file,'r')

    required_datasets = (
        'charge/raw_events/data',
        'charge/packets/data',
        )

    assert all([d in f for d in required_datasets])

@pytest.fixture
def charge_reco_file(charge_event_built_file, aux_files):
    output_filename = 'test.h5'
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

@pytest.fixture
def light_event_built_file(light_source_file, aux_files):
    output_filename = 'test.h5'
    h5flow.run('h5flow_yamls/light/light_event_building.yaml',
        output_filename,
        light_source_file,
        verbose=2,
        start_position=128*128,
        end_position=10*128*128)

    return output_filename

def test_light_event_building(light_event_built_file):
    f = h5py.File(light_event_built_file,'r')

    required_datasets = (
        'light/events/data',
        'light/wvfm/data',
        )

    assert all([d in f for d in required_datasets])

@pytest.fixture
def light_reco_file(light_event_built_file, aux_files):
    output_filename = 'test.h5'
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

@pytest.fixture
def charge_assoc_file(charge_reco_file, light_reco_file, aux_files):
    output_filename = 'test.h5'
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
