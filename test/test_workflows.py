import pytest
import h5py
import shutil

import h5flow

from .conftest import maybe_fetch_from_url


def check_dsets(filename, datasets, check_empty=False):
    with h5py.File(filename, 'r') as f:
        assert all([d in f for d in datasets]), ('Missing dataset(s)',
                                                 f.visit(print))
        if check_empty:
            assert all([len(f[d]) for d in datasets]), ('Empty dataset(s)',
                                                        f.visititems(print))


@pytest.fixture
def charge_event_built_file(charge_source_file, runlist_file, tmp_h5_file):
    print('Charge event building...')
    h5flow.run(['h5flow_yamls/workflows/charge/charge_event_building.yaml'],
               tmp_h5_file,
               charge_source_file,
               verbose=2,
               start_position=0,
               end_position=1000)

    check_dsets(tmp_h5_file, (
        'charge/raw_events/data',
        'charge/packets/data',
    ))

    return tmp_h5_file


@pytest.fixture
def charge_reco_file(charge_event_built_file, geometry_file, light_geometry_file, larpix_config_file,
                     larpix_pedestal_config_file, tmp_h5_file):
    print('Charge event reconstruction...')
    h5flow.run(['h5flow_yamls/workflows/charge/charge_event_reconstruction.yaml'],
               tmp_h5_file,
               charge_event_built_file,
               verbose=2)

    check_dsets(tmp_h5_file, (
        'charge/hits/data',
        'charge/ext_trigs/data',
        'charge/events/data'
    ), check_empty=False)

    return tmp_h5_file


@pytest.fixture
def light_event_built_file(pytestconfig, tmp_path_factory, light_source_file, runlist_file,
                           electron_lifetime_file, tmp_h5_file):
    filename = next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/merged/prod2/light_event/'
                                      'events_2021_04_04_00_41_40_CEST.gz.h5')))
    with h5py.File(filename, 'r') as fi:
        with h5py.File(tmp_h5_file,'a') as fo:
            fo.copy(fi['light'],'light')
    '''
    print('Light event building...')
    h5flow.run(['h5flow_yamls/workflows/light/light_event_building.yaml'],
               tmp_h5_file,
               light_source_file,
               verbose=2,
               start_position=153840,
               end_position=153840 + 10000)

    check_dsets(tmp_h5_file, (
        'light/events/data',
        'light/wvfm/data',
    ))
    '''
    return tmp_h5_file


@pytest.fixture
def light_reco_file(light_event_built_file, light_noise_file, light_signal_file,
                    light_impulse_file, light_geometry_file, tmp_h5_file):
    print('Light event reconstruction...')
    h5flow.run(['h5flow_yamls/workflows/light/light_event_reconstruction.yaml'],
               tmp_h5_file,
               light_event_built_file,
               verbose=2,
               start_position=0,
               end_position=1000)

    check_dsets(tmp_h5_file, (
        'light/hits/data',
        'light/t_ns/data',
    ))

    return tmp_h5_file


@pytest.fixture
def light_reco_wvfm_file(light_event_built_file, light_noise_file, light_signal_file,
                    light_impulse_file, light_geometry_file, tmp_h5_file):
    print('Light event reconstruction...')
    h5flow.run(['h5flow_yamls/workflows/light/light_event_reconstruction-keep_wvfm.yaml'],
               tmp_h5_file,
               light_event_built_file,
               verbose=2,
               start_position=0,
               end_position=1000)

    check_dsets(tmp_h5_file, (
        'light/hits/data',
        'light/t_ns/data',
        'light/fwvfm/data',
        'light/deconv/data',
        'light/swvfm/data'
    ))

    return tmp_h5_file


@pytest.fixture
def charge_assoc_file(charge_reco_file, light_reco_file, tmp_h5_file):
    print('Charge/light association...')
    h5flow.run(['h5flow_yamls/workflows/charge/charge_light_assoc.yaml'],
               tmp_h5_file,
               charge_reco_file,
               verbose=2)

    check_dsets(tmp_h5_file, (
        'charge/events/ref/light/events/ref',
        'charge/ext_trigs/ref/light/events/ref'
    ), check_empty=False)

    return tmp_h5_file


@pytest.fixture
def charge_assoc_wvfm_file(light_reco_wvfm_file, charge_reco_file, tmp_h5_file):
    print('Charge/light association...')
    h5flow.run(['h5flow_yamls/workflows/charge/charge_light_assoc.yaml'],
               tmp_h5_file,
               charge_reco_file,
               verbose=2)

    check_dsets(tmp_h5_file, (
        'charge/events/ref/light/events/ref',
        'charge/ext_trigs/ref/light/events/ref'
    ), check_empty=False)

    return tmp_h5_file


@pytest.fixture
def combined_file(charge_assoc_file, geometry_file, light_geometry_file, tmp_h5_file,
                  disabled_channels_list_file, missing_asic_list_file,
                  track_merging_pdf_file):
    print('Combined reconstruction...')
    h5flow.run(['h5flow_yamls/workflows/combined/combined_reconstruction.yaml'],
               tmp_h5_file,
               charge_assoc_file,
               verbose=2,
               end_position=64)

    check_dsets(tmp_h5_file, (
        'combined/t0/data',
        'combined/hit_drift/data',
        'combined/tracklets/data',
        'combined/tracklets/merged/data'
    ))

    return tmp_h5_file


@pytest.fixture
def combined_wvfm_file(charge_assoc_wvfm_file, geometry_file, light_geometry_file, tmp_h5_file,
                  disabled_channels_list_file, missing_asic_list_file,
                  track_merging_pdf_file):
    print('Combined reconstruction...')
    h5flow.run(['h5flow_yamls/workflows/combined/combined_reconstruction.yaml'],
               tmp_h5_file,
               charge_assoc_wvfm_file,
               verbose=2,
               end_position=64)

    check_dsets(tmp_h5_file, (
        'combined/t0/data',
        'combined/hit_drift/data',
        'combined/tracklets/data',
        'combined/tracklets/merged/data'
    ))

    return tmp_h5_file


@pytest.fixture
def combined_file_no_light(charge_reco_file, geometry_file, light_geometry_file, tmp_h5_file,
                           disabled_channels_list_file, missing_asic_list_file,
                           track_merging_pdf_file):
    print('Combined reconstruction...')
    h5flow.run(['h5flow_yamls/workflows/combined/combined_reconstruction.yaml'],
               tmp_h5_file,
               charge_reco_file,
               verbose=2,
               end_position=64)

    check_dsets(tmp_h5_file, (
        'combined/t0/data',
        'combined/hit_drift/data',
        'combined/tracklets/data',
        'combined/tracklets/merged/data'
    ))

    return tmp_h5_file


@pytest.fixture
def broken_track_sim_file(combined_file_no_light, geometry_file, light_geometry_file, tmp_h5_file,
                          disabled_channels_list_file, missing_asic_list_file):
    print('Broken track simulation...')
    h5flow.run(['h5flow_yamls/workflows/combined/broken_track_sim.yaml'],
               tmp_h5_file,
               combined_file_no_light,
               verbose=2)

    check_dsets(tmp_h5_file, (
        'misc/broken_track_sim/offset/data',
        'misc/broken_track_sim/label/data',
        'misc/broken_track_sim/tracklets/data',
    ))

    return tmp_h5_file


@pytest.fixture
def stopping_muon_analysis_file(combined_file, geometry_file, light_geometry_file,
                                proton_range_table, muon_range_table, michel_pdf_file,
                                electron_lifetime_file, tmp_h5_file):
    print('Stopping muon analysis...')
    h5flow.run(['h5flow_yamls/workflows/analysis/stopping_muons.yaml'],
               tmp_h5_file,
               combined_file,
               verbose=2)

    check_dsets(tmp_h5_file, (
        'analysis/stopping_muons/event_sel_reco/data',
        'analysis/stopping_muons/event_profile/data'
    ))

    return tmp_h5_file


@pytest.fixture
def delayed_signal_analysis_file(stopping_muon_analysis_file, triplet_response_data_256_file,
                                 triplet_response_sim_256_file, tmp_h5_file):
    print('Stopping muon analysis...')
    h5flow.run(['h5flow_yamls/workflows/analysis/delayed_signal.yaml'],
               tmp_h5_file,
               stopping_muon_analysis_file,
               verbose=2)

    check_dsets(tmp_h5_file, (
        'analysis/time_reco/prompt/data',
        'analysis/time_reco/delayed/data',
        'analysis/time_reco/fit/data'
    ))

    return tmp_h5_file


@pytest.fixture
def light_calib_file(combined_wvfm_file, tmp_h5_file):
    print('Stopping muon analysis...')
    h5flow.run(['h5flow_yamls/workflows/combined/light_gain_calibration.yaml'],
               tmp_h5_file,
               combined_wvfm_file,
               verbose=2)

    check_dsets(tmp_h5_file, (
        'light/calib/data',
    ))

    return tmp_h5_file


# def test_charge_event_building(charge_event_built_file):
#     pass


# def test_charge_reco(charge_reco_file):
#     pass


# def test_light_event_building(light_event_built_file):
#     pass


# def test_light_reco(light_reco_file):
#     pass


# def test_charge_assoc(charge_assoc_file):
#     pass


# def test_combined(combined_file):
#     pass


def test_light_calib(light_calib_file):
    pass


def test_broken_track_sim(broken_track_sim_file):
    pass


def test_chain(delayed_signal_analysis_file):
    pass

