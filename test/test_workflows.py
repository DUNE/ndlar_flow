import pytest
import h5py

import h5flow


def check_dsets(filename, datasets, check_empty=True):
    with h5py.File(filename, 'r') as f:
        assert all([d in f for d in datasets]), ('Missing dataset(s)',
                                                 f.visit(print))
        if check_empty:
            assert all([len(f[d]) for d in datasets]), ('Empty dataset(s)',
                                                        f.visititems(print))


@pytest.fixture
def charge_event_built_file(charge_source_file, runlist_file, tmp_h5_file):
    print('Charge event building...')
    h5flow.run('h5flow_yamls/reco/charge/charge_event_building.yaml',
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
def charge_reco_file(charge_event_built_file, geometry_file, larpix_config_file,
                     larpix_pedestal_config_file, tmp_h5_file):
    print('Charge event reconstruction...')
    h5flow.run('h5flow_yamls/reco/charge/charge_event_reconstruction.yaml',
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
def light_event_built_file(light_source_file, runlist_file,
                           electron_lifetime_file, tmp_h5_file):
    print('Light event building...')
    h5flow.run('h5flow_yamls/reco/light/light_event_building.yaml',
               tmp_h5_file,
               light_source_file,
               verbose=2,
               start_position=153840,
               end_position=153840 + 10000)

    check_dsets(tmp_h5_file, (
        'light/events/data',
        'light/wvfm/data',
    ))

    return tmp_h5_file


@pytest.fixture
def light_reco_file(light_event_built_file, light_noise_file, light_signal_file,
                    light_impulse_file, tmp_h5_file):
    print('Light event reconstruction...')
    h5flow.run('h5flow_yamls/reco/light/light_event_reconstruction.yaml',
               tmp_h5_file,
               light_event_built_file,
               verbose=2)

    check_dsets(tmp_h5_file, (
        'light/hits/data',
        'light/t_ns/data',
    ))

    return tmp_h5_file


@pytest.fixture
def charge_assoc_file(charge_reco_file, light_reco_file, tmp_h5_file):
    print('Charge/light association...')
    h5flow.run('h5flow_yamls/reco/charge/charge_light_assoc.yaml',
               tmp_h5_file,
               charge_reco_file,
               verbose=2)

    check_dsets(tmp_h5_file, (
        'charge/events/ref/light/events/ref',
        'charge/ext_trigs/ref/light/events/ref'
    ), check_empty=False)

    return tmp_h5_file


@pytest.fixture
def combined_file(charge_assoc_file, geometry_file, tmp_h5_file,
                  disabled_channels_list_file, missing_asic_list_file,
                  track_merging_pdf_file):
    print('Combined reconstruction...')
    h5flow.run('h5flow_yamls/reco/combined/combined_reconstruction.yaml',
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
def combined_file_no_light(charge_reco_file, geometry_file, tmp_h5_file,
                           disabled_channels_list_file, missing_asic_list_file,
                           track_merging_pdf_file):
    print('Combined reconstruction...')
    h5flow.run('h5flow_yamls/reco/combined/combined_reconstruction.yaml',
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
def broken_track_sim_file(combined_file_no_light, geometry_file, tmp_h5_file,
                          disabled_channels_list_file, missing_asic_list_file):
    print('Broken track simulation...')
    h5flow.run('h5flow_yamls/reco/combined/broken_track_sim.yaml',
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
def stopping_muon_analysis_file(combined_file, geometry_file,
                                proton_range_table, muon_range_table,
                                electron_lifetime_file, tmp_h5_file):
    print('Stopping muon analysis...')
    h5flow.run('h5flow_yamls/analysis/stopping_muons_data.yaml',
               tmp_h5_file,
               combined_file,
               verbose=2)

    check_dsets(tmp_h5_file, (
        'analysis/stopping_muons/event_sel_reco/data',
        'analysis/stopping_muons/event_profile/data'
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


def test_broken_track_sim(broken_track_sim_file):
    pass


def test_chain(stopping_muon_analysis_file):
    pass
