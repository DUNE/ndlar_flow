import pytest
import subprocess
import os
import shutil


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

    cached_filepath = os.path.join(dest, file)
    src_filepath = os.path.join(src, file) if src is not None else None
    dest_filepath = os.path.join('./h5flow_data/', file)

    # check if file exists in current cache
    if not os.path.exists(cached_filepath):
        if src_filepath is None or not os.path.exists(src_filepath):
            # copy from url
            print(f'Downloading {file} from {url}...')
            subprocess.run(['curl', '-f', '-o', cached_filepath, url], check=True)
        else:
            # copy from old cache
            print(f'Copying {file} from existing cache {src_filepath}...')
            shutil.copy(os.path.join(src, file), cached_filepath)
        print(f'Saved to current cache @ {cached_filepath}')
        pytestconfig.cache.set(f'cached_{url}', str(dest))

    # copy file from current cache to current h5flow data directory
    if os.path.exists(dest_filepath):
        print(f'Overwriting {dest_filepath}...')
        os.remove(dest_filepath)
    shutil.copy(cached_filepath, dest_filepath)

    yield dest_filepath


@pytest.fixture(params=[
    'data',
    'sim'
])
def charge_source_file(pytestconfig, tmp_path_factory, request):
    charge_source_file_url_lookup = {
        'data': ('https://portal.nersc.gov/project/dune/data/Module0/TPC1+2/'
                 'dataRuns/packetData/datalog_2021_04_04_00_41_40_CEST.h5'),
        'sim': ('https://portal.nersc.gov/project/dune/data/Module0/'
                'simulation/stopping_muons/stopping_muons.test.h5')
    }
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     charge_source_file_url_lookup[request.param]))


@ pytest.fixture
def light_source_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/LRS/Converted/'
                                      'rwf_20210404_004206.data.root')))


@ pytest.fixture
def geometry_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/'
                                      'multi_tile_layout-2.2.16.yaml')))


@ pytest.fixture
def light_geometry_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/'
                                      'light_module_desc-0.0.0.yaml')))


@ pytest.fixture
def larpix_config_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/TPC1+2/configFiles/'
                                      'evd_config_21-03-31_12-36-13.json')))


@ pytest.fixture
def larpix_pedestal_config_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/TPC1+2/configFiles/'
                                      'datalog_2021_04_02_19_00_46_CESTevd_ped.json')))


@ pytest.fixture
def runlist_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/runlist.txt')))


@ pytest.fixture
def disabled_channels_list_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/TPC1+2/badChannelLists/'
                                      'selftrigger_masked/'
                                      'module0-run1-selftrigger-disabled-list.json')))


@ pytest.fixture
def missing_asic_list_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/TPC1+2/badChannelLists/'
                                      'module0-network-absent-ASICs.json')))


@ pytest.fixture
def track_merging_pdf_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/merged/reco_data/'
                                      'joint_pdf-3_0_0.npz')))

@pytest.fixture
def michel_pdf_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/merged/reco_data/'
                                      'michel_pdf-0.1.0.npz')))

@pytest.fixture
def triplet_response_data_256_file(pytestconfig, tmp_path_factory):
    rv = next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/merged/reco_data/'
                                      'mod0_response.v0.data.256.npz')))
    if not os.path.exists('h5flow_data/mod0_response.data.256.npz'):
        os.rename(rv, 'h5flow_data/mod0_response.data.256.npz')    
    return rv

@pytest.fixture
def triplet_response_sim_256_file(pytestconfig, tmp_path_factory):
    rv = next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/merged/reco_data/'
                                      'mod0_response.v1.sim.256.npz')))
    if not os.path.exists('h5flow_data/mod0_response.sim.256.npz'):
        os.rename(rv, 'h5flow_data/mod0_response.sim.256.npz')
    return rv


@ pytest.fixture
def proton_range_table(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/merged/reco_data/'
                                      'NIST_proton_range_table_Ar.txt')))


@ pytest.fixture
def electron_lifetime_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/electronLifetime/'
                                      'ElecLifetimeFit_Module0.npz')))


@ pytest.fixture
def muon_range_table(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/merged/reco_data/'
                                      'PDG_muon_range_table_Ar.txt')))


@ pytest.fixture
def light_noise_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/merged/reco_data/'
                                      'events_2021_04_10_04_21_27_CEST.fwvfm.noise_power.npz')))


@ pytest.fixture
def light_signal_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/merged/prod2/'
                                      'light_noise_filtered/'
                                      'wvfm_deconv_signal_power.npz')))


@ pytest.fixture
def light_impulse_file(pytestconfig, tmp_path_factory):
    return next(maybe_fetch_from_url(pytestconfig, tmp_path_factory,
                                     ('https://portal.nersc.gov/project/dune/'
                                      'data/Module0/merged/prod2/'
                                      'light_noise_filtered/'
                                      'wvfm_deconv_signal_impulse.fit.npz')))
