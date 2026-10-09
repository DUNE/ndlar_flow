'''Settings shared by v_drift.py and vdrift_timeseries.py'''

from pathlib import Path

import yaml

IO_GROUPS = tuple(range(1, 9))
YAML_DIR = Path(__file__).resolve().parents[2] / 'yamls'
# detector option -> (flow yaml directory, plot label)
DETECTORS = {
    '2x2': ('proto_nd_flow', '2x2'),
    'fsd': ('fsd_flow', 'FSD'),
    'fsdcube': ('fsdcube_flow', 'FSD cube'),
    'ndlar': ('ndlar_flow', 'ND-LAr'),
}
DEFAULT_DETECTOR = '2x2'


def load_calibrated_vdrift(detector):
    '''Calibrated drift velocity in the detector's LArData.yaml [m/s]'''
    yaml_file = YAML_DIR / DETECTORS[detector][0] / 'resources/LArData.yaml'
    with open(yaml_file, encoding='utf-8') as f:
        vdrift = yaml.safe_load(f)['params'].get('vdrift')  # mm/us, one value or one per module
    if not vdrift:
        print(f'No vdrift set in {yaml_file} (computed from E field)')
        return []
    return sorted({float(v) * 1e3 for v in vdrift})
