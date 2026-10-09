import argparse
import copy
import json
import math
import os
import re
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from flufl.lock import Lock
from scipy.ndimage import gaussian_filter1d

try:
    from nearline_util import date_from_filename as nearline_date_from_filename
except ImportError:
    nearline_date_from_filename = None

from vdrift_timeseries import DEFAULT_DETECTOR, DETECTORS, load_calibrated_vdrift
from vdrift_timeseries import main as plot_timeseries


# Preferred first; calib_final_hits is the pre-#246 name of the filtered hits (e.g. 2x2 reflow v11),
# then fall back to prompt hits if the file was not run through the filtering stage
HIT_DSETS = ('charge/calib_filtered_hits', 'charge/calib_final_hits', 'charge/calib_prompt_hits')
EVENT_DSET = 'charge/events'
EXT_TRIG_DSET = 'charge/ext_trigs'
IO_GROUPS = tuple(range(1, 9))
LOCAL_TIMEZONE = ZoneInfo('America/Chicago')
# Mean raw velocity (lar_info v_drift x geometric/measured, calib_final_hits) over the 2x2
# reflow v11 beam july8_2024 + july10_2024 nominal_hv files; recompute if the method changes
BASELINE_RAW_MEAN_M_PER_S = 1573.4802914621064
BASELINE_FILE_COUNT = 486
BASELINE_MEASUREMENT_COUNT = 3881
BASELINE_START_TIMESTAMP = '2024-07-08T13:43:25-05:00'
BASELINE_END_TIMESTAMP = '2024-07-12T03:50:43-05:00'
X_BIN_WIDTH_CM = 0.01596 * 2.0
SEARCH_WINDOW_CM = 1.4
PLATEAU_MARGIN_ANODE_CM = 3.0
PLATEAU_MARGIN_CATHODE_CM = 3.0
THRESHOLD_FRACTION = 0.5
SMOOTH_SIGMA_CM = 0.08
STABILITY_GAP_CM = 0.04
STABILITY_WIDTH_CM = 0.12
STABILITY_MIN_BINS = 3
MIN_HITS = 100
N_Z_BINS = 10
N_Y_BINS = 10

GEOMETRY_PATH = 'geometry_info'
LAR_INFO_PATH = 'lar_info'
# Older flow layouts store cathode_thickness = 0 (cathode at module centre); the hit edge
# sits at the real cathode surface, so fall back to the 2x2 cathode thickness in that case
CATHODE_THICKNESS_FALLBACK_CM = 0.635


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def date_from_input_filename(input_file):
    if nearline_date_from_filename is not None:
        try:
            timestamp = nearline_date_from_filename(input_file)
            return timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=LOCAL_TIMEZONE)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    match = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', os.path.basename(input_file))
    if match is None:
        return None
    return datetime.strptime(match.group(), '%Y_%m_%d_%H_%M_%S').replace(tzinfo=LOCAL_TIMEZONE)


def json_file_lock(json_path):
    return Lock(str(json_path) + '.lock')


def ref_pairs(fh, parent, child):
    '''(parent index, child index) for every row of the h5flow reference parent -> child'''
    ref = fh[f'{parent}/ref/{child}/ref'][:]
    return ref[:, 0].astype(np.int64), ref[:, 1].astype(np.int64)


def event_filter_mask(fh):
    '''Boolean mask over events: first ext trigger on a real io_group, at most one ext
    trigger per io_group, and event duration <= 3300 ticks'''
    events = fh[f'{EVENT_DSET}/data'][:]
    n_events = len(events)
    evt_idx, trig_idx = ref_pairs(fh, EVENT_DSET, EXT_TRIG_DSET)
    iogroups = fh[f'{EXT_TRIG_DSET}/data']['iogroup'][trig_idx].astype(np.int64)

    # first trigger of each event (reference order) must be on a real io_group;
    # events without triggers stay False
    first_ok = np.zeros(n_events, dtype=bool)
    triggered, first = np.unique(evt_idx, return_index=True)
    first_ok[triggered] = iogroups[first] > 0

    # at most one external trigger per io_group (io_group 0 not counted)
    pairs, counts = np.unique(evt_idx * 256 + iogroups, return_counts=True)
    repeated = (counts > 1) & (pairs % 256 > 0)
    multi_trig = np.zeros(n_events, dtype=bool)
    multi_trig[pairs[repeated] // 256] = True

    too_long = (events['ts_end'] - events['ts_start']) > 3300
    return first_ok & ~multi_trig & ~too_long


def record_threshold_fraction(record):
    parameters = record.get('detection_parameters', {})
    if parameters.get('threshold_fraction') is not None:
        return validate_threshold_fraction(parameters['threshold_fraction'])
    for result in record.get('io_groups', {}).values():
        if result.get('threshold_fraction') is not None:
            return validate_threshold_fraction(result['threshold_fraction'])
        plateau, threshold = result.get('plateau_count'), result.get('threshold_count')
        if plateau is not None and plateau > 0 and threshold is not None:
            return validate_threshold_fraction(threshold / plateau)
    return None


def record_velocity_scale(record):
    return validate_velocity_scale(record.get('normalization', {}).get('velocity_scale', 1.0))


def update_json(output_file_json, record):
    parent = os.path.dirname(os.path.abspath(output_file_json))
    os.makedirs(parent, exist_ok=True)

    with json_file_lock(output_file_json):
        if os.path.exists(output_file_json):
            with open(output_file_json, encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'vdrifts': []}

        records = data.setdefault('vdrifts', [])

        source_file = record['source_file']
        fraction = record_threshold_fraction(record)
        velocity_scale = record_velocity_scale(record)
        for old in records:
            if old.get('source_file') == source_file:
                continue
            old_fraction = record_threshold_fraction(old)
            if (
                fraction is not None
                and old_fraction is not None
                and not math.isclose(fraction, old_fraction, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError(
                    'Threshold fractions differ within the JSON history; '
                    'use a separate output JSON for a different threshold'
                )
            if not math.isclose(
                velocity_scale, record_velocity_scale(old), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    'Velocity scales differ within the JSON history; '
                    'use a separate output JSON or normalize the existing records consistently'
                )
        replaced = False

        for i, old in enumerate(records):
            if old.get('source_file') == source_file:
                records[i] = record
                replaced = True
                break

        if not replaced:
            records.append(record)

        records.sort(
            key=lambda x: (
                x.get('timestamp') is None,
                x.get('timestamp') or '',
            )
        )

        safe_data = json_safe(data)

        fd, tmp_path = tempfile.mkstemp(
            prefix='.vdrift_',
            suffix='.json',
            dir=parent,
        )

        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(safe_data, f, indent=2, allow_nan=False)
                f.write('\n')

            os.replace(tmp_path, output_file_json)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return safe_data


def load_tpc_config(manager):
    from proto_nd_flow.util.lut import read_lut

    geometry = manager.get_attrs(GEOMETRY_PATH)
    module_bounds = np.asarray(geometry['module_RO_bounds'])
    cathode_thickness = float(geometry['cathode_thickness']) or CATHODE_THICKNESS_FALLBACK_CM

    # v_drift [mm/us] used by flow for this file: one value, or one per module
    v_drifts = np.atleast_1d(manager.get_attrs(LAR_INFO_PATH)['v_drift']) * 1e3

    tile_lut = read_lut(manager, GEOMETRY_PATH, 'tile_id')
    anode_lut = read_lut(manager, GEOMETRY_PATH, 'anode_drift_coordinate')
    drift_dir_lut = read_lut(manager, GEOMETRY_PATH, 'drift_dir')
    io_keys, channel_keys = tile_lut.keys()

    tpcs = {}
    for io in IO_GROUPS:
        tiles = tile_lut[(io_keys[io_keys == io], channel_keys[io_keys == io])]
        tiles = np.unique(tiles[tiles >= 0])
        anodes = np.unique(anode_lut[(tiles,)])
        drift_dirs = np.unique(drift_dir_lut[(tiles,)])
        if len(anodes) != 1 or len(drift_dirs) != 1:
            raise ValueError(f'io_group {io}: expected one anode plane, got {anodes} / {drift_dirs}')
        anode, drift_dir = float(anodes[0]), int(drift_dirs[0])

        # 2x2 convention: two io_groups per module (as in calib_prompt_hits)
        module = (io - 1) // 2
        bounds = module_bounds[module]
        if not np.isclose(anode, bounds[:, 0]).any():
            raise ValueError(f'io_group {io}: anode x={anode} is not on module {module} bounds')
        cathode = float(bounds[:, 0].mean() - drift_dir * cathode_thickness / 2)
        tpcs[io] = {
            'name': f'M{module}_{"Right" if io % 2 else "Left"}',
            'module': module,
            'cathode': cathode,
            'anode': anode,
            'nominal_v_m_per_s': float(v_drifts[module] if len(v_drifts) > 1 else v_drifts[0]),
            'x_range': sorted((cathode, anode)),
            'y_range': [float(bounds[0, 1]), float(bounds[1, 1])],
            'z_range': [float(bounds[0, 2]), float(bounds[1, 2])],
        }
    return tpcs


def histogram_edges(tpc):
    x_min, x_max = tpc['x_range']
    plot_min, plot_max = x_min - 2.0, x_max + 2.0
    edge_count = int(np.round((plot_max - plot_min) / X_BIN_WIDTH_CM)) + 1
    return np.linspace(plot_min, plot_max, edge_count)


def initialize_histograms(tpcs):
    histograms = {}
    for io in IO_GROUPS:
        config = tpcs[io]
        edges = histogram_edges(config)
        histograms[io] = {
            'edges': edges,
            'counts': np.zeros(len(edges) - 1, dtype=np.int64),
            'n_hits': 0,
        }
        for axis, bin_count in (('z', N_Z_BINS), ('y', N_Y_BINS)):
            histograms[io][axis] = {
                'edges': np.linspace(*config[f'{axis}_range'], bin_count + 1),
                'counts': np.zeros((bin_count, len(edges) - 1), dtype=np.int64),
                'n_hits': np.zeros(bin_count, dtype=np.int64),
            }
    return histograms


def accumulate_hits(histograms, hits):
    coordinates = {
        field: np.asarray(np.ma.getdata(hits[field])).reshape(-1)
        for field in ('x', 'y', 'z', 'io_group')
    }
    valid = ~(np.isnan(coordinates['x']) | np.isnan(coordinates['y']) | np.isnan(coordinates['z']))
    coordinates = {field: values[valid] for field, values in coordinates.items()}

    for io in IO_GROUPS:
        group_mask = coordinates['io_group'] == io
        x_values = coordinates['x'][group_mask]
        if not len(x_values):
            continue

        histogram = histograms[io]
        histogram['n_hits'] += len(x_values)
        histogram['counts'] += np.histogram(x_values, bins=histogram['edges'])[0]

        for axis in ('z', 'y'):
            values = coordinates[axis][group_mask]
            spatial = histogram[axis]
            in_range = (values >= spatial['edges'][0]) & (values < spatial['edges'][-1])
            spatial['n_hits'] += np.histogram(values[in_range], bins=spatial['edges'])[0]
            counts = np.histogram2d(
                values[in_range],
                x_values[in_range],
                bins=(spatial['edges'], histogram['edges']),
            )[0]
            spatial['counts'] += counts.astype(np.int64)


def select_hit_dset(manager):
    for hit_dset in HIT_DSETS:
        if hit_dset in manager.fh:
            return hit_dset
    raise KeyError(f'None of {HIT_DSETS} found in input file')


def load_hit_histograms(input_file):
    from h5flow.data import H5FlowDataManager

    with H5FlowDataManager(input_file, 'r', mpi=False) as manager:
        tpcs = load_tpc_config(manager)
        histograms = initialize_histograms(tpcs)
        hit_dset = select_hit_dset(manager)
        print(f'Using hits from: {hit_dset}')

        # read the references once and select with numpy instead of one lookup per event
        selected = event_filter_mask(manager.fh)
        evt_idx, hit_idx = ref_pairs(manager.fh, EVENT_DSET, hit_dset)
        hit_idx = hit_idx[selected[evt_idx]]
        hits = manager.fh[f'{hit_dset}/data'].fields(['x', 'y', 'z', 'io_group'])[:]
        accumulate_hits(histograms, hits[hit_idx])

    total_events = len(selected)
    selected_events = int(selected.sum())
    rejected_events = total_events - selected_events

    return {
        'histograms': histograms,
        'tpcs': tpcs,
        'hit_dset': hit_dset,
        'total_events': total_events,
        'selected_events': selected_events,
        'rejected_events': rejected_events,
    }


def find_stable_crossing(positions, counts, threshold, bin_width):
    if len(positions) < 2 * STABILITY_MIN_BINS:
        return None, 'insufficient_search_bins'
    if (
        np.median(counts[:STABILITY_MIN_BINS]) < threshold
        or np.median(counts[-STABILITY_MIN_BINS:]) >= threshold
    ):
        return None, 'unbracketed_edge'

    support_width = max(STABILITY_WIDTH_CM, STABILITY_MIN_BINS * bin_width)
    support_extent = STABILITY_GAP_CM + support_width
    for index in range(len(positions) - 2, -1, -1):
        inner_count, outer_count = counts[index : index + 2]
        if not inner_count >= threshold > outer_count:
            continue

        fraction = (threshold - inner_count) / (outer_count - inner_count)
        crossing = positions[index] + fraction * (positions[index + 1] - positions[index])
        if crossing - support_extent < positions[0] or crossing + support_extent > positions[-1]:
            continue

        interior = counts[
            (positions >= crossing - support_extent) & (positions <= crossing - STABILITY_GAP_CM)
        ]
        exterior = counts[
            (positions >= crossing + STABILITY_GAP_CM) & (positions <= crossing + support_extent)
        ]
        if min(len(interior), len(exterior)) < STABILITY_MIN_BINS:
            continue
        if np.all(interior >= threshold) and np.all(exterior < threshold):
            return float(crossing), 'ok'

    return None, 'no_stable_crossing'


def validate_threshold_fraction(value):
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError('Threshold fraction must be finite and strictly between 0 and 1')
    return fraction


def validate_velocity_scale(value):
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Velocity scale must be finite and positive')
    return scale


def detect_boundary(hist, edges, io_group, config, n_hits, threshold_fraction=None):
    threshold_fraction = validate_threshold_fraction(
        THRESHOLD_FRACTION if threshold_fraction is None else threshold_fraction
    )
    cathode = config['cathode']
    anode = config['anode']
    nominal_velocity = config['nominal_v_m_per_s']
    centers = 0.5 * (edges[:-1] + edges[1:])
    result = {
        'io_group': io_group,
        'status': 'insufficient_data',
        'n_hits': int(n_hits),
        'n_histogram_hits': int(hist.sum()),
        'nominal_v_m_per_s': nominal_velocity,
        'v_m_per_s': None,
        'v_m_per_s_error': None,
    }
    if hist.sum() < MIN_HITS:
        return result

    direction = np.sign(cathode - anode)
    plateau_low, plateau_high = sorted(
        (
            anode + direction * PLATEAU_MARGIN_ANODE_CM,
            cathode - direction * PLATEAU_MARGIN_CATHODE_CM,
        )
    )
    plateau_counts = hist[(centers >= plateau_low) & (centers <= plateau_high)]
    plateau = float(np.median(plateau_counts) if len(plateau_counts) >= 5 else np.max(hist))

    if plateau <= 0 or not np.isfinite(plateau):
        result['status'] = 'invalid_plateau'
        return result

    search_min, search_max = cathode - SEARCH_WINDOW_CM, cathode + SEARCH_WINDOW_CM
    search_mask = (centers >= search_min) & (centers <= search_max)
    if not np.any(search_mask):
        result['status'] = 'empty_search_range'
        return result

    bin_width = float(edges[1] - edges[0])
    smoothed = gaussian_filter1d(hist.astype(float), sigma=SMOOTH_SIGMA_CM / bin_width)
    positions = direction * (centers[search_mask] - cathode)
    order = np.argsort(positions)
    positions = positions[order]
    search_counts = smoothed[search_mask][order]
    threshold = plateau * threshold_fraction
    result.update(
        {
            'plateau_count': plateau,
            'threshold_count': float(threshold),
            'threshold_fraction': threshold_fraction,
            'search_range_cm': [float(search_min), float(search_max)],
            'smooth_sigma_cm': SMOOTH_SIGMA_CM,
        }
    )
    crossing, status = find_stable_crossing(positions, search_counts, threshold, bin_width)
    if status != 'ok':
        result['status'] = status
        return result

    boundary = cathode + direction * crossing
    geometric_drift = abs(cathode - anode)
    measured_drift = abs(boundary - anode)
    if measured_drift <= 0 or not np.isfinite(measured_drift):
        result['status'] = 'invalid_drift_length'
        return result

    scale = geometric_drift / measured_drift
    result.update(
        {
            'status': 'ok',
            'boundary_x_cm': float(boundary),
            'cathode_x_cm': float(cathode),
            'anode_x_cm': float(anode),
            'geometric_drift_length_cm': float(geometric_drift),
            'measured_drift_length_cm': float(measured_drift),
            'boundary_selection': 'stable_crossing',
            'velocity_ratio_percent': float((scale - 1.0) * 100.0),
            'v_m_per_s': float(nominal_velocity * scale),
        }
    )
    return result


def analyze_io(io_group, config, histogram, threshold_fraction=None):
    result = detect_boundary(
        histogram['counts'],
        histogram['edges'],
        io_group,
        config,
        histogram['n_hits'],
        threshold_fraction,
    )
    for axis in ('z', 'y'):
        spatial = histogram[axis]
        bin_results = []
        for index, counts in enumerate(spatial['counts']):
            n_hits = int(spatial['n_hits'][index])
            bin_result = detect_boundary(
                counts, histogram['edges'], io_group, config, n_hits, threshold_fraction
            )
            bin_result['bin_index'] = index + 1
            bin_result['range_cm'] = [
                float(spatial['edges'][index]),
                float(spatial['edges'][index + 1]),
            ]
            bin_results.append(bin_result)
        result[f'{axis}_bins'] = bin_results
    return result


def normalize_io_result(result, velocity_scale):
    velocity_scale = validate_velocity_scale(velocity_scale)
    normalized = copy.deepcopy(result)
    for current in (normalized, *normalized.get('y_bins', []), *normalized.get('z_bins', [])):
        raw_velocity = current.get('raw_v_m_per_s', current.get('v_m_per_s'))
        raw_error = current.get('raw_v_m_per_s_error', current.get('v_m_per_s_error'))
        current['raw_v_m_per_s'] = raw_velocity
        current['raw_v_m_per_s_error'] = raw_error
        current['v_m_per_s'] = (
            float(raw_velocity * velocity_scale) if raw_velocity is not None else None
        )
        current['v_m_per_s_error'] = (
            float(raw_error * velocity_scale) if raw_error is not None else None
        )
        if 'velocity_ratio_percent' in current:
            current['raw_velocity_ratio_percent'] = current.get(
                'raw_velocity_ratio_percent', current['velocity_ratio_percent']
            )
            current['velocity_ratio_percent'] = (
                float((current['v_m_per_s'] / current['nominal_v_m_per_s'] - 1.0) * 100.0)
                if current['v_m_per_s'] is not None
                else None
            )
    return normalized


def load_reference_velocity(detector):
    '''Normalization target: the calibrated v_drift in the detector's LArData.yaml [m/s]'''
    velocities = load_calibrated_vdrift(detector)
    if len(velocities) != 1:
        raise ValueError(f'Need exactly one calibrated vdrift for {detector}, got {velocities}')
    return velocities[0]


def normalization_metadata(velocity_scale, reference_velocity):
    return {
        'velocity_scale': validate_velocity_scale(velocity_scale),
        'reference_velocity_m_per_s': reference_velocity,
        'baseline': {
            'raw_mean_m_per_s': BASELINE_RAW_MEAN_M_PER_S,
            'source_file_count': BASELINE_FILE_COUNT,
            'successful_global_measurement_count': BASELINE_MEASUREMENT_COUNT,
            'start_timestamp': BASELINE_START_TIMESTAMP,
            'end_timestamp': BASELINE_END_TIMESTAMP,
            'threshold_fraction': 0.5,
            'weighting': 'equal_weight_per_successful_global_io_measurement',
        },
    }


def build_record(
    input_file,
    loaded,
    io_results,
    reference_velocity,
    threshold_fraction=THRESHOLD_FRACTION,
):
    timestamp = date_from_input_filename(input_file)
    # scale the raw velocities so the baseline period averages to the calibrated value
    velocity_scale = validate_velocity_scale(reference_velocity / BASELINE_RAW_MEAN_M_PER_S)
    if not math.isclose(threshold_fraction, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('The frozen normalization requires a 0.5 threshold fraction')
    io_results = {
        io: normalize_io_result(result, velocity_scale) for io, result in io_results.items()
    }
    # detect_boundary only sets 'ok' with a finite velocity, so both averages use the same TPCs
    ok_results = [result for result in io_results.values() if result.get('status') == 'ok']
    velocities = [result['v_m_per_s'] for result in ok_results]
    raw_velocities = [result['raw_v_m_per_s'] for result in ok_results]
    return {
        'timestamp': timestamp.isoformat() if timestamp is not None else None,
        'source_file': os.path.basename(input_file),
        'sample': f'{os.path.basename(loaded["hit_dset"])}_after_event_cuts',
        'normalization': normalization_metadata(velocity_scale, reference_velocity),
        'detection_parameters': {
            'threshold_fraction': validate_threshold_fraction(threshold_fraction),
            'nominal_x_bin_width_cm': X_BIN_WIDTH_CM,
            'smooth_sigma_cm': SMOOTH_SIGMA_CM,
            'search_half_width_cm': SEARCH_WINDOW_CM,
            'stability_gap_cm': STABILITY_GAP_CM,
            'stability_width_cm': STABILITY_WIDTH_CM,
            'stability_min_bins': STABILITY_MIN_BINS,
        },
        'total_events': int(loaded['total_events']),
        'selected_events': int(loaded['selected_events']),
        'rejected_events': int(loaded['rejected_events']),
        'io_groups': {str(io): io_results[io] for io in IO_GROUPS},
        'average_v_m_per_s': float(np.mean(velocities)) if velocities else None,
        'tpc_rms_m_per_s': float(np.std(velocities)) if velocities else None,
        'average_raw_v_m_per_s': float(np.mean(raw_velocities)) if raw_velocities else None,
        'raw_tpc_rms_m_per_s': float(np.std(raw_velocities)) if raw_velocities else None,
    }


def main(input_file, output_file_json, output_file_plot=None, detector=DEFAULT_DETECTOR):
    reference_velocity = load_reference_velocity(detector)
    print(f'Opening file: {input_file}')
    loaded = load_hit_histograms(input_file)
    io_results = {
        io: analyze_io(io, loaded['tpcs'][io], loaded['histograms'][io]) for io in IO_GROUPS
    }
    record = build_record(input_file, loaded, io_results, reference_velocity)
    update_json(output_file_json, record)
    print(f'Selected events: {loaded["selected_events"]}/{loaded["total_events"]}')
    print(f'Timestamp: {record["timestamp"]}, Velocity: {record["average_v_m_per_s"]} m/s')
    if output_file_plot is not None:
        plot_timeseries(output_file_json, output_file_plot, detector)


def build_parser():
    parser = argparse.ArgumentParser(description='Measure drift velocity from one FLOW file')
    parser.add_argument('--input_file', '--input-file', required=True, help='FLOW HDF5 file')
    parser.add_argument(
        '--output_file_json', '--output-file-json', required=True, help='JSON history'
    )
    parser.add_argument('--output_file_plot', '--output-file-plot', help='Time-series plot')
    parser.add_argument(
        '--detector',
        choices=DETECTORS,
        default=DEFAULT_DETECTOR,
        help='Selects LArData.yaml for the reference velocity',
    )
    return parser


if __name__ == '__main__':
    main(**vars(build_parser().parse_args()))
