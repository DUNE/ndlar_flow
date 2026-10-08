import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use('Agg')

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yaml

CENTRAL = ZoneInfo('America/Chicago')
IO_GROUPS = tuple(range(1, 9))
RECENT_RECORD_COUNT = 50
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
    '''Calibrated drift velocity used by flow, drawn as the plot reference [m/s]'''
    yaml_file = YAML_DIR / DETECTORS[detector][0] / 'resources/LArData.yaml'
    with open(yaml_file, encoding='utf-8') as f:
        vdrift = yaml.safe_load(f)['params'].get('vdrift')  # mm/us, one value or one per module
    if not vdrift:
        print(f'No vdrift set in {yaml_file} (computed from E field); no reference line drawn')
        return []
    return sorted({float(v) * 1e3 for v in vdrift})


def parse_timestamp(value):
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=CENTRAL)
    return timestamp.astimezone(CENTRAL)


def load_points(input_file):
    with open(input_file, encoding='utf-8') as json_file:
        data = json.load(json_file)['vdrifts']

    data.sort(
        key=lambda entry: (
            entry.get('timestamp') is None,
            entry.get('timestamp') or '',
        )
    )
    return data


def extract_series(data):
    series = {io: {'timestamps': [], 'velocities': [], 'raw_velocities': []} for io in IO_GROUPS}

    for entry in data:
        timestamp_value = entry.get('timestamp')
        if timestamp_value is None:
            continue

        timestamp = parse_timestamp(timestamp_value)
        io_groups = entry.get('io_groups', {})

        for io in IO_GROUPS:
            result = io_groups.get(str(io), {})
            if result.get('status') != 'ok' or result.get('v_m_per_s') is None:
                continue

            series[io]['timestamps'].append(timestamp)
            series[io]['velocities'].append(float(result['v_m_per_s']))
            series[io]['raw_velocities'].append(result.get('raw_v_m_per_s', result['v_m_per_s']))

    return series


def draw_static(series, reference_velocities, output_file, title):
    fig, ax = plt.subplots(figsize=(10, 5))

    for io in IO_GROUPS:
        timestamps = series[io]['timestamps']
        if timestamps:
            ax.plot(timestamps, series[io]['velocities'], 'o-', label=f'IO {io}')

    for velocity in reference_velocities:
        ax.axhline(
            velocity,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label=f'Calibrated v_drift: {velocity:g} m/s',
        )
    ax.set_xlabel('Timestamp [CT]')
    ax.set_ylabel('Drift velocity [m/s]')
    ax.set_title(title)
    ax.grid(True)
    ax.legend(ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y\n%H:%M CT', tz=CENTRAL))
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches='tight')
    plt.close(fig)


def draw_interactive(data, reference_velocities, output_file, title):
    series = extract_series(data)
    fig = go.Figure()

    for io in IO_GROUPS:
        timestamps = series[io]['timestamps']
        if timestamps:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=series[io]['velocities'],
                    mode='lines+markers',
                    name=f'IO {io}',
                    customdata=series[io]['raw_velocities'],
                    hovertemplate=(
                        '%{x}<br>Velocity: %{y:.2f} m/s'
                        '<br>Raw: %{customdata:.2f} m/s<extra>%{fullData.name}</extra>'
                    ),
                )
            )

    for velocity in reference_velocities:
        fig.add_hline(
            y=velocity,
            line_color='black',
            line_dash='dash',
            line_width=1.5,
            annotation_text=f'Calibrated v_drift: {velocity:g} m/s',
            annotation_position='top left',
        )
    fig.update_layout(
        title=title,
        xaxis_title='Timestamp [CT]',
        yaxis_title='Drift velocity [m/s]',
        template='plotly_white',
        width=1000,
        height=500,
    )
    fig.update_xaxes(tickformat='%m/%d/%Y<br>%H:%M CT')
    fig.write_html(f'{output_file}.html')


def main(input_file, output_file, detector=DEFAULT_DETECTOR):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data = load_points(input_file)
    reference_velocities = load_calibrated_vdrift(detector)
    title = f'{DETECTORS[detector][1]} Drift Velocity'

    draw_static(extract_series(data), reference_velocities, output_file, title)
    draw_static(
        extract_series(data[-RECENT_RECORD_COUNT:]),
        reference_velocities,
        f'{output_file}_last.png',
        f'{title} last {RECENT_RECORD_COUNT} points',
    )
    draw_interactive(data, reference_velocities, output_file, title)

    for path in (output_file, f'{output_file}_last.png', f'{output_file}.html'):
        print(f'Saved: {path}')


def build_parser():
    parser = argparse.ArgumentParser(description='Plot drift-velocity JSON history')
    parser.add_argument('--input_file', '--input-file', required=True, help='JSON history')
    parser.add_argument('--output_file', '--output-file', required=True, help='Time-series plot')
    parser.add_argument(
        '--detector', choices=DETECTORS, default=DEFAULT_DETECTOR, help='Selects LArData.yaml'
    )
    return parser


if __name__ == '__main__':
    main(**vars(build_parser().parse_args()))
