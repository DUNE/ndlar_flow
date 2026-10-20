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

from vdrift_config import DEFAULT_DETECTOR, DETECTORS, IO_GROUPS, load_calibrated_vdrift

CENTRAL = ZoneInfo('America/Chicago')
RECENT_RECORD_COUNT = 50
AVERAGE_LABEL = 'TPC average'


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
    '''Per-TPC series plus the TPC average (AVERAGE_LABEL), each with timestamps and
    normalized and raw velocities'''
    series = {
        name: {'timestamps': [], 'velocities': [], 'raw_velocities': []}
        for name in (*IO_GROUPS, AVERAGE_LABEL)
    }

    def append(name, velocity, raw_velocity, timestamp):
        series[name]['timestamps'].append(timestamp)
        series[name]['velocities'].append(float(velocity))
        series[name]['raw_velocities'].append(float(raw_velocity))

    for entry in data:
        if entry.get('timestamp') is None:
            continue
        timestamp = parse_timestamp(entry['timestamp'])
        if entry.get('average_v_m_per_s') is not None:
            append(
                AVERAGE_LABEL, entry['average_v_m_per_s'], entry['average_raw_v_m_per_s'], timestamp
            )
        for io in IO_GROUPS:
            result = entry.get('io_groups', {}).get(str(io), {})
            if result.get('status') == 'ok':
                append(io, result['v_m_per_s'], result['raw_v_m_per_s'], timestamp)

    return series


def draw_static(series, reference_velocities, output_file, title):
    fig, ax = plt.subplots(figsize=(10, 5))

    # TPCs thin and faded so the average stands out
    for io in IO_GROUPS:
        if series[io]['timestamps']:
            ax.plot(
                series[io]['timestamps'],
                series[io]['velocities'],
                'o-',
                markersize=2,
                linewidth=0.7,
                alpha=0.4,
                label=f'IO {io}',
            )
    average = series[AVERAGE_LABEL]
    if average['timestamps']:
        ax.plot(
            average['timestamps'],
            average['velocities'],
            'o-',
            color='black',
            markersize=3,
            linewidth=2.5,
            label=AVERAGE_LABEL,
        )

    for velocity in reference_velocities:
        ax.axhline(
            velocity,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label=f'Calibrated v_drift: {velocity:g} m/s',
        )
    ax.set_xlabel('Timestamp [CT]')
    ax.set_ylabel('Drift velocity [m/s]')
    ax.set_title(title)
    ax.grid(True)
    # legend order: calibrated value, average, then TPCs
    handles, labels = ax.get_legend_handles_labels()
    rank = {'Calibrated': 0, AVERAGE_LABEL: 1}
    order = sorted(range(len(labels)), key=lambda i: rank.get(labels[i].split(' v_drift')[0], 2))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y\n%H:%M CT', tz=CENTRAL))
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches='tight')
    plt.close(fig)


def draw_interactive(series, reference_velocities, output_file, title):
    fig = go.Figure()

    # TPCs start hidden (click the legend to show); the average is always drawn
    traces = [(io, f'IO {io}', 'legendonly', {}) for io in IO_GROUPS]
    traces.append((AVERAGE_LABEL, AVERAGE_LABEL, True, {'color': 'black', 'width': 3}))
    for name, label, visible, line in traces:
        if not series[name]['timestamps']:
            continue
        fig.add_trace(
            go.Scatter(
                x=series[name]['timestamps'],
                y=series[name]['velocities'],
                mode='lines+markers',
                name=label,
                visible=visible,
                line=line,
                legendrank=2 if name == AVERAGE_LABEL else 3,
                customdata=series[name]['raw_velocities'],
                hovertemplate=(
                    '%{x}<br>Velocity: %{y:.2f} m/s'
                    '<br>Raw: %{customdata:.2f} m/s<extra>%{fullData.name}</extra>'
                ),
            )
        )

    # calibrated value as a legend entry spanning the plotted time range
    timestamps = [t for name in series for t in series[name]['timestamps']]
    for velocity in reference_velocities if timestamps else []:
        fig.add_trace(
            go.Scatter(
                x=[min(timestamps), max(timestamps)],
                y=[velocity, velocity],
                mode='lines',
                name=f'Calibrated v_drift: {velocity:g} m/s',
                line={'color': 'red', 'dash': 'dash', 'width': 1.5},
                legendrank=1,
                hoverinfo='skip',
            )
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

    series = extract_series(data)
    draw_static(series, reference_velocities, output_file, title)
    draw_static(
        extract_series(data[-RECENT_RECORD_COUNT:]),
        reference_velocities,
        f'{output_file}_last.png',
        f'{title} last {RECENT_RECORD_COUNT} points',
    )
    draw_interactive(series, reference_velocities, output_file, title)

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
