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

CENTRAL = ZoneInfo('America/Chicago')
IO_GROUPS = tuple(range(1, 9))
REFERENCE_VELOCITY_M_PER_S = 1584.0
RECENT_RECORD_COUNT = 50
PLOT_TITLE = '2x2 Drift Velocity'


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


def draw_static(series, output_file, title):
    fig, ax = plt.subplots(figsize=(10, 5))

    for io in IO_GROUPS:
        timestamps = series[io]['timestamps']
        if timestamps:
            ax.plot(timestamps, series[io]['velocities'], 'o-', label=f'IO {io}')

    ax.axhline(
        REFERENCE_VELOCITY_M_PER_S,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label=f'Reference: {REFERENCE_VELOCITY_M_PER_S:g} m/s',
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


def draw_interactive(data, output_file):
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

    fig.add_hline(
        y=REFERENCE_VELOCITY_M_PER_S,
        line_color='black',
        line_dash='dash',
        line_width=1.5,
        annotation_text=f'Reference: {REFERENCE_VELOCITY_M_PER_S:g} m/s',
        annotation_position='top left',
    )
    fig.update_layout(
        title=PLOT_TITLE,
        xaxis_title='Timestamp [CT]',
        yaxis_title='Drift velocity [m/s]',
        template='plotly_white',
        width=1000,
        height=500,
    )
    fig.update_xaxes(tickformat='%m/%d/%Y<br>%H:%M CT')
    fig.write_html(f'{output_file}.html')


def main(input_file, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data = load_points(input_file)

    draw_static(extract_series(data), output_file, PLOT_TITLE)
    draw_static(
        extract_series(data[-RECENT_RECORD_COUNT:]),
        f'{output_file}_last.png',
        f'{PLOT_TITLE} last {RECENT_RECORD_COUNT} points',
    )
    draw_interactive(data, output_file)

    for path in (output_file, f'{output_file}_last.png', f'{output_file}.html'):
        print(f'Saved: {path}')


def build_parser():
    parser = argparse.ArgumentParser(description='Plot drift-velocity JSON history')
    parser.add_argument('--input_file', '--input-file', required=True, help='JSON history')
    parser.add_argument('--output_file', '--output-file', required=True, help='Time-series plot')
    return parser


if __name__ == '__main__':
    main(**vars(build_parser().parse_args()))
