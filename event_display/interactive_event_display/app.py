"""
A Dash app to display events in the 2x2 experiment
"""

# Import the required libraries
import dash_uploader as du
import plotly.graph_objects as go
import atexit
import shutil

from dash import dcc
from dash import html
from datetime import datetime

# from dash import no_update
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Output, DashProxy, Input, State

from display_utils import (
    parse_contents,
    parse_minerva_contents,
    is_beam_event,
    create_3d_figure,
    plot_waveform,
    plot_charge,
    plot_2d_charge,
    plot_charge_energy,
)

from os.path import basename
from pathlib import Path

# Settings and constants
UPLOAD_FOLDER_ROOT = "cache"

# Create the app
app = DashProxy(__name__, title="2x2 event display")
du.configure_upload(app, UPLOAD_FOLDER_ROOT)  # without this upload will not work

# App layout
app.layout = html.Div(
    [
        # Hidden divs to store data
        dcc.Location(id="url"),
        dcc.Store(id="filename", storage_type="local", data=""),
        dcc.Store(id="minerva-filename", storage_type="local", data=""),
        dcc.Store(id="event-id", data=0),
        dcc.Store(id="minerva-event-id", data=0),
        dcc.Store(id="data-length", data=0),
        dcc.Store(id="minerva-data-length", data=0),
        dcc.Store(id="event-time", data=0),
        dcc.Store(id="sim-version", data=0),
        # Header
        html.Div(
            [
                # Text and buttons on the left
                html.Div(
                    [
                        # Header
                        html.H2(
                            children="2x2 event display", style={"textAlign": "left"}
                        ),
                        html.Div(
                            children="", id="filename-div", style={"textAlign": "left"}
                        ),
                        html.Div(
                            children="",
                            id="minerva-filename-div",
                            style={"textAlign": "left"},
                        ),
                        # File input
                        dcc.Input(
                            id="file-path",
                            type="text",
                            placeholder="URL or NERSC path",
                            debounce=True,
                        ),
                        html.Button("Load Flow File", id="load-button", n_clicks=0),
                        dcc.Input(
                            id="minerva-file-path",
                            type="text",
                            placeholder="URL or NERSC path",
                            debounce=True,
                        ),
                        html.Button(
                            "Load Minerva File", id="load-minerva-button", n_clicks=0
                        ),
                        # Event ID input box
                        dcc.Input(
                            id="input-evid",
                            type="number",
                            placeholder="0",
                            debounce=True,
                            style={
                                "width": "6em",
                                "display": "inline-block",
                                "margin-right": "0.5em",
                                "margin-left": "0.5em",
                            },
                        ),
                        # Event ID buttons
                        html.Button("Previous Event", id="prev-button", n_clicks=0),
                        html.Button("Next Event", id="next-button", n_clicks=0),
                        html.Button("Next Beam Event", "beam-button", n_clicks=0),
                        html.Div(
                            [
                                html.Div(id="evid-div", style={"textAlign": "center", "display": "inline-block", "margin-right": "10px"}),
                                html.Div(children="", id="event-time-div", style={"display": "inline-block"}),
                            ]
                        ),
                    ],
                    style={"display": "inline-block"},
                ),
                # Logo on the right
                html.Div(
                    [
                        html.Img(src="assets/2x2logo.png", style={"height": "5vh"}),
                        html.Img(src="assets/DUNElogo.png", style={"height": "5vh"}),
                    ],
                    style={"display": "inline-block"},
                ),
            ],
            style={
                "display": "flex",
                "justify-content": "space-between",
                "align-items": "flex-start",
            },
        ),
        # Graphs
        html.Div(
            [
                # First row
                html.Div(
                    [
                        # Large 3D graph on the left
                        html.Div(
                            dcc.Graph(
                                id="3d-graph",
                                style={
                                    "height": "65vh",
                                    "width": "68vw",
                                    "float": "left",
                                },
                            ),
                        ),
                        # 2D plots on the right
                        html.Div(
                            dcc.Graph(id="2d-plots"),
                            style={"height": "65vh", "width": "28vw", "float": "right"},
                        ),
                    ],
                    style={"display": "flex"},
                ),
                # Second row
                html.Div(
                    [
                        # Light waveform graph on the left
                        html.Div(
                            dcc.Graph(
                                id="light-waveform",
                                style={
                                    "height": "25vh",
                                    "width": "29vw",
                                    "float": "left",
                                },
                            ),
                        ),
                        # Charge graph in the middle
                        html.Div(
                            dcc.Graph(
                                id="charge-hist",
                                style={
                                    "height": "25vh",
                                    "width": "29vw",
                                    "float": "left",
                                },
                            ),
                        ),
                        # Placeholder for the charge-energy graph on the right
                        html.Div(
                            dcc.Graph(
                                id="charge-energy-hist",
                                style={
                                    "height": "20vh",
                                    "width": "29vw",
                                    "float": "right",
                                },
                            ),
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justify-content": "space-between",
                        "align-items": "flex-start",
                    },
                ),
            ],
        ),
    ],
)

def resolve_url(url_or_path):
    prefix = 'https://portal.nersc.gov/project/dune/data'
    base_path = '/global/cfs/cdirs/dune/www/data'
    if not url_or_path.startswith('https://'):
        return url_or_path
    url = url_or_path
    assert(url.startswith(prefix))
    return base_path + url[len(prefix):]

# Callbacks


# Callback to handle file selection from path
# ============================================
@app.callback(
    Input("load-button", "n_clicks"),
    Input("file-path", "value"),
    [
        Output("filename", "data", allow_duplicate=True),
        Output("filename-div", "children", allow_duplicate=True),
        Output("event-id", "data", allow_duplicate=True),
        Output("data-length", "data", allow_duplicate=True),
    ],
    prevent_initial_call=True,
)
def load_file(n, file_path):
    file_path = resolve_url(file_path)
    print(file_path)
    if n > 0 and file_path is not None:
        _, num_events = parse_contents(file_path)
        return file_path, file_path, 0, num_events
    else:
        return None, None, None, None


@app.callback(
    Input("load-minerva-button", "n_clicks"),
    Input("minerva-file-path", "value"),
    [
        Output("minerva-filename", "data", allow_duplicate=True),
        Output("minerva-filename-div", "children", allow_duplicate=True),
        Output("minerva-event-id", "data", allow_duplicate=True),
        Output("minerva-data-length", "data", allow_duplicate=True),
    ],
    prevent_initial_call=True,
)
def load_minerva(n, minerva_file_path):
    minerva_file_path = resolve_url(minerva_file_path)
    print(minerva_file_path)
    if n > 0 and minerva_file_path is not None:
        _, minerva_num_events = parse_minerva_contents(minerva_file_path)
        return minerva_file_path, minerva_file_path, 0, minerva_num_events
    else:
        return None, None, None, None

# Callbacks to handle the event ID and time display
# ==================================================
@app.callback(
    Output("event-id", "data", allow_duplicate=True),
    Input("next-button", "n_clicks"),
    State("event-id", "data"),
    State("data-length", "data"),
    prevent_initial_call=True,
)
def increment(n, evid, max_value):
    """Increment the event ID with the button"""
    if n > 0:
        new_evid = evid + 1
        if new_evid > max_value:  # wrap around
            return 0
        else:
            return new_evid


@app.callback(
    Output("event-id", "data", allow_duplicate=True),
    Input("prev-button", "n_clicks"),
    State("event-id", "data"),
    State("data-length", "data"),
    prevent_initial_call=True,
)
def decrement(n, evid, max_value):
    """Decrement the event ID with the button"""
    if n > 0:
        if evid > 0:
            return evid - 1
        return max_value - 1  # wrap around


@app.callback(
    Output("event-id", "data", allow_duplicate=True),
    Input("input-evid", "value"),
    State("data-length", "data"),
    prevent_initial_call=True,
)
def set_evid(value, max_value):
    """Set the event ID in the input box"""
    if value is not None:
        if value > max_value:  # not possible to go higher than max value
            return max_value
        else:
            return value


@app.callback(
    Output("event-id", "data", allow_duplicate=True),
    Input("beam-button", "n_clicks"),
    Input("filename", "data"),
    State("event-id", "data"),
    State("data-length", "data"),
    prevent_initial_call=True,
)
def next_beam_event(n, filename, evid, max_value):
    """Increment the event ID with the button"""
    if n > 0:
        new_evid = (evid + 1) % max_value # wrap around
        # check if beam event
        # otherwise increment until beam event
        while new_evid < max_value:
            if is_beam_event(new_evid, filename):
                return new_evid
            new_evid = (new_evid + 1) % max_value
        return new_evid
    else:
        return 0


@app.callback(
    Output("evid-div", "children"),
    Input("event-id", "data"),
    State("data-length", "data"),
)
def update_div(evid, max_value):
    """Update the event ID display"""
    return f"Event ID: {evid}/{max_value}"


@app.callback(
    Output("event-time-div", "children"),
    Input("event-id", "data"),
    State("event-time", "data"),
)
def update_time(_, time):
    """Update the time display"""
    return f"Event time: {time}"


# Callback to display the event
# =============================
@app.callback(
    Output("3d-graph", "figure"),
    Output("sim-version", "data"),
    Output("event-time", "data"),
    Input("filename", "data"),
    Input("minerva-filename", "data"),
    Input("event-id", "data"),
    prevent_initial_call=True,
)
def update_graph(filename, minerva_filename, evid):
    """Update the 3D graph when the event ID is changed"""
    if minerva_filename is not None:
        minerva_data, _ = parse_minerva_contents(minerva_filename)
    else:
        minerva_data = None
    if filename is not None:
        data, _ = parse_contents(filename)
        graph, sim_version = create_3d_figure(minerva_data, data, filename, evid)
        event_datetime = datetime.utcfromtimestamp(
            data["charge/events", evid]["unix_ts"][0]
        ).strftime("%Y-%m-%d %H:%M:%S")
    else:
        raise PreventUpdate
    return (
        graph,
        sim_version,
        event_datetime,
    )  # TODO: move to utils


@app.callback(
    Output("2d-plots", "figure"),
    Input("filename", "data"),
    Input("event-id", "data"),
    prevent_initial_call=True,
)
def update_2d_plots(filename, evid):
    if filename is not None:
        data, _ = parse_contents(filename)
        return plot_2d_charge(data, evid)
    return go.Figure()


@app.callback(
    Input("filename", "data"),
    Input("event-id", "data"),
    Input("sim-version", "data"),
    Input("3d-graph", "figure"),
    Input("3d-graph", "clickData"),
    Output("light-waveform", "figure"),
    prevent_initial_call=True,
)
def update_light_waveform(filename, evid, sim_version, graph, click_data):
    """Update the light waveform graph when the 3D graph is clicked"""
    if click_data:
        curvenum = int(click_data["points"][0]["curveNumber"])
        try:
            det_id = int(
                graph["data"][curvenum]["ids"][0][0].split("_")[2]
            )  # det_id_{det_id}_tpc_{tpc}
            tpc_id = int(graph["data"][curvenum]["ids"][0][0].split("_")[4])
            opid = (det_id, tpc_id)
            if filename is not None:
                data, _ = parse_contents(filename)
                return plot_waveform(data, evid, opid, sim_version)
        except Exception as e:
            print("That is not a light trap, no waveform to plot")
    else:
        det_id = 0
        tpc_id = 0
        opid = (det_id, tpc_id)
        if filename is not None:
            data, _ = parse_contents(filename)
            return plot_waveform(data, evid, opid, sim_version)
    return go.Figure()


@app.callback(
    Input("filename", "data"),
    Input("event-id", "data"),
    Output("charge-hist", "figure"),
    prevent_initial_call=True,
)
def update_charge_histogram(filename, evid):
    """Update the charge graph when the event ID is changed"""
    if filename is not None:
        data, _ = parse_contents(filename)
        return plot_charge(data, evid)
    return go.Figure()

@app.callback(
    Output("charge-energy-hist", "figure"),
    Input("filename", "data"),
    Input("event-id", "data"),
    prevent_initial_call=True,
)
def update_charge_energy_histogram(filename, evid):
    """Update the charge-energy graph when the event ID is changed"""
    if filename is not None:
        data, _ = parse_contents(filename)
        return plot_charge_energy(data, evid)
    return go.Figure()


# Cleaning up
# ===========
@atexit.register
def clean_cache():
    """Delete uploaded files"""
    try:
        shutil.rmtree(Path(UPLOAD_FOLDER_ROOT))
    except OSError as err:
        print("Can't clean %s : %s" % (UPLOAD_FOLDER_ROOT, err.strerror))


# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8080)
