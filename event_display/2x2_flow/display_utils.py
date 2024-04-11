"""
Utility functions for displaying data in the app
"""

# TODO: remove dependency on h5flow
import h5flow
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go


def parse_contents(filename):
    data = h5flow.data.H5FlowDataManager(filename, "r")
    num_events = data["charge/events/data"].shape[0]
    return data, num_events


def create_3d_figure(data, evid):
    fig = go.Figure()
    # Select the hits for the current event
    prompthits_ev = data["charge/events", "charge/calib_prompt_hits", evid]
    finalhits_ev = data["charge/events", "charge/calib_final_hits", evid]
    # select the segments (truth) for the current event
    try:
        prompthits_segs = data[
            "charge/events",
            "charge/calib_prompt_hits",
            "charge/packets",
            "mc_truth/segments",  # called segments in minirun4
            evid,
        ]
        sim_version = "minirun5"
        print("Found truth info in minirun5 format")
    except:
        print("No truth info in minirun4 format found")
        try:
            prompthits_segs = data[
                "charge/events",
                "charge/calib_prompt_hits",
                "charge/packets",
                "mc_truth/tracks",  # called tracks in minirun3
                evid,
            ]
            sim_version = "minirun3"
            print("Found truth info in minirun3 format")
        except:
            print("No truth info in minirun3 format found")
            prompthits_segs = None

    # Plot the prompt hits
    prompthits_traces = go.Scatter3d(
        x=prompthits_ev.data["x"].flatten(),
        y=(prompthits_ev.data["y"].flatten()),
        z=(prompthits_ev.data["z"].flatten()),
        marker_color=prompthits_ev.data["E"].flatten(),  # convert to MeV from GeV for minirun4, not sure for minirun3
        marker={
            "size": 1.75,
            "opacity": 0.7,
            "colorscale": "cividis",
            "colorbar": {
                "title": "Hit energy [MeV]",
                "titlefont": {"size": 12},
                "tickfont": {"size": 10},
                "thickness": 15,
                "len": 0.5,
                "xanchor": "left",
                "x": 0,
            },
        },
        name="prompt hits",
        mode="markers",
        showlegend=True,
        opacity=0.7,
        customdata=prompthits_ev.data["E"].flatten(),
        hovertemplate="<b>x:%{x:.3f}</b><br>y:%{y:.3f}<br>z:%{z:.3f}<br>E:%{customdata:.3f}",
    )
    fig.add_traces(prompthits_traces)

    # Plot the final hits
    finalhits_traces = go.Scatter3d(
        x=finalhits_ev.data["x"].flatten(),
        y=(finalhits_ev.data["y"].flatten()),
        z=(finalhits_ev.data["z"].flatten()),
        marker_color=finalhits_ev.data["E"].flatten(),
        marker={
            "size": 1.75,
            "opacity": 0.7,
            "colorscale": "Plasma",
            "colorbar": {
                "title": "Hit energy [MeV]",
                "titlefont": {"size": 12},
                "tickfont": {"size": 10},
                "thickness": 15,
                "len": 0.5,
                "xanchor": "left",
                "x": 0,
            },
        },
        name="final hits",
        mode="markers",
        visible="legendonly",
        showlegend=True,
        opacity=0.7,
        customdata=finalhits_ev.data["E"].flatten(),
        hovertemplate="<b>x:%{x:.3f}</b><br>y:%{y:.3f}<br>z:%{z:.3f}<br>E:%{customdata:.3f}",
        # render_mode="svg",
    )
    fig.add_traces(finalhits_traces)

    if prompthits_segs is not None:
        segs_traces = plot_segs(
            prompthits_segs[0, :, 0, 0],
            sim_version=sim_version,
            mode="lines",
            name="edep segments",
            visible="legendonly",
            line_color="red",
            showlegend=True,
        )
        fig.add_traces(segs_traces)

    # Draw the TPC
    tpc_center, anodes, cathodes = draw_tpc(sim_version)
    light_detectors = draw_light_detectors(data, evid)

    fig.add_traces(tpc_center)
    fig.add_traces(anodes)
    fig.add_traces(cathodes)
    fig.add_traces(light_detectors)

    return fig


def plot_segs(segs, sim_version="minirun5", **kwargs):
    def to_list(axis):
        if sim_version == "minirun5" or sim_version == "minirun4":
            nice_array = np.column_stack(
                [segs[f"{axis}_start"], segs[f"{axis}_end"], np.full(len(segs), None)]
            ).flatten()
        if sim_version == "minirun3":
            nice_array = np.column_stack(
                [
                    segs[f"{axis}_start"] * 10,
                    segs[f"{axis}_end"] * 10,
                    np.full(len(segs), None),
                ]
            ).flatten()
        return nice_array

    x, y, z = (to_list(axis) for axis in "xyz")

    trace = go.Scatter3d(x=x, y=y, z=z, **kwargs)

    return trace


def draw_tpc(sim_version="minirun5"):
    anode_xs = np.array([-63.931, -3.069, 3.069, 63.931])
    anode_ys = np.array([-19.8543, 103.8543])  # two ys
    anode_zs = np.array([-64.3163, -2.6837, 2.6837, 64.3163])  # four zs
    if sim_version == "minirun4":  # hit coordinates are in cm
        detector_center = (0, -268, 1300)
        anode_ys = anode_ys - (268 + 42)
        anode_zs = anode_zs + 1300
    if sim_version == "minirun3":  # hit coordinates are in mm
        detector_center = (0, 42 * 10, 0)
        anode_xs = anode_xs * 10
        anode_ys = anode_ys * 10
        anode_zs = anode_zs * 10
    if sim_version == "minirun5":  # hit coordinates are in cm
        detector_center = (0, 0, 0)
        anode_ys = anode_ys - 42

    center = go.Scatter3d(
        x=[detector_center[0]],
        y=[detector_center[1]],
        z=[detector_center[2]],
        marker=dict(size=3, color="green", opacity=0.5),
        mode="markers",
        name="tpc center",
    )
    anodes = draw_anode_planes(
        anode_xs, anode_ys, anode_zs, colorscale="ice", showscale=False, opacity=0.1
    )
    cathodes = draw_cathode_planes(
        anode_xs, anode_ys, anode_zs, colorscale="burg", showscale=False, opacity=0.1
    )
    return center, anodes, cathodes


def draw_cathode_planes(x_boundaries, y_boundaries, z_boundaries, **kwargs):
    traces = []
    for i_z in range(int(len(z_boundaries) / 2)):
        for i_x in range(int(len(x_boundaries) / 2)):
            z, y = np.meshgrid(
                np.linspace(z_boundaries[i_z * 2], z_boundaries[i_z * 2 + 1], 2),
                np.linspace(y_boundaries.min(), y_boundaries.max(), 2),
            )
            x = (
                (x_boundaries[i_x * 2] + x_boundaries[i_x * 2 + 1])
                * 0.5
                * np.ones(z.shape)
            )
            traces.append(go.Surface(x=x, y=y, z=z, **kwargs))

    return traces


def draw_anode_planes(x_boundaries, y_boundaries, z_boundaries, **kwargs):
    traces = []
    for i_z in range(int(len(z_boundaries) / 2)):
        for i_x in range(int(len(x_boundaries))):
            z, y = np.meshgrid(
                np.linspace(z_boundaries[i_z * 2], z_boundaries[i_z * 2 + 1], 2),
                np.linspace(y_boundaries.min(), y_boundaries.max(), 2),
            )
            x = x_boundaries[i_x] * np.ones(z.shape)

            traces.append(go.Surface(x=x, y=y, z=z, **kwargs))

    return traces


def draw_light_detectors(data, evid):
    try:
        data["charge/events", "light/events", "light/wvfm", evid]
    except:
        print("No light information found, not plotting light detectors")
        return []
    
    match_light = match_light_to_charge_event(data, evid)
    if match_light is None:
        print(
            f"No light event matches found for charge event {evid}, not plotting light detectors"
        )
        return []

    waveforms_all_detectors = get_waveforms_all_detectors(match_light)

    drawn_objects = []
    drawn_objects.extend(plot_light_traps(data, waveforms_all_detectors))

    return drawn_objects


def match_light_to_charge_event(data, evid):
    """
    Match the light events to the charge event by looking at proximity in time.
    Use unix time for this, since it should refer to the same time in both readout systems.
    For now we just take all the light within 1s from the charge event time.
    """

    match_light = data["charge/events", "light/events", "light/wvfm", evid]

    return match_light


def get_waveforms_all_detectors(match_light):
    """
    Get the light waveforms for the matched light events.
    """
    n_matches = match_light["samples"].shape[1]
    waveforms_all_detectors = match_light['samples'].reshape(n_matches,8,64,1000)
    return waveforms_all_detectors


def plot_light_traps(data, waveforms_all_detectors):
    """Plot optical detectors"""
    drawn_objects = []

    det_bounds = data["/geometry_info/det_bounds/data"]

    channel_map = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 65, 73, 81, 89, 97, 105, 113, 121, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26,
                34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122, 67, 75, 83, 91, 99, 107, 115, 123, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52,
                  60, 68, 76, 84, 92, 100, 108, 116, 124, 69, 77, 85, 93, 101, 109, 117, 125, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 70,
                   78, 86, 94, 102, 110, 118, 126, 71, 79, 87, 95, 103, 111, 119, 127, 7, 15, 23, 31, 39, 47, 55, 63]) # this maps detector position to detector number
    # we need to invert the mapping because I'm stupid
    channel_map = np.argsort(channel_map)
    channel_map_deluxe = pd.read_csv('sipm_channel_map.csv')
    xs = []
    ys = []
    zs = []
    for i in range(len(det_bounds)):
        if det_bounds[i][1] == True:
            xs.append([det_bounds[i][0][0][0], det_bounds[i][0][1][0]])
            ys.append([det_bounds[i][0][0][1], det_bounds[i][0][1][1]])
            zs.append([det_bounds[i][0][0][2], det_bounds[i][0][1][2]])

    COLORSCALE = plotly.colors.make_colorscale(
        plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.YlOrRd)[0]
    )
    photon_sums = []
    for i in range(len(xs)):
        opid = channel_map[i]
        # get all adc, channel belonging to opid=det_id. We have a numpy array channel_map_deluxe with dtype (det_id, tpc, side, sipm_pos, adc, channel)
        sipms = channel_map_deluxe[channel_map_deluxe['det_id']==opid][['adc', 'channel']].values
        sum_photons = 0
        for adc, channel in sipms:
            wvfm = waveforms_all_detectors[:, adc, channel, :]
            sum_wvfm = np.sum(wvfm, axis=0) # sum over the events
            sum_photons += np.sum(sum_wvfm, axis=0) # sum over the time
        photon_sums.append(sum_photons)
    max_integral = np.max(photon_sums)
    
    for i in range(len(xs)):
        opid = channel_map[i]
        # get all adc, channel belonging to opid=det_id. We have a numpy array channel_map_deluxe with dtype (det_id, tpc, side, sipm_pos, adc, channel)
        sipms = channel_map_deluxe[channel_map_deluxe['det_id']==opid][['adc', 'channel']].values
        sum_photons = 0
        for adc, channel in sipms:
            wvfm = waveforms_all_detectors[:, adc, channel, :]
            sum_wvfm = np.sum(wvfm, axis=0)
            sum_photons += np.sum(sum_wvfm, axis=0) 
        opid_str = f"opid_{opid}"
        light_color = [
            [
                0.0,
                get_continuous_color(
                    COLORSCALE, intermed=max(0, sum_photons/max_integral)
                ),
            ],
            [
                1.0,
                get_continuous_color(
                    COLORSCALE, intermed=max(0, sum_photons/max_integral)
                ),
            ],
        ]
        light_plane = go.Surface(
            x=xs[i],
            y=ys[i],
            z=[zs[i], zs[i]],
            colorscale=light_color,
            showscale=False,
            showlegend=False,
            opacity=0.2,
            hoverinfo="text",
            ids=[[opid_str, opid_str], [opid_str, opid_str]],
            text=f"Optical detector {opid} waveform integral<br>{max(0,sum_photons):.2e}",
        )

        drawn_objects.append(light_plane)

    return drawn_objects


def plot_waveform(data, evid, opid):
    try:
        charge = data["charge/events", evid][["id", "unix_ts"]]
        num_light = data["light/events/data"].shape[0]
        light = data["light/events", slice(0, num_light)][
            ["id", "utime_ms"]
        ]  # we have to try them all, events may not be time ordered
    except:
        print("No light information found, not plotting light waveform")
        return []

    match_light = match_light_to_charge_event(data, evid)

    if match_light is None:
        print(
            f"No light event matches found for charge event {evid}, not plotting light waveform"
        )
        return []

    fig = go.Figure()
    waveforms_all_detectors = get_waveforms_all_detectors( match_light)

    channel_map_deluxe = pd.read_csv('sipm_channel_map.csv')
    sipms = channel_map_deluxe[channel_map_deluxe['det_id']==opid][['adc', 'channel']].values
    
    sum_wvfm = np.array([0]*1000)
    for adc, channel in sipms:
        wvfm = waveforms_all_detectors[:, adc, channel, :]
        event_sum_wvfm = np.sum(wvfm, axis=0) # sum over the events
        sum_wvfm += event_sum_wvfm # sum over the sipms

    x = np.arange(0, 1000, 1)
    y = sum_wvfm

    drawn_objects = go.Scatter(x=x, y=y, name=f"Channel sum for light trap {opid}", visible=True, showlegend=True)
    fig.add_traces(drawn_objects)
    for adc, channel in sipms:
        wvfm = waveforms_all_detectors[:, adc, channel, :]
        sum_wvfm = np.sum(wvfm, axis=0)
        fig.add_traces(go.Scatter(x=x, y=sum_wvfm, visible="legendonly", showlegend=True, name=f"Channel {adc, channel}"))

    fig.update_xaxes(title_text="Time [ticks] (16 ns)")
    fig.update_yaxes(title_text="Adc counts")
    fig.update_layout(title_text=f"Waveform for light trap {opid}")
    return fig


def plot_charge(data, evid):
    io_group = data[
        "charge/events", "charge/calib_prompt_hits", "charge/packets", evid
    ]["io_group"][0, :, 0]
    time = data["charge/events", "charge/calib_prompt_hits", "charge/packets", evid][
        "timestamp"
    ][0, :, 0]
    charge = data["charge/events", "charge/calib_prompt_hits", "charge/packets", evid][
        "dataword"
    ][0, :, 0]

    fig = go.Figure()

    for i in range(0, 8):
        time_io = time[io_group == i + 1]
        charge_io = charge[io_group == i + 1]

        # Create the plot
        fig.add_trace(
            go.Histogram(
                x=time_io,
                y=charge_io,
                nbinsx=20,
                name=f"IO group {i+1}",
                visible="legendonly",
                showlegend=True,
            ),
        )

        # Add labels and title
        fig.update_xaxes(
            title_text="packets timestamp",
        )
        fig.update_yaxes(
            title_text="charge [ke-]",
        )
    fig.update_layout()
    return fig


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        if intermed <= cutoff:
            high_cutoff, high_color = cutoff, color
            break

    # noinspection PyUnboundLocalVariable
    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )
