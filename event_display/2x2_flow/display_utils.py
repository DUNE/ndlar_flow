"""
Utility functions for displaying data in the app
"""

# TODO: remove dependency on h5flow
import h5flow
import numpy as np
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
        marker_color=prompthits_ev.data["E"].flatten()
        * 1000,  # convert to MeV from GeV for minirun4, not sure for minirun3
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
        customdata=prompthits_ev.data["E"].flatten() * 1000,
        hovertemplate="<b>x:%{x:.3f}</b><br>y:%{y:.3f}<br>z:%{z:.3f}<br>E:%{customdata:.3f}",
    )
    fig.add_traces(prompthits_traces)

    # Plot the final hits
    finalhits_traces = go.Scatter3d(
        x=finalhits_ev.data["x"].flatten(),
        y=(finalhits_ev.data["y"].flatten()),
        z=(finalhits_ev.data["z"].flatten()),
        marker_color=finalhits_ev.data["E"].flatten() * 1000,
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
        customdata=finalhits_ev.data["E"].flatten() * 1000,
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
        charge = data["charge/events", evid][["id", "unix_ts"]]
        num_light = data["light/events/data"].shape[0]
        light = data["light/events", slice(0, num_light)][
            ["id", "utime_ms"]
        ]  # we have to try them all, events may not be time ordered
    except:
        print("No light information found, not plotting light detectors")
        return []

    match_light = match_light_to_charge_event(charge, light, evid)

    if match_light is None:
        print(
            f"No light event matches found for charge event {evid}, not plotting light detectors"
        )
        return []

    waveforms_all_detectors = get_waveforms_all_detectors(data, match_light)

    # make a list of the sum of the waveform and the channel index
    integral = np.sum(np.sum(waveforms_all_detectors, axis=2), axis=0)
    max_integral = np.max(integral)
    index = np.arange(0, waveforms_all_detectors.shape[1], 1)

    # plot for each of the 96 channels per tpc the sum of the adc values
    drawn_objects = []
    drawn_objects.extend(plot_light_traps(data, integral, index, max_integral))

    return drawn_objects


def match_light_to_charge_event(charge, light, evid):
    """
    Match the light events to the charge event by looking at proximity in time.
    Use unix time for this, since it should refer to the same time in both readout systems.
    For now we just take all the light within 1s from the charge event time.
    """
    matches = []
    for i in range(len(light)):
        if np.abs(light["utime_ms"][i][0] / 1000 - charge["unix_ts"][0]) < 0.5:
            matches.append([charge["id"][0], light["id"][i]])

    match_light = []
    for i in range(len(matches)):
        if (
            matches[i][0] == evid
        ):  # just checking that we get light for the right charge event
            match_light.append(matches[i][1])
    if len(match_light) == 0:
        match_light = None  # no light for this charge event

    return match_light


def get_waveforms_all_detectors(data, match_light):
    """
    Get the light waveforms for the matched light events.
    """
    light_wvfm = data["/light/wvfm", match_light]

    samples_mod0 = light_wvfm["samples"][:, 0:2, :, :]
    samples_mod1 = light_wvfm["samples"][:, 2:4, :, :]
    samples_mod2 = light_wvfm["samples"][:, 4:6, :, :]
    samples_mod3 = light_wvfm["samples"][:, 6:8, :, :]

    sipm_channels_module0 = np.array(
        [2, 3, 4, 5, 6, 7]
        + [9, 10]
        + [11, 12]
        + [13, 14]
        + [18, 19, 20, 21, 22, 23]
        + [25, 26]
        + [27, 28]
        + [29, 30]
        + [34, 35, 36, 37, 38, 39]
        + [41, 42]
        + [43, 44]
        + [45, 46]
        + [50, 51, 52, 53, 54, 55]
        + [57, 58]
        + [59, 60]
        + [61, 62]
    )

    sipm_channels_modules = np.array(
        [4, 5, 6, 7, 8, 9]
        + [10, 11, 12, 13, 14, 15]
        + [20, 21, 22, 23, 24, 25]
        + [26, 27, 28, 29, 30, 31]
        + [36, 37, 38, 39, 40, 41]
        + [42, 43, 44, 45, 46, 47]
        + [52, 53, 54, 55, 56, 57]
        + [58, 59, 60, 61, 62, 63]
    )
    adcs_mod0 = samples_mod0[:, :, sipm_channels_module0, :]
    adcs_mod1 = samples_mod1[:, :, sipm_channels_modules, :]
    adcs_mod2 = samples_mod2[:, :, sipm_channels_modules, :]
    adcs_mod3 = samples_mod3[:, :, sipm_channels_modules, :]

    all_adcs = np.concatenate((adcs_mod0, adcs_mod1, adcs_mod2, adcs_mod3), axis=1)

    # instead of a (m, 8, 48, 1000) array, we want a (m, 4, 96, 1000) array
    # modules instead of tpcs, and 96 channels per module
    m = len(match_light)
    all_modules = all_adcs.reshape((m, 4, 96, 1000))

    # now we make a full array for all the modules
    # could have been done in one step, but this is easier to read
    all_detector = all_modules.reshape((m, 384, 1000))

    return all_detector


def plot_light_traps(data, n_photons, op_indeces, max_integral):
    """Plot optical detectors"""
    drawn_objects = []

    det_bounds = data["/geometry_info/det_bounds/data"]
    channel_map = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 65, 73, 81, 89, 97, 105, 113, 121, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26,
                34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122, 67, 75, 83, 91, 99, 107, 115, 123, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52,
                  60, 68, 76, 84, 92, 100, 108, 116, 124, 69, 77, 85, 93, 101, 109, 117, 125, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 70,
                    78, 86, 94, 102, 110, 118, 126, 71, 79, 87, 95, 103, 111, 119, 127, 7, 15, 23, 31, 39, 47, 55, 63])
    # we need to invert the mapping because I'm stupid
    channel_map = np.argsort(channel_map)

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

    channel_dict = {}
    for i in range(0, 128, 4):
        j=i*3
        channel_dict[i] = [j, j+1, j+2, j+3, j+4, j+5] # arclight channels
        channel_dict[i+1] = [j+6, j+7] # lcm channels
        channel_dict[i+2] = [j+8, j+9]
        channel_dict[i+3] = [j+10, j+11]

    for i in range(len(xs)):
        opid = channel_map[i]
        sipms = channel_dict[opid]
        opid_str = f"opid_{opid}"
        light_color = [
            [
                0.0,
                get_continuous_color(
                    COLORSCALE, intermed=max(0, sum(n_photons[sipms])) / max_integral
                ),
            ],
            [
                1.0,
                get_continuous_color(
                    COLORSCALE, intermed=max(0, sum(n_photons[sipms])) / max_integral
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
            text=f"Optical detector {opid} waveform integral<br>{sum(n_photons[sipms])/max_integral:.2e}",
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

    match_light = match_light_to_charge_event(charge, light, evid)

    if match_light is None:
        print(
            f"No light event matches found for charge event {evid}, not plotting light waveform"
        )
        return []

    fig = go.Figure()
    waveforms_all_detectors = get_waveforms_all_detectors(data, match_light)

    channel_dict = {}
    for i in range(0, 128, 4):
        j=i*3
        channel_dict[i] = [j, j+1, j+2, j+3, j+4, j+5, j+6] # arclight channels
        channel_dict[i+1] = [j+7, j+8] # lcm channels
        channel_dict[i+2] = [j+9, j+10]
        channel_dict[i+3] = [j+11, j+12]
    wvfm_opid = waveforms_all_detectors[:, channel_dict[opid], :]

    x = np.arange(0, 1000, 1)
    y = np.sum(np.sum(wvfm_opid, axis=0),axis=0)
    drawn_objects = go.Scatter(x=x, y=y, name=f"Channel sum for light trap {opid}", visible=True, showlegend=True)
    fig.add_traces(drawn_objects)
    for i in range(0, wvfm_opid.shape[1]):
        fig.add_traces(go.Scatter(x=x, y=np.sum(wvfm_opid, axis=0)[i, : ], visible="legendonly", showlegend=True, name=f"Channel {channel_dict[opid][i]}"))

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
            title_text="charge",
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
