"""
Utility functions for displaying data in the app
"""

# TODO: remove dependency on h5flow
import cmasher as cmr
import h5flow
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import uproot

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.proto_nd_flow.util.lut import LUT
from plotly.subplots import make_subplots

cmap_charge = cmr.get_sub_cmap('cmr.torch_r', 0.13, 0.95)
cmap_light = cmr.get_sub_cmap('cmr.sunburst_r', 0.0, 0.55)
cmap_bg = cmr.get_sub_cmap(cmr.torch_r, 0.01, 0.95)

colorscale_charge = [[i/255.0, 'rgb'+str(cmap_charge(i)[:3])] for i in range(256)]
colorscale_light = [[i/255.0, 'rgb'+str(cmap_light(i)[:3])] for i in range(256)]
bg_color = 'rgb'+str(cmap_bg(0.01)[:3])

def parse_contents(filename):
    if filename is None:
        return None, 0
    data = h5flow.data.H5FlowDataManager(filename, "r")
    num_events = data["charge/events/data"].shape[0]

    return data, num_events


def parse_minerva_contents(filename):
    if filename is None or filename == "":
        return None, 0
    minerva_data = uproot.open(filename)
    minerva_num_events = len(minerva_data["minerva"]["offsetX"].array(library="np"))

    return minerva_data, minerva_num_events


def is_beam_event(evid, filename):
    data, _ = parse_contents(filename)
    try:
        io_group = data["charge/ext_trigs", evid]["iogroup"][0]
        if io_group == 5:
            return True
        else:
            return False
    except:
        return False


def get_all_beam_triggers(filename):
    data, _ = parse_contents(filename)
    all_beam_triggers = []
    for ev_id, iogroup in enumerate(data["charge/ext_trigs/data"]["iogroup"]):
        if iogroup == 5:
            all_beam_triggers.append(ev_id)
    return all_beam_triggers


def create_3d_figure(minerva_data, data, filename, evid):
    fig = go.Figure()
    # Add watermark
    fig.add_annotation(
        text="DUNE 2x2",
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.05,
        showarrow=False,
        font=dict(size=30, color="#ffebc9", family="Arial Black"),
        textangle=-30,
        opacity=0.5,
    )
    # Select the hits for the current event
    beam_triggers = get_all_beam_triggers(filename)
    prompthits_ev = data["charge/events", "charge/calib_prompt_hits", evid]
    try:
        finalhits_ev = data["charge/events", "charge/calib_final_hits", evid]
    except:
        finalhits_ev = prompthits_ev
        print("No final hits found, plotting prompt hits")

    # select the segments (truth) for the current event
    try:
        prompthits_segs = data[
            "charge/events",
            "charge/calib_prompt_hits",
            "charge/packets",
            "mc_truth/segments",
            evid,
        ]
        sim_version = "minirun5"
        print("Found truth info in minirun5 format")
    except:
        try:
            prompthits_segs = data[
                "charge/events", "charge/calib_prompt_hits", "charge/packets", evid
            ]
            sim_version = "data"
            print("Found data")
        except:
            print("Cannot process this file type")
            prompthits_segs = None
    event = data["charge/events", evid]
    if (minerva_data is not None
        and evid in beam_triggers
        and sim_version == "data"):
        # trigger = beam_triggers.index(evid)
        minerva_times = (
            minerva_data["minerva"]["ev_gps_time_sec"].array(library="np")
            + minerva_data["minerva"]["ev_gps_time_usec"].array(library="np") / 1e6
        )
        charge_time = (event["unix_ts"][:] + event["ts_start"][:] / 1e7)[0]
        # find the index of the minerva_times that matches the charge_time
        trigger = np.argmin(np.abs(minerva_times - charge_time))
    if evid in beam_triggers and sim_version == "minirun5":
        trigger = ((event["unix_ts"][:] + event["ts_start"][:] / 1e7) / 1.2).astype(
            int
        )[0]
    if (
        minerva_data is not None
        # and evid < len(minerva_data["minerva"]["offsetX"].array(library="np"))
        and is_beam_event(evid, filename)
        and np.abs(minerva_times[trigger] - charge_time) < 1
    ):
        minerva = draw_minerva()
        fig.add_traces(minerva)

        minerva_hits_x_offset = minerva_data["minerva"]["offsetX"].array(library="np")
        minerva_hits_y_offset = minerva_data["minerva"]["offsetY"].array(library="np")
        minerva_hits_z_offset = minerva_data["minerva"]["offsetZ"].array(library="np")

        minerva_hits_x = minerva_data["minerva"]["trk_node_X"].array(library="np")
        minerva_hits_y = minerva_data["minerva"]["trk_node_Y"].array(library="np")
        minerva_hits_z = minerva_data["minerva"]["trk_node_Z"].array(library="np")

        minerva_trk_index = minerva_data["minerva"]["trk_index"].array(library="np")
        minerva_trk_nodes = minerva_data["minerva"]["trk_nodes"].array(library="np")
        minerva_trk_node_energy = minerva_data["minerva"]["clus_id_energy"].array(
            library="np"
        )

        xs = []
        ys = []
        zs = []
        qs = []

        for idx in minerva_trk_index[trigger]:

            n_nodes = minerva_trk_nodes[trigger][idx]
            if n_nodes > 0:
                x_nodes = (
                    minerva_hits_x[trigger][idx][:n_nodes]
                    # - minerva_hits_x_offset[trigger]
                )
                y_nodes = (
                    minerva_hits_y[trigger][idx][:n_nodes]
                    # - minerva_hits_y_offset[trigger]
                )
                z_nodes = minerva_hits_z[trigger][idx][
                    :n_nodes
                ]  # - minerva_hits_z_offset[trigger]
                q_nodes = minerva_trk_node_energy[trigger][:n_nodes]
            xs.append((x_nodes / 10).tolist())
            ys.append((y_nodes / 10 - 21.8338).tolist())
            zs.append((z_nodes / 10 - 691.3).tolist())
            qs.append((q_nodes).tolist())
        minerva_hit_traces = go.Scatter3d(
            x=[item for sublist in xs for item in sublist],
            y=[item for sublist in ys for item in sublist],
            z=[item for sublist in zs for item in sublist],
            marker_color=[item for sublist in qs for item in sublist],
            marker={
                "size": 1.75,
                "opacity": 0.9,
                "colorscale": colorscale_charge,
                "colorbar": {
                    "title": "Mx2 E [MeV]",
                    "titlefont": {"size": 12},
                    "tickfont": {"size": 10},
                    "thickness": 15,
                    "len": 0.5,
                    "xanchor": "right",
                    "x": 0,
                },
            },
            name="minerva hits",
            mode="markers",
            showlegend=True,
            opacity=0.9,
            customdata=[item for sublist in qs for item in sublist],
            hovertemplate="<b>x:%{x:.3f}</b><br>y:%{y:.3f}<br>z:%{z:.3f}<br>E:%{customdata:.3f}",
        )
        fig.add_traces(minerva_hit_traces)

    # Draw the TPC
    tpc_center, anodes, cathodes = draw_tpc(sim_version)
    light_detectors = draw_light_detectors(data, evid, sim_version)

    fig.add_traces(tpc_center)
    fig.add_traces(anodes)
    fig.add_traces(cathodes)
    fig.add_traces(light_detectors)

    # Draw an arrow for the beam direction
    fig.add_traces(
        go.Cone(
            x=[10],
            y=[20],
            z=[-75],
            u=[0],
            v=[0],
            w=[1],
            showscale=False,  # to hide the colorbar
            sizemode="absolute",
            sizeref=10,
            anchor="tail",
        )
    )

    fig.update_layout(
        font=dict(size=14),
        legend=dict(orientation="h"),
        margin=dict(
                l=50,  # left margin
                r=50,  # right margin
                b=20,  # bottom margin
                t=20,  # top margin
                pad=10,  # padding
            ),
        scene=dict(
            xaxis_title="x [cm]",
            annotations=[
                dict(
                    showarrow=False,
                    x=10,
                    y=20,
                    z=-75,
                    text="Beam",
                    xanchor="right",
                    xshift=10,
                    opacity=0.8,
                )
            ],
            xaxis=dict(  # to make the background white
                backgroundcolor=bg_color,
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            yaxis=dict(
                backgroundcolor=bg_color,
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            zaxis=dict(
                backgroundcolor=bg_color,
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
            ),
            yaxis_title="y [cm]",
            zaxis_title="z [cm]",
            camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=-1.25, y=1.0, z=-1.0)),
        ),
        # updatemenus=[ # this will make the plot rotate around the scene, but it's too slow
        #     dict(
        #         type="buttons",
        #         showactive=False,
        #         buttons=[
        #             dict(
        #                 label="Play",
        #                 method="animate",
        #                 args=[
        #                     None,
        #                     dict(
        #                         frame=dict(duration=25, redraw=True),
        #                         fromcurrent=True,
        #                         transition=dict(duration=5, easing="quadratic-in-out"),
        #                     ),
        #                 ],
        #             )
        #         ],
        #     )
        # ], 
    )

    # frames = [
    #     go.Frame(
    #         layout=dict(
    #             scene=dict(
    #                 camera=dict(
    #                     eye=dict(
    #                         x=-1.25 * np.cos(i * np.pi / 20),
    #                         y=1.0,
    #                         z=-1.0 * np.sin(i * np.pi / 20)
    #                     )
    #                 )
    #             )
    #         )
    #     )
    #     for i in range(20)
    # ]

    # fig.frames = frames # make the camera rotate around the scene

    # Plot the prompt hits
    if prompthits_ev["x"].mask[0][0]:
        print("No hits found for this event")
        return fig, sim_version
    prompthits_traces = go.Scatter3d(
        x=prompthits_ev.data["x"].flatten(),
        y=(prompthits_ev.data["y"].flatten()),
        z=(prompthits_ev.data["z"].flatten()),
        marker_color=prompthits_ev.data[
            "E"
        ].flatten(),  # convert to MeV from GeV for minirun4
        marker={
            "size": 1.75,
            "opacity": 0.9,
            "colorscale": colorscale_charge,
            "colorbar": {
                "title": "Hit E [MeV]",
                "titlefont": {"size": 12},
                "tickfont": {"size": 10},
                "thickness": 15,
                "len": 0.5,
                "xanchor": "left",
                "x": 0,
            },
            "cmin": 0.,
            "cmax": 5.,
        },
        name="prompt hits",
        mode="markers",
        showlegend=True,
        opacity=0.9,
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
            "opacity": 0.9,
            "colorscale": colorscale_charge,
            "colorbar": {
                "title": "Hit E [MeV]",
                "titlefont": {"size": 12},
                "tickfont": {"size": 10},
                "thickness": 15,
                "len": 0.5,
                "xanchor": "left",
                "x": 0,
            },
            "cmin": 0.,
            "cmax": 5.,
        },
        name="final hits",
        mode="markers",
        visible="legendonly",
        showlegend=True,
        opacity=0.9,
        customdata=finalhits_ev.data["E"].flatten(),
        hovertemplate="<b>x:%{x:.3f}</b><br>y:%{y:.3f}<br>z:%{z:.3f}<br>E:%{customdata:.3f}",
    )
    fig.add_traces(finalhits_traces)

    if prompthits_segs is not None and sim_version != "data":
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

    return fig, sim_version


def plot_segs(segs, sim_version="minirun5", **kwargs):
    def to_list(axis):
        if sim_version == "minirun5" or sim_version == "minirun4":
            nice_array = np.column_stack(
                [segs[f"{axis}_start"], segs[f"{axis}_end"], np.full(len(segs), None)]
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
    if sim_version == "minirun5" or sim_version == "data":  # hit coordinates are in cm
        detector_center = (0, 0, 0)
        anode_ys = anode_ys - 42
    if sim_version == "single_mod":  # module 1
        detector_center = (0, 0, 0)
        anode_xs = anode_xs[1:2]
        anode_ys = anode_ys
        anode_zs = anode_zs[0:2] + 33

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
    index_to_number = {(1, 1): 0, (1, 0): 1, (0, 1): 2, (0, 0): 3}
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
            # Get the module number for this plane
            number = index_to_number[(i_x, i_z)]
            trace = go.Surface(
                x=x, y=y, z=z, hovertemplate=f"Module {number}", **kwargs
            )
            traces.append(trace)

    return traces


def draw_anode_planes(x_boundaries, y_boundaries, z_boundaries, **kwargs):
    index_to_number = {
        (3, 1): 0,
        (2, 1): 1,
        (1, 1): 4,  # mod 2 tpcs are switched?
        (0, 1): 5,
        (3, 0): 2,
        (2, 0): 3,
        (1, 0): 6,
        (0, 0): 7,
    }
    traces = []
    for i_z in range(int(len(z_boundaries) / 2)):
        for i_x in range(int(len(x_boundaries))):
            z, y = np.meshgrid(
                np.linspace(z_boundaries[i_z * 2], z_boundaries[i_z * 2 + 1], 2),
                np.linspace(y_boundaries.min(), y_boundaries.max(), 2),
            )
            x = x_boundaries[i_x] * np.ones(z.shape)
            # Get the TPC number for this plane
            number = index_to_number[(i_x, i_z)]
            trace = go.Surface(x=x, y=y, z=z, hovertemplate=f"TPC {number}", **kwargs)
            traces.append(trace)

    return traces


def draw_minerva():
    x_base = [0, 108.0, 108.0, 0, -108.0, -108.0]
    shift = 245.0
    y_base = [
        -390.0 + shift,
        -330.0 + shift,
        -204.0 + shift,
        -145.0 + shift,
        -206.0 + shift,
        -330.0 + shift,
    ]

    z_base = {}
    z_base["ds"] = [164.0, 310.0]
    z_base["us"] = [-240.0, -190.0]

    traces = []
    # Define the vertices of the hexagon in the x-y plane
    hexagon_vertices = np.array([
        [x_base[0], y_base[0]],
        [x_base[1], y_base[1]],
        [x_base[2], y_base[2]],
        [x_base[3], y_base[3]],
        [x_base[4], y_base[4]],
        [x_base[5], y_base[5]]
    ])

    for z in [-240, -190, 164, 310]:
        # Create a Mesh3d object for each hexagonal plane
        hexagon = go.Mesh3d(
            x=hexagon_vertices[:, 0],
            y=hexagon_vertices[:, 1],
            z=z * np.ones(len(hexagon_vertices)),
            opacity=0.1,
            color='blue',
        )

        traces.append(hexagon)

    # Plot the cylindrical hexagon
    for j in ["ds", "us"]:
        for i in range(len(x_base)):
            traces.append(
                go.Scatter3d(
                    x=[x_base[i], x_base[(i + 1) % len(x_base)]],
                    y=[y_base[i], y_base[(i + 1) % len(x_base)]],
                    z=[z_base[j][0], z_base[j][0]],
                    mode="lines",
                    showlegend=False,
                    line=dict(color="grey"),
                )
            )  # Plot bottom face
            traces.append(
                go.Scatter3d(
                    x=[x_base[i], x_base[(i + 1) % len(x_base)]],
                    y=[y_base[i], y_base[(i + 1) % len(x_base)]],
                    z=[z_base[j][1], z_base[j][1]],
                    mode="lines",
                    showlegend=False,
                    line=dict(color="grey"),
                )
            )  # Plot top face
            traces.append(
                go.Scatter3d(
                    x=[x_base[i], x_base[i]],
                    y=[y_base[i], y_base[i]],
                    z=[z_base[j][0], z_base[j][1]],
                    mode="lines",
                    showlegend=False,
                    line=dict(color="blue"),
                )
            )  # Plot vertical edges

        # Plot the line connecting the bottom and top faces
        for i in range(len(x_base)):
            traces.append(
                go.Scatter3d(
                    x=[x_base[i], x_base[i]],
                    y=[y_base[i], y_base[i]],
                    z=[z_base[j][0], z_base[j][1]],
                    mode="lines",
                    showlegend=False,
                    line=dict(color="grey"),
                )
            )  # Plot vertical edges

    return traces


def draw_light_detectors(data, evid, sim_version):
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

    waveforms_all_detectors = get_waveforms_all_detectors(match_light, sim_version)

    drawn_objects = []
    drawn_objects.extend(plot_light_traps(data, waveforms_all_detectors))

    return drawn_objects


def match_light_to_charge_event(data, evid):
    """
    Match the light events to the charge event by looking at proximity in time.
    Use unix time for this, since it should refer to the same time in both readout systems.
    For now we just take all the light within 1s from the charge event time.
    """
    try:
        match_light = data["charge/events", "light/events", "light/wvfm", evid]
        if np.ma.all(match_light.mask) == True:
            try:
                match_light = match_light[match_light.mask == False]
            except:
                print("No matching light information found, plotting zeros")
    except Exception as e:
        return None

    return match_light


def get_waveforms_all_detectors(match_light, sim_version):
    """
    Get the light waveforms for the matched light events.
    """
    n_matches = match_light["samples"].shape[1]
    if sim_version != "single_mod":
        waveforms_all_detectors = match_light["samples"].reshape(n_matches, 8, 64, 1000)
        # compute the mean of the first 50 samples along the last axis
        baseline_mean = np.mean(waveforms_all_detectors[:, :, :, :50], axis=-1)
        # subtract the baseline mean from each waveform
        waveforms_all_detectors -= (baseline_mean[:, :, :, np.newaxis]).astype(int)
    if sim_version == "single_mod":
        waveforms_all_detectors = (
            match_light["samples"].reshape(n_matches, 2, 64, 1000).astype(np.float64)
        )
        # for each waveform subtract the mean of the first 50 samples
        # compute the mean of the first 50 samples along the last axis
        baseline_mean = np.mean(waveforms_all_detectors[:, :, :, :50], axis=-1)
        # subtract the baseline mean from each waveform
        waveforms_all_detectors -= baseline_mean[:, :, :, np.newaxis]
    return waveforms_all_detectors


def plot_light_traps(data, waveforms_all_detectors):
    """Plot optical detectors"""
    drawn_objects = []

    det_bounds = LUT.from_array(
        data["/geometry_info/det_bounds"].attrs["meta"],
        data["/geometry_info/det_bounds/data"],
    )
    sipm_rel_pos = LUT.from_array(
        data["geometry_info/sipm_rel_pos"].attrs["meta"],
        data["geometry_info/sipm_rel_pos/data"],
    )
    sipm_abs_pos = LUT.from_array(
        data["geometry_info/sipm_abs_pos"].attrs["meta"],
        data["geometry_info/sipm_abs_pos/data"],
    )
    det_ids = LUT.from_array(
        data["geometry_info/det_id"].attrs["meta"],
        data["geometry_info/det_id/data"],
    )

    sipm_keys = list(zip(sipm_rel_pos.keys()[0], sipm_rel_pos.keys()[1]))
    data = []
    for k in sipm_keys:
        sipm_x, sipm_y, sipm_z = sipm_abs_pos[k][0]
        tpc, side, sipm_pos = sipm_rel_pos[k][0]
        adc, channel = k
        det_id = det_ids[k][0]
        (min_x, min_y, min_z), (max_x, max_y, max_z) = det_bounds[(tpc, det_id)][0]
        row = {
            "det_id": det_id,
            "sipm_x": sipm_x,
            "sipm_y": sipm_y,
            "sipm_z": sipm_z,
            "tpc": tpc,
            "side": side,
            "sipm_pos": sipm_pos,
            "adc": adc,
            "channel": channel,
            "min_x": min_x,
            "min_y": min_y,
            "min_z": min_z,
            "max_x": max_x,
            "max_y": max_y,
            "max_z": max_z,
        }
        data.append(row)
    lut_based_channel_map = pd.DataFrame(data)

    COLORSCALE = colorscale_light #plotly.colors.make_colorscale(
        #plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.YlOrRd)[0]
    #)

    drawn_objects = []
    max_photons = 0
    for (det_id, tpc), group in lut_based_channel_map.groupby(["det_id", "tpc"]):
        for _, row in group.iterrows():
            adc, channel = row["adc"], row["channel"]
            wvfm = waveforms_all_detectors[:, int(adc), int(channel), :]
            if np.ma.all(np.ma.getmaskarray(wvfm)):
                wvfm = np.zeros(waveforms_all_detectors[:, 0, 0, :].shape, dtype=int)
            sum_wvfm = np.sum(wvfm, axis=0)  # sum over the events
            sum_photons = np.sum(sum_wvfm, axis=0)  # sum over the time
            if sum_photons > max_photons:
                max_photons = sum_photons

    for (det_id, tpc), group in lut_based_channel_map.groupby(["det_id", "tpc"]):
        xs = [group["min_x"].values[0], group["max_x"].values[0]]
        ys = [group["min_y"].values[0], group["max_y"].values[0]]
        zs = [group["min_z"].values[0], group["max_z"].values[0]]
        sum_photons = 0
        for _, row in group.iterrows():
            adc, channel = row["adc"], row["channel"]
            wvfm = waveforms_all_detectors[:, int(adc), int(channel), :]
            if np.ma.all(np.ma.getmaskarray(wvfm)):
                wvfm = np.zeros(waveforms_all_detectors[:, 0, 0, :].shape, dtype=int)
            sum_wvfm = np.sum(wvfm, axis=0)  # sum over the events
            sum_photons += np.sum(sum_wvfm, axis=0)  # sum over the time
        if max_photons == 0:
            max_photons = 1
        light_color = [
            [
                0.0,
                get_continuous_color(
                    COLORSCALE, intermed=np.abs(sum_photons) / max_photons
                ),
            ],
            [
                1.0,
                get_continuous_color(
                    COLORSCALE, intermed=np.abs(sum_photons) / max_photons
                ),
            ],
        ]
        det_label = f"det_id_{det_id}_tpc_{tpc}"

        light_plane = go.Surface(
            x=xs,
            y=ys,
            z=[zs, zs],
            colorscale=light_color,
            showscale=False,
            showlegend=False,
            opacity=0.2,
            hoverinfo="text",
            ids=[[det_label, det_label], [det_label, det_label]],
            text=f"Optical detector {det_label} waveform integral<br>{np.abs(sum_photons):.2e}",
        )
        drawn_objects.append(light_plane)

    return drawn_objects


def plot_waveform(data, evid, opid, sim_version):
    match_light = match_light_to_charge_event(data, evid)
    fig = go.Figure()
    # Add watermark
    fig.add_annotation(
        text="DUNE 2x2",
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.05,
        showarrow=False,
        font=dict(size=30, color="#ffebc9", family="Arial Black"),
        textangle=-30,
        opacity=0.5,
    )
    det_id_click, tpc_click = opid
    print(f"Plotting waveform for light trap {det_id_click} in TPC {tpc_click}")
    if match_light is None or np.ma.all(match_light.mask) == True:
        try:
            charge = data["charge/events", evid][["id", "unix_ts"]]
            num_light = data["light/events/data"].shape[0]
            light = data["light/events", slice(0, num_light)][
                ["id", "utime_ms"]
            ]  # we have to try them all, events may not be time ordered
            match_light = np.array(
                [
                    light[light["utime_ms"] - charge["unix_ts"][i] < 1e6]
                    for i in range(len(charge))
                ]
            )
        except:
            print(
                f"No light event matches found for charge event {evid}, not plotting light waveform"
            )
            return fig

    waveforms_all_detectors = get_waveforms_all_detectors(match_light, sim_version)

    det_bounds = LUT.from_array(
        data["/geometry_info/det_bounds"].attrs["meta"],
        data["/geometry_info/det_bounds/data"],
    )
    sipm_rel_pos = LUT.from_array(
        data["geometry_info/sipm_rel_pos"].attrs["meta"],
        data["geometry_info/sipm_rel_pos/data"],
    )
    sipm_abs_pos = LUT.from_array(
        data["geometry_info/sipm_abs_pos"].attrs["meta"],
        data["geometry_info/sipm_abs_pos/data"],
    )
    det_ids = LUT.from_array(
        data["geometry_info/det_id"].attrs["meta"],
        data["geometry_info/det_id/data"],
    )

    sipm_keys = list(zip(sipm_rel_pos.keys()[0], sipm_rel_pos.keys()[1]))
    data = []
    for k in sipm_keys:
        sipm_x, sipm_y, sipm_z = sipm_abs_pos[k][0]
        tpc, side, sipm_pos = sipm_rel_pos[k][0]
        adc, channel = k
        det_id = det_ids[k][0]
        (min_x, min_y, min_z), (max_x, max_y, max_z) = det_bounds[(tpc, det_id)][0]
        row = {
            "det_id": det_id,
            "sipm_x": sipm_x,
            "sipm_y": sipm_y,
            "sipm_z": sipm_z,
            "tpc": tpc,
            "side": side,
            "sipm_pos": sipm_pos,
            "adc": adc,
            "channel": channel,
            "min_x": min_x,
            "min_y": min_y,
            "min_z": min_z,
            "max_x": max_x,
            "max_y": max_y,
            "max_z": max_z,
        }
        data.append(row)
    lut_based_channel_map = pd.DataFrame(data)
    sipms = lut_based_channel_map[
        (lut_based_channel_map["det_id"] == det_id_click)
        & (lut_based_channel_map["tpc"] == tpc_click)
    ][["adc", "channel"]].values

    sum_wvfm = np.array([0.0] * 1000)
    for adc, channel in sipms:
        wvfm = waveforms_all_detectors[:, int(adc), int(channel), :]
        event_sum_wvfm = np.sum(wvfm, axis=0)  # sum over the events
        sum_wvfm += event_sum_wvfm  # sum over the sipms

    x = np.arange(0, 1000, 1)
    y = sum_wvfm

    drawn_objects = go.Scatter(
        x=x,
        y=y,
        name=f"Channel sum",
        visible=True,
        showlegend=True,
    )
    fig.add_traces(drawn_objects)
    for adc, channel in sipms:
        wvfm = waveforms_all_detectors[:, int(adc), int(channel), :]
        sum_wvfm = np.sum(wvfm, axis=0)
        fig.add_traces(
            go.Scatter(
                x=x,
                y=sum_wvfm,
                visible="legendonly",
                showlegend=True,
                name=f"Channel {adc, channel}",
            )
        )

    fig.update_xaxes(title_text="Time [ticks] (16 ns)", showgrid=False)
    fig.update_yaxes(title_text="Adc counts", showgrid=False)
    fig.update_layout(
        title_text=f"Waveforms for light trap {det_id_click} in TPC {tpc_click}",
        legend=dict(
            orientation="h", yref="container", yanchor="bottom", xanchor="center", x=0.5
        ),
        plot_bgcolor=bg_color,
        margin=dict(
            l=20,  # left margin
            r=20,  # right margin
            b=20,  # bottom margin
            t=30,  # top margin
            pad=10,  # padding
        ),
    )
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
    # Add watermark
    fig.add_annotation(
        text="DUNE 2x2",
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.05,
        showarrow=False,
        font=dict(size=30, color="#ffebc9", family="Arial Black"),
        textangle=-30,
        opacity=0.5,
    )

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
                showlegend=True,
            ),
        )

        # Add labels and title
        fig.update_xaxes(
            title_text="packets timestamp",
            showgrid=False,
        )
        fig.update_yaxes(
            title_text="charge [ke-]",
            showgrid=False,
        )
    fig.update_layout(
        title_text=f"Charge histogram for event {evid}",
        legend=dict(
            orientation="h", yref="container", yanchor="bottom", xanchor="center", x=0.5
        ),
        plot_bgcolor=bg_color,
        margin=dict(
            l=20,  # left margin
            r=20,  # right margin
            b=20,  # bottom margin
            t=30,  # top margin
            pad=10,  # padding
        ),
    )
    return fig


def plot_charge_energy(data, evid):
    prompthits_ev = data["charge/events", "charge/calib_prompt_hits", evid]
    energy = prompthits_ev.data["E"].flatten()
    fig = go.Figure()
    # Add watermark
    fig.add_annotation(
        text="DUNE 2x2",
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.05,
        showarrow=False,
        font=dict(size=30, color="#ffebc9", family="Arial Black"),
        textangle=-30,
        opacity=0.5,
    )

    fig.add_trace(
        go.Histogram(
            x=energy,
            nbinsx=20,
            name="Energy",
            showlegend=False,
        ),
    )

    fig.update_xaxes(
        title_text="Energy [MeV]",
        showgrid=False,
    )
    fig.update_yaxes(
        title_text="Hit count",
        showgrid=False,
    )
    fig.update_layout(
        title_text=f"Energy histogram for event {evid}",
        plot_bgcolor=bg_color,
        margin=dict(
            l=20,  # left margin
            r=20,  # right margin
            b=20,  # bottom margin
            t=30,  # top margin
            pad=10,  # padding
        ),
    )
    return fig


def plot_2d_charge(data, evid):
    # Create a subplot with 1 row and 3 columns
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.1, horizontal_spacing=0.15)
    # Add watermark
    fig.add_annotation(
        text="DUNE 2x2",
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.05,
        showarrow=False,
        font=dict(size=30, color="#ffebc9", family="Arial Black"),
        textangle=-30,
        opacity=0.5,
    )

    # Select the hits for the current event
    prompthits_ev = data["charge/events", "charge/calib_prompt_hits", evid]

    # Define a colorscale and colorbar for the plots
    colorbar = dict(
        title="Hit E [MeV]",
        ticks="outside",
        ticklen=3,
        tickwidth=1,
        showticklabels=True,
    )

    # Add 2D projections of the prompt hits
    prompthits_traces_xy = go.Scatter(
        x=prompthits_ev.data["x"].flatten(),
        y=prompthits_ev.data["y"].flatten(),
        mode="markers",
        marker=dict(
            size=2,
            opacity=0.9,
            color=prompthits_ev.data["E"].flatten(),
            colorscale=colorscale_charge,
            colorbar=colorbar,
            showscale=False,
        ),
    )

    prompthits_traces_xz = go.Scatter(
        x=prompthits_ev.data["z"].flatten(),
        y=prompthits_ev.data["x"].flatten(),
        mode="markers",
        marker=dict(
            size=2,
            opacity=0.9,
            color=prompthits_ev.data["E"].flatten(),
            colorscale=colorscale_charge,
            colorbar=colorbar,
            showscale=False,
        ),
    )

    prompthits_traces_yz = go.Scatter(
        x=prompthits_ev.data["z"].flatten(),
        y=prompthits_ev.data["y"].flatten(),
        mode="markers",
        marker=dict(
            size=2,
            opacity=0.9,
            color=prompthits_ev.data["E"].flatten(),
            colorscale=colorscale_charge,
            colorbar=colorbar,
            showscale=False,
        ),
    )

    # Add a dummy trace with the desired colorbar
    dummy_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            size=2,
            opacity=0.7,
            colorscale=colorscale_charge,
            colorbar=colorbar,
            showscale=True,
        ),
        showlegend=False,
    )

    cathode_line_1 = go.Scatter(
        x=[-63.931/2, -63.931/2],
        y=[-63.931, 63.931],
        mode="lines",
        line=dict(color="white", width=1),
    )
    cathode_line_2 = go.Scatter(
        x=[63.931/2, 63.931/2],
        y=[-63.931, 63.931],
        mode="lines",
        line=dict(color="white", width=1),
    )

    cathode_line_3 = go.Scatter(
        x=[-63.931, 63.931],
        y=[-63.931/2, -63.931/2],
        mode="lines",
        line=dict(color="white", width=1),
    )
    cathode_line_4 = go.Scatter(
        x=[-63.931, 63.931],
        y=[63.931/2, 63.931/2],
        mode="lines",
        line=dict(color="white", width=1),
    )

    # Add traces to the subplots
    fig.add_trace(prompthits_traces_xy, row=2, col=2)
    fig.add_trace(cathode_line_1, row=2, col=2)
    fig.add_trace(cathode_line_2, row=2, col=2)
    fig.add_trace(prompthits_traces_xz, row=1, col=1)
    fig.add_trace(cathode_line_3, row=1, col=1)
    fig.add_trace(cathode_line_4, row=1, col=1)
    fig.add_trace(prompthits_traces_yz, row=2, col=1)
    fig.add_trace(
        dummy_trace, row=2, col=2
    )  # Add the dummy trace to one of the subplots

    # Add x and y axis labels to the subplots
    fig.update_xaxes(
        title_text="z [cm]",
        row=1,
        col=1,
        showgrid=False,
        range=[-60, 60],
        zeroline=True,
        constrain="domain",
    )
    fig.update_yaxes(
        title_text="x [cm]",
        row=1,
        col=1,
        showgrid=False,
        range=[-60, 60],
        zeroline=True,
        scaleanchor="x1",
        scaleratio=1,
    )

    fig.update_xaxes(
        title_text="z [cm]",
        row=2,
        col=1,
        showgrid=False,
        range=[-60, 60],
        zeroline=True,
        constrain="domain",
    )
    fig.update_yaxes(
        title_text="y [cm]",
        row=2,
        col=1,
        showgrid=False,
        range=[-60, 60],
        zeroline=False,
        scaleanchor="x2",
        scaleratio=1,
    )

    fig.update_xaxes(
        title_text="x [cm]",
        row=2,
        col=2,
        showgrid=False,
        range=[-60, 60],
        zeroline=True,
        constrain="domain",
    )
    fig.update_yaxes(
        title_text="y [cm]",
        row=2,
        col=2,
        showgrid=False,
        range=[-60, 60],
        zeroline=False,
        scaleanchor="x3",
        scaleratio=1,
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor=bg_color,
        autosize=False,
        width=800,
        height=700,
        margin=dict(
            l=20,  # left margin
            r=20,  # right margin
            b=20,  # bottom margin
            t=20,  # top margin
            pad=10,  # padding
        ),
    )

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

    # Initialize low_color and high_color with default values
    low_color = colorscale[0][1]
    high_color = colorscale[-1][1]
    # Initialize low_cutoff and high_cutoff with default values
    low_cutoff = 0
    high_cutoff = 1

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        if intermed <= cutoff:
            high_cutoff, high_color = cutoff, color
            break

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )
