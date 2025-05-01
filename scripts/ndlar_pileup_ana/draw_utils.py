import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

def draw_tpc(geo_version="default", mod_bounds=np.array([]), det_bounds=np.array([])):
    if geo_version == "default":
        center = go.Scatter3d(
            x=[np.mean(det_bounds[:,0])],
            y=[np.mean(det_bounds[:,1])],
            z=[np.mean(det_bounds[:,2])],
            marker=dict(size=3, color="green", opacity=0.5),
            mode="markers",
            name="tpcs' center",
        )


        anodes = draw_anode_planes_default(
            mod_bounds, colorscale="ice", showscale=False, opacity=0.1
        )
        cathodes = draw_cathode_planes_default(
            mod_bounds, colorscale="burg", showscale=False, opacity=0.1
        )
        
        cages = draw_cage_planes_default(
            mod_bounds, colorscale="burg", showscale=False, opacity=0.1
        )
    else:
        anode_xs = np.array([-63.931, -3.069, 3.069, 63.931])
        anode_ys = np.array([-19.8543, 103.8543])  # two ys
        anode_zs = np.array([-64.3163, -2.6837, 2.6837, 64.3163])  # four zs
        if geo_version == "minirun4":  # hit coordinates are in cm
            detector_center = (0, -268, 1300)
            anode_ys = anode_ys - (268 + 42)
            anode_zs = anode_zs + 1300
        if geo_version == "minirun5" or geo_version == "data":  # hit coordinates are in cm
            detector_center = (0, 0, 0)
            anode_ys = anode_ys - 42
        if geo_version == "single_mod":  # module 1
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


    return center, anodes, cathodes, cages

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

def draw_anode_planes_default(mod_bounds, **kwargs):
    traces = []
    for i_mod, this_mod_bounds in enumerate(mod_bounds):
        for i_tpc in [1, 2]:
            z, y = np.meshgrid(this_mod_bounds[:,2], this_mod_bounds[:, 1])
            # the module numbering starts from 0, and the tpc numbering starts from 1
            # FIXME: In 2x2, the high x side is the odd number of the tpc. Need to check for single module, FSD and NDLAr
            x = this_mod_bounds[:,0][2-i_tpc] * np.ones(z.shape)
            trace = go.Surface(
                x=x, y=y, z=z, hovertemplate=f"Module {i_mod} TPC {i_mod*2 + i_tpc}", **kwargs
            )
            traces.append(trace)
    return traces

def draw_cathode_planes_default(mod_bounds, **kwargs):
    traces = []
    for i_mod, this_mod_bounds in enumerate(mod_bounds):
        z, y = np.meshgrid(this_mod_bounds[:,2], this_mod_bounds[:, 1])
        x = np.mean(this_mod_bounds[:,0]) * np.ones(z.shape)
        trace = go.Surface(
            x=x, y=y, z=z, hovertemplate=f"Module {i_mod}", **kwargs
        )
        traces.append(trace)
    return traces

def draw_cage_planes_default(mod_bounds, **kwargs):
    traces = []
    for i_mod, this_mod_bounds in enumerate(mod_bounds):
        for i_tpc in [0, 1]:
            x, y = np.meshgrid(this_mod_bounds[:,0], this_mod_bounds[:, 1])
            # the module numbering starts from 0, and the tpc numbering starts from 1
            # FIXME: In 2x2, the high x side is the odd number of the tpc. Need to check for single module, FSD and NDLAr
            z = this_mod_bounds[:,2][i_tpc] * np.ones(x.shape)
            trace = go.Surface(
                x=x, y=y, z=z, hovertemplate=f"Module {i_mod}", **kwargs
            )
            traces.append(trace)
    return traces

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

def plot_clusters(cluster_list, tpc_center, anodes, cathodes, cages, title='', ncol=6):
    fig = go.Figure()
    cmap = plt.get_cmap("gist_rainbow", ncol)

    fig = go.Figure()
    fig.add_traces(tpc_center)
    fig.add_traces(anodes)
    fig.add_traces(cathodes)
    fig.add_traces(cages)

    for i, cl in enumerate(cluster_list):
        color = to_hex(cmap(i%ncol))
        fig.add_trace(go.Scatter3d(
            x=cl[:, 0], y=cl[:, 1], z=cl[:, 2],
            mode='markers',
            marker=dict(size=1, color=color),
            showlegend=False,
            name=f'Cluster {i}'
        ))

    camera = dict(
        up=dict(x=1., y=1., z=0),
        eye=dict(x=2.5, y=0, z=0)
    )

    fig.update_layout(
        width=1024, height=768,
        legend_orientation="h",
        scene = dict(xaxis_title='x [cm]',
                    yaxis_title='y [cm]',
                    zaxis_title='z [cm]',
                    aspectmode='data'),
        scene_camera=camera,
        legend_title=title
    )
    return fig
