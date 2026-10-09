"""Beginner-friendly 2x2 charge-light event display + id lookups.

Self-contained: only needs the files in THIS folder plus read access to the
2x2 FLOW files on CFS.  No GPU, no torch.

Works with BOTH kinds of FLOW files:
  * regular charge+light files (beam reflows, MiniRun sim, ...);
  * LIGHT-ONLY files (e.g. nearline flowed_light) — any waveform window
    length; the channel map falls back to the bundled sipm_lut.json when the
    file carries no geometry_info.

    import cl_display as cl
    h5, tbl = cl.open_flow()                      # or open_flow(path)
    cl.show_pair(h5, tbl, charge_ev=286, light_ev=41)
    cl.show_light(h5, tbl, 41)                    # auto charge lookup
    cl.show_light_only(h5, tbl, 41)               # just the light
"""
from __future__ import annotations
import json
import os
from types import SimpleNamespace

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, _HERE)
import data_2x2 as data          # noqa: E402
import geometry_2x2 as geo       # noqa: E402

DEFAULT_FLOW = ("/global/cfs/cdirs/dune/www/data/2x2/reflows/v10/flow/beam/"
                "july10_2024/nominal_hv/"
                "packet-0050018-2024_07_10_09_36_12_CDT.FLOW.hdf5")
DEAD_YAML = os.path.join(_HERE, "dead_channels_2x2.yaml")
LUT_JSON = os.path.join(_HERE, "sipm_lut.json")

# pretty colormap if available, safe fallback otherwise
try:
    import cmasher as _cmr
    import cmcrameri                              # noqa: F401
    _CMAP = _cmr.get_sub_cmap("cmc.devon", 0.13, 0.95)
except Exception:
    _CMAP = plt.get_cmap("viridis")

# ORDERED_KEYS (side*24+y_rel) -> drawer's interleaved top-to-bottom order
_PERM = np.empty(48, np.int64)
for _i in range(24):
    _PERM[2 * _i] = 23 - _i
    _PERM[2 * _i + 1] = 24 + 23 - _i


# ---------------------------------------------------------------------------
# opening files (charge+light OR light-only)
# ---------------------------------------------------------------------------
def _light_only_tables(h5):
    """Minimal table bundle for files with no charge (and possibly no
    geometry_info — then the bundled channel map is used)."""
    if "geometry_info" in h5:
        lut = geo.build_sipm_lut(h5)
    else:
        lut = {(t, s, y): (a, c)
               for t, s, y, a, c in json.load(open(LUT_JSON))}
        print("note: file has no geometry_info — using the bundled "
              "july10_2024 channel map (verify if the run's cabling differs)")
    adc_idx, ch_idx, valid = geo.build_ordered_channel_index(lut)
    dead_mask = geo.ordered_dead_mask(lut, geo.parse_dead_channels(DEAD_YAML))
    empty = np.zeros((0, 2), np.int64)
    return SimpleNamespace(
        h5=h5, wvfm_dset=h5["light/wvfm/data"], lut=lut,
        adc_idx=adc_idx, ch_idx=ch_idx, valid=valid, dead_mask=dead_mask,
        charge_light_ref=empty, hits_ref=empty, hits_full=None,
        evt_flash_ref=empty, flash_data=None)


def open_flow(path: str = DEFAULT_FLOW):
    """Open any FLOW hdf5 (charge+light or light-only). Returns (h5, tbl)."""
    h5 = h5py.File(path, "r")
    n_lev = h5["light/events/data"].shape[0]
    T = h5["light/wvfm/data"].dtype["samples"].shape[-1]
    if "charge" in h5:
        tbl = data.get_tables(h5, dead_yaml=DEAD_YAML)
        print(f"opened {os.path.basename(path)}: "
              f"{h5['charge/events/data'].shape[0]} charge events, "
              f"{n_lev} light events ({T}-tick window), "
              f"{len(tbl.charge_light_ref)} charge-light pairs")
    else:
        tbl = _light_only_tables(h5)
        print(f"opened {os.path.basename(path)}: LIGHT-ONLY file, "
              f"{n_lev} light events ({T}-tick window)")
    return h5, tbl


# ---------------------------------------------------------------------------
# id lookups (all protected)
# ---------------------------------------------------------------------------
def find_light_for_charge(tbl, charge_ev: int):
    """Light event id associated with this charge event, or None."""
    rows = tbl.charge_light_ref[tbl.charge_light_ref[:, 0] == int(charge_ev)]
    return int(rows[0, 1]) if rows.size else None


def find_charge_for_light(tbl, light_ev: int):
    """Charge event id associated with this light event, or None."""
    rows = tbl.charge_light_ref[tbl.charge_light_ref[:, 1] == int(light_ev)]
    return int(rows[0, 0]) if rows.size else None


def charge_hits(h5, tbl, charge_ev: int):
    """(x, y, z, E, tpc) arrays for one charge event (may all be empty)."""
    if "charge" not in h5:
        print("this file has no charge data (light-only file)")
        return (np.array([]),) * 5
    n_ev = h5["charge/events/data"].shape[0]
    if not (0 <= int(charge_ev) < n_ev):
        print(f"charge event {charge_ev} out of range (file has {n_ev})")
        return (np.array([]),) * 5
    refs = tbl.hits_ref[tbl.hits_ref[:, 0] == int(charge_ev), 1]
    if refs.size == 0:
        return (np.array([]),) * 5
    hits = tbl.hits_full[refs]
    x = np.asarray(hits["x"], float)
    y = np.asarray(hits["y"], float)
    z = np.asarray(hits["z"], float)
    E = np.asarray(hits["E"], float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(E)
    tpc = geo.charge_tpc_from_io_group(hits["io_group"])
    return x[ok], y[ok], z[ok], E[ok], tpc[ok]


# ---------------------------------------------------------------------------
# the standard single-TPC display (blue measured light + charge in center)
# ---------------------------------------------------------------------------
def _draw_tpc(wf48, tpc, lut, ySet, zSet, xSet, title):
    T = wf48.shape[-1]
    is_left = tpc in (2, 3, 6, 7)
    ymin_g = float(np.nanmin(wf48)) - 3
    ymax_g = float(np.nanmax(wf48)) + 10

    fig = plt.figure(figsize=(13, 10))
    outer = gridspec.GridSpec(1, 7,
                              width_ratios=[1, 0.7, 1, .02, .02, .1, .03],
                              wspace=0.0)
    left_gs = gridspec.GridSpecFromSubplotSpec(24, 1, subplot_spec=outer[0],
                                               hspace=0)
    center_ax = fig.add_subplot(outer[1])
    right_gs = gridspec.GridSpecFromSubplotSpec(24, 1, subplot_spec=outer[2],
                                                hspace=0)
    cbar_ax = fig.add_subplot(outer[6])
    xticks = list(np.linspace(0, T, 5).astype(int))
    for i in range(24):
        vert = 23 - i
        adc_L, ch_L = lut[(tpc, 0, vert)]
        adc_R, ch_R = lut[(tpc, 1, vert)]
        for gs, wf, adc, ch, right in ((left_gs, wf48[2 * i], adc_L, ch_L, 0),
                                       (right_gs, wf48[2 * i + 1], adc_R,
                                        ch_R, 1)):
            ax = fig.add_subplot(gs[i])
            ax.plot(wf, color="#1f77b4")
            ax.set_xlim(0, T)
            ax.set_ylim(ymin_g, ymax_g)
            ax.set_yticks([])
            ax.set_xticks([] if i < 23 else xticks)
            if right:
                ax.yaxis.set_label_position("right")
            ax.set_ylabel(f"{adc}:{ch}", rotation=0,
                          labelpad=15 if right else 10, va="center")
    if is_left:
        rect = mpatches.Rectangle((-64.5, -65), 64, 130, linewidth=1,
                                  edgecolor="blue", facecolor="black")
        center_ax.set_xlim([-64.5, -0.5])
        center_ax.set_xticks([-49, -38, -27, -16])
    else:
        rect = mpatches.Rectangle((0.5, -65), 64, 130, linewidth=1,
                                  edgecolor="blue", facecolor="black")
        center_ax.set_xlim([0.5, 64.5])
        center_ax.set_xticks([16, 27, 38, 49])
    center_ax.add_patch(rect)
    if len(ySet):
        sc = center_ax.scatter(zSet, ySet, c=xSet, cmap=_CMAP, s=2, marker="s")
        cb = plt.colorbar(sc, cax=cbar_ax, orientation="vertical")
        cb.set_label("drift x [cm]")
    else:
        cbar_ax.axis("off")
    center_ax.set_ylim([-65, 65])
    center_ax.set_yticks([])
    center_ax.set_title("Single TPC Event Display", fontsize=14)
    fig.suptitle(title, fontsize=15)
    plt.show()


def show_pair(h5, tbl, charge_ev, light_ev: int, tpc=None,
              min_hits_tpc: int = 1):
    """Standard display(s): light waveforms of `light_ev` on the two walls,
    charge hits of `charge_ev` (may be None) in the center."""
    n_lev = h5["light/events/data"].shape[0]
    if not (0 <= int(light_ev) < n_lev):
        print(f"light event {light_ev} out of range (file has {n_lev})")
        return
    wf = data.format_light_waveform(tbl, int(light_ev))   # (8, 48, T)
    if charge_ev is not None:
        x, y, z, E, tpcs = charge_hits(h5, tbl, charge_ev)
        head = f"charge ev {charge_ev} + light ev {light_ev}"
    else:
        x = y = z = E = tpcs = np.array([])
        head = f"light ev {light_ev} (no charge)"
    if tpc is not None:
        tpc_list = list(range(8)) if tpc == "all" else [int(tpc)]
    elif len(tpcs):
        tpc_list = [t for t in range(8)
                    if int((tpcs == t).sum()) >= min_hits_tpc]
    else:
        if charge_ev is not None:
            print(f"charge event {charge_ev} has no hits — showing the "
                  f"brightest-light TPC instead")
        tpc_list = [int(np.argmax(wf.max(axis=(1, 2))))]
    for t in tpc_list:
        m = tpcs == t if len(tpcs) else np.array([], bool)
        nh = int(m.sum()) if len(tpcs) else 0
        _draw_tpc(np.asarray(wf[t], np.float32)[_PERM], t, tbl.lut,
                  ySet=y[m] if nh else np.array([]),
                  zSet=z[m] if nh else np.array([]),
                  xSet=x[m] if nh else np.array([]),
                  title=f"{head} | TPC {t} ({nh} hits)")


def show_light_only(h5, tbl, light_ev: int, tpc=None):
    """Display just the light of one light event (no charge needed).
    tpc = number, or "all" for all 8 TPCs, or None for the brightest."""
    show_pair(h5, tbl, None, light_ev, tpc=tpc)


def show_charge(h5, tbl, charge_ev: int, **kw):
    """Mode 2: look up the light partner automatically."""
    if "charge" not in h5:
        print("this file has no charge data (light-only file) — "
              "use show_light_only(h5, tbl, light_ev) instead")
        return
    lev = find_light_for_charge(tbl, charge_ev)
    if lev is None:
        print("no light found!")
        return
    print(f"charge ev {charge_ev} -> light ev {lev}")
    show_pair(h5, tbl, charge_ev, lev, **kw)


def show_light(h5, tbl, light_ev: int, **kw):
    """Mode 3: look up the charge partner automatically; on light-only
    files (or unassociated light events) falls back to a light-only view."""
    cev = find_charge_for_light(tbl, light_ev)
    if cev is None:
        if "charge" in h5:
            print("no charge event found for this light event — "
                  "showing light only")
        show_pair(h5, tbl, None, light_ev, **kw)
        return
    print(f"light ev {light_ev} -> charge ev {cev}")
    show_pair(h5, tbl, cev, light_ev, **kw)
