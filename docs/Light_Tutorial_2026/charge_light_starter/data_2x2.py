"""
Per-event data access for the 2x2 charge-light matcher.

Reads a FLOW HDF5 file and produces, for one charge event:
  * charge hits        : x, y, z (cm), E (MeV), tpc id (0..7), hit row refs
  * observed light     : fullLightWaveform (8, 48, 1000), baseline-subtracted,
                         assembled in ORDERED_KEYS order (perceiver convention)
  * light noise model  : fullLightVar (8, 48, 1000), a self-contained
                         heteroscedastic variance used as the chi2 denominator
  * flash t0 seeds     : per-TPC list of candidate t0 (matching ticks) from the
                         light/flash table

File-level arrays (refs, LUTs) are cached per open file handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import geometry_2x2 as geo


# ----------------------------------------------------------------------------
# File-level cache
# ----------------------------------------------------------------------------
@dataclass
class FileTables:
    h5: Any
    hits_full: Any                      # charge/<hits>/data dataset handle
    hits_ref: np.ndarray                # (Nhit,2) [charge_event_id, hit_row]
    charge_light_ref: np.ndarray        # (Npair,2) [charge_event_id, light_event_id]
    wvfm_dset: Any                       # light/wvfm/data dataset handle
    evt_flash_ref: np.ndarray           # (Nfl,2) [light_event_id, flash_id]
    flash_data: Any                      # light/flash/data dataset handle
    adc_idx: np.ndarray                  # (8,48)
    ch_idx: np.ndarray                   # (8,48)
    valid: np.ndarray                    # (8,48)
    dead_mask: np.ndarray                # (8,48) bool
    lut: Dict                            # (tpc,side,y_rel)->(adc,ch)


_FILE_CACHE: Dict[str, FileTables] = {}


def _file_key(h5) -> str:
    return str(getattr(h5, "filename", f"<h5-{id(h5)}>"))


def get_tables(h5, *, hits_dset: str = "calib_prompt_hits",
               dead_yaml: str = "") -> FileTables:
    key = _file_key(h5)
    cached = _FILE_CACHE.get(key)
    if cached is not None:
        return cached

    hits_full = h5[f"charge/{hits_dset}/data"]
    hits_ref = np.asarray(
        h5[f"charge/events/ref/charge/{hits_dset}/ref"][()], dtype=np.int64)
    charge_light_ref = np.asarray(
        h5["charge/events/ref/light/events/ref"][()], dtype=np.int64)
    wvfm_dset = h5["light/wvfm/data"]
    evt_flash_ref = np.asarray(
        h5["light/events/ref/light/flash/ref"][()], dtype=np.int64)
    flash_data = h5["light/flash/data"]

    lut = geo.build_sipm_lut(h5)
    adc_idx, ch_idx, valid = geo.build_ordered_channel_index(lut)
    dead_pairs = geo.parse_dead_channels(dead_yaml) if dead_yaml else set()
    dead_mask = geo.ordered_dead_mask(lut, dead_pairs)

    tbl = FileTables(
        h5=h5, hits_full=hits_full, hits_ref=hits_ref,
        charge_light_ref=charge_light_ref, wvfm_dset=wvfm_dset,
        evt_flash_ref=evt_flash_ref, flash_data=flash_data,
        adc_idx=adc_idx, ch_idx=ch_idx, valid=valid, dead_mask=dead_mask, lut=lut)
    _FILE_CACHE[key] = tbl
    return tbl


def clear_cache() -> None:
    _FILE_CACHE.clear()


# ----------------------------------------------------------------------------
# Per-event light formatting
# ----------------------------------------------------------------------------
def raw_baseline_subtracted(tbl: FileTables, light_event_id: int) -> np.ndarray:
    """Raw light/wvfm minus per-channel baseline: (8, 64, 1000), physical (adc,ch).

    Baseline = mean of the first ``BASELINE_TICKS`` samples.  This is the
    physical-channel form (indexed by adc, ch) used by the mc_truth validation,
    which keys off (adc, ch) visibility.
    """
    raw = tbl.wvfm_dset[int(light_event_id)]["samples"].astype(np.float32)  # (8,64,1000)
    bl = raw[..., :geo.BASELINE_TICKS].mean(axis=-1, keepdims=True)
    return (raw - bl).astype(np.float32)


def format_from_sub(tbl: FileTables, sub: np.ndarray) -> np.ndarray:
    """Reorder a (8,64,1000) baseline-subtracted waveform into the perceiver's
    (8, 48, 1000) ORDERED_KEYS layout (dead channels zeroed)."""
    out = np.zeros((geo.N_TPCS, geo.N_CHANNELS, sub.shape[-1]), dtype=np.float32)
    for t in range(geo.N_TPCS):
        good = tbl.valid[t]
        out[t, good] = sub[tbl.adc_idx[t][good], tbl.ch_idx[t][good]]
        out[t, tbl.dead_mask[t]] = 0.0
    return out


def format_light_waveform(tbl: FileTables, light_event_id: int) -> np.ndarray:
    """Convenience: raw light/wvfm -> (8, 48, 1000) ORDERED_KEYS waveform.

    peak(formatted) reproduces the perceiver's trained ``phi`` target (prep_2x2
    used the same baseline definition).
    """
    return format_from_sub(tbl, raw_baseline_subtracted(tbl, light_event_id))


def light_noise_variance(wvfm: np.ndarray, *, rel_sigma: float = 0.10,
                         var_floor: float = 2.5e3,
                         dead_mask: Optional[np.ndarray] = None,
                         big_var: float = 1.0e12,
                         sat_threshold: float = 6.07e4) -> np.ndarray:
    """Self-contained heteroscedastic variance (chi2 denominator), (8,48,1000).

    var[t,c,k] = max(baseline_var[t,c], var_floor) + (rel_sigma * max(obs,0))^2

    The signal-proportional term down-weights bright-channel ticks (where the
    perceiver's amplitude imperfection dominates) while keeping faint channels
    sharp — which is what makes small/low-energy blobs matchable.

    Saturated ticks (raw amplitude near the ADC rail) get a huge variance: a
    railed channel reads a censored lower bound, so penalising the model for not
    reaching it would REPEL the match from a bright (saturated) flash — the main
    cause of big-cluster wrong-flash assignments.  Dead channels likewise.
    """
    wvfm = np.asarray(wvfm, dtype=np.float32)
    base = wvfm[..., :geo.BASELINE_TICKS]
    base_var = base.var(axis=-1, keepdims=True)                    # (8,48,1)
    base_var = np.maximum(base_var, np.float32(var_floor))
    sig = np.clip(wvfm, 0.0, None)
    var = base_var + (np.float32(rel_sigma) * sig) ** 2
    if dead_mask is not None:
        var[dead_mask] = np.float32(big_var)
    return var.astype(np.float32)


# ----------------------------------------------------------------------------
# Flash t0 seeds
# ----------------------------------------------------------------------------
def flash_t0_seeds(tbl: FileTables, light_event_id: int, *,
                   min_tot_max: float = 0.0,
                   merge_ticks: float = geo.NS_PER_TICK and 5.0) -> List[List[float]]:
    """Per-TPC candidate t0 (matching ticks) from the flash table.

    t0_seed = hit_time_range[0] / NS_PER_TICK - FLASH_T0_OFFSET
            (== sample_range_start - PULSE_PEAK_TICK).

    Near-duplicate seeds within ``merge_ticks`` in the same TPC are merged to
    their median.  Returns a list of 8 lists (one per TPC).
    """
    seeds: List[List[float]] = [[] for _ in range(geo.N_TPCS)]
    fids = tbl.evt_flash_ref[tbl.evt_flash_ref[:, 0] == int(light_event_id), 1]
    if fids.size == 0:
        return seeds
    fids = np.unique(fids)
    tpc = tbl.flash_data["tpc"][:]
    htr = tbl.flash_data["hit_time_range"][:]
    tot_max = tbl.flash_data["tot_max"][:]
    for fid in fids:
        fid = int(fid)
        ltpc = int(tpc[fid])
        ctpc = geo.light_tpc_to_charge_tpc(ltpc)
        if not (0 <= ctpc < geo.N_TPCS):
            continue
        if float(tot_max[fid]) < float(min_tot_max):
            continue
        t_ns = float(htr[fid][0])
        t0 = t_ns / geo.NS_PER_TICK - geo.FLASH_T0_OFFSET
        if not np.isfinite(t0):
            continue
        # Drop flashes outside the matchable window rather than clipping them to
        # the boundary: a late flash clipped to SEARCH_RANGE becomes a spurious
        # edge seed that faint blobs wrongly snap to.
        if t0 < 0.0 or t0 > geo.SEARCH_RANGE:
            continue
        seeds[ctpc].append(t0)
    # merge near-duplicates per TPC
    for t in range(geo.N_TPCS):
        vals = sorted(seeds[t])
        if not vals:
            continue
        groups = [[vals[0]]]
        for v in vals[1:]:
            if abs(v - groups[-1][-1]) <= float(merge_ticks):
                groups[-1].append(v)
            else:
                groups.append([v])
        seeds[t] = [float(np.median(g)) for g in groups]
    return seeds


# ----------------------------------------------------------------------------
# Per-event bundle
# ----------------------------------------------------------------------------
@dataclass
class Event2x2:
    ev_id: int
    light_event_id: int
    hit_refs: np.ndarray
    xset: np.ndarray
    yset: np.ndarray
    zset: np.ndarray
    Eset: np.ndarray
    hitTPCid: np.ndarray
    fullLightWaveform: np.ndarray      # (8,48,1000)
    fullLightVar: np.ndarray           # (8,48,1000)
    flash_seeds: List[List[float]]
    extras: Dict[str, Any] = field(default_factory=dict)


def load_event(h5, ev_id: int, *, hits_dset: str = "calib_prompt_hits",
               dead_yaml: str = "", rel_sigma: float = 0.10,
               variance_model=None, unit_variance: bool = False) -> Optional[Event2x2]:
    """Assemble everything the matcher needs for one charge event.

    Returns None if the event has no associated light event.
    """
    tbl = get_tables(h5, hits_dset=hits_dset, dead_yaml=dead_yaml)
    ev_id = int(ev_id)

    rows = np.flatnonzero(tbl.charge_light_ref[:, 0] == ev_id)
    if rows.size == 0:
        return None
    light_event_id = int(tbl.charge_light_ref[rows[0], 1])

    hit_mask = tbl.hits_ref[:, 0] == ev_id
    hit_refs = tbl.hits_ref[hit_mask, 1]
    if hit_refs.size == 0:
        return None
    hits_evt = tbl.hits_full[hit_refs]

    xset = np.asarray(hits_evt["x"], dtype=np.float64)
    yset = np.asarray(hits_evt["y"], dtype=np.float64)
    zset = np.asarray(hits_evt["z"], dtype=np.float64)
    Eset = np.asarray(hits_evt["E"], dtype=np.float64)
    hitTPCid = geo.charge_tpc_from_io_group(hits_evt["io_group"])

    # drop non-finite hits (data calib can carry NaN positions / energy)
    finite = (np.isfinite(xset) & np.isfinite(yset) &
              np.isfinite(zset) & np.isfinite(Eset))
    if not finite.all():
        hit_refs = hit_refs[finite]
        xset, yset, zset, Eset = xset[finite], yset[finite], zset[finite], Eset[finite]
        hitTPCid = hitTPCid[finite]

    raw_sub = raw_baseline_subtracted(tbl, light_event_id)        # (8,64,1000)
    fullLightWaveform = format_from_sub(tbl, raw_sub)              # (8,48,1000)
    if unit_variance:
        # Plain unit variance (== ND vAlpha): chi2 is the raw sum of squared
        # residuals; dead channels still censored.
        fullLightVar = np.ones_like(fullLightWaveform, dtype=np.float32)
        fullLightVar[tbl.dead_mask] = np.float32(1.0e12)
    elif variance_model is not None:
        fullLightVar = variance_model.predict_variance(
            raw_sub, tbl, dead_mask=tbl.dead_mask)
        # A model trained on the (perceiver - observed) residual already captures
        # the perceiver's model error; an old measurement-only net does not, so
        # add a signal-proportional slack term in that case.
        if not getattr(variance_model, "includes_model_error", False):
            sig = np.clip(fullLightWaveform, 0.0, None)
            fullLightVar = fullLightVar + (np.float32(rel_sigma) * sig) ** 2
        fullLightVar[tbl.dead_mask] = np.float32(1.0e12)
    else:
        fullLightVar = light_noise_variance(
            fullLightWaveform, rel_sigma=rel_sigma, dead_mask=tbl.dead_mask)
    flash_seeds = flash_t0_seeds(tbl, light_event_id)

    return Event2x2(
        ev_id=ev_id, light_event_id=light_event_id, hit_refs=hit_refs,
        xset=xset, yset=yset, zset=zset, Eset=Eset, hitTPCid=hitTPCid,
        fullLightWaveform=fullLightWaveform, fullLightVar=fullLightVar,
        flash_seeds=flash_seeds,
        extras={"hits_evt": hits_evt, "n_hits": int(hit_refs.size),
                "raw_sub": raw_sub})


__all__ = [
    "FileTables", "get_tables", "clear_cache",
    "raw_baseline_subtracted", "format_from_sub", "format_light_waveform",
    "light_noise_variance", "flash_t0_seeds",
    "Event2x2", "load_event",
]
