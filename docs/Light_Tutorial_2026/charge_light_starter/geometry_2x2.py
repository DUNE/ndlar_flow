"""
Geometry, channel-ordering and timing constants for the 2x2 charge-light
matcher (vAlpha port).

Everything detector-specific that the matcher needs lives here:

  * The 8-TPC drift geometry (per-TPC x window, z sign) — kept identical to
    ``prep_2x2.py`` / ``ML_2x2_perceiver.py`` so the coordinates we feed the
    perceiver match the convention it was trained on.

  * The 48-channel light ordering.  This is the single most important
    convention in the whole package: the perceiver predicts the 48 light
    amplitudes in ``ORDERED_KEYS`` order ( [(0,0)..(0,23), (1,0)..(1,23)] ),
    so the OBSERVED light waveform must be assembled in exactly the same
    order or predicted-vs-observed channels silently misalign.  (The legacy
    v4 notebook used a *different* interleaved [L23,R23,...] order tied to the
    old 3D-CNN; we deliberately do NOT use that here.)

  * Time/tick conventions, derived empirically from the FLOW files:
        observed_flash_peak_sample = t_ns / NS_PER_TICK + 100
        template peaks at PULSE_PEAK_TICK (=105)
        => matched t0 (matching ticks) = peak_sample - PULSE_PEAK_TICK
        => flash/truth t0 seed         = t_ns / NS_PER_TICK - FLASH_T0_OFFSET
    with NS_PER_TICK=16, FLASH_T0_OFFSET=5 (mirrors the ND flash convention
    time/16 - 5).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from lut import LUT

# ----------------------------------------------------------------------------
# TPC / voxel geometry (must match prep_2x2.py and ML_2x2_perceiver.py)
# ----------------------------------------------------------------------------
N_TPCS = 8
N_CHANNELS = 48          # light channels per TPC
WVFM_LEN = 1000          # samples per light waveform
NX, NY, NZ = 32, 128, 64  # voxel grid (1 cm bins)

_X_RANGE_BY_TPC: Dict[int, Tuple[float, float]] = {
    0: (32.5, 64.5), 2: (32.5, 64.5),
    1: (2.5, 34.5),  3: (2.5, 34.5),
    4: (-34.5, -2.5), 6: (-34.5, -2.5),
    5: (-64.5, -32.5), 7: (-64.5, -32.5),
}
_POSZ_TPCS = {0, 1, 4, 5}
_NEGZ_TPCS = {2, 3, 6, 7}

# ----------------------------------------------------------------------------
# 48-channel light ordering — IDENTICAL to prep_2x2.ORDERED_KEYS
# ----------------------------------------------------------------------------
Y_RANGE = range(24)
ORDERED_KEYS: List[Tuple[int, int]] = (
    [(0, y) for y in Y_RANGE] + [(1, y) for y in Y_RANGE]
)  # len 48, index j -> (side, y_rel)

# ----------------------------------------------------------------------------
# Timing conventions (samples == "matching ticks")
# ----------------------------------------------------------------------------
PULSE_PEAK_TICK = 105     # avg_pulse.npy peaks here
NS_PER_TICK = 16.0        # light digitiser sampling (derived from flash table)
FLASH_T0_OFFSET = 5.0     # t0_seed = t_ns/NS_PER_TICK - FLASH_T0_OFFSET
SEARCH_RANGE = 895        # t0 scan range [0, SEARCH_RANGE]; 895 = last t0 with
                          # the full pulse peak (t0+105) inside the 1000-tick
                          # window — recovers late (e.g. delayed n-capture)
                          # flashes; A/B vs 700: efficiency unchanged (±0.1pp)
ADC_CLIP = 60780.0        # light ADC saturation cap used in chi2 model clip
BASELINE_TICKS = 75       # ticks used for per-channel baseline / noise estimate

# light TPC id == charge TPC id in the 2x2 (io_group - 1); no even/odd swap
def light_tpc_to_charge_tpc(light_tpc_id: int) -> int:
    return int(light_tpc_id)


def charge_tpc_to_light_tpc(charge_tpc_id: int) -> int:
    return int(charge_tpc_id)


def charge_tpc_from_io_group(io_group: np.ndarray) -> np.ndarray:
    """io_group (1..8) -> charge TPC id (0..7)."""
    return (np.asarray(io_group, dtype=np.int64) - 1).astype(np.int32)


# ----------------------------------------------------------------------------
# Light-channel lookup tables
# ----------------------------------------------------------------------------
def build_sipm_lut(h5) -> Dict[Tuple[int, int, int], Tuple[int, int]]:
    """(tpc, side, y_rel) -> (adc, ch) physical channel, from geometry_info."""
    rel_meta = h5["geometry_info/sipm_rel_pos"].attrs["meta"]
    rel_data = h5["geometry_info/sipm_rel_pos/data"]
    sipm_rel_pos = LUT.from_array(rel_meta, rel_data)
    lut: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
    for adc in range(8):
        for ch in range(64):
            try:
                tpc, side, y = sipm_rel_pos[(adc, ch)][0]
                lut[(int(tpc), int(side), int(y))] = (int(adc), int(ch))
            except Exception:
                pass
    return lut


def build_ordered_channel_index(lut: Dict[Tuple[int, int, int], Tuple[int, int]]):
    """Per-TPC (adc, ch) index arrays in ORDERED_KEYS order.

    Returns
    -------
    adc_idx, ch_idx : (8, 48) int arrays
        adc_idx[t, j], ch_idx[t, j] is the physical (adc, ch) feeding light
        channel j of TPC t, where ORDERED_KEYS[j] = (side, y_rel).  Entries
        with no mapping are -1 (those channels stay 0 in the formatted wvfm).
    valid : (8, 48) bool
        adc_idx >= 0.
    """
    adc_idx = np.full((N_TPCS, N_CHANNELS), -1, dtype=np.int64)
    ch_idx = np.full((N_TPCS, N_CHANNELS), -1, dtype=np.int64)
    for t in range(N_TPCS):
        for j, (side, y_rel) in enumerate(ORDERED_KEYS):
            key = (t, int(side), int(y_rel))
            if key in lut:
                adc_idx[t, j], ch_idx[t, j] = lut[key]
    valid = adc_idx >= 0
    return adc_idx, ch_idx, valid


def build_sipm_positions(h5) -> np.ndarray:
    """(N_TPCS, N_CHANNELS, 3) SiPM centers [cm] in the charge frame, in
    ORDERED_KEYS row order per TPC; NaN where the channel has no mapping."""
    lut = build_sipm_lut(h5)
    adc_idx, ch_idx, valid = build_ordered_channel_index(lut)
    abs_meta = h5["geometry_info/sipm_abs_pos"].attrs["meta"]
    abs_data = h5["geometry_info/sipm_abs_pos/data"]
    sipm_abs = LUT.from_array(abs_meta, abs_data)
    pos = np.full((N_TPCS, N_CHANNELS, 3), np.nan, dtype=np.float64)
    for t in range(N_TPCS):
        for j in range(N_CHANNELS):
            if not valid[t, j]:
                continue
            try:
                xyz = np.asarray(sipm_abs[(adc_idx[t, j], ch_idx[t, j])])[0]
            except Exception:
                continue
            if not np.all(xyz == -1.0):
                pos[t, j] = xyz
    return pos


def parse_dead_channels(yaml_path: str) -> set:
    """Read dead_channels_2x2.yaml -> set of (adc, ch)."""
    if not yaml_path:
        return set()
    try:
        import yaml
    except Exception:
        return set()
    with open(yaml_path, "r") as f:
        doc = yaml.safe_load(f) or {}
    out = set()
    for item in doc.get("dead_channels", []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.add((int(item[0]), int(item[1])))
    return out


def ordered_dead_mask(lut, dead_pairs: set) -> np.ndarray:
    """(8, 48) bool mask of dead light channels in ORDERED_KEYS order."""
    mask = np.zeros((N_TPCS, N_CHANNELS), dtype=bool)
    if not dead_pairs:
        return mask
    for t in range(N_TPCS):
        for j, (side, y_rel) in enumerate(ORDERED_KEYS):
            key = (t, int(side), int(y_rel))
            if key in lut and lut[key] in dead_pairs:
                mask[t, j] = True
    return mask


__all__ = [
    "N_TPCS", "N_CHANNELS", "WVFM_LEN", "NX", "NY", "NZ",
    "ORDERED_KEYS", "PULSE_PEAK_TICK", "NS_PER_TICK", "FLASH_T0_OFFSET",
    "SEARCH_RANGE", "ADC_CLIP", "BASELINE_TICKS",
    "light_tpc_to_charge_tpc", "charge_tpc_to_light_tpc", "charge_tpc_from_io_group",
    "build_sipm_lut", "build_ordered_channel_index",
    "parse_dead_channels", "ordered_dead_mask",
]
