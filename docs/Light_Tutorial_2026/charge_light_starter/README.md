# 2x2 charge-light starter kit

Self-contained event-display toolkit for 2x2 FLOW files (charge hits + light
waveforms + the charge<->light event association). No GPU needed.

Also works with LIGHT-ONLY files (nearline `flowed_light`, e.g. cold
commissioning / purity runs): any waveform window length is handled, and if
the file carries no `geometry_info` the bundled `sipm_lut.json` channel map
(from july10_2024) is used — a note is printed; verify if that run's cabling
differs. Use `show_light_only(h5, tbl, light_ev, tpc="all")` there
(notebook Mode 4).

## Get started (NERSC)

```bash
cp -r /pscratch/sd/y/yuxuan/2x2QLMatching/charge_light_starter  /your/dir/
```

Then open `quickstart.ipynb` at https://jupyter.nersc.gov (login node kernel
is fine) and run the cells. Three modes:

1. give a charge event id AND a light event id -> full display
   (light waveforms on the two walls, charge in the center);
2. give only a charge event id -> the light partner is looked up for you
   (prints `no light found!` if there is none);
3. give only a light event id -> reverse lookup of the charge partner
   (same protection).

Optional nicer colors: `pip install --user cmasher cmcrameri` (one time).

## What's in here

| file | what |
|---|---|
| `quickstart.ipynb` | the notebook — start here |
| `cl_display.py` | display + lookup functions (`open_flow`, `show_pair`, `show_charge`, `show_light`, `find_light_for_charge`, `find_charge_for_light`, `charge_hits`) |
| `data_2x2.py`, `geometry_2x2.py`, `lut.py` | FLOW readers: event tables, channel mapping (tpc, side, y) <-> (adc, ch), waveform formatting (8 TPC x 48 ch x 1000 ticks, baseline-subtracted) |
| `dead_channels_2x2.yaml` | 17 known dead light channels (zeroed in displays) |

## Conventions worth knowing

* Light waveforms: 1000 ticks x 16 ns = 16 us window; trigger at tick ~100.
  Beam light arrives at ~1.6 us (tick ~100); cosmic self-trigger flashes sit
  at tick ~433. Channel labels are `adc:channel`.
* Each TPC has 2 light walls x 24 channels; left boxes = side 0 (low z),
  right = side 1 (high z). ArCLight tiles = y_rel 0-5 & 12-17, LCM = 6-11 &
  18-23.
* The charge readout window is ~194 us after the trigger, the light window
  only 16 us — so some charge (slow drift, pre-trigger tracks) legitimately
  has no light in the window.

Data lives under `/global/cfs/cdirs/dune/www/data/2x2/` (beam data:
`reflows/v10/flow/beam/<day>/nominal_hv/*.FLOW.hdf5`; simulation:
`simulation/productions/MiniRun6.4_1E19_RHC/...`).

For the full charge-light MATCHER (ML light prediction + t0 association),
ask Yuxuan for the QLMatching2x2 package — this kit is the display/data
layer of it.
