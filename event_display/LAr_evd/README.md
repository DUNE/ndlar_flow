# ND LAr + TMS Event Display

This directory contains the ND LAr + TMS event display (`ndlar_evd.py`) and an example notebook (`ndlar_evd_example.ipynb`).

## Setup

### 1. Create and activate a conda environment

```bash
conda create -n ndlar_evd python=3.11
conda activate ndlar_evd
```

### 2. Install dependencies

From this directory, install all required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Note:** The `fitz` module is provided by the `PyMuPDF` package, which is listed in `requirements.txt`.

### 3. Register the environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name ndlar_evd --display-name "Python (ndlar_evd)"
```

## Running the notebook

### 1. Add `ndlar_flow` to your path

The event display relies on utilities from `ndlar_flow`. Before opening the notebook, make sure the path to `ndlar_flow/src` is accessible. The first cell of `ndlar_evd_example.ipynb` also includes a `sys.path.append` call — update it to point to your local copy of `ndlar_flow/event_display/LAr_evd/`.

### 2. Launch Jupyter

```bash
jupyter notebook ndlar_evd_example.ipynb
```

or, if you prefer JupyterLab:

```bash
jupyter lab ndlar_evd_example.ipynb
```

### 3. Select the kernel

In the Jupyter interface, select the **Python (ndlar_evd)** kernel you registered above.

### 4. Configure input files

In the second cell of the notebook, set the paths to your input files:

- `d` / `f`: directory and filename of the ndlar-flow HDF5 file.
- `tms_dir` / `tms_filename`: directory and filename of the TMS ROOT file (set `tms_file = None` to skip TMS).

Then run all cells to launch the interactive event display.

## Dependencies

| Package        | PyPI name       | Notes                                  |
|----------------|-----------------|----------------------------------------|
| `fitz`         | `PyMuPDF`       | PDF rendering for logo overlays        |
| `numpy`        | `numpy`         |                                        |
| `pandas`       | `pandas`        |                                        |
| `ipywidgets`   | `ipywidgets`    | Interactive widgets in Jupyter         |
| `h5py`         | `h5py`          | Reading ndlar-flow HDF5 files          |
| `cmasher`      | `cmasher`       | Colour maps                            |
| `matplotlib`   | `matplotlib`    |                                        |
| `PIL`          | `Pillow`        |                                        |
| `uproot`       | `uproot`        | Reading TMS ROOT files                 |
| `periodictable`| `periodictable` | Particle physics utilities             |
| `particle`     | `particle`      | PDG particle database                  |
| `jupyter`      | `jupyter`       | Notebook interface                     |
