# Interactive Event Display Setup and Usage

This README provides instructions on how to set up and run the `interactive event display` application on the NERSC system (`perlmutter.nersc.gov`). This interactive event display tool is developed for visualizing simulations and data for the 2x2 demonstrator (and single modules).

## Overview

The `event display` requires specific setup steps to ensure all dependencies are installed. This README provides a step-by-step guide for the setup on NERSC. If you want to run the event display locally instead, there are some indicated steps you can skip.

## Setup Instructions

### Prerequisites

Before setting up, ensure you have the following prerequisites installed:

- SSH client for tunneling (`ssh`)
- Git for cloning repositories (`git`)
- Python environment management tool (`conda`)
- MPI compiler (`mpicc`)

### Setup Steps
0. **Log on to NERSC (skip for running locally)**:
  
  ```bash
  ssh -L 9999:localhost:8080 username@perlmutter.nersc.gov
  ```

1. **Clone Repositories**:
Make sure you have git set up (on NERSC) before you try this

  ```bash
  git clone git@github.com:DUNE/ndlar_flow.git
  git clone git@github.com:lbl-neutrino/h5flow.git
  ```
   
2. **Checkout Branch**:

  ```bash
  cd ndlar_flow
  git checkout feature/2x2-event-display
  cd ..
  ```

3. **Setup conda environment**:

  ```bash
  module load python # skip this if not on NERSC
  conda create -n myenv pip
  conda activate myenv
  ```

4. **Install h5flow with MPI support**:

  ```bash
  cd h5flow
  export CC=mpicc
  export HDF5_MPI="ON"
  pip install .
  ```

5. **Install Requirements**:

  ```bash
  cd ../ndlar_flow/event_display/interactive_event_display
  pip install -r requirements.txt
  ```

6. **Running the application**:

  ```bash
  python app.py
  ```

Now, you can locally access the event display by going to `localhost:9999` in the browser, or, if you are not on NERSC, by going to `localhost:8080`. Use the file upload or provide a path to a flow file (on nersc), and optionally provide a minerva file path to plot matching minerva tracks.
