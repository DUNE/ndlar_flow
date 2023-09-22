# Flow File Event Displays

## Single Module Event Displays

### Overview 

There are currently two Python-based event displays available for visualizing single module data and simulation files run through flow. The event displays are version-specific, with one intended for use on files run through `module0_flow` and one for use on files run through `proto_nd_flow`. The source code for each event display is available in the corresponding subdirectory, e.g. `event_display/proto_nd_flow`. The event displays are set up to run inside Jupyter notebooks, and examples of such usage are also given in the corresponding subdirectories. The examples involve using files stored on NERSC, so if the notebooks are run outside of NERSC, you will need to download both the flow file you would like to visualize and a module geometry file.

### File Locations

In general, single module data files can be found [in this file system.](https://portal.nersc.gov/project/dune/data/) For the examples provided in the associated notebooks, the following files are used:

 - [Module 1 geometry file](https://portal.nersc.gov/project/dune/data/Module1/TPC12/module1_layout-2.3.16.yaml)
 - [Module 1 data file run through `module0_flow` (charge only)](https://portal.nersc.gov/project/dune/data/Module1/reco/charge_only/events_2022_02_09_17_23_09_CET.gz.h5)
 - Module 1 data file run through `proto_nd_flow` (charge only) (web link not currently available)
 
Additional Module 1 charge-only data files run through `module0_flow` can be found [here](https://portal.nersc.gov/project/dune/data/Module1/reco/charge_only/), and additional Module 1 charge-only data files run through `proto_nd_flow` can be found here (location will be updated when available). 
