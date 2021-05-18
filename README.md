# usage

## charge event builder

To run charge event builder::

    mpiexec h5flow -c h5flow_yamls/charge_event_building.yaml -i <input file> -o <output file>

This generates the ``charge/raw_events`` and ``charge/packets`` datasets (see
``h5flow_modules/charge/raw_event_generator.py::RawEventGenerator``).
