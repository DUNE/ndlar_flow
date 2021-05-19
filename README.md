# usage

## charge event builder

To run charge event builder::

    mpiexec h5flow -c h5flow_yamls/charge_event_building.yaml -i <input file> -o <output file>

This generates the ``charge/raw_events``, ``charge/packets``,
``charge/packets_corr_ts``, ``charge/ext_trigs``, ``charge/hits``,
and ``charge/events`` datasets.
