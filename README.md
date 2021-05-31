# environment

To install proper dependencies, use the provided conda environment file `env.yaml`::

    conda env create -f env.yaml -n <environment name>
    conda activate <environment>

To update an existing environment::

    conda env update -f env.yaml -n <environment name>

You will also need to install `h5flow` [https://github.com/peter-madigan/h5flow].

# usage

## charge event builder

To run charge event builder::

    mpiexec h5flow -c h5flow_yamls/charge_event_building.yaml -i <input file> -o <output file>

This generates the ``charge/raw_events`` and ``charge/packets`` datasets.

## charge reconstruction

To run charge reconstruction::

    mpiexec h5flow -c h5flow_yamls/charge_event_reconstruction.yaml -i <input file> -o <output file>

This generates ``charge/packets_corr_ts``, ``charge/ext_trigs``, ``charge/hits``,
and ``charge/events`` datasets.

## light event builder

To run light event builder::

    mpiexec h5flow -c h5flow_yamls/light_event_building.yaml -i <input file> -o <output file>

This generates the ``light/events`` and ``light/wvfm`` datasets

## charge-light association

To associate charge events to light events, run::

    mpiexec h5flow -c h5flow_yamls/charge_light_association.yaml -i <input file> -o <output file>

This creates references between ``charge/ext_trigs`` and ``light/events`` as well
as ``charge/events`` and ``light/events``.

# file format

Let's walk through a simple example of how to access and use the combined hdf5
file format. In particular, we will perform a mock analysis to compare the
light system waveform integrals to the larpix charge sum. First, we'll open up
the file::

    import h5py
    f = h5py.File('<example file>.h5','r')

And list the available datasets using `visititems`, which will call a specific
function on all datasets and groups within the file. In particular, let's
have it print out all available datasets::

    my_func = lambda name,dset : print(name) if isinstance(dset, h5py.Dataset) else None
    f.visititems(my_func)

You'll notice three different types of paths:
 1. paths that end in `.../data`
 2. paths that end in `.../ref`
 3. paths that end in `.../ref_valid`

The first contain the primitive data for that particular object as a 1D
structured array, so for our example we want to access the charge sum for each
event. So first, let's check what fields are available in the
`'charge/events/data'` dataset::

    print(f['charge/events/data'].dtype.names)

And then we can access the data by the field name::

    charge_qsum = f['charge/events/data']['q']

The second type of path (ending in `.../ref`) contain uni-directional references
between two datasets. In particular, the paths to these datasets are structured
like `<parent dataset name>/ref/<child dataset name>/ref`. Each entry in the
`.../ref` dataset has a 1:1 correspondence to the parent dataset::

    f['charge/events/data'].shape == f['charge/events/ref/light/events/ref'].shape
    f['charge/events/ref/light/events/ref'][26] # reference to light/events/data for event 26

The third type of path (ending in `.../ref_valid`) is a boolean array indicating
if the corresponding reference is non-null. This is needed due to how `h5py`
handles null reference, not how native HDF5 handles null references, so should
`h5py` be improved in the future this dataset will become irrelevant. But before
we dereference the charge -> light references, we will need to check if there
is a valid association::

    ref_mask = f['charge/events/ref/light/events/ref_valid'].astype(bool)
    ref_mask[0] # check if event 0 has light event(s) associated with it
    charge_qsum = charge_qsum[ref_mask[:]] # only keep charge sums from events that can be associated, [:] to load from the file

Now we can fetch the light data. First, h5py allows for the following access pattern::

    ref = f['charge/events/ref/light/events/ref']

    light_event_dset = f[ref[0]] # get the dataset associated with event 0, equivalent to f['light/events/data']
    light_event_dset[ref[0]] # dereference, equivalent to f['light/events/data'][<indices that are associated to the charge event>]

So we will use that to loop over the dataset and load all of the associated
light event ids::

    light_events = [light_event_dset[r] for r,m in zip(ref,ref_mask) if m]
    len(light_events) == len(charge_qsum)

Some datasets have trivial 1:1 relationships and can associated via
their `'id'` field rather than invoking the full HDF5 region reference
mechanics. We will use that to load and integrate the light waveforms for each
event::

    import numpy as np
    import numpy.ma as ma # use masked arrays for simpler math when using masks

    wvfm_dset = f['light/wvfm/data']
    print(wvfm_dset.dtype) # inspect the data type for the waveform data, `'samples'` shape is: (nadcs, nchannels, nsamples)

    # exclude certain channels from the event sum
    channel_mask = np.zeros((1,) + light_event_dset.dtype.fields['wvfm_valid'][0].shape + (1,), dtype=bool) # shape is (1,nadcs,nchannels)
    channel_mask[:,:,[31,24,15,8,63,56,47,40]] = True # just use channel sum waveforms for both adcs

    light_integral = np.zeros(len(charge_qsum), dtype='f8')
    for i,ev in enumerate(light_events):
        wvfms = wvfm_dset[ ev['id'] ]['samples'] # load raw waveforms, shape is (N, nadcs, nchannels, nsamples)
        mask = np.broadcast_to(channel_mask & np.expand_dims(ev['wvfm_valid'].astype(bool), -1), wvfms.shape) # mask for valid waveforms in event, reshape to match waveform dimensions
        wvfms = ma.masked_where(~mask, wvfms, copy=False) # create a masked array of waveforms, note that the masked array convention is True == invalid
        light_integral[i] = wvfms.sum() # sums over all light events, ADCs, channels, and samples (ignoring the masked channels)

Dereferencing may take a minute or two.

We can now plot the correlation between the charge and light systems::

    import matplotlib.pyplot as plt

    plt.ion()
    plt.figure()
    plt.hist2d(charge_qsum, light_integral, bins=(1000,1000))
    plt.xlabel('Charge sum [mV]')
    plt.ylabel('Light integral [ADC]')







