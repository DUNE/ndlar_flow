environment
===========

First, download this code::

    git clone https://github.com/peter-madigan/module0_flow
    cd module0_flow

To install proper dependencies, use the provided conda environment file ``env.yaml``::

    conda env create -f env.yaml -n <environment name>
    conda activate <environment>

To update an existing environment::

    conda env update -f env.yaml -n <environment name>

The module0 flow code is built off of ``h5flow``
[https://github.com/peter-madigan/h5flow], so you will also need to install this
in order to run any of the workflows described here.

usage
=====

The ``module0_flow`` reconstruction chain breaks up the reconstruction into the
following steps:

    1. charge event building -> charge event reconstruction
    2. light event building -> light event recontruction
    3. charge-to-light association -> merged reconstruction

charge event building
---------------------

To run charge event builder::

    mpiexec h5flow -c h5flow_yamls/charge_event_building.yaml -i <input file> -o <output file>

This generates the ``charge/raw_events`` and ``charge/packets`` datasets. The
input file is a "datalog"- (a.k.a "packet"-) formatted LArPix data file.

charge event reconstruction
---------------------------

To run charge reconstruction::

    mpiexec h5flow -c h5flow_yamls/charge_event_reconstruction.yaml -i <input file> -o <output file>

This generates ``charge/packets_corr_ts``, ``charge/ext_trigs``, ``charge/hits``,
and ``charge/events`` datasets. The input file is a charge event built ``module0_flow``
file.

light event building
--------------------

To run light event builder::

    mpiexec h5flow -c h5flow_yamls/light_event_building.yaml -i <input file> -o <output file>

This generates the ``light/events`` and ``light/wvfm`` datasets. The input file
is a PPS-timestamp corrected "rwf_XX" root file produced by the adapted ADCViewer
code here [https://github.com/peter-madigan/ADCViewer64-Module0].

light event reconstruction
--------------------------

This is a work in progress... but performs low-level waveform processing
for light events. It takes as input a light event built ``module0_flow`` file.

charge-to-light association
---------------------------

To associate charge events to light events, run::

    mpiexec h5flow -c h5flow_yamls/charge_light_association.yaml -i <input file> -o <output file>

This creates references between ``charge/ext_trigs`` and ``light/events`` as well
as ``charge/events`` and ``light/events``. Both charge and light reconstructed
events are expected in the input file, which can be facilitated by running both
charge and light reconstruction flows on the same output file or by using
and hdf5 tool::

    # copy light data from a source file
    h5copy -v -f ref -s light -d light -i <light event file> -o <destination file>

    # copy charge data from a source file
    h5copy -v -f ref -s charge -d charge -i <charge event file> -o <destination file>

merged event reconstruction
---------------------------

This is a work in progress... but performs mid-level analysis making use of both
light system information and charge system information.

file structure and access
=========================

Let's walk through a simple example of how to access and use the hdf5
file format containing both light `and` charge data. As an example, we will
perform a mock analysis to compare the light system waveform integrals to the
larpix charge sum. First, we'll open up the file::

    import h5py
    f = h5py.File('<example file>.h5','r')

And list the available datasets using ``visititems``, which will call a specific
function on all datasets and groups within the file. In particular, let's
have it print out all available datasets::

    my_func = lambda name,dset : print(name) if isinstance(dset, h5py.Dataset) else None
    f.visititems(my_func)

This will print out quite a number of things, but you'll notice three different
types of paths:

 1. paths that end in ``.../data``
 2. paths that end in ``.../ref``
 3. paths that end in ``.../ref_region``

The first contain the primitive data for that particular object as a 1D
structured array, so for our example we want to access the charge sum for each
event. So first, let's check what fields are available in the
``'charge/events/data'`` dataset::

    print(f['charge/events/data'].dtype.names)

And then we can access the data by the field name::

    charge_qsum = f['charge/events/data']['q']

The second type of path (ending in ``.../ref``) contain bi-directional references
between two datasets. In particular, the paths to these datasets are structured
like ``<parent dataset name>/ref/<child dataset name>/ref``. Each entry in the
``.../ref`` dataset corresponds to a single link between the parent and child
datasets::

    f['charge/events/ref/light/events/ref'][0]
    # returns something like [1, 2], i.e. charge event 1 is linked to light event 2

You can directly use these references as indices into the corresponding dataset::

    ref = f['charge/events/ref/light/events/ref'][0]
    # get the first charge event that has a light event associated with it
    f['charge/events/data'][ref[0]]
    # get the light event associated with the first charge event
    f['light/events/data'][ref[1]]

You could loop over these references and load the rows of the dataset in that
way, but it would be very slow. Luckily ``h5flow`` offers a helper function
(``dereference``) to load references::

    from h5flow.data import dereference

    # reference dataset you want to use
    ref = f['charge/events/ref/light/events/ref']
    # data you want to load
    dset = f['light/events/data']
    # parent indices you want to use (i.e. event id 0-99)
    sel = slice(0,100)

    # this will load *ALL* references
    # and then find the data related to your selection
    data = dereference(sel, ref, dset)

    # other selections are possible, either integers or iterables
    dereference(0, ref, dset)
    defererence([0,1,2,3,1,0], ref, dset)

Data is loaded as a ``numpy`` masked array with shape ``(len(sel), max_ref)``.
So if there are only up to 5 light events associated any of the 100 charge
events we wanted before::

    data.shape # (100, 5)

The first index corresponds to our charge event selection and the second index
corresponds to the light event(s) that are associated with a given charge event.

This will likely take some time if you have a very large reference dataset
(>500k). To speed things up, we can can use the ``../ref_region`` datasets to
find out where in the reference dataset we need to look for each item. In
particular, this dataset provides a ``'start'`` and ``'stop'`` index for each
item::

    # get the bounds for where the first charge event references exist within the ref dataset
    region = f['charge/events/ref_region'][0]

    region['start'] # the first index in ref that is associated with charge event 0
    region['stop']  # the last index + 1 in ref that is associated with charge event 0

    # gets all references that *might* be associated with charge event 0
    ref = f['charge/events/ref/light/events/ref'][region['start']:region['stop']]

You can use this dataset with the helper function to load referred data in an
efficient way (this is the recommended approach)::

    region = f['charge/events/ref_region']

    # this will load only necessary references and then find the data related to your selection
    data = dereference(sel, ref, dset, region=region)

For datasets with a trivial 1:1 relationship (``light/events/data`` and
``light/wvfm/data`` in this case), you can directly use the references for one
of the datasets for any of the others::

    light_events = dereference(sel, ref, f['light/events/data'], region=region)
    light_wvfms = dereference(sel, ref, f['light/wvfm/data'], region=region)

Now that we have both the event information and the waveform data, we can
compare the charge sum of an event to the integral of the raw waveforms::

    import numpy.ma as ma # use masked arrays

    # first get the data
    charge_events = f['charge/events/data'][sel]
    print('charge_events:',charge_events.shape)
    print('light_events:',light_events.shape)
    print('light_wvfms:',light_wvfms.shape)

    # now apply a channel mask to the waveforms
    valid_wvfm = light_events['wvfm_valid'].astype(bool)
    # (event index, light event index, adc index, channel index)
    print('valid_wvfm',valid_wvfm.shape)
    channel_mask = np.zeros_like(valid_wvfm)
    # only use channel sum waveforms for both adcs
    channel_mask[:,:,:,[31,24,15,8,63,56,47,40]] = True

    samples = light_wvfms['samples']
    # (event index, light event index, adc index, channel index, sample index)
    print('samples:',samples.shape)
    # numpy masked arrays use the mask convention True == invalid
    samples.mask = samples.mask | np.expand_dims(~channel_mask,-1) | np.expand_dims(~valid_wvfm,-1)

    # calculating the integrals is now trivial!
    # axis 4 = integral over waveform, axis 3 = integral over valid channels,
    # axis 2 = integral over valid adcs, axis 1 = sum over light events
    light_integrals = samples.sum(axis=4).sum(axis=3).sum(axis=2).sum(axis=1)

We can now plot the correlation between the charge and light systems::

    import matplotlib.pyplot as plt

    plt.ion()
    plt.figure()
    plt.hist2d(charge_qsum, light_integrals, bins=(1000,1000))
    plt.xlabel('Charge sum [mV]')
    plt.ylabel('Light integral [ADC]')

For more details on what different fields in the datatypes mean, look at the
module-specific documentation. For more details on how to use the dereferencing
schema, look at the h5flow documentation [https://h5flow.readthedocs.io/en/latest/].





