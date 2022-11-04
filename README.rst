.. image:: https://readthedocs.org/projects/module0-flow/badge/?version=latest
    :target: https://module0-flow.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/peter-madigan/module0_flow/actions/workflows/test.yml/badge.svg
    :target: https://github.com/peter-madigan/module0_flow/actions/
    :alt: Test Status

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

If MPI is not available, you may use the alternative environment file
(``env-nompi.yaml``) that does not install parallel-HDF5. The module0 flow code
is built off of ``h5flow`` [https://github.com/peter-madigan/h5flow], so you
will also need to install this in order to run any of the workflows described
here.

To install::

    pip install .

tips for installing at NERSC
----------------------------

Because CORI uses a special build of MPICH MPI, you will need to follow the
instructions at [https://docs.nersc.gov/development/languages/python/parallel-python/]
to install a parallelized version of h5py *before* installing ``h5flow``. Note
that this environment will then only be usable on the CORI compute nodes.

If you'd like to set up a debugging environment that works on the CORI login
nodes, install using the ``*-nompi`` environment files. This will not allow you
to take advantage of the parallelism of ``h5flow``, and so isn't recommended for
production jobs.

usage
=====

There is no "executable" for ``module0_flow``, instead, it uses ``h5flow``
workflow yaml files and one of the entry points to ``h5flow``:
``python -m h5flow -c ...``, ``run_h5flow.py -c ...``, or ``h5flow -c ...``.
When this is run, it will set up a loop using the yaml file and
following the ``h5flow`` specification, namely,
a loop dataset, a series of "stages" to run on that dataset, and a set of
"resources" to make available to each stage. ``h5flow`` handles the
instantiation of python objects, the access to the data file, and the workflow
sequencing, while ``module0_flow`` provides workflow descriptions (under
``h5flow_yamls/workflows/``), module configurations (under
``h5flow_yamls/{reco,resources,...}``), and the source code for each module
for those workflows (under ``module0_flow/``). The intention behind using
``h5flow`` and separating the reconstruction and calibration into modules is
to allow for:

 1. flexibility - a purely modular workflow has better separation of code which enables easier collaboration between multiple developers and can allow for code reuse by making modules generic and re-usable for different purposes.
 2. portability - the only dependency for reading ``h5flow`` files is HDF5 which makes interfacing with ``module0_flow`` files easier, module-wise versioning makes developers think about compatiblitity early and often, and module-wise persistency makes adding new data objects easier and often with no reprocessing of data
 3. incrementalism - intermediate data objects can be saved and built upon, so development can occur from any intermediate step and data processing can be checkpointed at any stage.

That is the intention at any rate - if you have questions/comments/suggestions
for improvement, please feel welcome to open an issue at
[https://github.com/peter-madigan/module0_flow/issues] :)

There is a tutorial repo at
[https://github.com/peter-madigan/module0_flow_tutorial] that should help you
get started, but below are the basic building blocks for ``module0_flow``:

The ``module0_flow`` reconstruction chain breaks up the reconstruction into the
following steps for each component of the reconstruction. For charge-only
reconstruction, there are two workflows:

    1. charge_event_building.yaml
    2. charge_event_reconstruction.yaml

which are run in sequence. For light-only reconstruction, there are
corresponding workflows:

    1. light_event_building_adc64.yaml (and light_event_building_mc.yaml)
    2. light_event_recontruction.yaml

Finally, to perform the combined reconstruction using information from both
sub-systems, first generate the list of associations between the detectors
using ``charge_to_light_association.yaml``. Then run the reconstruction with
``combined_reconstruction.yaml``.

charge event building
---------------------

To run charge event builder::

    mpiexec h5flow -c h5flow_yamls/workflows/charge/charge_event_building.yaml \
        -i <input file> -o <output file>

This generates the ``charge/raw_events`` and ``charge/packets`` datasets. The
input file is a "datalog"- (a.k.a "packet"-) formatted LArPix data file. This
workflow step is the same for data and simulation.

charge event reconstruction
---------------------------

To run charge reconstruction::

    mpiexec h5flow -c h5flow_yamls/workflows/charge/charge_event_reconstruction.yaml \
        -i <input file> -o <output file>

This generates ``charge/packets_corr_ts``, ``charge/ext_trigs``, ``charge/hits``,
and ``charge/events`` datasets. The input file is a charge event built ``module0_flow``
file.

light event building
--------------------

To run light event builder on data::

    mpiexec h5flow -c h5flow_yamls/workflows/light/light_event_building_adc64.yaml \
        -i <input file> -o <output file>

This generates the ``light/events`` and ``light/wvfm`` datasets. The input file
is a raw ADC64-formatted .data file.

To run light event builder on simulation::

    mpiexec h5flow -c h5flow_yamls/workflows/light/light_event_building_adcmc.yaml \
        -i <input file> -o <output file>

This generates the same ``light/events`` and ``light/wvfm`` datasets as the data, but the input file
is a larnd-sim HDF5 file.

light event reconstruction
--------------------------

To run light reconstruction::

    mpiexec h5flow -c h5flow_yamls/workflows/light/light_event_reconstruction.yaml \
        -i <input file> -o <output file>

This generates ``light/t_ns`` and ``light/hits`` datasets. The input file is a light event built ``module0_flow``
file. The light event reconstruction also removes raw waveforms from the file.

charge-to-light association
---------------------------

To associate charge events to light events, run::

    mpiexec h5flow -c h5flow_yamls/workflows/charge/charge_light_association.yaml \
        -i <input file> -o <output file>

This creates references between ``charge/ext_trigs`` and ``light/events`` as well
as ``charge/events`` and ``light/events``. Both charge and light reconstructed
events are expected in the input file, which can be facilitated by running both
charge and light reconstruction flows on the same output file or by using
the ``h5copy`` hdf5 tool::

    # copy light data from a source file
    h5copy -v -f ref -s light -d light -i <light event file> \
        -o <destination file>

    # copy charge data from a source file
    h5copy -v -f ref -s charge -d charge -i <charge event file> \
        -o <destination file>

merged event reconstruction
---------------------------

To generate T0s and tracks, run::

    mpiexec h5flow -c h5flow_yamls/workflows/combined/combined_reconstruction.yaml \
        -i <input file> -o <output file>

minimal staging
---------------

Running these commands one after the other can be tedious, but with ``h5flow``
version 0.1.8, you can combine them into only two commands::

    output_file=<output file>

    mpiexec h5flow -c \
        h5flow_yamls/workflows/light/light_event_building_adc64.yaml \
        h5flow_yamls/workflows/light/light_event_reconstruction.yaml \
        -i <input light file> \
        -o $output_file

    mpiexec h5flow -c \
        h5flow_yamls/workflows/charge/charge_event_building.yaml \
        h5flow_yamls/workflows/charge/charge_event_reconstruction.yaml \
        h5flow_yamls/workflows/charge/charge_light_association.yaml \
        h5flow_yamls/workflows/combined/combined_reconstruction.yaml \
        -i <input charge file> \
        -o $output_file


file structure and access
=========================

Let's walk through an example of how to access and use the hdf5
file format containing both light `and` charge data using two different approaches:
the first is much more verbose, but is more flexible, while the second requires
minimal code, but has some limitations. As an example, we will
perform a mock analysis to compare the light system waveform integrals to the
larpix charge sum.

So let's start with the first approach, we'll open up the file using ``h5py``::

    import h5py
    f = h5py.File('<example file>.h5','r')

And list the available datasets using ``visititems``, which will call a specific
function on all datasets and groups within the file. In particular, let's
have it print out all available datasets::

    my_func = lambda name,dset : print(name) if isinstance(dset, h5py.Dataset) \
        else None
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
    print(charge_qsum.shape, charge_qsum.dtype)

The second type of path (ending in ``.../ref``) contain bi-directional references
between two datasets. In particular, the paths to these datasets are structured
like ``<parent dataset name>/ref/<child dataset name>/ref``. Each entry in the
``.../ref`` dataset corresponds to a single link between the parent and child
datasets::

    f['charge/events/ref/light/events/ref'][0]
    # returns something like [1, 2]

By convention, the first value corresponds to the index into the ``charge/events/data``
dataset and the second value corresponds to the index into the ``light/events/data``
dataset. To use, you can directly pass these references as indices into the
corresponding datasets::

    ref = f['charge/events/ref/light/events/ref'][0]
    # get the first charge event that has a light event associated with it
    f['charge/events/data'][ref[0]]
    # get the light event associated with the first charge event
    f['light/events/data'][ref[1]]

You could loop over these references and load the rows of the dataset in that
way, but it would be very slow. Instead, ``h5flow`` offers a helper function
(``dereference``) to load references::

    from h5flow.data import dereference

    # reference dataset you want to use
    ref = f['charge/events/ref/light/events/ref']
    # data you want to load
    dset = f['light/events/data']
    # parent indices you want to use (i.e. event id 0)
    sel = 0

    # this will load *ALL* the references
    # and then find the data related to your selection
    data = dereference(sel, ref, dset)

    # other selections are possible, either slices or iterables
    dereference(slice(0,100), ref, dset)
    dereference([0,1,2,3,1,0], ref, dset)

Data is loaded as a ``numpy`` masked array with shape ``(len(sel), max_ref)``.
So if there are only up to 5 light events associated any of the 100 charge
events we wanted before::

    print(data.shape, data.dtype) # e.g. (100, 5)

The first dimension corresponds to our charge event selection and the second dimension
corresponds to the light event(s) that are associated with a given charge event.

We can also load references with the opposite orientation (e.g.
``light/events -> charge/events``), by using the ``ref_direction`` argument::

    # we use the same reference dataset as before
    ref = f['charge/events/ref/light/events/ref']
    # but now we load from the charge dataset
    dset = f['charge/events/data']
    # and the parent indices correspond to positions within the light events
    sel = 0 # get charge events associated with the first light event

    # to load, we modify the reference direction from (0,1) [default] to (1,0)
    # since we want to use the second index of the ref dset as the "parent" and
    # the first index as the "child"
    data = dereference(sel, ref, dset, ref_direction=(1,0))
    print(data.shape, data.dtype)

Loading references can take some time if you have a very large reference dataset
(>50k). To speed things up, we can can use the ``../ref_region`` datasets to
find out where in the reference dataset we need to look for each item. In
particular, this dataset provides a ``'start'`` and ``'stop'`` index for each
item::

    # get the bounds for where the first charge event references exist within
    # the ref dataset
    sel = 0
    region = f['charge/events/ref/light/events/ref_region'][sel]

    # the first index in ref that is associated with charge event 0
    print(region['start'])
    # the last index + 1 in ref that is associated with charge event 0
    print(region['stop'])

    # gets all references that *might* be associated with charge event 0
    ref = f['charge/events/ref/light/events/ref'][region['start']:region['stop']]
    print(ref)

You can use this dataset with the helper function to load referred data in an
efficient way (this is the recommended approach)::

    sel = 0
    ref = f['charge/events/ref/light/events/ref']
    dset = f['light/events/data']

    region = f['charge/events/ref/light/events/ref_region']

    # this will load only necessary references and then find the data related
    # to your selection
    data = dereference(sel, ref, dset, region=region)

For datasets with a trivial 1:1 relationship (``light/events/data`` and
``light/wvfm/data`` in this case), you can directly use the references for one
of the datasets for any of the others::

    light_events = dereference(sel, ref, f['light/events/data'], region=region)
    light_wvfms = dereference(sel, ref, f['light/wvfm/data'], region=region)

Now that we have both the event information and the waveform data, we can
compare the charge sum of an event to the integral of the raw waveforms::

    import numpy.ma as ma # use masked arrays

    # we'll only look at a events 0-1000 since the raw waveforms will use a
    # lot of memory
    sel = slice(0,1000)

    # first get the data
    ref = f['charge/events/ref/light/events/ref']
    dset = f['light/events/data']
    region = f['charge/events/ref/light/events/ref_region']

    charge_events = f['charge/events/data'][sel]
    light_events = dereference(sel, ref, f['light/events/data'], region=region)
    light_wvfms = dereference(sel, ref, f['light/wvfm/data'], region=region)

    print('charge_events:',charge_events.shape)
    print('light_events:',light_events.shape)
    print('light_wvfms:',light_wvfms.shape)

    # now apply a channel mask to the waveforms to ignore certain channels
    # and waveforms
    valid_wvfm = light_events['wvfm_valid'].astype(bool)
    # (event index, light event index, adc index, channel index)
    print('valid_wvfm',valid_wvfm.shape)
    channel_mask = np.zeros_like(valid_wvfm)
    sipm_channels = np.array(
        [2,3,4,5,6,7] + [18,19,20,21,22,23] + [34,35,36,37,38,39] + \
        [50,51,52,53,54,55] + \
        [9,10,11,12,13,14] + [25,26,27,28,29,30] + [41,42,43,44,45,46] + \
        [57,58,59,60,61,62]
    )
    channel_mask[:,:,:,sipm_channels] = True

    samples = light_wvfms['samples']
    # (event index, light event index, adc index, channel index, sample index)
    print('samples:',samples.shape)
    # numpy masked arrays use the mask convention: True == invalid
    samples.mask = samples.mask | np.expand_dims(~channel_mask,-1) | \
        np.expand_dims(~valid_wvfm,-1)

    # now we can subtract the pedestals (using the mean of the first 50 samples)
    samples = samples.astype(float) - samples[...,:50].mean(axis=-1, keepdims=True)

    # and we can integrate over each of the dimensions:
    # axis 4 = integral over waveform, axis 3 = sum over valid channels,
    # axis 2 = sum over valid adcs, axis 1 = sum over light events associated
    #          to a charge event
    light_integrals = samples.sum(axis=4).sum(axis=3).sum(axis=2).sum(axis=1)

    # we can either create a mask for only the valid entries (i.e. the charge-
    # to-light association exists)
    valid_event_mask = ~light_integrals.mask
    # or we can zero out the invalid entries (beware: this will update the
    # light_integral.mask to indicate that these are now valid entries)
    light_integrals[light_integrals.mask] = 0.

And we plot the correlation between the charge and light systems::

    import matplotlib.pyplot as plt

    plt.ion()
    plt.figure()
    plt.hist2d(charge_qsum[valid_event_mask], light_integrals[valid_event_mask],
        bins=(1000,1000))
    plt.xlabel('Charge sum [mV]')
    plt.ylabel('Light integral [ADC]')

``h5flow`` also has the capability of traversing multiple references using
the ``dereference_chain`` helper function. I will leave it to you to visit the
``h5flow`` docs and to play around with this functionality.

Ok, so that's how to access data using the verbose and flexible approach. Now
let's do it the quick and easy way.

We'll use an ``H5FlowDataManager`` object to help::

    from h5flow.data import H5FlowDataManager
    dm = H5FlowDataManager('<input file>', 'r', mpi=False)

This object has built-in smart reference traversal via the ``__getitem__``
special method. If one argument is specified, it acts as a pass-through to an
underlying ``h5py.File``::

    dm['light/events/data'] # get the light events dataset
    dm['light/events'] # get the light events group
    dm['light/events'].attrs # get light event attributes

But when using multiple arguments, it will load references::

    # again lets get the first 1000 charge events
    charge_events = dm['charge/events/data', sel]
    # (event index,)

    # and now we use the fancy access method
    light_events = dm['charge/events','light/events',sel]
    # (event index, light event index)

    # and we can also get the waveforms, but only if the light/events -> light/wvfm references exist
    light_wvfm = dm['charge/events','light/events','light/wvfm',sel]
    # (event index, light event index, light waveform index)

That's certainly much cleaner! But in this case, you are limited in only traversing
references that are explicitly defined so references can't do double duty for
multiple datasets. You also are not able to just load the reference index by
itself. So, this approach might not be suited for every situation.

There is also a plotting script at ``scripts/map_file.py`` which will generate
a map of all of the references included in the file. You will need ``networkx``
installed in order to run this. Run with::

    python scripts/map_file.py <file>

And that concludes the intro into the data access!

For more details on what different fields in the datatypes mean, look at the
module-specific documentation. For more details on how to use the dereferencing
schema, look at the h5flow documentation [https://h5flow.readthedocs.io/en/latest/].
