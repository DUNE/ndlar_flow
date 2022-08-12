Light geometry description file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Geometry`` resource uses a YAML config file to describe the existing light
detector and their locations within the module. This YAML file must have the
following keys:
 - ``format_version``: version number (``major.minor.fix``) of the YAML file formatting
 - ``geometry_version``: version number (``major.minor.fix``) of the geometry
 - ``geom``: dictionary describing the shape of each detector module (described in more detail below)
 - ``det_center``: dictionary describing the position of each detector module (described in more detail below)
 - ``tpc_center``: dictionary for the center coordinate of each TPC (``<TPC #>: [x,y,z]``)
 - ``det_geom``: dictionary for detector shape lookup (``<DET #>: <GEOM #>``), assumed to be the same for each TPC
 - ``det_adc``: dictionary for the ADC index used for each detector (``<TPC #>: <DET #>: <ADC INDEX>``)
 - ``det_chan``: dictionary for the channel indices used for each detector (``<TPC #>: <DET #>: [<CHANNEL #s>]``)

Generally each detector module is assigned two numbers: a ``TPC #`` and a ``DET #``.
The first refers to the drift region (2 per module) that the detector module
sees. The second refers the position along the field cage wall the detector
inhabits - with a convention of the (-x, -y) corner being ``DET # == 0`` and
increasing with with y position. Each detector module can consist of more than
one SiPM, as reflected in the ``det_chan``, and will be treated as though they
have the same position and acceptance.

The ``geom`` key describes each different light detector geometry, assuming they
are rectangular. The format of this dictionary is::

    <DET GEOM #>: {
        min: [x,y,z],
        max: [x,y,z]
    }

The ``det_center`` key describes the position and geometry of each light
detector. The format of this dictionary is::

    <DET #>: {
        geom: <DET GEOM #>,
        center: [x,y,z]
    }
