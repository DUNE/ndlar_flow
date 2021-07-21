from h5flow.core import H5FlowResource

class Units(H5FlowResource):
    '''
        Provides a common set of units for Module 0 data. Base units are::

            us == mm == keV == e == K == 1

        To convert from an external source into Module 0 flow units, multiply::

            ext_val = 1 # m/s
            module0_val = ext_val * (resources['Units'].m / resources['Units'].s)
            module0_val # 0.001 mm/us

            1000 / 1000000

        To export a number from Module 0 flow units to a particular unit system,
        divide::

            module0_val = 0.05 # kV/mm
            ext_val = module0_val / (resources['Units'].kV / resources['Units'].cm)
            ext_val # 0.5 kV/cm

    '''
    # ~~~ time units ~~~
    s   = 1000000*us
    ms  = 1000*us
    us  = 1
    ns  = 1e-3*us

    larpix_ticks    = 100*ns
    lds_ticks       = 10*ns

    # ~~~ length units ~~~
    mm  = 1
    cm  = 10*mm
    m   = 1000*mm
    km  = 1000000*mm

    # ~~~ energy units ~~~
    GeV     = 1000000*keV
    MeV     = 1000*keV
    keV     = 1
    eV      = 1e-3*keV

    # ~~~ electrodynamic units ~~~
    e   = 1
    MV  = 1000*kV
    kV  = keV/e
    V   = 1e-3*kV
    mV  = 1e-6*kV

    # ~~~ temperature units ~~~
    K   = 1
