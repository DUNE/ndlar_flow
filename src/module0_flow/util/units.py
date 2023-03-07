'''
    Provides a common set of units for Module 0 data. Base units are::

        us == mm == keV == e == K == 1

    To convert from an external source into Module 0 flow units, multiply::

        import module0_flow.util.units as units

        ext_val = 1 # m/s
        module0_val = ext_val * (units.m / units.s)
        module0_val # 0.001 mm/us

    To export a number from Module 0 flow units to a particular unit system,
    divide::

        import module0_flow.util.units as units

        module0_val = 0.05 # kV/mm
        ext_val = module0_val / (units.kV / units.cm)
        ext_val # 0.5 kV/cm

'''
# ~~~ time units ~~~
#: microseconds
us = 1
#: seconds
s = 1000000 * us
#: milliseconds
ms = 1000 * us
#: nanoseconds
ns = 1e-3 * us

# ~~~ length units ~~~
#: millimeter
mm = 1
#: centimeter
cm = 10 * mm
#: meter
m = 1000 * mm
#: kilometer
km = 1000000 * mm

# ~~~ energy units ~~~
#: kiloelectron-volts
keV = 1
#: electron-volts
eV = 1e-3 * keV
#: gigaelectron-volts
GeV = 1000000 * keV
#: megaelectron-volts
MeV = 1000 * keV

# ~~~ electrodynamic units ~~~
#: electron charge
e = 1
#: kilovolts
kV = keV / e
#: volts
V = 1e-3 * kV
#: millivolts
mV = 1e-6 * kV

# ~~~ temperature units ~~~
#: Kelvin
K = 1

# ~~~ mass units ~~~
#: gram
g = 1
#: kilogram
kg = 1000 * g
