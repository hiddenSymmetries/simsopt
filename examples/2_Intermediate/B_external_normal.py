#!/usr/bin/env python

import os
from simsopt.mhd import VirtualCasing

"""
This example illustrates the virtual casing calculation, in which
we compute the contribution to the magnetic field on the plasma
boundary due to current inside the boundary, excluding currents
outside the plasma. Typically this calculation is run once at the end
of a stage-1 optimization (optimizing the plasma shape) before
starting the stage-2 optimization (optimizing the coil shapes). It is
only required for finite-beta plasmas, not for vacuum fields.

This example requires that the python virtual_casing module is installed.
"""

filename = os.path.join(os.path.dirname(__file__), '..', '..',
                        'tests', 'test_files', 'wout_li383_low_res_reference.nc')

# The "trgt_" resolution is the resolution on which results are
# computed, which you will then use for the stage-2 coil
# optimization. The "src_" resolution is used internally for the
# virtual casing calculation, and is often higher than the "trgt_"
# resolution for better accuracy.  Typically "src_ntheta" is not
# specified; in this case it is computed automatically from "src_nphi"
# to minimize anisotropy of the grid.
vc = VirtualCasing.from_vmec(filename, src_nphi=48, trgt_ntheta=32, trgt_nphi=32)
print('automatically determined src_ntheta:', vc.src_ntheta)

# The VirtualCasing.from_vmec command writes a file
# simsopt/tests/test_files/vcasing_li383_low_res_reference.nc
# containing the results of the virtual casing calculation.

# You can generate a matplotlib plot of B_external_normal on the
# boundary surface by uncommenting the following line:

# vc.plot()

# B_external_normal is now available as an attribute:
print('B_external_normal:')
print(vc.B_external_normal[:4, :4])  # Just print the first few elements

# The saved virtual casing results can be loaded in later for stage-2
# coil optimization, as follows:
directory, wout_file = os.path.split(filename)
vcasing_file = os.path.join(directory, wout_file.replace('wout', 'vcasing'))
vc2 = VirtualCasing.load(vcasing_file)

print('B_external_normal, loaded from file:')
print(vc2.B_external_normal[:4, :4])

