#!/usr/bin/env python

import os
from simsopt.mhd.virtual_casing import VirtualCasing

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

# Only the phi resolution needs to be specified. The theta resolution
# is computed automatically to minimize anisotropy of the grid.
vc = VirtualCasing.from_vmec(filename, src_nphi=30)
print('automatically determined src_ntheta:', vc.src_ntheta)

# The above command writes a file
# simsopt/tests/test_files/vcasing_li383_low_res_reference.nc
# containing the results of the virtual casing calculation.

# You can generate a matplotlib plot of B_internal_normal on the
# boundary surface by uncommenting the following line:

# vc.plot()
