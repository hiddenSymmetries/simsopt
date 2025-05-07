#!/usr/bin/env python
r"""
In this example, the segment currents in a toroidal wireframe are optimized
to generate a specific target normal field on a user-provided plasma boundary.

For this example, the wireframe is constructed such that its nodes lie on a 
toroidal surface a certain fixed distance from the target plasma boundary. 
However, in principle the user may specify any toroidal geometry for the
construction of the wireframe using the `SurfaceRZFourier` class.
"""

import os
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as pl
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt.solve import optimize_wireframe

# Number of wireframe segments per half period in the toroidal dimension
wf_n_phi = 8

# Number of wireframe segments in the poloidal dimension
wf_n_theta = 12

# Distance between the plasma boundary and the wireframe
wf_surf_dist = 0.3

# Average magnetic field on axis, in Teslas, to be produced by the wireframe.
# This will be used for setting the poloidal current constraint. The radius
# of the magnetic axis will be estimated from the plasma boundary geometry.
field_on_axis = 1.0

# Weighting factor for Tikhonov regularization (used instead of a matrix)
regularization_w = 10**-10

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename_equil = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Load the geometry of the target plasma boundary
plas_n_phi = 32
plas_n_theta = 32
surf_plas = SurfaceRZFourier.from_vmec_input(filename_equil,
                                             nphi=plas_n_phi, ntheta=plas_n_theta, range='half period')

# Construct the wireframe on a toroidal surface
surf_wf = SurfaceRZFourier.from_vmec_input(filename_equil)
surf_wf.extend_via_projected_normal(wf_surf_dist)
wf = ToroidalWireframe(surf_wf, wf_n_phi, wf_n_theta)

# Calculate the required net poloidal current and set it as a constraint
mu0 = 4.0 * np.pi * 1e-7
pol_cur = -2.0*np.pi*surf_plas.get_rc(0, 0)*field_on_axis/mu0
wf.set_poloidal_current(pol_cur)

# Set the optimization parameters
opt_params = {'reg_W': regularization_w}

# Run the RCLS optimization
t0 = time.time()
res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas=surf_plas,
                         verbose=False)
t1 = time.time()
delta_t = t1 - t0

# Verify that the solution satisfies all constraints
assert wf.check_constraints()

# Post-processing
res['wframe_field'].set_points(surf_plas.gamma().reshape((-1, 3)))
Bfield = res['wframe_field'].B().reshape((plas_n_phi, plas_n_theta, 3))
Bnormal = np.sum(Bfield * surf_plas.unitnormal(), axis=2)
modB = np.sqrt(np.sum(Bfield**2, axis=2))
rel_Bnorm = Bnormal/modB
area = np.sqrt(np.sum(surf_plas.normal()**2, axis=2))/float(modB.size)
mean_rel_Bn = np.sum(np.abs(rel_Bnorm)*area)/np.sum(area)
max_cur = np.max(np.abs(res['x']))
ndof = wf.n_segments - wf.constraint_matrices()[0].shape[0]

# Print post-processing results
print('')
print('Post-processing')
print('---------------')
print('  # dof          %12d' % (ndof))
print('  opt time [s]   %12.3f' % (delta_t))
print('  f_B [T^2m^2]   %12.4e' % (res['f_B']))
print('  f_R [T^2m^2]   %12.4e' % (res['f_R']))
print('  <|Bn|/|B|>     %12.4e' % (mean_rel_Bn))
print('  I_max [MA]     %12.4e' % (max_cur))

# Save plots and visualization data to files
wf.make_plot_2d(coordinates='degrees')
pl.savefig(OUT_DIR + 'rcls_wireframe_curr2d.png')
wf.to_vtk(OUT_DIR + 'rcls_wireframe')
