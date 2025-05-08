#!/usr/bin/env python
r"""
In this example, the segment currents in a toroidal wireframe are optimized
using the Greedy Stellarator Coil Optimization (GSCO) procedure to produce
a design for modular coils for a stellarator plasma. This is done by 
initializing the wireframe to have a set of planar toroidal-field coils
and then using GSCO to optimize the shapes of those coils to produce the
field shaping required for the target plasma equilibrium.
"""

import os
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as pl
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt.solve import optimize_wireframe
from simsopt.util import in_github_actions

# Set to True generate a 3d rendering with the mayavi package
make_mayavi_plots = False

# Wireframe resolution
if not in_github_actions:

    # Number of wireframe segments per half period, toroidal dimension
    wf_n_phi = 48      # 96 to match reference

    # Number of wireframe segments per half period, poloidal dimension
    wf_n_theta = 50    # 100 to match reference

    # Maximum number of GSCO iterations
    max_iter = 2000   # 20000 to match reference

    # How often print progress
    print_interval = 100

    # Resolution of test points on plasma boundary (poloidal and toroidal)
    plas_n = 32       # 32 to match reference

else:

    # For GitHub CI tests, run at very low resolution
    wf_n_phi = 18
    wf_n_theta = 8
    max_iter = 100
    print_interval = 10
    plas_n = 4

# Number of modular coils in the solution per half period
n_mod_coils_hp = 6

# Average magnetic field on axis, in Teslas, to be produced by the wireframe.
# This will be used for initializing the modular coils. The radius
# of the magnetic axis will be estimated from the plasma boundary geometry.
field_on_axis = 1.0

# Weighting factor for the sparsity objective
lambda_S = 10**-6

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename_equil = TEST_DIR / 'input.LandremanPaul2021_QA'

# File specifying the geometry of the wireframe surface (made with BNORM)
filename_wf_surf = TEST_DIR / 'nescin.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Load the geometry of the target plasma boundary
plas_n_phi = plas_n
plas_n_theta = plas_n
surf_plas = SurfaceRZFourier.from_vmec_input(filename_equil,
                                             nphi=plas_n_phi, ntheta=plas_n_theta, range='half period')

# Construct the wireframe on a toroidal surface
surf_wf = SurfaceRZFourier.from_nescoil_input(filename_wf_surf, 'current')
wf = ToroidalWireframe(surf_wf, wf_n_phi, wf_n_theta)

# Calculate the required net poloidal current
mu0 = 4.0 * np.pi * 1e-7
pol_cur = -2.0*np.pi*surf_plas.get_rc(0, 0)*field_on_axis/mu0

# Initialize the wireframe with a set of planar TF coils
coil_current = pol_cur/(2*wf.nfp*n_mod_coils_hp)
wf.add_tfcoil_currents(n_mod_coils_hp, coil_current)

# Set constraint for net poloidal current (note: the constraint is not strictly
# necessary for GSCO to work properly, but it can be used as a consistency
# check for the solution)
wf.set_poloidal_current(pol_cur)

# Generate a 3D plot of the initialized wireframe and plasma if desired
if make_mayavi_plots:

    from mayavi import mlab
    mlab.options.offscreen = True

    mlab.figure(size=(1050, 800), bgcolor=(1, 1, 1))
    wf.make_plot_3d(to_show='all')
    surf_plas_plot = SurfaceRZFourier.from_vmec_input(filename_equil,
                                                      nphi=plas_n_phi, ntheta=plas_n_theta, range='full torus')
    surf_plas_plot.plot(engine='mayavi', show=False, close=True,
                        wireframe=False, color=(1, 0.75, 1))
    mlab.view(distance=5.5, focalpoint=(0, 0, -0.15))
    mlab.savefig(OUT_DIR + 'gsco_modular_wireframe_init_plot3d.png')

# Set the optimization parameters
opt_params = {'lambda_S': lambda_S,
              'max_iter': max_iter,
              'print_interval': print_interval,
              'no_crossing': True,
              'default_current': np.abs(coil_current),
              'max_current': 1.1 * np.abs(coil_current)
              }

# Run the GSCO optimization
t0 = time.time()
res = optimize_wireframe(wf, 'gsco', opt_params, surf_plas=surf_plas,
                         verbose=False)
t1 = time.time()
deltaT = t1 - t0

print('')
print('Post-processing')
print('---------------')
print('  opt time [s]   %12.3f' % (deltaT))

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

# Print post-processing results
print('  f_B [T^2m^2]   %12.4e' % (res['f_B']))
print('  f_S            %12.4e' % (res['f_S']))
print('  <|Bn|/|B|>     %12.4e' % (mean_rel_Bn))
print('  I_max [MA]     %12.4e' % (max_cur))

# Save plots and visualization data to files
wf.make_plot_2d(coordinates='degrees', quantity='nonzero currents')
pl.savefig(OUT_DIR + 'gsco_modular_wireframe_curr2d.png')
wf.to_vtk(OUT_DIR + 'gsco_modular_wireframe')

# Generate a 3D plot of the wireframe and plasma if desired
if make_mayavi_plots:

    from mayavi import mlab
    mlab.options.offscreen = True

    mlab.figure(size=(1050, 800), bgcolor=(1, 1, 1))
    wf.make_plot_3d(to_show='active')
    surf_plas_plot = SurfaceRZFourier.from_vmec_input(filename_equil,
                                                      nphi=plas_n_phi, ntheta=plas_n_theta, range='full torus')
    surf_plas_plot.plot(engine='mayavi', show=False, close=True,
                        wireframe=False, color=(1, 0.75, 1))
    mlab.view(distance=5.5, focalpoint=(0, 0, -0.15))
    mlab.savefig(OUT_DIR + 'gsco_modular_wireframe_plot3d.png')
