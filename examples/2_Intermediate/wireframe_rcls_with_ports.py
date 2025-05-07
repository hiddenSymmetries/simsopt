#!/usr/bin/env python
r"""
In this example, the segment currents of a toroidal wireframe are optimized
in such a way as to leave room for a set of user-specified ports that may
be necessary for diagnostic or heating systems.

For this example, the ports are placed on a grid of toroidal (phi) and 
poloidal (theta) angles within each half-period, and aligned to be locally
perpendicular to the toroidal surface used to specify the wireframe 
geometry. However, in general, ports may be specified with arbitrary
locations and orientations. Furthermore, the RCLS technique is used as the 
optimizer; however, any wireframe optimization method is compatible with 
ports.

Output files include a 2d plot of the current distribution and VTK files
with the wireframe data ('rcls_ports_wireframe') and the port geometry
('rcls_ports_port_geometry').
"""

import os
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as pl
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt.geo import PortSet, CircularPort
from simsopt.solve import optimize_wireframe

# Set to True generate a 3d rendering with the mayavi package
make_mayavi_plot = False

# Number of wireframe segments per half period in the toroidal dimension
wf_n_phi = 12

# Number of wireframe segments in the poloidal dimension
wf_n_theta = 22

# Distance between the plasma boundary and the wireframe
wf_surf_dist = 0.3

# Angular positions in each half-period where ports should be placed
port_phis = [np.pi/8, 3*np.pi/8]  # toroidal angles
port_thetas = [np.pi/4, 7*np.pi/4]  # poloidal angles

# Dimensions of each port
port_ir = 0.1       # inner radius [m]
port_thick = 0.005  # wall thickness [m]
port_gap = 0.04     # minimum gap between port and wireframe segments [m]
port_l0 = -0.15     # distance from origin to end, negative axis direction [m]
port_l1 = 0.15      # distance from origin to end, positive axis direction [m]

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

# Construct the port geometry
ports = PortSet()
for i in range(len(port_phis)):
    # For simplicity, adjust the angles to the positions of the nearest existing
    # quadrature points in the surf_wf class instance
    phi_nearest = np.argmin(np.abs((0.5/np.pi)*port_phis[i]
                                   - surf_wf.quadpoints_phi))
    for j in range(len(port_thetas)):
        theta_nearest = np.argmin(np.abs((0.5/np.pi)*port_thetas[j]
                                         - surf_wf.quadpoints_theta))
        ox = surf_wf.gamma()[phi_nearest, theta_nearest, 0]
        oy = surf_wf.gamma()[phi_nearest, theta_nearest, 1]
        oz = surf_wf.gamma()[phi_nearest, theta_nearest, 2]
        ax = surf_wf.normal()[phi_nearest, theta_nearest, 0]
        ay = surf_wf.normal()[phi_nearest, theta_nearest, 1]
        az = surf_wf.normal()[phi_nearest, theta_nearest, 2]
        ports.add_ports([CircularPort(ox=ox, oy=oy, oz=oz, ax=ax, ay=ay, az=az,
                                      ir=port_ir, thick=port_thick, l0=port_l0, l1=port_l1)])
ports = ports.repeat_via_symmetries(surf_wf.nfp, True)

# Constrain wireframe segments that collide with the ports
wf.constrain_colliding_segments(ports.collides, gap=port_gap)

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
wf.make_plot_2d(quantity='nonzero currents', coordinates='degrees')
pl.savefig(OUT_DIR + 'rcls_ports_wireframe_curr2d.png')
wf.to_vtk(OUT_DIR + 'rcls_ports_wireframe')
ports.to_vtk(OUT_DIR + 'rcls_ports_port_geometry')

# Generate a 3D plot if desired
if make_mayavi_plot:

    from mayavi import mlab
    mlab.options.offscreen = True

    mlab.figure(size=(1050, 800), bgcolor=(1, 1, 1))
    wf.make_plot_3d(to_show='active')
    ports.plot()
    surf_plas_plot = SurfaceRZFourier.from_vmec_input(filename_equil,
                                                      nphi=plas_n_phi, ntheta=plas_n_theta, range='full torus')
    surf_plas_plot.plot(engine='mayavi', show=False, close=True,
                        wireframe=False, color=(1, 0.75, 1))
    mlab.view(distance=5.5, focalpoint=(0, 0, -0.15))
    mlab.savefig(OUT_DIR + 'rcls_ports_wireframe_plot3d.png')
