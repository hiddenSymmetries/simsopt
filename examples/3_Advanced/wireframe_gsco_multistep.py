#!/usr/bin/env python
r"""
In this example, the segment currents in a toroidal wireframe are optimized
using the Greedy Stellarator Coil Optimization (GSCO) procedure to produce
a design for saddle coils that are confined to toroidal sectors (wedges). 
The optimization is carried out in multiple steps to enable a solution in
which different saddle coils may carry different currents.

For this solution, the toroidal field is provided by a fixed external set
of TF coils.
"""

import os
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as pl
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe, \
    create_equally_spaced_curves
from simsopt.solve import optimize_wireframe
from simsopt.field import WireframeField, BiotSavart, Current, \
    coils_via_symmetries
from simsopt.util import in_github_actions

# Set to True generate a 3d rendering with the mayavi package
make_mayavi_plots = False

# Wireframe resolution
if not in_github_actions:

    # Number of wireframe segments per half period, toroidal dimension
    wf_n_phi = 96

    # Number of wireframe segments per half period, poloidal dimension
    wf_n_theta = 100

    # Maximum number of GSCO iterations
    max_iter = 2500

    # How often to print progress
    print_interval = 100

    # Resolution of test points on plasma boundary (poloidal and toroidal)
    plas_n = 32

else:

    # For GitHub CI tests, run at very low resolution
    wf_n_phi = 24
    wf_n_theta = 8
    max_iter = 100
    print_interval = 10
    plas_n = 4

# Number of planar TF coils in the solution per half period
n_tf_coils_hp = 3

# Toroidal width, in cells, of the restricted regions (breaks) between sectors
break_width = 4

# GSCO loop current as a fraction of net TF coil current
init_gsco_cur_frac = 0.2

# Minimum size (in enclosed wireframe cells) for a saddle coil
min_coil_size = 20

# Average magnetic field on axis, in Teslas, to be produced by the wireframe.
# This will be used for initializing the TF coils. The radius of the
# magnetic axis will be estimated from the plasma boundary geometry.
field_on_axis = 1.0

# Weighting factor for the sparsity objective
lambda_S = 10**-7

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename_equil = TEST_DIR / 'input.LandremanPaul2021_QA'

# File specifying the geometry of the wireframe surface (made with BNORM)
filename_wf_surf = TEST_DIR / 'nescin.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

############################################################
# End of input parameters.
############################################################

############################################################
# Helper functions for the optimization
############################################################


def constrain_enclosed_segments(wframe, loop_count):
    """
    Applies constraints to any segment that is enclosed within a saddle coil.

    Parameters
    ----------
        wframe: ToroidalWireframe class instance
            The wireframe whose segments are to be constrained
        loop_count: integer array
            Array giving the net number of loops of current added to each
            cell in the wireframe (this is output by the GSCO function)

    Returns
    -------
        enclosed_segments: integer array
            Indices of the segments that were constrained
    """

    encl_loops = loop_count != 0
    encl_seg_inds = np.unique(wframe.cell_key[encl_loops, :].reshape((-1)))
    encl_segs = np.full(wframe.n_segments, False)
    encl_segs[encl_seg_inds] = True
    encl_segs[wframe.currents != 0] = False
    wframe.set_segments_constrained(np.where(encl_segs)[0])
    return np.where(encl_segs)[0]


def contiguous_coil_size(ind, count_in, coil_id, coil_ids, loop_count,
                         neighbors):
    """
    Recursively counts how many cells are enclosed by a saddle coil.

    Parameters
    ----------
        ind: integer
            Index of the first cell to be counted
        count_in: integer
            Running total of cells counted (typically zero; nonzero for
            recursive calls)
        coil_id: integer
            ID number to assign to the coil whose cells are being counted
        coil_ids: integer array
            Array to keep track of which coil ID each cell belongs to
        loop_count: integer array
            Array giving the net number of loops of current added to each
            cell in the wireframe (this is output by the GSCO function)
        neighbors: integer array
            Array giving the indices of the neighboring cells to each cell
            in the wireframe, provided by the `get_cell_neighbors` method

    Returns
    -------
        coil_size: integer
            Number of cells contained within the coil (running total for
            recursive calls)
    """

    # Return without incrementing the count if this cell is already counted
    if loop_count[ind] == 0 or coil_ids[ind] >= 0:
        return count_in

    # Label the current cell with the coil id
    coil_ids[ind] = coil_id

    # Recursively cycle through
    count_out = count_in
    for neighbor_id in neighbors[ind, :]:
        count_out = contiguous_coil_size(neighbor_id, count_out, coil_id,
                                         coil_ids, loop_count, neighbors)

    # Increment the count for the current cell
    return count_out + 1


def find_coil_sizes(loop_count, neighbors):
    """
    Determines the sizes of the saddle coils in a wireframe GSCO solution

    Parameters
    ----------
        loop_count: integer array
            Array giving the net number of loops of current added to each
            cell in the wireframe (this is output by the GSCO function)
        neighbors: integer array
            Array giving the indices of the neighboring cells to each cell
            in the wireframe, provided by the `get_cell_neighbors` method

    Returns
    -------
        coil_sizes: integer array
            For each wireframe cell, provides the size of the coil to which
            that cell belongs (zero if the cell is not part of a coil)
    """

    unique_coil_ids = []
    coil_ids = np.full(len(loop_count), -1)
    coil_sizes = np.zeros(len(loop_count))
    coil_id = -1

    for i in range(len(loop_count)):
        if loop_count[i] != 0:
            coil_id += 1
            unique_coil_ids.append(coil_id)
            count = contiguous_coil_size(i, 0, coil_id, coil_ids,
                                         loop_count, neighbors)
            coil_sizes[coil_ids == coil_id] = count

    return coil_sizes

############################################################
# Setting up and running the optimization
############################################################


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

# Initialize the wireframe
tfcoil_current = pol_cur/(2*wf.nfp*n_tf_coils_hp)

# Constrain toroidal segments around the TF coils to prevent new coils from
# being placed there (and to prevent the TF coils from being reshaped)
wf.set_toroidal_breaks(n_tf_coils_hp, break_width, allow_pol_current=True)

# Set constraint for net poloidal current (note: the constraint is not strictly
# necessary for GSCO to work properly, but it can be used as a consistency
# check for the solution)
wf.set_poloidal_current(0)

# Create an external set of TF coils
tf_curves = create_equally_spaced_curves(n_tf_coils_hp, surf_plas.nfp, True,
                                         R0=1.0, R1=0.85)
tf_curr = [Current(-pol_cur/(2*n_tf_coils_hp*surf_plas.nfp))
           for i in range(n_tf_coils_hp)]
tf_coils = coils_via_symmetries(tf_curves, tf_curr, surf_plas.nfp, True)
mf_tf = BiotSavart(tf_coils)

# Initialize loop variables
soln_prev = np.full(wf.currents.shape, np.nan)
soln_current = np.array(wf.currents)
cur_frac = init_gsco_cur_frac
loop_count = None
final_step = False
encl_segs = []
n_step = 0

# Multi-step optimization loop
while not final_step:

    n_step += 1

    if not final_step and np.all(soln_prev == soln_current):
        final_step = True
        wf.set_segments_free(encl_segs)

    step_name = 'step %d' % (n_step) if not final_step else 'final adjustment'
    print('------------------------------------------------------------------')
    print('Performing GSCO for ' + step_name)
    print('------------------------------------------------------------------')

    # Set the optimization parameters
    if not final_step:
        opt_params = {'lambda_S': lambda_S,
                      'max_iter': max_iter,
                      'print_interval': print_interval,
                      'no_crossing': True,
                      'max_loop_count': 1,
                      'loop_count_init': loop_count,
                      'default_current': np.abs(cur_frac*pol_cur),
                      'max_current': 1.1 * np.abs(cur_frac*pol_cur)
                      }
    else:
        opt_params = {'lambda_S': lambda_S,
                      'max_iter': max_iter,
                      'print_interval': print_interval,
                      'no_crossing': True,
                      'max_loop_count': 1,
                      'loop_count_init': loop_count,
                      'match_current': True,
                      'no_new_coils': True,
                      'default_current': 0,
                      'max_current': 1.1 * np.abs(init_gsco_cur_frac*pol_cur)
                      }

    # Run the GSCO optimization
    t0 = time.time()
    res = optimize_wireframe(wf, 'gsco', opt_params, surf_plas=surf_plas,
                             ext_field=mf_tf, verbose=False)
    t1 = time.time()
    deltaT = t1 - t0

    print('')
    print('  Post-processing for ' + step_name)
    print('  ------------------------------------')
    print('    opt time [s]   %12.3f' % (deltaT))

    if not final_step:

        # "Sweep" the solution to remove coils that are too small
        coil_sizes = find_coil_sizes(res['loop_count'], wf.get_cell_neighbors())
        small_inds = np.where(
            np.logical_and(coil_sizes > 0, coil_sizes < min_coil_size))[0]
        adjoining_segs = wf.get_cell_key()[small_inds, :]
        segs_to_zero = np.unique(adjoining_segs.reshape((-1)))

        # Modify the solution by removing the small coils
        loop_count = res['loop_count']
        wf.currents[segs_to_zero] = 0
        loop_count[small_inds] = 0

        # Prevent coils from being placed inside existing coils in subsequent
        # steps
        encl_segs = constrain_enclosed_segments(wf, loop_count)

    # Verify that the solution satisfies all constraints
    assert wf.check_constraints()

    # Re-calculate field after coil removal
    mf_post = WireframeField(wf) + mf_tf
    mf_post.set_points(surf_plas.gamma().reshape((-1, 3)))

    # Post-processing
    x_post = np.array(wf.currents).reshape((-1, 1))
    f_B_post = 0.5 * np.sum((res['Amat'] @ x_post - res['bvec'])**2)
    f_S_post = 0.5 * np.linalg.norm(x_post.ravel(), ord=0)
    Bfield = mf_post.B().reshape((plas_n_phi, plas_n_theta, 3))
    Bnormal = np.sum(Bfield * surf_plas.unitnormal(), axis=2)
    modB = np.sqrt(np.sum(Bfield**2, axis=2))
    rel_Bnorm = Bnormal/modB
    area = np.sqrt(np.sum(surf_plas.normal()**2, axis=2))/float(modB.size)
    mean_rel_Bn = np.sum(np.abs(rel_Bnorm)*area)/np.sum(area)
    max_cur = np.max(np.abs(res['x']))

    # Print post-processing results
    print('    f_B [T^2m^2]   %12.4e' % (f_B_post))
    print('    f_S            %12.4e' % (f_S_post))
    print('    <|Bn|/|B|>     %12.4e' % (mean_rel_Bn))
    print('    I_max [MA]     %12.4e' % (max_cur))
    print('')

    cur_frac *= 0.5

    soln_prev = soln_current
    soln_current = np.array(wf.currents)


# Save plots and visualization data to files
wf.make_plot_2d(coordinates='degrees', quantity='nonzero currents')
pl.savefig(OUT_DIR + 'gsco_multistep_curr2d.png')
pl.close(pl.gcf())
wf.to_vtk(OUT_DIR + 'gsco_multistep')

# Generate a 3D plot of the wireframe and plasma if desired
if make_mayavi_plots:

    from mayavi import mlab
    mlab.options.offscreen = True

    mlab.figure(size=(1050, 800), bgcolor=(1, 1, 1))
    wf.make_plot_3d(to_show='active')
    for tfc in tf_coils:
        tfc.curve.plot(engine='mayavi', show=False, color=(0.75, 0.75, 0.75),
                       close=True)
    surf_plas_plot = SurfaceRZFourier.from_vmec_input(filename_equil,
                                                      nphi=2*surf_plas.nfp*plas_n_phi, ntheta=plas_n_theta,
                                                      range='full torus')
    surf_plas_plot.plot(engine='mayavi', show=False, close=True,
                        wireframe=False, color=(1, 0.75, 1))
    mlab.view(distance=6, focalpoint=(0, 0, -0.15))
    mlab.savefig(OUT_DIR + 'gsco_multistep_plot3d.png')
