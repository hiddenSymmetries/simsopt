from __future__ import annotations #ToDo: Remove this

#!/usr/bin/env python
r"""
This example script uses the GPMO
greedy algorithm for solving permanent 
magnet optimization on the MUSE grid. This 
algorithm is described in the following paper:
    A. A. Kaptanoglu, R. Conlin, and M. Landreman, 
    Greedy permanent magnet optimization, 
    Nuclear Fusion 63, 036016 (2023)

The script should be run as:
    mpirun -n 1 python permanent_magnet_MUSE.py
on a cluster machine but 
    python permanent_magnet_MUSE.py
is sufficient on other machines. Note that this code does not use MPI, but is 
parallelized via OpenMP and XSIMD, so will run substantially
faster on multi-core machines (make sure that all the cores
are available to OpenMP, e.g. through setting OMP_NUM_THREADS).

For high-resolution and more realistic designs, please see the script files at
https://github.com/akaptano/simsopt_permanent_magnet_advanced_scripts.git
"""

import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve import GPMO
from simsopt.util import FocusData, discretize_polarizations, polarization_axes, in_github_actions
from simsopt.util.permanent_magnet_helper_functions import *

t_start = time.time()

# Set some parameters -- if doing CI, lower the resolution
if in_github_actions:
    nphi = 2
    nIter_max = 100
    nBacktracking = 50
    max_nMagnets = 20
    downsample = 100  # downsample the FAMUS grid of magnets by this factor
else:
    nphi = 16  # >= 64 for high-resolution runs
    nIter_max = 10000
    nBacktracking = 200
    max_nMagnets = 1000
    downsample = 8

ntheta = nphi  # same as above
dr = 0.01  # Radial extent in meters of the cylindrical permanent magnet bricks
input_name = 'input.muse'

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
famus_filename = TEST_DIR / 'zot80.focus'
surface_filename = TEST_DIR / input_name
s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_inner = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_outer = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

# Make the output directory -- warning, saved data can get big!
# On NERSC, recommended to change this directory to point to SCRATCH!
out_dir = Path("output_permanent_magnet_GPMO_MUSE")
out_dir.mkdir(parents=True, exist_ok=True)

# initialize the coils
base_curves, curves, coils = initialize_coils('muse_famus', TEST_DIR, s, out_dir)

# Set up BiotSavart fields
bs = BiotSavart(coils)

# Calculate average, approximate on-axis B field strength
calculate_modB_on_major_radius(bs, s)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_focus(
    surface_filename,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_initial")

# Set up correct Bnormal from TF coils
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Load a downsampled version of the magnet grid from a FAMUS file
mag_data = FocusData(famus_filename, downsample=downsample)

# Set the allowable orientations of the magnets to be face-aligned
# i.e. the local x, y, or z directions
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)
pol_axes_f, pol_type_f = polarization_axes(['face'])
ntype_f = int(len(pol_type_f)/2)
pol_axes_f = pol_axes_f[:ntype_f, :]
pol_type_f = pol_type_f[:ntype_f]
pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
pol_type = np.concatenate((pol_type, pol_type_f))

# Optionally add additional types of allowed orientations
PM4Stell_orientations = False
if PM4Stell_orientations:
    pol_axes_fe_ftri, pol_type_fe_ftri = polarization_axes(['fe_ftri'])
    ntype_fe_ftri = int(len(pol_type_fe_ftri)/2)
    pol_axes_fe_ftri = pol_axes_fe_ftri[:ntype_fe_ftri, :]
    pol_type_fe_ftri = pol_type_fe_ftri[:ntype_fe_ftri] + 1
    pol_axes = np.concatenate((pol_axes, pol_axes_fe_ftri), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_fe_ftri))
    pol_axes_fc_ftri, pol_type_fc_ftri = polarization_axes(['fc_ftri'])
    ntype_fc_ftri = int(len(pol_type_fc_ftri)/2)
    pol_axes_fc_ftri = pol_axes_fc_ftri[:ntype_fc_ftri, :]
    pol_type_fc_ftri = pol_type_fc_ftri[:ntype_fc_ftri] + 2
    pol_axes = np.concatenate((pol_axes, pol_axes_fc_ftri), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_fc_ftri))

ophi = np.arctan2(mag_data.oy, mag_data.ox)
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
pol_vectors = np.zeros((mag_data.nMagnets, len(pol_type), 3))
pol_vectors[:, :, 0] = mag_data.pol_x
pol_vectors[:, :, 1] = mag_data.pol_y
pol_vectors[:, :, 2] = mag_data.pol_z
print('pol_vectors_shape = ', pol_vectors.shape)

# pol_vectors is only used for the greedy algorithms with cartesian coordinate_flag
# which is the default, so no need to specify it here.
kwargs = {"pol_vectors": pol_vectors, "downsample": downsample, "dr": dr}

# Finally, initialize the permanent magnet class
pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs)

print('Number of available dipoles = ', pm_opt.ndipoles)

# Set some hyperparameters for the optimization
algorithm = 'ArbVec_backtracking_macromag_py'  # Algorithm to use
nAdjacent = 1  # How many magnets to consider "adjacent" to one another
nHistory = 20  # How often to save the algorithm progress
thresh_angle = np.pi  # The angle between two "adjacent" dipoles such that they should be removed
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = nIter_max  # Maximum number of GPMO iterations to run
kwargs['nhistory'] = nHistory
if algorithm == 'backtracking' or algorithm == 'ArbVec_backtracking_macromag_py':
    kwargs['backtracking'] = nBacktracking  # How often to perform the backtrackinig
    kwargs['Nadjacent'] = nAdjacent
    kwargs['dipole_grid_xyz'] = np.ascontiguousarray(pm_opt.dipole_grid_xyz)
    kwargs['max_nMagnets'] = max_nMagnets
    if algorithm == 'ArbVec_backtracking_macromag_py':
        kwargs['thresh_angle'] = thresh_angle

# Optimize the permanent magnets greedily
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs)
t2 = time.time()
print('GPMO took t = ', t2 - t1, ' s')

# plot the MSE history
iterations = np.linspace(0, kwargs['max_nMagnets'], len(R2_history), endpoint=False)
plt.figure()
plt.semilogy(iterations, R2_history, label=r'$f_B$')
plt.semilogy(iterations, Bn_history, label=r'$<|Bn|>$')
plt.grid(True)
plt.xlabel('K')
plt.ylabel('Metric values')
plt.legend()
plt.savefig(out_dir / 'GPMO_MSE_history.png')

# Set final m to the minimum achieved during the optimization
min_ind = np.argmin(R2_history)
pm_opt.m = np.ravel(m_history[:, :, min_ind])

# Print effective permanent magnet volume
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
M_max = B_max / mu0
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

save_plots = False
if save_plots:
    # Save the MSE history and history of the m vectors
    np.savetxt(
        out_dir / f"mhistory_K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}.txt",
        m_history.reshape(pm_opt.ndipoles * 3, kwargs['nhistory'] + 1)
    )
    np.savetxt(
        out_dir / f"R2history_K{kwargs['K']}_nphi{nphi}_ntheta{ntheta}.txt",
        R2_history
    )
    # Plot the SIMSOPT GPMO solution
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    make_Bnormal_plots(bs, s_plot, out_dir, "biot_savart_optimized")

    # Look through the solutions as function of K and make plots
    for k in range(0, kwargs["nhistory"] + 1, 50):
        mk = m_history[:, :, k].reshape(pm_opt.ndipoles * 3)
        b_dipole = DipoleField(
            pm_opt.dipole_grid_xyz,
            mk,
            nfp=s.nfp,
            coordinate_flag=pm_opt.coordinate_flag,
            m_maxima=pm_opt.m_maxima,
        )
        b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
        K_save = int(kwargs['K'] / kwargs['nhistory'] * k)
        b_dipole._toVTK(out_dir / f"Dipole_Fields_K{K_save}_nphi{nphi}_ntheta{ntheta}")
        print("Total fB = ", 0.5 * np.sum((pm_opt.A_obj @ mk - pm_opt.b_obj) ** 2))
        Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
        Bnormal_total = Bnormal + Bnormal_dipoles

        # For plotting Bn on the full torus surface at the end with just the dipole fields
        make_Bnormal_plots(b_dipole, s_plot, out_dir, "only_m_optimized_K{K_save}_nphi{nphi}_ntheta{ntheta}")
        pointData = {"B_N": Bnormal_total[:, :, None]}
        s_plot.to_vtk(out_dir / "m_optimized_K{K_save}_nphi{nphi}_ntheta{ntheta}", extra_data=pointData)

    # write solution to FAMUS-type file
    pm_opt.write_to_famus(out_dir)

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# Print optimized f_B and other metrics
### Note this will only agree with the optimization in the high-resolution
### limit where nphi ~ ntheta >= 64!
b_dipole = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m,
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima,
)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B = ', f_B_sf)
total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume = ', total_volume)

# Optionally make a QFM and pass it to VMEC
# This is worthless unless plasma
# surface is at least 64 x 64 resolution.
vmec_flag = False
if vmec_flag:
    from simsopt.mhd.vmec import Vmec
    from simsopt.util.mpi import MpiPartition
    mpi = MpiPartition(ngroups=1)

    # Make the QFM surfaces
    t1 = time.time()
    Bfield = bs + b_dipole
    Bfield.set_points(s_plot.gamma().reshape((-1, 3)))
    qfm_surf = make_qfm(s_plot, Bfield)
    qfm_surf = qfm_surf.surface
    t2 = time.time()
    print("Making the QFM surface took ", t2 - t1, " s")

    # Run VMEC with new QFM surface
    t1 = time.time()

    ### Always use the QA VMEC file and just change the boundary
    vmec_input = "../../tests/test_files/input.LandremanPaul2021_QA"
    equil = Vmec(vmec_input, mpi)
    equil.boundary = qfm_surf
    equil.run()

t_end = time.time()
print('Total time = ', t_end - t_start)
# plt.show()


### Armin ouput inspection (temporary) additions
def write_dipoles_vtu(
    fname: Path,
    points: np.ndarray,
    moments: np.ndarray,
    drop_zeros: bool = False,     
    nfp: int | None = None,
    replicate_full_torus: bool = False  # optional symmetry replication
) -> None:
    import math
    pts = np.asarray(points, dtype=float)
    m   = np.asarray(moments, dtype=float)
    if m.ndim != 2 or m.shape != pts.shape or m.shape[1] != 3:
        raise ValueError("points and moments must both be (N,3)")

    # Optional: replicate around torus (simple rotation about z by equal angles)
    if nfp is not None and replicate_full_torus:
        copies = 2 * nfp              # half-period → full torus replication
        all_p, all_m = [], []
        for k in range(copies):
            ang = k * (math.pi / nfp) # step = π/nfp
            ca, sa = math.cos(ang), math.sin(ang)
            Rz = np.array([[ca, -sa, 0.0],
                           [sa,  ca, 0.0],
                           [0.0, 0.0, 1.0]])
            all_p.append(pts @ Rz.T)
            all_m.append(m   @ Rz.T)
        pts = np.vstack(all_p)
        m   = np.vstack(all_m)

    # Mask & optional zero-drop
    m_mag = np.linalg.norm(m, axis=1)
    mask  = (m_mag > 0.0).astype(np.uint8)
    if drop_zeros:
        keep = mask.astype(bool)
        pts, m, m_mag, mask = pts[keep], m[keep], m_mag[keep], mask[keep]

    N = len(pts)
    if N == 0:
        print(f"[write_dipoles_vtu] no dipoles to write → {fname}")
        return

    # Direction (unit vector); safe for zeros
    m_dir = np.zeros_like(m)
    nz = m_mag > 0
    m_dir[nz] = m[nz] / m_mag[nz][:, None]

    # VTK arrays
    connectivity = " ".join(str(i) for i in range(N))
    offsets      = " ".join(str(i+1) for i in range(N))
    types        = " ".join("1" for _ in range(N))   # VTK_VERTEX

    fname = Path(fname); fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{N}" NumberOfCells="{N}">\n')

        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for x, y, z in pts:
            f.write(f'          {x:.15e} {y:.15e} {z:.15e}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        # Cells
        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write('          ' + connectivity + '\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write('          ' + offsets + '\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write('          ' + types + '\n')
        f.write('        </DataArray>\n')
        f.write('      </Cells>\n')

        # PointData
        f.write('      <PointData Scalars="m_mag" Vectors="m_dir">\n')

        # magnitude for coloring
        f.write('        <DataArray type="Float64" Name="m_mag" format="ascii">\n')
        for v in m_mag:
            f.write(f'          {v:.15e}\n')
        f.write('        </DataArray>\n')

        # unit direction for glyph orientation
        f.write('        <DataArray type="Float64" Name="m_dir" NumberOfComponents="3" format="ascii">\n')
        for vx, vy, vz in m_dir:
            f.write(f'          {vx:.15e} {vy:.15e} {vz:.15e}\n')
        f.write('        </DataArray>\n')

        # full moment vector (A·m^2) if you want it
        f.write('        <DataArray type="Float64" Name="m_full" NumberOfComponents="3" format="ascii">\n')
        for vx, vy, vz in m:
            f.write(f'          {vx:.15e} {vy:.15e} {vz:.15e}\n')
        f.write('        </DataArray>\n')

        # mask so you can threshold to active magnets
        f.write('        <DataArray type="UInt8" Name="mask_nonzero" format="ascii">\n')
        for q in mask:
            f.write(f'          {int(q)}\n')
        f.write('        </DataArray>\n')

        f.write('      </PointData>\n')
        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')
    print(f"[write_dipoles_vtu] wrote {fname}")

    
xyz     = pm_opt.dipole_grid_xyz
m_final = pm_opt.m.reshape(pm_opt.ndipoles, 3)

# all sites (zeros included), plus full-torus copies:
write_dipoles_vtu(
    out_dir / "dipoles_final.vtu",
    xyz, m_final,
    drop_zeros=False,
    nfp=s.nfp,
    replicate_full_torus=True
)

