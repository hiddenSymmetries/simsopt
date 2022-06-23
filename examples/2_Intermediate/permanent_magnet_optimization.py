#!/usr/bin/env python
r"""
This example script allows the user to explore building
permanent magnet configurations for several stage-1 optimized
plasma boundaries, including the Landreman/Paul QA/QH designs,
the MUSE stellarator, the NCSX stellarator, and variations.

The script should be run as:
    python permanent_magnet_optimization.py True
for an interactive command prompt that asks user for parameters. 

If the script is being run on slurm, the script should be run as
    python permanent_magnet_optimization.py False my_config my_resolution ...
where the command line parameters must be specified as is detailed
in permanent_magnet_helpers.py.

Note that if the run_type flag is set to post-processing, the script will generate
Poincare plots, QFMS, and VMEC files, so the script should then be run as:
    srun python permanent_magnet_optimization.py True
or the equivalent command for a slurm script. This is also true for initializing
the Landreman/Paul QA/QH surfaces, since these designs need some mpi
functionality to get them properly scaled to a particular on-axis magnetic field.

"""

import os
import pickle
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.field.magneticfieldclasses import DipoleField, ToroidalField
from simsopt.field.biotsavart import BiotSavart 
from simsopt.util.permanent_magnet_optimizer import PermanentMagnetOptimizer
from simsopt._core.optimizable import Optimizable
from permanent_magnet_helpers import *
import time

t_start = time.time()

# Read in all the required parameters
comm = None
config_flag, res_flag, run_type, reg_l2, epsilon, max_iter_MwPGP, min_fb, reg_l0, nu, max_iter_RS, dr, coff, poff, surface_flag, input_name, nphi, ntheta, pms_name, is_premade_famus_grid, cylindrical_flag = read_input()

# Add cori scratch path 
class_filename = "PM_optimizer_" + config_flag
scratch_path = '/global/cscratch1/sd/akaptano/'

# Read in the correct plasma equilibrium file
t1 = time.time()
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surface_filename = TEST_DIR / input_name
if surface_flag == 'focus':
    s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
elif surface_flag == 'wout':
    s = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
else:
    s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
t2 = time.time()
print("Done loading in plasma boundary surface, t = ", t2 - t1)

if run_type == 'initialization':
    # Make the output directory
    OUT_DIR = scratch_path + config_flag + "_nphi{0:d}_ntheta{1:d}_dr{2:.2e}_coff{3:.2e}_poff{4:.2e}/".format(nphi, ntheta, dr, coff, poff)
    print("Output directory = ", OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    t1 = time.time()

    # Don't have NCSX TF coils, just the Bn field on the surface 
    # so have to treat the NCSX example separately.
    quadpoints_phi = np.linspace(0, 1, 2 * nphi, endpoint=True)
    qphi = len(quadpoints_phi)
    quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
    if config_flag != 'ncsx':
        t1 = time.time()

        # initialize the coils
        base_curves, curves, coils = initialize_coils(config_flag, TEST_DIR, OUT_DIR, s)

        # Set up BiotSavart fields
        bs = BiotSavart(coils)

        # Calculate average, approximate on-axis B field strength
        calculate_on_axis_B(bs, s)

        t2 = time.time()
        print("Done setting up biot savart, ", t2 - t1, " s")

        # Make higher resolution surface for plotting Bnormal
        if surface_flag == 'focus': 
            s_plot = SurfaceRZFourier.from_focus(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
        elif surface_flag == 'wout': 
            s_plot = SurfaceRZFourier.from_wout(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
        else:
            s_plot = SurfaceRZFourier.from_vmec_input(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)

        # Plot initial Bnormal on plasma surface from un-optimized BiotSavart coils
        make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_initial")

        # If BiotSavart not yet optimized, optimize it
        if 'muse' not in config_flag: 
            s, bs = coil_optimization(s, bs, base_curves, curves, OUT_DIR, s_plot, config_flag)

            # check after-optimization average on-axis magnetic field strength
            calculate_on_axis_B(bs, s)

        # Save optimized BiotSavart object
        bs.set_points(s.gamma().reshape((-1, 3)))
        Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
        biotsavart_json_str = bs.save(filename=OUT_DIR + 'BiotSavart.json')

        # Plot Bnormal on plasma surface from optimized BiotSavart coils
        bs.set_points(s_plot.gamma().reshape((-1, 3)))
        Bnormal_plot = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
        f_B_sf = SquaredFlux(s_plot, bs).J()
        print('BiotSavart f_B = ', f_B_sf)
        make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")
    else:
        # Set up the contribution to Bnormal from a purely toroidal field.
        # Ampere's law for a purely toroidal field: 2 pi R B0 = mu0 I
        net_poloidal_current_Amperes = 3.7713e+6
        mu0 = 4 * np.pi * (1e-7)
        RB = mu0 * net_poloidal_current_Amperes / (2 * np.pi)
        print('B0 of toroidal field = ', RB)
        bs = ToroidalField(R0=1, B0=RB)

        # Check average on-axis magnetic field strength
        calculate_on_axis_B(bs, s)

        # Calculate Bnormal
        bs.set_points(s.gamma().reshape((-1, 3)))
        Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

        # Plot Bnormal on plasma surface from optimized BiotSavart coils
        s_plot = SurfaceRZFourier.from_wout(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
        bs.set_points(s_plot.gamma().reshape((-1, 3)))
        Bnormal_plot = np.sum(bs.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)
        make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")

    # Finally, initialize the permanent magnet class
    t1 = time.time()
    pm_opt = PermanentMagnetOptimizer(
        s, is_premade_famus_grid=is_premade_famus_grid, coil_offset=coff, 
        dr=dr, plasma_offset=poff, Bn=Bnormal, 
        filename=surface_filename, surface_flag=surface_flag, out_dir=OUT_DIR,
        cylindrical_flag=cylindrical_flag, pms_name=pms_name,
    )
    t2 = time.time()
    print('Done initializing the permanent magnet object')
    print('Process took t = ', t2 - t1, ' s')

    # If using a pre-made FAMUS grid of permanent magnet
    # locations, save the FAMUS grid and FAMUS solution.
    if is_premade_famus_grid:
        t1 = time.time()
        read_FAMUS_grid(pms_name, pm_opt, s, s_plot, Bnormal, Bnormal_plot, OUT_DIR)
        t2 = time.time()
        print('Saving FAMUS solution took ', t2 - t1, ' s')

        # Plot the REGCOIL_PM solution for the NCSX example
        if config_flag == 'ncsx':
            read_regcoil_pm('../../tests/test_files/regcoil_pm_ncsx.nc', surface_filename, OUT_DIR)

    # check nothing got mistranslated by plotting the initial dipole guess
    t1 = time.time()
    pm_opt.m = pm_opt.m0
    pm_opt.m_proxy = pm_opt.m0
    b_dipole_initial = DipoleField(pm_opt)
    b_dipole_initial.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole_initial._toVTK(OUT_DIR + "Dipole_Fields_initial")
    t2 = time.time()
    print('Done setting up the Dipole Field class')
    print('Process took t = ', t2 - t1, ' s')

    t1 = time.time()
    # Save PM class object to file for optimization
    file_out = open(OUT_DIR + class_filename + ".pickle", "wb")

    # SurfaceRZFourier objects not pickle-able, so set to None
    # Presumably can fix this with Bharats new json functionality
    pm_opt.plasma_boundary = None
    pm_opt.rz_inner_surface = None
    pm_opt.rz_outer_surface = None
    pickle.dump(pm_opt, file_out)
    t2 = time.time()
    print('Pickling took ', t2 - t1, ' s')
    t_end = time.time()
    print('In total, script took ', t_end - t_start, ' s')

# Do optimization on pre-made grid of dipoles
elif run_type == 'optimization':
    IN_DIR = scratch_path + config_flag + "_nphi{0:d}_ntheta{1:d}_dr{2:.2e}_coff{3:.2e}_poff{4:.2e}/".format(nphi, ntheta, dr, coff, poff)

    # Make a subdirectory for the optimization output
    OUT_DIR = IN_DIR + "output_regl2{5:.2e}_regl0{6:.2e}_nu{7:.2e}/".format(nphi, ntheta, dr, coff, poff, reg_l2, reg_l0, nu)
    os.makedirs(OUT_DIR, exist_ok=True)

    pickle_name = IN_DIR + class_filename + ".pickle"
    pm_opt = pickle.load(open(pickle_name, "rb", -1))
    pm_opt.out_dir = OUT_DIR 
    print("Cylindrical coordinates are being used = ", pm_opt.cylindrical_flag)

    # Check that you loaded the correct file with the same parameters
    assert (dr == pm_opt.dr)
    assert (nphi == pm_opt.nphi)
    assert (ntheta == pm_opt.ntheta)
    assert (coff == pm_opt.coil_offset)
    assert (poff == pm_opt.plasma_offset)
    assert (surface_flag == pm_opt.surface_flag)
    assert (surface_filename == pm_opt.filename)

    # Read in the Bnormal or BiotSavart fields from any coils 
    if config_flag != 'ncsx':
        bs = Optimizable.from_file(IN_DIR + 'BiotSavart.json')
        bs.set_points(s.gamma().reshape((-1, 3)))
        Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
    else:
        Bnormal = pm_opt.Bn

    # Set the pm_opt plasma boundary 
    pm_opt.plasma_boundary = s
    print('Done initializing the permanent magnet object')

    # Set some hyperparameters for the optimization
    t1 = time.time()

    # Set an initial condition
    #m0 = np.ravel(np.array([pm_opt.m_maxima, np.zeros(pm_opt.ndipoles), np.zeros(pm_opt.ndipoles)]).T)
    #m0 = np.ravel(np.array([pm_opt.m_maxima, pm_opt.m_maxima, pm_opt.m_maxima]).T) / np.sqrt(4)
    #m0 = np.ravel(np.array([pm_opt.m_maxima, pm_opt.m_maxima, pm_opt.m_maxima]).T) / np.sqrt(3)
    #m0 = np.ravel((np.random.rand(pm_opt.ndipoles, 3) - 0.5) * 2 * np.array([pm_opt.m_maxima, pm_opt.m_maxima, pm_opt.m_maxima]).T / np.sqrt(12))
    m0 = np.zeros(pm_opt.m0.shape)

    # Optimize the permanent magnets, increasing L0 threshold as converging
    for i in range(37):
        reg_l0_scaled = reg_l0 * (1 + i / 2.0)
        RS_history, m_history, m_proxy_history = pm_opt._optimize(
            max_iter_MwPGP=max_iter_MwPGP, epsilon=epsilon, min_fb=min_fb,
            reg_l2=reg_l2, reg_l0=reg_l0_scaled, nu=nu, max_iter_RS=max_iter_RS,
            m0=m0
        )
        m0 = pm_opt.m 

    t2 = time.time()
    print('Done optimizing the permanent magnet object')
    print('Process took t = ', t2 - t1, ' s')

    # Print effective permanent magnet volume
    M_max = 1.465 / (4 * np.pi * 1e-7)
    dipoles = pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3)
    print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
    print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

    # Plot the sparse and less sparse solutions from SIMSOPT 
    t1 = time.time()
    m_copy = np.copy(pm_opt.m)
    pm_opt.m = pm_opt.m_proxy
    b_dipole_proxy = DipoleField(pm_opt)
    b_dipole_proxy.set_points(s.gamma().reshape((-1, 3)))
    b_dipole_proxy._toVTK(OUT_DIR + "Dipole_Fields_Sparse")
    pm_opt.m = m_copy
    b_dipole = DipoleField(pm_opt)
    b_dipole.set_points(s.gamma().reshape((-1, 3)))
    b_dipole._toVTK(OUT_DIR + "Dipole_Fields")
    t2 = time.time()
    print('Done setting up the Dipole Field class')
    print('Process took t = ', t2 - t1, ' s')

    # Print optimized metrics
    t1 = time.time()
    dphi = (pm_opt.phi[1] - pm_opt.phi[0]) * 2 * np.pi
    dtheta = (pm_opt.theta[1] - pm_opt.theta[0]) * 2 * np.pi
    print("Average Bn without the PMs = ", 
          np.mean(np.abs(Bnormal * dphi * dtheta)))
    print("Total Bn without the PMs = ", 
          np.sum((pm_opt.b_obj) ** 2) / 2.0)
    print("Total Bn without the coils = ", 
          np.sum((pm_opt.A_obj @ pm_opt.m) ** 2) / 2.0)
    print("Total Bn = ", 
          0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))
    print("Total Bn (sparse) = ", 
          0.5 * np.sum((pm_opt.A_obj @ pm_opt.m_proxy - pm_opt.b_obj) ** 2))

    # Compute metrics with permanent magnet results
    Nnorms = np.ravel(np.sqrt(np.sum(pm_opt.plasma_boundary.normal() ** 2, axis=-1)))
    Ngrid = pm_opt.nphi * pm_opt.ntheta
    ave_Bn_proxy = np.mean(np.abs(pm_opt.A_obj.dot(pm_opt.m_proxy) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms)) / (2 * pm_opt.nphi * pm_opt.ntheta)
    Bn_Am = (pm_opt.A_obj.dot(pm_opt.m)) * np.sqrt(Ngrid / Nnorms) 
    Bn_opt = (pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms) 
    ave_Bn = np.mean(np.abs(Bn_opt) / (2 * pm_opt.nphi * pm_opt.ntheta))
    print('<B * n> with the optimized permanent magnets = {0:.8e}'.format(ave_Bn)) 
    print('<B * n> with the sparsified permanent magnets = {0:.8e}'.format(ave_Bn_proxy)) 

    Bnormal_dipoles = np.sum(b_dipole.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=-1)
    Bnormal_total = Bnormal + Bnormal_dipoles
    print("Average Bn with the PMs = ", 
          np.mean(np.abs(Bnormal_total) / (2 * pm_opt.nphi * pm_opt.ntheta)))
    print('F_B INITIAL = ', SquaredFlux(s, b_dipole, -Bnormal).J())
    print('F_B INITIAL * 2 * nfp = ', 2 * s.nfp * SquaredFlux(pm_opt.plasma_boundary, b_dipole, -pm_opt.Bn).J())

    dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
    num_nonzero = np.count_nonzero(dipoles_m[:, 0] ** 2 + dipoles_m[:, 1] ** 2 + dipoles_m[:, 2] ** 2) / pm_opt.ndipoles * 100
    print("Number of possible dipoles = ", pm_opt.ndipoles)
    print("% of dipoles that are nonzero = ", num_nonzero)
    dipoles = np.ravel(dipoles)
    print('Dipole field setup done')

    make_optimization_plots(RS_history, m_history, m_proxy_history, pm_opt, OUT_DIR)
    t2 = time.time()
    print("Done printing and plotting, ", t2 - t1, " s")

    if comm is None or comm.rank == 0:
        # double the plasma surface resolution for the vtk plots
        t1 = time.time()
        qphi = 2 * s.nfp * nphi + 1
        qtheta = ntheta + 1
        endpoint = True
        quadpoints_phi = np.linspace(0, 1, qphi, endpoint=endpoint) 
        quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=endpoint)
        srange = 'full torus' 

        if surface_flag == 'focus':
            s_plot = SurfaceRZFourier.from_focus(surface_filename, range=srange, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
        elif surface_flag == 'wout':
            s_plot = SurfaceRZFourier.from_wout(surface_filename, range=srange, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
        else:
            s_plot = SurfaceRZFourier.from_vmec_input(surface_filename, range=srange, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)

        if config_flag == 'ncsx':
            # Ampere's law for a purely toroidal field: 2 pi R B0 = mu0 I
            net_poloidal_current_Amperes = 3.7713e+6
            mu0 = 4 * np.pi * 1e-7
            RB = mu0 * net_poloidal_current_Amperes / (2 * np.pi)
            bs = ToroidalField(R0=1, B0=RB)
            bs.set_points(s_plot.gamma().reshape((-1, 3)))
            Bnormal = np.sum(bs.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)
        else:
            bs = Optimizable.from_file(IN_DIR + 'BiotSavart.json')
            bs.set_points(s_plot.gamma().reshape((-1, 3)))
            Bnormal = np.sum(bs.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)

        b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
        b_dipole_proxy.set_points(s_plot.gamma().reshape((-1, 3)))
        Bnormal_dipoles = np.sum(b_dipole.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)
        Bnormal_dipoles_proxy = np.sum(b_dipole_proxy.B().reshape((len(quadpoints_phi), len(quadpoints_theta), 3)) * s_plot.unitnormal(), axis=2)
        Bnormal_total = Bnormal + Bnormal_dipoles
        Bnormal_total_proxy = Bnormal + Bnormal_dipoles_proxy

        # For plotting Bn on the full torus surface at the end with just the dipole fields
        make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")
        make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized")
        make_Bnormal_plots(b_dipole_proxy, s_plot, OUT_DIR, "only_m_proxy_optimized")
        pointData = {"B_N": Bnormal_total[:, :, None]}
        s_plot.to_vtk(OUT_DIR + "m_optimized", extra_data=pointData)
        pointData = {"B_N": Bnormal_total_proxy[:, :, None]}
        s_plot.to_vtk(OUT_DIR + "m_proxy_optimized", extra_data=pointData)
        t2 = time.time()
        print('Done saving final vtk files, ', t2 - t1, " s")

        # Print optimized f_B and other metrics
        f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
        print('f_B = ', f_B_sf)
        B_max = 1.465
        mu0 = 4 * np.pi * 1e-7
        total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * s.stellsym * mu0 / B_max
        total_volume_sparse = np.sum(np.sqrt(np.sum(pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * s.stellsym * mu0 / B_max
        print('Total volume for m and m_proxy = ', total_volume, total_volume_sparse)
        pm_opt.m = pm_opt.m_proxy
        b_dipole = DipoleField(pm_opt)
        b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
        f_B_sp = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
        print('f_B_sparse = ', f_B_sp)
        dipoles = pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3)
        num_nonzero_sparse = np.count_nonzero(dipoles[:, 0] ** 2 + dipoles[:, 1] ** 2 + dipoles[:, 2] ** 2) / pm_opt.ndipoles * 100
        np.savetxt(OUT_DIR + 'final_stats.txt', [f_B_sf, f_B_sp, num_nonzero, num_nonzero_sparse, total_volume, total_volume_sparse])

    # Save optimized permanent magnet class object
    file_out = open(OUT_DIR + class_filename + "_optimized.pickle", "wb")

    # SurfaceRZFourier objects not pickle-able, so set to None
    pm_opt.plasma_boundary = None
    pm_opt.rz_inner_surface = None
    pm_opt.rz_outer_surface = None
    pickle.dump(pm_opt, file_out)
elif run_type == 'post-processing':
    # Load in MPI, VMEC, etc. if doing a final run 
    # to generate a VMEC wout file which can be 
    # used to plot symmetry-breaking bmn, the flux
    # surfaces, epsilon_eff, etc. 
    from mpi4py import MPI
    from simsopt.util.mpi import MpiPartition
    from simsopt.mhd.vmec import Vmec
    mpi = MpiPartition(ngroups=4)
    comm = MPI.COMM_WORLD

    # Load in optimized PMs
    IN_DIR = scratch_path + config_flag + "_nphi{0:d}_ntheta{1:d}_dr{2:.2e}_coff{3:.2e}_poff{4:.2e}/".format(nphi, ntheta, dr, coff, poff)

    # Read in the correct subdirectory with the optimization output
    OUT_DIR = IN_DIR + "output_regl2{5:.2e}_regl0{6:.2e}_nu{7:.2e}/".format(nphi, ntheta, dr, coff, poff, reg_l2, reg_l0, nu)
    os.makedirs(OUT_DIR, exist_ok=True)
    pickle_name = IN_DIR + class_filename + ".pickle"
    pm_opt = pickle.load(open(pickle_name, "rb", -1))
    pm_opt.out_dir = OUT_DIR 
    pm_opt.plasma_boundary = s

    b_dipole = DipoleField(pm_opt)
    b_dipole.set_points(s.gamma().reshape((-1, 3)))
    b_dipole._toVTK(OUT_DIR + "Dipole_Fields_fully_optimized")

    # run Poincare plots
    t1 = time.time()

    # Make higher resolution surface
    quadpoints_phi = np.linspace(0, 1, 2 * nphi, endpoint=True)
    qphi = len(quadpoints_phi)
    quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
    if surface_flag == 'focus': 
        s_plot = SurfaceRZFourier.from_focus(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    elif surface_flag == 'wout': 
        s_plot = SurfaceRZFourier.from_wout(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    else:
        s_plot = SurfaceRZFourier.from_vmec_input(surface_filename, range="full torus", quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)

    # Read in the Bnormal or BiotSavart fields from any coils 
    if config_flag != 'ncsx':
        pm_opt.m = pm_opt.m_proxy
        bs = Optimizable.from_file(IN_DIR + 'BiotSavart.json')
        bs.set_points(s.gamma().reshape((-1, 3)))
        Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
    
        # need to call set_points again here for the combined field
        Bfield = Optimizable.from_file(IN_DIR + 'BiotSavart.json') + DipoleField(pm_opt)
        Bfield_tf = Optimizable.from_file(IN_DIR + 'BiotSavart.json') + DipoleField(pm_opt)
        Bfield.set_points(s.gamma().reshape((-1, 3)))
    else:

        # Set up the contribution to Bnormal from a purely toroidal field.
        # Ampere's law for a purely toroidal field: 2 pi R B0 = mu0 I
        net_poloidal_current_Amperes = 3.7713e+6
        mu0 = 4 * np.pi * (1e-7)
        RB = mu0 * net_poloidal_current_Amperes / (2 * np.pi)
        print('B0 of toroidal field = ', RB)
        bs = ToroidalField(R0=1, B0=RB)

        # Calculate Bnormal
        bs.set_points(s.gamma().reshape((-1, 3)))
        Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
        
        Bfield = ToroidalField(R0=1, B0=RB) + DipoleField(pm_opt)
        Bfield_tf = ToroidalField(R0=1, B0=RB) + DipoleField(pm_opt)
        Bfield.set_points(s.gamma().reshape((-1, 3)))

    run_Poincare_plots(s_plot, bs, b_dipole, config_flag, comm)
    t2 = time.time()
    print('Done with Poincare plots with the permanent magnets, t = ', t2 - t1)

    # Make the QFM surfaces
    t1 = time.time()
    qfm_surf = make_qfm(s, Bfield, Bfield_tf)
    t2 = time.time()
    print("Making the QFM took ", t2 - t1, " s")

    # Run VMEC with new QFM surface
    t1 = time.time()
    #try:
    ### Always use the QA VMEC file and just change the boundary
    vmec_input = "../../tests/test_files/input.LandremanPaul2021_QA" 
    equil = Vmec(vmec_input, mpi)
    #equil.boundary(qfm_surf)
    equil.boundary = qfm_surf
    #    equil._boundary = qfm_surf
    #    equil.need_to_run_code = True
    equil.run()
    #except RuntimeError:
    #    print('VMEC cannot be initialized from a wout file for some reason.')

    t2 = time.time()
    print("VMEC took ", t2 - t1, " s")

t_end = time.time()
print('Total time = ', t_end - t_start)

# Show the figures
# plt.show()
