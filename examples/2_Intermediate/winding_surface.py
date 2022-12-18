from matplotlib import pyplot as plt
import numpy as np
from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.field import CurrentPotential, CurrentPotentialFourier, CurrentPotentialSolve
from simsopt.geo import SurfaceRZFourier
from pathlib import Path
import os
from pyevtk.hl import pointsToVTK

TEST_DIR = Path(__file__).parent / ".." / ".." / "tests" / "test_files"


def make_Bnormal_plots(cpst, OUT_DIR, filename):
    """
        Plot Bnormal on the full torus plasma surface using the optimized
        CurrentPotentialFourier object (cpst).
    """

    # redefine the plasma surface to the full torus
    s_plasma = cpst.plasma_surface
    nfp = s_plasma.nfp
    mpol = s_plasma.mpol
    ntor = s_plasma.ntor
    nphi = len(s_plasma.quadpoints_phi)
    ntheta = len(s_plasma.quadpoints_theta)
    quadpoints_phi = np.linspace(0, 1, nfp * nphi, endpoint=True)
    quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
    s_plasma_full = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor, stellsym=s_plasma.stellsym, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
    s_plasma_full.x = s_plasma.local_full_x

    # Compute the BiotSavart fields from the optimized current potential
    Bfield_opt = WindingSurfaceField(cpst.current_potential)
    Bfield_opt.set_points(s_plasma_full.gamma().reshape((-1, 3)))
    Bn_coil = np.sum(Bfield_opt.B().reshape((nfp * nphi, ntheta, 3)) * s_plasma_full.unitnormal(), axis=2)
    Bn_ext = - cpst.Bnormal_plasma.reshape(nphi, ntheta)

    # interpolate the known Bnormal_plasma onto new grid
    if nfp > 1:
        Bn_ext_full = np.vstack((Bn_ext, Bn_ext))
        for i in range(nfp - 2):
            Bn_ext_full = np.vstack((Bn_ext_full, Bn_ext))
    else:
        Bn_ext_full = Bn_ext

    # Save all the data to file
    pointData = {"Bn": Bn_coil[:, :, None], "Bn_ext": Bn_ext_full[:, :, None], "Bn_total": (Bn_ext_full + Bn_coil)[:, :, None]}
    s_plasma_full.to_vtk(OUT_DIR + filename, extra_data=pointData)


OUT_DIR = 'simsopt_winding_surface_example/'
os.makedirs(OUT_DIR, exist_ok=True)
for file in ['regcoil_out.w7x.nc', 'regcoil_out.axisymmetry.nc', 'regcoil_out.li383.nc']:
    filename = TEST_DIR / file
    cpst = CurrentPotentialSolve.from_netcdf(filename)
    cp = CurrentPotentialFourier.from_netcdf(filename)
    s_coil = cpst.winding_surface

    # function needed for saving to vtk after optimizing
    contig = np.ascontiguousarray

    # Solve the least-squares problem with the specified plasma
    # quadrature points, normal vector, and Bnormal at these quadrature points
    lambdas = [1, 1e-5, 1e-15, 1e-20, 1e-25, 1e-30]
    fB_tikhonov = []
    fB_lasso = []
    fK_tikhonov = []
    fK_lasso = []
    Kmax_tikhonov = []
    Kmean_tikhonov = []
    Kmax_lasso = []
    Kmean_lasso = []
    for i, lambda_reg in enumerate(lambdas):
        # Solve the REGCOIL problem that uses Tikhonov regularization (L2 norm)
        optimized_phi_mn, f_B, f_K = cpst.solve_tikhonov(lam=lambda_reg)
        fB_tikhonov.append(f_B)
        fK_tikhonov.append(f_K)
        make_Bnormal_plots(cpst, OUT_DIR, file + "_tikhonov_Bnormal_lambda{0:.2e}".format(lambda_reg)) 
        cp_opt = cpst.current_potential
        K = np.ascontiguousarray(cp_opt.K())
        Kmax_tikhonov.append(np.max(abs(K)))
        Kmean_tikhonov.append(np.mean(abs(K)))

        pointData = {"phi": contig(cp_opt.Phi()[:, :, None]), "K": (contig(K[..., 0]), contig(K[..., 1]), contig(K[..., 2]))}
        s_coil.to_vtk(OUT_DIR + file + "_tikhonov_winding_surface_lambda{0:.2e}".format(lambda_reg), extra_data=pointData)

        # Repeat with the L1 instead of the L2 norm!
        optimized_phi_mn, f_B, f_K = cpst.solve_lasso(lam=lambda_reg)
        fB_lasso.append(f_B)
        fK_lasso.append(f_K)
        make_Bnormal_plots(cpst, OUT_DIR, file + "_lasso_Bnormal_lambda{0:.2e}".format(lambda_reg))
        cp_opt = cpst.current_potential
        K = np.ascontiguousarray(cp_opt.K())
        Kmax_lasso.append(np.max(abs(K)))
        Kmean_lasso.append(np.mean(abs(K)))
        pointData = {"phi": contig(cp_opt.Phi()[:, :, None]), "K": (contig(K[..., 0]), contig(K[..., 1]), contig(K[..., 2]))}
        s_coil.to_vtk(OUT_DIR + file + "_lasso_winding_surface_lambda{0:.2e}".format(lambda_reg), extra_data=pointData)

    # plot cost function results
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(lambdas, fB_tikhonov, 'b', label='f_B Tikhonov')
    plt.plot(lambdas, fK_tikhonov, 'r', label='f_K Tikhonov')
    plt.plot(lambdas, fK_tikhonov + fB_tikhonov, 'm', label='Total f Tikhonov')
    plt.xscale('log')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(lambdas, fB_lasso, 'b', label='f_B Lasso')
    plt.plot(lambdas, fK_lasso, 'r', label='f_K Lasso')
    plt.plot(lambdas, fK_lasso + fB_lasso, 'm', label='Total f Lasso')
    plt.xscale('log')
    plt.grid(True)
    plt.show()
