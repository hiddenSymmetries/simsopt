from matplotlib import pyplot as plt
import numpy as np
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
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
files = ['regcoil_out.li383.nc']


def run_scan():
    mpol = 16
    ntor = 16
    for file in files:
        filename = TEST_DIR / file
        cpst = CurrentPotentialSolve.from_netcdf(filename)
        cp = CurrentPotentialFourier.from_netcdf(filename)
        cp = CurrentPotentialFourier(
            cpst.winding_surface, mpol=mpol, ntor=ntor,
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            stellsym=True)
        cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.Bnormal_plasma, cpst.B_GI)
        s_coil = cpst.winding_surface
        s_plasma = cpst.plasma_surface
        normal_coil = s_coil.normal().reshape(-1, 3)
        normN = np.linalg.norm(normal_coil, axis=-1)
        nfp = s_plasma.nfp
        nphi = len(s_plasma.quadpoints_phi)
        ntheta = len(s_plasma.quadpoints_theta)

        # function needed for saving to vtk after optimizing
        contig = np.ascontiguousarray

        # Loop through wide range of regularization values
        lambdas = np.logspace(-26, -8, 80)
        fB_tikhonov = np.zeros(len(lambdas))
        fB_lasso = np.zeros(len(lambdas))
        fK_tikhonov = np.zeros(len(lambdas))
        fK_lasso = np.zeros(len(lambdas))
        fK_l1_lasso = np.zeros(len(lambdas))
        Kmax_tikhonov = np.zeros(len(lambdas))
        Kmean_tikhonov = np.zeros(len(lambdas))
        Kmax_lasso = np.zeros(len(lambdas))
        Kmean_lasso = np.zeros(len(lambdas))
        Bmax_tikhonov = np.zeros(len(lambdas))
        Bmean_tikhonov = np.zeros(len(lambdas))
        Bmax_lasso = np.zeros(len(lambdas))
        Bmean_lasso = np.zeros(len(lambdas))
        for i, lambda_reg in enumerate(lambdas):
            print(i, lambda_reg)
            # Solve the REGCOIL problem that uses Tikhonov regularization (L2 norm)
            optimized_phi_mn, f_B, _ = cpst.solve_tikhonov(lam=lambda_reg)
            fB_tikhonov[i] = f_B
            cp_opt = cpst.current_potential
            K = cp_opt.K()
            K2 = np.sum(K ** 2, axis=2)
            f_K_direct = 0.5 * np.sum(np.ravel(K2) * normN) / (normal_coil.shape[0])
            fK_tikhonov[i] = f_K_direct
            K = np.ascontiguousarray(K)
            Kmax_tikhonov[i] = np.max(abs(K))
            Kmean_tikhonov[i] = np.mean(abs(K))

            Bfield_opt = WindingSurfaceField(cp_opt)
            Bfield_opt.set_points(s_plasma.gamma().reshape((-1, 3)))
            Bn_coil = np.sum(Bfield_opt.B().reshape((nphi, ntheta, 3)) * s_plasma.unitnormal(), axis=2)
            Bmax_tikhonov[i] = np.max(abs(Bn_coil))
            Bmean_tikhonov[i] = np.mean(abs(Bn_coil))

            print('f_B from least squares = ', f_B)
            f_B_sf = SquaredFlux(
                s_plasma, 
                Bfield_opt, 
                contig(cpst.Bnormal_plasma.reshape(nphi, ntheta))
            ).J()
            print('f_B from plasma surface = ', f_B_sf)

            #make_Bnormal_plots(cpst, OUT_DIR, file + "_tikhonov_Bnormal_lambda{0:.2e}".format(lambda_reg)) 
            #pointData = {"phi": contig(cp_opt.Phi()[:, :, None]), "K": (contig(K[..., 0]), contig(K[..., 1]), contig(K[..., 2]))}
            #s_coil.to_vtk(OUT_DIR + file + "_tikhonov_winding_surface_lambda{0:.2e}".format(lambda_reg), extra_data=pointData)

            # Repeat with the L1 instead of the L2 norm!
            optimized_phi_mn, f_B, f_K = cpst.solve_lasso(lam=lambda_reg)
            fB_lasso[i] = f_B
            fK_l1_lasso[i] = f_K 
            K = cp_opt.K()
            K2 = np.sum(K ** 2, axis=2)
            f_K_direct = 0.5 * np.sum(np.ravel(K2) * normN) / (normal_coil.shape[0])
            fK_lasso[i] = f_K_direct
            cp_opt = cpst.current_potential
            K = np.ascontiguousarray(cp_opt.K())
            Kmax_lasso[i] = np.max(abs(K))
            Kmean_lasso[i] = np.mean(abs(K))

            Bfield_opt = WindingSurfaceField(cp_opt)
            Bfield_opt.set_points(s_plasma.gamma().reshape((-1, 3)))
            Bn_coil = np.sum(Bfield_opt.B().reshape((nphi, ntheta, 3)) * s_plasma.unitnormal(), axis=2)
            Bmax_lasso[i] = np.max(abs(Bn_coil))
            Bmean_lasso[i] = np.mean(abs(Bn_coil))

            print('Results from Lasso: ')
            print('f_B from least squares = ', f_B)
            f_B_sf = SquaredFlux(
                s_plasma, 
                Bfield_opt, 
                contig(cpst.Bnormal_plasma.reshape(nphi, ntheta))
            ).J()
            print('f_B from plasma surface = ', f_B_sf)

            #make_Bnormal_plots(cpst, OUT_DIR, file + "_lasso_Bnormal_lambda{0:.2e}".format(lambda_reg))
            #pointData = {"phi": contig(cp_opt.Phi()[:, :, None]), "K": (contig(K[..., 0]), contig(K[..., 1]), contig(K[..., 2]))}
            #s_coil.to_vtk(OUT_DIR + file + "_lasso_winding_surface_lambda{0:.2e}".format(lambda_reg), extra_data=pointData)

        # plot cost function results
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 2, 1)
        plt.suptitle(file)
        plt.plot(lambdas, fB_tikhonov, 'b', label='f_B Tikhonov')
        plt.plot(lambdas, fK_tikhonov / 1e14, 'r', label='f_K Tikhonov / 1e14')
        plt.plot(lambdas, fK_tikhonov / 1e14 + fB_tikhonov, 'm', label='Total f Tikhonov')
        plt.plot(lambdas, fB_lasso, 'b--', label='f_B Lasso')
        plt.plot(lambdas, fK_lasso / 1e14, 'r--', label='f_K Lasso / 1e14')
        plt.plot(lambdas, fK_lasso / 1e14 + fB_lasso, 'm--', label='Total f Lasso')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel(r'$\lambda$')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(lambdas, Kmean_tikhonov / 1e6, 'c', label='Kmean (MA) Tikhonov')
        plt.plot(lambdas, Kmax_tikhonov / 1e6, 'k', label='Kmax (MA) Tikhonov')
        plt.plot(lambdas, Kmean_lasso / 1e6, 'c--', label='Kmean (MA) Lasso')
        plt.plot(lambdas, Kmax_lasso / 1e6, 'k--', label='Kmax (MA) Lasso')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.xlabel(r'$\lambda$')
        plt.legend()
        plt.savefig(OUT_DIR + file + '_lambda_scan.jpg')

        plt.figure()
        plt.suptitle(file)
        plt.plot(fK_tikhonov, fB_tikhonov, 'r', label='L2')
        plt.plot(fK_lasso, fB_lasso, 'b', label='L1 (same term as L2)')
        plt.plot(fK_l1_lasso, fB_lasso, 'm', label='L1 (using the L1 fK)')
        plt.xlabel(r'$f_K$')
        plt.ylabel(r'$f_B$')
        plt.grid(True)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(OUT_DIR + file + '_fK_fB.jpg')

        plt.figure()
        plt.suptitle(file)
        plt.plot(Kmax_tikhonov, Bmax_tikhonov, 'r', label='L2')
        plt.plot(Kmax_lasso, Bmax_lasso, 'b', label='L1')
        plt.ylabel(r'$max(|Bn|)$ on plasma surface')
        plt.xlabel(r'$max(K)$ on coil surface')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(OUT_DIR + file + '_Kmax_Bmax.jpg')
        np.savetxt(
            OUT_DIR + file + '_metrics.txt', 
            np.array(
                [lambdas,
                 fB_tikhonov, fK_tikhonov,
                 Bmax_tikhonov, Bmean_tikhonov,
                 Kmax_tikhonov, Kmean_tikhonov,
                 fB_lasso, fK_lasso, fK_l1_lasso,
                 Bmax_lasso, Bmean_lasso,
                 Kmax_lasso, Kmean_lasso,
                 ]).T
        )


def run_target():
    # Now repeat, but scan lambda for both until a target fB is achieved
    fB_target = 5e-5
    mpol = 16
    ntor = 16

    for file in files:
        filename = TEST_DIR / file
        cpst = CurrentPotentialSolve.from_netcdf(filename)
        cp = CurrentPotentialFourier.from_netcdf(filename)
        cp = CurrentPotentialFourier(
            cpst.winding_surface, mpol=mpol, ntor=ntor,
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            stellsym=True)
        cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.Bnormal_plasma, cpst.B_GI)

        s_coil = cpst.winding_surface

        # function needed for saving to vtk after optimizing
        contig = np.ascontiguousarray
        # Loop through wide range of regularization values
        lambdas = np.flip(np.logspace(-22, -10, 40))
        for i, lambda_reg in enumerate(lambdas):
            # Solve the REGCOIL problem that uses Tikhonov regularization (L2 norm)
            optimized_phi_mn, f_B, _ = cpst.solve_tikhonov(lam=lambda_reg)
            print(i, lambda_reg, f_B)
            cp_opt = cpst.current_potential
            print(optimized_phi_mn.shape)

            if f_B < fB_target:
                K = cp_opt.K()
                K = np.ascontiguousarray(K)
                print('f_B from least squares = ', f_B)
                print('lambda = ', lambda_reg)
                make_Bnormal_plots(cpst, OUT_DIR, file + "_tikhonov_fBtarget_Bnormal_lambda{0:.2e}".format(lambda_reg)) 
                pointData = {"phi": contig(cp_opt.Phi()[:, :, None]), "K": (contig(K[..., 0]), contig(K[..., 1]), contig(K[..., 2]))}
                s_coil.to_vtk(OUT_DIR + file + "_tikhonov_fBtarget_winding_surface_lambda{0:.2e}".format(lambda_reg), extra_data=pointData)
                break
        print('Now repeating for Lasso: ')
        #cpst = CurrentPotentialSolve.from_netcdf(filename)
        #cp = CurrentPotentialFourier.from_netcdf(filename)
        #cp = CurrentPotentialFourier(
        #    cpst.winding_surface, mpol=mpol, ntor=ntor,
        #    net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
        #    net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
        #    stellsym=True)
        #cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.Bnormal_plasma, cpst.B_GI)
        #cpst = CurrentPotentialSolve.from_netcdf(filename)
        #cp = CurrentPotentialFourier.from_netcdf(filename)
        #s_coil = cpst.winding_surface
        for i, lambda_reg in enumerate(lambdas):
            # Solve the REGCOIL problem with the Lasso 
            optimized_phi_mn, f_B, _ = cpst.solve_lasso(lam=lambda_reg)
            print(i, lambda_reg, f_B)
            cp_opt = cpst.current_potential

            if f_B < fB_target:
                K = cp_opt.K()
                K = np.ascontiguousarray(K)
                print('f_B from least squares = ', f_B)
                print('lambda = ', lambda_reg)
                make_Bnormal_plots(cpst, OUT_DIR, file + "_lasso_fBtarget_Bnormal_lambda{0:.2e}".format(lambda_reg)) 
                pointData = {"phi": contig(cp_opt.Phi()[:, :, None]), "K": (contig(K[..., 0]), contig(K[..., 1]), contig(K[..., 2]))}
                s_coil.to_vtk(OUT_DIR + file + "_lasso_fBtarget_winding_surface_lambda{0:.2e}".format(lambda_reg), extra_data=pointData)
                break


run_scan()
# run_target()
plt.show()
