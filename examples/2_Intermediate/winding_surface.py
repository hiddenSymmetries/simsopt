from matplotlib import pyplot as plt
import numpy as np
from simsopt.objectives import SquaredFlux
from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
from simsopt.geo import SurfaceRZFourier
from simsoptpp import WindingSurfaceBn_REGCOIL
from pathlib import Path
import os
import time
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
files = ['regcoil_out.hsx.nc']


def run_scan():
    """
        Run the REGCOIL and Lasso-regularized winding surface 
        problems across a wide range of regularization values,
        and generate comparison plots. 
    """
    mpol = 8
    ntor = 8
    for file in files:
        filename = TEST_DIR / file

        # Load current potential from NCSX configuration from REGCOIL
        cpst = CurrentPotentialSolve.from_netcdf(filename)
        cp = CurrentPotentialFourier.from_netcdf(filename)

        # Overwrite low-resolution NCSX config with higher-resolution
        # current potential now with mpol = 16, ntor = 16.
        cp = CurrentPotentialFourier(
            cpst.winding_surface, mpol=mpol, ntor=ntor,
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            stellsym=True)
        cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.B_GI)

        # define a number of geometric quantities from the plasma and coil surfaces
        s_coil = cpst.winding_surface
        s_plasma = cpst.plasma_surface
        normal_coil = s_coil.normal().reshape(-1, 3)
        normN = np.linalg.norm(normal_coil, axis=-1)
        nfp = s_plasma.nfp
        nphi = len(s_plasma.quadpoints_phi)
        ntheta = len(s_plasma.quadpoints_theta)
        points = s_plasma.gamma().reshape(-1, 3)
        normal = s_plasma.normal().reshape(-1, 3)
        normN_plasma = np.linalg.norm(normal, axis=-1)
        ws_points = s_coil.gamma().reshape(-1, 3)
        dtheta_coil = s_coil.quadpoints_theta[1]
        dzeta_coil = s_coil.quadpoints_phi[1]

        # function needed for saving to vtk after optimizing
        contig = np.ascontiguousarray

        # Loop through wide range of regularization values
        lambdas = np.logspace(-14, -10, 4)
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
            Kmax_tikhonov[i] = np.max(np.linalg.norm(K, axis=-1))
            Kmean_tikhonov[i] = np.mean(np.linalg.norm(K, axis=-1))

            Bfield_opt = WindingSurfaceField(cp_opt)
            Bfield_opt.set_points(s_plasma.gamma().reshape((-1, 3)))

            # For agreement at low regularization,
            # Bnormal MUST be computed with the function below
            # Since using Biot Savart will be discretized differently
            Bnormal_REGCOIL = WindingSurfaceBn_REGCOIL(
                points,
                ws_points,
                normal_coil,
                cp_opt.Phi(),
                normal
            ) * dtheta_coil * dzeta_coil
            Bnormal_REGCOIL += cpst.B_GI
            Bn = Bnormal_REGCOIL + cpst.Bnormal_plasma
            Bmax_tikhonov[i] = np.max(abs(Bn))
            Bmean_tikhonov[i] = np.mean(abs(Bn))
            res = (np.ravel(Bn) ** 2) @ normN_plasma
            f_B_manual = 0.5 * res / (nphi * ntheta)

            # Check that fB calculations are consistent
            print('f_B from least squares = ', f_B)
            print('f_B direct = ', f_B_manual)
            f_B_sf = SquaredFlux(
                s_plasma,
                Bfield_opt,
                -contig(cpst.Bnormal_plasma.reshape(nphi, ntheta))
            ).J()
            print('f_B from plasma surface = ', f_B_sf)

            # Repeat with the L1 instead of the L2 norm!
            optimized_phi_mn, f_B, f_K, fB_history, fK_history = cpst.solve_lasso(lam=lambda_reg, max_iter=10000, acceleration=True)

            # Make plots of the history so we can see convergence was achieved
            plt.figure(100)
            plt.semilogy(fB_history)
            plt.grid(True)
            plt.figure(101)
            plt.semilogy(lambda_reg * np.array(fK_history), label='{0:.2e}'.format(lambda_reg))
            plt.grid(True)
            plt.figure(102)
            plt.semilogy(fB_history + lambda_reg * np.array(fK_history), label='{0:.2e}'.format(lambda_reg))
            plt.grid(True)

            # repeat computing the metrics we defined
            fB_lasso[i] = f_B
            fK_l1_lasso[i] = f_K
            K = cp_opt.K()
            K2 = np.sum(K ** 2, axis=2)
            f_K_direct = 0.5 * np.sum(np.ravel(K2) * normN) / (normal_coil.shape[0])
            fK_lasso[i] = f_K_direct
            cp_opt = cpst.current_potential
            K = np.ascontiguousarray(cp_opt.K())
            Kmax_lasso[i] = np.max(np.linalg.norm(K, axis=-1))
            Kmean_lasso[i] = np.mean(np.linalg.norm(K, axis=-1))
            Bfield_opt = WindingSurfaceField(cp_opt)
            Bfield_opt.set_points(s_plasma.gamma().reshape((-1, 3)))
            Bnormal_REGCOIL = WindingSurfaceBn_REGCOIL(
                points,
                ws_points,
                normal_coil,
                cp_opt.Phi(),
                normal
            ) * dtheta_coil * dzeta_coil
            Bnormal_REGCOIL += cpst.B_GI
            Bn = Bnormal_REGCOIL + cpst.Bnormal_plasma
            Bmax_lasso[i] = np.max(abs(Bn))
            Bmean_lasso[i] = np.mean(abs(Bn))
            res = (np.ravel(Bn) ** 2) @ normN_plasma
            f_B_manual = 0.5 * res / (nphi * ntheta)

            print('Results from Lasso: ')
            print('f_B from least squares = ', f_B)
            print('f_B direct = ', f_B_manual)
            f_B_sf = SquaredFlux(
                s_plasma,
                Bfield_opt,
                -contig(cpst.Bnormal_plasma.reshape(nphi, ntheta))
            ).J()
            print('f_B from plasma surface = ', f_B_sf)

        # Finalize and save figures
        plt.figure(100)
        plt.savefig(OUT_DIR + 'fB_history.jpg')
        plt.figure(101)
        plt.legend()
        plt.savefig(OUT_DIR + 'fK_history.jpg')
        plt.figure(102)
        plt.legend()
        plt.savefig(OUT_DIR + 'f_history.jpg')

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
        # plt.plot(fK_l1_lasso, fB_lasso, 'm', label='L1 (using the L1 fK)')
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

        # Save results, useful if the run was intensive
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
    """
        Run REGCOIL (L2 regularization) and Lasso (L1 regularization)
        starting from high regularization to low. When fB < fB_target
        is achieved, the algorithms quit. This allows one to compare
        L2 and L1 results at comparable levels of fB, which seems
        like the fairest way to compare them.
    """

    fB_target = 1e-2
    mpol = 4
    ntor = 4
    coil_ntheta_res = 1
    coil_nzeta_res = coil_ntheta_res
    plasma_ntheta_res = coil_ntheta_res
    plasma_nzeta_res = coil_ntheta_res

    for file in files:
        filename = TEST_DIR / file

        # Load in low-resolution NCSX file from REGCOIL
        cpst = CurrentPotentialSolve.from_netcdf(
            filename, plasma_ntheta_res, plasma_nzeta_res, coil_ntheta_res, coil_nzeta_res
        )
        cp = CurrentPotentialFourier.from_netcdf(filename, coil_ntheta_res, coil_nzeta_res)

        # Overwrite low-resolution file with more mpol and ntor modes
        cp = CurrentPotentialFourier(
            cpst.winding_surface, mpol=mpol, ntor=ntor,
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            stellsym=True)
        cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.B_GI)
        s_coil = cpst.winding_surface

        nfp = s_coil.nfp
        mpol = s_coil.mpol
        ntor = s_coil.ntor
        nphi = len(s_coil.quadpoints_phi)
        ntheta = len(s_coil.quadpoints_theta)
        quadpoints_phi = np.linspace(0, 1, nphi + 1, endpoint=True)
        quadpoints_theta = np.linspace(0, 1, ntheta + 1, endpoint=True)
        s_coil_full = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor, stellsym=s_coil.stellsym, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
        s_coil_full.x = s_coil.local_full_x
        G = cp.net_poloidal_current_amperes
        I = cp.net_toroidal_current_amperes
        phi_secular, theta_secular = np.meshgrid(quadpoints_phi, quadpoints_theta, indexing='ij')
        Phi_secular = G * phi_secular + I * theta_secular
        # function needed for saving to vtk after optimizing
        contig = np.ascontiguousarray

        # Loop through wide range of regularization values
        lambdas = np.flip(np.logspace(-22, -10, 2))
        for i, lambda_reg in enumerate(lambdas):
            # Solve the REGCOIL problem that uses Tikhonov regularization (L2 norm)
            optimized_phi_mn, f_B, _ = cpst.solve_tikhonov(lam=lambda_reg)
            print(i, lambda_reg, f_B)
            cp_opt = cpst.current_potential

            if f_B < fB_target:
                K = cp_opt.K()
                print('fB < fB_target has been achieved: ')
                print('f_B from least squares = ', f_B)
                print('lambda = ', lambda_reg)
                make_Bnormal_plots(
                    cpst,
                    OUT_DIR,
                    file + "_tikhonov_fBtarget_Bnormal_lambda{0:.2e}".format(lambda_reg)
                )
                Phi = cp_opt.Phi()
                Phi = np.hstack((Phi, Phi[:, 0:1]))
                Phi = np.vstack((Phi, Phi[0, :])) + Phi_secular
                K = np.hstack((K, K[:, 0:1, :]))
                K = np.vstack((K, K[0:1, :, :]))
                pointData = {"phi": contig(Phi[:, :, None]),
                             "K": (contig(K[..., 0]), contig(K[..., 1]), contig(K[..., 2]))
                             }
                s_coil_full.to_vtk(
                    OUT_DIR + file + "_tikhonov_fBtarget_winding_surface_lambda{0:.2e}".format(lambda_reg),
                    extra_data=pointData
                )
                break
        print('Now repeating for Lasso: ')
        for i, lambda_reg in enumerate(lambdas):
            # Solve the REGCOIL problem with the Lasso
            optimized_phi_mn, f_B, _, fB_history, _ = cpst.solve_lasso(lam=lambda_reg, max_iter=5000, acceleration=True)
            print(i, lambda_reg, f_B)
            cp_opt = cpst.current_potential

            if f_B < fB_target:
                plt.figure(100)
                plt.semilogy(fB_history)
                plt.ylabel('fB')
                plt.xlabel('Iterations')
                plt.grid(True)
                K = contig(cp_opt.K())
                print('fB < fB_target has been achieved: ')
                print('f_B from Lasso = ', f_B)
                print('lambda = ', lambda_reg)
                make_Bnormal_plots(
                    cpst,
                    OUT_DIR,
                    file + "_lasso_fBtarget_Bnormal_lambda{0:.2e}".format(lambda_reg)
                )
                Phi = cp_opt.Phi()
                Phi = np.hstack((Phi, Phi[:, 0:1]))
                Phi = np.vstack((Phi, Phi[0, :]))
                K = np.hstack((K, K[:, 0:1, :]))
                K = np.vstack((K, K[0:1, :, :]))
                pointData = {"phi": contig(Phi[:, :, None]),
                             "K": (contig(K[..., 0]), contig(K[..., 1]), contig(K[..., 2]))
                             }
                s_coil_full.to_vtk(
                    OUT_DIR + file + "_lasso_fBtarget_winding_surface_lambda{0:.2e}".format(lambda_reg),
                    extra_data=pointData
                )
                break
        cpst.write_regcoil_out(filename='simsopt_' + file)


# Run one of the functions and time it
t1 = time.time()
# run_scan()
run_target()
t2 = time.time()
print('Total run time = ', t2 - t1)
plt.show()
