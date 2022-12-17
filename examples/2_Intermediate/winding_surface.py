from matplotlib import pyplot as plt
import numpy as np
from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.field import CurrentPotential, CurrentPotentialFourier, CurrentPotentialSolveTikhonov
from pathlib import Path
import os
from pyevtk.hl import pointsToVTK

TEST_DIR = Path(__file__).parent / ".." / ".." / "tests" / "test_files"


def make_Bnormal_plots(bs, s_plot, OUT_DIR, bs_filename):
    """
        Plot Bnormal on plasma surface from a MagneticField object.
        Do this quite a bit in the permanent magnet optimization
        and initialization so this is a wrapper function to reduce
        the amount of code.
    """


def make_winding_surface_plots(cp, OUT_DIR, bs_filename):
    """
    """
    s_plot = cp.winding_surface
    nphi = len(s_plot.quadpoints_phi)
    ntheta = len(s_plot.quadpoints_theta)
    K = np.ascontiguousarray(cp.K())
    pointData = {"phi": cp.Phi()[:, :, None], "K": K} 
    s_plot.to_vtk(OUT_DIR + bs_filename, extra_data=pointData)


OUT_DIR = 'simsopt_winding_surface_example/'
os.makedirs(OUT_DIR, exist_ok=True)
for file in ['regcoil_out.w7x.nc', 'regcoil_out.axisymmetry.nc', 'regcoil_out.li383.nc']:
    filename = TEST_DIR / file
    cpst = CurrentPotentialSolveTikhonov.from_netcdf(filename)
    cp = CurrentPotentialFourier.from_netcdf(filename)
    s_plasma = cpst.plasma_surface

    nphi = len(s_plasma.quadpoints_phi)
    ntheta = len(s_plasma.quadpoints_theta)

    # Solve the least-squares problem with the specified plasma
    # quadrature points, normal vector, and Bnormal at these quadrature points
    lambdas = [1e-15, 1e-21]
    for i, lambda_reg in enumerate(lambdas):

        optimized_phi_mn = cpst.solve(lam=lambda_reg)
        print(i, np.mean(abs(cp.K())))
        #cp.set_current_potential(optimized_phi_mn)
        cp.set_dofs(optimized_phi_mn)
        print(i, np.mean(abs(cp.K())))
        Bfield_opt = WindingSurfaceField(cp)
        Bfield_opt.set_points(s_plasma.gamma().reshape((-1, 3)))
        Bn_ext = - (cpst.B_GI + cpst.Bnormal_plasma).reshape(nphi, ntheta)
        Bn_coil = np.sum(Bfield_opt.B().reshape((nphi, ntheta, 3)) * s_plasma.unitnormal(), axis=2)
        pointData = {"Bn": Bn_coil[:, :, None], "Bn_ext": Bn_ext[:, :, None], "Bn_total": (Bn_ext + Bn_coil)[:, :, None]}
        s_plasma.to_vtk(OUT_DIR + file + "_Bnormal_lambda{0:.2e}".format(lambda_reg), extra_data=pointData)

        make_winding_surface_plots(cp, OUT_DIR, file + "_winding_surface_lambda{0:.2e}".format(lambda_reg))
