#!/usr/bin/env python

import numpy as np
from simsopt.util.mpi import MpiPartition, proc0_print, log
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.geo.surfacespline import SurfaceBSpline, vmec_from_surf
from simsopt.solve.mpi import least_squares_mpi_solve, bounded_least_squares_mpi_solve
from simsopt._core import Optimizable

from simsopt._core.types import RealArray
from typing import Union
import numpy as np
from simsopt.mhd.vmec import Vmec
from scipy.interpolate import interp1d

def alan_QuasisymmetryRatioResidual(vmec: Vmec,
                 surfaces: Union[float, RealArray],
                 helicity_m: int = 1,
                 helicity_n: int = 0,
                 weights: RealArray = None,
                 ntheta: int = 63,
                 nphi: int = 64):
    vmec = vmec
    ntheta = ntheta
    nphi = nphi
    helicity_m = helicity_m
    helicity_n = helicity_n

    # Make sure surfaces is a list:
    try:
        surfaces = list(surfaces)
    except:
        surfaces = [surfaces]

    if weights is None:
        weights = np.ones(len(surfaces))
    else:
        weights = weights
    assert len(weights) == len(surfaces)

    vmec.run()
    if vmec.wout.lasym:
        raise RuntimeError('Quasisymmetry class cannot yet handle non-stellarator-symmetric configs')

    ns = len(surfaces)
    ntheta = ntheta
    nphi = nphi
    nfp = vmec.wout.nfp
    d_psi_d_s = -vmec.wout.phi[-1] / (2 * np.pi)

    # First, interpolate in s to get the quantities we need on the surfaces we need.
    method = 'linear'

    interp = interp1d(vmec.s_half_grid, vmec.wout.iotas[1:], fill_value="extrapolate")
    iota = interp(surfaces)

    interp = interp1d(vmec.s_half_grid, vmec.wout.bvco[1:], fill_value="extrapolate")
    G = interp(surfaces)

    interp = interp1d(vmec.s_half_grid, vmec.wout.buco[1:], fill_value="extrapolate")
    I = interp(surfaces)

    interp = interp1d(vmec.s_half_grid, vmec.wout.gmnc[:, 1:], fill_value="extrapolate")
    gmnc = interp(surfaces)

    interp = interp1d(vmec.s_half_grid, vmec.wout.bmnc[:, 1:], fill_value="extrapolate")
    bmnc = interp(surfaces)

    interp = interp1d(vmec.s_half_grid, vmec.wout.bsubumnc[:, 1:], fill_value="extrapolate")
    bsubumnc = interp(surfaces)

    interp = interp1d(vmec.s_half_grid, vmec.wout.bsubvmnc[:, 1:], fill_value="extrapolate")
    bsubvmnc = interp(surfaces)

    interp = interp1d(vmec.s_half_grid, vmec.wout.bsupumnc[:, 1:], fill_value="extrapolate")
    bsupumnc = interp(surfaces)

    interp = interp1d(vmec.s_half_grid, vmec.wout.bsupvmnc[:, 1:], fill_value="extrapolate")
    bsupvmnc = interp(surfaces)

    theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    phi2d, theta2d = np.meshgrid(phi1d, theta1d)
    phi3d = phi2d.reshape((1, ntheta, nphi))
    theta3d = theta2d.reshape((1, ntheta, nphi))

    myshape = (ns, ntheta, nphi)
    modB = np.zeros(myshape)
    d_B_d_theta = np.zeros(myshape)
    d_B_d_phi = np.zeros(myshape)
    sqrtg = np.zeros(myshape)
    bsubu = np.zeros(myshape)
    bsubv = np.zeros(myshape)
    bsupu = np.zeros(myshape)
    bsupv = np.zeros(myshape)
    residuals3d = np.zeros(myshape)
    for jmn in range(len(vmec.wout.xm_nyq)):
        m = vmec.wout.xm_nyq[jmn]
        n = vmec.wout.xn_nyq[jmn]
        angle = m * theta3d - n * phi3d
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        modB += np.kron(bmnc[jmn, :].reshape((ns, 1, 1)), cosangle)
        d_B_d_theta += np.kron(bmnc[jmn, :].reshape((ns, 1, 1)), -m * sinangle)
        d_B_d_phi += np.kron(bmnc[jmn, :].reshape((ns, 1, 1)), n * sinangle)
        sqrtg += np.kron(gmnc[jmn, :].reshape((ns, 1, 1)), cosangle)
        bsubu += np.kron(bsubumnc[jmn, :].reshape((ns, 1, 1)), cosangle)
        bsubv += np.kron(bsubvmnc[jmn, :].reshape((ns, 1, 1)), cosangle)
        bsupu += np.kron(bsupumnc[jmn, :].reshape((ns, 1, 1)), cosangle)
        bsupv += np.kron(bsupvmnc[jmn, :].reshape((ns, 1, 1)), cosangle)

    B_dot_grad_B = bsupu * d_B_d_theta + bsupv * d_B_d_phi
    B_cross_grad_B_dot_grad_psi = d_psi_d_s * (bsubu * d_B_d_phi - bsubv * d_B_d_theta) / sqrtg

    dtheta = theta1d[1] - theta1d[0]
    dphi = phi1d[1] - phi1d[0]
    V_prime = nfp * dtheta * dphi * np.sum(sqrtg, axis=(1, 2))
    # Check that we can evaluate the flux surface average <1> and the result is 1:
    assert np.sum(np.abs(np.sqrt((1 / V_prime) * nfp * dtheta * dphi * np.sum(sqrtg, axis=(1, 2))) - 1)) < 1e-12

    meanB = np.abs((np.sum(modB * sqrtg, axis=(1, 2)) / V_prime * nfp * dtheta * dphi))
    nn = helicity_n * nfp
    for js in range(ns):
        residuals3d[js, :, :] = np.sqrt(weights[js] * nfp * dtheta * dphi / V_prime[js] * sqrtg[js, :, :]) \
            * (B_cross_grad_B_dot_grad_psi[js, :, :] * (nn - iota[js] * helicity_m) \
                - B_dot_grad_B[js, :, :] * (helicity_m * G[js] + nn * I[js])) \
            / (modB[js, :, :] ** 2 * meanB[js])

    residuals1d = residuals3d.reshape((ns * ntheta * nphi,))
    profile = np.sum(residuals3d * residuals3d, axis=(1, 2))
    total = np.sum(residuals1d * residuals1d)

    return residuals1d

def ar_target(vmec, target):
    val = vmec.aspect()
    return (val - target)

def iota_target(vmec, target):
    val = vmec.mean_iota()#np.abs(vmec.mean_iota())
    return (val - target)
import argparse
import matplotlib.pyplot as plt

mpi = MpiPartition()
mpi.write()

# log()

class PASOpt(Optimizable):
    def __init__(self, surf, spline_kwargs):
        self.surf = surf
        self.spline_kwargs = spline_kwargs
        Optimizable.__init__(self, depends_on=[surf])

    def J_qa(self):
        new_surf = SurfaceBSpline(
            **self.spline_kwargs
        )
        new_surf.set_dofs_from_vec(self.x)
        # new_surf.x = self.x
        rz_surf = new_surf.to_RZFourier(
            nu=64,
            nv=64,
            nv_interp=128,
            nu_interp=128,
            collocation='arclength',
            plot=False,
            spec_cond_options={
                'plot':False,
                'ftol':1e-4,
                'Mtol':1.1,
                'shapetol':None,
                'niters':2000,
                'verbose':False,
                'cutoff':1e-5
            }
        )

        vmec = vmec_from_surf(
            self.surf.nfp,
            rz_surf,
            ns=13,
            M=12, 
            N=12,
            ftol=1e-8
        )
        vmec.run()

        #qa
        ar_penalty = ar_target(vmec, 6)
        # iota_penalty = np.sqrt(10)*iota_target(vmec, 0.42)
        qs_penalty = alan_QuasisymmetryRatioResidual(vmec, np.linspace(0.02, 1, 20), 1, 0)

        qs_s1 = np.sqrt(5)*alan_QuasisymmetryRatioResidual(vmec, np.arange(1.0, 1.1, 0.1), 1, 0)
        iota_edge_penalty = np.sqrt(10)*(vmec.iota_edge() - 0.42)
        iota_axis_penalty = np.sqrt(10)*(vmec.iota_axis () - 0.42)

        residuals = np.array([ar_penalty] + [iota_edge_penalty] + [iota_axis_penalty])
        residuals = np.concatenate((residuals, qs_penalty, qs_s1))
        res = np.sum(residuals**2)
        return residuals

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--axis_points', type=int, help = 'number of control points for the axis per half-field period')
    parser.add_argument('--points_per_cs', type=int, help = 'points per cross section')
    parser.add_argument('--n_cs', type=int, help= 'number of cross sections')
    parser.add_argument('--nfp', type=int, help = 'number of field periods')
    parser.add_argument('--M', type=int, help = 'max poloidal mode number')
    parser.add_argument('--N', type=int, help = 'max toroidal mode number')
    parser.add_argument('--p_u', type=int, help='degree of spline in poloidal direction')
    parser.add_argument('--p_v', type=int, help='degree of spline in toroidal direction')
    parser.add_argument('--cs_equispaced', type=bool, help = 'boolean, whether the dofs are equispaced or not', action=argparse.BooleanOptionalAction)
    parser.add_argument('--axis_angles_fixed', type=bool, help = 'boolean, whether the axis angle dofs are fixed', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cs_global_angle_free', type=bool, help = 'boolean, whether there is a dof for the rotation of an entire cross section', action=argparse.BooleanOptionalAction)
    parser.add_argument('--r', type=int, help='random number seed')
    parser.add_argument('--optimizer', type=int, help = 'type of optimizer. 1: lsq-trf, 2: bounded lsq-trf, 2: minimize-bfgs')
    parser.add_argument('--abs_step', type=float, help='abs_step for finite difference')
    parser.add_argument('--rel_step', type=float, help='rel_step for finite difference')
    parser.add_argument('--diff_method', type=str, help='method for finite difference')
    parser.add_argument('--cs_basis', type=str, help='basis of cross section dofs (polar/cartesian)')
    parser.add_argument('--nurbs', type=bool, help='basis of cross section dofs (polar/cartesian)', action=argparse.BooleanOptionalAction)

    input_args = vars(parser.parse_args())

    default_args = {
        'axis_points':3,
        'points_per_cs':4,
        'n_cs':6,
        'nfp':2,
        'M':12,
        'N':12,
        'p_u':3,
        'p_v':3,
        'cs_equispaced':False,
        'cs_global_angle_free':False,
        'axis_angles_fixed':False,
        'cs_basis':'polar',
        'nurbs': True,
    }

    proc0_print(input_args)
    spline_kwargs = {key:input_args[key] if input_args[key] is not None else default_args[key] for key in default_args}
    proc0_print(f'spline_kwargs: {spline_kwargs}')

    spline_surf = SurfaceBSpline(
        **spline_kwargs
    )

    # spline_surf.plot()
    # plt.show()

    # rng for initial surface if applicable

    if input_args['r'] is not None:
        np.random.seed(input_args['r']) 
        new_x = np.random.uniform(spline_surf.lower_bounds, spline_surf.upper_bounds)
        spline_surf.x = new_x

       
    myopt = PASOpt(spline_surf, spline_kwargs)

    prob = LeastSquaresProblem.from_tuples([
        (myopt.J_qa, 0, 1)
    ])

    proc0_print(f'prob.bounds: {prob.bounds}')
    proc0_print(f'initial: {repr(spline_surf.x)}, ndofs = {len(spline_surf.x)}')

    default_optimizer_args = {
        'abs_step':1e-5,
        'rel_step':1e-8,
        'diff_method':'forward',
    }

    optimizer_args = {key:input_args[key] if (input_args[key]is not None) else default_optimizer_args[key] for key in default_optimizer_args}

    proc0_print(f'optimizer_args: {optimizer_args}')

    if input_args['optimizer']== 0:
        least_squares_mpi_solve(prob, mpi=mpi, grad=True, **optimizer_args)
    elif input_args['optimizer'] == 1 or input_args['optimizer'] is None:
        bounded_least_squares_mpi_solve(prob, mpi=mpi, grad=True, **optimizer_args)
    else:
        raise NameError

    proc0_print(f'final: {repr(myopt.x)}')

 
