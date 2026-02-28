import numpy as np
import torch
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import Vmec
from bo_utils import from_unit_cube
import matplotlib.pyplot as plt
from mpi4py import MPI

from simsopt._core.types import RealArray
from typing import Union
import numpy as np
from simsopt.mhd.vmec import Vmec
from scipy.interpolate import interp1d

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

INVALID_PENALTY = np.array([-2])
from bo_utils import from_unit_cube

from mpi4py import MPI

def parallel_batch_target(candidates, spline_kwargs, lb, ub, stopp):
    stopp[0]=comm.bcast(stopp[0], root=0)
    if stopp[0]==0:
        x = comm.scatter(candidates, root=0)
        val = target(x.flatten(), spline_kwargs, lb, ub)
        gathered = comm.gather(val)
        if rank==0:
            print(f'batch complete. ')
            Y_cand = np.array(gathered)
            return torch.Tensor(Y_cand).reshape(-1, 1)

def target(X, spline_kwargs, lb, ub):
    dofs = from_unit_cube(X, lb, ub)
    # print(dofs)

    surf = SurfaceBSpline(
        **spline_kwargs
    )
    assert len(surf.x) == len(dofs), f'len(surf.x): {len(surf.x)}, len(dofs): {len(dofs)}'
    surf.set_dofs_from_vec(np.array(dofs))

    try:
        rz_surf = surf.to_RZFourier(
            nu=64,
            nv=64,
            nv_interp=128,
            nu_interp=128,
            collocation='arclength',
            plot=False,
            spec_cond=True,
            spec_cond_options={
            'plot':False,
            'ftol':1e-4,
            'Mtol':1.1,
            'shapetol':None,
            'niters':5000,
            'verbose':False,
            'cutoff':1e-6
            }
        )
        # print_dofs_nicely(surf, lb, ub)
        # surf.plot()
        # plt.show()
        vmec = Vmec.vmec_from_surf(
            nfp=surf.nfp,
            surf=rz_surf,
            ns=13
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
        res = -np.log10(0.5*np.sum(residuals**2))
        ressq = np.nan_to_num(res, nan=INVALID_PENALTY, posinf=INVALID_PENALTY, neginf=INVALID_PENALTY)
        return np.maximum(ressq, INVALID_PENALTY)
    except Exception as e:
        print(f'Failed with exception {e}, appending invalid penalty')
        #surf.plot()
        #plt.show()
        return INVALID_PENALTY

def ar_target(vmec, target):
    val = vmec.aspect()
    return (val - target)

def iota_target(vmec, target):
    val = vmec.mean_iota()#np.abs(vmec.mean_iota())
    return (val - target)

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


