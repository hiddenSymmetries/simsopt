import numpy as np
import torch
from simsopt.geo import SurfaceBSpline
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from bo_utils import from_unit_cube
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

INVALID_PENALTY = np.array([-100])

mpi = MpiPartition()

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
    # surf.plot()
    # plt.show()

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
            nfp=rz_surf.nfp,
            surf=rz_surf,
            mpi=mpi,
            ns=13,
            M=12, 
            N=12,
            ftol=1e-7
        )
        # rz_surf.plot()
        vmec.run()

        # Same target function as stage_one_splines.py: QuasisymmetryRatioResidual
        # over radii 0..1 in steps of 0.1 (helicity (m,n)=(1,0), i.e. QA),
        # combined with aspect ratio (goal=6) and mean_iota (goal=0.42)
        # targets via the same weighted-least-squares formula
        # LeastSquaresProblem.from_tuples([(qs.residuals, 0, 1),
        # (vmec.aspect, 6, 10), (vmec.mean_iota, 0.42, 10)]) uses --
        # residuals = [unweighted_residual * sqrt(weight)], cost =
        # 0.5*sum(residuals**2) (scipy least_squares' own convention).
        qs = QuasisymmetryRatioResidual(
            vmec,
            np.arange(0, 1.01, 0.1),
            helicity_m=1,
            helicity_n=0,
        )
        residuals = np.concatenate([
            qs.residuals(),
            np.sqrt(10) * np.array([vmec.aspect() - 6]),
            np.sqrt(10) * np.array([vmec.mean_iota() - 0.42]),
        ])
        res = -0.5 * np.sum(residuals**2)
        resnn = np.nan_to_num(res, nan=INVALID_PENALTY, posinf=INVALID_PENALTY, neginf=INVALID_PENALTY)
        return np.maximum(resnn, INVALID_PENALTY)
    except Exception as e:
        print(f'Failed with exception {e}, appending invalid penalty')
        #surf.plot()
        #plt.show()
        return INVALID_PENALTY


