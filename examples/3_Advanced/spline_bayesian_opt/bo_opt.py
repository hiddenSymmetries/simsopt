#!/usr/bin/env python

from bo_model import VanillaBO
from simsopt.geo import SurfaceBSpline
from test_target import target, parallel_batch_target
import numpy as np
from bo_utils import write_doflist_maxlist_minlist, from_unit_cube
from mpi4py import MPI
from torch.quasirandom import SobolEngine
import torch

max_iter =  500

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()
    rank = comm.Get_rank()
    batch_size = nranks

    spline_kwargs = {
        'axis_points':3,
        'points_per_cs':5,
        'n_cs':4,
        'nfp':2,
        'M':6,
        'N':6,
        'p_u':3,
        'p_v':3,
        'cs_equispaced': True,
        'rays_equispaced': False,
        'cs_global_angle_free':False,
        'axis_angles_fixed':True,
        'cs_basis':'polar',
        'nurbs': False,
    }

    dof_list, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)

    n_init = 4*len(lb) #15*len(lb)

    if rank == 0:
        optimizer = VanillaBO(dof_list=dof_list, lb=lb, ub=ub, X_history = None, y_history = None, spline_kwargs=spline_kwargs, target = parallel_batch_target)

        print(f'lower bounds: {lb}')
        print(f'upper bounds: {ub}')

        # initial runs

        initial_X = []
        initial_y = []
        count = 0
        X_sobol = SobolEngine(dimension=len(optimizer.lb), scramble=True, seed=0)
        while count < n_init:#len(initial_y) < n_init:
            X = X_sobol.draw(batch_size).to(dtype=optimizer.dtype, device=optimizer.device)
            #print(f'X: {X}')
            initial_X.append(X)
            stop=[0]
            new_y = parallel_batch_target(X, optimizer.spline_kwargs, optimizer.lb, optimizer.ub, stop)
            initial_y.append(new_y)
            count += batch_size
        optimizer.X_history = torch.Tensor(np.array(initial_X).reshape(-1, optimizer.dims)).to(torch.double)
        optimizer.y_history = torch.Tensor(np.array(initial_y), device=optimizer.device).reshape(-1, 1).to(torch.double)
 
        print(f'Completed {count} initial runs. Beginning bayesian iterations. ')

        i = 0

        # bayesian runs

        while i < max_iter:
            stop=[0]
            new_x = optimizer.ask(batch_size=nranks)
            new_y = parallel_batch_target(new_x, spline_kwargs, lb, ub, stop)
            print(f'{i}/{max_iter}:\n')
            for k, _ in enumerate(new_x[:, 0]):
                # print(f'candidate {k}: {from_unit_cube(new_x[k, :], lb, ub)}')
                print(f'f(candidate{k}): {new_y[k, :]}')
            optimizer.tell(new_x, new_y, lb, ub)
            i += 1
        stop=[1]
        parallel_batch_target(new_x, spline_kwargs, lb, ub, stop)
    else:
        stop=[0]
        dummy_surf = SurfaceBSpline(
            **spline_kwargs
        )
        while stop[0]==0:
            parallel_batch_target(dummy_surf.x, spline_kwargs, lb, ub, stop)
    # except:
    #     optimizer.dump(f'opt_{max_iter}_2.pkl')
