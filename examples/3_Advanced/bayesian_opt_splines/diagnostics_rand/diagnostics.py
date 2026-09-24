#!/usr/bin/env python

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpi4py import MPI
from simsopt.geo import SurfaceBSpline

from test_target import parallel_batch_target
from bo_utils import write_doflist_maxlist_minlist

# -----------------------
# MPI setup
# -----------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()

# -----------------------
# CONFIG
# -----------------------
N_SWEEP = 100      # points per DOF for landscape
N_SOBOL = 500     # number of Sobol samples for PCA
SEED = 0          # Sobol random seed
DELTA = 0.01       # sweep +/- delta around known optimum (as fraction of DOF range)

# -----------------------
# spline setup
# -----------------------
spline_kwargs = {
    'axis_points':3,
    'points_per_cs':4,
    'n_cs':5,
    'nfp':2,
    'M':9,
    'N':4,
    'p_u':3,
    'p_v':3,
    'cs_equispaced':False,
    'rays_equispaced':False,
    'cs_global_angle_free':False,
    'axis_angles_fixed':True,
    'cs_basis':'polar',
    'nurbs': False,
}

dof_list, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)
dim = len(lb)

# -----------------------
# KNOWN OPTIMUM
# -----------------------
np.random.seed(0)
known_optimum = np.random.uniform(lb, ub)

surf = SurfaceBSpline(
    **spline_kwargs
)
surf.axis.fix('r_axis_0')
surf.set_dofs_from_vec(known_optimum)
surf.plot()
plt.show()

# -----------------------
# MPI BATCH EVALUATION
# -----------------------
def evaluate_batch(X: torch.Tensor):
    results = []
    total = X.shape[0]
    i = 0

    while i < total:
        batch = X[i:i+nranks]
        actual_size = batch.shape[0]

        # pad to exactly nranks
        if actual_size < nranks:
            pad = nranks - actual_size
            last_row = batch[-1:, :].repeat(pad, 1)
            batch = torch.cat([batch, last_row], dim=0)

        # convert to list for MPI scatter
        batch_list = [batch[j, :].clone() for j in range(nranks)]

        stop = [0]
        Y = parallel_batch_target(batch_list, spline_kwargs, lb, ub, stop)

        if rank == 0:
            Y = Y.cpu().numpy().flatten()
            results.extend(Y[:actual_size])  # only keep real outputs

        i += nranks

    if rank == 0:
        return np.array(results)
    else:
        return None
# -----------------------
# MAIN
# -----------------------
if rank == 0:
    print("Starting MPI diagnostics with bounds scaling...")

    # 1️⃣ Loss landscapes around known optimum (scaled)
    for i in range(dim):
        # number of points in each direction
        N_SWEEP_LOW = N_SWEEP // 2
        N_SWEEP_HIGH = N_SWEEP - N_SWEEP_LOW

        # sweep from lb to known optimum
        xs_low = np.linspace(lb[i], known_optimum[i], N_SWEEP_LOW, endpoint=False)
        # sweep from known optimum to ub
        xs_high = np.linspace(known_optimum[i], ub[i], N_SWEEP_HIGH, endpoint=True)

        # combine
        xs = np.concatenate([xs_low, xs_high])

        X = np.tile(known_optimum, (N_SWEEP, 1))
        X[:, i] = xs
        X = torch.tensor(X, dtype=torch.double)

        ys = evaluate_batch(X)

        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(f"DOF {i} around random value")
        plt.ylabel("Objective")
        plt.title(f"1D Landscape, DOF {i}")
        plt.grid()
        plt.savefig(f"landscape_opt_dof_{i}.png", dpi=150)
        plt.close()

        print(f"Finished landscape for DOF {i}")

    print("All 1D landscapes saved.")

    # # 2️⃣ Sobol sampling around known optimum (scaled)
    # sobol = SobolEngine(dimension=dim, scramble=True, seed=SEED)
    # X_sobol_unit = sobol.draw(N_SOBOL).to(dtype=torch.double)  # in [0,1]

    # # scale to DELTA window around known optimum
    # X_sobol = known_optimum + (X_sobol_unit - 0.5) * 2 * DELTA * (ub - lb)

    # Y_sobol = evaluate_batch(X_sobol)

    # # PCA
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_sobol.numpy())

    # plt.figure()
    # sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_sobol, s=20)
    # plt.colorbar(sc, label="Objective")
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.title("Sobol PCA Projection around known optimum")
    # plt.grid()
    # plt.savefig("pca_scatter_opt.png", dpi=150)
    # plt.close()

    # print("Sobol PCA saved. Explained variance:", pca.explained_variance_ratio_)

    # # 3️⃣ STOP workers
    # stop = [1]
    # dummy = torch.zeros((nranks, dim), dtype=torch.double)
    # parallel_batch_target([dummy[j,:].clone() for j in range(nranks)], spline_kwargs, lb, ub, stop)

else:
    # Worker loop
    stop = [0]
    dummy = torch.zeros(dim)
    while stop[0] == 0:
        parallel_batch_target(dummy, spline_kwargs, lb, ub, stop)