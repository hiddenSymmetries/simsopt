#!/usr/bin/env python

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from botorch.acquisition import qExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from simsopt.geo import SurfaceBSpline

from bo_utils import write_doflist_maxlist_minlist, from_unit_cube
from test_target import target, parallel_batch_target

import numpy as np

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cpu")
dtype = torch.double
SMOKE_TEST = False
# SMOKE_TEST = os.environ.get("SMOKE_TEST")
from mpi4py import MPI

@dataclass
class TurboState:
    """Turbo state used to track the recent history of the trust region."""
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 5e-7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_x = None
    restart_triggered: bool = False

    def __post_init__(self):
        """Post-initialize the state of the trust region."""
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_state(state: TurboState, X_next: torch.Tensor, Y_next: torch.Tensor, ub, lb) -> TurboState:
    """Update the state of the trust region based on the new function values."""
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0


    if max(Y_next).item() > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.best_x = from_unit_cube(X_next[torch.argmax(Y_next)], lb, ub)
    state.best_value = max(state.best_value, max(Y_next).item())

    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def get_initial_points(dim: int, n_pts: int, seed: int = 0) -> torch.Tensor:
    """Generate initial points using Sobol sequence."""
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    return sobol.draw(n=n_pts).to(dtype=dtype, device=device)

def generate_batch(
    state: TurboState,
    model: SingleTaskGP,  # GP model
    X: torch.Tensor,  # Evaluated points on the domain [0, 1]^d
    Y: torch.Tensor,  # Function values
    batch_size: int,
    n_candidates: Optional[int] = None,  # Number of candidates for Thompson sampling
    num_restarts: int = 20,
    raw_samples: int = 512,
    acqf: str = "ts",  # "ei" or "ts"
) -> torch.Tensor:
    """Generate a new batch of points."""
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0
    assert X.max() <= 1.0
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()
    rank = comm.Get_rank()
    batch_size = nranks

    spline_kwargs = {
        'axis_points':3,
        'points_per_cs':6,
        'n_cs':4,
        'nfp':2,
        'M':9,
        'N':4,
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
    
    dim = len(ub)
    n_init=5*batch_size if SMOKE_TEST else 2*dim
    max_cholesky_size = float("inf")

    if rank ==0:
        def eval_objective(x):
            return target(x, spline_kwargs, lb, ub)

        X_turbo = []
        Y_turbo = []
        X_sobol = SobolEngine(dimension=len(lb), scramble=True, seed=0)
        count=0
        while count < n_init:#len(initial_y) < n_init:
            X = X_sobol.draw(batch_size).to(dtype=dtype, device=device)
            #print(f'X: {X}')
            X_turbo.append(X)
            stop=[0]
            new_y, new_yvar = parallel_batch_target(X, spline_kwargs, lb, ub, stop)
            Y_turbo.append(new_y)
            count += batch_size
        #print(Y_turbo)
        X_turbo = torch.Tensor(np.array(X_turbo).reshape(-1, dim)).to(torch.double)
        Y_turbo = torch.Tensor(np.array(Y_turbo)).reshape(-1, 1).to(torch.double)

        print(f'Intitial function values: {Y_turbo}')
        
        state = TurboState(dim, batch_size=batch_size, best_value=max(Y_turbo).item())

        NUM_RESTARTS = 25 if not SMOKE_TEST else 2
        RAW_SAMPLES = 512 if not SMOKE_TEST else 4
        N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4

        torch.manual_seed(0)

        while not state.restart_triggered:  # Run until TuRBO converges
            # Fit a GP model
            train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
            )
            model = SingleTaskGP(X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # Do the fitting and acquisition function optimization inside the Cholesky context
            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                # Fit the model
                fit_gpytorch_mll(mll)

                # Create a batch
                X_next = generate_batch(
                    state=state,
                    model=model,
                    X=X_turbo,
                    Y=train_Y,
                    batch_size=batch_size,
                    n_candidates=N_CANDIDATES,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                    acqf="ts",
                )
            stop=[0]
            Y_next = parallel_batch_target(X_next, spline_kwargs, lb, ub, stop)[0]
                # , dtype=dtype, device=device
            #.unsqueeze(-1)

            # Update state
            state = update_state(state=state, X_next=X_next, Y_next=Y_next, ub=ub, lb=lb)

            # Append data
            X_turbo = torch.cat((X_turbo, X_next), dim=0)
            Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

            # Print current status
            print(f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}\nat point: {state.best_x}")
            print(f"length scales: {model.covar_module.base_kernel.lengthscale.squeeze().detach()}")
    else:
        stop=[0]
        dummy_surf = SurfaceBSpline(
            **spline_kwargs
        )
        while stop[0]==0:
            parallel_batch_target(dummy_surf.x, spline_kwargs, lb, ub, stop)
