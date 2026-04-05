from typing import Any
import torch

from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine
from botorch.models.transforms.input import Normalize

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import pickle
from typing import Optional, Sequence, Union
from bo_utils import from_unit_cube, to_unit_cube
import botorch 
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior

from gpytorch.kernels import ScaleKernel

from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition import qExpectedImprovement,  qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler

import numpy as np

WARMUP_STEPS = 512 
NUM_SAMPLES = 512 
THINNING = 16

tkwargs = {
    "device": torch.device("cpu"),
    "dtype": torch.double,
}

from math import log, sqrt
from typing import Optional, Sequence, Union

import torch
from gpytorch.constraints.constraints import GreaterThan, LessThan, Interval
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior

MIN_INFERRED_NOISE_LEVEL = 1e-7
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)

def get_covar_module_with_dim_scaled_prior_maxlengthscale_constrained(
    ard_num_dims: int,
    batch_shape: Optional[torch.Size] = None,
    use_rbf_kernel: bool = True,
    active_dims: Optional[Sequence[int]] = None,
) -> Union[MaternKernel, RBFKernel]:
    """Returns an RBF or Matern kernel with priors
    from  [Hvarfner2024vanilla]_.

    Args:
        ard_num_dims: Number of feature dimensions for ARD.
        batch_shape: Batch shape for the covariance module.
        use_rbf_kernel: Whether to use an RBF kernel. If False, uses a Matern kernel.
        active_dims: The set of input dimensions to compute the covariances on.
            By default, the covariance is computed using the full input tensor.
            Set this if you'd like to ignore certain dimensions.

    Returns:
        A Kernel constructed according to the given arguments. The prior is constrained
        to have lengthscales larger than 0.025 for numerical stability.
    """
    base_class = RBFKernel if use_rbf_kernel else MaternKernel
    lengthscale_prior = LogNormalPrior(loc=0.5, scale=0.1*SQRT3)
    base_kernel = base_class(
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=Interval(
            1e-6, 0.5, transform=None, initial_value=lengthscale_prior.mode
        ),
        eps=1e-8,# pyre-ignore[6] GPyTorch type is unnecessarily restrictive.
        active_dims=active_dims,
    )
    return base_kernel

class GlobalOptimizer():
    def __init__(self, dof_list, lb: Any, ub: Any, target: callable = None):
        self.dof_list = dof_list
        self.lb = lb
        self.ub = ub
        self.target = target
    def tell(self, X: Any, y: Any):
        pass
    def suggest(self):
        pass
    def dump(self):
        pass
    def initial_runs(self):
        pass
    def return_best(self):
        pass

class VanillaBO(GlobalOptimizer):
    def __init__(self, dof_list, lb, ub, X_history, y_history, spline_kwargs, target):
        super().__init__(dof_list, lb, ub, target)
        self.dims = len(lb)
        # all X_history are stored unscaled - this is NECESSARY for rebounding
        self.X_history = None
        self.y_history = None
        self.device = 'cpu'
        self.dtype = torch.double
        self.dof_list = dof_list
        self.spline_kwargs = spline_kwargs

    def _fitting_loop(self, batch_size):
        print(f'Current best: {self.y_history.max()}')
        print(f'at {from_unit_cube(self.X_history[np.argmax(self.y_history)], self.lb, self.ub)}')
        # bounds = torch.tensor(np.vstack([self.lb, self.ub]))
        gp = SingleTaskGP(
            train_X = self.X_history, 
            train_Y = self.y_history,            #train_Yvar=torch.full_like(scaled_y_history, 1e-6),
            outcome_transform=Standardize(m=1),
            covar_module = get_covar_module_with_dim_scaled_prior_maxlengthscale_constrained(ard_num_dims=len(self.lb), use_rbf_kernel=True)
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2048]), seed=0)
        MC_LogEI = qLogExpectedImprovement(gp, best_f=self.y_history.max(), sampler=sampler, fat=False)
        
        torch.manual_seed(seed=0)  # to keep the restart conditions the same
        candidates, _ = optimize_acqf(
            acq_function=MC_LogEI,
            bounds=torch.tensor([[0.0] * len(self.lb), [1.0] * len(self.lb)]),
            q=batch_size,
            num_restarts=128,
            raw_samples=1024,
            options={},
        )
        return gp, candidates

    def ask(self, batch_size)->np.ndarray:
        '''
        Currently only supporting single point per ask
        '''
        # assert batch_size == 1 , 'LogExpectedImprovement supports only one new point per ask'
        gp, candidates = self._fitting_loop(batch_size)
        print(f'Length scales: {gp.covar_module.lengthscale.detach()}')      
        return candidates
    
    def tell(self, X_new:np.ndarray, y_new:np.ndarray, lb:np.ndarray, ub:np.ndarray):
        '''
        Literally just concatenating to history and updating bounds
        '''
        X_new = torch.Tensor(X_new).reshape(-1, self.dims).to(torch.double)
        y_new = torch.Tensor(y_new).reshape(-1, 1).to(torch.double)

        self.X_history = torch.cat((self.X_history, X_new), dim = 0)
        self.y_history = torch.cat((self.y_history, y_new))
        self.lb = lb
        self.ub = ub

    def dump(self, fpath):
        pickle.dump(self, open(fpath, 'wb'), pickle.HIGHEST_PROTOCOL)

    def return_best(self):
        return self.y_history.max(), self.X_history[self.y_history.argmax()]

    def _custom_fit_mll(self, mll):
        mll.train()
        
    @classmethod
    def load(cls, fpath):
        with open(fpath, 'rb') as f:
            optimizer = pickle.load(f)
        return optimizer
