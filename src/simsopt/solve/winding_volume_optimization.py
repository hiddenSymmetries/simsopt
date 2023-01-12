import numpy as np
import simsoptpp as sopp

__all__ = ['projected_gradient_descent_Tikhonov']


def projected_gradient_descent_Tikhonov(winding_volume, lam=0.0, alpha0=None, max_iter=1000, P=None):
    """
        This function performed projected gradient descent for the Tikhonov-regularized
        winding volume optimization problem (if the flux jump constraints are not used,
        then it reduces to simple gradient descent). Without constraints, flux jumps
        across cells are not constrained and therefore J is not globally
        divergence-free (although J is locally divergence-free in each cell). 
        This function is primarily for making sure this sub-problem of the
        winding volume optimization is working correctly. Note that this function
        uses gradient descent even for the unconstrained problem
        because the matrices are too big for np.linalg.solve!

        Args:
            P : projection matrix onto the linear equality constraints representing
                the flux jump constraints. If C is the flux jump constraint matrix,
                P = I - C.T @ (C * C.T)^{-1} @ C. P is very large but very sparse
                so is assumed to be stored as a scipy csc_matrix. 
    """
    B = winding_volume.B_matrix
    L = np.linalg.svd(B, compute_uv=False)[0] ** 2
    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size = ', L, step_size)
    BT = B.T
    b = winding_volume.b_rhs
    BTb = step_size * (BT @ b)
    BT = step_size * BT
    lam = step_size * lam
    n = winding_volume.N_grid
    nfp = winding_volume.plasma_boundary.nfp
    num_basis = winding_volume.n_functions

    alpha_opt = np.random.rand(n * num_basis) - 0.5  # set some initial guess
    f_B = []
    f_K = []
    if P is None:
        for i in range(max_iter):
            alpha_opt = alpha_opt + BTb - BT @ (B @ alpha_opt) - lam * alpha_opt
            f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
            f_K.append(np.linalg.norm(alpha_opt, ord=2))
            if (i % 1000) == 0.0:
                print(i, f_B[i])
    else:
        for i in range(max_iter):
            alpha_opt = P.dot(alpha_opt + BTb - BT @ (B @ alpha_opt) - lam * alpha_opt)
            f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
            f_K.append(np.linalg.norm(alpha_opt, ord=2))
    return alpha_opt, 0.5 * np.array(f_B) ** 2 * nfp, 0.5 * np.array(f_K) ** 2
