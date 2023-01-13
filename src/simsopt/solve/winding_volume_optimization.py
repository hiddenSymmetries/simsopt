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
    # B = np.random.rand(winding_volume.B_matrix.shape[0], winding_volume.B_matrix.shape[1]) # winding_volume.B_matrix
    B = winding_volume.B_matrix
    BT = B.T
    L = np.linalg.svd(B @ BT + lam * np.eye(B.shape[0]), compute_uv=False)[0]
    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size = ', L, step_size)
    # b = np.random.rand(winding_volume.b_rhs.shape[0])
    b = winding_volume.b_rhs
    BTb = BT @ b
    n = winding_volume.N_grid
    nfp = winding_volume.plasma_boundary.nfp
    num_basis = winding_volume.n_functions
    #print('B = ', B)
    #print('b = ', b)

    alpha_opt = (np.random.rand(n * num_basis) - 0.5) * 1e4  # set some initial guess
    #print('alpha initial = ', alpha_opt)
    B_inv = np.linalg.pinv(B, rcond=1e-30)
    #print('B_inv @ b = ', B_inv @ b)
    #print('B @ B_inv @ b - b = ', B @ (B_inv @ b) - b)
    #print('B @ BT = ', B @ BT)
    f_B = []
    f_K = []
    #print(np.mean(abs(BTb)), np.mean(abs(BT @ (B @ alpha_opt))))
    if P is None:
        for i in range(max_iter):
            alpha_opt = alpha_opt + step_size * (BTb - BT @ (B @ alpha_opt) - lam * alpha_opt)
            f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
            f_K.append(np.linalg.norm(alpha_opt, ord=2))
            if (i % 100) == 0.0:
                print(i, f_B[i] ** 2 * nfp * 0.5, alpha_opt[:10])
                print(np.mean(abs(BTb)), np.mean(abs(BT @ (B @ alpha_opt))))
    else:
        for i in range(max_iter):
            alpha_opt = P.dot(alpha_opt + BTb - BT @ (B @ alpha_opt) - lam * alpha_opt)
            f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
            f_K.append(np.linalg.norm(alpha_opt, ord=2))
    return alpha_opt, 0.5 * np.array(f_B) ** 2 * nfp, 0.5 * np.array(f_K) ** 2
