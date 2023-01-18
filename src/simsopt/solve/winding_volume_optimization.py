import numpy as np
import simsoptpp as sopp

__all__ = ['projected_gradient_descent_Tikhonov']


def projected_gradient_descent_Tikhonov(winding_volume, lam=0.0, alpha0=None, max_iter=5000, P=None, acceleration=True):
    """
        This function performed projected gradient descent for the Tikhonov-regularized
        winding volume optimization problem (if the flux jump constraints are not used,
        then it reduces to simple gradient descent). Without constraints, flux jumps
        across cells are not constrained and therefore J is not globally
        divergence-free (although J is locally divergence-free in each cell). 
        This function is primarily for making sure this sub-problem of the
        winding volume optimization is working correctly. Note that this function
        uses gradient descent even for the unconstrained problem
        because the matrices are too big for np.linalg.solve. Also, by default it 
        uses Nesterov acceleration. 

        Args:
            P : projection matrix onto the linear equality constraints representing
                the flux jump constraints. If C is the flux jump constraint matrix,
                P = I - C.T @ (C * C.T)^{-1} @ C. P is very large but very sparse
                so is assumed to be stored as a scipy csc_matrix. 
    """
    B = winding_volume.B_matrix
    I = winding_volume.Itarget_matrix
    BT = B.T
    IT = I.T
    L = np.linalg.svd(B @ BT + I @ IT + lam * np.eye(B.shape[0]), compute_uv=False)[0]
    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size = ', L, step_size)
    # b = np.random.rand(winding_volume.b_rhs.shape[0])
    b = winding_volume.b_rhs
    b_I = winding_volume.Itarget_rhs
    BTb = BT @ b
    ITbI = IT * b_I 
    n = winding_volume.N_grid
    nfp = winding_volume.plasma_boundary.nfp
    num_basis = winding_volume.n_functions
    alpha_opt = (np.random.rand(n * num_basis) - 0.5) * 1e3  # set some initial guess
    B_inv = np.linalg.pinv(B, rcond=1e-30)
    f_B = []
    f_I = []
    f_K = []
    if P is None:
        if acceleration:  # Nesterov acceleration 
            # first iteration do regular GD 
            alpha_opt_prev = alpha_opt
            step_size_i = step_size
            alpha_opt = alpha_opt + step_size_i * (
                BTb + ITbI - BT @ (B @ alpha_opt) - IT * (I @ alpha_opt) - lam * alpha_opt
            )
            f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
            f_I.append((I @ alpha_opt - b_I) ** 2)
            f_K.append(np.linalg.norm(alpha_opt, ord=2))
            for i in range(1, max_iter):
                vi = alpha_opt + (i - 1) / (i + 2) * (alpha_opt - alpha_opt_prev)
                alpha_opt_prev = alpha_opt
                alpha_opt = vi + step_size_i * (BTb + ITbI - BT @ (B @ vi) - IT * (I @ vi) - lam * vi)
                step_size_i = (1 + np.sqrt(1 + 4 * step_size_i ** 2)) / 2.0
                f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
                f_I.append((I @ alpha_opt - b_I) ** 2)
                f_K.append(np.linalg.norm(alpha_opt, ord=2))
                if (i % 100) == 0.0:
                    print(i, f_B[i] ** 2 * nfp * 0.5, f_I[i], f_K[i])
        else:
            for i in range(max_iter):
                alpha_opt = alpha_opt + step_size * (BTb - BT @ (B @ alpha_opt) - lam * alpha_opt)
                f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
                f_K.append(np.linalg.norm(alpha_opt, ord=2))
                if (i % 100) == 0.0:
                    print(i, f_B[i] ** 2 * nfp * 0.5, alpha_opt[:10])
    else:
        for i in range(max_iter):
            alpha_opt = P.dot(alpha_opt + BTb - BT @ (B @ alpha_opt) - lam * alpha_opt)
            f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2))
            f_K.append(np.linalg.norm(alpha_opt, ord=2))
    return alpha_opt, 0.5 * np.array(f_B) ** 2 * nfp, 0.5 * np.array(f_K) ** 2, 0.5 * np.array(f_I)
