import numpy as np
import time
from scipy.sparse import spdiags
from scipy.sparse.linalg import LinearOperator, splu, minres
from functools import partial

__all__ = ['relax_and_split_minres']

def compute_J(current_voxels, alphas, ws):
    """
    Compute J at each quadrature point from the alphas
    determined through optimization. 

    Args:
        current_voxels: CurrentVoxelsGrid class object
          representing the geometry of the voxel grid.
        alphas: 2D numpy array, shape (num_voxels, num_basis_functions)
          representing the optimized voxel degrees of freedom.
        ws: 2D numpy array, shape (num_voxels, num_basis_functions) 
          representing the proxy voxel degrees of freedom.
    """
    current_voxels.J = np.sum(np.multiply(alphas.T[:, :, np.newaxis, np.newaxis], current_voxels.Phi), axis=0)
    current_voxels.J_sparse = np.sum(np.multiply(ws.T[:, :, np.newaxis, np.newaxis], current_voxels.Phi), axis=0)


def prox_group_l0(alpha, threshold, n, num_basis):
    """
    Computes the non-overlapping group l0 proximal operator,
    which is a very fancy way to say threshold the subgroups to 
    zero if the subgroup has norm < threshold.

    Args:
        alpha: 1D numpy array, shape (num_voxels, num_basis_functions)
          representing the optimization variables to threshold.
        threshold: float
          Value such that if a subgroup in alpha has norm < threshold,
          we set that whole subgroup to zeros.
        n: int
          Number of voxels in the unique part of the voxel grid.
        num_basis: int
          Number of basis functions in each voxel, hard-coded to 5 elsewhere.
    """
    alpha_mat = alpha.reshape(n, num_basis)
    alpha_mat[np.linalg.norm(alpha_mat.reshape(n, num_basis), axis=-1) < threshold, :] = 0.0
    return alpha_mat.reshape(n * num_basis, 1)


def relax_and_split_minres(
        current_voxels, 
        **kwargs,
):
    """
    This function performs a relax-and-split algorithm for solving the 
    current voxel problem. The convex part of the solve is done with
    (preconditioned) minres and the nonconvex part 
    can be solved with the prox of the group l0 norm... hard 
    thresholding on the whole set of alphas in a grid cell. 

    Args:
        current_voxels: CurrentVoxelGrid class object to be optimized.
        kwargs:
            Keyword arguments for this algorithm. 
            The following variables can be passed:

            kappa: float, defaults to 0.0
                Strength of Tikhonov regularization to use.
            nu: float, defaults to 1e100
                Strength of the "relaxing" term in relax-and-split algorithm.
            sigma: float, defaults to 0.0
                Strength of the Itarget loss term used to avoid the trivial solution.
            max_iter: int, defaults to 5000
                Maximum number of MINRES iterations to use during a call to MINRES.
            rs_max_iter: int, defaults to 1
                Maximum number of full relax-and-split iterations to do.
            l0_thresholds: list of floats, defaults to [0.0] 
                An increasing sequence of thresholds to use, to solve multiple
                relax-and-split problems in a row.
            alpha0: 1D numpy array, shape (N, 1)
                Initial guess for the alphas that are trying to be optimized for.
            OUT_DIR: string 
                Output directory to save MINRES progress to.
            precondition: bool, default False
                Whether to use the approximate Schur preconditioner for MINRES.
    """
    kappa = kwargs.pop("kappa", 0.0)
    sigma = kwargs.pop("sigma", 1.0)
    nu = kwargs.pop("nu", 1e100)
    max_iter = kwargs.pop("max_iter", 500)
    rs_max_iter = kwargs.pop("rs_max_iter", 1)
    l0_thresholds = kwargs.pop("l0_thresholds", [0.0])
    alpha0 = kwargs.pop("alpha0", None)
    OUT_DIR = kwargs.pop("OUT_DIR", '')
    precondition = kwargs.pop("precondition", False)
    if kappa < 0 or sigma < 0 or nu < 0 or np.any(np.array(l0_thresholds) < 0) or max_iter <= 0 or rs_max_iter <= 0:
        raise ValueError('Hyperparameters must be positive floats and max/print iterations must be positive integers')

    n = current_voxels.N_grid
    nfp = current_voxels.plasma_boundary.nfp
    num_basis = current_voxels.n_functions
    N = n * num_basis
    if alpha0 is not None:
        if alpha0.shape != (N, 1):
            if len(alpha0.shape) == 1 and alpha0.shape[0] == N:
                alpha_opt = alpha0.reshape(-1, 1)
            else:
                raise ValueError('Initial guess alpha0 has wrong shape, should be (5 * N_grid, 1)')
        else:
            alpha_opt = alpha0
    else:
        alpha_opt = np.zeros((N, 1))  # (np.random.rand(N, 1) - 0.5) * 1e4  

    B = current_voxels.B_matrix

    b_B = current_voxels.b_rhs.reshape(-1, 1)
    b_I = current_voxels.Itarget_rhs.reshape(-1, 1)
    I = current_voxels.Itarget_matrix.reshape(1, -1)

    kappa = kappa / n  # normalize kappa by the number of voxels
    kappa_nu = (kappa + 1.0 / nu)
    scale = 1e6
    C = current_voxels.C
    CT = C.T
    K = C.shape[0]  # K is the number of constraints

    # Rescale the fB and fI loss terms since they are small relative to
    # the magnitude of the constraints
    A = scale * np.vstack((B, np.sqrt(sigma) * I))
    b = scale * np.vstack((b_B, np.sqrt(sigma) * b_I))
    AT = A.T
    ATb = AT @ b

    def A_fun(x):
        """ The 2x2 KKT matrix used in minres with constraints """
        alpha = x[:N]
        return np.vstack(((AT @ (A @ alpha) + kappa_nu * alpha + CT @ x[N:])[:, np.newaxis], 
                          (C @ alpha)[:, np.newaxis]))

    A_operator = LinearOperator((N + K, N + K), matvec=A_fun)
    tol = 1e-20
    minres_kwargs = {"tol": tol, "maxiter": max_iter}

    t1 = time.time()
    if precondition:
        # setup matrices for the preconditioner
        AAdiag = np.sum(A ** 2, axis=0) + kappa_nu  # diagonal of A'*A + 1/nu*I
        AAdiag_inv_diag = (1 / AAdiag).reshape(-1, 1)
        AAdiag_inv = spdiags(1 / AAdiag.T, 0, N, N)

        # Schur complement is C*inv(A.'*A + (1/mu)*I)*C.'. 
        # Let's use the inverse diagonal of A:
        S = C @ AAdiag_inv @ CT
        perturbation_factor = abs(S).max() * 1e-7
        S = S + perturbation_factor * spdiags(np.ones(K), 0, K, K)  # regularized Schur complement
        S_inv_op = splu(S)

        def M_inv(x):
            """ 
            An approximate Schur complement preconditioner, where we 
            apply the inverse of M given by
            M_minres = [[spdiags((AAdiag'), 0, n, n), 0]
                        [sparse(k,n),                 S ]]
            to the vector [alpha, lam], or equivalently [x[:N], x[N:]].
            """
            return np.vstack((x[:N, np.newaxis] * AAdiag_inv_diag, S_inv_op.solve(x[N:, np.newaxis])))

        M_operator = LinearOperator((N + K, N + K), matvec=M_inv)
        minres_kwargs["M"] = M_operator
    t2 = time.time()
    t_preconditioner_setup = t2 - t1
    print('Time to setup precond = ', t_preconditioner_setup, ' s')

    def callback(xk, w_opt, rhs, minres_file):
        """ 
        Callback function for minres. Computes and then prints out all the 
        loss terms we are interested in.
        The output is written to a text file, rather than printed to stdout. 

        Args:
            xk: The solution vector in the current algorithm iteration.
            w_opt: The sparse proxy solution vector in the current algorithm iteration.
            rhs: The right-hand-side of the Ax=b problem that MINRES is solving.
            minres_file: The file to write the minres output to.
        """
        alpha = xk[:N, np.newaxis]
        fB = np.linalg.norm(B @ alpha - b_B) ** 2
        fI = sigma * np.linalg.norm(I @ alpha - b_I) ** 2
        fK = kappa * np.linalg.norm(alpha) ** 2 / scale ** 2
        fRS = np.linalg.norm(alpha - w_opt) ** 2 / nu / scale ** 2
        f = fB + fI + fK + fRS
        fC = np.linalg.norm(C.dot(alpha)) 
        fminres = np.linalg.norm(A_fun(xk) - rhs)
        minres_file.write(
            f"{f:.16f} {fB:.16f} {fI:.16f} {fK:.16f} {fRS:.16f} {fC:.16f} {fminres:.16f}\n"
        )

    # Now prepare to record optimization progress
    lam_opt = np.zeros((K, 1))
    t_minres = []
    current_voxels.alpha_history = []
    current_voxels.w_history = []
    set_point = True
    w_opt = alpha_opt  # prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)

    # We will write minres output to a text file so initialize the file now
    with open(OUT_DIR + 'minres_output.txt', 'w') as minres_file:
        minres_file.write('Total   fB   sigma * fI   kappa * fK   fRS/nu   C*alpha   MINRES residual\n')
        t0 = time.time()
        for j, threshold in enumerate(l0_thresholds):
            print('threshold iteration = ', j + 1, ' / ', len(l0_thresholds), ', threshold = ', threshold)

            # Line here to run extra iterations when nearing the end of the relax-and-split
            # iterations. Tends to polish up the solution a bit.
            if j > (2.0 * len(l0_thresholds) / 3.0) and set_point:
                rs_max_iter = 2 * rs_max_iter
                set_point = False
            for k in range(rs_max_iter):
                rhs = ATb + w_opt / nu
                rhs = np.vstack((rhs.reshape(len(rhs), 1), np.zeros((K, 1))))

                # now MINRES
                t1 = time.time()
                x0 = np.vstack((alpha_opt, lam_opt))
                alpha_opt, _ = minres(
                    A_operator, rhs, x0=x0, 
                    callback=partial(callback, w_opt=w_opt, rhs=rhs, minres_file=minres_file), 
                    **minres_kwargs
                )
                # Minres did zero iterations if this holds, so then use callback explicitly
                if np.allclose(alpha_opt, x0.ravel()):
                    callback(alpha_opt, w_opt, rhs, minres_file)
                t2 = time.time()
                t_minres.append(t2 - t1)
                alpha_opt = alpha_opt.reshape(-1, 1)
                lam_opt = alpha_opt[N:]
                alpha_opt = alpha_opt[:N]

                if (threshold > 0):
                    # current_voxels.w_history.append(w_opt)
                    # current_voxels.alpha_history.append(alpha_opt)
                    print('w : ', j, k,
                          np.linalg.norm(np.ravel(B @ w_opt - b_B), ord=2) ** 2,
                          np.linalg.norm(np.ravel(I @ w_opt - b_I)) ** 2 * sigma,
                          kappa * np.linalg.norm(np.ravel(w_opt)) ** 2 / scale ** 2,
                          np.linalg.norm(alpha_opt - w_opt) ** 2 / nu / scale ** 2,
                          np.linalg.norm(C.dot(w_opt)) / scale ** 2,
                          np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1)), 
                          ' / ', n, ', ',
                          )

                # now do the prox step
                w_opt = prox_group_l0(np.copy(alpha_opt), threshold, n, num_basis)

    t2 = time.time()
    relax_and_split_time = t2 - t0
    print('Time to run relax-and-split algo = ', relax_and_split_time, ' s')
    average_minres_time = np.mean(np.array(t_minres))
    print('Time on avg to complete a MINRES call = ', average_minres_time, ' s')
    t1 = time.time()
    alpha_opt = np.ravel(alpha_opt)
    print('Avg |alpha| in a cell = ', np.mean(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
    f0, fB, fI, fK, fRS, fC, fminres = np.loadtxt(OUT_DIR + 'minres_output.txt', skiprows=1, unpack=True)
    current_voxels.alphas = alpha_opt
    current_voxels.w = np.ravel(w_opt)
    current_voxels.J = np.zeros((n, current_voxels.Phi.shape[2], 3))
    current_voxels.J_sparse = np.zeros((n, current_voxels.Phi.shape[2], 3))
    alphas = current_voxels.alphas.reshape(n, num_basis)
    ws = current_voxels.w.reshape(n, num_basis)
    compute_J(current_voxels, alphas, ws)
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return_dict = {"alpha_opt": alpha_opt, "fB": 0.5 * 2 * np.array(fB) * nfp, "fK": 0.5 * np.array(fK),
                   "fI": 0.5 * np.array(fI), "fRS": 0.5 * np.array(fRS), "f0": f0, "fC": fC, "fminres": fminres,
                   "relax_and_split_time": relax_and_split_time, "t_preconditioner_setup": t_preconditioner_setup,
                   "average_minres_time": average_minres_time}
    return return_dict 
