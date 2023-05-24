import sys
import numpy as np
from simsoptpp import acc_prox_grad_descent, matmult, matmult_sparseA
import time
from matplotlib import pyplot as plt
from scipy.linalg import cholesky
from scipy.sparse import csc_matrix, spdiags, vstack, hstack, eye
from scipy.sparse.linalg import minres, LinearOperator

__all__ = ['relax_and_split_increasingl0', 'cstlsq', 'exact_convex_solve', 'ras_minres']


def prox_group_l0(alpha, threshold, n, num_basis):
    alpha_mat = alpha.reshape(n, num_basis)
    alpha_mat[np.linalg.norm(alpha_mat.reshape(n, num_basis), axis=-1) < threshold, :] = 0.0
    return alpha_mat.reshape(n * num_basis, 1)


def prox_group_l1(alpha, threshold, n, num_basis):
    alpha_mat = alpha.reshape(n, num_basis)
    #alpha_mat[np.linalg.norm(alpha_mat.reshape(n, num_basis), axis=-1) < threshold, :] = 0.0
    #alpha_mat = np.sign(alpha_mat, ax) * np.maximum(np.abs(alpha_mat) - reg_l1 * nu, 0) * np.ravel(mmax_vec)
    return alpha_mat.reshape(n * num_basis, 1)


def relax_and_split_increasingl0(
        current_voxels, lam=0.0, nu=1e20, max_iter=5000, sigma=1.0,
        rs_max_iter=1, l0_thresholds=[0.0], print_iter=10,
        alpha0=None):
    """
        This function performs a relax-and-split algorithm for solving the 
        current voxel problem. The convex part of the solve is done with
        accelerated projected gradient descent and the nonconvex part 
        can be solved with the prox of the group l0 norm... hard 
        thresholding on the whole set of alphas in a grid cell. 

        Args:
            P : projection matrix onto the linear equality constraints representing
                the flux jump constraints. If C is the flux jump constraint matrix,
                P = I - C.T @ (C * C.T)^{-1} @ C. P is very large but very sparse
                so is assumed to be stored as a scipy csc_matrix. 
    """
    t1 = time.time()
    P = current_voxels.P
    n = current_voxels.N_grid
    nfp = current_voxels.plasma_boundary.nfp
    num_basis = current_voxels.n_functions
    N = n * num_basis
    if alpha0 is not None:
        alpha_opt = alpha0
    else:
        alpha_opt = (np.random.rand(N, 1) - 0.5) * 1e4  

    if P is not None:
        alpha_opt = P.dot(alpha_opt)

    B = current_voxels.B_matrix
    I = current_voxels.Itarget_matrix
    BT = B.T
    b = current_voxels.b_rhs
    b_I = current_voxels.Itarget_rhs
    BTb = BT @ b
    IT = I.T
    ITbI = IT * b_I 
    contig = np.ascontiguousarray
    I = I.reshape(1, len(current_voxels.Itarget_matrix))
    IT = I.T
    b_I = b_I.reshape(1, 1)
    b = b.reshape(len(b), 1)
    ITbI = contig(IT * b_I)
    BTb = contig(BTb.reshape(len(BTb), 1))   
    alpha_opt = contig(alpha_opt.reshape(len(alpha_opt), 1))
    IT = contig(IT)
    BT = contig(BT)
    I = contig(I)
    B = contig(B)
    #optimization_dict = {'A': B, 'b': b, 'C': current_voxels.C.todense(), 'A_I': I, 'b_I': b_I}
    #np.save('optimization_matrices.npy', optimization_dict)
    t2 = time.time()

    lam_nu = (lam + 1.0 / nu)

    print('Time to setup the algo = ', t2 - t1, ' s')

    # upper bound on largest singular value from https://www.cs.yale.edu/homes/spielman/BAP/lect3.pdf
    t1 = time.time()
    L = np.linalg.svd(BT @ B + sigma * IT @ I + np.eye(N) * lam_nu, compute_uv=False, hermitian=True)[0]
    L0 = np.linalg.svd(BT @ B + sigma * IT @ I + np.eye(N) * lam, compute_uv=False, hermitian=True)[0]
    #if True:
    # L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + IT @ I + np.eye(N) * lam_nu, axis=0), axis=-1)
    # L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + np.eye(N) * lam_nu, axis=0), axis=-1)
    #else:
    #    L = 2e-12  # L = 1e-12 for N = 30, L = 2.4e-12 for N = 24, so not a bad guess

    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size = ', L, step_size) 
    t2 = time.time()

    print('Time to setup step size = ', t2 - t1, ' s')

    f_B = []
    f_I = []
    f_K = []
    f_RS = []
    f_Bw = []
    f_Iw = []
    f_Kw = []
    f_0 = []
    f_0w = []
    current_voxels.alpha_history = []
    current_voxels.w_history = []
    t1 = time.time()
    set_point = True
    #w_opt = alpha_opt  # np.zeros(alpha_opt.shape)  # prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)
    w_opt = np.zeros(alpha_opt.shape)  # prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)
    for j, threshold in enumerate(l0_thresholds):
        print('threshold iteration = ', j + 1, ' / ', len(l0_thresholds), ', threshold = ', threshold)
        if j > (2.0 * len(l0_thresholds) / 3.0) and set_point:
            rs_max_iter = 3 * rs_max_iter
            set_point = False
        for k in range(rs_max_iter):
            if j == 0 and k == 0 and len(l0_thresholds) > 1:
                nu_algo = 1e100
                #nu_algo = nu
                step_size_i = 1.0 / L0
                max_it = 5000
                #max_it = max_iter 
            else:
                nu_algo = nu
                step_size_i = step_size
                max_it = max_iter
            alpha_opt_prev = alpha_opt 
            BTb_ITbI_nuw = BTb + sigma * ITbI + w_opt / nu_algo
            lam_nu = (lam + 1.0 / nu_algo)
            # max_it = max_iter + 1
            # print(w_opt, alpha_opt)
            for i in range(max_it + 1):
                vi = contig(alpha_opt + i / (i + 3) * (alpha_opt - alpha_opt_prev))
                alpha_opt_prev = alpha_opt
                alpha_opt = P.dot(
                    vi + step_size_i * (BTb_ITbI_nuw - matmult(BT, matmult(B, vi)) - sigma * matmult(IT, matmult(I, vi)) - lam_nu * vi)
                )
                step_size_i = (1 + np.sqrt(1 + 4 * step_size_i ** 2)) / 2.0

                # print metrics
                if ((i % print_iter) == 0.0):
                    f_B.append(np.linalg.norm(np.ravel(B @ alpha_opt - b), ord=2) ** 2)
                    f_I.append(np.linalg.norm(np.ravel(I @ alpha_opt - b_I)) ** 2)
                    f_K.append(np.linalg.norm(np.ravel(alpha_opt)) ** 2) 
                    f_RS.append(np.linalg.norm(np.ravel(alpha_opt - w_opt) ** 2))
                    f_0.append(np.count_nonzero(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1) > threshold))
                    # f_0.append(np.count_nonzero(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
                    # ind = (j * (max_iter + 1) * rs_max_iter + k * (max_iter + 1) + i) // print_iter
                    print(j, k, i, 
                          f_B[-1], 
                          sigma * f_I[-1], 
                          lam * f_K[-1], 
                          f_RS[-1] / nu, 
                          f_0[-1], ' / ', n
                          )  
                    #if np.allclose(alpha_opt, alpha_opt_prev, rtol=1e-5):
                    #    break

            # now do the prox step
            w_opt = prox_group_l0(np.copy(alpha_opt), threshold, n, num_basis)
            if threshold > 0:
                f_Bw.append(np.linalg.norm(np.ravel(B @ w_opt - b), ord=2) ** 2)
                f_Iw.append(np.linalg.norm(np.ravel(I @ w_opt - b_I)) ** 2)
                f_Kw.append(np.linalg.norm(np.ravel(w_opt)) ** 2) 
                # f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1) > threshold))
                f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1)))
                # ind = (j * rs_max_iter + k)
                current_voxels.w_history.append(w_opt)
                current_voxels.alpha_history.append(alpha_opt)
                print('w : ', j, k, i, 
                      f_Bw[-1], 
                      f_Iw[-1] * sigma, 
                      lam * f_Kw[-1], 
                      f_0w[-1], ' / ', n, ', ',
                      )

    t2 = time.time()
    print('Time to run algo = ', t2 - t1, ' s')
    t1 = time.time()
    alpha_opt = np.ravel(alpha_opt)
    print('Avg |alpha| in a cell = ', np.mean(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
    current_voxels.alphas = alpha_opt
    current_voxels.w = np.ravel(w_opt)
    current_voxels.J = np.zeros((n, current_voxels.Phi.shape[2], 3))
    current_voxels.J_sparse = np.zeros((n, current_voxels.Phi.shape[2], 3))
    alphas = current_voxels.alphas.reshape(n, num_basis)
    ws = current_voxels.w.reshape(n, num_basis)
    for i in range(3):
        for j in range(current_voxels.Phi.shape[2]):
            for k in range(num_basis):
                current_voxels.J[:, j, i] += alphas[:, k] * current_voxels.Phi[k, :, j, i]
                current_voxels.J_sparse[:, j, i] += ws[:, k] * current_voxels.Phi[k, :, j, i]
    #current_voxels.J = current_voxels.Phi[0, :, :, :]
    # print('J = ', current_voxels.J)
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return (alpha_opt, 
            0.5 * 2 * np.array(f_B) * nfp, 
            0.5 * np.array(f_K), 
            0.5 * np.array(f_I), 
            0.5 * np.array(f_RS),
            f_0,
            0.5 * 2 * np.array(f_Bw) * nfp, 
            0.5 * np.array(f_Kw), 
            0.5 * np.array(f_Iw)
            )


def cstlsq(
        current_voxels, lam=0.0, max_iter=5000, sigma=1.0,
        stlsq_max_iter=1, l0_thresholds=[0.0], print_iter=10,
        alpha0=None):
    """
        This function performs a relax-and-split algorithm for solving the 
        current voxel problem. The convex part of the solve is done with
        accelerated projected gradient descent and the nonconvex part 
        can be solved with the prox of the group l0 norm... hard 
        thresholding on the whole set of alphas in a grid cell. 

        Args:
            P : projection matrix onto the linear equality constraints representing
                the flux jump constraints. If C is the flux jump constraint matrix,
                P = I - C.T @ (C * C.T)^{-1} @ C. P is very large but very sparse
                so is assumed to be stored as a scipy csc_matrix. 
    """
    t1 = time.time()
    P = current_voxels.P
    n = current_voxels.N_grid
    nfp = current_voxels.plasma_boundary.nfp
    num_basis = current_voxels.n_functions
    N = n * num_basis
    if alpha0 is not None:
        alpha_opt = alpha0
    else:
        alpha_opt = (np.random.rand(N, 1) - 0.5) * 1e4  

    if P is not None:
        alpha_opt = P.dot(alpha_opt)

    B = current_voxels.B_matrix
    I = current_voxels.Itarget_matrix
    BT = B.T
    b = current_voxels.b_rhs
    b_I = current_voxels.Itarget_rhs
    BTb = BT @ b
    IT = I.T
    ITbI = IT * b_I 
    contig = np.ascontiguousarray
    I = I.reshape(1, len(current_voxels.Itarget_matrix))
    IT = I.T
    b_I = b_I.reshape(1, 1)
    b = b.reshape(len(b), 1)
    ITbI = contig(IT * b_I)
    BTb = contig(BTb.reshape(len(BTb), 1))   
    alpha_opt = contig(alpha_opt.reshape(len(alpha_opt), 1))
    IT = contig(IT)
    BT = contig(BT)
    I = contig(I)
    B = contig(B)
    BTb_ITbI = BTb + sigma * ITbI
    C = current_voxels.C
    #optimization_dict = {'A': B, 'b': b, 'C': current_voxels.C.todense(), 'A_I': I, 'b_I': b_I}
    #np.save('optimization_matrices.npy', optimization_dict)
    t2 = time.time()

    print('Time to setup the algo = ', t2 - t1, ' s')

    # upper bound on largest singular value from https://www.cs.yale.edu/homes/spielman/BAP/lect3.pdf
    t1 = time.time()
    L = np.linalg.svd(BT @ B + sigma * IT @ I + np.eye(N) * lam, compute_uv=False, hermitian=True)[0]
    #if True:
    # L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + IT @ I + np.eye(N) * lam_nu, axis=0), axis=-1)
    # L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + np.eye(N) * lam_nu, axis=0), axis=-1)
    #else:
    #    L = 2e-12  # L = 1e-12 for N = 30, L = 2.4e-12 for N = 24, so not a bad guess

    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size = ', L, step_size) 
    t2 = time.time()

    print('Time to setup step size = ', t2 - t1, ' s')

    f_B = []
    f_I = []
    f_K = []
    f_0 = []
    current_voxels.alpha_history = []
    current_voxels.w_history = []
    t1 = time.time()
    removed_inds = []
    for j, threshold in enumerate(l0_thresholds):
        print('threshold iteration = ', j + 1, ' / ', len(l0_thresholds), ', threshold = ', threshold)
        for k in range(stlsq_max_iter):
            step_size_i = step_size
            alpha_opt_prev = alpha_opt 
            for i in range(max_iter + 1):
                vi = contig(alpha_opt)  # contig(alpha_opt + i / (i + 3) * (alpha_opt - alpha_opt_prev))
                #alpha_opt_prev = alpha_opt
                alpha_opt = P.dot(
                    vi + step_size_i * (BTb_ITbI - matmult(BT, matmult(B, vi)) - sigma * matmult(IT, matmult(I, vi)) - lam * vi)
                )
                step_size_i = (1 + np.sqrt(1 + 4 * step_size_i ** 2)) / 2.0

                # print metrics
                if ((i % print_iter) == 0.0):
                    f_B.append(np.linalg.norm(np.ravel(B @ alpha_opt - b), ord=2) ** 2)
                    f_I.append(np.linalg.norm(np.ravel(I @ alpha_opt - b_I)) ** 2)
                    f_K.append(np.linalg.norm(np.ravel(alpha_opt)) ** 2) 
                    f_0.append(np.count_nonzero(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1) > threshold))
                    print(j, k, i, 
                          f_B[-1], 
                          sigma * f_I[-1], 
                          lam * f_K[-1], 
                          f_0[-1], ' / ', n
                          )  
                    #if np.allclose(alpha_opt, alpha_opt_prev, rtol=1e-5):
                    #    break

            # now do the prox step
            C = C.reshape(C.shape[0], n, num_basis)
            print(alpha_opt.shape, C.shape)
            alpha_mat = alpha_opt.reshape(n, num_basis)
            small_inds = np.ravel(np.where(np.linalg.norm(alpha_mat, axis=-1) < threshold))
            removed_inds.append(small_inds)
            # alpha_mat[small_inds, :] = 0.0
            print(small_inds)
            alpha_opt = np.delete(alpha_mat, small_inds, axis=0)
            C = np.delete(C, small_inds, axis=1).reshape(C.shape[0], (n - len(small_inds)) * num_basis)
            CT = C.T
            CCT_inv = np.linalg.pinv(C @ CT, rcond=1e-8, hermitian=True)
            BTb_ITbI = np.delete(BTb_ITbI.reshape(n, num_basis), small_inds, axis=0)
            B = B.reshape(B.shape[0], n, num_basis)
            B = np.delete(B, small_inds, axis=1).reshape(B.shape[0], (n - len(small_inds)) * num_basis)
            BT = B.T
            I = I.reshape(I.shape[0], n, num_basis)
            I = np.delete(I, small_inds, axis=1).reshape(I.shape[0], (n - len(small_inds)) * num_basis)
            IT = I.T
            n -= len(small_inds)
            alpha_opt = alpha_opt.reshape(n * num_basis, 1)
            BTb_ITbI = BTb_ITbI.reshape(n * num_basis, 1)
            P = np.eye(n * num_basis) - C.T @ CCT_inv @ C
            print('P * alpha_opt - alpha_opt = ', P.dot(alpha_opt) - alpha_opt)
            print('||P * alpha_opt - alpha_opt|| / ||alpha_opt|| = ', np.linalg.norm(P.dot(alpha_opt) - alpha_opt) / np.linalg.norm(alpha_opt))

    t2 = time.time()
    print('Time to run algo = ', t2 - t1, ' s')
    t1 = time.time()
    alpha_opt = alpha_opt.reshape(n, num_basis)
    print(removed_inds)
    #remaining_inds = np.setdiff1d(np.arange(current_voxels.N_grid), removed_inds)

    ### Need to carefully reinsert if there were removals in multiple iterations of stlsq
    for removed_i in np.flip(removed_inds):
        for i in removed_i:
            alpha_opt = np.insert(alpha_opt, i, 0.0, axis=0)
    alpha_opt = np.ravel(alpha_opt)
    n = current_voxels.N_grid
    print('Avg |alpha| in a cell = ', np.mean(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
    current_voxels.alphas = alpha_opt
    current_voxels.w = alpha_opt 
    current_voxels.J = np.zeros((n, current_voxels.Phi.shape[2], 3))
    alphas = current_voxels.alphas.reshape(n, num_basis)
    for i in range(3):
        for j in range(current_voxels.Phi.shape[2]):
            for k in range(num_basis):
                current_voxels.J[:, j, i] += alphas[:, k] * current_voxels.Phi[k, :, j, i]
                #current_voxels.J[:, j, i] += alphas[:, k] * current_voxels.Phi[k, remaining_inds, j, i]
    current_voxels.J_sparse = current_voxels.J
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return (alpha_opt, 
            0.5 * 2 * np.array(f_B) * nfp, 
            0.5 * np.array(f_K), 
            0.5 * np.array(f_I), 
            f_0,
            )


def exact_convex_solve(current_voxels, lam=0.0, sigma=1.0):
    """
    """
    t1 = time.time()
    n = current_voxels.N_grid
    num_basis = current_voxels.n_functions
    N = n * num_basis
    B = current_voxels.B_matrix
    I = current_voxels.Itarget_matrix
    BT = B.T
    b = current_voxels.b_rhs
    b_I = current_voxels.Itarget_rhs
    BTb = BT @ b
    IT = I.T
    ITbI = IT * b_I 
    I = I.reshape(1, len(current_voxels.Itarget_matrix))
    IT = I.T
    b_I = b_I.reshape(1, 1)
    b = b.reshape(len(b), 1)
    ITbI = (IT * b_I).reshape(len(BTb), 1)   
    BTb = (BTb.reshape(len(BTb), 1))   
    C = current_voxels.C
    CT = C.T
    H = np.linalg.inv(BT @ B + lam * np.eye(N) + sigma * IT @ I)
    CHCT = C @ H @ CT
    CHCTinv = np.linalg.pinv(CHCT)
    alpha_opt = H @ (np.eye(N) - CT @ CHCTinv @ C @ H) @ (BTb + sigma * ITbI)
    current_voxels.alphas = alpha_opt
    current_voxels.w = alpha_opt
    current_voxels.J = np.zeros((n, current_voxels.Phi.shape[2], 3))
    alphas = current_voxels.alphas.reshape(n, num_basis)
    for i in range(3):
        for j in range(current_voxels.Phi.shape[2]):
            for k in range(num_basis):
                current_voxels.J[:, j, i] += alphas[:, k] * current_voxels.Phi[k, :, j, i]
    current_voxels.J_sparse = current_voxels.J
    t2 = time.time()
    print('Time to run algo = ', t2 - t1, ' s')
    return alpha_opt


def ras_minres(
        current_voxels, lam=0.0, nu=1e20, max_iter=5000, sigma=1.0,
        rs_max_iter=1, l0_thresholds=[0.0], print_iter=10,
        alpha0=None, OUT_DIR=''):
    """
        This function performs a relax-and-split algorithm for solving the 
        current voxel problem. The convex part of the solve is done with
        accelerated projected gradient descent and the nonconvex part 
        can be solved with the prox of the group l0 norm... hard 
        thresholding on the whole set of alphas in a grid cell. 
    """
    n = current_voxels.N_grid
    nfp = current_voxels.plasma_boundary.nfp
    num_basis = current_voxels.n_functions
    N = n * num_basis
    if alpha0 is not None:
        alpha_opt = alpha0
    else:
        alpha_opt = np.zeros((N, 1))  # (np.random.rand(N, 1) - 0.5) * 1e4  

    B = current_voxels.B_matrix
    I = current_voxels.Itarget_matrix
    BT = B.T
    b = current_voxels.b_rhs
    b_I = current_voxels.Itarget_rhs
    BTb = (BT @ b).reshape(-1, 1)
    IT = I.T
    ITbI = (IT * b_I).reshape(-1, 1) 
    contig = np.ascontiguousarray
    I = I.reshape(1, len(current_voxels.Itarget_matrix))
    IT = I.T
    b_I = b_I.reshape(1, 1)
    b_B = b.reshape(len(b), 1)
    alpha_opt = alpha_opt.reshape(len(alpha_opt), 1)
    C = current_voxels.C
    K = C.shape[0]
    CT = C.T
    lam_nu = (lam + 1.0 / nu)
    scale = 1e6
    A = scale * np.vstack((B, np.sqrt(sigma) * I))
    b = scale * np.vstack((b_B, np.sqrt(sigma) * b_I))
    AT = A.T
    ATb = AT @ b

    def A_fun(x):
        alpha = x[:N]
        Ax = np.vstack(((AT @ (A @ alpha) + lam_nu * alpha + CT @ x[N:])[:, np.newaxis], (C @ alpha)[:, np.newaxis]))
        return Ax

    def A_fun_nu0(x):
        alpha = x[:N]
        Ax = np.vstack(((AT @ (A @ alpha) + lam * alpha + CT @ x[N:])[:, np.newaxis], (C @ alpha)[:, np.newaxis]))
        return Ax

    A_operator = LinearOperator((N + K, N + K), matvec=A_fun)
    A_operator_nu0 = LinearOperator((N + K, N + K), matvec=A_fun_nu0)

    lam_opt = np.zeros((K, 1))
    tol = 1e-30
    f0 = []
    fB = []
    fI = []
    fK = []
    fRS = []
    fC = []
    f_Bw = []
    f_Iw = []
    f_Kw = []
    f_0w = []
    f_RS = []
    current_voxels.alpha_history = []
    current_voxels.w_history = []
    t1 = time.time()
    set_point = True
    w_opt = np.zeros(alpha_opt.shape)  # prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)
    for j, threshold in enumerate(l0_thresholds):
        print('threshold iteration = ', j + 1, ' / ', len(l0_thresholds), ', threshold = ', threshold)
        if j > (2.0 * len(l0_thresholds) / 3.0) and set_point:
            rs_max_iter = 2 * rs_max_iter
            set_point = False
        for k in range(rs_max_iter):
            if (j == 0) and (k == 0) and (len(l0_thresholds) > 1):
                A_op = A_operator_nu0
                nu_algo = 1e100
                max_it = 50000
            else:
                A_op = A_operator
                max_it = max_iter
                nu_algo = nu
            rhs = ATb + w_opt / nu_algo
            rhs = np.vstack((rhs.reshape(len(rhs), 1), np.zeros((K, 1))))

            def callback(xk):
                xk = xk[:N, np.newaxis]
                fB = np.linalg.norm(B @ xk - b_B) ** 2
                fI = sigma * np.linalg.norm(I @ xk - b_I) ** 2
                fK = lam * np.linalg.norm(xk) ** 2
                fRS = np.linalg.norm(xk - w_opt) ** 2 / nu_algo / scale ** 2
                f = fB + fI + fK + fRS
                print(f, fB, fI, fK, fRS,
                      np.linalg.norm(C.dot(xk)))

            # now MINRES
            t1 = time.time()
            x0 = np.vstack((alpha_opt, lam_opt)) 
            sys.stdout = open(OUT_DIR + 'minres_output.txt', 'w')
            print('Total ', 'fB ', 'sigma * fI ', 'kappa * fK ', 'C*alpha')
            alpha_opt, _ = minres(A_op, rhs, x0=x0, tol=tol, maxiter=max_it, callback=callback)
            t2 = time.time()
            sys.stdout.close()
            sys.stdout = open("/dev/stdout", "w")
            t3 = t2 - t1
            #print('MINRES took = ', t3, ' s')
            alpha_opt = alpha_opt.reshape(-1, 1)
            lam_opt = alpha_opt[N:]
            alpha_opt = alpha_opt[:N]
            #if k != 0:
            #    print(f_B[-1], f_I[-1], f_K[-1], f_C[-1])
            f_0, f_B, f_I, f_K, f_RS, f_C = np.loadtxt(OUT_DIR + 'minres_output.txt', skiprows=1, unpack=True)
            f0 += f_0.tolist()
            fB += f_B.tolist()
            fI += f_I.tolist()
            fK += f_K.tolist()
            fRS += f_RS.tolist()
            fC += f_C.tolist()
            #print(f_B[-1], f_I[-1], f_K[-1], f_C[-1])

            if threshold > 0:
                f_Bw.append(np.linalg.norm(np.ravel(B @ w_opt - b_B), ord=2) ** 2)
                f_Iw.append(np.linalg.norm(np.ravel(I @ w_opt - b_I)) ** 2)
                f_Kw.append(np.linalg.norm(np.ravel(w_opt)) ** 2) 
                #f_RS.append(np.linalg.norm(alpha_opt - w_opt) ** 2) 
                # f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1) > threshold))
                f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1)))
                current_voxels.w_history.append(w_opt)
                current_voxels.alpha_history.append(alpha_opt)
                print('w : ', j, k, 
                      f_Bw[-1], 
                      f_Iw[-1] * sigma, 
                      lam * f_Kw[-1], 
                      f_RS[-1],  # / nu,
                      f_0w[-1], ' / ', n, ', ',
                      )

            # now do the prox step
            w_opt = prox_group_l0(np.copy(alpha_opt), threshold, n, num_basis)

    t2 = time.time()
    print('Time to run algo = ', t2 - t1, ' s')
    t1 = time.time()
    alpha_opt = np.ravel(alpha_opt)
    print('Avg |alpha| in a cell = ', np.mean(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
    current_voxels.alphas = alpha_opt
    current_voxels.w = np.ravel(w_opt)
    current_voxels.J = np.zeros((n, current_voxels.Phi.shape[2], 3))
    current_voxels.J_sparse = np.zeros((n, current_voxels.Phi.shape[2], 3))
    alphas = current_voxels.alphas.reshape(n, num_basis)
    ws = current_voxels.w.reshape(n, num_basis)
    for i in range(3):
        for j in range(current_voxels.Phi.shape[2]):
            for k in range(num_basis):
                current_voxels.J[:, j, i] += alphas[:, k] * current_voxels.Phi[k, :, j, i]
                current_voxels.J_sparse[:, j, i] += ws[:, k] * current_voxels.Phi[k, :, j, i]
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return (alpha_opt, 
            0.5 * 2 * np.array(fB) * nfp, 
            0.5 * np.array(fK), 
            0.5 * np.array(fI), 
            0.5 * np.array(fRS),  # / nu,
            f0,
            fC,
            0.5 * 2 * np.array(f_Bw) * nfp, 
            0.5 * np.array(f_Kw), 
            0.5 * np.array(f_Iw)
            )
