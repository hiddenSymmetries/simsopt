import numpy as np
from simsoptpp import acc_prox_grad_descent, matmult, matmult_sparseA
import time
from matplotlib import pyplot as plt
# from scipy.linalg import svd

__all__ = ['relax_and_split_increasingl0', 'cstlsq', ]


def prox_group_l0(alpha, threshold, n, num_basis):
    alpha_mat = alpha.reshape(n, num_basis)
    alpha_mat[np.linalg.norm(alpha_mat.reshape(n, num_basis), axis=-1) < threshold, :] = 0.0
    return alpha_mat.reshape(n * num_basis, 1)


def relax_and_split_increasingl0(
        winding_volume, lam=0.0, nu=1e20, max_iter=5000, sigma=1.0,
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
    P = winding_volume.P
    n = winding_volume.N_grid
    nfp = winding_volume.plasma_boundary.nfp
    num_basis = winding_volume.n_functions
    N = n * num_basis
    if alpha0 is not None:
        alpha_opt = alpha0
    else:
        alpha_opt = (np.random.rand(N, 1) - 0.5) * 1e4  

    if P is not None:
        alpha_opt = P.dot(alpha_opt)

    B = winding_volume.B_matrix
    I = winding_volume.Itarget_matrix
    BT = B.T
    b = winding_volume.b_rhs
    b_I = winding_volume.Itarget_rhs
    BTb = BT @ b
    IT = I.T
    ITbI = IT * b_I 
    contig = np.ascontiguousarray
    I = I.reshape(1, len(winding_volume.Itarget_matrix))
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
    #optimization_dict = {'A': B, 'b': b, 'C': winding_volume.C.todense(), 'A_I': I, 'b_I': b_I}
    #np.save('optimization_matrices.npy', optimization_dict)
    t2 = time.time()

    lam_nu = (lam + 1.0 / nu)

    print('Time to setup the algo = ', t2 - t1, ' s')

    # upper bound on largest singular value from https://www.cs.yale.edu/homes/spielman/BAP/lect3.pdf
    t1 = time.time()
    L = np.linalg.svd(BT @ B + sigma * IT @ I + np.eye(N) * lam_nu, compute_uv=False, hermitian=True)[0]
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
    winding_volume.alpha_history = []
    winding_volume.w_history = []
    t1 = time.time()
    #w_opt = alpha_opt  # np.zeros(alpha_opt.shape)  # prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)
    w_opt = np.zeros(alpha_opt.shape)  # prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)
    for j, threshold in enumerate(l0_thresholds):
        print('threshold iteration = ', j + 1, ' / ', len(l0_thresholds), ', threshold = ', threshold)
        for k in range(rs_max_iter):
            step_size_i = step_size
            alpha_opt_prev = alpha_opt 
            BTb_ITbI_nuw = BTb + sigma * ITbI + w_opt / nu
            # lam_nu = (lam + 1.0 / nu)
            # max_it = max_iter + 1
            # print(w_opt, alpha_opt)
            for i in range(max_iter + 1):
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
                winding_volume.w_history.append(w_opt)
                winding_volume.alpha_history.append(alpha_opt)
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
    winding_volume.alphas = alpha_opt
    winding_volume.w = np.ravel(w_opt)
    winding_volume.J = np.zeros((n, winding_volume.Phi.shape[2], 3))
    winding_volume.J_sparse = np.zeros((n, winding_volume.Phi.shape[2], 3))
    alphas = winding_volume.alphas.reshape(n, num_basis)
    ws = winding_volume.w.reshape(n, num_basis)
    for i in range(3):
        for j in range(winding_volume.Phi.shape[2]):
            for k in range(num_basis):
                winding_volume.J[:, j, i] += alphas[:, k] * winding_volume.Phi[k, :, j, i]
                winding_volume.J_sparse[:, j, i] += ws[:, k] * winding_volume.Phi[k, :, j, i]
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
        winding_volume, lam=0.0, max_iter=5000, sigma=1.0,
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
    P = winding_volume.P
    n = winding_volume.N_grid
    nfp = winding_volume.plasma_boundary.nfp
    num_basis = winding_volume.n_functions
    N = n * num_basis
    if alpha0 is not None:
        alpha_opt = alpha0
    else:
        alpha_opt = (np.random.rand(N, 1) - 0.5) * 1e4  

    if P is not None:
        alpha_opt = P.dot(alpha_opt)

    B = winding_volume.B_matrix
    I = winding_volume.Itarget_matrix
    BT = B.T
    b = winding_volume.b_rhs
    b_I = winding_volume.Itarget_rhs
    BTb = BT @ b
    IT = I.T
    ITbI = IT * b_I 
    contig = np.ascontiguousarray
    I = I.reshape(1, len(winding_volume.Itarget_matrix))
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
    C = winding_volume.C
    #optimization_dict = {'A': B, 'b': b, 'C': winding_volume.C.todense(), 'A_I': I, 'b_I': b_I}
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
    winding_volume.alpha_history = []
    winding_volume.w_history = []
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
    #remaining_inds = np.setdiff1d(np.arange(winding_volume.N_grid), removed_inds)

    ### Need to carefully reinsert if there were removals in multiple iterations of stlsq
    for removed_i in np.flip(removed_inds):
        for i in removed_i:
            alpha_opt = np.insert(alpha_opt, i, 0.0, axis=0)
    alpha_opt = np.ravel(alpha_opt)
    n = winding_volume.N_grid
    print('Avg |alpha| in a cell = ', np.mean(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
    winding_volume.alphas = alpha_opt
    winding_volume.w = alpha_opt 
    winding_volume.J = np.zeros((n, winding_volume.Phi.shape[2], 3))
    alphas = winding_volume.alphas.reshape(n, num_basis)
    for i in range(3):
        for j in range(winding_volume.Phi.shape[2]):
            for k in range(num_basis):
                winding_volume.J[:, j, i] += alphas[:, k] * winding_volume.Phi[k, :, j, i]
                #winding_volume.J[:, j, i] += alphas[:, k] * winding_volume.Phi[k, remaining_inds, j, i]
    winding_volume.J_sparse = winding_volume.J
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return (alpha_opt, 
            0.5 * 2 * np.array(f_B) * nfp, 
            0.5 * np.array(f_K), 
            0.5 * np.array(f_I), 
            f_0,
            )
