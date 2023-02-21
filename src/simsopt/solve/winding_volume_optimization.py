import numpy as np
from simsoptpp import acc_prox_grad_descent, matmult, matmult_sparseA
import time
from matplotlib import pyplot as plt
# from scipy.linalg import svd

__all__ = ['projected_gradient_descent_Tikhonov', 'relax_and_split', 'relax_and_split_increasingl0', 'relax_and_split_analytic']


def prox_group_l0(alpha, threshold, n, num_basis):
    alpha_mat = alpha.reshape(n, num_basis)
    alpha_mat[np.linalg.norm(alpha_mat.reshape(n, num_basis), axis=-1) < threshold, :] = 0.0
    return alpha_mat.reshape(n * num_basis, 1)


def projected_gradient_descent_Tikhonov(winding_volume, lam=0.0, alpha0=None, max_iter=5000, P=None, acceleration=True, cpp=True):
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
    t1 = time.time()
    n = winding_volume.N_grid
    nfp = winding_volume.plasma_boundary.nfp
    num_basis = winding_volume.n_functions
    N = n * num_basis
    alpha_opt = (np.random.rand(N) - 0.5) * 1e3  # set some initial guess

    # Initial guess with projected gradient descent must start inside the
    # feasible region (the area allowed by the equality constraints)
    if P is not None:
        alpha_opt = P.dot(alpha_opt)
    alpha_opt_T = alpha_opt.T

    B = winding_volume.B_matrix
    I = winding_volume.Itarget_matrix
    BT = B.T
    b = winding_volume.b_rhs
    b_I = winding_volume.Itarget_rhs
    BTb = BT @ b
    IT = I.T
    ITbI = IT * b_I 
    contig = np.ascontiguousarray
    if cpp:
        I = I.reshape(1, len(winding_volume.Itarget_matrix))
        IT = I.T
        b_I = b_I.reshape(1, 1)
        b = b.reshape(len(b), 1)
        ITbI = IT * b_I 
        BTb = BTb.reshape(len(BTb), 1)     
        alpha_opt = alpha_opt.reshape(len(alpha_opt), 1)
    IT = contig(IT)
    BT = contig(BT)
    I = contig(I)
    B = contig(B)
    t2 = time.time()

    print('Time to setup the algo = ', t2 - t1, ' s')

    ######## Temporary
    # B = np.zeros(winding_volume.B_matrix.shape)
    # I = np.zeros(I.shape)
    # b = np.zeros(b.shape)
    # b_I = np.zeros(b_I.shape)

    # BTB = BT @ B
    # ITI = IT @ I
    #L = np.linalg.svd(BTB + ITI + lam * np.eye(B.shape[1]), compute_uv=False)[0]
    # L = svd(BT @ B + IT @ I, compute_uv=False, check_finite=False, full_matrices=False)[0] + lam

    # upper bound on largest singular value from https://www.cs.yale.edu/homes/spielman/BAP/lect3.pdf
    t1 = time.time()
    if False:
        L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + IT @ I + np.eye(N) * lam, axis=0), axis=-1)
    else:
        L = 2e-12  # L = 1e-12 for N = 30, L = 2.4e-12 for N = 24, so not a bad guess

    #cond_num = np.linalg.cond(BTB + ITI + lam * np.eye(B.shape[1]))
    # cond_num = np.linalg.cond(BT @ B + IT @ I + lam * np.eye(B.shape[1]))
    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size = ', L, step_size)  # , cond_num)
    t2 = time.time()

    print('Time to setup step size = ', t2 - t1, ' s')

    # Pdense = contig(P.todense())
    # nonzero_inds = P.nonzero()
    # rows = nonzero_inds[0]
    # cols = nonzero_inds[1]
    # print(rows, cols)
    # # print(Pdense.shape, B.shape, I.shape, BTb.shape, ITbI.shape)
    # t1 = time.time()
    # alpha_opt_cpp = acc_prox_grad_descent(
    #     P, 
    #     contig(B), 
    #     contig(I), 
    #     contig(b), 
    #     contig(b_I), 
    #     contig(alpha_opt),
    #     lam, 
    #     step_size, 
    #     max_iter,
    # )
    # t2 = time.time()
    # print('Time to run algo in C++ = ', t2 - t1, ' s')
    # print('f_B from c++ calculation = ', 0.5 * nfp * np.linalg.norm(B @ alpha_opt_cpp - b, ord=2) ** 2)
    # print('f_I from c++ calculation = ', np.linalg.norm(I @ alpha_opt_cpp - b, ord=2) ** 2)
    # print('f_K from c++ calculation = ', np.linalg.norm(alpha_opt_cpp, ord=2) ** 2)

    f_B = []
    f_I = []
    f_K = []
    print_iter = 10
    t1 = time.time()
    if P is None:
        if acceleration:  # Nesterov acceleration 
            # first iteration do regular GD 
            alpha_opt_prev = alpha_opt
            step_size_i = step_size
            # f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2) ** 2)
            # f_I.append((I @ alpha_opt - b_I) ** 2)
            # f_K.append(np.linalg.norm(alpha_opt, ord=2) ** 2)
            # print('0', f_B[0], f_I[0], f_K[0])
            # alpha_opt = alpha_opt + step_size_i * (
            #     BTb + ITbI - BT @ (B @ alpha_opt) - IT * (I @ alpha_opt) - lam * alpha_opt
            # )
            for i in range(max_iter):
                if (i == 0) or ((i % print_iter) == 0.0):
                    f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2) ** 2)
                    if cpp: 
                        f_I.append(np.linalg.norm(I @ alpha_opt - b_I, ord=2) ** 2)
                    else:
                        f_I.append((I @ alpha_opt - b_I) ** 2)
                    f_K.append(np.linalg.norm(alpha_opt, ord=2) ** 2)
                    print(i, f_B[i // print_iter], f_I[i // print_iter], lam * f_K[i // print_iter], step_size_i)
                vi = alpha_opt + i / (i + 3) * (alpha_opt - alpha_opt_prev)
                alpha_opt_prev = alpha_opt
                if cpp:
                    vi = vi.reshape(len(vi), 1)
                    alpha_opt = vi + step_size_i * (BTb + ITbI - matmult(BT, matmult(B, vi)) - matmult(IT, matmult(I, vi)) - lam * vi)
                else:
                    alpha_opt = vi + step_size_i * (BTb + ITbI - BT @ (B @ vi) - IT * (I @ vi) - lam * vi)
                # print(vi.shape, alpha_opt.shape, sopp.matmult(IT, sopp.matmult(I, vi)).shape, sopp.matmult(BT, sopp.matmult(B, vi)).shape)
                # print(sopp.matmult(I, vi.reshape(len(vi), 1)).shape, I.shape, vi.shape)
                # print(ITbI.shape, BTb.shape)
                # print(alpha_opt.shape, np.ravel(vi).shape)
                step_size_i = (1 + np.sqrt(1 + 4 * step_size_i ** 2)) / 2.0
        else:
            for i in range(max_iter):
                alpha_opt = alpha_opt + step_size * (BTb + ITbI - BT @ (B @ alpha_opt) - IT * (I @ alpha_opt) - lam * alpha_opt)
                f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2) ** 2)
                f_I.append((I @ alpha_opt - b_I) ** 2)
                f_K.append(np.linalg.norm(alpha_opt, ord=2) ** 2)
                if (i % print_iter) == 0.0:
                    print(i, f_B[i // print_iter], f_I[i // print_iter], f_K[i // print_iter])
    else:
        if acceleration:  # Nesterov acceleration 
            alpha_opt_prev = alpha_opt
            step_size_i = step_size
            BTB_ITbI = BTb + ITbI
            for i in range(max_iter):
                vi = alpha_opt + i / (i + 3) * (alpha_opt - alpha_opt_prev)
                alpha_opt_prev = alpha_opt
                #alpha_opt = P.dot(vi + step_size_i * (BTb + ITbI - (BTB + ITI + lam * ident) @ vi))
                if cpp:
                    # alpha_opt = matmult_sparseA(
                    #     P, N,
                    #     contig(vi + step_size_i * (BTB_ITbI - matmult(BT, matmult(B, contig(vi))) - matmult(IT, matmult(I, contig(vi))) - lam * vi))
                    # )
                    alpha_opt = P.dot(
                        vi + step_size_i * (BTB_ITbI - matmult(BT, matmult(B, contig(vi))) - matmult(IT, matmult(I, contig(vi))) - lam * vi)
                    )
                    # alpha_opt = P.dot(vi + step_size_i * (BTB_ITbI - BT @ (B @ vi) - IT * (I @ vi) - lam * vi))
                else:
                    alpha_opt = P.dot(vi + step_size_i * (BTB_ITbI - BT @ (B @ vi) - IT * (I @ vi) - lam * vi))

                step_size_i = (1 + np.sqrt(1 + 4 * step_size_i ** 2)) / 2.0
                if ((i % print_iter) == 0.0 or i == max_iter - 1):
                    f_B.append(np.linalg.norm(np.ravel(B @ alpha_opt - b), ord=2) ** 2)
                    if cpp: 
                        f_I.append(np.linalg.norm(np.ravel(I @ alpha_opt - b_I)) ** 2)
                    else:
                        f_I.append((I @ alpha_opt - b_I) ** 2)
                    f_K.append(np.linalg.norm(np.ravel(alpha_opt)) ** 2) 
                    print(i, f_B[(i + 1) // print_iter], f_I[(i + 1) // print_iter], lam * f_K[(i + 1) // print_iter], step_size_i)  # , alpha_opt.shape)
        else:
            for i in range(max_iter):
                alpha_opt = P.dot(alpha_opt + step_size * (BTb + ITbI - BT @ (B @ alpha_opt) - IT * (I @ alpha_opt) - lam * alpha_opt))
                if (i % print_iter) == 0.0:
                    f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2) ** 2)
                    f_I.append((I @ alpha_opt - b_I) ** 2)
                    f_K.append(np.linalg.norm(alpha_opt, ord=2) ** 2)
                    print(i, f_B[i // print_iter], f_I[i // print_iter], f_K[i // print_iter])
    t2 = time.time()
    print('Time to run algo = ', t2 - t1, ' s')
    t1 = time.time()
    alpha_opt = np.ravel(alpha_opt)
    winding_volume.alphas = alpha_opt
    winding_volume.J = np.zeros((n, winding_volume.Phi.shape[2], 3))
    alphas = winding_volume.alphas.reshape(n, num_basis)
    for i in range(3):
        for j in range(winding_volume.Phi.shape[2]):
            for k in range(num_basis):
                winding_volume.J[:, j, i] += alphas[:, k] * winding_volume.Phi[k, :, j, i]
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return alpha_opt, 0.5 * 2 * np.array(f_B) * nfp, 0.5 * np.array(f_K), 0.5 * np.array(f_I)


def relax_and_split(winding_volume, lam=0.0, nu=1e20, alpha0=None, max_iter=5000, P=None, rs_max_iter=1, l0_threshold=0.0):
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
    n = winding_volume.N_grid
    nfp = winding_volume.plasma_boundary.nfp
    num_basis = winding_volume.n_functions
    N = n * num_basis
    if alpha0 is not None:
        alpha_opt = alpha0
    else:
        alpha_opt = (np.random.rand(N) - 0.5) * 1e3  
    if P is not None:
        alpha_opt = P.dot(alpha_opt)
    alpha_opt_T = alpha_opt.T

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
    ITbI = IT * b_I 
    BTb = BTb.reshape(len(BTb), 1)     
    alpha_opt = alpha_opt.reshape(len(alpha_opt), 1)
    IT = contig(IT)
    BT = contig(BT)
    I = contig(I)
    B = contig(B)
    t2 = time.time()

    lam_nu = (lam + 1.0 / nu)

    print('Time to setup the algo = ', t2 - t1, ' s')

    # upper bound on largest singular value from https://www.cs.yale.edu/homes/spielman/BAP/lect3.pdf
    t1 = time.time()
    if True:
        L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + IT @ I + np.eye(N) * lam_nu, axis=0), axis=-1)
    else:
        L = 2e-12  # L = 1e-12 for N = 30, L = 2.4e-12 for N = 24, so not a bad guess

    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size = ', L, step_size) 
    t2 = time.time()

    print('Time to setup step size = ', t2 - t1, ' s')

    f_B = []
    f_I = []
    f_K = []
    f_RS = []
    f_0 = []
    f_0w = []
    print_iter = 10
    t1 = time.time()
    # w_opt = np.copy(alpha_opt)  # (np.random.rand(len(alpha_opt), 1) - 0.5) * 1e5
    w_opt = prox_group_l0(np.copy(alpha_opt), l0_threshold, n, num_basis)
    for k in range(rs_max_iter):
        f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1) < l0_threshold))
        step_size_i = step_size
        alpha_opt_prev = alpha_opt
        BTB_ITbI_nuw = BTb + ITbI + w_opt / nu
        # print(w_opt, alpha_opt)
        for i in range(max_iter):
            vi = alpha_opt + i / (i + 3) * (alpha_opt - alpha_opt_prev)
            alpha_opt_prev = alpha_opt
            alpha_opt = P.dot(
                vi + step_size_i * (BTB_ITbI_nuw - matmult(BT, matmult(B, contig(vi))) - matmult(IT, matmult(I, contig(vi))) - lam_nu * vi)
            )
            step_size_i = (1 + np.sqrt(1 + 4 * step_size_i ** 2)) / 2.0
            # alpha_opt = P.dot(alpha_opt + step_size * (BTB_ITbI_nuw - BT @ (B @ alpha_opt) - IT * (I @ alpha_opt) - lam_nu * alpha_opt))

            # print metrics
            if ((i % print_iter) == 0.0 or i == max_iter - 1):
                f_B.append(np.linalg.norm(np.ravel(B @ alpha_opt - b), ord=2) ** 2)
                f_I.append(np.linalg.norm(np.ravel(I @ alpha_opt - b_I)) ** 2)
                f_K.append(np.linalg.norm(np.ravel(alpha_opt)) ** 2) 
                f_RS.append(np.linalg.norm(np.ravel(alpha_opt - w_opt) ** 2))
                f_0.append(np.count_nonzero(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1) < l0_threshold))
                print(i, f_B[k * max_iter // print_iter + (i + 1) // print_iter], 
                      f_I[k * max_iter // print_iter + (i + 1) // print_iter], 
                      lam * f_K[k * max_iter // print_iter + (i + 1) // print_iter], 
                      f_RS[k * max_iter // print_iter + (i + 1) // print_iter] / nu, 
                      f_0[k * max_iter // print_iter + (i + 1) // print_iter], 
                      f_0w[k], step_size_i)  

        # now do the prox step
        # print(np.mean(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
        print('Number of cells below the threshold = ', 
              np.count_nonzero(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1) < l0_threshold), 
              ' / ', n
              )
        print('Relax-and-split iteration = ', (k + 1), ' / ', rs_max_iter)
        w_opt = prox_group_l0(np.copy(alpha_opt), l0_threshold, n, num_basis)

    t2 = time.time()
    print('Time to run algo = ', t2 - t1, ' s')
    t1 = time.time()
    alpha_opt = np.ravel(alpha_opt)
    winding_volume.alphas = alpha_opt
    winding_volume.w = np.ravel(w_opt)
    winding_volume.J = np.zeros((n, winding_volume.Phi.shape[2], 3))
    alphas = winding_volume.alphas.reshape(n, num_basis)
    for i in range(3):
        for j in range(winding_volume.Phi.shape[2]):
            for k in range(num_basis):
                winding_volume.J[:, j, i] += alphas[:, k] * winding_volume.Phi[k, :, j, i]
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return alpha_opt, 0.5 * 2 * np.array(f_B) * nfp, 0.5 * np.array(f_K), 0.5 * np.array(f_I), 0.5 * np.array(f_RS) / nu, f_0


def relax_and_split_increasingl0(
        winding_volume, lam=0.0, nu=1e20, max_iter=5000,
        rs_max_iter=1, l0_thresholds=[0.0], print_iter=1):
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
    # alpha0 = winding_volume.alpha0
    alpha0 = None
    n = winding_volume.N_grid
    nfp = winding_volume.plasma_boundary.nfp
    num_basis = winding_volume.n_functions
    N = n * num_basis
    if alpha0 is not None:
        alpha_opt = alpha0
    else:
        alpha_opt = (np.random.rand(N) - 0.5) * 1e3  

    print(alpha_opt.shape, P.shape)
    C = winding_volume.C
    d = winding_volume.d
    print(C.dot(alpha_opt)[-1])
    print(np.linalg.norm(winding_volume.C.dot(alpha_opt) - winding_volume.d))
    # if P is not None:
    #     alpha_opt = P.dot(alpha_opt)
    print(C.dot(alpha_opt)[-1])
    print(np.linalg.norm(winding_volume.C.dot(alpha_opt) - winding_volume.d))

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
    t2 = time.time()

    print(np.linalg.norm(np.ravel(I @ alpha_opt - b_I)) ** 2)

    lam_nu = (lam + 1.0 / nu)

    print('Time to setup the algo = ', t2 - t1, ' s')

    # upper bound on largest singular value from https://www.cs.yale.edu/homes/spielman/BAP/lect3.pdf
    t1 = time.time()
    if True:
        # L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + IT @ I + np.eye(N) * lam_nu, axis=0), axis=-1)
        L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + np.eye(N) * lam_nu, axis=0), axis=-1)
    else:
        L = 2e-12  # L = 1e-12 for N = 30, L = 2.4e-12 for N = 24, so not a bad guess

    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size = ', L, step_size) 
    t2 = time.time()

    print('Time to setup step size = ', t2 - t1, ' s')

    f_B = []
    f_I = []
    f_K = []
    f_RS = []
    f_0 = []
    f_0w = []
    t1 = time.time()
    # print('C @ alpha  (initial) =', C @ alpha_opt)
    # w_opt = np.copy(alpha_opt)  # (np.random.rand(len(alpha_opt), 1) - 0.5) * 1e5
    w_opt = prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)
    for j, threshold in enumerate(l0_thresholds):
        # alpha0 = alpha_opt
        # w_opt = prox_group_l0(np.copy(alpha0), l0_threshold, n, num_basis)
        print('threshold iteration = ', j + 1, ' / ', len(l0_thresholds), ', threshold = ', threshold)
        for k in range(rs_max_iter):
            f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1) > threshold))
            step_size_i = step_size
            alpha_opt_prev = alpha_opt
            BTB_ITbI_nuw = BTb + w_opt / nu
            # BTB_ITbI_nuw = BTb + ITbI + w_opt / nu
            # print(w_opt, alpha_opt)
            for i in range(max_iter):
                # vi = contig(alpha_opt + i / (i + 3) * (alpha_opt - alpha_opt_prev))
                # alpha_opt_prev = alpha_opt
                # alpha_opt = P.dot(
                #     vi + step_size_i * (BTB_ITbI_nuw - matmult(BT, matmult(B, vi)) - lam_nu * vi)
                #     # vi + step_size_i * (BTB_ITbI_nuw - matmult(BT, matmult(B, vi)) - matmult(IT, matmult(I, vi)) - lam_nu * vi)
                # )
                # step_size_i = (1 + np.sqrt(1 + 4 * step_size_i ** 2)) / 2.0
                # alpha_opt = P.dot(alpha_opt + step_size * (BTB_ITbI_nuw - matmult(BT, matmult(B, contig(alpha_opt))) - matmult(IT, matmult(I, contig(alpha_opt))) - lam_nu * alpha_opt))
                # alpha_opt = P.dot(alpha_opt + step_size * (BTB_ITbI_nuw - BT @ (B @ alpha_opt) - IT * (I @ alpha_opt) - lam_nu * alpha_opt))
                alpha_opt = alpha_opt + step_size * P.dot(BTB_ITbI_nuw - BT @ (B @ alpha_opt) - lam_nu * alpha_opt)
                # print(C @ (step_size * P.dot(BTB_ITbI_nuw - BT @ (B @ alpha_opt) - lam_nu * alpha_opt)))

                # print metrics
                # if ((i % print_iter) == 0.0 or i == max_iter - 1):
                f_B.append(np.linalg.norm(np.ravel(B @ alpha_opt - b), ord=2) ** 2)
                f_I.append(np.linalg.norm(np.ravel(I @ alpha_opt - b_I)) ** 2)
                f_K.append(np.linalg.norm(np.ravel(alpha_opt)) ** 2) 
                f_RS.append(np.linalg.norm(np.ravel(alpha_opt - w_opt) ** 2))
                f_0.append(np.count_nonzero(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1) > threshold))
                ind = i  # (j * max_iter * rs_max_iter + k * max_iter + (i + 1)) # // print_iter
                print(i, f_B[ind], 
                      f_I[ind], 
                      lam * f_K[ind], 
                      f_RS[ind] / nu, 
                      f_0[ind], 
                      f_0w[j * rs_max_iter + k], 
                      step_size_i,
                      # winding_volume.C @ alpha_opt
                      )  

            # now do the prox step
            print(np.mean(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1)))
            print('Number of cells below the threshold = ', 
                  np.count_nonzero(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1) < threshold), 
                  ' / ', n
                  )
            print('Relax-and-split iteration = ', (k + 1), ' / ', rs_max_iter)
            w_opt = prox_group_l0(np.copy(alpha_opt), threshold, n, num_basis)

    t2 = time.time()
    print('Time to run algo = ', t2 - t1, ' s')
    t1 = time.time()
    alpha_opt = np.ravel(alpha_opt)
    winding_volume.alphas = alpha_opt
    winding_volume.w = np.ravel(w_opt)
    winding_volume.J = np.zeros((n, winding_volume.Phi.shape[2], 3))
    alphas = winding_volume.alphas.reshape(n, num_basis)
    for i in range(3):
        for j in range(winding_volume.Phi.shape[2]):
            for k in range(num_basis):
                winding_volume.J[:, j, i] += alphas[:, k] * winding_volume.Phi[k, :, j, i]
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return alpha_opt, 0.5 * 2 * np.array(f_B) * nfp, 0.5 * np.array(f_K), 0.5 * np.array(f_I), 0.5 * np.array(f_RS) / nu, f_0


def relax_and_split_analytic(
        winding_volume, lam=0.0, nu=1e20, max_iter=5000,
        rs_max_iter=1, l0_thresholds=[0.0], print_iter=10):
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
    alpha0 = winding_volume.alpha0
    n = winding_volume.N_grid
    nfp = winding_volume.plasma_boundary.nfp
    num_basis = winding_volume.n_functions
    N = n * num_basis
    if alpha0 is not None:
        alpha_opt = alpha0
    else:
        alpha_opt = (np.random.rand(N) - 0.5) * 1e3  

    # print(alpha_opt.shape, P.shape)
    C = winding_volume.C
    CT = C.T
    d = winding_volume.d
    # print(C.dot(alpha_opt)[-1])
    # print(np.linalg.norm(winding_volume.C.dot(alpha_opt) - winding_volume.d))
    # if P is not None:
    #     alpha_opt = P.dot(alpha_opt)
    print(C.dot(alpha_opt)[-1])
    print(np.linalg.norm(winding_volume.C.dot(alpha_opt) - winding_volume.d))

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
    t2 = time.time()

    print(np.linalg.norm(np.ravel(I @ alpha_opt - b_I)) ** 2)

    lam_nu = (lam + 1.0 / nu)

    print('Time to setup the algo = ', t2 - t1, ' s')

    # upper bound on largest singular value from https://www.cs.yale.edu/homes/spielman/BAP/lect3.pdf
    t1 = time.time()
    if True:
        # L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + IT @ I + np.eye(N) * lam_nu, axis=0), axis=-1)
        L = np.sqrt(N) * np.max(np.linalg.norm(BT @ B + np.eye(N) * lam_nu, axis=0), axis=-1)
    else:
        L = 2e-12  # L = 1e-12 for N = 30, L = 2.4e-12 for N = 24, so not a bad guess

    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size = ', L, step_size) 
    t2 = time.time()

    print('Time to setup step size = ', t2 - t1, ' s')

    H = np.linalg.inv(BT @ B + np.eye(N) * lam_nu)
    CHCT = C @ H @ CT  # Tends to be not full rank!!!!
    S = np.linalg.svd(CHCT, compute_uv=False)
    S2 = np.linalg.svd(H, compute_uv=False)
    plt.figure(22)
    plt.semilogy(S)
    plt.semilogy(S2)
    CHCT_inv = np.linalg.pinv(CHCT, rcond=1e-30)
    analytic_mat = H @ (np.eye(N) - CT @ CHCT_inv @ C @ H)
    analytic_vec = (H @ CT @ CHCT_inv @ d).reshape(N, 1)

    f_B = []
    f_I = []
    f_K = []
    f_RS = []
    f_0 = []
    f_0w = []
    t1 = time.time()
    # print('C @ alpha  (initial) =', C @ alpha_opt)
    # w_opt = np.copy(alpha_opt)  # (np.random.rand(len(alpha_opt), 1) - 0.5) * 1e5
    w_opt = prox_group_l0(alpha_opt, l0_thresholds[0], n, num_basis)
    for j, threshold in enumerate(l0_thresholds):
        # alpha0 = alpha_opt
        # w_opt = prox_group_l0(np.copy(alpha0), l0_threshold, n, num_basis)
        print('threshold iteration = ', j + 1, ' / ', len(l0_thresholds), ', threshold = ', threshold)
        for k in range(rs_max_iter):
            if (k % print_iter) == 0.0 or k == 0:
                ind = (j * rs_max_iter + k) // print_iter
                f_B.append(np.linalg.norm(np.ravel(B @ alpha_opt - b), ord=2) ** 2)
                f_I.append(np.linalg.norm(np.ravel(I @ alpha_opt - b_I)) ** 2)
                f_K.append(np.linalg.norm(np.ravel(alpha_opt)) ** 2) 
                f_RS.append(np.linalg.norm(np.ravel(alpha_opt - w_opt) ** 2))
                f_0.append(np.count_nonzero(np.linalg.norm(alpha_opt.reshape(n, num_basis), axis=-1) > threshold))
                f_0w.append(np.count_nonzero(np.linalg.norm(w_opt.reshape(n, num_basis), axis=-1) > threshold))
                print(k, f_B[ind], 
                      f_I[ind], 
                      lam * f_K[ind], 
                      f_RS[ind] / nu, 
                      f_0[ind], 
                      f_0w[ind], 
                      # winding_volume.C @ alpha_opt
                      )  
            BTB_nuw = BTb + w_opt / nu
            alpha_opt = analytic_mat.dot(BTB_nuw) + analytic_vec
            w_opt = prox_group_l0(alpha_opt, threshold, n, num_basis)

    t2 = time.time()
    print('Time to run algo = ', t2 - t1, ' s')
    t1 = time.time()
    alpha_opt = np.ravel(alpha_opt)
    # print('alpha_opt = ', alpha_opt)
    winding_volume.alphas = alpha_opt
    winding_volume.w = np.ravel(w_opt)
    winding_volume.J = np.zeros((n, winding_volume.Phi.shape[2], 3))
    alphas = winding_volume.alphas.reshape(n, num_basis)
    for i in range(3):
        for j in range(winding_volume.Phi.shape[2]):
            for k in range(num_basis):
                winding_volume.J[:, j, i] += alphas[:, k] * winding_volume.Phi[k, :, j, i]
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return alpha_opt, 0.5 * 2 * np.array(f_B) * nfp, 0.5 * np.array(f_K), 0.5 * np.array(f_I), 0.5 * np.array(f_RS) / nu, f_0
