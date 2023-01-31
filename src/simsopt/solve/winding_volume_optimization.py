import numpy as np
import simsoptpp as sopp
import time
# from scipy.linalg import svd

__all__ = ['projected_gradient_descent_Tikhonov']


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
    alpha_opt = (np.random.rand(n * num_basis) - 0.5) * 1e3  # set some initial guess

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
    if cpp:
        I = I.reshape(1, len(winding_volume.Itarget_matrix))
        IT = I.T
        b_I = b_I.reshape(1, 1)
        b = b.reshape(len(b), 1)
        ITbI = IT * b_I 
        BTb = BTb.reshape(len(BTb), 1)     
        alpha_opt = alpha_opt.reshape(len(alpha_opt), 1)

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
    L = np.sqrt(B.shape[1]) * np.max(np.linalg.norm(BT @ B + IT @ I, axis=0), axis=-1)

    #cond_num = np.linalg.cond(BTB + ITI + lam * np.eye(B.shape[1]))
    # cond_num = np.linalg.cond(BT @ B + IT @ I + lam * np.eye(B.shape[1]))
    step_size = 1.0 / L  # largest step size suitable for gradient descent
    print('L, step_size = ', L, step_size)  # , cond_num)
    t2 = time.time()

    print('Time to setup the algo = ', t2 - t1, ' s')
    contig = np.ascontiguousarray
    # Pdense = P.todense()
    # nonzero_inds = P.nonzero()
    # rows = nonzero_inds[0]
    # cols = nonzero_inds[1]
    # print(rows, cols)
    # print(Pdense.shape, B.shape, I.shape, BTb.shape, ITbI.shape)
    # t1 = time.time()
    # alpha_opt_cpp = sopp.acc_prox_grad_descent(
    #     contig(Pdense), 
    #     contig(B), 
    #     contig(I), 
    #     contig(b), 
    #     contig(b_I), 
    #     contig(rows), 
    #     contig(cols), 
    #     contig(alpha_opt),
    #     lam, 
    #     step_size, 
    #     max_iter,
    #     P.nnz
    # )
    # t2 = time.time()
    # print('Time to run algo in C++ = ', t2 - t1, ' s')
    # print('f_B from c++ calculation = ', 0.5 * nfp * np.linalg.norm(B @ alpha_opt_cpp - b, ord=2) ** 2)
    # print('f_I from c++ calculation = ', np.linalg.norm(I @ alpha_opt_cpp - b, ord=2) ** 2)
    # print('f_K from c++ calculation = ', np.linalg.norm(alpha_opt_cpp, ord=2) ** 2)

    f_B = []
    f_I = []
    f_K = []
    print_iter = 100
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
                    print(i, f_B[i // print_iter], f_I[i // print_iter], f_K[i // print_iter], step_size_i)
                vi = alpha_opt + i / (i + 3) * (alpha_opt - alpha_opt_prev)
                alpha_opt_prev = alpha_opt
                if cpp:
                    vi = vi.reshape(len(vi), 1)
                    alpha_opt = vi + step_size_i * (BTb + ITbI - sopp.matmult(BT, sopp.matmult(B, vi)) - sopp.matmult(IT, sopp.matmult(I, vi)))
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
            for i in range(max_iter):
                if ((i == 0) or (i % print_iter) == 0.0):
                    f_B.append(np.linalg.norm(B @ alpha_opt - b, ord=2) ** 2)
                    if cpp: 
                        f_I.append(np.linalg.norm(I @ alpha_opt - b_I, ord=2) ** 2)
                    else:
                        f_I.append((I @ alpha_opt - b_I) ** 2)
                    f_K.append(np.linalg.norm(alpha_opt, ord=2) ** 2) 
                    print(i, f_B[i // print_iter], f_I[i // print_iter], f_K[i // print_iter], alpha_opt.shape)
                vi = alpha_opt + i / (i + 3) * (alpha_opt - alpha_opt_prev)
                alpha_opt_prev = alpha_opt
                #alpha_opt = P.dot(vi + step_size_i * (BTb + ITbI - (BTB + ITI + lam * ident) @ vi))
                if cpp:
                    vi = vi.reshape(len(vi), 1)
                    alpha_opt = P.dot(vi + step_size_i * (BTb + ITbI - sopp.matmult(BT, sopp.matmult(B, vi)) - sopp.matmult(IT, sopp.matmult(I, vi))))
                else:
                    alpha_opt = P.dot(vi + step_size_i * (BTb + ITbI - BT @ (B @ vi) - IT * (I @ vi) - lam * vi))

                step_size_i = (1 + np.sqrt(1 + 4 * step_size_i ** 2)) / 2.0
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
    winding_volume.alphas = alpha_opt 
    winding_volume.J = np.zeros((n, winding_volume.Phi.shape[2], 3))
    alphas = alpha_opt.reshape(n, num_basis)
    for i in range(3):
        for j in range(winding_volume.Phi.shape[2]):
            for k in range(num_basis):
                winding_volume.J[:, j, i] += alphas[:, k] * winding_volume.Phi[k, :, j, i]
    t2 = time.time()
    print('Time to compute J = ', t2 - t1, ' s')
    return alpha_opt, 0.5 * np.array(f_B) * nfp, 0.5 * np.array(f_K), 0.5 * np.array(f_I)
