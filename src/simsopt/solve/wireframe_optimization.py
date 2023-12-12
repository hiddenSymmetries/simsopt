"""
Contains functions for optimizing currents within wireframe structures.
"""

import numpy as np
import scipy

__all__ = ['regularized_constrained_least_squares']

def regularized_constrained_least_squares(A, c, T, B, d):
    """
    Solves a linear least squares problem with Tikhonov regularization
    subject to linear equality constraints on the variables.

    In other words, minimizes:
        0.5 * ((A*x - c)**2 + (T*x)**2

    such that: 
        B*x = d

    Here, A is the design matrix, c is the target vector, T is the 
    regularization matrix (normally diagonal), and B and d contain the 
    coefficients and constants of the constraint equations, respectively.

    Parameters
    ----------
        A: array with dimensions m*n
            Design matrix
        c: array with dimension m
            Target vector
        T: scalar, array with dimension n, or array with dimension n*n
            Regularization matrix
        B: array with dimension p*n
            Coefficients of the solution vector elements in each of the p
            constraint equations. Must be full rank.
        d: array with dimension p
            Constants appearing on the right-hand side of the constraint
            equations, i.e. B*x = d.

    Returns
    -------
        x: array with dimension n
            Solution to the least-squares problem
    """

    # Recast inputs as Numpy arrays
    Amat = np.array(A)
    cvec = np.array(c).reshape((-1,1))
    Btra = np.array(B).T # Transpose will be used for the calculations
    dvec = np.array(d).reshape((-1,1))

    # Check the inputs
    m, n = Amat.shape
    if cvec.shape[0] != m:
        raise ValueError('Number of elements in c must match rows in A')
    n_B, p = Btra.shape
    if n_B != n:
        raise ValueError('A and B must have the same number of columns')
    if dvec.shape[0] != p:
        raise ValueError('Number of elements in p must match rows in B')

    if np.isscalar(T):
        Tmat = T*np.eye(n)
    else:
        Tmat = np.squeeze(T)
        if len(Tmat.shape) == 1:
            if Tmat.shape[0] != n:
                raise ValueError('Number of elements in vector-form T ' \
                                 'must match columns in A')
            Tmat = np.diag(Tmat)
        elif len(Tmat.shape) == 2:
            if Tmat.shape[0] != n or Tmat.shape[1] != n:
                raise ValueError('Number of rows and columns in matrix-form T '\
                                 'must both equal number of columns in A')
        else:
            raise ValueError('T must be a scalar, 1d array, or 2d array')
            
    # Compute the QR factorization of the transpose of the constraint matrix
    Qfull, Rtall = scipy.linalg.qr(Btra)
    Q1mat = Qfull[:,:p]  # Orthonormal vectors in the constrained subspace
    Q2mat = Qfull[:,p:]  # Orthonormal vectors in the free subspace
    Rmat = Rtall[:p,:]

    # SOLVE: Rmat.T * uvec = dvec
    # uvec = coefficients for basis vectors from constrained subspace
    uvec = scipy.linalg.solve_triangular(Rmat.T, dvec, lower=True)

    # Form the LHS of the least-squares problem
    AQ2mat = np.matmul(Amat, Q2mat)
    TQ2mat = np.matmul(Tmat, Q2mat)
    LHS = np.matmul(AQ2mat.T, AQ2mat) + np.matmul(TQ2mat.T, TQ2mat)

    # Form the RHS of the least-squares problem
    AQ1mat = np.matmul(Amat, Q1mat)   
    TQ1mat = np.matmul(Tmat, Q1mat)
    AQ1uvec = np.matmul(AQ1mat, uvec)
    TQ1uvec = np.matmul(TQ1mat, uvec)
    AQ2cvec = np.matmul(AQ2mat.T, cvec)
    RHS = AQ2cvec - np.matmul(AQ2mat.T, AQ1uvec) - np.matmul(TQ2mat.T, TQ1uvec)

    # SOLVE: least-squares equation for the "v" vector
    # vvec = coefficients for basis vectors in the unconstrained subspace
    vvec = scipy.linalg.lstsq(LHS, RHS)[0]

    # Transform from "Q" basis back to the basis of individual segment currents
    return np.matmul(Qfull, np.concatenate((uvec, vvec), axis=0))


