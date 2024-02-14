"""
Contains functions for optimizing currents within wireframe structures.
"""

import numpy as np
import scipy
import time
from simsopt.geo.surface import Surface
from simsopt.geo.wireframe import ToroidalWireframe
from simsopt.field.magneticfield import MagneticField
from simsopt.field.wireframefield import WireframeField

__all__ = ['rcls_wireframe', 'regularized_constrained_least_squares']

def rcls_wireframe(wframe, reg_lambda, assume_no_crossings=False, \
                   surf_plas=None, ext_field=None, area_weighted=True, \
                   bn_plas_curr=None, Amat=None, cvec=None, verbose=True):
    """
    Performs a regularized constrained least squares optimization of the 
    segment currents in a wireframe class instance.

    The `currents` field of the wireframe class instance will be updated with
    the solution.

    There are two different modes of operation depending on which optional
    input parameters are set:

      (1) The user supplies a target plasma boundary and, if needed, a constant
          external magnetic field; in this case, the function calculates the
          inductance matrix and required normal field at the test points
          in preparation for performing the least-squares solve. In this mode,
          the parameter `surf_plas` must be supplied and, if relevant, 
          `ext_field`.

      (2) The user supplies a pre-computed inductance matrix and vector with
          the required normal field on the plasma boundary, in which case it
          is not necessary to perform a field calculation before the least-
          squares solve. In this mode, the parameters `Amat` and `cvec` must
          be supplied.

    IMPORTANT: the parameter `assume_no_crossings` MUST be set to True if 
    the wireframe is constrained to allow no crossing currents; otherwise,
    the optimization will not work properly!

    ----------
        wframe: instance of the ToroidalWireframe class
            Wireframe whose segment currents are to be optimized
        reg_lambda: scalar, 1d array, or 2d array
            Scalar, array, or matrix for regularization. If a 1d array, must
            have the same number of elements as wframe.nSegments. If a 2d 
            array, both dimensions must be equal to wframe.nSegments.
        assume_no_crossings: boolean (optional)
            If true, will assume that the wireframe is constrained such that
            its free segments form single-track loops with no forks or 
            crossings.  Default is False. 
        surf_plas: Surface class instance (optional)
            Surface of the target plasma, on which test points are placed for
            evaluation of the normal field. If supplied, a magnetic field
            calculation will be performed to determine the inductance matrix
            and target normal field vector prior to performing the least-
            squares solve as described in mode (1) above.
        ext_field: MagneticField class instance (optional)
            Constant external field assumed to be present in addition to the
            field produced by the wireframe. Used to calculate the target
            normal field vector in mode (1) as described above.
        bn_plas_curr: 1d double array (optional)
            Contributions of internal plasma currents to the normal field on
            the plasma boundary. Must have dimensions (nTestPoints, 1).   
        area_weighted: boolean (optional)
            Determines whether the inductance matrix and target normal field
            vector elements are weighted by the square root of the area 
            ascribed to each test point on the plasma boundary. If true, the
            weightings will be applied and therefore the optimization will 
            minimize the square integral of the normal field on the plasma 
            boundary. If false, the weightings will NOT be applied, and the
            optimization will minimize the sum of squares of the normal field
            at each test point. 
        Amat: 2d double array (optional)
            Inductance matrix relating normal field at test points on the
            plasma boundary to currents in each segment. This can be supplied
            along with `cvec` to skip the field calculation as describe in mode
            (2) above. Must have dimensions (nTestPoints, wFrame.nSegments), 
            where nTestPoints is the number of test points on the plasma 
            boundary. 
        cvec: double array (optional)
            Vector giving the target values of the normal field at each test
            point on the plasma boundary. This can be supplied along with `Amat`
            to skip the field calculation. Must have dimensions
            (nTestPoints, 1).
        verbose: boolean (optional)
            If true, will print progress to screen with durations of certain
            steps

    Returns
    -------
        results: dictionary with the following entries:
            x: 1d double array 
                Array of currents in each segment according to the solution.
                The elements of wframe.currents will be set to these values.
            Amat: 2d double array
                Inductance matrix used for optimization, area weighted if
                requested
            cvec: 1d double array (column vector)
                Target values of the normal field on the plasma boundary,
                area weighted if requested
            wframe_field: WireframeField class instance
                Magnetic field produced by the wireframe
    """

    if not isinstance(wframe, ToroidalWireframe):
        raise ValueError('Input `wframe` must be a ToroidalWireframe class ' \
                         + 'instance')

    if verbose:
        print('  Running Regularized Constrained Least Squares for a ' \
              + 'ToroidalWireframe')

    # Mode 1: plasma boundary supplied; field calculation necessary
    if surf_plas is not None:

        if not isinstance(surf_plas, Surface):
            raise ValueError('Input `surf_plas` must be a Surface class ' \
                             + 'instance')

        elif Amat is not None or cvec is not None:
            raise ValueError('Inputs `Amat` and `cvec` must not be supplied ' \
                             + 'if `surf_plas` is given')
    
        # Calculate the normal vectors for the surface
        n = surf_plas.normal()
        absn = np.linalg.norm(n, axis=2)[:,:,None]
        unitn = n * (1./absn)
        sqrt_area = np.sqrt(absn.reshape((-1,1))/float(absn.size))

        if area_weighted:
            area_weight = sqrt_area
        else:
            area_weight = np.ones(sqrt_area.shape)

        # Calculate the inductance matrix for the wireframe
        if verbose:
            print('    Calculating the wireframe field and inductance matrix')
        t0 = time.time()
        mf_wf = WireframeField(wframe)
        mf_wf.set_points(surf_plas.gamma().reshape((-1,3)))
        A = mf_wf.dBnormal_by_dsegmentcurrents_matrix(surf_plas, \
                area_weighted=area_weighted)
        t1 = time.time()
        if verbose:
            print('        Field and inductance matrix calc took %.2f seconds' \
                  % (t1 - t0))

        # Calculate the target normal field to cancel contributions from the
        # external field
        if ext_field is not None:
            if not isinstance(ext_field, MagneticField):
                raise ValueError('Input `ext_field` must be a MagneticField ' \
                                 + 'class instance')

            if verbose:
                print('    Determining contribution from external field')

            # Save the test points from the external field as input
            orig_points = ext_field.get_points_cart_ref()

            ext_field.set_points(surf_plas.gamma().reshape((-1,3)))
            B_ext = ext_field.B().reshape(n.shape)
            B_ext_norm = np.sum(B_ext * unitn, axis=2)[:,:,None]
            ext_norm_target = -B_ext_norm.reshape((-1,1))*area_weight
            
            # Restore the original test points
            ext_field.set_points(orig_points)

        else:

            ext_norm_target = 0*area_weight

        # Calculate the target normal field to cancel contributions from 
        # plasma currents
        if bn_plas_curr is not None:

            if bn_plas_curr.shape != (len(n), 1):
                raise ValueError('Input `bn_plas_curr` must have shape ' \
                    + '(nTestPoints,1), where nTestPoints\n is the number of ' \
                    + 'test points on the plasma boundary')
                  
            if verbose:
                print('    Adding contribution from plasma current')

            bn_plas_target = -bn_plas_curr*area_weight

        else:

            bn_plas_target = 0*area_weight

        # Calculate the target bnormal on the plasma boundary
        c = ext_norm_target + bn_plas_target

    # Mode 2: Inductance matrix and target bnormal vector supplied
    elif Amat is not None and cvec is not None:

        if surf_plas is not None or ext_field is not None or \
           bn_plas_curr is not None:
            raise ValueError('If `Amat` and `cvec` are provided, the ' \
                + 'following parameters must not be provided: \n' \
                + '    `surf_plas`, `ext_field`, `bn_plas_curr`')

        if verbose:
            print('    Using pre-calculated inductance and target field')

        # Check Amat and cvec inputs
        c = np.array(cvec).reshape((-1,1))
        nTestPoints = len(c)
        A = np.array(Amat)
        if np.shape(A) != (nTestPoints, wframe.nSegments):
            raise ValueError('Input `Amat` has inconsistent dimensions with ' \
                             'input `cvec` and/or `wframe`')

    # Obtain the constraint matrices
    if verbose:
        print('    Obtaining constraint matrices')
    B, d, free_segs = \
        wframe.constraint_matrices(assume_no_crossings=assume_no_crossings, \
                                   remove_constrained_segments=True)

    if np.shape(B)[0] >= len(free_segs):
        raise ValueError('Least-squares problem has as many or more ' \
            + 'constraints than degrees of freedom.\n ' \
            + 'Wireframe may have redundant constraints or the problem is '\
            + 'over-constrained.')

    # Trim constrained segments out of the A and T matrices
    Afree = A[:,free_segs]
    if np.isscalar(reg_lambda):
        Tfree = reg_lambda
    else:
        T = np.array(reg_lambda)
        if T.ndim == 1:
            Tfree = T[free_segs]
        elif T.ndim == 2:
            Tfree = T[free_segs,free_segs]
        else:
            raise ValueError('Input reg_lambda must be a scalar, 1d array, ' \
                             + 'or 2d array')

    # Solve the least-squares problem
    if verbose:
        print('    Solving the regularized constrained least-squares problem')
    t0 = time.time()
    xfree = regularized_constrained_least_squares(Afree, c, Tfree, B, d)
    t1 = time.time()
    if verbose:
        print('        Solver took %.2f seconds' % (t1 - t0))

    # Set wireframe currents to the solution vector
    wframe.currents[:] = 0
    wframe.currents[free_segs] = xfree.reshape((-1))[:]

    # Prepare the results to output
    x = np.zeros(wframe.nSegments)
    x[free_segs] = xfree.reshape((-1))[:]
    mf_wf = WireframeField(wframe)  # regenerate with solution currents
    results = {'x': x, \
               'Amat': A, \
               'cvec': c, \
               'wframe_field': mf_wf}

    return results
   

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


