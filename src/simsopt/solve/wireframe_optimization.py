"""
Contains functions for optimizing currents within wireframe structures.
"""

import numpy as np
import scipy
import time
import simsoptpp as sopp
from simsopt.geo import Surface, ToroidalWireframe
from simsopt.field.magneticfield import MagneticField
from simsopt.field.wireframefield import WireframeField

__all__ = ['optimize_wireframe', 'bnorm_obj_matrices',
           'rcls_wireframe', 'gsco_wireframe', 'get_gsco_iteration',
           'regularized_constrained_least_squares']


def optimize_wireframe(wframe, algorithm, params,
                       surf_plas=None, ext_field=None, area_weighted=True,
                       bnorm_target=None, Amat=None, bvec=None, verbose=True):
    """
    Optimizes the segment currents in a wireframe class instance.

    The ``currents`` field of the wireframe class instance will be updated with
    the solution.

    There are two different modes of operation depending on which optional
    input parameters are set:

      (1) The user supplies a target plasma boundary and, if needed, a constant
          external magnetic field; in this case, the function calculates the
          normal field matrix and required normal field at the test points
          in preparation for performing the least-squares solve. In this mode,
          the parameter ``surf_plas`` must be supplied and, if relevant, 
          ``ext_field``.

      (2) The user supplies a pre-computed normal field matrix and vector with
          the required normal field on the plasma boundary, in which case it
          is not necessary to perform a field calculation before the least-
          squares solve. In this mode, the parameters ``Amat`` and ``bvec`` must
          be supplied.

    IMPORTANT: for Regularized Constrained Least Squares ('rcls') optimizations,
    the parameter ``assume_no_crossings`` MUST be set to True if the wireframe 
    is constrained to allow no crossing currents; otherwise, the optimization 
    will not work properly!

    Parameters
    ----------
        wframe: instance of the ToroidalWireframe class
            Wireframe whose segment currents are to be optimized
        algorithm: string
            Optimization algorithm to use. Options are

            * ``"rcls"``: Regularized Constrained Least Squares

            * ``"gsco"``: Greedy Stellarator Coil Optimization
        params: dictionary
            As specified in the lists below under `Parameters for RCLS 
            optimizations` or `Parameters for GSCO optimizations`
        surf_plas: Surface class instance (optional)
            Surface of the target plasma, on which test points are placed for
            evaluation of the normal field. If supplied, a magnetic field
            calculation will be performed to determine the normal field matrix
            and target normal field vector prior to performing the least-
            squares solve as described in mode (1) above.
        ext_field: MagneticField class instance (optional)
            Constant external field assumed to be present in addition to the
            field produced by the wireframe. Used to calculate the target
            normal field vector in mode (1) as described above.
        bnorm_target: double array (optional)
            Target value of the normal field on the plasma boundary to be
            produced by the combination of the wireframe and ``ext_field``.
            Zero by default. Test points on the plasma boundary corresponding 
            to the elements of ``bnorm_target`` must agree with the test points 
            of ``surf_plas``.   
        area_weighted: boolean (optional)
            Determines whether the normal field matrix and target normal field
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
            along with ``bvec`` to skip the field calculation as describe in 
            mode (2) above. Must have dimensions (n_test_points, 
            wFrame.n_segments), where n_test_points is the number of test points
            on the plasma boundary. 
        bvec: double array (optional)
            Vector giving the target values of the normal field at each test
            point on the plasma boundary. This can be supplied along with 
            ``Amat`` to skip the field calculation. Must have dimensions
            (n_test_points, 1).
        verbose: boolean (optional)
            If true, will print progress to screen with durations of certain
            steps

    Returns
    -------
        results: dictionary
            As specified below under `Results dictionary`

    Parameters for RCLS optimizations:

    *   ``reg_W``: (*scalar, 1d array, or 2d array*) -
        Scalar, array, or matrix for regularization. If a 1d array, must
        have the same number of elements as wframe.n_segments. If a 2d 
        array, both dimensions must be equal to wframe.n_segments.
    *   ``assume_no_crossings``: (*boolean (optional)*) -
        If true, will assume that the wireframe is constrained such that
        its free segments form single-track loops with no forks or 
        crossings.  Default is False. 

    Parameters for GSCO optimizations:

    *   ``lambda_S``: (*double*) -
        Weighting factor for the objective function proportional to the
        number of active segments
    *   ``max_iter``: (*integer*) -
        Maximum number of iterations to perform
    *   ``print_inteval``: (*integer*) -
        Number of iterations between subsequent prints of progress to screen
    *   ``no_crossing``: (*boolean (optional)*) -
        If true, the solution will forbid currents from crossing within
        the wireframe; default is false
    *   ``match_current``: (*boolean (optional)*) -
        If true, added loops of current will match the current of the 
        loop(s) adjacent to where they are added; default is false
    *   ``default_current``: (*double (optional)*) -
        Loop current to add during each iteration to empty loops or to any
        loop if not matching existing current; default is 0
    *   ``max_current``: (*double (optional)*) -
        Maximum magnitude for the current in each segment; default is 
        infinity
    *   ``max_loop_count``: (*integer (optional)*) -
        If nonzero, sets the maximum number of current increments to add
        to a given loop in the wireframe (default is zero)
    *   ``x_init``: (*double array (optional)*) -
        Initial current values to impose on the wireframe segments; will
        overwrite wframe.currents if used
    *   ``loop_count_init``: (*integer array (optional)*) -
        Signed number of loops of current added to each loop in the 
        wireframe prior to the optimization (optimization will
        add to these numbers); zero by default

    Results dictionary:

    *   ``x``: (*1d double array*) -
        Array of currents in each segment according to the solution.
        The elements of wframe.currents will be set to these values.
    *   ``Amat``: (*2d double array*) -
        Inductance matrix used for optimization, area weighted if
        requested
    *   ``bvec``: (*1d double array (column vector)*) -
        Target values of the normal field on the plasma boundary,
        area weighted if requested
    *   ``wframe_field``: (*WireframeField class instance*) -
        Magnetic field produced by the wireframe
    *   ``f_B``: (*double*) -
        Values of the sub-objective function f_B
    *   ``f``: (*double*) -
        Values of the total objective function f

    For RCLS optimizations only:

    *   ``f_R``: (*double*) -
        Value of the sub-objective function f_R

    For GSCO optimizations only:

    *   ``loop_count``: (*integer array (1d columnn vector)*) -
        Signed number of current loops added to each loop in the 
        wireframe
    *   ``iter_hist``: (*integer array*) -
        Array with the iteration numbers of the data recorded in the 
        history arrays. The first index is 0, corresponding to the 
        initial guess `x_init`; the last is the final iteration. 
    *   ``curr_hist``: (*double array*) -
        Array with the signed loop current added at each iteration, 
        taken to be zero for the initial guess (iteration zero)
    *   ``loop_hist``: (*integer array*) -
        Array with the index of the loop to which current was added
        at each iteration, taken to be zero for the initial guess
        (iteration zero)
    *   ``f_B_hist``: (*1d double array (column vector)*) -
        Array with values of the f_B objective function at each 
        iteration
    *   ``f_S_hist``: (*1d double array (column vector)*) -
        Array with values of the f_S objective function at each 
        iteration
    *   ``f_hist``: (*1d double array (column vector)*) -
        Array with values of the f_S objective function at each
        iteration
    *   ``x_init``: (*1d double array (column vector)*) -
        Copy of the initial guess provided to the optimizer
    *   ``f_S``: (*double*) -
        Values of the sub-objective function f_S
    """

    if not isinstance(wframe, ToroidalWireframe):
        raise ValueError('Input `wframe` must be a ToroidalWireframe class '
                         + 'instance')

    if verbose:
        print('  Optimization of the segment currents in a '
              + 'ToroidalWireframe')

    # Mode 1: plasma boundary supplied; field calculation necessary
    if surf_plas is not None:

        if Amat is not None or bvec is not None:
            raise ValueError('Inputs `Amat` and `bvec` must not be supplied '
                             + 'if `surf_plas` is given')

        # Calculate the normal field matrix (A) and target field vector (c)
        A, b = bnorm_obj_matrices(wframe, surf_plas, ext_field=ext_field,
                                  area_weighted=area_weighted, bnorm_target=bnorm_target,
                                  verbose=verbose)

    # Mode 2: Inductance matrix and target bnormal vector supplied
    elif Amat is not None and bvec is not None:

        if surf_plas is not None or ext_field is not None or \
           bnorm_target is not None:
            raise ValueError('If `Amat` and `bvec` are provided, the '
                             + 'following parameters must not be provided: \n'
                             + '    `surf_plas`, `ext_field`, `bnorm_target`')

        if verbose:
            print('    Using pre-calculated normal field matrix '
                  + 'and target field')

        # Check Amat and bvec inputs
        b = np.array(bvec).reshape((-1, 1))
        n_test_points = len(b)
        A = np.array(Amat)
        if np.shape(A) != (n_test_points, wframe.n_segments):
            raise ValueError('Input `Amat` has inconsistent dimensions with '
                             'input `bvec` and/or `wframe`')

    else:

        raise ValueError('`surf_plas` or `Amat` and `bvec` must be supplied')

    results = dict()

    # Perform the optimization
    if algorithm.lower() == 'rcls':

        # Check supplied parameters
        if 'reg_W' not in params:
            raise ValueError('params dictionary must contain ''reg_W'' '
                             + 'for the RCLS algorithm')
        else:
            reg_W = params['reg_W']

        assume_no_crossings = False if 'assume_no_crossings' not in params \
            else params['assume_no_crossings']

        x, f_B, f_R, f = \
            rcls_wireframe(wframe, A, b, reg_W, assume_no_crossings, verbose)

        results['f_R'] = f_R  # f_B and f will be recorded later

    elif algorithm.lower() == 'gsco':

        # Check supplied parameters
        for v in ['lambda_S', 'max_iter', 'print_interval']:
            if v not in params:
                raise ValueError(('params dictionary must contain ''%s'' for '
                                  + 'the GSCO algorithm') % (v))

        # Set default values if necessary
        default_current = 0.0 if 'default_current' not in params \
            else params['default_current']
        max_current = np.inf if 'max_current' not in params \
            else params['max_current']
        match_current = False if 'match_current' not in params \
            else params['match_current']
        no_crossing = False if 'no_crossing' not in params \
            else params['no_crossing']
        no_new_coils = False if 'no_new_coils' not in params \
            else params['no_new_coils']
        max_loop_count = 0 if 'max_loop_count' not in params \
            else params['max_loop_count']
        x_init = None if 'x_init' not in params \
            else params['x_init']
        loop_count_init = None if 'loop_count_init' not in params \
            else params['loop_count_init']

        x, loop_count, iter_hist, curr_hist, loop_hist, f_B_hist, f_S_hist, \
            f_hist, x_init_out \
            = gsco_wireframe(wframe, A, b, params['lambda_S'], no_crossing,
                             match_current, default_current, max_current,
                             params['max_iter'], params['print_interval'],
                             no_new_coils=no_new_coils,
                             max_loop_count=max_loop_count, x_init=x_init,
                             loop_count_init=loop_count_init, verbose=verbose)

        f_B = f_B_hist[-1]
        f_S = f_S_hist[-1]
        f = f_hist[-1]

        results['loop_count'] = loop_count
        results['iter_hist'] = iter_hist
        results['curr_hist'] = curr_hist
        results['loop_hist'] = loop_hist
        results['f_B_hist'] = f_B_hist
        results['f_S_hist'] = f_S_hist
        results['f_hist'] = f_hist
        results['x_init'] = x_init_out
        results['f_S'] = f_S

    else:

        raise ValueError('Unrecognized algorithm %s' % (algorithm))

    # Prepare the results to output
    mf_wf = WireframeField(wframe)  # regenerate with solution currents
    results['x'] = x
    results['Amat'] = A
    results['bvec'] = b
    results['wframe_field'] = mf_wf
    results['f_B'] = f_B
    results['f'] = f

    return results


def bnorm_obj_matrices(wframe, surf_plas, ext_field=None,
                       area_weighted=True, bnorm_target=None, verbose=True):
    """
    Computes the normal field matrix and target field vector used for 
    determining the squared-flux objective for wireframe current optimizations.

    Parameters
    ----------
        wframe: instance of the ToroidalWireframe class
            Wireframe whose segment currents are to be optimized
        surf_plas: Surface class instance 
            Surface of the target plasma, on which test points are placed for
            evaluation of the normal field. 
        ext_field: MagneticField class instance (optional)
            Constant external field assumed to be present in addition to the
            field produced by the wireframe.
        bnorm_target: 1d double array (optional)
            Target value of the normal field on the plasma boundary to be
            produced by the combination of the wireframe and `ext_field`.
            Zero by default. Test points on the plasma boundary corresponding 
            to the elements of bnorm_target must agree with the test points of 
            `surf_plas`.   
        area_weighted: boolean (optional)
            Determines whether the normal field matrix and target normal field
            vector elements are weighted by the square root of the area 
            ascribed to each test point on the plasma boundary. If true, the
            weightings will be applied and therefore the optimization 
            objective will be the square integral of the normal field on the 
            plasma boundary. If false, the weightings will NOT be applied, and 
            the optimization objective will be the sum of squares of the normal 
            field at each test point. 
        verbose: boolean (optional)
            If true, will print progress to screen with durations of certain
            steps

    Returns
    -------
        Amat: 2d double array 
            Inductance matrix relating normal field at test points on the
            plasma boundary to currents in each segment.
        bvec: double array (column vector)
            Vector giving the target values of the normal field at each test
            point on the plasma boundary.
    """

    if not isinstance(surf_plas, Surface):
        raise ValueError('Input `surf_plas` must be a Surface class '
                         + 'instance')

    # Calculate the normal vectors for the surface
    n = surf_plas.normal()
    absn = np.linalg.norm(n, axis=2)[:, :, None]
    unitn = n * (1./absn)
    sqrt_area = np.sqrt(absn.reshape((-1, 1))/float(absn.size))

    if area_weighted:
        area_weight = sqrt_area
    else:
        area_weight = np.ones(sqrt_area.shape)

    # Calculate the normal field matrix for the wireframe
    if verbose:
        print('    Calculating the wireframe field and normal field matrix')
    t0 = time.time()
    mf_wf = WireframeField(wframe)
    mf_wf.set_points(surf_plas.gamma().reshape((-1, 3)))
    A = mf_wf.dBnormal_by_dsegmentcurrents_matrix(surf_plas,
                                                  area_weighted=area_weighted)
    t1 = time.time()
    if verbose:
        print('        Field and normal field matrix calc took %.2f seconds'
              % (t1 - t0))

    # Calculate the target normal field to cancel contributions from the
    # external field
    if ext_field is not None:
        if not isinstance(ext_field, MagneticField):
            raise ValueError('Input `ext_field` must be a MagneticField '
                             + 'class instance')

        if verbose:
            print('    Determining contribution from external field')

        # Save the test points from the external field as input
        orig_points = ext_field.get_points_cart_ref()

        ext_field.set_points(surf_plas.gamma().reshape((-1, 3)))
        B_ext = ext_field.B().reshape(n.shape)
        bnorm_ext = np.sum(B_ext * unitn, axis=2)[:, :, None]
        bnorm_ext_weighted = bnorm_ext.reshape((-1, 1))*area_weight

        # Restore the original test points
        ext_field.set_points(orig_points)

    else:

        bnorm_ext_weighted = 0*area_weight

    # Calculate the target normal field to cancel contributions from
    # plasma currents
    if bnorm_target is not None:

        if bnorm_target.size != area_weight.size:
            raise ValueError('Input `bnorm_target` must have the same'
                             + 'number of elements as the number of quadrature points'
                             + 'of `surf_plas`')

        if verbose:
            print('    Adding contribution from plasma current')

        bnorm_target_weighted = bnorm_target.reshape((-1, 1))*area_weight

    else:

        bnorm_target_weighted = 0*area_weight

    # Calculate the target bnormal on the plasma boundary
    b = np.ascontiguousarray(bnorm_target_weighted - bnorm_ext_weighted)

    return A, b


def rcls_wireframe(wframe, Amat, bvec, reg_W, assume_no_crossings, verbose):
    """
    Performs a Regularized Constrained Least Squares optimization for the 
    segments in a wireframe.

    Parameters
    ----------
        wframe: instance of the ToroidalWireframe class
            Wireframe whose segment currents are to be optimized. This function
            will update the `currents` instance variable to the solution.
            The optimizer will obey the constraints set in wframe.
        Amat: 2d double array 
            Inductance matrix relating normal field at test points on the
            plasma boundary to currents in each segment.
        bvec: double array (column vector)
            Vector giving the target values of the normal field at each test
            point on the plasma boundary.
        reg_W: scalar, 1d array, or 2d array
            Scalar, array, or matrix for regularization. If a 1d array, must
            have the same number of elements as wframe.n_segments. If a 2d 
            array, both dimensions must be equal to wframe.n_segments.
        assume_no_crossings: boolean (optional)
            If true, will assume that the wireframe is constrained such that
            its free segments form single-track loops with no forks or 
            crossings.  Default is False.
        verbose: boolean (optional)
            If true, will print progress to screen with durations of certain
            steps

    Returns
    -------
        x: double array (1d column vector)
            Solution (currents in each segment of the wirframe)
        f_B, f_R, f: doubles
            Values of the partial objective functions (f_B for field accuracy,
            f_R for regularization) and the total objective function
            (f = f_B + f_R)
    """

    # Obtain the constraint matrices
    if verbose:
        print('    Obtaining constraint matrices')
    C, d = wframe.constraint_matrices(assume_no_crossings=assume_no_crossings,
                                      remove_constrained_segments=True)
    free_segs = wframe.unconstrained_segments()

    if np.shape(C)[0] >= len(free_segs):
        raise ValueError('Least-squares problem has as many or more '
                         + 'constraints than degrees\nof freedom. '
                         + 'Wireframe may have redundant constraints or the problem is\n'
                         + 'over-constrained.')

    # Trim constrained segments out of the A and W matrices
    Afree = Amat[:, free_segs]
    if np.isscalar(reg_W):
        Wfree = reg_W
        W = reg_W
    else:
        W = np.array(reg_W)
        if W.ndim == 1:
            Wfree = W[free_segs]
        elif W.ndim == 2:
            Wfree = W[free_segs, free_segs]
        else:
            raise ValueError('Input reg_W must be a scalar, 1d array, '
                             + 'or 2d array')

    # Solve the least-squares problem
    if verbose:
        print('    Solving the regularized constrained least-squares problem')
    t0 = time.time()
    xfree = regularized_constrained_least_squares(Afree, bvec, Wfree, C, d)
    t1 = time.time()
    if verbose:
        print('        Solver took %.2f seconds' % (t1 - t0))

    # Construct the solution column vector
    x = np.zeros((wframe.n_segments, 1))
    x[free_segs] = xfree[:]

    # Set wireframe currents to the solution vector
    wframe.currents[:] = 0
    wframe.currents[free_segs] = xfree.reshape((-1))[:]

    # Calculate the objectives
    f_B = 0.5 * np.sum((Amat @ x - bvec)**2)
    if np.isscalar(W):
        f_R = 0.5 * W**2 * np.sum(x**2)
    elif W.ndim == 1:
        f_R = 0.5 * np.sum((W.ravel()*x.ravel())**2)
    else:
        f_R = 0.5 * np.sum((W @ x)**2)
    f = f_B + f_R

    return x, f_B, f_R, f


def gsco_wireframe(wframe, A, c, lambda_S, no_crossing, match_current,
                   default_current, max_current, max_iter, print_interval,
                   no_new_coils=False, max_loop_count=0, x_init=None,
                   loop_count_init=None, verbose=True):
    """
    Runs the Greedy Stellarator Coil Optimization algorithm to optimize the
    currents in a wireframe.

    Parameters
    ----------
        wframe: instance of the ToroidalWireframe class
            Wireframe whose segment currents are to be optimized. This function
            will update the `currents` instance variable to the solution.
            The optimizer will obey the constraints set in wframe.
        Amat: contiguous 2d double array 
            Inductance matrix relating normal field at test points on the
            plasma boundary to currents in each segment.
        bvec: contiguous double array (column vector)
            Vector giving the target values of the normal field at each test
            point on the plasma boundary.
        lambda_S: double
            Weighting factor for the objective function proportional to the
            number of active segments
        no_crossing: boolean
            If true, the solution will forbid currents from crossing within
            the wireframe 
        match_current: boolean
            If true, added loops of current will match the current of the 
            loop(s) adjacent to where they are added
        default_current: double
            Loop current to add during each iteration to empty loops or to any
            loop if not matching existing current
        max_current: double
            Maximum magnitude for the current in each segment
        max_iter: integer
            Maximum number of iterations to perform
        print_interval: integer
            Number of iterations between subsequent progress prints to screen
        no_new_coils: boolean (optional)
            If true, the solution will forbid the addition of currents to loops 
            where no segments already carry current (although it is still 
            possible for an existing coil to be split into two coils); False
            by default
        max_loop_count: integer (optional)
            If nonzero, sets the maximum number of current increments to add
            to a given loop in the wireframe (default is zero)
        x_init: contiguous double array (optional)
            Initial current values to impose on the wireframe segments; will
            overwrite wframe.currents if used
        loop_count_init: contiguous integer array (optional)
            Signed number of loops of current added to each loop in the 
            wireframe prior to the optimization (optimization will
            add to these numbers); zero by default
        verbose: boolean (optional)
            If true, will print progress to screen with durations of certain
            steps

    Returns
    -------
        x: double array (1d column vector)
            Solution (currents in each segment of the wirframe)
        loop_count: integer array (1d columnn vector)
            Signed number of current loops added to each loop in the wireframe
        iter_hist: integer array 
            Array with the iteration numbers of the data recorded in the 
            history arrays. The first index is 0, corresponding to the 
            initial guess `x_init`; the last is the final iteration. 
        curr_hist: double array
            Array with the signed loop current added at each iteration, 
            taken to be zero for the initial guess (iteration zero)
        loop_hist: integer array
            Array with the index of the loop to which current was added
            at each iteration, taken to be zero for the initial guess
            (iteration zero)
        f_B_hist: 1d double array (column vector)
            Array with values of the f_B objective function at each iteration
        f_S_hist: 1d double array (column vector)
            Array with values of the f_S objective function at each iteration
        f_hist: 1d double array (column vector)
            Array with values of the f_S objective function at each iteration
        x_init: 1d double array (column vector)
            Copy of the initial guess provided to the optimizer
    """

    # Obtain data from the wireframe class instance
    loops = wframe.get_cell_key()
    free_loops = wframe.get_free_cells(form='logical')
    segments = wframe.segments
    connections = wframe.connected_segments

    # Initialize x with wframe currents or user-provided values
    if x_init is None:
        x_init = np.ascontiguousarray(np.zeros((wframe.n_segments, 1)))
        x_init[:, 0] = wframe.currents[:]
    else:
        x_init = np.ascontiguousarray(np.reshape(x_init, (-1, 1)))

    if loop_count_init is None:
        loop_count_init = np.ascontiguousarray(np.zeros(len(free_loops)).astype(np.int64))
    else:
        loop_count_init = np.ascontiguousarray(loop_count_init).astype(np.int64)

    # Run the GSCO algorithm
    if verbose:
        print('    Running GSCO')
    t0 = time.time()
    x, loop_count, iter_hist, curr_hist, loop_hist, f_B_hist, f_S_hist, f_hist \
        = sopp.GSCO(no_crossing, no_new_coils, match_current, A, c,
                    np.abs(default_current), np.abs(max_current),
                    np.abs(max_loop_count), loops, free_loops, segments,
                    connections, lambda_S, max_iter, x_init, loop_count_init,
                    print_interval)
    t1 = time.time()
    if verbose:
        print('        GSCO took %.2d seconds' % (t1 - t0))

    # Set wireframe currents to the solution vector
    wframe.currents[:] = 0
    wframe.currents[:] = x.reshape((-1))[:]

    return x, loop_count, iter_hist, curr_hist, loop_hist, \
        f_B_hist, f_S_hist, f_hist, x_init


def get_gsco_iteration(iteration, res, wframe):
    """
    Returns the intermediate solution obtained by GSCO at a given iteration.

    Parameters
    ----------
        iteration: integer
            The iteration at which data are requested (0 corresponds to the
            initialization)
        res: dictionary
            Dictionary returned by `optimize_wireframe`
        wframe: wireframe class instance
            Wireframe whose segment currents were optimized

    Returns
    -------
        x_iter: double array (1d column vector)
            Solution at the requested iteration
    """

    if 'loop_hist' not in res or 'curr_hist' not in res or 'x_init' not in res:
        raise ValueError('`res` does not appear to contain data from a ' +
                         ' GSCO procedure')

    if wframe.n_segments != res['x_init'].size:
        raise ValueError('Input `wframe` is not consistent with the solution ' +
                         'in `res`')

    if iteration + 1 > res['loop_hist'].size:
        raise ValueError('`iteration` exceeds number of iterations for ' +
                         'solution')

    cells = wframe.get_cell_key()

    x_iter = np.array(res['x_init'])
    for i in range(iteration+1):

        curr_i = res['curr_hist'][i]
        cell_i = res['loop_hist'][i]

        x_iter[cells[cell_i, :2]] += curr_i
        x_iter[cells[cell_i, 2:]] -= curr_i

    return x_iter


def regularized_constrained_least_squares(A, b, W, C, d):
    """
    Solves a linear least squares problem with Tikhonov regularization
    subject to linear equality constraints on the variables.

    In other words, minimizes:

    0.5 * ((A * x - b)**2 + (W * x)**2

    such that:

    C * x = d

    Here, A is the design matrix, b is the target vector, W is the 
    regularization matrix (normally diagonal), and C and d contain the 
    coefficients and constants of the constraint equations, respectively.

    Parameters
    ----------
        A: array with dimensions m*n
            Design matrix
        b: array with dimension m
            Target vector
        W: scalar, array with dimension n, or array with dimension n*n
            Regularization matrix
        C: array with dimension p*n
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
    bvec = np.array(b).reshape((-1, 1))
    Ctra = np.array(C).T  # Transpose will be used for the calculations
    dvec = np.array(d).reshape((-1, 1))

    # Check the inputs
    m, n = Amat.shape
    if bvec.shape[0] != m:
        raise ValueError('Number of elements in b must match rows in A')
    n_C, p = Ctra.shape
    if n_C != n:
        raise ValueError('A and C must have the same number of columns')
    if dvec.shape[0] != p:
        raise ValueError('Number of elements in d must match rows in C')

    if np.isscalar(W):
        Wmat = W*np.eye(n)
    else:
        Wmat = np.squeeze(W)
        if len(Wmat.shape) == 1:
            if Wmat.shape[0] != n:
                raise ValueError('Number of elements in vector-form W '
                                 'must match columns in A')
            Wmat = np.diag(Wmat)
        elif len(Wmat.shape) == 2:
            if Wmat.shape[0] != n or Wmat.shape[1] != n:
                raise ValueError('Number of rows and columns in matrix-form W '
                                 'must both equal number of columns in A')
        else:
            raise ValueError('W must be a scalar, 1d array, or 2d array')

    # Compute the QR factorization of the transpose of the constraint matrix
    Qfull, Rtall = _qr_factorization_wrapper(Ctra)
    Q1mat = Qfull[:, :p]  # Orthonormal vectors in the constrained subspace
    Q2mat = Qfull[:, p:]  # Orthonormal vectors in the free subspace
    Rmat = Rtall[:p, :]

    # SOLVE: Rmat.T * uvec = dvec
    # uvec = coefficients for basis vectors from constrained subspace
    uvec = scipy.linalg.solve_triangular(Rmat.T, dvec, lower=True)

    # Form the LHS of the least-squares problem
    AQ2mat = Amat @ Q2mat
    WQ2mat = Wmat @ Q2mat
    LHS = AQ2mat.T @ AQ2mat + WQ2mat.T @ WQ2mat

    # Form the RHS of the least-squares problem
    AQ1mat = Amat @ Q1mat
    WQ1mat = Wmat @ Q1mat
    AQ1uvec = AQ1mat @ uvec
    WQ1uvec = WQ1mat @ uvec
    AQ2bvec = AQ2mat.T @ bvec
    RHS = AQ2bvec - AQ2mat.T @ AQ1uvec - WQ2mat.T @ WQ1uvec

    # SOLVE: least-squares equation for the "v" vector
    # vvec = coefficients for basis vectors in the unconstrained subspace
    vvec = scipy.linalg.lstsq(LHS, RHS)[0]

    # Transform from "Q" basis back to the basis of individual segment currents
    return Qfull @ np.concatenate((uvec, vvec), axis=0)


def _qr_factorization_wrapper(M):
    """
    Wrapper for the function ``scipy.linalg.qr`` that handles a bug that has 
    been observed in some installations of simsopt. If the bug is present, 
    performing certain actions (e.g. a magnetic field calculation) can
    inexplicably cause a subsequent (and possibly unrelated) call to 
    ``scipy.linalg.qr`` to return matrices with nan values. This issue can 
    often be resolved by simply calling the function a second time.

    For some discussions that are possibly relevant to this issue:

    https://github.com/scipy/scipy/issues/5586

    https://github.com/numpy/numpy/issues/20356

    Parameters
    ----------
        M: 2d double array
            matrix to factorize

    Returns
    -------
        Q, R: 2d double arrays
            matrices returned by scipy.linalg.qr
    """

    Q, R = scipy.linalg.qr(M)

    if not np.all(np.isfinite(R)):

        Q, R = scipy.linalg.qr(M)

        if not np.all(np.isfinite(R)):
            raise RuntimeError('Error in calculating QR factorization.')

    return Q, R
