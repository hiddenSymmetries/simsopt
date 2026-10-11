import numpy as np

def print_dofs_nicely(surf, lb=None, ub=None):
    '''
    Print a table of dof names, values and bounds.

    Parameters
    ----------
    surf : Optimizable
        Object whose `dof_names` and `x` are printed.
    lb : array-like, optional
        Lower bounds to show; defaults to `surf.lower_bounds` if both lb and ub are None.
    ub : array-like, optional
        Upper bounds to show; defaults to `surf.upper_bounds` if both lb and ub are None.
    '''
    if lb is None and ub is None:
        dofs_lb_ub = list(zip(surf.x, surf.lower_bounds, surf.upper_bounds))
    else:
        dofs_lb_ub = list(zip(surf.x, lb, ub))
    dofs_dict = dict(zip(surf.dof_names, dofs_lb_ub))
    print("{:<30} {:<20} {:<20} {:<20}".format('dof','value','lower bound', 'upper bound'))
    for k, v in dofs_dict.items():
        val, lb, ub = v
        print("{:<30} {:<20} {:<20} {:<20}".format(k, val, lb, ub))

def b_p(t, p, x, i=None):
    '''
    B-spline basis functions of degree p on knots t (Cox-de Boor recursion).

    Parameters
    ----------
    t : ndarray
        Knot vector.
    p : int
        Degree.
    x : ndarray
        1D array of evaluation points.
    i : int, optional
        Index of a single basis function to return; None returns all.

    Returns
    -------
    ndarray
        Shape (len(x), len(t) - p - 1), or (len(x),) if i is given.
    '''
    b = []

    for deg in range(0, p+1):
        if deg == 0:
            # x = x[(x >= t[0]) & (x <= t[-1])]
            x1d = x.copy()
            t1d = t.copy()
            t = np.outer(np.ones(len(x)), t)
            x = np.expand_dims(x, 0)
            b0 = ((x.T >= t[:, :-1]) & (x.T < t[:, 1:]))
            # include the rightmost knot in the last non-empty span
            b0[np.isclose(x1d, t1d[-1]), np.flatnonzero(t1d[:-1] < t1d[-1])[-1]] = 1
            b.append(b0)
        else:
            l_term_n = (x.T - t[:, :-deg-1])
            l_term_d = (t[:, deg:-1] - t[:, :-deg-1])
            l_term = b[-1][:, :-1]* np.divide(l_term_n, l_term_d, out=np.zeros_like(l_term_d), where=l_term_d != 0)

            r_term_n = (t[:, deg+1:] - x.T)
            r_term_d = (t[:, deg+1:] - t[:, 1:-deg])
            r_term = b[-1][:, 1:]* np.divide(r_term_n, r_term_d, out=np.zeros_like(r_term_d), where=r_term_d != 0)

            b.append(l_term + r_term)

    if i is None:
        return b[-1]
    else:
        return b[-1][:, i]

def b_p_deriv(t, p, x):
    '''
    B-spline basis and first derivative, same conventions as `b_p` (requires p >= 1).

    Parameters
    ----------
    t : ndarray
        Knot vector.
    p : int
        Degree.
    x : ndarray
        1D array of evaluation points.

    Returns
    -------
    basis : ndarray
        Basis functions, shape (len(x), len(t) - p - 1).
    deriv : ndarray
        First derivatives with respect to x, same shape.
    '''
    b = []
    tt = t
    xx = x
    for deg in range(0, p+1):
        if deg == 0:
            x1d = xx.copy()
            t1d = tt.copy()
            tt = np.outer(np.ones(len(xx)), tt)
            xx = np.expand_dims(xx, 0)
            b0 = ((xx.T >= tt[:, :-1]) & (xx.T < tt[:, 1:]))
            # include the rightmost knot in the last non-empty span
            b0[np.isclose(x1d, t1d[-1]), np.flatnonzero(t1d[:-1] < t1d[-1])[-1]] = 1
            b.append(b0)
        else:
            l_term_n = (xx.T - tt[:, :-deg-1])
            l_term_d = (tt[:, deg:-1] - tt[:, :-deg-1])
            l_term = b[-1][:, :-1] * np.divide(l_term_n, l_term_d, out=np.zeros_like(l_term_d), where=l_term_d != 0)

            r_term_n = (tt[:, deg+1:] - xx.T)
            r_term_d = (tt[:, deg+1:] - tt[:, 1:-deg])
            r_term = b[-1][:, 1:] * np.divide(r_term_n, r_term_d, out=np.zeros_like(r_term_d), where=r_term_d != 0)

            b.append(l_term + r_term)

    bp_minus1 = b[-2]
    l_d = (tt[:, p:-1] - tt[:, :-p-1])
    r_d = (tt[:, p+1:] - tt[:, 1:-p])
    deriv = p * (
        np.divide(bp_minus1[:, :-1], l_d, out=np.zeros_like(l_d), where=l_d != 0)
        - np.divide(bp_minus1[:, 1:], r_d, out=np.zeros_like(r_d), where=r_d != 0)
    )
    return b[-1], deriv

def b_p_deriv2(t, p, x):
    '''
    B-spline basis and its first and second derivatives, same conventions as `b_p`.

    Parameters
    ----------
    t : ndarray
        Knot vector.
    p : int
        Degree (requires p >= 1).
    x : ndarray
        1D array of evaluation points.

    Returns
    -------
    basis : ndarray
        Basis functions, shape (len(x), len(t) - p - 1).
    deriv1 : ndarray
        First derivatives with respect to x, same shape.
    deriv2 : ndarray
        Second derivatives with respect to x, same shape; zero for p < 2.
    '''
    b = []
    tt = t
    xx = x
    for deg in range(0, p+1):
        if deg == 0:
            x1d = xx.copy()
            t1d = tt.copy()
            tt = np.outer(np.ones(len(xx)), tt)
            xx = np.expand_dims(xx, 0)
            b0 = ((xx.T >= tt[:, :-1]) & (xx.T < tt[:, 1:]))
            # include the rightmost knot in the last non-empty span
            b0[np.isclose(x1d, t1d[-1]), np.flatnonzero(t1d[:-1] < t1d[-1])[-1]] = 1
            b.append(b0)
        else:
            l_term_n = (xx.T - tt[:, :-deg-1])
            l_term_d = (tt[:, deg:-1] - tt[:, :-deg-1])
            l_term = b[-1][:, :-1] * np.divide(l_term_n, l_term_d, out=np.zeros_like(l_term_d), where=l_term_d != 0)

            r_term_n = (tt[:, deg+1:] - xx.T)
            r_term_d = (tt[:, deg+1:] - tt[:, 1:-deg])
            r_term = b[-1][:, 1:] * np.divide(r_term_n, r_term_d, out=np.zeros_like(r_term_d), where=r_term_d != 0)

            b.append(l_term + r_term)

    def _deriv_level(b_lower, d):
        l_d = tt[:, d:-1] - tt[:, :-d-1]
        r_d = tt[:, d+1:] - tt[:, 1:-d]
        return d * (
            np.divide(b_lower[:, :-1], l_d, out=np.zeros_like(l_d), where=l_d != 0)
            - np.divide(b_lower[:, 1:], r_d, out=np.zeros_like(r_d), where=r_d != 0)
        )

    deriv1 = _deriv_level(b[p-1], p)
    if p >= 2:
        deriv1_pm1 = _deriv_level(b[p-2], p-1)
        deriv2 = _deriv_level(deriv1_pm1, p)
    else:
        deriv2 = np.zeros_like(deriv1)

    return b[-1], deriv1, deriv2

def uniform_knots(n, p, domain=2*np.pi):
    '''
    Uniform periodic knot vector for n+1 control points padded by p on each side.

    Parameters
    ----------
    n : int
        Number of distinct control points minus one.
    p : int
        Degree.
    domain : float
        Parameter period.

    Returns
    -------
    ndarray
        Knots of length n+3p+2, spaced by domain/(n+1), that include 0 and domain.
    '''
    interval = domain / (n + 1)
    k = np.arange(-2 * p, n + 2 * p + 2)
    knots_wide = k * interval
    n_needed = (n + 1 + 2 * p) + p + 1
    start = (len(knots_wide) - n_needed) // 2
    return knots_wide[start:start + n_needed]

def double_reflection_rmf(gamma, gammadash, normal0):
    '''
    Rotation-minimizing frame along sampled points via double reflection (Wang et al., ACM TOG 27(1), 2008).

    Parameters
    ----------
    gamma : ndarray, shape (n, 3)
        Points along the curve, in order.
    gammadash : ndarray, shape (n, 3)
        Tangents at those points (only direction matters).
    normal0 : ndarray, shape (3,)
        Initial normal, must already be orthogonal to gammadash[0].

    Returns
    -------
    T, N, B : ndarray, shape (n, 3)
        Right-handed orthonormal frame; closure on periodic curves is not enforced.
    '''
    n = gamma.shape[0]
    T = gammadash / np.linalg.norm(gammadash, axis=1, keepdims=True)

    N = np.zeros_like(gamma)
    N[0] = normal0 / np.linalg.norm(normal0)

    for i in range(n - 1):
        v1 = gamma[i + 1] - gamma[i]
        c1 = v1 @ v1
        r_L = N[i] - (2 / c1) * (v1 @ N[i]) * v1
        t_L = T[i] - (2 / c1) * (v1 @ T[i]) * v1

        v2 = T[i + 1] - t_L
        c2 = v2 @ v2
        if c2 > 1e-14:
            N[i + 1] = r_L - (2 / c2) * (v2 @ r_L) * v2
        else:
            N[i + 1] = r_L
        N[i + 1] /= np.linalg.norm(N[i + 1])

    B = np.cross(T, N)
    return T, N, B
