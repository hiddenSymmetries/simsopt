import numpy as np
from scipy.linalg import lu
from scipy.optimize import minimize

from .curveobjectives import CurveLength
from .curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from .._core.optimizable import Optimizable

__all__ = ['PeriodicFieldLine']

def field_line_residual(curve, length, field):
    pts = curve.gamma()
    field.set_points(pts.reshape((-1, 3)))
    B = field.B().reshape((-1, 3))
    modB = field.AbsB()
    res = curve.gammadash()/length - B/modB
    res = res.flatten()
    if not curve.stellsym:
        # set y=0
        res_y = curve.gamma()[0, 1]
        dres_y = np.concatenate([curve.dgamma_by_dcoeff()[0, 1, :], [0]])

    dres1_dcoeff = curve.dgammadash_by_dcoeff()/length
    
    idx = np.arange(3)
    diag = np.zeros((pts.shape[0], 3, 3))
    diag[:, idx, idx] = 1/modB
    dres2_dB = -B[:, None, :] * B[:, :, None]/modB[:, None]**3 + diag

    dB_by_dX = field.dB_by_dX()
    dB_dc = np.einsum('ikl,ikm->ilm', dB_by_dX, curve.dgamma_by_dcoeff(), optimize=True)
    dres2_dcoeff = np.einsum('ikl,ikm->ilm', dres2_dB, dB_dc, optimize=True)
    dres_dcoeff = dres1_dcoeff - dres2_dcoeff

    ncurve = dres1_dcoeff.shape[-1]
    dres_dcoeff = dres_dcoeff.reshape((-1, ncurve))
    
    dres_del = -curve.gammadash().reshape((-1, 1))/length**2
    dres = np.concatenate((dres_dcoeff, dres_del), axis=-1)
    
    dres_dB = -dres2_dB.reshape((-1, 3))
    if not curve.stellsym:
        res = np.concatenate((res, [res_y]))
        dres = np.concatenate((dres, dres_y[None, :]), axis=0)
        dres_dB = np.concatenate([dres_dB, np.zeros((1, 3))], axis=0)
    return res, dres, dres_dB

def periodicfieldline_dcoils_dcurrents_vjp(lm, biotsavart, fieldline):
    length = fieldline.res['length']
    curve = fieldline.curve
    res, dres, dres_dB = field_line_residual(curve, length, biotsavart)
    
    lmask = np.zeros(fieldline.res["mask"].shape)
    lmask[fieldline.res["mask"]] = lm
    if not curve.stellsym:
        dres_dB = dres_dB[:-1] # the final equation if stellsym==False is just the curve label, which doesn't depend on B
        lmask = lmask[:-1]

    dres_dB = dres_dB.reshape((-1, 3, 3)) # the final equation if stellsym==False is just the curve label, which doesn't depend on B
    lm_cons = lmask.reshape((-1, 3))

    lm_times_dres_dB = np.sum(lm_cons[:, :, None] * dres_dB, axis=1).reshape((-1, 3))
    lm_times_dres_dcoils = biotsavart.B_vjp(lm_times_dres_dB)
    return lm_times_dres_dcoils


class PeriodicFieldLine(Optimizable):
    r"""
    The PeriodicFieldLine class computes a periodic field line of a BiotSavart
    magnetic field, that is, a closed field line that is a fixed point of the
    field-line return map. The magnetic axis and the X-points of a stellarator
    are the most common examples.

    The field line is found by driving the residual

        .. math::

            \mathbf r(x) = \frac{\boldsymbol\gamma'(\theta)}{L}
                           - \frac{\mathbf B(\boldsymbol\gamma(\theta))}
                                  {|\mathbf B(\boldsymbol\gamma(\theta))|}

    to zero, where :math:`\boldsymbol\gamma` is the curve, :math:`L` is the
    length of the field line and :math:`\mathbf B` is the magnetic field. The
    residual vanishes when the tangent of the curve is everywhere parallel to
    :math:`\mathbf B`, so that the curve is a field line, and when the curve is
    parametrized proportionally to arclength. The degrees of freedom are the
    curve coefficients together with :math:`L`.

    Args:
        biotsavart: the :obj:`~simsopt.field.BiotSavart` magnetic field.
        curve: a :obj:`~simsopt.geo.CurveXYZFourierSymmetries` holding the
               initial guess. This is the only supported curve type: the solver
               reads ``curve.order`` and ``curve.stellsym`` to build the set of
               residual equations, and it exploits the discrete rotational
               symmetry of the representation. Use ``ntor > 1`` for a field line
               that closes only after several toroidal transits.
        options: an optional dict of solver options, with keys ``verbose``
                 (default ``False``), ``newton_tol`` (default ``1e-13``) and
                 ``newton_maxiter`` (default ``40``).

    The curve must be evaluated at exactly ``2 * curve.order + 1`` quadrature
    points, so that the Newton system is square: an order-8 curve therefore
    requires 17 points. The points should span a single field period,
    ``np.linspace(0, 1/nfp, 2*order+1, endpoint=False)``. A ``ValueError`` is
    raised otherwise.

    The recommended way to solve for the field line is the
    :obj:`~simsopt.geo.PeriodicFieldLine.run_code` method, which takes an
    initial guess for the field line length,

        :obj:`~simsopt.geo.PeriodicFieldLine.run_code(length_guess)`.

    It calls
    :obj:`~simsopt.geo.PeriodicFieldLine.solve_residual_equation_exactly_newton`,
    a Newton iteration on the residual above. It converges quadratically, but
    only from an initial curve that is already sufficiently close to the desired
    periodic field line. Such a guess is usually obtained by tracing a field line
    and fitting the traced points, parametrized by arclength, with
    :obj:`~simsopt.geo.CurveXYZFourierSymmetries.least_squares_fit`. The
    alternative
    :obj:`~simsopt.geo.PeriodicFieldLine.minimize_boozer_penalty_constraints_LBFGS`
    minimizes :math:`\frac{1}{2}\|\mathbf r\|^2` with L-BFGS, which is more
    robust to a poor initial guess but reaches a looser tolerance.
    """

    def __init__(self, biotsavart, curve, options=None):
        super().__init__(depends_on=[biotsavart])

        if not isinstance(curve, CurveXYZFourierSymmetries):
            raise ValueError(
                "PeriodicFieldLine only supports a CurveXYZFourierSymmetries, "
                f"but a {type(curve).__name__} was given.")

        expected = 2 * curve.order + 1
        if len(curve.quadpoints) != expected:
            raise ValueError(
                f"The curve must have exactly 2*order+1 = {expected} quadrature "
                f"points for the Newton system to be square, but it has "
                f"{len(curve.quadpoints)} (order={curve.order}). Rebuild it with "
                f"quadpoints=np.linspace(0, 1/nfp, {expected}, endpoint=False).")

        self.biotsavart = biotsavart
        self.curve = curve
        self.need_to_run_code = True

        if options is None:
            options={}

        # set the default options now
        if 'verbose' not in options:
            options['verbose'] = False
        # default solver options for the BoozerExact and BoozerLS solvers
        if 'newton_tol' not in options:
            options['newton_tol'] = 1e-13
        if 'newton_maxiter' not in options:
            options['newton_maxiter'] = 40
        self.options = options

        
    def recompute_bell(self, parent=None):
        self.need_to_run_code = True
    
    def run_code(self, length):
        if not self.need_to_run_code:
            return
        res = self.solve_residual_equation_exactly_newton(length=length, tol=self.options['newton_tol'], maxiter=self.options['newton_maxiter'], verbose=self.options['verbose'])
        return res
    
    def get_stellsym_mask(self):
        order = self.curve.order
        stellsym = self.curve.stellsym
        if not stellsym:
            mask = np.ones((2*order+1) * 3 + 1, dtype=bool)
            return mask

        mask = np.ones((2*order+1, 3), dtype=bool)
        if stellsym:
            mask[0, 0] = False
            mask[order+1:, :] = False
        mask = mask.flatten()
        return mask

    def minimize_boozer_penalty_constraints_LBFGS(self, tol=1e-3, maxiter=1000, length=None, limited_memory=True, verbose=False):
        if not self.need_to_run_code:
            return self.res
        curve = self.curve
        
        if length is None:
            length = CurveLength(curve).J()

        x = np.concatenate((curve.get_dofs(), [length]))
        def fun(x):
            curve.x = x[:-1]
            length = x[-1]
            r, J, _ = field_line_residual(self.curve, length, self.biotsavart)
            val = 0.5 * np.mean(r**2)
            dval = J.T@r/r.size
            return val, dval

        method = 'L-BFGS-B' if limited_memory else 'BFGS'
        options = {'maxiter': maxiter, 'gtol': tol}
        if limited_memory:
            options['maxcor'] = 200
            options['ftol'] = tol

        res = minimize(
            fun, x, jac=True, method=method,
            options=options)

        resdict = {
            "fun": res.fun, "gradient": res.jac, "iter": res.nit, "info": res, "success": res.success
        }
        self.curve.x = res.x[:-1]
        length = res.x[-1]
        resdict['length'] = length

        self.res = resdict
        self.need_to_run_code = False

        if verbose:
            print(f"{method} solve - {resdict['success']}  iter={resdict['iter']}, length={resdict['length']:.8f}, ||grad||_inf = {np.linalg.norm(resdict['gradient'], ord=np.inf):.3e}", flush=True)
        return resdict


    def solve_residual_equation_exactly_newton(self, tol=1e-10, maxiter=10, length=None, verbose=False):
        #verbose=True
        if not self.need_to_run_code:
            return self.res
        
        curve = self.curve
        mask = self.get_stellsym_mask()

        if length is None:
            length = CurveLength(self.curve).J()

        x = np.concatenate((curve.get_dofs(), [length]))
        i = 0

        r, J, _ = field_line_residual(curve, length, self.biotsavart)
        b = r[mask]
        J = J[mask]

        norm = 1e6
        while i < maxiter:
            norm = np.linalg.norm(b, ord=np.inf)
            if norm <= tol:
                break
            dx = np.linalg.solve(J, b)
            dx += np.linalg.solve(J, b-J@dx)
            x -= dx
            curve.set_dofs(x[:-1])
            length = x[-1]
            i += 1
            r, J, _ = field_line_residual(curve, length, self.biotsavart)
            b = r[mask]
            J = J[mask]

        P, L, U = lu(J)
        res = {
            "residual": r, "jacobian": J, "iter": i, "success": norm <= tol, "length": length, "PLU": (P, L, U),
            "mask": mask, "vjp":periodicfieldline_dcoils_dcurrents_vjp
        }
        if verbose:
            print(f"NEWTON solve - {res['success']}  iter={res['iter']}, length={res['length']:.8f}, ||residual||_inf = {np.linalg.norm(res['residual'], ord=np.inf):.3e}, cond(J) = {np.linalg.cond(J):.3e}", flush=True)

        self.res = res
        self.need_to_run_code = False
        return res
