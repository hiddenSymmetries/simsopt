from scipy.optimize import minimize, least_squares, NonlinearConstraint
import numpy as np
from monty.json import MontyDecoder, MSONable

from simsopt.geo.surfaceobjectives import QfmResidual

__all__ = ['QfmSurface']


class QfmSurface(MSONable):
    r"""
    QfmSurface is used to compute a quadratic-flux minimizing surface, defined
    as the minimum of the objective function,

    .. math::
        f(S) = \frac{\int_{S} d^2 x \, \left(\textbf{B} \cdot \hat{\textbf{n}}\right)^2}{\int_{S} d^2 x \, B^2}

    subject to a constraint on the surface label, such as the area, volume,
    or toroidal flux. This objective, :math:`f`, is computed in :class:`QfmResidual`
    (see :mod:`surfaceobjectives.py`). If magnetic surfaces exists, :math:`f`
    will be zero. If not, QFM surfaces approximate flux surfaces.

    The label constraint can be enforced with a penalty formulation, defined
    in :func:`qfm_penalty_constraints()`, and whose minimium is computed with
    LBFGS-B by :func:`minimize_qfm_penalty_constraints_LBFGS()`.

    Alternatively, the label constraint can be enforced with a constrained
    optimization algorithm. This constrained optimization problem is
    solved with the SLSQP algorithm by :func:`minimize_qfm_exact_constraints_SLSQP()`.
    """

    def __init__(self, biotsavart, surface, label, targetlabel):
        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.qfm = QfmResidual(surface, biotsavart)

    def qfm_label_constraint(self, x, derivatives=0):
        r"""
        Returns the residual

        .. math::
            0.5 (\texttt{label} - \texttt{labeltarget})^2

        and optionally the gradient wrt surface params.
        """
        assert derivatives in [0, 1]
        self.surface.x = x
        l = self.label.J()
        rl = l - self.targetlabel
        val = 0.5 * rl ** 2

        if derivatives:
            dl = self.label.dJ_by_dsurfacecoefficients()
            dval = rl * dl
            return val, dval
        else:
            return val

    def qfm_objective(self, x, derivatives=0):
        r"""
        Returns the residual

        .. math::
            f(S)

        and optionally the gradient wrt surface params where :math:`f(S)`` is
        the QFM residual defined in surfaceobjectives.py.
        """

        assert derivatives in [0, 1]
        self.surface.x = x
        qfm = self.qfm

        r = qfm.J()

        if not derivatives:
            return r
        else:
            dr = qfm.dJ_by_dsurfacecoefficients()
            return r, dr

    def qfm_penalty_constraints(self, x, derivatives=0, constraint_weight=1):
        r"""
        Returns the residual

        .. math::
            f(S) + 0.5 \texttt{constraint_weight} (\texttt{label} - \texttt{labeltarget})^2

        and optionally the gradient and Hessian wrt surface params where
        :math:`f(S)` is the QFM residual defined in :mod:`surfaceobjectives.py`.
        """

        assert derivatives in [0, 1]
        s = self.surface
        s.x = x

        qfm = self.qfm

        r = qfm.J()

        l = self.label.J()
        rl = l - self.targetlabel

        val = r + 0.5 * constraint_weight * rl**2
        if derivatives == 0:
            return val

        dr = qfm.dJ_by_dsurfacecoefficients()

        dl = self.label.dJ_by_dsurfacecoefficients()

        dval = dr + constraint_weight*rl*dl
        return val, dval

    def minimize_qfm_penalty_constraints_LBFGS(self, tol=1e-3, maxiter=1000,
                                               constraint_weight=1.):
        r"""
        This function tries to find the surface that approximately solves

        .. math::
            \min_S f(S) + 0.5 \texttt{constraint_weight} (\texttt{label} - \texttt{labeltarget})^2

        where :math:`f(S)` is the QFM residual defined in :mod:`surfaceobjectives.py.`
        This is done using LBFGS.
        """

        s = self.surface
        x = s.x
        fn = lambda x: self.qfm_penalty_constraints(
            x, derivatives=1, constraint_weight=constraint_weight)
        res = minimize(
            fn, x, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol,
                     'maxcor': 200})

        resdict = {
            "fun": res.fun, "gradient": res.jac, "iter": res.nit, "info": res,
            "success": res.success,
        }
        s.x = res.x
        resdict['s'] = s

        return resdict

    def minimize_qfm_exact_constraints_SLSQP(self, tol=1e-3, maxiter=1000):
        r"""
        This function tries to find the surface that approximately solves

        .. math::
            \min_{S} f(S)

        subject to

        .. math::
            \texttt{label} = \texttt{labeltarget}

        where :math:`f(S)` is the QFM residual. This is done using SLSQP.
        """
        s = self.surface
        x = s.x

        fun = lambda x: self.qfm_objective(x, derivatives=1)
        con = lambda x: self.qfm_label_constraint(x, derivatives=1)[0]
        dcon = lambda x: self.qfm_label_constraint(x, derivatives=1)[1]

        nlc = NonlinearConstraint(con, 0, 0)
        eq_constraints = [{'type': 'eq', 'fun': con, 'jac': dcon}]
        res = minimize(
            fun, x, jac=True, method='SLSQP', constraints=eq_constraints,
            options={'maxiter': maxiter, 'ftol': tol})

        resdict = {
            "fun": res.fun, "gradient": res.jac, "iter": res.nit, "info": res,
            "success": res.success,
        }
        s.x = res.x
        resdict['s'] = s

        return resdict

    def minimize_qfm(self, tol=1e-3, maxiter=1000, method='SLSQP',
                     constraint_weight=1.):
        r"""
        This function tries to find the surface that approximately solves

        .. math::
            \min_{S} f(S)

        subject to

        .. math::
            \texttt{label} = \texttt{labeltarget}

        where :math:`f(S)` is the QFM residual. This is done using method,
        either 'SLSQP' or 'LBFGS'. If 'LBFGS' is chosen, a penalized formulation
        is used defined by constraint_weight. If 'SLSQP' is chosen, constraint_weight
        is not used, and the constrained optimization problem is solved.
        """
        if method == 'SLSQP':
            return self.minimize_qfm_exact_constraints_SLSQP(
                tol=tol, maxiter=maxiter)
        elif method == 'LBFGS':
            return self.minimize_qfm_penalty_constraints_LBFGS(
                tol=tol, maxiter=maxiter, constraint_weight=constraint_weight)
        else:
            raise ValueError
