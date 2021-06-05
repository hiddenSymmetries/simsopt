from scipy.optimize import minimize, least_squares, NonlinearConstraint
import numpy as np
from simsopt.geo.surfaceobjectives import QfmResidual

class QfmSurface():

    def __init__(self, biotsavart, surface, label, targetlabel):
        self.bs = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel

    def qfm_label_constraint(self, x, derivatives=0):
        """
        Returns the residual

        0.5 * (label - labeltarget)^2

        and optionally the gradient wrt surface params.
        """
        assert derivatives in [0, 1]

        s = self.surface
        sdofs = x
        nsurfdofs = x.size
        bs = self.bs

        s.set_dofs(sdofs)
        l = self.label.J()
        rl = l-self.targetlabel
        val = 0.5 * rl ** 2

        if derivatives:
            dl = self.label.dJ_by_dsurfacecoefficients()
            dval = rl*dl
            return val, dval
        else:
            return val

    def qfm_objective(self, x, derivatives=0):
        """
        Returns the residual

        0.5 * f(x)^2

        and optionally the gradient wrt surface params where f(x) is the QFM
        residual defined in surfaceobjectives.py.
        """
        assert derivatives in [0, 1]

        s = self.surface
        sdofs = x
        nsurfdofs = x.size
        bs = self.bs

        s.set_dofs(sdofs)
        qfm = QfmResidual(s, bs)

        r = qfm.J()

        l = self.label.J()
        rl = l-self.targetlabel

        val = 0.5 * r**2
        if derivatives == 0:
            return val
        else:
            dr = qfm.dJ_by_dsurfacecoefficients()
            dval = r*dr
            return val, dval

    def qfm_penalty_constraints(self, x, derivatives=0, constraint_weight=1.):
        """
        Returns the residual

        0.5 * f(x)^2 + 0.5 * constraint_weight * (label - labeltarget)^2

        and optionally the gradient and Hessian wrt surface params where f(x)
        is the QFM residual defined in surfaceobjectives.py.
        """

        assert derivatives in [0, 1, 2]
        s = self.surface
        sdofs = x
        nsurfdofs = x.size
        bs = self.bs

        s.set_dofs(sdofs)

        qfm = QfmResidual(s, bs)

        r = qfm.J()

        l = self.label.J()
        rl = l-self.targetlabel

        val = 0.5 * (r**2 + constraint_weight * rl**2)
        if derivatives == 0:
            return val

        dr = qfm.dJ_by_dsurfacecoefficients()

        dl = self.label.dJ_by_dsurfacecoefficients()

        dval = r*dr + constraint_weight*rl*dl
        if derivatives == 1:
            return val, dval

        d2r = qfm.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
        d2l = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
        d2val =  r*d2r + dr[:,None]*dr[None,:] \
            + constraint_weight*rl*d2l + constraint_weight*dl[:,None]*dl[None,:]

        return val, dval, d2val

    def minimize_qfm_penalty_constraints_LBFGS(self, tol=1e-3, maxiter=1000,
        constraint_weight=1.):
        """
        This function tries to find the surface that approximately solves

        min 0.5 * f(x)^2 + 0.5 * constraint_weight * (label - labeltarget)^2

        where f(x) is the QFM residual defined in surfaceobjectives.py. This is
        done using LBFGS.
        """

        s = self.surface
        x = s.get_dofs()
        fun = lambda x: self.qfm_penalty_constraints(
            x, derivatives=1, constraint_weight=constraint_weight)
        res = minimize(
            fun, x, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol,
                'maxcor': 200})

        resdict = {
            "fun": res.fun, "gradient": res.jac, "iter": res.nit, "info": res,
                "success": res.success,
        }
        s.set_dofs(res.x)
        resdict['s'] = s

        return resdict

    def minimize_qfm_exact_constraints_SLSQP(self, tol=1e-3, maxiter=1000):
        """
        This function tries to find the surface that approximately solves

        min 0.5 * f(x)^2

        subject to

        label = labeltarget

        where f(x) is the QFM residual. This is done using SLSQP.
        """
        s = self.surface
        x = s.get_dofs()

        fun = lambda x: self.qfm_objective(x, derivatives=1)
        con = lambda x : self.qfm_label_constraint(x, derivatives=1)[0]
        dcon = lambda x : self.qfm_label_constraint(x, derivatives=1)[1]

        nlc = NonlinearConstraint(con, 0, 0)
        eq_constraints = [{'type' : 'eq', 'fun': con, 'jac': dcon}]
        res = minimize(
            fun, x, jac=True, method='SLSQP',constraints=eq_constraints,
            options={'maxiter': maxiter, 'ftol': tol})

        resdict = {
            "fun": res.fun, "gradient": res.jac, "iter": res.nit, "info": res,
                "success": res.success,
        }
        s.set_dofs(res.x)
        resdict['s'] = s

        return resdict
