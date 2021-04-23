from scipy.optimize import minimize, least_squares
import numpy as np
from simsopt.geo.surfaceobjectives import QfmResidual

class QfmSurface():

    def __init__(self, biotsavart, surface, label, targetlabel):
        self.bs = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel

    def qfm_penalty_constraints(self, x, derivatives=0, constraint_weight=1.):
        """
        Returns the residual

        0.5 * f(x)^2 + 0.5 * constraint_weight * (label - labeltarget)^2

        and optionally the gradient and Hessian wrt surface params.
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
        d2val =  r*d2r + dr[:,None]*dr[None,:] + constraint_weight*rl*d2l + constraint_weight*dl[:,None]*dl[None,:]

        return val, dval, d2val

    def qfm_exact_constraints(self, xl, derivatives=0):
        """
        This function returns the optimality conditions corresponding to the
            minimization problem

            min 0.5 * f(x)^2

            subject to

            label - targetlabel = 0

        as well as optionally the first derivatives for these optimality
        conditions.
        """
        sdofs = xl[:-1]
        lm = xl[-1]
        s = self.surface
        bs = self.bs
        s.set_dofs(sdofs)
        nsurfdofs = sdofs.size

        qfm = QfmResidual(s, bs)
        qfm_res = qfm.J()
        dqfm_res = qfm.dJ_by_dsurfacecoefficients();

        l = self.label.J()
        dl = self.label.dJ_by_dsurfacecoefficients()
        res = np.zeros(xl.shape)
        res[0:-1] = dqfm_res - lm * dl
        res[-1] = l - self.targetlabel
        if derivatives == 0:
            return res

        dres = np.zeros((xl.shape[0], xl.shape[0]))
        d2l = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
        d2_qfm_res = qfm.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        dres[0:-1,0:-1] = d2_qfm_res - lm * d2l
        dres[0:-1, -1] = -dl
        dres[-1,0:-1] = dl

        return res, dres

    def minimize_qfm_exact_constraints_newton(self, tol=1e-12, maxiter=10,
        lm=[0.]):
        """
        This function solves the constrained optimization problem
            min 0.5 * f(x)^2

            subject to

            label - targetlabel = 0

        using Lagrange multipliers and Newton's method
        """
        s = self.surface
        xl = np.concatenate((s.get_dofs(), lm))

        val, dval = self.qfm_exact_constraints(xl, derivatives=1)
        norm = np.linalg.norm(val)

        print('norm: ',norm)
        print('val: ',val)
        print('dval: ',dval)
        print(dval)

        i = 0
        while i < maxiter and norm > tol:
            print('norm = ',norm)
            print('iter = ',i)
            print('lambda = ',val[-1])
            dx = np.linalg.solve(dval, val)
            if norm < 1e-9:  # iterative refinement for higher accuracy. TODO: cache LU factorisation
                dx += np.linalg.solve(dval, val-dval@dx)
            xl = xl - dx
            val, dval = self.qfm_exact_constraints(xl, derivatives=1)
            norm = np.linalg.norm(val)
            normgrad = np.linalg.norm(dval)

            i = i + 1

        res = {
            "residual": val, "jacobian": dval, "iter": i, "success": norm <= tol, "lm": lm,
        }
        res['s'] = s
        return res

    def minimize_qfm_penalty_constraints_LBFGS(self, tol=1e-3, maxiter=1000,
        constraint_weight=1.):
        """
        This function tries to find the surface that approximately solves

        min 0.5 * f(x)^2 + 0.5 * constraint_weight * (label - labeltarget)^2

        where f(x) is the QFM residual. This is done using LBFGS.
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
