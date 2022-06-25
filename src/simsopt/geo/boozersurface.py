from scipy.optimize import minimize, least_squares
import numpy as np
from monty.json import MontyDecoder, MSONable

from simsopt.geo.surfaceobjectives import boozer_surface_residual

__all__ = ['BoozerSurface']


class BoozerSurface(MSONable):
    r"""
    BoozerSurface and its associated methods can be used to compute the Boozer
    angles on a surface. It takes a Surface representation (e.g. SurfaceXYZFourier,
    or SurfaceXYZTensorFourier), a magnetic field evaluator, surface label evaluator,
    and a target surface label.

    The Boozer angles are computed by solving a constrained least squares problem.
    The least squares objective is given by :math:`J(x) = \frac{1}{2} \mathbf r^T(x) \mathbf r(x)`, 
    where :math:`\mathbf r` is a vector of residuals computed by :mod:`boozer_surface_residual` 
    (see :mod:`surfaceobjectives.py`), and some constraints.  This objective is zero when the surface corresponds 
    to a magnetic surface of the field and :math:`(\phi,\theta)` that parametrize the surface correspond to 
    Boozer angles, and the constraints are satisfied.

    The surface label can be area, volume, or toroidal flux. The label on the computed surface will be equal or close
    to the user-provided ``targetlabel``, depending on how the label constraint is imposed.  This 
    constrained least squares problem can be solved by scalarizing and adding the constraint as 
    an additional penalty term to the objective.  This is done in

        #. :mod:`minimize_boozer_penalty_constraints_LBFGS`
        #. :mod:`minimize_boozer_penalty_constraints_newton`
        #. :mod:`minimize_boozer_penalty_constraints_ls`

    where LBFGS, Newton, or :mod:`scipy.optimize.least_squares` optimizers are used, respectively.
    Alternatively, the exactly constrained least squares optimization problem can be solved.
    This is done in

        #. :mod:`minimize_boozer_exact_constraints_newton`

    where Newton is used to solve the first order necessary conditions for optimality.
    """

    def __init__(self, biotsavart, surface, label, targetlabel):
        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel

    def boozer_penalty_constraints(self, x, derivatives=0, constraint_weight=1., scalarize=True, optimize_G=False):
        r"""
        Define the residual

        .. math::
            \mathbf r(x) = [r_1(x),...,r_n(x), \sqrt{w_c}  (l-l_0), \sqrt{w_c}  (z(\varphi=0, \theta=0) - 0)]

        where :math:`w_c` is the constraint weight, :math:`r_i` are the Boozer residuals 
        at quadrature points :math:`1,\dots,n`, :math:`l` is the surface label, and :math:`l_0` is
        the target surface label.

        For ``scalarized=False``, this function returns :math:`\mathbf r(x)` and optionally the Jacobian 
        of :math:`\mathbf r(x)`.

        for ``scalarized=True``, this function returns

        .. math::
            J(x) = \frac{1}{2}\mathbf r(x)^T \mathbf r(x),

        i.e. the least squares residual and optionally the gradient and the Hessian of :math:`J(x)`.
        """

        assert derivatives in [0, 1, 2]
        if optimize_G:
            sdofs = x[:-2]
            iota = x[-2]
            G = x[-1]
        else:
            sdofs = x[:-1]
            iota = x[-1]
            G = None

        nsurfdofs = sdofs.size
        s = self.surface
        biotsavart = self.biotsavart

        s.set_dofs(sdofs)

        boozer = boozer_surface_residual(s, iota, G, biotsavart, derivatives=derivatives)

        r = boozer[0]

        l = self.label.J()
        rl = (l-self.targetlabel)
        rz = (s.gamma()[0, 0, 2] - 0.)
        r = np.concatenate((r, [
            np.sqrt(constraint_weight) * rl,
            np.sqrt(constraint_weight) * rz
        ]))

        val = 0.5 * np.sum(r**2)
        if derivatives == 0:
            if scalarize:
                return val
            else:
                return r

        J = boozer[1]

        dl = np.zeros(x.shape)
        drz = np.zeros(x.shape)

        dl[:nsurfdofs] = self.label.dJ_by_dsurfacecoefficients()

        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]
        J = np.concatenate((
            J,
            np.sqrt(constraint_weight) * dl[None, :],
            np.sqrt(constraint_weight) * drz[None, :]), axis=0)
        dval = np.sum(r[:, None]*J, axis=0)
        if derivatives == 1:
            if scalarize:
                return val, dval
            else:
                return r, J
        if not scalarize:
            raise NotImplementedError('Can only return Hessian for scalarized version.')

        H = boozer[2]

        d2l = np.zeros((x.shape[0], x.shape[0]))
        d2l[:nsurfdofs, :nsurfdofs] = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        H = np.concatenate((
            H,
            np.sqrt(constraint_weight) * d2l[None, :, :],
            np.zeros(d2l[None, :, :].shape)), axis=0)
        d2val = J.T @ J + np.sum(r[:, None, None] * H, axis=0)
        return val, dval, d2val

    def boozer_exact_constraints(self, xl, derivatives=0, optimize_G=True):
        r"""
        This function returns the optimality conditions corresponding to the minimization problem

        .. math::
            \text{min}_x ~J(x)

        subject to 

        .. math::
            l - l_0 &= 0 \\
            z(\varphi=0,\theta=0) - 0 &= 0

        where :math:`l` is the surface label and :math:`l_0` is the target surface label, 
        :math:`J(x) = \frac{1}{2}\mathbf r(x)^T \mathbf r(x)`, and :math:`\mathbf r(x)` contains
        the Boozer residuals at quadrature points :math:`1,\dots,n`.
        We can also optionally return the first derivatives of these optimality conditions.
        """
        assert derivatives in [0, 1]
        if optimize_G:
            sdofs = xl[:-4]
            iota = xl[-4]
            G = xl[-3]
        else:
            sdofs = xl[:-3]
            iota = xl[-3]
            G = None
        lm = xl[-2:]
        s = self.surface
        biotsavart = self.biotsavart
        s.set_dofs(sdofs)
        nsurfdofs = sdofs.size

        boozer = boozer_surface_residual(s, iota, G, biotsavart, derivatives=derivatives+1)
        r, J = boozer[0:2]

        dl = np.zeros((xl.shape[0]-2,))

        l = self.label.J()
        dl[:nsurfdofs] = self.label.dJ_by_dsurfacecoefficients()
        drz = np.zeros((xl.shape[0]-2,))
        g = [l-self.targetlabel]
        rz = (s.gamma()[0, 0, 2] - 0.)
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]

        res = np.zeros(xl.shape)
        res[:-2] = np.sum(r[:, None]*J, axis=0) - lm[-2] * dl - lm[-1] * drz
        res[-2] = g[0]
        res[-1] = rz
        if derivatives == 0:
            return res

        H = boozer[2]

        d2l = np.zeros((xl.shape[0]-2, xl.shape[0]-2))
        d2l[:nsurfdofs, :nsurfdofs] = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        dres = np.zeros((xl.shape[0], xl.shape[0]))
        dres[:-2, :-2] = J.T @ J + np.sum(r[:, None, None] * H, axis=0) - lm[-2]*d2l
        dres[:-2, -2] = -dl
        dres[:-2, -1] = -drz

        dres[-2, :-2] = dl
        dres[-1, :-2] = drz
        return res, dres

    def minimize_boozer_penalty_constraints_LBFGS(self, tol=1e-3, maxiter=1000, constraint_weight=1., iota=0., G=None):
        r"""
        This function tries to find the surface that approximately solves

        .. math::
            \text{min}_x ~J(x) + \frac{1}{2} w_c (l - l_0)^2
                                 + \frac{1}{2} w_c (z(\varphi=0, \theta=0) - 0)^2

        where :math:`J(x) = \frac{1}{2}\mathbf r(x)^T \mathbf r(x)`, and :math:`\mathbf r(x)` contains
        the Boozer residuals at quadrature points :math:`1,\dots,n`.
        This is done using LBFGS.
        """

        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        fun = lambda x: self.boozer_penalty_constraints(
            x, derivatives=1, constraint_weight=constraint_weight, optimize_G=G is not None)
        res = minimize(
            fun, x, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol, 'maxcor': 200})

        resdict = {
            "fun": res.fun, "gradient": res.jac, "iter": res.nit, "info": res, "success": res.success, "G": None,
        }
        if G is None:
            s.set_dofs(res.x[:-1])
            iota = res.x[-1]
        else:
            s.set_dofs(res.x[:-2])
            iota = res.x[-2]
            G = res.x[-1]
            resdict['G'] = G
        resdict['s'] = s
        resdict['iota'] = iota

        return resdict

    def minimize_boozer_penalty_constraints_newton(self, tol=1e-12, maxiter=10, constraint_weight=1., iota=0., G=None, stab=0.):
        """
        This function does the same as :mod:`minimize_boozer_penalty_constraints_LBFGS`, but instead of LBFGS it uses
        Newton's method.
        """
        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0

        val, dval, d2val = self.boozer_penalty_constraints(
            x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None)
        norm = np.linalg.norm(dval)
        while i < maxiter and norm > tol:
            d2val += stab*np.identity(d2val.shape[0])
            dx = np.linalg.solve(d2val, dval)
            if norm < 1e-9:
                dx += np.linalg.solve(d2val, dval - d2val@dx)
            x = x - dx
            val, dval, d2val = self.boozer_penalty_constraints(
                x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None)
            norm = np.linalg.norm(dval)
            i = i+1

        r = self.boozer_penalty_constraints(
            x, derivatives=0, constraint_weight=constraint_weight, scalarize=False, optimize_G=G is not None)
        res = {
            "residual": r, "jacobian": dval, "hessian": d2val, "iter": i, "success": norm <= tol, "G": None,
        }
        if G is None:
            s.set_dofs(x[:-1])
            iota = x[-1]
        else:
            s.set_dofs(x[:-2])
            iota = x[-2]
            G = x[-1]
            res['G'] = G
        res['s'] = s
        res['iota'] = iota
        return res

    def minimize_boozer_penalty_constraints_ls(self, tol=1e-12, maxiter=10, constraint_weight=1., iota=0., G=None, method='lm'):
        """
        This function does the same as :mod:`minimize_boozer_penalty_constraints_LBFGS`, but instead of LBFGS it
        uses a nonlinear least squares algorithm when ``method='lm'``.  Options for the method 
        are the same as for :mod:`scipy.optimize.least_squares`. If ``method='manual'``, then a 
        damped Gauss-Newton method is used.
        """
        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        norm = 1e10
        if method == 'manual':
            i = 0
            lam = 1.
            r, J = self.boozer_penalty_constraints(
                x, derivatives=1, constraint_weight=constraint_weight, scalarize=False, optimize_G=G is not None)
            b = J.T@r
            JTJ = J.T@J
            while i < maxiter and norm > tol:
                dx = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), b)
                x -= dx
                r, J = self.boozer_penalty_constraints(
                    x, derivatives=1, constraint_weight=constraint_weight, scalarize=False, optimize_G=G is not None)
                b = J.T@r
                JTJ = J.T@J
                norm = np.linalg.norm(b)
                lam *= 1/3
                i += 1
            resdict = {
                "residual": r, "gradient": b, "jacobian": JTJ, "success": norm <= tol
            }
            if G is None:
                s.set_dofs(x[:-1])
                iota = x[-1]
            else:
                s.set_dofs(x[:-2])
                iota = x[-2]
                G = x[-1]
                resdict['G'] = G
            resdict['s'] = s
            resdict['iota'] = iota
            return resdict
        fun = lambda x: self.boozer_penalty_constraints(
            x, derivatives=0, constraint_weight=constraint_weight, scalarize=False, optimize_G=G is not None)
        jac = lambda x: self.boozer_penalty_constraints(
            x, derivatives=1, constraint_weight=constraint_weight, scalarize=False, optimize_G=G is not None)[1]
        res = least_squares(fun, x, jac=jac, method=method, ftol=tol, xtol=tol, gtol=tol, x_scale=1.0, max_nfev=maxiter)
        resdict = {
            "info": res, "residual": res.fun, "gradient": res.grad, "jacobian": res.jac, "success": res.status > 0,
            "G": None,
        }
        if G is None:
            s.set_dofs(res.x[:-1])
            iota = res.x[-1]
        else:
            s.set_dofs(res.x[:-2])
            iota = res.x[-2]
            G = res.x[-1]
            resdict['G'] = G
        resdict['s'] = s
        resdict['iota'] = iota
        return resdict

    def minimize_boozer_exact_constraints_newton(self, tol=1e-12, maxiter=10, iota=0., G=None, lm=[0., 0.]):
        r"""
        This function solves the constrained optimization problem

        .. math::
            \text{min}_x ~ J(x)

        subject to

        .. math::
            l - l_0 &= 0 \\
            z(\varphi=0,\theta=0) - 0 &= 0

        using Lagrange multipliers and Newton's method. In the above,
        :math:`J(x) = \frac{1}{2}\mathbf r(x)^T \mathbf r(x)`, and :math:`\mathbf r(x)` contains
        the Boozer residuals at quadrature points :math:`1,\dots,n`.

        The final constraint is not necessary for stellarator symmetric surfaces as it is automatically
        satisfied by the stellarator symmetric surface parametrization.
        """
        s = self.surface
        if G is not None:
            xl = np.concatenate((s.get_dofs(), [iota, G], lm))
        else:
            xl = np.concatenate((s.get_dofs(), [iota], lm))
        val, dval = self.boozer_exact_constraints(xl, derivatives=1, optimize_G=G is not None)
        norm = np.linalg.norm(val)
        i = 0
        while i < maxiter and norm > tol:
            if s.stellsym:
                A = dval[:-1, :-1]
                b = val[:-1]
                dx = np.linalg.solve(A, b)
                if norm < 1e-9:  # iterative refinement for higher accuracy. TODO: cache LU factorisation
                    dx += np.linalg.solve(A, b-A@dx)
                xl[:-1] = xl[:-1] - dx
            else:
                dx = np.linalg.solve(dval, val)
                if norm < 1e-9:  # iterative refinement for higher accuracy. TODO: cache LU factorisation
                    dx += np.linalg.solve(dval, val-dval@dx)
                xl = xl - dx
            val, dval = self.boozer_exact_constraints(xl, derivatives=1, optimize_G=G is not None)
            norm = np.linalg.norm(val)
            i = i + 1

        if s.stellsym:
            lm = xl[-2]
        else:
            lm = xl[-2:]

        res = {
            "residual": val, "jacobian": dval, "iter": i, "success": norm <= tol, "lm": lm, "G": None,
        }
        if G is not None:
            s.set_dofs(xl[:-4])
            iota = xl[-4]
            G = xl[-3]
            res['G'] = G
        else:
            s.set_dofs(xl[:-3])
            iota = xl[-3]
        res['s'] = s
        res['iota'] = iota
        return res

    def solve_residual_equation_exactly_newton(self, tol=1e-10, maxiter=10, iota=0., G=None):
        """
        This function solves the Boozer Surface residual equation exactly.  For
        this to work, we need the right balance of quadrature points, degrees
        of freedom and constraints.  For this reason, this only works for
        surfaces of type SurfaceXYZTensorFourier right now.

        Given ntor, mpol, nfp and stellsym, the surface is expected to be
        created in the following way::

            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
            s = SurfaceXYZTensorFourier(
                mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                quadpoints_phi=phis, quadpoints_theta=thetas)

        Or the following two are also possible in the stellsym case::

            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 0.5, mpol+1, endpoint=False)

        or::

            phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)

        and then::

            s = SurfaceXYZTensorFourier(
                mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                quadpoints_phi=phis, quadpoints_theta=thetas)

        For the stellsym case, there is some redundancy between dofs.  This is
        taken care of inside this function.

        In the non-stellarator-symmetric case, the surface has
        ``(2*ntor+1)*(2*mpol+1)`` many quadrature points and
        ``3*(2*ntor+1)*(2*mpol+1)`` many dofs.

        Equations:
            - Boozer residual in x, y, and z at all quadrature points
            - z(0, 0) = 0
            - label constraint (e.g. volume or flux)

        Unknowns:
            - Surface dofs
            - iota
            - G

        So we end up having ``3*(2*ntor+1)*(2*mpol+1) + 2`` equations and the
        same number of unknowns.

        In the stellarator-symmetric case, we have
        ``D = (ntor+1)*(mpol+1)+ ntor*mpol + 2*(ntor+1)*mpol + 2*ntor*(mpol+1)
        = 6*ntor*mpol + 3*ntor + 3*mpol + 1``
        many dofs in the surface. After calling ``surface.get_stellsym_mask()`` we have kicked out
        ``2*ntor*mpol + ntor + mpol``
        quadrature points, i.e. we have
        ``2*ntor*mpol + ntor + mpol + 1``
        quadrature points remaining. In addition we know that the x coordinate of the
        residual at phi=0=theta is also always satisfied. In total this
        leaves us with
        ``3*(2*ntor*mpol + ntor + mpol) + 2`` equations for the boozer residual, plus
        1 equation for the label,
        which is the same as the number of surface dofs + 2 extra unknowns
        given by iota and G.
        """

        from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
        s = self.surface
        if not isinstance(s, SurfaceXYZTensorFourier):
            raise RuntimeError('Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier')

        # In the case of stellarator symmetry, some of the information is
        # redundant, since the coordinates at (-phi, -theta) are the same (up
        # to sign changes) to those at (phi, theta). In addition, for stellsym
        # surfaces and stellsym magnetic fields, the residual in the x
        # component is always satisfied at phi=theta=0, so we ignore that one
        # too. The mask object below is True for those parts of the residual
        # that we need to keep, and False for those that we ignore.
        m = s.get_stellsym_mask()
        mask = np.concatenate((m[..., None], m[..., None], m[..., None]), axis=2)
        if s.stellsym:
            mask[0, 0, 0] = False
        mask = mask.flatten()

        label = self.label
        if G is None:
            G = 2. * np.pi * np.sum(np.abs(self.biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0
        r, J = boozer_surface_residual(s, iota, G, self.biotsavart, derivatives=1)
        norm = 1e6
        while i < maxiter:
            if s.stellsym:
                b = np.concatenate((r[mask], [(label.J()-self.targetlabel)]))
            else:
                b = np.concatenate((r[mask], [(label.J()-self.targetlabel), s.gamma()[0, 0, 2]]))
            norm = np.linalg.norm(b)
            if norm <= tol:
                break
            if s.stellsym:
                J = np.vstack((
                    J[mask, :],
                    np.concatenate((label.dJ_by_dsurfacecoefficients(), [0., 0.])),
                ))
            else:
                J = np.vstack((
                    J[mask, :],
                    np.concatenate((label.dJ_by_dsurfacecoefficients(), [0., 0.])),
                    np.concatenate((s.dgamma_by_dcoeff()[0, 0, 2, :], [0., 0.]))
                ))
            dx = np.linalg.solve(J, b)
            dx += np.linalg.solve(J, b-J@dx)
            x -= dx
            s.set_dofs(x[:-2])
            iota = x[-2]
            G = x[-1]
            i += 1
            r, J = boozer_surface_residual(s, iota, G, self.biotsavart, derivatives=1)

        res = {"residual": r, "jacobian": J, "iter": i, "success": norm <= tol,
               "G": G, "s": s, "iota": iota}
        return res

    @classmethod
    def from_dict(cls, d):
        decoder = MontyDecoder()
        bs = decoder.process_decoded(d["biotsavart"])
        surf = decoder.process_decoded(d["surface"])
        return cls(bs, surf, d["label"], d["targetlabel"])
