from scipy.optimize import minimize, least_squares
from scipy.linalg import lu
import numpy as np
from simsopt.geo.surfaceobjectives import boozer_surface_residual
from simsopt.geo.surfaceobjectives import boozer_surface_residual_accumulate
from simsopt.geo.surfaceobjectives import residual
from simsopt._core.graph_optimizable import Optimizable


class BoozerSurface(Optimizable):
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

    def __init__(self, biotsavart, surface, label, targetlabel, reg=None):
        Optimizable.__init__(self, depends_on=[biotsavart])
        self.bs = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.need_to_compute_surface = True
        self.reg = reg

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def boozer_penalty(self, x, derivatives=0, constraint_weight=1., scalarize=True, optimize_G=False):
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

        s = self.surface
        nphi = s.quadpoints_phi.size
        ntheta = s.quadpoints_theta.size
        nsurfdofs = sdofs.size

        s.set_dofs(sdofs)

        boozer = residual(s, iota, G, self.bs, derivatives=derivatives)
        lab = self.label.J()
        
        rnl = boozer[0]
        rl = np.sqrt(constraint_weight) * (lab-self.targetlabel)
        rz = np.sqrt(constraint_weight) * (s.gamma()[0, 0, 2] - 0.)
        r = rnl + 0.5*rl**2 + 0.5*rz**2

        if self.reg is not None:
            r += self.reg.J()
        
        if derivatives == 0:
            return r

        dl = np.zeros(x.shape)
        drz = np.zeros(x.shape)
        dl[:nsurfdofs] = self.label.dJ_by_dsurfacecoefficients()
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]
        
        Jnl = boozer[1]
        drl = np.sqrt(constraint_weight) * dl
        drz = np.sqrt(constraint_weight) * drz
        J = Jnl + rl * drl + rz * drz
        if self.reg is not None:
            J[:nsurfdofs] += self.reg.dJ()
        
        if derivatives == 1:
            return r, J
 
        Hnl = boozer[2]
        d2rl = np.zeros((x.shape[0], x.shape[0]))
        d2rl[:nsurfdofs, :nsurfdofs] = np.sqrt(constraint_weight)*self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
        H = Hnl + drl[:,None] @ drl[None,:] + drz[:,None] @ drz[None, :] + rl * d2rl 
        if self.reg is not None:
            H[:nsurfdofs, :nsurfdofs] += self.reg.d2J()
        return r, J, H



    def boozer_penalty_constraints_accumulate(self, x, derivatives=0, constraint_weight=1., scalarize=True, optimize_G=False, weighting=None):
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
        bs = self.bs
        nphi = s.quadpoints_phi.size
        ntheta = s.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        s.set_dofs(sdofs)
        

        boozer = boozer_surface_residual_accumulate(s, iota, G, self.bs, derivatives=derivatives, weighting=weighting)
        lab = self.label.J()
        
        rnl = boozer[0]
        rl = np.sqrt(constraint_weight) * (lab-self.targetlabel)
        rz = np.sqrt(constraint_weight) * (s.gamma()[0, 0, 2] - 0.)
        r = rnl/num_points + 0.5*rl**2 + 0.5*rz**2

        if self.reg is not None:
            r += self.reg.J()
        
        if derivatives == 0:
            return r
        dl = np.zeros(x.shape)
        drz = np.zeros(x.shape)
        dl[:nsurfdofs] = self.label.dJ_by_dsurfacecoefficients()
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]
        
        Jnl = boozer[1]
        drl = np.sqrt(constraint_weight) * dl
        drz = np.sqrt(constraint_weight) * drz
        J = Jnl/num_points + rl * drl + rz * drz
        if self.reg is not None:
            J[:nsurfdofs] += self.reg.dJ()
        
        if derivatives == 1:
            return r, J
        
        Hnl = boozer[2]
        d2rl = np.zeros((x.shape[0], x.shape[0]))
        d2rl[:nsurfdofs, :nsurfdofs] = np.sqrt(constraint_weight)*self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
        H = Hnl/num_points + drl[:,None] @ drl[None,:] + drz[:,None] @ drz[None, :] + rl * d2rl 
        if self.reg is not None:
            H[:nsurfdofs, :nsurfdofs] += self.reg.d2J()
        return r, J, H
        



    def boozer_penalty_constraints(self, x, derivatives=0, constraint_weight=1., scalarize=True, optimize_G=False, weighting=None):
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

        assert derivatives in [0, 1]
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
        bs = self.bs

        s.set_dofs(sdofs)

        boozer = boozer_surface_residual(s, iota, G, bs, derivatives=derivatives, weighting=weighting)

        r = boozer[0]
        num_points = r.size 


        l = self.label.J()
        rl = (l-self.targetlabel)
        rz = (s.gamma()[0, 0, 2] - 0.)
        r = np.concatenate((r, [
            np.sqrt(constraint_weight) * rl,
            np.sqrt(constraint_weight) * rz
        ]))

        val = 0.5 * np.sum(r[:num_points]**2)/num_points + np.sum(0.5*r[num_points:]**2)
        
        if self.reg is not None:
            val+=self.reg.J()



#        if reg:
#            if weighting == 'dS' or weighting == 'dS/B':
#                wtol = 0.033
#                regweight=1e3
#                nor = s.normal()
#                num_points = nor.size
#                dA = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
#                w = np.sqrt(dA / num_points)
#                regval = 0.5*regweight*np.mean(np.maximum(wtol-w, 0)**2)
#                val+=regval
#                print('regval', regval, 'minw', np.min(w))

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
        dval = np.sum(r[:num_points, None]*J[:num_points, :], axis=0)/num_points + np.sum(r[num_points:, None]*J[num_points:, :], axis=0)

#        if reg:
#            if weighting == 'dS' or weighting == 'dS/B':
#                nor = s.normal()
#                num_points = nor.size
#                dA = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
#                w = np.sqrt(dA / num_points)
#                dA_dc = s.d_dS_by_dcoeff()
#                dw_dc = 0.5*dA_dc/np.sqrt(dA[:, :, None]*num_points)
#                dregval = regweight*np.mean(np.maximum(wtol-w[:, :, None], 0)*-dw_dc, axis=(0,1))
#                dval[:nsurfdofs]+=dregval
                
        if self.reg is not None:
            dval[:nsurfdofs]+=self.reg.dJ()
        #print(val, np.linalg.norm(dval))

        if scalarize:
            return val, dval
        else:
            return r, J

    def boozer_exact_constraints(self, xl, derivatives=0, optimize_G=True, weighting=None):
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
        bs = self.bs
        s.set_dofs(sdofs)
        nsurfdofs = sdofs.size
        nphi = s.quadpoints_phi.size
        ntheta = s.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        boozer = boozer_surface_residual_accumulate(s, iota, G, bs, derivatives=derivatives+1, weighting=weighting)
        val, dval = boozer[:2]

        if self.reg is not None:
            val += self.reg.J()
            dval[:nsurfdofs]+=self.reg.dJ()


        dl = np.zeros((xl.shape[0]-2,))
        l = self.label.J()
        dl[:nsurfdofs] = self.label.dJ_by_dsurfacecoefficients()
        drz = np.zeros((xl.shape[0]-2,))
        g = [l-self.targetlabel]
        rz = (s.gamma()[0, 0, 2] - 0.)
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]

        res = np.zeros(xl.shape)
        res[:-2] = dval/num_points- lm[-2] * dl - lm[-1] * drz
        res[-2] = g[0]
        res[-1] = rz
        if derivatives == 0:
            return res

        d2val = boozer[2]
        if self.reg is not None:
            d2val[:nsurfdofs, :nsurfdofs]+=self.reg.d2J()

        d2l = np.zeros((xl.shape[0]-2, xl.shape[0]-2))
        d2l[:nsurfdofs, :nsurfdofs] = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()

        dres = np.zeros((xl.shape[0], xl.shape[0]))
        dres[:-2, :-2] = d2val/num_points - lm[-2]*d2l
        dres[:-2, -2] = -dl
        dres[:-2, -1] = -drz

        dres[-2, :-2] = dl
        dres[-1, :-2] = drz
        return res, dres

    def minimize_boozer_penalty_constraints_LBFGS(self, tol=1e-3, maxiter=1000, constraint_weight=1., iota=0., G=None, weighting=None):
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
            x, derivatives=1, constraint_weight=constraint_weight, optimize_G=G is not None, weighting=weighting)
        res = minimize(
            fun, x, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol, 'maxcor': 200})

        if G is None:
            s.set_dofs(res.x[:-1])
            iota = res.x[-1]
        else:
            s.set_dofs(res.x[:-2])
            iota = res.x[-2]
            G = res.x[-1]

        resdict = {
            "residual": res.fun, "gradient": res.jac, "success": res.success,
            "firstorderop":res.jac,
            "iota":iota,"G":G, "iter":res.nit, "weighting":weighting, "type":"ls", "constraint_weight":constraint_weight,
            "labelerr":np.abs((self.label.J()-self.targetlabel)/self.targetlabel)
        }



        return resdict

    def minimize_boozer_penalty_constraints_BFGS(self, tol=1e-3, maxiter=1000, constraint_weight=1., iota=0., G=None, weighting=None, hessian=False):
        r"""
        This function tries to find the surface that approximately solves

        .. math::
            \text{min}_x ~J(x) + \frac{1}{2} w_c (l - l_0)^2
                                 + \frac{1}{2} w_c (z(\varphi=0, \theta=0) - 0)^2

        where :math:`J(x) = \frac{1}{2}\mathbf r(x)^T \mathbf r(x)`, and :math:`\mathbf r(x)` contains
        the Boozer residuals at quadrature points :math:`1,\dots,n`.
        This is done using BFGS.
        """
        if not self.need_to_run_code:
            return self.res
        
        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))

        fun = lambda x: self.boozer_penalty(
            x, derivatives=1, constraint_weight=constraint_weight, optimize_G=G is not None)
        res = minimize(
            fun, x, jac=True, method='BFGS',
            options={'maxiter': maxiter, 'gtol': tol})
        

        resdict = {
                "residual": res.fun, "gradient": res.jac, "iter": res.nit, "info": res, "success": res.success, "G": None, 'type':'ls', 'solver':'BFGS',
                "firstorderop":res.jac, "weighting":weighting, "constraint_weight":constraint_weight, "labelerr":np.abs((self.label.J()-self.targetlabel)/self.targetlabel),
                "reg": self.reg.J() if self.reg is not None else 0.
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

        if hessian:
            val, dval, d2val = self.boozer_penalty(
                x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None)
            P, L, U = lu(d2val)
            resdict["PLU"] = (P, L, U)
        self.res = resdict
        self.need_to_run_code = False
        
        print(f"BFGS - {resdict['success']}  iter={resdict['iter']}, iota={resdict['iota']:.16f}, ||grad||_inf = {np.linalg.norm(resdict['firstorderop'], ord=np.inf):.3e}", flush=True)
        return resdict



    def minimize_boozer_penalty_constraints_newton(self, tol=1e-12, maxiter=10, constraint_weight=1., iota=0., G=None, stab=0., weighting=None):
        """
        This function does the same as :mod:`minimize_boozer_penalty_constraints_LBFGS`, but instead of LBFGS it uses
        Newton's method.
        """
        if not self.need_to_run_code:
            return self.res
 
        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0

        val, dval, d2val = self.boozer_penalty(
            x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None)
        norm = np.linalg.norm(dval, ord=np.inf)
        #print(np.linalg.norm(dval, ord=np.inf), np.linalg.cond(d2val), flush=True)
        
        while i < maxiter and norm > tol:
            d2val += stab*np.identity(d2val.shape[0])
            dx = np.linalg.solve(d2val, dval)
            if norm < 1e-9:
                dx += np.linalg.solve(d2val, dval - d2val@dx)
            x = x - dx
            val, dval, d2val = self.boozer_penalty(
                x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None)
            norm = np.linalg.norm(dval, ord=np.inf)
            #print(np.linalg.norm(dval, ord=np.inf), np.linalg.cond(d2val), flush=True)

            if norm > 1e2:
                break

            i = i+1

        P, L, U = lu(d2val)
        if G is None:
            s.set_dofs(x[:-1])
            iota = x[-1]
        else:
            s.set_dofs(x[:-2])
            iota = x[-2]
            G = x[-1]
        res = {
            "residual": val, "jacobian": dval, "hessian": d2val, "iter": i, "success": norm <= tol, "G": None, "type": "ls",
            "PLU":(P,L,U), "firstorderop":dval, "weighting":weighting, "constraint_weight":constraint_weight, "iota":iota, "G":G,
            "type":'ls', "labelerr":np.abs((self.label.J()-self.targetlabel)/self.targetlabel), "cond":np.linalg.cond(d2val), 
            "reg": self.reg.J() if self.reg is not None else 0.
        }

        
        self.res = res
        self.need_to_run_code = False
        
        return res

    def minimize_boozer_penalty_constraints_newton_cg(self, tol=1e-3, maxiter=1000, constraint_weight=1., iota=0., G=None, weighting=None, hessian=False):
        r"""
        This function tries to find the surface that approximately solves

        .. math::
            \text{min}_x ~J(x) + \frac{1}{2} w_c (l - l_0)^2
                                 + \frac{1}{2} w_c (z(\varphi=0, \theta=0) - 0)^2

        where :math:`J(x) = \frac{1}{2}\mathbf r(x)^T \mathbf r(x)`, and :math:`\mathbf r(x)` contains
        the Boozer residuals at quadrature points :math:`1,\dots,n`.
        This is done using BFGS.
        """
        if not self.need_to_run_code:
            return self.res
        
        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        
        
        def fun(x):
            f = self.boozer_penalty_constraints_accumulate(x, derivatives=0, constraint_weight=constraint_weight, optimize_G=G is not None, weighting=weighting)
            return f
        def jac(x):
            fg = self.boozer_penalty_constraints_accumulate(x, derivatives=1, constraint_weight=constraint_weight, optimize_G=G is not None, weighting=weighting)
            return fg[-1] 
        def hess(x):
            fgh = self.boozer_penalty_constraints_accumulate(x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None, weighting=weighting)
            return fgh[-1] 

        #fun = lambda x: self.boozer_penalty_constraints_accumulate(
        #    x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None, weighting=weighting)
        res = minimize(fun, x, jac=jac, hess=hess, method='Newton-CG',options={'maxiter': maxiter, 'xtol':1e-20})
        

        resdict = {
                "residual": res.fun, "gradient": res.jac, "iter": res.nit, "info": res, "success": res.success, "G": None, 'type':'ls', 'solver':'BFGS',
                "firstorderop":res.jac, "weighting":weighting, "constraint_weight":constraint_weight, "labelerr":np.abs((self.label.J()-self.targetlabel)/self.targetlabel)
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

        if hessian:
            val, dval, d2val = self.boozer_penalty_constraints_accumulate(
                x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None, weighting=weighting)
            P, L, U = lu(d2val)
            resdict["PLU"] = (P, L, U)
        self.res = resdict
        self.need_to_run_code = False
        
        #print(f"BFGS - {resdict['success']}  iter={resdict['iter']}, iota={resdict['iota']:.16f}, ||grad||_inf = {np.linalg.norm(resdict['firstorderop'], ord=np.inf):.3e}", flush=True)
        return resdict





    def minimize_boozer_penalty_constraints_ls(self, tol=1e-10, maxiter=10, constraint_weight=1., iota=0., G=None, method='lm', hessian=False, weighting=None):
        """
        This function does the same as :mod:`minimize_boozer_penalty_constraints_LBFGS`, but instead of LBFGS it
        uses a nonlinear least squares algorithm when ``method='lm'``.  Options for the method 
        are the same as for :mod:`scipy.optimize.least_squares`. If ``method='manual'``, then a 
        damped Gauss-Newton method is used.
        """
        if not self.need_to_run_code:
            return self.res
        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))

        nphi = s.quadpoints_phi.size
        ntheta = s.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        nsurfdofs = s.x.size

        norm = 1e10
        i = 0
        lam = 1.
        r, J = self.boozer_penalty_constraints(
            x, derivatives=1, constraint_weight=constraint_weight, scalarize=False, optimize_G=G is not None, weighting=weighting)
        
        r[:num_points]/=np.sqrt(num_points)
        J[:num_points, :]/=np.sqrt(num_points)
        
        b = J.T@r
        JTJ = J.T@J
        while i < maxiter and norm > tol:
            dx = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), b)
            x -= dx
            r, J = self.boozer_penalty_constraints(
                x, derivatives=1, constraint_weight=constraint_weight, scalarize=False, optimize_G=G is not None, weighting=weighting)
            r[:num_points]/=np.sqrt(num_points)
            J[:num_points, :]/=np.sqrt(num_points)
            

            b = J.T@r
            JTJ = J.T@J
            if self.reg is not None:
                b[:nsurfdofs]+=self.reg.dJ()
                JTJ[:nsurfdofs, :nsurfdofs]+=self.reg.dJ()[:, None] * self.reg.dJ()[None, :]

            norm = np.linalg.norm(b, ord=np.inf)
            lam *= 1/3
            i += 1

        if G is None:
            s.set_dofs(x[:-1])
            iota = x[-1]
        else:
            s.set_dofs(x[:-2])
            iota = x[-2]
            G = x[-1]
        
        resdict = {
            "residual": r, "gradient": b, "success": norm <= tol,
            "firstorderop":b,
            "iota":iota,"G":G, "iter":i, "weighting":weighting, "type":"ls", "constraint_weight":constraint_weight,
            "labelerr":np.abs((self.label.J()-self.targetlabel)/self.targetlabel)
        }
        #print(f"LVM - {resdict['success']}  iter={resdict['iter']}, iota={resdict['iota']:.16f}, ||grad||_inf = {np.linalg.norm(resdict['firstorderop'], ord=np.inf):.3e}", flush=True)
        if self.reg is not None:
            resdict['residual'] = np.linalg.norm(r)**2 + self.reg.J()

        self.res=resdict
        if hessian:
            val, dval, d2val = self.boozer_penalty_constraints_accumulate(
                x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None, weighting=weighting)
            P, L, U = lu(d2val)
            resdict["PLU"] = (P, L, U)
        return resdict


    def minimize_boozer_exact_constraints_newton(self, tol=1e-12, maxiter=10, iota=0., G=None, lm=[0., 0.], weighting=None):
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
        if not self.need_to_run_code:
            return self.res

        s = self.surface
        if G is not None:
            xl = np.concatenate((s.get_dofs(), [iota, G], lm))
        else:
            xl = np.concatenate((s.get_dofs(), [iota], lm))
        val, dval = self.boozer_exact_constraints(xl, derivatives=1, optimize_G=G is not None, weighting=weighting)

        norm = np.linalg.norm(val, ord=np.inf)
        print(np.linalg.norm(val, ord=np.inf), np.linalg.cond(dval[:-1, :-1]), flush=True)
        i = 0

        if s.stellsym:
            b = val[:-1]
        else:
            b = val

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
            val, dval = self.boozer_exact_constraints(xl, derivatives=1, optimize_G=G is not None, weighting=weighting)
            norm = np.linalg.norm(val, ord=np.inf)
            i = i + 1
            print(np.linalg.norm(val, ord=np.inf), np.linalg.cond(dval[:-1, :-1]), flush=True)

        if s.stellsym:
            lm = xl[-2]
            P, L, U = lu(dval[:-1,:-1])
        else:
            lm = xl[-2:]
            P, L, U = lu(dval)
        
        if G is not None:
            x = xl[:-2]
            s.set_dofs(xl[:-4])
            iota = xl[-4]
            G = xl[-3]
        else:
            x = xl[:-2]
            s.set_dofs(xl[:-3])
            iota = xl[-3]
        
        r = self.boozer_penalty_constraints(
            x, derivatives=0, constraint_weight=0, scalarize=True, optimize_G=G is not None, weighting=weighting)

        res = {
            "residual": r, "gradient": val, "success": norm <= tol,
            "iota":iota,"G":G, "iter":i, "weighting":weighting, "type":"lscons", "PLU": (P, L, U),
            "firstorderop":val, "labelerr":np.abs((self.label.J()-self.targetlabel)/self.targetlabel), 
        }
        self.res = res
        self.need_to_run_code = False
        
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
        if not self.need_to_run_code:
            return self.res
        
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
            G = 2. * np.pi * np.sum(np.abs(self.bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
        x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0
        r, J = boozer_surface_residual(s, iota, G, self.bs, derivatives=1)
        norm = 1e6
        while i < maxiter:
            if s.stellsym:
                b = np.concatenate((r[mask], [(label.J()-self.targetlabel)]))
            else:
                b = np.concatenate((r[mask], [(label.J()-self.targetlabel), s.gamma()[0, 0, 2]]))
            norm = np.linalg.norm(b, ord=np.inf)
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
            r, J = boozer_surface_residual(s, iota, G, self.bs, derivatives=1)
        J = np.vstack((
            J[mask, :],
            np.concatenate((label.dJ_by_dsurfacecoefficients(), [0., 0.])),
        ))
        
        P, L, U = lu(J)
        res = {
                "residual": r, "jacobian": J, "iter": i, "success": norm <= tol, "G": G, "s": s, "iota": iota, "type": "exact", "PLU":(P,L,U),
                "mask":mask, "solver":"NEWTON"
        }
        
        self.res = res
        self.need_to_run_code=False
        return res
