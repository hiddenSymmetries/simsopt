from scipy.optimize import minimize, least_squares
import numpy as np
from simsopt.geo.surfaceobjectives import boozer_surface_residual

class BoozerSurface():
    def __init__(self, biotsavart, surface, label, targetlabel):
        self.bs = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel

    def boozer_penalty_constraints(self, x, derivatives=0,  constraint_weight=1., scalarize=True, optimize_G=False):
        """
        Define the residual

        r(x) = [
            f_1(x),...,f_n(x),
            sqrt(constraint_weight) * (label-targetlabel),
            sqrt(constraint_weight) * (y(varphi=0, theta=0) - 0),
            sqrt(constraint_weight) * (z(varphi=0, theta=0) - 0),
        ]
        
        where {f_i}_i are the Boozer residuals at quadrature points 1,...,n.

        For scalarized=False, this function returns r(x) and optionally the Jacobian of r.

        for scalarized=True, this function returns 

            g(x) = 0.5 * r(x)^T * r(x),

        i.e. the least squares residual and optionally the gradient and the Hessian of g.
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

        s.set_dofs(sdofs)

        boozer = boozer_surface_residual(s, iota, G, bs, derivatives=derivatives)

        r = boozer[0]
        
        l = self.label.J()
        rl = (l-self.targetlabel)
        ry = (s.gamma()[0,0,1] - 0.)
        rz = (s.gamma()[0,0,2] - 0.)
        r = np.concatenate( (r, [np.sqrt(constraint_weight) *rl,np.sqrt(constraint_weight) *ry,np.sqrt(constraint_weight) *rz]) )
        
        val = 0.5 * np.sum(r**2)  
        if derivatives == 0:
            if scalarize:
                return val
            else:
                return r

        J = boozer[1]

        dl  = np.zeros( x.shape )
        dry  = np.zeros( x.shape )
        drz  = np.zeros( x.shape )

        dl[:nsurfdofs]      = self.label.dJ_by_dsurfacecoefficients()
        
        dry[:nsurfdofs] = s.dgamma_by_dcoeff()[0,0,1,:]
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0,0,2,:]
        J = np.concatenate( (J,  np.sqrt(constraint_weight) *dl[None,:] ,np.sqrt(constraint_weight) *dry[None,:],np.sqrt(constraint_weight) *drz[None,:]  ),  axis = 0)
        dval = np.sum(r[:, None]*J, axis=0) 
        if derivatives == 1:
            if scalarize:
                return val, dval
            else:
                return r, J
        if not scalarize:
            raise NotImplementedError('Can only return Hessian for scalarized version.')

        H = boozer[2]

        d2l = np.zeros( (x.shape[0], x.shape[0] ) )
        d2l[:nsurfdofs,:nsurfdofs] = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients() 

        H = np.concatenate( (H,  np.sqrt(constraint_weight) *d2l[None,:,:], np.zeros(d2l[None,:,:].shape), np.zeros(d2l[None,:,:].shape) ), axis = 0)
        d2val = J.T @ J + np.sum( r[:,None,None] * H, axis = 0) 
        return val, dval, d2val


    def boozer_exact_constraints(self, xl, derivatives=0, optimize_G=True):
        """
        This function returns the optimality conditions corresponding to the minimisation problem

            min 0.5 * || f(x) ||^2_2

            subject to 

            label - targetlabel = 0
            y(varphi=0,theta=0) - 0 = 0
            z(varphi=0,theta=0) - 0 = 0

        as well as optionally the first derivatives for these optimality conditions.
        """
        assert derivatives in [0, 1]
        if optimize_G:
            sdofs = xl[:-5]
            iota = xl[-5]
            G = xl[-4]
        else:
            sdofs = xl[:-4]
            iota = xl[-4]
            G = None
        lm = xl[-3:]
        s = self.surface
        bs = self.bs
        s.set_dofs(sdofs)
        nsurfdofs = sdofs.size

        boozer = boozer_surface_residual(s, iota, G, bs, derivatives=derivatives+1)
        r, J = boozer[0:2]

        dl  = np.zeros( (xl.shape[0]-3,) )

        l             = self.label.J()
        dl[:nsurfdofs] = self.label.dJ_by_dsurfacecoefficients()
        dry  = np.zeros((xl.shape[0]-3,) )
        drz  = np.zeros((xl.shape[0]-3,) )
        g = [l-self.targetlabel]
        ry = (s.gamma()[0,0,1] - 0.)
        rz = (s.gamma()[0,0,2] - 0.)
        dry[:nsurfdofs] = s.dgamma_by_dcoeff()[0,0,1,:]
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0,0,2,:]

        res = np.zeros( xl.shape )
        res[:-3] = np.sum(r[:, None]*J, axis=0)  - lm[-3] * dl - lm[-2] * dry - lm[-1] * drz 
        res[-3]  = g[0]
        res[-2] = ry
        res[-1] = rz
        if derivatives == 0:
            return res

        H = boozer[2]
 
        d2l =  np.zeros((xl.shape[0]-3, xl.shape[0]-3) )
        d2l[:nsurfdofs,:nsurfdofs] = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients() 
        
        dres = np.zeros( (xl.shape[0], xl.shape[0]) )
        dres[:-3,:-3] = J.T @ J + np.sum( r[:,None,None] * H, axis = 0) - lm[-3]*d2l
        dres[:-3, -3] = -dl
        dres[:-3, -2] = -dry
        dres[:-3, -1] = -drz

        dres[ -3,:-3] =  dl
        dres[ -2,:-3] =  dry
        dres[ -1,:-3] =  drz
        return res, dres
 
    def minimize_boozer_penalty_constraints_LBFGS(self, tol=1e-3, maxiter=1000, constraint_weight=1., iota=0., G=None):
        """
        This function tries to find the surface that approximately solves

        min 0.5 * || f(x) ||^2_2 + 0.5 * constraint_weight * (label - labeltarget)^2
                                 + 0.5 * constraint_weight * (y(varphi=0, theta=0) - 0)^2
                                 + 0.5 * constraint_weight * (z(varphi=0, theta=0) - 0)^2

        where || f(x)||^2_2 is the sum of squares of the Boozer residual at
        the quadrature points.  This is done using LBFGS. 
        """

        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        res = minimize(lambda x: self.boozer_penalty_constraints(x,derivatives = 1, constraint_weight = constraint_weight, optimize_G=G is not None), \
                x, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol' : tol,'gtol' : tol, 'maxcor' : 200})

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
        This function does the same as the above, but instead of LBFGS it uses
        Newton's method.
        """
        s = self.surface 
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0
        
        val, dval, d2val = self.boozer_penalty_constraints(x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None)
        norm = np.linalg.norm(dval) 
        while i < maxiter and norm > tol:
            d2val += stab*np.identity(d2val.shape[0])
            dx = np.linalg.solve(d2val, dval)
            if norm < 1e-9:
                dx += np.linalg.solve(d2val, dval - d2val@dx)
            x = x - dx
            val, dval, d2val = self.boozer_penalty_constraints(x, derivatives=2, constraint_weight=constraint_weight, optimize_G=G is not None)
            norm = np.linalg.norm(dval) 
            print("norm", norm)
            i = i+1

        res = { 
            "residual": val, "jacobian": dval, "hessian": d2val, "iter": i, "success": norm <= tol, "G": None,
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

    def minimize_boozer_penalty_constraints_ls(self, tol=1e-12, maxiter=10,constraint_weight=1., iota=0., G=None, method='lm'):
        """
        This function does the same as the above, but instead of LBFGS it uses a nonlinear least squares algorithm.
        Options for method are the same as for scipy.optimize.least_squares.
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
            r, J = self.boozer_penalty_constraints(x, derivatives=1, constraint_weight=constraint_weight, scalarize=False, optimize_G=G is not None)
            b = J.T@r
            JTJ = J.T@J 
            while i < maxiter and norm > tol:
                dx = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), b)
                x -= dx
                r, J = self.boozer_penalty_constraints(x, derivatives=1, constraint_weight=constraint_weight, scalarize=False, optimize_G=G is not None)
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
        fun = lambda x: self.boozer_penalty_constraints(x, derivatives=0, constraint_weight=constraint_weight, scalarize=False, optimize_G=G is not None)
        jac = lambda x: self.boozer_penalty_constraints(x, derivatives=1, constraint_weight=constraint_weight, scalarize=False, optimize_G=G is not None)[1]
        res = least_squares(fun, x, jac=jac, method=method, ftol=tol, xtol=tol, gtol=tol, x_scale=1.0, max_nfev=maxiter)
        resdict = { 
            "info": res, "residual": res.fun, "gradient": res.grad, "jacobian": res.jac, "success": res.status > 0,  "G": None,
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

    def minimize_boozer_exact_constraints_newton(self, tol=1e-12, maxiter=10, iota=0., G=None, lm=[0.,0.,0.]):
        """
        This function solves the constrained optimization problem

            min 0.5 * || f(x) ||^2_2

            subject to 

            label - targetlabel = 0
            y(varphi=0,theta=0) - 0 = 0
            z(varphi=0,theta=0) - 0 = 0

        using Lagrange multipliers and Newton's method.  The final two constraints
        are not necessary for stellarator symmetric surfaces as they are automatically
        satisfied by the SurfaceXYZFourier parameterization.
        """
        s = self.surface 
        if G is not None:
            xl = np.concatenate( (s.get_dofs(), [iota, G], lm) )
        else:
            xl = np.concatenate( (s.get_dofs(), [iota], lm) )
        val, dval = self.boozer_exact_constraints(xl, derivatives=1, optimize_G=G is not None)
        norm = np.linalg.norm(val) 
        i = 0   
        while i < maxiter and norm > tol:
            if s.stellsym:
                A = dval[:-2,:-2]
                b = val[:-2]
                dx = np.linalg.solve(A, b)
                if norm < 1e-9: # iterative refinement for higher accuracy. TODO: cache LU factorisation
                    dx += np.linalg.solve(A, b-A@dx)
                xl[:-2] = xl[:-2] - dx
            else:
                dx = np.linalg.solve(dval, val)
                if norm < 1e-9: # iterative refinement for higher accuracy. TODO: cache LU factorisation
                    dx += np.linalg.solve(dval, val-dval@dx)
                xl = xl - dx
            val, dval = self.boozer_exact_constraints(xl, derivatives=1, optimize_G=G is not None)
            norm = np.linalg.norm(val)
            i = i + 1

        if s.stellsym:
            lm = xl[-3]
        else:
            lm = xl[-3:]

        res = { 
            "residual": val, "jacobian": dval, "iter": i, "success": norm <= tol, "lm": lm, "G": None,
        }
        if G is not None:
            s.set_dofs(xl[:-5])
            iota = xl[-5]
            G = xl[-4]
            res['G'] = G
        else:
            s.set_dofs(xl[:-4])
            iota = xl[-4]
        res['s'] = s
        res['iota'] = iota
        return res
