from scipy.optimize import minimize
import numpy as np
from simsopt.geo.surfaceobjectives import ToroidalFlux 
from simsopt.geo.surfaceobjectives import BoozerSurfaceResidual 


class BoozerSurface():
    def __init__(self, biotsavart, surface, label, targetlabel):
        self.bs = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel

    def BoozerConstrainedScalarized(self,x, derivatives = 0,  constraint_weight = 1.):
        sdofs = x[:-1]
        iota = x[-1]
        s = self.surface
        bs = self.bs

        s.set_dofs(sdofs)

        if derivatives == 0:
            r = BoozerSurfaceResidual(s, iota, bs, derivatives = 0)
            
            l = self.label.J()
            rl = (l-self.targetlabel) 
            r = np.concatenate( (r, [np.sqrt(constraint_weight) *rl]) )
            
            val = 0.5 * np.sum(r**2)  
            return val

        elif derivatives == 1:
            r, Js, Jiota = BoozerSurfaceResidual(s, iota, bs, derivatives = 1)
            J = np.concatenate((Js, Jiota), axis=1)
            dl  = np.zeros( x.shape )
            d2l = np.zeros( (x.shape[0], x.shape[0] ) )
 
            l = self.label.J()
            dl[:-1]      = self.label.dJ_by_dsurfacecoefficients()
            
            rl = (l-self.targetlabel) 
            r = np.concatenate( (r, [np.sqrt(constraint_weight) *rl]) )
            J = np.concatenate( (J,  np.sqrt(constraint_weight) *dl[None,:]),  axis = 0)
            
            val = 0.5 * np.sum(r**2) 
            dval = np.sum(r[:, None]*J, axis=0) 

            return val,dval

        elif derivatives == 2:
            r, Js, Jiota, Hs, Hsiota, Hiota = BoozerSurfaceResidual(s, iota, bs, derivatives = 2)
            J = np.concatenate((Js, Jiota), axis=1)
            Htemp = np.concatenate((Hs,Hsiota[...,None]),axis=2)
            col = np.concatenate( (Hsiota, Hiota), axis = 1)
            H = np.concatenate( (Htemp,col[...,None,:]), axis = 1) 
 
            dl  = np.zeros( x.shape )
            d2l = np.zeros( (x.shape[0], x.shape[0] ) )
    
            l            = self.label.J()
            dl[:-1]      = self.label.dJ_by_dsurfacecoefficients()
            d2l[:-1,:-1] = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients() 
    
            rl = l-self.targetlabel 
            r = np.concatenate( (r, [np.sqrt(constraint_weight) *rl]) )
            J = np.concatenate( (J,  np.sqrt(constraint_weight) *dl[None,:]),  axis = 0)
            H = np.concatenate( (H,  np.sqrt(constraint_weight) *d2l[None,:,:]), axis = 0)
    
            val = 0.5 * np.sum(r**2) 
            dval = np.sum(r[:, None]*J, axis=0) 
            d2val = J.T @ J + np.sum( r[:,None,None] * H, axis = 0) 
            return val, dval, d2val
        else:
            raise "Not implemented"


    def BoozerConstrained(self,xl):
        sdofs = xl[:-2]
        iota = xl[-2]
        lm = xl[-1]
        s.set_dofs(sdofs)
        r, Js, Jiota, Hs, Hsiota, Hiota = BoozerSurfaceResidual(s, iota, bs, derivatives = 2)
        J = np.concatenate((Js, Jiota), axis=1)
        Htemp = np.concatenate((Hs,Hsiota[...,None]),axis=2)
        col = np.concatenate( (Hsiota, Hiota), axis = 1)
        H = np.concatenate( (Htemp,col[...,None,:]), axis = 1) 
     
        dl  = np.zeros( (xl.shape[0]-1,) )
        d2l = np.zeros( (xl.shape[0]-1, xl.shape[0]-1 ) )
    
        tf.invalidate_cache()
        l            = tf.J()
        dl[:-1]      = tf.dJ_by_dsurfacecoefficients()
        d2l[:-1,:-1] = tf.d2J_by_dsurfacecoefficientsdsurfacecoefficients() 
        g = [l-tf0]
    
        res = np.zeros( xl.shape )
        res[:-1] = np.sum(r[:, None]*J, axis=0) - lm * dl 
        res[-1]  = g[0]
        
        dres = np.zeros( (xl.shape[0], xl.shape[0]) )
        dres[:-1,:-1] = J.T @ J + np.sum( r[:,None,None] * H, axis = 0) - lm*d2l
        dres[:-1, -1] = -dl
        dres[ -1,:-1] =  dl
        
        return res,dres
 
#    def minimizeBoozerScalarizedLBFGS(tol = 1e-3, maxiter = 1000, constraint_weight = 1.):
#    """
#    This function tries to find the surface that approximately solves
#    min || f(x) ||^2_2 + cweight * (label - labeltarget)^2
#    where || f(x)||^2_2 is the sum of squares of the Boozer residual at
#    the quadrature points.  This is done using LBFGS.
#    """
#    def callback(x, state):
#        print(np.linalg(x))
#    x = np.concatenate((s.get_dofs(), [iota]))
#    res = minimize(BoozerConstrainedScalarized_Gradient, x, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'tol' : tol, 'callback' : callback})
#    s.set_dofs(res.x[:-1]) 
#
#
#    def minimizeBoozerScalarizedNewton(tol = 1e-13, maxiter = 10):
#    """
#    This function does the same as the above, but instead of LBFGS it uses
#    Newton's method.
#    """
#       
#    for i in range(10):
#        val,dval,d2val = f(x)
#        dx = np.linalg.solve(d2val,dval)
#        x = x - dx
#        print(np.linalg.norm(dval) )
#    
#    s.set_dofs(res.x[:-1])
#
#
#    def minimizeBoozerConstraintedNewton(tol = 1e-13, maxiter = 10):
#    """
#    This function solves the constrained optimization problem
#    min || f(x) ||^2_2
#    subject to label - targetlabel = 0
#    using Lagrange multipliers and Newton's method.
#    """
#    # \nabla L = 0
#    # \nabla f - \lambda \nabla g = 0
#    # g = 0
#   
#    xl = np.concatenate( (x, [0.]) )
#    for i in range(10):
#        val,dval = f(xl)
#        dx = np.linalg.solve(dval,val)
#        xl = xl - dx
#        print(np.linalg.norm(val) )
#



