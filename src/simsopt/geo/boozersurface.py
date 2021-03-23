from scipy.optimize import minimize
import numpy as np
from simsopt.geo.surfaceobjectives import ToroidalFlux 
from simsopt.geo.surfaceobjectives import BoozerSurfaceResidual 
import ipdb

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
            ry = (s.gamma()[0,0,1] - 0.)
            rz = (s.gamma()[0,0,2] - 0.)
            r = np.concatenate( (r, [np.sqrt(constraint_weight) *rl,np.sqrt(constraint_weight) *ry,np.sqrt(constraint_weight) *rz]) )
            
            val = 0.5 * np.sum(r**2)  
            return val

        elif derivatives == 1:
            r, Js, Jiota = BoozerSurfaceResidual(s, iota, bs, derivatives = 1)
            J = np.concatenate((Js, Jiota), axis=1)
            dl  = np.zeros( x.shape )
            dry  = np.zeros( x.shape )
            drz  = np.zeros( x.shape )
            d2l = np.zeros( (x.shape[0], x.shape[0] ) )
 
            l = self.label.J()
            dl[:-1]      = self.label.dJ_by_dsurfacecoefficients()
            
            rl = (l-self.targetlabel) 
            ry = (s.gamma()[0,0,1] - 0.)
            rz = (s.gamma()[0,0,2] - 0.)
            dry[:-1] = s.dgamma_by_dcoeff()[0,0,1,:]
            drz[:-1] = s.dgamma_by_dcoeff()[0,0,2,:]
            

            r = np.concatenate( (r, [np.sqrt(constraint_weight) *rl,np.sqrt(constraint_weight) *ry,np.sqrt(constraint_weight) *rz]) )
            J = np.concatenate( (J,  np.sqrt(constraint_weight) *dl[None,:] ,np.sqrt(constraint_weight) *dry[None,:],np.sqrt(constraint_weight) *drz[None,:]  ),  axis = 0)
            
            val = 0.5 * np.sum(r**2) 
            dval = np.sum(r[:, None]*J, axis=0) 
            #print(val, np.linalg.norm(dval))    
            return val,dval

        elif derivatives == 2:
            r, Js, Jiota, Hs, Hsiota, Hiota = BoozerSurfaceResidual(s, iota, bs, derivatives = 2)
            J = np.concatenate((Js, Jiota), axis=1)
            Htemp = np.concatenate((Hs,Hsiota[...,None]),axis=2)
            col = np.concatenate( (Hsiota, Hiota), axis = 1)
            H = np.concatenate( (Htemp,col[...,None,:]), axis = 1) 
 
            dry  = np.zeros( x.shape )
            drz  = np.zeros( x.shape )
            dl  = np.zeros( x.shape )
            d2l = np.zeros( (x.shape[0], x.shape[0] ) )
    
            l            = self.label.J()
            dl[:-1]      = self.label.dJ_by_dsurfacecoefficients()
            d2l[:-1,:-1] = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients() 
    
            rl = l-self.targetlabel 
            ry = (s.gamma()[0,0,1] - 0.)
            rz = (s.gamma()[0,0,2] - 0.)
            dry[:-1] = s.dgamma_by_dcoeff()[0,0,1,:]
            drz[:-1] = s.dgamma_by_dcoeff()[0,0,2,:]
            r = np.concatenate( (r, [np.sqrt(constraint_weight) *rl,np.sqrt(constraint_weight) *ry,np.sqrt(constraint_weight) *rz]) )
            J = np.concatenate( (J,  np.sqrt(constraint_weight) *dl[None,:] ,np.sqrt(constraint_weight) *dry[None,:],np.sqrt(constraint_weight) *drz[None,:]  ),  axis = 0)
            H = np.concatenate( (H,  np.sqrt(constraint_weight) *d2l[None,:,:], np.zeros(d2l[None,:,:].shape), np.zeros(d2l[None,:,:].shape) ), axis = 0)
    
            val = 0.5 * np.sum(r**2) 
            dval = np.sum(r[:, None]*J, axis=0) 
            d2val = J.T @ J + np.sum( r[:,None,None] * H, axis = 0) 
            return val, dval, d2val
        else:
            raise "Not implemented"


    def BoozerConstrained(self,xl, derivatives = 0):
        sdofs = xl[:-4]
        iota = xl[-4]
        lm = xl[-3:]
        s = self.surface
        bs = self.bs
        s.set_dofs(sdofs)
        
        if derivatives == 0:
            r, Js, Jiota = BoozerSurfaceResidual(s, iota, bs, derivatives = 1)
            J = np.concatenate((Js, Jiota), axis=1)
            dl  = np.zeros( (xl.shape[0]-3,) )
    
            l            = self.label.J()
            dl[:-1]      = self.label.dJ_by_dsurfacecoefficients()
            dry  = np.zeros((xl.shape[0]-3,) )
            drz  = np.zeros((xl.shape[0]-3,) )
            g = [l-self.targetlabel]
            ry = (s.gamma()[0,0,1] - 0.)
            rz = (s.gamma()[0,0,2] - 0.)
            dry[:-1] = s.dgamma_by_dcoeff()[0,0,1,:]
            drz[:-1] = s.dgamma_by_dcoeff()[0,0,2,:]
    
            res = np.zeros( xl.shape )
            res[:-3] = np.sum(r[:, None]*J, axis=0)  - lm[-3] * dl - lm[-2] * dry - lm[-1] * drz 
            res[-3]  = g[0]
            res[-2] = ry
            res[-1] = rz
            return res
        elif derivatives == 1:
            r, Js, Jiota, Hs, Hsiota, Hiota = BoozerSurfaceResidual(s, iota, bs, derivatives = 2)
            J = np.concatenate((Js, Jiota), axis=1)
            Htemp = np.concatenate((Hs,Hsiota[...,None]),axis=2)
            col = np.concatenate( (Hsiota, Hiota), axis = 1)
            H = np.concatenate( (Htemp,col[...,None,:]), axis = 1) 
     
            dl  =  np.zeros((xl.shape[0]-3,) )
            dry  = np.zeros((xl.shape[0]-3,) )
            drz  = np.zeros((xl.shape[0]-3,) )
            d2l =  np.zeros((xl.shape[0]-3, xl.shape[0]-3) )
    
            l            = self.label.J()
            dl[:-1]      = self.label.dJ_by_dsurfacecoefficients()
            d2l[:-1,:-1] = self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients() 
            g = [l-self.targetlabel]
            ry = (s.gamma()[0,0,1] - 0.)
            rz = (s.gamma()[0,0,2] - 0.)
            dry[:-1] = s.dgamma_by_dcoeff()[0,0,1,:]
            drz[:-1] = s.dgamma_by_dcoeff()[0,0,2,:]
            

            res = np.zeros( xl.shape )
            res[:-3] = np.sum(r[:, None]*J, axis=0) - lm[-3] * dl - lm[-2] * dry - lm[-1] * drz 
            res[-3]  = g[0]
            res[-2]  = ry 
            res[-1]  = rz
            
            dres = np.zeros( (xl.shape[0], xl.shape[0]) )
            dres[:-3,:-3] = J.T @ J + np.sum( r[:,None,None] * H, axis = 0) - lm[-3]*d2l
            dres[:-3, -3] = -dl
            dres[:-3, -2] = -dry
            dres[:-3, -1] = -drz

            dres[ -3,:-3] =  dl
            dres[ -2,:-3] =  dry
            dres[ -1,:-3] =  drz
            return res,dres
 
    def minimizeBoozerScalarizedLBFGS(self,tol = 1e-3, maxiter = 1000, constraint_weight = 1., iota = 0.):
        """
        This function tries to find the surface that approximately solves
        min || f(x) ||^2_2 + 0.5 * constraint_weight * (label - labeltarget)^2
        +constraint_weight * (y(varphi=0,theta=0) -0)^2
        +constraint_weight * (z(varphi=0,theta=0) -0)^2
        where || f(x)||^2_2 is the sum of squares of the Boozer residual at
        the quadrature points.  This is done using LBFGS. 
        """

        s = self.surface
        x = np.concatenate((s.get_dofs(), [iota]))
        res = minimize(lambda x: self.BoozerConstrainedScalarized(x,derivatives = 1, constraint_weight = constraint_weight), \
                x, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol' : tol,'gtol' : tol, 'maxcor' : 50})
        print(res.fun)
        s.set_dofs(res.x[:-1])
        iota = res.x[-1]
        return s,iota
       

    def minimizeBoozerScalarizedNewton(self,tol = 1e-12, maxiter = 10,constraint_weight = 1., iota = 0.):
        """
        This function does the same as the above, but instead of LBFGS it uses
        Newton's method.
        """
        s = self.surface 
        x = np.concatenate((s.get_dofs(), [iota]))
        i = 0
        
        val,dval = self.BoozerConstrainedScalarized(x, derivatives = 1, constraint_weight = constraint_weight)
        norm =np.linalg.norm(dval) 
        while i < maxiter and norm > tol:
            val,dval,d2val = self.BoozerConstrainedScalarized(x, derivatives = 2, constraint_weight = constraint_weight)
            dx = np.linalg.solve(d2val,dval)
            x = x - dx
            norm = np.linalg.norm(dval) 
            print(norm)
            i = i +1
        s.set_dofs(x[:-1])
        iota = x[-1]
        return s, iota

    def minimizeBoozerConstrainedNewton(self, tol = 1e-12, maxiter = 10, iota = 0., lm = [0.,0.,0.] ):
        """
        This function solves the constrained optimization problem
        min || f(x) ||^2_2
        subject to 
        label - targetlabel = 0
        y(varphi=0,theta=0) - 0 = 0
        z(varphi=0,theta=0) - 0 = 0
        using Lagrange multipliers and Newton's method.
        """
        s = self.surface 
        xl = np.concatenate( (s.get_dofs(), [iota], lm) )
        val,dval = self.BoozerConstrained(xl, derivatives = 1)
        norm =np.linalg.norm(val) 
        i = 0   
        while i < maxiter and norm > tol:
            val,dval = self.BoozerConstrained(xl, derivatives = 1)
            if s.stellsym:
                dx = np.linalg.solve(dval[:-2,:-2],val[:-2])
                xl[:-2] = xl[:-2] - dx
            else:
                dx = np.linalg.solve(dval,val)
                xl = xl - dx
            norm = np.linalg.norm(val)
            print( norm )
            i = i + 1
        s.set_dofs(xl[:-4])
        iota = xl[-4]
        lm = xl[-3:]
        return s, iota, lm


