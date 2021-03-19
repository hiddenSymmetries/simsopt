import numpy as np
class ToroidalFlux(object):

    r"""
    This objective calculates
        J = \int_{varphi = constant} B \cdot n ds
          = \int_{varphi = constant} curlA \cdot n ds
          from Stokes' theorem
          = \int_{curve on surface where varphi = constant} A \cdot n dl
    given a surface and Biot Savart kernel.
    """

    def __init__(self, surface, biotsavart, idx = 0):
        self.surface = surface
        self.biotsavart = biotsavart
        self.idx = idx 
        self.surface.dependencies.append(self)

    def invalidate_cache(self):
        x = self.surface.gamma()[self.idx]
        self.biotsavart.set_points(x)
        
    def J(self):
        xtheta = self.surface.gammadash2()[self.idx]
        ntheta = self.surface.gamma().shape[1]
        
        A = self.biotsavart.A()
        tf = np.sum(A * xtheta)/ntheta
        return tf


    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients
        """
        ntheta = self.surface.gamma().shape[1]
        A = self.biotsavart.A()
        dA_by_dX = self.biotsavart.dA_by_dX()
        dgammadash2 = self.surface.gammadash2()[self.idx,:]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx,:]

        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        dA_dc = np.sum( dA_by_dX[...,:,None] * dx_dc[...,None,:],axis=1)
        term1 = np.sum( dA_dc * dgammadash2[...,None], axis = (0,1) ) 
        term2 = np.sum( A[...,None] * dgammadash2_by_dc, axis = (0,1) )

        out = (term1+term2)/ntheta
        return out
    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivatives with respect to the surface coefficients
        """
        ntheta = self.surface.gamma().shape[1]
        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        dA_by_dX = self.biotsavart.dA_by_dX()
        d2A_by_dXdX = self.biotsavart.d2A_by_dXdX().reshape((ntheta,3,3,3))
        dA_dc = np.sum(dA_by_dX[...,:,None] * dx_dc[...,None,:],axis=1)
        d2A_dcdc = np.einsum('jkpl,jpn,jkm->jlmn', d2A_by_dXdX, dx_dc, dx_dc)

        dgammadash2 = self.surface.gammadash2()[self.idx]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx]

        term1 = np.sum(d2A_dcdc * dgammadash2[...,None,None],axis = -3)
        term2 = np.sum(dA_dc[...,:,None] * dgammadash2_by_dc[...,None,:], axis = -3)
        term3 = np.sum(dA_dc[...,None,:] * dgammadash2_by_dc[...,:,None], axis = -3)
        
        out = (1/ntheta) * np.sum(term1+term2+term3, axis = 0)
        return out



