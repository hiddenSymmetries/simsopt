from typing import Any

import numpy as np
from nptyping import NDArray, Float

import simsoptpp as sopp
from .._core.graph_optimizable import Optimizable
from simsopt.geo.surface import Surface
import gc

class Area(Optimizable):
    """
    Wrapper class for surface area label.
    """

    def __init__(self, surface):
        self.surface = surface
        super().__init__(depends_on=[surface])

    def J(self):
        """
        Compute the area of a surface.
        """
        return self.surface.area()

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients.
        """
        return self.surface.darea_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivatives with respect to the surface coefficients.
        """
        return self.surface.d2area_by_dcoeffdcoeff()


class Volume(Optimizable):
    """
    Wrapper class for volume label.
    """

    def __init__(self, surface):
        self.surface = surface
        super().__init__(depends_on=[surface])

    def J(self):
        """
        Compute the volume enclosed by the surface.
        """
        return self.surface.volume()

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients.
        """
        return self.surface.dvolume_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivatives with respect to the surface coefficients.
        """
        return self.surface.d2volume_by_dcoeffdcoeff()


class ToroidalFlux(Optimizable):
    r"""
    Given a surface and Biot Savart kernel, this objective calculates

    .. math::
       J &= \int_{S_{\varphi}} \mathbf{B} \cdot \mathbf{n} ~ds, \\
       &= \int_{S_{\varphi}} \text{curl} \mathbf{A} \cdot \mathbf{n} ~ds, \\
       &= \int_{\partial S_{\varphi}} \mathbf{A} \cdot \mathbf{t}~dl,

    where :math:`S_{\varphi}` is a surface of constant :math:`\varphi`, and :math:`\mathbf A`
    is the magnetic vector potential.
    """

    def __init__(self, surface, biotsavart, idx=0):
        self.surface = surface
        self.biotsavart = biotsavart
        self.idx = idx
        super().__init__(depends_on=[surface, biotsavart])

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def invalidate_cache(self):
        x = self.surface.gamma()[self.idx]
        self.biotsavart.set_points(x)

    def J(self):
        r"""
        Compute the toroidal flux on the surface where
        :math:`\varphi = \texttt{quadpoints_varphi}[\texttt{idx}]`.
        """
        xtheta = self.surface.gammadash2()[self.idx]
        ntheta = self.surface.gamma().shape[1]
        A = self.biotsavart.A()
        tf = np.sum(A * xtheta)/ntheta
        return tf

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients.
        """
        ntheta = self.surface.gamma().shape[1]
        dA_by_dX = self.biotsavart.dA_by_dX()
        A = self.biotsavart.A()
        dgammadash2 = self.surface.gammadash2()[self.idx, :]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx, :]

        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        dA_dc = np.sum(dA_by_dX[..., :, None] * dx_dc[..., None, :], axis=1)
        term1 = np.sum(dA_dc * dgammadash2[..., None], axis=(0, 1))
        term2 = np.sum(A[..., None] * dgammadash2_by_dc, axis=(0, 1))

        out = (term1+term2)/ntheta
        return out

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivatives with respect to the surface coefficients.
        """
        ntheta = self.surface.gamma().shape[1]
        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        d2A_by_dXdX = self.biotsavart.d2A_by_dXdX().reshape((ntheta, 3, 3, 3))
        dA_by_dX = self.biotsavart.dA_by_dX()
        dA_dc = np.sum(dA_by_dX[..., :, None] * dx_dc[..., None, :], axis=1)
        d2A_dcdc = np.einsum('jkpl,jpn,jkm->jlmn', d2A_by_dXdX, dx_dc, dx_dc)

        dgammadash2 = self.surface.gammadash2()[self.idx]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx]

        term1 = np.sum(d2A_dcdc * dgammadash2[..., None, None], axis=-3)
        term2 = np.sum(dA_dc[..., :, None] * dgammadash2_by_dc[..., None, :], axis=-3)
        term3 = np.sum(dA_dc[..., None, :] * dgammadash2_by_dc[..., :, None], axis=-3)

        out = (1/ntheta) * np.sum(term1+term2+term3, axis=0)
        return out


def boozer_surface_residual_accumulate(surface, iota, G, biotsavart, derivatives=0, weighting=None, reg=False):
    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum([np.abs(c.current.x) for c in biotsavart.coils]) * (4 * np.pi * 10**(-7) / (2 * np.pi))
    
    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]
    

    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)

    biotsavart.compute(derivatives)
    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    B2 = np.sum(B**2, axis=2)
    residual = G*B - B2[..., None] * tang
    
    if weighting == 'dS':
        nor = surface.normal()
        dA = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
        w = np.sqrt(dA)
    elif weighting == '1/B':
        modB = np.sqrt(B2)
        w = 1./modB
    elif weighting == 'dS/B':
        nor = surface.normal()
        dA = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
        modB = np.sqrt(B2)
        w1 = np.sqrt(dA)
        w2 = 1./modB
        w = w1*w2

    if weighting is not None:
        rtil = w[:, :, None] * residual
    else:
        rtil = residual.copy()

    rtil_flattened = rtil.reshape((nphi*ntheta*3, ))
    r = rtil_flattened
    val = 0.5 * np.sum(r**2)

    if reg:
        tik_weight=1e-3
        tik =0.5* tik_weight * np.sum(surface.x**2)
        val+=tik


    if derivatives == 0:
        return val,

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()
    nsurfdofs = dx_dc.shape[-1]

    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)

    dresidual_dc = sopp.boozer_dresidual_dc(G, dB_dc, B, tang, B2, dxphi_dc, iota, dxtheta_dc)
    dresidual_diota = -B2[..., None] * xtheta
    
    # initialize the derivative of the functional form of the weighting function
    if weighting == 'dS':
        dA_dc = surface.d_dS_by_dcoeff()
        dw_dc = 0.5*dA_dc/np.sqrt(dA[:, :, None])
    elif weighting == '1/B':
        dB2_dc = 2*np.einsum('ijk,ijkl->ijl', B, dB_dc, optimize=True)
        dmodB_dc = 0.5*dB2_dc/np.sqrt(B2[:, :, None])
        dw_dc =  -dmodB_dc/modB[:, :, None]**2
    elif weighting == 'dS/B':
        dA_dc = surface.d_dS_by_dcoeff()
        dB2_dc = 2*np.einsum('ijk,ijkl->ijl', B, dB_dc, optimize=True)
        dmodB_dc = 0.5*dB2_dc/np.sqrt(B2[:, :, None])
        dw1_dc = 0.5*dA_dc/np.sqrt(dA[:, :, None])
        dw2_dc =  -dmodB_dc/modB[:, :, None]**2
        dw_dc = dw1_dc * w2[:, :, None] + dw2_dc * w1[:, :, None]

    # modify derivative of residual
    if weighting is not None:
        # d/dc(r*dS) = dr/dc * dS + dS_dc * r
        drtil_dc = residual[..., None] * dw_dc[:, :, None, :] + w[:, :, None, None] * dresidual_dc
        drtil_diota = w[:, :, None] * dresidual_diota
    else:
        drtil_dc    = dresidual_dc.copy() 
        drtil_diota = dresidual_diota.copy()


    drtil_dc_flattened    = drtil_dc.reshape((nphi*ntheta*3, nsurfdofs))
    drtil_diota_flattened = drtil_diota.reshape((nphi*ntheta*3, 1))

    if user_provided_G:
        dresidual_dG = B
        
        if weighting is not None:
            drtil_dG = w[:, :, None] * dresidual_dG
        else:
            drtil_dG = dresidual_dG.copy()

        drtil_dG_flattened = drtil_dG.reshape((nphi*ntheta*3, 1))
        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened, drtil_dG_flattened), axis=1)
    else:
        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened), axis=1)
    
    dval = np.sum(r[:, None]*J, axis=0)
    


    if reg:
        dtik_dc = tik_weight * surface.x
        dval[:nsurfdofs]+=dtik_dc



    
    if derivatives == 1:
        return val, dval

    if weighting == 'dS':
        d2A_dc2 = surface.d2_dS_by_dcoeffdcoeff()
        d2w_dc2 = (2 * dA[:, :, None, None] * d2A_dc2 - dA_dc[:, :, :, None]*dA_dc[:, :, None, :])/(4*dA[:, :, None, None]**1.5)
    elif weighting == '1/B':
        d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))
        #d2B_dcdc = np.einsum('ijkpl,ijpn,ijkm->ijlmn', d2B_by_dXdX, dx_dc, dx_dc, optimize=True)
        #term1 = np.einsum('ijlm,ijln->ijmn', dB_dc, dB_dc, optimize=True)
        #term2 = np.einsum('ijlmn,ijl->ijmn', d2B_dcdc, B, optimize=True)
        #d2B2_dcdc = 2*(term1 + term2)
        
        term1 = np.einsum('ijlm,ijln->ijmn', dB_dc, dB_dc, optimize=True)
        term2 = np.einsum('ijkpl,ijpn,ijkm,ijl->ijmn', d2B_by_dXdX, dx_dc, dx_dc, B, optimize=True)
        d2B2_dcdc = 2*(term1+term2)
        
        d2modB_dc2 = (2*B2[:, :, None, None] * d2B2_dcdc -dB2_dc[:, :, :, None]*dB2_dc[:, :, None, :])/(4*B2[:, :, None, None]**1.5)
        d2w_dc2 = (2*dmodB_dc[:, :, :, None] * dmodB_dc[:, :, None, :] - modB[:, :, None, None] * d2modB_dc2)/modB[:, :, None, None]**3.
        
    elif weighting =='dS/B':
        d2A_dc2 = surface.d2_dS_by_dcoeffdcoeff()
        d2w1_dc2 = (2 * dA[:, :, None, None] * d2A_dc2 - dA_dc[:, :, :, None]*dA_dc[:, :, None, :])/(4*dA[:, :, None, None]**1.5)
        
        d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))
        d2B_dcdc = np.einsum('ijkpl,ijpn,ijkm->ijlmn', d2B_by_dXdX, dx_dc, dx_dc)
        term1 = np.einsum('ijlm,ijln->ijmn', dB_dc, dB_dc)
        term2 = np.einsum('ijlmn,ijl->ijmn', d2B_dcdc, B)
        d2B2_dcdc = 2*(term1 + term2)
        
        d2modB_dc2 = (2*B2[:, :, None, None] * d2B2_dcdc -dB2_dc[:, :, :, None]*dB2_dc[:, :, None, :])/(4*B2[:, :, None, None]**1.5)
        d2w2_dc2 = (2*dmodB_dc[:, :, :, None] * dmodB_dc[:, :, None, :] - modB[:, :, None, None] * d2modB_dc2)/modB[:, :, None, None]**3.

        d2w_dc2 = dw1_dc[:, :, :, None] * dw2_dc[:, :, None, :] + dw1_dc[:, :, None, :] * dw2_dc[:, :, :, None] + d2w1_dc2 * w2[:, :, None, None] + d2w2_dc2 * w1[:, :, None, None]

    if weighting is None:
        d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))
        resd2B_dcdc = np.einsum('ijkpl,ijpn,ijkm,ijl->mn', d2B_by_dXdX, dx_dc, dx_dc, residual, optimize=True)
        
        dB2_dc = 2. * np.einsum('ijl,ijlm->ijm', B, dB_dc, optimize=True)
        resterm1 = np.einsum('ijlm,ijln,ijq->mn', dB_dc, dB_dc, residual*tang, optimize=True)
        resterm2 = np.einsum('ijkpl,ijpn,ijkm,ijl,ijq->mn', d2B_by_dXdX, dx_dc, dx_dc, B, residual*tang, optimize=True)
        resd2B2_dcdc = -2*(resterm1 + resterm2)
        
        tang_dc = dxphi_dc + iota * dxtheta_dc
        resterm1 = -np.einsum('ijm,ijkn,ijk->mn', dB2_dc, tang_dc, residual, optimize=True)
        resterm2 = -np.einsum('ijn,ijkm,ijk->mn', dB2_dc, tang_dc, residual, optimize=True)
        resd2residual_by_dcdc = G * resd2B_dcdc + resterm1 + resterm2 + resd2B2_dcdc
        
        d2residual_by_dcdiota = -(dB2_dc[..., None, :] * xtheta[..., :, None] + B2[..., None, None] * dxtheta_dc)
        resd2residual_by_dcdiota = np.sum(residual[...,None]*d2residual_by_dcdiota, axis=(0,1,2) )
        resd2residual_by_diotadiota = 0.

        d2val = J.T@J 
        d2val[:nsurfdofs, :nsurfdofs] += resd2residual_by_dcdc        # noqa (0, 0) dcdc
        d2val[:nsurfdofs, nsurfdofs]  += resd2residual_by_dcdiota     # noqa (0, 1) dcdiota
        d2val[nsurfdofs, :nsurfdofs]  += resd2residual_by_dcdiota     # noqa (1, 0) diotadc
        d2val[nsurfdofs, nsurfdofs]   += resd2residual_by_diotadiota  # noqa (1, 1) diotadiota

        if user_provided_G:
            resd2residual_by_dcdG = np.sum(residual[...,None]*dB_dc, axis=(0,1,2))
            resd2residual_by_diotadG = 0.
            resd2residual_by_dGdG = 0.
            # noqa turns out linting so that we can align everything neatly
            d2val[:nsurfdofs, nsurfdofs+1]  += resd2residual_by_dcdG        # noqa (0, 2) dcdG
            d2val[nsurfdofs, nsurfdofs+1]   += resd2residual_by_diotadiota  # noqa (1, 2) diotadG
            d2val[nsurfdofs+1, :nsurfdofs]  += resd2residual_by_dcdG        # noqa (2, 0) dGdc
            d2val[nsurfdofs+1, nsurfdofs]   += resd2residual_by_diotadG     # noqa (2, 1) dGdiota
            d2val[nsurfdofs+1, nsurfdofs+1] += resd2residual_by_dGdG        # noqa (2, 2) dGdG
    else:
        d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))
        resd2B_dcdc = np.einsum('ijkpl,ijpn,ijkm,ijl,ij->mn', d2B_by_dXdX, dx_dc, dx_dc, residual, w**2, optimize=True)
        
        dB2_dc = 2. * np.einsum('ijl,ijlm->ijm', B, dB_dc, optimize=True)
        resterm1 = np.einsum('ijlm,ijln,ijq,ij->mn', dB_dc, dB_dc, residual*tang, w**2, optimize=True)
        resterm2 = np.einsum('ijkpl,ijpn,ijkm,ijl,ijq,ij->mn', d2B_by_dXdX, dx_dc, dx_dc, B, residual*tang, w**2, optimize=True)
        resd2B2_dcdc = -2*(resterm1 + resterm2)
        
        tang_dc = dxphi_dc + iota * dxtheta_dc
        resterm1 = -np.einsum('ijm,ijkn,ijk,ij->mn', dB2_dc, tang_dc, residual, w**2, optimize=True)
        resterm2 = -np.einsum('ijn,ijkm,ijk,ij->mn', dB2_dc, tang_dc, residual, w**2, optimize=True)
        resd2residual_by_dcdc = G * resd2B_dcdc + resterm1 + resterm2 + resd2B2_dcdc
        
        d2residual_by_dcdiota = -(dB2_dc[..., None, :] * xtheta[..., :, None] + B2[..., None, None] * dxtheta_dc)
        resd2residual_by_dcdiota = np.sum(residual[...,None]*d2residual_by_dcdiota*w[:, :, None, None]**2, axis=(0,1,2) )
        resd2residual_by_diotadiota = 0.
        
        d2val = J.T@J

        d2val[:nsurfdofs, :nsurfdofs] += resd2residual_by_dcdc        # noqa (0, 0) dcdc
        d2val[:nsurfdofs, :nsurfdofs] += np.einsum('ijk,ij,ijmn->mn', residual**2, w, d2w_dc2, optimize=True)  # noqa (0, 0) dcdc
        d2val[:nsurfdofs, :nsurfdofs] += np.einsum('ijk,ij,ijm,ijkn->mn', residual, w, dw_dc, dresidual_dc, optimize=True)  # noqa (0, 0) dcdc
        d2val[:nsurfdofs, :nsurfdofs] += np.einsum('ijk,ij,ijn,ijkm->mn', residual, w, dw_dc, dresidual_dc, optimize=True)  # noqa (0, 0) dcdc
        
        corr = np.einsum('ijk,ij,ijm,ijk->m', residual, w, dw_dc, dresidual_diota, optimize=True)
        d2val[:nsurfdofs, nsurfdofs] += resd2residual_by_dcdiota     # noqa (0, 1) dcdiota
        d2val[nsurfdofs, :nsurfdofs] += resd2residual_by_dcdiota     # noqa (1, 0) diotadc
        d2val[:nsurfdofs, nsurfdofs] += corr 
        d2val[nsurfdofs, :nsurfdofs] += corr
        
        d2val[nsurfdofs, nsurfdofs] += resd2residual_by_diotadiota  # noqa (1, 1) diotadiota


        if user_provided_G:
            resd2residual_by_dcdG  = np.sum(residual[...,None]*dB_dc*w[:, :, None, None]**2, axis=(0,1,2))
            resd2residual_by_dcdG += np.einsum('ijk,ij,ijm,ijk->m', residual, w, dw_dc, dresidual_dG, optimize=True)
            
            d2val[:nsurfdofs, nsurfdofs+1]  += resd2residual_by_dcdG        # noqa (0, 2) dcdG
            d2val[nsurfdofs, nsurfdofs+1]   += 0.                           # noqa (1, 2) diotadG
            d2val[nsurfdofs+1, :nsurfdofs]  += resd2residual_by_dcdG        # noqa (2, 0) dGdc
            d2val[nsurfdofs+1, nsurfdofs]   += 0.                           # noqa (2, 1) dGdiota
            d2val[nsurfdofs+1, nsurfdofs+1] += 0.                           # noqa (2, 2) dGdG


    if reg:
        d2tik_dc2 = tik_weight * np.eye(nsurfdofs, nsurfdofs)
        d2val[:nsurfdofs, :nsurfdofs]+=d2tik_dc2



    return val, dval, d2val

def boozer_surface_residual(surface, iota, G, biotsavart, derivatives=0, weighting=None):
    r"""
    For a given surface, this function computes the
    residual

    .. math::
        G\mathbf B_\text{BS}(\mathbf x) - ||\mathbf B_\text{BS}(\mathbf x)||^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)

    as well as the derivatives of this residual with respect to surface dofs,
    iota, and G.  In the above, :math:`\mathbf x` are points on the surface, :math:`\iota` is the
    rotational transform on that surface, and :math:`\mathbf B_{\text{BS}}` is the magnetic field
    computed using the Biot-Savart law.

    :math:`G` is known for exact boozer surfaces, so if ``G=None`` is passed, then that
    value is used instead.
    """
    
    assert derivatives in [0, 1]

    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum([np.abs(c.current.get_value()) for c in biotsavart.coils]) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]

    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)

    biotsavart.compute(derivatives)
    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    B2 = np.sum(B**2, axis=2)
    residual = G*B - B2[..., None] * tang

    if weighting == 'dS':
        nor = surface.normal()
        dA = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
        w = np.sqrt(dA)
    elif weighting == '1/B':
        modB = np.sqrt(B2)
        w = 1./modB
    elif weighting == 'dS/B':
        nor = surface.normal()
        dA = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
        modB = np.sqrt(B2)
        w1 = np.sqrt(dA)
        w2 = 1./modB
        w = w1*w2

    if weighting is not None:
        rtil = w[:, :, None] * residual
    else:
        rtil = residual.copy()

    rtil_flattened = rtil.reshape((nphi*ntheta*3, ))
    r = rtil_flattened
    if derivatives == 0:
        return r,

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()
    nsurfdofs = dx_dc.shape[-1]

    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc)

    # dresidual_dc = G*dB_dc - 2*np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] - B2[..., None, None] * (dxphi_dc + iota * dxtheta_dc)
    dresidual_dc = sopp.boozer_dresidual_dc(G, dB_dc, B, tang, B2, dxphi_dc, iota, dxtheta_dc)
    dresidual_diota = -B2[..., None] * xtheta


    if weighting == 'dS':
        dA_dc = surface.d_dS_by_dcoeff()
        dw_dc = 0.5*dA_dc/np.sqrt(dA[:, :, None])
    elif weighting == '1/B':
        dB2_dc = 2*np.einsum('ijk,ijkl->ijl', B, dB_dc, optimize=True)
        dmodB_dc = 0.5*dB2_dc/np.sqrt(B2[:, :, None])
        dw_dc =  -dmodB_dc/modB[:, :, None]**2
    elif weighting == 'dS/B':
        dA_dc = surface.d_dS_by_dcoeff()
        dB2_dc = 2*np.einsum('ijk,ijkl->ijl', B, dB_dc, optimize=True)
        dmodB_dc = 0.5*dB2_dc/np.sqrt(B2[:, :, None])
        dw1_dc = 0.5*dA_dc/np.sqrt(dA[:, :, None])
        dw2_dc =  -dmodB_dc/modB[:, :, None]**2
        dw_dc = dw1_dc * w2[:, :, None] + dw2_dc * w1[:, :, None]

    # modify derivative of residual
    if weighting is not None:
        # d/dc(r*dS) = dr/dc * dS + dS_dc * r
        drtil_dc = residual[..., None] * dw_dc[:, :, None, :] + w[:, :, None, None] * dresidual_dc
        drtil_diota = w[:, :, None] * dresidual_diota
    else:
        drtil_dc    = dresidual_dc.copy() 
        drtil_diota = dresidual_diota.copy()


    drtil_dc_flattened    = drtil_dc.reshape((nphi*ntheta*3, nsurfdofs))
    drtil_diota_flattened = drtil_diota.reshape((nphi*ntheta*3, 1))

    if user_provided_G:
        dresidual_dG = B
        
        if weighting is not None:
            drtil_dG = w[:, :, None] * dresidual_dG
        else:
            drtil_dG = dresidual_dG.copy()

        drtil_dG_flattened = drtil_dG.reshape((nphi*ntheta*3, 1))
        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened, drtil_dG_flattened), axis=1)
    else:
        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened), axis=1)
    
    return r, J


def parameter_derivatives(surface: Surface,
                          shape_gradient: NDArray[(Any, Any), Float]
                          ) -> NDArray[(Any,), Float]:
    r"""
    Converts the shape gradient of a given figure of merit, :math:`f`,
    to derivatives with respect to parameters defining a surface.  For
    a perturbation to the surface :math:`\delta \vec{x}`, the
    resulting perturbation to the objective function is

    .. math::
      \delta f(\delta \vec{x}) = \int d^2 x \, G \delta \vec{x} \cdot \vec{n}

    where :math:`G` is the shape gradient and :math:`\vec{n}` is the
    unit normal. Given :math:`G`, the parameter derivatives are then
    computed as

    .. math::
      \frac{\partial f}{\partial \Omega} = \int d^2 x \, G \frac{\partial\vec{x}}{\partial \Omega} \cdot \vec{n},

    where :math:`\Omega` is any parameter of the surface.

    Args:
        surface: The surface to use for the computation
        shape_gradient: 2d array of size (numquadpoints_phi,numquadpoints_theta)

    Returns:
        1d array of size (ndofs)
    """
    N = surface.normal()
    norm_N = np.linalg.norm(N, axis=2)
    dx_by_dc = surface.dgamma_by_dcoeff()
    N_dot_dx_by_dc = np.einsum('ijk,ijkl->ijl', N, dx_by_dc)
    nphi = surface.gamma().shape[0]
    ntheta = surface.gamma().shape[1]
    return np.einsum('ijk,ij->k', N_dot_dx_by_dc, shape_gradient) / (ntheta * nphi)


class QfmResidual(Optimizable):
    r"""
    For a given surface :math:`S`, this class computes the residual

    .. math::
        f(S) = \frac{\int_{S} d^2 x \, (\textbf{B} \cdot \hat{\textbf{n}})^2}{\int_{S} d^2 x \, B^2}

    where :math:`\textbf{B}` is the magnetic field from :mod:`biotsavart`,
    :math:`\hat{\textbf{n}}` is the unit normal on a given surface, and the
    integration is performed over the surface. Derivatives are computed wrt the
    surface dofs.
    """

    def __init__(self, surface, biotsavart):
        self.surface = surface
        self.biotsavart = biotsavart
        self.biotsavart.append_parent(self.surface)
        super().__init__(depends_on=[surface, biotsavart])

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def invalidate_cache(self):
        x = self.surface.gamma()
        xsemiflat = x.reshape((-1, 3))
        self.biotsavart.set_points(xsemiflat)

    def J(self):
        N = self.surface.normal()
        norm_N = np.linalg.norm(N, axis=2)
        n = N/norm_N[:, :, None]
        x = self.surface.gamma()
        nphi = x.shape[0]
        ntheta = x.shape[1]
        B = self.biotsavart.B().reshape((nphi, ntheta, 3))
        B_n = np.sum(B * n, axis=2)
        norm_B = np.linalg.norm(B, axis=2)
        return np.sum(B_n**2 * norm_N)/np.sum(norm_B**2 * norm_N)

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients
        """

        # we write the objective as J = J1/J2, then we compute the partial derivatives
        # dJ1_by_dgamma, dJ1_by_dN, dJ2_by_dgamma, dJ2_by_dN and then use the vjp functions
        # to get the derivatives wrt to the surface dofs
        x = self.surface.gamma()
        nphi = x.shape[0]
        ntheta = x.shape[1]
        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        B = self.biotsavart.B().reshape((nphi, ntheta, 3))
        N = self.surface.normal()
        norm_N = np.linalg.norm(N, axis=2)

        B_N = np.sum(B * N, axis=2)
        dJ1dx = (2*B_N/norm_N)[:, :, None] * (np.sum(dB_by_dX*N[:, :, None, :], axis=3))
        dJ1dN = (2*B_N/norm_N)[:, :, None] * B - (B_N**2/norm_N**3)[:, :, None] * N

        dJ2dx = 2 * np.sum(dB_by_dX*B[:, :, None, :], axis=3) * norm_N[:, :, None]
        dJ2dN = (np.sum(B*B, axis=2)/norm_N)[:, :, None] * N

        J1 = np.sum(B_N**2 / norm_N)  # same as np.sum(B_n**2 * norm_N)
        J2 = np.sum(B**2 * norm_N[:, :, None])

        # d_J1 = self.surface.dnormal_by_dcoeff_vjp(dJ1dN) + self.surface.dgamma_by_dcoeff_vjp(dJ1dx)
        # d_J2 = self.surface.dnormal_by_dcoeff_vjp(dJ2dN) + self.surface.dgamma_by_dcoeff_vjp(dJ2dx)
        # deriv = d_J1/J2 - d_J2*J1/(J2*J2)

        deriv = self.surface.dnormal_by_dcoeff_vjp(dJ1dN/J2 - dJ2dN*J1/(J2*J2)) \
            + self.surface.dgamma_by_dcoeff_vjp(dJ1dx/J2 - dJ2dx*J1/(J2*J2))
        return deriv
