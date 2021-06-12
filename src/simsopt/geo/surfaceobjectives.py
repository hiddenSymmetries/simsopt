import numpy as np
import scipy


class Area(object):

    def __init__(self, surface, stellarator):
        self.surface = surface
        self.stellarator = stellarator
    def J(self):
        return self.surface.area()

    def dJ_by_dsurfacecoefficients(self):
        return self.surface.darea_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        return self.surface.d2area_by_dcoeffdcoeff()
    
    def dJ_by_dcoilcoefficients(self):
        return [np.zeros((c.num_dofs(),)) for c in self.stellarator.coils]

class Volume(object):

    def __init__(self, surface, stellarator):
        self.surface = surface
        self.stellarator = stellarator

    def J(self):
        return self.surface.volume()

    def dJ_by_dsurfacecoefficients(self):
        return self.surface.dvolume_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        return self.surface.d2volume_by_dcoeffdcoeff()

    def dJ_by_dcoilcoefficients(self):
        return [np.zeros((c.num_dofs(),)) for c in self.stellarator.coils]


class ToroidalFlux(object):

    r"""
    This objective calculates
        J = \int_{varphi = constant} B \cdot n ds
          = \int_{varphi = constant} curlA \cdot n ds
          from Stokes' theorem
          = \int_{curve on surface where varphi = constant} A \cdot n dl
    given a surface and Biot Savart kernel.
    """

    def __init__(self, surface, biotsavart, stellarator, idx=0):
        self.stellarator = stellarator
        self.surface = surface
        self.biotsavart = biotsavart
        self.idx = idx

    def J(self):
        x = self.surface.gamma()[self.idx]
        self.biotsavart.set_points(x)
        xtheta = self.surface.gammadash2()[self.idx]
        ntheta = self.surface.gamma().shape[1]
        A = self.biotsavart.A()
        tf = np.sum(A * xtheta)/ntheta
        return tf

    def dJ_by_dA(self):
        xtheta = self.surface.gammadash2()[self.idx]
        nphi = self.surface.gamma().shape[0]
        ntheta = self.surface.gamma().shape[1]
        dJ_by_dA = np.zeros((nphi, ntheta, 3))
        dJ_by_dA[self.idx, :] = xtheta/ntheta
        return dJ_by_dA.reshape((-1, 3))

    def dJ_by_dcoilcoefficients(self):
        return [np.zeros((c.num_dofs(),)) for c in self.stellarator.coils]

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients
        """
        x = self.surface.gamma()[self.idx]
        self.biotsavart.set_points(x)

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
        Calculate the second derivatives with respect to the surface coefficients
        """
        x = self.surface.gamma()[self.idx]
        self.biotsavart.set_points(x)

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



def boozer_surface_dexactresidual_dcoils_vjp(lm, booz_surf, iota, G, biotsavart):
    """
    For a given surface with points x on it, this function computes the
    vector-Jacobian product of \lm^T * dresidual_dcoils:

    lm^T dresidual_dcoils = [G*lm - lm(2*||B_BS(x)|| (x_phi + iota * x_theta) ]^T * dB_dcoils
    
    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """
    surface = booz_surf.surface
    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum(np.abs(biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    res, dres_dB = boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=0)
    dres_dB = dres_dB.reshape((-1, 3, 3))

    lmask = np.zeros(booz_surf.res["mask"].shape)
    lmask[booz_surf.res["mask"]] = lm[:-1]
    lm = lmask.reshape( (-1,3) )

    lm_times_dres_dB = np.sum(lm[:,:,None] * dres_dB, axis=1).reshape( (-1,3) )
    dres_dcoils = biotsavart.B_vjp(lm_times_dres_dB)
    return dres_dcoils
    
def boozer_surface_dlsqgrad_dcoils_vjp(lm, booz_surf, iota, G, biotsavart):
    """
    For a given surface with points x on it, this function computes the
    vector-Jacobian product of \lm^T * dlsqgrad_dcoils:

    lm^T dresidual_dcoils = lm^T [dr_dsurface]^T[dr_dcoils] + sum r_i lm^T d2ri_dsdc
    
    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """

    surface = booz_surf.surface
    #import ipdb;ipdb.set_trace()
    # r, dr_dB, J, d2residual_dsurfacedB, d2residual_dsurfacedgradB
    boozer = boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=1)
    r = boozer[0]
    dr_dB = boozer[1].reshape((-1, 3, 3))
    dr_ds = boozer[2]
    d2r_dsdB = boozer[3]
    d2r_dsdgradB = boozer[4]
    
    v1 = np.sum(np.sum(lm[:, None]*dr_ds.T, axis=0).reshape((-1, 3, 1)) * dr_dB, axis=1)
    v2 = np.sum(r.reshape((-1, 3, 1))*np.sum(lm[None,None,:]*d2r_dsdB, axis=-1).reshape((-1, 3, 3)), axis=1)
    v3 = np.sum(r.reshape((-1, 3, 1, 1))*np.sum(lm[None,None,None,:]*d2r_dsdgradB, axis=-1).reshape((-1, 3, 3, 3)), axis=1)
    vjp = biotsavart.B_and_dB_vjp(v1+v2, v3)
    vjp = [a + b for a,b in zip(vjp[0], vjp[1])]
    return vjp

def boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=0):
    """
    For a given surface with points x on it, this function computes the
    differentiated residual

       d/dB[ G*B_BS(x) - ||B_BS(x)||^2 * (x_phi + iota * x_theta) ]

    as well as the derivatives of this residual with respect to surface dofs,
    iota, and G.

    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """

    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum(np.abs(biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]
    
    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)

    B = biotsavart.B(compute_derivatives=derivatives).reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    residual = G*B - np.sum(B**2, axis=2)[..., None] * tang
    
    GI = np.eye(3,3) * G
    dresidual_dB = GI[None,None,:,:] - 2. * tang[:,:,:,None] * B[:,:,None,:]

    residual_flattened = residual.reshape((nphi*ntheta*3, ))
    dresidual_dB_flattened = dresidual_dB.reshape((nphi*ntheta*3,3))
    r = residual_flattened
    dr_dB = dresidual_dB_flattened
    if derivatives == 0:
        return r, dr_dB

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()
    nsurfdofs = dx_dc.shape[-1]

    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc)
    dtang_dc = dxphi_dc + iota * dxtheta_dc
    dresidual_dc = G*dB_dc \
        - 2*np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] \
        - np.sum(B**2, axis=2)[..., None, None] * dtang_dc 
    dresidual_diota = -np.sum(B**2, axis=2)[..., None] * xtheta

    d2residual_dcdB = -2*dB_dc[:, :, None, :, :] * tang[:, :, :, None, None] - 2*B[:, :, None, :, None] * dtang_dc[:, :, :, None, :]
    d2residual_diotadB = -2.*B[:, :, None, :] * xtheta[:, :, :, None]
    d2residual_dcdgradB = -2.*B[:, :, None, None, :, None]*dx_dc[:, :, None, :, None, :]*tang[:, :, :, None, None, None]
    idx = np.arange( 3 )
    d2residual_dcdgradB[:, :, idx, :, idx, :] += dx_dc * G

    dresidual_dc_flattened = dresidual_dc.reshape((nphi*ntheta*3, nsurfdofs))
    dresidual_diota_flattened = dresidual_diota.reshape((nphi*ntheta*3, 1))
    d2residual_dcdB_flattened = d2residual_dcdB.reshape((nphi*ntheta*3, 3, nsurfdofs))
    d2residual_diotadB_flattened = d2residual_diotadB.reshape((nphi*ntheta*3, 3, 1))
    d2residual_dcdgradB_flattened = d2residual_dcdgradB.reshape((nphi*ntheta*3, 3, 3, nsurfdofs))
    d2residual_diotadgradB_flattened = np.zeros((nphi*ntheta*3, 3, 3, 1)) 

    if user_provided_G:
        dresidual_dG = B
        dresidual_dG_flattened = dresidual_dG.reshape((nphi*ntheta*3, 1))
        J = np.concatenate((dresidual_dc_flattened, dresidual_diota_flattened, dresidual_dG_flattened), axis=1)
        
        d2residual_dGdB = np.ones((nphi*ntheta, 3, 3))
        d2residual_dGdB[:,:,:] = np.eye(3)[None,:,:]
        d2residual_dGdB = d2residual_dGdB.reshape((3*nphi*ntheta, 3, 1))
        d2residual_dGdgradB = np.zeros((3*nphi*ntheta, 3, 3, 1))

        d2residual_dsurfacedB = np.concatenate((d2residual_dcdB_flattened, 
                                                d2residual_diotadB_flattened, 
                                                d2residual_dGdB), axis=-1)
        d2residual_dsurfacedgradB = np.concatenate((d2residual_dcdgradB_flattened, 
                                                    d2residual_diotadgradB_flattened, 
                                                    d2residual_dGdgradB), axis=-1)
    else:
        J = np.concatenate((dresidual_dc_flattened, dresidual_diota_flattened), axis=1)
        d2residual_dsurfacedB = np.concatenate((d2residual_dcdB_flattened, d2residual_diotadB_flattened), axis=-1)
        d2residual_dsurfacedgradB = np.concatenate((d2residual_dcdgradB_flattened, 
                                                    d2residual_diotadgradB_flattened), axis=-1)
    
    if derivatives == 1:
        return r, dr_dB, J, d2residual_dsurfacedB, d2residual_dsurfacedgradB
    

def boozer_surface_residual(surface, iota, G, biotsavart, derivatives=0):
    """
    For a given surface with points x on it, this function computes the
    residual

        G*B_BS(x) - ||B_BS(x)||^2 * (x_phi + iota * x_theta)

    as well as the derivatives of this residual with respect to surface dofs,
    iota, and G.

    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """

    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum(np.abs(biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]
    
    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)

    B = biotsavart.B(compute_derivatives=derivatives).reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    residual = G*B - np.sum(B**2, axis=2)[..., None] * tang

    residual_flattened = residual.reshape((nphi*ntheta*3, ))
    r = residual_flattened
    if derivatives == 0:
        return r,

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()
    nsurfdofs = dx_dc.shape[-1]

    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc)

    dresidual_dc = G*dB_dc \
        - 2*np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] \
        - np.sum(B**2, axis=2)[..., None, None] * (dxphi_dc + iota * dxtheta_dc)
    dresidual_diota = -np.sum(B**2, axis=2)[..., None] * xtheta

    dresidual_dc_flattened = dresidual_dc.reshape((nphi*ntheta*3, nsurfdofs))
    dresidual_diota_flattened = dresidual_diota.reshape((nphi*ntheta*3, 1))

    if user_provided_G:
        dresidual_dG = B
        dresidual_dG_flattened = dresidual_dG.reshape((nphi*ntheta*3, 1))
        J = np.concatenate((dresidual_dc_flattened, dresidual_diota_flattened, dresidual_dG_flattened), axis=1)
    else:
        J = np.concatenate((dresidual_dc_flattened, dresidual_diota_flattened), axis=1)
    if derivatives == 1:
        return r, J


    d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))
    B2 = np.sum(B**2, axis=-1)
    d2B_dcdc = np.einsum('ijkpl,ijpn,ijkm->ijlmn', d2B_by_dXdX, dx_dc, dx_dc)
    dB2_dc = 2. * np.einsum('ijl,ijlm->ijm', B, dB_dc)

    term1 = np.einsum('ijlm,ijln->ijmn', dB_dc, dB_dc)
    term2 = np.einsum('ijlmn,ijl->ijmn', d2B_dcdc, B)
    d2B2_dcdc = 2*(term1 + term2)

    term1 = -(dxphi_dc[..., None, :] + iota * dxtheta_dc[..., None, :]) * dB2_dc[..., None, :, None]
    term2 = -(dxphi_dc[..., :, None] + iota * dxtheta_dc[..., :, None]) * dB2_dc[..., None, None, :]
    term3 = -(xphi[..., None, None] + iota * xtheta[..., None, None]) * d2B2_dcdc[..., None, :, :]
    d2residual_by_dcdc = G * d2B_dcdc + term1 + term2 + term3
    d2residual_by_dcdiota = -(dB2_dc[..., None, :] * xtheta[..., :, None] + B2[..., None, None] * dxtheta_dc)
    d2residual_by_diotadiota = np.zeros(dresidual_diota.shape)

    d2residual_by_dcdc_flattened = d2residual_by_dcdc.reshape((nphi*ntheta*3, nsurfdofs, nsurfdofs))
    d2residual_by_dcdiota_flattened = d2residual_by_dcdiota.reshape((nphi*ntheta*3, nsurfdofs))
    d2residual_by_diotadiota_flattened = d2residual_by_diotadiota.reshape((nphi*ntheta*3,))

    if user_provided_G:
        d2residual_by_dcdG = dB_dc
        d2residual_by_diotadG = np.zeros(dresidual_diota.shape)
        d2residual_by_dGdG = np.zeros(dresidual_dG.shape)
        d2residual_by_dcdG_flattened = d2residual_by_dcdG.reshape((nphi*ntheta*3, nsurfdofs))
        d2residual_by_diotadG_flattened = d2residual_by_diotadG.reshape((nphi*ntheta*3,))
        d2residual_by_dGdG_flattened = d2residual_by_dGdG.reshape((nphi*ntheta*3,))
        H = np.zeros((nphi*ntheta*3, nsurfdofs + 2, nsurfdofs + 2))
        H[:, :nsurfdofs, :nsurfdofs]   = d2residual_by_dcdc_flattened       # (0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs]    = d2residual_by_dcdiota_flattened    # (0, 1) dcdiota
        H[:, :nsurfdofs, nsurfdofs+1]  = d2residual_by_dcdG_flattened       # (0, 2) dcdG
        H[:, nsurfdofs, :nsurfdofs]    = d2residual_by_dcdiota_flattened    # (1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs]     = d2residual_by_diotadiota_flattened # (1, 1) diotadiota
        H[:, nsurfdofs, nsurfdofs+1]   = d2residual_by_diotadiota_flattened # (1, 2) diotadG
        H[:, nsurfdofs+1, :nsurfdofs]  = d2residual_by_dcdG_flattened       # (2, 0) dGdc
        H[:, nsurfdofs+1, nsurfdofs]   = d2residual_by_diotadG_flattened    # (2, 1) dGdiota
        H[:, nsurfdofs+1, nsurfdofs+1] = d2residual_by_dGdG_flattened       # (2, 2) dGdG
    else:
        H = np.zeros((nphi*ntheta*3, nsurfdofs + 1, nsurfdofs + 1))
        H[:, :nsurfdofs, :nsurfdofs]   = d2residual_by_dcdc_flattened       # (0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs]    = d2residual_by_dcdiota_flattened    # (0, 1) dcdiota
        H[:, nsurfdofs, :nsurfdofs]    = d2residual_by_dcdiota_flattened    # (1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs]     = d2residual_by_diotadiota_flattened # (1, 1) diotadiota

    return r, J, H


class MajorRadius(object): 
    r"""
    This objective computes the major radius of a toroidal surface with the following formula:
    
    R_major = (1/2*pi) * (V / mean_cross_sectional_area)
    
    where mean_cross_sectional_area = \int^1_0 \int^1_0 z_\theta(xy_\varphi -yx_\varphi) - z_\varphi(xy_\theta-yx_\theta) ~d\theta d\varphi

    """
    def __init__(self, surface):
        self.surface = surface

    def J(self):
        """
        Return the objective value
        """
        surface = self.surface

        g  = surface.gamma()
        g1 = surface.gammadash1()
        g2 = surface.gammadash2()
        
        x = g[:,:,0]
        y = g[:,:,1]
        
        r = np.sqrt(x**2+y**2)

        xvarphi = g1[:,:,0]
        yvarphi = g1[:,:,1]
        zvarphi = g1[:,:,2]

        xtheta = g2[:,:,0]
        ytheta = g2[:,:,1]
        ztheta = g2[:,:,2]

        mean_area = np.mean((ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta))/r)/(2*np.pi)
        R_major = surface.volume() / (2. * np.pi * mean_area)
        return R_major

    def dJ(self):
        """
        Return the objective value
        """

        surface = self.surface

        g  = surface.gamma()
        g1 = surface.gammadash1()
        g2 = surface.gammadash2()

        dg_ds = surface.dgamma_by_dcoeff()
        dg1_ds = surface.dgammadash1_by_dcoeff()
        dg2_ds = surface.dgammadash2_by_dcoeff()

        
        x = g[:,:,0, None]
        y = g[:,:,1, None]

        dx_ds = dg_ds[:,:,0,:]
        dy_ds = dg_ds[:,:,1,:]

        r = np.sqrt(x**2+y**2)
        dr_ds =  (x*dx_ds+y*dy_ds)/r

        xvarphi = g1[:,:,0, None]
        yvarphi = g1[:,:,1, None]
        zvarphi = g1[:,:,2, None]

        xtheta = g2[:,:,0, None]
        ytheta = g2[:,:,1, None]
        ztheta = g2[:,:,2, None]

        dxvarphi_ds = dg1_ds[:,:,0,:]
        dyvarphi_ds = dg1_ds[:,:,1,:]
        dzvarphi_ds = dg1_ds[:,:,2,:]

        dxtheta_ds = dg2_ds[:,:,0,:]
        dytheta_ds = dg2_ds[:,:,1,:]
        dztheta_ds = dg2_ds[:,:,2,:]

        mean_area = np.mean((1/r) * (ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta)))
        dmean_area_ds = np.mean((1/(r**2))*((xvarphi * y * ztheta - xtheta * y * zvarphi + x * (-yvarphi * ztheta + ytheta * zvarphi)) * dr_ds + r * (-zvarphi * (ytheta* dx_ds - y * dxtheta_ds - xtheta * dy_ds + x * dytheta_ds) + ztheta * (yvarphi * dx_ds - y * dxvarphi_ds - xvarphi* dy_ds + x * dyvarphi_ds) + (-xvarphi * y + x * yvarphi) * dztheta_ds + (xtheta * y - x * ytheta) * dzvarphi_ds)), axis = (0,1) )

        dR_major_ds = (-surface.volume() * dmean_area_ds + surface.dvolume_by_dcoeff() * mean_area) / mean_area**2
        return dR_major_ds



class NonQuasiAxisymmetricComponentPenalty(object):
    r"""
    This objective decomposes the field magnitude :math:`B(\varphi,\theta)` into quasiaxisymmetric and
    non-quasiaxisymmetric components.
    
    .. math::
        J &= \frac{1}{2}\int_{\Gamma_{s}} (B-B_{\text{QS}})^2~dS
          &= \frac{1}{2}\int_0^1 \int_0^1 (B - B_{\text{QS}})^2 \|\mathbf n\| ~d\varphi~\d\theta 
    where
    
    .. math::
        B &= \| \mathbf B(\varphi,\theta) \|_2
        B_{\text{QS}} &= \frac{\int_0^1 \int_0^1 B \| n\| ~d\varphi ~d\theta}{\int_0^1 \int_0^1 \|\mathbf n\| ~d\varphi ~d\theta}
    
    """
    def __init__(self, boozer_surface, stellarator, target_iota = None, iota_weight = 0.,
                 target_major_radius = None, major_radius_weight = 0.):
        self.stellarator = stellarator
        self.boozer_surface = boozer_surface
        if boozer_surface.res is not None:
            self.boozer_surface_reference = {"dofs": self.boozer_surface.surface.get_dofs(),
                                             "iota": self.boozer_surface.res["iota"],
                                              "G": self.boozer_surface.res["G"]}
        else:
            raise Exception("Please solve for the Boozer surface before initializing.")

        if target_iota is not None:
            self.target_iota = target_iota
            self.iota_weight = iota_weight
        else:
            self.target_iota = boozer_surface.res['iota']
            self.iota_weight = 1.

        self.biotsavart = boozer_surface.bs
        self.major_radius = MajorRadius(boozer_surface.surface)
        self.major_radius_weight = major_radius_weight
        if target_major_radius is None:
            self.target_major_radius = self.major_radius.J()
        else:
            self.target_major_radius = target_major_radius

#        x = self.boozer_surface.surface.gamma().reshape((-1,3))
#        self.biotsavart.set_points(x)
        self.boozer_surface.surface.dependencies.append(self)
        self.invalidate_cache()
    
    def invalidate_cache(self):
        x = self.boozer_surface.surface.gamma().reshape((-1,3))
        self.biotsavart.set_points(x)
    
    def J(self):
        """
        Return the objective value
        """
        surface = self.boozer_surface.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size
        iota = self.boozer_surface.res["iota"]

        B = self.biotsavart.B()
        B = B.reshape( (nphi,ntheta,3) )
        modB = np.sqrt( B[:,:,0]**2 + B[:,:,1]**2 + B[:,:,2]**2)
        
        nor = surface.normal()
        dS = np.sqrt(nor[:,:,0]**2 + nor[:,:,1]**2 + nor[:,:,2]**2)

        B_QS = np.mean(modB * dS, axis = 0) / np.mean(dS, axis = 0)
        B_nonQS = modB - B_QS[None,:]
        J = 0.5 * np.mean( dS * B_nonQS**2 ) + 0.5 * self.iota_weight * (iota - self.target_iota)**2. + 0.5 * self.major_radius_weight * ( self.major_radius.J() - self.target_major_radius)**2.
        return J

    def dJ(self):
        bs = self.biotsavart
        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        jac = booz_surf.res['jacobian']
        dconstraint_dcoils_vjp = booz_surf.res['dconstraint_dcoils_vjp']

        # tack on dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.concatenate((self.dJ_by_dsurfacecoefficients(), [self.iota_weight * (iota - self.target_iota), 0.]))
        adj = np.linalg.solve(jac.T, dJ_ds)
        adj += np.linalg.solve(jac.T, dJ_ds-jac.T@adj)
        
        adj_times_dg_dc = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, bs)
        dJ = [dj_dc - adj_dg_dc for dj_dc,adj_dg_dc in zip(self.dJ_by_dcoilcoefficients(), adj_times_dg_dc)]
        dJ = self.stellarator.reduce_coefficient_derivatives(dJ)

        return dJ


    def dJ_by_dB(self):
        """
        Return the derivative of the objective with respect to the magnetic field
        """
        surface = self.boozer_surface.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape( (nphi,ntheta,3) )
        
        modB = np.sqrt( B[:,:,0]**2 + B[:,:,1]**2 + B[:,:,2]**2)
        nor = surface.normal()
        dS = np.sqrt(nor[:,:,0]**2 + nor[:,:,1]**2 + nor[:,:,2]**2)

        denom = np.mean( dS, axis = 0)
        B_QS = np.mean(modB * dS, axis = 0) / denom
        B_nonQS = modB - B_QS[None,:]
        
        dmodB_dB = B / modB[...,None]
        dJ_by_dB = B_nonQS[...,None] * dmodB_dB * dS[:,:,None] / (nphi * ntheta)
        return dJ_by_dB
       
   
    def dJ_by_dcoilcoefficients(self):
        """
        Return the derivative of the objective with respect to the coil coefficients
        """
        dJ_by_dB = self.dJ_by_dB().reshape( (-1,3) )
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)
        return dJ_by_dcoils

    def dJ_by_dsurfacecoefficients(self):
        """
        Return the derivative of the objective with respect to the surface coefficients
        """
        surface = self.boozer_surface.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape( (nphi,ntheta,3) )
        modB = np.sqrt( B[:,:,0]**2 + B[:,:,1]**2 + B[:,:,2]**2)
        
        nor = surface.normal()
        dnor_dc = surface.dnormal_by_dcoeff()
        dS = np.sqrt(nor[:,:,0]**2 + nor[:,:,1]**2 + nor[:,:,2]**2)
        dS_dc = (nor[:,:,0,None]*dnor_dc[:,:,0,:] + nor[:,:,1,None]*dnor_dc[:,:,1,:] + nor[:,:,2,None]*dnor_dc[:,:,2,:])/dS[:,:,None]

        B_QS = np.mean(modB * dS, axis = 0) / np.mean(dS, axis = 0)
        B_nonQS = modB - B_QS[None,:]
        
        dB_by_dX    = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        dx_dc = surface.dgamma_by_dcoeff()
        dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc)
        
        modB = np.sqrt( B[:,:,0]**2 + B[:,:,1]**2 + B[:,:,2]**2)
        dmodB_dc = (B[:,:,0, None] * dB_dc[:,:,0,:] + B[:,:,1, None] * dB_dc[:,:,1,:] + B[:,:,2, None] * dB_dc[:,:,2,:])/modB[:,:,None]
        
        num = np.mean(modB * dS, axis = 0)
        denom = np.mean(dS, axis = 0)
        dnum_dc = np.mean(dmodB_dc * dS[...,None] + modB[...,None] * dS_dc, axis = 0) 
        ddenom_dc = np.mean(dS_dc, axis = 0)
        B_QS_dc = (dnum_dc * denom[:,None] - ddenom_dc * num[:,None])/denom[:,None]**2
        B_nonQS_dc = dmodB_dc - B_QS_dc[None,:]
        
        dJ_by_dc = np.mean( 0.5 * dS_dc * B_nonQS[...,None]**2 + dS[...,None] * B_nonQS[...,None] * B_nonQS_dc , axis = (0,1) ) + self.major_radius_weight * ( self.major_radius.J() - self.target_major_radius) * self.major_radius.dJ()
        return dJ_by_dc


