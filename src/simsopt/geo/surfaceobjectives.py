import numpy as np
import scipy


class Area(object):

    def __init__(self, surface):
        self.surface = surface

    def J(self):
        return self.surface.area()

    def dJ_by_dsurfacecoefficients(self):
        return self.surface.darea_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        return self.surface.d2area_by_dcoeffdcoeff()


class Volume(object):
    def __init__(self, surface):
        self.surface = surface

    def J(self):
        return self.surface.volume()

    def dJ_by_dsurfacecoefficients(self):
        return self.surface.dvolume_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        return self.surface.d2volume_by_dcoeffdcoeff()


class ToroidalFlux(object):

    r"""
    This objective calculates
        J = \int_{varphi = constant} B \cdot n ds
          = \int_{varphi = constant} curlA \cdot n ds
          from Stokes' theorem
          = \int_{curve on surface where varphi = constant} A \cdot n dl
    given a surface and Biot Savart kernel.
    """

    def __init__(self, surface, biotsavart, idx=0):
        self.surface = surface
        self.biotsavart = biotsavart
        self.idx = idx
        self.surface.dependencies.append(self)
        self.invalidate_cache()

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
        H[:, :nsurfdofs, :nsurfdofs]   = d2residual_by_dcdc_flattened       #(0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs]    = d2residual_by_dcdiota_flattened    #(0, 1) dcdiota
        H[:, :nsurfdofs, nsurfdofs+1]  = d2residual_by_dcdG_flattened       #(0, 2) dcdG
        H[:, nsurfdofs, :nsurfdofs]    = d2residual_by_dcdiota_flattened    #(1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs]     = d2residual_by_diotadiota_flattened #(1, 1) diotadiota
        H[:, nsurfdofs, nsurfdofs+1]   = d2residual_by_diotadiota_flattened #(1, 2) diotadG
        H[:, nsurfdofs+1, :nsurfdofs]  = d2residual_by_dcdG_flattened       #(2, 0) dGdc
        H[:, nsurfdofs+1, nsurfdofs]   = d2residual_by_diotadG_flattened    #(2, 1) dGdiota
        H[:, nsurfdofs+1, nsurfdofs+1] = d2residual_by_dGdG_flattened       #(2, 2) dGdG
    else:
        H = np.zeros((nphi*ntheta*3, nsurfdofs + 1, nsurfdofs + 1))
        H[:, :nsurfdofs, :nsurfdofs]   = d2residual_by_dcdc_flattened       #(0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs]    = d2residual_by_dcdiota_flattened    #(0, 1) dcdiota
        H[:, nsurfdofs, :nsurfdofs]    = d2residual_by_dcdiota_flattened    #(1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs]     = d2residual_by_diotadiota_flattened #(1, 1) diotadiota

    return r, J, H

class NonQuasiSymmetricComponentPenalty(object):
    """
    This class computes a measure of quasisymmetry on a given surface.  It does so by taking a 
    Fourier transform of the field magnitude.  Then, it separates the quasisymmetric harmonics 
    from the non-quasisymmetric ones.  From these harmonics, we can define a 
    quasisymmetric and non-quasisymmetric component of the field strength:

    modB(varphi,theta) = modB_QS + modB_nonQS
    where modB_QS    = ifft2(quasisymmetric harmonics)
    and   modB_nonQS = ifft2(nonquasisymmetric harmonics)

    The quasisymmetric harmonics have indices (n,m) = (m*N,m) and the non quasisymmetric
    harmonics have indices (n,m) != (m*N,m)

    This penalty term returns J = 0.5\int_{surface} modB_nonQS^2 dS and its derivatives with respect
    to the surface degrees of freedom.
    """
    def __init__(self,surface, biotsavart, N = 0):
        self.surface = surface
        self.biotsavart = biotsavart
        self.N = N # what the variety of quasisymmetry we're optimizing for
        self.surface.dependencies.append(self)
        self.invalidate_cache()
        
        original_dim = surface.gamma().shape 
        if original_dim[1] % 2 == 0:
            rfft_dim2 = original_dim[1] // 2 + 1
        else:
            rfft_dim2 = (original_dim[1] + 1) // 2
        rfft_dim = (original_dim[0], rfft_dim2)

        # the quasisymmetric harmonics have index (N*m, m)
        qs_idx1 = self.N*np.arange( min(rfft_dim) )
        qs_idx2 =        np.arange( min(rfft_dim) )
        
        qs_filter = np.zeros( rfft_dim )
        qs_filter[qs_idx1,qs_idx2] = 1.

        non_qs_filter = np.ones( rfft_dim )
        non_qs_filter[qs_idx1,qs_idx2] = 0.
        
        self.nqs_filter = non_qs_filter
        self.qs_filter = qs_filter

        nphi = surface.gamma().shape[0]
        ntheta = surface.gamma().shape[1]
        def get_op_col(in_vec):
            modB = in_vec.reshape( (nphi,ntheta) )
            return np.fft.irfft2(np.fft.rfft2(modB) * self.nqs_filter, s=modB.shape).flatten()
        
        self.op_mat = np.zeros( (nphi*ntheta,nphi*ntheta) )
        for i in range(nphi*ntheta):
            ei = np.zeros( (nphi*ntheta,) )
            ei[i] = 1.
            self.op_mat[:,i] = get_op_col(ei)
        
    
    def invalidate_cache(self):
        x = self.surface.gamma().reshape((-1,3))
        self.biotsavart.set_points(x)


    def J(self):
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape( (nphi,ntheta,3) )
        modB = np.sqrt( B[:,:,0]**2 + B[:,:,1]**2 + B[:,:,2]**2)
        
        non_qs_func = np.fft.irfft2(np.fft.rfft2(modB) * self.nqs_filter, s = modB.shape)
        nor = self.surface.normal()
        dS = np.sqrt(nor[:,:,0]**2 + nor[:,:,1]**2 + nor[:,:,2]**2)

        J = 0.5 * np.mean( dS * non_qs_func**2 )
        return J
    def dJ_by_dB(self):
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape( (nphi,ntheta,3) )
        
        modB = np.sqrt( B[:,:,0]**2 + B[:,:,1]**2 + B[:,:,2]**2)
        dmodB_dB = (B / modB[...,None]).reshape( (-1,3) )
        
        nor = self.surface.normal()
        dS = np.sqrt(nor[:,:,0]**2 + nor[:,:,1]**2 + nor[:,:,2]**2).flatten()
        
        non_qs_func = np.fft.irfft2(np.fft.rfft2(modB) * self.nqs_filter, s = modB.shape).flatten()
        dnon_qs_func_dB = self.op_mat[...,None] * dmodB_dB[None,...]
        dJ_by_dB = np.mean( dS[:,None,None] * non_qs_func[:,None,None] * dnon_qs_func_dB, axis = 0 )

        return dJ_by_dB
    
    def dJ_by_dcoilcoefficients(self):
        dJ_by_dB = self.dJ_by_dB().reshape( (-1,3) )
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)
        return dJ_by_dcoils

    def dJ_by_dsurfacecoefficients(self):
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size

        B = self.biotsavart.B().reshape( (nphi,ntheta,3) )
        dB_by_dX    = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        dx_dc = self.surface.dgamma_by_dcoeff()
        dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc)

        modB = np.sqrt( B[:,:,0]**2 + B[:,:,1]**2 + B[:,:,2]**2)
        dmodB_dc = (B[:,:,0, None] * dB_dc[:,:,0,:] + B[:,:,1, None] * dB_dc[:,:,1,:] + B[:,:,2, None] * dB_dc[:,:,2,:])/modB[:,:,None]
        
        non_qs_func     = np.fft.irfft2(np.fft.rfft2(modB) * self.nqs_filter, s = modB.shape)
        dnon_qs_func_dc = np.fft.irfft2(
                          np.fft.rfft2(dmodB_dc, axes=(0,1)) * self.nqs_filter[:,:,None], \
                          axes=(0,1), s = modB.shape)
        
        nor = self.surface.normal()
        dnor_dc = self.surface.dnormal_by_dcoeff()
        dS = np.sqrt(nor[:,:,0]**2 + nor[:,:,1]**2 + nor[:,:,2]**2)
        dS_dc = (nor[:,:,0,None]*dnor_dc[:,:,0,:] + nor[:,:,1,None]*dnor_dc[:,:,1,:] + nor[:,:,2,None]*dnor_dc[:,:,2,:])/dS[:,:,None]

        dJ_dc = np.mean(  dmodB_dc, axis = (0,1) )
        dJ_dc = np.mean( 0.5 * dS_dc * non_qs_func[:,:,None]**2 
                         + non_qs_func[:,:,None] * dnon_qs_func_dc * dS[:,:,None] , axis = (0,1) )
        return dJ_dc





