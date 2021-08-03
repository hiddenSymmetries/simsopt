import numpy as np
import simsoptpp as sopp


class Area(object):
    """
    Wrapper class for surface area label.
    """

    def __init__(self, surface):
        self.surface = surface

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


class Volume(object):
    """
    Wrapper class for volume label.
    """

    def __init__(self, surface):
        self.surface = surface

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


class ToroidalFlux(object):
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
        self.surface.dependencies.append(self)
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


def boozer_surface_residual(surface, iota, G, biotsavart, derivatives=0):
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

    biotsavart.compute(derivatives)
    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    B2 = np.sum(B**2, axis=2)
    residual = G*B - B2[..., None] * tang

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

    # dresidual_dc = G*dB_dc - 2*np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] - B2[..., None, None] * (dxphi_dc + iota * dxtheta_dc)
    dresidual_dc = sopp.boozer_dresidual_dc(G, dB_dc, B, tang, B2, dxphi_dc, iota, dxtheta_dc)
    dresidual_diota = -B2[..., None] * xtheta

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
        # noqa turns out linting so that we can align everything neatly
        H[:, :nsurfdofs, :nsurfdofs] = d2residual_by_dcdc_flattened        # noqa (0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs] = d2residual_by_dcdiota_flattened     # noqa (0, 1) dcdiota
        H[:, :nsurfdofs, nsurfdofs+1] = d2residual_by_dcdG_flattened        # noqa (0, 2) dcdG
        H[:, nsurfdofs, :nsurfdofs] = d2residual_by_dcdiota_flattened     # noqa (1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs] = d2residual_by_diotadiota_flattened  # noqa (1, 1) diotadiota
        H[:, nsurfdofs, nsurfdofs+1] = d2residual_by_diotadiota_flattened  # noqa (1, 2) diotadG
        H[:, nsurfdofs+1, :nsurfdofs] = d2residual_by_dcdG_flattened        # noqa (2, 0) dGdc
        H[:, nsurfdofs+1, nsurfdofs] = d2residual_by_diotadG_flattened     # noqa (2, 1) dGdiota
        H[:, nsurfdofs+1, nsurfdofs+1] = d2residual_by_dGdG_flattened        # noqa (2, 2) dGdG
    else:
        H = np.zeros((nphi*ntheta*3, nsurfdofs + 1, nsurfdofs + 1))

        H[:, :nsurfdofs, :nsurfdofs] = d2residual_by_dcdc_flattened        # noqa (0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs] = d2residual_by_dcdiota_flattened     # noqa (0, 1) dcdiota
        H[:, nsurfdofs, :nsurfdofs] = d2residual_by_dcdiota_flattened     # noqa (1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs] = d2residual_by_diotadiota_flattened  # noqa (1, 1) diotadiota

    return r, J, H


class QfmResidual(object):
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
        self.surface.dependencies.append(self)
        self.invalidate_cache()

    def invalidate_cache(self):
        x = self.surface.gamma()
        xsemiflat = x.reshape((x.size//3, 3)).copy()
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
        x = self.surface.gamma()
        nphi = x.shape[0]
        ntheta = x.shape[1]
        dx_by_dc = self.surface.dgamma_by_dcoeff()
        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        B = self.biotsavart.B().reshape((nphi, ntheta, 3))
        d_B = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_by_dc)

        N = self.surface.normal()
        norm_N = np.linalg.norm(N, axis=2)
        n = N/norm_N[:, :, None]
        d_N = self.surface.dnormal_by_dcoeff()
        d_norm_N = np.einsum('ijkl,ijk->ijl', d_N, n)
        d_n = d_N/norm_N[:, :, None, None] \
            - N[:, :, :, None] * d_norm_N[:, :, None, :]/norm_N[:, :, None, None]**2

        B_n = np.sum(B * n, axis=2)
        norm_B = np.linalg.norm(B, axis=2)

        d_B_n = np.einsum('ijkl,ijk->ijl', d_B, n) + np.einsum('ijk,ijkl->ijl', B, d_n)
        d_norm_B = np.einsum('ijkl,ijk->ijl', d_B, B/norm_B[:, :, None])

        num = np.sum(B_n**2 * norm_N)
        denom = np.sum(norm_B**2 * norm_N)
        d_num = np.sum(2 * d_B_n * B_n[:, :, None] * norm_N[:, :, None]
                       + B_n[:, :, None]**2 * d_norm_N, axis=(0, 1))
        d_denom = np.sum(2 * d_norm_B * norm_B[:, :, None] * norm_N[:, :, None]
                         + norm_B[:, :, None]**2 * d_norm_N, axis=(0, 1))
        return d_num/denom - d_denom*num/(denom*denom)

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivative wrt the surface coefficients
        """
        x = self.surface.gamma()
        nphi = x.shape[0]
        ntheta = x.shape[1]
        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        d2B_by_dXdX = self.biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))
        B = self.biotsavart.B().reshape((nphi, ntheta, 3))
        dx_by_dc = self.surface.dgamma_by_dcoeff()

        d_B = np.einsum('ijkl,ijkm->ijml', dx_by_dc, dB_by_dX)
        d2_B = np.einsum('ijkpl,ijpn,ijkm->ijlmn', d2B_by_dXdX, dx_by_dc, dx_by_dc)

        N = self.surface.normal()
        norm_N = np.linalg.norm(N, axis=2)
        n = N/norm_N[:, :, None]

        d_N = self.surface.dnormal_by_dcoeff()
        d2_N = self.surface.d2normal_by_dcoeffdcoeff()

        d_norm_N = np.einsum('ijkl,ijk->ijl', d_N, n)
        d2_norm_N = (np.einsum('imjk,imjl->imkl', d_N, d_N)
                     + np.einsum('imjkl,imj->imkl', d2_N, N)
                     - d_norm_N[:, :, :, None]*d_norm_N[:, :, None, :])/norm_N[:, :, None, None]
        d_n = d_N/norm_N[:, :, None, None] \
            - N[:, :, :, None] * d_norm_N[:, :, None, :]/norm_N[:, :, None, None]**2
        d2_n = d2_N/norm_N[:, :, None, None, None] \
            - np.einsum('imjk,iml->imjkl', d_N, d_norm_N)/norm_N[:, :, None, None, None]**2 \
            - np.einsum('imjk,iml->imjlk', d_N, d_norm_N)/norm_N[:, :, None, None, None]**2 \
            + 2 * np.einsum('imj,iml,imk->imjlk', N, d_norm_N, d_norm_N)/norm_N[:, :, None, None, None]**3 \
            - np.einsum('imj,imkl->imjkl', N, d2_norm_N)/norm_N[:, :, None, None, None]**2

        B_n = np.sum(B * n, axis=2)
        norm_B = np.linalg.norm(B, axis=2)

        d_B_n = np.einsum('ijkl,ijk->ijl', d_B, n) + np.einsum('ijk,ijkl->ijl', B, d_n)
        d_norm_B = np.einsum('ijkl,ijk->ijl', d_B, B/norm_B[:, :, None])

        d2_B_n = np.einsum('imjkl,imj->imkl', d2_B, n) \
            + np.einsum('imj,imjkl->imkl', B, d2_n) \
            + np.einsum('imjk,imjl->imkl', d_B, d_n) \
            + np.einsum('imjk,imjl->imlk', d_B, d_n)
        d2_norm_B = (np.einsum('imjkl,imj->imkl', d2_B, B)
                     + np.einsum('imjk,imjl->imkl', d_B, d_B)
                     - np.einsum('imk,iml->imkl', d_norm_B, d_norm_B))/norm_B[:, :, None, None]

        num = np.sum(B_n**2 * norm_N)
        denom = np.sum(norm_B**2 * norm_N)
        d_num = np.sum(2 * d_B_n * B_n[:, :, None] * norm_N[:, :, None]
                       + B_n[:, :, None]**2 * d_norm_N, axis=(0, 1))
        d2_num = np.sum(2*(d2_B_n * B_n[:, :, None, None] \
                           + d_B_n[:, :, :, None] * d_B_n[:, :, None, :]) * norm_N[:, :, None, None]
                        + 2 * d_B_n[:, :, None, :] * B_n[:, :, None, None] * d_norm_N[:, :, :, None]
                        + 2 * d_B_n[:, :, :, None] * B_n[:, :, None, None] * d_norm_N[:, :, None, :]
                        + B_n[:, :, None, None]**2 * d2_norm_N, axis=(0, 1))
        d_denom = np.sum(2 * d_norm_B * norm_B[:, :, None] * norm_N[:, :, None]
                         + norm_B[:, :, None]**2 * d_norm_N, axis=(0, 1))
        d2_denom = np.sum(2*(d2_norm_B * norm_B[:, :, None, None] \
                             + d_norm_B[:, :, :, None] * d_norm_B[:, :, None, :]) * norm_N[:, :, None, None]
                          + 2 * d_norm_B[:, :, None, :] * norm_B[:, :, None, None] * d_norm_N[:, :, :, None]
                          + 2 * d_norm_B[:, :, :, None] * norm_B[:, :, None, None] * d_norm_N[:, :, None, :]
                          + norm_B[:, :, None, None]**2 * d2_norm_N, axis=(0, 1))

        return d2_num/denom - d_num[:, None]*d_denom[None, :]/denom**2 \
            - d_num[None, :]*d_denom[:, None]/denom**2 \
            - d2_denom*num/(denom*denom) \
            + 2 * d_denom[:, None]*d_denom[None, :]*num/(denom**3)
