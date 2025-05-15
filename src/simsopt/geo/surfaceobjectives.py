import numpy as np

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import Derivative, derivative_dec
from .._core.types import RealArray
from .surface import Surface
from .surfacexyztensorfourier import SurfaceXYZTensorFourier
from ..objectives.utilities import forward_backward

__all__ = ['Area', 'Volume', 'ToroidalFlux', 'PrincipalCurvature',
           'QfmResidual', 'boozer_surface_residual', 'Iotas',
           'MajorRadius', 'NonQuasiSymmetricRatio', 'BoozerResidual',
           'AspectRatio']


class AspectRatio(Optimizable):
    """
    Wrapper class for surface aspect ratio.
    """

    def __init__(self, surface, range=None, nphi=None, ntheta=None):

        if range is not None or nphi is not None or ntheta is not None:
            if range is None:
                if surface.stellsym:
                    range = Surface.RANGE_HALF_PERIOD
                else:
                    range = Surface.RANGE_FIELD_PERIOD
            if nphi is None:
                nphi = len(surface.quadpoints_phi)
            if ntheta is None:
                ntheta = len(surface.quadpoints_theta)
            self.surface = surface.__class__.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=range, nfp=surface.nfp, stellsym=surface.stellsym,
                                                              mpol=surface.mpol, ntor=surface.ntor, dofs=surface.dofs)
        else:
            self.surface = surface

        self.range = range
        self.nphi = nphi
        self.ntheta = ntheta

        super().__init__(depends_on=[self.surface])

    def J(self):
        """
        Compute the aspect ratio of a surface.
        """
        return self.surface.aspect_ratio()

    @derivative_dec
    def dJ(self):
        return Derivative({self.surface: self.dJ_by_dsurfacecoefficients()})

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the partial derivatives with respect to the surface coefficients.
        """
        return self.surface.daspect_ratio_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second partial derivatives with respect to the surface coefficients.
        """
        return self.surface.d2aspect_ratio_by_dcoeff_dcoeff()


class Area(Optimizable):
    """
    Wrapper class for surface area label.
    """

    def __init__(self, surface, range=None, nphi=None, ntheta=None):

        if range is not None or nphi is not None or ntheta is not None:
            if range is None:
                if surface.stellsym:
                    range = Surface.RANGE_HALF_PERIOD
                else:
                    range = Surface.RANGE_FIELD_PERIOD
            if nphi is None:
                nphi = len(surface.quadpoints_phi)
            if ntheta is None:
                ntheta = len(surface.quadpoints_theta)
            self.surface = surface.__class__.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=range, nfp=surface.nfp, stellsym=surface.stellsym,
                                                              mpol=surface.mpol, ntor=surface.ntor, dofs=surface.dofs)
        else:
            self.surface = surface

        self.range = range
        self.nphi = nphi
        self.ntheta = ntheta

        super().__init__(depends_on=[self.surface])

    def J(self):
        """
        Compute the area of a surface.
        """
        return self.surface.area()

    @derivative_dec
    def dJ(self):
        return Derivative({self.surface: self.dJ_by_dsurfacecoefficients()})

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the partial derivatives with respect to the surface coefficients.
        """
        return self.surface.darea_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second partial derivatives with respect to the surface coefficients.
        """
        return self.surface.d2area_by_dcoeffdcoeff()


class Volume(Optimizable):
    """
    Wrapper class for volume label.
    """

    def __init__(self, surface, range=None, nphi=None, ntheta=None):

        if range is not None or nphi is not None or ntheta is not None:
            if range is None:
                if surface.stellsym:
                    range = Surface.RANGE_HALF_PERIOD
                else:
                    range = Surface.RANGE_FIELD_PERIOD
            if nphi is None:
                nphi = len(surface.quadpoints_phi)
            if ntheta is None:
                ntheta = len(surface.quadpoints_theta)
            self.surface = surface.__class__.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=range, nfp=surface.nfp, stellsym=surface.stellsym,
                                                              mpol=surface.mpol, ntor=surface.ntor, dofs=surface.dofs)
        else:
            self.surface = surface

        self.range = range
        self.nphi = nphi
        self.ntheta = ntheta

        super().__init__(depends_on=[self.surface])

    def J(self):
        """
        Compute the volume enclosed by the surface.
        """
        return self.surface.volume()

    @derivative_dec
    def dJ(self):
        return Derivative({self.surface: self.dJ_by_dsurfacecoefficients()})

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

    def __init__(self, surface, biotsavart, idx=0, range=None, nphi=None, ntheta=None):

        if range is not None or nphi is not None or ntheta is not None:
            if range is None:
                if surface.stellsym:
                    range = Surface.RANGE_HALF_PERIOD
                else:
                    range = Surface.RANGE_FIELD_PERIOD
            if nphi is None:
                nphi = len(surface.quadpoints_phi)
            if ntheta is None:
                ntheta = len(surface.quadpoints_theta)
            self.surface = surface.__class__.from_nphi_ntheta(nphi=nphi, ntheta=ntheta, range=range, nfp=surface.nfp, stellsym=surface.stellsym,
                                                              mpol=surface.mpol, ntor=surface.ntor, dofs=surface.dofs)
        else:
            self.surface = surface

        self.biotsavart = biotsavart
        self.idx = idx

        self.range = range
        self.nphi = nphi
        self.ntheta = ntheta

        super().__init__(depends_on=[self.surface, biotsavart])

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

    @derivative_dec
    def dJ(self):
        d_s = Derivative({self.surface: self.dJ_by_dsurfacecoefficients()})
        d_c = self.dJ_by_dcoils()
        return d_s + d_c

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the partial derivatives with respect to the surface coefficients.
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
        Calculate the second partial derivatives with respect to the surface coefficients.
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

    def dJ_by_dcoils(self):
        """
        Calculate the partial derivatives with respect to the coil coefficients.
        """
        xtheta = self.surface.gammadash2()[self.idx]
        ntheta = self.surface.gamma().shape[1]
        dJ_by_dA = xtheta/ntheta
        dJ_by_dcoils = self.biotsavart.A_vjp(dJ_by_dA)
        return dJ_by_dcoils


class PrincipalCurvature(Optimizable):
    r"""

    Given a Surface, evaluates a metric based on the principal curvatures,
    :math:`\kappa_1` and :math:`\kappa_2`, where :math:`\kappa_1>\kappa_2`.
    This metric is designed to penalize :math:`\kappa_1 > \kappa_{\max,1}` and
    :math:`-\kappa_2 > \kappa_{\max,2}`.

    .. math::
       J &= \int d^2 x \exp \left(- ( \kappa_1 - \kappa_{\max,1})/w_1) \right) \\
         &+ \int d^2 x \exp \left(- (-\kappa_2 - \kappa_{\max,2})/w_2) \right).

    This metric can be used as a regularization within fixed-boundary optimization
    to prevent, for example, surfaces with concave regions
    (large values of :math:`|\kappa_2|`) or surfaces with large elongation
    (large values of :math:`\kappa_1`).

    """

    def __init__(self, surface, kappamax1=1, kappamax2=1, weight1=0.05, weight2=0.05):
        super().__init__(depends_on=[surface])
        self.surface = surface
        self.kappamax1 = kappamax1
        self.kappamax2 = kappamax2
        self.weight1 = weight1
        self.weight2 = weight2

    def J(self):
        curvature = self.surface.surface_curvatures()
        k1 = curvature[:, :, 2]  # larger
        k2 = curvature[:, :, 3]  # smaller
        normal = self.surface.normal()
        norm_normal = np.sqrt(normal[:, :, 0]**2 + normal[:, :, 1]**2 + normal[:, :, 2]**2)
        return np.sum(norm_normal * np.exp(-(k1 - self.kappamax1)/self.weight1)) + \
            np.sum(norm_normal * np.exp(-(-k2 - self.kappamax2)/self.weight2))

    @derivative_dec
    def dJ(self):
        curvature = self.surface.surface_curvatures()
        k1 = curvature[:, :, 2]  # larger
        k2 = curvature[:, :, 3]  # smaller
        normal = self.surface.normal()
        norm_normal = np.sqrt(normal[:, :, 0]**2 + normal[:, :, 1]**2 + normal[:, :, 2]**2)
        dcurvature_dc = self.surface.dsurface_curvatures_by_dcoeff()
        dk1_dc = dcurvature_dc[:, :, 2, :]
        dk2_dc = dcurvature_dc[:, :, 3, :]
        dnormal_dc = self.surface.dnormal_by_dcoeff()
        dnorm_normal_dc = normal[:, :, 0, None]*dnormal_dc[:, :, 0, :]/norm_normal[:, :, None] + \
            normal[:, :, 1, None]*dnormal_dc[:, :, 1, :]/norm_normal[:, :, None] + \
            normal[:, :, 2, None]*dnormal_dc[:, :, 2, :]/norm_normal[:, :, None]
        deriv = np.sum(dnorm_normal_dc * np.exp(-(k1[:, :, None] - self.kappamax1)/self.weight1), axis=(0, 1)) + \
            np.sum(norm_normal[:, :, None] * np.exp(-(k1[:, :, None] - self.kappamax1)/self.weight1) * (- dk1_dc/self.weight1), axis=(0, 1)) + \
            np.sum(dnorm_normal_dc * np.exp(-(-k2[:, :, None] - self.kappamax2)/self.weight2), axis=(0, 1)) + \
            np.sum(norm_normal[:, :, None] * np.exp(-(-k2[:, :, None] - self.kappamax2)/self.weight2) * (dk2_dc/self.weight2), axis=(0, 1))
        return Derivative({self.surface: deriv})


def boozer_surface_residual(surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=False):
    r"""
    For a given surface, this function computes the
    residual

    .. math::
        G\mathbf B_\text{BS}(\mathbf x) - \|\mathbf B_\text{BS}(\mathbf x)\|^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)

    as well as the derivatives of this residual with respect to surface dofs,
    iota, and G.  In the above, :math:`\mathbf x` are points on the surface, :math:`\iota` is the
    rotational transform on that surface, and :math:`\mathbf B_{\text{BS}}` is the magnetic field
    computed using the Biot-Savart law.

    :math:`G` is known for exact boozer surfaces, so if ``G=None`` is passed, then that
    value is used instead.

    Args:
        surface: The surface to use for the computation
        iota: the surface rotational transform
        G: a constant that is a function of the coil currents in vacuum field
        biotsavart: the Biot-Savart magnetic field
        derivatives: how many spatial derivatives of the residual to compute
        weight_inv_modB: whether or not to weight the residual by :math:`1/\|\mathbf B\|`.  This 
                         is useful to activate so that the residual does not scale with the 
                         coil currents.

    Returns:
        the residual at the surface quadrature points and optionally the spatial derivatives
        of the residual.


    """

    assert derivatives in [0, 1, 2]

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

    if weight_inv_modB:
        modB = np.sqrt(B2)
        w = 1./modB
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

    if weight_inv_modB:
        dB2_dc = 2*np.einsum('ijk,ijkl->ijl', B, dB_dc, optimize=True)
        dmodB_dc = 0.5*dB2_dc/np.sqrt(B2[:, :, None])
        dw_dc = -dmodB_dc/modB[:, :, None]**2
        drtil_dc = residual[..., None] * dw_dc[:, :, None, :] + w[:, :, None, None] * dresidual_dc
        drtil_diota = w[:, :, None] * dresidual_diota
    else:
        drtil_dc = dresidual_dc.copy()
        drtil_diota = dresidual_diota.copy()

    drtil_dc_flattened = drtil_dc.reshape((nphi*ntheta*3, nsurfdofs))
    drtil_diota_flattened = drtil_diota.reshape((nphi*ntheta*3, 1))

    if user_provided_G:
        dresidual_dG = B

        if weight_inv_modB:
            drtil_dG = w[:, :, None] * dresidual_dG
        else:
            drtil_dG = dresidual_dG.copy()

        drtil_dG_flattened = drtil_dG.reshape((nphi*ntheta*3, 1))
        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened, drtil_dG_flattened), axis=1)
    else:
        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened), axis=1)

    if derivatives == 1:
        return r, J

    d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))
    d2B_dcdc = np.einsum('ijkpl,ijpn,ijkm->ijlmn', d2B_by_dXdX, dx_dc, dx_dc, optimize=True)
    dB2_dc = 2. * np.einsum('ijl,ijlm->ijm', B, dB_dc, optimize=True)

    term1 = np.einsum('ijlm,ijln->ijmn', dB_dc, dB_dc, optimize=True)
    term2 = np.einsum('ijlmn,ijl->ijmn', d2B_dcdc, B, optimize=True)
    d2B2_dcdc = 2*(term1 + term2)

    term1 = -(dxphi_dc[..., None, :] + iota * dxtheta_dc[..., None, :]) * dB2_dc[..., None, :, None]
    term2 = -(dxphi_dc[..., :, None] + iota * dxtheta_dc[..., :, None]) * dB2_dc[..., None, None, :]
    term3 = -(xphi[..., None, None] + iota * xtheta[..., None, None]) * d2B2_dcdc[..., None, :, :]
    d2residual_by_dcdc = G * d2B_dcdc + term1 + term2 + term3
    d2residual_by_dcdiota = -(dB2_dc[..., None, :] * xtheta[..., :, None] + B2[..., None, None] * dxtheta_dc)
    d2residual_by_diotadiota = np.zeros(dresidual_diota.shape)

    if weight_inv_modB:
        d2B2_dcdc = 2*(np.einsum('ijlm,ijln->ijmn', dB_dc, dB_dc, optimize=True)+np.einsum('ijkpl,ijpn,ijkm,ijl->ijmn', d2B_by_dXdX, dx_dc, dx_dc, B, optimize=True))
        d2modB_dc2 = (2*B2[:, :, None, None] * d2B2_dcdc - dB2_dc[:, :, :, None]*dB2_dc[:, :, None, :])*(1/(4*B2[:, :, None, None]**1.5))
        d2w_dc2 = (2*dmodB_dc[:, :, :, None] * dmodB_dc[:, :, None, :] - modB[:, :, None, None] * d2modB_dc2)/modB[:, :, None, None]**3.

        d2rtil_dcdc = residual[..., None, None] * d2w_dc2[:, :, None, ...] \
            + dw_dc[:, :, None, :, None] * dresidual_dc[:, :, :, None, :] \
            + dw_dc[:, :, None, None, :] * dresidual_dc[:, :, :, :, None] \
            + w[:, :, None, None, None] * d2residual_by_dcdc
        d2rtil_dcdiota = w[:, :, None, None] * d2residual_by_dcdiota + dw_dc[:, :, None, :] * dresidual_diota[..., None]
        d2rtil_diotadiota = np.zeros(dresidual_diota.shape)
    else:
        d2rtil_dcdc = d2residual_by_dcdc.copy()
        d2rtil_dcdiota = d2residual_by_dcdiota.copy()
        d2rtil_diotadiota = d2residual_by_diotadiota.copy()

    d2rtil_dcdc_flattened = d2rtil_dcdc.reshape((nphi*ntheta*3, nsurfdofs, nsurfdofs))
    d2rtil_dcdiota_flattened = d2rtil_dcdiota.reshape((nphi*ntheta*3, nsurfdofs))
    d2rtil_diotadiota_flattened = d2rtil_diotadiota.reshape((nphi*ntheta*3,))

    if user_provided_G:
        d2residual_by_dcdG = dB_dc
        if weight_inv_modB:
            d2rtil_dcdG = dw_dc[:, :, None, :] * dresidual_dG[..., None] + w[:, :, None, None] * d2residual_by_dcdG
        else:
            d2rtil_dcdG = d2residual_by_dcdG.copy()

        d2rtil_dGdG = np.zeros(dresidual_dG.shape)
        d2rtil_dcdG_flattened = d2rtil_dcdG.reshape((nphi*ntheta*3, nsurfdofs))
        d2rtil_diotadG_flattened = np.zeros((nphi*ntheta*3,))
        d2rtil_dGdG_flattened = d2rtil_dGdG.reshape((nphi*ntheta*3,))

        H = np.zeros((nphi*ntheta*3, nsurfdofs + 2, nsurfdofs + 2))
        # noqa turns out linting so that we can align everything neatly
        H[:, :nsurfdofs, :nsurfdofs] = d2rtil_dcdc_flattened        # noqa (0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs] = d2rtil_dcdiota_flattened     # noqa (0, 1) dcdiota
        H[:, :nsurfdofs, nsurfdofs+1] = d2rtil_dcdG_flattened        # noqa (0, 2) dcdG
        H[:, nsurfdofs, :nsurfdofs] = d2rtil_dcdiota_flattened     # noqa (1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs] = d2rtil_diotadiota_flattened  # noqa (1, 1) diotadiota
        H[:, nsurfdofs, nsurfdofs+1] = d2rtil_diotadiota_flattened  # noqa (1, 2) diotadG
        H[:, nsurfdofs+1, :nsurfdofs] = d2rtil_dcdG_flattened        # noqa (2, 0) dGdc
        H[:, nsurfdofs+1, nsurfdofs] = d2rtil_diotadG_flattened     # noqa (2, 1) dGdiota
        H[:, nsurfdofs+1, nsurfdofs+1] = d2rtil_dGdG_flattened        # noqa (2, 2) dGdG
    else:
        H = np.zeros((nphi*ntheta*3, nsurfdofs + 1, nsurfdofs + 1))
        H[:, :nsurfdofs, :nsurfdofs] = d2rtil_dcdc_flattened        # noqa (0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs] = d2rtil_dcdiota_flattened     # noqa (0, 1) dcdiota
        H[:, nsurfdofs, :nsurfdofs] = d2rtil_dcdiota_flattened     # noqa (1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs] = d2rtil_diotadiota_flattened  # noqa (1, 1) diotadiota

    return r, J, H


def parameter_derivatives(surface: Surface,
                          shape_gradient: RealArray
                          ) -> RealArray:
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


class MajorRadius(Optimizable):
    r"""
    This wrapper objective computes the major radius of a toroidal Boozer surface and supplies
    its derivative with respect to coils

    Args:
        boozer_surface: The surface to use for the computation
    """

    def __init__(self, boozer_surface):
        super().__init__(depends_on=[boozer_surface])
        self.boozer_surface = boozer_surface
        self.surface = boozer_surface.surface
        self.recompute_bell()

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        surface = self.surface
        self._J = surface.major_radius()

        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = self.boozer_surface.res['vjp']

        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.zeros(L.shape[0])
        dj_ds = surface.dmajor_radius_by_dcoeff()
        dJ_ds[:dj_ds.size] = dj_ds
        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = -1 * adj_times_dg_dcoil


class NonQuasiSymmetricRatio(Optimizable):
    r"""
    This objective decomposes the field magnitude :math:`B(\varphi,\theta)` into quasisymmetric and
    non-quasisymmetric components.  For quasi-axisymmetry, we compute

    .. math::
        B_{\text{QS}} &= \frac{\int_0^1 B \|\mathbf n\| ~d\varphi}{\int_0^1 \|\mathbf n\| ~d\varphi} \\
        B_{\text{non-QS}} &= B - B_{\text{QS}}

    where :math:`B = \| \mathbf B(\varphi,\theta) \|_2`.  
    For quasi-poloidal symmetry, an analagous formula is used, but the integral is computed in the :math:`\theta` direction.
    The objective computed by this penalty is

    .. math::
        J &= \frac{\int_{\Gamma_{s}} B_{\text{non-QS}}^2~dS}{\int_{\Gamma_{s}} B_{\text{QS}}^2~dS} \\

    When :math:`J` is zero, then there is perfect QS on the given boozer surface. The ratio of the QS and non-QS components
    of the field is returned to avoid dependence on the magnitude of the field strength.  Note that this penalty is computed
    on an auxilliary surface with quadrature points that are different from those on the input Boozer surface.  This is to allow
    for a spectrally accurate evaluation of the above integrals. Note that if boozer_surface.surface.stellsym == True, 
    computing this term on the half-period with shifted quadrature points is ~not~ equivalent to computing on the full-period 
    with unshifted points.  This is why we compute on an auxilliary surface with quadrature points on the full period.

    Args:
        boozer_surface: input boozer surface on which the penalty term is evaluated,
        biotsavart: biotsavart object (not necessarily the same as the one used on the Boozer surface). 
        sDIM: integer that determines the resolution of the quadrature points placed on the auxilliary surface.  
        quasi_poloidal: `False` for quasiaxisymmetry and `True` for quasipoloidal symmetry
    """

    def __init__(self, boozer_surface, bs, sDIM=20, quasi_poloidal=False):
        # only SurfaceXYZTensorFourier for now
        assert type(boozer_surface.surface) is SurfaceXYZTensorFourier

        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface

        surface = in_surface
        phis = np.linspace(0, 1/in_surface.nfp, 2*sDIM, endpoint=False)
        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        surface = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas, dofs=in_surface.dofs)

        self.axis = 1 if quasi_poloidal else 0
        self.in_surface = in_surface
        self.surface = surface
        self.biotsavart = bs
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))
        axis = self.axis

        # compute J
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)

        nor = surface.normal()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        B_QS = np.mean(modB * dS, axis=axis) / np.mean(dS, axis=axis)

        if axis == 0:
            B_QS = B_QS[None, :]
        else:
            B_QS = B_QS[:, None]

        B_nonQS = modB - B_QS
        self._J = np.mean(dS * B_nonQS**2) / np.mean(dS * B_QS**2)

        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = self.boozer_surface.res['vjp']

        dJ_by_dB = self.dJ_by_dB().reshape((-1, 3))
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.zeros(L.shape[0])
        dj_ds = self.dJ_by_dsurfacecoefficients()
        dJ_ds[:dj_ds.size] = dj_ds
        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils-adj_times_dg_dcoil

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size
        axis = self.axis

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))

        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        nor = surface.normal()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        denom = np.mean(dS, axis=axis)
        B_QS = np.mean(modB * dS, axis=axis) / denom

        if axis == 0:
            B_QS = B_QS[None, :]
        else:
            B_QS = B_QS[:, None]

        B_nonQS = modB - B_QS

        dmodB_dB = B / modB[..., None]
        dnum_by_dB = B_nonQS[..., None] * dmodB_dB * dS[:, :, None] / (nphi * ntheta)  # d J_nonQS / dB_ijk
        ddenom_by_dB = B_QS[..., None] * dmodB_dB * dS[:, :, None] / (nphi * ntheta)  # dJ_QS/dB_ijk
        num = 0.5*np.mean(dS * B_nonQS**2)
        denom = 0.5*np.mean(dS * B_QS**2)
        return (denom * dnum_by_dB - num * ddenom_by_dB) / denom**2

    def dJ_by_dsurfacecoefficients(self):
        """
        Return the partial derivative of the objective with respect to the surface coefficients
        """
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size
        axis = self.axis

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)

        nor = surface.normal()
        dnor_dc = surface.dnormal_by_dcoeff()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
        dS_dc = (nor[:, :, 0, None]*dnor_dc[:, :, 0, :] + nor[:, :, 1, None]*dnor_dc[:, :, 1, :] + nor[:, :, 2, None]*dnor_dc[:, :, 2, :])/dS[:, :, None]

        B_QS = np.mean(modB * dS, axis=axis) / np.mean(dS, axis=axis)

        if axis == 0:
            B_QS = B_QS[None, :]
        else:
            B_QS = B_QS[:, None]

        B_nonQS = modB - B_QS

        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        dx_dc = surface.dgamma_by_dcoeff()
        dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)

        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        dmodB_dc = (B[:, :, 0, None] * dB_dc[:, :, 0, :] + B[:, :, 1, None] * dB_dc[:, :, 1, :] + B[:, :, 2, None] * dB_dc[:, :, 2, :])/modB[:, :, None]

        num = np.mean(modB * dS, axis=axis)
        denom = np.mean(dS, axis=axis)
        dnum_dc = np.mean(dmodB_dc * dS[..., None] + modB[..., None] * dS_dc, axis=axis)
        ddenom_dc = np.mean(dS_dc, axis=axis)
        B_QS_dc = (dnum_dc * denom[:, None] - ddenom_dc * num[:, None])/denom[:, None]**2

        if axis == 0:
            B_QS_dc = B_QS_dc[None, :, :]
        else:
            B_QS_dc = B_QS_dc[:, None, :]

        B_nonQS_dc = dmodB_dc - B_QS_dc

        num = 0.5*np.mean(dS * B_nonQS**2)
        denom = 0.5*np.mean(dS * B_QS**2)
        dnum_by_dc = np.mean(0.5*dS_dc * B_nonQS[..., None]**2 + dS[..., None] * B_nonQS[..., None] * B_nonQS_dc, axis=(0, 1))
        ddenom_by_dc = np.mean(0.5*dS_dc * B_QS[..., None]**2 + dS[..., None] * B_QS[..., None] * B_QS_dc, axis=(0, 1))
        dJ_by_dc = (denom * dnum_by_dc - num * ddenom_by_dc) / denom**2
        return dJ_by_dc


class Iotas(Optimizable):
    """
    This term returns the rotational transform on a boozer surface as well as its derivative
    with respect to the coil degrees of freedom.
    """

    def __init__(self, boozer_surface):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[boozer_surface])
        self.boozer_surface = boozer_surface
        self.biotsavart = boozer_surface.biotsavart
        self.recompute_bell()

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None
        self._dJ_by_dcoefficients = None
        self._dJ_by_dcoilcurrents = None

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self._J = self.boozer_surface.res['iota']

        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = self.boozer_surface.res['vjp']

        dJ_ds = np.zeros(L.shape[0])
        if G is not None:
            # tack on dJ_diota = 1, and  dJ_dG = 0 to the end of dJ_ds
            dJ_ds[-2] = 1.
        else:
            # tack on dJ_diota = 1 to the end of dJ_ds
            dJ_ds[-1] = 1.

        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = -1.*adj_times_dg_dcoil


class BoozerResidual(Optimizable):
    r"""
    This term returns the Boozer residual penalty term

    .. math::
       J = \int_0^{1/n_{\text{fp}}} \int_0^1 \| \mathbf r \|^2 ~d\theta ~d\varphi + w (\text{label.J()-boozer_surface.constraint_weight})^2.

    where

    .. math::
        \mathbf r = \frac{1}{\|\mathbf B\|}[G\mathbf B_\text{BS}(\mathbf x) - ||\mathbf B_\text{BS}(\mathbf x)||^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)]

    """

    def __init__(self, boozer_surface, bs):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface

        # same number of points as on the solved surface
        phis = in_surface.quadpoints_phi
        thetas = in_surface.quadpoints_theta

        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.constraint_weight = boozer_surface.constraint_weight
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def J(self):
        """
        Return the value of the penalty function.
        """

        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        """
        Return the derivative of the penalty function with respect to the coil degrees of freedom.
        """

        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))

        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        # compute J
        surface = self.surface
        iota = self.boozer_surface.res['iota']
        G = self.boozer_surface.res['G']
        r, J = boozer_surface_residual(surface, iota, G, self.biotsavart, derivatives=1, weight_inv_modB=self.boozer_surface.res['weight_inv_modB'])
        rtil = np.concatenate((r/np.sqrt(num_points), [np.sqrt(self.constraint_weight)*(self.boozer_surface.label.J()-self.boozer_surface.targetlabel)]))
        self._J = 0.5*np.sum(rtil**2)

        booz_surf = self.boozer_surface
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = booz_surf.res['vjp']

        dJ_by_dB = self.dJ_by_dB()
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
        dl = np.zeros((J.shape[1],))
        dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        dl[:dlabel_dsurface.size] = dlabel_dsurface
        Jtil = np.concatenate((J/np.sqrt(num_points), np.sqrt(self.constraint_weight) * dl[None, :]), axis=0)
        dJ_ds = Jtil.T@rtil

        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """

        surface = self.surface
        res = self.boozer_surface.res
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        r, r_dB = boozer_surface_residual_dB(surface, self.boozer_surface.res['iota'], self.boozer_surface.res['G'], self.biotsavart, derivatives=0, weight_inv_modB=res['weight_inv_modB'])

        r /= np.sqrt(num_points)
        r_dB /= np.sqrt(num_points)

        dJ_by_dB = r[:, None]*r_dB
        dJ_by_dB = np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)
        return dJ_by_dB


def boozer_surface_dexactresidual_dcoils_dcurrents_vjp(lm, booz_surf, iota, G):
    r"""
    For a given surface with points :math:`x` on it, this function computes the
    vector-Jacobian product of:

    .. math::
        \lambda^T \frac{d\mathbf{r}}{d\text{coils   }} &= [G\lambda - 2\lambda\|\mathbf B(\mathbf x)\| (\mathbf{x}_\varphi + \iota \mathbf{x}_\theta) ]^T \frac{d\mathbf B}{d\text{coils}} \\ 
        \lambda^T \frac{d\mathbf{r}}{d\text{currents}} &= [G\lambda - 2\lambda\|\mathbf B(\mathbf x)\| (\mathbf{x}_\varphi + \iota \mathbf{x}_\theta) ]^T \frac{d\mathbf B}{d\text{currents}}

    where :math:`\mathbf{r}` is the Boozer residual.

    Args:
        lm: adjoint variable,
        booz_surf: boozer surface,
        iota: rotational transform on the boozer surface,
        G: constant on boozer surface,
    """
    surface = booz_surf.surface
    biotsavart = booz_surf.biotsavart

    # G must be provided here
    assert G is not None

    res, dres_dB = boozer_surface_residual_dB(surface, iota, G, biotsavart)
    dres_dB = dres_dB.reshape((-1, 3, 3))

    lm_label = lm[-1]
    lmask = np.zeros(booz_surf.res["mask"].shape)
    lmask[booz_surf.res["mask"]] = lm[:-1]
    lm_cons = lmask.reshape((-1, 3))

    lm_times_dres_dB = np.sum(lm_cons[:, :, None] * dres_dB, axis=1).reshape((-1, 3))
    lm_times_dres_dcoils = biotsavart.B_vjp(lm_times_dres_dB)
    lm_times_dlabel_dcoils = lm_label*booz_surf.label.dJ(partials=True)(biotsavart, as_derivative=True)

    return lm_times_dres_dcoils+lm_times_dlabel_dcoils


def boozer_surface_dlsqgrad_dcoils_vjp(lm, booz_surf, iota, G, weight_inv_modB=True):
    """
    For a given surface with points x on it, this function computes the
    vector-Jacobian product of \lm^T * dlsqgrad_dcoils, \lm^T * dlsqgrad_dcurrents:
    lm^T dresidual_dcoils    = lm^T [dr_dsurface]^T[dr_dcoils]    + sum r_i lm^T d2ri_dsdc
    lm^T dresidual_dcurrents = lm^T [dr_dsurface]^T[dr_dcurrents] + sum r_i lm^T d2ri_dsdcurrents

    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """

    surface = booz_surf.surface
    biotsavart = booz_surf.biotsavart
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size
    num_points = 3 * nphi * ntheta
    # r, dr_dB, J, d2residual_dsurfacedB, d2residual_dsurfacedgradB
    boozer = boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=1, weight_inv_modB=weight_inv_modB)
    r = boozer[0]/np.sqrt(num_points)
    dr_dB = boozer[1].reshape((-1, 3, 3))/np.sqrt(num_points)
    dr_ds = boozer[2]/np.sqrt(num_points)
    d2r_dsdB = boozer[3]/np.sqrt(num_points)
    d2r_dsdgradB = boozer[4]/np.sqrt(num_points)

    v1 = np.sum(np.sum(lm[:, None]*dr_ds.T, axis=0).reshape((-1, 3, 1)) * dr_dB, axis=1)
    v2 = np.sum(r.reshape((-1, 3, 1))*np.sum(lm[None, None, :]*d2r_dsdB, axis=-1).reshape((-1, 3, 3)), axis=1)
    v3 = np.sum(r.reshape((-1, 3, 1, 1))*np.sum(lm[None, None, None, :]*d2r_dsdgradB, axis=-1).reshape((-1, 3, 3, 3)), axis=1)
    dres_dcoils = biotsavart.B_and_dB_vjp(v1+v2, v3)
    return dres_dcoils[0]+dres_dcoils[1]


def boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=False):
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
        G = 2. * np.pi * np.sum(np.abs([c.current.get_value() for c in biotsavart.coils])) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]

    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)

    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    residual = G*B - np.sum(B**2, axis=2)[..., None] * tang

    GI = np.eye(3, 3) * G
    dresidual_dB = GI[None, None, :, :] - 2. * tang[:, :, :, None] * B[:, :, None, :]

    if weight_inv_modB:
        B2 = np.sum(B**2, axis=2)
        modB = np.sqrt(B2)
        w = 1./modB
        dw_dB = -B/B2[:, :, None]**1.5
        rtil = w[:, :, None] * residual
        drtil_dB = residual[:, :, :, None] * dw_dB[:, :, None, :] + dresidual_dB * w[:, :, None, None]
    else:
        rtil = residual.copy()
        drtil_dB = dresidual_dB.copy()

    rtil_flattened = rtil.reshape((nphi*ntheta*3, ))
    drtil_dB_flattened = drtil_dB.reshape((nphi*ntheta*3, 3))

    if derivatives == 0:
        return rtil_flattened, drtil_dB_flattened

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()
    nsurfdofs = dx_dc.shape[-1]

    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)
    dtang_dc = dxphi_dc + iota * dxtheta_dc
    dresidual_dc = G*dB_dc \
        - 2*np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] \
        - np.sum(B**2, axis=2)[..., None, None] * dtang_dc
    dresidual_diota = -np.sum(B**2, axis=2)[..., None] * xtheta

    d2residual_dcdB = -2*dB_dc[:, :, None, :, :] * tang[:, :, :, None, None] - 2*B[:, :, None, :, None] * dtang_dc[:, :, :, None, :]
    d2residual_diotadB = -2.*B[:, :, None, :] * xtheta[:, :, :, None]
    d2residual_dcdgradB = -2.*B[:, :, None, None, :, None]*dx_dc[:, :, None, :, None, :]*tang[:, :, :, None, None, None]
    idx = np.arange(3)
    d2residual_dcdgradB[:, :, idx, :, idx, :] += dx_dc * G

    if weight_inv_modB:
        dB2_dc = 2*np.einsum('ijk,ijkl->ijl', B, dB_dc, optimize=True)
        dmodB_dc = 0.5*dB2_dc/modB[:, :, None]
        dw_dc = -dmodB_dc/B2[:, :, None]

        d2w_dcdB = -(dB_dc * B2[:, :, None, None]**1.5 - 1.5*dB2_dc[:, :, None, :]*modB[:, :, None, None]*B[:, :, :, None])/B2[:, :, None, None]**3
        d2w_dcdgradB = dw_dB[:, :, None, :, None] * dx_dc[:, :, :, None, :]

        drtil_dc = dresidual_dc * w[:, :, None, None] + dw_dc[:, :, None, :] * residual[..., None]
        drtil_diota = w[:, :, None] * dresidual_diota
        d2rtil_dcdB = dresidual_dc[:, :, :, None, :]*dw_dB[:, :, None, :, None]  \
            + dresidual_dB[:, :, :, :, None]*dw_dc[:, :, None, None, :] \
            + d2residual_dcdB*w[:, :, None, None, None] \
            + residual[:, :, :, None, None]*d2w_dcdB[:, :, None, :, :]
        d2rtil_diotadB = dw_dB[:, :, None, :]*dresidual_diota[:, :, :, None] + w[:, :, None, None]*d2residual_diotadB
        d2rtil_dcdgradB = d2w_dcdgradB[:, :, None, :, :, :]*residual[:, :, :, None, None, None] + d2residual_dcdgradB*w[:, :, None, None, None, None]
    else:
        drtil_dc = dresidual_dc.copy()
        drtil_diota = dresidual_diota.copy()
        d2rtil_dcdB = d2residual_dcdB.copy()
        d2rtil_diotadB = d2residual_diotadB.copy()
        d2rtil_dcdgradB = d2residual_dcdgradB.copy()

    drtil_dc_flattened = drtil_dc.reshape((nphi*ntheta*3, nsurfdofs))
    drtil_diota_flattened = drtil_diota.reshape((nphi*ntheta*3, 1))
    d2rtil_dcdB_flattened = d2rtil_dcdB.reshape((nphi*ntheta*3, 3, nsurfdofs))
    d2rtil_diotadB_flattened = d2rtil_diotadB.reshape((nphi*ntheta*3, 3, 1))
    d2rtil_dcdgradB_flattened = d2rtil_dcdgradB.reshape((nphi*ntheta*3, 3, 3, nsurfdofs))
    d2rtil_diotadgradB_flattened = np.zeros((nphi*ntheta*3, 3, 3, 1))

    if user_provided_G:
        dresidual_dG = B
        d2residual_dGdB = np.ones((nphi*ntheta, 3, 3))
        d2residual_dGdB[:, :, :] = np.eye(3)[None, :, :]
        d2residual_dGdB = d2residual_dGdB.reshape((nphi, ntheta, 3, 3))
        d2residual_dGdgradB = np.zeros((nphi, ntheta, 3, 3, 3))

        if weight_inv_modB:
            drtil_dG = dresidual_dG * w[:, :, None]
            d2rtil_dGdB = d2residual_dGdB * w[:, :, None, None] + dw_dB[:, :, None, :]*dresidual_dG[:, :, :, None]
            d2rtil_dGdgradB = d2residual_dGdgradB.copy()
        else:
            drtil_dG = dresidual_dG.copy()
            d2rtil_dGdB = d2residual_dGdB.copy()
            d2rtil_dGdgradB = d2residual_dGdgradB.copy()

        drtil_dG_flattened = drtil_dG.reshape((nphi*ntheta*3, 1))
        d2rtil_dGdB_flattened = d2rtil_dGdB.reshape((nphi*ntheta*3, 3, 1))
        d2rtil_dGdgradB_flattened = d2rtil_dGdgradB.reshape((nphi*ntheta*3, 3, 3, 1))

        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened, drtil_dG_flattened), axis=1)
        d2rtil_dsurfacedB = np.concatenate((d2rtil_dcdB_flattened,
                                            d2rtil_diotadB_flattened,
                                            d2rtil_dGdB_flattened), axis=-1)
        d2rtil_dsurfacedgradB = np.concatenate((d2rtil_dcdgradB_flattened,
                                                d2rtil_diotadgradB_flattened,
                                                d2rtil_dGdgradB_flattened), axis=-1)
    else:
        J = np.concatenate((drtil_dc_flattened, drtil_diota_flattened), axis=1)
        d2rtil_dsurfacedB = np.concatenate((d2rtil_dcdB_flattened, d2rtil_diotadB_flattened), axis=-1)
        d2rtil_dsurfacedgradB = np.concatenate((d2rtil_dcdgradB_flattened, d2rtil_diotadgradB_flattened), axis=-1)

    if derivatives == 1:
        return rtil_flattened, drtil_dB_flattened, J, d2rtil_dsurfacedB, d2rtil_dsurfacedgradB
