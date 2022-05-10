from typing import Any

import numpy as np
from nptyping import NDArray, Float

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from simsopt.geo.surface import Surface
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.util.zoo import forward_backward 


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
    
    def dJ_by_dcoils(self):
        return Derivative()


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


def parameter_derivatives(surface: Surface,
                          shape_gradient: NDArray[Any, Float]
                          ) -> NDArray[Any, Float]:
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


class MajorRadius(Optimizable): 
    r"""
    This objective computes the major radius of a toroidal surface with the following formula:
    
    R_major = (1/2*pi) * (V / mean_cross_sectional_area)
    
    where mean_cross_sectional_area = \int^1_0 \int^1_0 z_\theta(xy_\varphi -yx_\varphi) - z_\varphi(xy_\theta-yx_\theta) ~d\theta d\varphi
    
    """
    def __init__(self, boozer_surface, sDIM=15):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        phis = np.linspace(0, 1/(2*in_surface.nfp), sDIM, endpoint=False)
        phis += phis[1]/2

        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())
        
        self.volume = Volume(s)
        self.boozer_surface = boozer_surface
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = boozer_surface.bs
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

    def compute(self, compute_derivatives=0):

        self.surface.set_dofs(self.in_surface.get_dofs())
        surface = self.surface
    
        g = surface.gamma()
        g1 = surface.gammadash1()
        g2 = surface.gammadash2()
        
        x = g[:, :, 0]
        y = g[:, :, 1]
        
        r = np.sqrt(x**2+y**2)
    
        xvarphi = g1[:, :, 0]
        yvarphi = g1[:, :, 1]
        zvarphi = g1[:, :, 2]
    
        xtheta = g2[:, :, 0]
        ytheta = g2[:, :, 1]
        ztheta = g2[:, :, 2]
    
        mean_area = np.mean((ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta))/r)/(2*np.pi)
        R_major = self.volume.J() / (2. * np.pi * mean_area)
        self._J = R_major
        
        bs = self.biotsavart
        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = boozer_surface_dexactresidual_dcoils_dcurrents_vjp
        
        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.zeros(L.shape[0])
        dj_ds = self.dJ_dsurfacecoefficients()
        dJ_ds[:dj_ds.size] = dj_ds
        adj = forward_backward(P, L, U, dJ_ds)
        
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, bs)
        self._dJ = -1 * adj_times_dg_dcoil

    def dJ_dsurfacecoefficients(self):
        """
        Return the objective value
        """
    
        self.surface.set_dofs(self.in_surface.get_dofs())
        surface = self.surface
    
        g = surface.gamma()
        g1 = surface.gammadash1()
        g2 = surface.gammadash2()
    
        dg_ds = surface.dgamma_by_dcoeff()
        dg1_ds = surface.dgammadash1_by_dcoeff()
        dg2_ds = surface.dgammadash2_by_dcoeff()
    
        x = g[:, :, 0, None]
        y = g[:, :, 1, None]
    
        dx_ds = dg_ds[:, :, 0, :]
        dy_ds = dg_ds[:, :, 1, :]
    
        r = np.sqrt(x**2+y**2)
        dr_ds = (x*dx_ds+y*dy_ds)/r
    
        xvarphi = g1[:, :, 0, None]
        yvarphi = g1[:, :, 1, None]
        zvarphi = g1[:, :, 2, None]
    
        xtheta = g2[:, :, 0, None]
        ytheta = g2[:, :, 1, None]
        ztheta = g2[:, :, 2, None]
    
        dxvarphi_ds = dg1_ds[:, :, 0, :]
        dyvarphi_ds = dg1_ds[:, :, 1, :]
        dzvarphi_ds = dg1_ds[:, :, 2, :]
    
        dxtheta_ds = dg2_ds[:, :, 0, :]
        dytheta_ds = dg2_ds[:, :, 1, :]
        dztheta_ds = dg2_ds[:, :, 2, :]
    
        mean_area = np.mean((1/r) * (ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta)))
        dmean_area_ds = np.mean((1/(r**2))*((xvarphi * y * ztheta - xtheta * y * zvarphi + x * (-yvarphi * ztheta + ytheta * zvarphi)) * dr_ds + r * (-zvarphi * (ytheta * dx_ds - y * dxtheta_ds - xtheta * dy_ds + x * dytheta_ds) + ztheta * (yvarphi * dx_ds - y * dxvarphi_ds - xvarphi * dy_ds + x * dyvarphi_ds) + (-xvarphi * y + x * yvarphi) * dztheta_ds + (xtheta * y - x * ytheta) * dzvarphi_ds)), axis=(0, 1))
    
        dR_major_ds = (-self.volume.J() * dmean_area_ds + self.volume.dJ_by_dsurfacecoefficients() * mean_area) / mean_area**2
        return dR_major_ds


class NonQuasiAxisymmetricRatio(Optimizable):
    r"""
    This objective decomposes the field magnitude :math:`B(\varphi,\theta)` into quasiaxisymmetric and
    non-quasiaxisymmetric components, then returns a penalty on the latter component.
    
    .. math::
        J &= \frac{1}{2}\int_{\Gamma_{s}} (B-B_{\text{QS}})^2~dS
          &= \frac{1}{2}\int_0^1 \int_0^1 (B - B_{\text{QS}})^2 \|\mathbf n\| ~d\varphi~\d\theta 
    where
    
    .. math::
        B &= \| \mathbf B(\varphi,\theta) \|_2
        B_{\text{QS}} &= \frac{\int_0^1 \int_0^1 B \| n\| ~d\varphi ~d\theta}{\int_0^1 \int_0^1 \|\mathbf n\| ~d\varphi ~d\theta}
    
    """
    def __init__(self, boozer_surface, bs, sDIM=15):
        # only BoozerExact surfaces work for now
        assert boozer_surface.res['type'] == 'exact'
        # only SurfaceXYZTensorFourier for now
        assert type(boozer_surface.surface) is SurfaceXYZTensorFourier 

        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface
        
        phis = np.linspace(0, 1/in_surface.nfp, 2*sDIM, endpoint=False)
        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.in_surface = in_surface
        self.surface = s
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
            res = self.boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=res['iota'], G=res['G'])

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))
        
        # compute J
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size
        
        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        
        nor = surface.normal()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
        B_nonQS = modB - B_QS[None, :]
        self._J = np.mean(dS * B_nonQS**2) / np.mean(dS * B_QS**2)

        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = boozer_surface_dexactresidual_dcoils_dcurrents_vjp

        dJ_by_dB = self.dJ_by_dB().reshape((-1, 3))
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.concatenate((self.dJ_by_dsurfacecoefficients(), [0., 0.]))
        adj = forward_backward(P, L, U, dJ_ds)
        
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, booz_surf.bs)
        self._dJ = dJ_by_dcoils-adj_times_dg_dcoil

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        nor = surface.normal()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        denom = np.mean(dS, axis=0)
        B_QS = np.mean(modB * dS, axis=0) / denom
        B_nonQS = modB - B_QS[None, :]
        
        dmodB_dB = B / modB[..., None]
        dnum_by_dB = B_nonQS[..., None] * dmodB_dB * dS[:, :, None] / (nphi * ntheta)  # d J_nonQS / dB_ijk
        ddenom_by_dB = B_QS[None, :, None] * dmodB_dB * dS[:, :, None] / (nphi * ntheta)  # dJ_QS/dB_ijk
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

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        
        nor = surface.normal()
        dnor_dc = surface.dnormal_by_dcoeff()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
        dS_dc = (nor[:, :, 0, None]*dnor_dc[:, :, 0, :] + nor[:, :, 1, None]*dnor_dc[:, :, 1, :] + nor[:, :, 2, None]*dnor_dc[:, :, 2, :])/dS[:, :, None]

        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
        B_nonQS = modB - B_QS[None, :]
        
        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        dx_dc = surface.dgamma_by_dcoeff()
        dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)
        
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        dmodB_dc = (B[:, :, 0, None] * dB_dc[:, :, 0, :] + B[:, :, 1, None] * dB_dc[:, :, 1, :] + B[:, :, 2, None] * dB_dc[:, :, 2, :])/modB[:, :, None]
        
        num = np.mean(modB * dS, axis=0)
        denom = np.mean(dS, axis=0)
        dnum_dc = np.mean(dmodB_dc * dS[..., None] + modB[..., None] * dS_dc, axis=0) 
        ddenom_dc = np.mean(dS_dc, axis=0)
        B_QS_dc = (dnum_dc * denom[:, None] - ddenom_dc * num[:, None])/denom[:, None]**2
        B_nonQS_dc = dmodB_dc - B_QS_dc[None, :]
        
        num = 0.5*np.mean(dS * B_nonQS**2)
        denom = 0.5*np.mean(dS * B_QS**2)
        dnum_by_dc = np.mean(0.5*dS_dc * B_nonQS[..., None]**2 + dS[..., None] * B_nonQS[..., None] * B_nonQS_dc, axis=(0, 1)) 
        ddenom_by_dc = np.mean(0.5*dS_dc * B_QS[..., None]**2 + dS[..., None] * B_QS[..., None] * B_QS_dc, axis=(0, 1)) 
        dJ_by_dc = (denom * dnum_by_dc - num * ddenom_by_dc) / denom**2 
        return dJ_by_dc


class Iotas(Optimizable):
    def __init__(self, boozer_surface):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[boozer_surface])
        self.boozer_surface = boozer_surface
        self.biotsavart = boozer_surface.bs 
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
        self._dJ_by_dcoefficients = None
        self._dJ_by_dcoilcurrents = None
    
    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=res['iota'], G=res['G'])

        self._J = self.boozer_surface.res['iota']
        
        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = boozer_surface_dexactresidual_dcoils_dcurrents_vjp

        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.zeros(L.shape[0])
        dJ_ds[-2] = 1.
        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, booz_surf.bs)
        self._dJ = -1.*adj_times_dg_dcoil


def boozer_surface_dexactresidual_dcoils_dcurrents_vjp(lm, booz_surf, iota, G, biotsavart):
    """
    For a given surface with points x on it, this function computes the
    vector-Jacobian product of \lm^T * dresidual_dcoils:
    lm^T dresidual_dcoils    = [G*lm - lm(2*||B_BS(x)|| (x_phi + iota * x_theta) ]^T * dB_dcoils
    lm^T dresidual_dcurrents = [G*lm - lm(2*||B_BS(x)|| (x_phi + iota * x_theta) ]^T * dB_dcurrents
    
    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """
    surface = booz_surf.surface
    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum(np.abs(biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    res, dres_dB = boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=0)
    dres_dB = dres_dB.reshape((-1, 3, 3))

    lm_label = lm[-1]
    lmask = np.zeros(booz_surf.res["mask"].shape)
    lmask[booz_surf.res["mask"]] = lm[:-1]
    lm_cons = lmask.reshape((-1, 3))
   
    lm_times_dres_dB = np.sum(lm_cons[:, :, None] * dres_dB, axis=1).reshape((-1, 3))
    lm_times_dres_dcoils = biotsavart.B_vjp(lm_times_dres_dB)
    lm_times_dlabel_dcoils = lm_label*booz_surf.label.dJ_by_dcoils()
    return lm_times_dres_dcoils+lm_times_dlabel_dcoils


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

    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    residual = G*B - np.sum(B**2, axis=2)[..., None] * tang

    GI = np.eye(3, 3) * G
    dresidual_dB = GI[None, None, :, :] - 2. * tang[:, :, :, None] * B[:, :, None, :]

    residual_flattened = residual.reshape((nphi*ntheta*3, ))
    dresidual_dB_flattened = dresidual_dB.reshape((nphi*ntheta*3, 3))
    r = residual_flattened
    dr_dB = dresidual_dB_flattened
    if derivatives == 0:
        return r, dr_dB

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
        d2residual_dGdB[:, :, :] = np.eye(3)[None, :, :]
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
