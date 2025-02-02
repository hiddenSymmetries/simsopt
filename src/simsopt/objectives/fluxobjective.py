import numpy as np

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec
from ..objectives.utilities import forward_backward
from simsopt.geo import SurfaceXYZTensorFourier


__all__ = ['BoozerSquaredFlux', 'SquaredFlux']

class BoozerSquaredFlux(Optimizable):

    r"""
    Objective representing quadratic-flux-like quantities, useful for stage-2
    coil optimization. Several variations are available, which can be selected
    using the ``definition`` argument. For ``definition="quadratic flux"`` 
    (the default), the objective is defined as

    .. math::
        J = \frac12 \int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds,

    where :math:`\mathbf{n}` is the surface unit normal vector and
    :math:`B_T` is an optional (zero by default) target value for the
    magnetic field. Also :math:`\int_{S} ds` indicates a surface integral.
    For ``definition="normalized"``, the objective is defined as

    .. math::
        J = \frac12 \frac{\int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds}
                         {\int_{S} |\mathbf{B}|^2 ds}.

    For ``definition="local"``, the objective is defined as

    .. math::
        J = \frac12 \int_{S} \frac{(\mathbf{B}\cdot \mathbf{n} - B_T)^2}{|\mathbf{B}|^2} ds.

    The definition ``"quadratic flux"`` has the advantage of simplicity, and it
    is used in other contexts such as REGCOIL. However for stage-2 optimization,
    the optimizer can "cheat", lowering this objective by reducing the magnitude
    of the field. The definitions ``"normalized"`` and ``"local"`` close this loophole.

    Args:
        surface: A :obj:`simsopt.geo.surface.Surface` object on which to compute the flux
        field: A :obj:`simsopt.field.magneticfield.MagneticField` for which to compute the flux.
        target: A ``nphi x ntheta`` numpy array containing target values for the flux. Here 
          ``nphi`` and ``ntheta`` correspond to the number of quadrature points on `surface` 
          in ``phi`` and ``theta`` direction.
        definition: A string to select among the definitions above. The
          available options are ``"quadratic flux"``, ``"normalized"``, and ``"local"``.
    """

    def __init__(self, boozer_surface, field, target=None, definition="quadratic flux"):
        self.boozer_surface = boozer_surface
        ins = boozer_surface.surface
        self.surface = SurfaceXYZTensorFourier(mpol=ins.mpol, ntor=ins.ntor, stellsym=ins.stellsym, nfp=ins.nfp,\
                quadpoints_phi=np.linspace(0, 1, ins.nfp*(2*ins.mpol+1)*2, endpoint=False), \
                quadpoints_theta=np.linspace(0, 1, (2*ins.ntor+1)*2, endpoint=False),\
                dofs=ins.dofs)
        if target is not None:
            self.target = np.ascontiguousarray(target)
        else:
            self.target = np.zeros(self.surface.normal().shape[:2])
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        return sopp.integral_BdotN(Bcoil, self.target, n, self.definition)

    @derivative_dec
    def dJ(self):
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))

        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil * unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n

        if self.definition == "quadratic flux":
            dJdB = (B_n[..., None] * unitn * absn[..., None]) / absn.size
            dJdB = dJdB.reshape((-1, 3))

        elif self.definition == "local":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            dJdB = ((
                (B_n/mod_Bcoil)[..., None] * (
                    unitn / mod_Bcoil[..., None] - (B_n / mod_Bcoil**3)[..., None] * Bcoil
                )) * absn[..., None]) / absn.size
            
            nphi, ntheta = self.surface.quadpoints_phi.size, self.surface.quadpoints_theta.size
            B = self.field.B().reshape((nphi, ntheta, 3))
            dB_by_dX = self.field.dB_by_dX().reshape((nphi, ntheta, 3, 3))
            dx_ds = self.surface.dgamma_by_dcoeff()
            dB_ds = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_ds, optimize=True)
            modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
            dmodB_ds = (B[:, :, 0, None] * dB_ds[:, :, 0, :] + B[:, :, 1, None] * dB_ds[:, :, 1, :] + B[:, :, 2, None] * dB_ds[:, :, 2, :])/modB[:, :, None]

            nor = self.surface.normal()
            dnor_ds = self.surface.dnormal_by_dcoeff()
            dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
            dS_ds = (nor[:, :, 0, None]*dnor_ds[:, :, 0, :] + nor[:, :, 1, None]*dnor_ds[:, :, 1, :] + nor[:, :, 2, None]*dnor_ds[:, :, 2, :])/dS[:, :, None]
            

            N = np.sum(B * nor, axis=2)**2
            dN_ds = 2*np.sum(B * nor, axis=2)[..., None]*np.sum(dB_ds * nor[..., None] + dnor_ds * B[..., None], axis=2)
            
            D = modB**2 * dS
            dD_ds = 2 * modB[...,  None] * dmodB_ds * dS[..., None] + dS_ds * modB[..., None]**2
            dj_ds = 0.5*np.mean((dN_ds * D[..., None] - dD_ds * N[..., None]) / D[..., None]**2, axis=(0, 1))
            
            booz_surf = self.boozer_surface
            iota = booz_surf.res['iota']
            G = booz_surf.res['G']
            P, L, U = booz_surf.res['PLU']
            dconstraint_dcoils_vjp = self.boozer_surface.res['vjp']

            # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
            dJ_ds = np.zeros(L.shape[0])
            dJ_ds[:dj_ds.size] = dj_ds
            adj = forward_backward(P, L, U, dJ_ds)

            adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        elif self.definition == "normalized":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            num = np.mean(B_n**2 * absn)
            denom = np.mean(mod_Bcoil**2 * absn)

            dnum = 2 * (B_n[..., None] * unitn * absn[..., None]) / absn.size
            ddenom = 2 * (Bcoil * absn[..., None]) / absn.size
            dJdB = 0.5 * (dnum / denom - num * ddenom / denom**2)
        else:
            raise ValueError("Should never get here")
        
        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB) - adj_times_dg_dcoil

class SquaredFlux(Optimizable):

    r"""
    Objective representing quadratic-flux-like quantities, useful for stage-2
    coil optimization. Several variations are available, which can be selected
    using the ``definition`` argument. For ``definition="quadratic flux"`` 
    (the default), the objective is defined as

    .. math::
        J = \frac12 \int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds,

    where :math:`\mathbf{n}` is the surface unit normal vector and
    :math:`B_T` is an optional (zero by default) target value for the
    magnetic field. Also :math:`\int_{S} ds` indicates a surface integral.
    For ``definition="normalized"``, the objective is defined as

    .. math::
        J = \frac12 \frac{\int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds}
                         {\int_{S} |\mathbf{B}|^2 ds}.

    For ``definition="local"``, the objective is defined as

    .. math::
        J = \frac12 \int_{S} \frac{(\mathbf{B}\cdot \mathbf{n} - B_T)^2}{|\mathbf{B}|^2} ds.

    The definition ``"quadratic flux"`` has the advantage of simplicity, and it
    is used in other contexts such as REGCOIL. However for stage-2 optimization,
    the optimizer can "cheat", lowering this objective by reducing the magnitude
    of the field. The definitions ``"normalized"`` and ``"local"`` close this loophole.

    Args:
        surface: A :obj:`simsopt.geo.surface.Surface` object on which to compute the flux
        field: A :obj:`simsopt.field.magneticfield.MagneticField` for which to compute the flux.
        target: A ``nphi x ntheta`` numpy array containing target values for the flux. Here 
          ``nphi`` and ``ntheta`` correspond to the number of quadrature points on `surface` 
          in ``phi`` and ``theta`` direction.
        definition: A string to select among the definitions above. The
          available options are ``"quadratic flux"``, ``"normalized"``, and ``"local"``.
    """

    def __init__(self, surface, field, target=None, definition="quadratic flux"):
        self.surface = surface
        if target is not None:
            self.target = np.ascontiguousarray(target)
        else:
            self.target = np.zeros(self.surface.normal().shape[:2])
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        return sopp.integral_BdotN(Bcoil, self.target, n, self.definition)

    @derivative_dec
    def dJ(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil * unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n

        if self.definition == "quadratic flux":
            dJdB = (B_n[..., None] * unitn * absn[..., None]) / absn.size
            dJdB = dJdB.reshape((-1, 3))

        elif self.definition == "local":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            dJdB = ((
                (B_n/mod_Bcoil)[..., None] * (
                    unitn / mod_Bcoil[..., None] - (B_n / mod_Bcoil**3)[..., None] * Bcoil
                )) * absn[..., None]) / absn.size

        elif self.definition == "normalized":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            num = np.mean(B_n**2 * absn)
            denom = np.mean(mod_Bcoil**2 * absn)

            dnum = 2 * (B_n[..., None] * unitn * absn[..., None]) / absn.size
            ddenom = 2 * (Bcoil * absn[..., None]) / absn.size
            dJdB = 0.5 * (dnum / denom - num * ddenom / denom**2)
            
        else:
            raise ValueError("Should never get here")

        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB)

