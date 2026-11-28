from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from .magneticfield import MagneticField

if TYPE_CHECKING:
    from ..mhd.vmec import Vmec

__all__ = ['VmecField']


class VmecField(MagneticField):
    r"""
    Magnetic field of a VMEC equilibrium at Cartesian points inside its boundary, and NaN outside.

    The VMEC coordinates of each point are found by Newton's method, started from the closest
    point of a coarse grid. The Fourier coefficients of :math:`R, Z, B^\theta, B^\phi` are
    interpolated by cubic splines in :math:`\rho = \sqrt{s}`. At :math:`s = 1` the field has the
    linear extrapolation of :func:`~simsopt.mhd.vmec_diagnostics.B_cartesian`, so it agrees with
    the boundary field used by :class:`~simsopt.field.VmecVirtualCasingField`. Only
    stellarator-symmetric equilibria are supported.

    Args:
        vmec: a :obj:`~simsopt.mhd.vmec.Vmec` object, or the name of a VMEC input or ``wout`` file.
        tol: points farther than ``tol`` (in metres) outside the boundary are outside.
    """

    def __init__(self, vmec: 'Vmec', tol: float = 1e-8):
        from ..mhd.vmec import Vmec

        if not isinstance(vmec, Vmec):
            vmec = Vmec(vmec)
        self._vmec = vmec
        self.tol = float(tol)
        self._need_build = True
        MagneticField.__init__(self, depends_on=[vmec])

    @property
    def vmec(self):
        """The :obj:`~simsopt.mhd.vmec.Vmec` object."""
        return self._vmec

    def recompute_bell(self, parent=None):
        self._need_build = True
        self.clear_cached_properties()

    def _build(self):
        if not self._need_build:
            return
        self._vmec.run()
        w = self._vmec.wout
        if w.lasym:
            raise RuntimeError("VmecField supports only stellarator-symmetric equilibria")
        ns = int(w.ns)
        self.nfp = int(w.nfp)
        self._signgs = float(w.signgs)
        self._xm, self._xn = np.asarray(w.xm, dtype=float), np.asarray(w.xn, dtype=float)
        self._xm_nyq, self._xn_nyq = np.asarray(w.xm_nyq, dtype=float), np.asarray(w.xn_nyq, dtype=float)
        rho = np.sqrt(np.linspace(0, 1, ns))
        self._R = CubicSpline(rho, np.asarray(w.rmnc).T, axis=0)
        self._Z = CubicSpline(rho, np.asarray(w.zmns).T, axis=0)
        self._dR, self._dZ = self._R.derivative(), self._Z.derivative()
        knots = np.append(np.sqrt((np.arange(1, ns) - 0.5) / (ns - 1)), 1.0)
        for name in ("bsupumnc", "bsupvmnc"):
            f = np.asarray(getattr(w, name))
            values = np.vstack([f[:, 1:].T, 1.5 * f[:, -1] - 0.5 * f[:, -2]])
            setattr(self, "_" + name, CubicSpline(knots, values, axis=0))
        r, t, p = np.meshgrid(np.linspace(0.05, 1, 16), np.linspace(0, 2 * np.pi, 32, endpoint=False),
                              np.linspace(0, 2 * np.pi, 32 * self.nfp, endpoint=False), indexing="ij")
        self._guesses = np.stack([r.ravel(), t.ravel(), p.ravel()], axis=-1)
        self._tree = cKDTree(self._geometry(*self._guesses.T)[0])
        self._need_build = False

    def _geometry(self, rho, theta, phi):
        """Position and the basis vectors d/drho, d/dtheta, d/dphi, each of shape (n, 3)."""
        ang = self._xm[None] * theta[:, None] - self._xn[None] * phi[:, None]
        c, s = np.cos(ang), np.sin(ang)
        Rm, Zm, dRm, dZm = self._R(rho), self._Z(rho), self._dR(rho), self._dZ(rho)
        R, Z = np.sum(Rm * c, axis=1), np.sum(Zm * s, axis=1)
        Rr, Zr = np.sum(dRm * c, axis=1), np.sum(dZm * s, axis=1)
        Rt, Zt = -np.sum(self._xm * Rm * s, axis=1), np.sum(self._xm * Zm * c, axis=1)
        Rp, Zp = np.sum(self._xn * Rm * s, axis=1), -np.sum(self._xn * Zm * c, axis=1)
        cp, sp = np.cos(phi), np.sin(phi)
        x = np.stack([R * cp, R * sp, Z], axis=-1)
        e_r = np.stack([Rr * cp, Rr * sp, Zr], axis=-1)
        e_t = np.stack([Rt * cp, Rt * sp, Zt], axis=-1)
        e_p = np.stack([Rp * cp - R * sp, Rp * sp + R * cp, Zp], axis=-1)
        return x, e_r, e_t, e_p

    def _field(self, rho, theta, phi):
        _, _, e_t, e_p = self._geometry(rho, theta, phi)
        c = np.cos(self._xm_nyq[None] * theta[:, None] - self._xn_nyq[None] * phi[:, None])
        Bt = np.sum(self._bsupumnc(rho) * c, axis=1)
        Bp = np.sum(self._bsupvmnc(rho) * c, axis=1)
        return Bt[:, None] * e_t + Bp[:, None] * e_p

    def _outward_normal(self, theta, phi):
        """Boundary point and outward unit normal at (theta, phi)."""
        x, _, e_t, e_p = self._geometry(np.ones_like(theta), theta, phi)
        n = self._signgs * np.cross(e_t, e_p)
        return x, n / np.linalg.norm(n, axis=1)[:, None]

    def vmec_coordinates(self, points, maxiter=40):
        r"""
        VMEC coordinates of Cartesian points.

        Args:
            points: array of shape ``(n, 3)``.
            maxiter: Newton iterations.

        Returns:
            ``(s, theta, phi, inside)``. For points outside the boundary, ``s = 1`` and
            ``theta, phi`` belong to a nearby boundary point.
        """
        self._build()
        x = np.atleast_2d(np.asarray(points, dtype=float))
        rho, theta, phi = self._guesses[self._tree.query(x)[1]].T.copy()
        for _ in range(maxiter):
            r, e_r, e_t, e_p = self._geometry(rho, theta, phi)
            res = x - r
            if np.max(np.abs(res)) < 1e-3 * self.tol:
                break
            d = np.linalg.solve(np.stack([e_r, e_t, e_p], axis=-1), res[..., None])[..., 0]
            rho = np.clip(rho + np.clip(d[:, 0], -0.2, 0.2), 1e-8, 1.0)
            theta = theta + np.clip(d[:, 1], -0.5, 0.5)
            phi = phi + np.clip(d[:, 2], -0.5, 0.5)
        inside = np.linalg.norm(x - self._geometry(rho, theta, phi)[0], axis=1) < self.tol
        return rho ** 2, np.mod(theta, 2 * np.pi), np.mod(phi, 2 * np.pi), inside

    def _B_impl(self, B):
        s, theta, phi, inside = self.vmec_coordinates(self.get_points_cart_ref())
        B[:] = np.nan
        if inside.any():
            B[inside] = self._field(np.sqrt(s[inside]), theta[inside], phi[inside])
