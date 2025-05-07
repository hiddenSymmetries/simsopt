import numpy as np

from .framedcurve import FramedCurve, FrameRotation, ZeroRotation, FramedCurveCentroid, FramedCurveFrenet

"""
The functions and classes in this model are used to deal with multifilament
approximation of finite build coils.
"""

__all__ = ['create_multifilament_grid', 'CurveFilament']


class CurveFilament(FramedCurve):

    def __init__(self, framedcurve, dn, db):
        """
        Given a FramedCurve, defining a normal and
        binormal vector, create a grid of curves by shifting 
        along the normal and binormal vector. 

        The idea is explained well in Figure 1 in the reference:

        Singh et al, "Optimization of finite-build stellarator coils",
        Journal of Plasma Physics 86 (2020),
        doi:10.1017/S0022377820000756. 

        Args:
            curve: the underlying curve
            dn: how far to move in normal direction
            db: how far to move in binormal direction
            rotation: angle along the curve to rotate the frame.
        """
        self.curve = framedcurve.curve
        self.dn = dn
        self.db = db
        self.rotation = framedcurve.rotation
        self.framedcurve = framedcurve
        FramedCurve.__init__(self, self.curve, self.rotation)

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        t, n, b = self.framedcurve.rotated_frame()
        gamma[:] = self.curve.gamma() + self.dn * n + self.db * b

    def gammadash_impl(self, gammadash):
        td, nd, bd = self.framedcurve.rotated_frame_dash()
        gammadash[:] = self.curve.gammadash() + self.dn * nd + self.db * bd

    def dgamma_by_dcoeff_vjp(self, v):
        vg = self.framedcurve.rotated_frame_dcoeff_vjp(v, self.dn, self.db, 0)
        vgd = self.framedcurve.rotated_frame_dcoeff_vjp(v, self.dn, self.db, 1)
        vgdd = self.framedcurve.rotated_frame_dcoeff_vjp(v, self.dn, self.db, 2)
        va = self.framedcurve.rotated_frame_dcoeff_vjp(v, self.dn, self.db, 3)
        out = self.curve.dgamma_by_dcoeff_vjp(v + vg) \
            + self.curve.dgammadash_by_dcoeff_vjp(vgd) \
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, va)
        if vgdd is not None:
            out += self.curve.dgammadashdash_by_dcoeff_vjp(vgdd)
        return out


    def dgammadash_by_dcoeff_vjp(self, v):
        vg = self.framedcurve.rotated_frame_dash_dcoeff_vjp(v, self.dn, self.db, 0)
        vgd = self.framedcurve.rotated_frame_dash_dcoeff_vjp(v, self.dn, self.db, 1)
        vgdd = self.framedcurve.rotated_frame_dash_dcoeff_vjp(v, self.dn, self.db, 2)
        vgddd = self.framedcurve.rotated_frame_dash_dcoeff_vjp(v, self.dn, self.db, 3)
        va = self.framedcurve.rotated_frame_dash_dcoeff_vjp(v, self.dn, self.db, 4)
        vad = self.framedcurve.rotated_frame_dash_dcoeff_vjp(v, self.dn, self.db, 5)
        out = self.curve.dgamma_by_dcoeff_vjp(vg) \
            + self.curve.dgammadash_by_dcoeff_vjp(v+vgd) \
            + self.curve.dgammadashdash_by_dcoeff_vjp(vgdd) \
            + self.rotation.dalpha_by_dcoeff_vjp(self.curve.quadpoints, va) \
            + self.rotation.dalphadash_by_dcoeff_vjp(self.curve.quadpoints, vad)
        if vgddd is not None:
            out += self.curve.dgammadashdashdash_by_dcoeff_vjp(vgddd)
        return out


def create_multifilament_grid(curve, numfilaments_n, numfilaments_b, gapsize_n, gapsize_b,
                              rotation_order=None, rotation_scaling=None, frame='centroid'):
    """
    Create a regular grid of ``numfilaments_n * numfilaments_b`` many
    filaments to approximate a finite-build coil.

    Note that "normal" and "binormal" in the function arguments here
    refer to either the Frenet frame or the "coil centroid
    frame" defined by Singh et al., before rotation.

    Args:
        curve: The underlying curve.
        numfilaments_n: number of filaments in normal direction.
        numfilaments_b: number of filaments in bi-normal direction.
        gapsize_n: gap between filaments in normal direction.
        gapsize_b: gap between filaments in bi-normal direction.
        rotation_order: Fourier order (maximum mode number) to use in the expression for the rotation
                        of the filament pack. ``None`` means that the rotation is not optimized.
        rotation_scaling: scaling for the rotation degrees of freedom. good
                           scaling improves the convergence of first order optimization
                           algorithms. If ``None``, then the default of ``1 / max(gapsize_n, gapsize_b)``
                           is used.
        frame: orthonormal frame to define normal and binormal before rotation (either 'centroid' or 'frenet')
    """
    assert frame in ['centroid', 'frenet']
    if numfilaments_n % 2 == 1:
        shifts_n = np.arange(numfilaments_n) - numfilaments_n//2
    else:
        shifts_n = np.arange(numfilaments_n) - numfilaments_n/2 + 0.5
    shifts_n = shifts_n * gapsize_n
    if numfilaments_b % 2 == 1:
        shifts_b = np.arange(numfilaments_b) - numfilaments_b//2
    else:
        shifts_b = np.arange(numfilaments_b) - numfilaments_b/2 + 0.5
    shifts_b = shifts_b * gapsize_b

    if rotation_scaling is None:
        rotation_scaling = 1/max(gapsize_n, gapsize_b)
    if rotation_order is None:
        rotation = ZeroRotation(curve.quadpoints)
    else:
        rotation = FrameRotation(curve.quadpoints, rotation_order, scale=rotation_scaling)
    if frame == 'frenet':
        framedcurve = FramedCurveFrenet(curve, rotation)
    else:
        framedcurve = FramedCurveCentroid(curve, rotation)

    filaments = []
    for i in range(numfilaments_n):
        for j in range(numfilaments_b):
            filaments.append(CurveFilament(framedcurve, shifts_n[i], shifts_b[j]))
    return filaments
