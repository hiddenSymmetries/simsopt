import os
import unittest

import numpy as np
from monty.tempfile import ScratchDir

from simsopt import load
from simsopt.field import CircularCoil, ToroidalField, VirtualCasingField, VmecPlasmaField, VmecVirtualCasingField
from simsopt.geo import SurfaceRZFourier
from simsopt.mhd import Vmec

try:
    import virtual_casing
    offsurf = hasattr(virtual_casing.VirtualCasing, "compute_internal_B_offsurf")
except ImportError:
    offsurf = False

from . import TEST_DIR

LBD_QH = os.path.join(TEST_DIR, "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc")


def rotating_ellipse(n):
    surf = SurfaceRZFourier(nfp=3, stellsym=True, mpol=1, ntor=1,
                            quadpoints_phi=SurfaceRZFourier.get_phi_quadpoints(n, "half period", 3),
                            quadpoints_theta=SurfaceRZFourier.get_theta_quadpoints(n))
    surf.set_rc(0, 0, 1.0)
    surf.set_rc(1, 0, 0.2)
    surf.set_zs(1, 0, 0.2)
    surf.set_rc(1, 1, 0.05)
    surf.set_zs(1, 1, 0.05)
    return surf


def inner_field():
    return CircularCoil(r0=1.0, center=[0, 0, 0], I=2e5, normal=[0, 0, 1])


def outer_field():
    return ToroidalField(R0=1.0, B0=1.0) + CircularCoil(r0=3.0, center=[0, 0, 0], I=-1e6, normal=[0, 0, 1])


def field_at(field, x):
    field.set_points(np.ascontiguousarray(x))
    return field.B().copy()


def boundary_points(surf, n=40, seed=1):
    """Random points on surf and the outward unit normals there."""
    rng = np.random.default_rng(seed)
    probe = surf.copy(quadpoints_phi=rng.uniform(0, 1, n), quadpoints_theta=rng.uniform(0, 1, n))
    x = np.diagonal(probe.gamma(), axis1=0, axis2=1).T
    nhat = np.diagonal(probe.unitnormal(), axis1=0, axis2=1).T
    return x, np.sign(surf.volume()) * nhat


@unittest.skipIf(not offsurf, "Need a virtual_casing version with off-surface evaluation")
class VirtualCasingFieldTests(unittest.TestCase):
    n = 48

    def make(self, currents_inside=True, digits=6):
        surf = rotating_ellipse(self.n)
        x = surf.gamma().reshape(-1, 3)
        B = (field_at(inner_field(), x) + field_at(outer_field(), x)).reshape(self.n, self.n, 3)
        return VirtualCasingField(surf, B, digits=digits, currents_inside=currents_inside)

    def test_internal_currents_outside(self):
        vc = self.make(currents_inside=True)
        x0, nhat = boundary_points(vc.surf)
        scale = np.max(np.linalg.norm(field_at(inner_field(), x0), axis=1))
        for t in (1e-4, 1e-2, 0.2):
            x = x0 + t * nhat
            err = np.max(np.linalg.norm(field_at(vc, x) - field_at(inner_field(), x), axis=1)) / scale
            self.assertLess(err, 1e-5, f"t = {t}")

    def test_external_currents_inside(self):
        vc = self.make(currents_inside=False)
        x0, nhat = boundary_points(vc.surf)
        scale = np.max(np.linalg.norm(field_at(outer_field(), x0), axis=1))
        for t in (1e-4, 1e-2, 0.1):
            x = x0 - t * nhat
            err = np.max(np.linalg.norm(field_at(vc, x) - field_at(outer_field(), x), axis=1)) / scale
            self.assertLess(err, 1e-5, f"t = {t}")

    def test_on_surface(self):
        vc = self.make()
        x = vc.trgt_surf.gamma().reshape(-1, 3)
        B_int = field_at(inner_field(), x).reshape(vc.onsurf_field_due_int.shape)
        B_ext = field_at(outer_field(), x).reshape(vc.onsurf_field_due_ext.shape)
        np.testing.assert_allclose(vc.onsurf_field_due_int, B_int, atol=1e-5 * np.abs(B_int).max())
        np.testing.assert_allclose(vc.onsurf_field_due_ext, B_ext, atol=1e-5 * np.abs(B_ext).max())
        n = vc.trgt_surf.unitnormal()
        np.testing.assert_allclose(vc.Bnormal_due_int + vc.Bnormal_due_ext, np.sum((B_int + B_ext) * n, axis=-1),
                                   atol=1e-5 * np.abs(B_ext).max())

    def test_update_and_flat_input(self):
        vc = self.make()
        x0, nhat = boundary_points(vc.surf)
        x = x0 + 0.05 * nhat
        B = field_at(vc, x)
        vc.total_B_in_surf = 2 * vc.total_B_in_surf
        np.testing.assert_allclose(field_at(vc, x), 2 * B, rtol=1e-12, atol=0)
        flat = VirtualCasingField(vc.surf, vc.total_B_in_surf.transpose(2, 0, 1).ravel(), digits=6)
        np.testing.assert_allclose(field_at(flat, x), 2 * B, rtol=1e-12, atol=0)
        with self.assertRaises(ValueError):
            vc.total_B_in_surf = np.zeros((self.n, self.n + 1, 3))

    def test_save_load(self):
        vc = self.make()
        x0, nhat = boundary_points(vc.surf)
        x = x0 + 0.05 * nhat
        with ScratchDir("."):
            vc.save("vc.json")
            vc2 = load("vc.json")
        self.assertEqual(vc2.currents_inside, vc.currents_inside)
        np.testing.assert_allclose(field_at(vc2, x), field_at(vc, x), rtol=1e-12, atol=0)


@unittest.skipIf(not offsurf, "Need a virtual_casing version with off-surface evaluation")
class VmecVirtualCasingFieldTests(unittest.TestCase):

    def test_matches_on_surface_virtual_casing(self):
        """Normal fields agree with simsopt.mhd.VirtualCasing on the same grids."""
        from simsopt.mhd import VirtualCasing

        vc = VmecVirtualCasingField(LBD_QH, src_nphi=32, src_ntheta=32, trgt_nphi=16, trgt_ntheta=16, digits=6)
        ref = VirtualCasing.from_vmec(LBD_QH, src_nphi=32, src_ntheta=32, trgt_nphi=16, trgt_ntheta=16, digits=6,
                                      filename=None)
        np.testing.assert_allclose(vc.Bnormal_due_ext, ref.B_external_normal, atol=1e-10)
        self.assertLess(np.max(np.abs(vc.Bnormal_due_ext + vc.Bnormal_due_int)), 1e-5)

    def test_continuous_with_on_surface_field(self):
        """Just outside the boundary, B tends to the on-surface field of the plasma currents."""
        vc = VmecVirtualCasingField(LBD_QH, src_nphi=96, src_ntheta=96, trgt_nphi=8, trgt_ntheta=8, digits=6)
        surf = vc.trgt_surf
        x0 = surf.gamma().reshape(-1, 3)
        n = np.sign(surf.volume()) * surf.unitnormal().reshape(-1, 3)
        B = field_at(vc, x0 + 1e-6 * n)
        B0 = vc.onsurf_field_due_int.reshape(-1, 3)
        self.assertLess(np.max(np.linalg.norm(B - B0, axis=1)) / np.max(np.linalg.norm(B0, axis=1)), 1e-3)

    def test_plasma_field_inside_and_outside(self):
        """VmecPlasmaField: virtual casing outside, B_VMEC - B_ext inside, continuous on the boundary."""
        pf = VmecPlasmaField(LBD_QH, src_nphi=64, src_ntheta=64, trgt_nphi=8, trgt_ntheta=8, digits=6)
        x0 = pf.trgt_surf.gamma().reshape(-1, 3)
        n = np.sign(pf.trgt_surf.volume()) * pf.trgt_surf.unitnormal().reshape(-1, 3)
        B = {t: field_at(pf, x0 + t * n) for t in (0.05, 1e-9, 0.0, -1e-9, -0.05)}
        scale = np.max(np.linalg.norm(B[1e-9], axis=1))

        outside = VmecVirtualCasingField(LBD_QH, src_nphi=64, src_ntheta=64, trgt_nphi=8, trgt_ntheta=8, digits=6)
        np.testing.assert_allclose(B[0.05], field_at(outside, x0 + 0.05 * n), rtol=0, atol=1e-12 * scale)
        ext = VirtualCasingField(pf.surf, pf.total_B_in_surf, trgt_nphi=8, trgt_ntheta=8, digits=6,
                                 currents_inside=False)
        xi = x0 - 0.05 * n
        np.testing.assert_allclose(B[-0.05] + field_at(ext, xi), field_at(pf.vmec_field, xi), rtol=0,
                                   atol=1e-12 * np.abs(field_at(pf.vmec_field, xi)).max())
        for a, b in ((1e-9, -1e-9), (0.0, 1e-9), (0.0, -1e-9)):
            self.assertLess(np.max(np.linalg.norm(B[a] - B[b], axis=1)) / scale, 1e-6)

        # near the magnetic axis
        rng = np.random.default_rng(0)
        npts = 2000
        xc = pf.vmec_field._geometry(rng.uniform(0.02, 0.2, npts), rng.uniform(0, 2 * np.pi, npts),
                                     rng.uniform(0, 2 * np.pi, npts))[0]
        Bv = field_at(pf.vmec_field, xc)
        np.testing.assert_allclose(field_at(pf, xc) + field_at(ext, xc), Bv, rtol=0, atol=1e-12 * np.abs(Bv).max())

    def test_default_source_grid(self):
        """The default grid resolves n x B on the boundary to 10^-(digits+2); explicit sizes take precedence."""
        vmec = Vmec(LBD_QH)
        for digits, grid in ((5, (29, 114)), (6, (34, 134)), (7, (39, 154))):
            self.assertEqual(VmecVirtualCasingField.default_source_grid(vmec, digits), grid)
        vc = VmecVirtualCasingField(vmec, digits=6)
        self.assertEqual((vc.src_nphi, vc.src_ntheta), (34, 134))
        vc = VmecVirtualCasingField(vmec, src_nphi=16, digits=6)
        self.assertEqual((vc.src_nphi, vc.src_ntheta), (16, 134))
        vc = VmecVirtualCasingField(vmec, src_nphi=16, src_ntheta=40)
        self.assertEqual((vc.src_nphi, vc.src_ntheta), (16, 40))

    def test_depends_on_vmec(self):
        vc = VmecVirtualCasingField(LBD_QH, src_nphi=16, src_ntheta=32)
        self.assertIn(vc.vmec, vc.parents)
        vc.Bnormal_due_ext
        self.assertFalse(vc._need_vmec)
        vc.recompute_bell()
        self.assertTrue(vc._need_vmec)


if __name__ == "__main__":
    unittest.main()
