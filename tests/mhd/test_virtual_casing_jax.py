import os
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

import simsopt.mhd.virtual_casing_jax as virtual_casing_jax_module

try:
    import virtual_casing_jax
except ImportError:
    virtual_casing_jax = None

try:
    from virtual_casing_jax.functional import (
        compute_external_B_normal_functional,
        compute_external_B_normal_jvp_columns_functional,
    )
except ImportError:
    compute_external_B_normal_functional = None
    compute_external_B_normal_jvp_columns_functional = None

from simsopt.mhd import (
    B_external_normal_from_data,
    B_external_normal_jacobian_from_surface,
    B_external_normal_jvp_from_data,
    VirtualCasingJax,
    Vmec,
    VmecJax,
    clear_virtual_casing_jax_cache,
    local_squared_flux_surface_gradient,
)
from simsopt.geo import SurfaceRZFourier

from . import TEST_DIR


VARIABLES = [
    "src_nphi",
    "src_ntheta",
    "src_phi",
    "src_theta",
    "trgt_nphi",
    "trgt_ntheta",
    "trgt_phi",
    "trgt_theta",
    "gamma",
    "unit_normal",
    "B_total",
    "B_external",
    "B_external_normal",
]


def torus_data(nphi=5, ntheta=4):
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    theta2d, phi2d = np.meshgrid(theta, phi)
    gamma = np.zeros((nphi, ntheta, 3))
    gamma[:, :, 0] = (2.0 + 0.3 * np.cos(theta2d)) * np.cos(phi2d)
    gamma[:, :, 1] = (2.0 + 0.3 * np.cos(theta2d)) * np.sin(phi2d)
    gamma[:, :, 2] = 0.3 * np.sin(theta2d)
    B_total = 0.02 * gamma + 0.05
    unit_normal = np.zeros_like(gamma)
    unit_normal[:, :, 0] = np.cos(theta2d) * np.cos(phi2d)
    unit_normal[:, :, 1] = np.cos(theta2d) * np.sin(phi2d)
    unit_normal[:, :, 2] = np.sin(theta2d)
    return gamma, B_total, unit_normal


def make_axisymmetric_surface(nphi=5, ntheta=4):
    surf = SurfaceRZFourier.from_nphi_ntheta(
        mpol=1,
        ntor=0,
        nfp=1,
        nphi=nphi,
        ntheta=ntheta,
        range="field period",
    )
    surf.set_rc(0, 0, 2.0)
    surf.set_rc(1, 0, 0.3)
    surf.set_zs(1, 0, 0.3)
    surf.fix("rc(0,0)")
    return surf


class LinearField:
    def __init__(self):
        self.matrix = np.asarray(
            [
                [0.07, -0.02, 0.03],
                [0.01, 0.05, -0.04],
                [-0.02, 0.03, 0.06],
            ]
        )
        self.offset = np.asarray([0.4, -0.2, 0.3])
        self.points = None

    def set_points(self, points):
        self.points = np.asarray(points)

    def B(self):
        return self.points @ self.matrix.T + self.offset

    def dB_by_dX(self):
        return np.broadcast_to(
            self.matrix,
            (self.points.shape[0], 3, 3),
        ).copy()


@unittest.skipIf(virtual_casing_jax is None, "virtual_casing_jax not found")
class VirtualCasingJaxTests(unittest.TestCase):
    def test_vmec_and_vmec_jax_inputs_agree(self):
        filename = os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
        vc = VirtualCasingJax.from_vmec(
            Vmec(filename),
            src_nphi=6,
            src_ntheta=7,
            trgt_nphi=6,
            trgt_ntheta=7,
            digits=3,
            filename=None,
        )
        vc_j = VirtualCasingJax.from_vmec(
            VmecJax(filename),
            src_nphi=6,
            src_ntheta=7,
            trgt_nphi=6,
            trgt_ntheta=7,
            digits=3,
            filename=None,
        )

        for variable in VARIABLES:
            np.testing.assert_allclose(getattr(vc_j, variable), getattr(vc, variable))

    def test_vacuum_has_small_normal_field(self):
        filename = os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
        vc = VirtualCasingJax.from_vmec(
            VmecJax(filename),
            src_nphi=8,
            src_ntheta=9,
            trgt_nphi=8,
            trgt_ntheta=9,
            digits=4,
            filename=None,
        )
        np.testing.assert_allclose(vc.B_external_normal, 0, atol=0.005)

    def test_save_load(self):
        filename = os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
        with TemporaryDirectory() as tmp:
            outfile = os.path.join(tmp, "vcasing.nc")
            vc1 = VirtualCasingJax.from_vmec(
                VmecJax(filename),
                src_nphi=6,
                src_ntheta=7,
                trgt_nphi=5,
                trgt_ntheta=6,
                digits=3,
                filename=outfile,
            )
            vc2 = VirtualCasingJax.load(outfile)

        for variable in VARIABLES:
            np.testing.assert_allclose(getattr(vc1, variable), getattr(vc2, variable))

    @unittest.skipIf(
        compute_external_B_normal_functional is None,
        "virtual_casing_jax functional normal-field API not found",
    )
    def test_normal_field_from_data_matches_class(self):
        filename = os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
        vc = VirtualCasingJax.from_vmec(
            VmecJax(filename),
            src_nphi=6,
            src_ntheta=7,
            trgt_nphi=6,
            trgt_ntheta=7,
            digits=3,
            filename=None,
        )
        Bnormal = B_external_normal_from_data(
            vc.gamma,
            vc.B_total,
            vc.nfp,
            True,
            digits=3,
            trgt_nphi=vc.trgt_nphi,
            trgt_ntheta=vc.trgt_ntheta,
            unit_normal=vc.unit_normal,
        )

        np.testing.assert_allclose(Bnormal, vc.B_external_normal, rtol=1e-12, atol=1e-12)

    @unittest.skipIf(
        compute_external_B_normal_functional is None,
        "virtual_casing_jax functional normal-field API not found",
    )
    def test_normal_field_from_data_without_explicit_normal(self):
        gamma, B_total, _ = torus_data()
        tangent_gamma = np.zeros_like(gamma)
        tangent_gamma[0, 0, 0] = 1.0

        Bnormal, dBnormal = B_external_normal_jvp_from_data(
            gamma,
            B_total,
            tangent_gamma,
            nfp=1,
            stellsym=False,
            digits=4,
        )
        eps = 1e-5
        Bnormal_plus = B_external_normal_from_data(
            gamma + eps * tangent_gamma,
            B_total,
            1,
            False,
            digits=4,
        )
        Bnormal_minus = B_external_normal_from_data(
            gamma - eps * tangent_gamma,
            B_total,
            1,
            False,
            digits=4,
        )
        fd = (np.sum(Bnormal_plus**2) - np.sum(Bnormal_minus**2)) / (2 * eps)
        jvp = np.sum(2 * Bnormal * dBnormal)

        self.assertEqual(Bnormal.shape, gamma.shape[:2])
        np.testing.assert_allclose(jvp, fd, rtol=5e-3, atol=1e-6)

    @unittest.skipIf(
        compute_external_B_normal_functional is None,
        "virtual_casing_jax functional normal-field API not found",
    )
    def test_normal_field_from_data_validation(self):
        gamma, B_total, unit_normal = torus_data()

        with self.assertRaisesRegex(ValueError, "gamma must have shape"):
            B_external_normal_from_data(
                gamma[:, :, 0],
                B_total,
                1,
                False,
                digits=4,
            )
        with self.assertRaisesRegex(ValueError, "B_total must have shape"):
            B_external_normal_from_data(
                gamma,
                B_total[:-1],
                1,
                False,
                digits=4,
            )
        with self.assertRaisesRegex(ValueError, "unit_normal must have shape"):
            B_external_normal_from_data(
                gamma,
                B_total,
                1,
                False,
                digits=4,
                unit_normal=unit_normal[:-1],
            )
        with self.assertRaisesRegex(ValueError, "tangent_unit_normal"):
            B_external_normal_jvp_from_data(
                gamma,
                B_total,
                np.zeros_like(gamma),
                nfp=1,
                stellsym=False,
                digits=4,
                tangent_unit_normal=unit_normal,
            )

    @unittest.skipIf(
        compute_external_B_normal_functional is None,
        "virtual_casing_jax functional normal-field API not found",
    )
    def test_normal_field_jvp_with_fixed_target_normal(self):
        gamma, B_total, unit_normal = torus_data()
        tangent_gamma = np.zeros_like(gamma)
        tangent_gamma[0, 0, 0] = 1.0
        tangent_B_total = np.zeros_like(B_total)
        tangent_B_total[1, 2, 1] = -0.3

        Bnormal, dBnormal = B_external_normal_jvp_from_data(
            gamma,
            B_total,
            tangent_gamma,
            tangent_B_total,
            nfp=1,
            stellsym=False,
            digits=4,
            unit_normal=unit_normal,
        )
        eps = 1e-5
        Bnormal_plus = B_external_normal_from_data(
            gamma + eps * tangent_gamma,
            B_total + eps * tangent_B_total,
            1,
            False,
            digits=4,
            unit_normal=unit_normal,
        )
        Bnormal_minus = B_external_normal_from_data(
            gamma - eps * tangent_gamma,
            B_total - eps * tangent_B_total,
            1,
            False,
            digits=4,
            unit_normal=unit_normal,
        )
        fd = (np.sum(Bnormal_plus**2) - np.sum(Bnormal_minus**2)) / (2 * eps)
        jvp = np.sum(2 * Bnormal * dBnormal)

        np.testing.assert_allclose(jvp, fd, rtol=5e-3, atol=1e-6)

    @unittest.skipIf(
        compute_external_B_normal_functional is None,
        "virtual_casing_jax functional normal-field API not found",
    )
    def test_normal_field_jvp_from_data(self):
        gamma, B_total, unit_normal = torus_data()

        tangent_gamma = np.zeros_like(gamma)
        tangent_gamma[0, 0, 0] = 1.0
        tangent_B_total = np.zeros_like(B_total)
        tangent_B_total[1, 2, 1] = -0.3
        tangent_unit_normal = np.zeros_like(unit_normal)
        tangent_unit_normal[2, 1, 2] = 0.2

        Bnormal, dBnormal = B_external_normal_jvp_from_data(
            gamma,
            B_total,
            tangent_gamma,
            tangent_B_total,
            nfp=1,
            stellsym=False,
            digits=4,
            unit_normal=unit_normal,
            tangent_unit_normal=tangent_unit_normal,
        )
        eps = 1e-5
        Bnormal_plus = B_external_normal_from_data(
            gamma + eps * tangent_gamma,
            B_total + eps * tangent_B_total,
            1,
            False,
            digits=4,
            unit_normal=unit_normal + eps * tangent_unit_normal,
        )
        Bnormal_minus = B_external_normal_from_data(
            gamma - eps * tangent_gamma,
            B_total - eps * tangent_B_total,
            1,
            False,
            digits=4,
            unit_normal=unit_normal - eps * tangent_unit_normal,
        )
        fd = (np.sum(Bnormal_plus**2) - np.sum(Bnormal_minus**2)) / (2 * eps)
        jvp = np.sum(2 * Bnormal * dBnormal)

        np.testing.assert_allclose(jvp, fd, rtol=5e-3, atol=1e-6)

    @unittest.skipIf(
        compute_external_B_normal_jvp_columns_functional is None,
        "virtual_casing_jax functional normal-field API not found",
    )
    def test_normal_field_jacobian_from_surface(self):
        surf = make_axisymmetric_surface()
        gamma = surf.gamma()
        B_total = 0.02 * gamma + 0.05
        B_total_tangents = np.zeros(gamma.shape + (len(surf.x),))
        B_total_tangents[:, :, :, 0] = 0.01 * gamma
        B_total_tangents[:, :, :, 1] = -0.02

        Bnormal, jacobian = B_external_normal_jacobian_from_surface(
            surf,
            B_total,
            nfp=1,
            stellsym=False,
            digits=4,
            B_total_tangents=B_total_tangents,
        )
        direction = np.asarray([0.2, -0.1])
        dBnormal = np.tensordot(jacobian, direction, axes=([2], [0]))
        dB_total = np.tensordot(B_total_tangents, direction, axes=([3], [0]))

        x0 = np.copy(surf.x)
        eps = 1e-5
        surf.x = x0 + eps * direction
        Bnormal_plus = B_external_normal_from_data(
            surf.gamma(),
            B_total + eps * dB_total,
            1,
            False,
            digits=4,
            unit_normal=surf.unitnormal(),
        )
        surf.x = x0 - eps * direction
        Bnormal_minus = B_external_normal_from_data(
            surf.gamma(),
            B_total - eps * dB_total,
            1,
            False,
            digits=4,
            unit_normal=surf.unitnormal(),
        )
        surf.x = x0

        fd = (Bnormal_plus - Bnormal_minus) / (2 * eps)
        Bnormal_reference = B_external_normal_from_data(
            surf.gamma(),
            B_total,
            1,
            False,
            digits=4,
            unit_normal=surf.unitnormal(),
        )
        np.testing.assert_allclose(Bnormal, Bnormal_reference)
        np.testing.assert_allclose(dBnormal, fd, rtol=5e-3, atol=1e-6)

    @unittest.skipIf(
        compute_external_B_normal_jvp_columns_functional is None,
        "virtual_casing_jax functional normal-field API not found",
    )
    def test_normal_field_jacobian_defaults_and_validation(self):
        surf = make_axisymmetric_surface()
        gamma = surf.gamma()
        B_total = 0.02 * gamma + 0.05

        Bnormal, jacobian = B_external_normal_jacobian_from_surface(
            surf,
            B_total,
            digits=4,
        )
        self.assertEqual(Bnormal.shape, gamma.shape[:2])
        self.assertEqual(jacobian.shape, gamma.shape[:2] + (np.count_nonzero(surf.dofs_free_status),))

        _, all_jacobian = B_external_normal_jacobian_from_surface(
            surf,
            B_total,
            digits=4,
            free_only=False,
            B_total_tangents=np.zeros(gamma.shape + (surf.dgamma_by_dcoeff().shape[-1],)),
        )
        self.assertEqual(all_jacobian.shape, gamma.shape[:2] + (surf.dgamma_by_dcoeff().shape[-1],))

        with self.assertRaisesRegex(ValueError, "B_total_tangents must have shape"):
            B_external_normal_jacobian_from_surface(
                surf,
                B_total,
                digits=4,
                B_total_tangents=np.zeros(gamma.shape + (1,)),
            )

    @unittest.skipIf(
        compute_external_B_normal_jvp_columns_functional is None,
        "virtual_casing_jax functional normal-field API not found",
    )
    def test_local_squared_flux_surface_gradient_with_target_jacobian(self):
        surf = make_axisymmetric_surface()
        gamma = surf.gamma()
        B_total = 0.02 * gamma + 0.05
        B_total_tangents = np.zeros(gamma.shape + (len(surf.x),))
        B_total_tangents[:, :, :, 0] = 0.01 * gamma
        B_total_tangents[:, :, :, 1] = -0.02

        target, target_jacobian = B_external_normal_jacobian_from_surface(
            surf,
            B_total,
            nfp=1,
            stellsym=False,
            digits=4,
            B_total_tangents=B_total_tangents,
        )
        field = LinearField()
        grad = local_squared_flux_surface_gradient(
            surf,
            field,
            target,
            target_jacobian,
        )

        def objective(target_arg):
            n = surf.normal()
            absn = np.linalg.norm(n, axis=2)
            unitn = n * (1.0 / absn)[:, :, None]
            field.set_points(surf.gamma().reshape((-1, 3)))
            Bcoil = field.B().reshape(n.shape)
            B_n = np.sum(Bcoil * unitn, axis=2) - target_arg
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            return 0.5 * np.mean(B_n**2 * absn / mod_Bcoil**2)

        direction = np.asarray([0.2, -0.1])
        dB_total = np.tensordot(B_total_tangents, direction, axes=([3], [0]))
        x0 = np.copy(surf.x)
        eps = 1e-5
        surf.x = x0 + eps * direction
        target_plus = B_external_normal_from_data(
            surf.gamma(),
            B_total + eps * dB_total,
            1,
            False,
            digits=4,
            unit_normal=surf.unitnormal(),
        )
        J_plus = objective(target_plus)
        surf.x = x0 - eps * direction
        target_minus = B_external_normal_from_data(
            surf.gamma(),
            B_total - eps * dB_total,
            1,
            False,
            digits=4,
            unit_normal=surf.unitnormal(),
        )
        J_minus = objective(target_minus)
        surf.x = x0

        finite_difference = (J_plus - J_minus) / (2 * eps)
        np.testing.assert_allclose(
            np.dot(grad, direction),
            finite_difference,
            rtol=5e-3,
            atol=1e-8,
        )

    @unittest.skipIf(
        compute_external_B_normal_jvp_columns_functional is None,
        "virtual_casing_jax functional normal-field API not found",
    )
    def test_local_squared_flux_surface_gradient_fixed_target(self):
        surf = make_axisymmetric_surface()
        gamma = surf.gamma()
        target = B_external_normal_from_data(
            gamma,
            0.02 * gamma + 0.05,
            1,
            False,
            digits=4,
            unit_normal=surf.unitnormal(),
        )
        field = LinearField()
        grad = local_squared_flux_surface_gradient(surf, field, target)

        def objective():
            n = surf.normal()
            absn = np.linalg.norm(n, axis=2)
            unitn = n * (1.0 / absn)[:, :, None]
            field.set_points(surf.gamma().reshape((-1, 3)))
            Bcoil = field.B().reshape(n.shape)
            B_n = np.sum(Bcoil * unitn, axis=2) - target
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            return 0.5 * np.mean(B_n**2 * absn / mod_Bcoil**2)

        direction = np.asarray([0.2, -0.1])
        x0 = np.copy(surf.x)
        eps = 1e-5
        surf.x = x0 + eps * direction
        J_plus = objective()
        surf.x = x0 - eps * direction
        J_minus = objective()
        surf.x = x0

        finite_difference = (J_plus - J_minus) / (2 * eps)
        np.testing.assert_allclose(
            np.dot(grad, direction),
            finite_difference,
            rtol=5e-3,
            atol=1e-8,
        )

        with self.assertRaisesRegex(ValueError, "target must have shape"):
            local_squared_flux_surface_gradient(surf, field, target[:-1])
        with self.assertRaisesRegex(ValueError, "target_jacobian must have shape"):
            local_squared_flux_surface_gradient(
                surf,
                field,
                target,
                np.zeros(target.shape + (1,)),
            )

    def test_jvp_column_cache_is_bounded_and_clearable(self):
        class DummyJax:
            def jit(self, fn):
                return fn

        options = {
            "chunk_size": "auto",
            "target_chunk_size": "auto",
            "pou_dtype": None,
            "patch_dtype": None,
            "interp_block_size": "auto",
            "remat": None,
        }
        kwargs = {
            "digits": 4,
            "nfp": 1,
            "half_period": False,
            "surf_nt": 5,
            "surf_np": 4,
            "src_nt": 5,
            "src_np": 4,
            "trg_nt": 5,
            "trg_np": 4,
            "quad_nt": 5,
            "quad_np": 4,
            "patch_dim0": 1,
            "orient": 1.0,
        }
        old_cache_size = os.environ.get("SIMSOPT_VIRTUAL_CASING_JAX_CACHE_SIZE")
        try:
            os.environ["SIMSOPT_VIRTUAL_CASING_JAX_CACHE_SIZE"] = "2"
            clear_virtual_casing_jax_cache()
            for digits in range(4, 7):
                call_kwargs = dict(kwargs)
                call_kwargs["digits"] = digits
                virtual_casing_jax_module._cached_external_B_columns(
                    SimpleNamespace(),
                    DummyJax(),
                    call_kwargs,
                    options,
                    ((3, 5, 4),),
                    "unused_function",
                )
            self.assertLessEqual(
                len(virtual_casing_jax_module._B_EXTERNAL_JVP_COLUMNS_CACHE),
                2,
            )

            clear_virtual_casing_jax_cache()
            self.assertEqual(
                len(virtual_casing_jax_module._B_EXTERNAL_JVP_COLUMNS_CACHE),
                0,
            )
        finally:
            if old_cache_size is None:
                os.environ.pop("SIMSOPT_VIRTUAL_CASING_JAX_CACHE_SIZE", None)
            else:
                os.environ["SIMSOPT_VIRTUAL_CASING_JAX_CACHE_SIZE"] = old_cache_size
            clear_virtual_casing_jax_cache()

    def test_from_vmec_defaults_auto_filename_and_error_paths(self):
        filename = os.path.join(TEST_DIR, "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
        with TemporaryDirectory() as tmp:
            tmp_wout = os.path.join(tmp, "wout_auto.nc")
            shutil.copy2(filename, tmp_wout)
            vc = VirtualCasingJax.from_vmec(
                tmp_wout,
                src_nphi=6,
                digits=3,
                filename="auto",
            )

            self.assertTrue(os.path.exists(os.path.join(tmp, "vcasing_auto.nc")))
            self.assertEqual(vc.trgt_nphi, vc.src_nphi)
            self.assertEqual(vc.trgt_ntheta, vc.src_ntheta)
            self.assertEqual(vc.B_external_normal.shape, (vc.trgt_nphi, vc.trgt_ntheta))

        old_vc_mod = virtual_casing_jax_module.virtual_casing_jax_mod
        try:
            virtual_casing_jax_module.virtual_casing_jax_mod = None
            with self.assertRaisesRegex(RuntimeError, "requires the virtual_casing_jax package"):
                VirtualCasingJax.from_vmec("dummy", src_nphi=2)
        finally:
            virtual_casing_jax_module.virtual_casing_jax_mod = old_vc_mod

        class AsymmetricVmec:
            def __init__(self):
                self.wout = SimpleNamespace(lasym=True, nfp=1)

            def run(self):
                pass

        old_vmec = virtual_casing_jax_module.Vmec
        old_vmec_jax = virtual_casing_jax_module.VmecJax
        try:
            virtual_casing_jax_module.Vmec = AsymmetricVmec
            virtual_casing_jax_module.VmecJax = AsymmetricVmec
            with self.assertRaisesRegex(RuntimeError, "stellarator symmetry"):
                VirtualCasingJax.from_vmec(AsymmetricVmec(), src_nphi=2)
        finally:
            virtual_casing_jax_module.Vmec = old_vmec
            virtual_casing_jax_module.VmecJax = old_vmec_jax

    def test_functional_api_guard_errors(self):
        gamma, B_total, _ = torus_data()
        old_functional = virtual_casing_jax_module.virtual_casing_jax_functional
        try:
            virtual_casing_jax_module.virtual_casing_jax_functional = None
            with self.assertRaisesRegex(RuntimeError, "functional normal-field API"):
                B_external_normal_from_data(gamma, B_total, 1, False)

            virtual_casing_jax_module.virtual_casing_jax_functional = SimpleNamespace(
                prepare_functional_setup=object()
            )
            with self.assertRaisesRegex(RuntimeError, "functional normal-field API"):
                B_external_normal_from_data(gamma, B_total, 1, False)
        finally:
            virtual_casing_jax_module.virtual_casing_jax_functional = old_functional


if __name__ == "__main__":
    unittest.main()
