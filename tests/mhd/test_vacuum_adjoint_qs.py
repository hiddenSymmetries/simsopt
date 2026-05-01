import logging
import os
import unittest

import numpy as np

try:
    import py_spec
    from py_spec import SPECout
    py_spec_available = True
except ImportError:
    py_spec_available = False

try:
    import spec as spec_mod
    spec_available = spec_mod is not None
except ImportError:
    spec_available = False

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from . import TEST_DIR

logger = logging.getLogger(__name__)

# Path to adjoint_QS example SPEC file for integration tests
ADJQS_EXAMPLE = os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    '..', 'adjoint_QS', 'examples',
    'quasisymmetry_fom_and_shape_gradient', 'run_SPEC_main', 'vacuum.sp')
ADJQS_EXAMPLE = os.path.abspath(ADJQS_EXAMPLE)
adjqs_example_available = os.path.isfile(ADJQS_EXAMPLE)


class TestVacuumAdjointQSUnit(unittest.TestCase):
    """Unit tests for pure-math helpers — no SPEC binary or py_spec required."""

    def test_import(self):
        """VacuumAdjointQS should be importable from simsopt.mhd."""
        from simsopt.mhd import VacuumAdjointQS
        self.assertTrue(callable(VacuumAdjointQS))

    def test_fourier_helpers(self):
        """_sine_modes and _cosine_modes return correct mode counts."""
        from simsopt.mhd.vacuum_adjoint_qs import _sine_modes, _cosine_modes
        for mpol, ntor in [(3, 3), (5, 5), (2, 4)]:
            nms, xm_s, xn_s = _sine_modes(mpol, ntor)
            nmc, xm_c, xn_c = _cosine_modes(mpol, ntor)
            self.assertEqual(nms, ntor + mpol * (2 * ntor + 1))
            self.assertEqual(nmc, (ntor + 1) + mpol * (2 * ntor + 1))

    def test_basis_and_derivs_sin(self):
        """Sine basis derivatives match finite-difference approximation."""
        from simsopt.mhd.vacuum_adjoint_qs import _sine_modes, _basis_and_derivs
        nfp = 3
        thetas = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        phis = np.linspace(0, 2 * np.pi / nfp, 32, endpoint=False)
        th, ph = np.meshgrid(thetas, phis, indexing='ij')
        mpol, ntor = 2, 2
        nm, xm, xn = _sine_modes(mpol, ntor)
        basis, db_dt, db_dp, d2b_dt2, d2b_dtp, d2b_dp2 = _basis_and_derivs(
            nm, xm, xn, 'sin', th, ph, nfp, second_deriv=True)

        eps = 1e-6
        th_p = th + eps
        basis_p, _, _ = _basis_and_derivs(nm, xm, xn, 'sin', th_p, ph, nfp)
        db_dt_fd = (basis_p - basis) / eps
        np.testing.assert_allclose(db_dt, db_dt_fd, atol=1e-4)

    def test_basis_and_derivs_cos(self):
        """Cosine basis derivatives match finite-difference approximation."""
        from simsopt.mhd.vacuum_adjoint_qs import _cosine_modes, _basis_and_derivs
        nfp = 2
        thetas = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        phis = np.linspace(0, 2 * np.pi / nfp, 48, endpoint=False)
        th, ph = np.meshgrid(thetas, phis, indexing='ij')
        mpol, ntor = 3, 3
        nm, xm, xn = _cosine_modes(mpol, ntor)
        basis, db_dt, db_dp = _basis_and_derivs(nm, xm, xn, 'cos', th, ph, nfp)

        eps = 1e-6
        th_p = th + eps
        basis_p, _, _ = _basis_and_derivs(nm, xm, xn, 'cos', th_p, ph, nfp)
        db_dt_fd = (basis_p - basis) / eps
        np.testing.assert_allclose(db_dt, db_dt_fd, atol=1e-4)

    def test_solve_stfl_sinusoidal(self):
        """SFL solver recovers known iota for a sinusoidally-perturbed field."""
        from simsopt.mhd.vacuum_adjoint_qs import _solve_stfl
        # B^phi = 2 + eps*cos(theta), B^theta = 1 → iota ≈ 0.5 to leading order
        nfp = 1
        ntheta, nphi = 64, 64
        thetas = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        th, ph = np.meshgrid(thetas, phis, indexing='ij')
        eps = 0.01
        Bsupphi = 2.0 + eps * np.cos(th)
        Bsuptheta = np.ones_like(th)
        iota, lam, *_ = _solve_stfl(Bsuptheta, Bsupphi, 4, 4, th, ph, nfp)
        # Leading-order iota = <B^theta>/<B^phi> = 1/2
        self.assertAlmostEqual(iota, 0.5, places=3)


@unittest.skipIf(not (spec_available and py_spec_available and adjqs_example_available),
                 "SPEC binary, py_spec, or adjoint_QS example not available")
class TestVacuumAdjointQSIntegration(unittest.TestCase):
    """
    Integration tests that require a SPEC binary, py_spec, and the example
    SPEC file from the adjoint_QS repository.
    """

    def _make_spec(self, filename):
        from simsopt.mhd import Spec
        from monty.tempfile import ScratchDir
        import shutil, tempfile
        tmpdir = tempfile.mkdtemp()
        dst = os.path.join(tmpdir, os.path.basename(filename))
        shutil.copy(filename, dst)
        return Spec(dst), tmpdir

    def test_J_smoke(self):
        """J() returns a positive scalar and vQS has correct shape."""
        import shutil
        from simsopt.mhd import Spec, VacuumAdjointQS
        from monty.tempfile import ScratchDir

        with ScratchDir("."):
            shutil.copy(ADJQS_EXAMPLE, '.')
            spfile = os.path.basename(ADJQS_EXAMPLE)
            spec = Spec(spfile)
            qs = VacuumAdjointQS(spec, helicity_m=1, helicity_n=0,
                                  mpol_adj=4, ntor_adj=4, ndiscrete=5)
            J = qs.J()
            self.assertIsInstance(J, float)
            self.assertGreater(J, 0.0)
            self.assertEqual(qs.vQS.shape, (qs.ntheta, qs.nphi))

    def test_f_iota_at_target(self):
        """f_iota ≈ 0 when iota_target equals the computed iota."""
        import shutil
        from simsopt.mhd import Spec, VacuumAdjointQS
        from monty.tempfile import ScratchDir

        with ScratchDir("."):
            shutil.copy(ADJQS_EXAMPLE, '.')
            spfile = os.path.basename(ADJQS_EXAMPLE)
            spec = Spec(spfile)
            qs = VacuumAdjointQS(spec, helicity_m=1, helicity_n=0,
                                  iota_target=None, mpol_adj=4, ntor_adj=4,
                                  ndiscrete=5)
            qs.J()
            iota = qs.iota
            qs2 = VacuumAdjointQS(spec, helicity_m=1, helicity_n=0,
                                   iota_target=iota, iota_weight=1.0,
                                   qs_weight=0.0,
                                   mpol_adj=4, ntor_adj=4, ndiscrete=5)
            self.assertAlmostEqual(qs2.J(), 0.0, places=10)

    def test_cache_invalidation(self):
        """Perturbing the boundary invalidates the cache and changes J."""
        import shutil
        from simsopt.mhd import Spec, VacuumAdjointQS
        from monty.tempfile import ScratchDir

        with ScratchDir("."):
            shutil.copy(ADJQS_EXAMPLE, '.')
            spfile = os.path.basename(ADJQS_EXAMPLE)
            spec = Spec(spfile)
            qs = VacuumAdjointQS(spec, helicity_m=1, helicity_n=0,
                                  mpol_adj=4, ntor_adj=4, ndiscrete=5)
            J1 = qs.J()
            # Perturb a DOF
            x = spec.boundary.x.copy()
            x[0] += 0.01
            spec.boundary.x = x
            J2 = qs.J()
            self.assertNotAlmostEqual(J1, J2)


if __name__ == '__main__':
    unittest.main()
