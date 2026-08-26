"""
Regressions of the VMEC++ backend against vmecpp's own simsopt shim,
``vmecpp.simsopt_compat``.
"""

import glob
import os
import subprocess
import sys
import textwrap
import unittest

import numpy as np
import vmecpp
from simsopt._core.util import ObjectiveFailure
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmecpp_solver import VmecppSolver

from . import TEST_DIR


def multigrid(vmec):
    """ Give ``vmec`` a two-step ``ns_array``, as a hot restart needs. """
    vmec.indata.ns_array = np.array([5, 17])
    vmec.indata.ftol_array = np.array([1.0e-12, 1.0e-20])
    vmec.indata.niter_array = np.array([1000, 3000])


@unittest.skipIf(vmecpp is None, "vmecpp is not installed")
class VmecppSolverRegressionTests(unittest.TestCase):
    def tearDown(self):
        for name in glob.glob("wout_*_000_??????.nc"):
            os.remove(name)

    def vmec(self, name="input.li383_low_res"):
        return Vmec(os.path.join(TEST_DIR, name), solver=VmecppSolver, verbose=False)

    # ---------------------------------------------------------------
    # Fix 1: raising or lowering indata.mpol/ntor by plain assignment
    # ---------------------------------------------------------------

    def test_raising_mpol_reallocates_indata_and_runs(self):
        """ ``vmec.indata.mpol = 3 + step`` is the documented simsopt idiom. """
        v = self.vmec()
        rbc = np.array(v.indata.rbc)
        ntor = v.indata.ntor
        v.indata.mpol = 5
        v.run()
        self.assertEqual(v.indata.rbc.shape, (5, 2 * ntor + 1))
        # The coefficients the file supplied are still there, and the new
        # row is zero:
        np.testing.assert_allclose(v.indata.rbc[:4, :], rbc)
        np.testing.assert_allclose(v.indata.rbc[4, :], 0.0)
        self.assertTrue(np.isfinite(v.wout.aspect))

    def test_raising_ntor_recentres_the_n_axis(self):
        v = self.vmec()
        rbc = np.array(v.indata.rbc)
        mpol, ntor = v._solver.resolution
        v.indata.ntor = ntor + 2
        v.run()
        self.assertEqual(v.indata.rbc.shape, (mpol, 2 * (ntor + 2) + 1))
        # Existing modes keep their physical n, i.e. they move by 2 along
        # the padded axis:
        np.testing.assert_allclose(v.indata.rbc[:, 2:-2], rbc)
        self.assertEqual(v.indata.raxis_c.size, ntor + 3)
        self.assertTrue(np.isfinite(v.wout.aspect))

    def test_lowering_mpol_and_ntor_runs(self):
        v = self.vmec()
        v.indata.mpol = 2
        v.indata.ntor = 2
        v.run()
        self.assertEqual(v.indata.rbc.shape, (2, 5))
        self.assertTrue(np.isfinite(v.wout.aspect))

    def test_set_mpol_ntor(self):
        """ The method vmecpp.simsopt_compat.Vmec.set_mpol_ntor provides. """
        v = self.vmec()
        rbc = np.array(v.indata.rbc)
        v._solver.set_mpol_ntor(7, 5)
        self.assertEqual(v.indata.mpol, 7)
        self.assertEqual(v.indata.ntor, 5)
        self.assertEqual(v.indata.rbc.shape, (7, 11))
        np.testing.assert_allclose(v.indata.rbc[:4, 2:-2], rbc)
        v.run()
        self.assertTrue(np.isfinite(v.wout.aspect))

    def test_resize_keeps_a_fourier_continuation_schedule(self):
        """ A sequence-valued mpol is resized by its last entry, not replaced. """
        v = self.vmec("input.circular_tokamak")
        multigrid(v)
        v.indata.mpol = np.array([4, 6])
        v._solver._ensure_indata_resolution()
        np.testing.assert_array_equal(v.indata.mpol, [4, 6])
        self.assertEqual(v.indata.rbc.shape[0], 6)

    def test_boundary_readable_after_raising_mpol(self):
        """ Reading the boundary back must not index past the old arrays. """
        solver = VmecppSolver(os.path.join(TEST_DIR, "input.li383_low_res"), None,
                              verbose=False)
        solver.indata.mpol = 6
        self.assertEqual(solver.boundary.mpol, 6)

    def test_repr_reports_the_final_resolution(self):
        v = self.vmec("input.circular_tokamak")
        multigrid(v)
        v.indata.mpol = np.array([4, 6])
        self.assertIn("mpol=6", repr(v))

    # ---------------------------------------------------------------
    # Fix 2: hot restart
    # ---------------------------------------------------------------

    def test_hot_restart_on_a_multi_ns_array_input(self):
        cold = self.vmec("input.circular_tokamak")
        multigrid(cold)
        cold.run()

        hot = self.vmec("input.circular_tokamak")
        multigrid(hot)
        hot._solver.restart_from = cold._solver.output_quantities
        hot.run()
        self.assertAlmostEqual(hot.wout.aspect, cold.wout.aspect, places=8)
        np.testing.assert_allclose(hot.wout.iotaf, cold.wout.iotaf, atol=1.0e-8)
        # The user's indata was not truncated to run the last step only:
        np.testing.assert_array_equal(hot.indata.ns_array, [5, 17])

    def test_restart_from_is_consumed_by_the_solve(self):
        v = self.vmec("input.circular_tokamak")
        multigrid(v)
        v.run()
        v._solver.restart_from = v._solver.output_quantities
        v.need_to_run_code = True
        v.run()
        self.assertIsNone(v._solver.restart_from)

    def test_incompatible_hot_restart_raises_objective_failure(self):
        """ vmecpp signals this with an AttributeError, which must be mapped. """
        v = self.vmec("input.circular_tokamak")
        multigrid(v)
        v.run()
        restart_from = v._solver.output_quantities
        other = self.vmec("input.li383_low_res")
        other._solver.restart_from = restart_from
        with self.assertRaises(ObjectiveFailure):
            other.run()

    # ---------------------------------------------------------------
    # Fix 3: mpi4py is optional for backends that do not use MPI
    # ---------------------------------------------------------------

    def test_runs_without_mpi4py(self):
        """
        VMEC++ is OpenMP-parallel, so simsopt must not require mpi4py to
        use it. Restores vmecpp commit ff79130b.
        """
        script = textwrap.dedent("""
            import sys

            class Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name == "mpi4py" or name.startswith("mpi4py."):
                        raise ImportError("mpi4py is blocked for this test")
                    return None

            sys.meta_path.insert(0, Blocker())
            import simsopt.mhd.vmec as vmec_module
            assert vmec_module.MPI is None
            from simsopt.mhd.vmecpp_solver import VmecppSolver

            v = vmec_module.Vmec(sys.argv[1], solver=VmecppSolver, verbose=False)
            assert v.mpi is None
            assert v._solver.mpi is None
            assert v._solver.group == 0
            v.update_mpi(None)
            assert v.aspect() > 0
            repr(v)

            # The VMEC2000 backend still requires mpi4py, as before:
            try:
                vmec_module.Vmec(sys.argv[1])
            except RuntimeError as e:
                assert "mpi4py needs to be installed" in str(e), e
            else:
                raise AssertionError("expected a RuntimeError")
            print("OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script,
             os.path.join(TEST_DIR, "input.circular_tokamak")],
            capture_output=True, text=True, check=False)
        self.assertIn("OK", result.stdout, msg=result.stderr)

    # ---------------------------------------------------------------
    # Fix 4: flipping lasym on a stellarator-symmetric input
    # ---------------------------------------------------------------

    def test_lasym_without_asymmetric_arrays_raises_value_error(self):
        v = self.vmec()
        v.indata.lasym = True
        for call in [v.run, v._solver._boundary_from_indata]:
            with self.subTest(call=call.__name__):
                with self.assertRaises(ValueError) as cm:
                    call()
                message = str(cm.exception)
                self.assertIn("lasym", message)
                self.assertIn("rbs", message)
                self.assertIn("stellarator-symmetric", message)
                self.assertIn("LASYM = T", message)


if __name__ == "__main__":
    unittest.main()
