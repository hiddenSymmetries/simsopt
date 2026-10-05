"""Regression guards for refactoring Vmec."""

import unittest

import numpy as np
from monty.tempfile import ScratchDir

try:
    import vmec as vmec_mod
except ImportError:
    vmec_mod = None

from simsopt.mhd.profiles import ProfilePolynomial, ProfileSpline
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_solver import Vmec2000Solver, fit_profile, profile_curtor

from . import TEST_DIR


class ProfileFittingTests(unittest.TestCase):
    """ Profile fitting, which needs no fortran. """

    def test_power_series_fit(self):
        coeffs, knots = fit_profile(ProfilePolynomial([1.0e5, 0.0, -1.0e5]), 10, "power_series")
        self.assertIsNone(knots)
        self.assertEqual(len(coeffs), 10)
        np.testing.assert_allclose(coeffs[:3], [1.0e5, 0.0, -1.0e5], atol=1e-4)
        np.testing.assert_allclose(coeffs[3:], 0, atol=1e-4)

    def test_spline_sampling(self):
        """ A spline is sampled at n uniform nodes, even a ProfileSpline with its own knots. """
        profile = ProfileSpline([0.0, 0.25, 0.75, 1.0], [1.0e6, 0.9e6, 0.5e6, 0.0])
        for profile_type in ["cubic_spline", "cubic_spline_i", "akima_spline", "line_segment"]:
            with self.subTest(profile_type=profile_type):
                values, knots = fit_profile(profile, 6, profile_type)
                np.testing.assert_allclose(knots, np.linspace(0, 1, 6))
                np.testing.assert_allclose(values, profile(np.linspace(0, 1, 6)))

    def test_unsupported_type_raises(self):
        with self.assertRaises(RuntimeError):
            fit_profile(ProfilePolynomial([1.0]), 10, "gauss_trunc")

    def test_curtor(self):
        """ I'(s) types integrate the profile, I(s) types evaluate it at s = 1. """
        current = ProfilePolynomial([2.0, -2.0])
        self.assertAlmostEqual(profile_curtor(current, "power_series"), 1.0)
        self.assertAlmostEqual(profile_curtor(current, "cubic_spline_ip"), 1.0)
        self.assertAlmostEqual(profile_curtor(current, "cubic_spline_i"), 0.0)


@unittest.skipIf(vmec_mod is None, "vmec python extension is not installed")
class Vmec2000SolverTests(unittest.TestCase):
    def test_set_indata_writes_indata(self):
        """ set_indata() transfers the boundary and profiles into indata at once. """
        v = Vmec(str(TEST_DIR / "input.li383_low_res"), verbose=False)
        v.boundary.set_rc(1, 1, 0.123)
        v.boundary.set_zs(2, -1, -0.045)
        v.pressure_profile = ProfilePolynomial([2.0e4, -2.0e4])
        v.indata.raxis_cc[0] = 1.5
        v.set_indata()
        self.assertEqual(v.indata.rbc[101 + 1, 1], 0.123)
        self.assertEqual(v.indata.zbs[101 - 1, 2], -0.045)
        np.testing.assert_allclose(v.indata.am[:2], [2.0e4, -2.0e4], atol=1e-6)
        self.assertEqual(v.indata.raxis_cc[0], 0.0)

    def test_n_pressure_sets_the_number_of_spline_nodes(self):
        v = Vmec(str(TEST_DIR / "input.li383_low_res"), verbose=False)
        v.indata.pmass_type = "cubic_spline"
        v.pressure_profile = ProfilePolynomial([2.0e4, -2.0e4])
        v.n_pressure = 6
        v.set_indata()
        np.testing.assert_allclose(v.indata.am_aux_s[:6], np.linspace(0, 1, 6))
        np.testing.assert_allclose(v.indata.am_aux_s[6:], 0)

    def test_get_max_mn_and_repr(self):
        v = Vmec(str(TEST_DIR / "input.li383_low_res"), verbose=False)
        self.assertEqual(v.get_max_mn(), (6, 4))
        self.assertIn("nfp=3 mpol=4 ntor=3", repr(v))
        self.assertIn("nfp=3 mpol=4 ntor=3", repr(v._solver))

    def test_assigned_boundary_is_read_back(self):
        v = Vmec(str(TEST_DIR / "input.li383_low_res"), verbose=False)
        v.set_indata()
        self.assertIs(v._solver.boundary, v._solver._boundary)

    def test_resolution_limits(self):
        v = Vmec(str(TEST_DIR / "input.li383_low_res"), verbose=False)
        for name in ["mpol", "ntor"]:
            with self.subTest(name=name):
                old = getattr(v.indata, name)
                setattr(v.indata, name, 102)
                with self.assertRaises(ValueError):
                    v.set_indata()
                setattr(v.indata, name, old)

    def test_push_without_boundary_raises(self):
        v = Vmec(str(TEST_DIR / "input.li383_low_res"), verbose=False)
        solver = Vmec2000Solver()
        solver.initialize(str(TEST_DIR / "input.li383_low_res"), v.mpi, verbose=False)
        with self.assertRaises(RuntimeError):
            solver.get_input()

    def test_json_input_raises(self):
        """ VMEC2000 cannot read a VMEC++ JSON input file. """
        with self.assertRaisesRegex(ValueError, "VmecppSolver"):
            Vmec2000Solver().initialize("input.li383_low_res.json", None)

    def test_second_initialize_raises(self):
        v = Vmec(str(TEST_DIR / "input.li383_low_res"), verbose=False)
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            Vmec(str(TEST_DIR / "input.li383_low_res"), verbose=False, solver=v.solver)

    def test_update_mpi(self):
        from simsopt.util.mpi import MpiPartition
        v = Vmec(str(TEST_DIR / "input.li383_low_res"), verbose=False)
        mpi = MpiPartition(ngroups=1)
        v.update_mpi(mpi)
        self.assertIs(v.mpi, mpi)
        self.assertIs(v._solver.mpi, mpi)
        self.assertEqual(v._solver.fcomm, mpi.comm_groups.py2f())
        self.assertEqual(v.iter, -1)

    def test_result_is_independent_of_history(self):
        """ Boundary A, B, A gives the same answer for A, via the axis reset. """
        with ScratchDir("."):
            v = Vmec(str(TEST_DIR / "input.li383_low_res"), verbose=False)
            x0 = v.boundary.x.copy()
            first = (v.aspect(), v.iota_axis(), v.volume())
            v.boundary.x = x0 * (1 + 1.0e-3 * np.cos(np.arange(len(x0))))
            v.run()
            v.boundary.x = x0
            second = (v.aspect(), v.iota_axis(), v.volume())
            np.testing.assert_allclose(second, first, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
