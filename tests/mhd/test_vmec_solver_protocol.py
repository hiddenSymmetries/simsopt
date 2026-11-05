"""Tests of the Vmec solver protocols."""

import os
import unittest

from simsopt.mhd.profiles import ProfilePolynomial, ProfileSpline
from simsopt.mhd.vmec import (
    REQUIRED_WOUT_FIELDS,
    REQUIRED_WOUT_FIELDS_ASYM,
    FourierMode,
    ProfileProtocol,
    SurfaceRZFourierProtocol,
    Vmec,
    VmecBoundary,
    VmecSolverProtocol,
)

from . import TEST_DIR


class MinimalSolver:
    """ Exactly the members VmecSolverProtocol asks for, and no simsopt import. """

    def __init__(self):
        self.boundary = VmecBoundary()
        self.pressure = None
        self.current = None
        self.iota = None
        self.phiedge = 1.0
        self.curtor = 0.0
        self.pres_scale = 1.0
        self.indata = None
        self.wout = None
        self.output_file = None
        self.verbose = False

    def initialize(self, filename, mpi, keep_all_files=False, verbose=True):
        pass

    def solve(self):
        pass

    def load_wout(self):
        return 0

    def save_wout(self, filename):
        pass

    def update_mpi(self, new_mpi):
        pass


class VmecSolverProtocolTests(unittest.TestCase):
    def test_minimal_solver_conforms(self):
        self.assertIsInstance(MinimalSolver(), VmecSolverProtocol)

    def _assert_initialized_solver_conforms(self, solver):
        # Boundary and scalars are views onto indata, so exist only after initialize():
        solver.initialize(os.path.join(TEST_DIR, "input.li383_low_res"), None, verbose=False)
        self.assertIsInstance(solver, VmecSolverProtocol)

    def test_vmec2000_solver_conforms(self):
        try:
            from simsopt.mhd.vmec_solver import Vmec2000Solver
            import vmec  # noqa: F401
        except ImportError:
            self.skipTest("VMEC2000 is not installed")
        self._assert_initialized_solver_conforms(Vmec2000Solver())

    def test_vmecpp_solver_conforms(self):
        try:
            from simsopt.mhd.vmecpp_solver import VmecppSolver
        except ImportError:
            self.skipTest("vmecpp is not installed")
        self._assert_initialized_solver_conforms(VmecppSolver())

    def test_missing_member_does_not_conform(self):
        solver = MinimalSolver()
        del solver.indata
        self.assertNotIsInstance(solver, VmecSolverProtocol)
        self.assertNotIsInstance(object(), VmecSolverProtocol)

    def test_vmec_boundary_conforms(self):
        self.assertIsInstance(VmecBoundary(), SurfaceRZFourierProtocol)

    def test_simsopt_profiles_conform(self):
        """ Profiles cross the interface as callables of s. """
        for profile in [ProfilePolynomial([1.0, -1.0]), ProfileSpline([0.0, 0.5, 1.0], [1.0, 0.5, 0.0])]:
            self.assertIsInstance(profile, ProfileProtocol)
        self.assertNotIsInstance(1.0, ProfileProtocol)

    def test_fourier_mode(self):
        """ Mode numbers are named, and a key equals the plain (m, n) tuple. """
        mode = FourierMode(m=2, n=-1)
        self.assertEqual((mode.m, mode.n), (2, -1))
        self.assertEqual(mode, (2, -1))
        self.assertEqual({mode: 0.1}[(2, -1)], 0.1)
        self.assertNotEqual(mode, FourierMode(m=-1, n=2))

    def test_required_wout_fields(self):
        """ Reference wout files written by VMEC2000 have every required field. """
        cases = [("wout_li383_low_res_reference.nc", REQUIRED_WOUT_FIELDS),
                 ("wout_LandremanSenguptaPlunk_section5p3_reference.nc",
                  REQUIRED_WOUT_FIELDS + REQUIRED_WOUT_FIELDS_ASYM)]
        for filename, fields in cases:
            with self.subTest(filename=filename):
                wout = Vmec(str(TEST_DIR / filename)).wout
                missing = [name for name in fields if not hasattr(wout, name)]
                self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
