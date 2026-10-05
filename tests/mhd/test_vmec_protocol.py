"""Vmec driven by a fake VmecSolverProtocol backend, without VMEC2000."""

import os
import unittest

import numpy as np
from simsopt._core.util import Struct
from simsopt.mhd.profiles import ProfilePolynomial, ProfileSpline
from simsopt.mhd.vmec import (
    FourierMode,
    ProfileProtocol,
    Vmec,
    VmecBoundary,
    VmecSolverProtocol,
)
from simsopt.mhd.vmec_solver import load_wout_file

from . import TEST_DIR

#: The boundary the fake backend reports at initialization.
FAKE_RBC = {FourierMode(0, 0): 1.5, FourierMode(1, 0): 0.3, FourierMode(1, 1): 0.05}
FAKE_ZBS = {FourierMode(1, 0): 0.31, FourierMode(1, 1): -0.04}


class FakeIndata:
    """ The minimal solver settings ``Vmec`` reads through ``indata``. """

    def __init__(self):
        self.nfp = 3
        self.mpol = 2
        self.ntor = 1
        self.ncurr = 1
        self.phiedge = 0.5
        self.curtor = 1.0e5
        self.pres_scale = 2.0
        self.pmass_type = "power_series"
        self.pcurr_type = "power_series"
        self.piota_type = "power_series"


class FakeVmecSolver:
    """ ``solve()`` loads a stored wout file and counts the calls. """

    def __init__(self, filename, mpi, keep_all_files=False, verbose=True):
        self.input_file = filename
        self.verbose = verbose
        self.indata = FakeIndata()
        self.wout = Struct()
        self.output_file = os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc")

        self.boundary = VmecBoundary(nfp=self.indata.nfp, stellsym=True,
                                     mpol=self.indata.mpol, ntor=self.indata.ntor,
                                     rbc=dict(FAKE_RBC), zbs=dict(FAKE_ZBS))
        self.pressure = None
        self.current = None
        self.iota = None

        self.n_solve = 0
        self.mpi = mpi

    # phiedge, curtor and pres_scale must be views onto indata:
    @property
    def phiedge(self):
        return self.indata.phiedge

    @phiedge.setter
    def phiedge(self, phiedge):
        self.indata.phiedge = phiedge

    @property
    def curtor(self):
        return self.indata.curtor

    @curtor.setter
    def curtor(self, curtor):
        self.indata.curtor = curtor

    @property
    def pres_scale(self):
        return self.indata.pres_scale

    @pres_scale.setter
    def pres_scale(self, pres_scale):
        self.indata.pres_scale = pres_scale

    def solve(self):
        self.n_solve += 1
        self.load_wout()

    def load_wout(self):
        return load_wout_file(self.output_file, self.wout)

    def update_mpi(self, new_mpi):
        self.mpi = new_mpi


def fake_vmec():
    """ A ``Vmec`` driven by a fresh :obj:`FakeVmecSolver`. """
    filename = os.path.join(TEST_DIR, "input.li383_low_res")
    return Vmec(filename, solver=FakeVmecSolver(filename, None))


class VmecSolverProtocolTests(unittest.TestCase):
    def test_fake_solver_conforms(self):
        solver = FakeVmecSolver("input.dummy", None)
        self.assertIsInstance(solver, VmecSolverProtocol)

    def test_non_conforming_solver_raises(self):
        with self.assertRaises(TypeError) as cm:
            Vmec(os.path.join(TEST_DIR, "input.li383_low_res"), solver=object())
        self.assertIn("VmecSolverProtocol", str(cm.exception))

    def test_solver_instance_is_used(self):
        filename = os.path.join(TEST_DIR, "input.li383_low_res")
        solver = FakeVmecSolver(filename, None)
        self.assertIs(Vmec(filename, solver=solver)._solver, solver)

    def test_solver_instance_gets_mpi(self):
        filename = os.path.join(TEST_DIR, "input.li383_low_res")
        solver = FakeVmecSolver(filename, None)
        Vmec(filename, mpi="a partition", solver=solver)
        self.assertEqual(solver.mpi, "a partition")

    def test_settings_with_solver_instance_raise(self):
        """ keep_all_files and verbose are set on the solver itself. """
        filename = os.path.join(TEST_DIR, "input.li383_low_res")
        for kwargs in [dict(keep_all_files=True), dict(verbose=False)]:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                Vmec(filename, solver=FakeVmecSolver(filename, None), **kwargs)

    def test_boundary_read_back_at_init(self):
        v = fake_vmec()
        self.assertEqual(v.boundary.nfp, 3)
        self.assertTrue(v.boundary.stellsym)
        self.assertEqual(v.boundary.mpol, 2)
        self.assertEqual(v.boundary.ntor, 1)
        for (m, n), value in FAKE_RBC.items():
            self.assertAlmostEqual(v.boundary.get_rc(m, n), value)
        for (m, n), value in FAKE_ZBS.items():
            self.assertAlmostEqual(v.boundary.get_zs(m, n), value)

    def test_run_and_caching(self):
        v = fake_vmec()
        solver = v._solver
        self.assertAlmostEqual(v.aspect(), v.wout.aspect)
        self.assertEqual(solver.n_solve, 1)

        # A second call is served from the cache:
        v.aspect()
        self.assertEqual(solver.n_solve, 1)

        # Changing a boundary dof re-triggers the solve:
        v.boundary.set_rc(1, 0, v.boundary.get_rc(1, 0) * 1.01)
        v.aspect()
        self.assertEqual(solver.n_solve, 2)

    def test_boundary_pushed_to_solver(self):
        v = fake_vmec()
        v.boundary.set_rc(1, 1, 0.123)
        v.run()
        pushed = v._solver.boundary
        self.assertAlmostEqual(pushed.rbc[(1, 1)], 0.123)
        self.assertEqual(pushed.nfp, v.boundary.nfp)
        self.assertIsNotNone(pushed.surface)

    def test_boundary_keys_are_fourier_modes(self):
        """ Modes are keyed by FourierMode. """
        v = fake_vmec()
        v.boundary.set_rc(2, -1, 0.012)
        v.set_indata()
        boundary = v._solver.boundary
        self.assertTrue(all(isinstance(mode, FourierMode) for mode in boundary.rbc))
        self.assertEqual(boundary.rbc[FourierMode(m=2, n=-1)], 0.012)
        self.assertEqual(boundary.rbc[FourierMode(m=1, n=1)], v.boundary.get_rc(1, 1))
        # A plain (m, n) tuple is the same key:
        self.assertEqual(boundary.rbc[(2, -1)], 0.012)

    def test_dofs_are_views_onto_indata(self):
        v = fake_vmec()
        np.testing.assert_allclose(v.get_dofs(), [0.5, 1.0e5, 2.0])
        v.indata.curtor = 3.3e5
        np.testing.assert_allclose(v.get_dofs(), [0.5, 3.3e5, 2.0])
        v.set_dofs([0.7, 4.4e5, 1.5])
        self.assertAlmostEqual(v.indata.phiedge, 0.7)
        self.assertAlmostEqual(v.indata.curtor, 4.4e5)
        self.assertAlmostEqual(v.indata.pres_scale, 1.5)

    def test_profiles_reach_the_solver_as_callables(self):
        """ Profiles reach the solver unchanged. """
        v = fake_vmec()
        v.pressure_profile = ProfilePolynomial([1.0e5, 0.0, -1.0e5])
        v.current_profile = ProfileSpline([0.0, 0.25, 0.75, 1.0], [1.0e6, 0.9e6, 0.5e6, 0.0])
        curtor = v._solver.curtor

        v.set_indata()
        self.assertIs(v._solver.pressure, v.pressure_profile)
        self.assertIs(v._solver.current, v.current_profile)
        self.assertIsInstance(v._solver.pressure, ProfileProtocol)
        # iota is unused here, so None must reach the solver:
        self.assertIsNone(v._solver.iota)
        # A pressure profile object always takes over the scaling:
        self.assertAlmostEqual(v._solver.pres_scale, 1.0)
        # curtor follows from the current profile only through the solver:
        self.assertEqual(v._solver.curtor, curtor)
        # This solver does not fit profiles, so it is not given n_pressure:
        self.assertFalse(hasattr(v._solver, "n_pressure"))

    def test_methods_outside_the_protocol_raise(self):
        """ A backend need not provide get_input/write_input/get_max_mn. """
        v = fake_vmec()
        for name in ["get_input", "get_max_mn"]:
            with self.assertRaises(AttributeError) as cm:
                getattr(v, name)()
            self.assertIn(name, str(cm.exception))
        with self.assertRaises(AttributeError):
            v.write_input("input.should_not_be_written")
        self.assertFalse(os.path.exists("input.should_not_be_written"))

    def test_solver_class_raises(self):
        """ A class must be instantiated by the user, even if its class attributes satisfy the protocol. """
        with self.assertRaises(TypeError) as cm:
            Vmec(os.path.join(TEST_DIR, "input.li383_low_res"), solver=FakeVmecSolver)
        self.assertIn("instance", str(cm.exception))

    def test_indata_of_wout_initialized_object(self):
        """ As before the split, a wout-initialized Vmec has no indata attribute. """
        v = Vmec(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
        self.assertFalse(hasattr(v, "indata"))

    def test_attributes_forwarded_to_solver(self):
        v = fake_vmec()
        for name, value in [("input_file", "input.other"), ("iter", 7), ("keep_all_files", True),
                            ("files_to_delete", ["a"]), ("free_boundary", True), ("ictrl", [1]),
                            ("fcomm", 3), ("output_file", "wout_other.nc"), ("wout", Struct())]:
            with self.subTest(name=name):
                setattr(v, name, value)
                self.assertIs(getattr(v._solver, name), value)
                self.assertIs(getattr(v, name), value)

    def test_load_wout_through_solver(self):
        v = fake_vmec()
        self.assertEqual(v.load_wout(), 0)
        self.assertEqual(v.wout.ns, 16)
        self.assertEqual(len(v.s_half_grid), 15)

    def test_wout_initialized_object_has_no_solver(self):
        v = Vmec(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
        with self.assertRaises(AttributeError):
            _ = v.iter
        wout = Struct()
        v.wout = wout
        v.output_file = "wout_other.nc"
        self.assertIs(v.wout, wout)
        self.assertEqual(v.output_file, "wout_other.nc")

    def test_update_mpi_forwarded(self):
        v = fake_vmec()
        v.update_mpi("a new partition")
        self.assertEqual(v._solver.mpi, "a new partition")
        self.assertEqual(v.mpi, "a new partition")

    def test_verbose_forwarded(self):
        v = fake_vmec()
        v.verbose = False
        self.assertFalse(v._solver.verbose)
        self.assertFalse(v.verbose)


if __name__ == "__main__":
    unittest.main()
