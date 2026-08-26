"""
Drive :obj:`~simsopt.mhd.vmec.Vmec` with a backend that implements only
:obj:`~simsopt.mhd.vmec.VmecSolverProtocol`. Nothing here may touch the
VMEC2000 fortran extension, so this module must pass when VMEC2000 is
not installed.
"""

import os
import unittest

import numpy as np
from simsopt._core.util import Struct
from simsopt.mhd.profiles import ProfilePolynomial, ProfileSpline
from simsopt.mhd.vmec import Vmec, VmecBoundary, VmecProfile, VmecSolverProtocol
from simsopt.mhd.vmec_solver import load_wout_file

from . import TEST_DIR

#: The boundary the fake backend reports at initialization.
FAKE_RBC = {(0, 0): 1.5, (1, 0): 0.3, (1, 1): 0.05}
FAKE_ZBS = {(1, 0): 0.31, (1, 1): -0.04}


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
    """
    A backend implementing exactly ``VmecSolverProtocol``: no fortran, no
    MPI and no VMEC2000 import. ``solve()`` just loads a stored wout file
    and counts how many times it was called.
    """

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
        self.n_update_mpi = 0

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
        self.n_update_mpi += 1


def fake_vmec():
    """ A ``Vmec`` driven by a fresh :obj:`FakeVmecSolver`. """
    return Vmec(os.path.join(TEST_DIR, "input.li383_low_res"), solver=FakeVmecSolver)


class VmecSolverProtocolTests(unittest.TestCase):
    def test_fake_solver_conforms(self):
        solver = FakeVmecSolver("input.dummy", None)
        self.assertIsInstance(solver, VmecSolverProtocol)

    def test_non_conforming_solver_raises(self):
        with self.assertRaises(TypeError):
            Vmec(os.path.join(TEST_DIR, "input.li383_low_res"), solver=object)

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

    def test_dofs_are_views_onto_indata(self):
        v = fake_vmec()
        np.testing.assert_allclose(v.get_dofs(), [0.5, 1.0e5, 2.0])
        v.indata.curtor = 3.3e5
        np.testing.assert_allclose(v.get_dofs(), [0.5, 3.3e5, 2.0])
        v.set_dofs([0.7, 4.4e5, 1.5])
        self.assertAlmostEqual(v.indata.phiedge, 0.7)
        self.assertAlmostEqual(v.indata.curtor, 4.4e5)
        self.assertAlmostEqual(v.indata.pres_scale, 1.5)

    def test_profile_exact_passthrough(self):
        """ A simsopt profile matching the vmec parametrization is passed through. """
        v = fake_vmec()
        coeffs = [1.0e5, 0.0, -1.0e5]
        v.pressure_profile = ProfilePolynomial(coeffs)
        v.indata.pmass_type = "power_series"

        knots = [0.0, 0.25, 0.75, 1.0]
        values = [1.0e6, 0.9e6, 0.5e6, 0.0]
        v.current_profile = ProfileSpline(knots, values)
        v.indata.pcurr_type = "cubic_spline_i"

        v.set_indata()
        pressure = v._solver.pressure
        self.assertIsInstance(pressure, VmecProfile)
        self.assertEqual(pressure.profile_type, "power_series")
        np.testing.assert_allclose(pressure.coeffs, coeffs)
        self.assertIsNone(pressure.knots)

        current = v._solver.current
        self.assertEqual(current.profile_type, "cubic_spline_i")
        np.testing.assert_allclose(current.knots, knots)
        np.testing.assert_allclose(current.coeffs, values)
        # cubic_spline_i means I(s), so curtor is the profile at s=1:
        self.assertAlmostEqual(v._solver.curtor, values[-1])
        # A pressure profile object always takes over the scaling:
        self.assertAlmostEqual(v._solver.pres_scale, 1.0)

        # iota is unused here, so None must reach the solver:
        self.assertIsNone(v._solver.iota)

    def test_profile_sampled_fallback(self):
        """ A mismatched parametrization is resampled onto vmec's. """
        v = fake_vmec()
        v.pressure_profile = ProfilePolynomial([1.0e5, 0.0, -1.0e5])
        v.indata.pmass_type = "cubic_spline"
        v.n_pressure = 6
        v.set_indata()

        pressure = v._solver.pressure
        self.assertEqual(pressure.profile_type, "cubic_spline")
        s = np.linspace(0, 1, 6)
        np.testing.assert_allclose(pressure.knots, s)
        np.testing.assert_allclose(pressure.coeffs, v.pressure_profile(s))

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

    def test_update_mpi_forwarded(self):
        v = fake_vmec()
        v.update_mpi("a new partition")
        self.assertEqual(v._solver.n_update_mpi, 1)
        self.assertEqual(v.mpi, "a new partition")

    def test_verbose_forwarded(self):
        v = fake_vmec()
        v.verbose = False
        self.assertFalse(v._solver.verbose)
        self.assertFalse(v.verbose)


if __name__ == "__main__":
    unittest.main()
