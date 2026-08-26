"""Test simsopt with the VMEC++ backend."""

import glob
import json
import os
import tempfile
import unittest

import numpy as np
import vmecpp
from simsopt._core.util import ObjectiveFailure, Struct
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.mhd.vmec import (
    REQUIRED_WOUT_FIELDS,
    REQUIRED_WOUT_FIELDS_ASYM,
    Vmec,
    VmecProfile,
    VmecSolverProtocol,
)
from simsopt.mhd.vmec_solver import load_wout_file
from simsopt.mhd.vmecpp_solver import (
    AXIS_ALIASES,
    COMMON_INDATA_FIELDS,
    VmecppIndata,
    VmecppSolver,
)

from . import TEST_DIR

#: Fixtures VMEC++ can load. Stellarator-symmetric and asymmetric,
#: multi-period and tokamak.
BOUNDARY_FIXTURES = [
    "input.li383_low_res",
    "input.circular_tokamak",
    "input.LandremanSenguptaPlunk_section5p3",
    "input.rotating_ellipse",
]


@unittest.skipIf(vmecpp is None, "vmecpp is not installed")
class VmecppSolverTests(unittest.TestCase):
    def tearDown(self):
        for name in glob.glob("wout_*_000_??????.nc"):
            os.remove(name)

    def solver(self, name="input.li383_low_res"):
        return VmecppSolver(os.path.join(TEST_DIR, name), None, verbose=False)

    def vmec(self, name="input.li383_low_res"):
        return Vmec(os.path.join(TEST_DIR, name), solver=VmecppSolver, verbose=False)

    def test_conforms_to_protocol(self):
        self.assertIsInstance(self.solver(), VmecSolverProtocol)

    def test_indata_is_a_vmecpp_vmecinput(self):
        """ The whole point: users get vmecpp's own type, hints and docstrings. """
        solver = self.solver()
        self.assertIsInstance(solver.indata, vmecpp.VmecInput)
        self.assertIsInstance(solver.indata, VmecppIndata)
        # Fields, and hence their type hints and docstrings, are
        # inherited rather than re-declared:
        self.assertEqual(set(VmecppIndata.model_fields), set(vmecpp.VmecInput.model_fields))
        for name, field in vmecpp.VmecInput.model_fields.items():
            self.assertEqual(VmecppIndata.model_fields[name].annotation, field.annotation)

    def test_bad_filename_raises(self):
        with self.assertRaises(ValueError):
            VmecppSolver("not_an_input_file", None)

    def test_boundary_matches_from_vmec_input(self):
        """
        The boundary read back from indata agrees with simsopt's own
        reader, for m < mpol. VMEC uses m = 0, ..., mpol - 1, and VMEC++
        drops the higher modes at parse time.
        """
        for name in BOUNDARY_FIXTURES:
            with self.subTest(name=name):
                path = os.path.join(TEST_DIR, name)
                solver = self.solver(name)
                mpol, ntor = solver.resolution
                reference = SurfaceRZFourier.from_vmec_input(path)
                boundary = solver.boundary
                self.assertEqual(boundary.nfp, reference.nfp)
                self.assertEqual(boundary.stellsym, reference.stellsym)
                self.assertEqual(boundary.mpol, mpol)
                self.assertEqual(boundary.ntor, ntor)
                # from_vmec_input sizes its surface by the modes actually
                # present in the file, which may be a lower resolution.
                m_max = min(mpol - 1, reference.mpol)
                n_max = min(ntor, reference.ntor)
                for m in range(m_max + 1):
                    for n in range(-n_max, n_max + 1):
                        self.assertAlmostEqual(boundary.rbc[(m, n)], reference.get_rc(m, n))
                        self.assertAlmostEqual(boundary.zbs[(m, n)], reference.get_zs(m, n))
                        if not reference.stellsym:
                            self.assertAlmostEqual(boundary.rbs[(m, n)], reference.get_rs(m, n))
                            self.assertAlmostEqual(boundary.zbc[(m, n)], reference.get_zc(m, n))

    def test_boundary_dof_count_matches_indata_mpol(self):
        """
        A Vmec on this backend reports the same mpol, and hence the same
        number of boundary dofs, as the VMEC2000 backend does.
        """
        v = self.vmec()
        self.assertEqual(v.indata.mpol, 4)
        self.assertEqual(v.boundary.mpol, 4)
        self.assertEqual(len(v.boundary.x), 63)

    def test_m_equals_mpol_row_reads_back_as_zero(self):
        """
        Pinning a known difference from the VMEC2000 backend, not a bug
        to fix here: input.li383_low_res specifies boundary coefficients
        at m == mpol, which VmecInput has no slot for, so this backend
        reports them as zero where VMEC2000 reports the file values. VMEC
        ignores those modes. Tracked in simsopt PR #437.
        """
        path = os.path.join(TEST_DIR, "input.li383_low_res")
        # from_vmec_input reads the file directly, as VMEC2000 does:
        reference = SurfaceRZFourier.from_vmec_input(path)
        v = self.vmec()
        mpol, ntor = v._solver.resolution
        self.assertEqual(v.boundary.mpol, mpol)
        self.assertGreaterEqual(reference.mpol, mpol)

        n_zeroed = 0
        for m in range(mpol + 1):
            for n in range(-ntor, ntor + 1):
                for get in ["get_rc", "get_zs"]:
                    if m == 0 and n < 0:
                        continue
                    expected = getattr(reference, get)(m, n)
                    actual = getattr(v.boundary, get)(m, n)
                    if m == mpol:
                        self.assertEqual(actual, 0.0)
                        n_zeroed += int(expected != 0.0)
                    else:
                        self.assertAlmostEqual(actual, expected)
        # 7 rbc + 7 zbs coefficients at m == mpol = 4, |n| <= ntor = 3:
        self.assertEqual(n_zeroed, 14)

    def test_boundary_round_trip_through_indata(self):
        """ Pushing a boundary and reading it back is the identity for m < mpol. """
        v = self.vmec()
        v.boundary.set_rc(1, 1, 0.0321)
        v.set_indata()
        v._solver._push_to_indata()
        read_back = v._solver._boundary_from_indata()
        mpol, _ = v._solver.resolution
        for (m, n), value in read_back.rbc.items():
            if m < mpol:
                self.assertAlmostEqual(value, v.boundary.get_rc(m, n))
        self.assertAlmostEqual(read_back.rbc[(1, 1)], 0.0321)

    def test_m_equals_mpol_row_is_dropped(self):
        """ indata has no slot for m == mpol, so that row is silently dropped. """
        solver = self.solver()
        mpol, ntor = solver.resolution
        boundary = solver.boundary
        boundary.rbc[(mpol, 0)] = 1.234
        solver.boundary = boundary
        solver._push_to_indata()
        self.assertEqual(solver.indata.rbc.shape, (mpol, 2 * ntor + 1))

    def test_run_preserves_an_assigned_low_resolution_boundary(self):
        """
        A boundary assigned at a lower resolution than ``indata`` is
        neither resized nor replaced by a run, so its dofs and its place
        in the dependency graph survive. Compare vmecpp/simsopt_compat
        #429, which had to resize a copy to get this.
        """
        v = self.vmec()
        surface = SurfaceRZFourier(mpol=1, ntor=1, nfp=v.indata.nfp)
        surface.set_rc(0, 0, 1.4)
        surface.set_rc(1, 0, 0.2)
        surface.set_zs(1, 0, 0.2)
        v.boundary = surface
        n_dofs = len(v.x)

        v.run()
        self.assertIs(v.boundary, surface)
        self.assertEqual(len(v.x), n_dofs)
        self.assertFalse(v.need_to_run_code)
        # The dependency chain still reaches the Vmec object:
        surface.set_rc(1, 0, 0.21)
        self.assertTrue(v.need_to_run_code)

        # Modes the surface does not have were zeroed in indata:
        ntor = v._solver.resolution[1]
        np.testing.assert_allclose(v.indata.rbc[2:, :], 0.0)
        self.assertAlmostEqual(v.indata.rbc[0, ntor], 1.4)
        self.assertAlmostEqual(v.indata.rbc[1, ntor], 0.2)

    def test_common_indata_fields_read_and_write(self):
        """ Every field in COMMON_INDATA_FIELDS is readable and writable. """
        indata = self.solver().indata
        for name in COMMON_INDATA_FIELDS:
            with self.subTest(name=name):
                value = getattr(indata, name)
                setattr(indata, name, value)
                new_value = getattr(indata, name)
                if isinstance(value, np.ndarray):
                    np.testing.assert_allclose(new_value, value)
                else:
                    self.assertEqual(new_value, value)

    def test_axis_aliases(self):
        """ The fortran axis names alias vmecpp's. """
        indata = self.solver().indata
        for fortran_name, vmecpp_name in AXIS_ALIASES.items():
            with self.subTest(name=fortran_name):
                self.assertIs(getattr(indata, fortran_name), getattr(indata, vmecpp_name))
        ntor = indata.raxis_c.size
        indata.raxis_cc = np.linspace(1.0, 2.0, ntor)
        np.testing.assert_allclose(indata.raxis_c, np.linspace(1.0, 2.0, ntor))
        indata.zaxis_cs = np.zeros(ntor)
        np.testing.assert_allclose(indata.zaxis_s, 0.0)

    def test_bytes_accepted_for_string_fields(self):
        """ fortran-facing code assigns b'...', which must be decoded. """
        indata = self.solver().indata
        indata.mgrid_file = b"mgrid_foo.nc   "
        self.assertEqual(indata.mgrid_file, "mgrid_foo.nc")
        for name in ["pmass_type", "pcurr_type", "piota_type"]:
            setattr(indata, name, b"cubic_spline  ")
            self.assertEqual(getattr(indata, name), "cubic_spline")

    def test_dofs_alias_indata(self):
        v = self.vmec()
        v.indata.curtor = 1.23e5
        self.assertAlmostEqual(v.get_dofs()[1], 1.23e5)
        v.indata.phiedge = 0.44
        v.indata.pres_scale = 3.0
        np.testing.assert_allclose(v.get_dofs(), [0.44, 1.23e5, 3.0])
        v.set_dofs([0.5, 2.0e5, 1.5])
        self.assertAlmostEqual(v.indata.phiedge, 0.5)
        self.assertAlmostEqual(v.indata.curtor, 2.0e5)
        self.assertAlmostEqual(v.indata.pres_scale, 1.5)

    def test_profiles_pushed_to_indata(self):
        solver = self.solver()
        solver.boundary = solver.boundary
        solver.pressure = VmecProfile("power_series", [1.0e5, 0.0, -1.0e5])
        solver.current = VmecProfile("cubic_spline_i", [1.0e6, 0.0], [0.0, 1.0])
        solver._push_to_indata()
        np.testing.assert_allclose(solver.indata.am, [1.0e5, 0.0, -1.0e5])
        self.assertEqual(solver.indata.pmass_type, "power_series")
        np.testing.assert_allclose(solver.indata.ac_aux_s, [0.0, 1.0])
        np.testing.assert_allclose(solver.indata.ac_aux_f, [1.0e6, 0.0])
        self.assertEqual(solver.indata.pcurr_type, "cubic_spline_i")

    def test_unsupported_profile_type_raises_value_error(self):
        solver = self.solver()
        with self.assertRaises(ValueError) as cm:
            solver.set_profile(VmecProfile("sum_atan", [1.0]), "m")
        self.assertIn("VMEC++", str(cm.exception))
        self.assertIn("power_series", str(cm.exception))
        self.assertIn("cubic_spline", str(cm.exception))

    def test_update_mpi_tolerates_none(self):
        solver = self.solver()
        solver.update_mpi(None)
        self.assertIsNone(solver.mpi)
        self.assertEqual(solver.group, 0)

    def test_required_wout_fields_present(self):
        """ VmecWOut provides every field downstream simsopt code reads. """
        fields = set(vmecpp.VmecWOut.model_fields) | set(dir(vmecpp.VmecWOut))
        missing = [name for name in REQUIRED_WOUT_FIELDS + REQUIRED_WOUT_FIELDS_ASYM
                   if name not in fields]
        self.assertEqual(missing, [])
        # Vmec.volume() reads wout.volume, which vmecpp aliases from volume_p:
        self.assertIn("volume", fields)

    def test_run_circular_tokamak(self):
        v = self.vmec("input.circular_tokamak")
        v._solver.max_threads = 1
        for value in [v.aspect(), v.volume(), v.mean_iota(), v.iota_axis(),
                      v.iota_edge(), v.vacuum_well()]:
            self.assertTrue(np.isfinite(value))
        self.assertEqual(v.wout.ier_flag, 0)
        self.assertEqual(v.wout.rmnc.shape[1], v.wout.ns)
        # The wout file was saved, even though keep_all_files is False:
        self.assertTrue(os.path.isfile(v.output_file))

        # A second call is served from the cache:
        self.assertFalse(v.need_to_run_code)
        aspect = v.aspect()
        self.assertEqual(v.iter, 0)
        # Changing a dof re-triggers the solve:
        v.boundary.set_rc(0, 0, v.boundary.get_rc(0, 0) * 1.01)
        self.assertNotAlmostEqual(v.aspect(), aspect, delta=0.0)
        self.assertEqual(v.iter, 1)

    def test_load_wout_round_trip(self):
        v = self.vmec("input.circular_tokamak")
        v.run()
        aspect = v.wout.aspect
        v.load_wout()
        self.assertAlmostEqual(v.wout.aspect, aspect)
        np.testing.assert_allclose(np.linspace(0, 1, v.wout.ns), v.s_full_grid)

    def test_json_input_file(self):
        """ VMEC++ JSON input files work too, and are named wout_<name>.nc. """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "circtok.json")
            source = vmecpp.VmecInput.from_file(
                os.path.join(TEST_DIR, "input.circular_tokamak"))
            with open(path, "w") as f:
                f.write(source.model_dump_json())
            solver = VmecppSolver(path, None, verbose=False)
            self.assertEqual(solver.indata.nfp, source.nfp)
            v = Vmec(path, solver=solver, verbose=False)
            self.assertTrue(np.isfinite(v.aspect()))
            self.assertEqual(os.path.basename(v.output_file),
                             "wout_circtok_000_000000.nc")

    def test_saved_wout_is_readable_as_a_fortran_wout(self):
        """ The file VMEC++ writes is what virtual_casing and Boozer expect. """
        v = self.vmec("input.circular_tokamak")
        v.run()
        wout = Struct()
        load_wout_file(v.output_file, wout)
        self.assertAlmostEqual(wout.aspect, v.wout.aspect)
        self.assertEqual(wout.rmnc.shape, v.wout.rmnc.shape)
        np.testing.assert_allclose(wout.rmnc, v.wout.rmnc)
        np.testing.assert_allclose(wout.gmnc, v.wout.gmnc)

    def test_non_convergence_raises_objective_failure(self):
        v = self.vmec("input.circular_tokamak")
        v.indata.niter_array = np.array([2])
        with self.assertRaises(ObjectiveFailure):
            v.run()

    def test_vmec2000_only_methods_raise(self):
        v = self.vmec()
        for name in ["get_max_mn"]:
            with self.assertRaises(NotImplementedError):
                getattr(v, name)()

    def test_get_input_returns_vmecpp_json(self):
        """ get_input() emits VMEC++ JSON, as vmecpp/simsopt_compat did. """
        v = self.vmec()
        v.boundary.set_rc(1, 1, 0.0321)
        indata = json.loads(v.get_input())
        self.assertEqual(indata["nfp"], v.indata.nfp)
        self.assertEqual(indata["mpol"], v.indata.mpol)
        # The boundary was transferred to indata first:
        rbc = {(mode["m"], mode["n"]): mode["value"] for mode in indata["rbc"]}
        self.assertAlmostEqual(rbc[(1, 1)], 0.0321)
        self.assertIsInstance(vmecpp.VmecInput.model_validate(indata),
                              vmecpp.VmecInput)

    def test_write_input_json_and_namelist(self):
        """ A '*.json' name gets JSON; an 'input.*' name gets an INDATA namelist. """
        v = self.vmec()
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "li383.json")
            v.write_input(json_path)
            with open(json_path) as f:
                self.assertEqual(json.load(f)["nfp"], v.indata.nfp)

            namelist_path = os.path.join(tmpdir, "input.li383")
            v.write_input(namelist_path)
            with open(namelist_path) as f:
                self.assertIn("&INDATA", f.read())
            # Both files describe the same equilibrium:
            for path in [json_path, namelist_path]:
                written = vmecpp.VmecInput.from_file(path)
                self.assertEqual(written.mpol, v.indata.mpol)
                np.testing.assert_allclose(written.rbc, v.indata.rbc, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
