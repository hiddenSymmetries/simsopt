"""VMEC2000 and VMEC++ on the same input files."""

import glob
import os
import unittest

import numpy as np
from monty.tempfile import ScratchDir

try:
    import vmec as vmec_mod
except ImportError:
    vmec_mod = None

try:
    from simsopt.mhd.vmecpp_solver import VmecppSolver
except ImportError:
    VmecppSolver = None

from simsopt.mhd.profiles import ProfilePolynomial
from simsopt.mhd.vmec import Vmec

from . import TEST_DIR

#: Measured differences with some headroom.
RTOL = {"aspect": 1e-12, "volume": 1e-12, "mean_iota": 1e-5, "iota_axis": 2e-4,
        "iota_edge": 1e-4, "vacuum_well": 2e-4, "external_current": 1e-6}


def indata_files():
    """ All VMEC INDATA fixtures, i.e. input.* files that are not FOCUS files. """
    for path in sorted(glob.glob(os.path.join(TEST_DIR, "input.*"))):
        with open(path) as f:
            if not f.read(100).lstrip().startswith("#"):
                yield path


def trim(array):
    """ ``array`` up to its last nonzero entry. """
    array = np.asarray(array, dtype=float).ravel()
    nonzero = np.nonzero(array)[0]
    return array[:nonzero[-1] + 1] if len(nonzero) else array[:0]


def tag(value):
    return (value.decode() if isinstance(value, bytes) else value).strip().lower()


@unittest.skipIf(vmec_mod is None or VmecppSolver is None,
                 "both the vmec python extension and vmecpp are needed")
class VmecBackendConversionTests(unittest.TestCase):
    def test_input_conversion(self):
        """ Every INDATA fixture reads into the same Vmec state on both backends. """
        for path in indata_files():
            name = os.path.basename(path)
            with self.subTest(name=name):
                a = Vmec(path, verbose=False)
                b = Vmec(path, solver=VmecppSolver(path, None, verbose=False))
                ia, ib = a.indata, b.indata

                for field in ["nfp", "lasym", "ncurr", "lfreeb"]:
                    self.assertEqual(bool(getattr(ia, field)) if field in ["lasym", "lfreeb"]
                                     else getattr(ia, field), getattr(ib, field), field)
                for field in ["phiedge", "pres_scale", "delt", "gamma"]:
                    self.assertEqual(getattr(ia, field), getattr(ib, field), field)
                if ia.ncurr == 1:
                    # VMEC++ reads CURTOR as 0 when NCURR = 0, where it is unused.
                    self.assertEqual(ia.curtor, ib.curtor)
                for field in ["pmass_type", "pcurr_type", "piota_type"]:
                    self.assertEqual(tag(getattr(ia, field)), tag(getattr(ib, field)), field)
                profiles = ["am", "ac"] if ia.ncurr == 1 else ["am", "ai"]
                for field in profiles:
                    np.testing.assert_array_equal(trim(getattr(ia, field)),
                                                  trim(getattr(ib, field)), field)
                np.testing.assert_array_equal(trim(ia.ns_array), trim(ib.ns_array))

                self.assertEqual(a.boundary.mpol, b.boundary.mpol)
                self.assertEqual(a.boundary.ntor, b.boundary.ntor)
                self.assertEqual(a.boundary.local_full_dof_names,
                                 b.boundary.local_full_dof_names)
                if ia.lasym:
                    # VMEC2000's readin theta-shifts the m = 1 modes; the surface is the same:
                    np.testing.assert_allclose(b.boundary.volume(), a.boundary.volume(), rtol=1e-10)
                    np.testing.assert_allclose(b.boundary.area(), a.boundary.area(), rtol=1e-10)
                    continue
                # VMEC++ has no slot for m == mpol, so that row reads back as zero:
                mpol = a.boundary.mpol
                for dof, xa, xb in zip(a.boundary.local_full_dof_names,
                                       a.boundary.x, b.boundary.x):
                    if dof[3:].startswith(f"{mpol},"):
                        self.assertEqual(xb, 0.0, dof)
                    else:
                        self.assertEqual(xa, xb, dof)

    def test_profile_fit_agrees(self):
        """ An attached profile is fit into the same indata arrays by both backends. """
        path = os.path.join(TEST_DIR, "input.li383_low_res")
        profile = ProfilePolynomial([2.0e4, 0.0, -2.0e4])
        for pmass_type, fields in [("power_series", ["am"]),
                                   ("cubic_spline", ["am_aux_s", "am_aux_f"])]:
            with self.subTest(pmass_type=pmass_type):
                arrays = []
                for v in [Vmec(path, verbose=False),
                          Vmec(path, solver=VmecppSolver(path, None, verbose=False))]:
                    v.indata.pmass_type = pmass_type
                    v.pressure_profile = profile
                    v.n_pressure = 7
                    v.get_input()
                    arrays.append([trim(getattr(v.indata, name)) for name in fields])
                for a, b in zip(*arrays):
                    np.testing.assert_array_equal(a, b)


@unittest.skipIf(vmec_mod is None or VmecppSolver is None,
                 "both the vmec python extension and vmecpp are needed")
class VmecBackendResultTests(unittest.TestCase):
    def outputs(self, v):
        return {name: getattr(v, name)() for name in RTOL}

    def test_outputs_agree(self):
        rtol = RTOL
        for name in ["input.li383_low_res", "input.LandremanPaul2021_QA_lowres"]:
            with self.subTest(name=name), ScratchDir("."):
                path = os.path.join(TEST_DIR, name)
                a = self.outputs(Vmec(path, verbose=False))
                v = Vmec(path, solver=VmecppSolver(path, None, verbose=False))
                b = self.outputs(v)
                self.assertEqual(np.shape(v.wout.rmnc), (v.wout.mnmax, v.wout.ns))
                self.assertEqual(np.shape(v.wout.bmnc), (v.wout.mnmax_nyq, v.wout.ns))
                for key in a:
                    np.testing.assert_allclose(b[key], a[key], rtol=rtol[key], err_msg=key)

    def test_vmecpp_is_independent_of_history(self):
        """ Boundary A, then B, then A again gives the same answer for A. """
        with ScratchDir("."):
            path = os.path.join(TEST_DIR, "input.li383_low_res")
            v = Vmec(path, solver=VmecppSolver(path, None, verbose=False))
            x0 = v.boundary.x.copy()
            first = (v.aspect(), v.iota_axis(), v.volume())
            v.boundary.x = x0 * (1 + 1.0e-3 * np.cos(np.arange(len(x0))))
            v.run()
            v.boundary.x = x0
            second = (v.aspect(), v.iota_axis(), v.volume())
            np.testing.assert_allclose(second, first, rtol=1e-12)

    def test_asymmetric_fixture_converges(self):
        """ input.basic_non_stellsym converges, with the same aspect and volume. """
        with ScratchDir("."):
            path = os.path.join(TEST_DIR, "input.basic_non_stellsym")
            a = Vmec(path, verbose=False)
            b = Vmec(path, solver=VmecppSolver(path, None, verbose=False))
            np.testing.assert_allclose(b.aspect(), a.aspect(), rtol=1e-12)
            np.testing.assert_allclose(b.volume(), a.volume(), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
