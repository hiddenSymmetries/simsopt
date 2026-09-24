"""
Compare the VMEC2000 and VMEC++ backends on the same input files.

Known differences are listed explicitly with their cause. Entries that
record a VMEC++ limitation are asserted to still fail, so that fixing
the limitation upstream makes the test point at the stale entry.
"""

import glob
import os
import subprocess
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

from simsopt.mhd.vmec import Vmec

from . import TEST_DIR

#: INDATA files VMEC++ cannot read, with the reason.
VMECPP_UNREADABLE = {
    # indata2json's namelist declares bcrit/at/ah but not pt_type/ph_type:
    "input.LandremanPaul2021_QA_reactorScale_lowres": "ANI/FLOW block",
    "input.LandremanPaul2021_QH_reactorScale_lowres": "ANI/FLOW block",
    "input.LandremanSengupta2019_section5.4_B2_A80": "ANI/FLOW block",
    "input.LandremanSenguptaPlunk_section5p3": "ANI/FLOW block",
    # VMEC2000 accepts ac_aux_f and ac_aux_s of different lengths:
    "input.20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs":
        "aux array length mismatch",
}


#: Tolerance on the relative difference between the backends, per
#: output, set from the measured differences with some headroom. VMEC++
#: is a reimplementation, so results agree to convergence tolerance,
#: not to round-off.
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
        """
        Every INDATA fixture reads into the same Vmec state under both
        backends, apart from the documented exceptions.
        """
        for path in indata_files():
            name = os.path.basename(path)
            with self.subTest(name=name):
                if name in VMECPP_UNREADABLE:
                    with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                        VmecppSolver(path, None, verbose=False)
                    continue
                a = Vmec(path, verbose=False)
                b = Vmec(path, verbose=False, solver=VmecppSolver)
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
                    # VMEC2000's readin applies its theta shift to the m = 1
                    # modes of an asymmetric boundary, VMEC++ keeps the file's
                    # parametrization. The dofs differ, the surface does not.
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


@unittest.skipIf(vmec_mod is None or VmecppSolver is None,
                 "both the vmec python extension and vmecpp are needed")
class VmecBackendResultTests(unittest.TestCase):
    def outputs(self, v):
        return {name: getattr(v, name)() for name in RTOL}

    def test_outputs_agree(self):
        """
        Measured for input.li383_low_res / LandremanPaul2021_QA_lowres:
        aspect and volume agree to ~1e-15, iota to 2e-5 / 1.2e-4 at the
        axis, vacuum_well to 1.7e-5 / 1.0e-4, external_current to
        8e-7 / 1e-8.
        """
        rtol = RTOL
        for name in ["input.li383_low_res", "input.LandremanPaul2021_QA_lowres"]:
            with self.subTest(name=name), ScratchDir("."):
                path = os.path.join(TEST_DIR, name)
                a = self.outputs(Vmec(path, verbose=False))
                v = Vmec(path, verbose=False, solver=VmecppSolver)
                b = self.outputs(v)
                self.assertEqual(np.shape(v.wout.rmnc), (v.wout.mnmax, v.wout.ns))
                self.assertEqual(np.shape(v.wout.bmnc), (v.wout.mnmax_nyq, v.wout.ns))
                for key in a:
                    np.testing.assert_allclose(b[key], a[key], rtol=rtol[key], err_msg=key)

    def test_vmecpp_is_independent_of_history(self):
        """ Boundary A, then B, then A again gives the same answer for A. """
        with ScratchDir("."):
            v = Vmec(os.path.join(TEST_DIR, "input.li383_low_res"), verbose=False,
                     solver=VmecppSolver)
            x0 = v.boundary.x.copy()
            first = (v.aspect(), v.iota_axis(), v.volume())
            v.boundary.x = x0 * (1 + 1.0e-3 * np.cos(np.arange(len(x0))))
            v.run()
            v.boundary.x = x0
            second = (v.aspect(), v.iota_axis(), v.volume())
            np.testing.assert_allclose(second, first, rtol=1e-12)

    def test_asymmetric_fixture_converges(self):
        """
        input.basic_non_stellsym, the only non-stellarator-symmetric
        equilibrium in the suite, converges on VMEC++ >= 0.7.4 with the
        same aspect ratio and volume as on VMEC2000.
        """
        with ScratchDir("."):
            path = os.path.join(TEST_DIR, "input.basic_non_stellsym")
            a = Vmec(path, verbose=False)
            b = Vmec(path, verbose=False, solver=VmecppSolver)
            np.testing.assert_allclose(b.aspect(), a.aspect(), rtol=1e-12)
            np.testing.assert_allclose(b.volume(), a.volume(), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
