import unittest
from pathlib import Path

import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt.geo import Surface, SurfaceRZFourier
from simsopt.mhd.profiles import Profile, ProfileScaled, ProfilePolynomial
from simsopt.mhd.gvec import Gvec, GVECSurfaceDoFs

try:
    import gvec
except ImportError:
    gvec = None

TEST_DIR = Path(__file__).parent / ".." / "test_files"

@unittest.skipIf(gvec is None, "gvec package not installed")
class GvecTests(unittest.TestCase):

    # === helpers === #

    def check_Optimizable(self, eq):
        """assert that eq is a properly configured optimizable"""
        self.assertIsInstance(eq, Gvec)
        self.assertIsInstance(eq, Optimizable)

        self.assertListEqual(eq.local_full_dof_names, ["phiedge"])
        self.assertEqual(eq.local_full_x.size, 1)

        self.assertEqual(len(eq.parents), 3)
        self.assertIn(eq.boundary, eq.parents)
        self.assertIn(eq.pressure_profile, eq.parents)
        if eq.current_profile is None:
            self.assertIn(eq.iota_profile, eq.parents)
        else:
            self.assertIn(eq.current_profile, eq.parents)
            # with prescribed current, iota_profile is only used as an initial condition
            # with only 3 parents, we don't need to check explicitly

        self.assertIsInstance(eq.boundary, (Surface, GVECSurfaceDoFs))
        self.assertIsInstance(eq.pressure_profile, Profile)
        if eq.iota_profile is not None:
            self.assertIsInstance(eq.iota_profile, Profile)
        if eq.current_profile is not None:
            self.assertIsInstance(eq.current_profile, Profile)
    
    def check_consistency(self, eq):
        # check consistency: phiedge
        #   In GVEC the input 'phiedge' is the flux through the cross-section,
        #   but the 'Phi' profile refers to the component of the vector potential.
        #   These are related by a factor 2π.
        phiedge = eq.state.evaluate("Phi", rho=1.0)["Phi"].item() * 2 * np.pi
        self.assertEqual(eq.phiedge, phiedge)
        
        # check consistency: pressure profile
        rho = np.linspace(0, 1, 11)
        pressure = eq.state.evaluate("p", rho=rho).p
        np.testing.assert_allclose(eq.pressure_profile(rho**2), pressure)

        # check consistency: rotational transform profile (iota)
        if eq.current_profile is None:
            iota = eq.state.evaluate("iota", rho=rho).iota
            np.testing.assert_allclose(eq.iota_profile(rho**2), iota)

        # check consistency: current profile
        # absolute tolerance of 1kA
        if eq.current_profile is not None:
            I_tor = eq.state.evaluate("I_tor", rho=rho).I_tor
            np.testing.assert_allclose(eq.current_profile(rho**2), I_tor, atol=1e3)
        
        # check consistency: boundary
        if isinstance(eq.boundary, SurfaceRZFourier):
            theta = eq.boundary.quadpoints_theta * 2 * np.pi
            zeta = -eq.boundary.quadpoints_phi * 2 * np.pi
            boundary = eq.state.evaluate("pos", rho=1.0, theta=theta, zeta=zeta)["pos"].squeeze().transpose("tor", "pol", "xyz")
            np.testing.assert_allclose(eq.boundary.gamma(), boundary)
    
    def check_return_functions(self, eq):
        # check that the return functions work and return a float
        run_count0 = eq.run_count
        self.assertFalse(eq.run_required)

        self.assertIsInstance(eq.aspect(), float)
        self.assertIsInstance(eq.volume(), float)
        self.assertIsInstance(eq.iota_axis(), float)
        self.assertIsInstance(eq.iota_edge(), float)
        self.assertIsInstance(eq.mean_iota(), float)
        self.assertIsInstance(eq.mean_shear(), float)
        self.assertIsInstance(eq.vacuum_well(), float)

        # no new run should have been triggered
        self.assertEqual(eq.run_count, run_count0)

    # === tests === #

    def test_init_defaults(self):
        eq = Gvec()
        self.check_Optimizable(eq)
        self.assertTrue(eq.run_required)

        self.assertIsInstance(eq.boundary, SurfaceRZFourier)
        reference_surf = SurfaceRZFourier()
        self.assertEqual(eq.boundary.nfp, reference_surf.nfp)
        self.assertEqual(eq.boundary.stellsym, reference_surf.stellsym)
        self.assertEqual(eq.boundary.mpol, reference_surf.mpol)
        self.assertEqual(eq.boundary.ntor, reference_surf.ntor)
        np.testing.assert_allclose(eq.boundary.x, reference_surf.x)
        
        self.assertIsInstance(eq.pressure_profile, ProfilePolynomial)
        np.testing.assert_equal(eq.pressure_profile.local_full_x, [0.0])

        self.assertIsInstance(eq.current_profile, ProfilePolynomial)
        np.testing.assert_equal(eq.current_profile.local_full_x, [0.0])

        self.assertEqual(eq.phiedge, 1.0)

    def test_init_from_parameter_file(self):
        eq = Gvec.from_parameter_file(TEST_DIR / "parameter-LandremanPaul2021_QA.gvec.toml")
        self.check_Optimizable(eq)
        self.assertTrue(eq.run_required)

        self.assertIsInstance(eq.boundary, SurfaceRZFourier)
        self.assertEqual(eq.boundary.nfp, 2)
        self.assertEqual(eq.boundary.stellsym, True)
        self.assertEqual(eq.boundary.mpol, 15)
        self.assertEqual(eq.boundary.ntor, 12)
        self.assertAlmostEqual(eq.boundary.get("rc(2,-4)"), 4.850684989433037e-05)

        self.assertIsInstance(eq.pressure_profile, ProfileScaled)
        self.assertIsInstance(eq.pressure_profile.base, ProfilePolynomial)
        np.testing.assert_equal(eq.pressure_profile.base.local_full_x, [0.0])

        self.assertIsInstance(eq.current_profile, ProfileScaled)
        self.assertIsInstance(eq.current_profile.base, ProfilePolynomial)
        np.testing.assert_equal(eq.current_profile.base.local_full_x, [0.0])

        self.assertEqual(eq.phiedge, -0.08385727554)

        self.assertEqual(eq.parameters["sgrid"]["nElems"], 5)
    
    def test_init_from_rundir(self):
        eq = Gvec.from_rundir(TEST_DIR / "gvec-W7-X_standard_configuration")
        self.check_Optimizable(eq)
        self.assertFalse(eq.run_required)
        self.assertTrue(eq.run_successful)
        self.check_consistency(eq)
        self.check_return_functions(eq)

        self.assertIsInstance(eq.boundary, SurfaceRZFourier)
        self.assertEqual(eq.boundary.nfp, 5)
        self.assertEqual(eq.boundary.stellsym, True)
        self.assertEqual(eq.boundary.mpol, 11)
        self.assertEqual(eq.boundary.ntor, 12)
        self.assertAlmostEqual(eq.boundary.get("rc(2,-4)"), -0.000133285510379407)

        self.assertIsInstance(eq.pressure_profile, ProfileScaled)
        self.assertAlmostEqual(eq.pressure_profile.local_full_x[0], 1.0)
        self.assertIsInstance(eq.pressure_profile.base, ProfilePolynomial)
        np.testing.assert_equal(eq.pressure_profile.base.local_full_x, [1e-6, -1e-6])

        self.assertIsInstance(eq.current_profile, ProfileScaled)
        self.assertIsInstance(eq.current_profile.base, ProfilePolynomial)
        np.testing.assert_equal(eq.current_profile.base.local_full_x, [0.0])

        self.assertEqual(eq.phiedge, 2.1907427)

        self.assertEqual(eq.parameters["sgrid"]["nElems"], 5)
        self.assertEqual(eq.parameters["X1X2_deg"], 5)
        self.assertEqual(eq.parameters["LA_deg"], 5)