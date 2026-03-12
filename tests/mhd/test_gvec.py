import unittest
from pathlib import Path

import numpy as np
from monty.tempfile import ScratchDir

from simsopt._core.optimizable import Optimizable
from simsopt.geo import Surface, SurfaceRZFourier
from simsopt.mhd.profiles import Profile, ProfileScaled, ProfilePolynomial, ProfileSpline
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
        """assert that eq is internally consistent"""
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
        # absolute tolerance of 10kA
        if eq.current_profile is not None:
            # np.testing.assert_allclose(eq.state.evaluate("iota_curr", rho=rho).iota_curr, 0.0)
            I_tor = eq.state.evaluate("I_tor", rho=rho).I_tor
            np.testing.assert_allclose(eq.current_profile(rho**2), I_tor, atol=1e4)
        
        # check consistency: boundary
        if isinstance(eq.boundary, SurfaceRZFourier):
            theta = eq.boundary.quadpoints_theta * 2 * np.pi
            zeta = -eq.boundary.quadpoints_phi * 2 * np.pi
            boundary = eq.state.evaluate("pos", rho=1.0, theta=theta, zeta=zeta)["pos"].squeeze().transpose("tor", "pol", "xyz")
            np.testing.assert_allclose(eq.boundary.gamma(), boundary)
    
    def check_return_functions(self, eq):
        """check that the return functions work and return a float"""
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
    
    def test_run_from_rundir(self):
        with ScratchDir("."):
            eq = Gvec.from_rundir(TEST_DIR / "gvec-W7-X_standard_configuration")
            self.check_Optimizable(eq)
            self.assertFalse(eq.run_required)
            self.assertTrue(eq.run_successful)
            self.check_consistency(eq)
            self.check_return_functions(eq)

            eq.parameters["totalIter"] = 10
            eq.run(force=True)
            self.assertFalse(eq.run_required)
            self.assertTrue(eq.run_successful)
    
    def test_set_pressure_profile(self):
        s_spline = np.linspace(0, 1, 5)
        profiles = [
            ProfilePolynomial(1.0e2 * np.array([1, 1, -2.0])),
            ProfileScaled(ProfilePolynomial([1, 1, -2.0]), 1.5e2),
            ProfileSpline(s_spline, 1.0e2 * (2.0 + 0.6 * s_spline - 1.5 * s_spline ** 2)),
        ]
        pressure_on_axis = [1.0e2, 1.5e2, 2.0e2]
        profile_types = ["polynomial", "polynomial", "interpolation"]

        for profile, p0, profile_type in zip(profiles, pressure_on_axis, profile_types):
            with ScratchDir("."), self.subTest(profile=profile):
                eq = Gvec.from_parameter_file(TEST_DIR / "parameter-LandremanPaul2021_QA_lowres.gvec.toml")
                eq.parameters["totalIter"] = 10
                eq.pressure_profile = profile
                self.assertEqual(eq.pressure_profile, profile)
                self.assertTrue(eq.run_required)
                eq.run()
                self.assertTrue(eq.run_successful)
                self.check_consistency(eq)
                self.check_return_functions(eq)
                self.assertEqual(eq.state.parameters["pres"]["type"], profile_type)
                p_axis = eq.state.evaluate("p", rho=0.0).p.item()
                self.assertAlmostEqual(p_axis, p0)
    
    def test_set_iota_profile(self):
        s_spline = np.linspace(0, 1, 5)
        profiles = [
            ProfilePolynomial(np.array([1, 1, -2.0])),
            ProfileSpline(s_spline, (2.0 + 0.6 * s_spline - 1.5 * s_spline ** 2)),
        ]
        profile_types = ["polynomial", "interpolation"]

        for profile, profile_type in zip(profiles, profile_types):
            with ScratchDir("."), self.subTest(profile=profile):
                eq = Gvec.from_parameter_file(TEST_DIR / "parameter-LandremanPaul2021_QA_lowres.gvec.toml")
                eq.iota_profile = profile
                eq.current_profile = None  # -> fixed iota
                self.assertEqual(eq.iota_profile, profile)
                self.assertTrue(eq.run_required)
                eq.run()
                self.assertTrue(eq.run_successful)
                self.check_consistency(eq)
                self.check_return_functions(eq)
                self.assertEqual(eq.state.parameters["iota"]["type"], profile_type)
                self.assertAlmostEqual(eq.iota_axis(), eq.iota_profile(0.0))
                self.assertAlmostEqual(eq.iota_edge(), eq.iota_profile(1.0))

    def test_set_current_profile(self):
        s_spline = np.linspace(0, 1, 5)
        factor = 1.0e4  # 10kA
        profiles = [
            ProfilePolynomial(factor * np.array([0, 1.1, -0.1])),
            ProfileScaled(ProfilePolynomial(np.array([0, 1.1, -0.1])), factor),
            ProfileSpline(s_spline, factor * (1.1 * s_spline - 0.1 * s_spline ** 2)),
            ProfileScaled(ProfileSpline(s_spline, 1.1 * s_spline - 0.1 * s_spline ** 2), factor),
        ]
        profile_types = 2 * ["polynomial"] + 2 * ["interpolation"]

        for profile, profile_type in zip(profiles, profile_types):
            with ScratchDir("."), self.subTest(profile=profile):
                eq = Gvec.from_parameter_file(TEST_DIR / "parameter-LandremanPaul2021_QA_lowres.gvec.toml")
                eq.parameters["minimize_tol"] = 1e-3
                eq.current_profile = profile
                eq.iota_profile = None  # -> no initial guess
                self.assertEqual(eq.current_profile, profile)
                self.assertTrue(eq.run_required)
                eq.run()
                self.assertTrue(eq.run_successful)
                self.check_consistency(eq)
                self.check_return_functions(eq)
                rho = np.array([0.25, 0.5, 0.75, 1.0])
                Itor = eq.state.evaluate("I_tor", rho=rho).I_tor
                ref = [eq.current_profile(r**2) for r in rho]
                np.testing.assert_allclose(Itor, ref, atol=1e4)
