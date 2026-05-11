import logging
import unittest

import numpy as np

try:
    import matplotlib
except ImportError:
    matplotlib = None

try:
    from desc.profiles import (
        PowerSeriesProfile as DescPowerSeriesProfile,
        SplineProfile as DescSplineProfile,
    )
except ImportError:
    DescPowerSeriesProfile = None
    DescSplineProfile = None

from simsopt.mhd import (ProfilePolynomial, ProfileScaled, ProfileSpline,
                         ProfilePressure, ProfileSpec)

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)


class ProfilesTests(unittest.TestCase):
    def test_spec_profile(self):
        """
        Test initialization and modification of a spec profile
        """

        prof = ProfileSpec(np.zeros((8,)))

        mvol = 8
        self.assertAlmostEqual(prof.f(0), 0)
        self.assertAlmostEqual(prof.f(mvol - 1), 0)

        # Test that out of range volumes raises errors:
        with self.assertRaises(ValueError):
            prof.f(-1)
        with self.assertRaises(ValueError):
            prof.f(mvol)

        # Test modification of profile in fourth volume
        lvol = 3
        value = 1
        prof.set(lvol, value)
        self.assertEqual(prof.f(lvol), value)

        # Check that other values have not changed:
        self.assertEqual(prof.f(lvol - 1), 0)
        self.assertEqual(prof.f(lvol + 1), 0)

    def test_polynomial(self):
        """
        Test the value and derivative of a polynomial profile
        """
        # Function: f(s) = 3 * (1 - s^3)
        prof = ProfilePolynomial([3, 0, 0, -3])
        s = np.linspace(0, 1, 10)
        np.testing.assert_allclose(prof.f(s), 3 * (1 - s ** 3))
        np.testing.assert_allclose(prof(s), 3 * (1 - s ** 3))
        np.testing.assert_allclose(prof.dfds(s), 3 * (- 3 * s ** 2))

    @unittest.skipIf(matplotlib is None,
                     "Matplotlib python module not found")
    def test_plot(self):
        """
        Test the plotting function.
        """
        # Function: f(s) = 3 * (1 - s^3)
        prof = ProfilePolynomial([3, 0, 0, -3])
        prof.plot(show=False)

    def test_scaled(self):
        """
        Test ProfileScaled
        """
        prof1 = ProfilePolynomial([3, 0, 0, -3])
        scalefac = 0.6
        prof2 = ProfileScaled(prof1, scalefac)
        prof2.local_unfix_all()
        s = np.linspace(0, 1, 10)
        np.testing.assert_allclose(prof2(s), scalefac * prof1(s))
        np.testing.assert_allclose(prof2.dfds(s), scalefac * prof1.dfds(s))
        newscalefac = 1.3
        prof2.x = [newscalefac]
        np.testing.assert_allclose(prof2(s), newscalefac * prof1(s))
        np.testing.assert_allclose(prof2.dfds(s), newscalefac * prof1.dfds(s))

    def test_spline(self):
        """
        Test ProfileSpline.
        """
        s = np.linspace(0, 1, 5)
        s_fine = np.linspace(0, 1, 100)
        f = 1.2 + 0.7 * s + 0.3 * s * s
        f_fine = 1.2 + 0.7 * s_fine + 0.3 * s_fine * s_fine

        profile = ProfileSpline(s, f)
        np.testing.assert_allclose(profile(s), f)
        np.testing.assert_allclose(profile.dfds(s), 0.7 + 2 * 0.3 * s)
        np.testing.assert_allclose(profile(s_fine), f_fine)
        np.testing.assert_allclose(
            profile.dfds(s_fine),
            0.7 + 2 * 0.3 * s_fine)

        # Try changing the dofs
        f2 = -2.1 - 0.4 * s + 0.7 * s * s
        f2_fine = -2.1 - 0.4 * s_fine + 0.7 * s_fine * s_fine
        profile.local_unfix_all()
        profile.x = f2
        np.testing.assert_allclose(profile(s), f2)
        np.testing.assert_allclose(profile.dfds(s), -0.4 + 2 * 0.7 * s)
        np.testing.assert_allclose(profile(s_fine), f2_fine)
        np.testing.assert_allclose(
            profile.dfds(s_fine), -0.4 + 2 * 0.7 * s_fine)

        profile2 = profile.resample(s_fine)
        self.assertEqual(profile2.degree, profile.degree)
        np.testing.assert_allclose(profile2.full_x, f2_fine)
        np.testing.assert_allclose(profile2(s_fine), f2_fine)
        np.testing.assert_allclose(
            profile2.dfds(s_fine), -0.4 + 2 * 0.7 * s_fine)

        profile2 = profile.resample(s_fine, degree=4)
        self.assertEqual(profile2.degree, 4)
        np.testing.assert_allclose(profile2.full_x, f2_fine)
        np.testing.assert_allclose(profile2(s_fine), f2_fine)
        np.testing.assert_allclose(
            profile2.dfds(s_fine), -0.4 + 2 * 0.7 * s_fine)

        # Try a case with extrapolation
        s = np.linspace(0.1, 0.9, 2)
        f = 0.3 - 0.5 * s
        f_fine = 0.3 - 0.5 * s_fine
        profile3 = ProfileSpline(s, f, degree=1)
        np.testing.assert_allclose(profile3(s), f)
        np.testing.assert_allclose(profile3.dfds(s), -0.5 * np.ones_like(s))
        np.testing.assert_allclose(profile3(s_fine), f_fine)
        np.testing.assert_allclose(
            profile3.dfds(s_fine), -0.5 * np.ones_like(s_fine))

        profile4 = profile3.resample(s_fine)
        self.assertEqual(profile4.degree, 1)
        np.testing.assert_allclose(profile4.full_x, f_fine)
        np.testing.assert_allclose(profile4(s_fine), f_fine)
        np.testing.assert_allclose(
            profile4.dfds(s_fine), -0.5 * np.ones_like(s_fine))

    def test_pressure(self):
        """
        Try typical usage of ProfilePressure.
        """
        # Try a case with 2 species:
        ne = ProfilePolynomial(1.0e20 * np.array([1.0, 0.0, 0.0, 0.0, -1.0]))
        Te = ProfilePolynomial(8.0e3 * np.array([1.0, -1.0]))
        nH = ne
        TH = ProfilePolynomial(7.0e3 * np.array([1.0, -1.0]))
        pressure = ProfilePressure(ne, Te, nH, TH)

        s = np.linspace(0, 1, 20)
        atol = 1e-14
        np.testing.assert_allclose(
            pressure(s), ne(s) * (Te(s) + TH(s)), atol=atol)
        np.testing.assert_allclose(pressure.dfds(s),
                                   ne.dfds(s) * (Te(s) + TH(s))
                                   + ne(s) * (Te.dfds(s) + TH.dfds(s)),
                                   atol=atol)

        # Now try a case with 3 species:
        nD = ProfileScaled(ne, 0.55)
        nT = ProfileScaled(ne, 0.45)
        TD = ProfilePolynomial(12.0e3 * np.array([1.0, -1.0]))
        TT = TD
        pressure = ProfilePressure(ne, Te, nD, TD, nT, TT)
        ne.local_unfix_all()
        nD.local_unfix_all()
        nT.local_unfix_all()
        Te.local_unfix_all()
        TD.local_unfix_all()
        TT.local_unfix_all()
        for j in range(2):
            np.testing.assert_allclose(
                pressure(s),
                ne(s) * Te(s) + nD(s) * TD(s) + nT(s) * TT(s),
                atol=atol)
            np.testing.assert_allclose(
                pressure.dfds(s),
                ne.dfds(s) * Te(s) + nT.dfds(s) * TT(s) + nD.dfds(s) * TD(s) +
                ne(s) * Te.dfds(s) + nT(s) * TT.dfds(s) + nD(s) * TD.dfds(s),
                atol=atol)
            # Try changing some dofs before the 2nd loop iteration:
            ne.x = 2.0e20 * np.array([1.0, 0.0, 0.0, -1.0, 0.0])
            nD.local_x = [0.51]
            nT.local_x = [0.49]
            Te.x = 15.0e3 * np.array([1.0, -0.9])
            TD.x = 14.0e3 * np.array([1.0, -0.3])

    def test_pressure_exception(self):
        """
        ProfilePressure should raise an exception if an odd number of
        profiles are provided.
        """
        ne = ProfilePolynomial(1.0e20 * np.array([1.0, 0.0, 0.0, 0.0, -1.0]))
        Te = ProfilePolynomial(8.0e3 * np.array([1.0, -1.0]))
        nD = ProfileScaled(ne, 0.55)
        nT = ProfileScaled(ne, 0.45)
        TD = ProfilePolynomial(12.0e3 * np.array([1.0, -1.0]))
        # Try zero profiles:
        with self.assertRaises(ValueError):
            ProfilePressure()
        with self.assertRaises(ValueError):
            ProfilePressure(ne)
        with self.assertRaises(ValueError):
            ProfilePressure(ne, Te, nD)
        with self.assertRaises(ValueError):
            ProfilePressure(ne, Te, nD, TD, nT)


@unittest.skipUnless(DescPowerSeriesProfile is not None, "DESC not installed")
class TestProfilePolynomialDesc(unittest.TestCase):
    """Tests for ProfilePolynomial <-> DESC PowerSeriesProfile conversion."""

    def test_to_desc_sym(self):
        """to_desc produces a DESC profile with matching values (sym=True)."""
        # f(s) = 1 + 0.5*s - 0.3*s^2
        coeffs = np.array([1.0, 0.5, -0.3])
        prof = ProfilePolynomial(coeffs)
        prof_desc = prof.to_desc()

        # DESC uses rho; simsopt uses s = rho^2
        rho = np.linspace(0, 1, 11)
        np.testing.assert_allclose(prof(rho**2), prof_desc(rho), rtol=1e-12)

    def test_from_desc_sym(self):
        """from_desc with sym=True roundtrips coefficients exactly."""
        params = np.array([1.0, 0.5, -0.3])
        prof_desc = DescPowerSeriesProfile(params=params, sym=True)
        prof = ProfilePolynomial.from_desc(prof_desc)

        self.assertIsInstance(prof, ProfilePolynomial)
        np.testing.assert_array_equal(prof.local_full_x, params)

        rho = np.linspace(0, 1, 11)
        np.testing.assert_allclose(prof(rho**2), prof_desc(rho), rtol=1e-12)

    def test_from_desc_no_sym(self):
        """from_desc with sym=False drops odd-power terms."""
        # f(rho) = 1 + 0.5*rho - 0.3*rho^2 + 0.1*rho^3
        # Even terms: a0=1, a2=-0.3 -> ProfilePolynomial([1, -0.3])
        params = np.array([1.0, 0.5, -0.3, 0.1])
        prof_desc = DescPowerSeriesProfile(params=params, sym=False)
        prof = ProfilePolynomial.from_desc(prof_desc)

        np.testing.assert_array_equal(prof.local_full_x, params[::2])

    def test_roundtrip(self):
        """simsopt -> DESC -> simsopt preserves values."""
        coeffs = np.array([1.0, 0.5, -0.3, 0.0, 0.2])
        prof_orig = ProfilePolynomial(coeffs)
        prof_rt = ProfilePolynomial.from_desc(prof_orig.to_desc())

        s = np.linspace(0, 1, 20)
        np.testing.assert_allclose(prof_orig(s), prof_rt(s), rtol=1e-12)


@unittest.skipUnless(DescSplineProfile is not None, "DESC not installed")
class TestProfileSplineDesc(unittest.TestCase):
    """Tests for ProfileSpline <-> DESC SplineProfile conversion."""

    def test_to_desc(self):
        """to_desc maps s-space knots to rho-space and preserves values."""
        s_knots = np.linspace(0, 1, 6)
        f_values = 1.0 - s_knots
        prof = ProfileSpline(s_knots, f_values, degree=3)
        prof_desc = prof.to_desc()

        rho = np.linspace(0, 1, 11)
        np.testing.assert_allclose(prof(rho**2), prof_desc(rho), rtol=1e-6)

    def test_from_desc(self):
        """from_desc maps rho-space knots to s-space and preserves values."""
        # f(rho) = 1 - 1.5*rho^2 + 0.5*rho^4 uses even rho powers only,
        # so f(s) = 1 - 1.5*s + 0.5*s^2 is polynomial — exactly representable
        # by the s-spline. Error only comes from the DESC cubic spline
        # approximating the degree-4 rho term.
        rho_knots = np.linspace(0, 1, 20)
        f_values = 1.0 - 1.5 * rho_knots**2 + 0.5 * rho_knots**4
        prof_desc = DescSplineProfile(values=f_values, knots=rho_knots, method="cubic2")
        prof = ProfileSpline.from_desc(prof_desc, degree=3)

        rho = np.linspace(0, 1, 50)
        np.testing.assert_allclose(prof(rho**2), prof_desc(rho), atol=1e-5)

    def test_roundtrip(self):
        """simsopt -> DESC -> simsopt preserves values."""
        # f(s) = 1 - s maps to 1 - rho^2 in rho-space (degree 2),
        # which a cubic spline represents exactly — making the roundtrip exact.
        s_knots = np.linspace(0, 1, 6)
        f_values = 1.3252 - s_knots
        prof_orig = ProfileSpline(s_knots, f_values, degree=3)
        prof_rt = ProfileSpline.from_desc(prof_orig.to_desc(), degree=3)

        s = np.linspace(0, 1, 20)
        np.testing.assert_allclose(prof_orig(s), prof_rt(s), atol=1e-12)
