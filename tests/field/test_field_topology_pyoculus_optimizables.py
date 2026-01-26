import unittest
import json

import numpy as np

from simsopt._core.dev import SimsoptRequires
from simsopt.configs import get_data
from simsopt.field import (PyOculusFixedPoint, 
                           PyOculusFixedPointLocationTarget,
                           PyOculusEllipticityTarget,
                           PyOculusResidueTarget,
                           PyOculusTraceTarget,
                           PyOculusTraceInRangeTarget,
                           PyOculusLocationAtOtherAngleTarget,
                           PyOculusTwoFixedPointLocationDifference,
                           ClinicConnection)

from simsopt._core import ObjectiveFailure
from simsopt.field import CircularCoil


try :
    from pyoculus.solvers import FixedPoint, Manifold
    from pyoculus.fields import SimsoptBfield
    from pyoculus.maps import CylindricalBfieldSection
    from pyoculus import __version__ as pyoc_version
except ImportError:
    pyoc_version = -1
    pyoculus = None


try:
    newpyoculus = int(str(pyoc_version).split('.')[0]) >= 1
except Exception:
    newpyoculus = False


@unittest.skipUnless(newpyoculus, "pyOculus>=1.0.0 is required for this test")
class TestPyOculusFixedPoints(unittest.TestCase):
    attributes = ['loc_RZ', 'loc_xyz', 'R', 'Z', 'x_coord', 'y_coord', 'z_coord', 'residue', 'jacobian', 'trace']

    def setUp(self):
        base_curves, base_currents, ma, nfp, bs = get_data('w7x', coil_order=12, points_per_period=4)
        self.base_curves = base_curves
        self.base_currents = base_currents
        self.ma = ma
        self.nfp = nfp
        self.bs = bs
        self.axis_guess = self.ma.gamma()[0][::2] # R, Z (actually x, z, but axis starts at phi=0)
        self.integration_args = {'rtol':1e-8, 'atol':1e-8}
        # create the pyoculus interface to the simsopt field
        self.pyoculus_field_interface = SimsoptBfield(nfp, bs)  
        # create the pyoculus map
        self.pyoculus_map = CylindricalBfieldSection(self.pyoculus_field_interface, 
                                                     R0=self.axis_guess[0],
                                                     Z0=self.axis_guess[1],
                                                     phi0=0.,
                                                     **self.integration_args)

    def test_init_on_axis(self):
        """
        Test that the axis is a 1 field period fixed point
        """
        fp1 = PyOculusFixedPoint(self.pyoculus_map, # needs a map
                        start_guess=self.axis_guess, 
                        fp_order=1) 
        self.assertIsInstance(fp1, PyOculusFixedPoint)
        fp2 = PyOculusFixedPoint(self.pyoculus_map, # needs a map
                        start_guess=self.axis_guess, 
                        fp_order=1,
                        with_axis=True) 
        self.assertIsInstance(fp2, PyOculusFixedPoint)
        fp3 = PyOculusFixedPoint.from_field(self.bs, 
                                            self.nfp, 
                                            start_guess=self.axis_guess,
                                            fp_order=1,
                                            phi0=0,
                                            axis=False,
                                            integration_args=self.integration_args)
        self.assertIsInstance(fp3, PyOculusFixedPoint)

        for attribute_name in self.attributes:
            val1 = getattr(fp1, attribute_name)
            val2 = getattr(fp2, attribute_name)
            val3 = getattr(fp3, attribute_name)
            # If the attribute is a callable (method), call it to get the returned value
            if callable(val1):
                val1 = val1()
            if callable(val2):
                val2 = val2()
            if callable(val3):
                val3 = val3()
            print(attribute_name, val1, val2, val3)
            np.testing.assert_allclose(val1, val2, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(val1, val3, rtol=1e-7, atol=1e-7)

    def test_dofchange_invalidates_triggers_recompute(self):
        """
        Test that changing a degree of freedom in the coils
        triggers recomputation, and that accessing any attribute 
        triggers recomputation. 
        """
        # Create a fixed point
        fp = PyOculusFixedPoint.from_field(self.bs, 
                                           self.nfp, 
                                           start_guess=self.axis_guess,
                                           fp_order=1,
                                           phi0=0,
                                           axis=False,
                                           integration_args=self.integration_args)
        
        # Get initial values
        initial_loc = fp.loc_RZ().copy()
        initial_residue = fp.residue()
        initial_trace = fp.trace()
        
        # Check that _refind_fp is False (no recomputation needed)
        self.assertFalse(fp._refind_fp)
        
        # Modify one of the base currents by a miniscule amount
        original_current = self.base_currents[0].x.copy()
        self.base_currents[0].x = original_current * 1.0001  # tiny change
        
        # Check that _refind_fp is now True (recomputation triggered)
        self.assertTrue(fp._refind_fp)
        
        # Access an attribute to trigger recomputation
        new_loc = fp.loc_RZ()
        
        # Check that _refind_fp is now False (recomputation occurred)
        self.assertFalse(fp._refind_fp)
        
        # Verify that the location changed slightly (due to field change)
        # but should still be close since the change is small
        np.testing.assert_allclose(initial_loc, new_loc, rtol=1e-3, atol=1e-4)
        
        # Restore original current
        self.base_currents[0].x = original_current
        
        # Verify recompute bell was rung again
        self.assertTrue(fp._refind_fp)
        
        # Access residue to trigger recomputation 
        new_residue = fp.residue()
        self.assertFalse(fp._refind_fp)
        
        # Values should be close to original after restoring the current
        np.testing.assert_allclose(initial_residue, new_residue, rtol=1e-4, atol=1e-7)

    def test_map_and_fixed_point_properties_readonly(self):
        """
        Test that the map and fixed_point properties are read-only.
        """
        fp = PyOculusFixedPoint(self.pyoculus_map, 
                                start_guess=self.axis_guess, 
                                fp_order=1)
        
        # Test that map property returns the pyoculus map
        self.assertEqual(fp.map, fp._pyocmap)
        
        # Test that map property is read-only
        with self.assertRaises(ValueError):
            fp.map = None
        
        # Test that fixed_point property returns the fixed point object
        self.assertEqual(fp.fixed_point, fp._fixed_point)
        
        # Test that fixed_point property is read-only
        with self.assertRaises(ValueError):
            fp.fixed_point = None

    def test_loc_at_other_angle(self):
        """
        Test the loc_at_other_angle method.
        """
        fp = PyOculusFixedPoint.from_field(self.bs, 
                                           self.nfp, 
                                           start_guess=self.axis_guess,
                                           fp_order=1,
                                           phi0=0,
                                           axis=False,
                                           integration_args=self.integration_args)
        
        # Get location at phi=0
        loc_phi0 = fp.loc_RZ()
        
        # Get location at a small angle
        phi_test = 0.1
        loc_other = fp.loc_at_other_angle(phi_test)
        
        # Location should be an array of length 2
        self.assertEqual(len(loc_other), 2)
        
        # R should still be positive
        self.assertGreater(loc_other[0], 0)

    def test_ellipticity(self):
        """
        Test the ellipticity method.
        """
        fp = PyOculusFixedPoint.from_field(self.bs, 
                                           self.nfp, 
                                           start_guess=self.axis_guess,
                                           fp_order=1,
                                           phi0=0,
                                           axis=False,
                                           integration_args=self.integration_args)
        
        # Get ellipticity
        ell = fp.ellipticity()
        
        # Ellipticity should be a scalar
        self.assertIsInstance(ell, (float, np.floating))
        
        # For an elliptic fixed point (like the axis), ellipticity should be between -1 and 1
        self.assertGreaterEqual(ell, -1)
        self.assertLessEqual(ell, 1)

    def test_from_field_with_axis(self):
        """
        Test PyOculusFixedPoint.from_field with axis parameter set to array.
        When axis=True is passed but no explicit guess, pyoculus needs a good guess.
        This test verifies that passing the actual axis location works.
        """
        # Pass a good axis guess (the actual axis location)
        fp = PyOculusFixedPoint.from_field(self.bs, 
                                           self.nfp, 
                                           start_guess=self.axis_guess,
                                           fp_order=1,
                                           phi0=0,
                                           axis=self.axis_guess,  # Use good axis guess
                                           integration_args=self.integration_args)
        self.assertIsInstance(fp, PyOculusFixedPoint)
        self.assertTrue(fp._with_axis)

    def test_from_field_with_axis_array(self):
        """
        Test PyOculusFixedPoint.from_field with axis parameter as an array.
        """
        axis_location = np.array([self.axis_guess[0], self.axis_guess[1]])
        fp = PyOculusFixedPoint.from_field(self.bs, 
                                           self.nfp, 
                                           start_guess=self.axis_guess,
                                           fp_order=1,
                                           phi0=0,
                                           axis=axis_location,
                                           integration_args=self.integration_args)
        self.assertIsInstance(fp, PyOculusFixedPoint)
        self.assertTrue(fp._with_axis)
    
    def test_island_fixed_point(self):
        """
        test to find the o-point of the island chain in the standard
        configuration also works
        """
        # Create a fixed point for the island chain
        fp_island = PyOculusFixedPoint.from_field(self.bs, 
                                                  self.nfp, 
                                                  start_guess=[6.2345, 0.0],
                                                  fp_order=5,
                                                  phi0=0,
                                                  axis=False,
                                                  integration_args=self.integration_args)
        self.assertIsInstance(fp_island, PyOculusFixedPoint)
        self.assertFalse(np.allclose(fp_island._fixed_point.coords[0], fp_island._fixed_point.coords[1]))

    def test_whacky_field_raises_objectivefailure(self):
        """
        Test that a whacky field configuration raises an ObjectiveFailure
        when trying to find a fixed point.
        """
        # Modify the field by adding a circular coil that makes most field lines not pass around
        
        fp = PyOculusFixedPoint.from_field(self.bs, 
                                            self.nfp, 
                                            start_guess=[6.2345, 0.0], 
                                            fp_order=5,
                                            phi0=0,
                                            axis=False,
                                            integration_args=self.integration_args)
        # set the currents to lowiota so that 5fp fixed point search fails
        start_currents = [curr.get_value() for curr in self.base_currents]
        coil_currents_high_iota = np.array([1, 1, 1, 1, 1, -0.23, -0.23, 0])

        for current, highvalue in zip(self.base_currents, coil_currents_high_iota):
            print(highvalue)
            current.x = np.atleast_1d(highvalue)

        with self.assertRaises(ObjectiveFailure):
            fp.loc_RZ()
        
        for i, current in enumerate(self.base_currents):
            current.x = start_currents[i]  # Restore original current values    

@unittest.skipUnless(newpyoculus, "pyOculus>=1.0.0 is required for this test")
class TestPyOculusTargets(unittest.TestCase):
    """
    Test class for all PyOculus*Target classes.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up the field and fixed point once for all tests.
        """
        base_curves, base_currents, ma, nfp, bs = get_data('w7x', coil_order=12, points_per_period=4)
        cls.base_curves = base_curves
        cls.base_currents = base_currents
        cls.ma = ma
        cls.nfp = nfp
        cls.bs = bs
        cls.axis_guess = ma.gamma()[0][::2]
        cls.integration_args = {'rtol':1e-8, 'atol':1e-8}
        
        # Create a fixed point for testing
        cls.fp = PyOculusFixedPoint.from_field(bs, 
                                               nfp, 
                                               start_guess=cls.axis_guess,
                                               fp_order=1,
                                               phi0=0,
                                               axis=False,
                                               integration_args=cls.integration_args)

    def test_location_target(self):
        """
        Test PyOculusFixedPointLocationTarget.
        """
        # Target is the current location
        target_RZ = self.fp.loc_RZ().copy()
        loc_target = PyOculusFixedPointLocationTarget(self.fp, target_RZ)
        
        # J should be ~0 when target equals current location
        J_val = loc_target.J()
        self.assertAlmostEqual(J_val, 0.0, places=5)
        
        # Test with a different target
        different_target = target_RZ + np.array([0.1, 0.05])
        loc_target2 = PyOculusFixedPointLocationTarget(self.fp, different_target)
        J_val2 = loc_target2.J()
        
        # J should be the distance between the fp location and target
        expected_J = np.linalg.norm(self.fp.loc_RZ() - different_target)
        np.testing.assert_allclose(J_val2, expected_J, rtol=1e-10)

    def test_ellipticity_target(self):
        """
        Test PyOculusEllipticityTarget.
        """
        current_ellipticity = self.fp.ellipticity()
        
        # Test with target equal to current ellipticity
        ell_target = PyOculusEllipticityTarget(self.fp, target_ellipticity=current_ellipticity)
        J_val = ell_target.J()
        self.assertAlmostEqual(J_val, 0.0, places=7)
        
        # Test with a different target
        ell_target2 = PyOculusEllipticityTarget(self.fp, target_ellipticity=0.5)
        J_val2 = ell_target2.J()
        expected_J = current_ellipticity - 0.5
        np.testing.assert_allclose(J_val2, expected_J, rtol=1e-10)
        
        # Test default target (0.0)
        ell_target3 = PyOculusEllipticityTarget(self.fp)
        J_val3 = ell_target3.J()
        self.assertEqual(J_val3, current_ellipticity)

    def test_residue_target(self):
        """
        Test PyOculusResidueTarget.
        """
        current_residue = self.fp.residue()
        
        # Test with target equal to current residue
        res_target = PyOculusResidueTarget(self.fp, target_residue=current_residue)
        J_val = res_target.J()
        self.assertAlmostEqual(J_val, 0.0, places=7)
        
        # Test with a different target
        res_target2 = PyOculusResidueTarget(self.fp, target_residue=0.0)
        J_val2 = res_target2.J()
        expected_J = np.linalg.norm(current_residue - 0.0)
        np.testing.assert_allclose(J_val2, expected_J, rtol=1e-10)

    def test_trace_target_difference(self):
        """
        Test PyOculusTraceTarget with 'difference' penalize mode.
        """
        current_trace = self.fp.trace()
        
        # Test with target equal to current trace
        trace_target = PyOculusTraceTarget(self.fp, target_trace=current_trace, penalize="difference")
        J_val = trace_target.J()
        self.assertAlmostEqual(J_val, 0.0, places=7)
        
        # Test with a different target
        trace_target2 = PyOculusTraceTarget(self.fp, target_trace=0.0, penalize="difference")
        J_val2 = trace_target2.J()
        expected_J = current_trace - 0.0
        np.testing.assert_allclose(J_val2, expected_J, rtol=1e-10)

    def test_trace_target_above(self):
        """
        Test PyOculusTraceTarget with 'above' penalize mode.
        """
        current_trace = self.fp.trace()
        
        # Target below current trace - should penalize
        trace_target = PyOculusTraceTarget(self.fp, target_trace=current_trace - 1.0, penalize="above")
        J_val = trace_target.J()
        self.assertGreater(J_val, 0)
        
        # Target above current trace - should not penalize
        trace_target2 = PyOculusTraceTarget(self.fp, target_trace=current_trace + 1.0, penalize="above")
        J_val2 = trace_target2.J()
        self.assertEqual(J_val2, 0.0)

    def test_trace_target_below(self):
        """
        Test PyOculusTraceTarget with 'below' penalize mode.
        """
        current_trace = self.fp.trace()
        
        # Target above current trace - should penalize (negative)
        trace_target = PyOculusTraceTarget(self.fp, target_trace=current_trace + 1.0, penalize="below")
        J_val = trace_target.J()
        self.assertLess(J_val, 0)
        
        # Target below current trace - should not penalize
        trace_target2 = PyOculusTraceTarget(self.fp, target_trace=current_trace - 1.0, penalize="below")
        J_val2 = trace_target2.J()
        self.assertEqual(J_val2, 0.0)

    def test_trace_in_range_target(self):
        """
        Test PyOculusTraceInRangeTarget.
        """
        current_trace = self.fp.trace()
        
        # Current trace is within range - J should be 0
        range_target = PyOculusTraceInRangeTarget(self.fp, 
                                                   min_trace=current_trace - 1.0, 
                                                   max_trace=current_trace + 1.0)
        J_val = range_target.J()
        self.assertEqual(J_val, 0.0)
        
        # Current trace is below range
        range_target2 = PyOculusTraceInRangeTarget(self.fp, 
                                                    min_trace=current_trace + 0.5, 
                                                    max_trace=current_trace + 2.0)
        J_val2 = range_target2.J()
        self.assertGreater(J_val2, 0)
        self.assertAlmostEqual(J_val2, 0.5, places=5)
        
        # Current trace is above range
        range_target3 = PyOculusTraceInRangeTarget(self.fp, 
                                                    min_trace=current_trace - 2.0, 
                                                    max_trace=current_trace - 0.5)
        J_val3 = range_target3.J()
        self.assertGreater(J_val3, 0)
        self.assertAlmostEqual(J_val3, 0.5, places=5)

    def test_two_fixed_point_location_difference(self):
        """
        Test PyOculusTwoFixedPointLocationDifference.
        """
        # Create a second fixed point (same as first for simplicity)
        fp2 = PyOculusFixedPoint.from_field(self.bs, 
                                            self.nfp, 
                                            start_guess=self.axis_guess,
                                            fp_order=1,
                                            phi0=0,
                                            axis=False,
                                            integration_args=self.integration_args)
        
        # Distance between same locations should be 0
        diff_target = PyOculusTwoFixedPointLocationDifference(self.fp, fp2, 
                                                               target_distance=0.0,
                                                               other_phi=0,
                                                               penalize="difference")
        J_val = diff_target.J()
        self.assertAlmostEqual(J_val, 0.0, places=5)

    def test_location_at_other_angle_target(self):
        """
        Test PyOculusLocationAtOtherAngleTarget.
        """
        # Get location at a small angle
        phi_test = 0.1
        loc_other = self.fp.loc_at_other_angle(phi_test)
        
        # Test with target equal to the actual location at that angle
        angle_target = PyOculusLocationAtOtherAngleTarget(self.fp, 
                                                          other_phi=phi_test,
                                                          target_RZ=loc_other)
        J_val = angle_target.J()
        self.assertAlmostEqual(J_val, 0.0, places=5)
        
        # Test with a different target
        different_target = loc_other + np.array([0.05, 0.02])
        angle_target2 = PyOculusLocationAtOtherAngleTarget(self.fp, 
                                                           other_phi=phi_test,
                                                           target_RZ=different_target)
        J_val2 = angle_target2.J()
        expected_J = np.linalg.norm(loc_other - different_target)
        np.testing.assert_allclose(J_val2, expected_J, rtol=1e-10)


@unittest.skipUnless(newpyoculus, "pyOculus>=1.0.0 is required for this test")  
class TestClinicConnection(unittest.TestCase):
    """
    Test class for ClinicConnection.
    These tests are placeholders as ClinicConnection requires hyperbolic fixed points.
    """

    def setUp(self): 
        """
        load w7x data and generate two fixed points in the island chain
        """
        base_curves, base_currents, ma, nfp, bs = get_data('w7x', coil_order=12, points_per_period=4)
        self.base_curves = base_curves
        self.base_currents = base_currents
        self.ma = ma
        self.nfp = nfp
        self.bs = bs

        # Create a fixed point for testing
        self.fp1 = PyOculusFixedPoint.from_field(bs,
                                               nfp,
                                               start_guess=[5.],
                                               fp_order=5,
                                               phi0=0,
                                               axis=False,
                                               integration_args=self.integration_args)

    def test_clinic_connection_placeholder(self):
        """
        Placeholder test for ClinicConnection.
        ClinicConnection requires hyperbolic fixed points which are difficult
        to set up in a simple test case. Comprehensive testing of this class
        would require a configuration with known hyperbolic fixed points.
        """
        pass


