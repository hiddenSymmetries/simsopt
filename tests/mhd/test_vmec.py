import unittest
import numpy as np
import os

from simsopt.core.least_squares_problem import LeastSquaresProblem
from simsopt.mhd.vmec import vmec_found
if vmec_found:
    from simsopt.mhd.vmec import Vmec
from . import TEST_DIR

@unittest.skipIf(not vmec_found, "Valid Python interface to VMEC not found")
class VmecTests(unittest.TestCase):
    def test_init_defaults(self):
        """
        Just create a Vmec instance using the standard constructor,
        and make sure we can read some of the attributes.
        """
        v = Vmec()
        self.assertEqual(v.indata.nfp, 5)
        self.assertFalse(v.indata.lasym)
        self.assertEqual(v.indata.mpol, 5)
        self.assertEqual(v.indata.ntor, 4)
        self.assertEqual(v.indata.delt, 0.5)
        self.assertEqual(v.indata.tcon0, 2.0)
        self.assertEqual(v.indata.phiedge, 1.0)
        self.assertEqual(v.indata.curtor, 0.0)
        self.assertEqual(v.indata.gamma, 0.0)
        self.assertEqual(v.indata.ncurr, 1)
        self.assertFalse(v.free_boundary)
        self.assertTrue(v.need_to_run_code)

    def test_init_from_file(self):
        """
        Try creating a Vmec instance from a specified input file.
        """

        filename = os.path.join(TEST_DIR, 'input.li383_low_res')

        v = Vmec(filename)
        self.assertEqual(v.indata.nfp, 3)
        self.assertEqual(v.indata.mpol, 4)
        self.assertEqual(v.indata.ntor, 3)
        self.assertEqual(v.boundary.mpol, 4)
        self.assertEqual(v.boundary.ntor, 3)

        # n = 0, m = 0:
        self.assertAlmostEqual(v.boundary.get_rc(0, 0), 1.3782)

        # n = 0, m = 1:
        self.assertAlmostEqual(v.boundary.get_zs(1, 0), 4.6465E-01)

        # n = 1, m = 1:
        self.assertAlmostEqual(v.boundary.get_zs(1, 1), 1.6516E-01)

        self.assertEqual(v.indata.ncurr, 1)
        self.assertFalse(v.free_boundary)
        self.assertTrue(v.need_to_run_code)

    def test_vmec_failure(self):
        """
        Verify that failures of VMEC are correctly caught and represented
        by large values of the objective function.
        """
        filename = os.path.join(TEST_DIR, 'input.li383_low_res')
        vmec = Vmec(filename)
        # Use the objective function from
        # stellopt_scenarios_2DOF_targetIotaAndVolume:
        prob = LeastSquaresProblem([(vmec.iota_axis, 0.41, 1),
                                    (vmec.volume, 0.15, 1)])
        r00 = vmec.boundary.get_rc(0, 0)
        # The first evaluation should succeed.
        f = prob.f()
        print(f[0], f[1])
        correct_f = [-0.004577338528148067, 2.8313872701632925]
        # Don't worry too much about accuracy here.
        np.testing.assert_allclose(f, correct_f, rtol=0.1)
        
        # Now set a crazy boundary shape to make VMEC fail. This
        # boundary causes VMEC to hit the max number of iterations
        # without meeting ftol.
        vmec.boundary.set_rc(0, 0, 0.2)
        vmec.need_to_run_code = True
        f = prob.f()
        print(f)
        np.testing.assert_allclose(f, np.full(2, 1.0e12))

        # Restore a reasonable boundary shape. VMEC should work again.
        vmec.boundary.set_rc(0, 0, r00)
        vmec.need_to_run_code = True
        f = prob.f()
        print(f)
        np.testing.assert_allclose(f, correct_f, rtol=0.1)
        
        # Now set a self-intersecting boundary shape. This causes VMEC
        # to fail with "ARNORM OR AZNORM EQUAL ZERO IN BCOVAR" before
        # it even starts iterating.
        orig_mode = vmec.boundary.get_rc(1, 3)
        vmec.boundary.set_rc(1, 3, 0.5)
        vmec.need_to_run_code = True
        f = prob.f()
        print(f)
        np.testing.assert_allclose(f, np.full(2, 1.0e12))
        
        # Restore a reasonable boundary shape. VMEC should work again.
        vmec.boundary.set_rc(1, 3, orig_mode)
        vmec.need_to_run_code = True
        f = prob.f()
        print(f)
        np.testing.assert_allclose(f, correct_f, rtol=0.1)
        
    #def test_stellopt_scenarios_1DOF_circularCrossSection_varyR0_targetVolume(self):
        """
        This script implements the "1DOF_circularCrossSection_varyR0_targetVolume"
        example from
        https://github.com/landreman/stellopt_scenarios

        This optimization problem has one independent variable, representing
        the mean major radius. The problem also has one objective: the plasma
        volume. There is not actually any need to run an equilibrium code like
        VMEC since the objective function can be computed directly from the
        boundary shape. But this problem is a fast way to test the
        optimization infrastructure with VMEC.

        Details of the optimum and a plot of the objective function landscape
        can be found here:
        https://github.com/landreman/stellopt_scenarios/tree/master/1DOF_circularCrossSection_varyR0_targetVolume
        """
"""
        # Start with a default surface, which is axisymmetric with major
        # radius 1 and minor radius 0.1.
        equil = Vmec()
        surf = equil.boundary

        # Set the initial boundary shape. Here is one way to do it:
        surf.set('rc(0,0)', 1.0)
        # Here is another syntax that works:
        surf.set_rc(0, 1, 0.1)
        surf.set_zs(0, 1, 0.1)

        surf.set_rc(1, 0, 0.1)
        surf.set_zs(1, 0, 0.1)

        # VMEC parameters are all fixed by default, while surface
        # parameters are all non-fixed by default. You can choose
        # which parameters are optimized by setting their 'fixed'
        # attributes.
        surf.all_fixed()
        surf.set_fixed('rc(0,0)', False)

        # Each Target is then equipped with a shift and weight, to become a
        # term in a least-squares objective function
        desired_volume = 0.15
        term1 = LeastSquaresTerm(equil.volume, desired_volume, 1)

        # A list of terms are combined to form a nonlinear-least-squares
        # problem.
        prob = LeastSquaresProblem([term1])

        # Check that the problem was set up correctly:
        self.assertEqual(prob.names, ['rc(0,0)'])
        np.testing.assert_allclose(prob.x, [1.0])
        self.assertEqual(prob.all_owners, [equil, surf])
        self.assertEqual(prob.dof_owners, [surf])
            
        # Solve the minimization problem:
        prob.solve()

        print("At the optimum,")
        print(" rc(m=0,n=0) = ", surf.get_rc(0, 0))
        print(" volume, according to VMEC    = ", equil.volume())
        print(" volume, according to Surface = ", surf.volume())
        print(" objective function = ", prob.objective)

        self.assertAlmostEqual(surf.get_rc(0, 0), 0.7599088773175, places=5)
        self.assertAlmostEqual(equil.volume(), 0.15, places=6)
        self.assertAlmostEqual(surf.volume(), 0.15, places=6)
        self.assertLess(np.abs(prob.objective), 1.0e-15)

        equil.finalize()
"""     
if __name__ == "__main__":
    unittest.main()
