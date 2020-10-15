import unittest
import numpy as np
import os
from simsopt.mhd.vmec import *
from . import TEST_DIR

@unittest.skipIf(not vmec_found, "Valid Python interface to VMEC not found")
class VmecTests(unittest.TestCase):
    def test_init_defaults(self):
        """
        Just create a Vmec instance using the standard constructor,
        and make sure we can read some of the attributes.
        """
        v = Vmec()
        self.assertEqual(v.nfp, 5)
        self.assertTrue(v.stelsym)
        self.assertEqual(v.mpol, 5)
        self.assertEqual(v.ntor, 4)
        self.assertEqual(v.delt, 0.5)
        self.assertEqual(v.tcon0, 2.0)
        self.assertEqual(v.phiedge, 1.0)
        self.assertEqual(v.curtor, 0.0)
        self.assertEqual(v.gamma, 0.0)
        self.assertEqual(v.ncurr, 1)
        self.assertFalse(v.free_boundary)
        self.assertTrue(v.need_to_run_code)
        v.finalize()

    def test_init_from_file(self):
        """
        Try creating a Vmec instance from a specified input file.
        """

        filename = os.path.join(TEST_DIR, 'input.li383_low_res')

        v = Vmec(filename)
        self.assertEqual(v.nfp, 3)
        self.assertEqual(v.mpol, 4)
        self.assertEqual(v.ntor, 3)
        self.assertEqual(v.boundary.mpol, 4)
        self.assertEqual(v.boundary.ntor, 3)

        # n = 0, m = 0:
        self.assertAlmostEqual(v.boundary.get_rc(0, 0), 1.3782)

        # n = 0, m = 1:
        self.assertAlmostEqual(v.boundary.get_zs(1, 0), 4.6465E-01)

        # n = 1, m = 1:
        self.assertAlmostEqual(v.boundary.get_zs(1, 1), 1.6516E-01)

        self.assertEqual(v.ncurr, 1)
        self.assertFalse(v.free_boundary)
        self.assertTrue(v.need_to_run_code)

        v.finalize()

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
