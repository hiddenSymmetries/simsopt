import unittest
import numpy as np
from simsopt import SurfaceRZFourier, optimizable, LeastSquaresTerm, LeastSquaresProblem

class IntegratedTests(unittest.TestCase):
    def test_2dof_surface_opt(self):
        """
        Optimize the minor radius and elongation of an axisymmetric torus to
        obtain a desired volume and area.
        """

        desired_volume = 0.6
        desired_area = 8.0

        # Start with a default surface, which is axisymmetric with major
        # radius 1 and minor radius 0.1.
        surf = optimizable(SurfaceRZFourier())

        # Set initial surface shape. It helps to make zs(1,0) larger
        # than rc(1,0) since there are two solutions to this
        # optimization problem, and for testing we want to find one
        # rather than the other.
        surf.set_zs(1, 0, 0.2)

        # Parameters are all non-fixed by default, meaning they will be
        # optimized.  You can choose to exclude any subset of the variables
        # from the space of independent variables by setting their 'fixed'
        # property to True.
        surf.set_fixed('rc(0,0)')
        
        # Each Target is then equipped with a shift and weight, to become a
        # term in a least-squares objective function
        term1 = LeastSquaresTerm(surf.volume, desired_volume, 1)
        term2 = LeastSquaresTerm(surf.area,   desired_area,   1)

        # A list of terms are combined to form a nonlinear-least-squares
        # problem.
        prob = LeastSquaresProblem([term1, term2])

        # Verify the state vector and names are what we expect
        np.testing.assert_allclose(prob.x, [0.1, 0.2])
        self.assertEqual(prob.dofs.names[0][:28], 'rc(1,0) of SurfaceRZFourier ')
        self.assertEqual(prob.dofs.names[1][:28], 'zs(1,0) of SurfaceRZFourier ')
        
        # Solve the minimization problem:
        prob.solve()

        # Check results
        self.assertAlmostEqual(surf.get_rc(0, 0), 1.0, places=13)
        self.assertAlmostEqual(surf.get_rc(1, 0), 0.10962565115956417, places=13)
        self.assertAlmostEqual(surf.get_zs(0, 0), 0.0, places=13)
        self.assertAlmostEqual(surf.get_zs(1, 0), 0.27727411213693337, places=13)
        self.assertAlmostEqual(surf.volume(), desired_volume, places=8)
        self.assertAlmostEqual(surf.area(), desired_area, places=8)
        self.assertLess(np.abs(prob.objective), 1.0e-15)

if __name__ == "__main__":
    unittest.main()
