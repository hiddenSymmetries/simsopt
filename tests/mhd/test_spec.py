import unittest
import numpy as np
import os
import logging
import shutil
from simsopt.mhd.spec import Spec, nested_lists_to_array
from simsopt.core.least_squares_problem import LeastSquaresProblem
from simsopt.solve.serial_solve import least_squares_serial_solve
from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

# See if a spec executable can be found. If not, we will skip certain
# tests below.
exe = shutil.which('xspec')
if exe is None:
    print('Trying to find xspec')
    # Locations of xspec on Matt's laptop and the Github actions CI:
    try_exes = ['/Users/mattland/SPEC/xspec',
               '/home/runner/work/simsopt/simsopt/SPEC/xspec']
    for try_exe in try_exes:
        if os.path.isfile(try_exe):
            exe = try_exe
spec_found = (exe is not None)
logger.info("spec standalone executable: {}".format(exe))


class SpecTests(unittest.TestCase):
    def test_init_defaults(self):
        """
        Just create a Spec instance using the standard constructor,
        and make sure we can read some of the attributes.
        """
        spec = Spec()
        self.assertEqual(spec.nml['physicslist']['nfp'], 5)
        self.assertEqual(spec.nml['physicslist']['nvol'], 1)
        self.assertTrue(spec.need_to_run_code)

    def test_nested_lists_to_array(self):
        """
        Test the utility function used to convert the rbc and zbs data
        from f90nml to a 2D numpy array.
        """
        list_of_lists = [[42]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[42], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42, 0, 0],
                         [ 1, 2, 3]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[None, 42], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[0, 42, 0],
                         [1,  2, 3]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[42, 43, 44], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42, 43, 44],
                         [ 1,  2,  3]])
        np.testing.assert_allclose(arr1, arr2)

        list_of_lists = [[42, 43, 44, 45], [1, 2, 3]]
        arr1 = nested_lists_to_array(list_of_lists)
        arr2 = np.array([[42, 43, 44, 45],
                         [ 1,  2,  3,  0]])
        np.testing.assert_allclose(arr1, arr2)

        
    def test_init_from_file(self):
        """
        Try creating a Spec instance from a specified input file.
        """

        filename = os.path.join(TEST_DIR, '1DOF_Garabedian.sp')

        s = Spec(filename)
        self.assertEqual(s.nml['physicslist']['nfp'], 5)
        self.assertEqual(s.nml['physicslist']['nvol'], 1)
        self.assertTrue(s.need_to_run_code)

        places = 5
        
        # n = 0, m = 0:
        self.assertAlmostEqual(s.boundary.get_rc(0, 0), 1.0, places=places)
        self.assertAlmostEqual(s.boundary.get_zs(0, 0), 0.0, places=places)

        # n = 0, m = 1:
        self.assertAlmostEqual(s.boundary.get_rc(1, 0), 0.01, places=places)
        self.assertAlmostEqual(s.boundary.get_zs(1, 0), 0.01, places=places)

        # n = 1, m = 0:
        self.assertAlmostEqual(s.boundary.get_rc(0, 1), 0.1, places=places)
        self.assertAlmostEqual(s.boundary.get_zs(0, 1), 0.1, places=places)


    @unittest.skipIf(not spec_found, "SPEC standalone executable not found")
    def test_run(self):
        """
        Try running SPEC and reading in the output.
        """
        filename = os.path.join(TEST_DIR, '1DOF_Garabedian.sp')

        for new_mpol in [2, 3]:
            for new_ntor in [2, 3]:
                s = Spec(filename, exe=exe)
                print('new_mpol: {}, new_ntor: {}'.format(new_mpol, new_ntor))
                s.update_resolution(new_mpol, new_ntor)
                s.run()
        
                self.assertAlmostEqual(s.volume(), 0.001973920880217874, places=4)
        
                self.assertAlmostEqual(s.results.output.helicity, 0.435225, places=3)

                self.assertAlmostEqual(s.iota(), 0.544176, places=3)
    
    @unittest.skipIf(not spec_found, "SPEC standalone executable not found")
    def test_integrated_stellopt_scenarios_1dof(self):
        """
        This script implements the "1DOF_circularCrossSection_varyR0_targetVolume"
        example from
        https://github.com/landreman/stellopt_scenarios

        This optimization problem has one independent variable, representing
        the mean major radius. The problem also has one objective: the plasma
        volume. There is not actually any need to run an equilibrium code like
        SPEC since the objective function can be computed directly from the
        boundary shape. But this problem is a fast way to test the
        optimization infrastructure with SPEC.

        Details of the optimum and a plot of the objective function landscape
        can be found here:
        https://github.com/landreman/stellopt_scenarios/tree/master/1DOF_circularCrossSection_varyR0_targetVolume
        """
        for grad in [True, False]:
            # Start with a default surface.
            equil = Spec(exe=exe)
            surf = equil.boundary

            # Set the initial boundary shape. Here is one way to do it:
            surf.set('rc(0,0)', 1.0)
            # Here is another syntax that works:
            surf.set_rc(0, 1, 0.1)
            surf.set_zs(0, 1, 0.1)

            surf.set_rc(1, 0, 0.1)
            surf.set_zs(1, 0, 0.1)
            
            surf.set_rc(1, 1, 0)
            surf.set_zs(1, 1, 0)

            # SPEC parameters are all fixed by default, while surface
            # parameters are all non-fixed by default. You can choose
            # which parameters are optimized by setting their 'fixed'
            # attributes.
            surf.all_fixed()
            surf.set_fixed('rc(0,0)', False)

            # Turn off Poincare plots and use low resolution, for speed:
            equil.nml['diagnosticslist']['nPtrj'] = 0
            equil.nml['physicslist']['lrad'] = [2]
            
            # Each Target is then equipped with a shift and weight, to become a
            # term in a least-squares objective function
            desired_volume = 0.15
            term1 = (equil.volume, desired_volume, 1)

            # A list of terms are combined to form a nonlinear-least-squares
            # problem.
            prob = LeastSquaresProblem([term1])

            # Check that the problem was set up correctly:
            self.assertEqual(len(prob.dofs.names), 1)
            self.assertEqual(prob.dofs.names[0][:7], 'rc(0,0)')
            np.testing.assert_allclose(prob.x, [1.0])
            self.assertEqual(prob.dofs.all_owners, [equil, surf])
            self.assertEqual(prob.dofs.dof_owners, [surf])

            # Solve the minimization problem:
            least_squares_serial_solve(prob, grad=grad)
            
            self.assertAlmostEqual(surf.get_rc(0, 0), 0.7599088773175, places=5)
            self.assertAlmostEqual(equil.volume(), 0.15, places=6)
            self.assertAlmostEqual(surf.volume(), 0.15, places=6)
            self.assertLess(np.abs(prob.objective()), 1.0e-15)
    
    @unittest.skipIf(not spec_found, "SPEC standalone executable not found")
    def test_integrated_stellopt_scenarios_1dof_Garabedian(self):
        """
        This script implements the "1DOF_circularCrossSection_varyAxis_targetIota"
        example from
        https://github.com/landreman/stellopt_scenarios

        This example demonstrates optimizing a surface shape using the
        Garabedian representation instead of VMEC's RBC/ZBS representation.
        This optimization problem has one independent variable, the Garabedian
        Delta_{m=1, n=-1} coefficient, representing the helical excursion of
        the magnetic axis. The objective function is (iota - iota_target)^2,
        where iota is measured on the magnetic axis.

        Details of the optimum and a plot of the objective function landscape
        can be found here:
        https://github.com/landreman/stellopt_scenarios/tree/master/1DOF_circularCrossSection_varyAxis_targetIota
        """
        filename = os.path.join(TEST_DIR, '1DOF_Garabedian.sp')

        for mpol_ntor in [2, 4]:
            # Start with a default surface.
            equil = Spec(filename, exe=exe)
            equil.update_resolution(mpol_ntor, mpol_ntor)

            # We will optimize in the space of Garabedian coefficients
            # rather than RBC/ZBS coefficients. To do this, we convert the
            # boundary to the Garabedian representation:
            surf = equil.boundary.to_Garabedian()
            equil.boundary = surf

            # SPEC parameters are all fixed by default, while surface
            # parameters are all non-fixed by default. You can choose
            # which parameters are optimized by setting their 'fixed'
            # attributes.
            surf.all_fixed()
            surf.set_fixed('Delta(1,-1)', False)

            # Use low resolution, for speed:
            equil.nml['physicslist']['lrad'] = [4]
            equil.nml['diagnosticslist']['nppts'] = 100
            
            # Each Target is then equipped with a shift and weight, to become a
            # term in a least-squares objective function
            desired_iota = 0.41 # Sign was + for VMEC
            prob = LeastSquaresProblem([(equil.iota, desired_iota, 1)])

            # Check that the problem was set up correctly:
            self.assertEqual(len(prob.dofs.names), 1)
            self.assertEqual(prob.dofs.names[0][:11], 'Delta(1,-1)')
            np.testing.assert_allclose(prob.x, [0.1])
            self.assertEqual(prob.dofs.all_owners, [equil, surf])
            self.assertEqual(prob.dofs.dof_owners, [surf])

            # Solve the minimization problem:
            least_squares_serial_solve(prob)
            
            self.assertAlmostEqual(surf.get_Delta(1, -1), 0.08575, places=4)
            self.assertAlmostEqual(equil.iota(), desired_iota, places=5)
            self.assertLess(np.abs(prob.objective()), 1.0e-15)
    
    @unittest.skipIf(not spec_found, "SPEC standalone executable not found")
    def test_integrated_stellopt_scenarios_2dof(self):
        """
        This script implements the "2DOF_vmecOnly_targetIotaAndVolume" example from
        https://github.com/landreman/stellopt_scenarios

        This optimization problem has two independent variables, representing
        the helical shape of the magnetic axis. The problem also has two
        objectives: the plasma volume and the rotational transform on the
        magnetic axis.

        The resolution in this example (i.e. ns, mpol, and ntor) is somewhat
        lower than in the stellopt_scenarios version of the example, just so
        this example runs fast.

        Details of the optimum and a plot of the objective function landscape
        can be found here:
        https://github.com/landreman/stellopt_scenarios/tree/master/2DOF_vmecOnly_targetIotaAndVolume
        """
        filename = os.path.join(TEST_DIR, '2DOF_targetIotaAndVolume.sp')

        # Initialize SPEC from an input file
        equil = Spec(filename, exe=exe)
        surf = equil.boundary

        # VMEC parameters are all fixed by default, while surface parameters are all non-fixed by default.
        # You can choose which parameters are optimized by setting their 'fixed' attributes.
        surf.all_fixed()
        surf.set_fixed('rc(1,1)', False)
        surf.set_fixed('zs(1,1)', False)

        # Each Target is then equipped with a shift and weight, to become a
        # term in a least-squares objective function.  A list of terms are
        # combined to form a nonlinear-least-squares problem.
        desired_volume = 0.15
        volume_weight = 1
        term1 = (equil.volume, desired_volume, volume_weight)

        desired_iota = -0.41
        iota_weight = 1
        term2 = (equil.iota, desired_iota, iota_weight)

        prob = LeastSquaresProblem([term1, term2])

        # Solve the minimization problem:
        least_squares_serial_solve(prob)

        # The tests here are based on values from the VMEC version in
        # https://github.com/landreman/stellopt_scenarios/tree/master/2DOF_vmecOnly_targetIotaAndVolume
        # Due to this and the fact that we don't yet have iota on axis from SPEC, the tolerances are wide.
        """
        assert np.abs(surf.get_rc(1, 1) - 0.0313066948) < 0.001
        assert np.abs(surf.get_zs(1, 1) - (-0.031232391)) < 0.001
        assert np.abs(equil.volume() - 0.178091) < 0.001
        assert np.abs(surf.volume()  - 0.178091) < 0.001
        assert np.abs(equil.iota() - (-0.4114567)) < 0.001
        assert (prob.objective() - 7.912501330E-04) < 0.2e-4
        """
        self.assertAlmostEqual(surf.get_rc(1, 1), 0.0313066948, places=3)
        self.assertAlmostEqual(surf.get_zs(1, 1), -0.031232391, places=3)
        self.assertAlmostEqual(equil.volume(), 0.178091, places=3)
        self.assertAlmostEqual(surf.volume(), 0.178091, places=3)
        self.assertAlmostEqual(equil.iota(), -0.4114567, places=3)
        self.assertAlmostEqual(prob.objective(), 7.912501330E-04, places=3)


if __name__ == "__main__":
    unittest.main()
