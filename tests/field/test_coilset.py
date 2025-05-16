import unittest
import os
from simsopt.field.coilset import CoilSet, ReducedCoilSet
from simsopt.field.coil import Coil
from simsopt.geo import SurfaceRZFourier, CurveLength
from simsopt.configs import get_ncsx_data
import numpy as np
from monty.tempfile import ScratchDir
from simsopt import load, save


class TestCoilSet(unittest.TestCase):
    def setUp(self):
        # Create a CoilSet object for testing
        self.coilset = CoilSet()

    def test_default_properties(self):
        coilset = CoilSet()
        self.assertEqual(len(coilset.base_coils), 10)
        self.assertEqual(len(coilset.coils), 20)

    def test_wrong_init(self):
        curves, currents, _ = get_ncsx_data()
        coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
        with self.assertRaises(ValueError):
            CoilSet(coils=coils)

    def test_for_surface_classmethod(self):
        s = SurfaceRZFourier(nfp=2, mpol=3, ntor=3)
        coilset1 = CoilSet.for_surface(s, current_constraint='fix_all')
        for coil in coilset1.base_coils:
            self.assertEqual(coil.current.dof_size, 0)
        coilset2 = CoilSet.for_surface(s, current_constraint='fix_one')
        self.assertEqual(coilset2.base_coils[0].current.dof_size, 0)
        for coil in coilset2.base_coils[1:]:
            self.assertEqual(coil.current.dof_size, 1)
        coilset3 = CoilSet.for_surface(s, current_constraint='free_all')
        for coil in coilset3.base_coils:
            self.assertEqual(coil.current.dof_size, 1)

    def test_to_from_mgrid(self):
        order = 25
        ppp = 10
        with ScratchDir("."):
            self.coilset.to_makegrid_file("coils.file_to_load")
            loaded_coilset = CoilSet.from_makegrid_file("coils.file_to_load", self.coilset.surface, order=order, ppp=ppp)

        np.random.seed(1)

        points = np.asarray(17 * [[0.9, 0.4, -0.85]])
        points += 0.01 * (np.random.rand(*points.shape) - 0.5)
        self.coilset.bs.set_points(points)
        loaded_coilset.bs.set_points(points)

        B = self.coilset.bs.B()
        loaded_B = loaded_coilset.bs.B()
        np.testing.assert_allclose(B, loaded_B)

    def test_surface(self):
        # Test the surface property
        surface = self.coilset.surface
        self.assertIsNotNone(surface)

    def test_surface_setter_different_nfp(self):
        with self.assertRaises(ValueError):
            self.coilset.surface = SurfaceRZFourier(nfp=3)

    def test_surface_setter_nonstellsym(self):
        # Test the surface setter method
        first_surface = SurfaceRZFourier(nfp=2, stellsym=True)
        coilset = CoilSet(surface=first_surface)
        self.assertEqual(coilset.surface.deduced_range, SurfaceRZFourier.RANGE_HALF_PERIOD)
        second_surface = SurfaceRZFourier(nfp=2, stellsym=False)
        coilset.surface = second_surface
        self.assertEqual(coilset.surface.deduced_range, SurfaceRZFourier.RANGE_FIELD_PERIOD)

    def test_surface_setter_stellsym(self):
        # Test the surface setter method
        new_surface = SurfaceRZFourier(nfp=1, stellsym=True)
        self.coilset.surface = new_surface
        self.assertEqual(self.coilset.surface.deduced_range, SurfaceRZFourier.RANGE_HALF_PERIOD)

    def test_surface_setter_field_period(self):
        s = SurfaceRZFourier(nfp=2, stellsym=False).copy(range='half period')
        coilset = CoilSet(surface=s)
        self.assertEqual(coilset.surface.deduced_range, SurfaceRZFourier.RANGE_FIELD_PERIOD)
        s2 = SurfaceRZFourier(nfp=2, stellsym=True).copy(range='field period')
        coilset = CoilSet(surface=s2)
        self.assertEqual(coilset.surface.deduced_range, SurfaceRZFourier.RANGE_HALF_PERIOD)

    def test_base_coils(self):
        # Test the base_coils property
        base_coils = self.coilset.base_coils
        self.assertIsNotNone(base_coils)

    def test_base_coils_setter(self):
        # Test the base_coils setter method
        curves, currents, _ = get_ncsx_data()
        new_base_coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
        self.coilset.base_coils = new_base_coils
        self.assertEqual(self.coilset.base_coils, new_base_coils)

    def test_reduce(self):
        # Test the reduce method on simple collocation points
        copysurf = self.coilset.surface.copy(quadpoints_theta=np.random.random(5), quadpoints_phi=np.random.random(5))
        gammas = copysurf.gamma().reshape(-1, 3)
        normals = copysurf.normal().reshape(-1, 3)

        def target_function(coilset):
            coilset.bs.set_points(gammas)
            return np.sum(coilset.bs.B() * normals, axis=-1)
        reduced_coilset = self.coilset.reduce(target_function, nsv='nonzero')
        self.assertIsNotNone(reduced_coilset)

    def test_flux_penalty(self):
        # Test the flux_penalty function
        penalty = self.coilset.flux_penalty()
        self.assertIsNotNone(penalty.J())

    def test_length_penalty(self):
        # Test the length_penalty function
        TOTAL_LENGTH = 50
        penalty = self.coilset.length_penalty(TOTAL_LENGTH, f='identity')
        self.assertIsNotNone(penalty.J())

    def test_cc_distance_penalty(self):
        # Test the coil-coil distance penalty function
        DISTANCE_THRESHOLD = 0.1
        penalty = self.coilset.cc_distance_penalty(DISTANCE_THRESHOLD)
        self.assertIsNotNone(penalty.J())

    def test_cs_distance_penalty(self):
        # Test the coil-surface distance penalty function
        DISTANCE_THRESHOLD = 0.1
        penalty = self.coilset.cs_distance_penalty(DISTANCE_THRESHOLD)
        self.assertIsNotNone(penalty.J())

    def test_lp_curvature_penalty(self):
        # Test the lp_curvature_penalty function
        CURVATURE_THRESHOLD = 0.1
        penalty = self.coilset.lp_curvature_penalty(CURVATURE_THRESHOLD)
        self.assertIsNotNone(penalty.J())

    def test_meansquared_curvature_penalty(self):
        # Test the meansquared_curvature_penalty function
        penalty = self.coilset.meansquared_curvature_penalty()
        self.assertIsNotNone(penalty.J())

    def test_meansquared_curvature_threshold_penalty(self):
        # Test the meansquared_curvature_penalty function
        penalty = self.coilset.meansquared_curvature_threshold(CURVATURE_THRESHOLD=0.1)
        self.assertIsNotNone(penalty.J())

    def test_arc_length_variation_penalty(self):
        # Test the arc_length_variation_penalty function
        penalty = self.coilset.arc_length_variation_penalty()
        self.assertIsNotNone(penalty.J())

    def test_total_length_penalty(self):
        # Test the total_length function
        length = self.coilset.total_length_penalty()
        self.assertIsNotNone(length.J())
        self.assertTrue(CurveLength(self.coilset.coils[0].curve).J() < length.J())

    def test_total_length_property(self):
        # Test the total_length property
        length = self.coilset.total_length
        self.assertIsNotNone(length)
        self.assertTrue(length > 0)

    def test_coilset_to_vtk(self):
        with ScratchDir("."):
            self.coilset.to_vtk("test")
            self.assertTrue(os.path.exists("test_coils.vtu"))
            self.assertTrue(os.path.exists("test_surface.vts"))
        with ScratchDir("."):
            self.coilset.to_vtk("test2", add_biotsavart=False)
            self.assertTrue(os.path.exists("test2_coils.vtu"))
            self.assertTrue(os.path.exists("test2_surface.vts"))

    def test_save_load(self):
        with ScratchDir("."):
            save(self.coilset, "test.json")
            loaded_coilset = load("test.json")
            self.assertEqual(self.coilset.total_length, loaded_coilset.total_length)

    def test_dof_orders(self):
        # Test the dof_orders property
        dof_orders = self.coilset.get_dof_orders()
        self.assertIsNotNone(dof_orders)
        self.assertTrue(len(dof_orders) == self.coilset.dof_size)

#### Inherit from TestCoilSet (all tests run exept those overloaded)


class TestReducedCoilSet(TestCoilSet):
    def setUp(self):
        # Create a ReducedCoilSet object using collocation points
        self.unreduced_coilset = CoilSet()
        copysurf = self.unreduced_coilset.surface.copy(quadpoints_theta=np.random.random(7), quadpoints_phi=np.random.random(7))
        gammas = copysurf.gamma().reshape(-1, 3)
        normals = copysurf.normal().reshape(-1, 3)

        def target_function(coilset):
            coilset.bs.set_points(gammas)
            return np.sum(coilset.bs.B() * normals, axis=-1)
        self.test_target_function = target_function
        self.coilset = ReducedCoilSet.from_function(self.unreduced_coilset, target_function, nsv='nonzero')

    def test_partially_empty_init(self):
        #also test rebasing with none target function
        reduced_coilset = ReducedCoilSet()
        self.assertIsNotNone(reduced_coilset)
        with self.assertRaises(ValueError):
            reduced_coilset.recalculate_reduced_basis()  # target function not set; fail
        reduced_coilset2 = ReducedCoilSet(self.unreduced_coilset)
        self.assertIsNotNone(reduced_coilset2)
        self.assertEqual(len(reduced_coilset2.x), len(self.unreduced_coilset.x))
        reduced_coilset3 = ReducedCoilSet(self.unreduced_coilset, nsv=10)
        self.assertEqual(len(reduced_coilset3.x), 10)
        with self.assertRaises(ValueError):
            ReducedCoilSet(u_matrix=np.random.random((20, 20)))
        with self.assertRaises(ValueError):
            ReducedCoilSet(self.unreduced_coilset, nsv=1000)
        with self.assertRaises(TypeError):
            ReducedCoilSet(self.unreduced_coilset, nsv=np.pi)

    def test_nsv_setter(self):
        self.coilset.nsv = 10
        self.assertEqual(self.coilset.nsv, 10)
        self.assertEqual(len(self.coilset.x), 10)
        self.coilset.recalculate_reduced_basis()
        self.assertEqual(len(self.coilset.x), 10)
        with self.assertRaises(ValueError):
            self.coilset.nsv = 1000
        with self.assertRaises(TypeError):
            self.coilset.nsv = np.pi
        self.coilset.nsv = 'nonzero'

    def test_surface_setter(self):
        with self.assertRaises(ValueError):
            self.coilset.surface = SurfaceRZFourier(nfp=3, stellsym=False)
        self.coilset.surface = SurfaceRZFourier(nfp=1, stellsym=True)

    def test_recalculate_reduced_basis(self):
        reduced_coilset = ReducedCoilSet()
        reduced_coilset.recalculate_reduced_basis(self.test_target_function)
        # test if the reduced basis has the correct length
        self.assertEqual(len(reduced_coilset.x), len(self.test_target_function(self.unreduced_coilset)))

    def test_dof_orders(self):
        with self.assertRaises(ValueError):
            self.coilset.get_dof_orders()

    def test_taylor_test_of_value_decomposition(self):
        # test that small displacements along the singular vectors give the correct change in the target function evaluation.
        self.coilset.nsv = 10
        # calculate what the normal field on the collocation points is.
        initial_function_value = np.copy(self.test_target_function(self.unreduced_coilset))
        initial_coilset_x = np.copy(self.unreduced_coilset.x)
        epsilon = 1e-6
        for index, (rsv, lsv, singular_value) in enumerate(zip(self.coilset.rsv, self.coilset.lsv, self.coilset.singular_values)):
            newx = np.zeros_like(self.coilset.x)
            newx[index] = epsilon
            self.coilset.x = newx
            # test if coilset has been updated with rsv
            np.testing.assert_allclose(self.unreduced_coilset.x, initial_coilset_x+epsilon*rsv, atol=1e-6)

            new_function_value = self.test_target_function(self.coilset.coilset)
            function_diff = new_function_value - initial_function_value
            np.testing.assert_allclose(lsv, function_diff/(epsilon*singular_value), atol=1e-4)

        def test_wrong_setters(self):
            with self.assertRaises(ValueError):
                self.coilset.rsv = np.random.random(10)
            with self.assertRaises(ValueError):
                self.coilset.lsv = np.random.random(10)
            with self.assertRaises(ValueError):
                self.coilset.singular_values = np.random.random(10)
            with self.assertRaises(ValueError):
                self.coilset.coilset = self.unreduced_coilset
            with self.assertRaises(ValueError):
                self.coilset.coils = self.unreduced_coilset.coils
            with self.assertRaises(ValueError):
                self.set_dofs(np.random.random(self.coilset.nsv+1))


if __name__ == '__main__':
    unittest.main()
