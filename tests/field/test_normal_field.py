import os
import unittest
import logging
import re

import numpy as np
from monty.tempfile import ScratchDir

from simsopt._core.util import DofLengthMismatchError
from simsopt.field import NormalField, CoilNormalField, CoilSet, Coil, Current
from simsopt.mhd import Spec
from simsopt.geo import SurfaceRZFourier

try:
    import py_spec
except ImportError:
    py_spec = None

from . import TEST_DIR

logger = logging.getLogger(__name__)


class NormalFieldTests(unittest.TestCase):
    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_initialize_normal_field_from_spec(self):
        """
        Test the initialization of an instance of NormalField using a SPEC input file
        """

        filename = os.path.join(TEST_DIR, "Dommaschk.sp")

        normal_field = NormalField.from_spec(filename)

        self.assertAlmostEqual(normal_field.get_vns(m=3, n=-1), 1.71466651e-03)
        self.assertAlmostEqual(normal_field.get_vns(m=5, n=1), -3.56991494e-05)

    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_dofs(self):
        """
        Test access to degrees of freedom, its setter
        """

        # Init from SPEC input file
        filename = os.path.join(TEST_DIR, "Dommaschk.sp")
        normal_field = NormalField.from_spec(filename)

        # Get dofs
        dofs = normal_field.local_full_x

        # Check size of array
        self.assertEqual(dofs.size, normal_field.ndof)

        # Check value of an element
        m = 5
        n = 1
        ii = normal_field.get_index_in_dofs(m=m, n=n)
        self.assertEqual(dofs[ii], normal_field.get_vns(m=m, n=n))

        # Set dofs to zeros

        # ... Check test of size
        with self.assertRaises(DofLengthMismatchError):
            normal_field.local_full_x = np.zeros((normal_field.ndof - 1,))

        # ... Actually set them to zeros
        normal_field.local_full_x = np.zeros((normal_field.ndof,))
        dofs = normal_field.local_full_x

        for el in dofs:
            self.assertEqual(el, 0)

    def test_get_index(self):
        """
        Test that the correct index is returned by get_index_in_dofs
        """

        normal_field = NormalField(stellsym=False, mpol=4, ntor=4)

        normal_field.set_vns(m=3, n=-1, value=1)
        normal_field.set_vnc(m=2, n=1, value=-1)

        dofs = normal_field.local_full_x

        ii = normal_field.get_index_in_dofs(m=3, n=-1, even=False)
        self.assertEqual(dofs[ii], 1)

        ii = normal_field.get_index_in_dofs(m=2, n=1, even=True)
        self.assertEqual(dofs[ii], -1)

    def test_getter_setter(self):
        """
        Check setters / getters of vns, vnc harmonics.
        """

        normal_field = NormalField(stellsym=False, mpol=2, ntor=1)

        normal_field.set_vns(m=1, n=-1, value=1)
        self.assertEqual(normal_field.get_vns(m=1, n=-1), 1)
        i, j = normal_field.get_index_in_array(m=1, n=-1)
        self.assertEqual(normal_field.vns[i, j], 1)

        normal_field.set_vnc(m=2, n=1, value=0.5)
        self.assertEqual(normal_field.get_vnc(m=2, n=1), 0.5)
        i, j = normal_field.get_index_in_array(m=2, n=1)
        self.assertEqual(normal_field.vnc[i, j], 0.5)

    def test_check_mn(self):
        """
        Check built in test routine for values of m, n
        """

        normal_field = NormalField(stellsym=False, mpol=2, ntor=1)

        with self.assertRaises(ValueError):
            normal_field.check_mn(m=-1, n=0)
        with self.assertRaises(ValueError):
            normal_field.check_mn(m=3, n=0)
        with self.assertRaises(ValueError):
            normal_field.check_mn(m=1, n=-2)
        with self.assertRaises(ValueError):
            normal_field.check_mn(m=1, n=2)
        with self.assertRaises(ValueError):
            normal_field.check_mn(m=0, n=-1)

    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_make_names(self):
        """
        Test routine make_names.

        We verify that first element is vns(0,1), and that
        there are the correct amount of elements
        """

        # Init from SPEC input file
        filename = os.path.join(TEST_DIR, "Dommaschk.sp")
        normal_field = NormalField.from_spec(filename)

        # Make names
        names = normal_field._make_names()

        # Check
        self.assertTrue(names[0] == "vns(0,1)")
        self.assertTrue(names[-1] == "vns(5,5)")

        # Check length
        self.assertEqual(len(names), normal_field.ndof)

    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_change_resolution(self):
        """
        Test that object resolution can be changed.

        We proceed as follow:
        1. initialize a normal field instance from a SPEC input file;
        2. save locally some selected non-zero harmonics
        3. change resolution of the normal field
        4. Check, using the get_vns routine, that harmonics are still available.
        5. Check, that new harmonics are indeed zeros
        6. Check, that deleted harmonics are no longer accessible
        """

        # 1. Init from SPEC input file
        filename = os.path.join(TEST_DIR, "Dommaschk.sp")
        normal_field = NormalField.from_spec(filename)

        # 2. Save some selected non-zero harmonics
        val1 = normal_field.get_vns(m=3, n=-1)
        val2 = normal_field.get_vns(m=5, n=1)

        # 3. Change resolution
        new_mpol = 6
        new_ntor = 2
        normal_field.change_resolution(mpol=new_mpol, ntor=new_ntor)

        # ... Check resolution has been changed
        self.assertEqual(normal_field.mpol, new_mpol)
        self.assertEqual(normal_field.ntor, new_ntor)

        # 4. Check harmonics
        self.assertEqual(normal_field.get_vns(m=3, n=-1), val1)
        self.assertEqual(normal_field.get_vns(m=5, n=1), val2)

        # 5. Check new harmonics are zeros
        self.assertEqual(normal_field.get_vns(m=6, n=2), 0)
        self.assertEqual(normal_field.get_vns(m=6, n=-1), 0)

        # 6. Check deleted harmonics are no longer accessible
        with self.assertRaises(ValueError):
            normal_field.check_mn(m=0, n=3)
        with self.assertRaises(ValueError):
            normal_field.check_mn(m=4, n=-4)

    def test_fixed_range(self):
        """
        Test the ability to fix ranges of dofs.
        """

        # Initialize an instance of Normal field
        normal_field = NormalField(stellsym=True, mpol=6, ntor=4)

        # Check that dofs are free at initialization
        [
            self.assertTrue(normal_field.is_free(ii))
            for ii in range(0, normal_field.ndof)
        ]

        # Check fixed range
        normal_field.fixed_range(mmin=3, mmax=4, nmin=-2, nmax=1, fixed=True)

        for mm in range(0, normal_field.mpol + 1):
            for nn in range(-normal_field.ntor, normal_field.ntor + 1):
                if mm == 0 and nn <= 0:
                    continue

                ii = normal_field.get_index_in_dofs(m=mm, n=nn)
                if (mm == 3 or mm == 4) and (nn >= -2 and nn <= 1):
                    self.assertTrue(normal_field.is_fixed(ii))
                else:
                    self.assertTrue(normal_field.is_free(ii))

    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_from_spec_object(self):
        """
        test classmethod to instantiate from an existing SPEC object
        """
        from simsopt.mhd import Spec

        # Init from SPEC input file
        filename = os.path.join(TEST_DIR, "Dommaschk.sp")
        spec = Spec(filename)
        normal_field = NormalField.from_spec_object(spec)
        normal_field_2 = NormalField.from_spec(filename)

        self.assertAlmostEqual(normal_field.get_vns(m=3, n=-1), 1.71466651e-03)
        self.assertAlmostEqual(normal_field.get_vns(m=5, n=1), -3.56991494e-05)

        self.assertTrue(
            np.allclose(normal_field.local_full_x, normal_field_2.local_full_x)
        )

    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_fail_for_fixedb(self):
        """
        test if instantiation fails if spec is not freeboundary
        """
        from simsopt.mhd import Spec

        # Init from SPEC input file
        spec = Spec()
        with self.assertRaises(ValueError):
            NormalField.from_spec_object(spec)

    def test_wrong_index(self):
        """
        test if accessing a wrong m or n raises an error
        """
        normal_field = NormalField(stellsym=False, mpol=3, ntor=2)
        with self.assertRaises(ValueError):
            normal_field.get_index_in_array(m=4, n=1)
        with self.assertRaises(ValueError):
            normal_field.get_index_in_array(m=1, n=3)
        with self.assertRaises(ValueError):
            normal_field.get_index_in_array(m=0, n=-1)
        with self.assertRaises(ValueError):
            normal_field.get_index_in_array(m=3, n=-3)
        with self.assertRaises(ValueError):
            normal_field.get_index_in_array(m=4, n=-3)

    def test_asarray_getter_setter_raises(self):
        """
        test that if wrong bounds are given, an error is raised
        """
        normal_field = NormalField(stellsym=False, mpol=3, ntor=2)
        with self.assertRaises(ValueError):
            normal_field.get_vns_asarray(mpol=4, ntor=5)
        with self.assertRaises(ValueError):
            normal_field.get_vns_asarray(mpol=3, ntor=4)
        with self.assertRaises(ValueError):
            normal_field.get_vnc_asarray(mpol=4, ntor=5)
        with self.assertRaises(ValueError):
            normal_field.get_vnc_asarray(mpol=3, ntor=4)
        with self.assertRaises(ValueError):
            normal_field.get_vns_vnc_asarray(mpol=4, ntor=5)
        with self.assertRaises(ValueError):
            normal_field.get_vns_vnc_asarray(mpol=3, ntor=4)
        with self.assertRaises(ValueError):
            normal_field.set_vns_asarray(np.zeros((3, 7)), mpol=2, ntor=3)
        with self.assertRaises(ValueError):
            normal_field.set_vnc_asarray(np.zeros((4, 7)), mpol=2, ntor=3)
        with self.assertRaises(ValueError):
            normal_field.set_vns_vnc_asarray(
                np.zeros((4, 7)), np.zeros((4, 7)), mpol=2, ntor=3
            )

    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_get_set_vns_vnc_asarray(self):
        """
        test the array-wise getter and setter functions for the
        Vns and Vnc arrays
        """
        filename = os.path.join(TEST_DIR, "M16N08.sp")
        normal_field = NormalField.from_spec(filename)
        vns = normal_field.get_vns_asarray()
        vnc = normal_field.get_vnc_asarray()
        vns2, vnc2 = normal_field.get_vns_vnc_asarray()
        self.assertTrue(np.allclose(vns, vns2))
        self.assertTrue(np.allclose(vnc, vnc2))
        vns3 = np.copy(vnc)
        vnc3 = np.copy(vnc)
        i, j = normal_field.get_index_in_array(m=3, n=-1)
        vns3[i, j] = 0.5
        normal_field.set_vns_asarray(vns3)
        self.assertEqual(normal_field.get_vns(m=3, n=-1), 0.5)
        dof_index = normal_field.get_index_in_dofs(m=3, n=-1)
        self.assertEqual(normal_field.local_full_x[dof_index], 0.5)
        i, j = normal_field.get_index_in_array(m=2, n=1)
        vnc3[i, j] = 1.5
        normal_field.set_vnc_asarray(vnc3)
        self.assertEqual(normal_field.get_vnc(m=2, n=1), 1.5)
        dof_index = normal_field.get_index_in_dofs(m=2, n=1, even=True)
        self.assertEqual(normal_field.local_full_x[dof_index], 1.5)
        normal_field.set_vns_vnc_asarray(vns2, vnc2)

    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_get_real_space_field(self):
        """
        test the conversion to real space
        """
        filename = os.path.join(TEST_DIR, "Dommaschk.sp")

        normal_field = NormalField.from_spec(filename)
        real_space_field = normal_field.get_real_space_field()
        self.assertTrue(real_space_field is not None)
        self.assertEqual(
            real_space_field.shape,
            (
                normal_field.surface.quadpoints_phi.size,
                normal_field.surface.quadpoints_theta.size,
            ),
        )

    def test_vns_vnc_setter(self):
        normal_field = NormalField()
        with self.assertRaises(AttributeError):
            normal_field.vns = 1
        with self.assertRaises(AttributeError):
            normal_field.vnc = 1


class CoilNormalFieldTests(unittest.TestCase):
    def test_empty_init(self):
        coil_normal_field = CoilNormalField()
        self.assertIsNotNone(coil_normal_field)

    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_spec_coil_correspondence_on_converged_output(self):
        # Init from SPEC input file
        with ScratchDir("."):
            filespec = os.path.join(TEST_DIR, "M16N08.sp")
            spec = Spec(filespec)
            filecoils = os.path.join(TEST_DIR, "coils.M16N08")
            coilset = CoilSet.from_makegrid_file(filecoils, spec.computational_boundary)
            coil_normal_field = CoilNormalField(coilset)
            for m in range(0, spec.mpol + 1):
                for n in range(-spec.ntor, spec.ntor + 1):
                    if m == 0 and n <= 0:
                        continue
                    self.assertAlmostEqual(
                        coil_normal_field.get_vns(m=m, n=n),
                        spec.normal_field.get_vns(m=m, n=n),
                        msg=f"m={m}, n={n}",
                        places=6,
                    )
                    self.assertAlmostEqual(
                        coil_normal_field.get_vnc(m=m, n=n),
                        spec.normal_field.get_vnc(m=m, n=n),
                        msg=f"m={m}, n={n}",
                        places=6,
                    )

    def test_vns_vns_setter_raises(self):
        coil_normal_field = CoilNormalField()
        with self.assertRaises(AttributeError):
            coil_normal_field.vns = coil_normal_field.vns
        with self.assertRaises(AttributeError):
            coil_normal_field.set_vns_asarray(coil_normal_field.vns)
        with self.assertRaises(AttributeError):
            coil_normal_field.set_vns(0, 1, .1)

        with self.assertRaises(AttributeError):
            coil_normal_field.vnc = coil_normal_field.vnc
        with self.assertRaises(AttributeError):
            coil_normal_field.set_vnc_asarray(coil_normal_field.vnc)
        with self.assertRaises(AttributeError):
            coil_normal_field.set_vnc(0, 1, .1)

        with self.assertRaises(AttributeError):
            coil_normal_field.set_vns_vnc_asarray(coil_normal_field.vns, coil_normal_field.vnc)

    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_reduce_coilset(self):
        """
        test if the coilset can be reduced, and
        if the"""
        with ScratchDir("."):
            spec = Spec.default_freeboundary(copy_to_pwd=True)
            spec.need_to_run_code = False
            base_curves = CoilSet._circlecurves_around_surface(
                spec.computational_boundary, coils_per_period=2, order=4
            )
            base_coils = [Coil(curve, Current(1e5)) for curve in base_curves]
            for coil in base_coils:
                coil.current.fix_all()  # fix the currents
            coilset = CoilSet(
                surface=spec.computational_boundary, base_coils=base_coils
            )
            cnf = CoilNormalField(coilset)
            self.assertEqual(cnf.dof_names, coilset.dof_names)
            self.assertEqual(cnf.dof_size, coilset.dof_size)
            self.assertTrue(cnf.dof_names[0].startswith("CurveXYZFourier"))
            spec.normal_field = cnf
            # test if spec's recompute bell has rung:
            self.assertTrue(spec.need_to_run_code)
            spec.need_to_run_code = False

            cnf.reduce_coilset()
            self.assertEqual(cnf.dof_size, spec.computational_boundary.ntor + spec.computational_boundary.mpol * (spec.computational_boundary.ntor * 2 + 1))
            self.assertTrue(cnf.dof_names[0].startswith("ReducedCoilSet"))
            # test if specs recompute bell has rung:
            self.assertTrue(spec.need_to_run_code)
            # test if the degrees of freedom are al zeros:
            np.testing.assert_equal(cnf.x, 0)

            # perform a taylor test; see if a small change in the first dof leads to the change in the fourier components predicted by the first l.s.v.
            epsilon = 1e-5
            initial_vns = np.copy(cnf.vns)
            dofs = np.zeros(cnf.dof_size)
            dofs[0] = epsilon
            cnf.x = dofs
            vnsdiff = initial_vns - cnf.vns
            vnsdiff_unraveled = vnsdiff.ravel()[
                cnf.coilset.surface.ntor + 1:
            ]  # hardcoded stellsym part
            np.testing.assert_allclose(
                cnf.coilset.lsv[0],
                -1*vnsdiff_unraveled / (epsilon * cnf.coilset.singular_values[0]),
                atol=1e-4,
            )

    def test_optimize_coils(self):
        surface = SurfaceRZFourier(nfp=3, stellsym=False)
        coilset = CoilSet(surface=surface)
        cnf = CoilNormalField(coilset)
        cnf.optimize_coils(targetvns=np.zeros_like(cnf.vns), targetvnc=np.zeros_like(cnf.vnc), TARGET_LENGTH=500, MAXITER=10)

    def test_inherited_methods_handled_correctly(self):
        cnf = CoilNormalField()
        with self.assertRaises(ValueError):
            CoilNormalField.from_spec()
        with self.assertRaises(ValueError):
            CoilNormalField.from_spec_object()
        with self.assertRaises(ValueError):
            cnf.get_dofs()
        with self.assertRaises(ValueError):
            cnf.get_index_in_dofs()
        with self.assertRaises(ValueError):
            cnf.fixed_range()
        with self.assertRaises(ValueError):
            cnf.change_resolution()

    def test_real_space_field(self):
        surface = SurfaceRZFourier(nfp=3, stellsym=False, mpol=12, ntor=12)
        base_curves = CoilSet._circlecurves_around_surface(
            surface, coils_per_period=2, order=4
        )
        base_coils = [Coil(curve, Current(1e5)) for curve in base_curves]
        coilset = CoilSet(base_coils=base_coils, surface=surface)
        cnf = CoilNormalField(coilset)
        real_space_field = cnf.get_real_space_field()  # fourier-transform vns and vnc to real space
        thetasize, phisize = cnf.surface.quadpoints_theta.size, cnf.surface.quadpoints_phi.size
        directly_evaluated = np.copy(np.sum(cnf.coilset.bs.B().reshape(phisize, thetasize, 3) * cnf.surface.unitnormal(), axis=2))  # evaluate the field on the surface
        np.testing.assert_allclose(real_space_field, directly_evaluated, atol=1e-3)

    def test_cache_vns(self):
        cnf = CoilNormalField()
        self.assertIsNone(cnf._vns)
        tmp = cnf.vns
        np.testing.assert_equal(cnf._vns, tmp)
        self.assertIsNotNone(cnf._vnc)

    def test_cache_vnc(self):
        cnf = CoilNormalField()
        self.assertIsNone(cnf._vnc)
        tmp = cnf.vnc
        np.testing.assert_equal(cnf._vnc, tmp)
        self.assertIsNotNone(cnf._vns)

    def test_nonstellsym_reduce(self):
        """
        nonstellaratorsymmetric fields have both vns and vnc components on the field
        and a different function is used to reduce the coilset. 
        This test ensures that the reduction works for non-stellaratorsymmetric fields
        """
        surface = SurfaceRZFourier(nfp=2, stellsym=False, mpol=6, ntor=6)
        base_curves = CoilSet._circlecurves_around_surface(
            surface, coils_per_period=2, order=4
        )
        base_coils = [Coil(curve, Current(1e5)) for curve in base_curves]
        coilset = CoilSet(base_coils=base_coils, surface=surface)
        cnf = CoilNormalField(coilset)
        cnf.reduce_coilset()

        initial_vns = np.copy(cnf.vns)
        initial_vnc = np.copy(cnf.vnc)

        epsilon = 1e-5
        dofs = np.zeros(cnf.dof_size)
        dofs[0] = epsilon
        cnf.x = dofs

        vnsdiff = initial_vns - cnf.vns
        vncdiff = initial_vnc - cnf.vnc

        diff_unraveled = np.concatenate(
            (vnsdiff.ravel()[
                cnf.coilset.surface.ntor + 1:
            ],
                vncdiff.ravel()[
                cnf.coilset.surface.ntor:
            ]
            )
        )
        np.testing.assert_allclose(
            cnf.coilset.lsv[0],
            -1*diff_unraveled / (epsilon * cnf.coilset.singular_values[0]),
            atol=1e-4,
        )

    def test_vns_vnc_asarray(self):
        cnf = CoilNormalField()
        vns = cnf.get_vns_asarray()
        vnc = cnf.get_vnc_asarray()
        self.assertIsNotNone(vns)
        self.assertIsNotNone(vnc)

    def test_wrong_index(self):
        """
        test if accessing a wrong m or n raises an error
        """
        surface = SurfaceRZFourier(nfp=3, stellsym=False, mpol=3, ntor=2)
        base_curves = CoilSet._circlecurves_around_surface(
            surface, coils_per_period=2, order=4
        )
        base_coils = [Coil(curve, Current(1e5)) for curve in base_curves]
        coilset = CoilSet(base_coils=base_coils, surface=surface)
        cnf = CoilNormalField(coilset)
        with self.assertRaises(ValueError):
            cnf.get_index_in_array(m=4, n=1)
        with self.assertRaises(ValueError):
            cnf.get_index_in_array(m=1, n=3)
        with self.assertRaises(ValueError):
            cnf.get_index_in_array(m=0, n=-1)
        with self.assertRaises(ValueError):
            cnf.get_index_in_array(m=3, n=-3)
        with self.assertRaises(ValueError):
            cnf.get_index_in_array(m=4, n=-3)

    def test_double_reduction(self):
        """
        test if reducing a coilset twice works
        """
        surface = SurfaceRZFourier(nfp=2, stellsym=False, mpol=6, ntor=6)
        base_curves = CoilSet._circlecurves_around_surface(
            surface, coils_per_period=2, order=4
        )
        base_coils = [Coil(curve, Current(1e5)) for curve in base_curves]
        coilset = CoilSet(base_coils=base_coils, surface=surface)
        cnf = CoilNormalField(coilset)
        cnf.reduce_coilset()
        num1 = int(re.search(r'\d+', cnf.dof_names[0]).group())  # grab 'ReducedCoilSet*N* from first dof name
        cnf.reduce_coilset()
        num2 = int(re.search(r'\d+', cnf.dof_names[0]).group())  # grab 'ReducedCoilSet*N* from first dof name
        self.assertEqual(num2-num1, 1)  # Coilset sucessfully replaced


if __name__ == "__main__":
    unittest.main()
