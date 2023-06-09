import os
import unittest
import logging

import numpy as np

from simsopt._core.util import DofLengthMismatchError
from simsopt.field import NormalField

try:
    import py_spec
except ImportError as e:
    py_spec = None

from . import TEST_DIR

logger = logging.getLogger(__name__)


class NormalFieldTests(unittest.TestCase):
    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_initialize_normal_field_from_spec(self):
        """
        Test the initialization of an instance of NormalField using a SPEC input file
        """

        filename = os.path.join(TEST_DIR, 'Dommaschk.sp')

        normal_field = NormalField.from_spec(filename)

        self.assertAlmostEqual(normal_field.get_vns(m=3, n=-1), 1.71466651E-03)
        self.assertAlmostEqual(normal_field.get_vns(m=5, n=1), -3.56991494E-05)

    @unittest.skipIf(py_spec is None, "py_spec not found")
    def test_dofs(self):
        """
        Test access to degrees of freedom, its setter
        """

        # Init from SPEC input file
        filename = os.path.join(TEST_DIR, 'Dommaschk.sp')
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

        normal_field.set_vnc(m=2, n=1, value=0.5)
        self.assertEqual(normal_field.get_vnc(m=2, n=1), 0.5)

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
        filename = os.path.join(TEST_DIR, 'Dommaschk.sp')
        normal_field = NormalField.from_spec(filename)

        # Make names
        names = normal_field._make_names()

        # Check
        self.assertTrue(names[0] == 'vns(0,1)')
        self.assertTrue(names[-1] == 'vns(5,5)')

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
        filename = os.path.join(TEST_DIR, 'Dommaschk.sp')
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
        [self.assertTrue(normal_field.is_free(ii))
         for ii in range(0, normal_field.ndof)]

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
