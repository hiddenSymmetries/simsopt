import unittest
from simsopt.shape import *

class ShapeTests(unittest.TestCase):
    def test_init(self):
        """
        This test case checks the most common use cases.
        """
        # Try initializing a Shape without or with the optional
        # arguments:
        s = Shape()
        self.assertEqual(s.nfp.val, 1)
        self.assertTrue(s.stelsym.val)

        s = Shape(nfp=3)
        self.assertEqual(s.nfp.val, 3)
        self.assertTrue(s.stelsym.val, True)

        s = Shape(stelsym=False)
        self.assertEqual(s.nfp.val, 1)
        self.assertFalse(s.stelsym.val)

        # Now let's check that we can change nfp and stelsym by
        # setting them to new Parameter instances:
        s.nfp = Parameter(5)
        self.assertEqual(s.nfp.val, 5)
        self.assertFalse(s.stelsym.val)

        s.stelsym = Parameter(True)
        self.assertEqual(s.nfp.val, 5)
        self.assertTrue(s.stelsym.val)

        # Now let's check that we can change nfp and stelsym by
        # directly manipulating the val attributes:
        s.nfp.val = 7
        self.assertEqual(s.nfp.val, 7)
        self.assertTrue(s.stelsym.val)

        s.stelsym.val = False
        self.assertEqual(s.nfp.val, 7)
        self.assertFalse(s.stelsym.val)

if __name__ == "__main__":
    unittest.main()
