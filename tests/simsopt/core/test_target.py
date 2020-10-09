import unittest
from simsopt.core.parameter import Parameter
from simsopt.core.target import Target, Identity

def my_function():
    return 7

class TargetTests(unittest.TestCase):

    def another_function(self):
        return self.p1.val + 100

    def test_basic(self):
        """
        Test basic usage
        """

        # Try a Target that depends on 1 Parameter:
        p1 = Parameter(6)
        t = Target({p1}, my_function)
        self.assertEqual(t.evaluate(), 7)
        p = t.parameters
        self.assertIsInstance(p, set)
        self.assertEqual(len(p), 1)
        self.assertIs(p.pop(), p1)

        # Try a Target that depends on 2 Parameters:
        p2 = Parameter(3.14)
        t2 = Target({p1, p2}, my_function)
        self.assertEqual(t2.evaluate(), 7)
        p = t2.parameters
        self.assertIsInstance(p, set)
        self.assertEqual(len(p), 2)
        self.assertEqual(p, {p1, p2})

        # Try a class method:
        self.p1 = p1
        t3 = Target({p1}, self.another_function)
        self.assertEqual(t3.evaluate(), 106)

    def test_exceptions(self):
        """
        Test that exceptions are raised if invalid parameters are
        supplied.
        """

        p1 = Parameter(2)
        p2 = Parameter(6.0)

        # An exception should be raised if we supply something
        # non-callable in the 2nd argument:
        with self.assertRaises(ValueError):
            t3 = Target({p1}, 7)

        # An exception should be raised if we supply something that is
        # not a set in the 1st argument:
        with self.assertRaises(ValueError):
            t4 = Target(p1, my_function)
        with self.assertRaises(ValueError):
            t4 = Target(3, my_function)

        # An exception should be raised if we supply a set for which
        # not all of the entries are Parameter objects:
        with self.assertRaises(ValueError):
            t4 = Target({3.0}, my_function)
        with self.assertRaises(ValueError):
            t4 = Target({p1, 4}, my_function)
        with self.assertRaises(ValueError):
            t4 = Target({p1, 4, p2}, my_function)


class IdentityTests(unittest.TestCase):
    def test_basic(self):

        iden = Identity()
        iden.x.val = 7
        self.assertEqual(iden.target.evaluate(), 7)

        iden.x.val = -3.14
        self.assertAlmostEqual(iden.target.evaluate(), -3.14, places=13)

        iden2 = Identity()
        iden2.x.val = 42
        self.assertEqual(iden2.target.evaluate(), 42)

if __name__ == "__main__":
    unittest.main()
