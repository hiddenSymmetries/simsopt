import unittest
import re
import json

import numpy as np
from monty.json import MontyDecoder, MontyEncoder
from monty.serialization import loadfn, dumpfn

from simsopt._core.optimizable import Optimizable, make_optimizable, \
    ScaledOptimizable, OptimizableSum, load, save
from simsopt.objectives.functions import Identity, Rosenbrock, TestObject1, \
    TestObject2, Beale
from simsopt.objectives.functions import Adder as FAdder


class Adder(Optimizable):
    """This class defines a minimal object that can be optimized. It has
    n degrees of freedom, and has a function that just returns the sum
    of these dofs. This class is used for testing.
    """

    def __init__(self, n=3, x0=None, dof_names=None, dof_fixed=None):
        self.n = n
        x = x0 if x0 is not None else np.zeros(n)
        super().__init__(x, names=dof_names, fixed=dof_fixed)

    def sum(self):
        return np.sum(self.local_full_x)

    return_fn_map = {'sum': sum}

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["dof_names"] = d["names"]
        d["dof_fixed"] = d["fixed"]
        del d["names"]
        del d["fixed"]
        d["n"] = self.n
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["n"],
                   d.get("x0", None),
                   d.get("dof_names", None),
                   d.get("dof_fixed", None))


class OptClassWithParents(Optimizable):
    def __init__(self, val, depends_on=None):
        if depends_on is None:
            depends_on = [Adder(3), Adder(2)]
        super().__init__(x0=[val], names=['val'], depends_on=depends_on)

    def f(self):
        return (self.local_full_x[0] + 2 * self.parents[0](child=self)) \
            / (10.0 + self.parents[1](child=self))

    return_fn_map = {'f': f}

    def as_dict(self) -> dict:
        d = super().as_dict()
        del d["x0"]
        del d["names"]
        d['val'] = self.local_full_x[0]
        return d

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d["val"], d["depends_on"])


class N_No(Optimizable):
    """This class defines a minimal object that can be optimized. It has
    n degrees of freedom, and has couple of functions that return the sum
    and product of these dofs. This class is used for testing.
    """

    def __init__(self, n=3, x0=None, dof_names=None, dof_fixed=None):
        self.n = n
        x = x0 if x0 is not None else np.zeros(n)
        super().__init__(x, names=dof_names, fixed=dof_fixed)

    def sum(self):
        return np.sum(self.local_full_x)

    def product(self):
        return np.product(self.local_full_x)

    return_fn_map = {'sum': sum, 'prod': product}


class OptClassWithParentsReturnFns(Optimizable):
    def __init__(self, val):
        self.opt1 = N_No(3, x0=[2, 3, 4])  # Computes to [9, 24]
        self.opt2 = N_No(2, x0=[1, 2])    # Computes to [3, 2]
        super().__init__(x0=[val], names=['val'],
                         depends_on=[self.opt1, self.opt2],
                         opt_return_fns=[['sum'], ['sum', 'prod']])

    # Pay attention to the arguments passed in f1 and f2
    def f1(self):
        return (self.local_full_x[0] + 2 * self.opt1(child=self)) \
            / (10.0 + np.sum(self.opt2(child=self)))

    # If child=self is not passed, full return array is returned, because
    # parent class is not aware of the calling child class
    def f2(self):
        return (self.local_full_x[0] + 2 * self.opt1()[0]) \
            / (10.0 + np.sum(self.opt2()))

    return_fn_map = {'f1': f1, 'f2': f2}


class OptimizableTestsWithParentsReturnFns(unittest.TestCase):
    def setUp(self) -> None:
        self.opt = OptClassWithParentsReturnFns(10)

    def tearDown(self) -> None:
        self.opt = None

    def test_name(self):
        self.assertTrue('OptClassWithParentsReturnFns' in self.opt.name)
        self.assertNotEqual(self.opt.name,
                            OptClassWithParentsReturnFns(10).name)

    def test_hash(self):
        hash1 = hash(self.opt)
        hash2 = hash(OptClassWithParentsReturnFns(10))
        self.assertNotEqual(hash1, hash2)

    def test_eq(self):
        self.assertNotEqual(self.opt, OptClassWithParentsReturnFns(10))

    def test_get_return_fn_names(self):
        ret_fn_names = self.opt.get_return_fn_names()
        self.assertEqual(ret_fn_names[0], 'f1')
        self.assertEqual(ret_fn_names[1], 'f2')

    def test_add_return_fn_by_name(self):
        opt1 = self.opt.opt1
        self.assertEqual(len(opt1.return_fns[self.opt]), 1)
        opt2 = self.opt.opt2
        self.assertEqual(len(opt2.return_fns[self.opt]), 2)
        opt1.add_return_fn(self.opt, 'prod')
        self.assertEqual(len(opt1.return_fns[self.opt]), 2)

    def test_add_return_fn_by_reference(self):
        opt1 = self.opt.opt1
        self.assertEqual(len(opt1.return_fns[self.opt]), 1)
        opt2 = self.opt.opt2
        self.assertEqual(len(opt2.return_fns[self.opt]), 2)
        opt1.add_return_fn(self.opt, opt1.product)
        self.assertEqual(len(opt1.return_fns[self.opt]), 2)

    def test_call(self):
        # Test for leaf nodes
        self.assertAlmostEqual(self.opt.f1(), 28.0/15)
        self.assertAlmostEqual(self.opt.f2(), 28.0/15)
        np.allclose(self.opt(), [28.0/15, 28.0/15])

        # Change parent objects and see if the DOFs are propagated
        opt1 = self.opt.opt1
        opt1.set('x1', 5)
        opt2 = self.opt.opt2
        opt2.set('x1', 4)
        np.allclose(self.opt(), 34.0/24)


class OptClassWithDirectParentFnCalls(Optimizable):
    def __init__(self, val):
        self.opt1 = N_No(3, x0=[2, 3, 4])
        self.opt2 = N_No(2, x0=[1, 2])
        super().__init__(x0=[val], names=['val'],
                         depends_on=[self.opt1, self.opt2])

    # The value returned by f3 should be identical to f1
    def f(self):
        return (self.local_full_x[0] + 2 * self.opt1.sum()) \
            / (10.0 + self.opt2.sum() + self.opt2.product())

    return_fn_map = {'f': f}


class OptimizableTestsWithDirectParentFnCalls(unittest.TestCase):
    def setUp(self) -> None:
        self.opt = OptClassWithDirectParentFnCalls(10)

    def tearDown(self) -> None:
        self.opt = None

    def test_name(self):
        self.assertTrue('OptClassWithDirectParentFnCalls' in self.opt.name)

    def test_equal(self):
        self.assertNotEqual(self.opt.name,
                            OptClassWithDirectParentFnCalls(10).name)

    def test_hash(self):
        hash1 = hash(self.opt)
        hash2 = hash(OptClassWithDirectParentFnCalls(10))
        self.assertNotEqual(hash1, hash2)

    def test_eq(self):
        self.assertNotEqual(self.opt, OptClassWithDirectParentFnCalls(10))

    def test_get_return_fn_names(self):
        ret_fn_names = self.opt.get_return_fn_names()
        self.assertEqual(ret_fn_names[0], 'f')

    @unittest.skip
    def test_add_return_fn_by_name(self):
        """
        This test is not needed
        """
        pass

    @unittest.skip
    def test_add_return_fn_by_reference(self):
        """
        This test is not need for the used function
        """
        pass

    def test_call(self):
        # Test for leaf nodes
        self.assertAlmostEqual(self.opt.f(), 28.0/15)
        np.allclose(self.opt(), 28.0/15)

        # Change parent objects and see if the DOFs are propagated
        opt1 = self.opt.opt1
        opt1.set('x1', 5)
        opt2 = self.opt.opt2
        opt2.set('x1', 4)
        np.allclose(self.opt(), 34.0/24)


class OptClassWithDirectRegisterParentFn(Optimizable):
    def __init__(self, val):
        self.opt1 = N_No(3, x0=[2, 3, 4])
        self.opt2 = N_No(2, x0=[1, 2])
        super().__init__(x0=[val], names=['val'],
                         funcs_in=[self.opt1.sum, self.opt2.sum,
                                   self.opt2.product])

    # Pay attention to the arguments passed in f1 and f2
    def f1(self):
        return (self.local_full_x[0] + 2 * self.opt1(child=self)) \
            / (10.0 + np.sum(self.opt2(child=self)))

    # If child=self is not passed, full return array is returned, because
    # parent class is not aware of the calling child class
    def f2(self):
        return (self.local_full_x[0] + 2 * self.opt1()[0]) \
            / (10.0 + np.sum(self.opt2()))

    return_fn_map = {'f1': f1, 'f2': f2}


class OptimizableTestsWithDirectRegisterParentFns(unittest.TestCase):
    def setUp(self) -> None:
        self.opt = OptClassWithDirectRegisterParentFn(10)

    def tearDown(self) -> None:
        self.opt = None

    def test_name(self):
        self.assertTrue('OptClassWithDirectRegisterParentFn' in self.opt.name)
        self.assertNotEqual(self.opt.name,
                            OptClassWithDirectRegisterParentFn(10).name)

    def test_hash(self):
        hash1 = hash(self.opt)
        hash2 = hash(OptClassWithDirectRegisterParentFn(10))
        self.assertNotEqual(hash1, hash2)

    def test_eq(self):
        self.assertNotEqual(self.opt, OptClassWithDirectRegisterParentFn(10))

    def test_get_return_fn_names(self):
        ret_fn_names = self.opt.get_return_fn_names()
        self.assertEqual(ret_fn_names[0], 'f1')
        self.assertEqual(ret_fn_names[1], 'f2')

    def test_add_return_fn_by_name(self):
        opt1 = self.opt.opt1
        self.assertEqual(len(opt1.return_fns[self.opt]), 1)
        opt2 = self.opt.opt2
        self.assertEqual(len(opt2.return_fns[self.opt]), 2)
        opt1.add_return_fn(self.opt, 'prod')
        self.assertEqual(len(opt1.return_fns[self.opt]), 2)

    def test_add_return_fn_by_reference(self):
        opt1 = self.opt.opt1
        self.assertEqual(len(opt1.return_fns[self.opt]), 1)
        opt2 = self.opt.opt2
        self.assertEqual(len(opt2.return_fns[self.opt]), 2)
        opt1.add_return_fn(self.opt, opt1.product)
        self.assertEqual(len(opt1.return_fns[self.opt]), 2)

    def test_call(self):
        # Test for leaf nodes
        self.assertAlmostEqual(self.opt.f1(), 28.0/15)
        self.assertAlmostEqual(self.opt.f2(), 28.0/15)
        np.allclose(self.opt(), [28.0/15, 28.0/15])

        # Change parent objects and see if the DOFs are propagated
        opt1 = self.opt.opt1
        opt1.set('x1', 5)
        opt2 = self.opt.opt2
        opt2.set('x1', 4)
        np.allclose(self.opt(), 34.0/24)


class OptClassWith2LevelParents(Optimizable):
    def __init__(self, val1, val2):
        x = [val1, val2]
        names = ['v1', 'v2']
        opts = [OptClassWithParents(0.0), Adder(2)]
        super().__init__(x0=x, names=names, depends_on=opts)

    def f(self):
        x = self.local_full_x
        v1 = x[0]
        v2 = x[1]
        t = self.parents[0](self)
        a = self.parents[1](self)
        return v1 + a * np.cos(v2 + t)

    return_fn_map = {'f': f}


class OptWithInputParent(Optimizable):
    def __init__(self, obj):
        super().__init__(depends_on=[obj])


class ThreeDofOpt(Optimizable):
    def __init__(self):
        super().__init__(x0=[1, 2, 3])


class TwoDofOpt(Optimizable):
    def __init__(self):
        super().__init__(x0=[10, 20])


class OptimizableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.iden = Identity(x=10)
        self.adder = Adder(n=3, dof_names=['x', 'y', 'z'])
        self.rosen = Rosenbrock()

    def tearDown(self) -> None:
        self.iden = None
        self.adder = None
        self.rosen = None

    def test_name(self):
        self.assertTrue('Identity' in self.iden.name)
        self.assertTrue('Adder' in self.adder.name)
        self.assertTrue('Rosenbrock' in self.rosen.name)
        self.assertNotEqual(self.iden.name, Identity().name)
        self.assertNotEqual(self.adder.name, Adder().name)
        self.assertNotEqual(self.rosen.name, Rosenbrock().name)

    def test_hash(self):
        hash1 = hash(self.adder)
        hash2 = hash(Adder())
        self.assertNotEqual(hash1, hash2)

    def test_add_parent(self):
        opt1 = Adder(3, x0=[2, 3, 4])
        opt2 = Adder(2, x0=[1, 2])
        opt_with_parents = OptClassWithParents(10, depends_on=[opt1])

        with self.assertRaises(IndexError):  # Missing second parent
            opt_with_parents()

        opt_with_parents.add_parent(1, opt2)
        self.assertAlmostEqual(opt_with_parents(), 28.0/13.0)

    def test_append_parent(self):
        opt1 = Adder(3, x0=[2, 3, 4])
        opt2 = Adder(2, x0=[1, 2])
        opt_with_parents = OptClassWithParents(10, depends_on=[opt1])

        with self.assertRaises(IndexError):  # Missing second parent
            opt_with_parents()

        opt_with_parents.append_parent(opt2)
        self.assertAlmostEqual(opt_with_parents(), 28.0/13.0)

    def test_append_parent_dof_sizes(self):
        # vmec is the parent, prob is the child
        three_dof_opt = ThreeDofOpt()
        opt_with_inp_parent = OptWithInputParent(three_dof_opt)

        # Test dof sizes before adding grandparent
        self.assertEqual(three_dof_opt.dof_size, 3)
        self.assertEqual(opt_with_inp_parent.dof_size, 3)

        two_dof_opt = TwoDofOpt()
        three_dof_opt.append_parent(two_dof_opt)
        # Now two_dof_opt is the grandparent

        # Test dof sizes after adding grandparent
        self.assertEqual(three_dof_opt.dof_size, 5)
        self.assertEqual(three_dof_opt.dof_size, 5)

    def test_pop_parent(self):
        opt1 = Adder(3, x0=[2, 3, 4])
        opt2 = Adder(2, x0=[1, 2])
        opt_with_parents = OptClassWithParents(10, depends_on=[opt1, opt2])

        self.assertEqual(len(opt_with_parents.parents), 2)
        self.assertAlmostEqual(opt_with_parents(), 28.0/13.0)
        opt_with_parents.pop_parent()
        self.assertEqual(len(opt_with_parents.parents), 1)
        with self.assertRaises(IndexError):  # Missing second parent
            opt_with_parents()

    def test_remove_parent(self):
        opt1 = Adder(3, x0=[2, 3, 4])
        opt2 = Adder(2, x0=[1, 2])
        opt_with_parents = OptClassWithParents(10, depends_on=[opt1, opt2])

        self.assertEqual(len(opt_with_parents.parents), 2)
        self.assertAlmostEqual(opt_with_parents(), 28.0/13.0)
        opt_with_parents.remove_parent(opt1)
        self.assertEqual(len(opt_with_parents.parents), 1)
        with self.assertRaises(IndexError):  # Missing second parent
            opt_with_parents()

    def test_dof_size(self):
        # Define Null class
        class EmptyOptimizable(Optimizable):
            def f(self):
                return 0
            return_fn_map = {'f': f}

        opt = EmptyOptimizable()
        self.assertEqual(opt.dof_size, 0)

        self.assertEqual(self.iden.dof_size, 1)
        self.assertEqual(self.adder.dof_size, 3)
        self.assertEqual(self.rosen.dof_size, 2)

        iden2 = Identity(x=10, dof_fixed=True)
        self.assertEqual(iden2.dof_size, 0)

        # Use Optimizable object with parents
        test_obj = OptClassWithParents(10)
        self.assertEqual(test_obj.dof_size, 6)

        test_obj1 = OptClassWithParents(10,
                                        depends_on=[Identity(x=10, dof_fixed=True),
                                                    Adder(n=3, x0=[1, 2, 3])])
        self.assertEqual(test_obj1.dof_size, 4)

    def test_full_dof_size(self):

        # Define Null class
        class EmptyOptimizable(Optimizable):
            def f(self):
                return 0

        opt = EmptyOptimizable()
        self.assertEqual(opt.full_dof_size, 0)

        self.assertEqual(self.iden.full_dof_size, 1)
        self.assertEqual(self.adder.full_dof_size, 3)
        self.assertEqual(self.rosen.full_dof_size, 2)

        iden2 = Identity(x=10, dof_fixed=True)
        self.assertEqual(iden2.full_dof_size, 1)

        # Use Optimizable object with parents
        test_obj = OptClassWithParents(10)
        self.assertEqual(test_obj.full_dof_size, 6)

        test_obj1 = OptClassWithParents(10,
                                        depends_on=[Identity(x=10, dof_fixed=True),
                                                    Adder(3)])
        self.assertEqual(test_obj1.full_dof_size, 5)

    def test_local_dof_size(self):
        # Define Null class
        class EmptyOptimizable(Optimizable):
            def f(self):
                return 0

        opt = EmptyOptimizable()
        self.assertEqual(opt.local_dof_size, 0)

        self.assertEqual(self.iden.local_dof_size, 1)
        self.assertEqual(self.adder.local_dof_size, 3)
        self.assertEqual(self.rosen.local_dof_size, 2)

        iden2 = Identity(x=10, dof_fixed=True)
        self.assertEqual(iden2.local_dof_size, 0)

        # Use Optimizable object with parents
        test_obj = OptClassWithParents(10)
        self.assertEqual(test_obj.local_dof_size, 1)

        test_obj1 = OptClassWithParents(10,
                                        depends_on=[Identity(x=10, dof_fixed=True),
                                                    Adder(3)])
        self.assertEqual(test_obj1.local_dof_size, 1)

    def test_local_full_dof_size(self):
        # Define Null class
        class EmptyOptimizable(Optimizable):
            def f(self):
                return 0

        opt = EmptyOptimizable()
        self.assertEqual(opt.local_full_dof_size, 0)

        self.assertEqual(self.iden.local_full_dof_size, 1)
        self.assertEqual(self.adder.local_full_dof_size, 3)
        self.assertEqual(self.rosen.local_full_dof_size, 2)

        iden2 = Identity(x=10, dof_fixed=True)
        self.assertEqual(iden2.local_full_dof_size, 1)

        # Use Optimizable object with parents
        test_obj = OptClassWithParents(10)
        self.assertEqual(test_obj.local_full_dof_size, 1)

        test_obj1 = OptClassWithParents(10,
                                        depends_on=[Identity(x=10, dof_fixed=True),
                                                    Adder(3)])
        self.assertEqual(test_obj1.local_full_dof_size, 1)

    def test_x(self):
        # Check with leaf type Optimizable objects
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_dofs = adder.x
        iden_dofs = iden.x
        self.assertAlmostEqual(adder_dofs[0], 1)
        self.assertAlmostEqual(adder_dofs[1], 2)
        self.assertAlmostEqual(adder_dofs[2], 3)
        self.assertEqual(len(iden_dofs), 0)

        adder.x = [4, 5, 6]
        self.assertAlmostEqual(adder.local_x[0], 4)
        self.assertAlmostEqual(adder.local_x[1], 5)
        self.assertAlmostEqual(adder.local_x[2], 6)
        with self.assertRaises(ValueError):
            iden.x = np.array([11, ], dtype=float)
        self.assertAlmostEqual(iden.full_x[0], 10)

        # Check with Optimizable objects containing parents
        adder2 = Adder(3)
        iden2 = Identity(x=10)
        test_obj1 = OptClassWithParents(10, depends_on=[iden2, adder2])
        with self.assertRaises(ValueError):
            test_obj1.x = np.array([20])

        test_obj1.x = np.array([4, 5, 6, 20, 25])
        self.assertAlmostEqual(iden2.local_x[0], 20)
        self.assertAlmostEqual(adder2.local_x[0], 4)
        self.assertAlmostEqual(adder2.local_x[1], 5)
        self.assertAlmostEqual(adder2.local_x[2], 6)
        self.assertAlmostEqual(test_obj1.local_x[0], 25)

        adder3 = Adder(3)
        test_obj2 = OptClassWithParents(10, depends_on=[iden, adder3])
        with self.assertRaises(ValueError):
            test_obj2.x = np.array([20, 4, 5, 6, 25])

        test_obj2.x = np.array([4, 5, 6, 25])
        self.assertAlmostEqual(iden.local_full_x[0], 10)
        self.assertAlmostEqual(adder3.local_x[0], 4)
        self.assertAlmostEqual(adder3.local_x[1], 5)
        self.assertAlmostEqual(adder3.local_x[2], 6)
        self.assertAlmostEqual(test_obj2.local_x[0], 25)

    def test_local_x(self):
        # Check with leaf type Optimizable objects
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_x = adder.local_x
        iden_x = iden.local_x
        self.assertAlmostEqual(adder_x[0], 1)
        self.assertAlmostEqual(adder_x[1], 2)
        self.assertAlmostEqual(adder_x[2], 3)
        self.assertTrue(len(iden_x) == 0)

        adder.local_x = [4, 5, 6]
        with self.assertRaises(ValueError):
            iden.local_x = np.array([11, ], dtype=float)
        self.assertAlmostEqual(iden.full_x[0], 10)

        # Check with Optimizable objects containing parents
        adder2 = Adder(3)
        iden2 = Identity(x=10)
        test_obj1 = OptClassWithParents(10, depends_on=[iden2, adder2])
        test_obj1.local_x = np.array([25])

        adder3 = Adder(3)
        test_obj2 = OptClassWithParents(10, depends_on=[iden, adder3])

        test_obj2.local_x = np.array([25])

    def test_full_x(self):
        # Check with leaf type Optimizable objects
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_full_x = adder.full_x
        self.assertAlmostEqual(adder_full_x[0], 1)
        self.assertAlmostEqual(adder_full_x[1], 2)
        self.assertAlmostEqual(adder_full_x[2], 3)
        self.assertEqual(len(iden.full_x), 1)
        self.assertAlmostEqual(iden.full_x[0], 10)

        # Check with Optimizable objects containing parents
        test_obj1 = OptClassWithParents(20, depends_on=[iden, adder])
        full_x = test_obj1.full_x
        self.assertTrue(np.allclose(full_x, np.array([1, 2, 3, 10, 20])))

        test_obj1.x = np.array([4, 5, 6, 25])
        full_x = test_obj1.full_x
        self.assertTrue(np.allclose(full_x, np.array([4, 5, 6, 10, 25])))

    def test_local_full_x(self):
        # Check with leaf type Optimizable objects
        # Check with Optimizable objects containing parents
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_local_full_x = adder.local_full_x
        self.assertAlmostEqual(adder_local_full_x[0], 1)
        self.assertAlmostEqual(adder_local_full_x[1], 2)
        self.assertAlmostEqual(adder_local_full_x[2], 3)
        self.assertEqual(len(iden.local_full_x), 1)
        self.assertAlmostEqual(iden.local_full_x[0], 10)

        # Check with Optimizable objects containing parents
        test_obj1 = OptClassWithParents(20, depends_on=[iden, adder])
        local_full_x = test_obj1.local_full_x
        self.assertTrue(np.allclose(local_full_x, np.array([20])))

        test_obj1.x = np.array([4, 5, 6, 25])
        local_full_x = test_obj1.local_full_x
        self.assertTrue(np.allclose(local_full_x, np.array([25])))

    def test_get(self):
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=True)

        self.assertAlmostEqual(adder.get(0), 1.)
        self.assertAlmostEqual(adder.get('y'), 2.)
        self.assertAlmostEqual(iden.get('x0'), 10.)

    def test_set(self):
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=True)

        adder.set(0, 2)
        adder.set('y', 20)
        iden.set('x0', 20)
        self.assertAlmostEqual(adder.full_x[0], 2)
        self.assertAlmostEqual(adder.full_x[1], 20)
        self.assertAlmostEqual(iden.full_x[0], 20)

    def test_dofs_free_status(self):
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=True)
        test_obj = OptClassWithParents(20, depends_on=[iden, adder])

        adder_status = [False, True, True]
        self.assertTrue(np.equal(adder.dofs_free_status, adder_status).all())
        self.assertTrue(np.equal(iden.dofs_free_status, [False]).all())
        obj_status = [False, True, True, False, True]
        self.assertTrue(np.equal(test_obj.dofs_free_status, obj_status).all())

    def test_local_dofs_free_status(self):
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=True)

        self.assertTrue(
            np.equal(adder.local_dofs_free_status, [False, True, True]).all())
        self.assertTrue(np.equal(iden.local_dofs_free_status, [False]).all())

    def test_call(self):
        # Test for leaf nodes
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        self.assertAlmostEqual(adder(), 6.0)
        adder.fix('y')
        self.assertAlmostEqual(adder(), 6.0)

        iden = Identity(x=10, dof_fixed=True)
        self.assertAlmostEqual(iden(), 10.0)

        # Set dofs and call
        adder.x = [6]
        self.assertAlmostEqual(adder(), 9.0)
        adder.local_unfix_all()
        adder.x = [4, 5, 6]
        self.assertAlmostEqual(adder(), 15.0)
        iden.local_unfix_all()
        iden.x = [20]
        self.assertAlmostEqual(iden(), 20.0)

        # Call with arguments
        self.assertAlmostEqual(adder(x=[10, 11, 12]), 33)
        self.assertAlmostEqual(iden(x=[20]), 20)

        # Now call without arguments to make sure the previous value is returned
        self.assertAlmostEqual(adder(), 33)
        self.assertAlmostEqual(iden(), 20)

        # Fix dofs and now call
        adder.fix('x')
        self.assertAlmostEqual(adder([1, 2]), 13)
        adder.local_fix_all()
        self.assertAlmostEqual(adder(), 13)
        iden.local_fix_all()
        self.assertAlmostEqual(iden(), 20)

        # Check with Optimizable objects containing parents
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=True)
        test_obj1 = OptClassWithParents(20, depends_on=[iden, adder])
        # Value returned by test_obj1 is (val + 2*iden())/(10.0 + adder())
        self.assertAlmostEqual(test_obj1(), 2.5)

        # Set the parents nodes' x and call
        adder.x = [4, 5]
        self.assertAlmostEqual(test_obj1(), 2.0)

        # Set the dofs and call
        test_obj1.x = np.array([14, 15, 30])
        self.assertAlmostEqual(test_obj1(), 1.25)

        # Set only the node  local dofs and call
        test_obj1.local_x = [20]
        self.assertAlmostEqual(test_obj1(), 1.0)

        # Call with arguments
        self.assertAlmostEqual(test_obj1([2, 3, 20]), 2.5)
        # Followed by Call with no arguments
        self.assertAlmostEqual(test_obj1(), 2.5)

    def test_bounds(self):
        pass

    def test_local_bounds(self):
        pass

    def test_lower_bounds(self):
        pass

    def test_local_lower_bounds(self):
        pass

    def test_upper_bounds(self):
        pass

    def test_local_upper_bounds(self):
        pass

    def test_dof_names(self):
        iden = Identity(x=10, dof_fixed=True)
        adder = Adder(n=3, x0=[1, 2, 3])
        self.assertEqual(len(iden.dof_names), 0)
        self.assertEqual(len(adder.dof_names), 3)
        patt = re.compile("Adder\d+:x\d+")
        for name in adder.dof_names:
            self.assertTrue(patt.match(name))

        patt1 = "Adder\d+:x\d+"
        patt2 = "OptClassWithParents\d+:val"
        comb_patt = re.compile("|".join([patt1, patt2]))
        test_obj = OptClassWithParents(10, depends_on=[iden, adder])
        self.assertEqual(len(test_obj.dof_names), 4)
        for name in test_obj.dof_names:
            self.assertTrue(comb_patt.match(name))

        test_obj.fix('val')
        self.assertEqual(len(test_obj.dof_names), 3)
        for name in test_obj.dof_names:
            self.assertTrue(comb_patt.match(name))
        exc_patt = re.compile("OptClassWithParents\d+:val")
        for name in test_obj.dof_names:
            self.assertFalse(exc_patt.match(name))

        adder.fix('x1')
        self.assertEqual(len(test_obj.dof_names), 2)
        for name in test_obj.dof_names:
            self.assertTrue(comb_patt.match(name))
        exc_patt = re.compile("Adder\d+:x1")
        for name in test_obj.dof_names:
            self.assertFalse(exc_patt.match(name))

        test_obj2 = OptClassWith2LevelParents(10, 20)
        patt1 = "Adder\d+:x\d+"
        patt2 = "OptClassWithParents\d+:val"
        patt3 = "OptClassWith2LevelParents\d+:v\d"
        comb_patt = re.compile("|".join([patt1, patt2, patt3]))
        self.assertEqual(len(test_obj2.dof_names), 10)
        for name in test_obj2.dof_names:
            self.assertTrue(comb_patt.match(name))

        test_obj2.fix(0)
        self.assertEqual(len(test_obj2.dof_names), 9)
        for name in test_obj2.dof_names:
            self.assertTrue(comb_patt.match(name))
        exc_patt = re.compile("OptClassWith2LevelParents\d+:v1")
        for name in test_obj.dof_names:
            self.assertFalse(exc_patt.match(name))

    def test_full_dof_names(self):
        iden = Identity(x=10, dof_fixed=True)
        adder = Adder(n=3, x0=[1, 2, 3])
        self.assertEqual(len(iden.full_dof_names), 1)
        self.assertEqual(len(adder.full_dof_names), 3)

        test_obj = OptClassWithParents(10, depends_on=[iden, adder])
        self.assertEqual(len(test_obj.full_dof_names), 5)
        test_obj.fix('val')
        self.assertEqual(len(test_obj.full_dof_names), 5)
        adder.fix('x1')
        self.assertEqual(len(test_obj.full_dof_names), 5)

        test_obj2 = OptClassWith2LevelParents(10, 20)
        self.assertEqual(len(test_obj2.full_dof_names), 10)
        test_obj2.fix(0)
        self.assertEqual(len(test_obj2.full_dof_names), 10)
        patt1 = "Adder\d+:x\d+"
        patt2 = "OptClassWithParents\d+:val"
        patt3 = "OptClassWith2LevelParents\d+:v\d"
        comb_patt = re.compile("|".join([patt1, patt2, patt3]))
        for name in test_obj2.full_dof_names:
            self.assertTrue(comb_patt.match(name))

    def test_local_dof_names(self):
        # Test in DOFs class is sufficient
        pass

    def test_local_full_dof_names(self):
        # Test in DOFs class is sufficient
        pass

    def test_is_fixed(self):
        iden = Identity(x=10, dof_fixed=True)
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        self.assertTrue(adder.is_fixed(0))
        self.assertTrue(adder.is_fixed('x'))
        self.assertFalse(adder.is_fixed(1))
        self.assertFalse(adder.is_fixed('y'))
        self.assertTrue(iden.is_fixed(0))
        self.assertTrue(iden.is_fixed('x0'))

    def test_is_free(self):
        iden = Identity(x=10, dof_fixed=True)
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        self.assertFalse(adder.is_free(0))
        self.assertFalse(adder.is_free('x'))
        self.assertTrue(adder.is_free(1))
        self.assertTrue(adder.is_free('y'))
        self.assertFalse(iden.is_free(0))
        self.assertFalse(iden.is_free('x0'))

    def test_fix(self):
        self.iden.fix(0)
        self.adder.fix('x')
        self.rosen.fix('y')

        self.assertEqual(self.iden.dof_size, 0)
        self.assertEqual(self.adder.dof_size, 2)
        self.assertEqual(self.rosen.dof_size, 1)

    def test_local_fix_all(self):
        self.iden.local_fix_all()
        self.adder.local_fix_all()
        self.rosen.local_fix_all()

        self.assertEqual(self.iden.dof_size, 0)
        self.assertEqual(self.adder.dof_size, 0)
        self.assertEqual(self.rosen.dof_size, 0)

    def test_fix_all(self):
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=False)
        adder_x = adder.x
        iden_x = iden.x
        self.assertEqual(len(adder_x), 2)
        self.assertAlmostEqual(adder_x[0], 2)
        self.assertAlmostEqual(adder_x[1], 3)
        self.assertEqual(len(iden_x), 1)

        iden.fix_all()
        adder.fix_all()
        with self.assertRaises(ValueError):
            iden.x = [10]
        with self.assertRaises(ValueError):
            adder.x = [4, 5, 6]

        self.assertEqual(iden.dof_size, 0)
        self.assertEqual(adder.dof_size, 0)

        # Check with Optimizable objects containing parents
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=False)
        test_obj = OptClassWithParents(10, depends_on=[iden, adder])

        test_x = test_obj.x
        self.assertEqual(len(test_x), 4)
        test_obj.fix_all()

        with self.assertRaises(ValueError):
            test_obj.x = np.array([20, 5, 6, 25])

        self.assertEqual(test_obj.dof_size, 0)
        self.assertEqual(adder.dof_size, 0)
        self.assertEqual(iden.dof_size, 0)

    def test_unfix(self):
        pass

    def test_local_unfix_all(self):
        # Test with leaf nodes
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=True)
        adder_x = adder.x
        iden_x = iden.x
        self.assertEqual(len(adder_x), 2)
        self.assertEqual(adder.dof_size, 2)
        self.assertAlmostEqual(adder_x[0], 2)
        self.assertAlmostEqual(adder_x[1], 3)
        self.assertEqual(len(iden_x), 0)

        with self.assertRaises(ValueError):
            iden.x = [10]
        with self.assertRaises(ValueError):
            adder.x = [4, 5, 6]

        iden.local_unfix_all()
        adder.local_unfix_all()
        iden.x = [10]
        adder.x = [4, 5, 6]
        self.assertEqual(iden.dof_size, 1)
        self.assertEqual(adder.dof_size, 3)

        # Check with Optimizable objects containing parents
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=True)
        test_obj = OptClassWithParents(10, depends_on=[iden, adder])

        with self.assertRaises(ValueError):
            test_obj.x = np.array([20, 4, 5, 6, 25])

        adder.local_unfix_all()
        test_obj.x = np.array([4, 5, 6, 25])
        self.assertAlmostEqual(adder.local_full_x[0], 4)
        self.assertAlmostEqual(adder.local_full_x[1], 5)
        self.assertAlmostEqual(adder.local_full_x[2], 6)
        self.assertAlmostEqual(test_obj.local_full_x[0], 25)

        iden.local_unfix_all()
        test_obj.x = np.array([1, 2, 3, 1, 10])

        self.assertAlmostEqual(adder.local_full_x[0], 1)
        self.assertAlmostEqual(adder.local_full_x[1], 2)
        self.assertAlmostEqual(adder.local_full_x[2], 3)
        self.assertAlmostEqual(iden.local_full_x[0], 1)
        self.assertAlmostEqual(test_obj.local_full_x[0], 10)

    def test_unfix_all(self):
        # Test with leaf nodes
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=True)
        adder_x = adder.x
        iden_x = iden.x
        self.assertEqual(len(adder_x), 2)
        self.assertEqual(adder.dof_size, 2)
        self.assertAlmostEqual(adder_x[0], 2)
        self.assertAlmostEqual(adder_x[1], 3)
        self.assertEqual(len(iden_x), 0)

        with self.assertRaises(ValueError):
            iden.x = [10]
        with self.assertRaises(ValueError):
            adder.x = [4, 5, 6]

        iden.unfix_all()
        adder.unfix_all()
        iden.x = [10]
        adder.x = [4, 5, 6]
        self.assertEqual(iden.dof_size, 1)
        self.assertEqual(adder.dof_size, 3)

        # Check with Optimizable objects containing parents
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=True)
        test_obj = OptClassWithParents(10, depends_on=[iden, adder])
        test_obj.fix('val')

        with self.assertRaises(ValueError):
            test_obj.x = np.array([20, 4, 5, 6, 25])

        test_obj.unfix_all()
        test_obj.x = np.array([4, 5, 6, 20, 25])
        print(iden.x)
        self.assertAlmostEqual(adder.local_full_x[0], 4)
        self.assertAlmostEqual(adder.local_full_x[1], 5)
        self.assertAlmostEqual(adder.local_full_x[2], 6)
        self.assertAlmostEqual(iden.local_full_x[0], 20)
        self.assertAlmostEqual(test_obj.local_full_x[0], 25)

    def test_get_ancestors(self):
        iden = Identity(x=10, dof_fixed=True)
        adder = Adder(n=3, x0=[1, 2, 3])
        self.assertEqual(len(iden._get_ancestors()), 0)
        self.assertEqual(len(adder._get_ancestors()), 0)

        test_obj = OptClassWithParents(10, depends_on=[iden, adder])
        ancestors = test_obj._get_ancestors()
        self.assertEqual(len(ancestors), 2)
        self.assertIn(iden, ancestors)
        self.assertIn(adder, ancestors)

        test_obj2 = OptClassWith2LevelParents(10, 20)
        ancestors = test_obj2._get_ancestors()
        self.assertEqual(len(ancestors), 4)

    def test_plot(self):
        """
        Verify that a DAG can be plotted.
        The ``show`` argument is set to ``False`` so the
        tests do not require human intervention to close plot windows.
        However, if you do want to actually display the figure, you
        can change ``show`` to ``True`` in the first line of this
        function.
        """
        show = False

        try:
            import matplotlib
        except ImportError:
            return
        try:
            import networkx
        except ImportError:
            return
        try:
            import pygraphviz
        except ImportError:
            return

        # optimizable with no parents
        adder = Adder(n=3, x0=[1.0, 2.0, 3.0])
        G1, pos1 = adder.plot_graph(show=show)

        # optimizable with two parents
        opt1 = OptClassWithParentsReturnFns(10)
        G2, pos2 = opt1.plot_graph(show=show)

        # optimizable with many parents
        opt2 = OptClassWith2LevelParents(10, 20)
        G3, pos3 = opt2.plot_graph(show=show)


class OptClassExternalDofs(Optimizable):
    def __init__(self):
        self.vals = [1, 2]
        Optimizable.__init__(self, external_dof_setter=OptClassExternalDofs.set_dofs,
                             x0=self.get_dofs())

    def get_dofs(self):
        return self.vals

    def set_dofs(self, x):
        self.vals = x

    def recompute_bell(self, parent=None):
        pass


class OptimizableTestsExternalDofs(unittest.TestCase):
    def setUp(self) -> None:
        self.opt = OptClassExternalDofs()

    def tearDown(self) -> None:
        self.opt = None

    def test_get_dofs(self):
        vals = self.opt.get_dofs()
        self.assertTrue((vals == np.array([1, 2])).all())

    def test_set_dofs(self):
        self.opt.set_dofs([3, 4])
        vals = self.opt.get_dofs()
        self.assertTrue((vals == np.array([3, 4])).all())

    def test_set_x(self):
        self.opt.x = [3, 4]
        vals = self.opt.get_dofs()
        self.assertTrue((vals == np.array([3, 4])).all())

    def test_set_local_x(self):
        self.opt.local_x = [3, 4]
        vals = self.opt.get_dofs()
        self.assertTrue((vals == np.array([3, 4])).all())


class TestMakeOptimizable(unittest.TestCase):
    def setUp(self) -> None:
        def arb_fun_dofs_noopts(a, b, c):
            return a ** 2 + 2 * b ** 2 + 3 * c ** 2 - 10

        def arb_fun_nodofs_opts(adder):
            return adder.sum()**2 - 10

        def arb_fun_dofs_opts(a, b, adder):
            return a**2 + b**2 + adder.sum()**2 - 10

        self.arb_fun_dofs_noopts = arb_fun_dofs_noopts
        self.arb_fun_nodofs_opts = arb_fun_nodofs_opts
        self.arb_fun_dofs_opts = arb_fun_dofs_opts

    def test_arb_func_dofs_noopts(self):
        x, y, z = 1, 2, 3
        opt = make_optimizable(self.arb_fun_dofs_noopts,
                               x, y, z,
                               dof_indicators=["dof", "dof", "dof"])
        self.assertAlmostEqual(opt.J(), 26.0)
        opt.x = np.array([1.2, 0.8, 0.5])
        self.assertAlmostEqual(opt.J(), -6.53)

    def test_arb_func_nodofs_opts(self):
        adder = Adder(n=3, x0=[1.0, 2.0, 3.0])
        opt = make_optimizable(self.arb_fun_nodofs_opts,
                               adder,
                               dof_indicators=["opt"])
        self.assertAlmostEqual(opt.J(), 26.0)
        x = opt.x   # Length of x is 3
        opt.x = x / 2.0
        self.assertAlmostEqual(opt.J(), -1.0)

        # When dof_indicators argument is not passed only opts and non_dofs are
        # considered. Below a and b are treated as non-dofs and the adder should
        # be recognized as optimizable object
        a = 2.0
        b = 3.0
        adder = Adder(n=3, x0=[1.0, 2.0, 3.0])
        opt = make_optimizable(self.arb_fun_dofs_opts, a, b, adder)
        self.assertAlmostEqual(opt.J(), 39.0)
        x = opt.x   # Length of x is 3
        self.assertEqual(len(x), 3)
        opt.x = x / 2.0
        self.assertAlmostEqual(opt.J(), 12.0)

    def test_arb_func_dofs_opts(self):
        # Below a is passed as dof b is passed as non-dof and the adder is
        # passed as optimizable object
        a = 2.0
        b = 3.0
        adder = Adder(n=3, x0=[1.0, 2.0, 3.0])
        opt = make_optimizable(self.arb_fun_dofs_opts,
                               a, b, adder,
                               dof_indicators=['dof', 'non-dof', 'opt'])
        self.assertAlmostEqual(opt.J(), 39.0)
        x = opt.x  # Length of x is 4
        self.assertEqual(len(x), 4)
        opt.x = x / 2.0
        self.assertAlmostEqual(opt.J(), 9.0)


class TestOptimizableSerialize(unittest.TestCase):
    """
    Test the serialization of the Optimizable class based on as_dict and
    from_dict methods using various sub-classes
    """

    def test_adder_serialize(self):
        adder_orig = FAdder(n=3, x0=[1, 2, 3], names=["x", "y", "z"],
                            fixed=[True, False, True])
        s = json.dumps(adder_orig, cls=MontyEncoder)
        adder = json.loads(s, cls=MontyDecoder)
        self.assertEqual(adder.n, adder_orig.n)
        self.assertTrue(np.allclose(adder.full_x, adder_orig.full_x))
        self.assertTrue(np.array_equal(adder.dofs_free_status,
                                       adder_orig.dofs_free_status))
        self.assertEqual(adder.local_full_dof_names,
                         adder_orig.local_full_dof_names)

    def test_identity_serialize(self):
        iden_orig = Identity(x=10.0, dof_name="x", dof_fixed=False)
        s = json.dumps(iden_orig, cls=MontyEncoder)
        iden = json.loads(s, cls=MontyDecoder)
        self.assertAlmostEqual(iden.x[0], iden_orig.x[0])
        self.assertEqual(iden.local_full_dof_names[0],
                         iden_orig.local_full_dof_names[0])
        self.assertEqual(iden.dofs_free_status[0],
                         iden_orig.dofs_free_status[0])

    def test_rosenbrock_serialize(self):
        r_orig = Rosenbrock(b=100.0, x=10.0, y=20.0)
        s = json.dumps(r_orig, cls=MontyEncoder)
        r = json.loads(s, cls=MontyDecoder)
        self.assertAlmostEqual(r.term1, r_orig.term1)
        self.assertAlmostEqual(r.term2, r_orig.term2)

    def test_twolevel_serialize(self):
        adder1 = FAdder(n=3, x0=[1, 2, 3], names=["x", "y", "z"],
                        fixed=[True, False, True])
        adder2 = FAdder(n=2, x0=[10, 11], names=["a", "b"], fixed=[True, False])
        test_opt_orig = TestObject1(100.0, depends_on=[adder1, adder2])
        s = json.dumps(test_opt_orig, cls=MontyEncoder)
        test_opt = json.loads(s, cls=MontyDecoder)
        self.assertAlmostEqual(test_opt.f(), test_opt_orig.f())

    def test_scaled_optimizer_serialize(self):
        beale = Beale(x0=[2.2, 3.3])
        scaled_beale = ScaledOptimizable(2.0, beale)
        s = json.dumps(scaled_beale, cls=MontyEncoder)
        scaled_beale_regen = json.loads(s, cls=MontyDecoder)
        self.assertTrue(np.allclose(scaled_beale_regen.J(), 2*beale.J()))

    def test_optimizable_sum_serializer(self):
        adder1 = FAdder(n=3, x0=[1, 2, 3], names=["x", "y", "z"],
                        fixed=[True, False, True])
        adder2 = FAdder(n=2, x0=[10, 11], names=["a", "b"], fixed=[True, False])
        opt_sum = OptimizableSum(opts=[adder1, adder2])
        s = json.dumps(opt_sum, cls=MontyEncoder)
        opt_sum_regen = json.loads(s, cls=MontyDecoder)
        self.assertAlmostEqual(opt_sum_regen.J(), adder1.J() + adder2.J())

    def test_load_save(self):
        import tempfile
        from pathlib import Path

        adder1 = FAdder(n=3, x0=[1, 2, 3], names=["x", "y", "z"],
                        fixed=[True, False, True])
        adder2 = FAdder(n=2, x0=[10, 11], names=["a", "b"], fixed=[True, False])
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "adders.json"
            save([adder1, adder2], fpath, indent=2)
            self.assertTrue(fpath.is_file())

            adders = load(fpath)
            self.assertAlmostEqual(adder1.J(), adders[0].J())
            self.assertAlmostEqual(adder2.J(), adders[1].J())

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "adder.json"
            adder_str = adder1.save(fpath, indent=2)
            self.assertTrue(fpath.is_file())

            adder1_str_regen = FAdder.from_str(adder_str)
            self.assertAlmostEqual(adder1.J(), adder1_str_regen.J())
            adder1_file_regen = FAdder.from_file(fpath)
            self.assertAlmostEqual(adder1.J(), adder1_file_regen.J())

        adder_str1 = adder1.save(fmt='json', indent=2)
        adder1_str_regen1 = Optimizable.from_str(adder_str1)
        self.assertAlmostEqual(adder1.J(), adder1_str_regen1.J())


if __name__ == "__main__":
    unittest.main()
