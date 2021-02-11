import unittest
import numpy as np
from simsopt.core.new_optimizable import Optimizable
from simsopt.core.new_functions import Identity, Rosenbrock, TestObject2

class Adder(Optimizable):
    """This class defines a minimal object that can be optimized. It has
    n degrees of freedom, and has a function that just returns the sum
    of these dofs. This class is used for testing.
    """
    def __init__(self, n=3, x0=None, dof_names=None, dof_fixed=None):
        self.n = n
        x = x0 if x0 is not None else np.zeros(n)
        super().__init__(x, names=dof_names, fixed=dof_fixed)

    def f(self):
        return np.sum(self._dofs.full_x)


class OptClassWithParents(Optimizable):
    def __init__(self, val, opts=None):
        if opts is None:
            opts = [Adder(3), Adder(2)]
        super().__init__(x0=[val], names=['val'], funcs_in=opts)

    def f(self):
        return (self._dofs.full_x[0] + 2 * self.parents[0]()) \
                / (10.0 + self.parents[1]())


class OptClassWith2LevelParents(Optimizable):
    def __init__(self, val1, val2):
        x = [val1, val2]
        names = ['v1', 'v2']
        funcs = [OptClassWithParents(0.0), Adder(2)]
        super().__init__(x0=x, names=names, funcs_in=funcs)

    def f(self):
        x = self.local_full_x
        v1 = x[0]
        v2 = x[1]
        t = self.parents[0]()
        a = self.parents[1]()
        return v1 + a * np.cos(v2 + t)


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

    def test_dof_size(self):
        # Define Null class
        class EmptyOptimizable(Optimizable):
            def f(self):
                return 0

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
                                        opts=[Identity(x=10, dof_fixed=True),
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
                                        opts=[Identity(x=10, dof_fixed=True),
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
                                        opts=[Identity(x=10, dof_fixed=True),
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
                                        opts=[Identity(x=10, dof_fixed=True),
                                              Adder(3)])
        self.assertEqual(test_obj1.local_full_dof_size, 1)

    def test_dofs(self):
        # Check with leaf type Optimizable objects
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_dofs = adder.dofs
        iden_dofs = iden.dofs
        self.assertAlmostEqual(adder_dofs[0], 1)
        self.assertAlmostEqual(adder_dofs[1], 2)
        self.assertAlmostEqual(adder_dofs[2], 3)
        self.assertEqual(len(iden_dofs), 0)

        adder.dofs = [4, 5, 6]
        self.assertAlmostEqual(adder._dofs.loc['x', '_x'], 4)
        self.assertAlmostEqual(adder._dofs.loc['y', '_x'], 5)
        self.assertAlmostEqual(adder._dofs.loc['z', '_x'], 6)
        with self.assertRaises(ValueError):
            iden.dofs = np.array([11], dtype=float)
        self.assertAlmostEqual(iden.full_dofs[0], 10)

        # Check with Optimizable objects containing parents
        adder2 = Adder(3)
        iden2 = Identity(x=10)
        test_obj1 = OptClassWithParents(10, opts=[iden2, adder2])
        with self.assertRaises(ValueError):
            test_obj1.x = np.array([20])

        test_obj1.x = np.array([20, 4, 5, 6, 25])
        self.assertAlmostEqual(test_obj1._dofs.loc['val', '_x'], 25)
        self.assertAlmostEqual(iden2._dofs.loc['x0', '_x'], 20)
        self.assertAlmostEqual(adder2._dofs.loc['x0', '_x'], 4)
        self.assertAlmostEqual(adder2._dofs.loc['x1', '_x'], 5)
        self.assertAlmostEqual(adder2._dofs.loc['x2', '_x'], 6)

        adder3 = Adder(3)
        test_obj2 = OptClassWithParents(10, opts=[iden, adder3])
        with self.assertRaises(ValueError):
            test_obj2.dofs = np.array([20, 4, 5, 6, 25])

        test_obj2.dofs = np.array([4, 5, 6, 25])
        self.assertAlmostEqual(iden._dofs.loc['x0', '_x'], 10)
        self.assertAlmostEqual(adder3._dofs.loc['x0', '_x'], 4)
        self.assertAlmostEqual(adder3._dofs.loc['x1', '_x'], 5)
        self.assertAlmostEqual(adder3._dofs.loc['x2', '_x'], 6)
        self.assertAlmostEqual(test_obj2._dofs.loc['val', '_x'], 25)

    def test_local_dofs(self):
        # Check with leaf type Optimizable objects
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_dofs = adder.local_dofs
        iden_dofs = iden.local_dofs
        self.assertAlmostEqual(adder_dofs[0], 1)
        self.assertAlmostEqual(adder_dofs[1], 2)
        self.assertAlmostEqual(adder_dofs[2], 3)
        self.assertTrue(len(iden_dofs) == 0)

        adder.local_dofs = [4, 5, 6]
        self.assertAlmostEqual(adder._dofs.loc['x', '_x'], 4)
        self.assertAlmostEqual(adder._dofs.loc['y', '_x'], 5)
        self.assertAlmostEqual(adder._dofs.loc['z', '_x'], 6)
        with self.assertRaises(ValueError):
            iden.local_dofs = np.array([11], dtype=float)
        self.assertAlmostEqual(iden.full_dofs[0], 10)

        # Check with Optimizable objects containing parents
        adder2 = Adder(3)
        iden2 = Identity(x=10)
        test_obj1 = OptClassWithParents(10, opts=[iden2, adder2])
        test_obj1.local_dofs = np.array([25])
        self.assertAlmostEqual(test_obj1._dofs.loc['val', '_x'], 25)

        adder3 = Adder(3)
        test_obj2 = OptClassWithParents(10, opts=[iden, adder3])

        test_obj2.local_dofs = np.array([25])
        self.assertAlmostEqual(test_obj2._dofs.loc['val', '_x'], 25)

    def test_state(self):
        # Check with leaf type Optimizable objects
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_dofs = adder.state
        iden_dofs = iden.state
        self.assertAlmostEqual(adder_dofs[0], 1)
        self.assertAlmostEqual(adder_dofs[1], 2)
        self.assertAlmostEqual(adder_dofs[2], 3)
        self.assertEqual(len(iden_dofs), 0)

        adder.state = [4, 5, 6]
        self.assertAlmostEqual(adder._dofs.loc['x', '_x'], 4)
        self.assertAlmostEqual(adder._dofs.loc['y', '_x'], 5)
        self.assertAlmostEqual(adder._dofs.loc['z', '_x'], 6)
        with self.assertRaises(ValueError):
            iden.state = np.array([11,], dtype=float)
        self.assertAlmostEqual(iden.full_state[0], 10)

        # Check with Optimizable objects containing parents
        adder2 = Adder(3)
        iden2 = Identity(x=10)
        test_obj1 = OptClassWithParents(10, opts=[iden2, adder2])
        with self.assertRaises(ValueError):
            test_obj1.state = np.array([20])

        test_obj1.state = np.array([20, 4, 5, 6, 25])
        self.assertAlmostEqual(test_obj1._dofs.loc['val', '_x'], 25)
        self.assertAlmostEqual(iden2._dofs.loc['x0', '_x'], 20)
        self.assertAlmostEqual(adder2._dofs.loc['x0', '_x'], 4)
        self.assertAlmostEqual(adder2._dofs.loc['x1', '_x'], 5)
        self.assertAlmostEqual(adder2._dofs.loc['x2', '_x'], 6)

        adder3 = Adder(3)
        test_obj2 = OptClassWithParents(10, opts=[iden, adder3])
        with self.assertRaises(ValueError):
            test_obj2.state = np.array([20, 4, 5, 6, 25])

        test_obj2.state = np.array([4, 5, 6, 25])
        self.assertAlmostEqual(iden._dofs.loc['x0', '_x'], 10)
        self.assertAlmostEqual(adder3._dofs.loc['x0', '_x'], 4)
        self.assertAlmostEqual(adder3._dofs.loc['x1', '_x'], 5)
        self.assertAlmostEqual(adder3._dofs.loc['x2', '_x'], 6)
        self.assertAlmostEqual(test_obj2._dofs.loc['val', '_x'], 25)

    def test_local_state(self):
        # Check with leaf type Optimizable objects
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_dofs = adder.local_state
        iden_dofs = iden.local_state
        self.assertAlmostEqual(adder_dofs[0], 1)
        self.assertAlmostEqual(adder_dofs[1], 2)
        self.assertAlmostEqual(adder_dofs[2], 3)
        self.assertTrue(len(iden_dofs) == 0)

        adder.local_state = [4, 5, 6]
        self.assertAlmostEqual(adder._dofs.loc['x', '_x'], 4)
        self.assertAlmostEqual(adder._dofs.loc['y', '_x'], 5)
        self.assertAlmostEqual(adder._dofs.loc['z', '_x'], 6)
        with self.assertRaises(ValueError):
            iden.local_state = np.array([11,], dtype=float)
        self.assertAlmostEqual(iden.full_dofs[0], 10)

        # Check with Optimizable objects containing parents
        adder2 = Adder(3)
        iden2 = Identity(x=10)
        test_obj1 = OptClassWithParents(10, opts=[iden2, adder2])
        test_obj1.local_state = np.array([25])
        self.assertAlmostEqual(test_obj1._dofs.loc['val', '_x'], 25)

        adder3 = Adder(3)
        test_obj2 = OptClassWithParents(10, opts=[iden, adder3])

        test_obj2.local_state = np.array([25])
        self.assertAlmostEqual(test_obj2._dofs.loc['val', '_x'], 25)

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
        self.assertAlmostEqual(adder._dofs.loc['x', '_x'], 4)
        self.assertAlmostEqual(adder._dofs.loc['y', '_x'], 5)
        self.assertAlmostEqual(adder._dofs.loc['z', '_x'], 6)
        with self.assertRaises(ValueError):
            iden.x = np.array([11,], dtype=float)
        self.assertAlmostEqual(iden.full_state[0], 10)

        # Check with Optimizable objects containing parents
        adder2 = Adder(3)
        iden2 = Identity(x=10)
        test_obj1 = OptClassWithParents(10, opts=[iden2, adder2])
        with self.assertRaises(ValueError):
            test_obj1.x = np.array([20])

        test_obj1.x = np.array([20, 4, 5, 6, 25])
        self.assertAlmostEqual(test_obj1._dofs.loc['val', '_x'], 25)
        self.assertAlmostEqual(iden2._dofs.loc['x0', '_x'], 20)
        self.assertAlmostEqual(adder2._dofs.loc['x0', '_x'], 4)
        self.assertAlmostEqual(adder2._dofs.loc['x1', '_x'], 5)
        self.assertAlmostEqual(adder2._dofs.loc['x2', '_x'], 6)

        adder3 = Adder(3)
        test_obj2 = OptClassWithParents(10, opts=[iden, adder3])
        with self.assertRaises(ValueError):
            test_obj2.x = np.array([20, 4, 5, 6, 25])

        test_obj2.x = np.array([4, 5, 6, 25])
        self.assertAlmostEqual(iden._dofs.loc['x0', '_x'], 10)
        self.assertAlmostEqual(adder3._dofs.loc['x0', '_x'], 4)
        self.assertAlmostEqual(adder3._dofs.loc['x1', '_x'], 5)
        self.assertAlmostEqual(adder3._dofs.loc['x2', '_x'], 6)
        self.assertAlmostEqual(test_obj2._dofs.loc['val', '_x'], 25)

    def test_local_x(self):
        # Check with leaf type Optimizable objects
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_dofs = adder.local_x
        iden_dofs = iden.local_x
        self.assertAlmostEqual(adder_dofs[0], 1)
        self.assertAlmostEqual(adder_dofs[1], 2)
        self.assertAlmostEqual(adder_dofs[2], 3)
        self.assertTrue(len(iden_dofs) == 0)

        adder.local_x = [4, 5, 6]
        self.assertAlmostEqual(adder._dofs.loc['x', '_x'], 4)
        self.assertAlmostEqual(adder._dofs.loc['y', '_x'], 5)
        self.assertAlmostEqual(adder._dofs.loc['z', '_x'], 6)
        with self.assertRaises(ValueError):
            iden.local_x = np.array([11,], dtype=float)
        self.assertAlmostEqual(iden.full_dofs[0], 10)

        # Check with Optimizable objects containing parents
        adder2 = Adder(3)
        iden2 = Identity(x=10)
        test_obj1 = OptClassWithParents(10, opts=[iden2, adder2])
        test_obj1.local_x = np.array([25])
        self.assertAlmostEqual(test_obj1._dofs.loc['val', '_x'], 25)

        adder3 = Adder(3)
        test_obj2 = OptClassWithParents(10, opts=[iden, adder3])

        test_obj2.local_x = np.array([25])
        self.assertAlmostEqual(test_obj2._dofs.loc['val', '_x'], 25)

    def test_full_dofs(self):
        # Check with leaf type Optimizable objects
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_full_dofs = adder.full_dofs
        self.assertAlmostEqual(adder_full_dofs[0], 1)
        self.assertAlmostEqual(adder_full_dofs[1], 2)
        self.assertAlmostEqual(adder_full_dofs[2], 3)
        self.assertEqual(len(iden.full_dofs), 1)
        self.assertAlmostEqual(iden.full_dofs[0], 10)

        # Check with Optimizable objects containing parents
        test_obj1 = OptClassWithParents(20, opts=[iden, adder])
        full_dofs = test_obj1.full_dofs
        self.assertTrue(np.allclose(full_dofs, np.array([10, 1, 2, 3, 20])))

        test_obj1.x = np.array([4, 5, 6, 25])
        full_dofs = test_obj1.full_dofs
        self.assertTrue(np.allclose(full_dofs, np.array([10, 4, 5, 6, 25])))

    def test_local_full_dofs(self):
        # Check with leaf type Optimizable objects
        # Check with Optimizable objects containing parents
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_full_dofs = adder.local_full_dofs
        self.assertAlmostEqual(adder_full_dofs[0], 1)
        self.assertAlmostEqual(adder_full_dofs[1], 2)
        self.assertAlmostEqual(adder_full_dofs[2], 3)
        self.assertEqual(len(iden.full_dofs), 1)
        self.assertAlmostEqual(iden.local_full_dofs[0], 10)

        # Check with Optimizable objects containing parents
        test_obj1 = OptClassWithParents(20, opts=[iden, adder])
        local_full_dofs = test_obj1.local_full_dofs
        self.assertTrue(np.allclose(local_full_dofs, np.array([20])))

        test_obj1.x = np.array([4, 5, 6, 25])
        local_full_dofs = test_obj1.local_full_dofs
        self.assertTrue(np.allclose(local_full_dofs, np.array([25])))

    def test_full_state(self):
        # Check with leaf type Optimizable objects
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_full_state = adder.full_state
        self.assertAlmostEqual(adder_full_state[0], 1)
        self.assertAlmostEqual(adder_full_state[1], 2)
        self.assertAlmostEqual(adder_full_state[2], 3)
        self.assertEqual(len(iden.full_state), 1)
        self.assertAlmostEqual(iden.full_state[0], 10)

        # Check with Optimizable objects containing parents
        test_obj1 = OptClassWithParents(20, opts=[iden, adder])
        full_state = test_obj1.full_state
        self.assertTrue(np.allclose(full_state, np.array([10, 1, 2, 3, 20])))

        test_obj1.x = np.array([4, 5, 6, 25])
        full_state = test_obj1.full_state
        self.assertTrue(np.allclose(full_state, np.array([10, 4, 5, 6, 25])))

    def test_local_full_state(self):
        # Check with leaf type Optimizable objects
        # Check with Optimizable objects containing parents
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'])
        iden = Identity(x=10, dof_fixed=True)
        adder_full_state = adder.local_full_state
        self.assertAlmostEqual(adder_full_state[0], 1)
        self.assertAlmostEqual(adder_full_state[1], 2)
        self.assertAlmostEqual(adder_full_state[2], 3)
        self.assertEqual(len(iden.full_state), 1)
        self.assertAlmostEqual(iden.local_full_state[0], 10)

        # Check with Optimizable objects containing parents
        test_obj1 = OptClassWithParents(20, opts=[iden, adder])
        local_full_state = test_obj1.local_full_state
        self.assertTrue(np.allclose(local_full_state, np.array([20])))

        test_obj1.state = np.array([4, 5, 6, 25])
        local_full_state = test_obj1.local_full_state
        self.assertTrue(np.allclose(local_full_state, np.array([25])))

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
        test_obj1 = OptClassWithParents(20, opts=[iden, adder])
        full_x = test_obj1.full_x
        self.assertTrue(np.allclose(full_x, np.array([10, 1, 2, 3, 20])))

        test_obj1.x = np.array([4, 5, 6, 25])
        full_x = test_obj1.full_x
        self.assertTrue(np.allclose(full_x, np.array([10, 4, 5, 6, 25])))

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
        test_obj1 = OptClassWithParents(20, opts=[iden, adder])
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
        adder.unfix_all()
        adder.x = [4, 5, 6]
        self.assertAlmostEqual(adder(), 15.0)
        iden.unfix_all()
        iden.x = [20]
        self.assertAlmostEqual(iden(), 20.0)

        # Call with arguments
        self.assertAlmostEqual(adder([10, 11, 12]), 33)
        self.assertAlmostEqual(iden([20]), 20)

        # Now call without arguments to make sure the previous value is returned
        self.assertAlmostEqual(adder(), 33)
        self.assertAlmostEqual(iden(), 20)

        # Fix dofs and now call
        adder.fix('x')
        self.assertAlmostEqual(adder([1, 2]), 13)
        adder.fix_all()
        self.assertAlmostEqual(adder(), 13)
        iden.fix_all()
        self.assertAlmostEqual(iden(), 20)

        # Check with Optimizable objects containing parents
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=True)
        test_obj1 = OptClassWithParents(20, opts=[iden, adder])
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

    def test_fix_all(self):
        self.iden.fix_all()
        self.adder.fix_all()
        self.rosen.fix_all()

        self.assertEqual(self.iden.dof_size, 0)
        self.assertEqual(self.adder.dof_size, 0)
        self.assertEqual(self.rosen.dof_size, 0)

    def test_unfix(self):
        pass

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
        self.assertAlmostEqual(adder._dofs.loc['x', '_x'], 4)
        self.assertAlmostEqual(adder._dofs.loc['y', '_x'], 5)
        self.assertAlmostEqual(adder._dofs.loc['z', '_x'], 6)
        self.assertAlmostEqual(iden._dofs.loc['x0', '_x'], 10)

        # Check with Optimizable objects containing parents
        adder = Adder(n=3, x0=[1, 2, 3], dof_names=['x', 'y', 'z'],
                      dof_fixed=[True, False, False])
        iden = Identity(x=10, dof_fixed=True)
        test_obj = OptClassWithParents(10, opts=[iden, adder])

        with self.assertRaises(ValueError):
            test_obj.x = np.array([20, 4, 5, 6, 25])

        adder.unfix_all()
        test_obj.x = np.array([4, 5, 6, 25])
        self.assertAlmostEqual(test_obj._dofs.loc['val', '_x'], 25)
        self.assertAlmostEqual(adder._dofs.loc['x', '_x'], 4)
        self.assertAlmostEqual(adder._dofs.loc['y', '_x'], 5)
        self.assertAlmostEqual(adder._dofs.loc['z', '_x'], 6)

        iden.unfix_all()
        test_obj.x = np.array([1, 1, 2, 3, 10])

        self.assertAlmostEqual(iden._dofs.loc['x0', '_x'], 1)
        self.assertAlmostEqual(adder._dofs.loc['x', '_x'], 1)
        self.assertAlmostEqual(adder._dofs.loc['y', '_x'], 2)
        self.assertAlmostEqual(adder._dofs.loc['z', '_x'], 3)
        self.assertAlmostEqual(test_obj._dofs.loc['val', '_x'], 10)

    def test_get_ancestors(self):
        iden = Identity(x=10, dof_fixed=True)
        adder = Adder(n=3, x0=[1, 2, 3])
        self.assertEqual(len(iden.get_ancestors()), 0)
        self.assertEqual(len(adder.get_ancestors()), 0)

        test_obj = OptClassWithParents(10, opts=[iden, adder])
        ancestors = test_obj.get_ancestors()
        self.assertEqual(len(ancestors), 2)
        self.assertIn(iden, ancestors)
        self.assertIn(adder, ancestors)

        test_obj2 = OptClassWith2LevelParents(10, 20)
        ancestors = test_obj2.get_ancestors()
        self.assertEqual(len(ancestors), 4)

       
if __name__ == "__main__":
    unittest.main()
