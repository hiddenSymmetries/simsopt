import unittest

try:
    import numpy as np
except ImportError:
    np = None

from simsopt._core.dev import SimsoptRequires, deprecated
from simsopt._core.optimizable import Optimizable


@SimsoptRequires(np is not None, "numpy is not installed.")
class TestClass(Optimizable):
    def __init__(self):
        x = np.array([1.2, 0.9, -0.4])
        fixed = np.full(3, False)
        super().__init__(x0=x, fixed=fixed)

    def J(self):
        return np.exp(self.full_x[0] ** 2 - np.exp(self.full_x[1]) \
                      + np.sin(self.full_x[2]))

    return_fn_map = {'J': J}


class SimsoptRequiresTest(unittest.TestCase):
    def test_subclass_check(self):
        tf = TestClass()
        self.assertTrue(issubclass(type(tf), Optimizable))


if __name__ == '__main__':
    unittest.main()
