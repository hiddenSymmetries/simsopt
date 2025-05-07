import unittest
import numpy as np
from pathlib import Path

from monty.tempfile import ScratchDir

from simsopt.field import CircularCoil
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve.serial import least_squares_serial_solve
from simsopt import make_optimizable

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()


class Testing(unittest.TestCase):

    def test_circularcoil_current_optimization(self):

        # test optimization of I and r0
        coil = CircularCoil()
        x0 = np.random.rand(coil.dof_size)
        x0[1] = 0
        x0[2] = 0
        x0[3] = 0
        x0[5] = 0
        x0[6] = 0
        coil.x = x0
        coil.fix([1, 2, 3, 5, 6])

        print('Initial coil radius: ', coil.x[0])
        print('Initial coil current: ', coil.x[1])

        points = np.array([[0, 0, 0]])
        coil.set_points(points)

        def get_I(c):
            return c.I

        def get_B(c):
            return c.B()[0][2]

        I_coil = make_optimizable(get_I, coil)
        B_coil = make_optimizable(get_B, coil)

        Bmag = 1.2 * 2 * np.pi / 1.12345
        prob = LeastSquaresProblem.from_tuples([(I_coil.J, 1.2e7, 2e6), (B_coil.J, Bmag, 1.0)])

        with ScratchDir("."):
            least_squares_serial_solve(prob)

        print(' Final coil radius: ', coil.x[0])
        print(' Final coil current: ', coil.x[1])
        assert np.allclose(coil.x, [1.12345, 4.8], atol=1e-6)

    def test_circularcoil_position_optimization(self):

        # test center optimization
        coil = CircularCoil()
        x0 = np.random.rand(coil.dof_size)
        x0[0] = 1.12345
        x0[4] = 4.8
        x0[5] = 0
        x0[6] = 0
        coil.x = x0
        coil.fix([0, 4, 5, 6])

        print('Initial coil position: ', coil.x)

        def Bx(c):
            return c.B()[0][0]

        def By(c):
            return c.B()[0][1]

        def Bz(c):
            return c.B()[0][2]

        Bx_coil = make_optimizable(Bx, coil)
        By_coil = make_optimizable(By, coil)
        Bz_coil = make_optimizable(Bz, coil)

        Bmag = 1.2 * 2 * np.pi / 1.12345
        prob = LeastSquaresProblem.from_tuples(
            [(Bx_coil.J, 0, 1.0),
             (By_coil.J, 0, 1.0),
             (Bz_coil.J, Bmag, 1.0)]
        )

        points = np.array([[1, 1, 1]])
        coil.set_points(points)

        with ScratchDir("."):
            least_squares_serial_solve(prob, ftol=1e-15, xtol=1e-15, gtol=1e-15)

        coil.unfix('y0')
        coil.unfix('z0')

        print(' Final coil position: ', coil.x)
        assert np.allclose(coil.x, np.ones(3), atol=1e-6)

    def test_circularcoil_orientation_optimization(self):

        # test normal optimization
        coil = CircularCoil()
        x0 = np.random.rand(coil.dof_size)
        x0[0] = 1.12345
        x0[1] = 0
        x0[2] = 0
        x0[3] = 0
        x0[4] = 4.8
        coil.x = x0
        coil.fix([0, 1, 2, 3, 4])

        print('Initial coil normal: ', coil.x)

        points = np.array([[0, 0, 0]])
        coil.set_points(points)

        def Bx(c):
            return c.B()[0][0]

        def By(c):
            return c.B()[0][1]

        def Bz(c):
            return c.B()[0][2]

        Bx_coil = make_optimizable(Bx, coil)
        By_coil = make_optimizable(By, coil)
        Bz_coil = make_optimizable(Bz, coil)

        Bmag = np.sqrt(2) / 2 * 1.2 * 2 * np.pi / 1.12345
        prob = LeastSquaresProblem.from_tuples(
            [(Bx_coil.J, Bmag, 1.0),
             (By_coil.J, 0, 1.0),
             (Bz_coil.J, Bmag, 1.0)]
        )

        with ScratchDir("."):
            least_squares_serial_solve(prob)

        print(' Final coil normal: ', coil.x)
        assert np.allclose(coil.x, [0, np.pi / 4], atol=1e-6)
