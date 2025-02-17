import unittest
import json

import numpy as np

from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curveobjectives import CurveLength, LpCurveTorsion
from simsopt.objectives.utilities import MPIObjective, QuadraticPenalty, MPIOptimizable
from simsopt.geo import parameters
from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON
from simsopt._core.util import parallel_loop_bounds
parameters['jit'] = False
try:
    from mpi4py import MPI
except:
    MPI = None


class UtilityObjectiveTesting(unittest.TestCase):

    def create_curve(self):
        np.random.seed(1)
        rand_scale = 0.01
        order = 4
        nquadpoints = 200
        curve = CurveXYZFourier(nquadpoints, order)
        dofs = np.zeros((curve.dof_size, ))
        dofs[1] = 1.
        dofs[2*order+3] = 1.
        dofs[4*order+3] = 1.
        curve.x = dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape)
        return curve

    def subtest_quadratic_penalty(self, curve, constant, f):
        J = QuadraticPenalty(CurveLength(curve), constant, f)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        err = 1e6
        for i in range(5, 15):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            print("err_new %s" % (err_new))
            assert err_new < 0.6 * err or err_new < 1e-13
            err = err_new

        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_quadratic_penalty(self):
        curve = self.create_curve()
        J = CurveLength(curve)
        for f in ['min', 'max', 'identity']:
            self.subtest_quadratic_penalty(curve, J.J()+0.1, f)
            self.subtest_quadratic_penalty(curve, J.J()-0.1, f)
        with self.assertRaises(Exception):
            self.subtest_quadratic_penalty(curve, J.J()+0.1, 'NotInList')

    @unittest.skipIf(MPI is None, "mpi4py not found")
    def test_mpi_objective(self):
        comm = MPI.COMM_WORLD

        c = self.create_curve()
        Js = [
            CurveLength(c),
            QuadraticPenalty(CurveLength(c)),
            LpCurveTorsion(c, p=2),
            LpCurveTorsion(c, p=2)
        ]
        n = len(Js)

        Jmpi0 = MPIObjective(Js, comm, needs_splitting=True)
        assert abs(Jmpi0.J() - sum(J.J() for J in Js)/n) < 1e-14
        assert np.sum(np.abs(Jmpi0.dJ() - sum(J.dJ() for J in Js)/n)) < 1e-14
        if comm.size == 2:
            Js1subset = Js[:2] if comm.rank == 0 else Js[2:]
            Jmpi1 = MPIObjective(Js1subset, comm, needs_splitting=False)
            assert abs(Jmpi1.J() - sum(J.J() for J in Js)/n) < 1e-14
            assert np.sum(np.abs(Jmpi1.dJ() - sum(J.dJ() for J in Js)/n)) < 1e-14

    @unittest.skipIf(MPI is None, "mpi4py not found")
    def test_mpi_optimizable(self):
        """
        This test checks that the `x` attribute of the surfaces is correctly communicated across the ranks.
        """

        comm = MPI.COMM_WORLD
        for size in [1, 2, 3, 4, 5]:
            surfaces = [SurfaceXYZTensorFourier(mpol=1, ntor=1, stellsym=True) for i in range(size)]

            equal_to = []
            for i in range(size):
                x = np.zeros(surfaces[i].x.size)
                x[:] = i
                equal_to.append(x)

            startidx, endidx = parallel_loop_bounds(comm, len(surfaces))
            for idx in range(startidx, endidx):
                surfaces[idx].x = equal_to[idx]

            mpi_surfaces = MPIOptimizable(surfaces, ["x"], comm)
            for s, sx in zip(mpi_surfaces, equal_to):
                np.testing.assert_allclose(s.x, sx, atol=1e-14)

            # this should raise an exception
            mpi_surfaces = [SurfaceXYZTensorFourier(mpol=1, ntor=1, stellsym=True) for i in range(size)]
            with self.assertRaises(Exception):
                _ = MPIOptimizable(surfaces, ["y"], comm)
