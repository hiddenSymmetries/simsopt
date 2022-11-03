import unittest
import json

import numpy as np
from monty.json import MontyDecoder, MontyEncoder

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curveobjectives import CurveLength, LpCurveCurvature, LpCurveTorsion
from simsopt.objectives.utilities import MPIObjective, QuadraticPenalty, PrecalculatedObjective
from simsopt.geo import parameters


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

    def subtest_quadratic_penalty(self, curve, threshold):
        J = QuadraticPenalty(CurveLength(curve), threshold)
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
            assert err_new < 0.55 * err or err_new < 1e-13
            err = err_new

        J_str = json.dumps(J, cls=MontyEncoder)
        J_regen = json.loads(J_str, cls=MontyDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_quadratic_penalty(self):
        curve = self.create_curve()
        J = CurveLength(curve)
        self.subtest_quadratic_penalty(curve, J.J()+0.1)
        self.subtest_quadratic_penalty(curve, J.J()-0.1)

    def test_precalculated_objective(self):
        class Myopt(Optimizable):
            def __init__(self):
                Optimizable.__init__(self,x0=[0.0,0.0], ames = ["x0","x1"], fixed=[False,False])

            def J(self):
                print("Performing objective calculation")
                return self.x[0]**2 + np.cos(self.x[1])

            @derivative_dec
            def dJ(self):
                print("Performing derivative calculation")
                return np.array([2*self.x[0], - np.sin(self.x[1])])

        wrapped_obj = Myopt()
        precalc_x = [[0.0,0.0],[1.25,9.0],[0.6,0.5]]
        precalc_J = [1.0, 0.6513697381153231, 1.2375825618903726]
        precalc_dJ = [[0.0, 0.0],[2.5, -0.4121184852417566],[1.2, -0.479425538604203]]
        obj = PrecalculatedObjective(wrapped_obj,precalc_x,precalc_J,precalc_dJ,radius = 1e-6)
        
        # objective function is evaluated away from precalculated points
        obj.x = [7.5,2.3]
        self.assertAlmostEqual(obj.J(), 55.58372397872017)
        
        # precalculated value is used when x is closer than 'radius' to a precalculated x
        # where radius is applied to all coordinates.
        obj.x = [0.99e-6, 0.99e-6]
        # Equal to 1 within delta=1e-14 because precalculate is used
        self.assertAlmostEqual(obj.J(), 1.0, delta = 1e-14)
        # Not equal to the actual value of wrapped_obj at obj.x because precalculated value is used
        self.assertNotAlmostEqual(obj.J(), 1.00000000000049, delta = 1e-14) # 

        obj.x = [1.01e-6, 1.01e-6]
        # Precalculated is not used because we are away from 0
        self.assertAlmostEqual(obj.J(), 1.00000000000051, delta = 1e-14)
        # Not equal to 1 within delta=1e-14 because precalculate is not used
        self.assertNotAlmostEqual(obj.J(), 1.0, delta = 1e-14)

        # test vector radii
        obj2 = PrecalculatedObjective(wrapped_obj,precalc_x,precalc_J,precalc_dJ,radius = [1e-6, 1e-7])
        # not precalculated
        obj2.x = [1.25 + 1.1e-7, 9.0 - 1.1e-7]
        self.assertAlmostEqual(obj2.dJ(partials=True)[0], 2.50000022, delta=1-7)
        # precalculated
        obj2.x = [1.25 + 1.1e-7, 9.0 - 0.99e-7]
        self.assertAlmostEqual(obj2.dJ(partials=True)[0], 2.5, delta=1-7)
        
    def test_mpi_objective(self):
        if MPI is None:
            print("skip test_mpi_objective")
            return
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

if __name__ == "__main__":
    unittest.main()
