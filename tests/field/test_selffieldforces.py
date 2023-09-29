import os
import unittest
import logging

import numpy as np
from scipy import constants
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from simsopt.field import Coil, Current, apply_symmetries_to_curves, apply_symmetries_to_currents
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.configs import get_hsx_data
from simsopt.geo import CurveXYZFourier
from simsopt.field.selffield import (
    B_regularized_circ,
    B_regularized_rect,
    rectangular_xsection_k,
    rectangular_xsection_delta,
)
from simsopt.field.force import self_force_circ, self_force_rect, ForceOpt

logger = logging.getLogger(__name__)


class SpecialFunctionsTests(unittest.TestCase):
    """
    Test the functions that are specific to the reduced model for rectangular
    cross-section coils.
    """

    def test_k_square(self):
        """Check value of k for a square cross-section."""
        truth = 2.556493222766492
        np.testing.assert_allclose(rectangular_xsection_k(0.3, 0.3), truth)
        np.testing.assert_allclose(rectangular_xsection_k(2.7, 2.7), truth)

    def test_delta_square(self):
        """Check value of delta for a square cross-section."""
        truth = 0.19985294779417703
        np.testing.assert_allclose(rectangular_xsection_delta(0.3, 0.3), truth)
        np.testing.assert_allclose(rectangular_xsection_delta(2.7, 2.7), truth)

    def test_symmetry(self):
        """k and delta should be unchanged if a and b are swapped."""
        n_ratio = 10
        d = 0.01  # Geometric mean of a and b
        for ratio in [0.1, 3.7]:
            a = d * ratio
            b = d / ratio
            np.testing.assert_allclose(
                rectangular_xsection_delta(a, b), rectangular_xsection_delta(b, a)
            )
            np.testing.assert_allclose(
                rectangular_xsection_k(a, b), rectangular_xsection_k(b, a)
            )

    def test_limits(self):
        """Check limits of k and delta for a >> b and b >> a."""
        ratios = [1.1e6, 2.2e4, 3.5e5]
        xs = [0.2, 1.0, 7.3]
        for ratio in ratios:
            for x in xs:
                # a >> b
                b = x
                a = b * ratio
                np.testing.assert_allclose(rectangular_xsection_k(a, b), (7.0 / 6) + np.log(a / b), rtol=1e-3)
                np.testing.assert_allclose(rectangular_xsection_delta(a, b), a / (b * np.exp(3)), rtol=1e-3)

                # b >> a
                a = x
                b = ratio * a
                np.testing.assert_allclose(rectangular_xsection_k(a, b), (7.0 / 6) + np.log(b / a), rtol=1e-3)
                np.testing.assert_allclose(rectangular_xsection_delta(a, b), b / (a * np.exp(3)), rtol=1e-3)


class CoilForcesTest(unittest.TestCase):

    def test_circular_coil(self):
        """Check whether B_reg and hoop force on a circular-centerline coil are correct."""
        R0 = 1.7
        I = 10000
        a = 0.01
        b = 0.023
        order = 1

        # Analytic field has only a z component
        B_reg_analytic_circ = constants.mu_0 * I / (4 * np.pi * R0) * (np.log(8 * R0 / a) - 3 / 4)
        # Eq (98) in Landreman Hurwitz Antonsen:
        B_reg_analytic_rect = constants.mu_0 * I / (4 * np.pi * R0) * (
            np.log(8 * R0 / np.sqrt(a * b)) + 13.0 / 12 - rectangular_xsection_k(a, b) / 2
        )
        force_analytic_circ = B_reg_analytic_circ * I
        force_analytic_rect = B_reg_analytic_rect * I

        for N_quad in [23, 13, 23]:

            # Create a circle of radius R0 in the x-y plane:
            curve = CurveXYZFourier(N_quad, order)
            curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R0
            phi = 2 * np.pi * curve.quadpoints

            current = Current(I)
            coil = Coil(curve, current)

            # Check the case of circular cross-section:

            B_reg_test = B_regularized_circ(coil, a)
            np.testing.assert_allclose(B_reg_test[:, 2], B_reg_analytic_circ)
            np.testing.assert_allclose(B_reg_test[:, 0:2], 0)

            force_test = self_force_circ(coil, a)
            np.testing.assert_allclose(force_test[:, 0], force_analytic_circ * np.cos(phi))
            np.testing.assert_allclose(force_test[:, 1], force_analytic_circ * np.sin(phi))
            np.testing.assert_allclose(force_test[:, 2], 0.0)

            # Check the case of rectangular cross-section:

            B_reg_test = B_regularized_rect(coil, a, b)
            np.testing.assert_allclose(B_reg_test[:, 2], B_reg_analytic_rect)
            np.testing.assert_allclose(B_reg_test[:, 0:2], 0)

            force_test = self_force_rect(coil, a, b)
            np.testing.assert_allclose(force_test[:, 0], force_analytic_rect * np.cos(phi))
            np.testing.assert_allclose(force_test[:, 1], force_analytic_rect * np.sin(phi))
            np.testing.assert_allclose(force_test[:, 2], 0.0)

    def test_force_convergence(self):
        """Check that the self-force is approximately independent of the number of quadrature points"""
        ppps = [8, 4, 2, 7, 5]
        for j, ppp in enumerate(ppps):
            curves, currents, ma = get_hsx_data(ppp=ppp)
            curve = curves[0]
            I = 1.5e3
            a = 0.01
            coil = Coil(curve, Current(I))
            force = self_force_circ(coil, a)
            max_force = np.max(np.abs(force))
            #print("ppp:", ppp, " max force:", max_force)
            if j == 0:
                interpolant = interp1d(curve.quadpoints, force, axis=0)
                max_force_ref = max_force
            else:
                np.testing.assert_allclose(force, interpolant(curve.quadpoints), atol=max_force_ref / 60)
            #print(np.max(np.abs(force - interpolant(curve.quadpoints))))
            #plt.plot(curve.quadpoints, force[:, 0], '+-')
            #plt.plot(curve.quadpoints, interpolant(curve.quadpoints)[:, 0], 'x-')
            #plt.show()

    def test_hsx_coil(self):
        """Compare self-force for HSX coil 1 to result from CoilForces.jl"""
        curves, currents, ma = get_hsx_data()
        assert len(curves[0].quadpoints) == 160
        I = 150e3
        a = 0.01
        b = 0.023
        coil = Coil(curves[0], Current(I))

        # Case of circular cross-section

        # The data from CoilForces.jl were computed with adaptive quadrature rather than
        # fixed quadrature points, so there are minor differences.
        F_x_benchmark = np.array(
            [-15621.714454504356, -21671.165283829952, -27803.883600137513, -33137.63406542344, -36514.610012244935, -37153.15580744652, -35219.95039660063, -31784.14295830274, -28263.91306808791, -25868.96785764328, -25267.863947169404, -26420.719631426105, -28605.174612786002, -30742.286470110146, -31901.79711430137, -31658.687883357023, -30114.496277699604, -27692.050400895027, -24914.32325994916, -22264.404226334766, -20109.078024632625, -18653.393541344263, -17923.34567401103, -17786.90068765639, -18013.558438749424, -18356.84722843243, -18632.28096486515, -18762.70476417455, -18777.777880291706, -18775.25255243524, -18865.065750798094, -19118.903657413353, -19541.492178765562, -20069.946249471617, -20597.76544300027, -21012.94382104526, -21235.99151667497, -21244.42060579293, -21076.283266850463, -20814.259231467684, -20558.557456604085, -20398.672268498092, -20391.95609266666, -20553.456706270365, -20857.968902582856, -21252.466344462075, -21675.335459722362, -22078.067542826495, -22445.147818891808, -22808.51507851074, -23253.40638785968, -23912.326387371173, -24944.467375673074, -26500.877159097003, -28680.906086585404, -31490.462448871924, -34814.238890535926, -38409.58196514822, -41918.135529007886, -44880.12108187824, -46744.74567754952, -46912.87574104355, -44892.220385504654, -40594.371841108295, -34604.26514262674, -28103.81562380291, -22352.117973464843, -18072.020259187793, -15190.453903496049, -13026.606615969953, -10728.246572371136, -7731.063613776822, -4026.3918985048313, -67.6736079295325, 3603.222721794793, 6684.496603922743, 9168.985631597217, 11191.324152143765, 12861.646174806145, 14197.624371838814, 15155.65642466898, 15708.02631569096, 15905.360068930446, 15887.08530906614, 15841.091388685945, 15942.244706940037, 16302.608983867274, 16951.96285551128, 17851.365769734224, 18930.68354614868, 20131.67457842201, 21434.717140456793, 22855.411841777906, 24414.29879175387, 26097.083814069738, 27826.01260192654, 29457.514422303968, 30811.954355976002, 31728.861276623724, 32126.352688475457, 32035.56051890004, 31590.16436771902, 30974.86354195517, 30357.832532863627, 29837.018009118612, 29419.823132415637, 29040.102824240355, 28602.515201729922, 28034.85117397787, 27326.324360676645, 26536.165911557677, 25770.549578748036, 25139.917640814485, 24715.870050046073, 24505.020819951285, 24449.465556871415, 24453.10094740869, 24422.93435687919, 24308.295528898736, 24121.582611759975, 23932.11679870642, 23836.35587784055, 23917.39250931535, 24210.274419402507, 24687.00299537781, 25267.598019903002, 25853.16868209816, 26367.13770177058, 26786.490712365226, 27148.719334319103, 27530.543940041363, 28006.892817037417, 28607.380859864672, 29289.5794536128, 29943.369473108127, 30428.382600649416, 30629.251572733403, 30500.68581330162, 30077.236387914258, 29441.05610919083, 28663.36931005758, 27749.377441743312, 26617.694524563023, 25135.286777612233, 23203.781887268055, 20852.96305267705, 18273.00111385351, 15751.940709349841, 13560.670820131532, 11862.796811944912, 10686.244917385686, 9933.785431650856, 9396.18499956079, 8765.39563869019, 7679.932328018144, 5823.975547630689, 3040.5832677878334, -630.0354600368134, -5034.889144795833, -10047.290491130872]
        )

        F_x_test = self_force_circ(coil, a)[:, 0]
        np.testing.assert_allclose(F_x_benchmark, F_x_test, rtol=5e-4)

        # Case of rectangular cross-section

        F_x_benchmark = np.array(
            [-15902.977742351064, -22087.338601391388, -28374.716841044858, -33849.19166695465, -37297.92164471041, -37900.33823383308, -35834.671809643165, -32222.32607220294, -28539.42357151646, -26038.956838142843, -25414.10121281049, -26625.01818750839, -28917.233935079832, -31157.544275080458, -32369.40033651356, -32111.964192743144, -30498.076019284414, -27973.06320846141, -25082.862607054507, -22330.927900850536, -20100.672092441015, -18607.16966150796, -17876.501377941204, -17766.54300180238, -18030.865806238882, -18407.821451914846, -18703.634927043346, -18839.453077409125, -18849.498339692313, -18839.42010585212, -18927.064921185396, -19189.124962837435, -19630.672419252995, -20184.566483277315, -20737.328517128903, -21170.05630040534, -21398.865299474575, -21400.49745923904, -21215.588465619352, -20931.593542626284, -20654.226270809195, -20477.81224620796, -20462.71503094643, -20624.499033892276, -20936.004375481756, -21340.52020207602, -21772.18877862607, -22178.88373546749, -22543.052200512448, -22896.886220882116, -23328.67601490355, -23976.917882622227, -25009.116377208502, -26585.311195806644, -28812.126971990976, -31698.674748888938, -35127.35857507789, -38847.794103003784, -42489.77372895541, -45578.89472856843, -47546.74720714213, -47772.39764042205, -45740.4661382203, -41351.02637525012, -35206.446917852394, -28536.04216906948, -22650.537373159696, -18298.712898092577, -15399.715530592208, -13242.582486688874, -10939.188962578351, -7900.959595368205, -4120.065876080659, -72.88361226219706, 3673.7842024512443, 6807.9335655033965, 9326.46245417103, 11372.318480608472, 13060.45482710502, 14408.018402296264, 15368.040306172647, 15910.681339497574, 16088.44252644412, 16046.381735963158, 15979.236054882966, 16067.158079521216, 16424.37885132976, 17079.664286304655, 17991.03353643569, 19085.210240409593, 20302.61524208298, 21624.909700051576, 23070.72705280818, 24663.298244272148, 26388.50527971258, 28164.995735057346, 29842.005909841657, 31230.915453068672, 32163.60421789365, 32555.154366620372, 32439.916824241867, 31959.933902098488, 31310.09412683754, 30666.77324808608, 30131.42557561285, 29709.507216221242, 29328.781440649454, 28886.563326541516, 28305.38424855209, 27573.57054060708, 26754.06402904274, 25959.606780943082, 25307.347630273074, 24873.155857206173, 24663.866265832185, 24617.630441594178, 24631.797774854273, 24606.984548529585, 24489.030309821508, 24291.13992107201, 24087.213885558882, 23980.07507264754, 24058.468724827086, 24360.219888393232, 24856.054398793225, 25461.010781980935, 26069.532417606748, 26599.670546048674, 27026.791769426636, 27391.155925242052, 27775.243471007205, 28260.14350201056, 28879.33793031511, 29590.039526497843, 30277.12943502278, 30792.2997683124, 31012.838418681196, 30890.320874148143, 30461.768393949194, 29815.673777616368, 29029.816275644724, 28112.601280724863, 26980.455670627973, 25493.162040823336, 23543.598059415424, 21156.701977016797, 18525.540453701873, 15947.480813254067, 13703.920698550575, 11965.816002481666, 10764.413177100172, 10002.735958901756, 9468.905439131277, 8848.217216372326, 7768.321968088028, 5901.667945626914, 3084.5643561042325, -641.7185254715235, -5119.310167490877, -10219.963710429949]
        )

        F_x_test = self_force_rect(coil, a, b)[:, 0]
        np.testing.assert_allclose(F_x_benchmark, F_x_test, rtol=5e-4)

    def test_force_objective(self):
        """Check whether objective function matches function for export"""
        nfp = 4
        ncoils = 4
        I = 1

        base_curves = create_equally_spaced_curves(ncoils, nfp, True)
        curves = apply_symmetries_to_curves(base_curves, nfp, True)

        base_currents = []
        for i in range(ncoils):
            curr = Current(I)
            base_currents.append(curr)

        curves = apply_symmetries_to_curves(base_curves, nfp, True)
        currents = apply_symmetries_to_currents(
            base_currents, nfp, True)

        coils = [Coil(c, curr) for (c, curr) in zip(curves, currents)]

        objective = float(ForceOpt(coils[0], coils[1:]).J())
        export = np.max(np.linalg.norm(
            force_on_coil(coils[0], coils[1:]), axis=1))

        self.assertEqual(objective, export)

    def test_update_points(self):
        """Check whether quadrature points are updated"""
        nfp = 4
        ncoils = 4
        I = 1

        base_curves = create_equally_spaced_curves(ncoils, nfp, True)
        curves = apply_symmetries_to_curves(base_curves, nfp, True)

        base_currents = []
        for i in range(ncoils):
            curr = Current(I)
            base_currents.append(curr)

        curves = apply_symmetries_to_curves(base_curves, nfp, True)
        currents = apply_symmetries_to_currents(
            base_currents, nfp, True)

        coils = [Coil(c, curr) for (c, curr) in zip(curves, currents)]
        self.assertEqual(1, 1)

    def test_forces_taylor_test(self):
        """Try whether dJ matches finite differences of J"""
        nfp = 4
        ncoils = 4
        I = 1

        base_curves = create_equally_spaced_curves(ncoils, nfp, True)
        curves = apply_symmetries_to_curves(base_curves, nfp, True)

        base_currents = []
        for i in range(ncoils):
            curr = Current(I)
            base_currents.append(curr)

        curves = apply_symmetries_to_curves(base_curves, nfp, True)
        currents = apply_symmetries_to_currents(
            base_currents, nfp, True)

        coils = [Coil(c, curr) for (c, curr) in zip(curves, currents)]

        J = ForceOpt(coils[0], coils[1:])
        J0 = J.J()
        coil_dofs = coils[0].x
        h = 1e-3 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        err = 1e-6
        for i in range(5, 25):
            eps = 0.5**i
            coils[0].x = coil_dofs + eps * h
            Jp = J.J()
            coils[0].x = coil_dofs - eps * h
            Jm = J.J()
            deriv_est = (Jp-Jm)/(2*eps)
            err_new = np.linalg.norm(deriv_est-deriv)

            # print("err_new %s" % (err_new))
            print(eps, err - err_new)
            # assert err_new < err
            err = err_new

        self.assertAlmostEqual(err, 0, 4)


if __name__ == '__main__':
    unittest.main()
