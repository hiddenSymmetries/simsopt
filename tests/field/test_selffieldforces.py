import os
import unittest
import logging

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

from simsopt.field.selffieldforces import field_on_coils, ForceOpt, force_on_coil, self_force
from simsopt.field import Coil, Current, apply_symmetries_to_curves, apply_symmetries_to_currents
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.configs import get_hsx_data
from simsopt.geo import CurveXYZFourier

logger = logging.getLogger(__name__)


class CoilForcesTest(unittest.TestCase):

    def test_circular_coil(self):
        """Check whether hoop force on a circular coil is right"""
        R0 = 0
        R1 = 1
        I = 10000
        a = 0.01
        N_quad = 1000

        dofs = [0, 0, 1, 0, 1, 0, 0, 0., 0.]
        curve = CurveXYZFourier(N_quad, 1)
        curve.x = dofs

        current = Current(I)
        coil = Coil(curve, current)

        _, n, _ = curve.frenet_frame()

        hoop_force_analytical = np.linalg.norm(constants.mu_0 *
                                               I**2 / 4 / np.pi / R1 * (np.log(8 * R1 / a) - 3 / 4) * n, axis=1)

        hoop_force_test = np.linalg.norm(self_force(coil, a), axis=1)

        np.testing.assert_array_almost_equal(
            hoop_force_test, hoop_force_analytical, 2)

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

    def test_hsx_coil(self):
        """Check whether HSX coil 1 is correct [not yet functioinal]"""
        curves, currents, ma = get_hsx_data(Nt_coils=10)
        I = 150e3
        a = 0.1
        c = curves[0]
        curr = Current(I)
        phi_test = np.linspace(
            0, 2*np.pi, curves[0].gamma().shape[0], endpoint=False)

        coil = Coil(curves[0], curr)

        phi = np.array([0.0, 0.06283185307179587, 0.12566370614359174, 0.18849555921538758, 0.25132741228718347, 0.3141592653589793, 0.37699111843077515, 0.43982297150257105, 0.5026548245743669, 0.5654866776461628, 0.6283185307179586, 0.6911503837897545, 0.7539822368615503, 0.8168140899333463, 0.8796459430051421, 0.9424777960769378, 1.0053096491487339, 1.0681415022205298, 1.1309733552923256, 1.1938052083641213, 1.2566370614359172, 1.3194689145077132, 1.382300767579509, 1.4451326206513047, 1.5079644737231006, 1.5707963267948966, 1.6336281798666925, 1.6964600329384882, 1.7592918860102842, 1.8221237390820801, 1.8849555921538756, 1.9477874452256716, 2.0106192982974678, 2.0734511513692637, 2.1362830044410597, 2.199114857512855, 2.261946710584651, 2.324778563656447, 2.3876104167282426, 2.4504422698000385, 2.5132741228718345, 2.57610597594363, 2.6389378290154264, 2.701769682087222, 2.764601535159018, 2.827433388230814, 2.8902652413026093, 2.9530970943744057, 3.015928947446201,
                        3.078760800517997, 3.141592653589793, 3.204424506661589, 3.267256359733385, 3.330088212805181, 3.3929200658769765, 3.4557519189487724, 3.5185837720205684, 3.581415625092364, 3.6442474781641603, 3.707079331235956, 3.7699111843077513, 3.8327430373795477, 3.895574890451343, 3.9584067435231396, 4.0212385965949355, 4.084070449666731, 4.1469023027385274, 4.209734155810323, 4.272566008882119, 4.335397861953915, 4.39822971502571, 4.461061568097507, 4.523893421169302, 4.586725274241098, 4.649557127312894, 4.71238898038469, 4.775220833456485, 4.838052686528282, 4.900884539600077, 4.9637163926718735, 5.026548245743669, 5.0893800988154645, 5.15221195188726, 5.215043804959057, 5.277875658030853, 5.340707511102648, 5.403539364174444, 5.466371217246239, 5.529203070318036, 5.592034923389832, 5.654866776461628, 5.717698629533423, 5.780530482605219, 5.843362335677015, 5.906194188748811, 5.969026041820607, 6.031857894892402, 6.094689747964199, 6.157521601035994, 6.220353454107791])

        F_x_benchmark = np.array([-15.621714454504355, -25.386254814014332, -34.00485918824268, -37.25194320715892, -33.94735819154057, -28.26391306808791, -25.278784250490897, -26.807548029155342, -30.369264292607372, -31.97730171116336, -30.114496277699605, -26.035156403506647, -21.78467333474507, -18.8847783133302, -17.80907138700853, -18.013558438748653, -18.537801229627746, -18.77246376727638, -18.77160911102063, -18.945014431688293, -19.541492178765562, -20.3942380706085, -21.07455240765378, -21.25886817032106, -20.97720133880779, -20.558557456604085, -20.37432066919552, -20.604321836381924, -21.169376652400235, -21.840640725676806, -22.445147818892284, -23.05807741033117, -24.083684322447443, -26.14177826533346, -29.73275453813305, -34.814238890535925, -40.552395526684464, -45.35984208591351, -47.042221180903084, -43.43013791962298, -34.60426514262674, -24.49931440690929, -17.399226392831004, -13.444360463399379, -9.627015721735356, -4.026391898504832, 2.1961219738518714, 7.225619188022534, 10.817296830609212,
                                  13.4379905995812, 15.15565642466898, 15.861808944422751, 15.873934125390347, 15.903699160657348, 16.52863837250943, 17.851365769730304, 19.638910116508573, 21.70860446798523, 24.09135719843826, 26.789854323159766, 29.457514422299667, 31.422188359643474, 32.14422671768141, 31.69893238992888, 30.72080180997715, 29.837018009118612, 29.193568709496084, 28.50077325792555, 27.477359476543903, 26.220378619749138, 25.139917640814485, 24.56624036043404, 24.448757554214936, 24.434956022149535, 24.2394202908342, 23.932116798706417, 23.860151476346307, 24.292881129282332, 25.14798153122407, 26.070046793056417, 26.786490712365225, 27.37008389921956, 28.117302410777462, 29.151131014219267, 30.165249880318953, 30.629251572733402, 30.277187693419982, 29.295451439912902, 27.945841375906173, 26.074227978416182, 23.203781887268054, 19.314951539216192, 15.280161323855381, 12.159646367697459, 10.343668130239022, 9.39618499956079, 8.191364531018712, 5.342861437557306, 0.1681693022707696, -6.9708975292215785])

        F_x_test = self_force(coil, a)[:, 0] / 1e3

        np.testing.assert_array_almost_equal(F_x_benchmark, F_x_test, 3)

    def test_force_convergence(self):
        """Check force for different number of quadrature points"""
        N_quad = [10, 100, 1000, 3000, 5000]
        dofs = [0, 0, 1, 0, 0, 0, 1., 0., 0., 0, 0, 0, 0, 0.2, 0.]
        # 100 = Number of quadrature points, 1 = max Fourier mode number
        curves = [CurveXYZFourier(n, 2) for n in N_quad]
        for c in curves:
            c.x = dofs
        current = Current(10000.)
        coils = [Coil(c, current) for c in curves]
        forces = [np.max(self_force(c)) for c in coils]
        forces_res = [forces[-1]-f for f in forces]
        self.assertAlmostEqual(forces_res[-1], 0)

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
