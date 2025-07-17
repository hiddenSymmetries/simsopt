import unittest
import logging

import numpy as np
from scipy import constants
from scipy.interpolate import interp1d

from simsopt.field import Coil, Current, coils_via_symmetries
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.configs import get_hsx_data, get_ncsx_data
from simsopt.geo import CurveXYZFourier
from simsopt.field.selffield import (
    B_regularized_circ,
    B_regularized_rect,
    rectangular_xsection_k,
    rectangular_xsection_delta,
    regularization_circ,
)
from simsopt.field.force import (
    coil_force,
    self_force_circ,
    self_force_rect,
    MeanSquaredForce,
    LpCurveForce)

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

        # The data from CoilForces.jl were generated with the command
        # CoilForces.reference_HSX_force_for_simsopt_test()
        F_x_benchmark = np.array(
            [-15624.06752062059, -21673.892879345873, -27805.92218896322, -33138.2025931857, -36514.62850757798, -37154.811045050716, -35224.36483811566, -31790.6909934216, -28271.570764376913, -25877.063414550663, -25275.54000792784, -26426.552957555898, -28608.08732785721, -30742.66146788618, -31901.1192650387, -31658.2982018783, -30115.01252455622, -27693.625158453917, -24916.97602450875, -22268.001550194127, -20113.123569572494, -18657.02934190755, -17925.729621918534, -17787.670352261383, -18012.98424762069, -18355.612668419068, -18631.130455525174, -18762.19098176415, -18778.162916012046, -18776.500656205895, -18866.881771744567, -19120.832848894337, -19543.090214569205, -20070.954769137115, -20598.194181114803, -21013.020202255055, -21236.028702664324, -21244.690600996386, -21076.947768954156, -20815.355048694666, -20560.007956111527, -20400.310604802795, -20393.566682281307, -20554.83647318684, -20858.986285059094, -21253.088938981215, -21675.620708707665, -22078.139271497712, -22445.18444801059, -22808.75225496607, -23254.130115531163, -23913.827617806084, -24946.957266144746, -26504.403695291898, -28685.32300927181, -31495.471071978012, -34819.49374359714, -38414.82789487393, -41923.29333627555, -44885.22293635466, -46749.75134352123, -46917.59025432583, -44896.50887106118, -40598.462003586974, -34608.57105847433, -28108.332731765862, -22356.321253373, -18075.405570497107, -15192.820251877345, -13027.925896696135, -10728.68775277632, -7731.104577216556, -4026.458734812997, -67.65800705092924, 3603.7480987311537, 6685.7274727329805, 9170.743233515725, 11193.25631660189, 12863.446736995473, 14199.174999621611, 15157.063376046968, 15709.513692788054, 15907.086239630167, 15889.032882713132, 15843.097529146156, 15944.109516240991, 16304.199171854023, 16953.280592130628, 17852.57440796256, 18932.066168700923, 20133.516941300426, 21437.167716977303, 22858.402963585464, 24417.568974489524, 26100.277202379944, 27828.811426061613, 29459.771430218898, 30813.7836860175, 31730.62350657151, 32128.502820609796, 32038.429339023023, 31593.803847403953, 30979.028723505002, 30362.077268204735, 29840.850204702965, 29422.877198133527, 29042.28057709125, 28604.02774189412, 28036.121314230902, 27327.793860493435, 26538.11580899982, 25773.01411179288, 25142.696104375616, 24718.6066327647, 24507.334842447635, 24451.10991168722, 24454.085831995577, 24423.536258237124, 24308.931868210013, 24122.627773352768, 23933.764307662732, 23838.57162949479, 23919.941100154054, 24212.798983180386, 24689.158548635372, 25269.212310785344, 25854.347267952628, 26368.228758087153, 26787.918123459167, 27150.79244000832, 27533.348289627098, 28010.279752667528, 28611.021858534772, 29293.073660468486, 29946.40958260143, 30430.92513540546, 30631.564524187717, 30503.197269324868, 30080.279217014842, 29444.6938562621, 28667.38229651914, 27753.348490269695, 26621.137071620036, 25137.82866539427, 23205.371963209964, 20853.92976118877, 18273.842305983166, 15753.018584850472, 13562.095187201534, 11864.517807863573, 10688.16332321768, 9935.766441264674, 9398.023223792645, 8766.844594289494, 7680.841209848606, 5824.4042671660145, 3040.702284846631, -630.2054351866387, -5035.57692055936, -10048.785939525675]
        )

        F_x_test = self_force_circ(coil, a)[:, 0]
        np.testing.assert_allclose(F_x_benchmark, F_x_test, rtol=1e-9, atol=0)

        # Case of rectangular cross-section

        F_x_benchmark = np.array(
            [-15905.20099921593, -22089.84960387874, -28376.348489470365, -33849.08438046449, -37297.138833218974, -37901.3580214951, -35838.71064362283, -32228.643120480687, -28546.9118841109, -26046.96628692484, -25421.777194138715, -26630.791911489407, -28919.842325785943, -31157.40078884933, -32368.19957740524, -32111.184287572887, -30498.330514718982, -27974.45692852191, -25085.400672446423, -22334.49737678633, -20104.78648017159, -18610.931535243944, -17878.995292047493, -17767.35330442759, -18030.259902092654, -18406.512856357545, -18702.39969540496, -18838.862854941028, -18849.823944445518, -18840.62799920807, -18928.85330885538, -19191.02138695175, -19632.210519767978, -20185.474968977625, -20737.621297822592, -21169.977809582055, -21398.747768091078, -21400.62658689198, -21216.133558586924, -20932.595132161085, -20655.60793743372, -20479.40191077005, -20464.28582628529, -20625.83431400738, -20936.962932518098, -21341.067527434556, -21772.38656616101, -22178.862986210577, -22542.999300185398, -22897.045487538875, -23329.342412912913, -23978.387795050137, -25011.595805992223, -26588.8272541588, -28816.499234411625, -31703.566987071903, -35132.3971671138, -38852.71510558583, -42494.50815372789, -45583.48852415488, -47551.1577527285, -47776.415427331594, -45743.97982645536, -41354.37991615283, -35210.20495138465, -28540.23742988024, -22654.55869049082, -18301.96907423793, -15401.963398143102, -13243.762349314706, -10939.450828758423, -7900.820612170931, -4120.028225769904, -72.86209546891608, 3674.253747922276, 6809.0803070326565, 9328.115750414787, 11374.122069162511, 13062.097330371573, 14409.383808494194, 15369.251684718018, 15911.988418337934, 16090.021555975769, 16048.21613878066, 15981.151899412167, 16068.941633738388, 16425.88464448961, 17080.88532516404, 17992.129241265648, 19086.46631302506, 20304.322975363317, 21627.219065732254, 23073.563938875737, 24666.38845701993, 26391.47816311481, 28167.521012668185, 29843.93199662863, 31232.367301229497, 32164.969954389788, 32556.923587447265, 32442.446350951064, 31963.284032424053, 31314.01211399212, 30670.79551082286, 30135.039340095944, 29712.330052677768, 29330.71025802117, 28887.8200773726, 28306.412420411067, 27574.83013193789, 26755.843397583598, 25961.936385889934, 25310.01540139794, 24875.789463354584, 24666.066357125907, 24619.136261928328, 24632.619408002214, 24607.413073397413, 24489.503028993608, 24292.044623409187, 24088.74651990258, 23982.195361428472, 24060.929104794097, 24362.6460843878, 24858.082439252874, 25462.457564195745, 26070.50973682213, 26600.547196554344, 27028.01270305341, 27393.03996450607, 27777.872708277075, 28263.357416931998, 28882.7902495421, 29593.307386932454, 30279.887846398404, 30794.507327329207, 31014.791285198782, 30892.485429183558, 30464.50108998591, 29819.03800239511, 29033.577206319136, 28116.32127507844, 26983.626000124084, 25495.394951521277, 23544.852551314456, 21157.350595114454, 18526.131317622883, 15948.394109661942, 13705.248433750054, 11967.480036214449, 10766.293968812726, 10004.685998499026, 9470.706025372589, 8849.607342610005, 7769.149525451194, 5902.017638994769, 3084.6416074691333, -641.878548205229, -5119.944566458021, -10221.371299891642]
        )

        F_x_test = self_force_rect(coil, a, b)[:, 0]
        np.testing.assert_allclose(F_x_benchmark, F_x_test, rtol=1e-9, atol=0)

    def test_force_objectives(self):
        """Check whether objective function matches function for export"""
        nfp = 3
        ncoils = 4
        I = 1.7e4
        regularization = regularization_circ(0.05)

        base_curves = create_equally_spaced_curves(ncoils, nfp, True)
        base_currents = [Current(I) for j in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True)

        # Test LpCurveForce

        p = 2.5
        threshold = 1.0e3
        objective = float(LpCurveForce(coils[0], coils, regularization, p=p, threshold=threshold).J())

        # Now compute the objective a different way, using the independent
        # coil_force function
        gammadash_norm = np.linalg.norm(coils[0].curve.gammadash(), axis=1)
        force_norm = np.linalg.norm(coil_force(coils[0], coils, regularization), axis=1)
        print("force_norm mean:", np.mean(force_norm), "max:", np.max(force_norm))
        objective_alt = (1 / p) * np.sum(np.maximum(force_norm - threshold, 0)**p * gammadash_norm)

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt)

        # Test MeanSquaredForce

        objective = float(MeanSquaredForce(coils[0], coils, regularization).J())

        # Now compute the objective a different way, using the independent
        # coil_force function
        force_norm = np.linalg.norm(coil_force(coils[0], coils, regularization), axis=1)
        objective_alt = np.sum(force_norm**2 * gammadash_norm) / np.sum(gammadash_norm)

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt)

    def test_update_points(self):
        """Confirm that Biot-Savart evaluation points are updated when the
        curve shapes change."""
        nfp = 4
        ncoils = 3
        I = 1.7e4
        regularization = regularization_circ(0.05)

        for objective_class in [MeanSquaredForce, LpCurveForce]:

            base_curves = create_equally_spaced_curves(ncoils, nfp, True, order=2)
            base_currents = [Current(I) for j in range(ncoils)]
            coils = coils_via_symmetries(base_curves, base_currents, nfp, True)

            objective = objective_class(coils[0], coils, regularization)
            old_objective_value = objective.J()
            old_biot_savart_points = objective.biotsavart.get_points_cart()

            # A deterministic random shift to the coil dofs:
            shift = np.array([-0.06797948, -0.0808704, -0.02680599, -0.02775893, -0.0325402,
                              0.04382695, 0.06629717, 0.05050437, -0.09781039, -0.07473099,
                              0.03492035, 0.08474462, 0.06076695, 0.02420473, 0.00388997,
                              0.06992079, 0.01505771, -0.09350505, -0.04637735, 0.00321853,
                              -0.04975992, 0.01802391, 0.09454193, 0.01964133, 0.09205931,
                              -0.09633654, -0.01449546, 0.07617653, 0.03008342, 0.00636141,
                              0.09065833, 0.01628199, 0.02683667, 0.03453558, 0.03439423,
                              -0.07455501, 0.08084003, -0.02490166, -0.05911573, -0.0782221,
                              -0.03001621, 0.01356862, 0.00085723, 0.06887564, 0.02843625,
                              -0.04448741, -0.01301828, 0.01511824])

            objective.x = objective.x + shift
            assert abs(objective.J() - old_objective_value) > 1e-6
            new_biot_savart_points = objective.biotsavart.get_points_cart()
            assert not np.allclose(old_biot_savart_points, new_biot_savart_points)
            # Objective2 is created directly at the new points after they are moved:
            objective2 = objective_class(coils[0], coils, regularization)
            print("objective 1:", objective.J(), "objective 2:", objective2.J())
            np.testing.assert_allclose(objective.J(), objective2.J())

    def test_meansquaredforces_taylor_test(self):
        """Verify that dJ matches finite differences of J"""
        # The Fourier spectrum of the NCSX coils is truncated - we don't need the
        # actual coil shapes from the experiment, just a few nonzero dofs.

        curves, currents, axis = get_ncsx_data(Nt_coils=2)
        coils = [Coil(curve, current) for curve, current in zip(curves, currents)]

        J = MeanSquaredForce(coils[0], coils, regularization_circ(0.05))
        dJ = J.dJ()
        deriv = np.sum(dJ * np.ones_like(J.x))
        dofs = J.x
        h = np.ones_like(dofs)
        err = 100
        for i in range(10, 18):
            eps = 0.5**i
            J.x = dofs + eps * h
            Jp = J.J()
            J.x = dofs - eps * h
            Jm = J.J()
            deriv_est = (Jp - Jm) / (2 * eps)
            err_new = np.abs(deriv_est - deriv) / np.abs(deriv)
            # print("i:", i, "deriv_est:", deriv_est, "deriv:", deriv, "err_new:", err_new, "err:", err, "ratio:", err_new / err)
            np.testing.assert_array_less(err_new, 0.3 * err)
            err = err_new

    def test_lpcurveforces_taylor_test(self):
        """Verify that dJ matches finite differences of J"""
        # The Fourier spectrum of the NCSX coils is truncated - we don't need the
        # actual coil shapes from the experiment, just a few nonzero dofs.

        curves, currents, axis = get_ncsx_data(Nt_coils=2)
        coils = [Coil(curve, current) for curve, current in zip(curves, currents)]

        J = LpCurveForce(coils[0], coils, regularization_circ(0.05), 2.5)
        dJ = J.dJ()
        deriv = np.sum(dJ * np.ones_like(J.x))
        dofs = J.x
        h = np.ones_like(dofs)
        err = 100
        for i in range(10, 18):
            eps = 0.5**i
            J.x = dofs + eps * h
            Jp = J.J()
            J.x = dofs - eps * h
            Jm = J.J()
            deriv_est = (Jp - Jm) / (2 * eps)
            err_new = np.abs(deriv_est - deriv) / np.abs(deriv)
            print("test_lpcurveforces_taylor_test i:", i, "deriv_est:", deriv_est, "deriv:", deriv, "err_new:", err_new, "err:", err, "ratio:", err_new / err)
            np.testing.assert_array_less(err_new, 0.31 * err)
            err = err_new


if __name__ == '__main__':
    unittest.main()
