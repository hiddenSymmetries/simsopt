import unittest
import logging

import numpy as np
from scipy import constants
from scipy.interpolate import interp1d
from scipy.special import ellipk, ellipe

from simsopt.field import Coil, Current, coils_via_symmetries
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.configs import get_hsx_data
from simsopt.geo import CurveXYZFourier, CurvePlanarFourier
from simsopt.field.selffield import (
    B_regularized_circ,
    B_regularized_rect,
    rectangular_xsection_k,
    rectangular_xsection_delta,
    regularization_circ,
    regularization_rect,
)
from simsopt.field import (
    coil_force,
    coil_torque,
    self_force_circ,
    self_force_rect,
    coil_coil_inductances_pure,
    coil_coil_inductances_full_pure,
    coil_coil_inductances_inv_pure,
    NetFluxes,
    TVE,
    MeanSquaredForce_deprecated,
    LpCurveTorque_deprecated,
    LpCurveTorque,
    SquaredMeanTorque_deprecated,
    SquaredMeanTorque,
    LpCurveForce_deprecated,
    LpCurveForce,
    SquaredMeanForce_deprecated,
    SquaredMeanForce,
)

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
        I = 1e5
        a = 0.01
        b = 0.023
        order = 1
        R1 = 40.0
        R2 = 3.0
        d = 5.0

        # Analytic field has only a z component
        B_reg_analytic_circ = constants.mu_0 * I / (4 * np.pi * R0) * (np.log(8 * R0 / a) - 3 / 4)
        force_analytic_circ = B_reg_analytic_circ * I

        # For two concentric circular coils, only "analytic" for R1 >> R0
        Lij_analytic = constants.mu_0 * np.pi * R0 ** 2 / (2 * R1)

        # For two coils that share a common axis
        k = np.sqrt(4.0 * R0 * R2 / ((R0 + R2) ** 2 + d ** 2))
        Lij_analytic2 = constants.mu_0 * np.sqrt(R0 * R2) * (
            (2 / k - k) * ellipk(k ** 2) - (2 / k) * ellipe(k ** 2)
        )

        # Eq (98) in Landreman Hurwitz Antonsen:
        B_reg_analytic_rect = constants.mu_0 * I / (4 * np.pi * R0) * (
            np.log(8 * R0 / np.sqrt(a * b)) + 13.0 / 12 - rectangular_xsection_k(a, b) / 2
        )
        force_analytic_rect = B_reg_analytic_rect * I

        for N_quad in [23, 13, 23, 500]:

            # Create a circle of radius R0 in the x-y plane:
            curve = CurveXYZFourier(N_quad, order)
            curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R0
            curve2 = CurveXYZFourier(N_quad, order)
            curve2.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R0
            phi = 2 * np.pi * curve.quadpoints

            current = Current(I)
            coil = Coil(curve, current)

            # Check the case of circular cross-section:

            B_reg_test = B_regularized_circ(coil, a)
            np.testing.assert_allclose(B_reg_test[:, 2], B_reg_analytic_circ)
            np.testing.assert_allclose(B_reg_test[:, 0:2], 0)

            # Test self-force for circular coil
            force_test = self_force_circ(coil, a)
            np.testing.assert_allclose(force_test[:, 0], force_analytic_circ * np.cos(phi))
            np.testing.assert_allclose(force_test[:, 1], force_analytic_circ * np.sin(phi))
            np.testing.assert_allclose(force_test[:, 2], 0.0)

            # Test self-torque for circular coil (should be zero by symmetry)
            # Analytic torque for a perfect circle is zero
            torque_test = coil_torque(coil, [coil], regularization_circ(a))
            np.testing.assert_allclose(torque_test, 0.0, atol=1e-9)

            normal = [0, 0, 1]
            alpha = np.arcsin(normal[1])
            delta = np.arccos(normal[2] / np.cos(alpha))
            curve = CurvePlanarFourier(N_quad, 0, nfp=1, stellsym=False)
            dofs = np.zeros(8)
            dofs[0] = R0
            dofs[1] = np.cos(alpha / 2.0) * np.cos(delta / 2.0)
            dofs[2] = np.sin(alpha / 2.0) * np.cos(delta / 2.0)
            dofs[3] = np.cos(alpha / 2.0) * np.sin(delta / 2.0)
            dofs[4] = -np.sin(alpha / 2.0) * np.sin(delta / 2.0)
            # Now specify the center
            dofs[5] = 0.0
            dofs[6] = 0.0
            dofs[7] = 0.0
            curve.set_dofs(dofs)

            # Make concentric coil with larger radius
            curve2 = CurvePlanarFourier(N_quad, 0, nfp=1, stellsym=False)
            dofs[0] = R1
            curve2.set_dofs(dofs)

            # Make circular coil with shared axis
            curve3 = CurvePlanarFourier(N_quad, 0, nfp=1, stellsym=False)
            dofs[0] = R2
            dofs[7] = d
            curve3.set_dofs(dofs)

            Lij = coil_coil_inductances_pure(
                curve.gamma(),
                curve.gammadash(),
                np.array([curve2.gamma()]),
                np.array([curve2.gammadash()]),
                a=a,
                b=a,
                downsample=1,
                cross_section='circular',
            )
            Lij_full = coil_coil_inductances_full_pure(
                np.array([c.gamma() for c in [curve, curve2]]),
                np.array([c.gammadash() for c in [curve, curve2]]),
                a_list=np.array([a, a]),
                b_list=np.array([a, a]),
                downsample=1,
                cross_section='circular',
            )
            # np.testing.assert_allclose(Lij[0], Lii_analytic)
            assert np.allclose(Lij, Lij_full[0, :])

            # Test rectangular cross section for a << R
            Lij_rect = coil_coil_inductances_pure(
                curve.gamma(),
                curve.gammadash(),
                np.array([curve2.gamma()]),
                np.array([curve2.gammadash()]),
                a=a,
                b=a,
                downsample=1,
                cross_section='rectangular',
            )
            Lij_rect_full = coil_coil_inductances_full_pure(
                np.array([c.gamma() for c in [curve, curve2]]),
                np.array([c.gammadash() for c in [curve, curve2]]),
                a_list=np.array([a, a]),
                b_list=np.array([a, a]),
                downsample=1,
                cross_section='rectangular',
            )

            np.testing.assert_allclose(Lij[1], Lij_analytic, rtol=1e-2)
            assert np.allclose(Lij_rect, Lij_rect_full[0, :])

            # retry but swap the coils
            Lij = coil_coil_inductances_pure(
                curve2.gamma(),
                curve2.gammadash(),
                np.array([curve.gamma()]),
                np.array([curve.gammadash()]),
                a=a,
                b=a,
                downsample=1,
                cross_section='circular',
            )
            assert np.allclose(np.flip(Lij), Lij_full[1, :])
            np.testing.assert_allclose(Lij[1], Lij_analytic, rtol=1e-2)

            # now test coils with shared axis
            Lij = coil_coil_inductances_pure(
                curve.gamma(),
                curve.gammadash(),
                np.array([curve3.gamma()]),
                np.array([curve3.gammadash()]),
                a=a,
                b=a,
                downsample=1,
                cross_section='circular',
            )
            np.testing.assert_allclose(Lij[1], Lij_analytic2, rtol=1e-2)

            Lij_full = coil_coil_inductances_full_pure(
                np.array([c.gamma() for c in [curve, curve3]]),
                np.array([c.gammadash() for c in [curve, curve3]]),
                a_list=np.array([a, a]),
                b_list=np.array([a, a]),
                downsample=1,
                cross_section='circular',
            )
            assert np.allclose(Lij, Lij_full[0, :])

            # Test cholesky computation of the inverse works on simple case
            Lij_inv = coil_coil_inductances_inv_pure(
                np.array([c.gamma() for c in [curve, curve3]]),
                np.array([c.gammadash() for c in [curve, curve3]]),
                a_list=np.array([a, a]),
                b_list=np.array([a, a]),
                downsample=1,
                cross_section='circular',
            )
            assert np.allclose(np.linalg.inv(Lij_full), Lij_inv)

            # Check the case of rectangular cross-section:

            B_reg_test = B_regularized_rect(coil, a, b)
            np.testing.assert_allclose(B_reg_test[:, 2], B_reg_analytic_rect)
            np.testing.assert_allclose(B_reg_test[:, 0:2], 0)

            force_test = self_force_rect(coil, a, b)
            np.testing.assert_allclose(force_test[:, 0], force_analytic_rect * np.cos(phi))
            np.testing.assert_allclose(force_test[:, 1], force_analytic_rect * np.sin(phi))
            np.testing.assert_allclose(force_test[:, 2], 0.0)

            # Test self-torque for rectangular cross-section coil (should also be zero by symmetry)
            # Analytic torque for a perfect rectangle is zero
            torque_test_rect = coil_torque(coil, [coil], regularization_rect(a, b))
            np.testing.assert_allclose(torque_test_rect, 0.0, atol=1e-9)

            # --- Two concentric circular coils: test mutual torque ---
            # Both in xy-plane, same center, different radii
            curve_inner = CurveXYZFourier(N_quad, order)
            curve_inner.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R0
            curve_outer = CurveXYZFourier(N_quad, order)
            curve_outer.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R1
            current_inner = Current(I)
            current_outer = Current(I)
            coil_inner = Coil(curve_inner, current_inner)
            coil_outer = Coil(curve_outer, current_outer)
            reg_inner = regularization_circ(a)
            reg_outer = regularization_circ(a)
            # Compute torque on each coil due to both
            torque_inner = coil_torque(coil_inner, [coil_inner, coil_outer], reg_inner)
            torque_outer = coil_torque(coil_outer, [coil_inner, coil_outer], reg_outer)
            # By symmetry, both should be zero
            np.testing.assert_allclose(torque_inner, 0.0, atol=1e-8)
            np.testing.assert_allclose(torque_outer, 0.0, atol=1e-8)

            # --- LpCurveTorque objective should also be zero ---
            obj1 = LpCurveTorque(coil_inner, coil_outer, reg_inner, p=2.0, threshold=0.0)
            val1 = obj1.J()
            np.testing.assert_allclose(val1, 0.0, atol=1e-1)
            # Outer as group 1, inner as group 2
            obj2 = LpCurveTorque(coil_outer, coil_inner, reg_outer, p=2.0, threshold=0.0)
            val2 = obj2.J()
            np.testing.assert_allclose(val2, 0.0, atol=1e-1)

            # --- Net force on each coil should also be zero ---
            net_force_inner = np.sum(coil_force(coil_inner, [coil_inner, coil_outer], reg_inner), axis=0)
            net_force_outer = np.sum(coil_force(coil_outer, [coil_inner, coil_outer], reg_outer), axis=0)
            np.testing.assert_allclose(net_force_inner, 0.0, atol=1e-6)
            np.testing.assert_allclose(net_force_outer, 0.0, atol=1e-6)

            # --- Two circular coils, separated along z but sharing a common axis: torque should be zero ---
            coil_z1 = Coil(curve2, Current(I))
            coil_z2 = Coil(curve3, Current(I))
            reg_z1 = regularization_circ(a)
            reg_z2 = regularization_circ(a)
            torque_z1 = coil_torque(coil_z1, [coil_z1, coil_z2], reg_z1)
            torque_z2 = coil_torque(coil_z2, [coil_z1, coil_z2], reg_z2)
            np.testing.assert_allclose(np.sum(torque_z1, axis=0), 0.0, atol=1e-8)
            np.testing.assert_allclose(np.sum(torque_z2, axis=0), 0.0, atol=1e-8)
            np.testing.assert_allclose(curve3.center(curve3.gamma(), curve3.gammadash()), [0, 0, 5], atol=1e-10)

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
            if j == 0:
                interpolant = interp1d(curve.quadpoints, force, axis=0)
                max_force_ref = max_force
            else:
                np.testing.assert_allclose(force, interpolant(curve.quadpoints), atol=max_force_ref / 60)

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

        # Test TVE
        objective = float(TVE(coils[0], coils, a=0.05).J())

        # Test LpCurveForce

        p = 2.5
        threshold = 1.0e3
        objective = float(LpCurveForce_deprecated(coils[0], coils, regularization, p=p, threshold=threshold).J())

        # Now compute the objective a different way, using the independent
        # coil_force function
        gammadash_norm = np.linalg.norm(coils[0].curve.gammadash(), axis=1)
        force_norm = np.linalg.norm(coil_force(coils[0], coils, regularization), axis=1)
        print("force_norm mean:", np.mean(force_norm), "max:", np.max(force_norm))
        objective_alt = (1 / p) * np.sum(np.maximum(force_norm - threshold, 0)**p * gammadash_norm) / np.shape(gammadash_norm)[0]

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-6)

        # Test MeanSquaredForce_deprecated

        objective = float(MeanSquaredForce_deprecated(coils[0], coils, regularization).J())

        # Now compute the objective a different way, using the independent
        # coil_force function
        force_norm = np.linalg.norm(coil_force(coils[0], coils, regularization), axis=1)
        objective_alt = np.sum(force_norm**2 * gammadash_norm) / np.sum(gammadash_norm)

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-6)

        # Test SquaredMeanForce_deprecated
        p = 2.5
        threshold = 1.0e3
        objective = float(SquaredMeanForce_deprecated(coils[0], coils).J())

        # Now compute the objective a different way, using the independent
        # coil_force function
        gammadash_norm = np.linalg.norm(coils[0].curve.gammadash(), axis=1)
        forces = coil_force(coils[0], coils, regularization)
        objective_alt = np.linalg.norm(np.sum(forces * gammadash_norm[:, None], axis=0) / gammadash_norm.shape[0], axis=-1) ** 2

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-6)

        # Test SquaredMeanForce vs SquaredMeanForce
        p = 2.5
        threshold = 1.0e3
        objective = 0.0
        objective2 = 0.0
        objective_mixed = 0.0
        for i in range(len(coils)):
            objective += float(SquaredMeanForce_deprecated(coils[i], coils).J())
            objective2 += float(SquaredMeanForce_deprecated(coils[i], coils, downsample=2).J())
            objective_mixed += float(SquaredMeanForce(coils[i], coils).J())

        print("objective:", objective, "mixed:", objective_mixed)
        np.testing.assert_allclose(objective, objective_mixed, rtol=1e-6)

        print("objective:", objective, "downsampled:", objective2)
        np.testing.assert_allclose(objective, objective2, rtol=1e-6)

        # # Test LpCurveForce
        objective = 0.0
        objective2 = 0.0
        objective_alt = 0.0
        objective_mixed = 0.0
        objective_mixed_downsampled = 0.0
        for i in range(len(coils)):
            objective += float(LpCurveForce_deprecated(coils[i], coils, regularization, p=p, threshold=threshold).J())
            objective2 += float(LpCurveForce_deprecated(coils[i], coils, regularization, p=p, threshold=threshold, downsample=2).J())
            objective_mixed += float(LpCurveForce(coils[i], coils, regularization, p=p, threshold=threshold).J())
            objective_mixed_downsampled += float(LpCurveForce(coils[i], coils, regularization, p=p, threshold=threshold, downsample=2).J())
            force_norm = np.linalg.norm(coil_force(coils[i], coils, regularization), axis=1)
            gammadash_norm = np.linalg.norm(coils[i].curve.gammadash(), axis=1)
            objective_alt += (1 / p) * np.sum(np.maximum(force_norm - threshold, 0)**p * gammadash_norm) / gammadash_norm.shape[0]

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-6)

        print("objective:", objective, "objective2:", objective2, "diff:", objective - objective2)
        np.testing.assert_allclose(objective, objective2, rtol=1e-2)

        print("objective:", objective, "objective_mixed:", objective_mixed, "diff:", objective - objective_mixed)
        np.testing.assert_allclose(objective, objective_mixed, rtol=1e-6)

        print("objective:", objective, "objective_mixed_downsampled:", objective_mixed_downsampled, "diff:", objective - objective_mixed)
        np.testing.assert_allclose(objective, objective_mixed_downsampled, rtol=1e-2)

        # Test SquaredMeanTorque

        # Scramble the orientations so the torques are nonzero
        for i in range(len(base_curves)):
            x_new = base_curves[i].x
            x_new[3] += 0.1
            base_curves[i].x = x_new

        objective = float(SquaredMeanTorque_deprecated(coils[0], coils).J())

        # Now compute the objective a different way, using the independent
        # coil_force function
        gammadash_norm = np.linalg.norm(coils[0].curve.gammadash(), axis=1)
        torques = coil_torque(coils[0], coils, regularization)
        objective_alt = np.linalg.norm(np.sum(torques * gammadash_norm[:, None], axis=0) / gammadash_norm.shape[0], axis=-1) ** 2

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-2)

        # Test SquaredMeanTorque_deprecated vs SquaredMeanTorque
        objective = 0.0
        objective2 = 0.0
        objective_alt = 0.0
        objective_mixed = 0.0
        for i in range(len(coils)):
            objective += float(SquaredMeanTorque_deprecated(coils[i], coils).J())
            objective2 += float(SquaredMeanTorque_deprecated(coils[i], coils, downsample=2).J())
            gammadash_norm = np.linalg.norm(coils[i].curve.gammadash(), axis=1)
            objective_alt += np.linalg.norm(np.sum(coil_torque(coils[i], coils, regularization) * gammadash_norm[:, None], axis=0) / gammadash_norm.shape[0]) ** 2
            objective_mixed += float(SquaredMeanTorque(coils[i], coils).J())
        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-2)

        print("objective:", objective, "downsampled:", objective2, "diff:", objective - objective2)
        np.testing.assert_allclose(objective, objective2, rtol=1e-2)

        print("objective:", objective, "objective_mixed:", objective_mixed, "diff:", objective - objective_mixed)
        np.testing.assert_allclose(objective, objective_mixed, rtol=1e-2)

        # Test LpCurveTorque
        objective = 0.0
        objective2 = 0.0
        objective_alt = 0.0
        threshold = 0.0
        objective_mixed = 0.0
        for i in range(len(coils)):
            objective += float(LpCurveTorque_deprecated(coils[i], coils, regularization, p=p, threshold=threshold).J())
            objective2 += float(LpCurveTorque_deprecated(coils[i], coils, regularization, p=p, threshold=threshold, downsample=2).J())
            torque_norm = np.linalg.norm(coil_torque(coils[i], coils, regularization), axis=1)
            gammadash_norm = np.linalg.norm(coils[i].curve.gammadash(), axis=1)
            objective_alt += (1 / p) * np.sum(np.maximum(torque_norm - threshold, 0)**p * gammadash_norm) / gammadash_norm.shape[0]
            objective_mixed += float(LpCurveTorque(coils[i], coils, regularization, p=p, threshold=threshold).J())

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-6)

        print("objective:", objective, "downsampled:", objective2, "diff:", objective - objective2)
        np.testing.assert_allclose(objective, objective2, rtol=1e-4)

        print("objective:", objective, "objective_mixed:", objective_mixed, "diff:", objective - objective_mixed)
        np.testing.assert_allclose(objective, objective_mixed, rtol=1e-6)

    def test_force_and_torque_objectives_with_different_quadpoints(self):
        """Check that force and torque objectives work with coils having different numbers of quadrature points."""
        I = 1.7e4
        regularization = regularization_circ(0.05)
        # Create two coils with different numbers of quadrature points
        curve1 = CurveXYZFourier(30, 1)
        curve1.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * 1.0
        curve2 = CurveXYZFourier(50, 1)
        curve2.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * 1.2
        curve3 = CurveXYZFourier(70, 1)
        curve3.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * 0.8
        current1 = Current(I)
        current2 = Current(I)
        current3 = Current(I)
        coil1 = Coil(curve1, current1)
        coil2 = Coil(curve2, current2)
        coil3 = Coil(curve3, current3)
        coils = [coil1, coil2, coil3]
        regularization_list = [regularization, regularization, regularization]
        # LpCurveForce
        val = LpCurveForce(coil1, coil2, regularization_list[0], p=2.5, threshold=1.0e3).J()
        self.assertTrue(np.isfinite(val))
        val = LpCurveForce(coil1, [coil2, coil3], regularization_list[0], p=2.5, threshold=1.0e3).J()
        self.assertTrue(np.isfinite(val))
        val = LpCurveForce(coil1, coils, regularization_list[0], p=2.5, threshold=1.0e3).J()
        self.assertTrue(np.isfinite(val))
        # SquaredMeanForce
        val = SquaredMeanForce(coil1, coil3).J()
        self.assertTrue(np.isfinite(val))
        val = SquaredMeanForce([coil1, coil2], coil3).J()
        self.assertTrue(np.isfinite(val))
        val = SquaredMeanForce(coil1, coils).J()
        self.assertTrue(np.isfinite(val))
        # LpCurveTorque
        val = LpCurveTorque(coil1, coils, regularization_list[0], p=2.5, threshold=1.0e3).J()
        self.assertTrue(np.isfinite(val))
        val = LpCurveTorque([coil1, coil2], coils, regularization_list[0:2], p=2.5, threshold=1.0e3).J()
        self.assertTrue(np.isfinite(val))
        val = LpCurveTorque(coil3, coil1, regularization_list[2], p=2.5, threshold=1.0e3).J()
        self.assertTrue(np.isfinite(val))
        # SquaredMeanTorque
        val = SquaredMeanTorque(coil1, coils).J()
        self.assertTrue(np.isfinite(val))
        val = SquaredMeanTorque(coil3, [coil1, coil2]).J()
        self.assertTrue(np.isfinite(val))
        val = SquaredMeanTorque(coil3, coils).J()
        self.assertTrue(np.isfinite(val))

    def test_Taylor(self):
        import matplotlib.pyplot as plt
        ncoils_list = [2]
        nfp_list = [1, 2, 3]
        stellsym_list = [False, True]
        p_list = [2.5]
        threshold_list = [0.0]
        downsample_list = [1, 2]
        numquadpoints_list = [30]
        I = 1.7e5
        a = 0.05
        b = 0.05
        np.random.seed(1234)
        regularization_types = [
            ("circular", lambda: regularization_circ(a)),
            ("rectangular", lambda: regularization_rect(a, b)),
        ]
        all_errors = []
        all_labels = []
        all_eps = []
        for ncoils in ncoils_list:
            for nfp in nfp_list:
                for stellsym in stellsym_list:
                    for p in p_list:
                        for threshold in threshold_list:
                            for reg_name, reg_func in regularization_types:
                                regularization = reg_func()
                                for downsample in downsample_list:
                                    for numquadpoints in numquadpoints_list:
                                        base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym, numquadpoints=numquadpoints)
                                        base_curves2 = create_equally_spaced_curves(ncoils, nfp, stellsym, numquadpoints=numquadpoints)
                                        base_currents = [Current(I) for j in range(ncoils)]
                                        coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)
                                        for ii in range(ncoils):
                                            base_curves2[ii].x = base_curves2[ii].x + np.ones(len(base_curves2[ii].x)) * 0.1
                                        coils2 = coils_via_symmetries(base_curves2, base_currents, nfp, stellsym)
                                        regularization_list = [regularization for _ in coils]
                                        objectives = [
                                            sum([NetFluxes(coils[i], coils+coils2) for i in range(len(coils))]),
                                            sum([TVE(coils[i], coils+coils2, a=a) for i in range(len(coils))]),
                                            LpCurveTorque(coils, coils2, regularization_list,
                                                          p=p, threshold=threshold, downsample=downsample),
                                            sum([LpCurveTorque(coils[i], coils+coils2, regularization_list[i],
                                                               p=p, threshold=threshold, downsample=downsample) for i in range(len(coils))]),
                                            SquaredMeanTorque(coils, coils2, downsample=downsample),
                                            sum([SquaredMeanTorque(coils[i], coils+coils2, downsample=downsample) for i in range(len(coils))]),
                                            LpCurveForce(coils, coils2, regularization_list,
                                                         p=p, threshold=threshold, downsample=downsample),
                                            sum([LpCurveForce(coils[i], coils+coils2, regularization_list[i],
                                                              p=p, threshold=threshold, downsample=downsample) for i in range(len(coils))]),
                                            SquaredMeanForce(coils, coils2, downsample=downsample),
                                            sum([SquaredMeanForce(coils[i], coils+coils2, downsample=downsample) for i in range(len(coils))]),
                                        ]
                                        dofs = np.copy(LpCurveTorque(coils, coils2, regularization_list,
                                                                     p=p, threshold=threshold, downsample=downsample).x)
                                        h = np.ones_like(dofs)
                                        for J in objectives:
                                            print(f"ncoils={ncoils}, nfp={nfp}, stellsym={stellsym}, p={p}, threshold={threshold}, reg={reg_name}, downsample={downsample}, objective={type(J).__name__}")
                                            J.x = dofs  # Need to reset Jf.x for each objective
                                            dJ = J.dJ()
                                            deriv = np.sum(dJ * h)
                                            errors = []
                                            epsilons = []
                                            label = f"{type(J).__name__}, ncoils={ncoils}, nfp={nfp}, stellsym={stellsym}, p={getattr(J, 'p', p)}, threshold={getattr(J, 'threshold', threshold)}, reg={reg_name}, downsample={downsample}"
                                            for i in range(10, 16):
                                                eps = 0.5**i
                                                J.x = dofs + eps * h
                                                Jp = J.J()
                                                J.x = dofs - eps * h
                                                Jm = J.J()
                                                deriv_est = (Jp - Jm) / (2 * eps)
                                                if np.abs(deriv) < 1e-8:
                                                    err_new = np.abs(deriv_est - deriv)  # compute absolute error instead
                                                else:
                                                    err_new = np.abs(deriv_est - deriv) / np.abs(deriv)
                                                # Check error decrease by at least a factor of 0.3
                                                if len(errors) > 0 and err_new > 1e-10:
                                                    print(f"err: {err_new}, jac: {np.abs(deriv)}, jac_est: {np.abs(deriv_est)}, ratio: {(err_new + 1e-12) / (errors[-1] + 1e-12)}")
                                                    assert err_new < 0.5 * errors[-1], f"Error did not decrease by factor 0.5: prev={errors[-1]}, curr={err_new}"
                                                errors.append(err_new)
                                                epsilons.append(eps)
                                            all_errors.append(errors)
                                            all_labels.append(label)
                                            all_eps.append(epsilons)
        # Plot all errors
        plt.figure(figsize=(14, 8))
        for errors, label, epsilons in zip(all_errors, all_labels, all_eps):
            plt.loglog(epsilons, errors, marker='o', label=label)
        plt.xlabel('eps')
        plt.ylabel('Relative Taylor error')
        plt.title('Taylor test errors for all objectives and parameter sweeps')
        plt.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('taylor_errors.png')

    def test_Taylor_PSC(self):
        import matplotlib.pyplot as plt
        from simsopt.field import PSCArray
        from simsopt.objectives import SquaredFlux
        from simsopt.geo import SurfaceRZFourier
        from pathlib import Path
        from monty.tempfile import ScratchDir
        from simsopt.field.selffield import regularization_circ, regularization_rect
        TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
        filename = TEST_DIR / 'input.LandremanPaul2021_QA'
        nphi, ntheta = 8, 8
        ncoils_list = [2]
        nfp_list = [1, 2, 3]
        stellsym_list = [False, True]
        p_list = [2.5]
        threshold_list = [0.0]
        downsample_list = [1, 2]
        I = 1.7e5
        a = 0.05
        b = 0.05
        regularization_types = [
            ("circular", lambda: regularization_circ(a)),
            ("rectangular", lambda: regularization_rect(a, b)),
        ]
        all_errors = []
        all_labels = []
        all_eps = []
        with ScratchDir("."):
            s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            eval_points = s.gamma().reshape(-1, 3)
            for ncoils in ncoils_list:
                for nfp in nfp_list:
                    for stellsym in stellsym_list:
                        for p in p_list:
                            for threshold in threshold_list:
                                for reg_name, reg_func in regularization_types:
                                    regularization = reg_func()
                                    for downsample in downsample_list:
                                        base_curves_TF = create_equally_spaced_curves(ncoils, nfp, stellsym)
                                        base_currents_TF = [Current(I) for j in range(ncoils)]
                                        for i in range(ncoils):
                                            base_currents_TF[i].fix_all()
                                        coils_TF = coils_via_symmetries(base_curves_TF, base_currents_TF, nfp, stellsym)
                                        base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym, R0=0.5, R1=0.1)
                                        a_list = np.ones(len(base_curves)) * a
                                        b_list = a_list
                                        psc_array = PSCArray(base_curves, coils_TF, eval_points, a_list, b_list, nfp=nfp, stellsym=stellsym)
                                        coils = psc_array.coils
                                        coils_TF = psc_array.coils_TF
                                        btot = psc_array.biot_savart_total
                                        btot.set_points(eval_points)
                                        regularization_list = [regularization for _ in coils]
                                        objectives = [
                                            SquaredFlux(s, btot),
                                            LpCurveTorque(coils, coils_TF, regularization_list,
                                                          p=p, threshold=threshold, psc_array=psc_array, downsample=downsample),
                                            SquaredMeanTorque(coils, coils_TF, psc_array=psc_array, downsample=downsample),
                                            LpCurveForce(coils, coils_TF, regularization_list,
                                                         p=p, threshold=threshold, psc_array=psc_array, downsample=downsample),
                                            SquaredMeanForce(coils, coils_TF, psc_array=psc_array, downsample=downsample),
                                        ]
                                        dofs = np.copy(SquaredFlux(s, btot).x)
                                        h = np.ones_like(dofs)
                                        for J in objectives:
                                            print(f"ncoils={ncoils}, nfp={nfp}, stellsym={stellsym}, p={p}, threshold={threshold}, reg={reg_name}, downsample={downsample}, objective={type(J).__name__}")
                                            J.x = dofs  # Need to reset Jf.x for each objective
                                            psc_array.recompute_currents()
                                            dJ = J.dJ()
                                            deriv = np.sum(dJ * np.ones_like(J.x))
                                            errors = []
                                            epsilons = []
                                            label = f"{type(J).__name__}, ncoils={ncoils}, nfp={nfp}, stellsym={stellsym}, p={getattr(J, 'p', p)}, threshold={getattr(J, 'threshold', threshold)}, reg={reg_name}, downsample={downsample}"
                                            for i in range(11, 18):
                                                eps = 0.5**i
                                                J.x = dofs + eps * h
                                                psc_array.recompute_currents()
                                                Jp = J.J()
                                                J.x = dofs - eps * h
                                                psc_array.recompute_currents()
                                                Jm = J.J()
                                                deriv_est = (Jp - Jm) / (2 * eps)
                                                if np.abs(deriv) < 1e-8:
                                                    err_new = np.abs(deriv_est - deriv)
                                                else:
                                                    err_new = np.abs(deriv_est - deriv) / np.abs(deriv)
                                                if len(errors) > 0:
                                                    print(f"err: {err_new}, eps: {eps}, ratio: {err_new / errors[-1]}")
                                                    assert err_new < 0.5 * errors[-1], f"Error did not decrease by factor 0.5: prev={errors[-1]}, curr={err_new}"
                                                errors.append(err_new)
                                                epsilons.append(eps)
                                            all_errors.append(errors)
                                            all_labels.append(label)
                                            all_eps.append(epsilons)
            # TVE objectives
            for ncoils in ncoils_list:
                for nfp in nfp_list:
                    for stellsym in stellsym_list:
                        base_curves_TF = create_equally_spaced_curves(ncoils, nfp, stellsym)
                        base_currents_TF = [Current(I) for j in range(ncoils)]
                        for i in range(ncoils):
                            base_currents_TF[i].fix_all()
                        coils_TF = coils_via_symmetries(base_curves_TF, base_currents_TF, nfp, stellsym)
                        base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym, R0=0.5, R1=0.1)
                        a_list = np.ones(len(base_curves)) * a
                        b_list = a_list
                        psc_array = PSCArray(base_curves, coils_TF, eval_points, a_list, b_list, nfp=nfp, stellsym=stellsym)
                        coils = psc_array.coils
                        coils_TF = psc_array.coils_TF
                        objectives = [
                            TVE(coils[0], coils[1:], a=a, psc_array=psc_array),
                        ]
                        dofs = np.copy(TVE(coils[0], coils[1:], a=a, psc_array=psc_array).x)
                        h = np.ones_like(dofs)
                        for J in objectives:
                            print(f"ncoils={ncoils}, nfp={nfp}, stellsym={stellsym}, objective={type(J).__name__}")
                            J.x = dofs
                            psc_array.recompute_currents()
                            dJ = J.dJ()
                            deriv = np.sum(dJ * np.ones_like(J.x))
                            errors = []
                            epsilons = []
                            label = f"{type(J).__name__}, ncoils={ncoils}, nfp={nfp}, stellsym={stellsym}"
                            for i in range(11, 18):
                                eps = 0.5**i
                                J.x = J.x + eps * h
                                psc_array.recompute_currents()
                                Jp = J.J()
                                J.x = J.x - eps * h
                                psc_array.recompute_currents()
                                Jm = J.J()
                                deriv_est = (Jp - Jm) / (2 * eps)
                                if np.abs(deriv) < 1e-8:
                                    err_new = np.abs(deriv_est - deriv)
                                else:
                                    err_new = np.abs(deriv_est - deriv) / np.abs(deriv)
                                if errors[-1] > 0:
                                    print(f"err: {err_new}, eps: {eps}, ratio: {err_new / errors[-1]}")
                                    assert err_new < 0.5 * errors[-1], f"Error did not decrease by factor 0.5: prev={errors[-1]}, curr={err_new}"
                                errors.append(err_new)
                                epsilons.append(eps)
                            all_errors.append(errors)
                            all_labels.append(label)
                            all_eps.append(epsilons)
        # Plot all errors
        plt.figure(figsize=(14, 8))
        for errors, label, epsilons in zip(all_errors, all_labels, all_eps):
            plt.loglog(epsilons, errors, marker='o', label=label)
        plt.xlabel('eps')
        plt.ylabel('Relative Taylor error')
        plt.title('Taylor test errors for all PSC objectives and parameter sweeps')
        plt.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('taylor_errors_psc.png')

    def test_objectives_time(self):
        import time
        import matplotlib.pyplot as plt
        import numpy as np

        nfp = 3
        I = 1.7e4

        p = 2.5
        threshold = 1.0e3
        regularization = regularization_circ(0.05)

        # List of objective classes to test
        objective_classes = [
            "LpCurveForce_deprecated (sum all)",
            "LpCurveForce",
            "LpCurveForce (one sum)",
            "LpCurveTorque_deprecated (sum all)",
            "LpCurveTorque",
            "LpCurveTorque (one sum)",
            "SquaredMeanForce_deprecated (sum all)",
            "SquaredMeanForce",
            "SquaredMeanForce (one sum)",
            "SquaredMeanTorque_deprecated (sum all)",
            "SquaredMeanTorque",
            "SquaredMeanTorque (one sum)",
        ]

        ncoils_list = [2, 4]
        runtimes_J = np.zeros((len(objective_classes), len(ncoils_list)))
        runtimes_dJ = np.zeros((len(objective_classes), len(ncoils_list)))
        compile_times_J = np.zeros((len(objective_classes), len(ncoils_list)))
        compile_times_dJ = np.zeros((len(objective_classes), len(ncoils_list)))

        for idx_n, ncoils in enumerate(ncoils_list):
            print(f"\n--- Timing tests for ncoils = {ncoils} ---")
            base_curves = create_equally_spaced_curves(ncoils, nfp, True)
            base_currents = [Current(I) for j in range(ncoils)]
            coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
            base_curves2 = create_equally_spaced_curves(ncoils, nfp, True)
            for i in range(ncoils):
                base_curves2[i].x = base_curves2[i].x + np.ones(len(base_curves2[i].x)) * 0.01
            coils2 = coils_via_symmetries(base_curves2, base_currents, nfp, True)
            regularization_list = [regularization for _ in coils]

            # Prepare objectives for each class
            # LpCurveForce, LpCurveTorque, SquaredMeanForce, SquaredMeanTorque: sum over all coils
            # Mixed objectives are faster if coils are split evenly into two groups
            objectives = [
                sum([LpCurveForce_deprecated(c, coils, regularization, p=p, threshold=threshold, downsample=2) for c in coils]),
                sum([LpCurveForce(c, coils, regularization, p=p, threshold=threshold, downsample=2) for c in coils]),
                LpCurveForce(coils, coils2, regularization_list, p=p, threshold=threshold, downsample=2),
                sum([LpCurveTorque_deprecated(c, coils, regularization, p=p, threshold=threshold, downsample=2) for c in coils]),
                sum([LpCurveTorque(c, coils, regularization, p=p, threshold=threshold, downsample=2) for c in coils]),
                LpCurveTorque(coils, coils2, regularization_list, p=p, threshold=threshold, downsample=2),
                sum([SquaredMeanForce_deprecated(c, coils, downsample=2) for c in coils]),
                sum([SquaredMeanForce(c, coils, downsample=2) for c in coils]),
                SquaredMeanForce(coils, coils2, downsample=2),
                sum([SquaredMeanTorque_deprecated(c, coils, downsample=2) for c in coils]),
                sum([SquaredMeanTorque(c, coils, downsample=2) for c in coils]),
                SquaredMeanTorque(coils, coils2, downsample=2),
            ]

            # Compilation time (first call)
            print("Timing compilation (first call):")
            for i, (obj, obj_label) in enumerate(zip(objectives, objective_classes)):
                t1 = time.time()
                obj.J()
                t2 = time.time()
                compile_times_J[i, idx_n] = t2 - t1
                print(f'{obj_label}: Compilation (J) took {t2 - t1:.6f} seconds')
                t1 = time.time()
                obj.dJ()
                t2 = time.time()
                compile_times_dJ[i, idx_n] = t2 - t1
                print(f'{obj_label}: Compilation (dJ) took {t2 - t1:.6f} seconds')

                # Run time (second call)
                print("Timing run (second call):")
                t1 = time.time()
                obj.J()
                t2 = time.time()
                runtimes_J[i, idx_n] = t2 - t1
                print(f'{obj_label}: Run (J) took {t2 - t1:.6f} seconds')
                t1 = time.time()
                obj.dJ()
                t2 = time.time()
                runtimes_dJ[i, idx_n] = t2 - t1
                print(f'{obj_label}: Run (dJ) took {t2 - t1:.6f} seconds')

        # Optionally, plot the results
        plt.figure(figsize=(10, 7))
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'x', '+']
        colors = plt.cm.tab10.colors
        for i, label in enumerate(objective_classes):
            plt.semilogy(ncoils_list, runtimes_J[i], marker=markers[i % len(markers)], color=colors[i % len(colors)], label=f"{label} J()")
            plt.semilogy(ncoils_list, runtimes_dJ[i], marker=markers[i % len(markers)], linestyle='--', color=colors[i % len(colors)], label=f"{label} dJ()")
        plt.xlabel("Number of coils")
        plt.ylabel("Run time (s)")
        plt.title("Objective run times as a function of number of coils")
        plt.legend(fontsize=8)
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig("objective_runtimes_semilogy.png")
        print("Run times saved to objective_runtimes_semilogy.png")

    def test_update_points(self):
        """Confirm that Biot-Savart evaluation points are updated when the
        curve shapes change."""
        from simsopt.field import BiotSavart
        nfp = 4
        ncoils = 3
        I = 1.7e4
        regularization = regularization_circ(0.05)

        for objective_class in [MeanSquaredForce_deprecated, LpCurveForce_deprecated, LpCurveTorque_deprecated]:

            base_curves = create_equally_spaced_curves(ncoils, nfp, True, order=2)
            base_currents = [Current(I) for j in range(ncoils)]
            coils = coils_via_symmetries(base_curves, base_currents, nfp, True)

            objective = objective_class(coils[0], coils, regularization)
            old_objective_value = objective.J()
            biotsavart = BiotSavart(objective.othercoils)
            old_biot_savart_points = biotsavart.get_points_cart()
            print(old_biot_savart_points)

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
            biotsavart = BiotSavart(objective.othercoils)

            # Don't understand the commented out test below --
            # the biot savart evaluation points actually
            # do not change here -- only the coil points are changing
            # new_biot_savart_points = biotsavart.get_points_cart()
            # assert not np.allclose(old_biot_savart_points, new_biot_savart_points)
            objective2 = objective_class(coils[0], coils, regularization)
            print("objective 1:", objective.J(), "objective 2:", objective2.J())
            np.testing.assert_allclose(objective.J(), objective2.J())


if __name__ == '__main__':
    unittest.main()
