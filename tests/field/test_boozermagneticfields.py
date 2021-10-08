from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField, BoozerAnalytic
import numpy as np
import unittest
from pathlib import Path
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename = str((TEST_DIR / 'input.LandremanPaul2021_QA_lowres').resolve())

try:
    import vmec
except ImportError as e:
    vmec = None

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None

if (MPI is not None) and (vmec is not None):
    from simsopt.mhd.vmec import Vmec


class TestingAnalytic(unittest.TestCase):
    def test_boozeranalytic(self):
        # Test that perfect derivatives integrate to zero
        etabar = 1.1
        B0 = 1.0
        Bbar = 1.0
        N = 0
        G0 = 1.1
        psi0 = 0.8
        iota0 = 0.4
        ba = BoozerAnalytic(etabar, B0, Bbar, N, G0, psi0, iota0)
        # Check that angular derivatives integrate to zero
        ntheta = 101
        nzeta = 100
        thetas = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        zetas = np.linspace(0, 2*np.pi, nzeta, endpoint=False)
        [zetas, thetas] = np.meshgrid(zetas, thetas)
        points = np.zeros((len(thetas.flatten()), 3))
        points[:, 0] = 0.5*np.ones_like(thetas.flatten())
        points[:, 1] = thetas.flatten()
        points[:, 2] = zetas.flatten()
        ba.set_points(points)
        # Check that get_points returns correct points
        points_get = ba.get_points()
        thetas_get = points_get[:, 1]
        assert np.allclose(thetas_get, thetas.flatten())
        assert np.allclose(np.sum(ba.dmodBdtheta().reshape(np.shape(thetas)), axis=0), 0, rtol=1e-12)
        assert np.allclose(np.sum(ba.dmodBdzeta().reshape(np.shape(thetas)), axis=1), 0, rtol=1e-12)
        # Check that zeta derivatives are small since we are QA
        assert np.allclose(ba.dmodBdzeta(), 0, atol=1e-12)
        ba.set_N(1)
        # Check that (theta + zeta) derivatives are small since we are QH
        assert np.allclose(ba.dmodBdtheta()+ba.dmodBdzeta(), 0, atol=1e-12)
        # Check that G = G0
        assert np.allclose(ba.G(), G0, atol=1e-12)
        # Check that dGds = 0 since G1 = 0
        assert np.allclose(ba.dGds(), 0, atol=1e-12)
        # Check that I = 0 and I'=0 since I0 = I1 = 0
        assert np.allclose(ba.I(), 0, atol=1e-12)
        assert np.allclose(ba.dIds(), 0, atol=1e-12)
        # Check that iota = iota0
        assert np.allclose(ba.iota(), iota0, atol=1e-12)
        # Check that diotads = 0
        assert np.allclose(ba.diotads(), 0, atol=1e-12)
        # Check that if etabar = 0, dBdtheta = dBdzeta = dBds = 0
        ba.set_etabar(0.)
        assert np.allclose(ba.dmodBdtheta(), 0, atol=1e-12)
        assert np.allclose(ba.dmodBdzeta(), 0, atol=1e-12)
        assert np.allclose(ba.dmodBds(), 0, atol=1e-12)
        # Check other set_ functions
        ba.set_B0(1.3)
        assert(ba.B0 == 1.3)
        ba.set_Bbar(3.)
        assert(ba.Bbar == 3.)
        ba.set_G0(3.1)
        assert(ba.G0 == 3.1)
        ba.set_I0(3.2)
        assert(ba.I0 == 3.2)
        ba.set_G1(3.3)
        assert(ba.G1 == 3.3)
        ba.set_I1(3.4)
        assert(ba.I1 == 3.4)
        ba.set_iota0(3.5)
        assert(ba.iota0 == 3.5)
        ba.set_psi0(3.6)
        assert(ba.psi0 == 3.6)


@unittest.skipIf(vmec is None, "vmec python package is not found")
class TestingVmec(unittest.TestCase):
    def test_boozerradialinterpolant(self):
        # Test that perfect derivatives integrate to zero
        vmec = Vmec(filename)
        order = 3
        ns_delete = 1
        for rescale in [True,False]:
            bri = BoozerRadialInterpolant(vmec, order, rescale=rescale, ns_delete=ns_delete)

            # Perform interpolation from full grid
            points = np.zeros((len(vmec.s_half_grid)-1, 3))
            points[:, 0] = vmec.s_full_grid[1:-1]
            bri.set_points(points)
            # Check with linear interpolation from half grid
            G_full = (vmec.wout.bvco[1:-1]+vmec.wout.bvco[2::])/2.
            iota_full = (vmec.wout.iotas[1:-1]+vmec.wout.iotas[2::])/2.
            modB00 = np.sum(bri.booz.bx.bmnc_b,axis=0)
            modB_full = (modB00[0:-1]+modB00[1::])/2

            # Compare splines of derivatives with spline derivatives
            from scipy.interpolate import InterpolatedUnivariateSpline
            G_spline = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bvco[1::])
            iota_spline = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1::])
            modB00_spline = InterpolatedUnivariateSpline(vmec.s_half_grid, modB00)

            assert np.allclose(bri.G(), G_full, atol=1e-4)
            assert np.allclose(bri.iota(), iota_full, atol=1e-2)
            assert np.allclose(bri.modB()[:,0], modB_full, atol=1e-2)
            assert np.allclose(bri.dGds(), G_spline.derivative()(vmec.s_full_grid[1:-1]), atol=1e-3)
            assert np.allclose(bri.diotads(), iota_spline.derivative()(vmec.s_full_grid[1:-1]), atol=1e-2)
            assert np.allclose(bri.dmodBds()[ns_delete+1::,0], modB00_spline.derivative()(vmec.s_full_grid[ns_delete+2:-1]), atol=1e-2)

            points = np.zeros((len(vmec.s_half_grid), 3))
            points[:, 0] = vmec.s_half_grid
            bri.set_points(points)

            # Check that angular derivatives integrate to zero
            ntheta = 101
            nzeta = 100
            thetas = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
            zetas = np.linspace(0, 2*np.pi, nzeta, endpoint=False)
            [zetas, thetas] = np.meshgrid(zetas, thetas)
            points = np.zeros((len(thetas.flatten()), 3))
            points[:, 0] = 0.5*np.ones_like(thetas.flatten())
            points[:, 1] = thetas.flatten()
            points[:, 2] = zetas.flatten()
            bri.set_points(points)
            # Check that get_points returns correct points
            points_get = bri.get_points()
            thetas_get = points_get[:, 1]
            assert np.allclose(thetas_get, thetas.flatten())
            assert np.allclose(np.sum(bri.dmodBdtheta().reshape(np.shape(thetas)), axis=0), 0, rtol=1e-12)
            assert np.allclose(np.sum(bri.dmodBdzeta().reshape(np.shape(thetas)), axis=1), 0, rtol=1e-12)
            # Check that zeta derivatives are small since we are close to QA
            assert np.allclose(bri.dmodBdzeta(), 0, atol=1e-2)

    def test_interpolatedboozerfield_sym(self):
        vmec = Vmec(filename)
        order = 3
        bri = BoozerRadialInterpolant(vmec, order)

        # Perform interpolation from full grid
        points = np.zeros((len(vmec.s_half_grid)-1, 3))
        points[:, 0] = vmec.s_full_grid[1:-1]
        bri.set_points(points)

        nfp = vmec.indata.nfp
        n = 12
        smin = 0.4
        smax = 0.6
        ssteps = n
        thetamin = np.pi*(1/4)
        thetamax = np.pi*(3/4)
        thetasteps = n
        zetamin = -2*np.pi/(4*nfp)
        zetamax = 2*np.pi/(4*nfp)
        zetasteps = n*2
        bsh = InterpolatedBoozerField(
            bri, 4, [smin, smax, ssteps], [thetamin, thetamax, thetasteps], [zetamin, zetamax, zetasteps],
            True, nfp=nfp, stellsym=True)
        N = 10
        np.random.seed(2)
        points = np.random.uniform(size=(N, 3))
        points[:, 0] = points[:, 0]*(smax-smin) + smin
        points[:, 1] = points[:, 1]*(thetamax-thetamin) + thetamin
        points[:, 2] = points[:, 2]*(zetamax-zetamin) + zetamin

        bri.set_points(points)
        modB = bri.modB()
        dmodBds = bri.dmodBds()
        dmodBdtheta = bri.dmodBdtheta()
        dmodBdzeta = bri.dmodBdzeta()
        G = bri.G()
        I = bri.I()
        iota = bri.iota()
        diotads = bri.diotads()
        dGds = bri.dGds()
        dIds = bri.dIds()

        bsh.set_points(points)
        modBh = bsh.modB()
        dmodBdsh = bsh.dmodBds()
        dmodBdthetah = bsh.dmodBdtheta()
        dmodBdzetah = bsh.dmodBdzeta()
        Gh = bsh.G()
        Ih = bsh.I()
        iotah = bsh.iota()
        diotadsh = bsh.diotads()
        dGdsh = bsh.dGds()
        dIdsh = bsh.dIds()

        assert np.allclose(modB, modBh, rtol=1e-3)
        assert np.allclose((dmodBds - dmodBdsh)/np.mean(np.abs(dmodBds)), 0, atol=1e-2)
        assert np.allclose((dmodBdtheta - dmodBdthetah)/np.mean(np.abs(dmodBdtheta)), 0, atol=1e-2)
        assert np.allclose(dmodBdzeta, 0, atol=1e-3)
        assert np.allclose(dmodBdzetah, 0, atol=1e-3)
        assert np.allclose(G, Gh, rtol=1e-3)
        assert np.allclose(iota, iotah, rtol=1e-3)
        assert np.allclose(diotads, diotadsh, rtol=1e-3)
        assert np.allclose(I, 0, atol=1e-3)
        assert np.allclose(Ih, 0, atol=1e-3)
        assert np.allclose(dGds, 0, atol=1e-3)
        assert np.allclose(dGdsh, 0, atol=1e-3)
        assert np.allclose(dIds, 0, atol=1e-3)
        assert np.allclose(dIdsh, 0, atol=1e-3)

    def test_interpolatedboozerfield_no_sym(self):
        vmec = Vmec(filename)
        order = 3
        bri = BoozerRadialInterpolant(vmec, order)

        # Perform interpolation from full grid
        points = np.zeros((len(vmec.s_half_grid)-1, 3))
        points[:, 0] = vmec.s_full_grid[1:-1]
        bri.set_points(points)

        nfp = vmec.indata.nfp

        n = 12
        smin = 0.4
        smax = 0.6
        ssteps = n
        thetamin = np.pi*(1/4)
        thetamax = np.pi*(3/4)
        thetasteps = n
        zetamin = -2*np.pi/(4*nfp)
        zetamax = 2*np.pi/(4*nfp)
        zetasteps = n*2
        bsh = InterpolatedBoozerField(
            bri, 4, [smin, smax, ssteps], [thetamin, thetamax, thetasteps], [zetamin, zetamax, zetasteps],
            True)
        N = 10
        np.random.seed(2)
        points = np.random.uniform(size=(N, 3))
        points[:, 0] = points[:, 0]*(smax-smin) + smin
        points[:, 1] = points[:, 1]*(thetamax-thetamin) + thetamin
        points[:, 2] = points[:, 2]*(zetamax-zetamin) + zetamin

        bri.set_points(points)
        modB = bri.modB()
        dmodBds = bri.dmodBds()
        dmodBdtheta = bri.dmodBdtheta()
        dmodBdzeta = bri.dmodBdzeta()
        G = bri.G()
        I = bri.I()
        iota = bri.iota()
        diotads = bri.diotads()
        dGds = bri.dGds()
        dIds = bri.dIds()

        bsh.set_points(points)
        modBh = bsh.modB()
        dmodBdsh = bsh.dmodBds()
        dmodBdthetah = bsh.dmodBdtheta()
        dmodBdzetah = bsh.dmodBdzeta()
        Gh = bsh.G()
        Ih = bsh.I()
        iotah = bsh.iota()
        diotadsh = bsh.diotads()
        dGdsh = bsh.dGds()
        dIdsh = bsh.dIds()

        assert np.allclose(modB, modBh, rtol=1e-3)
        assert np.allclose((dmodBds - dmodBdsh)/np.mean(np.abs(dmodBds)), 0, atol=1e-2)
        assert np.allclose((dmodBdtheta - dmodBdthetah)/np.mean(np.abs(dmodBdtheta)), 0, atol=1e-2)
        assert np.allclose(dmodBdzeta, 0, atol=1e-3)
        assert np.allclose(dmodBdzetah, 0, atol=1e-3)
        assert np.allclose(G, Gh, rtol=1e-3)
        assert np.allclose(iota, iotah, rtol=1e-3)
        assert np.allclose(diotads, diotadsh, rtol=1e-3)
        assert np.allclose(I, 0, atol=1e-3)
        assert np.allclose(Ih, 0, atol=1e-3)
        assert np.allclose(dGds, 0, atol=1e-3)
        assert np.allclose(dGdsh, 0, atol=1e-3)
        assert np.allclose(dIds, 0, atol=1e-3)
        assert np.allclose(dIdsh, 0, atol=1e-3)

    def test_interpolated_field_convergence_rate(self):
        vmec = Vmec(filename)
        order = 3
        bri = BoozerRadialInterpolant(vmec, order)

        # Perform interpolation from full grid
        points = np.zeros((len(vmec.s_half_grid)-1, 3))
        points[:, 0] = vmec.s_full_grid[1:-1]
        bri.set_points(points)

        nfp = vmec.indata.nfp
        smin = 0.1
        smax = 0.9
        thetamin = np.pi*(1/4)
        thetamax = np.pi*(3/4)
        zetamin = -2*np.pi/(4*nfp)
        zetamax = 2*np.pi/(4*nfp)
        old_err_1 = 1e6
        old_err_3 = 1e6
        for n in [4, 8, 16]:
            ssteps = n
            thetasteps = n
            zetasteps = n
            bsh = InterpolatedBoozerField(
                bri, 1, [smin, smax, ssteps], [thetamin, thetamax, thetasteps], [zetamin, zetamax, zetasteps],
                True, nfp=nfp, stellsym=True)
            err_1 = np.mean(bsh.estimate_error_modB(1000))
            err_3 = np.mean(bsh.estimate_error_iota(1000))

            assert err_1 < 0.6**2 * old_err_1
            assert err_3 < 0.6**2 * old_err_3

            old_err_1 = err_1
            old_err_3 = err_3


if __name__ == "__main__":
    unittest.main()
