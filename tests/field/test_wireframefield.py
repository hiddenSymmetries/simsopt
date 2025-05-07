import unittest
from pathlib import Path
import numpy as np
from simsopt.geo import CurveXYZFourier, SurfaceRZFourier, ToroidalWireframe
from simsopt.field import WireframeField, enclosed_current
from simsopt.objectives import SquaredFlux

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()


def surf_torus(nfp, rmaj, rmin):
    surf = SurfaceRZFourier(nfp=nfp, mpol=1, ntor=0)
    surf.set_rc(0, 0, rmaj)
    surf.set_rc(1, 0, rmin)
    surf.set_zs(1, 0, rmin)
    return surf


class WireframeFieldTests(unittest.TestCase):

    def test_toroidal_wireframe_field_from_square_loop(self):
        '''
        Compares field calculated at the center of a current-carrying square 
        loop with the analytical result. The square lies in the XY plane with 
        its vertices at (1,0), (0,1), (-1,0), and (0,-1).
        '''

        curr = 1e6

        surf_wf = SurfaceRZFourier(nfp=1, mpol=1, ntor=0)
        surf_wf.set_rc(0, 0, 2)
        surf_wf.set_rc(1, 0, 1)
        surf_wf.set_zs(1, 0, 1)

        test_wf = ToroidalWireframe(surf_wf, 2, 4)
        test_wf.currents[[2, 6]] = curr
        self.assertTrue(test_wf.check_constraints())

        pts_test = np.zeros((1, 3))
        B_analytic = np.array([0, 0, 8*1e-7*curr])
        dBdX_analytic = np.zeros((3, 3))

        field_wf = WireframeField(test_wf)
        field_wf.set_points(pts_test)
        assert np.allclose(field_wf.B(), B_analytic)
        assert np.allclose(field_wf.dB_by_dX(), dBdX_analytic)

    def test_toroidal_wireframe_toroidal_field_convergence(self):
        '''
        Verifies that the field inside a ToroidalWireframe with poloidal 
        currents approaches an ideal toroidal field as the toroidal wireframe 
        resolution increases.
        '''

        curr = 1e6

        # Test points inside the wireframe where field is evaluated
        n_test_pts = 10
        rtest = np.linspace(1.7, 2.3, n_test_pts)
        ztest = np.linspace(-0.3, 0.3, n_test_pts)
        ptest = np.linspace(0, 0.5*np.pi, n_test_pts)
        xtest = rtest*np.cos(ptest)
        ytest = rtest*np.sin(ptest)

        test_pts = np.zeros((n_test_pts, 3))
        test_pts[:, 0] = xtest[:]
        test_pts[:, 1] = ytest[:]
        test_pts[:, 2] = ztest[:]

        # Analytic field at each test point
        modB = -2*1e-7*curr/rtest
        B_analytic = np.zeros((n_test_pts, 3))
        B_analytic[:, 0] = -modB*np.sin(ptest)
        B_analytic[:, 1] = modB*np.cos(ptest)

        # Analytic field Jacobian at each test point
        dBdX_analytic = np.zeros((n_test_pts, 3, 3))
        dBdX_analytic[:, 0, 0] = modB/rtest * np.sin(2*ptest)
        dBdX_analytic[:, 0, 1] = modB/rtest * (np.sin(ptest)**2-np.cos(ptest)**2)
        dBdX_analytic[:, 1, 0] = modB/rtest * (np.sin(ptest)**2-np.cos(ptest)**2)
        dBdX_analytic[:, 1, 1] = -modB/rtest * np.sin(2*ptest)

        err_B = np.inf
        err_dBdX = np.inf
        for i in range(10, 15):

            # Set up a wireframe on an ideal torus
            n_phi = 2*i
            test_wf = ToroidalWireframe(surf_torus(2, 2, 1), n_phi, 4)
            test_wf.currents[-test_wf.n_pol_segments:] = \
                curr/(2*n_phi*test_wf.nfp)
            self.assertTrue(test_wf.check_constraints())

            field_wf = WireframeField(test_wf)
            field_wf.set_points(test_pts)
            new_err_B = np.max(np.abs(field_wf.B() - B_analytic))
            new_err_dBdX = np.max(np.abs(field_wf.dB_by_dX() - dBdX_analytic))
            #print('%.8e  %.8e  %.8e  %.8e' % (new_err_B, new_err_dBdX, \
            #          new_err_B/err_B, new_err_dBdX/err_dBdX))
            assert new_err_B < 0.5*err_B
            assert new_err_dBdX < 0.5*err_dBdX
            err_B = new_err_B
            err_dBdX = new_err_dBdX

    def test_toroidal_wireframe_curlB(self):
        '''
        Check that the curl of the magnetic field produced by a wireframe is
        zero if current continuity is satisfied and nonzero otherwise.
        '''

        cur_pol = 1e6
        cur_tor = 2e6

        # Test points inside the wireframe where field is evaluated
        n_test_pts = 10
        rtest = np.linspace(1.7, 2.3, n_test_pts)
        ztest = np.linspace(-0.3, 0.3, n_test_pts)
        ptest = np.linspace(0, 0.5*np.pi, n_test_pts)
        xtest = rtest*np.cos(ptest)
        ytest = rtest*np.sin(ptest)

        test_pts = np.zeros((n_test_pts, 3))
        test_pts[:, 0] = xtest[:]
        test_pts[:, 1] = ytest[:]
        test_pts[:, 2] = ztest[:]

        # Set up the wireframe
        n_phi = 4
        n_theta = 6
        test_wf = ToroidalWireframe(surf_torus(2, 2, 1), n_phi, n_theta)

        # Currents satisfying the constraints
        test_wf.currents[-test_wf.n_pol_segments:] = \
            -cur_pol/(2*n_phi*test_wf.nfp)
        test_wf.currents[:test_wf.n_tor_segments] = -cur_tor/n_theta
        self.assertTrue(test_wf.check_constraints())

        # Check that curlB is zero at each test point
        field_wf = WireframeField(test_wf)
        field_wf.set_points(test_pts)
        dBdX = field_wf.dB_by_dX()
        for i in range(n_test_pts):
            self.assertTrue(np.allclose(dBdX[i, :, :], dBdX[i, :, :].T))

        # Replace currents with values that do NOT satisfy the constraints
        test_wf.currents[:] = 0
        test_wf.currents[-int(0.5*test_wf.n_pol_segments):] = -cur_pol
        test_wf.currents[:int(0.5*test_wf.n_tor_segments)] = -cur_tor
        self.assertFalse(test_wf.check_constraints())

        # Check that curlB is not zero at each test point
        field_wf = WireframeField(test_wf)
        field_wf.set_points(test_pts)
        dBdX = field_wf.dB_by_dX()
        for i in range(n_test_pts):
            self.assertFalse(np.allclose(dBdX[i, :, :], dBdX[i, :, :].T))

    def test_toroidal_wireframe_amperian_loops(self):
        '''
        Verifies that the prescribed total toroidal and poloidal currents in 
        a wireframe agree with calculations of the field along Amperian loops
        '''

        cur_pol = 1e6
        cur_tor = 2e6

        # Surface on which wireframe is constructed
        surf = surf_torus(2, 2, 1)

        # Set up the wireframe
        n_phi = 4
        n_theta = 6
        test_wf = ToroidalWireframe(surf, n_phi, n_theta)
        test_wf.currents[-test_wf.n_pol_segments:] = \
            -cur_pol/(2*n_phi*test_wf.nfp)
        test_wf.currents[:test_wf.n_tor_segments] = -cur_tor/n_theta
        self.assertTrue(test_wf.check_constraints())

        # Wireframe field class instance
        field_wf = WireframeField(test_wf)

        # Amperian loop through which poloidal current flows
        amploop = CurveXYZFourier(10, 1)
        amploop.set('xc(1)', surf.get_rc(0, 0))
        amploop.set('ys(1)', surf.get_rc(0, 0))
        amploop_curr = enclosed_current(amploop, field_wf, 200)
        #print('    Enclosed poloidal current: %.8e' % (amploop_curr))
        #print('                    (expected: %.8e)' % (cur_pol))
        assert np.allclose(amploop_curr, cur_pol)

        # Amperian loop through which toroidal current flows
        amploop = CurveXYZFourier(10, 1)
        amploop.set('xc(0)', surf.get_rc(0, 0))
        amploop.set('xc(1)', 2*surf.get_rc(1, 0))
        amploop.set('zs(1)', 2*surf.get_zs(1, 0))
        amploop_curr = enclosed_current(amploop, field_wf, 200)
        #print('    Enclosed toroidal current: %.8e' % (amploop_curr))
        #print('                    (expected: %.8e)' % (cur_tor))
        assert np.allclose(amploop_curr, cur_tor)

    def test_toroidal_wireframe_add_tfcoil_currents(self):
        '''
        Verifies the functionality of the add_tfcoil_currents method with an
        Amperian loop.
        '''

        cur_pol = 1e6

        # Surface on which wireframe is constructed
        surf = surf_torus(2, 2, 1)

        # Set up the wireframe
        n_phi = 12
        n_theta = 4
        test_wf = ToroidalWireframe(surf, n_phi, n_theta)

        test_wf.set_poloidal_current(-cur_pol)

        # Amperian loop through which poloidal current flows
        amploop = CurveXYZFourier(10, 1)
        amploop.set('xc(1)', surf.get_rc(0, 0))
        amploop.set('ys(1)', surf.get_rc(0, 0))

        for n_tf in [2, 4, 6]:

            test_wf.currents[:] = 0
            test_wf.add_tfcoil_currents(n_tf, -cur_pol/(n_tf*2*test_wf.nfp))

            # Make sure the constraints are fulfilled
            self.assertTrue(test_wf.check_constraints())

            # Calculate the field from the wireframe
            field_wf = WireframeField(test_wf)

            # Verify correctness of field along Amperian loop
            amploop_curr = enclosed_current(amploop, field_wf, 200)
            assert np.allclose(amploop_curr, cur_pol)

    def test_toroidal_wireframe_squared_flux(self):
        '''
        Verifies that the squared flux on a plasma boundary calculated through
        the inductance matrix matches the calculation of the SquaredFlux class
        '''

        # Net poloidal and toroidal current (just needs to be nontrivial)
        cur_pol = 1e6
        cur_tor = 2e6

        # Use the rotating ellipse as the plasma boundary
        plas_fname = TEST_DIR / 'input.rotating_ellipse'
        surf_plas = SurfaceRZFourier.from_vmec_input(plas_fname)

        # Set up the wireframe
        n_phi = 2
        n_theta = 4
        surf_wf = SurfaceRZFourier.from_vmec_input(plas_fname)
        surf_wf.extend_via_normal(1.0)
        test_wf = ToroidalWireframe(surf_wf, n_phi, n_theta)
        test_wf.currents[-test_wf.n_pol_segments:] = \
            -cur_pol/(2*n_phi*test_wf.nfp)
        test_wf.currents[:test_wf.n_tor_segments] = -cur_tor/n_theta
        self.assertTrue(test_wf.check_constraints())

        # Get the inductance matrix
        field_wf = WireframeField(test_wf)
        field_wf.set_points(surf_plas.gamma().reshape((-1, 3)))
        Amat = field_wf.dBnormal_by_dsegmentcurrents_matrix(
            surf_plas, area_weighted=True)

        # Squared flux integral calculated in two ways
        sq_flux_mat = 0.5*np.sum(
            np.matmul(Amat, test_wf.currents.reshape((-1, 1)))**2)
        sq_flux_ref = SquaredFlux(surf_plas, field_wf).J()

        assert np.allclose(sq_flux_mat, sq_flux_ref)

    def test_wireframefield_initialization(self):
        '''
        Tests of the initialization of a wireframe field
        '''

        # Initialization for a ToroidalWireframe
        surf = surf_torus(2, 2, 1)
        n_phi = 4
        n_theta = 6
        test_wf_tor = ToroidalWireframe(surf, n_phi, n_theta)
        field_wf_tor = WireframeField(test_wf_tor)

        with self.assertRaises(ValueError):
            field_wf_tor.dBnormal_by_dsegmentcurrents_matrix(None)


if __name__ == "__main__":
    unittest.main()
