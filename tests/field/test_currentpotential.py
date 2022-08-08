import unittest
from pathlib import Path
import json

import numpy as np
from monty.json import MontyDecoder, MontyEncoder

from simsopt.field.currentpotential import CurrentPotential, CurrentPotentialFourier
from simsopt.geo import SurfaceRZFourier, SurfaceHenneberg
from scipy.io import netcdf_file

TEST_DIR = Path(__file__).parent / ".." / "test_files"

stellsym_list = [True, False]

try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False

class CurrentPotentialTests(unittest.TestCase):
    def test_compare_K_with_regcoil(self):
        # This is an axisymmetric case with no Phimns
        nfp = 1
        stellsym = True
        mpol = 1
        ntor = 0
        mpol_potential=6
        ntor_potential=7
        ntheta = 32
        nphi = 32
        s = SurfaceRZFourier(nfp=nfp, ntheta = nphi, nphi = nphi,
                mpol=mpol, ntor=ntor, stellsym=stellsym)
        s.set_dofs(0*s.get_dofs())
        s.set_rc(0, 0, 6.5)
        s.set_rc(1, 0, 4.0)
        s.set_zs(1, 0, 4.0)

        cp = CurrentPotentialFourier(s, nfp=nfp, stellsym=stellsym, mpol=mpol_potential,
            ntor=ntor_potential, nphi=nphi, ntheta=ntheta,
            net_poloidal_current_amperes=1.0e6, net_toroidal_current_amperes=0.0)

        filename = TEST_DIR / 'regcoil_out.axisymmetry.nc'
        f = netcdf_file(filename, 'r')
        K2_regcoil = f.variables['K2'][()][-1,:,:]
        K = cp.K()
        K2 = np.sum(K*K,axis=2)
        K2_average = np.mean(K2,axis=(0,1))
        assert np.allclose(K2/K2_average, K2_regcoil/K2_average)

        # Now a non-axisymmetric test
        filename = TEST_DIR / 'regcoil_out.w7x.nc'
        f = netcdf_file(filename, 'r')
        rmnc_coil = f.variables['rmnc_coil'][()]
        zmns_coil = f.variables['zmns_coil'][()]
        xm_coil = f.variables['xm_coil'][()]
        xn_coil = f.variables['xn_coil'][()]
        nfp = f.variables['nfp'][()]
        ntheta = f.variables['ntheta_coil'][()]
        nzeta = f.variables['nzeta_coil'][()]
        mpol_potential = f.variables['mpol_potential'][()]
        ntor_potential = f.variables['ntor_potential'][()]
        net_poloidal_current_amperes = f.variables['net_poloidal_current_Amperes'][()]
        net_toroidal_current_amperes = f.variables['net_toroidal_current_Amperes'][()]
        xm_potential = f.variables['xm_potential'][()]
        xn_potential = f.variables['xn_potential'][()]
        single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()][-1,:]
        mpol = int(np.max(xm_coil))
        ntor = int(np.max(xn_coil)/nfp)

        s = SurfaceRZFourier(nfp=nfp, ntheta=ntheta, nphi=nzeta,
                mpol=mpol, ntor=ntor, stellsym=stellsym,range="field period")
        s.set_dofs(0*s.get_dofs())
        for im in range(len(xm_coil)):
            s.set_rc(xm_coil[im],int(xn_coil[im]/nfp),rmnc_coil[im])
            s.set_zs(xm_coil[im],int(xn_coil[im]/nfp),zmns_coil[im])

        cp = CurrentPotentialFourier(s, nfp=nfp, stellsym=stellsym, mpol=mpol_potential,
            ntor=ntor_potential, nphi=nzeta, ntheta=ntheta,
            net_poloidal_current_amperes=net_poloidal_current_amperes,
            net_toroidal_current_amperes=net_toroidal_current_amperes,range="field period")
        for im in range(len(xm_potential)):
            cp.set_phis(xm_potential[im],int(xn_potential[im]/nfp),single_valued_current_potential_mn[im])

        K2_regcoil = f.variables['K2'][()][-1,:,:]
        K = cp.K()
        K2 = np.sum(K*K,axis=2)
        K2_average = np.mean(K2,axis=(0,1))

        assert np.allclose(K2/K2_average, K2_regcoil/K2_average)

def get_currentpotential(cptype, stellsym, phis=None, thetas=None):
    np.random.seed(2)
    mpol = 4
    ntor = 3
    nfp = 2
    phis = phis if phis is not None else np.linspace(0, 1, 31, endpoint=False)
    thetas = thetas if thetas is not None else np.linspace(0, 1, 31, endpoint=False)

    if cptype == "CurrentPotentialFourier":
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, nfp=nfp, stellsym=stellsym, mpol=mpol, ntor=ntor,
                                     quadpoints_phi=phis, quadpoints_theta=thetas)
        cp.x = cp.x * 0.
        cp.phis[1, ntor + 0] = 0.3
    else:
        assert False

    dofs = cp.get_dofs()
    np.random.seed(2)
    rand_scale = 0.01
    cp.x = dofs + rand_scale * np.random.rand(len(dofs))  # .reshape(dofs.shape)
    return cp

class CurrentPotentialFourierTests(unittest.TestCase):

    def test_init(self):
        """
        Modeled after test_init in test_surface_rzfourier
        """

        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, nfp=2, mpol=3, ntor=2)
        self.assertEqual(cp.phis.shape, (4, 5))
        self.assertEqual(cp.phic.shape, (4, 5))

        cp = CurrentPotentialFourier(s, nfp=10, mpol=1, ntor=3, stellsym=False)
        self.assertEqual(cp.phis.shape, (2, 7))
        self.assertEqual(cp.phic.shape, (2, 7))

    def test_get_dofs(self):
        """
        Modeled after test_get_dofs from test_surface_rzfourier
        Test that we can convert the degrees of freedom into a 1D vector
        """

        # axisymmetric current potential
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=3, ntor=0)
        cp.phis[1, 0] = 0.2
        cp.phis[2, 0] = 1.3
        cp.phis[3, 0] = 0.7
        dofs = cp.get_dofs()
        self.assertEqual(dofs.shape, (3,))
        self.assertAlmostEqual(dofs[0], 0.2)
        self.assertAlmostEqual(dofs[1], 1.3)
        self.assertAlmostEqual(dofs[2], 0.7)

        # non-axisymmetric current potential
        cp = CurrentPotentialFourier(s, mpol=3, ntor=1)
        cp.phis[:, :] = [[100, 101, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        dofs = cp.get_dofs()
        self.assertEqual(dofs.shape, (3*(2*1 + 1) + 1,))
        for j in range(len(dofs)):
            self.assertAlmostEqual(dofs[j], j + 3)

    def test_set_dofs(self):
        """
        Modeled after test_get_dofs from test_surface_rzfourier
        Test that we can set the shape from a 1D vector
        """

        # First try an axisymmetric current potential for simplicity:
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=3, ntor=0)
        cp.set_dofs([-1.1, 0.7, 0.5])
        self.assertAlmostEqual(cp.phis[1, 0], -1.1)
        self.assertAlmostEqual(cp.phis[2, 0], 0.7)
        self.assertAlmostEqual(cp.phis[3, 0], 0.5)

        # Now try a nonaxisymmetric shape:
        cp = CurrentPotentialFourier(s, mpol=3, ntor=1)
        ndofs = len(cp.get_dofs())
        cp.set_dofs(np.array(list(range(ndofs))) + 1)
        self.assertAlmostEqual(cp.phis[0, 0], 0)
        self.assertAlmostEqual(cp.phis[0, 1], 0)
        self.assertAlmostEqual(cp.phis[0, 2], 1)
        self.assertAlmostEqual(cp.phis[1, 0], 2)
        self.assertAlmostEqual(cp.phis[1, 1], 3)
        self.assertAlmostEqual(cp.phis[1, 2], 4)
        self.assertAlmostEqual(cp.phis[2, 0], 5)
        self.assertAlmostEqual(cp.phis[2, 1], 6)
        self.assertAlmostEqual(cp.phis[2, 2], 7)
        self.assertAlmostEqual(cp.phis[3, 0], 8)
        self.assertAlmostEqual(cp.phis[3, 1], 9)
        self.assertAlmostEqual(cp.phis[3, 2], 10)

        # Now try a nonaxisymmetric shape without stell sym
        cp = CurrentPotentialFourier(s, mpol=3, ntor=1, stellsym=False)
        ndofs = len(cp.get_dofs())
        cp.set_dofs(np.array(list(range(ndofs))) + 1)
        self.assertAlmostEqual(cp.phis[0, 0], 0)
        self.assertAlmostEqual(cp.phis[0, 1], 0)
        self.assertAlmostEqual(cp.phis[0, 2], 1)
        self.assertAlmostEqual(cp.phis[1, 0], 2)
        self.assertAlmostEqual(cp.phis[1, 1], 3)
        self.assertAlmostEqual(cp.phis[1, 2], 4)
        self.assertAlmostEqual(cp.phis[2, 0], 5)
        self.assertAlmostEqual(cp.phis[2, 1], 6)
        self.assertAlmostEqual(cp.phis[2, 2], 7)
        self.assertAlmostEqual(cp.phis[3, 0], 8)
        self.assertAlmostEqual(cp.phis[3, 1], 9)
        self.assertAlmostEqual(cp.phis[3, 2], 10)

        self.assertAlmostEqual(cp.phic[0, 0], 0)
        self.assertAlmostEqual(cp.phic[0, 1], 11)
        self.assertAlmostEqual(cp.phic[0, 2], 12)
        self.assertAlmostEqual(cp.phic[1, 0], 13)
        self.assertAlmostEqual(cp.phic[1, 1], 14)
        self.assertAlmostEqual(cp.phic[1, 2], 15)
        self.assertAlmostEqual(cp.phic[2, 0], 16)
        self.assertAlmostEqual(cp.phic[2, 1], 17)
        self.assertAlmostEqual(cp.phic[2, 2], 18)
        self.assertAlmostEqual(cp.phic[3, 0], 19)
        self.assertAlmostEqual(cp.phic[3, 1], 20)
        self.assertAlmostEqual(cp.phic[3, 2], 21)

    def test_change_resolution(self):
        """
        Modeled after test_get_dofs from test_surface_rzfourier
        Check that we can change mpol and ntor.
        """
        for mpol in [1, 2]:
            for ntor in [0, 1]:
                for stellsym in [True, False]:
                    s = SurfaceRZFourier()
                    cp = CurrentPotentialFourier(s, mpol=mpol, ntor=ntor)
                    ndofs = len(cp.get_dofs())
                    cp.set_dofs((np.random.rand(ndofs) - 0.5) * 0.01)
                    cp.set_phis(1, 0, 0.13)
                    phi1 = cp.Phi()

                    cp.change_resolution(mpol+1, ntor)
                    s.recalculate = True
                    phi2 = cp.Phi()
                    self.assertTrue(np.allclose(phi1, phi2))

    def test_get_phic(self):
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=1, ntor=0, stellsym=False)
        cp.x = [0.7, 2.9, -1.1]
        self.assertAlmostEqual(cp.get_phic(0, 0), 2.9)
        self.assertAlmostEqual(cp.get_phic(1, 0), -1.1)

    def test_get_phis(self):
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=3, ntor=1)
        ndofs = len(cp.get_dofs())
        cp.x = np.array(list(range(ndofs))) + 1

        self.assertAlmostEqual(cp.get_phis(0, -1), 0)
        self.assertAlmostEqual(cp.get_phis(0, 0), 0)
        self.assertAlmostEqual(cp.get_phis(0, 1), 1)
        self.assertAlmostEqual(cp.get_phis(1, -1), 2)
        self.assertAlmostEqual(cp.get_phis(1, 0), 3)
        self.assertAlmostEqual(cp.get_phis(1, 1), 4)
        self.assertAlmostEqual(cp.get_phis(2, -1), 5)
        self.assertAlmostEqual(cp.get_phis(2, 0), 6)
        self.assertAlmostEqual(cp.get_phis(2, 1), 7)
        self.assertAlmostEqual(cp.get_phis(3, -1), 8)
        self.assertAlmostEqual(cp.get_phis(3, 0), 9)
        self.assertAlmostEqual(cp.get_phis(3, 1), 10)

    def test_set_phic(self):
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=1, ntor=0, stellsym=False)
        cp.x = [2.9, -1.1, 0.7]
        cp.set_phic(0, 0, 3.1)
        self.assertAlmostEqual(cp.x[1], 3.1)

    def test_set_phis(self):
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, mpol=1, ntor=0, stellsym=True)
        s.x = [2.9, -1.1, 0.7]
        cp.set_phis(1, 0, 1.4)
        self.assertAlmostEqual(cp.x[0], 1.4)

    def test_names_order(self):
        """
        Verify that the order of phis and phic in the dof names is
        correct. This requires that the order of these four arrays in
        ``_make_names()`` matches the order in the C++ functions
        ``set_dofs_impl()`` and ``get_dofs()`` in
        ``src/simsoptpp/currentpotentialfourier.h``.
        """
        mpol = 1
        ntor = 1
        nfp = 4
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, nfp=nfp, mpol=mpol, ntor=ntor, stellsym=False)
        cp.set_phic(0, 0, 100.0)
        cp.set_phis(0, 1, 200.0)
        self.assertAlmostEqual(cp.get('Phic(0,0)'), 100.0)
        self.assertAlmostEqual(cp.get('Phis(0,1)'), 200.0)

    def test_mn(self):
        """
        Test the arrays of mode numbers m and n.
        """
        mpol = 3
        ntor = 2
        nfp = 4
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, nfp=nfp, mpol=mpol, ntor=ntor)
        m_correct = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        n_correct = [1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]
        np.testing.assert_array_equal(cp.m, m_correct)
        np.testing.assert_array_equal(cp.n, n_correct)

        m_correct = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        n_correct = [1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]

        cp = CurrentPotentialFourier(s, nfp=nfp, mpol=mpol, ntor=ntor, stellsym=False)
        np.testing.assert_array_equal(cp.m, m_correct + [0] + m_correct)
        np.testing.assert_array_equal(cp.n, n_correct + [0] + n_correct)

    def test_mn_matches_names(self):
        """
        Verify that the m and n attributes match the dof names.
        """
        mpol = 2
        ntor = 3
        nfp = 5
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, nfp=nfp, mpol=mpol, ntor=ntor, stellsym=True)
        names = [name[4:] for name in cp.local_dof_names]
        names2 = [f'({m},{n})' for m, n in zip(cp.m, cp.n)]
        self.assertEqual(names, names2)

        # Now try a non-stellarator-symmetric case:
        cp = CurrentPotentialFourier(s, nfp=nfp, mpol=mpol, ntor=ntor, stellsym=False)
        assert 'Phic(0,0)' in cp.local_dof_names
        names = [name[4:] for name in cp.local_dof_names]
        names2 = [f'({m},{n})' for m, n in zip(cp.m, cp.n)]
        self.assertEqual(names, names2)
