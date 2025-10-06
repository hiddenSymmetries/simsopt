import unittest
import numpy as np

from simsopt.field.magneticfieldclasses import ToroidalField
from simsopt.field.tracing import (
    PoincarePlotter,
    SimsoptFieldlineIntegrator,
    ScipyFieldlineIntegrator,
)
from simsopt.configs.zoo import get_data
from monty.tempfile import ScratchDir
import os


class TestPoincarePlotterSimsopt(unittest.TestCase):
    def setUp(self):
        """ set up a simple toroidal field and a plotter
        with two fieldlines"""
        self.R0 = 1.25
        self.B0 = 0.9
        self.field = ToroidalField(self.R0, self.B0)
        # 2 fieldlines on midplane
        self.R0s = np.array([self.R0 + 0.02, self.R0 + 0.05])
        self.Z0s = np.zeros_like(self.R0s)
        # integrators expect shape (nlines, 2) with columns (R,Z)
        self.start_points_RZ = np.column_stack([self.R0s, self.Z0s])
        self.intg = SimsoptFieldlineIntegrator(self.field, nfp=1, stellsym=True, R0=self.R0, tmax=200.0, tol=1e-9)
        self.pp = PoincarePlotter(self.intg, self.start_points_RZ, phis=4, n_transits=2, add_symmetry_planes=False)

    def test_res_properties_and_invariants(self):
        """
        test that the results are of the correct shape
        for the test parameters
        """
        tys = self.pp.res_tys
        hits = self.pp.res_phi_hits
        self.assertEqual(len(tys), self.start_points_RZ.shape[0])
        self.assertEqual(len(hits), self.start_points_RZ.shape[0])
        # invariants for ToroidalField: R const, Z const
        for i, h in enumerate(hits):
            r = np.sqrt(h[:, 2] ** 2 + h[:, 3] ** 2)
            z = h[:, 4]
            self.assertTrue(np.allclose(r, self.R0s[i], atol=1e-8))
            self.assertTrue(np.allclose(z, self.Z0s[i], atol=1e-10))

    def test_plane_hits_methods(self):
        """
        test that the plane hits are correctly 
        deduced from the results
        """
        # plane 0 exists since phis=4
        hits_cart = self.pp.plane_hits_cart(0)
        hits_cyl = self.pp.plane_hits_cyl(0)
        self.assertEqual(len(hits_cart), self.start_points_RZ.shape[0])
        self.assertEqual(len(hits_cyl), self.start_points_RZ.shape[0])
        for i in range(len(hits_cart)):
            self.assertGreater(hits_cart[i].shape[0], 0)
            self.assertEqual(hits_cart[i].shape[1], 3)
            self.assertEqual(hits_cyl[i].shape[1], 2)

    def test_plotting_methods_matplotlib(self):
        # All plotting should be non-interactive and not raise
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        # Ensure non-interactive backend
        import matplotlib
        try:
            matplotlib.use('Agg')
        except Exception:
            pass
        # Single plane by index
        fig, ax = self.pp.plot_poincare_plane_idx(0)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        # Single plane by value using existing phi to avoid modifying internal array
        phi_exist = self.pp.phis[0]
        fig2, ax2 = self.pp.plot_poincare_single(phi_exist, prevent_recompute=True)
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(ax2)
        # All planes
        fig3, axs = self.pp.plot_poincare_all()
        self.assertIsNotNone(fig3)
    # 3D trajectories and 3D poincare in matplotlib backend
    # Upstream code has an undefined 'color' variable for matplotlib branch; skip these calls for now.
    # Once upstream is fixed, these can be re-enabled.

    def test_randomcolors_and_lost(self):
        """
        test some methbods used in the plotting
        """
        colors = self.pp.randomcolors
        self.assertEqual(colors.shape[0], self.start_points_RZ.shape[0])
        self.assertEqual(colors.shape[1], 3)
        # ToroidalField should not trigger loss
        self.assertTrue(all(not x for x in self.pp.lost))


class TestPoincarePlotterScipy(unittest.TestCase):
    def setUp(self):
        """set up a simple toroidal field and a plotter with three field lines using scipy integrator"""
        self.R0 = 1.1
        self.B0 = 0.7
        self.field = ToroidalField(self.R0, self.B0)
        self.R0s = np.array([self.R0 + 0.01, self.R0 + 0.03, self.R0 + 0.05])
        self.Z0s = np.zeros_like(self.R0s)
        self.start_points_RZ = np.column_stack([self.R0s, self.Z0s])
        self.intg = ScipyFieldlineIntegrator(
            self.field,
            nfp=1,
            stellsym=True,
            R0=self.R0,
            integrator_type='RK45',
            integrator_args={'rtol': 1e-9, 'atol': 1e-11},
        )
        self.pp = PoincarePlotter(self.intg, self.start_points_RZ, phis=None, n_transits=2, add_symmetry_planes=False)

    def test_res_properties_and_plane_hits(self):
        tys = self.pp.res_tys
        hits = self.pp.res_phi_hits
        self.assertEqual(len(tys), self.start_points_RZ.shape[0])
        self.assertEqual(len(hits), self.start_points_RZ.shape[0])
        # Check plane hits for a couple of planes
        for plane_idx in [0]:
            hits_cart = self.pp.plane_hits_cart(plane_idx)
            hits_cyl = self.pp.plane_hits_cyl(plane_idx)
            self.assertEqual(len(hits_cart), self.start_points_RZ.shape[0])
            self.assertEqual(len(hits_cyl), self.start_points_RZ.shape[0])

    def test_plotting_methods_matplotlib(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        # Ensure non-interactive backend
        import matplotlib
        try:
            matplotlib.use('Agg')
        except Exception:
            pass
        # Index-based plot
        fig, ax = self.pp.plot_poincare_plane_idx(0)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        # Value-based plot for an existing phi
        phi_exist = self.pp.phis[0]
        fig2, ax2 = self.pp.plot_poincare_single(phi_exist, prevent_recompute=True)
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(ax2)
        # Grid of planes
        fig3, axs = self.pp.plot_poincare_all()
        self.assertIsNotNone(fig3)


class TestPoincarePlotterFactory(unittest.TestCase):
    def test_from_field_factory(self):
        """
        test that the classmethod to skip integrator creation works
        """
        R0 = 1.15
        B0 = 0.85
        field = ToroidalField(R0, B0)
        start_points_RZ = np.array([[R0 + 0.02, 0.0], [R0 + 0.04, 0.0]])
        pp = PoincarePlotter.from_field(field, start_points_RZ, phis=4, n_transits=2, add_symmetry_planes=False)
        # Basic shape checks
        self.assertEqual(len(pp.res_phi_hits), start_points_RZ.shape[0])
        self.assertEqual(pp.res_phi_hits[0].shape[1], 5)
        # Ensure phis interpreted correctly (int -> equally spaced)
        self.assertEqual(len(pp.plot_phis), 4)


class TestPoincarePlotterRealField(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ same but with NCSX coils, also testing cache invalidation on coil current change """
        # Load a realistic configuration (NCSX)
        base_curves, base_currents, ma, nfp, bs = get_data('ncsx', coil_order=5, magnetic_axis_order=6, points_per_period=4)
        cls.bs = bs
        # Build field from BiotSavart
        cls.field = bs  # BiotSavart acts as a field
        # Choose a couple of starting points near magnetic axis radius
        cls.R0 = float(ma.gamma()[0, 0])  # R of magnetic axis
        cls.n_transits = 1
        cls.n_planes = 3
        cls.start_points_RZ = np.array([
            [cls.R0 + 0.01, 0.0],
            [cls.R0 + 0.03, 0.0],
        ])
        cls.intg = SimsoptFieldlineIntegrator(cls.field, nfp=nfp, stellsym=True, R0=cls.R0, tmax=50.0, tol=1e-7)
        cls.pp = PoincarePlotter(cls.intg, cls.start_points_RZ, phis=cls.n_planes, n_transits=cls.n_transits, add_symmetry_planes=True)

    def test_basic_hits_exist(self):
        hits = self.pp.res_phi_hits
        self.assertEqual(len(hits), self.start_points_RZ.shape[0])
        for arr in hits:
            self.assertGreater(arr.shape[0], 0)
            self.assertEqual(arr.shape[1], 5)

    def test_cache_invalidation_on_current_change(self):
        # Prime cache
        hits_before = [h.copy() for h in self.pp.res_phi_hits]
        # Modify a coil current DOF: set first current to zero
        old_val = self.bs.coils[0].current.x.copy()
        self.bs.coils[0].current.x = old_val * 1.01  # small change but trigger cache invalidation.
        # After DOF change, force recompute by toggling start_points (setter) or phis
        # Accessing res_phi_hits again should recompute and differ
        new_hits = self.pp.res_phi_hits
        # Compare number of rows or any shape difference first; if same shape, compare subset limited to min length
        diff_any = False
        for a, b in zip(hits_before, new_hits):
            m = min(a.shape[0], b.shape[0])
            if not np.allclose(a[:m, :], b[:m, :]):
                diff_any = True
                break
        self.assertTrue(diff_any, "Poincare hits did not appear to change after modifying a coil current; cache not invalidated?")
        # restore value to avoid side effects
        self.bs.coils[0].current.x = old_val


class TestPoincarePlotter3DBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R0 = 1.2
        cls.B0 = 0.8
        cls.field = ToroidalField(cls.R0, cls.B0)
        cls.start_points_RZ = np.array([[cls.R0 + 0.02, 0.0]])
        cls.intg = SimsoptFieldlineIntegrator(cls.field, nfp=1, stellsym=True, R0=cls.R0, tmax=50.0, tol=1e-9)
        cls.pp = PoincarePlotter(cls.intg, cls.start_points_RZ, phis=3, n_transits=1, add_symmetry_planes=False)

    def test_matplotlib_3d(self):
        try:
            import matplotlib  # noqa: F401
            matplotlib.use('Agg')
        except Exception:
            self.skipTest('matplotlib not available')
        # Should not raise
        self.pp.plot_fieldline_trajectories_3d(engine='matplotlib', show=False)
        self.pp.plot_poincare_in_3d(engine='matplotlib', show=False)

    def test_plotly_3d(self):
        try:
            import plotly  # noqa: F401
        except Exception:
            self.skipTest('plotly not installed')
        self.pp.plot_fieldline_trajectories_3d(engine='plotly', show=False)
        self.pp.plot_poincare_in_3d(engine='plotly', show=False)

    def test_mayavi_3d(self):
        try:
            import mayavi  # noqa: F401
        except Exception:
            self.skipTest('mayavi not installed')
        # Use show=False for trajectories; poincare skip show
        self.pp.plot_fieldline_trajectories_3d(engine='mayavi', show=False)
        self.pp.plot_poincare_in_3d(engine='mayavi', show=False)


class TestPoincarePlotterSaveLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_curves, base_currents, ma, nfp, bs = get_data('ncsx', coil_order=4, magnetic_axis_order=4, points_per_period=3)
        cls.bs = bs
        cls.nfp = nfp
        cls.R0 = float(ma.gamma()[0, 0])
        cls.start_points_RZ = np.array([
            [cls.R0 + 0.02, 0.0],
            [cls.R0 + 0.04, 0.0],
        ])
        cls.intg = SimsoptFieldlineIntegrator(cls.bs, nfp=nfp, stellsym=True, R0=cls.R0, tmax=40.0, tol=1e-7)

    def test_save_and_load_with_dof_change(self):
        """
        test that the hashing, saving and loading works as intended. 
        """
        with ScratchDir('.'):
            archive = 'poincare_data.npz'
            pp = PoincarePlotter(self.intg, self.start_points_RZ, phis=4, n_transits=1, add_symmetry_planes=True, store_results=True)
            _ = pp.res_phi_hits  # prime cache and save to disk
            # Check archive created with hashed datasets
            self.assertTrue(os.path.exists(archive))
            with np.load(archive, allow_pickle=True) as data:
                hash_key = str(pp.poincare_hash)
                self.assertIn(f'res_phi_{hash_key}', data.files)
                self.assertIn(f'res_tys_{hash_key}', data.files)

            # A new instance with same params should already have data
            pp2 = PoincarePlotter(self.intg, self.start_points_RZ, phis=4, n_transits=1, add_symmetry_planes=True, store_results=True)
            # comparing to hidden attributes to avoid recompute.
            self.assertTrue(np.array_equal(pp.res_phi_hits, pp2._res_phi_hits))
            self.assertTrue(np.array_equal(pp.res_tys, pp2._res_tys))

            # change an element of the res_phi_hits and res tys, to make sure this modification is the one that is read:
            pp2._res_phi_hits[0][0, 0] = 1e5
            pp2._res_tys[0][0, 0] = 1e5
            pp2.save_to_disk()  # overwrite datasets inside archive

            # Change a dof to invalidate cache of both plotters:
            old_val = self.bs.coils[0].current.x.copy()
            self.bs.coils[0].current.x *= old_val * 1.02
            self.assertIsNone(pp._res_tys)
            self.assertIsNone(pp2._res_tys)

            # return the dof and see that the modified file is read from disk
            self.bs.coils[0].current.x = old_val
            pp2_from_disk = pp2.res_phi_hits  # should read from disk
            pp2_tys_from_disk = pp2.res_tys
            self.assertTrue(pp2_from_disk[0][0, 0] == 1e5)
            self.assertTrue(pp2_tys_from_disk[0][0, 0] == 1e5)
            
            #test removing the poincare cache file
            pp2.remove_poincare_data()
            self.assertFalse(os.path.exists(archive))

    def test_save_to_vtk(self):
        """
        test that the hashing, saving and loading works as intended. 
        """
        with ScratchDir('.'):
            pp = PoincarePlotter(self.intg, self.start_points_RZ, phis=4, n_transits=2, add_symmetry_planes=True, store_results=False)
            filename = "test"
            pp.particles_to_vtk(filename)
            self.assertTrue(os.path.exists(f"{filename}.vtu"))
    


if __name__ == '__main__':
    unittest.main()
