import sys
import unittest
from unittest import mock
import numpy as np

from simsopt.field.magneticfieldclasses import ToroidalField
from simsopt.field.integrator import SimsoptFieldlineIntegrator, ScipyFieldlineIntegrator
from simsopt.field.poincareplotter import PoincarePlotter
from simsopt.field.tracing import MinRStoppingCriterion
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
        self.intg = SimsoptFieldlineIntegrator(self.field, tmax=200.0, tol=1e-9)
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
        fig, ax = self.pp.plot_poincare_plane_idx(0, color='blue')
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

    def test_setters(self):
        """
        test that the setters for start points and phis work
        """
        self.pp.need_to_recompute = False  # reset flag
        # change start points
        new_start_points = np.array([[self.R0 + 0.03, 0.0], [self.R0 + 0.06, 0.0], [self.R0 + 0.09, 0.0]])
        self.pp.start_points_RZ = new_start_points
        self.assertEqual(self.pp.start_points_RZ.shape[0], 3)
        self.assertIsNone(self.pp._res_phi_hits)
        self.assertIsNone(self.pp._res_tys)
        self.assertTrue(self.pp.need_to_recompute)

        self.pp.need_to_recompute = False  # reset flag
        # change phis
        new_phis = np.array([0.0, 0.2, 0.4])
        self.pp.phis = new_phis
        self.assertTrue(np.array_equal(self.pp.phis, new_phis))
        self.assertIsNone(self.pp._res_phi_hits)
        self.assertIsNone(self.pp._res_tys)
        self.assertTrue(self.pp.need_to_recompute)

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

    def test_raise_error(self):
        """
        test that an error is raised when trying to plot a plane
        that does not exist
        """
        with self.assertRaises(ValueError):
            self.pp.plot_poincare_plane_idx(10)  # out of bounds
        with self.assertRaises(ValueError):
            self.pp.plot_poincare_single(0.123, prevent_recompute=True)  # not in phis

    def test_simple_plots(self):
        """
        test that the simple plotting functions run without error
        """
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
        from simsopt.geo import SurfaceRZFourier
        surface = SurfaceRZFourier()
        num_planes = self.pp.phis.shape[0]

        _ = self.pp.res_phi_hits
        self.pp._res_phi_hits[-1][-1,1] = -2  #simulate a lost particle
        # Simple plots
        fig1, ax1 = self.pp.plot_poincare_single(self.pp.phis[0], include_symmetry_planes=False, mark_lost=True)
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(ax1)
        self.assertEqual(self.pp.phis.shape[0], num_planes)

        fig1, ax1 = self.pp.plot_poincare_single(0.123, prevent_recompute=False, surf=surface)
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(ax1)
        self.assertEqual(self.pp.phis.shape[0], num_planes+1)



        fig2, axs2 = self.pp.plot_poincare_all()
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(axs2)


class TestPoincarePlotterFactory(unittest.TestCase):
    def test_from_field_factory(self):
        """
        test that the classmethod to skip integrator creation works
        """
        R0 = 1.15
        B0 = 0.85
        field = ToroidalField(R0, B0)
        start_points_RZ = np.array([[R0 + 0.02, 0.0], [R0 + 0.04, 0.0]])
        pp = PoincarePlotter.from_field(field, start_points_RZ, phis=None, n_transits=2, add_symmetry_planes=False)
        # Basic shape checks
        self.assertEqual(len(pp.res_phi_hits), start_points_RZ.shape[0])
        self.assertEqual(pp.res_phi_hits[0].shape[1], 5)
        # Ensure phis interpreted correctly (int -> equally spaced)
        self.assertEqual(len(pp.phis_for_plotting), 4)
        
        #set with array of phis
        phis_array = np.array([0, 0.1, 0.2])
        pp2 = PoincarePlotter.from_field(field, start_points_RZ, phis=phis_array, integrator_type='scipy', n_transits=2, add_symmetry_planes=False)
        self.assertTrue(np.array_equal(pp2.phis[:3], phis_array))

        with self.assertRaises(ValueError):
            PoincarePlotter.from_field(field, start_points_RZ, n_transits=2, add_symmetry_planes=False, integrator_type='unknown')

        # stopping criteria are passed to either integrator
        criterion = MinRStoppingCriterion(0.0)
        for integrator_type in ['simsopt', 'scipy']:
            pp3 = PoincarePlotter.from_field(field, start_points_RZ, integrator_type=integrator_type, stopping_criteria=[criterion])
            self.assertEqual(pp3.integrator.stopping_criteria, [criterion])



class TestPoincarePlotterStellsym(unittest.TestCase):
    """
    Stellarator symmetry, (R, phi, Z) -> (R, -phi, -Z): the cross section at
    phi is the mirror image in Z of the one at 2*pi*k/nfp - phi.
    """

    def test_mirror_planes(self):
        """
        With stellsym, the mirror planes 2*pi*k/nfp - phi are added to the
        field period planes phi + 2*pi*k/nfp. Planes that coincide up to
        round-off, also across 2*pi, are merged: the planes 0 and pi/nfp are
        their own mirror image and are not duplicated.
        """
        nfp = 3
        planes = PoincarePlotter.generate_symmetry_planes([0.1], nfp=nfp, stellsym=True)
        expected = np.sort(np.concatenate([0.1 + 2*np.pi*np.arange(nfp)/nfp, 2*np.pi*np.arange(1, nfp + 1)/nfp - 0.1]))
        np.testing.assert_allclose(planes, expected)
        planes = PoincarePlotter.generate_symmetry_planes([0.0, np.pi/nfp], nfp=nfp, stellsym=True)
        np.testing.assert_allclose(planes, np.pi*np.arange(2*nfp)/nfp)
        # a plane that already is the mirror of another one, up to round-off, is not duplicated
        planes = PoincarePlotter.generate_symmetry_planes([0.1, 2*np.pi/nfp - 0.1 + 1e-13], nfp=nfp, stellsym=True)
        self.assertEqual(len(planes), 2*nfp)
        # nor is an angle just below 2*pi, which is the plane 0
        for stellsym in [False, True]:
            planes = PoincarePlotter.generate_symmetry_planes([np.nextafter(2*np.pi/nfp, 0)], nfp=nfp, stellsym=stellsym)
            np.testing.assert_allclose(planes, 2*np.pi*np.arange(nfp)/nfp, atol=1e-12)

    def test_flip_z(self):
        """
        A field line of a purely toroidal field at Z = 0.05 crosses every plane
        at Z = 0.05. With stellsym, the plot of a cross section also shows the
        crossings of its mirror planes, at Z = -0.05. The planes 0 and pi/nfp
        are their own mirror image, so they are drawn twice, once flipped.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            self.skipTest('matplotlib not available')
        intg = SimsoptFieldlineIntegrator(ToroidalField(1.2, 0.8), tmax=50.0, tol=1e-9)
        nfp = 2
        for stellsym in [False, True]:
            pp = PoincarePlotter(intg, [[1.25, 0.05]], phis=[0.0, 0.4], n_transits=1, nfp=nfp, stellsym=stellsym)
            for phi in [0.0, 0.4]:
                with self.subTest(stellsym=stellsym, phi=phi):
                    fig, ax = pp.plot_poincare_single(phi, prevent_recompute=True)
                    Z = np.concatenate([collection.get_offsets()[:, 1] for collection in ax.collections])
                    # one crossing per field period plane, and as many from the mirror planes
                    self.assertEqual(np.sum(np.isclose(Z, 0.05)), nfp)
                    self.assertEqual(np.sum(np.isclose(Z, -0.05)), nfp if stellsym else 0)
                    plt.close(fig)


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
        cls.intg = SimsoptFieldlineIntegrator(cls.field, tmax=50.0, tol=1e-7)
        cls.pp = PoincarePlotter(cls.intg, cls.start_points_RZ, phis=cls.n_planes, n_transits=cls.n_transits, add_symmetry_planes=True, nfp=nfp)

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
        cls.intg = SimsoptFieldlineIntegrator(cls.field, tmax=50.0, tol=1e-9)
        cls.pp = PoincarePlotter(cls.intg, cls.start_points_RZ, phis=3, n_transits=1, add_symmetry_planes=False)

    def test_matplotlib_3d(self):
        try:
            import matplotlib  # noqa: F401
            matplotlib.use('Agg')
        except Exception:
            self.skipTest('matplotlib not available')
        # Should not raise, also when marking lost field lines
        self.pp.plot_fieldline_trajectories_3d(engine='matplotlib', show=False)
        self.pp.plot_poincare_in_3d(engine='matplotlib', show=False)
        self.pp._lost = [True] + [False]*(len(self.pp.start_points_RZ) - 1)
        self.pp.plot_fieldline_trajectories_3d(engine='matplotlib', show=False, mark_lost=True)
        self.pp.plot_poincare_in_3d(engine='matplotlib', show=False, mark_lost=True)
        self.pp._lost = None

    def test_unknown_engine(self):
        with self.assertRaises(ValueError):
            self.pp.plot_fieldline_trajectories_3d(engine='unknown', show=False)
        with self.assertRaises(ValueError):
            self.pp.plot_poincare_in_3d(engine='unknown', show=False)

    def test_plotly_3d(self):
        try:
            import plotly  # noqa: F401
        except Exception:
            self.skipTest('plotly not installed')
        self.pp.plot_fieldline_trajectories_3d(engine='plotly', show=False)
        self.pp.plot_poincare_in_3d(engine='plotly', show=False)

    def test_mayavi_3d(self):
        try:
            from mayavi import mlab  # noqa: F401
        except Exception:
            self.skipTest('mayavi not installed')
        # Use show=False for trajectories; poincare skip show
        self.pp.plot_fieldline_trajectories_3d(engine='mayavi', show=False)
        self.pp.plot_poincare_in_3d(engine='mayavi', show=False)


    def test_color_and_show(self):
        """
        A user-supplied color replaces the random colors, and show=True shows
        the figure, for the matplotlib and plotly engines.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
        except Exception:
            self.skipTest('matplotlib not available')
        with mock.patch('matplotlib.pyplot.show') as show:
            self.pp.plot_fieldline_trajectories_3d(engine='matplotlib', color='blue', show=True)
            self.pp.plot_poincare_in_3d(engine='matplotlib', color='blue', show=True)
            self.assertEqual(show.call_count, 2)
        try:
            import plotly.graph_objects as go
        except Exception:
            self.skipTest('plotly not installed')
        with mock.patch.object(go.Figure, 'show') as show:
            self.pp.plot_fieldline_trajectories_3d(engine='plotly', color='blue', show=True)
            self.pp.plot_poincare_in_3d(engine='plotly', color='blue', show=True)
            self.assertEqual(show.call_count, 2)

    def test_mayavi_3d_mocked(self):
        """
        The mayavi engine, with mayavi replaced by a mock so that the test
        runs without it: one tube per field line and one set of points per
        field line and plane, in random or given colors, with lost field
        lines marked in red.
        """
        mayavi = mock.MagicMock()
        with mock.patch.dict(sys.modules, {'mayavi': mayavi, 'mayavi.mlab': mayavi.mlab}):
            mlab = mayavi.mlab
            self.pp.plot_fieldline_trajectories_3d(engine='mayavi', show=True)
            self.assertEqual(mlab.plot3d.call_count, len(self.start_points_RZ))
            self.pp.plot_poincare_in_3d(engine='mayavi', show=True)
            self.assertEqual(mlab.points3d.call_count, len(self.start_points_RZ)*len(self.pp.phis))
            self.assertEqual(mlab.show.call_count, 2)
            self.pp._lost = [True]
            self.pp.plot_fieldline_trajectories_3d(engine='mayavi', color=(0, 0, 1), mark_lost=True, show=False)
            self.pp.plot_poincare_in_3d(engine='mayavi', color=(0, 0, 1), mark_lost=True, show=False)
            self.pp._lost = None
            self.assertEqual(mlab.plot3d.call_args.kwargs['color'], (1, 0, 0))
            self.assertEqual(mlab.points3d.call_args.kwargs['color'], (1, 0, 0))
            self.assertEqual(mlab.show.call_count, 2)


class TestPoincarePlotterRanks(unittest.TestCase):
    """
    Behaviour that does not depend on the field: MPI ranks other than 0,
    invalid plane indices, and results provided without trajectories.
    """

    def setUp(self):
        self.field = ToroidalField(1.2, 0.8)
        self.start_points_RZ = np.array([[1.25, 0.0]])
        self.intg = SimsoptFieldlineIntegrator(self.field, tmax=50.0, tol=1e-9)

    def test_non_plotting_rank(self):
        """
        Only rank 0 plots and writes cache files. On other ranks the plot
        methods return (None, None), and saving or clearing the cache does
        nothing.
        """
        with ScratchDir('.'):
            pp = PoincarePlotter(self.intg, self.start_points_RZ, phis=2, n_transits=1,
                                 add_symmetry_planes=False, cache_file='cache.npz')
            _ = pp.res_phi_hits
            self.assertTrue(os.path.exists('cache.npz'))
            pp.is_plotter = False
            self.assertEqual(pp.plot_poincare_plane_idx(0), (None, None))
            self.assertEqual(pp.plot_poincare_single(pp.phis[0]), (None, None))
            self.assertEqual(pp.plot_poincare_all(), (None, None))
            pp.save_cache(filename='other.npz')
            self.assertFalse(os.path.exists('other.npz'))
            pp.clear_cache()
            self.assertTrue(os.path.exists('cache.npz'))

    def test_plane_index_out_of_range(self):
        """Asking for the hits on a plane that does not exist raises."""
        pp = PoincarePlotter(self.intg, self.start_points_RZ, phis=2, n_transits=1, add_symmetry_planes=False)
        with self.assertRaises(ValueError):
            pp.plane_hits_cart(2)

    def test_trajectories_computed_when_missing(self):
        """
        A plotter made from plane hits only computes the trajectories when
        they are asked for.
        """
        _, res_phi_hits = self.intg.compute_poincare_hits(self.start_points_RZ, 1, phis=[0.0])
        pp = PoincarePlotter.from_poincare_data(self.intg, self.start_points_RZ, res_phi_hits,
                                                phis=[0.0], n_transits=1, add_symmetry_planes=False)
        self.assertIsNone(pp._res_tys)
        res_tys = pp.res_tys
        self.assertEqual(len(res_tys), 1)
        np.testing.assert_allclose(np.linalg.norm(res_tys[0][:, 1:3], axis=1), 1.25)

    def test_fix_axes_title(self):
        """fix_axes labels the axes and sets the title when one is given."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            self.skipTest('matplotlib not available')
        fig, ax = plt.subplots()
        PoincarePlotter.fix_axes(ax, title='phi = 0')
        self.assertEqual(ax.get_title(), 'phi = 0')
        self.assertEqual(ax.get_xlabel(), 'R')
        plt.close(fig)


class TestPoincarePlotterSaveLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_curves, base_currents, ma, nfp, bs = get_data('ncsx', coil_order=4, magnetic_axis_order=4, points_per_period=3)
        cls.bs = bs
        cls.nfp = nfp
        cls.R0 = float(ma.gamma()[0, 0])
        cls.start_points_RZ = np.array([
            [cls.R0 + 0.01, 0.0],
            [cls.R0 + 0.02, 0.0],
        ])
        cls.intg_sopp = SimsoptFieldlineIntegrator(cls.bs, tmax=40.0, tol=1e-7)
        cls.intg_scipy = ScipyFieldlineIntegrator(cls.bs, integrator_type='RK45')

    def test_save_and_load_with_dof_change(self):
        """
        test that the cache key, saving and loading work as intended.
        """
        with ScratchDir('.'):
            archive = 'poincare_data.npz'
            kwargs = dict(phis=4, n_transits=1, add_symmetry_planes=True, cache_file=archive, nfp=self.nfp)
            pp = PoincarePlotter(self.intg_sopp, self.start_points_RZ, **kwargs)
            _ = pp.res_phi_hits  # compute and save to disk
            self.assertTrue(os.path.exists(archive))
            with np.load(archive, allow_pickle=True) as data:
                self.assertIn(f'res_phi_{pp.cache_key}', data.files)
                self.assertIn(f'res_tys_{pp.cache_key}', data.files)

            # A new instance with the same settings loads the data on construction
            pp2 = PoincarePlotter(self.intg_sopp, self.start_points_RZ, **kwargs)
            self.assertFalse(pp2.need_to_recompute)
            for hits1, hits2 in zip(pp.res_phi_hits, pp2._res_phi_hits):
                np.testing.assert_array_equal(hits1, hits2)
            for ty1, ty2 in zip(pp.res_tys, pp2._res_tys):
                np.testing.assert_array_equal(ty1, ty2)

            # modify the results, to check that this modification is what is read back
            pp2._res_phi_hits[0][0, 0] = 1e5
            pp2._res_tys[0][0, 0] = 1e5
            pp2.save_cache()

            # Change a dof to invalidate the results of both plotters
            old_val = self.bs.coils[0].current.x.copy()
            self.bs.coils[0].current.x = old_val * 1.02
            self.assertIsNone(pp._res_tys)
            self.assertIsNone(pp2._res_tys)

            # restore the dof and check that the modified results are read from disk
            self.bs.coils[0].current.x = old_val
            self.assertEqual(pp2.res_phi_hits[0][0, 0], 1e5)
            self.assertEqual(pp2.res_tys[0][0, 0], 1e5)

            # a different integrator has a different key, and computes
            pp3 = PoincarePlotter(self.intg_scipy, self.start_points_RZ, **kwargs)
            self.assertNotEqual(pp3.cache_key, pp.cache_key)
            self.assertTrue(pp3.need_to_recompute)
            _ = pp3.res_tys
            pp3.save_cache(filename='othername_no_suffix')
            self.assertTrue(os.path.exists('othername_no_suffix.npz'))
            # results can be loaded into another plotter under an explicit key
            self.assertTrue(pp.load_cache(filename='othername_no_suffix', key=pp3.cache_key))
            self.assertFalse(pp.load_cache(filename='othername_no_suffix'))

            pp2.clear_cache()
            self.assertFalse(os.path.exists(archive))

    def test_cache_key(self):
        pp = PoincarePlotter(self.intg_sopp, self.start_points_RZ, phis=4, n_transits=1, nfp=self.nfp)
        key = pp.cache_key
        self.assertEqual(key, PoincarePlotter(self.intg_sopp, self.start_points_RZ, phis=4, n_transits=1, nfp=self.nfp).cache_key)
        self.assertNotEqual(key, PoincarePlotter(self.intg_sopp, self.start_points_RZ, phis=4, n_transits=1, nfp=self.nfp,
                                                 phi0=0.1).cache_key)
        self.assertNotEqual(key, PoincarePlotter(SimsoptFieldlineIntegrator(self.bs, tmax=40.0, tol=1e-9), self.start_points_RZ,
                                                 phis=4, n_transits=1, nfp=self.nfp).cache_key)
        self.assertNotEqual(key, PoincarePlotter(self.intg_sopp, self.start_points_RZ, phis=4, n_transits=2, nfp=self.nfp).cache_key)
        with self.assertRaises(ValueError):
            pp.save_cache()  # no cache_file and no filename

    def test_from_poincare_data(self):
        res_tys, res_phi_hits = self.intg_scipy.compute_poincare_hits(self.start_points_RZ, 1, phis=[0.0])
        pp = PoincarePlotter.from_poincare_data(self.intg_scipy, self.start_points_RZ, res_phi_hits, res_tys,
                                                phis=[0.0], n_transits=1, add_symmetry_planes=False)
        self.assertFalse(pp.need_to_recompute)
        self.assertEqual(pp.res_phi_hits[0].shape[1], 5)
        np.testing.assert_array_equal(pp.res_phi_hits[1], res_phi_hits[1])
        with self.assertRaises(ValueError):
            PoincarePlotter.from_poincare_data(self.intg_scipy, self.start_points_RZ, res_phi_hits[:1])

    def test_save_to_vtk(self):
        """
        test that the hashing, saving and loading works as intended. 
        """
        with ScratchDir('.'):
            pp = PoincarePlotter(self.intg_sopp, self.start_points_RZ, phis=4, n_transits=2, add_symmetry_planes=True, nfp=self.nfp)
            filename = "test"
            pp.particles_to_vtk(filename)
            self.assertTrue(os.path.exists(f"{filename}.vtu"))
    


if __name__ == '__main__':
    unittest.main()
