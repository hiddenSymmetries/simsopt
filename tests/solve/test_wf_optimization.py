import unittest
import numpy as np
from pathlib import Path
from simsopt.geo import SurfaceRZFourier, CurveXYZFourier, ToroidalWireframe
from simsopt.field import WireframeField, enclosed_current, ToroidalField
from simsopt.solve import optimize_wireframe, bnorm_obj_matrices, \
    get_gsco_iteration
from simsopt.objectives import SquaredFlux

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()


class WireframeOptimizationTests(unittest.TestCase):

    def test_toroidal_wireframe_bnorm_obj_matrices(self):
        """
        Tests the correctness of the bnormal and objective matrix calculations
        for ToroidalWireframe class instances
        """

        # Use the rotating ellipse as the plasma boundary
        plas_fname = TEST_DIR / 'input.rotating_ellipse'
        surf_plas = SurfaceRZFourier.from_vmec_input(plas_fname)

        # Set up the wireframe
        n_phi = 4
        n_theta = 8
        surf_wf = SurfaceRZFourier.from_vmec_input(plas_fname)
        surf_wf.extend_via_normal(1.0)
        wf = ToroidalWireframe(surf_wf, n_phi, n_theta)

        # Incorporate currents to create a toroidal field
        cur_pol = 1e6
        n_tf = 2
        curr_per_tf = cur_pol/(2*wf.nfp*n_tf)
        wf.add_tfcoil_currents(n_tf, curr_per_tf)
        mf_wf = WireframeField(wf)

        # Verify that matrices produce same bnormal as the SquaredFlux metric
        Amat, cvec = bnorm_obj_matrices(wf, surf_plas, verbose=False)
        sq_flux_mat = \
            0.5 * np.sum((Amat @ wf.currents.reshape((-1, 1)) - cvec)**2)
        sq_flux_ref = SquaredFlux(surf_plas, mf_wf).J()
        self.assertAlmostEqual(sq_flux_mat, sq_flux_ref)

        # Redo comparison with an external field added in
        mf_tor = ToroidalField(1.0, 2e-7*cur_pol)
        Amat2, cvec2 = bnorm_obj_matrices(wf, surf_plas, ext_field=mf_tor,
                                          verbose=False)
        self.assertTrue(np.allclose(Amat, Amat2))
        self.assertFalse(np.allclose(cvec, cvec2))
        sq_flux_mat2 = \
            0.5 * np.sum((Amat2 @ wf.currents.reshape((-1, 1)) - cvec2)**2)
        sq_flux_ref2 = SquaredFlux(surf_plas, mf_wf+mf_tor).J()
        self.assertAlmostEqual(sq_flux_mat2, sq_flux_ref2)

    def test_toroidal_wireframe_rcls(self):
        """
        Tests the Regularized Constrained Least Squares (RCLS) optimizer for
        ToroidalWireframe class instances
        """

        # Use the rotating ellipse as the plasma boundary
        plas_fname = TEST_DIR / 'input.rotating_ellipse'
        surf_plas = SurfaceRZFourier.from_vmec_input(plas_fname)

        # Set up the wireframe
        n_phi = 4
        n_theta = 8
        surf_wf = SurfaceRZFourier.from_vmec_input(plas_fname)
        surf_wf.extend_via_normal(1.0)
        wf = ToroidalWireframe(surf_wf, n_phi, n_theta)

        # Define Amperian loops for checking current constraints
        n_pts_amploop = 200  # number of quadrature points in the loop
        amploop_pol = CurveXYZFourier(n_pts_amploop, 1)
        amploop_pol.set('xc(1)', surf_wf.get_rc(0, 0))
        amploop_pol.set('ys(1)', surf_wf.get_rc(0, 0))
        amploop_tor = CurveXYZFourier(n_pts_amploop, 1)
        amploop_tor.set('xc(0)', surf_wf.get_rc(0, 0))
        amploop_tor.set('xc(1)', 2*surf_wf.get_rc(1, 0))
        amploop_tor.set('zs(1)', 2*surf_wf.get_zs(1, 0))

        # Trivial optimization: no constraint requiring non-zero current
        reg_W = 1e-10
        opt_params = {'reg_W': reg_W}
        res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas,
                                 verbose=False)

        self.assertTrue(np.allclose(wf.currents, res['x'].reshape((-1))))
        self.assertTrue(np.allclose(wf.currents, 0))

        # Case with a poloidal current constraint
        cur_pol = 1e6
        wf.set_poloidal_current(cur_pol)

        res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas,
                                 verbose=False)

        self.assertTrue(np.allclose(wf.currents, res['x'].reshape((-1))))
        self.assertTrue(wf.check_constraints())
        self.assertTrue(np.isclose(cur_pol,
                                   -enclosed_current(amploop_pol, res['wframe_field'], n_pts_amploop)))

        # Case with a poloidal and a toroidal current constraint
        cur_tor = 1e6
        wf.set_toroidal_current(cur_tor)

        res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas,
                                 verbose=False)

        self.assertTrue(wf.check_constraints())
        self.assertTrue(np.isclose(cur_pol,
                                   -enclosed_current(amploop_pol, res['wframe_field'], n_pts_amploop)))
        self.assertTrue(np.isclose(cur_tor,
                                   -enclosed_current(amploop_tor, res['wframe_field'], n_pts_amploop)))

        # Constrain some segments to have zero current
        constr_segs = [9, 17, 44]
        zero_segs = [9, 17, 44, 45]  # no. 45 should be implicitly constrained
        wf.set_segments_constrained(constr_segs)

        res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas,
                                 verbose=True)

        self.assertTrue(np.allclose(wf.currents[zero_segs], 0))
        self.assertTrue(np.isclose(cur_pol,
                                   -enclosed_current(amploop_pol, res['wframe_field'], n_pts_amploop)))
        self.assertTrue(np.isclose(cur_tor,
                                   -enclosed_current(amploop_tor, res['wframe_field'], n_pts_amploop)))
        self.assertTrue(wf.check_constraints())

        wf.free_all_segments()

        # Field error should decrease as wireframe resolution increases
        n_phi_arr = [4, 6, 8, 10]
        n_theta_arr = [8, 10, 12, 14]
        bnormal_prev = 0
        for i in range(len(n_phi_arr)):

            wf = ToroidalWireframe(surf_wf, n_phi_arr[i], n_theta_arr[i])
            wf.set_poloidal_current(cur_pol)
            res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas,
                                     verbose=False)
            bnormal = np.sum((res['Amat'] @ res['x'])**2)

            if i > 0:
                self.assertTrue(bnormal < 0.25*bnormal_prev)

            bnormal_prev = bnormal

        # RCLS optimizations in the presence of an external field
        wf = ToroidalWireframe(surf_wf, n_phi, n_theta)
        wf.set_toroidal_current(0)
        wf.set_poloidal_current(0)
        mf_tor = ToroidalField(1.0, -2e-7*cur_pol)
        res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas,
                                 ext_field=mf_tor, verbose=False)
        self.assertFalse(np.allclose(0, wf.currents))
        self.assertTrue(wf.check_constraints())
        self.assertTrue(np.isclose(0,
                                   enclosed_current(amploop_pol, res['wframe_field'], n_pts_amploop),
                                   atol=cur_pol*1e-6))
        self.assertTrue(np.isclose(cur_pol,
                                   -enclosed_current(amploop_pol, res['wframe_field'] + mf_tor,
                                                     n_pts_amploop)))
        self.assertTrue(SquaredFlux(surf_plas, res['wframe_field'] + mf_tor).J()
                        < 0.01*SquaredFlux(surf_plas, res['wframe_field']).J())

        # Check consistency of stored objective function values
        self.assertTrue(np.isclose(
            SquaredFlux(surf_plas, res['wframe_field'] + mf_tor).J(),
            res['f_B']))
        self.assertTrue(np.isclose(res['f'], res['f_B'] + res['f_R']))
        self.assertTrue(np.isclose(res['f_R'],
                                   0.5 * opt_params['reg_W']**2 * np.sum(res['x']**2)))

        # Verify that same solution is obtained when the bnormal and objective
        # matrices are supplied by the user
        res2 = optimize_wireframe(wf, 'rcls', opt_params, Amat=res['Amat'],
                                  bvec=res['bvec'], verbose=True)
        self.assertTrue(np.allclose(res2['x'], res['x']))

        # Tests with non-scalar regularization parameter
        opt_params_vectorW = {'reg_W': reg_W * np.ones((2*n_phi*n_theta))}
        res3 = optimize_wireframe(wf, 'rcls', opt_params_vectorW,
                                  Amat=res['Amat'], bvec=res['bvec'], verbose=False)
        self.assertTrue(np.allclose(res3['x'], res['x']))

        opt_params_matrixW = {'reg_W': reg_W * np.eye((2*n_phi*n_theta))}
        res4 = optimize_wireframe(wf, 'rcls', opt_params_matrixW,
                                  Amat=res['Amat'], bvec=res['bvec'], verbose=False)
        self.assertTrue(np.allclose(res4['x'], res['x']))

        opt_params_errorVecW = {'reg_W': reg_W * np.ones((2*n_phi*n_theta+1))}
        opt_params_errorMatW = {'reg_W': reg_W * np.eye((2*n_phi*n_theta+1))}
        with self.assertRaises(ValueError):
            optimize_wireframe(wf, 'rcls', opt_params_errorVecW,
                               Amat=res['Amat'], bvec=res['bvec'], verbose=False)
        with self.assertRaises(ValueError):
            optimize_wireframe(wf, 'rcls', opt_params_errorMatW,
                               Amat=res['Amat'], bvec=res['bvec'], verbose=False)

    def test_toroidal_wireframe_gsco(self):
        """
        Tests the Greedy Stellarator Coil Optimization algorithm for 
        ToroidalWireframe class instances
        """

        # Use the rotating ellipse as the plasma boundary
        plas_fname = TEST_DIR / 'input.rotating_ellipse'
        surf_plas = SurfaceRZFourier.from_vmec_input(plas_fname)

        # Set up the wireframe
        n_phi = 4
        n_theta = 8
        surf_wf = SurfaceRZFourier.from_vmec_input(plas_fname)
        surf_wf.extend_via_normal(1.0)
        wf = ToroidalWireframe(surf_wf, n_phi, n_theta)
        wf.set_poloidal_current(0)
        wf.set_toroidal_current(0)

        # External toroidal field
        cur_pol = 1e6
        mf_tor = ToroidalField(1.0, -2e-7*cur_pol)

        std_params = {'lambda_S': 1e-10,
                      'default_current': 0.02*cur_pol,
                      'max_current': 0.1*cur_pol,
                      'max_iter': 120,
                      'print_interval': 20}

        # Verify that suitable errors are raised for faulty input
        with self.assertRaises(ValueError):
            optimize_wireframe(wf, 'gsco', std_params, verbose=False)
        with self.assertRaises(ValueError):
            optimize_wireframe(wf, 'gsco', dict(), surf_plas=surf_plas,
                               verbose=False)
        params_no_lambda = dict(std_params)
        del (params_no_lambda['lambda_S'])
        with self.assertRaises(ValueError):
            optimize_wireframe(wf, 'gsco', params_no_lambda,
                               surf_plas=surf_plas, verbose=False)

        # Basic optimization in an external toroidal field
        res0 = optimize_wireframe(wf, 'gsco', std_params, surf_plas=surf_plas,
                                  ext_field=mf_tor, verbose=False)

        # Consistency checks for the solution
        self.assertFalse(np.allclose(wf.currents, 0))
        self.assertTrue(np.allclose(wf.currents, res0['x'].reshape((-1))))
        self.assertTrue(np.max(np.abs(wf.currents))
                        <= std_params['max_current'])
        self.assertTrue(wf.check_constraints())
        with self.assertRaises(RuntimeError):
            # Crossed currents aren't guaranteed to occur, but should appear
            # in this particular test case unless the algorithm has changed
            C, d = wf.constraint_matrices(assume_no_crossings=True)

        # Check correctness of 'loop_count' array by using it to reconstruct
        # the current distribution of the solution
        cell_key = wf.get_cell_key()
        test_currents = np.zeros(wf.n_segments)
        curr_added = res0['loop_count'] * std_params['default_current']
        for i in range(wf.n_theta*wf.n_phi):
            # Note: cannot be (easily) vectorized because slices of cell_key
            # contain repeated indices for test_currents
            test_currents[cell_key[i, 0]] += curr_added[i]
            test_currents[cell_key[i, 1]] += curr_added[i]
            test_currents[cell_key[i, 2]] -= curr_added[i]
            test_currents[cell_key[i, 3]] -= curr_added[i]
        self.assertTrue(np.allclose(test_currents, wf.currents))

        # Verify consistency of the history data
        currents_soln = np.array(wf.currents)
        currents_0 = get_gsco_iteration(0, res0, wf)
        self.assertTrue(np.allclose(currents_0, 0))
        currents_final = get_gsco_iteration(res0['iter_hist'][-1], res0, wf)
        self.assertTrue(np.allclose(currents_final.ravel(), currents_soln))
        for i in range(len(res0['iter_hist'])):
            f_B_i = res0['f_B_hist'][i]
            f_S_i = res0['f_S_hist'][i]
            currents_i = get_gsco_iteration(i, res0, wf).ravel()
            wf.currents[:] = currents_i[:]
            mf_i = WireframeField(wf) + mf_tor
            self.assertTrue(np.isclose(SquaredFlux(surf_plas, mf_i).J(), f_B_i))
            self.assertEqual(0.5*np.sum(currents_i != 0), f_S_i)

        # Check consistency of stored objective function values
        self.assertTrue(np.isclose(
            SquaredFlux(surf_plas, res0['wframe_field'] + mf_tor).J(),
            res0['f_B']))
        self.assertTrue(np.isclose(res0['f'],
                                   res0['f_B'] + std_params['lambda_S']*res0['f_S']))
        self.assertTrue(np.isclose(res0['f_S'],
                                   0.5 * np.sum(np.abs(res0['x']) > wf.constraint_atol)))

        # Verify that no iterations take place if default current is 0
        wf.currents[:] = 0
        params_0_curr = dict(std_params)
        params_0_curr['default_current'] = 0
        res1 = optimize_wireframe(wf, 'gsco', params_0_curr,
                                  surf_plas=surf_plas, ext_field=mf_tor, verbose=False)
        self.assertEqual(len(res1['iter_hist']), 1)

        # Verify that no iterations take place if no loops can be added
        wf.currents[:] = 0
        wf.set_segments_constrained(np.arange(wf.n_segments))
        res2 = optimize_wireframe(wf, 'gsco', std_params, surf_plas=surf_plas,
                                  ext_field=mf_tor, verbose=False)
        self.assertEqual(len(res2['iter_hist']), 1)
        wf.free_all_segments()

        # Redo optimization restricting current from forming loops
        params_no_xing = dict(std_params)
        params_no_xing['no_crossing'] = True
        params_no_xing['max_iter'] = 10
        params_no_xing['print_interval'] = 5
        res3 = optimize_wireframe(wf, 'gsco', params_no_xing,
                                  surf_plas=surf_plas, ext_field=mf_tor, verbose=False)

        # No-crossing assumption shoud *not* raise an error this time
        self.assertTrue(wf.check_constraints())
        wf.set_segments_constrained(np.where(res3['x'].reshape((-1)) == 0)[0])
        self.assertTrue(wf.check_constraints())
        C, d = wf.constraint_matrices(assume_no_crossings=True)
        wf.free_all_segments()

        # Continue previous optimization & ensure it was initialized correctly
        params_no_xing_contd = dict(params_no_xing)
        params_no_xing_contd['loop_count_init'] = res3['loop_count']
        res4 = optimize_wireframe(wf, 'gsco', params_no_xing_contd,
                                  surf_plas=surf_plas, ext_field=mf_tor, verbose=False)
        self.assertTrue(np.allclose(
            get_gsco_iteration(res3['iter_hist'][-1], res3, wf),
            get_gsco_iteration(0, res4, wf)))
        self.assertTrue(np.allclose(res3['x'], get_gsco_iteration(0, res4, wf)))
        self.assertTrue(np.isclose(res3['f_B_hist'][-1], res4['f_B_hist'][0]))
        self.assertTrue(np.isclose(res3['f_S_hist'][-1], res4['f_S_hist'][0]))
        cell_key = wf.get_cell_key()
        test_currents = np.zeros(wf.n_segments)
        curr_added = res4['loop_count'] * params_no_xing['default_current']
        for i in range(wf.n_theta*wf.n_phi):
            test_currents[cell_key[i, 0]] += curr_added[i]
            test_currents[cell_key[i, 1]] += curr_added[i]
            test_currents[cell_key[i, 2]] -= curr_added[i]
            test_currents[cell_key[i, 3]] -= curr_added[i]
        self.assertTrue(np.allclose(test_currents, wf.currents))

        # Repeat the previous optimization, this time using the x_init argument
        wf.currents[:] = 0
        params_no_xing_x_init = dict(params_no_xing)
        params_no_xing_x_init['x_init'] = res3['x']
        res5 = optimize_wireframe(wf, 'gsco', params_no_xing_x_init,
                                  surf_plas=surf_plas, ext_field=mf_tor, verbose=False)
        for i in res5['iter_hist']:
            self.assertTrue(np.allclose(get_gsco_iteration(i, res4, wf),
                                        get_gsco_iteration(i, res5, wf)))

        # Higher-resolution wireframe for additional testing
        n_phi2 = 8
        n_theta2 = 16
        wf2 = ToroidalWireframe(surf_wf, n_phi2, n_theta2)

        # No-crossing optimization with a single allowable current magnitude
        params_no_xing_1_curr = dict(params_no_xing)
        seg_curr = 0.01*cur_pol
        params_no_xing_1_curr['default_current'] = seg_curr
        params_no_xing_1_curr['max_current'] = seg_curr
        params_no_xing_1_curr['max_iter'] = 100
        params_no_xing_1_curr['print_interval'] = 10
        res6 = optimize_wireframe(wf2, 'gsco', params_no_xing_1_curr,
                                  surf_plas=surf_plas, ext_field=mf_tor, verbose=False)
        self.assertTrue(wf2.check_constraints())
        self.assertAlmostEqual(np.max(np.abs(wf2.currents)), seg_curr)
        nz_inds = np.where(wf2.currents != 0)[0]
        self.assertAlmostEqual(np.max(np.abs(wf2.currents)), seg_curr)
        self.assertTrue(np.allclose(np.abs(wf2.currents[nz_inds]), seg_curr))

        # Redo previous case with restricted loop count
        wf2.currents[:] = 0
        params_no_xing_1_curr_1_loop = dict(params_no_xing_1_curr)
        params_no_xing_1_curr_1_loop['max_loop_count'] = 1
        res7 = optimize_wireframe(wf2, 'gsco', params_no_xing_1_curr_1_loop,
                                  surf_plas=surf_plas, ext_field=mf_tor, verbose=False)
        self.assertEqual(np.max(np.abs(res7['loop_count'])),
                         params_no_xing_1_curr_1_loop['max_loop_count'])

        # Consistency check: using no_new_coils=T should be
        # equivalent to match_currents=T + default_current=0
        params_no_new = dict(params_no_xing_1_curr)
        params_no_new['no_new_coils'] = True
        params_no_new['default_current'] = 0.01*cur_pol
        params_no_new['x_init'] = get_gsco_iteration(40, res6, wf2)
        res8 = optimize_wireframe(wf2, 'gsco', params_no_new,
                                  surf_plas=surf_plas, ext_field=mf_tor, verbose=False)

        params_0_curr = dict(params_no_new)
        params_0_curr['no_new_coils'] = False
        params_0_curr['match_current'] = True
        params_0_curr['default_current'] = 0
        res9 = optimize_wireframe(wf2, 'gsco', params_0_curr,
                                  surf_plas=surf_plas, ext_field=mf_tor, verbose=False)

        self.assertTrue(np.allclose(res8['x'], res9['x']))

    def test_optimize_wireframe_errors_and_bnorm_target(self):
        """
        Tests error conditions in optimize_wireframe and the bnorm_target functionality
        """
        # Set up test objects
        plas_fname = TEST_DIR / 'input.rotating_ellipse'
        surf_plas = SurfaceRZFourier.from_vmec_input(plas_fname)
        surf_wf = SurfaceRZFourier.from_vmec_input(plas_fname)
        surf_wf.extend_via_normal(1.0)
        wf = ToroidalWireframe(surf_wf, 4, 8)

        # Test 1: wframe must be a ToroidalWireframe instance
        with self.assertRaisesRegex(ValueError, 'Input `wframe` must be a ToroidalWireframe class instance'):
            optimize_wireframe(None, 'rcls', {'reg_W': 1e-10}, surf_plas=surf_plas)

        # Test 2: If surf_plas is given, Amat and bvec must not be supplied
        with self.assertRaisesRegex(ValueError, 'Inputs `Amat` and `bvec` must not be supplied if `surf_plas` is given'):
            optimize_wireframe(wf, 'rcls', {'reg_W': 1e-10},
                               surf_plas=surf_plas,
                               Amat=np.eye(10),
                               bvec=np.ones(10))

        # Test 3: If Amat and bvec provided without surf_plas, other parameters must not be provided
        dummy_matrix = np.random.rand(10, wf.n_segments)
        dummy_vector = np.random.rand(10, 1)
        with self.assertRaisesRegex(ValueError, 'If `Amat` and `bvec` are provided, the following parameters must not be provided'):
            optimize_wireframe(wf, 'rcls', {'reg_W': 1e-10},
                               Amat=dummy_matrix,
                               bvec=dummy_vector,
                               ext_field=True)  # Using ext_field to trigger this error

        # Test 4: Either surf_plas or both Amat and bvec must be supplied
        with self.assertRaisesRegex(ValueError, '`surf_plas` or `Amat` and `bvec` must be supplied'):
            optimize_wireframe(wf, 'rcls', {'reg_W': 1e-10})

        # Test 5: Amat dimensions must be consistent
        wrong_matrix = np.random.rand(5, 5)  # Wrong dimensions
        with self.assertRaisesRegex(ValueError, 'Input `Amat` has inconsistent dimensions'):
            optimize_wireframe(wf, 'rcls', {'reg_W': 1e-10},
                               Amat=wrong_matrix,
                               bvec=np.ones((5, 1)))

        # Test 6: bnorm_target must have correct dimensions
        n_quadrature_points = surf_plas.gamma().shape[0]
        wrong_bnorm_target = np.ones(n_quadrature_points + 1) * 0.1
        with self.assertRaisesRegex(ValueError, 'Input `bnorm_target` must have the same'):
            optimize_wireframe(wf, 'rcls', {'reg_W': 1e-10},
                               surf_plas=surf_plas,
                               bnorm_target=wrong_bnorm_target,
                               verbose=False)

        # Test bnorm_target functionality

        # Poloidal current constraint
        cur_pol = 1e6
        wf.set_poloidal_current(cur_pol)

        # Create a non-zero target normal field that would arise from a
        # uniform vertical external field with strength 0.1 T
        n = surf_plas.normal()
        absn = np.linalg.norm(n, axis=2)[:, :, None]
        unitn = n * (1./absn)
        bnorm_target = 0.1 * unitn[:, :, 2]

        # Run optimization with bnorm_target
        reg_W = 0.0
        opt_params = {'reg_W': reg_W}
        # Define Amperian loops for checking current constraints
        n_pts_amploop = 200  # number of quadrature points in the loop
        amploop_pol = CurveXYZFourier(n_pts_amploop, 1)
        amploop_pol.set('xc(1)', surf_wf.get_rc(0, 0))
        amploop_pol.set('ys(1)', surf_wf.get_rc(0, 0))
        amploop_tor = CurveXYZFourier(n_pts_amploop, 1)
        amploop_tor.set('xc(0)', surf_wf.get_rc(0, 0))
        amploop_tor.set('xc(1)', 2*surf_wf.get_rc(1, 0))
        amploop_tor.set('zs(1)', 2*surf_wf.get_zs(1, 0))
        res_baseline = optimize_wireframe(wf, 'rcls', opt_params,
                                          surf_plas=surf_plas,
                                          area_weighted=False, verbose=False)
        self.assertTrue(wf.check_constraints())
        self.assertTrue(np.isclose(cur_pol,
                                   -enclosed_current(amploop_pol, res_baseline['wframe_field'], n_pts_amploop)))

        # Now run with a nonzero target field
        res_with_target = optimize_wireframe(wf, 'rcls', opt_params,
                                             surf_plas=surf_plas,
                                             bnorm_target=bnorm_target,
                                             area_weighted=False, verbose=False)

        self.assertTrue(wf.check_constraints())
        self.assertTrue(np.isclose(cur_pol,
                                   -enclosed_current(amploop_pol, res_with_target['wframe_field'], n_pts_amploop)))

        # The bvec should differ between the two runs
        self.assertFalse(np.allclose(res_baseline['bvec'], res_with_target['bvec']),
                         "Target field should change the optimization target vector")

        bvec_diff = res_with_target['bvec'] - res_baseline['bvec']
        assert np.allclose(bvec_diff, bnorm_target.reshape((-1, 1)))

        # Run with a different target field
        bnorm_target2 = 3 * bnorm_target
        res_with_target2 = optimize_wireframe(wf, 'rcls', opt_params,
                                              surf_plas=surf_plas,
                                              bnorm_target=bnorm_target2,
                                              area_weighted=False, verbose=False)
        self.assertTrue(wf.check_constraints())
        self.assertTrue(np.isclose(cur_pol,
                                   -enclosed_current(amploop_pol, res_with_target2['wframe_field'], n_pts_amploop)))

        print('x', res_with_target['x'] - res_with_target2['x'])
        print('bvec', res_with_target['bvec'] - res_with_target2['bvec'])
        print('Amat @ x - bvec', res_with_target['Amat'] @ res_with_target['x'] - res_with_target['bvec'],
              res_with_target2['Amat'] @ res_with_target2['x'] - res_with_target2['bvec'])
        self.assertFalse(np.allclose(res_with_target['x'], res_with_target2['x']),
                         "Different target fields should produce different solutions")

        # The difference in bvec should match the difference in target values
        bvec_diff2 = res_with_target2['bvec'] - res_with_target['bvec']
        assert np.allclose(bvec_diff2, 2*bnorm_target.reshape((-1, 1)))


if __name__ == "__main__":
    unittest.main()
