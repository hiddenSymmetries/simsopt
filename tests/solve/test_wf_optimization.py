import unittest
import numpy as np
from pathlib import Path
from simsopt.geo import SurfaceRZFourier, CurveXYZFourier, ToroidalWireframe
from simsopt.field import WireframeField, enclosed_current, ToroidalField
from simsopt.solve import optimize_wireframe, bnorm_obj_matrices
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
        nPhi = 4
        nTheta = 8
        surf_wf = SurfaceRZFourier.from_vmec_input(plas_fname)
        surf_wf.extend_via_normal(1.0)
        wf = ToroidalWireframe(surf_wf, nPhi, nTheta)

        # Incorporate currents to create a toroidal field
        cur_pol = 1e6
        n_tf = 2
        curr_per_tf = cur_pol/(2*wf.nfp*n_tf)
        wf.add_tfcoil_currents(n_tf, curr_per_tf)
        mf_wf = WireframeField(wf)

        # Verify that matrices produce same bnormal as the SquaredFlux metric
        Amat, cvec = bnorm_obj_matrices(wf, surf_plas, verbose=False)
        SqFlux_mat = \
            0.5 * np.sum((Amat @ wf.currents.reshape((-1,1)) - cvec)**2)
        SqFlux_ref = SquaredFlux(surf_plas, mf_wf).J()
        self.assertAlmostEqual(SqFlux_mat, SqFlux_ref)

        # Redo comparison with an external field added in
        mf_tor = ToroidalField(1.0, 2e-7*cur_pol)
        Amat2, cvec2 = bnorm_obj_matrices(wf, surf_plas, ext_field=mf_tor,
                                          verbose=False)
        self.assertTrue(np.allclose(Amat, Amat2))
        self.assertFalse(np.allclose(cvec, cvec2))
        SqFlux_mat2 = \
            0.5 * np.sum((Amat2 @ wf.currents.reshape((-1,1)) - cvec2)**2)
        SqFlux_ref2 = SquaredFlux(surf_plas, mf_wf+mf_tor).J()
        self.assertAlmostEqual(SqFlux_mat2, SqFlux_ref2)

    def test_toroidal_wireframe_rcls(self):
        """
        Tests the Regularized Constrained Least Squares (RCLS) optimizer for
        ToroidalWireframe class instances
        """

        # Use the rotating ellipse as the plasma boundary
        plas_fname = TEST_DIR / 'input.rotating_ellipse'
        surf_plas = SurfaceRZFourier.from_vmec_input(plas_fname)

        # Set up the wireframe
        nPhi = 4
        nTheta = 8
        surf_wf = SurfaceRZFourier.from_vmec_input(plas_fname)
        surf_wf.extend_via_normal(1.0)
        wf = ToroidalWireframe(surf_wf, nPhi, nTheta)

        # Define Amperian loops for checking current constraints
        nPtsAmpLoop = 200 # number of quadrature points in the loop
        amploop_pol = CurveXYZFourier(nPtsAmpLoop, 1)
        amploop_pol.set('xc(1)', surf_wf.get_rc(0,0))
        amploop_pol.set('ys(1)', surf_wf.get_rc(0,0))
        amploop_tor = CurveXYZFourier(nPtsAmpLoop, 1)
        amploop_tor.set('xc(0)', surf_wf.get_rc(0,0))
        amploop_tor.set('xc(1)', 2*surf_wf.get_rc(1,0))
        amploop_tor.set('zs(1)', 2*surf_wf.get_zs(1,0))

        # Trivial optimization: no constraint requiring non-zero current
        opt_params = {'reg_W': 1e-10}
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
            -enclosed_current(amploop_pol, res['wframe_field'], nPtsAmpLoop)))

        # Case with a poloidal and a toroidal current constraint
        cur_tor = 1e6
        wf.set_toroidal_current(cur_tor)

        res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas,
                                 verbose=False)

        self.assertTrue(wf.check_constraints())
        self.assertTrue(np.isclose(cur_pol, 
            -enclosed_current(amploop_pol, res['wframe_field'], nPtsAmpLoop)))
        self.assertTrue(np.isclose(cur_tor, 
            -enclosed_current(amploop_tor, res['wframe_field'], nPtsAmpLoop)))

        # Constrain some segments to have zero current
        constr_segs = [9, 17, 44]
        zero_segs = [9, 17, 44, 45]  # no. 45 should be implicitly constrained
        wf.set_segments_constrained(constr_segs)

        res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas,
                                 verbose=False)

        self.assertTrue(np.allclose(wf.currents[zero_segs], 0))
        self.assertTrue(np.isclose(cur_pol, 
            -enclosed_current(amploop_pol, res['wframe_field'], nPtsAmpLoop)))
        self.assertTrue(np.isclose(cur_tor, 
            -enclosed_current(amploop_tor, res['wframe_field'], nPtsAmpLoop)))
        self.assertTrue(wf.check_constraints())

        wf.free_all_segments()

        # Field error should decrease as wireframe resolution increases
        nPhi_arr = [4, 6, 8, 10]
        nTheta_arr = [8, 10, 12, 14]
        bnormal_prev = 0
        for i in range(len(nPhi_arr)):

            wf = ToroidalWireframe(surf_wf, nPhi_arr[i], nTheta_arr[i])
            wf.set_poloidal_current(cur_pol)
            res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas,
                                     verbose=False)
            bnormal = np.sum((res['Amat'] @ res['x'])**2)

            if i > 0:
                self.assertTrue(bnormal < 0.25*bnormal_prev)

            bnormal_prev = bnormal

        # RCLS optimizations in the presence of an external field
        wf = ToroidalWireframe(surf_wf, nPhi, nTheta)
        wf.set_toroidal_current(0)
        wf.set_poloidal_current(0)
        mf_tor = ToroidalField(1.0, -2e-7*cur_pol)
        res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas, 
                                 ext_field=mf_tor, verbose=False)
        self.assertFalse(np.allclose(0, wf.currents))
        self.assertTrue(wf.check_constraints())
        self.assertTrue(np.isclose(0, 
            enclosed_current(amploop_pol, res['wframe_field'], nPtsAmpLoop), 
            atol=cur_pol*1e-6))
        self.assertTrue(np.isclose(cur_pol, 
            -enclosed_current(amploop_pol, res['wframe_field'] + mf_tor, 
                              nPtsAmpLoop)))
        self.assertTrue(SquaredFlux(surf_plas, res['wframe_field'] + mf_tor).J()
                        < 0.01*SquaredFlux(surf_plas, res['wframe_field']).J())

    def test_toroidal_wireframe_gsco(self):
        """
        Tests the Greedy Stellarator Coil Optimization algorithm for 
        ToroidalWireframe class instances
        """

        # Use the rotating ellipse as the plasma boundary
        plas_fname = TEST_DIR / 'input.rotating_ellipse'
        surf_plas = SurfaceRZFourier.from_vmec_input(plas_fname)

        # Set up the wireframe
        nPhi = 4
        nTheta = 8
        surf_wf = SurfaceRZFourier.from_vmec_input(plas_fname)
        surf_wf.extend_via_normal(1.0)
        wf = ToroidalWireframe(surf_wf, nPhi, nTheta)
        wf.set_poloidal_current(0)
        wf.set_toroidal_current(0)

        # External toroidal field
        cur_pol = 1e6
        mf_tor = ToroidalField(1.0, -2e-7*cur_pol)

        std_params = {'lambda_S': 1e-10, 
                      'default_current': 0.02*cur_pol,        
                      'max_current': 0.1*cur_pol, 
                      'nIter': 120, 
                      'nHistory': 6}

        # Verify that suitable errors are raised for faulty input
        with self.assertRaises(ValueError):
            optimize_wireframe(wf, 'gsco', std_params, verbose=False)
        with self.assertRaises(ValueError):
            optimize_wireframe(wf, 'gsco', dict(), surf_plas=surf_plas,
                               verbose=False)
        params_no_lambda = dict(std_params)
        del(params_no_lambda['lambda_S'])
        with self.assertRaises(ValueError):
            optimize_wireframe(wf, 'gsco', params_no_lambda, \
                               surf_plas=surf_plas, verbose=False)
        

        # Basic optimization in an external toroidal field
        res0 = optimize_wireframe(wf, 'gsco', std_params, surf_plas=surf_plas,
                                  ext_field=mf_tor, verbose=False)

        # Consistency checks for the solution
        self.assertFalse(np.allclose(wf.currents, 0))
        self.assertTrue(np.allclose(wf.currents, res0['x'].reshape((-1))))
        self.assertTrue(np.max(np.abs(wf.currents)) \
                        <= std_params['max_current'])
        self.assertTrue(wf.check_constraints())
        with self.assertRaises(RuntimeError):
            # Crossed currents aren't guaranteed to occur, but should appear
            # in this particular test case unless the algorithm has changed
            C, d = wf.constraint_matrices(assume_no_crossings=True)

        # Check correctness of 'loop_count' array by using it to reconstruct
        # the current distribution of the solution
        cellKey = wf.get_cell_key()
        testCurrents = np.zeros(wf.nSegments)
        curr_added = res0['loop_count'] * std_params['default_current']
        for i in range(wf.nTheta*wf.nPhi):
            # Note: cannot be (easily) vectorized because slices of cellKey 
            # contain repeated indices for testCurrents
            testCurrents[cellKey[i,0]] += curr_added[i]
            testCurrents[cellKey[i,1]] += curr_added[i]
            testCurrents[cellKey[i,2]] -= curr_added[i]
            testCurrents[cellKey[i,3]] -= curr_added[i]
        self.assertTrue(np.allclose(testCurrents, wf.currents))

        # Verify consistency of the history data
        currents_soln = np.array(wf.currents)
        currents_0 = res0['x_hist'][:,0].reshape((-1))
        self.assertTrue(np.allclose(currents_0, 0))
        currents_final = res0['x_hist'][:,-1].reshape((-1))
        self.assertTrue(np.allclose(currents_final, currents_soln))
        for i in range(len(res0['iter_hist'])):
            f_B_i = res0['f_B_hist'][i]
            f_S_i = res0['f_S_hist'][i]
            currents_i = res0['x_hist'][:,i].reshape((-1))
            wf.currents[:] = currents_i[:]
            mf_i = WireframeField(wf) + mf_tor
            self.assertTrue(np.isclose(SquaredFlux(surf_plas, mf_i).J(), f_B_i))
            self.assertEqual(0.5*np.sum(currents_i != 0), f_S_i)

        # Verify that no iterations take place if default current is 0
        wf.currents[:] = 0
        params_0_curr = dict(std_params)
        params_0_curr['default_current'] = 0
        res1 = optimize_wireframe(wf, 'gsco', params_0_curr, 
                   surf_plas=surf_plas, ext_field=mf_tor, verbose=False)
        self.assertEqual(len(res1['iter_hist']), 1)
        self.assertEqual(res1['x_hist'].shape[1], 1)

        # Verify that no iterations take place if no loops can be added
        wf.currents[:] = 0
        wf.set_segments_constrained(np.arange(wf.nSegments))
        res2 = optimize_wireframe(wf, 'gsco', std_params, surf_plas=surf_plas,
                                  ext_field=mf_tor, verbose=False)
        self.assertEqual(len(res2['iter_hist']), 1)
        self.assertEqual(res2['x_hist'].shape[1], 1)
        wf.free_all_segments()
            
        # Redo optimization restricting current from forming loops
        params_no_xing = dict(std_params)
        params_no_xing['no_crossing'] = True
        params_no_xing['nIter'] = 10
        params_no_xing['nHistory'] = 2
        res3 = optimize_wireframe(wf, 'gsco', params_no_xing, 
            surf_plas=surf_plas, ext_field=mf_tor, verbose=False)

        # No-crossing assumption shoud *not* raise an error this time
        self.assertTrue(wf.check_constraints())
        wf.set_segments_constrained(np.where(res3['x'].reshape((-1))==0)[0])
        self.assertTrue(wf.check_constraints())
        C, d = wf.constraint_matrices(assume_no_crossings=True)
        wf.free_all_segments()

        # Continue previous optimization & ensure it was initialized correctly
        params_no_xing_contd = dict(params_no_xing)
        params_no_xing_contd['loop_count_init'] = res3['loop_count']
        res4 = optimize_wireframe(wf, 'gsco', params_no_xing_contd, 
            surf_plas=surf_plas, ext_field=mf_tor, verbose=False)
        self.assertTrue(np.allclose(res3['x_hist'][:,-1], res4['x_hist'][:,0]))
        self.assertTrue(np.allclose(res3['x'][:,0], res4['x_hist'][:,0]))
        self.assertTrue(np.isclose(res3['f_B_hist'][-1], res4['f_B_hist'][0]))
        self.assertTrue(np.isclose(res3['f_S_hist'][-1], res4['f_S_hist'][0]))
        cellKey = wf.get_cell_key()
        testCurrents = np.zeros(wf.nSegments)
        curr_added = res4['loop_count'] * params_no_xing['default_current']
        for i in range(wf.nTheta*wf.nPhi):
            testCurrents[cellKey[i,0]] += curr_added[i]
            testCurrents[cellKey[i,1]] += curr_added[i]
            testCurrents[cellKey[i,2]] -= curr_added[i]
            testCurrents[cellKey[i,3]] -= curr_added[i]
        self.assertTrue(np.allclose(testCurrents, wf.currents))

        # Repeat the previous optimization, this time using the x_init argument
        wf.currents[:] = 0
        params_no_xing_x_init = dict(params_no_xing)
        params_no_xing_x_init['x_init'] = res3['x']
        res5 = optimize_wireframe(wf, 'gsco', params_no_xing_x_init, 
            surf_plas=surf_plas, ext_field=mf_tor, verbose=False)
        self.assertTrue(np.allclose(res4['x_hist'], res5['x_hist']))

        # Higher-resolution wireframe for additional testing
        nPhi2 = 8
        nTheta2 = 16
        wf2 = ToroidalWireframe(surf_wf, nPhi2, nTheta2)

        # No-crossing optimization with a single allowable current magnitude
        params_no_xing_1_curr = dict(params_no_xing)
        seg_curr = 0.01*cur_pol
        params_no_xing_1_curr['default_current'] = seg_curr
        params_no_xing_1_curr['max_current'] = seg_curr
        params_no_xing_1_curr['nIter'] = 100
        params_no_xing_1_curr['nHistory'] = 10
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
        params_no_new['x_init'] = res6['x_hist'][:,5]
        res8 = optimize_wireframe(wf2, 'gsco', params_no_new, 
            surf_plas=surf_plas, ext_field=mf_tor, verbose=False)

        params_0_curr = dict(params_no_new)
        params_0_curr['no_new_coils'] = False
        params_0_curr['match_current'] = True
        params_0_curr['default_current'] = 0
        res9 = optimize_wireframe(wf2, 'gsco', params_0_curr, 
            surf_plas=surf_plas, ext_field=mf_tor, verbose=False)

        self.assertTrue(np.allclose(res8['x'], res9['x']))

if __name__ == "__main__":
    unittest.main() 

