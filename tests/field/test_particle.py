import simsoptpp as sopp
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from simsopt.field.boozermagneticfield import BoozerAnalytic
from simsopt.field.tracing import \
    IterationStoppingCriterion, trace_particles_boozer, \
    MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion, ToroidalTransitStoppingCriterion, \
    compute_poloidal_transits, compute_toroidal_transits, compute_resonances
import numpy as np
import unittest
import logging

logging.basicConfig()


class BoozerGuidingCenterTracingTesting(unittest.TestCase):

    def test_energy_momentum_conservation_boozer(self):
        """
        Trace 100 particles in a BoozerAnalytic field for 10 toroidal
        transits or 1e-3s. Check for energy and momentum conservation for a QA/QH
        configuration with and without current terms.
        """
        for solver_options in [{'solveSympl': False}, {'solveSympl': True, 'dt': 1e-7, 'roottol': 1e-15, 'predictor_step': True}]:
            # First, test energy and momentum conservation in a QA vacuum field
            etabar = 1.2/1.2
            B0 = 1.0
            Bbar = 1.0
            G0 = 1.1
            psi0 = 0.8
            iota0 = 0.4
            bsh = BoozerAnalytic(etabar, B0, 0, G0, psi0, iota0)

            nparticles = 100
            m = PROTON_MASS
            q = ELEMENTARY_CHARGE
            tmax = 1e-3
            Ekin = 100.*ONE_EV
            vpar = np.sqrt(2*Ekin/m)

            np.random.seed(1)
            stz_inits = np.random.uniform(size=(nparticles, 3))
            vpar_inits = vpar*np.random.uniform(size=(nparticles, 1))
            smin = 0.2
            smax = 0.6
            thetamin = 0
            thetamax = np.pi
            zetamin = 0
            zetamax = np.pi
            stz_inits[:, 0] = stz_inits[:, 0]*(smax-smin) + smin
            stz_inits[:, 1] = stz_inits[:, 1]*(thetamax-thetamin) + thetamin
            stz_inits[:, 2] = stz_inits[:, 2]*(zetamax-zetamin) + zetamin

            bsh.set_points(stz_inits)
            modB_inits = bsh.modB()
            assert np.all(modB_inits > 0)  # Make sure all modBs are positive
            G_inits = bsh.G()
            mu_inits = (Ekin/m - 0.5*vpar_inits**2)/modB_inits
            psip_inits = bsh.psip()
            p_inits = vpar_inits*G_inits/modB_inits - q*psip_inits/m

            gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits,
                                                          vpar_inits,
                                                          tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc_vac',
                                                          stopping_criteria=[MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(
                                                              0.99), ToroidalTransitStoppingCriterion(10)],
                                                          tol=1e-12, solver_options=solver_options)

            # pick 100 random points on each trace, and ensure that
            # the energy is being conserved up to some precision

            N = 100
            max_energy_gc_error = np.array([])
            max_p_gc_error = np.array([])
            np.seterr(divide='ignore')
            for i in range(nparticles):
                gc_ty = gc_tys[i]
                idxs = np.random.randint(0, gc_ty.shape[0], size=(N, ))
                gc_xyzs = gc_ty[idxs, 1:4]
                bsh.set_points(gc_xyzs)
                AbsBs_gc = np.squeeze(bsh.modB())
                G_gc = np.squeeze(bsh.G())
                psip = np.squeeze(bsh.psip())

                energy_gc = np.array([])
                mu_gc = np.array([])
                p_gc = np.array([])
                vParInitial = gc_ty[0, 4]
                muInitial = mu_inits[i]
                pInitial = p_inits[i]
                for j in range(N):
                    v_gc = gc_ty[idxs[j], 4]
                    energy_gc = np.append(
                        energy_gc, m*(0.5*v_gc**2 + muInitial*AbsBs_gc[j]))
                    p_gc = np.append(
                        p_gc, v_gc*G_gc[j]/AbsBs_gc[j] - q*psip[j]/m)
                energy_gc_error = np.log10(
                    np.abs(energy_gc-Ekin)/np.abs(energy_gc[0]))
                p_gc_error = np.log10(np.abs(p_gc-pInitial)/np.abs(p_gc[0]))
                max_energy_gc_error = np.append(
                    max_energy_gc_error, max(energy_gc_error[3::]))
                max_p_gc_error = np.append(
                    max_p_gc_error, max(p_gc_error[3::]))

            if not solver_options['solveSympl']:
                assert max(max_energy_gc_error) < -8
                assert max(max_p_gc_error) < -8
            else:
                assert max(max_energy_gc_error) < -1.0
                assert max(max_p_gc_error) < -8

            # Now perform same tests for QH field with G and I terms added

            bsh.set_N(1)
            bsh.set_G1(0.2)
            bsh.set_I1(0.1)
            bsh.set_I0(0.5)

            stz_inits = np.random.uniform(size=(nparticles, 3))
            vpar_inits = vpar*np.random.uniform(size=(nparticles, 1))
            stz_inits[:, 0] = stz_inits[:, 0]*(smax-smin) + smin
            stz_inits[:, 1] = stz_inits[:, 1]*(thetamax-thetamin) + thetamin
            stz_inits[:, 2] = stz_inits[:, 2]*(zetamax-zetamin) + zetamin

            bsh.set_points(stz_inits)
            modB_inits = bsh.modB()[:, 0]
            assert np.all(modB_inits > 0)
            G_inits = bsh.G()[:, 0]
            I_inits = bsh.I()[:, 0]
            mu_inits = (Ekin/m - 0.5*vpar_inits[:, 0]**2)/modB_inits
            psip_inits = bsh.psip()[:, 0]
            psi_inits = bsh.psi0*stz_inits[:, 0]
            p_inits = vpar_inits[:, 0]*(G_inits + I_inits) / \
                modB_inits + q*(psi_inits - psip_inits)/m

            gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits,
                                                          vpar_inits,
                                                          tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc_noK',
                                                          stopping_criteria=[MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(
                                                              0.99), ToroidalTransitStoppingCriterion(10)],
                                                          tol=1e-12, solver_options=solver_options)

            max_energy_gc_error = np.array([])
            max_p_gc_error = np.array([])
            for i in range(nparticles):
                gc_ty = gc_tys[i]
                idxs = np.random.randint(0, gc_ty.shape[0], size=(N, ))
                gc_xyzs = gc_ty[idxs, 1:4]
                bsh.set_points(gc_xyzs)
                AbsBs_gc = np.squeeze(bsh.modB())
                G_gc = np.squeeze(bsh.G())
                I_gc = np.squeeze(bsh.I())
                psip = np.squeeze(bsh.psip())
                psi = np.squeeze(bsh.psi0*gc_ty[idxs, 1])

                energy_gc = np.array([])
                p_gc = np.array([])
                vParInitial = gc_ty[0, 4]
                muInitial = mu_inits[i]
                pInitial = p_inits[i]
                for j in range(N):
                    v_gc = gc_ty[idxs[j], 4]
                    energy_gc = np.append(
                        energy_gc, m*(0.5*v_gc**2 + muInitial*AbsBs_gc[j]))
                    p_gc = np.append(
                        p_gc, v_gc*(G_gc[j]+I_gc[j])/AbsBs_gc[j] + q*(psi[j] - psip[j])/m)
                energy_gc_error = np.log10(
                    np.abs(energy_gc-Ekin)/np.abs(energy_gc[0]))
                p_gc_error = np.log10(np.abs(p_gc-pInitial)/np.abs(p_gc[0]))
                max_energy_gc_error = np.append(
                    max_energy_gc_error, max(energy_gc_error[3::]))
                max_p_gc_error = np.append(
                    max_p_gc_error, max(p_gc_error[3::]))

            if not solver_options['solveSympl']:
                assert max(max_energy_gc_error) < -8
                assert max(max_p_gc_error) < -8
            else:
                assert max(max_energy_gc_error) < -1.0
                assert max(max_p_gc_error) < -8

            # Now perform same tests for QH field with G, I, and K terms added
            bsh.set_K1(0.6)

            bsh.set_points(stz_inits)
            modB_inits = bsh.modB()[:, 0]
            assert np.all(modB_inits > 0)
            G_inits = bsh.G()[:, 0]
            I_inits = bsh.I()[:, 0]
            mu_inits = (Ekin/m - 0.5*vpar_inits[:, 0]**2)/modB_inits
            psip_inits = bsh.psip()[:, 0]
            psi_inits = bsh.psi0*stz_inits[:, 0]
            p_inits = vpar_inits[:, 0]*(G_inits + I_inits) / \
                modB_inits + q*(psi_inits - psip_inits)/m

            gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                          tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc',
                                                          stopping_criteria=[MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(
                                                              0.99), ToroidalTransitStoppingCriterion(10)],
                                                          tol=1e-12, solver_options=solver_options)
            max_energy_gc_error = np.array([])
            max_p_gc_error = np.array([])
            for i in range(nparticles):
                gc_ty = gc_tys[i]
                idxs = np.random.randint(0, gc_ty.shape[0], size=(N, ))
                gc_xyzs = gc_ty[idxs, 1:4]
                bsh.set_points(gc_xyzs)
                AbsBs_gc = np.squeeze(bsh.modB())
                G_gc = np.squeeze(bsh.G())
                I_gc = np.squeeze(bsh.I())
                psip = np.squeeze(bsh.psip())
                psi = np.squeeze(bsh.psi0*gc_ty[idxs, 1])

                energy_gc = np.array([])
                p_gc = np.array([])
                muInitial = mu_inits[i]
                pInitial = p_inits[i]
                for j in range(N):
                    v_gc = gc_ty[idxs[j], 4]
                    energy_gc = np.append(
                        energy_gc, m*(0.5*v_gc**2 + muInitial*AbsBs_gc[j]))
                    p_gc = np.append(
                        p_gc, v_gc*(G_gc[j]+I_gc[j])/AbsBs_gc[j] + q*(psi[j] - psip[j])/m)
                energy_gc_error = np.log10(
                    np.abs(energy_gc-Ekin)/np.abs(energy_gc[0]))
                p_gc_error = np.log10(np.abs(p_gc-pInitial)/np.abs(p_gc[0]))
                max_energy_gc_error = np.append(
                    max_energy_gc_error, max(energy_gc_error[3::]))
                max_p_gc_error = np.append(
                    max_p_gc_error, max(p_gc_error[3::]))

            if not solver_options['solveSympl']:
                assert max(max_energy_gc_error) < -8
                assert max(max_p_gc_error) < -8
            else:
                assert max(max_energy_gc_error) < -1.0
                assert max(max_p_gc_error) < -8

            # Now trace with forget_exact_path = True. Check that gc_zeta_hits is the same
            gc_tys, gc_zeta_hits_2 = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                            tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc',
                                                            stopping_criteria=[MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(
                                                                0.99), ToroidalTransitStoppingCriterion(10)],
                                                            tol=1e-12, forget_exact_path=True, solver_options=solver_options)

            for i in range(len(gc_zeta_hits_2)):
                assert np.allclose(gc_zeta_hits[i], gc_zeta_hits_2[i])

            assert len(gc_tys) == N
            assert np.shape(gc_tys[0])[0] == 2
            np.seterr(divide='warn')

    def test_compute_poloidal_toroidal_transits(self):
        """
        Trace low-energy particle on an iota=1 field line for one toroidal
        transit. Check that mpol = ntor = 1. This is performed for both a
        BoozerAnalytic field and a PoloidalField + ToroidalField
        """
        # First test for BoozerAnalytic field
        etabar = 1.2
        B0 = 1.0
        Bbar = 1.0
        G0 = 1.1
        psi0 = 0.8
        iota0 = 1.0
        bsh = BoozerAnalytic(etabar, B0, 4, G0, psi0, iota0)

        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        tmax = 1e-4
        Ekin = 100.*ONE_EV
        vpar = np.sqrt(2*Ekin/m)

        np.random.seed(1)
        stz_inits = np.random.uniform(size=(1, 3))
        vpar_inits = vpar*np.ones((1, 1))
        smin = 0.2
        smax = 0.6
        thetamin = 0
        thetamax = np.pi
        zetamin = 0
        zetamax = np.pi
        stz_inits[:, 0] = stz_inits[:, 0]*(smax-smin) + smin
        stz_inits[:, 1] = stz_inits[:, 1]*(thetamax-thetamin) + thetamin
        stz_inits[:, 2] = stz_inits[:, 2]*(zetamax-zetamin) + zetamin

        for solver_options in [{'solveSympl': True, 'dt': 5e-7, 'roottol': 1e-15, 'predictor_step': True}, {'solveSympl': False}]:
            gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                          tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc_vac',
                                                          stopping_criteria=[MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(
                                                              0.99), ToroidalTransitStoppingCriterion(1)],
                                                          tol=1e-12, solver_options=solver_options)

            mpol = compute_poloidal_transits(gc_tys)
            ntor = compute_toroidal_transits(gc_tys)

            assert mpol == 1
            assert ntor == 1

    def test_toroidal_flux_stopping_criterion(self):
        """
        Trace particles in a BoozerAnalyticField at high energy so that
        particle orbit widths are large. Ensure that trajectories remain
        within MinToroidalFlux and MaxToroidalFlux.
        """
        etabar = 1.2/1.2
        B0 = 1.0
        Bbar = 1.0
        G0 = 1.1
        psi0 = 0.8
        iota0 = 1.0
        bsh = BoozerAnalytic(etabar, B0, 4, G0, psi0, iota0)

        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        tmax = 1e-4
        Ekin = 100000.*ONE_EV
        vpar = np.sqrt(2*Ekin/m)

        Nparticles = 20
        np.random.seed(1)
        stz_inits = np.random.uniform(size=(Nparticles, 3))
        vpar_inits = vpar*np.random.uniform(size=(Nparticles, 1))
        smin = 0.4
        smax = 0.6
        thetamin = 0
        thetamax = np.pi
        zetamin = 0
        zetamax = np.pi
        stz_inits[:, 0] = stz_inits[:, 0]*(smax-smin) + smin
        stz_inits[:, 1] = stz_inits[:, 1]*(thetamax-thetamin) + thetamin
        stz_inits[:, 2] = stz_inits[:, 2]*(zetamax-zetamin) + zetamin

        for solver_options in [{'solveSympl': False}, {'solveSympl': True, 'dt': 1e-8, 'roottol': 1e-8, 'predictor_step': True}]:
            gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits,
                                                          vpar_inits,
                                                          tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc_vac',
                                                          stopping_criteria=[MinToroidalFluxStoppingCriterion(
                                                              0.4), MaxToroidalFluxStoppingCriterion(0.6)],
                                                          tol=1e-5, solver_options=solver_options, forget_exact_path=False)

            for i in range(Nparticles):
                assert np.all(gc_tys[i][:, 1] > 0.4)
                assert np.all(gc_tys[i][:, 1] < 0.6)

    def test_compute_resonances(self):
        """
        Compute particle resonances for low energy particles in a BoozerAnalytic
        field with zero magnetic shear. Check that helicity of resonances
        matches the rotational transform. 
        """
        for solver_options in [{'solveSympl': True, 'dt': 1e-7, 'roottol': 1e-7, 'predictor_step': True}, {'solveSympl': False}]:
            etabar = 1.2
            B0 = 1.0
            Bbar = 1.0
            R0 = 1.0
            G0 = R0*B0
            psi0 = B0*(0.1)**2/2
            iota0 = 0.5
            bsh = BoozerAnalytic(etabar, B0, 0, G0, psi0, iota0)

            m = PROTON_MASS
            q = ELEMENTARY_CHARGE
            tmax = 1e-4
            Ekin = 1e3*ONE_EV
            vpar = np.sqrt(2*Ekin/m)

            Nparticles = 10
            np.random.seed(1)
            stz_inits = np.random.uniform(size=(Nparticles, 3))
            vpar_inits = vpar*np.ones((Nparticles, 1))
            smin = 0.1
            smax = 0.2
            thetamin = 0
            thetamax = np.pi
            stz_inits[:, 0] = stz_inits[:, 0]*(smax-smin) + smin
            stz_inits[:, 1] = stz_inits[:, 1]*(thetamax-thetamin) + thetamin
            stz_inits[:, 2] = 0.

            gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits,
                                                          vpar_inits,
                                                          tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[0], mode='gc_vac',
                                                          stopping_criteria=[MinToroidalFluxStoppingCriterion(
                                                              0.01), MaxToroidalFluxStoppingCriterion(0.99), ToroidalTransitStoppingCriterion(100)],
                                                          tol=1e-7, solver_options=solver_options)

            resonances = compute_resonances(gc_tys, gc_zeta_hits, delta=0.01)

            iota_min = iota0-0.01
            iota_max = iota0+0.01
            for i in range(len(resonances)):
                h = resonances[i][5]/resonances[i][6]
                assert h > iota_min
                assert h < iota_max

    def test_vpars_zetas_stop(self):
        """
        Trace particles in a BoozerAnalyticField and test for vpars_stop 
        and phis(zetas)_stop.
        """
        # Same field from test_energy_momentum_conservation_boozer
        etabar = 1.2/1.2
        B0 = 1.0
        Bbar = 1.0
        G0 = 1.1
        psi0 = 0.8
        iota0 = 0.4
        bsh = BoozerAnalytic(etabar, B0, 0, G0, psi0, iota0)

        nparticles = 100
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        tmax = 1e-4
        Ekin = 100.*ONE_EV
        vpar = np.sqrt(2*Ekin/m)

        np.random.seed(1)
        stz_inits = np.random.uniform(size=(nparticles, 3))
        vpar_inits = vpar*np.random.uniform(size=(nparticles, 1))
        smin = 0.2
        smax = 0.6
        thetamin = 0
        thetamax = np.pi
        zetamin = 0
        zetamax = np.pi
        stz_inits[:, 0] = stz_inits[:, 0]*(smax-smin) + smin
        stz_inits[:, 1] = stz_inits[:, 1]*(thetamax-thetamin) + thetamin
        stz_inits[:, 2] = stz_inits[:, 2]*(zetamax-zetamin) + zetamin

        bsh.set_points(stz_inits)
        modB_inits = bsh.modB()
        assert np.all(modB_inits > 0)  # Make sure all modBs are positive

        for solver_options in [{'solveSympl': False}, {'solveSympl': True, 'dt': 1e-5, 'roottol': 1e-5, 'predictor_step': True}]:
            # zetas_stop with mismatched zetas and omegas
            solver_options['zetas_stop'] = True
            zetas = [1, 2, 3, 4]
            omegas = [0]
            zetas = np.array(zetas)
            with self.assertRaises(ValueError):
                gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits,
                                                              vpar_inits,
                                                              tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=zetas, omegas=omegas, mode='gc_vac',
                                                              stopping_criteria=[
                                                                  MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99)],
                                                              tol=1e-7,
                                                              dt_save=1e-5,
                                                              solver_options=solver_options)

            # zetas_stop with no omegas
            gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits,
                                                          vpar_inits,
                                                          tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=zetas, mode='gc_vac',
                                                          stopping_criteria=[
                                                              MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99)],
                                                          tol=1e-12, solver_options=solver_options, dt_save=1e-8)
            for i in range(nparticles):
                if len(gc_zeta_hits[i]):
                    # Check that we hit a stopping criteria
                    if (gc_zeta_hits[i][-1][1] >= 0):
                        idx = int(gc_zeta_hits[i][-1][1])
                        assert np.isclose(
                            gc_zeta_hits[i][-1, 4] % (2*np.pi), zetas[idx], rtol=1e-12, atol=0)
                        if gc_tys[i][-2, 3] > gc_tys[i][-1, 3]:
                            lower_bound = zetas[idx]
                            upper_bound = lower_bound + 1
                        else:
                            upper_bound = zetas[idx]
                            lower_bound = upper_bound - 1
                        assert np.all(gc_tys[i][1:-1, 3] < upper_bound)
                        assert np.all(gc_tys[i][1:-1, 3] > lower_bound)

            # add omegas
            omegas = [0.1, 0.2, 0.3, 0.4]
            omegas = np.array(omegas)
            gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits,
                                                          vpar_inits,
                                                          tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=zetas, omegas=omegas, mode='gc_vac',
                                                          stopping_criteria=[
                                                              MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99)],
                                                          tol=1e-12, solver_options=solver_options, dt_save=1e-8)
            for i in range(nparticles):
                if len(gc_zeta_hits[i]):
                    # Check that we hit a zeta plane
                    if (gc_zeta_hits[i][-1][1] >= 0):
                        idx = int(gc_zeta_hits[i][-1][1])
                        assert np.isclose(gc_zeta_hits[i][-1, 4] % (
                            2*np.pi), zetas[idx]+omegas[idx]*gc_zeta_hits[i][-1, 0], rtol=1e-12, atol=0)
                        if gc_tys[i][-2, 3] > gc_tys[i][-1, 3]:
                            lower_bound = zetas[idx] + \
                                omegas[idx]*gc_tys[i][1:-1, 0]
                            upper_bound = (
                                zetas[idx]+1) + (omegas[idx]+0.1)*gc_tys[i][1:-1, 0]
                        else:
                            upper_bound = zetas[idx] + \
                                omegas[idx]*gc_tys[i][1:-1, 0]
                            lower_bound = (
                                zetas[idx]-1) + (omegas[idx]-0.1)*gc_tys[i][1:-1, 0]
                        assert np.all(gc_tys[i][1:-1, 3] < upper_bound)
                        assert np.all(gc_tys[i][1:-1, 3] > lower_bound)

            # vpars_stop
            solver_options['zetas_stop'] = False
            solver_options['vpars_stop'] = True
            vpars = [vpar*0.25, vpar*0.5, vpar*0.75]
            vpars = np.array(vpars)
            gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits,
                                                          vpar_inits,
                                                          tmax=tmax, mass=m, charge=q, Ekin=Ekin, vpars=vpars, mode='gc_vac',
                                                          stopping_criteria=[
                                                              MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99)],
                                                          tol=1e-7, solver_options=solver_options, dt_save=1e-8)
            for i in range(nparticles):
                if len(gc_zeta_hits[i]):
                    idx = int(gc_zeta_hits[i][-1][1])
                    if idx >= 0:
                        assert np.isclose(
                            gc_zeta_hits[i][-1, 5], vpars[idx], rtol=1e-10, atol=0)
                        # vpar decreasing with time
                        if gc_tys[i][-2, 4] > gc_tys[i][-1, 4]:
                            lower_bound = vpars[idx]
                            upper_bound = lower_bound + 0.25*vpar
                        # vpar increasing with time
                        else:
                            upper_bound = vpars[idx]
                            lower_bound = upper_bound - 0.25*vpar
                        assert np.all(gc_tys[i][:-1, 4] < upper_bound)
                        assert np.all(gc_tys[i][:-1, 4] > lower_bound)

            # test vpars_stop and zetas_stop together
            solver_options['zetas_stop'] = True
            solver_options['vpars_stop'] = True

            gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                          tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=zetas, omegas=omegas, vpars=vpars, mode='gc_vac',
                                                          stopping_criteria=[
                                                              MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99)],
                                                          tol=1e-7, solver_options=solver_options, dt_save=1e-8)
            for i in range(nparticles):
                if len(gc_zeta_hits[i]):
                    idx = int(gc_zeta_hits[i][-1][1])
                    if idx >= 0:
                        if idx >= len(zetas):
                            res = gc_zeta_hits[i][-1, 5]
                            stop = vpars[idx-len(zetas)]
                        else:
                            res = gc_zeta_hits[i][-1, 4] % (2*np.pi)
                            stop = zetas[idx]+omegas[idx] * \
                                gc_zeta_hits[i][-1, 0]
                        assert np.isclose(res, stop, rtol=1e-10, atol=0)

    def test_sympl_dense_output(self):
        """
        Check symplectic dense output against integration using smaller steps.
        """
        # Same field from test_energy_momentum_conservation_boozer
        etabar = 1.2/1.2
        B0 = 1.0
        Bbar = 1.0
        G0 = 1.1
        psi0 = 0.8
        iota0 = 0.4
        bsh = BoozerAnalytic(etabar, B0, 0, G0, psi0, iota0)

        nparticles = 100
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        tmax = 1e-4
        Ekin = 100.*ONE_EV
        vpar = np.sqrt(2*Ekin/m)

        np.random.seed(1)
        stz_inits = np.random.uniform(size=(nparticles, 3))
        vpar_inits = vpar*np.random.uniform(size=(nparticles, 1))
        smin = 0.2
        smax = 0.6
        thetamin = 0
        thetamax = np.pi
        zetamin = 0
        zetamax = np.pi
        stz_inits[:, 0] = stz_inits[:, 0]*(smax-smin) + smin
        stz_inits[:, 1] = stz_inits[:, 1]*(thetamax-thetamin) + thetamin
        stz_inits[:, 2] = stz_inits[:, 2]*(zetamax-zetamin) + zetamin

        bsh.set_points(stz_inits)
        modB_inits = bsh.modB()
        assert np.all(modB_inits > 0)  # Make sure all modBs are positive

        solver_options = {'solveSympl': True, 'dt': 1e-7,
                          'roottol': 1e-15, 'predictor_step': True}
        gc_tys, gc_zeta_hits = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                      tmax=tmax+5e-8, mass=m, charge=q, Ekin=Ekin, mode='gc_vac',
                                                      stopping_criteria=[
                                                          MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99)],
                                                      tol=1e-12, solver_options=solver_options)
        diffs = np.array([])
        test_options = {'solveSympl': True, 'dt': 1e-8,
                        'roottol': 1e-15, 'predictor_step': True}
        for i in range(nparticles):
            stz_init = np.array([gc_tys[i][-2, 1:4]])
            vpar_init = np.array([gc_tys[i][-2, 4]])
            test_tmax = gc_tys[i][-1, 0] - gc_tys[i][-2, 0]
            gc_ty, gc_phi_hit = trace_particles_boozer(bsh, stz_init, vpar_init,
                                                       tmax=test_tmax, mass=m, charge=q, Ekin=Ekin, mode='gc_vac',
                                                       stopping_criteria=[
                                                           MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99)],
                                                       tol=1e-12, solver_options=test_options)

            diffs = np.append(diffs, np.log10(
                np.abs(gc_tys[i][-1, 1:]-gc_ty[0][-1, 1:])/np.abs(gc_ty[0][-1, 1:])))

        s_diff, theta_diff, zeta_diff, vpar_diff = [
            diffs[i::4] for i in range(4)]
        assert max(s_diff) < -5
        assert max(theta_diff) < -5
        assert max(zeta_diff) < -5
        assert max(vpar_diff) < -5


if __name__ == "__main__":
    unittest.main()
