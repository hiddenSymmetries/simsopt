import unittest
import logging

logging.basicConfig()

import numpy as np

from simsopt.field.coil import coils_via_symmetries
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.configs.zoo import get_ncsx_data
from simsopt.field.tracing import trace_particles_starting_on_curve, SurfaceClassifier, \
    particles_to_vtk, LevelsetStoppingCriterion, compute_gc_radius, gc_to_fullorbit_initial_guesses, \
    IterationStoppingCriterion, trace_particles_starting_on_surface, trace_particles_boozer, \
    MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion, ToroidalTransitStoppingCriterion, \
    compute_poloidal_transits, compute_toroidal_transits, trace_particles, compute_resonances
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.boozermagneticfield import BoozerAnalytic
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule, ToroidalField, PoloidalField
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from simsopt.geo.curverzfourier import CurveRZFourier
import simsoptpp as sopp


try:
    import pyevtk
    with_evtk = True
except ImportError:
    with_evtk = False

try:
    from mpi4py import MPI
except:
    MPI = None

try:
    from simsopt.mhd.vmec import Vmec
    vmec_found = True
except ImportError:
    vmec_found = False


def validate_phi_hits(phi_hits, bfield, nphis):
    """
    After hitting one of the phi planes, three things can happen:
    A: We can hit the same phi plane again (i.e. the particle bounces)
    B: We can hit one of the stopping criteria
    C: We can hit the next phi plane. Note that the definition of 'next'
       depends on the direction the particle is going. By looking at the
       direction of the B field and the tangential velocity, we can figure out
       if the particle was going clockwise or anti clockwise, and hence we know
       which is the 'next' phi plane.
    """
    for i in range(len(phi_hits)-1):
        this_idx = int(phi_hits[i][1])
        this_B = bfield.set_points(np.asarray(
            phi_hits[i][2:5]).reshape((1, 3))).B_cyl()
        vtang = phi_hits[i][5]
        forward = np.sign(vtang * this_B[0, 1]) > 0
        next_idx = int(phi_hits[i+1][1])
        hit_same_phi = next_idx == this_idx
        hit_stopping_criteria = next_idx < 0
        hit_next_phi = next_idx == (this_idx + 1) % nphis
        hit_previous_phi = next_idx == (this_idx - 1) % nphis
        if forward:
            if not (hit_next_phi or hit_same_phi or hit_stopping_criteria):
                return False
        else:
            if not (hit_previous_phi or hit_same_phi or hit_stopping_criteria):
                return False
    return True


class ParticleTracingTesting(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ParticleTracingTesting, self).__init__(*args, **kwargs)
        logger = logging.getLogger('simsopt.field.tracing')
        logger.setLevel(1)
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        n = 16

        rrange = (1.1, 1.8, n)
        phirange = (0, 2*np.pi/3, n*2)
        zrange = (0, 0.3, n//2)
        bsh = InterpolatedField(
            bs, UniformInterpolationRule(5),
            rrange, phirange, zrange, True, nfp=3, stellsym=True
        )
        print(bsh.estimate_error_B(1000))
        print(bsh.estimate_error_GradAbsB(1000))
        # trick to clear the cache in the biot savart class to reduce memory usage
        bs.set_points(np.asarray([[0., 0., 0.]])).GradAbsB()
        self.bsh = bsh
        self.ma = ma
        if with_evtk:
            bsh.to_vtk('/tmp/bfield')

    def test_guidingcenter_vs_fullorbit(self):
        bsh = self.bsh
        ma = self.ma
        nparticles = 2
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        Ekin = 1000*ONE_EV
        np.random.seed(1)
        umin = 0.25
        umax = 0.75
        tmax = 2e-5
        with self.assertRaises(RuntimeError):
            gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
                ma, bsh, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
                Ekin=Ekin, umin=umin, umax=umax,
                phis=[], mode='gc')

        gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
            ma, bsh, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
            Ekin=Ekin, umin=umin, umax=umax,
            phis=[], mode='gc_vac')
        fo_tys, fo_phi_hits = trace_particles_starting_on_curve(
            ma, bsh, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
            Ekin=Ekin, umin=umin, umax=umax,
            phis=[], mode='full')
        if with_evtk:
            particles_to_vtk(gc_tys, '/tmp/particles_gc')
            particles_to_vtk(fo_tys, '/tmp/particles_fo')

        # pick 100 random points on each trace, and ensure that the guiding
        # center and the full orbit simulation are close to each other

        N = 100
        for i in range(nparticles):
            gc_ty = gc_tys[i]
            fo_ty = fo_tys[i]
            idxs = np.random.randint(0, gc_ty.shape[0], size=(N, ))
            ts = gc_ty[idxs, 0]
            gc_xyzs = gc_ty[idxs, 1:4]
            bsh.set_points(gc_xyzs)
            Bs = bsh.B()
            AbsBs = bsh.AbsB()
            for j in range(N):
                jdx = np.argmin(np.abs(ts[j]-fo_ty[:, 0]))
                v = fo_ty[jdx, 4:]
                Bunit = Bs[j, :]/AbsBs[j, 0]
                vperp = np.linalg.norm(v - np.sum(v*Bunit)*Bunit)
                r = compute_gc_radius(m, vperp, q, AbsBs[j, 0])
                dist = np.linalg.norm(gc_xyzs[j, :] - fo_ty[jdx, 1:4])
                assert dist < 8*r

    def test_guidingcenterphihits(self):
        bsh = self.bsh
        ma = self.ma
        nparticles = 2
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        Ekin = 9000 * ONE_EV
        nphis = 10
        phis = np.linspace(0, 2*np.pi, nphis, endpoint=False)
        mpol = 5
        ntor = 5
        nfp = 3
        s = SurfaceRZFourier(
            mpol=mpol, ntor=ntor, stellsym=True, nfp=nfp,
            quadpoints_phi=np.linspace(0, 1, nfp*2*ntor+1, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 2*mpol+1, endpoint=False))
        s.fit_to_curve(ma, 0.10, flip_theta=False)
        sc = SurfaceClassifier(s, h=0.1, p=2)
        if with_evtk:
            sc.to_vtk('/tmp/classifier')
        # check that the axis is classified as inside the domain
        assert sc.evaluate_xyz(ma.gamma()[:1, :]) > 0
        assert sc.evaluate_xyz(2*ma.gamma()[:1, :]) < 0
        np.random.seed(1)
        gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
            ma, bsh, nparticles, tmax=1e-4, seed=1, mass=m, charge=q,
            Ekin=Ekin, umin=-0.1, umax=+0.1, phis=phis, mode='gc_vac',
            stopping_criteria=[LevelsetStoppingCriterion(sc)], forget_exact_path=True)

        for i in range(nparticles):
            assert gc_tys[i].shape[0] == 2
            assert validate_phi_hits(gc_phi_hits[i], bsh, nphis)

    def test_gc_to_full(self):
        N = 100
        etas = np.linspace(0, 2*np.pi, N, endpoint=False)
        ma = self.ma
        bsh = self.bsh
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        Ekin = 9000*ONE_EV
        vtotal = np.sqrt(2*Ekin/m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
        xyz_gc = ma.gamma()[:10, :]
        np.random.seed(1)
        vtangs = np.random.uniform(size=(xyz_gc.shape[0], )) * vtotal
        xyz, vxyz, _ = gc_to_fullorbit_initial_guesses(bsh, xyz_gc, vtangs, vtotal, m, q, eta=etas[0])
        for eta in etas[1:]:
            tmp = gc_to_fullorbit_initial_guesses(bsh, xyz_gc, vtangs, vtotal, m, q, eta=eta)
            xyz += tmp[0]
            vxyz += tmp[1]
            radius = tmp[2]
            assert np.allclose(np.abs(np.linalg.norm(xyz_gc-tmp[0], axis=1) - radius), 0)
            assert np.all(np.abs(np.linalg.norm(tmp[1], axis=1) - vtotal)/vtotal < 1e-10)
        xyz *= 1/N
        vxyz *= 1/N
        assert np.linalg.norm(xyz - xyz_gc) < 1e-5
        B = bsh.set_points(xyz_gc).B()
        B *= 1./bsh.AbsB()
        assert np.linalg.norm((vxyz - B * vtangs[:, None])/vtotal) < 1e-5

    def test_energy_conservation(self):
        # Test conservation of Energy = m v^2/2=(m/2) (vparallel^2+ 2muB)
        # where mu=v_perp^2/(2B) is the adiabatic invariant. In the
        # the presence of a strong magnetic field, mu should also be
        # conserved. Energy is conserved regardless of the magnitude
        # of the magnetic field.
        bsh = self.bsh
        ma = self.ma
        nparticles = 1
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        tmax = 1e-5
        Ekin = 100*ONE_EV
        np.random.seed(1)

        gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
            ma, bsh, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
            Ekin=Ekin, umin=-0.5, umax=-0.25,  # pitch angle so that we have both par and perp contribution
            phis=[], mode='gc_vac', tol=1e-11)
        fo_tys, fo_phi_hits = trace_particles_starting_on_curve(
            ma, bsh, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
            Ekin=Ekin, umin=-0.5, umax=-0.25,
            phis=[], mode='full', tol=1e-11)
        if with_evtk:
            particles_to_vtk(gc_tys, '/tmp/particles_gc')
            particles_to_vtk(fo_tys, '/tmp/particles_fo')

        # pick 100 random points on each trace, and ensure that
        # the energy is being conserved both in the guiding center
        # and full orbit cases up to some precision

        N = 100
        max_energy_gc_error = np.array([])
        max_energy_fo_error = np.array([])
        max_mu_gc_error = np.array([])
        max_mu_fo_error = np.array([])
        for i in range(nparticles):
            fo_ty = fo_tys[i]
            gc_ty = gc_tys[i]
            idxs = np.random.randint(0, gc_ty.shape[0], size=(N, ))
            gc_xyzs = gc_ty[idxs, 1:4]
            fo_xyzs = fo_ty[idxs, 1:4]
            bsh.set_points(gc_xyzs)
            AbsBs_gc = bsh.AbsB()
            bsh.set_points(fo_xyzs)
            AbsBs = bsh.AbsB()
            Bs = bsh.B()

            energy_fo = np.array([])
            energy_gc = np.array([])
            mu_fo = np.array([])
            mu_gc = np.array([])
            vParInitial = gc_ty[0, 4]
            muInitial = Ekin/(m*AbsBs_gc[0, 0])-vParInitial**2/(2*AbsBs_gc[0, 0])
            for j in range(N):
                v_fo = fo_ty[idxs[j], 4:]
                v_gc = gc_ty[idxs[j], 4]
                Bunit = Bs[j, :]/AbsBs[j, 0]
                vperp = np.linalg.norm(v_fo - np.sum(v_fo*Bunit)*Bunit)
                energy_fo = np.append(energy_fo, (m/2)*(v_fo[[0]]**2+v_fo[[1]]**2+v_fo[[2]]**2))
                energy_gc = np.append(energy_gc, (m/2)*(v_gc**2+2*muInitial*AbsBs_gc[j, 0]))
                mu_fo = np.append(mu_fo, vperp**2/(2*AbsBs[j, 0]))
                mu_gc = np.append(mu_gc, Ekin/(m*AbsBs_gc[j, 0])-v_gc**2/(2*AbsBs_gc[j, 0]))
            energy_fo_error = np.log10(np.abs(energy_fo-energy_fo[0]+1e-30)/energy_fo[0])
            energy_gc_error = np.log10(np.abs(energy_gc-energy_gc[0]+1e-30)/energy_gc[0])
            mu_fo_error = np.log10(np.abs(mu_fo-mu_fo[0]+1e-30)/mu_fo[0])
            mu_gc_error = np.log10(np.abs(mu_gc-mu_gc[0]+1e-30)/mu_gc[0])
            max_energy_fo_error = np.append(max_energy_fo_error, max(energy_fo_error[3::]))
            max_energy_gc_error = np.append(max_energy_gc_error, max(energy_gc_error[3::]))
            max_mu_fo_error = np.append(max_mu_fo_error, max(mu_fo_error[3::]))
            max_mu_gc_error = np.append(max_mu_gc_error, max(mu_gc_error[3::]))
        print("Max Energy Error for full orbit     = ", max_energy_fo_error)
        print("Max Energy Error for guiding center = ", max_energy_gc_error)
        print("Max mu Error for full orbit         = ", max_mu_fo_error)
        print("Max mu Error for guiding center     = ", max_mu_gc_error)
        assert max(max_energy_fo_error) < -8
        assert max(max_energy_gc_error) < -3
        assert max(max_mu_fo_error) < -3
        assert max(max_mu_gc_error) < -6

    def test_angularmomentum_conservation(self):
        # Test conservation of canonical angular momentum pphi
        # in an axisymmetric geometry where
        # pphi=q*poloidal_magnetic_flux+m*v_parallel*G/B
        R0test = 1.0
        B0test = 2.0
        qtest = 3.1
        bsh = ToroidalField(R0test, B0test)+PoloidalField(R0test, B0test, qtest)
        ma = CurveXYZFourier(300, 1)
        ma.set_dofs([0, 0, R0test+0.01, 0, R0test+0.01, 0., 0, 0., 0.])

        nparticles = 4
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        tmax = 1e-4
        Ekin = 100*ONE_EV
        np.random.seed(1)

        gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
            ma, bsh, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
            Ekin=Ekin, umin=-0.5, umax=-0.25,  # pitch angle so that we have both par and perp contribution
            phis=[], mode='gc_vac', tol=1e-11)
        if with_evtk:
            particles_to_vtk(gc_tys, '/tmp/particles_gc')

        # pick 100 random points on each trace

        N = 100
        max_pphi1_gc_error = np.array([])
        max_pphi2_gc_error = np.array([])
        max_pphi_gc_error = np.array([])
        for i in range(nparticles):
            gc_ty = gc_tys[i]
            idxs = np.random.randint(0, gc_ty.shape[0], size=(N, ))
            gc_xyzs = gc_ty[idxs, 1:4]
            bsh.set_points(gc_xyzs)
            AbsBs = bsh.AbsB()

            pphi1_gc = np.array([])
            pphi2_gc = np.array([])
            for j in range(N):
                v_gc = gc_ty[idxs[j], 4]
                r_squared = (np.sqrt(gc_xyzs[j][0]**2+gc_xyzs[j][1]**2)-R0test)**2+gc_xyzs[j][2]**2
                psi_poloidal = B0test*r_squared/qtest/2
                p_phi1 = q*psi_poloidal
                p_phi2 = m*v_gc*B0test*R0test/AbsBs[j]
                pphi1_gc = np.append(pphi1_gc, p_phi1)
                pphi2_gc = np.append(pphi2_gc, p_phi2)
            pphi_gc = pphi1_gc+pphi2_gc
            pphi1_error = np.log10(np.abs(pphi1_gc-pphi1_gc[0]+1e-30)/np.abs(pphi1_gc[0]))
            pphi2_error = np.log10(np.abs(pphi2_gc-pphi2_gc[0]+1e-30)/np.abs(pphi2_gc[0]))
            pphi_error = np.log10(np.abs(pphi_gc-pphi_gc[0]+1e-30)/np.abs(pphi_gc[0]))
            max_pphi1_gc_error = np.append(max_pphi1_gc_error, max(pphi1_error[3::]))
            max_pphi2_gc_error = np.append(max_pphi2_gc_error, max(pphi2_error[3::]))
            max_pphi_gc_error = np.append(max_pphi_gc_error, max(pphi_error[3::]))
        print("Max log10 variation of pphi1_gc = ", max_pphi1_gc_error)
        print("Max log10 variation of pphi2_gc = ", max_pphi2_gc_error)
        print("Max log10 variation of pphi1_gc + pphi2_gc = ", max_pphi_gc_error)
        assert max(max_pphi_gc_error) < -3

    def test_stopping_criteria(self):
        bsh = self.bsh
        ma = self.ma
        nparticles = 1
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        tmax = 1e-3
        Ekin = 9000*ONE_EV
        np.random.seed(1)

        gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
            ma, bsh, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
            Ekin=Ekin, umin=-0.80, umax=-0.70,
            phis=[], mode='gc_vac', tol=1e-11, stopping_criteria=[IterationStoppingCriterion(10)])
        assert len(gc_tys[0]) == 11

        # consider a particle with mostly perpendicular velocity so that it get's lost
        ntor = 1
        mpol = 1
        stellsym = True
        nfp = 3
        phis = np.linspace(0, 1, 50, endpoint=False)
        thetas = np.linspace(0, 1, 50, endpoint=False)
        s = SurfaceRZFourier(
            mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.fit_to_curve(ma, 0.03, flip_theta=False)
        sc = SurfaceClassifier(s, h=0.1, p=2)

        gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
            ma, bsh, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
            Ekin=Ekin, umin=-0.01, umax=+0.01,
            phis=[], mode='gc_vac', tol=1e-11, stopping_criteria=[LevelsetStoppingCriterion(sc)])
        if with_evtk:
            particles_to_vtk(gc_tys, '/tmp/particles_gc')
        assert gc_phi_hits[0][-1][1] == -1
        assert np.all(sc.evaluate_xyz(gc_tys[0][:, 1:4]) > 0)

    def test_tracing_on_surface_runs(self):
        bsh = self.bsh
        ma = self.ma
        nparticles = 1
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        tmax = 1e-3
        Ekin = 9000*ONE_EV
        np.random.seed(1)

        ntor = 1
        mpol = 1
        stellsym = True
        nfp = 3
        phis = np.linspace(0, 1, 50, endpoint=False)
        thetas = np.linspace(0, 1, 50, endpoint=False)
        s = SurfaceRZFourier(
            mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.fit_to_curve(ma, 0.03, flip_theta=False)

        gc_tys, gc_phi_hits = trace_particles_starting_on_surface(
            s, bsh, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
            Ekin=Ekin, umin=-0.80, umax=-0.70,
            phis=[], mode='gc_vac', tol=1e-11, stopping_criteria=[IterationStoppingCriterion(10)])
        assert len(gc_tys[0]) == 11


class BoozerGuidingCenterTracingTesting(unittest.TestCase):

    def test_energy_momentum_conservation_boozer(self):
        """
        Trace 100 particles in a BoozerAnalytic field for 100 toroidal
        transits. Check for energy and momentum conservation for a QA/QH
        configuration with and without current terms.
        """
        # First, test energy and momentum conservation in a QA vacuum field
        etabar = 1.2
        B0 = 1.0
        Bbar = 1.0
        G0 = 1.1
        psi0 = 0.8
        iota0 = 0.4
        bsh = BoozerAnalytic(etabar, B0, 0, G0, psi0, iota0)

        nparticles = 100
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        tmax = 1e-5
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
        G_inits = bsh.G()
        mu_inits = (Ekin/m - 0.5*vpar_inits**2)/modB_inits
        psip_inits = bsh.psip()
        p_inits = vpar_inits*G_inits/modB_inits - q*psip_inits/m

        gc_tys, gc_phi_hits = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                     tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc_vac',
                                                     stopping_criteria=[MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99), ToroidalTransitStoppingCriterion(100, True)],
                                                     tol=1e-12)

        # pick 100 random points on each trace, and ensure that
        # the energy is being conserved up to some precision

        N = 100
        max_energy_gc_error = np.array([])
        max_mu_gc_error = np.array([])
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
                energy_gc = np.append(energy_gc, m*(0.5*v_gc**2 + muInitial*AbsBs_gc[j]))
                mu_gc = np.append(mu_gc, Ekin/(m*AbsBs_gc[j]) - 0.5*v_gc**2/AbsBs_gc[j])
                p_gc = np.append(p_gc, v_gc*G_gc[j]/AbsBs_gc[j] - q*psip[j]/m)
            energy_gc_error = np.log10(np.abs(energy_gc-Ekin)/np.abs(energy_gc[0]))
            mu_gc_error = np.log10(np.abs(mu_gc-muInitial)/np.abs(mu_gc[0]))
            p_gc_error = np.log10(np.abs(p_gc-pInitial)/np.abs(p_gc[0]))
            max_energy_gc_error = np.append(max_energy_gc_error, max(energy_gc_error[3::]))
            max_mu_gc_error = np.append(max_mu_gc_error, max(mu_gc_error[3::]))
            max_p_gc_error = np.append(max_p_gc_error, max(p_gc_error[3::]))
        assert max(max_energy_gc_error) < -8
        assert max(max_mu_gc_error) < -8
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
        G_inits = bsh.G()[:, 0]
        I_inits = bsh.I()[:, 0]
        mu_inits = (Ekin/m - 0.5*vpar_inits[:, 0]**2)/modB_inits
        psip_inits = bsh.psip()[:, 0]
        psi_inits = bsh.psi0*stz_inits[:, 0]
        p_inits = vpar_inits[:, 0]*(G_inits + I_inits)/modB_inits + q*(psi_inits - psip_inits)/m

        gc_tys, gc_phi_hits = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                     tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc_noK',
                                                     stopping_criteria=[MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99), ToroidalTransitStoppingCriterion(100, True)],
                                                     tol=1e-12)

        max_energy_gc_error = np.array([])
        max_mu_gc_error = np.array([])
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
            mu_gc = np.array([])
            p_gc = np.array([])
            vParInitial = gc_ty[0, 4]
            muInitial = mu_inits[i]
            pInitial = p_inits[i]
            for j in range(N):
                v_gc = gc_ty[idxs[j], 4]
                energy_gc = np.append(energy_gc, m*(0.5*v_gc**2 + muInitial*AbsBs_gc[j]))
                mu_gc = np.append(mu_gc, Ekin/(m*AbsBs_gc[j]) - 0.5*v_gc**2/AbsBs_gc[j])
                p_gc = np.append(p_gc, v_gc*(G_gc[j]+I_gc[j])/AbsBs_gc[j] + q*(psi[j] - psip[j])/m)
            energy_gc_error = np.log10(np.abs(energy_gc-Ekin)/np.abs(energy_gc[0]))
            mu_gc_error = np.log10(np.abs(mu_gc-muInitial)/np.abs(mu_gc[0]))
            p_gc_error = np.log10(np.abs(p_gc-pInitial)/np.abs(p_gc[0]))
            max_energy_gc_error = np.append(max_energy_gc_error, max(energy_gc_error[3::]))
            max_mu_gc_error = np.append(max_mu_gc_error, max(mu_gc_error[3::]))
            max_p_gc_error = np.append(max_p_gc_error, max(p_gc_error[3::]))
        assert max(max_energy_gc_error) < -8
        assert max(max_mu_gc_error) < -8
        assert max(max_p_gc_error) < -8

        # Now perform same tests for QH field with G, I, and K terms added

        bsh.set_K1(0.6)

        bsh.set_points(stz_inits)
        modB_inits = bsh.modB()[:, 0]
        G_inits = bsh.G()[:, 0]
        I_inits = bsh.I()[:, 0]
        mu_inits = (Ekin/m - 0.5*vpar_inits[:, 0]**2)/modB_inits
        psip_inits = bsh.psip()[:, 0]
        psi_inits = bsh.psi0*stz_inits[:, 0]
        p_inits = vpar_inits[:, 0]*(G_inits + I_inits)/modB_inits + q*(psi_inits - psip_inits)/m

        gc_tys, gc_phi_hits = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                     tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc',
                                                     stopping_criteria=[MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99), ToroidalTransitStoppingCriterion(100, True)],
                                                     tol=1e-12)

        max_energy_gc_error = np.array([])
        max_mu_gc_error = np.array([])
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
            mu_gc = np.array([])
            p_gc = np.array([])
            vParInitial = gc_ty[0, 4]
            muInitial = mu_inits[i]
            pInitial = p_inits[i]
            for j in range(N):
                v_gc = gc_ty[idxs[j], 4]
                energy_gc = np.append(energy_gc, m*(0.5*v_gc**2 + muInitial*AbsBs_gc[j]))
                mu_gc = np.append(mu_gc, Ekin/(m*AbsBs_gc[j]) - 0.5*v_gc**2/AbsBs_gc[j])
                p_gc = np.append(p_gc, v_gc*(G_gc[j]+I_gc[j])/AbsBs_gc[j] + q*(psi[j] - psip[j])/m)
            energy_gc_error = np.log10(np.abs(energy_gc-Ekin)/np.abs(energy_gc[0]))
            mu_gc_error = np.log10(np.abs(mu_gc-muInitial)/np.abs(mu_gc[0]))
            p_gc_error = np.log10(np.abs(p_gc-pInitial)/np.abs(p_gc[0]))
            max_energy_gc_error = np.append(max_energy_gc_error, max(energy_gc_error[3::]))
            max_mu_gc_error = np.append(max_mu_gc_error, max(mu_gc_error[3::]))
            max_p_gc_error = np.append(max_p_gc_error, max(p_gc_error[3::]))
        assert max(max_energy_gc_error) < -8
        assert max(max_mu_gc_error) < -8
        assert max(max_p_gc_error) < -8

        # Now trace with forget_exact_path = False. Check that gc_phi_hits is the same
        gc_tys, gc_phi_hits_2 = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                       tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc_noK',
                                                       stopping_criteria=[MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99), ToroidalTransitStoppingCriterion(100, True)],
                                                       tol=1e-12, forget_exact_path=True)

        for i in range(len(gc_phi_hits_2)):
            assert np.allclose(gc_phi_hits[i], gc_phi_hits_2[i])

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

        gc_tys, gc_phi_hits = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                     tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc_vac',
                                                     stopping_criteria=[MinToroidalFluxStoppingCriterion(.01), MaxToroidalFluxStoppingCriterion(0.99), ToroidalTransitStoppingCriterion(1, True)],
                                                     tol=1e-12)

        mpol = compute_poloidal_transits(gc_tys)
        ntor = compute_toroidal_transits(gc_tys)

        assert mpol == 1
        assert ntor == 1

        # Now test for ToroidalField + PoloidalField
        R0 = 1
        B0 = 1
        q = 1
        bsh = ToroidalField(R0, B0)+PoloidalField(R0, B0, q)
        nquadrature = 300
        nfourier = 1
        nfp = 1
        stellsym = True
        ma = CurveRZFourier(nquadrature, nfourier, nfp, stellsym)
        x0 = np.zeros(ma.dof_size)
        x0[0] = R0
        ma.set_dofs(x0)

        xyz_inits = np.random.uniform(size=(1, 3))
        xyz_inits[:, 0] = xyz_inits[:, 0]+R0
        xyz_inits[:, 1] = 0.
        xyz_inits[:, 2] = 0.

        gc_tys, gc_phi_hits = trace_particles(bsh, xyz_inits, vpar_inits,
                                              tmax=tmax, mass=m, charge=q, Ekin=Ekin, phis=[], mode='gc_vac',
                                              stopping_criteria=[ToroidalTransitStoppingCriterion(1, False)],
                                              tol=1e-12)

        mpol = compute_poloidal_transits(gc_tys, ma=ma, flux=False)
        ntor = compute_toroidal_transits(gc_tys, flux=False)

        assert mpol == 1
        assert ntor == 1

    def test_toroidal_flux_stopping_criterion(self):
        """
        Trace particles in a BoozerAnalyticField at high energy so that
        particle orbit widths are large. Ensure that trajectories remain
        within MinToroidalFlux and MaxToroidalFlux.
        """
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
        Ekin = 100000.*ONE_EV
        vpar = np.sqrt(2*Ekin/m)

        Nparticles = 100
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

        gc_tys, gc_phi_hits = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                     tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[], mode='gc_vac',
                                                     stopping_criteria=[MinToroidalFluxStoppingCriterion(0.4), MaxToroidalFluxStoppingCriterion(0.6)],
                                                     tol=1e-12)

        for i in range(Nparticles):
            assert np.all(gc_tys[i][:, 1] > 0.4)
            assert np.all(gc_tys[i][:, 1] < 0.6)

    def test_compute_resonances(self):
        """
        Compute particle resonances for low energy particles in a BoozerAnalytic
        field with zero magnetic shear. Check that helicity of resonances
        matches the rotational transform. Perform same test for
        a ToroidalField + PoloidalField.
        """
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

        gc_tys, gc_phi_hits = trace_particles_boozer(bsh, stz_inits, vpar_inits,
                                                     tmax=tmax, mass=m, charge=q, Ekin=Ekin, zetas=[0], mode='gc_vac',
                                                     stopping_criteria=[MinToroidalFluxStoppingCriterion(0.01), MaxToroidalFluxStoppingCriterion(0.99), ToroidalTransitStoppingCriterion(100, True)],
                                                     tol=1e-8)

        resonances = compute_resonances(gc_tys, gc_phi_hits, delta=0.01)

        iota_min = iota0-0.01
        iota_max = iota0+0.01
        for i in range(len(resonances)):
            h = resonances[i][5]/resonances[i][6]
            assert h > iota_min
            assert h < iota_max

        R0 = 1
        B0 = 1
        q = 1
        bsh = ToroidalField(R0, B0)+PoloidalField(R0, B0, q)
        nquadrature = 300
        nfourier = 1
        nfp = 1
        stellsym = True
        ma = CurveRZFourier(nquadrature, nfourier, nfp, stellsym)
        x0 = np.zeros(ma.dof_size)
        x0[0] = R0
        ma.set_dofs(x0)

        xyz_inits = np.random.uniform(size=(Nparticles, 3))
        xyz_inits[:, 0] = 0.01*xyz_inits[:, 0]+R0
        xyz_inits[:, 1] = 0.
        xyz_inits[:, 2] = 0.

        gc_tys, gc_phi_hits = trace_particles(bsh, xyz_inits, vpar_inits,
                                              tmax=tmax, mass=m, charge=q, Ekin=Ekin, phis=[0], mode='gc_vac',
                                              stopping_criteria=[ToroidalTransitStoppingCriterion(10, False)],
                                              tol=1e-12)

        resonances = compute_resonances(gc_tys, gc_phi_hits, delta=0.01, ma=ma)

        for i in range(len(resonances)):
            h = resonances[i][5]/resonances[i][6]
            assert h == 1
