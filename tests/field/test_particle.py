from simsopt.geo.coilcollection import CoilCollection
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.util.zoo import get_ncsx_data
from simsopt.field.tracing import trace_particles_starting_on_curve, SurfaceClassifier, \
    particles_to_vtk, LevelsetStoppingCriterion, compute_gc_radius, gc_to_fullorbit_initial_guesses, \
    IterationStoppingCriterion, trace_particles_starting_on_surface
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule, ToroidalField, PoloidalField
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
import simsoptpp as sopp
import numpy as np
import unittest
import logging
logging.basicConfig()
try:
    import pyevtk
    with_evtk = True
except ImportError:
    with_evtk = False


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
        coils, currents, ma = get_ncsx_data(Nt_coils=8)
        currents = [3 * c for c in currents]
        stellarator = CoilCollection(coils, currents, 3, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)
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
        assert sc.evaluate(ma.gamma()[:1, :]) > 0
        assert sc.evaluate(2*ma.gamma()[:1, :]) < 0
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
        assert np.all(sc.evaluate(gc_tys[0][:, 1:4]) > 0)

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
