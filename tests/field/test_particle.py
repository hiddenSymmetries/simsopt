from simsopt.geo.coilcollection import CoilCollection
from simsopt.field.biotsavart import BiotSavart
from simsopt.util.zoo import get_ncsx_data
from simsopt.field.tracing import trace_particles_starting_on_axis, SurfaceClassifier, \
    particles_to_vtk, LevelsetStoppingCriterion, compute_gc_radius
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
import numpy as np
import unittest
import logging
logging.basicConfig()


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
        n = 10
        rrange = (1.1, 1.8, n)
        phirange = (0, 2*np.pi, n*6)
        zrange = (-0.3, 0.3, n)
        bsh = InterpolatedField(
            bs, UniformInterpolationRule(4),
            rrange, phirange, zrange, True
        )
        self.bsh = bsh
        self.ma = ma
        bsh.to_vtk('/tmp/bfield')

    def test_guidingcenter_vs_fullorbit(self):
        bsh = self.bsh
        ma = self.ma
        nparticles = 2
        m = 1.67e-27
        np.random.seed(1)
        with self.assertRaises(RuntimeError):
            gc_tys, gc_phi_hits = trace_particles_starting_on_axis(
                ma.gamma(), bsh, nparticles, tmax=1e-4, seed=1, mass=m, charge=1,
                Ekin=9000, umin=-0.1, umax=+0.1,
                phis=[], mode='gc')

        gc_tys, gc_phi_hits = trace_particles_starting_on_axis(
            ma.gamma(), bsh, nparticles, tmax=1e-4, seed=1, mass=m, charge=1,
            Ekin=9000, umin=-0.1, umax=+0.1,
            phis=[], mode='gc_vac')
        fo_tys, fo_phi_hits = trace_particles_starting_on_axis(
            ma.gamma(), bsh, nparticles, tmax=1e-4, seed=1, mass=m, charge=1,
            Ekin=9000, umin=-0.1, umax=+0.1,
            phis=[], mode='full')
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
                r = compute_gc_radius(m, vperp, 1.6e-19, AbsBs[j, 0])
                dist = np.linalg.norm(gc_xyzs[j, :] - fo_ty[jdx, 1:4])
                assert dist < 4*r
                assert dist > 0.5*r

    def test_guidingcenterphihits(self):
        bsh = self.bsh
        ma = self.ma
        nparticles = 2
        m = 1.67e-27
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
        sc.to_vtk('/tmp/classifier')
        # check that the axis is classified as inside the domain
        assert sc.evaluate(ma.gamma()[:1, :]) > 0
        assert sc.evaluate(2*ma.gamma()[:1, :]) < 0
        np.random.seed(1)
        gc_tys, gc_phi_hits = trace_particles_starting_on_axis(
            ma.gamma(), bsh, nparticles, tmax=1e-4, seed=1, mass=m, charge=1,
            Ekin=9000, umin=-0.1, umax=+0.1, phis=phis, mode='gc_vac',
            stopping_criteria=[LevelsetStoppingCriterion(sc.dist)])

        particles_to_vtk(gc_tys, '/tmp/particles_gc')
        for i in range(nparticles):
            assert validate_phi_hits(gc_phi_hits[i], bsh, nphis)

