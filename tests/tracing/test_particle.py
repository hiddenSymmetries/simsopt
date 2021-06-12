from simsopt.geo.coilcollection import CoilCollection
from simsopt.geo.biotsavart import BiotSavart
from simsopt.util.zoo import get_ncsx_data
from simsopt.tracing.tracing import trace_particles_starting_on_axis, SurfaceClassifier, \
    particles_to_vtk, LevelsetStoppingCriterion, compute_gc_radius
from simsopt.geo.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
import numpy as np
import unittest
import logging
logging.basicConfig()


class FieldlineTesting(unittest.TestCase):

    def test_poincare(self):
        logger = logging.getLogger('simsopt.tracing.tracing')
        logger.setLevel(1)
        # Test a toroidal magnetic field with no rotational transform
        coils, currents, ma = get_ncsx_data()
        currents = [3 * c for c in currents]
        stellarator = CoilCollection(coils, currents, 3, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        n = 16
        rrange = (1.0, 2.0, n)
        phirange = (0, 2*np.pi, n*6)
        zrange = (-0.7, 0.7, n)
        bsh = InterpolatedField(
            bs, UniformInterpolationRule(3),
            rrange, phirange, zrange, True
        )
        nparticles = 2
        m = 1.67e-27
        nphis = 4
        phis = np.linspace(0, 2*np.pi, nphis, endpoint=False)
        gc_tys, gc_phi_hits = trace_particles_starting_on_axis(
            ma.gamma(), bsh, nparticles, tmax=1e-4, seed=1, mass=m, charge=1,
            Ekinev=9000, umin=-0.1, umax=+0.1,
            phis=phis, mode='gc')
        # stopping_criteria=[LevelsetStoppingCriterion(sc.dist)], mode=mode)
        fo_tys, fo_phi_hits = trace_particles_starting_on_axis(
            ma.gamma(), bsh, nparticles, tmax=1e-4, seed=1, mass=m, charge=1,
            Ekinev=9000, umin=-0.1, umax=+0.1,
            phis=phis, mode='full')
        # stopping_criteria=[LevelsetStoppingCriterion(sc.dist)], mode=mode)
        particles_to_vtk(gc_tys, '/tmp/particles_gc')
        particles_to_vtk(fo_tys, '/tmp/particles_fo')

        # pick 100 random points on each trace, and ensure that the guiding
        # center and the full orbit simulation are close to each other

        np.random.seed(1)

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
