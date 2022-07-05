import unittest
import logging
logging.basicConfig()

import numpy as np
try:
    from mpi4py import MPI
    with_mpi = True
except ImportError:
    with_mpi = False

from simsopt.field.coil import coils_via_symmetries
from simsopt.field.biotsavart import BiotSavart
from simsopt.configs.zoo import get_ncsx_data
from simsopt.field.tracing import trace_particles_starting_on_curve, compute_fieldlines
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
import simsoptpp as sopp


class MPITracingTesting(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MPITracingTesting, self).__init__(*args, **kwargs)
        logger = logging.getLogger('simsopt.field.tracing')
        logger.setLevel(1)
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        n = 10
        rrange = (1.1, 1.8, n)
        phirange = (0, 2*np.pi/nfp, n*2)
        zrange = (0, 0.3, n//2)
        bsh = InterpolatedField(
            bs, UniformInterpolationRule(3),
            rrange, phirange, zrange, True, nfp=3, stellsym=True
        )
        bsh.estimate_error_GradAbsB(1000)
        bsh.estimate_error_B(1000)
        # trick to clear the cache in the biot savart class to reduce memory usage
        bs.set_points(np.asarray([[0., 0., 0.]])).GradAbsB()
        self.bsh = bsh
        self.ma = ma

    @unittest.skipIf(not with_mpi, "mpi not found")
    def test_parallel_fieldline(self):
        nlines = 4
        r0 = np.linalg.norm(self.ma.gamma()[0, :2])
        z0 = self.ma.gamma()[0, 2]
        R0 = [r0 + i*0.01 for i in range(nlines)]
        Z0 = [z0 for i in range(nlines)]
        phis = []
        comm = MPI.COMM_WORLD
        print('comm.size', comm.size)
        res_tys_mpi, res_phi_hits_mpi = compute_fieldlines(
            self.bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[], comm=comm)

        res_tys, res_phi_hits = compute_fieldlines(
            self.bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[], comm=None)
        for i in range(nlines):
            assert np.allclose(res_phi_hits_mpi[i], res_phi_hits[i], atol=1e-9, rtol=1e-9)

    @unittest.skipIf(not with_mpi, "mpi not found")
    def test_parallel_guiding_center(self):
        nparticles = 4
        m = PROTON_MASS
        q = ELEMENTARY_CHARGE
        Ekin = 1000*ONE_EV
        umin = 0.25
        umax = 0.75
        tmax = 1e-5
        np.set_printoptions(precision=20)
        comm = MPI.COMM_WORLD

        print('comm.size', comm.size)
        gc_tys_mpi, gc_phi_hits_mpi = trace_particles_starting_on_curve(
            self.ma, self.bsh, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
            Ekin=Ekin, umin=umin, umax=umax,
            phis=[], mode='gc_vac', comm=comm)

        gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
            self.ma, self.bsh, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
            Ekin=Ekin, umin=umin, umax=umax,
            phis=[], mode='gc_vac', comm=None)
        for i in range(nparticles):
            assert np.allclose(gc_phi_hits_mpi[i], gc_phi_hits[i], atol=1e-9, rtol=1e-9)
