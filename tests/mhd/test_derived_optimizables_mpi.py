import unittest
import logging
import os

import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
try:
    import vmec
except ImportError:
    vmec = None
try:
    from simsopt.mhd import Spec
except ImportError:
    Spec = None

from simsopt.objectives.least_squares import LeastSquaresProblem

if MPI is not None:
    from simsopt.mhd import Boozer, Quasisymmetry
    from simsopt.mhd import Vmec
    from simsopt.mhd import Spec, Residue
    from simsopt.solve.mpi import least_squares_mpi_solve
    from simsopt.util.mpi import MpiPartition

from . import TEST_DIR

#logging.basicConfig(level=logging.DEBUG)

#@unittest.skip("This test won't work until a low-level issue with VMEC is fixed to allow multiple readins.")


@unittest.skipIf((MPI is None) or (vmec is None) or (Spec is None), "VMEC or Spec not found")
class IntegratedTests(unittest.TestCase):
    def test_Quasisymmetry_paralellization(self):
        """
        this test checks if the Quasisymmetry objective evaluation is correctly 
        implemented to run in MPI tasks with different numbers of leaders and
        workers. 
        """
        # logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)
        logging.getLogger(__name__)
        for ngroups in range(1, 1 + MPI.COMM_WORLD.Get_size()):
            for grad in [False, True]:
                # In the next line, we can adjust how many groups the pool of MPI
                # processes is split into.
                mpi = MpiPartition(ngroups=ngroups)
                mpi.write()
                
                # Start with a default surface, which is axisymmetric with major
                # radius 1 and minor radius 0.1.
                equil = Vmec(mpi=mpi)
                surf = equil.boundary

                # Set the initial boundary shape. Here is one syntax:
                surf.set('rc(0,0)', 1.0)
                # Here is another syntax:
                surf.set_rc(0, 1, 0.1)
                surf.set_zs(0, 1, 0.1)

                surf.set_rc(1, 0, 0.1)
                surf.set_zs(1, 0, 0.1)

                # VMEC parameters are all fixed by default, while surface parameters are all non-fixed by default.
                # You can choose which parameters are optimized by setting their 'fixed' attributes.
                surf.local_fix_all()
                surf.unfix('rc(0,0)')
                surf.unfix('rc(1,1)')
                surf.unfix('rc(0,1)')

                qs = Quasisymmetry(Boozer(equil),
                                   0.5,  # Radius to target
                                   1, 1)  # (M, N) you want in |B|

                prob = LeastSquaresProblem.from_tuples([(equil.aspect, 7, 1),
                                                        (qs.J, 0, 1)])
                # Make sure all procs run vmec:
                least_squares_mpi_solve(prob, mpi, grad=grad, max_nfev=2)
                ## we just want to test if the problem runs for the two fevs

    def test_Residue_parallelization(self):
        """
        This test checks if the Residue objective evaluation is correctly implemented
        to run on in MPI tasks with different numbers of leaders and workers
        """
        logging.getLogger(__name__)
        for ngroups in range(1, 1 + MPI.COMM_WORLD.Get_size()):
            for grad in [False, True]:
                # In the next line, we can adjust how many groups the pool of MPI
                # processes is split into.
                mpi = MpiPartition(ngroups=ngroups)
                mpi.write()
                filename = os.path.join(TEST_DIR, 'QH-residues.sp')
                s = Spec(filename, mpi=mpi)

                # Expand number of Fourier modes to include larger poloidal mode numbers:
                s.boundary.change_resolution(6, s.boundary.ntor)
                # To make this example run relatively quickly, we will optimize in a
                # small parameter space. Here we pick out just 2 Fourier modes to vary
                # in the optimization:
                s.boundary.fix_all()
                s.boundary.unfix('zs(6,1)')
                s.boundary.unfix('zs(6,2)')
                logging.info(f"Initial zs(6,1):  {s.boundary.get('zs(6,1)')}")
                logging.info(f"Initial zs(6,2):  {s.boundary.get('zs(6,2)')}")

                # The main resonant surface is iota = p / q:
                p = -8
                q = 7
                # Guess for radial location of the island chain:
                s_guess = 0.9

                residue1 = Residue(s, p, q, s_guess=s_guess)
                residue2 = Residue(s, p, q, s_guess=s_guess, theta=np.pi)
                # Objective function is \sum_j residue_j ** 2
                prob = LeastSquaresProblem.from_tuples([(residue1.J, 0, 1),
                                                        (residue2.J, 0, 1)])
               
                # Solve for two nfevs to test if runs
                least_squares_mpi_solve(prob, mpi=mpi, grad=grad, max_nfev=2)
                
                # No assertions, run is to short to complete, just testing if it does run

                
