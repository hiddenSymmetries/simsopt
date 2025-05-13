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
    import spec
except ImportError:
    spec = None

from simsopt.objectives import LeastSquaresProblem

if MPI is not None:
    from simsopt.mhd import Boozer, Quasisymmetry, Vmec, Spec, Residue
    from simsopt.solve import least_squares_mpi_solve
    from simsopt.util.mpi import MpiPartition

from . import TEST_DIR

#logging.basicConfig(level=logging.DEBUG)

#@unittest.skip("This test won't work until a low-level issue with VMEC is fixed to allow multiple readins.")


@unittest.skipIf((MPI is None), "Valid Python interface to VMEC not found")
class IntegratedTests(unittest.TestCase):
    @unittest.skipIf((vmec is None), "VMEC not found")
    def test_stellopt_scenarios_1DOF_circularCrossSection_varyR0_targetVolume(self):
        """
        This script implements the "1DOF_circularCrossSection_varyR0_targetVolume"
        example from
        https://github.com/landreman/stellopt_scenarios

        This optimization problem has one independent variable, representing
        the mean major radius. The problem also has one objective: the plasma
        volume. There is not actually any need to run an equilibrium code like
        VMEC since the objective function can be computed directly from the
        boundary shape. But this problem is a fast way to test the
        optimization infrastructure with VMEC.

        Details of the optimum and a plot of the objective function landscape
        can be found here:
        https://github.com/landreman/stellopt_scenarios/tree/master/1DOF_circularCrossSection_varyR0_targetVolume
        """

        # logging.basicConfig(level=logging.DEBUG)
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

                # Each Target is then equipped with a shift and weight, to become a
                # term in a least-squares objective function
                desired_volume = 0.15
                prob = LeastSquaresProblem.from_tuples(
                    [(equil.volume, desired_volume, 1)])

                # Solve the minimization problem. We can choose whether to use a
                # derivative-free or derivative-based algorithm.
                least_squares_mpi_solve(prob, mpi=mpi, grad=grad)

                # Make sure all procs call VMEC:
                objective = prob.objective()
                if mpi.proc0_world:
                    print("At the optimum,")
                    print(" rc(m=0,n=0) = ", surf.get_rc(0, 0))
                    print(" volume, according to VMEC    = ", equil.volume())
                    print(" volume, according to Surface = ", surf.volume())
                    print(" objective function = ", objective)

                assert np.abs(surf.get_rc(0, 0) - 0.7599088773175) < 1.0e-5
                assert np.abs(equil.volume() - 0.15) < 1.0e-6
                assert np.abs(surf.volume() - 0.15) < 1.0e-6
                assert prob.objective() < 1.0e-15

    @unittest.skipIf((vmec is None), "VMEC not found")
    def test_Quasisymmetry_paralellization(self):
        """
        this test checks if the Quasisymmetry objective evaluation is correctly 
        implemented to run in MPI tasks with different numbers of leaders and
        workers. 
        """
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

    @unittest.skipIf((spec is None), "SPEC not found")
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
                least_squares_mpi_solve(prob, mpi=mpi, grad=grad, save_residuals=True, max_nfev=2)

                # No assertions, run is too short to complete, just testing if it does run


if __name__ == "__main__":
    unittest.main()
