import unittest
import logging

import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
try:
    import vmec
except ImportError:
    vmec = None

from simsopt.objectives.least_squares import LeastSquaresProblem

if MPI is not None:
    from simsopt.mhd.vmec import Vmec
    from simsopt.solve.mpi import least_squares_mpi_solve
    from simsopt.util.mpi import MpiPartition

#logging.basicConfig(level=logging.DEBUG)

#@unittest.skip("This test won't work until a low-level issue with VMEC is fixed to allow multiple readins.")


@unittest.skipIf((MPI is None) or (vmec is None), "Valid Python interface to VMEC not found")
class IntegratedTests(unittest.TestCase):
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
        logger = logging.getLogger(__name__)

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


if __name__ == "__main__":
    unittest.main()
