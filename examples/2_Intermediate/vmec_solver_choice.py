#!/usr/bin/env python

"""
Run the same fixed-boundary equilibrium with each available VMEC backend.

:obj:`~simsopt.mhd.vmec.Vmec` runs VMEC2000 unless it is given another
solver. Pass an instance of any solver that satisfies
:obj:`~simsopt.mhd.vmec.VmecSolverProtocol` as ``solver``, e.g. a
:obj:`~simsopt.mhd.vmecpp_solver.VmecppSolver`. The rest of the script
does not change.
"""

from pathlib import Path

from simsopt.mhd import Vmec

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
input_file = str(TEST_DIR / "input.li383_low_res")

solvers = {}
try:
    from simsopt.mhd.vmec_solver import Vmec2000Solver
    import vmec  # noqa: F401
    solvers["VMEC2000"] = Vmec2000Solver()
except ImportError:
    print("VMEC2000 is not installed, skipping it.")
try:
    from simsopt.mhd.vmecpp_solver import VmecppSolver
    solvers["VMEC++"] = VmecppSolver(max_threads=1)
except ImportError:
    print("vmecpp is not installed, skipping it.")

for name, solver in solvers.items():
    vmec = Vmec(input_file, solver=solver, verbose=False)
    print(f"{name}: aspect ratio = {vmec.aspect():.6f}, "
          f"volume = {vmec.volume():.6f}")
