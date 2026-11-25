from pathlib import Path

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()


def Vmec2000(*args, **kwargs):
    """ :obj:`~simsopt.mhd.vmec.Vmec` that runs VMEC2000 unless told otherwise. """
    from simsopt.mhd.vmec import Vmec
    from simsopt.mhd.vmec_solver import Vmec2000Solver

    if "solver" not in kwargs:
        kwargs["solver"] = Vmec2000Solver()
    return Vmec(*args, **kwargs)
