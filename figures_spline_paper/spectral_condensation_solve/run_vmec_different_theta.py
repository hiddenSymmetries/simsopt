#!/usr/bin/env python
"""
Runs VMEC (to ftol=1e-12) for the two theta parametrizations compared by
vmec_solves_different_theta.py:

  - "exact": the spline's raw u parameter used directly as theta
    (SurfaceBSpline.ft's collocation="exact" -- zero interpolation error,
    but not a physical polar angle), no spectral condensation.
  - "cond": the same collocation, with variational spectral condensation
    (SurfaceBSpline.to_RZFourier's spec_cond="variational") applied on top.

Uses the same SurfaceBSpline (spline_kwargs + dofs) as the equilibrium
currently plotted in examples/2_Intermediate/spline_surface_plot.py. Run
this script from this directory (spline_helpers file naming below is
relative to the current working directory) before running
vmec_solves_different_theta.py.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from simsopt._core.util import ObjectiveFailure
from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.util.mpi import MpiPartition

# ngroups=1 pools every launched MPI rank into a single group, so a single
# VMEC solve gets all of them (VMEC2000/PARVMEC parallelizes internally
# across the radial (ns) grid using whatever communicator it's given --
# MpiPartition()'s default, one process per group, gives it nothing to
# parallelize with). Run this script as `mpiexec -n <K> python
# run_vmec_different_theta.py` to actually use K processes per solve; the
# two cases below still run one after another, not concurrently, so there's
# no benefit to splitting ranks across multiple groups instead.
mpi = MpiPartition(ngroups=1)
mpi.write()


def report_best_residual(label):
    """
    Print the best force residual VMEC reached when it ran out of
    iterations without hitting ftol (runvmec's ierr=more_iter_flag=2,
    which Vmec.run() turns into ObjectiveFailure). This failure mode
    does NOT produce a wout_*.nc file at all -- unlike a converged run,
    vmec.wout is never populated and there's nothing on disk under that
    name to load. The only place the achieved residuals survive is
    fort.9, the plain-text per-iteration force-residual log VMEC always
    writes (and which Vmec.run() only deletes on a *successful* run) --
    parsed here for its last data row. fort.9 is a fixed filename
    (not tied to input_basename) that gets overwritten by the next
    run, so this must be called immediately after the failure, before
    the next case runs.
    """
    try:
        with open("fort.9") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(
            f"{label}: did not converge, and fort.9 was not found to "
            "report the best residual reached"
        )
        return

    last_row = None
    for line in lines:
        fields = line.split()
        if len(fields) >= 4:
            try:
                int(fields[0])
            except ValueError:
                continue
            last_row = fields

    if last_row is None:
        print(
            f"{label}: did not converge, and no iteration data was "
            "found in fort.9"
        )
        return

    iteration, fsqr, fsqz, fsql = last_row[:4]
    print(
        f"{label}: did not converge -- best residual reached at "
        f"iteration {iteration}: FSQR={fsqr} FSQZ={fsqz} FSQL={fsql}"
    )


if __name__ == "__main__":
    # Identical to examples/2_Intermediate/spline_surface_plot.py.
    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 4,
        "n_cs": 6,
        "nfp": 3,
        "M": 12,
        "N": 12,
        "p_u": 3,
        "p_v": 3,
        "cs_equispaced": True,
        "rays_equispaced": False,
        "cs_global_angle_free": False,
        "axis_angles_fixed": True,
        "cs_basis": "polar",
        "nurbs": False,
        "use_bishop_frame": True,
        "knot_parametrization": "uniform",
    }

    spline_surf = SurfaceBSpline(default_r=0.3, **spline_kwargs)
    # spline_surf.axis.fix("r_axis_0")

    new_x = np.array(
        [
            5.3896343573543060e-02,
            5.0883230048103856e-01,
            2.2489121027379422e-01,
            1.5929768581422210e00,
            1.0304048534365813e-01,
            4.3412658709604501e-01,
            1.9194536849729077e-01,
            5.3647438961678784e-01,
            1.7794171743465446e00,
            3.6038044630128279e00,
            4.9259924741397159e00,
            2.5036189767220018e-01,
            3.5802891724900809e-01,
            9.6212580154516142e-02,
            5.2064914206501489e-01,
            2.0474558259649123e00,
            3.8964954930212143e00,
            5.3228745694944024e00,
            3.2185070391451231e-01,
            2.5146308466749123e-01,
            2.1954123698502295e-01,
            4.5420692619007164e-01,
            2.2214282741552713e00,
            3.0009540172725946e00,
            5.4975172012762998e00,
            3.6991589429711308e-01,
            1.6285912057838300e-01,
            3.6446964917388275e-01,
            3.2242251631177343e-01,
            1.7920172479908414e00,
            3.0210349920179866e00,
            5.4977786325813431e00,
            3.8958534284832108e-01,
            1.9630111359305188e-01,
            4.0847103311261590e-01,
            1.0489177365984679e00,
            2.0043666242346596e00,
            1.3580534688161376e00,
            5.0279307449500388e-01,
            -7.0064725504961656e-01,
        ]
    )
    spline_surf.x = new_x
    # seed(888)
    # random_dofs = uniform(spline_surf.lower_bounds, spline_surf.upper_bounds)
    # spline_surf.x = random_dofs

    rz_surf = spline_surf.to_RZFourier(
        collocation="exact",
        spec_cond=None,
    )
    shapetol = rz_surf.minor_radius() / 100
    print(f"shapetol: {shapetol}")

    if MPI.COMM_WORLD.rank == 0:
        spline_surf.plot()
        plt.show()

    def run_case(spec_cond, spec_cond_options, input_basename):
        rz_surf = spline_surf.to_RZFourier(
            collocation="exact",
            nu=64,
            nv=64,
            spec_cond=spec_cond,
            spec_cond_options=spec_cond_options,
        )
        vmec = Vmec.vmec_from_surf(
            nfp=rz_surf.nfp,
            surf=rz_surf,
            mpi=mpi,
            ns=100,
            M=12,
            N=12,
            ftol=1e-11,
            verbose=True,
            niter=8000,
            ntheta=64,
            nzeta=64,
            delt=9e-1,
        )
        vmec.input_file = input_basename
        try:
            vmec.run()
            print(f"{input_basename}: converged, wrote {vmec.output_file}")
        except ObjectiveFailure:
            report_best_residual(input_basename)

    run_case(
        spec_cond="variational",
        spec_cond_options={
            "plot": True,
            "ftol": 1e-6,
            "Mtol": 1.1,
            "shapetol": shapetol,
            "niters": 5000,
            "verbose": True,
            "cutoff": 1e-6,
        },
        input_basename="input.cond",
    )
    if MPI.COMM_WORLD.rank == 0:
        plt.show()

    run_case(
        spec_cond=None, spec_cond_options=None, input_basename="input.exact"
    )
