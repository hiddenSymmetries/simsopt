"""
This module contains the a number of useful functions for using
the permanent magnets functionality in the SIMSOPT code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simsopt.geo import CurveXYZFourier
    from simsopt.geo.current_voxels_grid import CurrentVoxelsGrid

__all__ = [
    "read_focus_coils",
    "coil_optimization",
    "trace_fieldlines",
    "make_qfm",
    "vacuum_stage_II_optimization",
    "calculate_modB_on_major_radius",
    "initial_vacuum_stage_II_optimizations",
    "continuation_vacuum_stage_II_optimizations",
    "make_stage_II_pareto_plots",
    "build_stage_II_data_array",
    "make_filament_from_voxels",
    "make_curve_at_theta0",
]

import numpy as np
from scipy.optimize import minimize
from pathlib import Path

# Necessary imports for the pareto scans
import matplotlib.pyplot as plt
import glob
import json
import os
import time


def read_focus_coils(filename):
    """
    Reads in the coils from a FOCUS file. For instance, this is
    used for loading in the MUSE phased TF coils.

    Args:
        filename: String denoting the name of the coils file.

    Returns:
        coils: List of CurveXYZFourier class objects.
        base_currents: List of Current class objects.
        ncoils: Integer representing the number of coils.
    """
    from simsopt.geo import CurveXYZFourier
    from simsopt.field import Current

    ncoils = np.loadtxt(filename, skiprows=1, max_rows=1, dtype=int)
    order = int(np.loadtxt(filename, skiprows=8, max_rows=1, dtype=int))
    coilcurrents = np.zeros(ncoils)
    xc = np.zeros((ncoils, order + 1))
    xs = np.zeros((ncoils, order + 1))
    yc = np.zeros((ncoils, order + 1))
    ys = np.zeros((ncoils, order + 1))
    zc = np.zeros((ncoils, order + 1))
    zs = np.zeros((ncoils, order + 1))
    # load in coil currents and fourier representations of (x, y, z)
    for i in range(ncoils):
        coilcurrents[i] = np.loadtxt(
            filename, skiprows=6 + 14 * i, max_rows=1, usecols=1
        )
        xc[i, :] = np.loadtxt(
            filename, skiprows=10 + 14 * i, max_rows=1, usecols=range(order + 1)
        )
        xs[i, :] = np.loadtxt(
            filename, skiprows=11 + 14 * i, max_rows=1, usecols=range(order + 1)
        )
        yc[i, :] = np.loadtxt(
            filename, skiprows=12 + 14 * i, max_rows=1, usecols=range(order + 1)
        )
        ys[i, :] = np.loadtxt(
            filename, skiprows=13 + 14 * i, max_rows=1, usecols=range(order + 1)
        )
        zc[i, :] = np.loadtxt(
            filename, skiprows=14 + 14 * i, max_rows=1, usecols=range(order + 1)
        )
        zs[i, :] = np.loadtxt(
            filename, skiprows=15 + 14 * i, max_rows=1, usecols=range(order + 1)
        )

    # CurveXYZFourier wants data in order sin_x, cos_x, sin_y, cos_y, ...
    coil_data = np.zeros((order + 1, ncoils * 6))
    for i in range(ncoils):
        coil_data[:, i * 6 + 0] = xs[i, :]
        coil_data[:, i * 6 + 1] = xc[i, :]
        coil_data[:, i * 6 + 2] = ys[i, :]
        coil_data[:, i * 6 + 3] = yc[i, :]
        coil_data[:, i * 6 + 4] = zs[i, :]
        coil_data[:, i * 6 + 5] = zc[i, :]

    # Set the degrees of freedom in the coil objects
    base_currents = [Current(coilcurrents[i]) for i in range(ncoils)]
    ppp = 20
    coils = [CurveXYZFourier(order * ppp, order) for i in range(ncoils)]
    for ic in range(ncoils):
        dofs = coils[ic].dofs_matrix
        dofs[0][0] = coil_data[0, 6 * ic + 1]
        dofs[1][0] = coil_data[0, 6 * ic + 3]
        dofs[2][0] = coil_data[0, 6 * ic + 5]
        for io in range(0, min(order, coil_data.shape[0] - 1)):
            dofs[0][2 * io + 1] = coil_data[io + 1, 6 * ic + 0]
            dofs[0][2 * io + 2] = coil_data[io + 1, 6 * ic + 1]
            dofs[1][2 * io + 1] = coil_data[io + 1, 6 * ic + 2]
            dofs[1][2 * io + 2] = coil_data[io + 1, 6 * ic + 3]
            dofs[2][2 * io + 1] = coil_data[io + 1, 6 * ic + 4]
            dofs[2][2 * io + 2] = coil_data[io + 1, 6 * ic + 5]
        coils[ic].local_x = np.concatenate(dofs)
    return coils, base_currents, ncoils


def coil_optimization(s, bs, base_curves, curves, **kwargs):
    """
    Basic stellarator coil optimization wrapper function that can be reused in many 
    different scripts and different configurations. The objective function is given by:

    JF = Jf \
        + LENGTH_WEIGHT * QuadraticPenalty(sum([CurveLength(c) for c in base_curves]), LENGTH_TARGET, "max") \
        + CC_WEIGHT * CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(base_curves)) \
        + CS_WEIGHT * CurveSurfaceDistance(curves, s, CS_THRESHOLD) \
        + CURVATURE_WEIGHT * sum([LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]) \
        + MSC_WEIGHT * sum([QuadraticPenalty(MeanSquaredCurvature(c), MSC_THRESHOLD) for c in base_curves]) \
        + LINKING_NUMBER_WEIGHT * LinkingNumber(curves) \
        + FORCE_WEIGHT * sum([LpCurveForce(c, coils, regularization_circ(0.03 * R0), p=2, threshold=FORCE_THRESHOLD) for c in base_curves])

    Args:
        s (Surface): Surface representing the plasma boundary
        bs (BiotSavart): Biot Savart class object, presumably representing the
          magnetic fields generated by the coils.
        base_curves (list[Curve]): List of Curve class objects.
        curves (list[Curve]): List of Curve class objects obtained after applying symmetries to the base_curves.
        **kwargs (dict): Keyword arguments for the optimization.
            LENGTH_WEIGHT (float, default=1): Weight on the curve lengths in the objective function.
            LENGTH_THRESHOLD (float, default=18.0): Threshold for the curve lengths in the objective function.
            CC_THRESHOLD (float, default=0.1): Threshold for the coil-to-coil distance penalty in the objective function.
            CC_WEIGHT (float, default=1): Weight for the coil-to-coil distance penalty in the objective function.
            CS_THRESHOLD (float, default=1.5): Threshold for the coil-to-surface distance penalty in the objective function.
            CS_WEIGHT (float, default=1e-2): Weight for the coil-to-surface distance penalty in the objective function.
            CURVATURE_THRESHOLD (float, default=0.1): Threshold for the curvature penalty in the objective function.
            CURVATURE_WEIGHT (float, default=1e-9): Weight for the curvature penalty in the objective function.
            MSC_THRESHOLD (float, default=0.1): Threshold for the mean squared curvature penalty in the objective function.
            MSC_WEIGHT (float, default=1e-9): Weight for the mean squared curvature penalty in the objective function.
            LINKING_NUMBER_WEIGHT (float, default=0): Weight for the linking number penalty in the objective function.
            FORCE_THRESHOLD (float, default=0): Threshold for the force penalty in the objective function.
            FORCE_WEIGHT (float, default=0): Weight for the force penalty in the objective function.
            MAXITER (int, default=500): Maximum number of iterations for the optimization.

    Returns:
        bs (BiotSavart): Biot Savart class object, representing the
          OPTIMIZED magnetic fields generated by the coils.
    """

    from simsopt.geo import (
        CurveLength,
        CurveCurveDistance,
        MeanSquaredCurvature,
        LpCurveCurvature,
        CurveSurfaceDistance,
        LinkingNumber,
    )
    from simsopt.objectives import QuadraticPenalty, SquaredFlux
    from simsopt.field.force import LpCurveForce
    from simsopt.field.selffield import regularization_circ

    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    R0 = s.get_rc(0, 0)  # rescale all the thresholds by major radius of the plasma

    # Weight on the curve lengths in the objective function:
    LENGTH_WEIGHT = kwargs.get("LENGTH_WEIGHT", 1)
    LENGTH_THRESHOLD = kwargs.get("LENGTH_THRESHOLD", 18.0 * R0)

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    CC_THRESHOLD = kwargs.get("CC_THRESHOLD", 0.1 * R0)
    CC_WEIGHT = kwargs.get("CC_WEIGHT", 1)

    # Threshold and weight for the coil-to-surface distance penalty in the objective function:
    CS_THRESHOLD = kwargs.get("CS_THRESHOLD", 0.15 * R0)
    CS_WEIGHT = kwargs.get("CS_WEIGHT", 1e-2)

    # Threshold and weight for the curvature penalty in the objective function:
    CURVATURE_THRESHOLD = kwargs.get("CURVATURE_THRESHOLD", 0.1 * R0)
    CURVATURE_WEIGHT = kwargs.get("CURVATURE_WEIGHT", 1e-6)

    # Threshold and weight for the mean squared curvature penalty in the objective function:
    MSC_THRESHOLD = kwargs.get("MSC_THRESHOLD", 0.1 * R0)
    MSC_WEIGHT = kwargs.get("MSC_WEIGHT", 1e-6)

    # Linking number penalty in the objective function:
    LINKING_NUMBER_WEIGHT = kwargs.get("LINKING_NUMBER_WEIGHT", 0)

    # Force penalty in the objective function:
    FORCE_WEIGHT = kwargs.get("FORCE_WEIGHT", 0)
    FORCE_THRESHOLD = kwargs.get("FORCE_THRESHOLD", 0)
    coils = bs.coils
    base_coils = [coils[i] for i, c in enumerate(base_curves)]

    MAXITER = kwargs.get("MAXITER", 500)  # number of iterations for minimize

    # Define the objective function:
    Jf = SquaredFlux(s, bs)
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(base_curves))
    Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    linking_number = LinkingNumber(curves)
    # Hard-coded finite widths below -- 3 cm width for 1m device, ~ 30 cm width for 10m device
    Jforce = [
        LpCurveForce(
            c, coils, regularization_circ(0.03 * R0), p=2, threshold=FORCE_THRESHOLD
        )
        for c in base_coils
    ]

    # Form the total objective function.
    JF = (
        Jf
        + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), LENGTH_THRESHOLD, "max")
        + CC_WEIGHT * Jccdist
        + CS_WEIGHT * Jcsdist
        + CURVATURE_WEIGHT * sum(Jcs)
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
        + LINKING_NUMBER_WEIGHT * linking_number
        + FORCE_WEIGHT * sum(Jforce)
    )

    def fun(dofs):
        """Function for coil optimization grabbed from stage_two_optimization.py"""
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.mean(
            np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
        )
        outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", Linking Number={linking_number.J():.1f}"
        outstr += f", Force={sum(Jforce).J():.1f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        valuestr = f"J={J:.2e}, Jf={jf:.2e}"
        valuestr += f", LenObj={LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), LENGTH_THRESHOLD, 'max').J():.2e}"
        valuestr += f", ccObj={CC_WEIGHT * Jccdist.J():.2e}"
        valuestr += f", csObj={CS_WEIGHT * Jcsdist.J():.2e}"
        valuestr += f", curvatureObj={CURVATURE_WEIGHT * sum(Jcs).J():.2e}"
        valuestr += f", mscObj={MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs).J():.2e}"
        valuestr += (
            f", linkingNumberObj={LINKING_NUMBER_WEIGHT * linking_number.J():.2e}"
        )
        valuestr += f", forceObj={FORCE_WEIGHT * sum(Jforce).J():.2e}"
        print(valuestr)
        return J, grad

    dofs = JF.x
    minimize(
        fun,
        dofs,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAXITER, "maxcor": 300},
        tol=1e-15,
    )
    bs.set_points(s.gamma().reshape((-1, 3)))
    return bs


def trace_fieldlines(
    bfield, label, s, comm, out_dir="", nfieldlines=4, tmax_fl=10000, num_iters=20000
):
    """
    Make Poincare plots on a surface as in the trace_fieldlines
    example in the examples/1_Simple/ directory.

    Args:
        bfield (MagneticField or InterpolatedField): MagneticField or InterpolatedField class object.
        label (str): Name of the file to write to.
        s (Surface): plasma boundary surface.
        comm (MPI COMM_WORLD): MPI COMM_WORLD object for using MPI for tracing.
        out_dir (str): Path or string for the output directory for saved files.
        nfieldlines (int): Number of fieldlines to trace.
        tmax_fl (float): Maximum time to trace the fieldlines.
        num_iters (int): Maximum number of iterations for the fieldline tracing.
    """
    from simsopt.field.tracing import (
        compute_fieldlines,
        plot_poincare_data,
        IterationStoppingCriterion,
    )

    out_dir = Path(out_dir)

    R0 = np.linspace(s.get_rc(0, 0), s.get_rc(0, 0) + s.get_rc(1, 0) / 2.0, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i / 4) * (2 * np.pi / s.nfp) for i in range(4)]

    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield,
        R0,
        Z0,
        tmax=tmax_fl,
        tol=1e-16,
        comm=comm,
        phis=phis,
        stopping_criteria=[IterationStoppingCriterion(num_iters)],
    )

    # make the poincare plots
    if comm is None or comm.rank == 0:
        plot_poincare_data(
            fieldlines_phi_hits,
            phis,
            out_dir / f"poincare_fieldline_{label}.png",
            dpi=100,
            surf=s,
        )


def make_qfm(s, Bfield, n_iters=200):
    """
    Given some Bfield, compute a quadratic flux-minimizing surface (QFMS)
    for the field configuration on the surface s.

    Args:
        s (Surface): plasma boundary surface.
        Bfield (MagneticField or InterpolatedField): MagneticField or InterpolatedField class object.
        n_iters (int): Maximum number of iterations for the optimization.

    Returns:
        qfm_surface (QfmSurface): The identified QfmSurface class object.
    """
    from simsopt.geo.qfmsurface import QfmSurface
    from simsopt.geo.surfaceobjectives import QfmResidual, Volume

    # weight for the optimization
    constraint_weight = 1e-2

    # First optimize at fixed volume
    qfm = QfmResidual(s, Bfield)
    qfm.J()

    vol = Volume(s)
    vol_target = vol.J()
    qfm_surface = QfmSurface(Bfield, s, vol, vol_target)

    qfm_surface.minimize_qfm_penalty_constraints_LBFGS(
        tol=1e-15, maxiter=n_iters, constraint_weight=constraint_weight
    )
    print(
        f"||vol constraint||={0.5 * (s.volume() - vol_target) ** 2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}"
    )

    constraint_weight = 1e-4
    # repeat the optimization for further convergence
    qfm_surface.minimize_qfm_penalty_constraints_LBFGS(
        tol=1e-15, maxiter=n_iters, constraint_weight=constraint_weight
    )
    print(
        f"||vol constraint||={0.5 * (s.volume() - vol_target) ** 2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}"
    )
    return qfm_surface


def calculate_modB_on_major_radius(bs, s, plot=False, savepath=None):
    """
    Check the average magnetic field strength along the major radius
    (m=n=0 mode of a SurfaceRZFourier object)
    to make sure the configuration is scaled correctly. For highly shaped
    stellarators, this can deviate a bit from the on-axis B field strength.

    Args:
        bs (BiotSavart): MagneticField or BiotSavart class object.
        s (SurfaceRZFourier): plasma boundary surface.
        plot (bool): If True, create a figure of |B| vs toroidal angle φ.
        savepath (str, optional): If given, save the |B|(φ) figure to this path.

    Returns:
        B0avg (float): Average magnetic field strength along
          the major radius (m=n=0 mode of a SurfaceRZFourier object) of the device.
    """
    nphi = len(s.quadpoints_phi)
    bspoints = np.zeros((nphi, 3))

    # rescale phi from [0, 1) to [0, 2 * pi)
    phi = s.quadpoints_phi * 2 * np.pi

    R0 = s.get_rc(0, 0)
    for i in range(nphi):
        bspoints[i] = np.array([R0 * np.cos(phi[i]), R0 * np.sin(phi[i]), 0.0])
    bs.set_points(bspoints)
    B0 = np.linalg.norm(bs.B(), axis=-1)
    B0avg = np.mean(B0)
    print("Bmag at R = ", R0, ", Z = 0: ", B0)
    print("toroidally averaged Bmag at R = ", R0, ", Z = 0: ", B0avg)

    if plot or savepath is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(phi, B0, "b-", linewidth=1.5)
        ax.axhline(
            B0avg,
            color="r",
            linestyle="--",
            label=rf"$\langle B \rangle$ = {B0avg:.4f} T",
        )
        ax.set_xlabel(r"Toroidal angle $\varphi$ (rad)")
        ax.set_ylabel(r"$|B|$ (T)")
        ax.set_title(rf"$|B|$ on major radius R = {R0:.3f} m, Z = 0")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath, dpi=150)
        # Figure remains in pyplot state for display when script calls plt.show()

    return B0avg


"""
Following functions provide tools for large-scale parameter scans and optimization 
workflows for stellarator coil design using SIMSOPT. It includes routines for:
- Continuation methods for parameter sweeps based on previous optimizations
- Initial random parameter scans for coil optimization
- Stage II force-based optimization with flexible objective construction
- Utility functions for random sampling and data export

Functions:
- continuation_vacuum_stage_II_optimizations: Perform continuation-based parameter sweeps from previous results
- initial_vacuum_stage_II_optimizations: Run a batch of initial random parameter optimizations
- vacuum_stage_II_optimization: Main routine for vacuum stage II force-based coil optimization
- rand: Wrapper function for uniform random sampling

Intended for advanced users automating coil optimization studies and data generation for analysis or machine learning.
"""


def continuation_vacuum_stage_II_optimizations(
    N=10000,
    dx=0.05,
    config="QA",
    INPUT_DIR=None,
    OUTPUT_DIR=None,
    INPUT_FILE=None,
    FORCE_OBJ=None,
    debug=False,
    MAXITER=14000,
):
    """
    Perform a continuation method on a set of previous vacuum stage II optimizations,
    perturbing parameters and rerunning optimizations.

    Parameters
    ----------
    N : int
        Number of new optimizations to run.
    dx : float
        Relative perturbation size for parameters.
    INPUT_DIR : str
        Directory containing previous optimization results (with results.json).
    OUTPUT_DIR : str
        Directory to save new optimization results.
    INPUT_FILE : str
        Path to VMEC input file for the target surface.
    MAXITER : int
        Maximum number of optimizer iterations per job.
    """
    if config == "QA":
        if INPUT_DIR is None:
            INPUT_DIR = "./output/QA/optimizations/"
        if OUTPUT_DIR is None:
            OUTPUT_DIR = "./output/QA/optimizations/continuation/"
        if INPUT_FILE is None:
            INPUT_FILE = "./inputs/input.LandremanPaul2021_QA"
    elif config == "QH":
        if INPUT_DIR is None:
            INPUT_DIR = "./output/QH/optimizations/"
        if OUTPUT_DIR is None:
            OUTPUT_DIR = "./output/QH/optimizations/continuation/"
        if INPUT_FILE is None:
            INPUT_FILE = "./inputs/input.LandremanPaul2021_QH_magwell_R0=1"
    else:
        raise ValueError(f"Invalid configuration: {config}")
    if FORCE_OBJ is None:
        with_force = False
    else:
        with_force = True

    # Read in input optimizations
    results = glob.glob(f"{INPUT_DIR}*/results.json")
    df = []
    for results_file in results:
        with open(results_file, "r") as f:
            data = json.load(f)

        # Check if the corresponding directory and biot_savart.json exist
        uuid_from_path = os.path.basename(os.path.dirname(results_file))
        biot_savart_path = os.path.join(INPUT_DIR, uuid_from_path, "biot_savart.json")

        # Only include results that have both results.json and biot_savart.json
        if os.path.exists(biot_savart_path):
            # Wrap lists in another list
            for key, value in data.items():
                if isinstance(value, list):
                    data[key] = [value]
            df.append(data)
        else:
            print(f"DEBUG: Skipping {uuid_from_path} - no biot_savart.json found")

    def perturb(init, parameter):
        "Perturbs a parameter from an initial optimization."
        val = init[parameter]
        if isinstance(val, list):
            val = val[0]
        return np.random.uniform(1 - dx, 1 + dx) * val

    for i in range(N):
        if len(df) == 0:
            print("ERROR: Data list is empty, cannot sample from it")
            return
        init = df[np.random.randint(0, len(df))]

        # FIXED PARAMETERS
        ARCLENGTH_WEIGHT = 0.01
        UUID_init_from = (
            init["UUID"] if not isinstance(init["UUID"], list) else init["UUID"][0]
        )
        ncoils = (
            init["ncoils"]
            if not isinstance(init["ncoils"], list)
            else init["ncoils"][0]
        )
        order = (
            init["order"] if not isinstance(init["order"], list) else init["order"][0]
        )
        R1 = init["R1"] if not isinstance(init["R1"], list) else init["R1"][0]

        # RANDOM PARAMETERS
        CURVATURE_THRESHOLD = perturb(init, "max_κ_threshold")
        MSC_THRESHOLD = perturb(init, "msc_threshold")
        CS_THRESHOLD = perturb(init, "cs_threshold")
        CC_THRESHOLD = perturb(init, "cc_threshold")
        FORCE_THRESHOLD = perturb(init, "force_threshold")
        LENGTH_TARGET = perturb(init, "length_target")

        LENGTH_WEIGHT = perturb(init, "length_weight")
        CURVATURE_WEIGHT = perturb(init, "max_κ_weight")
        MSC_WEIGHT = perturb(init, "msc_weight")
        CS_WEIGHT = perturb(init, "cs_weight")
        CC_WEIGHT = perturb(init, "cc_weight")
        FORCE_WEIGHT = perturb(init, "force_weight")

        kwargs = {}
        kwargs["CURVATURE_THRESHOLD"] = CURVATURE_THRESHOLD
        kwargs["MSC_THRESHOLD"] = MSC_THRESHOLD
        kwargs["CS_THRESHOLD"] = CS_THRESHOLD
        kwargs["CC_THRESHOLD"] = CC_THRESHOLD
        kwargs["FORCE_THRESHOLD"] = FORCE_THRESHOLD
        kwargs["LENGTH_TARGET"] = LENGTH_TARGET
        kwargs["LENGTH_WEIGHT"] = LENGTH_WEIGHT
        kwargs["CURVATURE_WEIGHT"] = CURVATURE_WEIGHT
        kwargs["MSC_WEIGHT"] = MSC_WEIGHT
        kwargs["CS_WEIGHT"] = CS_WEIGHT
        kwargs["CC_WEIGHT"] = CC_WEIGHT
        kwargs["FORCE_WEIGHT"] = FORCE_WEIGHT
        kwargs["ARCLENGTH_WEIGHT"] = ARCLENGTH_WEIGHT
        kwargs["with_force"] = with_force
        kwargs["dx"] = dx
        kwargs["MAXITER"] = MAXITER
        kwargs["FORCE_OBJ"] = FORCE_OBJ
        kwargs["dx"] = dx
        kwargs["debug"] = debug

        # RUNNING THE JOBS
        results = vacuum_stage_II_optimization(
            config=config,
            OUTPUT_DIR=OUTPUT_DIR,
            INPUT_FILE=INPUT_FILE,
            R1=R1,
            order=order,
            ncoils=ncoils,
            UUID_init_from=UUID_init_from,
            **kwargs,
        )

        print(f"Job {i + 1} completed with UUID={results['UUID']}")


def initial_vacuum_stage_II_optimizations(
    N=10000,
    FORCE_OBJ=None,
    with_force=False,
    debug=False,
    MAXITER=14000,
    config="QA",
    OUTPUT_DIR=None,
    INPUT_FILE=None,
    ncoils=5,
):
    """
    Perform a batch of initial random parameter optimizations for coil design for the
    Landreman-Paul 2021 QA or QH configurations.

    Parameters
    ----------
    N : int
        Number of optimizations to run.
    OUTPUT_DIR : str
        Directory to save optimization results.
    INPUT_FILE : str
        Path to VMEC input file for the target surface.
    debug : bool
        If True, print extra diagnostics.
    ncoils : int
        Number of unique coil shapes.
    **kwargs (dict): Keyword arguments for the optimization.
        See vacuum_stage_II_optimization for more details.
    """
    if config == "QA":
        if OUTPUT_DIR is None:
            OUTPUT_DIR = "./output/QA/optimizations/"
        if INPUT_FILE is None:
            INPUT_FILE = "./inputs/input.LandremanPaul2021_QA"
    elif config == "QH":
        if OUTPUT_DIR is None:
            OUTPUT_DIR = "./output/QH/optimizations/"
        if INPUT_FILE is None:
            INPUT_FILE = "./inputs/input.LandremanPaul2021_QH_magwell_R0=1"
    else:
        raise ValueError(f"Invalid configuration: {config}")

    kwargs = {}
    kwargs["FORCE_OBJ"] = FORCE_OBJ
    kwargs["debug"] = debug
    kwargs["MAXITER"] = MAXITER
    kwargs["with_force"] = with_force

    for i in range(N):
        # FIXED PARAMETERS
        ARCLENGTH_WEIGHT = 0.01
        UUID_init_from = None  # not starting from prev. optimization
        order = 16  # 16 is very high!!!

        # RANDOM PARAMETERS
        R1 = np.random.uniform(0.35, 0.75)
        CURVATURE_THRESHOLD = np.random.uniform(5, 12)
        MSC_THRESHOLD = np.random.uniform(4, 6)
        CS_THRESHOLD = np.random.uniform(0.166, 0.300)
        CC_THRESHOLD = np.random.uniform(0.083, 0.120)
        FORCE_THRESHOLD = np.random.uniform(0, 5e04)
        LENGTH_TARGET = np.random.uniform(4.9, 5.0)
        CURVATURE_WEIGHT = 10.0 ** np.random.uniform(-9, -5)
        CS_WEIGHT = 10.0 ** np.random.uniform(-1, 4)
        CC_WEIGHT = 10.0 ** np.random.uniform(2, 5)

        # Parameters that differ between QA and QH
        if config == "QA":
            LENGTH_WEIGHT = 10.0 ** np.random.uniform(-4, -2)
            MSC_WEIGHT = 10.0 ** np.random.uniform(-7, -3)
        elif config == "QH":
            LENGTH_WEIGHT = 10.0 ** np.random.uniform(-3, -1)
            MSC_WEIGHT = 10.0 ** np.random.uniform(-5, -1)

        if with_force:
            FORCE_WEIGHT = 10.0 ** np.random.uniform(-14, -8)
        else:
            FORCE_WEIGHT = 0

        kwargs["LENGTH_WEIGHT"] = LENGTH_WEIGHT
        kwargs["MSC_WEIGHT"] = MSC_WEIGHT
        kwargs["CS_WEIGHT"] = CS_WEIGHT
        kwargs["CC_WEIGHT"] = CC_WEIGHT
        kwargs["FORCE_WEIGHT"] = FORCE_WEIGHT
        kwargs["ARCLENGTH_WEIGHT"] = ARCLENGTH_WEIGHT
        kwargs["CURVATURE_THRESHOLD"] = CURVATURE_THRESHOLD
        kwargs["MSC_THRESHOLD"] = MSC_THRESHOLD
        kwargs["CS_THRESHOLD"] = CS_THRESHOLD
        kwargs["CC_THRESHOLD"] = CC_THRESHOLD
        kwargs["FORCE_THRESHOLD"] = FORCE_THRESHOLD
        kwargs["LENGTH_TARGET"] = LENGTH_TARGET
        kwargs["CURVATURE_WEIGHT"] = CURVATURE_WEIGHT

        # RUNNING THE JOBS
        results = vacuum_stage_II_optimization(
            config=config,
            OUTPUT_DIR=OUTPUT_DIR,
            INPUT_FILE=INPUT_FILE,
            R1=R1,
            order=order,
            ncoils=ncoils,
            UUID_init_from=UUID_init_from,
            **kwargs,
        )

        print(f"Job {i + 1} completed with UUID={results['UUID']}")


def vacuum_stage_II_optimization(
    config="QA",
    OUTPUT_DIR=None,
    INPUT_FILE=None,
    R1=0.5,
    order=5,
    ncoils=5,
    UUID_init_from=None,
    **kwargs,
):
    """
    Perform a vacuum stage II force-based coil optimization with specified parameters and objectives.

    Parameters
    ----------
    config (str): Configuration to use. Must be 'QA' or 'QH'.
    R1 (float):
        Minor radius for initial coil geometry.
    order (int): Number of Fourier modes for coil parameterization.
    ncoils (int): Number of unique coil shapes.
    UUID_init_from (str or None):
        Universally Unique Identifiers (UUIDs) are standardized 128-bit identifiers that
        provide a practical way to ensure uniqueness across systems and time.
        UUID of previous optimization to initialize from, or None for random.
    **kwargs (dict): Keyword arguments for the optimization.
        LENGTH_TARGET (float): Target length for coil optimization.
        LENGTH_WEIGHT (float): Weight for length penalty.
        CURVATURE_THRESHOLD (float): Threshold for curvature penalty.
        CURVATURE_WEIGHT (float): Weight for curvature penalty.
        MSC_THRESHOLD (float): Threshold for mean squared curvature penalty.
        MSC_WEIGHT (float): Weight for mean squared curvature penalty.
        CC_THRESHOLD (float): Threshold for coil-coil distance penalty.
        CC_WEIGHT (float): Weight for coil-coil distance penalty.
        CS_THRESHOLD (float): Threshold for coil-surface distance penalty.
        Parameters and weights for objective terms.
        FORCE_OBJ (class): Force objective class to use (e.g., LpCurveForce).
        ARCLENGTH_WEIGHT (float): Weight for arclength variation penalty.
        dx (float or None): Optional perturbation for initialization.
        with_force (bool): Whether to include force terms in the objective.
        debug (bool): If True, print extra diagnostics.
        MAXITER (int): Maximum number of optimizer iterations.

    Returns
    -------
    res : OptimizeResult
        Result object from scipy.optimize.minimize.
    results : dict
        Dictionary of exported optimization results and metrics.
    coils : list
        List of optimized coil objects.
    """
    import uuid
    from simsopt._core.optimizable import load
    from simsopt.field import Current, coils_via_symmetries, BiotSavart  # coils_to_vtk
    from simsopt.geo import (
        CurveLength,
        CurveCurveDistance,
        CurveSurfaceDistance,
        MeanSquaredCurvature,
        LpCurveCurvature,
        ArclengthVariation,
        create_equally_spaced_curves,
        SurfaceRZFourier,
    )
    from simsopt.objectives import SquaredFlux, QuadraticPenalty

    start_time = time.perf_counter()

    if config == "QA":
        if OUTPUT_DIR is None:
            OUTPUT_DIR = "./output/QA/optimizations/"
        if INPUT_FILE is None:
            INPUT_FILE = "./inputs/input.LandremanPaul2021_QA"
    elif config == "QH":
        if OUTPUT_DIR is None:
            OUTPUT_DIR = "./output/QH/optimizations/"
        if INPUT_FILE is None:
            INPUT_FILE = "./inputs/input.LandremanPaul2021_QH_magwell_R0=1"
    else:
        raise ValueError(f"Invalid configuration: {config}. Must be 'QA' or 'QH'.")

    # Initialize the boundary magnetic surface:
    nphi = 32
    ntheta = 32
    s = SurfaceRZFourier.from_vmec_input(
        INPUT_FILE, range="half period", nphi=nphi, ntheta=ntheta
    )
    nfp = s.nfp
    R0 = s.get_rc(0, 0)

    def initial_base_curves(R0, R1, order, ncoils):
        return create_equally_spaced_curves(
            ncoils,
            nfp,
            stellsym=True,
            R0=R0,
            R1=R1,
            order=order,
        )

    from simsopt.field.force import LpCurveForce
    from simsopt.field.selffield import regularization_circ
    from simsopt.field import RegularizedCoil

    if UUID_init_from is None:  # No previous optimization to initialize from
        base_curves = initial_base_curves(R0, R1, order, ncoils)
        total_current = 3e5
        # Since we know the total sum of currents, we only optimize for ncoils-1
        # currents, and then pick the last one so that they all add up to the correct
        # value.
        base_currents = [
            Current(total_current / ncoils * 1e-5) * 1e5 for _ in range(ncoils - 1)
        ]
        # Above, the factors of 1e-5 and 1e5 are included so the current
        # degrees of freedom are O(1) rather than ~ MA.  The optimization
        # algorithm may not perform well if the dofs are scaled badly.
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]

        coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
        base_coils = coils[:ncoils]
        curves = [c.curve for c in coils]

        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))
    else:
        # Load the previous optimization results
        path = glob.glob(f"../**/{UUID_init_from}/biot_savart.json", recursive=True)[0]
        bs = load(path)
        coils = bs.coils
        curves = [c.curve for c in coils]
        base_coils = coils[:ncoils]
        base_curves = [base_coils[i].curve for i in range(ncoils)]
        base_currents = [base_coils[i].current for i in range(ncoils)]
        bs.set_points(s.gamma().reshape((-1, 3)))

    # Define the individual terms objective function:
    LENGTH_TARGET = kwargs.get("LENGTH_TARGET", 5.00)
    LENGTH_WEIGHT = kwargs.get("LENGTH_WEIGHT", 1e-03)
    CURVATURE_THRESHOLD = kwargs.get("CURVATURE_THRESHOLD", 12.0)
    CURVATURE_WEIGHT = kwargs.get("CURVATURE_WEIGHT", 1e-08)
    MSC_THRESHOLD = kwargs.get("MSC_THRESHOLD", 5.00)
    MSC_WEIGHT = kwargs.get("MSC_WEIGHT", 1e-04)
    CC_THRESHOLD = kwargs.get("CC_THRESHOLD", 0.083)
    CC_WEIGHT = kwargs.get("CC_WEIGHT", 1e03)
    CS_THRESHOLD = kwargs.get("CS_THRESHOLD", 0.166)
    CS_WEIGHT = kwargs.get("CS_WEIGHT", 1e03)
    FORCE_THRESHOLD = kwargs.get("FORCE_THRESHOLD", 2e04)
    FORCE_WEIGHT = kwargs.get("FORCE_WEIGHT", 1e-10)
    FORCE_OBJ = kwargs.get("FORCE_OBJ", None)
    ARCLENGTH_WEIGHT = kwargs.get("ARCLENGTH_WEIGHT", 1e-2)
    dx = kwargs.get("dx", 0.05)
    with_force = kwargs.get("with_force", False)
    debug = kwargs.get("debug", False)
    MAXITER = kwargs.get("MAXITER", 14000)

    Jf = SquaredFlux(s, bs)
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    if with_force:
        try:
            Jforce = FORCE_OBJ(base_coils, coils, p=2, threshold=FORCE_THRESHOLD)
        except:
            try:
                Jforce = FORCE_OBJ(base_coils, coils)
            except:
                Jforce = FORCE_OBJ(coils)  # For B2Energy
    else:

        class dummyObjective:
            def __init__(self):
                pass

            def J(self):
                return 0.0

            def dJ(self):
                return 0.0

        Jforce = dummyObjective()

    Jals = [ArclengthVariation(c) for c in base_curves]
    Jlength = sum(QuadraticPenalty(Jl, LENGTH_TARGET, "max") for Jl in Jls)

    # Form the total objective function.
    JF = (
        Jf
        + LENGTH_WEIGHT * Jlength
        + CC_WEIGHT * Jccdist
        + CS_WEIGHT * Jcsdist
        + CURVATURE_WEIGHT * sum(Jcs)
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
        + ARCLENGTH_WEIGHT * sum(Jals)
    )

    if with_force:
        JF += FORCE_WEIGHT * Jforce

    ###########################################################################
    ## PERFORM OPTIMIZATION ###################################################
    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        if debug:
            jf = Jf.J()
            length_val = LENGTH_WEIGHT * Jlength.J()
            cc_val = CC_WEIGHT * Jccdist.J()
            cs_val = CS_WEIGHT * Jcsdist.J()
            forces_val = Jforce.J()
            arc_val = sum(Jals).J()
            BdotN = np.mean(
                np.abs(
                    np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
                )
            )
            BdotN_over_B = np.mean(
                np.abs(
                    np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
                )
            ) / np.mean(bs.AbsB())
            outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, ⟨B·n⟩/⟨B⟩={BdotN_over_B:.1e}"
            valuestr = f"J={J:.2e}, Jf={jf:.2e}"
            cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
            kap_string = ", ".join(f"{np.max(c.kappa()):.2f}" for c in base_curves)
            msc_string = ", ".join(f"{J.J():.2f}" for J in Jmscs)
            outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
            outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.2f}"
            valuestr += f", LenObj={length_val:.2e}"
            valuestr += f", ccObj={cc_val:.2e}"
            valuestr += f", csObj={cs_val:.2e}"
            valuestr += f", forceObj={FORCE_WEIGHT * forces_val:.2e}"
            valuestr += f", arcObj={ARCLENGTH_WEIGHT * arc_val:.2e}"
            outstr += f", F={forces_val:.2e}"
            outstr += f", arclengthvar={arc_val:.2e}"
            outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
            outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
            print(outstr)
            print(valuestr)
        return J, grad

    res = minimize(
        fun,
        JF.x,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAXITER, "maxcor": 200},
        tol=1e-15,
    )
    JF.x = res.x

    ###########################################################################
    ## EXPORT OPTIMIZATION DATA ###############################################

    # MAKE DIRECTORY FOR EXPORT
    UUID = uuid.uuid4().hex  # unique id for each optimization
    OUTPUT_DIR = OUTPUT_DIR + UUID + "/"  # Directory for output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # SAVE DATA TO JSON
    BdotN = np.mean(
        np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
    )
    mean_AbsB = np.mean(bs.AbsB())
    # Create RegularizedCoil objects for force calculations
    reg_param = regularization_circ(0.05)
    base_coils_reg = [
        RegularizedCoil(c.curve, c.current, reg_param) for c in base_coils
    ]

    lpcurveforce = sum(
        [
            LpCurveForce(
                c, coils, p=2, threshold=FORCE_THRESHOLD, regularization=reg_param
            )
            for c in base_coils
        ]
    ).J()
    max_forces = [
        np.max(np.linalg.norm(c_reg.force(coils), axis=1)) for c_reg in base_coils_reg
    ]
    min_forces = [
        np.min(np.linalg.norm(c_reg.force(coils), axis=1)) for c_reg in base_coils_reg
    ]
    RMS_forces = [
        np.sqrt(np.mean(np.square(np.linalg.norm(c_reg.force(coils), axis=1))))
        for c_reg in base_coils_reg
    ]
    results = {
        "nfp": nfp,
        "ncoils": int(ncoils),
        "order": int(order),
        "nphi": nphi,
        "ntheta": ntheta,
        "R0": R0,  # if initialized from circles, else None
        "R1": R1,  # if initialized from circles, else None
        "UUID_init": UUID_init_from,  # if initialized from optimization, else None
        "length_target": LENGTH_TARGET,
        "length_weight": LENGTH_WEIGHT,
        "max_κ_threshold": CURVATURE_THRESHOLD,
        "max_κ_weight": CURVATURE_WEIGHT,
        "msc_threshold": MSC_THRESHOLD,
        "msc_weight": MSC_WEIGHT,
        "cc_threshold": CC_THRESHOLD,
        "cc_weight": CC_WEIGHT,
        "cs_threshold": CS_THRESHOLD,
        "cs_weight": CS_WEIGHT,
        "force_threshold": FORCE_THRESHOLD,
        "force_weight": FORCE_WEIGHT,
        "arclength_weight": ARCLENGTH_WEIGHT,
        # For some reason, the following values need to be converted explicitly to floats
        # to avoid issues with JSON serialization.
        "lpcurveforce": float(lpcurveforce),
        "JF": float(JF.J()),
        "Jf": float(Jf.J()),
        "gradient_norm": float(np.linalg.norm(JF.dJ())),
        "lengths": [float(J.J()) for J in Jls],
        "max_length": max(float(J.J()) for J in Jls),
        "max_κ": [float(np.max(c.kappa())) for c in base_curves],
        "max_max_κ": max(np.max(c.kappa()) for c in base_curves),
        "MSCs": [float(J.J()) for J in Jmscs],
        "max_MSC": max(float(J.J()) for J in Jmscs),
        "max_forces": [float(f) for f in max_forces],
        "max_max_force": max(float(f) for f in max_forces),
        "min_forces": [float(f) for f in min_forces],
        "min_min_force": min(float(f) for f in min_forces),
        "RMS_forces": [float(f) for f in RMS_forces],
        "mean_RMS_force": np.mean([float(f) for f in RMS_forces]),
        "arclength_variances": [float(J.J()) for J in Jals],
        "max_arclength_variance": max(float(J.J()) for J in Jals),
        "BdotN": BdotN,
        "mean_AbsB": mean_AbsB,
        "normalized_BdotN": BdotN / mean_AbsB,
        "coil_coil_distance": Jccdist.shortest_distance(),
        "coil_surface_distance": Jcsdist.shortest_distance(),
        "message": res.message,
        "success": res.success,
        "iterations": res.nit,
        "function_evaluations": res.nfev,
        "coil_currents": [c.get_value() for c in base_currents],
        "UUID": UUID,
        "eval_time": time.perf_counter() - start_time,
        "dx": dx,
    }

    with open(OUTPUT_DIR + "results.json", "w") as outfile:
        json.dump(results, outfile, indent=2)
    bs.save(
        OUTPUT_DIR + "biot_savart.json"
    )  # save the optimized coil shapes and currents
    print(time.perf_counter() - start_time)
    # return res, base_coils

    JF._children = set()
    Jf._children = set()
    bs._children = set()
    for c in coils:
        c._children = set()
        c.curve._children = set()
        c.current._children = set()

    return results


"""
The below set of scripts performs analysis and visualization for stage-two coil optimization results using force metrics. 
It processes optimization results stored in JSON files, applies engineering and quality filters, 
computes Pareto fronts for selected objectives, and generates summary plots. 
The script is intended for use in advanced coil design studies, although current functionality has been
limited to the Landreman-Paul QA and QH configurations. There is no reason why it cannot be used for other
configurations, but the parameters used for the scans defined in optimization_tools.py would need to be changed.

The assumption is that a large number of optimizations
have already been performed with "python initialization.py" (or sbatch cold_starts.sh to run batch jobs on 
a supercomputer) and the results are stored in the ./output/QA/B2Energy/ or a similar directory. This pareto
scan script was used to generate the results in the papers:

    Hurwitz, S., Landreman, M., Huslage, P. and Kaptanoglu, A., 2025. 
    Electromagnetic coil optimization for reduced Lorentz forces.
    Nuclear Fusion, 65(5), p.056044.
    https://iopscience.iop.org/article/10.1088/1741-4326/adc9bf/meta

    Kaptanoglu, A.A., Wiedman, A., Halpern, J., Hurwitz, S., Paul, E.J. and Landreman, M., 2025. 
    Reactor-scale stellarators with force and torque minimized dipole coils. 
    Nuclear Fusion, 65(4), p.046029.
    https://iopscience.iop.org/article/10.1088/1741-4326/adc318/meta

Main steps:
- Load and concatenate optimization results from a specified directory.
- Filter results based on engineering and quality constraints.
- Compute Pareto-optimal sets for force and field error objectives.
- Optionally copy Pareto-optimal result folders to a new output directory.
- Generate and save histograms and scatter plots for key metrics, colored by various parameters.

Usage:
    python coil_force_pareto_scans.py

Dependencies:
    - optimization_tools (local module)
    - matplotlib
    - numpy
    - paretoset (optional, for Pareto front calculation)
    - glob, json, os

"""


def make_stage_II_pareto_plots(df: list, df_filtered: list, OUTPUT_DIR: str = "./"):
    """
    Generate and save histograms comparing distributions of key metrics before and
    after filtering for vacuum stage II coil optimizations.

    Parameters
    ----------
    df (list): List of dictionaries containing all raw optimization results.
    df_filtered (list): List of dictionaries containing filtered optimization results.
    OUTPUT_DIR (str): Directory to save the histogram.
    """
    plt.figure(1, figsize=(14.5, 11))
    nrows = 5
    ncols = 5

    def plot_2d_hist(field: str, subplot_index: int = 0):
        """
        Plot a histogram for a given field, comparing before and after filtering.

        Parameters
        ----------
        field (str): The column name to plot.
        subplot_index (int): The subplot index in the figure.
        """
        plt.subplot(nrows, ncols, subplot_index)
        nbins = 20

        def get_field_values(data_list, field_name):
            """Extract field values from list of dicts, handling list values.

            Parameters
            ----------
            data_list (list): List of dictionaries containing optimization results.
            field_name (str): The column name to extract values from.

            Returns
            -------
            np.array: Array of field values.
            """
            values = []
            for item in data_list:
                val = item.get(field_name)
                if val is None:
                    # Field doesn't exist in this item, skip it (will be filtered out later)
                    continue
                if isinstance(val, list):
                    val = val[0] if len(val) > 0 else None
                if val is not None:
                    values.append(val)
            return np.array(values) if values else np.array([])

        data = get_field_values(df, field)
        data_filtered = get_field_values(df_filtered, field)

        # Filter out NaN and infinite values
        data = data[np.isfinite(data)]
        data_filtered = data_filtered[np.isfinite(data_filtered)]

        # Skip plotting if no valid data
        if len(data) == 0:
            plt.text(
                0.5,
                0.5,
                f"No valid data for {field}",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.xlabel(field)
            return

        if np.min(data) > 0:
            bins = np.logspace(np.log10(data.min()), np.log10(data.max()), nbins)
        else:
            bins = nbins
        _, bins, _ = plt.hist(data, bins=bins, label="before filtering")
        plt.hist(data_filtered, bins=bins, alpha=1, label="after filtering")
        plt.xlabel(field)
        plt.legend(loc=0, fontsize=6)
        plt.xlim(0, np.mean(data) * 2)
        if np.min(data) > 0:
            plt.xscale("log")

    fields = [
        "R1",
        "order",
        "max_max_force",
        "max_length",
        "max_max_κ",
        "max_MSC",
        "coil_coil_distance",
        "coil_surface_distance",
        "length_target",
        "force_threshold",
        "max_κ_threshold",
        "msc_threshold",
        "cc_threshold",
        "cs_threshold",
        "length_weight",
        "max_κ_weight",
        "msc_weight",
        "cc_weight",
        "cs_weight",
        "force_weight",
        "ncoils",
    ]

    for i, field in enumerate(fields):
        plot_2d_hist(field, i + 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "histograms.pdf")
    plt.close()


def build_stage_II_data_array(
    INPUT_DIR="./", margin_up: float = 1.5, margin_low: float = 0.5, **kwargs
):
    """
    Load, filter, and compute Pareto front for coil optimization results. Filtering is
    done based on the following engineering constraints. Max denotes the maximum over all coils,
    max denotes the maximum over a single coil, mean denotes an average over the plasma surface.
    These numbers will need to be adjusted for other stellarator configurations and rescaled if
    the major radius is scaled from the 1 m baseline:

    Parameters
    ----------
    INPUT_DIR (str, optional):
        Directory containing optimization result folders with results.json files.
    margin_up (float, default 1.5):
        The upper bound margin to consider for tolerable engineering constraints.
    margin_low (float, default 0.5):
        The lower bound margin to consider for tolerable engineering constraints.
    **kwargs (dict): Keyword arguments for the optimization.
        max_coil_length (float, default 5): Maximum coil length.
        max_max_kappa (float, default 12): Maximum curvature over all coils.
        max_mean_squared_curvature (float, default 6): Maximum mean squared curvature.
        max_coil_coil_distance (float, default 0.083): Maximum coil-coil distance.
        max_coil_surface_distance (float, default 0.166): Maximum coil-surface distance.
        min_coil_coil_distance (float, default 0.083): Minimum coil-coil distance.
        min_coil_surface_distance (float, default 0.166): Minimum coil-surface distance.
        mean_abs_B (float, default 0.22): Mean absolute magnetic field.
        max_arclength_variance (float, default 1e-2): Maximum arclength variance.
        max_normalized_BdotN (float, default 4e-2): Maximum normalized BdotN.
        max_max_force (float, default 50000): Maximum maximum force.
        min_coil_length (float, default 3): Minimum coil length.

    Returns
    -------
    df (list): List of dictionaries containing all raw optimization results.
    df_filtered (list): List of dictionaries containing filtered optimization results.
    df_pareto (list): List of dictionaries containing Pareto-optimal results (minimizing normalized_BdotN and max_max_force).
        If paretoset is not available, returns an empty list.
    """
    # Try to import paretoset, but make it optional
    try:
        from paretoset import paretoset

        paretoset_available = True
    except ImportError:
        paretoset_available = False
        import warnings

        warnings.warn(
            "paretoset package not available. Pareto front calculation will be skipped. "
            "Install with 'pip install paretoset' to enable this feature.",
            ImportWarning,
        )

    min_coil_length = kwargs.get("min_coil_length", 3)
    max_coil_length = kwargs.get("max_coil_length", 5)
    max_max_kappa = kwargs.get("max_max_kappa", 12)
    max_mean_squared_curvature = kwargs.get("max_mean_squared_curvature", 6)
    min_coil_coil_distance = kwargs.get("min_coil_coil_distance", 0.083)
    min_coil_surface_distance = kwargs.get("min_coil_surface_distance", 0.166)
    mean_abs_B = kwargs.get("mean_abs_B", 0.22)
    max_arclength_variance = kwargs.get("max_arclength_variance", 1e-2)
    max_normalized_BdotN = kwargs.get("max_normalized_BdotN", 4e-2)
    max_max_force = kwargs.get("max_max_force", 50000)
    max_coil_surface_distance = kwargs.get("max_coil_surface_distance", 0.375)
    max_coil_coil_distance = kwargs.get("max_coil_coil_distance", 0.15)

    def get_field_value(item, field_name):
        """Extract field value from dict, handling list values.

        Parameters
        ----------
        item (dict): Dictionary containing optimization results.
        field_name (str): The column name to extract values from.

        Returns
        -------
        val (float): The field value.
        """
        val = item.get(field_name)
        if isinstance(val, list):
            val = val[0] if len(val) > 0 else 0
        return val

    # Import raw data
    inputs = f"{INPUT_DIR}**/results.json"
    results = glob.glob(inputs, recursive=True)
    df = []
    for results_file in results:
        with open(results_file, "r") as f:
            data = json.load(f)
        # Wrap lists in another list
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [value]
        df.append(data)

    # Filter the data
    df_filtered = []
    for item in df:
        max_length = get_field_value(item, "max_length")
        max_max_κ = get_field_value(item, "max_max_κ")
        max_MSC = get_field_value(item, "max_MSC")
        coil_coil_distance = get_field_value(item, "coil_coil_distance")
        coil_surface_distance = get_field_value(item, "coil_surface_distance")
        mean_AbsB = get_field_value(item, "mean_AbsB")
        max_arclength_variance = get_field_value(item, "max_arclength_variance")
        normalized_BdotN = get_field_value(item, "normalized_BdotN")
        max_max_force = get_field_value(item, "max_max_force")

        # ENGINEERING CONSTRAINTS:
        if (
            max_length < max_coil_length * margin_up
            and max_max_κ < max_max_kappa * margin_up
            and max_MSC < max_mean_squared_curvature * margin_up
            and coil_coil_distance > min_coil_coil_distance * margin_low
            and coil_surface_distance > min_coil_surface_distance * margin_low
            and mean_AbsB
            > mean_abs_B
            * margin_low  # prevent coils from becoming detached from LCFS when margin_low ~ 1
            # FILTERING OUT BAD/UNNECESSARY DATA:
            and max_arclength_variance < max_arclength_variance * margin_up
            and coil_surface_distance < max_coil_surface_distance * margin_up
            and coil_coil_distance < max_coil_coil_distance * margin_up
            and max_length > min_coil_length * margin_low
            and normalized_BdotN < max_normalized_BdotN * margin_up
            and max_max_force < max_max_force * margin_up
        ):
            df_filtered.append(item)

    ### STEP 3: Generate Pareto front and export UUIDs as .txt
    if len(df_filtered) > 0 and paretoset_available:
        # Extract normalized_BdotN and max_max_force arrays for paretoset
        normalized_BdotN_array = np.array(
            [get_field_value(item, "normalized_BdotN") for item in df_filtered]
        )
        max_max_force_array = np.array(
            [get_field_value(item, "max_max_force") for item in df_filtered]
        )
        pareto_data = np.column_stack([normalized_BdotN_array, max_max_force_array])
        pareto_mask = paretoset(pareto_data, sense=[min, min])
        df_pareto = [df_filtered[i] for i in range(len(df_filtered)) if pareto_mask[i]]
    else:
        df_pareto = []

    ### Return statement
    return df, df_filtered, df_pareto


def make_curve_at_theta0(s, numquadpoints):
    """
    Make a curve at theta = 0 for a given surface. This is used in the current
    voxels optimization to define a given I_target and avoid the trivial solution.

    Parameters
    ----------
    s (Surface): Surface class object.
    numquadpoints (int): Number of quadrature points to use.

    Returns
    -------
    curve: CurveRZFourier class object.
    """
    from simsopt.geo import CurveRZFourier

    order = s.ntor + 1
    quadpoints = np.linspace(0, 1, numquadpoints, endpoint=True)
    curve = CurveRZFourier(quadpoints, order, nfp=s.nfp, stellsym=s.stellsym)
    r_mn = np.zeros((s.mpol + 1, 2 * s.ntor + 1))
    z_mn = np.zeros((s.mpol + 1, 2 * s.ntor + 1))
    for m in range(s.mpol + 1):
        if m == 0:
            nmin = 0
        else:
            nmin = -s.ntor
        for n in range(nmin, s.ntor + 1):
            r_mn[m, n + s.ntor] = s.get_rc(m, n)
            z_mn[m, n + s.ntor] = s.get_zs(m, n)
    r_n = np.sum(r_mn, axis=0)
    z_n = np.sum(z_mn, axis=0)
    for n in range(s.ntor + 1):
        if n == 0:
            curve.rc[n] = r_n[n + s.ntor]
        else:
            curve.rc[n] = r_n[n + s.ntor] + r_n[-n + s.ntor]
            curve.zs[n - 1] = -z_n[n + s.ntor] + z_n[-n + s.ntor]

    curve.x = curve.get_dofs()
    curve.x = curve.x  # need to do this to transfer data to C++
    return curve


def make_filament_from_voxels(
    current_voxels_grid: "CurrentVoxelsGrid",
    final_threshold: float,
    truncate: bool = False,
    num_fourier: int = 16,
) -> "CurveXYZFourier":
    """
    Build a filamentary curve from an optimized current voxel solution.

    Fits the nonzero voxel centers with a Fourier series. Assumes the solution
    is curve-like (sparse, path-connected).

    Parameters
    ----------
    current_voxels_grid : CurrentVoxelsGrid
        Optimized voxel grid (must have alphas and XYZ_flat set).
    final_threshold : float
        Largest L0 threshold used; voxels with norm > this are retained.
    truncate : bool, optional
        Whether to truncate small Fourier coefficients. Defaults to False.
    num_fourier : int, optional
        Number of Fourier modes per coordinate. Defaults to 16.

    Returns
    -------
    CurveXYZFourier
        Filament curve approximating the voxel current path.
    """
    from simsopt.geo import CurveXYZFourier

    # Find where voxels are nonzero
    alphas = current_voxels_grid.alphas.reshape(
        current_voxels_grid.N_grid, current_voxels_grid.n_functions
    )
    nonzero_inds = np.linalg.norm(alphas, axis=-1) > final_threshold
    xyz_curve = current_voxels_grid.XYZ_flat[nonzero_inds, :]
    nfp = current_voxels_grid.plasma_boundary.nfp
    stellsym = current_voxels_grid.plasma_boundary.stellsym
    n = xyz_curve.shape[0]

    # complete the curve via the symmetries
    xyz_total = np.zeros((n * nfp * (stellsym + 1), 3))
    if stellsym:
        stell_list = [1, -1]
    else:
        stell_list = [1]
    index = 0
    for stell in stell_list:
        for fp in range(nfp):
            phi0 = (2 * np.pi / nfp) * fp
            xyz_total[index : index + n, 0] = (
                xyz_curve[:, 0] * np.cos(phi0) - xyz_curve[:, 1] * np.sin(phi0) * stell
            )
            xyz_total[index : index + n, 1] = (
                xyz_curve[:, 0] * np.sin(phi0) + xyz_curve[:, 1] * np.cos(phi0) * stell
            )
            xyz_total[index : index + n, 2] = xyz_curve[:, 2] * stell
            index += n

    # number of fourier modes to use for the fourier expansion in each coordinate
    _ = plt.figure().add_subplot(projection="3d")
    xyz_curve = xyz_total
    plt.plot(xyz_curve[:, 0], xyz_curve[:, 1], xyz_curve[:, 2])

    # Let's assume curve is unique w/ respect to theta or phi
    # in the 1 / 2 * nfp part of the surface
    phi, unique_inds = np.unique(
        np.arctan2(xyz_curve[:, 1], xyz_curve[:, 0]), return_index=True
    )
    xyz_curve = xyz_curve[unique_inds, :]
    plt.plot(xyz_curve[:, 0], xyz_curve[:, 1], xyz_curve[:, 2])

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    x_curve = moving_average(xyz_curve[:, 0], 10)
    y_curve = moving_average(xyz_curve[:, 1], 10)
    z_curve = moving_average(xyz_curve[:, 2], 10)

    # Stitch together missing points lost during the moving average
    xn = len(x_curve)
    extra_x = -x_curve[xn // 2 : xn // 2 + 6]
    extra_y = -y_curve[xn // 2 : xn // 2 + 6]
    extra_z = z_curve[xn // 2 : xn // 2 + 6]
    x_curve = np.concatenate((extra_x, x_curve))
    y_curve = np.concatenate((extra_y, y_curve))
    z_curve = np.concatenate((extra_z, z_curve))
    x_curve = np.append(x_curve, np.flip(extra_x))
    y_curve = np.append(y_curve, -np.flip(extra_y))
    z_curve = np.append(z_curve, -np.flip(extra_z))
    xyz_curve = np.transpose([x_curve, y_curve, z_curve])
    plt.plot(xyz_curve[:, 0], xyz_curve[:, 1], xyz_curve[:, 2])

    # get phi and reorder it from -pi to pi (sometimes arctan returns different quadrant)
    phi = np.arctan2(xyz_curve[:, 1], xyz_curve[:, 0])
    phi_inds = np.argsort(phi)
    phi = phi[phi_inds] + np.pi
    xyz_curve = xyz_curve[phi_inds, :]
    plt.figure()
    plt.plot(phi, xyz_curve[:, 0])
    plt.plot(phi, xyz_curve[:, 1])
    plt.plot(phi, xyz_curve[:, 2])
    # plt.show()

    # Initialize CurveXYZFourier object
    quadpoints = 2000
    coil = CurveXYZFourier(quadpoints, num_fourier)
    dofs = coil.dofs_matrix
    # trapz is removed in favor of trapezoid as of numpy 2.0
    dofs[0][0] = np.trapezoid(xyz_curve[:, 0], phi) / np.pi

    if truncate and abs(dofs[0][0]) < 1e-2:
        dofs[0][0] = 0.0
    for f in range(num_fourier):
        for i in range(3):
            if i == 0:
                dofs[i][f * 2 + 2] = (
                    np.trapezoid(xyz_curve[:, i] * np.cos((f + 1) * phi), phi) / np.pi
                )
                if truncate and abs(dofs[i][f * 2 + 2]) < 1e-2:
                    dofs[i][f * 2 + 2] = 0.0
            else:
                dofs[i][f * 2 + 1] = (
                    np.trapezoid(xyz_curve[:, i] * np.sin((f + 1) * phi), phi) / np.pi
                )
                if truncate and abs(dofs[i][f * 2 + 1]) < 1e-2:
                    dofs[i][f * 2 + 1] = 0.0
    coil.local_x = np.concatenate(dofs)

    # now make the stellarator symmetry fixed in the optimization
    for f in range(num_fourier):
        coil.fix(f"xs({f + 1})")
    for f in range(num_fourier + 1):
        coil.fix(f"yc({f})")
        coil.fix(f"zc({f})")

    # attempt to fix all the non-nfp-symmetric coefficients to zero during optimiziation
    for f in range(num_fourier * 2 + 1):
        for i in range(3):
            if i == 0:
                if np.isclose(dofs[i][f], 0.0) and (f % 2) == 0 and (f // 2) % 2 == 0:
                    coil.fix(f"xc({f // 2})")
            elif i == 1:
                if np.isclose(dofs[i][f], 0.0) and (f % 2) != 0 and (f // 2) % 2 != 0:
                    coil.fix(f"ys({f // 2 + 1})")
            if i == 2:
                if np.isclose(dofs[i][f], 0.0) and (f % 2) != 0:
                    coil.fix(f"zs({f // 2 + 1})")
    return coil
