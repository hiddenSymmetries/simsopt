#!/usr/bin/env python3
"""
Cut Coils: Extract discrete coils from a current potential on a winding surface.

This example demonstrates how to extract coil contours from a regularized current
potential (from REGCOIL or simsopt-regcoil) and convert them into simsopt Coil
objects. The workflow includes:

1. Loading current potential data from a NetCDF file (simsopt-regcoil or legacy REGCOIL)
2. Computing the current potential φ and |∇φ| on a (θ, ζ) grid
3. Selecting contours by:
   - Interactive double-click selection
   - Specifying (θ, ζ) points that lie on desired contours
   - Specifying contour level values
   - Choosing N contours per field period
4. Classifying contours as window-pane (closed), modular, or helical
5. Computing currents for each contour
6. Converting contours to 3D curves and Coil objects
7. Applying stellarator symmetry to get the full coil set

Usage
-----
From the simsopt root directory:

    python examples/3_Advanced/cut_coils.py --surface regcoil_out.hsx.nc

With custom options:

    python examples/3_Advanced/cut_coils.py \\
        --surface /path/to/simsopt_regcoil_out.nc \\
        --ilambda 2 \\
        --points "0.5,1.0" "1.0,0.5" \\
        --output my_output_dir

For interactive mode (double-click to select contours):

    python examples/3_Advanced/cut_coils.py --surface file.nc --interactive

Requirements
------------
- A simsopt-regcoil or REGCOIL NetCDF output file
- For legacy REGCOIL: a surface file (nescin format) for 3D mapping
- matplotlib for contour visualization
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent to path for standalone run
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simsopt.field import RegularizedCoil, Current, coils_to_vtk, regularization_circ
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.geo import plot
from simsopt.geo.curveobjectives import CurveLength
from simsopt.util import (
    load_simsopt_regcoil_data,
    load_regcoil_data,
    load_CP_and_geometries,
    current_potential_at_point,
    genCPvals,
    genKvals,
    sortLevels,
    minDist,
    chooseContours_matching_coilType,
    is_periodic_lines,
    ID_and_cut_contour_types,
    ID_halfway_contour,
    compute_baseline_WP_currents,
    check_and_compute_nested_WP_currents,
    SIMSOPT_line_XYZ_RZ,
    writeToCurve,
    set_axes_equal,
    make_onclick,
    make_onpick,
    make_on_key,
)
from simsopt import save

# Backward compatibility alias
func = current_potential_at_point


def _contour_paths(cdata, i: int) -> list:
    """Get contour path vertices from contour set (matplotlib version-agnostic)."""
    if hasattr(cdata, "allsegs") and i < len(cdata.allsegs):
        return list(cdata.allsegs[i])
    return [p.vertices for p in cdata.collections[i].get_paths()]

# Default test file
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
DEFAULT_REGCOIL = TEST_DIR / "regcoil_out.hsx.nc"


def run_cut_coils(
    surface_filename: Path,
    ilambda: int = -1,
    num_of_contours: int = 100,
    ntheta: int = 128,
    nzeta: int = 128,
    single_valued: bool = True,
    coil_type: str = "free",
    points: list = None,
    levels: list = None,
    contours_per_period: int = None,
    interactive: bool = False,
    curve_fourier_cutoff: int = 20,
    map_to_full_torus: bool = True,
    modular_coils_via_symmetries: bool = True,
    helical_coils_via_symmetries: bool = True,
    show_final_coilset: bool = True,
    show_plots: bool = True,
    write_coils_to_file: bool = False,
    output_path: Path = None,
):
    """
    Run the cut coils workflow to extract coils from a current potential.

    Parameters
    ----------
    surface_filename : Path
        Path to the REGCOIL or simsopt-regcoil NetCDF file.
    ilambda : int
        Index of regularization parameter to use (0-based).
    num_of_contours : int
        Number of contour levels for visualization.
    ntheta, nzeta : int
        Grid resolution for contour computation.
    single_valued : bool
        If True, use single-valued current potential (zero net currents).
    coil_type : str
        'free' (all), 'wp' (window-pane only), 'mod' or 'hel' (open only).
    points : list of [theta, zeta]
        (θ, ζ) points on desired contours. Overrides levels and contours_per_period.
    levels : list of float
        Contour level values. Overrides contours_per_period.
    contours_per_period : int
        Number of contours to distribute per field period.
    interactive : bool
        If True, use double-click to select contours; press Delete to remove.
    show_plots : bool
        If True, display contour and coil plots; if False, close figures (for testing).
    curve_fourier_cutoff : int
        Fourier mode cutoff for curve representation.
    map_to_full_torus : bool
        If True, use nfp symmetry to map coils to full torus.
    modular_coils_via_symmetries : bool
        If True, duplicate modular/WP coils via stellarator symmetry.
    helical_coils_via_symmetries : bool
        If True, also duplicate helical coils via symmetry.
    show_final_coilset : bool
        If True, show 3D plot of final coils.
    write_coils_to_file : bool
        If True, save coils to JSON.
    output_path : Path, optional
        Output directory for all outputs (VTK, JSON). Default: winding_surface_<surface_filename_stem>/

    Returns
    -------
    coils : list of Coil
        The extracted coil objects.
    """
    surface_filename = Path(surface_filename)
    output_path = output_path or Path(f"winding_surface_{surface_filename.stem}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Load current potential data (auto-detect format)
    try:
        current_potential_vars = load_simsopt_regcoil_data(str(surface_filename), sparse=False)
    except KeyError:
        current_potential_vars = load_regcoil_data(str(surface_filename))

    # Load geometries for 3D mapping (works for both formats)
    cpst, s_coil_fp, s_coil_full, s_plasma_fp, s_plasma_full = load_CP_and_geometries(
        str(surface_filename), plot_flags=(0, 0, 0, 0)
    )

    (
        ntheta_coil,
        nzeta_coil,
        theta_coil,
        zeta_coil,
        r_coil,
        xm_coil,
        xn_coil,
        xm_potential,
        xn_potential,
        nfp,
        Ip,
        It,
        current_potential_allLambda,
        single_valued_current_potential_thetazeta_allLambda,
        phi_mn_allLambda,
        lambdas,
        chi2_B,
        chi2_K,
        K2,
    ) = current_potential_vars

    nfp_val = nfp[0]
    print(f"ilambda={ilambda + 1} of {len(lambdas)}, λ={lambdas[ilambda]:.2e}, χ²_B={chi2_B[ilambda]:.2e}")


    # Build args for current potential evaluation
    mn_max = len(xm_potential)
    phi_mn = phi_mn_allLambda[ilambda]
    phi_cos = np.zeros(mn_max)
    phi_sin = phi_mn[:mn_max]
    current_potential_ = current_potential_allLambda[ilambda]
    args = (Ip, It, xm_potential, xn_potential, phi_cos, phi_sin, nfp)

    if single_valued:
        Ip, It = 0, 0
        args = (0, 0, xm_potential, xn_potential, phi_cos, phi_sin, nfp)

    # Base grid: full winding surface [0, 2π] × [0, 2π] for contour selection.
    # Using the full torus ensures all helical contours are visible (helical
    # coils wrap across field-period boundaries, so a single-period view may
    # show fewer contours than the actual number of coils).
    theta_coil = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    zeta_coil = np.linspace(0, 2 * np.pi, nzeta * nfp_val, endpoint=False)
    theta_range, zeta_range = [0, 2 * np.pi], [0, 2 * np.pi]
    grid_shape = (ntheta, nzeta * nfp_val)

    # Compute current potential and |K|
    _, _, current_potential_, current_potential_SV, _, _ = genCPvals(
        theta_range, zeta_range, grid_shape, args
    )
    _, _, K_full, K_SV, _, ARGS = genKvals(
        theta_range, zeta_range, grid_shape, args
    )
    args, args_SV, _ = ARGS
    if single_valued:
        current_potential_ = current_potential_SV.copy()
        args = args_SV
        K_vals = K_SV
    else:
        K_vals = K_full

    # Initialize contour selection
    contours = []
    lines = []
    totalCurrent = np.sum(Ip + (It / nfp_val))

    # Contour plot: full winding surface (θ, ζ ∈ [0, 2π]) with current potential
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$\theta$", fontsize=14)
    ax.set_ylabel(r"$\zeta$", fontsize=14)
    cf = ax.contourf(theta_coil, zeta_coil, K_vals, levels=50, cmap="viridis")
    ax.contour(theta_coil, zeta_coil, current_potential_, num_of_contours, linewidths=0.5, colors="k", alpha=0.5)
    _ = fig.colorbar(cf, ax=ax, label=r"Current density $|\nabla\varphi|$")
    ax.set_xlim([0, 2 * np.pi])
    ax.set_ylim([0, 2 * np.pi])
    ax.set_aspect("equal")
    ax.set_title(rf"Full winding surface  $\lambda$={lambdas[ilambda]:.2e}  $\chi^2_B$={chi2_B[ilambda]:.2e}")

    # Contour selection
    if interactive:
        print("Interactive mode: Double-click to select contours. Press Delete to remove. Close figure when done.")
        _ = fig.canvas.mpl_connect("button_press_event", make_onclick(ax, args, contours, theta_coil, zeta_coil, current_potential_))
        fig.canvas.mpl_connect("pick_event", make_onpick(contours))
        fig.canvas.mpl_connect("key_press_event", make_on_key(contours))
        (plt.show() if show_plots else plt.close("all"))
    else:
        if points is not None:
            Pts = [np.array(pt) for pt in points]
            LvlsTemp = [func(pt, args) for pt in Pts]
            Lvls, Pts = sortLevels(LvlsTemp, Pts)
            for i in range(len(Lvls)):
                cdata = plt.contour(theta_coil, zeta_coil, current_potential_, Lvls, linestyles="dashed")
                paths = _contour_paths(cdata, i)
                minDists = [minDist(Pts[i], line) for line in paths]
                chosen_line_idx = np.argmin(minDists)
                chosen_line = paths[chosen_line_idx]
                if coil_type == "wp" and not is_periodic_lines(chosen_line):
                    print(f"Skipping non-window-pane contour at {Pts[i]}")
                    continue
                if coil_type in ("mod", "hel") and is_periodic_lines(chosen_line):
                    print(f"Skipping window-pane contour at {Pts[i]}")
                    continue
                lines.append(chosen_line)
            contours = lines
        elif levels is not None:
            Lvls = levels
            Pts = [None] * len(Lvls)
            Lvls, Pts = sortLevels(Lvls, Pts)
            for i in range(len(Lvls)):
                cdata = plt.contour(theta_coil, zeta_coil, current_potential_, Lvls, linestyles="dashed")
                paths = _contour_paths(cdata, i)
                for p in paths:
                    lines.append(p)
            contours = chooseContours_matching_coilType(lines, coil_type)
        elif contours_per_period is not None:
            Lvls = np.linspace(np.min(current_potential_), np.max(current_potential_), contours_per_period)
            Lvls, Pts = sortLevels(list(Lvls), list(Lvls))
            for i in range(len(Lvls)):
                cdata = plt.contour(theta_coil, zeta_coil, current_potential_, Lvls, linestyles="dashed")
                paths = _contour_paths(cdata, i)
                for p in paths:
                    lines.append(p)
            contours = chooseContours_matching_coilType(lines, coil_type)
        else:
            # Default: use a few points for demonstration
            Pts = [[0.5, 0.5], [1.0, 0.3], [1.5, 0.5]]
            LvlsTemp = [func(np.array(pt), args) for pt in Pts]
            Lvls, Pts = sortLevels(LvlsTemp, Pts)
            for i in range(len(Lvls)):
                cdata = plt.contour(theta_coil, zeta_coil, current_potential_, Lvls, linestyles="dashed")
                paths = _contour_paths(cdata, i)
                minDists = [minDist(Pts[i], line) for line in paths]
                chosen_line_idx = np.argmin(minDists)
                chosen_line = paths[chosen_line_idx]
                lines.append(chosen_line)
            contours = lines

        for icoil in contours:
            ax.plot(icoil[:, 0], icoil[:, 1], linestyle="--", color="r", picker=True, linewidth=1)
        fig.canvas.mpl_connect("pick_event", make_onpick(contours))
        fig.canvas.mpl_connect("key_press_event", make_on_key(contours))
        (plt.show() if show_plots else plt.close("all"))

    if len(contours) == 0:
        print("No contours selected. Exiting.")
        return []

    # Get contours from 3×3 periodic extension for proper marching-squares extraction.
    # Since the base grid now covers the full torus [0,2π]×[0,2π], we tile with
    # period 2π in both θ and ζ (not 2π/nfp).
    def _map_data_full_torus_3x3(xdata, ydata, zdata, args_):
        """Tile full-torus SV data 3×3 and add secular (NSV) contribution."""
        Ip_, It_, xm_, xn_, phi_cos_, phi_sin_, nfp_ = args_
        xf = 2 * np.pi
        yf = 2 * np.pi  # full torus period
        xN = np.hstack([xdata - xf, xdata, xdata + xf])
        yN = np.hstack([ydata - yf, ydata, ydata + yf])
        XX, YY = np.meshgrid(xN, yN)
        NSV = (It_ / (2 * np.pi)) * XX + (Ip_ / (2 * np.pi)) * YY
        z0_tiled = np.tile(zdata, (3, 3))
        return (xN, yN, z0_tiled + NSV)
    theta_coil_3x3, zeta_coil_3x3, current_potential_3x3 = _map_data_full_torus_3x3(
        theta_coil, zeta_coil, current_potential_SV, args
    )
    Pts = [contour[0, :] for contour in contours]
    LvlsTemp = [func(pt, args) for pt in Pts]
    Lvls, Pts = sortLevels(LvlsTemp, Pts)
    lines_3x3 = []
    for i in range(len(Lvls)):
        cdata = plt.contour(
            theta_coil_3x3, zeta_coil_3x3, current_potential_3x3,
            [Lvls[i]], linestyles="dashed"
        )
        paths = _contour_paths(cdata, 0)
        minDists = [minDist(Pts[i], line) for line in paths]
        chosen_line_idx = np.argmin(minDists)
        lines_3x3.append(paths[chosen_line_idx])
    plt.close("all")

    # Classify and fix contours
    _, _, types_lines_3x3 = ID_and_cut_contour_types(lines_3x3)
    open_contours, closed_contours, types_contours = ID_and_cut_contour_types(contours)

    for i in range(len(types_contours)):
        if types_contours[i] in (1, 2, 3) and types_contours[i] != types_lines_3x3[i]:
            contours[i] = lines_3x3[i]
    for i in range(len(types_contours)):
        if types_lines_3x3[i] in (1, 2, 3):
            icoil = lines_3x3[i]
            contour_theta = icoil[:, 0]
            contour_zeta = icoil[:, 1]
            if types_lines_3x3[i] in (1, 2):
                arg_theta0 = np.argmin(np.abs(contour_theta))
                arg_theta2pi = np.argmin(np.abs(contour_theta - 2 * np.pi))
                if arg_theta0 > arg_theta2pi:
                    contours[i] = lines_3x3[i][arg_theta2pi:arg_theta0, :]
                else:
                    contours[i] = lines_3x3[i][arg_theta0:arg_theta2pi, :]
            if types_lines_3x3[i] == 3:
                zeta_fp = 2 * np.pi / nfp_val
                arg_zeta0 = np.argmin(np.abs(contour_zeta))
                arg_zeta2pi_nfp = np.argmin(np.abs(contour_zeta - zeta_fp))
                if arg_zeta0 > arg_zeta2pi_nfp:
                    contours[i] = lines_3x3[i][arg_zeta2pi_nfp:arg_zeta0, :]
                else:
                    contours[i] = lines_3x3[i][arg_zeta0:arg_zeta2pi_nfp, :]

    open_contours, closed_contours, types_contours = ID_and_cut_contour_types(contours)

    # Delete repeated contours (marching squares can produce duplicates for periodic data)
    delete_indices = []
    marching_squares_tol = 0.001
    for i in range(len(types_contours) - 1):
        if (types_contours[i] == 1 and types_contours[i + 1] == 1) or (
            types_contours[i] == 2 and types_contours[i + 1] == 2
        ):
            contour1 = contours[i]
            contour2 = contours[i + 1]
            minDists = [minDist(pt, contour1) for pt in contour2]
            if np.min(minDists) < marching_squares_tol:
                delete_indices.append(i + 1)
    for idx in sorted(delete_indices, reverse=True):
        print("Repeated contour found. Deleting.")
        del contours[idx]
    open_contours, closed_contours, types_contours = ID_and_cut_contour_types(contours)

    # Delete degenerate single-point contours
    for i in range(len(contours) - 1, -1, -1):
        if len(contours[i]) <= 1:
            print("Degenerate single-point contour found. Deleting.")
            del contours[i]
    open_contours, closed_contours, types_contours = ID_and_cut_contour_types(contours)

    # Compute currents
    WP_currents, _, _ = compute_baseline_WP_currents(closed_contours, args, plot=False)
    WP_currents, _, _, _ = check_and_compute_nested_WP_currents(closed_contours, WP_currents, args, plot=False)

    computeHalfwayContours = len(open_contours) > 1
    if computeHalfwayContours:
        if not single_valued:
            # Multi-valued (sv=False): use direct level-based formula for correct helical/modular currents.
            # The halfway-contour method can give zero for helical coils when mixing mod/hel with secular.
            open_levels = np.array([func(cont[0, :], args) for cont in open_contours])
            # Sort by level so currents = level[i+1] - level[i] are consistent
            sort_idx = np.argsort(open_levels)
            open_levels_sorted = open_levels[sort_idx]
            NWP_from_diff = np.diff(open_levels_sorted)
            # Last contour carries current between last and first (periodic); ensure sum = totalCurrent
            last_current = totalCurrent - np.sum(NWP_from_diff)
            NWP_currents = np.concatenate([NWP_from_diff, [last_current]])
            # Restore original contour order for current assignment (j iterates over contours in types order)
            inv_sort = np.argsort(sort_idx)
            NWP_currents = NWP_currents[inv_sort]
        else:
            # Single-valued: use halfway-contour method.
            # Since the base grid covers the full torus, tile 3×1 in ζ by 2π.
            def _map_data_full_torus_3x1(xdata, ydata, zdata, args_):
                Ip_, It_, xm_, xn_, phi_cos_, phi_sin_, nfp_ = args_
                xN = xdata
                yf = 2 * np.pi  # full torus period
                yN = np.hstack([ydata - yf, ydata, ydata + yf])
                XX, YY = np.meshgrid(xN, yN)
                NSV = (It_ / (2 * np.pi)) * XX + (Ip_ / (2 * np.pi)) * YY
                z0_tiled = np.tile(zdata, (3, 1))
                return (xN, yN, z0_tiled + NSV)
            theta_3x1, zeta_3x1, current_potential_3x1 = _map_data_full_torus_3x1(
                theta_coil, zeta_coil, current_potential_SV, args
            )
            open_Pts = [contour[0, :] for contour in open_contours]
            pt_offset = np.array([0, 2 * np.pi])  # full torus offset
            Pts_3x1 = [open_Pts[0] - pt_offset] + open_Pts + [open_Pts[-1] + pt_offset]
            LvlsTemp_3x1 = [func(pt, args) for pt in Pts_3x1]
            Lvls_3x1, Pts_3x1 = sortLevels(LvlsTemp_3x1, Pts_3x1)
            lines_3x1 = []
            for i in range(len(Lvls_3x1)):
                cdata_3x1 = plt.contour(theta_3x1, zeta_3x1, current_potential_3x1, Lvls_3x1, linestyles="dashed", alpha=0.01)
                paths = _contour_paths(cdata_3x1, i)
                minDists = [minDist(Pts_3x1[i], line) for line in paths]
                chosen_line_idx = np.argmin(minDists)
                lines_3x1.append(paths[chosen_line_idx])
            open_contours_3x1, _, _ = ID_and_cut_contour_types(lines_3x1)
            halfway_contours_3x1 = ID_halfway_contour(open_contours_3x1, [theta_3x1, zeta_3x1, current_potential_3x1], False, args)
            halfway_contours_Lvls = np.array([func(cont[0, :], args) for cont in halfway_contours_3x1])
            NWP_currents = np.diff(halfway_contours_Lvls)
    elif len(open_contours) == 1:
        NWP_currents = [totalCurrent]
    else:
        NWP_currents = []

    # Tile helical contours (type 2 or 3) nfp times to span full torus.
    # Even though the base grid covers [0, 2π] in ζ, each helical contour at a
    # given level only spans one field period in ζ (the secular term shifts the
    # level). We tile nfp copies at shifted ζ to build the full helical coil.
    d_zeta = 2 * np.pi / nfp_val
    for i in range(len(contours)):
        if types_contours[i] in (2, 3):
            contour = contours[i]
            L = len(contour[:, 0])
            Contour = np.zeros((L * nfp_val, 2))
            for j in range(nfp_val):
                j1, j2 = j * L, (j + 1) * L
                Contour[j1:j2, 0] = contour[:, 0]
                Contour[j1:j2, 1] = contour[:, 1] - j * d_zeta
            contours[i] = Contour

    # Map to Cartesian (s_coil_full from load_CP_and_geometries works for both formats)
    contours_xyz = []
    for contour in contours:
        contour_theta, contour_zeta = contour[:, 0], contour[:, 1]
        X, Y, R, Z = SIMSOPT_line_XYZ_RZ(s_coil_full, [contour_theta, contour_zeta])
        contours_xyz.append([X, Y, R, Z])

    # Build curves and coils (writeToCurve for all; 3D closure handles helical)
    curves = []
    for i in range(len(contours)):
        curve = writeToCurve(contours[i], contours_xyz[i], curve_fourier_cutoff, [0], s_coil_full)
        curves.append(curve)

    currents = []
    j, k = 0, 0
    for i in range(len(contours)):
        if types_contours[i] in (1, 2, 3):
            currents.append(NWP_currents[j])
            j += 1
        else:
            currents.append(WP_currents[k])
            k += 1

    # Filter out zero-current contours (can arise from nearly-degenerate levels or indexing)
    zero_current_tol = 1e-14
    keep = [i for i in range(len(contours)) if abs(currents[i]) > zero_current_tol]
    if len(keep) < len(contours):
        for i in sorted(range(len(contours)), reverse=True):
            if i not in keep:
                ctype = "mod" if types_contours[i] == 1 else "hel" if types_contours[i] in (2, 3) else "wp"
                print(f"Removing contour {i} (type={ctype}): zero current ({currents[i]:.2e})")
        contours = [contours[i] for i in keep]
        contours_xyz = [contours_xyz[i] for i in keep]
        types_contours = [types_contours[i] for i in keep]
        currents = [currents[i] for i in keep]
        curves = []
        for i in range(len(contours)):
            curve = writeToCurve(contours[i], contours_xyz[i], curve_fourier_cutoff, [0], s_coil_full)
            curves.append(curve)

    Currents = [Current(c) for c in currents]
    coils = [RegularizedCoil(curve, curr, regularization_circ(0.05)) for curve, curr in zip(curves, Currents)]
    
    if map_to_full_torus and modular_coils_via_symmetries:
        foo = []
        for i in range(len(contours)):
            if types_contours[i] < 2 or helical_coils_via_symmetries:
                # Use nfp_val for both modular and helical: nfp rotational copies + stellarator flip.
                # nfp_sym=1 for helical caused an X artifact (two coils crossing at symmetry plane).
                coils_symm = coils_via_symmetries([curves[i]], [Currents[i]], nfp_val, stellsym=True, regularizations=[regularization_circ(0.05)])
                for c in coils_symm:
                    foo.append(c)
            if types_contours[i] >= 2 and not helical_coils_via_symmetries:
                foo.append(coils[i])
        curves = [c.curve for c in foo]
        Currents = [c.current for c in foo]
        currents = [c.get_value() for c in Currents]
        coils = foo

    print(f"\n{len(coils)} coils extracted.")

    # Print currents and lengths
    print("\nCoil currents (A) and lengths (m):")
    for i, c in enumerate(coils):
        curr = c.current.get_value()
        length = float(CurveLength(c.curve).J())
        print(f"  Coil {i+1:3d}:  current = {curr:14.6e}  length = {length:.6f}")

    if show_final_coilset:
        ax = plt.figure().add_subplot(projection="3d")
        # Build plot list: plasma first (background), then coils. Prefer full torus plasma.
        Plot = []
        if s_plasma_full is not None:
            Plot.append(s_plasma_full)
        elif s_plasma_fp is not None:
            Plot.append(s_plasma_fp)
        Plot.extend(curves)
        ax = plot(Plot, ax=ax, alpha=1, color="b", close=True)
        set_axes_equal(ax)
        (plt.show() if show_plots else plt.close("all"))

    if write_coils_to_file:
        coils_path = output_path / "coils.json"
        save(coils, str(coils_path))
        print(f"Coils saved to {coils_path}")

    # Bn from WindingSurfaceField (current potential solution)
    phi_mn = np.atleast_2d(phi_mn_allLambda)[ilambda]
    cpst.current_potential.set_dofs(phi_mn)
    wsfield = WindingSurfaceField(cpst.current_potential)
    points_plasma = s_plasma_full.gamma().reshape(-1, 3)
    wsfield.set_points_cart(points_plasma)
    B_ws = wsfield.B().reshape(s_plasma_full.gamma().shape)
    Bn_ws = np.sum(B_ws * s_plasma_full.unitnormal(), axis=2)
    s_plasma_full.to_vtk(str(output_path / "plasma_Bn_WindingSurfaceField"), extra_data={"B_N": Bn_ws[:, :, None]})

    # Bn from BiotSavart (extracted coils)
    bs = BiotSavart(coils)
    bs.set_points_cart(points_plasma)
    B_bs = bs.B().reshape(s_plasma_full.gamma().shape)
    Bn_bs = np.sum(B_bs * s_plasma_full.unitnormal(), axis=2)
    s_plasma_full.to_vtk(str(output_path / "plasma_Bn_BiotSavart"), extra_data={"B_N": Bn_bs[:, :, None]})
    s_coil_full.to_vtk(str(output_path / "winding_surface"))
    coils_to_vtk(coils, str(output_path / "final_coils"), close=True)
    return coils


def main():
    parser = argparse.ArgumentParser(
        description="Extract coils from a current potential (REGCOIL / simsopt-regcoil output).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--surface",
        type=Path,
        default=DEFAULT_REGCOIL,
        help="REGCOIL or simsopt-regcoil NetCDF file (relative paths are resolved against tests/test_files).",
    )
    parser.add_argument(
        "--ilambda",
        type=int,
        default=-1,
        help="Lambda index (0-based) to use.",
    )
    parser.add_argument(
        "--points",
        nargs="*",
        type=str,
        default=None,
        help='(θ,ζ) points as "theta,zeta" for contour selection.',
    )
    parser.add_argument(
        "--no-sv",
        action="store_true",
        help="Use full (multi-valued) current potential with net toroidal/poloidal currents.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive double-click contour selection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for VTK and JSON (default: winding_surface_<surface_stem>/).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not show final coil plot.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save coils to JSON file.",
    )
    args = parser.parse_args()

    points = None
    if args.points:
        points = []
        for s in args.points:
            parts = s.split(",")
            if len(parts) != 2:
                parser.error(f"Invalid point format: {s}. Use theta,zeta")
            points.append([float(parts[0]), float(parts[1])])

    surface_filename = args.surface if args.surface.is_absolute() else TEST_DIR / args.surface
    output_path = args.output or Path(f"winding_surface_{surface_filename.stem}")
    run_cut_coils(
        surface_filename=surface_filename,
        ilambda=args.ilambda,
        points=points,
        single_valued=not args.no_sv,
        interactive=args.interactive,
        show_final_coilset=not args.no_plot,
        write_coils_to_file=args.save,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
