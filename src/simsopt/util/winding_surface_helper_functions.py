"""
Winding surface and current potential contour utilities for coil extraction.

This module provides helper functions for extracting coil contours from a current
potential defined on a winding surface. It supports both REGCOIL (legacy) and
simsopt-regcoil NetCDF output formats, as well as loading directly from
:class:`simsopt.field.CurrentPotentialSolve` objects.

The workflow typically involves:
1. Loading current potential data (from file or object)
2. Computing current potential and |∇φ| (surface current density) on a grid
3. Selecting contours (interactively or by level/points)
4. Classifying contours as window-pane (closed), modular, or helical
5. Computing currents for each contour
6. Converting contours to 3D curves and simsopt Coil objects

See the cut_coils example in examples/3_Advanced/ for a complete usage.
"""

from __future__ import annotations

__all__ = ["run_cut_coils"]

import numpy as np
from matplotlib import path
from matplotlib import pyplot as plt
from scipy.io import netcdf_file


# =============================================================================
# Current potential evaluation
# =============================================================================


def _current_potential_at_point(x: np.ndarray, args: tuple) -> float:
    """
    Evaluate the current potential φ at a point (θ, ζ) in flux coordinates.

    The potential has the form:
        φ = Σ (φ_cos[m,n] cos(mθ - nζ) + φ_sin[m,n] sin(mθ - nζ)) + I_t θ/(2π) + I_p ζ/(2π)

    where the sum is over Fourier modes (m, n), I_t is net toroidal current,
    and I_p is net poloidal current.

    Args:
        x: Point [θ, ζ] (poloidal, toroidal angle in radians). Can be list or array.
        args: Tuple (Ip, It, xm_potential, xn_potential, phi_cos, phi_sin, nfp)
              with phi_cos, phi_sin arrays of same length as xm_potential, xn_potential.

    Returns:
        Current potential value at the point (float).
    """
    Ip, It, xm_potential, xn_potential, phi_cos, phi_sin, nfp = args
    t, z = x[0], x[1]
    angle = xm_potential * t - xn_potential * z
    MV = It * t / (2 * np.pi) + Ip * z / (2 * np.pi)
    phi = np.sum(phi_cos * np.cos(angle) + phi_sin * np.sin(angle)) + MV
    return float(phi[0]) if isinstance(phi, np.ndarray) else float(phi)


def _grad_current_potential_at_point(x: np.ndarray, args: tuple) -> float:
    """
    Evaluate |∇φ| (magnitude of current potential gradient) at (θ, ζ).

    This approximates the surface current density magnitude |K|.

    Args:
        x: Array [θ, ζ].
        args: Tuple (Ip, It, xm_potential, xn_potential, phi_cos, phi_sin, nfp).

    Returns:
        |∇φ| at the point.
    """
    Ip, It, xm_potential, xn_potential, phi_cos, phi_sin, nfp = args
    t, z = x[0], x[1]
    angle = xm_potential * t - xn_potential * z
    dMV_dth = It / (2 * np.pi)
    dMV_dze = Ip / (2 * np.pi)
    dphi_dth = (
        np.sum(
            -xm_potential * phi_cos * np.sin(angle)
            + xm_potential * phi_sin * np.cos(angle)
        )
        + dMV_dth
    )
    dphi_dze = (
        np.sum(
            xn_potential * phi_cos * np.sin(angle)
            - xn_potential * phi_sin * np.cos(angle)
        )
        + dMV_dze
    )
    normK = np.sqrt(dphi_dth**2 + dphi_dze**2)
    return float(normK[0]) if isinstance(normK, np.ndarray) else float(normK)


def _genCPvals(
    thetaVals: tuple,
    zetaVals: tuple,
    numbers: tuple,
    args: tuple,
) -> tuple:
    """
    Compute current potential on a (θ, ζ) grid.

    Evaluates full, single-valued, and non-single-valued components.

    Args:
        thetaVals: (θ_min, θ_max).
        zetaVals: (ζ_min, ζ_max).
        numbers: (n_theta, n_zeta) grid sizes.
        args: (Ip, It, xm, xn, phi_cos, phi_sin, nfp).

    Returns:
        (thetas, zetas, phi_full, phi_SV, phi_NSV, (args, args_SV, args_NSV)).
    """
    Ip, It, xm_potential, xn_potential, phi_cos, phi_sin, nfp = args
    args_SV = (0 * Ip, 0 * It, xm_potential, xn_potential, phi_cos, phi_sin, nfp)
    args_NSV = (Ip, It, xm_potential, xn_potential, 0 * phi_cos, 0 * phi_sin, nfp)
    ARGS = (args, args_SV, args_NSV)
    nT, nZ = numbers
    thetas = np.linspace(thetaVals[0], thetaVals[1], nT, endpoint=False)
    zetas = np.linspace(zetaVals[0], zetaVals[1], nZ, endpoint=False)
    foo = np.zeros((nT, nZ))
    foo_SV = np.zeros((nT, nZ))
    foo_NSV = np.zeros((nT, nZ))
    for i in range(nT):
        for j in range(nZ):
            foo[i, j] = _current_potential_at_point([thetas[i], zetas[j]], args)
            foo_SV[i, j] = _current_potential_at_point([thetas[i], zetas[j]], args_SV)
            foo_NSV[i, j] = _current_potential_at_point([thetas[i], zetas[j]], args_NSV)
    return (thetas, zetas, foo.T, foo_SV.T, foo_NSV.T, ARGS)


def _genKvals(
    thetaVals: tuple,
    zetaVals: tuple,
    numbers: tuple,
    args: tuple,
) -> tuple:
    """
    Compute |∇φ| on a (θ, ζ) grid.

    Args:
        thetaVals: (θ_min, θ_max).
        zetaVals: (ζ_min, ζ_max).
        numbers: (n_theta, n_zeta).
        args: (Ip, It, xm, xn, phi_cos, phi_sin, nfp).

    Returns:
        (thetas, zetas, K_full, K_SV, K_NSV, (args, args_SV, args_NSV)).
    """
    Ip, It, xm_potential, xn_potential, phi_cos, phi_sin, nfp = args
    args_SV = (0 * Ip, 0 * It, xm_potential, xn_potential, phi_cos, phi_sin, nfp)
    args_NSV = (Ip, It, xm_potential, xn_potential, 0 * phi_cos, 0 * phi_sin, nfp)
    ARGS = (args, args_SV, args_NSV)
    nT, nZ = numbers
    thetas = np.linspace(thetaVals[0], thetaVals[1], nT, endpoint=False)
    zetas = np.linspace(zetaVals[0], zetaVals[1], nZ, endpoint=False)
    foo = np.zeros((nT, nZ))
    foo_SV = np.zeros((nT, nZ))
    foo_NSV = np.zeros((nT, nZ))
    for i in range(nT):
        for j in range(nZ):
            foo[i, j] = _grad_current_potential_at_point([thetas[i], zetas[j]], args)
            foo_SV[i, j] = _grad_current_potential_at_point(
                [thetas[i], zetas[j]], args_SV
            )
            foo_NSV[i, j] = _grad_current_potential_at_point(
                [thetas[i], zetas[j]], args_NSV
            )
    return (thetas, zetas, foo.T, foo_SV.T, foo_NSV.T, ARGS)


# =============================================================================
# Data loading
# =============================================================================


def _load_regcoil_data(regcoilName: str) -> list:
    """
    Load current potential data from a legacy REGCOIL NetCDF file.

    Args:
        regcoilName: Path to the .nc file.

    Returns:
        List of [ntheta_coil, nzeta_coil, theta_coil, zeta_coil, r_coil,
        xm_coil, xn_coil, xm_potential, xn_potential, nfp, Ip, It,
        current_potential_allLambda, single_valued_..., phi_mn_allLambda,
        lambdas, chi2_B, chi2_K, K2].
    """
    with netcdf_file(regcoilName, "r", mmap=False) as f:
        nfp = f.variables["nfp"][()]
        ntheta_coil = f.variables["ntheta_coil"][()]
        nzeta_coil = f.variables["nzeta_coil"][()]
        theta_coil = f.variables["theta_coil"][()]
        zeta_coil = f.variables["zeta_coil"][()]
        r_coil = f.variables["r_coil"][()]
        xm_coil = f.variables["xm_coil"][()]
        xn_coil = f.variables["xn_coil"][()]
        xm_potential = f.variables["xm_potential"][()]
        xn_potential = f.variables["xn_potential"][()]
        single_valued_current_potential_thetazeta_allLambda = f.variables[
            "single_valued_current_potential_thetazeta"
        ][()]
        current_potential_allLambda = f.variables["current_potential"][()]
        net_poloidal_current_Amperes = f.variables["net_poloidal_current_Amperes"][()]
        net_toroidal_current_Amperes = f.variables["net_toroidal_current_Amperes"][()]
        phi_mn_allLambda = f.variables["single_valued_current_potential_mn"][()]
        lambdas = f.variables["lambda"][()]
        K2 = f.variables["K2"][()]
        chi2_B = f.variables["chi2_B"][()]
        chi2_K = f.variables["chi2_K"][()]

    nfp = np.array([nfp])
    return [
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
        net_poloidal_current_Amperes,
        net_toroidal_current_Amperes,
        current_potential_allLambda,
        single_valued_current_potential_thetazeta_allLambda,
        phi_mn_allLambda,
        lambdas,
        chi2_B,
        chi2_K,
        K2,
    ]


def _load_simsopt_regcoil_data(regcoilName: str, sparse: bool = False) -> list:
    """
    Load current potential data from a simsopt-regcoil NetCDF file.

    Args:
        regcoilName: Path to the .nc file.
        sparse: If True, load L1/sparse solution variables.

    Returns:
        Same structure as load_regcoil_data.
    """
    with netcdf_file(regcoilName, "r", mmap=False) as f:
        nfp = f.variables["nfp"][()]
        ntheta_coil = f.variables["ntheta_coil"][()]
        nzeta_coil = f.variables["nzeta_coil"][()]
        theta_coil = f.variables["theta_coil"][()]
        zeta_coil = f.variables["zeta_coil"][()]
        r_coil = f.variables["r_coil"][()]
        xm_coil = f.variables["xm_coil"][()]
        xn_coil = f.variables["xn_coil"][()]
        xm_potential = f.variables["xm_potential"][()]
        xn_potential = f.variables["xn_potential"][()]
        net_poloidal_current_Amperes = f.variables["net_poloidal_current_amperes"][()]
        net_toroidal_current_Amperes = f.variables["net_toroidal_current_amperes"][()]

        if sparse:
            phi_mn_allLambda = f.variables["single_valued_current_potential_mn_l1"][()]
            single_valued_current_potential_thetazeta_allLambda = f.variables[
                "single_valued_current_potential_thetazeta_l1"
            ][()]
            current_potential_allLambda = np.zeros_like(
                single_valued_current_potential_thetazeta_allLambda
            )
            lambdas = f.variables["lambda"][()]
            K2 = f.variables["K2_l1"][()]
            chi2_B = f.variables["chi2_B_l1"][()]
            chi2_K = f.variables["chi2_K_l1"][()]
        else:
            phi_mn_allLambda = f.variables["single_valued_current_potential_mn"][()]
            single_valued_current_potential_thetazeta_allLambda = f.variables[
                "single_valued_current_potential_thetazeta"
            ][()]
            current_potential_allLambda = np.zeros_like(
                single_valued_current_potential_thetazeta_allLambda
            )
            lambdas = f.variables["lambda"][()]
            K2 = f.variables["K2"][()]
            chi2_B = f.variables["chi2_B"][()]
            chi2_K = f.variables["chi2_K"][()]

    return [
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
        net_poloidal_current_Amperes[0],
        net_toroidal_current_Amperes[0],
        np.asarray(current_potential_allLambda),
        np.asarray(single_valued_current_potential_thetazeta_allLambda),
        np.array(phi_mn_allLambda),
        lambdas,
        chi2_B,
        chi2_K,
        K2,
    ]


def _load_surface_dofs_properly(s, s_new):
    """
    Copy surface Fourier coefficients from s to s_new with correct indexing.

    Handles the mapping between full-torus and field-period surface
    representations (quadpoint ordering differs).

    Args:
        s: Source SurfaceRZFourier (full torus).
        s_new: Target SurfaceRZFourier (field period).

    Returns:
        s_new with DOFs set from s.
    """
    xm_coil = s.m
    xn_coil = s.n
    s_new.set_dofs(0 * s.get_dofs())
    rc_ = s.rc.copy()
    rc_[0, :] = rc_[0, ::-1]
    rs_ = s.rs.copy()
    rs_[0, :] = rs_[0, ::-1]
    zc_ = s.zc.copy()
    zc_[0, :] = zc_[0, ::-1]
    zs_ = s.zs.copy()
    zs_[0, :] = zs_[0, ::-1]
    for im in range(len(xm_coil)):
        m = xm_coil[im]
        n = xn_coil[im]
        rc = rc_[m, n + s.ntor]
        rs = rs_[m, n + s.ntor]
        zc = zc_[m, n + s.ntor]
        zs = zs_[m, n + s.ntor]
        s_new.set_rc(xm_coil[im], xn_coil[im], rc)
        s_new.set_zs(xm_coil[im], xn_coil[im], zs)
        if not s.stellsym:
            s_new.set_rs(xm_coil[im], xn_coil[im], rs)
            s_new.set_zc(xm_coil[im], xn_coil[im], zc)
    return s_new


def _load_CP_and_geometries(
    filename: str,
    plot_flags: tuple = (0, 0, 0, 0),
    plasma_resolution_mult: tuple = (1.0, 1.0),
    wsurf_resolution_mult: tuple = (1.0, 1.0),
    loadDOFsProperly: bool = True,
) -> list:
    """
    Load CurrentPotentialSolve and plasma/winding surfaces from a simsopt-regcoil NetCDF.

    Args:
        filename: Path to the .nc file.
        plot_flags: (plot coil fp, plot coil full, plot plasma fp, plot plasma full).
        plasma_resolution_mult: Resolution multipliers for plasma surface.
        wsurf_resolution_mult: Resolution multipliers for winding surface.
        loadDOFsProperly: Whether to copy surface DOFs with correct indexing.

    Returns:
        [cpst, s_coil_fp, s_coil_full, s_plasma_fp, s_plasma_full].
    """
    from simsopt.geo import SurfaceRZFourier, plot
    from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve

    plasma_ntheta_res, plasma_nzeta_res = plasma_resolution_mult
    coil_ntheta_res, coil_nzeta_res = wsurf_resolution_mult

    cpst = CurrentPotentialSolve.from_netcdf(
        filename, plasma_ntheta_res, plasma_nzeta_res, coil_ntheta_res, coil_nzeta_res
    )
    cp = CurrentPotentialFourier.from_netcdf(filename, coil_ntheta_res, coil_nzeta_res)
    cp = CurrentPotentialFourier(
        cpst.winding_surface,
        mpol=cp.mpol,
        ntor=cp.ntor,
        net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
        stellsym=True,
    )
    cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.Bnormal_plasma)

    s_coil_full = cpst.winding_surface
    s_plasma_fp = cpst.plasma_surface

    s_plasma_full = SurfaceRZFourier(
        nfp=s_plasma_fp.nfp,
        stellsym=s_plasma_fp.stellsym,
        mpol=s_plasma_fp.mpol,
        ntor=s_plasma_fp.ntor,
    )
    s_plasma_full = s_plasma_full.from_nphi_ntheta(
        nfp=s_plasma_fp.nfp,
        ntheta=len(s_plasma_fp.quadpoints_theta),
        nphi=len(s_plasma_fp.quadpoints_phi) * s_plasma_fp.nfp,
        mpol=s_plasma_fp.mpol,
        ntor=s_plasma_fp.ntor,
        stellsym=s_plasma_fp.stellsym,
        range="full torus",
    )
    s_plasma_full.set_dofs(s_plasma_fp.x)

    s_coil_fp = SurfaceRZFourier(
        nfp=s_coil_full.nfp,
        stellsym=s_coil_full.stellsym,
        mpol=s_coil_full.mpol,
        ntor=s_coil_full.ntor,
    )
    s_coil_fp = s_coil_fp.from_nphi_ntheta(
        nfp=s_coil_full.nfp,
        ntheta=len(s_coil_full.quadpoints_theta),
        nphi=len(s_coil_full.quadpoints_phi) // s_coil_full.nfp,
        mpol=s_coil_full.mpol,
        ntor=s_coil_full.ntor,
        stellsym=s_coil_full.stellsym,
        range="field period",
    )
    s_coil_fp.set_dofs(s_coil_full.x)

    if loadDOFsProperly:
        s_coil_fp = _load_surface_dofs_properly(s=s_coil_full, s_new=s_coil_fp)

    if plot_flags[0]:
        plot([s_coil_fp], alpha=0.5)
    if plot_flags[1]:
        plot([s_coil_full], alpha=0.5)
    if plot_flags[2]:
        plot([s_plasma_fp], alpha=0.5)
    if plot_flags[3]:
        plot([s_plasma_full], alpha=0.5)

    return [cpst, s_coil_fp, s_coil_full, s_plasma_fp, s_plasma_full]


# =============================================================================
# Contour utilities
# =============================================================================


def is_periodic_lines(line: np.ndarray, tol: float = 0.01) -> bool:
    """
    Check if a contour is closed (start and end points coincide within tolerance).

    Args:
        line: Nx2 array of (θ, ζ) points.
        tol: Distance tolerance for closure.

    Returns:
        True if closed (window-pane type).
    """
    p_sta = line[0, :]
    p_end = line[-1, :]
    return bool(np.linalg.norm(p_sta - p_end) < tol)


def minDist(pt: np.ndarray, line: np.ndarray) -> float:
    """
    Minimum distance from a point to the nearest vertex in a polyline.

    Args:
        pt: Point [θ, ζ] or (x, y).
        line: Nx2 array of polyline vertices.

    Returns:
        Minimum Euclidean distance from pt to any vertex in line.
    """
    diffs = pt[np.newaxis, :] - line
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.min(dists))


def _removearray(L: list, arr: np.ndarray) -> None:
    """
    Remove the first list element that equals arr (by value).

    Args:
        L: List of arrays (modified in place).
        arr: Array to remove (matched by np.array_equal).

    Raises:
        ValueError: If arr is not found in L.
    """
    for ind in range(len(L)):
        if np.array_equal(L[ind], arr):
            L.pop(ind)
            return
    raise ValueError("array not found in list.")


def sortLevels(levels: list, points: list) -> tuple:
    """
    Sort levels and points by level value (ascending).

    Args:
        levels: List of contour level values.
        points: List of (θ, ζ) points corresponding to each level.

    Returns:
        Tuple of (levels_sorted, points_sorted) with same lengths as inputs.
    """
    indexing = np.argsort(levels)
    levels = [levels[i] for i in indexing]
    points = [np.asarray(points[i]) for i in indexing]
    return (levels, points)


def chooseContours_matching_coilType(lines: list, ctype: str) -> list:
    """
    Filter contours by coil type: 'free' (all), 'wp' (closed), 'mod'/'hel' (open).

    Args:
        lines: List of contour arrays.
        ctype: 'free', 'wp', 'mod', or 'hel'.

    Returns:
        Filtered list of contours.
    """
    if ctype == "free":
        return lines
    closed = [is_periodic_lines(line) for line in lines]
    ret = []
    if ctype == "wp":
        for i, c in enumerate(closed):
            if c:
                ret.append(lines[i])
    if ctype in ("mod", "hel"):
        for i, c in enumerate(closed):
            if not c:
                ret.append(lines[i])
    return ret


def _points_in_polygon(
    coords: tuple,
    zdata,
    polygon: np.ndarray,
) -> tuple:
    """
    Find grid points inside a polygon.

    Args:
        coords: (t_values, z_values) 1D arrays defining the grid.
        zdata: Unused (legacy parameter for API compatibility).
        polygon: Nx2 array of vertices defining the polygon (θ, ζ or x, y).

    Returns:
        Tuple (i1_ret, i2_ret, BOOL) where i1_ret, i2_ret are indices into
        coords[0] and coords[1] for points inside the polygon, and BOOL is
        a flattened mask of shape (len(coords[0])*len(coords[1]),).
    """
    p = path.Path(polygon)
    XX, YY = np.meshgrid(coords[0], coords[1])
    I1, I2 = np.meshgrid(np.arange(len(coords[0])), np.arange(len(coords[1])))
    xx = XX.flatten()
    yy = YY.flatten()
    i1 = I1.flatten()
    i2 = I2.flatten()
    pts = np.vstack([xx, yy]).T
    BOOL = p.contains_points(pts)
    i1_ret = i1[BOOL]
    i2_ret = i2[BOOL]
    return (i1_ret, i2_ret, BOOL)


def map_data_full_torus_3x3(
    xdata: np.ndarray,
    ydata: np.ndarray,
    zdata: np.ndarray,
    args: tuple,
) -> tuple:
    """
    Tile full-torus (θ, ζ ∈ [0, 2π]) SV data 3×3 and add secular (NSV) contribution.

    Used when the base grid already covers the full torus [0, 2π] × [0, 2π].
    Tiles with period 2π in both θ and ζ (not 2π/nfp).

    Args:
        xdata, ydata: 1D θ and ζ arrays.
        zdata: 2D single-valued current potential on the base grid.
        args: (Ip, It, xm, xn, phi_cos, phi_sin, nfp).

    Returns:
        (xN, yN, ret) extended grid and data.
    """
    Ip, It, xm_potential, xn_potential, phi_cos, phi_sin, nfp = args
    xf = 2 * np.pi
    yf = 2 * np.pi
    xN = np.hstack([xdata - xf, xdata, xdata + xf])
    yN = np.hstack([ydata - yf, ydata, ydata + yf])
    XX, YY = np.meshgrid(xN, yN)
    NSV = (It / (2 * np.pi)) * XX + (Ip / (2 * np.pi)) * YY
    z0_tiled = np.tile(zdata, (3, 3))
    return (xN, yN, z0_tiled + NSV)


def map_data_full_torus_3x1(
    xdata: np.ndarray,
    ydata: np.ndarray,
    zdata: np.ndarray,
    args: tuple,
) -> tuple:
    """
    Tile full-torus data 3×1 in ζ (period 2π) for halfway-contour extraction.

    One period in θ, three periods in ζ by 2π. Used for single-valued
    current potential when finding halfway contours between open contours.

    Args:
        xdata, ydata: 1D θ and ζ arrays.
        zdata: 2D single-valued current potential on the base grid.
        args: (Ip, It, xm, xn, phi_cos, phi_sin, nfp).

    Returns:
        (xN, yN, ret) extended grid and data.
    """
    Ip, It, xm_potential, xn_potential, phi_cos, phi_sin, nfp = args
    xN = xdata
    yf = 2 * np.pi
    yN = np.hstack([ydata - yf, ydata, ydata + yf])
    XX, YY = np.meshgrid(xN, yN)
    NSV = (It / (2 * np.pi)) * XX + (Ip / (2 * np.pi)) * YY
    z0_tiled = np.tile(zdata, (3, 1))
    return (xN, yN, z0_tiled + NSV)


def tile_helical_contours(contours: list, types_contours: list, nfp: int) -> None:
    """
    Tile helical contours (type 2 or 3) nfp times to span full torus.

    Modifies contours in place. Each helical contour at a given level spans
    one field period in ζ; we tile nfp copies at shifted ζ to build the
    full helical coil.

    Args:
        contours: List of contour arrays (modified in place).
        types_contours: Contour type codes (1=modular, 2/3=helical).
        nfp: Number of field periods.
    """
    d_zeta = 2 * np.pi / nfp
    for i in range(len(contours)):
        if types_contours[i] in (2, 3):
            contour = contours[i]
            L = len(contour[:, 0])
            Contour = np.zeros((L * nfp, 2))
            for j in range(nfp):
                j1, j2 = j * L, (j + 1) * L
                Contour[j1:j2, 0] = contour[:, 0]
                Contour[j1:j2, 1] = contour[:, 1] - j * d_zeta
            contours[i] = Contour


def _ID_mod_hel(contours: list, tol: float = 0.05) -> tuple:
    """
    Classify open contours as modular (1), helical (2), or vacuum-field (3).

    Args:
        contours: List of contour arrays.
        tol: Tolerance for cos(θ) and ζ matching.

    Returns:
        (names, ints) e.g. (['mod','hel'], [1,2]).
    """
    ret = []
    ints = []
    for contour in contours:
        th0 = contour[0, 0]
        thf = contour[-1, 0]
        ze0 = contour[0, 1]
        zef = contour[-1, 1]
        start = np.array([np.cos(th0), ze0])
        end = np.array([np.cos(thf), zef])
        if np.linalg.norm(start - end) < tol:
            ret.append("mod")
            ints.append(1)
        elif np.isclose(np.abs(th0 - thf), 2 * np.pi, rtol=1e-2):
            ret.append("hel")
            ints.append(2)
        else:
            ret.append("vf")
            ints.append(3)
    return (ret, ints)


def ID_and_cut_contour_types(contours: list) -> tuple:
    """
    Classify contours as closed (type 0) or open, and sub-classify open as mod(1)/hel(2)/vf(3).

    Args:
        contours: List of contour arrays.

    Returns:
        (open_contours, closed_contours, type_array).
    """
    closed = [is_periodic_lines(c) for c in contours]
    surfInt = []
    lineInt = []
    type_array = []
    for i in range(len(contours)):
        if closed[i]:
            surfInt.append(i)
            type_array.append(0)
        else:
            lineInt.append(i)
            type_array.append(1)
    Open_contours = [contours[i] for i in lineInt]
    Closed_contours = [contours[i] for i in surfInt]
    names, idcs = _ID_mod_hel(Open_contours)
    for i in range(len(Open_contours)):
        type_array[lineInt[i]] = idcs[i]
    return (Open_contours, Closed_contours, type_array)


def ID_halfway_contour(
    contours: list,
    data: tuple,
    do_plot: bool,
    args: tuple,
) -> list:
    """
    Find "halfway" contours between consecutive open contours for current integration.

    Args:
        contours: List of open contour arrays.
        data: (theta, zeta, current_potential) on extended grid.
        do_plot: Whether to show debug plot.
        args: Current potential args.

    Returns:
        List of halfway contour arrays.
    """
    theta, zeta, cpd = data
    halfway_contours = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if do_plot:
        ax.set_title('Identification of "halfway" contours', fontsize=25)
        ax.set_xlabel(r"$\theta$", fontsize=45)
        ax.set_ylabel(r"$\zeta$", fontsize=45)
        ax.tick_params(axis="both", which="major", labelsize=30)
        for icoil in contours:
            plt.plot(
                icoil[:, 0], icoil[:, 1], linestyle="-", color="orangered", picker=True
            )
    for i in range(len(contours) - 1):
        eye = i % len(contours)
        eyep1 = (i + 1) % len(contours)
        coil1 = contours[eye]
        coil2 = contours[eyep1]
        Lvl1 = _current_potential_at_point([coil1[0, 0], coil1[0, 1]], args)
        Lvl2 = _current_potential_at_point([coil2[0, 0], coil2[0, 1]], args)
        Lvlnew = (Lvl1 + Lvl2) / 2
        CS = ax.contour(theta, zeta, cpd, [Lvlnew], linestyles="dashed")
        paths = _contour_paths(CS, 0)
        halfway_contours.append(paths[0])
    if not do_plot:
        plt.close()
    else:
        ax.figure.canvas.draw()
        plt.show()
    return halfway_contours


# =============================================================================
# Current computation
# =============================================================================


def compute_baseline_WP_currents(
    closed_contours: list,
    args: tuple,
    plot: bool = False,
) -> tuple:
    """
    Compute currents for window-pane (closed) contours from potential difference.

    For each closed contour, samples points inside and uses max potential difference
    to estimate the current enclosed.

    Args:
        closed_contours: List of closed contour arrays.
        args: Current potential args.
        plot: Whether to plot sampling points.

    Returns:
        (wp_currents, max_closed_contours_func_val, closed_contours_func_vals).
    """
    COORDS = []
    res = 20
    closed_contours_func_vals = []
    max_closed_contours_func_val = []
    for cont in closed_contours:
        t = np.linspace(np.min(cont[:, 0]), np.max(cont[:, 0]), res)
        z = np.linspace(np.min(cont[:, 1]), np.max(cont[:, 1]), res)
        i1_ret, i2_ret, BOOL = _points_in_polygon([t, z], None, cont)
        coordinates = np.vstack([t[i1_ret], z[i2_ret]]).T
        COORDS.append(coordinates)
        func_vals = [_current_potential_at_point(pt, args) for pt in coordinates]
        contour_value = _current_potential_at_point(cont[0, :], args)
        Z = np.abs(contour_value - np.array(func_vals))
        idx = np.argmax(Z)
        max_closed_contours_func_val.append([coordinates[idx, :], np.max(Z)])
        closed_contours_func_vals.append(contour_value)

    if plot:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlabel(r"$\theta$", fontsize=35)
        ax.set_ylabel(r"$\zeta$", fontsize=35)
        cs = ["r", "g", "b", "m"]
        ss = ["s", "o", "x", "^"]
        for i in range(len(closed_contours)):
            ax.plot(
                closed_contours[i][:, 0],
                closed_contours[i][:, 1],
                marker="",
                linestyle="--",
                color=cs[i % len(cs)],
            )
            ax.plot(
                COORDS[i][:, 0],
                COORDS[i][:, 1],
                marker=ss[i % len(ss)],
                linestyle="",
                color=cs[i % len(cs)],
            )
            ax.plot(
                max_closed_contours_func_val[i][0][0],
                max_closed_contours_func_val[i][0][1],
                marker="o",
                linestyle="",
                color="c",
            )
        plt.show()

    wp_currents = []
    for i in range(len(closed_contours_func_vals)):
        wp_currents.append(
            max_closed_contours_func_val[i][1] - closed_contours_func_vals[i]
        )
    return (wp_currents, max_closed_contours_func_val, closed_contours_func_vals)


def check_and_compute_nested_WP_currents(
    closed_contours: list,
    wp_currents: list,
    args: tuple,
    plot: bool = False,
) -> tuple:
    """
    Adjust window-pane currents when contours are nested (one inside another).

    Args:
        closed_contours: List of closed contours.
        wp_currents: Initial current estimates.
        args: Current potential args.
        plot: Whether to plot.

    Returns:
        (wp_currents, nested_BOOL, func_vals, numContsContained).
    """
    nested_BOOL = np.zeros((len(closed_contours), len(closed_contours)), dtype=bool)
    func_vals = np.zeros((len(closed_contours), len(closed_contours), 2))
    for i, cont1 in enumerate(closed_contours):
        for j, cont2 in enumerate(closed_contours):
            if i == j:
                continue
            test_pt = cont2[0, :]
            p = path.Path(cont1)
            BOOL = p.contains_points([test_pt])
            if np.sum(BOOL) > 0:
                nested_BOOL[i, j] = True
                func_vals[i, j, :] = np.array(
                    [
                        _current_potential_at_point(cont1[0, :], args),
                        _current_potential_at_point(test_pt, args),
                    ]
                )
    numContsContained = np.sum(nested_BOOL, axis=1)
    if np.sum(numContsContained) == 0:
        return (wp_currents, None, None, None)
    for i in range(len(closed_contours)):
        for j in range(len(closed_contours)):
            if nested_BOOL[i, j]:
                wp_currents[i] = func_vals[i, j, 0] - func_vals[i, j, 1]
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(r"$\kappa$ chosen contours", fontsize=25)
        ax.tick_params(axis="both", which="major", labelsize=30)
        ax.set_xlabel(r"$\theta$", fontsize=45)
        ax.set_ylabel(r"$\zeta$", fontsize=45)
        for i in range(len(closed_contours)):
            M = f"${_current_potential_at_point(closed_contours[i][0, :], args):.0f}$"
            ax.plot(
                closed_contours[i][0, 0],
                closed_contours[i][0, 1],
                marker=M,
                linestyle="--",
                color="r",
                markersize=50,
            )
            ax.plot(
                closed_contours[i][:, 0],
                closed_contours[i][:, 1],
                marker="",
                linestyle="--",
                color="k",
            )
        plt.show()
    return (wp_currents, nested_BOOL, func_vals, numContsContained)


# =============================================================================
# Geometry and curve conversion
# =============================================================================


def _SIMSOPT_line_XYZ_RZ(surf, coords: tuple) -> tuple:
    """
    Map (θ, ζ) to (X, Y, R, Z) on a simsopt SurfaceRZFourier.

    Uses the surface's gamma_lin for correct evaluation. Contour angles (theta, zeta)
    are in radians; theta is poloidal [0, 2π), zeta is toroidal.

    Args:
        surf: SurfaceRZFourier instance.
        coords: (theta_array, zeta_array) in radians.

    Returns:
        (X, Y, R, Z) Cartesian and cylindrical coordinates on the surface.
    """
    coords_the, coords_ze = np.asarray(coords[0]), np.asarray(coords[1])
    # Surface quadpoints are in [0, 1); physical angles = 2π * quadpoints
    # theta (poloidal): quadpoints_theta = theta / (2π)
    # phi (toroidal): SurfaceRZFourier uses phi_phys = 2π * quadpoints_phi (one turn = 2π)
    #     so quadpoints_phi = zeta / (2π). zeta in [0, 2π] maps to full torus.
    #     Wrap into [0, 1) for closing points with zeta > 2π.
    quadpoints_theta = np.mod(coords_the / (2 * np.pi), 1.0)
    quadpoints_phi = np.mod(coords_ze / (2 * np.pi), 1.0)
    data = np.zeros((len(coords_the), 3))
    surf.gamma_lin(data, quadpoints_phi, quadpoints_theta)
    XX = data[:, 0]
    YY = data[:, 1]
    RR = np.sqrt(XX**2 + YY**2)
    ZZ = data[:, 2]
    return (XX, YY, RR, ZZ)


def _writeToCurve(
    contour: np.ndarray,
    contour_xyz: list,
    fourier_trunc: int,
    plotting_args: tuple = (0,),
    winding_surface=None,
    fix_stellarator_symmetry: bool = None,
):
    """
    Convert a (θ, ζ) contour to a simsopt CurveXYZFourier.

    Args:
        contour: Nx2 (θ, ζ) points.
        contour_xyz: [X, Y, R, Z] from _SIMSOPT_line_XYZ_RZ or real_space.
        fourier_trunc: Number of Fourier modes.
        plotting_args: (plot_comparison,).
        winding_surface: If contour_xyz empty, compute from this surface.
        fix_stellarator_symmetry: If True, fix xs, yc, zc so the curve stays
            stellarator-symmetric during optimization. If None, auto-detect for
            helical contours (3D closed but not θζ closed).

    Returns:
        CurveXYZFourier instance.
    """
    from simsopt.geo import CurveXYZFourier

    contour_theta = contour[:, 0]
    contour_zeta = contour[:, 1]
    if len(contour_xyz) == 0 and winding_surface is not None:
        X, Y, R, Z = _SIMSOPT_line_XYZ_RZ(
            winding_surface, [contour_theta, contour_zeta]
        )
        contour_xyz = [X, Y, R, Z]
    X, Y, R, Z = contour_xyz
    XYZ = np.column_stack([X, Y, Z])

    # Arc-length parametrization along the 3D contour (not theta-zeta space)
    ds = np.sqrt(np.sum(np.diff(XYZ, axis=0) ** 2, axis=1))
    ds_sum = np.sum(ds) + 1e-12
    closed_theta_zeta = np.allclose(contour[0], contour[-1])
    closed_3d = np.linalg.norm(XYZ[-1] - XYZ[0]) < 1e-6 * ds_sum

    if closed_theta_zeta or closed_3d:
        ds = np.append(ds, np.linalg.norm(XYZ[-1] - XYZ[0]))
    else:
        ds = np.append(ds, 0)
    S = np.cumsum(ds)
    S = S / S[-1] if S[-1] > 0 else np.linspace(0, 1, len(S), endpoint=False)
    # Enforce periodicity for closed curves (smooth closure for Fourier fit)
    if closed_theta_zeta or closed_3d:
        XYZ = np.asarray(XYZ, dtype=float)
        XYZ[-1] = XYZ[0]
    # Resample to uniform arc-length spacing for least-squares fit
    from scipy.interpolate import interp1d

    n_quad = max(4 * len(contour_theta), 128)
    s_uniform = np.linspace(0, 1, n_quad, endpoint=False)
    target_xyz = np.column_stack(
        [
            interp1d(S, XYZ[:, 0], kind="linear", fill_value="extrapolate")(s_uniform),
            interp1d(S, XYZ[:, 1], kind="linear", fill_value="extrapolate")(s_uniform),
            interp1d(S, XYZ[:, 2], kind="linear", fill_value="extrapolate")(s_uniform),
        ]
    )
    quadpoints = list(s_uniform)
    curve = CurveXYZFourier(quadpoints, fourier_trunc)
    curve.least_squares_fit(target_xyz)
    ORD = fourier_trunc

    # For helical coils: fix stellarator-symmetric coefficients (xs, yc, zc)
    if fix_stellarator_symmetry is None:
        fix_stellarator_symmetry = closed_3d and not closed_theta_zeta
    if fix_stellarator_symmetry:
        for m in range(1, ORD + 1):
            curve.fix(f"xs({m})")
        for m in range(ORD + 1):
            curve.fix(f"yc({m})")
            curve.fix(f"zc({m})")

    if plotting_args[0]:
        ax = curve.plot(show=False)
        ax.plot(
            target_xyz[:, 0],
            target_xyz[:, 1],
            target_xyz[:, 2],
            marker="",
            color="r",
            markersize=10,
        )
        XYZcurve = curve.gamma()
        ax.plot(
            XYZcurve[:, 0],
            XYZcurve[:, 1],
            XYZcurve[:, 2],
            marker="",
            color="k",
            markersize=10,
            linestyle="--",
        )
        plt.show()
    return curve


# =============================================================================
# Plotting and I/O
# =============================================================================


def set_axes_equal(ax) -> None:
    """
    Make 3D axes have equal scale (spheres appear spherical).

    Workaround for matplotlib's set_aspect('equal') not working in 3D.

    Args:
        ax: Matplotlib 3D axes instance.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def _contour_paths(cdata, level_idx: int = 0) -> list:
    """
    Get contour path vertices (matplotlib version-agnostic).

    Args:
        cdata: Matplotlib contour object (ContourSet).
        level_idx: Index of the contour level.

    Returns:
        List of Nx2 arrays, one per contour path at that level.
    """
    if hasattr(cdata, "allsegs") and level_idx < len(cdata.allsegs):
        return list(cdata.allsegs[level_idx])
    return [p.vertices for p in cdata.collections[level_idx].get_paths()]


# Interactive contour selection callbacks (used by CutCoils example)
def _make_onclick(ax, args, contours, theta_coil, zeta_coil, current_potential):
    """
    Factory for double-click contour selection callback.

    Args:
        ax: Matplotlib axes.
        args: Current potential args.
        contours: List to append selected contour to.
        theta_coil, zeta_coil: 1D grids for contour plot.
        current_potential: 2D current potential array.

    Returns:
        Callback function for connect('button_press_event').
    """

    def onclick(event):
        if event.dblclick:
            xdata, ydata = event.xdata, event.ydata
            click = np.array([xdata, ydata])
            tmp_value = _current_potential_at_point(click, args)
            tmp_level = np.array([tmp_value])
            print(
                f"click coords theta,zeta=({xdata:.2f},{ydata:.2f}) with contour value {tmp_value:.3f}"
            )
            cdata = plt.contour(
                theta_coil, zeta_coil, current_potential, tmp_level, linestyles="dashed"
            )
            lines = _contour_paths(cdata, 0)
            minDists = [minDist(click, line) for line in lines]
            chosen_line_idx = np.argmin(minDists)
            chosen_line = [lines[chosen_line_idx]]
            for icoil in chosen_line:
                plt.plot(
                    icoil[:, 0], icoil[:, 1], linestyle="--", color="r", picker=True
                )
            ax.figure.canvas.draw()
            contours.append(chosen_line[0])

    return onclick


def _make_onpick(contours):
    """
    Factory for pick event (store picked artist).

    Args:
        contours: Unused; kept for API compatibility with _make_on_key.

    Returns:
        Callback function for connect('pick_event').
    """

    def onpick(event):
        plt.gca().picked_object = event.artist

    return onpick


def _make_on_key(contours):
    """
    Factory for key press (delete picked contour).

    Args:
        contours: List of contours; removes the picked contour on 'delete' key.

    Returns:
        Callback function for connect('key_press_event').
    """

    def on_key(event):
        if event.key == "delete":
            ax = plt.gca()
            if hasattr(ax, "picked_object") and ax.picked_object is not None:
                arr = ax.picked_object.get_xydata()
                _removearray(contours, arr)
                ax.picked_object.remove()
                ax.picked_object = None
                ax.figure.canvas.draw()

    return on_key


def _contour_paths(cdata, i: int) -> list:
    """Get contour path vertices from contour set (matplotlib version-agnostic)."""
    if hasattr(cdata, "allsegs") and i < len(cdata.allsegs):
        return list(cdata.allsegs[i])
    return [p.vertices for p in cdata.collections[i].get_paths()]


def _run_cut_coils_select_contours_non_interactive(
    theta_coil,
    zeta_coil,
    current_potential_,
    args,
    coil_type,
    points,
    levels,
    contours_per_period,
    ax,
):
    """
    Select coil contours non-interactively from the current potential.

    Uses one of four selection modes (in priority order):
    1. **points**: (θ, ζ) coordinates on desired contours. For each point, finds the
       nearest contour at that level and filters by coil_type (wp/mod/hel).
    2. **levels**: Explicit contour level values. Extracts all contours at those levels
       and filters by coil_type.
    3. **contours_per_period**: Number of contours to distribute uniformly between
       min and max potential. Extracts all and filters by coil_type.
    4. **default**: Uses three fixed (θ, ζ) points for demonstration.

    Plots selected contours on the given axes in red dashed lines.

    Args:
        theta_coil: 1D array of poloidal angles (θ) for the grid.
        zeta_coil: 1D array of toroidal angles (ζ) for the grid.
        current_potential_: 2D current potential on the grid.
        args: Tuple (Ip, It, xm, xn, phi_cos, phi_sin, nfp) for potential evaluation.
        coil_type: 'free', 'wp', 'mod', or 'hel' to filter contour types.
        points: Optional list of [θ, ζ] points; None to use levels or default.
        levels: Optional list of contour level values; None to use contours_per_period.
        contours_per_period: Optional int; None to use default points.
        ax: Matplotlib axes to plot contours on.

    Returns:
        List of contour arrays, each Nx2 (θ, ζ) in radians.
    """
    contours = []
    lines = []
    if points is not None:
        Pts = [np.array(pt) for pt in points]
        LvlsTemp = [_current_potential_at_point(pt, args) for pt in Pts]
        Lvls, Pts = sortLevels(LvlsTemp, Pts)
        for i in range(len(Lvls)):
            cdata = plt.contour(
                theta_coil, zeta_coil, current_potential_, Lvls, linestyles="dashed"
            )
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
            cdata = plt.contour(
                theta_coil, zeta_coil, current_potential_, Lvls, linestyles="dashed"
            )
            paths = _contour_paths(cdata, i)
            for p in paths:
                lines.append(p)
        contours = chooseContours_matching_coilType(lines, coil_type)
    elif contours_per_period is not None:
        Lvls = np.linspace(
            np.min(current_potential_), np.max(current_potential_), contours_per_period
        )
        Lvls, Pts = sortLevels(list(Lvls), list(Lvls))
        for i in range(len(Lvls)):
            cdata = plt.contour(
                theta_coil, zeta_coil, current_potential_, Lvls, linestyles="dashed"
            )
            paths = _contour_paths(cdata, i)
            for p in paths:
                lines.append(p)
        contours = chooseContours_matching_coilType(lines, coil_type)
    else:
        Pts = [[0.5, 0.5], [1.0, 0.3], [1.5, 0.5]]
        LvlsTemp = [_current_potential_at_point(np.array(pt), args) for pt in Pts]
        Lvls, Pts = sortLevels(LvlsTemp, Pts)
        for i in range(len(Lvls)):
            cdata = plt.contour(
                theta_coil, zeta_coil, current_potential_, Lvls, linestyles="dashed"
            )
            paths = _contour_paths(cdata, i)
            minDists = [minDist(Pts[i], line) for line in paths]
            chosen_line_idx = np.argmin(minDists)
            chosen_line = paths[chosen_line_idx]
            lines.append(chosen_line)
        contours = lines

    for icoil in contours:
        ax.plot(
            icoil[:, 0],
            icoil[:, 1],
            linestyle="--",
            color="r",
            picker=True,
            linewidth=1,
        )
    return contours


def _run_cut_coils_refine_contours_3x3(
    contours,
    theta_coil,
    zeta_coil,
    current_potential_SV,
    args,
    nfp_val,
):
    """
    Refine contours using a 3×3 periodic extension for proper marching-squares extraction.

    The base grid [0, 2π] × [0, 2π] can produce ambiguous contours at boundaries.
    Tiling to 3×3 in both θ and ζ yields clean closed/open contours. This function:

    1. Tiles the single-valued current potential 3×3 and adds the secular (NSV) term.
    2. Re-extracts contours at the same levels as the input contours.
    3. Replaces input contours with the 3×3 versions when types differ (better classification).
    4. Cuts modular (type 1) and helical (type 2) contours to one θ-period [0, 2π].
    5. Cuts vacuum-field (type 3) contours to one ζ field-period [0, 2π/nfp].
    6. Deletes repeated contours (marching squares duplicates for periodic data).
    7. Deletes degenerate single-point contours.

    Modifies contours in place.

    Args:
        contours: List of Nx2 (θ, ζ) contour arrays; modified in place.
        theta_coil: 1D poloidal grid.
        zeta_coil: 1D toroidal grid.
        current_potential_SV: 2D single-valued current potential.
        args: (Ip, It, xm, xn, phi_cos, phi_sin, nfp) for potential evaluation.
        nfp_val: Number of field periods (int).

    Returns:
        Tuple (contours, open_contours, closed_contours, types_contours) where
        types_contours[i] is 0 (closed), 1 (modular), 2 (helical), or 3 (vacuum-field).
    """
    theta_coil_3x3, zeta_coil_3x3, current_potential_3x3 = map_data_full_torus_3x3(
        theta_coil, zeta_coil, current_potential_SV, args
    )
    Pts = [contour[0, :] for contour in contours]
    LvlsTemp = [_current_potential_at_point(pt, args) for pt in Pts]
    Lvls, Pts = sortLevels(LvlsTemp, Pts)
    lines_3x3 = []
    for i in range(len(Lvls)):
        cdata = plt.contour(
            theta_coil_3x3,
            zeta_coil_3x3,
            current_potential_3x3,
            [Lvls[i]],
            linestyles="dashed",
        )
        paths = _contour_paths(cdata, 0)
        minDists = [minDist(Pts[i], line) for line in paths]
        chosen_line_idx = np.argmin(minDists)
        lines_3x3.append(paths[chosen_line_idx])
    plt.close("all")

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

    # Delete repeated contours
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

    return contours, open_contours, closed_contours, types_contours


def _run_cut_coils_compute_NWP_currents(
    open_contours,
    single_valued,
    totalCurrent,
    theta_coil,
    zeta_coil,
    current_potential_SV,
    args,
):
    """
    Compute currents for non-window-pane (open) contours.

    For a single open contour, assigns the full total current. For multiple open
    contours, distributes current based on the potential difference between
    adjacent contours:

    - **Multi-valued (single_valued=False)**: Current between contours i and i+1
      is proportional to the potential difference. Contours are sorted by level;
      the last contour gets the remainder to conserve total current.
    - **Single-valued (single_valued=True)**: Uses a 3×1 periodic extension in ζ
      to find halfway contours between consecutive open contours. Current in each
      band is the potential difference between halfway contours.

    Args:
        open_contours: List of open contour arrays (Nx2 θ, ζ).
        single_valued: If True, use halfway-contour method; else use level differences.
        totalCurrent: Total net current (Ip + It/nfp) for normalization.
        theta_coil: 1D poloidal grid.
        zeta_coil: 1D toroidal grid.
        current_potential_SV: 2D single-valued current potential.
        args: (Ip, It, xm, xn, phi_cos, phi_sin, nfp) for potential evaluation.

    Returns:
        List of currents, one per open contour, in the same order as open_contours.
        Empty list if no open contours.
    """
    if len(open_contours) == 1:
        return [totalCurrent]
    if len(open_contours) == 0:
        return []

    if not single_valued:
        open_levels = np.array(
            [_current_potential_at_point(cont[0, :], args) for cont in open_contours]
        )
        sort_idx = np.argsort(open_levels)
        open_levels_sorted = open_levels[sort_idx]
        NWP_from_diff = np.diff(open_levels_sorted)
        last_current = totalCurrent - np.sum(NWP_from_diff)
        NWP_currents = np.concatenate([NWP_from_diff, [last_current]])
        inv_sort = np.argsort(sort_idx)
        return list(NWP_currents[inv_sort])

    theta_3x1, zeta_3x1, current_potential_3x1 = map_data_full_torus_3x1(
        theta_coil, zeta_coil, current_potential_SV, args
    )
    open_Pts = [contour[0, :] for contour in open_contours]
    pt_offset = np.array([0, 2 * np.pi])
    Pts_3x1 = [open_Pts[0] - pt_offset] + open_Pts + [open_Pts[-1] + pt_offset]
    LvlsTemp_3x1 = [_current_potential_at_point(pt, args) for pt in Pts_3x1]
    Lvls_3x1, Pts_3x1 = sortLevels(LvlsTemp_3x1, Pts_3x1)
    lines_3x1 = []
    for i in range(len(Lvls_3x1)):
        cdata_3x1 = plt.contour(
            theta_3x1,
            zeta_3x1,
            current_potential_3x1,
            Lvls_3x1,
            linestyles="dashed",
            alpha=0.01,
        )
        paths = _contour_paths(cdata_3x1, i)
        minDists = [minDist(Pts_3x1[i], line) for line in paths]
        chosen_line_idx = np.argmin(minDists)
        lines_3x1.append(paths[chosen_line_idx])
    open_contours_3x1, _, _ = ID_and_cut_contour_types(lines_3x1)
    halfway_contours_3x1 = ID_halfway_contour(
        open_contours_3x1, [theta_3x1, zeta_3x1, current_potential_3x1], False, args
    )
    halfway_contours_Lvls = np.array(
        [_current_potential_at_point(cont[0, :], args) for cont in halfway_contours_3x1]
    )
    return list(np.diff(halfway_contours_Lvls))


def run_cut_coils(
    surface_filename,
    ilambda: int = -1,
    num_of_contours: int = 100,
    ntheta: int = 128,
    nzeta: int = 128,
    single_valued: bool = False,
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
    output_path=None,
):
    """
    Run the cut coils workflow to extract coils from a current potential.

    Parameters
    ----------
    surface_filename : Path or str
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
    from pathlib import Path
    from simsopt.field import (
        RegularizedCoil,
        Current,
        coils_to_vtk,
        regularization_circ,
    )
    from simsopt.field import BiotSavart, coils_via_symmetries
    from simsopt.field.magneticfieldclasses import WindingSurfaceField
    from simsopt.geo import plot
    from simsopt.geo.curveobjectives import CurveLength
    from simsopt import save
    from simsopt.util import in_github_actions

    surface_filename = Path(surface_filename)
    output_path = output_path or Path(f"winding_surface_{surface_filename.stem}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Load current potential data (auto-detect format)
    try:
        current_potential_vars = _load_simsopt_regcoil_data(
            str(surface_filename), sparse=False
        )
    except KeyError:
        current_potential_vars = _load_regcoil_data(str(surface_filename))

    # Load geometries for 3D mapping (works for both formats)
    cpst, s_coil_fp, s_coil_full, s_plasma_fp, s_plasma_full = _load_CP_and_geometries(
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
    print(
        f"ilambda={ilambda + 1} of {len(lambdas)}, λ={lambdas[ilambda]:.2e}, χ²_B={chi2_B[ilambda]:.2e}"
    )

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
    theta_coil = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    zeta_coil = np.linspace(0, 2 * np.pi, nzeta * nfp_val, endpoint=False)
    theta_range, zeta_range = [0, 2 * np.pi], [0, 2 * np.pi]
    grid_shape = (ntheta, nzeta * nfp_val)

    # Compute current potential and |K|
    _, _, current_potential_, current_potential_SV, _, _ = _genCPvals(
        theta_range, zeta_range, grid_shape, args
    )
    _, _, K_full, K_SV, _, ARGS = _genKvals(theta_range, zeta_range, grid_shape, args)
    args, args_SV, _ = ARGS
    if single_valued:
        current_potential_ = current_potential_SV.copy()
        args = args_SV
        K_vals = K_SV
    else:
        K_vals = K_full

    # Initialize contour selection
    contours = []
    totalCurrent = np.sum(Ip + (It / nfp_val))

    # Contour plot: full winding surface (θ, ζ ∈ [0, 2π]) with current potential
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$\theta$", fontsize=14)
    ax.set_ylabel(r"$\zeta$", fontsize=14)
    cf = ax.contourf(theta_coil, zeta_coil, K_vals, levels=50, cmap="viridis")
    ax.contour(
        theta_coil,
        zeta_coil,
        current_potential_,
        num_of_contours,
        linewidths=0.5,
        colors="k",
        alpha=0.5,
    )
    _ = fig.colorbar(cf, ax=ax, label=r"Current density $|\nabla\varphi|$")
    ax.set_xlim([0, 2 * np.pi])
    ax.set_ylim([0, 2 * np.pi])
    ax.set_aspect("equal")
    ax.set_title(
        rf"Full winding surface  $\lambda$={lambdas[ilambda]:.2e}  $\chi^2_B$={chi2_B[ilambda]:.2e}"
    )

    # Contour selection
    if interactive:
        print(
            "Interactive mode: Double-click to select contours. Press Delete to remove. Close figure when done."
        )
        _ = fig.canvas.mpl_connect(
            "button_press_event",
            _make_onclick(
                ax, args, contours, theta_coil, zeta_coil, current_potential_
            ),
        )
        fig.canvas.mpl_connect("pick_event", _make_onpick(contours))
        fig.canvas.mpl_connect("key_press_event", _make_on_key(contours))
        (plt.show() if show_plots and not in_github_actions else plt.close("all"))
    else:
        contours = _run_cut_coils_select_contours_non_interactive(
            theta_coil,
            zeta_coil,
            current_potential_,
            args,
            coil_type,
            points,
            levels,
            contours_per_period,
            ax,
        )
        fig.canvas.mpl_connect("pick_event", _make_onpick(contours))
        fig.canvas.mpl_connect("key_press_event", _make_on_key(contours))
        (plt.show() if show_plots and not in_github_actions else plt.close("all"))

    if len(contours) == 0:
        print("No contours selected. Exiting.")
        return []

    # Refine contours via 3×3 periodic extension
    contours, open_contours, closed_contours, types_contours = (
        _run_cut_coils_refine_contours_3x3(
            contours, theta_coil, zeta_coil, current_potential_SV, args, nfp_val
        )
    )

    # Compute currents
    WP_currents, _, _ = compute_baseline_WP_currents(closed_contours, args, plot=False)
    WP_currents, _, _, _ = check_and_compute_nested_WP_currents(
        closed_contours, WP_currents, args, plot=False
    )

    NWP_currents = _run_cut_coils_compute_NWP_currents(
        open_contours,
        single_valued,
        totalCurrent,
        theta_coil,
        zeta_coil,
        current_potential_SV,
        args,
    )

    # Tile helical contours (type 2 or 3) nfp times to span full torus.
    tile_helical_contours(contours, types_contours, nfp_val)

    # Map to Cartesian
    contours_xyz = []
    for contour in contours:
        contour_theta, contour_zeta = contour[:, 0], contour[:, 1]
        X, Y, R, Z = _SIMSOPT_line_XYZ_RZ(s_coil_full, [contour_theta, contour_zeta])
        contours_xyz.append([X, Y, R, Z])

    # Build curves and coils
    curves = []
    for i in range(len(contours)):
        curve = _writeToCurve(
            contours[i], contours_xyz[i], curve_fourier_cutoff, [0], s_coil_full
        )
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

    # Filter out zero-current contours
    zero_current_tol = 1e-14
    keep = [i for i in range(len(contours)) if abs(currents[i]) > zero_current_tol]
    if len(keep) < len(contours):
        for i in sorted(range(len(contours)), reverse=True):
            if i not in keep:
                ctype = (
                    "mod"
                    if types_contours[i] == 1
                    else "hel"
                    if types_contours[i] in (2, 3)
                    else "wp"
                )
                print(
                    f"Removing contour {i} (type={ctype}): zero current ({currents[i]:.2e})"
                )
        contours = [contours[i] for i in keep]
        contours_xyz = [contours_xyz[i] for i in keep]
        types_contours = [types_contours[i] for i in keep]
        currents = [currents[i] for i in keep]
        curves = []
        for i in range(len(contours)):
            curve = _writeToCurve(
                contours[i], contours_xyz[i], curve_fourier_cutoff, [0], s_coil_full
            )
            curves.append(curve)

    Currents = [Current(c) for c in currents]
    coils = [
        RegularizedCoil(curve, curr, regularization_circ(0.05))
        for curve, curr in zip(curves, Currents)
    ]

    if map_to_full_torus and modular_coils_via_symmetries:
        foo = []
        for i in range(len(contours)):
            if types_contours[i] < 2 or helical_coils_via_symmetries:
                coils_symm = coils_via_symmetries(
                    [curves[i]],
                    [Currents[i]],
                    nfp_val,
                    stellsym=True,
                    regularizations=[regularization_circ(0.05)],
                )
                for c in coils_symm:
                    foo.append(c)
            if types_contours[i] >= 2 and not helical_coils_via_symmetries:
                foo.append(coils[i])
        curves = [c.curve for c in foo]
        Currents = [c.current for c in foo]
        currents = [c.get_value() for c in Currents]
        coils = foo

    print(f"\n{len(coils)} coils extracted.")

    print("\nCoil currents (A) and lengths (m):")
    for i, c in enumerate(coils):
        curr = c.current.get_value()
        length = float(CurveLength(c.curve).J())
        print(f"  Coil {i + 1:3d}:  current = {curr:14.6e}  length = {length:.6f}")

    if show_final_coilset:
        ax = plt.figure().add_subplot(projection="3d")
        Plot = []
        if s_plasma_full is not None:
            Plot.append(s_plasma_full)
        elif s_plasma_fp is not None:
            Plot.append(s_plasma_fp)
        Plot.extend(curves)
        ax = plot(Plot, ax=ax, alpha=1, color="b", close=True)
        set_axes_equal(ax)
        (plt.show() if show_plots and not in_github_actions else plt.close("all"))

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
    s_plasma_full.to_vtk(
        str(output_path / "plasma_Bn_WindingSurfaceField"),
        extra_data={"B_N": Bn_ws[:, :, None]},
    )

    # Bn from BiotSavart (extracted coils)
    bs = BiotSavart(coils)
    bs.set_points_cart(points_plasma)
    B_bs = bs.B().reshape(s_plasma_full.gamma().shape)
    Bn_bs = np.sum(B_bs * s_plasma_full.unitnormal(), axis=2)
    s_plasma_full.to_vtk(
        str(output_path / "plasma_Bn_BiotSavart"), extra_data={"B_N": Bn_bs[:, :, None]}
    )
    s_coil_full.to_vtk(str(output_path / "winding_surface"))
    coils_to_vtk(coils, str(output_path / "final_coils"), close=True)
    return coils
