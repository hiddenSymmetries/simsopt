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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simsopt.field import CurrentPotentialSolve
    from simsopt.geo import CurveXYZFourierSymmetries

__all__ = [
    'current_potential_at_point',
    'grad_current_potential_at_point',
    'genCPvals',
    'genKvals',
    'load_regcoil_data',
    'load_simsopt_regcoil_data',
    'load_from_CP_object',
    'load_CP_and_geometries',
    'is_periodic_lines',
    'minDist',
    'chooseContours_matching_coilType',
    'sortLevels',
    'points_in_polygon',
    'map_data_to_more_periods',
    'map_data_to_more_periods_3x1',
    'ID_and_cut_contour_types',
    'ID_mod_hel',
    'ID_halfway_contour',
    'compute_baseline_WP_currents',
    'check_and_compute_nested_WP_currents',
    'real_space',
    'SIMSOPT_line_XYZ_RZ',
    'REGCOIL_line_XYZ_RZ',
    'writeToCurve',
    'writeToCurve_helical',
    'gen_parametrization',
    'splice_curve_fourier_coeffs',
    'get1DFourierSinCosComps',
    'getCurveOrder',
    'set_axes_equal',
    'box_field_period',
    'ClosePlot',
    'open_writeto_file',
    'close_writeto_file',
    'writeContourToFile',
    'read_surface',
    'curves_to_vtk_w_pointdata',
    'load_surface_dofs_properly',
    'load_coils_from_json',
    'removearray',
    'generate_sunflower',
    'make_onclick',
    'make_onpick',
    'make_on_key',
]

import numpy as np
from matplotlib import path
from matplotlib import pyplot as plt
from scipy.io import netcdf_file



# =============================================================================
# Current potential evaluation
# =============================================================================

def current_potential_at_point(x: np.ndarray, args: tuple) -> float:
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


def grad_current_potential_at_point(x: np.ndarray, args: tuple) -> float:
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
    dphi_dth = np.sum(-xm_potential * phi_cos * np.sin(angle) + xm_potential * phi_sin * np.cos(angle)) + dMV_dth
    dphi_dze = np.sum(xn_potential * phi_cos * np.sin(angle) - xn_potential * phi_sin * np.cos(angle)) + dMV_dze
    normK = np.sqrt(dphi_dth**2 + dphi_dze**2)
    return float(normK[0]) if isinstance(normK, np.ndarray) else float(normK)


def genCPvals(
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
            foo[i, j] = current_potential_at_point([thetas[i], zetas[j]], args)
            foo_SV[i, j] = current_potential_at_point([thetas[i], zetas[j]], args_SV)
            foo_NSV[i, j] = current_potential_at_point([thetas[i], zetas[j]], args_NSV)
    return (thetas, zetas, foo.T, foo_SV.T, foo_NSV.T, ARGS)


def genKvals(
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
            foo[i, j] = grad_current_potential_at_point([thetas[i], zetas[j]], args)
            foo_SV[i, j] = grad_current_potential_at_point([thetas[i], zetas[j]], args_SV)
            foo_NSV[i, j] = grad_current_potential_at_point([thetas[i], zetas[j]], args_NSV)
    return (thetas, zetas, foo.T, foo_SV.T, foo_NSV.T, ARGS)


# =============================================================================
# Data loading
# =============================================================================

def load_regcoil_data(regcoilName: str) -> list:
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
    with netcdf_file(regcoilName, 'r', mmap=False) as f:
        nfp = f.variables['nfp'][()]
        ntheta_coil = f.variables['ntheta_coil'][()]
        nzeta_coil = f.variables['nzeta_coil'][()]
        theta_coil = f.variables['theta_coil'][()]
        zeta_coil = f.variables['zeta_coil'][()]
        r_coil = f.variables['r_coil'][()]
        xm_coil = f.variables['xm_coil'][()]
        xn_coil = f.variables['xn_coil'][()]
        xm_potential = f.variables['xm_potential'][()]
        xn_potential = f.variables['xn_potential'][()]
        single_valued_current_potential_thetazeta_allLambda = f.variables['single_valued_current_potential_thetazeta'][()]
        current_potential_allLambda = f.variables['current_potential'][()]
        net_poloidal_current_Amperes = f.variables['net_poloidal_current_Amperes'][()]
        net_toroidal_current_Amperes = f.variables['net_toroidal_current_Amperes'][()]
        phi_mn_allLambda = f.variables['single_valued_current_potential_mn'][()]
        lambdas = f.variables['lambda'][()]
        K2 = f.variables['K2'][()]
        chi2_B = f.variables['chi2_B'][()]
        chi2_K = f.variables['chi2_K'][()]

    nfp = np.array([nfp])
    return [
        ntheta_coil, nzeta_coil, theta_coil, zeta_coil, r_coil,
        xm_coil, xn_coil, xm_potential, xn_potential,
        nfp, net_poloidal_current_Amperes, net_toroidal_current_Amperes,
        current_potential_allLambda, single_valued_current_potential_thetazeta_allLambda, phi_mn_allLambda,
        lambdas, chi2_B, chi2_K, K2,
    ]


def load_simsopt_regcoil_data(regcoilName: str, sparse: bool = False) -> list:
    """
    Load current potential data from a simsopt-regcoil NetCDF file.

    Args:
        regcoilName: Path to the .nc file.
        sparse: If True, load L1/sparse solution variables.

    Returns:
        Same structure as load_regcoil_data.
    """
    with netcdf_file(regcoilName, 'r', mmap=False) as f:
        nfp = f.variables['nfp'][()]
        ntheta_coil = f.variables['ntheta_coil'][()]
        nzeta_coil = f.variables['nzeta_coil'][()]
        theta_coil = f.variables['theta_coil'][()]
        zeta_coil = f.variables['zeta_coil'][()]
        r_coil = f.variables['r_coil'][()]
        xm_coil = f.variables['xm_coil'][()]
        xn_coil = f.variables['xn_coil'][()]
        xm_potential = f.variables['xm_potential'][()]
        xn_potential = f.variables['xn_potential'][()]
        net_poloidal_current_Amperes = f.variables['net_poloidal_current_amperes'][()]
        net_toroidal_current_Amperes = f.variables['net_toroidal_current_amperes'][()]

        if sparse:
            phi_mn_allLambda = f.variables['single_valued_current_potential_mn_l1'][()]
            single_valued_current_potential_thetazeta_allLambda = f.variables['single_valued_current_potential_thetazeta_l1'][()]
            current_potential_allLambda = np.zeros_like(single_valued_current_potential_thetazeta_allLambda)
            lambdas = f.variables['lambda'][()]
            K2 = f.variables['K2_l1'][()]
            chi2_B = f.variables['chi2_B_l1'][()]
            chi2_K = f.variables['chi2_K_l1'][()]
        else:
            phi_mn_allLambda = f.variables['single_valued_current_potential_mn'][()]
            single_valued_current_potential_thetazeta_allLambda = f.variables['single_valued_current_potential_thetazeta'][()]
            current_potential_allLambda = np.zeros_like(single_valued_current_potential_thetazeta_allLambda)
            lambdas = f.variables['lambda'][()]
            K2 = f.variables['K2'][()]
            chi2_B = f.variables['chi2_B'][()]
            chi2_K = f.variables['chi2_K'][()]

    return [
        ntheta_coil, nzeta_coil, theta_coil, zeta_coil, r_coil,
        xm_coil, xn_coil, xm_potential, xn_potential,
        nfp, net_poloidal_current_Amperes[0], net_toroidal_current_Amperes[0],
        np.asarray(current_potential_allLambda),
        np.asarray(single_valued_current_potential_thetazeta_allLambda),
        np.array(phi_mn_allLambda),
        lambdas, chi2_B, chi2_K, K2,
    ]


def load_from_CP_object(
    cpst: CurrentPotentialSolve,
    use_l1: bool = False,
) -> list:
    """
    Load current potential data from a CurrentPotentialSolve object.

    Used when the current potential was solved in-memory rather than loaded from file.

    Args:
        cpst: CurrentPotentialSolve instance with solved current potential.
        use_l1: If True, use L1/sparse solution; else use L2/Tikhonov.

    Returns:
        Same structure as load_regcoil_data.
    """
    w = cpst.winding_surface
    net_poloidal_current_Amperes = cpst.current_potential.net_poloidal_current_amperes
    net_toroidal_current_Amperes = cpst.current_potential.net_toroidal_current_amperes
    xn_potential = cpst.current_potential.n * w.nfp
    xm_potential = cpst.current_potential.m
    xn_coil = w.n * w.nfp
    xm_coil = w.m
    nfp = np.array([w.nfp])

    if use_l1:
        phi_mn_allLambda = cpst.dofs_l1
        current_potential_allLambda = cpst.current_potential_l1
        lambdas = cpst.ilambdas_l1
        chi2_B = cpst.fBs_l1
        chi2_K = cpst.fKs_l1
        K2 = cpst.K2s_l1
    else:
        phi_mn_allLambda = cpst.dofs_l2
        current_potential_allLambda = cpst.current_potential_l2
        lambdas = cpst.ilambdas_l2
        chi2_B = cpst.fBs_l2
        chi2_K = cpst.fKs_l2
        K2 = cpst.K2s_l2

    theta_coil = 2 * np.pi * w.quadpoints_theta
    zeta_coil = 2 * np.pi * w.quadpoints_phi
    zeta_coil = w.quadpoints_phi[: len(zeta_coil) // 4]
    ntheta_coil = len(theta_coil)
    nzeta_coil = len(zeta_coil)
    r_coil = np.zeros_like(theta_coil)
    single_valued_current_potential_thetazeta_allLambda = np.zeros_like(current_potential_allLambda)

    return [
        ntheta_coil, nzeta_coil, theta_coil, zeta_coil, r_coil,
        xm_coil, xn_coil, xm_potential, xn_potential,
        nfp, net_poloidal_current_Amperes, net_toroidal_current_Amperes,
        current_potential_allLambda, single_valued_current_potential_thetazeta_allLambda, phi_mn_allLambda,
        lambdas, chi2_B, chi2_K, K2,
    ]


def load_CP_and_geometries(
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
        nfp=s_plasma_fp.nfp, stellsym=True, mpol=s_plasma_fp.mpol, ntor=s_plasma_fp.ntor
    )
    s_plasma_full = s_plasma_full.from_nphi_ntheta(
        nfp=s_plasma_fp.nfp,
        ntheta=len(s_plasma_fp.quadpoints_theta),
        nphi=len(s_plasma_fp.quadpoints_phi) * s_plasma_fp.nfp,
        mpol=s_plasma_fp.mpol,
        ntor=s_plasma_fp.ntor,
        stellsym=True,
        range='full torus',
    )
    s_plasma_full.set_dofs(s_plasma_fp.x)

    s_coil_fp = SurfaceRZFourier(nfp=s_coil_full.nfp, stellsym=True, mpol=s_coil_full.mpol, ntor=s_coil_full.ntor)
    s_coil_fp = s_coil_fp.from_nphi_ntheta(
        nfp=s_coil_full.nfp,
        ntheta=len(s_coil_full.quadpoints_theta),
        nphi=len(s_coil_full.quadpoints_phi) // 4,
        mpol=s_coil_full.mpol,
        ntor=s_coil_full.ntor,
        stellsym=True,
        range='field period',
    )
    s_coil_fp.set_dofs(s_coil_full.x)

    if loadDOFsProperly:
        s_coil_fp = load_surface_dofs_properly(s=s_coil_full, s_new=s_coil_fp)

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


def removearray(L: list, arr: np.ndarray) -> None:
    """Remove the first list element that equals arr (by value)."""
    for ind in range(len(L)):
        if np.array_equal(L[ind], arr):
            L.pop(ind)
            return
    raise ValueError('array not found in list.')


def sortLevels(levels: list, points: list) -> tuple:
    """Sort levels and points by level value (ascending)."""
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
    if ctype == 'free':
        return lines
    closed = [is_periodic_lines(line) for line in lines]
    ret = []
    if ctype == 'wp':
        for i, c in enumerate(closed):
            if c:
                ret.append(lines[i])
    if ctype in ('mod', 'hel'):
        for i, c in enumerate(closed):
            if not c:
                ret.append(lines[i])
    return ret


def points_in_polygon(
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


def map_data_to_more_periods(
    xdata: np.ndarray,
    ydata: np.ndarray,
    zdata: np.ndarray,
    args: tuple,
) -> tuple:
    """
    Extend (θ, ζ, φ) data to a 3×3 periodic grid for contour extraction.

    Args:
        xdata, ydata: 1D θ and ζ arrays.
        zdata: 2D current potential on the base grid.
        args: (Ip, It, xm, xn, phi_cos, phi_sin, nfp).

    Returns:
        (xN, yN, ret) extended grid and data.
    """
    Ip, It, xm_potential, xn_potential, phi_cos, phi_sin, nfp = args
    xf = 2 * np.pi
    yf = 2 * np.pi / nfp[0]
    xN = np.hstack([xdata - xf, xdata, xdata + xf])
    yN = np.hstack([ydata - yf, ydata, ydata + yf])
    z0 = zdata.copy()
    XX, YY = np.meshgrid(xN, yN)
    NSV = (It / (2 * np.pi)) * XX + (Ip / (2 * np.pi)) * YY
    z0_tiled = np.tile(z0, (3, 3))
    ret = 1 * z0_tiled + 1 * NSV
    return (xN, yN, ret)


def map_data_to_more_periods_3x1(
    xdata: np.ndarray,
    ydata: np.ndarray,
    zdata: np.ndarray,
    args: tuple,
) -> tuple:
    """
    Extend (θ, ζ, φ) data to 3×1 in ζ for modular/helical contour extraction.

    One period in θ, three periods in ζ. Used to find halfway contours
    between open contours for current integration.

    Args:
        xdata, ydata: 1D θ and ζ arrays.
        zdata: 2D current potential on the base grid.
        args: (Ip, It, xm, xn, phi_cos, phi_sin, nfp).

    Returns:
        (xN, yN, ret) extended grid and data.
    """
    Ip, It, xm_potential, xn_potential, phi_cos, phi_sin, nfp = args
    yf = 2 * np.pi / nfp[0]
    xN = xdata
    yN = np.hstack([ydata - yf, ydata, ydata + yf])
    z0 = zdata.copy()
    XX, YY = np.meshgrid(xN, yN)
    NSV = (It / (2 * np.pi)) * XX + (Ip / (2 * np.pi)) * YY
    z0_tiled = np.tile(z0, (3, 1))
    ret = 1 * z0_tiled + 1 * NSV
    return (xN, yN, ret)


def ID_mod_hel(contours: list, tol: float = 0.05) -> tuple:
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
            ret.append('mod')
            ints.append(1)
        elif np.isclose(np.abs(th0 - thf), 2 * np.pi, rtol=1e-2):
            ret.append('hel')
            ints.append(2)
        else:
            ret.append('vf')
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
    names, idcs = ID_mod_hel(Open_contours)
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
        ax.set_xlabel(r'$\theta$', fontsize=45)
        ax.set_ylabel(r'$\zeta$', fontsize=45)
        ax.tick_params(axis='both', which='major', labelsize=30)
        for icoil in contours:
            plt.plot(icoil[:, 0], icoil[:, 1], linestyle='-', color='orangered', picker=True)
    for i in range(len(contours) - 1):
        eye = i % len(contours)
        eyep1 = (i + 1) % len(contours)
        coil1 = contours[eye]
        coil2 = contours[eyep1]
        Lvl1 = current_potential_at_point([coil1[0, 0], coil1[0, 1]], args)
        Lvl2 = current_potential_at_point([coil2[0, 0], coil2[0, 1]], args)
        Lvlnew = (Lvl1 + Lvl2) / 2
        CS = ax.contour(theta, zeta, cpd, [Lvlnew], linestyles='dashed')
        paths = _contour_paths(CS, 0)
        halfway_contours.append(paths[0])
    if not do_plot:
        ClosePlot(plt)
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
        i1_ret, i2_ret, BOOL = points_in_polygon([t, z], None, cont)
        coordinates = np.vstack([t[i1_ret], z[i2_ret]]).T
        COORDS.append(coordinates)
        func_vals = [current_potential_at_point(pt, args) for pt in coordinates]
        contour_value = current_potential_at_point(cont[0, :], args)
        Z = np.abs(contour_value - np.array(func_vals))
        idx = np.argmax(Z)
        max_closed_contours_func_val.append([coordinates[idx, :], np.max(Z)])
        closed_contours_func_vals.append(contour_value)

    if plot:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlabel(r'$\theta$', fontsize=35)
        ax.set_ylabel(r'$\zeta$', fontsize=35)
        cs = ['r', 'g', 'b', 'm']
        ss = ['s', 'o', 'x', '^']
        for i in range(len(closed_contours)):
            ax.plot(closed_contours[i][:, 0], closed_contours[i][:, 1], marker='', linestyle='--', color=cs[i % len(cs)])
            ax.plot(COORDS[i][:, 0], COORDS[i][:, 1], marker=ss[i % len(ss)], linestyle='', color=cs[i % len(cs)])
            ax.plot(max_closed_contours_func_val[i][0][0], max_closed_contours_func_val[i][0][1], marker='o', linestyle='', color='c')
        plt.show()

    wp_currents = []
    for i in range(len(closed_contours_func_vals)):
        wp_currents.append(max_closed_contours_func_val[i][1] - closed_contours_func_vals[i])
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
                func_vals[i, j, :] = np.array([
                    current_potential_at_point(cont1[0, :], args),
                    current_potential_at_point(test_pt, args),
                ])
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
        ax.set_title(r'$\kappa$ chosen contours', fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.set_xlabel(r'$\theta$', fontsize=45)
        ax.set_ylabel(r'$\zeta$', fontsize=45)
        for i in range(len(closed_contours)):
            M = f'${current_potential_at_point(closed_contours[i][0, :], args):.0f}$'
            ax.plot(closed_contours[i][0, 0], closed_contours[i][0, 1], marker=M, linestyle='--', color='r', markersize=50)
            ax.plot(closed_contours[i][:, 0], closed_contours[i][:, 1], marker='', linestyle='--', color='k')
        plt.show()
    return (wp_currents, nested_BOOL, func_vals, numContsContained)


# =============================================================================
# Geometry and curve conversion
# =============================================================================

def real_space(
    nharmonics: int,
    coeff_array: np.ndarray,
    polAng: np.ndarray,
    torAng: np.ndarray,
) -> tuple:
    """
    Map (θ, ζ) to (X, Y, R, Z) using REGCOIL-style RZFourier coefficients.

    Args:
        nharmonics: Number of Fourier modes.
        coeff_array: Coefficient array (REGCOIL format).
        polAng: Poloidal angles θ.
        torAng: Toroidal angles ζ.

    Returns:
        (X, Y, R, Z) Cartesian and cylindrical coordinates.
    """
    nn = len(torAng)
    RR = np.zeros(nn)
    ZZ = np.zeros(nn)
    XX = np.zeros(nn)
    YY = np.zeros(nn)
    nnum = 4 * coeff_array[:, 0]
    mnum = coeff_array[:, 1]
    crc = coeff_array[:, 2]
    crs = coeff_array[:, 4]
    czc = coeff_array[:, 5]
    czs = coeff_array[:, 3]
    for i in range(nn):
        for k in range(nharmonics):
            RR[i] += crc[k] * np.cos(mnum[k] * polAng[i] + nnum[k] * torAng[i]) + crs[k] * np.sin(mnum[k] * polAng[i] + nnum[k] * torAng[i])
            ZZ[i] += czc[k] * np.cos(mnum[k] * polAng[i] + nnum[k] * torAng[i]) + czs[k] * np.sin(mnum[k] * polAng[i] + nnum[k] * torAng[i])
        XX[i] = RR[i] * np.cos(torAng[i])
        YY[i] = RR[i] * np.sin(torAng[i])
    return (XX, YY, RR, ZZ)


def REGCOIL_line_XYZ_RZ(
    nharmonics: int,
    coeff_array: np.ndarray,
    polAng: np.ndarray,
    torAng: np.ndarray,
) -> tuple:
    """Map (θ, ζ) to (X, Y, R, Z) using REGCOIL coefficient format."""
    nn = len(torAng)
    RR = np.zeros(nn)
    ZZ = np.zeros(nn)
    XX = np.zeros(nn)
    YY = np.zeros(nn)
    nnum = coeff_array[:, 0]
    mnum = coeff_array[:, 1]
    crc = coeff_array[:, 2]
    crs = coeff_array[:, 3]
    czc = coeff_array[:, 4]
    czs = coeff_array[:, 5]
    for i in range(nn):
        for k in range(nharmonics):
            RR[i] += crc[k] * np.cos(mnum[k] * polAng[i] + nnum[k] * torAng[i]) + crs[k] * np.sin(mnum[k] * polAng[i] + nnum[k] * torAng[i])
            ZZ[i] += czc[k] * np.cos(mnum[k] * polAng[i] + nnum[k] * torAng[i]) + czs[k] * np.sin(mnum[k] * polAng[i] + nnum[k] * torAng[i])
        XX[i] = RR[i] * np.cos(torAng[i])
        YY[i] = RR[i] * np.sin(torAng[i])
    return (XX, YY, RR, ZZ)


def SIMSOPT_line_XYZ_RZ(surf, coords: tuple) -> tuple:
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


def gen_parametrization(xdata: np.ndarray) -> np.ndarray:
    """Arc-length parametrization s from cumulative |dx|."""
    ds = np.abs(np.diff(xdata))
    ds = np.insert(ds, 0, 0)
    s = np.cumsum(ds)
    return s


def splice_curve_fourier_coeffs(Ws: np.ndarray, Wc: np.ndarray) -> np.ndarray:
    """Interleave cos and sin coefficients for CurveXYZFourier (stellarator-symmetric)."""
    coeffs = np.empty(Ws.size + Wc.size - 1, dtype=Ws.dtype)
    coeffs[0::2] = Wc
    coeffs[1::2] = Ws[1:]
    return coeffs


def get1DFourierSinCosComps(
    numT: int,
    signal: np.ndarray,
    plot: tuple,
    forward: bool,
    trunc: int,
) -> list:
    """
    FFT-based Fourier (cos, sin) decomposition of a 1D signal.

    Args:
        numT: Number of sample points.
        signal: 1D signal.
        plot: (plot_error, plot_recreation).
        forward: Unused (legacy).
        trunc: Number of modes to keep (-1 for all).

    Returns:
        [Amat, Bmat, eM, recreation].
    """
    from scipy.fftpack import fft, fftfreq, fftshift

    pi = np.pi
    T = np.linspace(0, 2 * pi, numT, endpoint=False)
    signal2 = signal
    signal2_fft2 = fft(signal2)
    dT = T[1] - T[0]
    wavenumT = fftfreq(numT, dT)
    m = 2 * pi * wavenumT
    M = m.copy()
    power2 = np.sqrt(np.real(signal2_fft2)**2 + np.imag(signal2_fft2)**2) / (numT / 2)
    phase2 = np.arctan2(np.imag(signal2_fft2), np.real(signal2_fft2))
    Amps = power2
    varphi = -phase2
    A = Amps * np.cos(varphi)
    B = Amps * np.sin(varphi)
    Ash = fftshift(A)
    Bsh = fftshift(B)
    Msh = fftshift(M)
    Amat = Ash[numT // 2:]
    Bmat = Bsh[numT // 2:]
    eM = Msh[numT // 2:]
    Amat[0] /= 2
    Bmat[0] /= 2
    if trunc != -1:
        trunc = int(trunc)
        Amat = Amat[:trunc]
        Bmat = Bmat[:trunc]
        eM = eM[:trunc]
    recreation = np.zeros(len(signal), dtype=float)
    for i in range(numT):
        recreation[i] = np.sum(Amat * np.cos(eM * T[i]) + Bmat * np.sin(eM * T[i]))
    if plot[0] or plot[1]:
        plt.figure()
        plt.plot(T, np.abs(signal2 - recreation))
        plt.plot(T, np.abs(signal2))
        plt.show()
    return [Amat, Bmat, eM, recreation]


def getCurveOrder(xdata: np.ndarray, number_coeffs: int, verb: bool = False) -> int:
    """Find CurveXYZFourier order that matches given number of DOFs."""
    from simsopt.geo import CurveXYZFourier
    for i in range(100):
        curve = CurveXYZFourier(quadpoints=xdata, order=i)
        if verb:
            print('ORD', i, len(curve.x))
        if len(curve.x) == number_coeffs:
            return i
        if i > len(curve.x):
            print('Cannot match order! Check number of dofs passed to curve.')
            return -1
    if verb:
        print('NO MATCHING CURVE ORDER!')
    return -1


def writeToCurve(
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
        contour_xyz: [X, Y, R, Z] from SIMSOPT_line_XYZ_RZ or real_space.
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
        X, Y, R, Z = SIMSOPT_line_XYZ_RZ(winding_surface, [contour_theta, contour_zeta])
        contour_xyz = [X, Y, R, Z]
    X, Y, R, Z = contour_xyz
    XYZ = np.column_stack([X, Y, Z])

    # Arc-length parametrization along the 3D contour (not theta-zeta space)
    ds = np.sqrt(np.sum(np.diff(XYZ, axis=0)**2, axis=1))
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
    target_xyz = np.column_stack([
        interp1d(S, XYZ[:, 0], kind='linear', fill_value='extrapolate')(s_uniform),
        interp1d(S, XYZ[:, 1], kind='linear', fill_value='extrapolate')(s_uniform),
        interp1d(S, XYZ[:, 2], kind='linear', fill_value='extrapolate')(s_uniform),
    ])
    quadpoints = list(s_uniform)
    curve = CurveXYZFourier(quadpoints, fourier_trunc)
    curve.least_squares_fit(target_xyz)
    ORD = fourier_trunc

    # For helical coils: fix stellarator-symmetric coefficients (xs, yc, zc)
    if fix_stellarator_symmetry is None:
        fix_stellarator_symmetry = closed_3d and not closed_theta_zeta
    if fix_stellarator_symmetry:
        for m in range(1, ORD + 1):
            curve.fix(f'xs({m})')
        for m in range(ORD + 1):
            curve.fix(f'yc({m})')
            curve.fix(f'zc({m})')

    if plotting_args[0]:
        ax = curve.plot(show=False)
        ax.plot(target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2], marker='', color='r', markersize=10)
        XYZcurve = curve.gamma()
        ax.plot(XYZcurve[:, 0], XYZcurve[:, 1], XYZcurve[:, 2], marker='', color='k', markersize=10, linestyle='--')
        plt.show()
    return curve


def writeToCurve_helical(
    contour_xyz: list,
    fourier_order: int,
    nfp: int,
    stellsym: bool = True,
    ntor: int = 1,
    quadpoints=None,
) -> CurveXYZFourierSymmetries:
    """
    Convert a helical contour (already mapped to full torus) to CurveXYZFourierSymmetries.

    Uses arc-length parametrization along the 3D contour (preserving the original
    point order). The Fourier coefficients are restricted to obey nfp and
    stellarator symmetry.

    Args:
        contour_xyz: [X, Y, R, Z] from SIMSOPT_line_XYZ_RZ (extended helical contour).
        fourier_order: Number of Fourier modes for xhat, yhat, z.
        nfp: Field-period symmetry number.
        stellsym: If True, enforce stellarator symmetry.
        ntor: Toroidal winding number (default 1).
        quadpoints: Quadrature points for the curve (default: 2000).

    Returns:
        CurveXYZFourierSymmetries instance.
    """
    from math import gcd
    from simsopt.geo import CurveXYZFourierSymmetries

    if gcd(ntor, nfp) != 1:
        raise ValueError("ntor and nfp must be coprime")

    X, Y, R, Z = contour_xyz
    xyz = np.column_stack([X, Y, Z])

    # Arc-length parametrization along the 3D contour (preserve original order)
    ds = np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))
    close_tol = 1e-6 * (np.sum(ds) + 1e-12)
    if np.linalg.norm(xyz[-1] - xyz[0]) < close_tol:
        ds = np.append(ds, np.linalg.norm(xyz[-1] - xyz[0]))
    else:
        ds = np.append(ds, 0.0)
    S = np.cumsum(ds)
    S = S / S[-1] if S[-1] > 0 else np.linspace(0, 1, len(S), endpoint=False)
    theta = S.copy()
    theta[-1] = 1.0
    # Ensure closure for interpolation
    xyz_closed = xyz.copy()
    if np.linalg.norm(xyz[-1] - xyz[0]) >= close_tol:
        xyz_closed[-1] = xyz[0]

    # CurveXYZFourierSymmetries: x = xhat*cos(2π ntor θ) - yhat*sin(...), y = xhat*sin(...) + yhat*cos(...)
    # Inverse: xhat = x*cos(angle) + y*sin(angle), yhat = -x*sin(angle) + y*cos(angle)
    angle = 2 * np.pi * ntor * theta
    xhat = xyz_closed[:, 0] * np.cos(angle) + xyz_closed[:, 1] * np.sin(angle)
    yhat = -xyz_closed[:, 0] * np.sin(angle) + xyz_closed[:, 1] * np.cos(angle)
    z_vals = xyz_closed[:, 2]

    # Fit Fourier coefficients. CurveXYZFourierSymmetries (stellsym):
    #   xhat = xc0 + Σ_m xc_m cos(2π nfp m θ),  yhat = Σ_m ys_m sin(2π nfp m θ),  z = Σ_m zs_m sin(2π nfp m θ)
    order = min(fourier_order, max(1, len(theta) // 4))

    if quadpoints is None:
        quadpoints = np.linspace(0, 1, 2000, endpoint=False)
    elif isinstance(quadpoints, int):
        quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

    curve = CurveXYZFourierSymmetries(quadpoints, order, nfp, stellsym, ntor=ntor)

    # Trapezoid integration over [0, 1]; use endpoint=True so we integrate full period
    from scipy.interpolate import interp1d
    theta_fit = np.linspace(0, 1, len(theta), endpoint=True)
    theta_fit[-1] = 1.0
    # Periodic: f(0)=f(1); use fill_value for safety at boundaries
    xhat_interp = interp1d(theta_fit, xhat, kind="linear", fill_value=(xhat[0], xhat[-1]), bounds_error=False)
    yhat_interp = interp1d(theta_fit, yhat, kind="linear", fill_value=(yhat[0], yhat[-1]), bounds_error=False)
    z_interp = interp1d(theta_fit, z_vals, kind="linear", fill_value=(z_vals[0], z_vals[-1]), bounds_error=False)

    n_quad = max(4 * order * nfp, 128)
    th = np.linspace(0, 1, n_quad + 1, endpoint=True)
    xh = xhat_interp(th)
    yh = yhat_interp(th)
    zh = z_interp(th)

    dofs = np.zeros(curve.num_dofs())
    idx = 0
    # xc0 = ∫_0^1 xhat(θ) dθ
    dofs[idx] = np.trapezoid(xh, th)
    idx += 1
    # xc_m = 2 ∫_0^1 xhat(θ) cos(2π nfp m θ) dθ  for m=1..order
    for m in range(1, order + 1):
        dofs[idx] = 2 * np.trapezoid(xh * np.cos(2 * np.pi * nfp * m * th), th)
        idx += 1
    # ys_m = 2 ∫_0^1 yhat(θ) sin(2π nfp m θ) dθ
    for m in range(1, order + 1):
        dofs[idx] = 2 * np.trapezoid(yh * np.sin(2 * np.pi * nfp * m * th), th)
        idx += 1
    # zs_m = 2 ∫_0^1 z(θ) sin(2π nfp m θ) dθ
    for m in range(1, order + 1):
        dofs[idx] = 2 * np.trapezoid(zh * np.sin(2 * np.pi * nfp * m * th), th)
        idx += 1

    curve.set_dofs(dofs)
    return curve


# =============================================================================
# Plotting and I/O
# =============================================================================

def set_axes_equal(ax) -> None:
    """
    Make 3D axes have equal scale (spheres appear spherical).

    Workaround for matplotlib's set_aspect('equal') not working in 3D.
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


def box_field_period(ax, lims: list, ret: bool = False):
    """Draw a dashed box on ax for the field period [x1,y1] or [x0,x1,y0,y1]."""
    if len(lims) == 2:
        x0, y0 = 0, 0
        x1, y1 = lims
    else:
        x0, x1, y0, y1 = lims
    ax.plot([x0, x1], [y0, y0], color='k', linestyle='--', linewidth=3)
    ax.plot([x0, x1], [y1, y1], color='k', linestyle='--', linewidth=3)
    ax.plot([x0, x0], [y0, y1], color='k', linestyle='--', linewidth=3)
    ax.plot([x1, x1], [y0, y1], color='k', linestyle='--', linewidth=3)
    if ret:
        return np.vstack([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])


def _contour_paths(cdata, level_idx: int = 0) -> list:
    """Get contour path vertices (matplotlib version-agnostic)."""
    if hasattr(cdata, 'allsegs') and level_idx < len(cdata.allsegs):
        return list(cdata.allsegs[level_idx])
    return [p.vertices for p in cdata.collections[level_idx].get_paths()]


def ClosePlot(plt_module) -> None:
    """Close figure without blocking."""
    plt_module.show(block=False)
    plt_module.pause(0.0001)
    plt_module.close()


def open_writeto_file(filename: str, nfp: np.ndarray):
    """Open a coil output file (legacy format) and write header."""
    f = open(filename, 'w')
    f.write('periods ' + str(nfp[0]) + '\n')
    f.write('begin filament\n')
    f.write('mirror NIL\n')
    return f


def close_writeto_file(filehandler) -> None:
    """Close coil output file."""
    filehandler.write('END \n')
    filehandler.close()


def writeContourToFile(
    file,
    contour: np.ndarray,
    level: float,
    args: tuple,
) -> None:
    """Write one contour to a coil file (legacy format)."""
    mtotal, surface_array, ctype = args
    contour_theta = contour[:, 0]
    contour_zeta = contour[:, 1]
    X, Y, R, Z = real_space(mtotal, surface_array, contour_theta, contour_zeta)
    for ii in range(len(X) - 1):
        file.write(f'{X[ii]:23.15E} {Y[ii]:23.15E} {Z[ii]:23.15E} {level:23.15E}\n')
    file.write(f'{X[0]:23.15E} {Y[0]:23.15E} {Z[0]:23.15E} {0.0:23.15E} 1 {ctype}\n')


def read_surface(file_name: str) -> tuple:
    """Read REGCOIL-style surface from nescin file. Returns (nharmonics, coeff_array)."""
    with open(file_name, 'r') as fo:
        fo.readline()
        nharmonics = int(fo.readline().split()[0])
        fo.readline()
        fo.readline()
        read_array = np.zeros((nharmonics, 6))
        for i in range(nharmonics):
            element = fo.readline().split()
            for j in range(6):
                read_array[i, j] = float(element[j])
    return (nharmonics, read_array)


def curves_to_vtk_w_pointdata(
    curves: list,
    filename: str,
    close: bool = False,
    pointData: list = None,
) -> None:
    """
    Export curves to VTK with optional point data (e.g. currents).

    Requires pyevtk.
    """
    from pyevtk.hl import polyLinesToVTK

    def wrap(data):
        return np.concatenate([data, [data[0]]])

    if close:
        x = np.concatenate([wrap(c.gamma()[:, 0]) for c in curves])
        y = np.concatenate([wrap(c.gamma()[:, 1]) for c in curves])
        z = np.concatenate([wrap(c.gamma()[:, 2]) for c in curves])
        ppl = np.asarray([c.gamma().shape[0] + 1 for c in curves])
    else:
        x = np.concatenate([c.gamma()[:, 0] for c in curves])
        y = np.concatenate([c.gamma()[:, 1] for c in curves])
        z = np.concatenate([c.gamma()[:, 2] for c in curves])
        ppl = np.asarray([c.gamma().shape[0] for c in curves])
    if pointData is not None:
        data = np.concatenate([pointData[i] * np.ones((ppl[i],)) for i in range(len(curves))])
        polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={"Currents": data})
    else:
        polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl)


def load_surface_dofs_properly(s, s_new):
    """Copy surface Fourier coefficients with correct (m,n) indexing."""
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


def load_coils_from_json(file: str, do_plot: bool = False) -> tuple:
    """Load coils from a JSON file (simsopt save format). Returns (Curves, Currents, Coils)."""
    import json
    from simsopt.geo import CurveXYZFourier, plot
    from simsopt.field import Current, Coil

    with open(file) as f:
        data = json.load(f)
    Curves = []
    Currents = []
    for i in range(len(data)):
        x0 = data[i]['curve']['x0']
        order = data[i]['curve']['order']
        quap = data[i]['curve']['quadpoints']
        II = data[i]['current']['current']
        newCurve = CurveXYZFourier(quap, order)
        newCurve.x = x0
        Curves.append(newCurve)
        Currents.append(Current(II))
    Coils = [Coil(curv, curr) for (curv, curr) in zip(Curves, Currents)]
    if do_plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title(file, fontsize=10)
        plot(Coils, engine="matplotlib", ax=ax, close=True, show=True, linewidth=3.5)
        plt.show()
    return (Curves, Currents, Coils)


def generate_sunflower(number: int, radius: float, plot: bool = False) -> np.ndarray:
    """Generate sunflower (Fibonacci) point distribution in a disk. Returns 2xN array."""
    a = radius
    golden = (1 + 5**0.5) / 2
    _theta = (2 * np.pi / (golden**2)) * np.arange(number)
    c = a / np.sqrt(number)
    _r = c * np.sqrt(np.arange(number))
    Sunflower = np.zeros((2, number))
    Sunflower[0, :] = _r * np.cos(_theta)
    Sunflower[1, :] = _r * np.sin(_theta)
    return Sunflower


# Interactive contour selection callbacks (used by CutCoils example)
def make_onclick(ax, args, contours, theta_coil, zeta_coil, current_potential):
    """Factory for double-click contour selection callback."""

    def onclick(event):
        if event.dblclick:
            xdata, ydata = event.xdata, event.ydata
            click = np.array([xdata, ydata])
            tmp_value = current_potential_at_point(click, args)
            tmp_level = np.array([tmp_value])
            print(f'click coords theta,zeta=({xdata:.2f},{ydata:.2f}) with contour value {tmp_value:.3f}')
            cdata = plt.contour(theta_coil, zeta_coil, current_potential, tmp_level, linestyles='dashed')
            lines = _contour_paths(cdata, 0)
            minDists = [minDist(click, line) for line in lines]
            chosen_line_idx = np.argmin(minDists)
            chosen_line = [lines[chosen_line_idx]]
            for icoil in chosen_line:
                plt.plot(icoil[:, 0], icoil[:, 1], linestyle='--', color='r', picker=True)
            ax.figure.canvas.draw()
            contours.append(chosen_line[0])

    return onclick


def make_onpick(contours):
    """Factory for pick event (store picked artist)."""

    def onpick(event):
        plt.gca().picked_object = event.artist

    return onpick


def make_on_key(contours):
    """Factory for key press (delete picked contour)."""

    def on_key(event):
        if event.key == 'delete':
            ax = plt.gca()
            if hasattr(ax, 'picked_object') and ax.picked_object is not None:
                arr = ax.picked_object.get_xydata()
                removearray(contours, arr)
                ax.picked_object.remove()
                ax.picked_object = None
                ax.figure.canvas.draw()

    return on_key
