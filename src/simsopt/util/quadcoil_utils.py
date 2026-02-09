# Import packages.
import jax.numpy as jnp
import cvxpy
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
import os
import subprocess
import glob
import matplotlib.pyplot as plt
import itertools
from scipy.fft import rfft
from scipy import interpolate
from matplotlib import cm, colors
from simsopt.geo import SurfaceRZFourier
try:
    from shapely.geometry import LineString, MultiPolygon
    from shapely.ops import unary_union, polygonize
except ImportError:
    LineString = None
    unary_union = None
    polygonize = None

__all__ = [
    'avg_order_of_magnitude',
    'callable_RZ_conv',
    'callable_RZ_union',
    'change_cp_resolution',
    'find_most_similar',
    'gen_callable_winding_surface',
    'gen_conv_winding_surface',
    'gen_normal_winding_surface',
    'gen_union_winding_surface',
    'last_exact_i_X_list',
    'project_arr_cylindrical',
    'project_arr_coord',
    'run_nescoil',
    'run_nescoil_legacy',
    'self_outer_prod_matrix',
    'self_outer_prod_vec',
    'sdp_helper_expand_and_add_diag',
    'sdp_helper_get_elem',
    'sdp_helper_last_col',
    'sdp_helper_p',
    'sdp_helper_p_inequality',
    'shallow_copy_cp_and_set_dofs',
    'sin_or_cos',
    'cvxpy_create_integrated_L1_from_array',
    'cvxpy_create_Linf_from_array',
    'cvxpy_create_Linf_leq_from_array',
    'cvxpy_no_windowpane',
    'cvxpy_create_X',
    'plot_coil_contours',
    'plot_coil_Phi_IG',
    'plot_comps',
    'plot_trade_off',
    'generate_video',
    'squared_flux',
    'coil_zeta_theta_from_cp',
    'ifft_simsopt_legacy',
    'ifft_simsopt',
    'coil_xyz_from_cp',
    'load_curves_from_xyz_legacy',
    'load_curves_from_xyz',
    'load_coils_from_xyz',
    'cut_coil_from_cp',
    'staircase'
]


def avg_order_of_magnitude(x):
    """
    Estimating the maximum magnitude for normalizing. 
    The appended 1 prevents this from going to zero.

    Args:
        x: The input array.
    Returns:
        The average order of magnitude.
    """
    return (jnp.max(jnp.append(jnp.abs(x).flatten(), 1.)))

    # The below implementation is not robust in autdiff.
    # Remove zeros
    # x_nonzero = jnp.where(x==0, jnp.nan, x)
    # return(jnp.exp(jnp.nanmean(jnp.log(jnp.abs(x_nonzero)))))
# A helper method. When mode=0, calculates sin(x).
# Otherwise calculates cos(x)


def sin_or_cos(x, mode): 
    """
    """
    return jnp.where(mode == 1, jnp.sin(x), jnp.cos(x))

# Winding surface
# @SimsoptRequires(LineString is not None, "gen_union_winding_surface requires shapely package")


def callable_RZ_union(vertices_R, vertices_Z):
    """
    Frank please add documentation on what this does!

    Args:
        vertices_R: 
        vertices_Z:

    Returns:
        union_vertices_R_periodic:
        union_vertices_Z_periodic:
    """
    vertices_R_wrapped = jnp.pad(vertices_R, (0, 1), mode='wrap')
    vertices_Z_wrapped = jnp.pad(vertices_Z, (0, 1), mode='wrap')
    vertices_RZ_i = jnp.stack((vertices_R_wrapped, vertices_Z_wrapped)).T
    linestring = LineString(vertices_RZ_i)
    polygon = unary_union(polygonize(unary_union(linestring)))
    # This union may contain multiple polygons.
    # When this is the case, we keep only the largest.
    if type(polygon) is MultiPolygon:
        area_list = []
        poly_list = list(polygon.geoms)
        for p in poly_list:
            area_list.append(p.area)
        polygon = poly_list[jnp.argmax(area_list)]
    union_vertices = jnp.array(polygon.exterior.coords).T
    union_vertices_R_periodic = union_vertices[0]
    union_vertices_Z_periodic = union_vertices[1]
    return (
        union_vertices_R_periodic,
        union_vertices_Z_periodic,
    )


def callable_RZ_conv(vertices_R, vertices_Z):
    """
    Frank please add documentation on what this does!

    Args:
        vertices_R: 
        vertices_Z:
    Returns:
        vertices_R_periodic:
        vertices_Z_periodic:
    """
    ConvexHull_i = ConvexHull(
        jnp.array([
            vertices_R,
            vertices_Z
        ]).T
    )
    vertices_i = ConvexHull_i.vertices
    # Obtaining vertices
    vertices_R_new = vertices_R[vertices_i]
    vertices_Z_new = vertices_Z[vertices_i]
    vertices_R_periodic = jnp.append(vertices_R_new, vertices_R_new[0])
    vertices_Z_periodic = jnp.append(vertices_Z_new, vertices_Z_new[0])
    return (vertices_R_periodic, vertices_Z_periodic)


def gen_conv_winding_surface(
    plasma_surface,
    d_expand,
    mpol=None,
    ntor=None,
    n_phi=None,
    n_theta=None,
):
    """
    Generate a convex winding surface.

    Args:
        plasma_surface: 
            The plasma surface as an array of points
        d_expand: 
            The plasma-winding surface offset distance desired.
        mpol: 
            The number of poloidal modes.
        ntor: 
            The number of toroidal modes.
        n_phi: 
            The number of phi points.
        n_theta: 
            The number of theta points.
    """
    return (gen_callable_winding_surface(
        callable_RZ_conv,
        plasma_surface,
        d_expand,
        mpol,
        ntor,
        n_phi,
        n_theta,
    ))


def gen_union_winding_surface(
    plasma_surface,
    d_expand,
    mpol=None,
    ntor=None,
    n_phi=None,
    n_theta=None,
):
    """
    Generate a winding surface using the union technique.

    Args:
        plasma_surface: 
            The plasma surface as an array of points
        d_expand: 
            The plasma-winding surface offset distance desired.
        mpol: 
            The number of poloidal modes.
        ntor: 
            The number of toroidal modes.
        n_phi: 
            The number of phi points.
        n_theta: 
            The number of theta points.
    """
    return (gen_callable_winding_surface(
        callable_RZ_union,
        plasma_surface,
        d_expand,
        mpol,
        ntor,
        n_phi,
        n_theta,
    ))


def gen_callable_winding_surface(
    callable_process_and_wrap_RZ,
    plasma_surface,
    d_expand,
    mpol=None,
    ntor=None,
    n_phi=None,
    n_theta=None,
):
    """
    Generate a winding surface.
    lambda_RZ must converts two arrays containing R and Z (without repeating endpoints)
    into two arrays containing R and Z WITH REPEATING ENDPOINTS

    Args:
        callable_process_and_wrap_RZ: 
            The callable function that processes R and Z.
        plasma_surface: 
            The plasma surface as an array of points
        d_expand: 
            The plasma-winding surface offset distance desired.
        mpol: 
            The number of poloidal modes.
        ntor: 
            The number of toroidal modes.
        n_phi: 
            The number of phi points.
        n_theta: 
            The number of theta

    Returns:
        winding_surface_out: 
            The winding surface
    """
    from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier
    if mpol is None:
        mpol = plasma_surface.mpol
    if ntor is None:
        ntor = plasma_surface.ntor
    if n_phi is None:
        n_phi = len(plasma_surface.quadpoints_phi)
    if n_theta is None:
        n_theta = len(plasma_surface.quadpoints_theta)
    offset_surface = SurfaceRZFourier(
        nfp=plasma_surface.nfp,
        stellsym=plasma_surface.stellsym,
        mpol=plasma_surface.mpol,
        ntor=plasma_surface.ntor,
        quadpoints_phi=jnp.linspace(0, 1, n_phi, endpoint=False),
        quadpoints_theta=jnp.linspace(0, 1, n_theta, endpoint=False),
    )
    offset_surface.set_dofs(plasma_surface.to_RZFourier().get_dofs())
    offset_surface.extend_via_normal(d_expand)
    # A naively expanded surface usually has self-intersections
    # in the poloidal cross section when the plasma is bean-shaped.
    # To avoid this, we create a new surface by taking each poloidal cross section's
    # convex hull.
    gamma = offset_surface.gamma().copy()
    gamma_R = jnp.linalg.norm([gamma[:, :, 0], gamma[:, :, 1]], axis=0)
    gamma_Z = gamma[:, :, 2]
    gamma_new = jnp.zeros_like(gamma)

    for i_phi in range(gamma.shape[0]):
        phi_i = offset_surface.quadpoints_phi[i_phi]
        cross_sec_R_i = gamma_R[i_phi]
        cross_sec_Z_i = gamma_Z[i_phi]

        # Create periodic array for interpolation
        (
            vertices_R_periodic_i,
            vertices_Z_periodic_i
        ) = callable_process_and_wrap_RZ(cross_sec_R_i, cross_sec_Z_i)

        # Temporarily vertically the cross section.
        # For centering the theta=0 curve to be along
        # the projection of the axis to the outboard side.
        Z_center_i = jnp.average(vertices_Z_periodic_i[:-1])
        vertices_Z_periodic_i = vertices_Z_periodic_i - Z_center_i

        # Parameterize the series of vertices with
        # arc length
        delta_R = jnp.diff(vertices_R_periodic_i)
        delta_Z = jnp.diff(vertices_Z_periodic_i)
        segment_length = jnp.sqrt(delta_R**2 + delta_Z**2)
        arc_length = jnp.cumsum(segment_length)
        arc_length_periodic = jnp.concatenate(([0], arc_length))
        arc_length_periodic_norm = arc_length_periodic/arc_length_periodic[-1]
        # Interpolate
        spline_i = CubicSpline(
            arc_length_periodic_norm,
            jnp.array([vertices_R_periodic_i, vertices_Z_periodic_i]).T,
            bc_type='periodic'
        )
        # Calculating the phase shift in quadpoints_theta needed
        # to center the theta=0 point to the intersection between
        # the Z=0 plane and the ourboard of the winding surface.
        Z_roots = spline_i.roots(extrapolate=False)[-1]
        root_RZ_i = spline_i(Z_roots)
        root_R_i = root_RZ_i[:, 0]
        # Choose the outboard root as theta=0
        phase_shift = Z_roots[jnp.argmax(root_R_i)]
        # Re-calculate R and Z from uniformly spaced theta
        conv_gamma_RZ_i = spline_i(offset_surface.quadpoints_theta + phase_shift)
        conv_gamma_R_i = conv_gamma_RZ_i[:, 0]
        conv_gamma_Z_i = conv_gamma_RZ_i[:, 1]
        # Remove the temporary offset introduced earlier
        conv_gamma_Z_i = conv_gamma_Z_i + Z_center_i
        # Calculate X and Y
        conv_gamma_X_i = conv_gamma_R_i*jnp.cos(phi_i*jnp.pi*2)
        conv_gamma_Y_i = conv_gamma_R_i*jnp.sin(phi_i*jnp.pi*2)
        gamma_new[i_phi, :, 0] = conv_gamma_X_i
        gamma_new[i_phi, :, 1] = conv_gamma_Y_i
        gamma_new[i_phi, :, 2] = conv_gamma_Z_i

    # Fitting to XYZ tensor fourier surface
    winding_surface_new = SurfaceXYZTensorFourier(
        nfp=offset_surface.nfp,
        stellsym=offset_surface.stellsym,
        mpol=mpol,
        ntor=ntor,
        quadpoints_phi=offset_surface.quadpoints_phi,
        quadpoints_theta=offset_surface.quadpoints_theta,
    )
    winding_surface_new.least_squares_fit(gamma_new)
    winding_surface_new = winding_surface_new.to_RZFourier()

    # Copying to all field periods
    len_phi_full = len(plasma_surface.quadpoints_phi) * plasma_surface.nfp
    winding_surface_out = SurfaceRZFourier(
        nfp=plasma_surface.nfp,
        stellsym=plasma_surface.stellsym,
        mpol=mpol,
        ntor=ntor,
        quadpoints_phi=jnp.arange(len_phi_full)/len_phi_full,
        quadpoints_theta=plasma_surface.quadpoints_theta,
    )
    winding_surface_out.set_dofs(winding_surface_new.get_dofs())
    return (winding_surface_out)


def gen_normal_winding_surface(source_surface, d_expand):
    """
    Generate a winding surface by expanding the plasma surface
    along the normal direction.

    Assumes that source_surface only contains 1 field period!

    Args:
        source_surface: 
            The source surface.
        d_expand: 
            The plasma-winding surface offset distance desired.
    """
    # Expanding plasma surface to winding surface
    len_phi = len(source_surface.quadpoints_phi)
    len_phi_full_fp = len_phi * source_surface.nfp
    len_theta = len(source_surface.quadpoints_theta)

    winding_surface = SurfaceRZFourier(
        nfp=source_surface.nfp,
        stellsym=source_surface.stellsym,
        mpol=source_surface.mpol,
        ntor=source_surface.ntor,
        quadpoints_phi=jnp.arange(len_phi_full_fp)/len_phi_full_fp,
        quadpoints_theta=jnp.arange(len_theta)/len_theta,
    )
    winding_surface.set_dofs(source_surface.get_dofs())
    winding_surface.extend_via_normal(d_expand)
    return (winding_surface)

# Operator Projection


def project_arr_coord(
        operator,
        unit1, unit2, unit3):
    """
    Project a (n_phi, n_theta, 3, <shape>) array in a given basis (unit1, unit2, unit3) 
    with shape (n_phi, n_theta, 3). 
    Outputs: (n_phi, n_theta, 3, <shape>)
    Sample the first field period when one_field_period is True.

    Args:
        operator: 
            The operator to project.
        unit1: 
            The first unit vector.
        unit2: 
            The second unit vector.
        unit3: 
            The third unit vector

    Returns:
        operator_comp_arr: 
            The projected operator.
    """
    # Memorizing shape of the last dimensions of the array
    len_phi = operator.shape[0]
    len_theta = operator.shape[1]
    operator_shape_rest = list(operator.shape[3:])
    operator_reshaped = operator.reshape((len_phi, len_theta, 3, -1))
    # Calculating components
    # shape of operator is
    # (n_grid_phi, n_grid_theta, 3, n_dof, n_dof)
    # We take the dot product between K and unit vectors.
    operator_1 = jnp.sum(unit1[:, :, :, None]*operator_reshaped, axis=2)
    operator_2 = jnp.sum(unit2[:, :, :, None]*operator_reshaped, axis=2)
    operator_3 = jnp.sum(unit3[:, :, :, None]*operator_reshaped, axis=2)

    operator_1_nfp_recovered = operator_1.reshape([len_phi, len_theta] + operator_shape_rest)
    operator_2_nfp_recovered = operator_2.reshape([len_phi, len_theta] + operator_shape_rest)
    operator_3_nfp_recovered = operator_3.reshape([len_phi, len_theta] + operator_shape_rest)
    operator_comp_arr = jnp.stack([
        operator_1_nfp_recovered,
        operator_2_nfp_recovered,
        operator_3_nfp_recovered
    ], axis=2)
    return (operator_comp_arr)


def project_arr_cylindrical(
    gamma,
    operator,
):
    """
    Project a (n_phi, n_theta, 3, <shape>) array in cylindrical coordinates.

    Args:
        gamma: 
            The gamma array.
        operator: 
            The operator to project.
    Returns:
        operator_comp_arr: 
            The projected operator.
    """
    # Keeping only the x, y components
    r_unit = jnp.zeros_like(gamma)
    r_unit = r_unit.at[:, :, -1].set(0)
    # Calculating the norm and dividing the x, y components by it
    r_unit = r_unit.at[:, :, :-1].set(gamma[:, :, :-1] / jnp.linalg.norm(gamma, axis=2)[:, :, None])

    # Setting Z unit to 1
    z_unit = jnp.zeros_like(gamma)
    z_unit = z_unit.at[:, :, -1].set(1)

    phi_unit = jnp.cross(z_unit, r_unit)
    return (
        project_arr_coord(
            operator,
            unit1=r_unit,
            unit2=phi_unit,
            unit3=z_unit,
        )
    )

# Miscellaneous functions


def self_outer_prod_vec(arr_1d):
    """
    Calculates the outer product of a 1d array with itself.

    Args:
        arr_1d: 
            The 1d array.
    Returns:
        The outer product of the 1d array.
    """
    return (arr_1d[:, None]*arr_1d[None, :])


def self_outer_prod_matrix(arr_2d):
    """
    Calculates the outer product of a matrix with itself. 
    Has the effect (Return)@x@x = (Return)@(xx^T) = (Input@x)(Input@x)^T

    Args:
        arr_2d: 
            The 2d array.
    Returns:
        The outer product of the 2d array.
    """
    return (arr_2d[None, :, :, None] * arr_2d[:, None, None, :])


def run_nescoil_legacy(
        filename,
        mpol=4,
        ntor=4,
        coil_ntheta_res=1,
        coil_nzeta_res=1,
        plasma_ntheta_res=1,
        plasma_nzeta_res=1):
    """
    Loads a CurrentPotentialFourier, a CurrentPotentialSolve 
    and a CurrentPotentialFourier containing the NESCOIL result.

    Works for 
    '/simsopt/tests/test_files/regcoil_out.hsx.nc'
    '/simsopt/tests/test_files/regcoil_out.li383.nc'

    Args:
        filename: 
            The filename.
        mpol: 
            The number of poloidal modes.
        ntor: 
            The number of toroidal modes.
        coil_ntheta_res: 
            The coil theta resolution.
        coil_nzeta_res: 
            The coil zeta resolution.
        plasma_ntheta_res: 
            The plasma theta resolution.
        plasma_nzeta_res: 
            The plasma zeta resolution.

    Returns:
        cp: 
            The current potential.
        cpst: 
            The current potential solve.
        cp_opt: 
            The optimized current potential.
        optimized_phi_mn: 
            The optimized phi_mn.
    """
    from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
    # Load in low-resolution NCSX file from REGCOIL
    cp_temp = CurrentPotentialFourier.from_netcdf(filename, coil_ntheta_res, coil_nzeta_res)
    coil_nzeta_res *= cp_temp.nfp

    cpst = CurrentPotentialSolve.from_netcdf(
        filename, plasma_ntheta_res, plasma_nzeta_res, coil_ntheta_res, coil_nzeta_res
    )
    cp = CurrentPotentialFourier.from_netcdf(filename, coil_ntheta_res, coil_nzeta_res)

    # Overwrite low-resolution file with more mpol and ntor modes
    cp = CurrentPotentialFourier(
        cpst.winding_surface, mpol=mpol, ntor=ntor,
        net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
        stellsym=True)

    cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.Bnormal_plasma)
    # cpst = CurrentPotentialSolve(cp, cpst.plasma_surface, cpst.Bnormal_plasma, cpst.B_GI)

    # Discard L2 regularization for testing against linear relaxation
    lambda_reg = 0
    optimized_phi_mn, f_B, _ = cpst.solve_tikhonov(lam=lambda_reg)
    cp_opt = cpst.current_potential

    return (cp, cpst, cp_opt, optimized_phi_mn)


def run_nescoil(
        filename,
        mpol=4,
        ntor=4,
        d_expand_norm=2,
        coil_ntheta_res=1,
        coil_nzeta_res=1,
        plasma_ntheta_res=1,
        plasma_nzeta_res=1):
    """
    Loads a CurrentPotentialFourier, a CurrentPotentialSolve 
    and a CurrentPotentialFourier containing the NESCOIL result.

    Works for 
    '/simsopt/tests/test_files/regcoil_out.hsx.nc'
    '/simsopt/tests/test_files/regcoil_out.li383.nc'

    Args:
        filename: 
            The filename.
        mpol: 
            The number of poloidal modes.
        ntor: 
            The number of toroidal modes.
        d_expand_norm: 
            The (minor-radius normalized) plasma-winding surface offset distance desired.
        coil_ntheta_res: 
            The coil theta resolution.
        coil_nzeta_res: 
            The coil zeta resolution.
        plasma_ntheta_res: 
            The plasma theta resolution.
        plasma_nzeta_res: 
            The plasma zeta resolution.
    """
    # Load in low-resolution NCSX file from REGCOIL
    from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve
    cp_temp = CurrentPotentialFourier.from_netcdf(filename, coil_ntheta_res, coil_nzeta_res)
    coil_nzeta_res *= cp_temp.nfp

    cpst = CurrentPotentialSolve.from_netcdf(
        filename, plasma_ntheta_res, plasma_nzeta_res, coil_ntheta_res, coil_nzeta_res
    )

    plasma_surface = cpst.plasma_surface
    d_expand = d_expand_norm * plasma_surface.minor_radius()
    winding_surface_conv = gen_conv_winding_surface(plasma_surface, d_expand)

    # Overwrite low-resolution file with more mpol and ntor modes
    cp = CurrentPotentialFourier(
        winding_surface_conv, mpol=mpol, ntor=ntor,
        net_poloidal_current_amperes=cp_temp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp_temp.net_toroidal_current_amperes,
        stellsym=True)

    cpst = CurrentPotentialSolve(cp, plasma_surface, cpst.Bnormal_plasma)

    # Discard L2 regularization for testing against linear relaxation
    lambda_reg = 0
    optimized_phi_mn, f_B, _ = cpst.solve_tikhonov(lam=lambda_reg)
    cp_opt = cpst.current_potential

    return (cp, cpst, cp_opt, optimized_phi_mn)


def change_cp_resolution(cp, n_phi: int, n_theta: int):
    """
    Takes a CurrentPotentialFourier, keeps its Fourier
    components but changes the phi (zeta), theta grid numbers.

    Args:
        cp:
            The current potential.
        n_phi:
            The number of phi points.
        n_theta:
            The number of theta points.

    Returns:
        cp_new:
            The new current potential.
    """
    from simsopt.geo import SurfaceRZFourier
    from simsopt.field import CurrentPotentialFourier
    winding_surface_new = SurfaceRZFourier(
        nfp=cp.winding_surface.nfp,
        stellsym=cp.winding_surface.stellsym,
        mpol=cp.winding_surface.mpol,
        ntor=cp.winding_surface.ntor,
        quadpoints_phi=jnp.linspace(0, 1, n_phi),
        quadpoints_theta=jnp.linspace(0, 1, n_theta)
    )
    winding_surface_new.set_dofs(cp.winding_surface.get_dofs())

    cp_new = CurrentPotentialFourier(
        winding_surface_new,
        mpol=cp.mpol,
        ntor=cp.ntor,
        net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
        stellsym=True)
    return (cp_new)


"""
Conic helper methods
Helpful matrices used during the construction of SDP problems.
"""


def sdp_helper_last_col(n_item, n_X):
    """
    Creates an operator on X=
    Phi Phi^T, Phi
    Phi      , 1
    Tr(OX) = Phi^T A Phi + b Phi + c.
    def quad_operator(A, b, c, dim_phi):

    Trace of (this matrix)@X contains 
    only the n_item-th item in the 
    last row/col of X.

    Args:
        n_item: 
            The item number to pick out.
        n_X: 
            The size of X.

    Returns:
        The operator.
    """
    return (sdp_helper_get_elem(n_item, -1, n_X))


def sdp_helper_get_elem(a, b, n_X):
    """
    Trace of (this matrix)@X contains 
    only the (a, b) item in the 
    last row/col of X.

    Args:
        a: 
            The row index.
        b: 
            The column index.
        n_X: 
            The size of X.
    Returns:
        matrix: 
            The operator to compute.
    """
    matrix = jnp.zeros((n_X, n_X))
    matrix = matrix.at[a, b].set(1)
    return (matrix)


def sdp_helper_p_inequality(matrix_A, p_sign):
    """
    This matrix creates an (n+1, n+1) matrix B
    from (n, n) matrix A so that
    tr(B@X) = tr(A@Phi)+p_sign*p
    For defining the inequalities:
    tr(A_ij@Phi)-p <= b_ij
    tr(A_ij@Phi)+p >= b_ij

    Args:
        matrix_A: 
            The matrix A.
        p_sign: 
            The sign of p.
    Returns:
        An (n+1, n+1) matrix B
        from (n, n) matrix A so that
        tr(B@X) = tr(A@Phi)+p_sign*p
    """
    n_X = jnp.shape(matrix_A)[0]+1
    return (sdp_helper_expand_and_add_diag(matrix_A, -1, p_sign, n_X))


def sdp_helper_expand_and_add_diag(matrix_A, n_p, p_sign, n_X):
    """
    This matrix creates an (n_X, n_X) matrix B
    from (n, n) matrix A so that
    tr(B@X) = tr(A@Phi)+p_sign*p
    where p is the n_p-th diagonal element.
    For defining the inequalities:
    tr(A_ij@Phi)-p <= b_ij
    tr(A_ij@Phi)+p >= b_ij

    Args:
        matrix_A: 
            The matrix A.
        n_p: 
            The n_p-th diagonal element.
        p_sign: 
            The sign of p.
        n_X: 
            The size of X.

    Returns:
        matrix:
            An (n_X, n_X) matrix B
            from (n, n) matrix A so that
            tr(B@X) = tr(A@Phi)+p_sign*p
    """
    n_A = jnp.shape(matrix_A)[0]
    matrix = jnp.zeros((n_X, n_X))
    matrix[:n_A, :n_A] = matrix_A
    matrix[n_p, n_p] = p_sign
    return (matrix)


def sdp_helper_p(n_X):
    """
    This matrix C satisfies
    tr(C@X)=p (p is X's last diagonal element)
    n_X is the size of X

    Args:
        n_X: 
            The size of X.
    Returns:
        C:
            The matrix C.
    """
    return (sdp_helper_get_elem(-1, -1, n_X))

# Analysis helper methods


def last_exact_i_X_list(X_value_list, theshold=1e-5):
    """
    Obtaining a list of second greatest eigenvalue
    and find the index of the last item where the
    eigenvalue <= threshold

    Args:
        X_value_list: 
            The list of X values.
        theshold: 
            The threshold value.

    Returns:
        last_exact_i: 
            The last index where the eigenvalue <= threshold
    """
    second_max_eig_list = []
    eig_list = []
    for i in range(len(X_value_list)):
        eigvals, _ = jnp.linalg.eig(X_value_list[i][:-1, :-1])
        eig_list.append(eigvals)
        second_max_eig_list.append(jnp.sort(jnp.abs(eigvals))[-2])
    last_exact_i = jnp.argwhere(
        jnp.max(jnp.abs(eig_list)[:, 1:], axis=1) < theshold
    ).flatten()[-1]
    return (last_exact_i)


def find_most_similar(Phi_list, f, f_value):
    """
    Loop over a list of arrays, call a function f over its elements, 
    and find the index of Phi with the closest f(Phi) to f_value

    Args:
        Phi_list: 
            The list of Phi values.
        f: 
            The function to call.
        f_value: 
            The f value.
    Returns:
        most_similar_index:
            The most similar index.
    """
    min_f_diff = float('inf')
    most_similar_index = 0
    for i_case in range(len(Phi_list)):
        Phi_l2_i = Phi_list[i_case]
        f_i = f(Phi_l2_i)
        f_diff = jnp.abs(f_i - f_value)
        if f_diff < min_f_diff:
            most_similar_index = i_case
            min_f_diff = f_diff
    return (most_similar_index)


def shallow_copy_cp_and_set_dofs(cp, dofs):
    """
    Shallow copy a CurrentPotential and set new DOF.
    Note that the winding_surface is still the 
    same instance and will change when the original's
    is modified.

    Args:
        cp: 
            The current potential.
        dofs: 
            The new DOF.
    Returns:
        cp_new: 
            The new current potential.
    """
    from simsopt.field import CurrentPotentialFourier
    cp_new = CurrentPotentialFourier(
        cp.winding_surface,
        cp.net_poloidal_current_amperes,
        cp.net_toroidal_current_amperes,
        cp.nfp,
        cp.stellsym,
        cp.mpol,
        cp.ntor,
    )
    cp_new.set_dofs(dofs)  # 28 for w7x
    return (cp_new)

# CVXPY helper methods


def cvxpy_create_X(n_dof,):
    """
    Define and solve the CVXPY problem.
    Create a symmetric matrix variable.
    One additional row and col each are added
    they have blank elements except for 
    the diagonal element.
    X contains:
    Phi  , Phi y, 
    Phi y,     y, 

    Args:
        n_dof: 
            The number of degrees of freedom (- 1) in the cvxpy problem.
    Returns:
        X: 
            The cvxpy variable.
        constraints: 
            The cvxpy constraints.
    """
    X = cvxpy.Variable((n_dof+1, n_dof+1), symmetric=True)

    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]

    # This constraint sets the last diagonal item of
    # x\otimes x to 1. When a rank-1 solution is feasible,
    # the SDP will produce a rank-1 solution. This allows
    # us to exactly control y despite having no way
    # to constrain the Phi_i y terms.
    constraints += [
        cvxpy.trace(sdp_helper_get_elem(-1, -1, n_dof+1) @ X) == 1
    ]
    return (X, constraints)


def cvxpy_no_windowpane(cp, current_scale, X):
    """
    Constructing cvxpy constraints and variables necessary for an inequality
    constraint the only poloidal contours of phi are produced. This means no window panes.

    Args:
        cp: 
            The current potential.
        current_scale: 
            The current scale.
        X: 
            cvxpy Variable
    Returns:
        constraints: 
            A list of cvxpy constraints.
        K_theta_operator:
            The operator K_theta involved in the inequality constraint.
        K_theta_scale:
            The scale of K_theta.
    """
    from simsopt.objectives import K_theta, A_b_c_to_block_operator
    (A_K_theta, b_K_theta, c_K_theta) = K_theta(
        cp.net_poloidal_current_amperes,
        cp.quadpoints_phi,
        cp.quadpoints_theta,
        cp.nfp, cp.m, cp.n,
        cp.stellsym,
    )
    A_K_theta = A_K_theta.reshape((-1, A_K_theta.shape[-2], A_K_theta.shape[-1]))
    b_K_theta = b_K_theta.reshape((-1, b_K_theta.shape[-1]))
    c_K_theta = c_K_theta.flatten()
    K_theta_operator, K_theta_scale = A_b_c_to_block_operator(
        A=A_K_theta, b=b_K_theta, c=c_K_theta,
        current_scale=current_scale,
        normalize=True
    )
    constraints = []
    if False:
        # if cp.stellsym:
        loop_size = K_theta_operator.shape[0]
    else:
        loop_size = K_theta_operator.shape[0]
    # This if statement distinguishes the sign of
    # tot. pol. current. The total current should never
    # change sign.
    print('Testing net current sign')
    if cp.net_poloidal_current_amperes > 0:
        for i in range(loop_size):
            constraints.append(
                cvxpy.trace(
                    K_theta_operator[i, :, :] @ X
                ) >= 0
            )
    else:
        print('Net current is negative')
        for i in range(loop_size):
            constraints.append(
                cvxpy.trace(
                    K_theta_operator[i, :, :] @ X
                ) <= 0
            )
    return (constraints, K_theta_operator, K_theta_scale)


def cvxpy_create_integrated_L1_from_array(cpst, grid_3d_operator, X, stellsym):
    """
    Constructing cvxpy constraints and variables necessary for an
    L1 norm term.

    Args:
        grid_3d_operator: 
            Array, has shape (n_grid, 3, ndof+1, ndof+1)
        X: 
            cvxpy Variable
        stellsym: 
            Whether the grid the operator lives on has stellarator 
            symmetry.

    Returns:
        constraints: 
            A list of cvxpy constraints. 
        L1_comps_to_sum: 
            Adding a lam*cvxpy.sum(L1_comps_to_sum) term in the 
            objective adds an L1 norm term.
    """

    normN_prime = np.linalg.norm(cpst.winding_surface.normal(), axis=-1)
    normN_prime = normN_prime.flatten()
    # if stellsym:
    if False:
        loop_size = grid_3d_operator.shape[0]//2
        jacobian_prime = normN_prime[:normN_prime.shape[0]//cpst.winding_surface.nfp//2]
    else:
        loop_size = grid_3d_operator.shape[0]
        jacobian_prime = normN_prime[:normN_prime.shape[0]//cpst.winding_surface.nfp]
    # q is used for L1 norm.
    L1_comps_to_sum = cvxpy.Variable(loop_size*3, nonneg=True)

    constraints = []
    for i in range(loop_size):
        for j in range(3):
            # K dot nabla K L1
            if np.all(grid_3d_operator[i, j, :, :] == 0):
                continue
            constraints.append(
                cvxpy.trace(
                    jacobian_prime[i] * grid_3d_operator[i, j, :, :] @ X
                ) <= L1_comps_to_sum[3*i+j]
            )
            constraints.append(
                cvxpy.trace(
                    jacobian_prime[i] * grid_3d_operator[i, j, :, :] @ X
                ) >= -L1_comps_to_sum[3*i+j]
            )
    # The L1 norm is given by cvxpy.sum(L1_comps_to_sum)
    return (constraints, L1_comps_to_sum)


def cvxpy_create_Linf_from_array(grid_3d_operator, X, stellsym):
    """
    Constructing cvxpy constraints and variables necessary for an
    L-inf norm term.

    Args:
        grid_3d_operator: 
            Array, has shape (n_grid, 3, ndof+1, ndof+1)
        X: 
            cvxpy Variable
        stellsym: 
            Whether the grid the operator lives on has stellarator 
            symmetry.

    Returns:
        constraints: 
            A list of cvxpy constraints. 
        Linf: 
            Adding a lam*Linf term in the 
            objective adds an L1 norm term.
    """
    # if False:
    #     loop_size = grid_3d_operator.shape[0]//2
    # else:
    loop_size = grid_3d_operator.shape[0]

    Linf = cvxpy.Variable(nonneg=True)
    constraints = []
    for i in range(loop_size):
        for j in range(3):
            # K dot nabla K L1
            if np.all(grid_3d_operator[i, j, :, :] == 0):
                continue
            constraints.append(
                cvxpy.trace(
                    grid_3d_operator[i, j, :, :] @ X
                ) <= Linf
            )
            constraints.append(
                cvxpy.trace(
                    grid_3d_operator[i, j, :, :] @ X
                ) >= -Linf
            )
    return (constraints, Linf)


def cvxpy_create_Linf_leq_from_array(grid_3d_operator, grid_3d_operator_scale, grid_1d_operator, grid_1d_operator_scale, k_param, X, stellsym):
    """
    Constructing cvxpy constraints and variables necessary for the following constraint:

    -kg(x) <= ||f(x)||_\infty <= kg(x)

    Args:
        grid_3d_operator: Array, has shape (n_grid, 3, ndof+1, ndof+1)
        grid_3d_operator_scale: 
            The scale of the 3d operator
        grid_1d_operator: Array, has shape (n_grid, ndof+1, ndof+1)
        grid_1d_operator_scale:
            The scale of the 1d operator
        k_param:
            The k parameter.
        X: cvxpy Variable
        stellsym:
            Whether the grid the operator lives on has stellarator 
            symmetry.

    Returns:
        constraints: 
            A list of cvxpy constraints. 
    """
    # if False:
    #     loop_size = grid_3d_operator.shape[0]//2
    # else:
    loop_size = grid_3d_operator.shape[0]

    #k_param = cvxpy.Variable()
    k_param_eff = k_param * grid_1d_operator_scale / grid_3d_operator_scale
    constraints = []
    for i in range(loop_size):
        for j in range(3):
            # K dot nabla K L1
            if np.all(grid_3d_operator[i, j, :, :] == 0):
                continue
            # constraints.append(
            #     cvxpy.trace(grid_1d_operator[i, :, :] @ X)
            #     >=0
            # )
            constraints.append(
                cvxpy.trace(
                    (
                        grid_3d_operator[i, j, :, :]
                        - k_param_eff * grid_1d_operator[i, :, :]
                    ) @ X
                ) <= 0
            )
            constraints.append(
                cvxpy.trace(
                    (
                        -grid_3d_operator[i, j, :, :]
                        - k_param_eff * grid_1d_operator[i, :, :]
                    ) @ X
                ) <= 0
            )
    return (constraints)

# Plotting helper methods


def plot_coil_contours(
        cp_opt,
        nlevels=40,
        plot_sv_only=False,
        plot_1fp=False):
    """
    Plots a coil configuration on 3d surface. 
    Args:
        cp_opt:
            The optimized current potential class instantation.
        nlevels:
            The number of contour levels.
        plot_sv_only:
            When True, plots Phi_sv only.
        plot_1fp:
            When True, plot only the first field period.
    Returns:
        quad_contour_set:
            The pyplot contour object that was plotted.
        Phi:
            The current potential values.
    """
    theta1d, phi1d = cp_opt.quadpoints_theta, cp_opt.quadpoints_phi
    theta2d, phi2d = np.meshgrid(theta1d, phi1d)

    # Calculating Phi
    G = cp_opt.net_poloidal_current_amperes
    I = cp_opt.net_toroidal_current_amperes
    if plot_sv_only:
        Phi = cp_opt.Phi()
    else:
        Phi = cp_opt.Phi() \
            + phi2d*G \
            + theta2d*I
    print(phi2d.shape)
    # print(theta2d)
    phi2d = np.pad(phi2d, ((0, 0), (0, 1)), 'edge')
    phi2d = np.pad(phi2d, ((0, 1), (0, 0)), constant_values=1)
    theta2d = np.pad(theta2d, ((0, 1), (0, 0)), 'edge')
    theta2d = np.pad(theta2d, ((0, 0), (0, 1)), constant_values=1)
    Phi = np.pad(Phi, ((0, 1), (0, 1)), 'wrap')
    # Correct wrapping
    Phi[-1, :] += G
    Phi[:, -1] += I
    # print(theta2d)
    # Making 2d contour plot
    # Calculating the correct contour levels
    if plot_1fp:
        len_new = len(phi2d)//cp_opt.nfp
        phi2d = phi2d[:len_new, :]
        theta2d = theta2d[:len_new, :]
        Phi = Phi[:len_new, :]
    min_phi = np.min(Phi)
    max_phi = np.max(Phi)
    interval = (I + G) / nlevels / cp_opt.nfp
    min_interval_num = round(min_phi / interval)
    max_interval_num = round(max_phi / interval)
    levels = interval * np.arange(min_interval_num, max_interval_num+1)
    quad_contour_set = plt.contour(
        phi2d,
        theta2d,
        Phi,
        levels=levels,
        algorithm='threaded',
        cmap='plasma'
    )
    plt.xlabel('Toroidal angle')
    plt.ylabel('Poloidal angle')
    return (quad_contour_set, Phi)


def plot_coil_Phi_IG(
    cp_opt,
    nlevels=40,
    plot_sv_only=False,
    plot_2d_contour=False,
    cmap=cm.plasma,
    **kwargs
):
    """
    Plots a coil configuration on 3d surface. 
    Args:
        cp_opt:
            The optimized current potential class instantion.
        nlevels:
            Number of contour levels to use.
        plot_sv_only:
            When True, plots Phi_sv only. 
        plot_2d_contour:
            When True, plots 2d contour also.
        cmap:
            The colormap to use.
        **kwargs:
            Additional arguments to pass to plt.figure.
    Returns:
        fig:
            The figure.
        ax:
            The figure axis.
    """
    theta1d, phi1d = cp_opt.quadpoints_theta, cp_opt.quadpoints_phi
    # Creating interpolation for mapping 2d contours onto 3d surface
    gamma = cp_opt.winding_surface.gamma()
    # Wrapping gamma and theta for periodic interpolation
    gamma_periodic = np.pad(gamma, ((0, 1), (0, 1), (0, 0)), 'wrap')
    theta1d_periodic = np.concatenate((theta1d, [1+theta1d[0]]))
    phi1d_periodic = np.concatenate((phi1d, [1+phi1d[0]]))
    # Wrapped meshgrid
    theta2d_periodic, phi2d_periodic = np.meshgrid(theta1d_periodic, phi1d_periodic)
    # Creating interpolation
    phi_theta_to_xyz = interpolate.LinearNDInterpolator(
        np.array([phi2d_periodic.flatten(), theta2d_periodic.flatten()]).T,
        gamma_periodic.reshape(-1, 3)
    )
    # Making 2d contour plot
    quad_contour_set, Phi = plot_coil_contours(
        cp_opt=cp_opt,
        nlevels=nlevels,
        plot_sv_only=plot_sv_only
    )
    if plot_2d_contour:
        plt.show()
    else:
        plt.close()
    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(projection='3d')

    norm = colors.Normalize(vmin=np.min(Phi), vmax=np.max(Phi), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    level_color = mapper.to_rgba(quad_contour_set.levels)

    # Phi contours:

    for i in range(len(quad_contour_set.allsegs)):
        # Loop over all contour levels
        seg_i = quad_contour_set.allsegs[i]
        if len(seg_i) > 0:
            # seg_i[kind_i==1] = np.nan
            list_of_levels = [
                list(g) for m, g in itertools.groupby(
                    seg_i, key=lambda x: not np.all(np.isnan(x))
                ) if m
            ]
            for level in list_of_levels:
                # A level ideally contains only one segment.
                for segment in level:
                    xyz_seg_i = phi_theta_to_xyz(segment)
                    ax.plot(
                        xyz_seg_i[:, 0],
                        xyz_seg_i[:, 1],
                        xyz_seg_i[:, 2],
                        # facecolors=facecolors
                        c=level_color[i],
                        linewidth=0.5
                    )

    ax.axis('equal')
    return (fig, ax)


def plot_comps(cp, comps, clim_lower, clim_upper):
    """
    Plot the components of K??? (Frank please clarify)

    Args:
        cp: 
            The current potential, used to just get the winding surface quadpoints.
        comps: 
            The components of K
        clim_lower: 
            The lower limit of the colorbar.
        clim_upper: 
            The upper limit of the colorbar
    """
    len_phi = len(cp.winding_surface.quadpoints_phi)//cp.winding_surface.nfp
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    ax1 = axes[0]
    _pcolor1 = ax1.pcolor(
        cp.winding_surface.quadpoints_phi[:len_phi]*np.pi*2,
        cp.winding_surface.quadpoints_theta*np.pi*2,
        comps[:, :, 0],
        cmap='seismic',
        vmin=clim_lower,
        vmax=clim_upper
    )

    ax2 = axes[1]
    _pcolor2 = ax2.pcolor(
        cp.winding_surface.quadpoints_phi[:len_phi]*np.pi*2,
        cp.winding_surface.quadpoints_theta*np.pi*2,
        comps[:, :, 1],
        cmap='seismic',
        vmin=clim_lower,
        vmax=clim_upper
    )

    ax3 = axes[2]
    pcolor3 = ax3.pcolor(
        cp.winding_surface.quadpoints_phi[:len_phi]*np.pi*2,
        cp.winding_surface.quadpoints_theta*np.pi*2,
        comps[:, :, 2],
        cmap='seismic',
        vmin=clim_lower,
        vmax=clim_upper
    )
    fig.text(0.5, 0, r'Toroidal angle $\zeta$', ha='center')
    fig.text(0.08, 0.5, r'Poloidal angle $\theta$', va='center', rotation='vertical')
    cb_ax = fig.add_axes([0.91, 0.05, 0.01, 0.9])
    _cbar = fig.colorbar(pcolor3, cax=cb_ax, label=r'$(K\cdot\nabla K)_{R, \phi, Z} (A^2/m^3)$')
    plt.show()
    print('Max comp:', np.max(np.abs(comps)))
    print('Avg comp:', np.average(np.abs(comps)))
    print('Max l2:', np.max(np.linalg.norm(comps, axis=-1)))
    print('Avg l2:', np.average(np.linalg.norm(comps, axis=-1)))


def plot_trade_off(
    Phi_list,
    f_x, f_y,
    xlabel, ylabel,
    plot=False,
    **kwargs
):
    """
    Plot the tradeoff in f_x and f_y for a list of Phi values.

    Args:
        Phi_list: 
            The list of Phi values.
        f_x: 
        f_y: 
        xlabel: 
            The x label.
        ylabel: 
            The y label.
        plot: 
            Whether to plot or scatter plot.
        **kwargs:
            Additional arguments.
    """
    x_list = list(map(f_x, Phi_list))
    y_list = list(map(f_y, Phi_list))
    if plot:
        plt.plot(x_list, y_list, **kwargs)
    else:
        plt.scatter(x_list, y_list, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def generate_video(cp, Phi_list, vid_name):
    """
    Generate a video of the current potential solution as a function
    of the iteration number in the algorithm.

    Args:
        cp: 
            The current potential.
        Phi_list: 
            The list of Phi values.
        vid_name: 
            The video name.
    """
    from simsopt.field import CurrentPotentialFourier
    # img_list = []
    for i in range(len(Phi_list)):
        cp_opt_temp_cp = CurrentPotentialFourier(
            cp.winding_surface,
            cp.net_poloidal_current_amperes,
            cp.net_toroidal_current_amperes,
            cp.nfp,
            cp.stellsym,
            cp.mpol,
            cp.ntor,
        )
        cp_opt_temp_cp.set_dofs(Phi_list[i])  # i_best_both
        fig, ax = plot_coil_Phi_IG(
            cp_opt=cp_opt_temp_cp,
            nlevels=100,
            plot_sv_only=False,
            plot_2d_contour=False
        )
        # img_list.append(ax)
    # ani = animation.ArtistAnimation(fig, img_list, interval=50, blit=True,
    #                         repeat_delay=1000)
        plt.savefig("%05d.vid_temp_file.png" % i, dpi=500)

    subprocess.call([
        'ffmpeg', '-y', '-framerate', '20', '-i', '%05d.vid_temp_file.png', '-r', '15', '-pix_fmt', 'yuv420p',
        vid_name+'.mp4'
    ])
    for file_name in glob.glob("*.vid_temp_file.png"):
        os.remove(file_name)


def squared_flux(coils, plasma_surface):
    """
    Wrapped for making a BiotSavart object from given coils,
    setting the points to the plasma surface, and evaluating
    the squared flux using this BiotSavart object.

    Args:
        coils: 
            The coils.
        plasma_surface: 
            The plasma surface.
    Returns:
        squared_flux: 
            The squared flux associated with the magnetic field
            generated by the given coils, evaluated at the given surface.
    """
    from simsopt.field import BiotSavart
    from simsopt.objectives import SquaredFlux
    bs = BiotSavart(coils)
    # Evaluate on surface
    bs.set_points(plasma_surface.gamma().reshape((-1, 3)))
    return (SquaredFlux(plasma_surface, bs).J())


def coil_zeta_theta_from_cp(
        cp,
        coilsPerHalfPeriod=1,
        thetaShift=0):
    """
    Get (zeta, theta) contours from the current potential.

    Args:
        cp:
            The current potential.
        coilsPerHalfPeriod:
            The number of coils per half period.
        thetaShift:
            The theta shift.
    Returns:
        contour_zeta:
            The zeta contours.
        contour_theta:
            The theta contours.
    """
    nzeta_coil = len(cp.winding_surface.quadpoints_phi)
    # f.variables['nfp'][()]
    nfp = cp.winding_surface.nfp
    # f.variables['theta_coil'][()]
    theta = cp.winding_surface.quadpoints_theta * 2 * np.pi
    # f.variables['zeta_coil'][()]
    zeta = cp.winding_surface.quadpoints_phi[:nzeta_coil // nfp] * 2 * np.pi
    # f.variables['net_poloidal_current_amperes'][()]
    net_poloidal_current_amperes = cp.net_poloidal_current_amperes

    # ------------------------
    # Load current potential
    # ------------------------
    current_potential = np.copy(cp.Phi()[:nzeta_coil // nfp, :])\
        + cp.current_potential_secular[:nzeta_coil // nfp, :]

    if abs(net_poloidal_current_amperes) > np.finfo(float).eps:
        data = current_potential / net_poloidal_current_amperes * nfp
    else:
        data = current_potential / np.max(current_potential)

    # First apply 'roll' to be sure I use the same convention as numpy:
    theta = np.roll(theta, thetaShift)
    # Now just generate a new monotonic array with the correct first value:
    theta = theta[0] + np.linspace(0, 2*np.pi, len(theta), endpoint=False)
    data = np.roll(data, thetaShift, axis=1)

    d = 2*np.pi/nfp
    zeta_3 = np.concatenate((zeta-d, zeta, zeta+d))
    data_3 = np.concatenate((data-1, data, data+1))

    # Repeat with just the contours we care about:
    contours = np.linspace(0, 1, coilsPerHalfPeriod*2, endpoint=False)
    d = contours[1]-contours[0]
    contours = contours + d/2
    cdata = plt.contour(zeta_3, theta, np.transpose(data_3), contours, colors='k')
    plt.close()

    numCoilsFound = len(cdata.collections)
    if numCoilsFound != 2*coilsPerHalfPeriod:
        print("WARNING!!! The expected number of coils was not the number found.")

    contour_zeta = []
    contour_theta = []
    numCoils = 0
    for j in range(numCoilsFound):
        p = cdata.collections[j].get_paths()[0]
        v = p.vertices
        # Make sure the contours have increasing theta:
        if v[1, 1] < v[0, 1]:
            v = np.flipud(v)

        for jfp in range(nfp):
            d = 2*np.pi/nfp*jfp
            contour_zeta.append(v[:, 0]+d)
            contour_theta.append(v[:, 1])
            numCoils += 1
    print('numCoils', numCoils)
    print('contour_zeta', len(contour_zeta))
    print('contour_theta', len(contour_theta))
    return (contour_zeta, contour_theta)


def ifft_simsopt_legacy(x, order):
    """
    IFFT a array in real space to a sin/cos series used by sinsopt.geo.curve

    Args:
        x: 
            The array to IFFT.
        order: 
            The order of the IFFT.
    Returns:
        dof: 
            The coefficients of the sin/cos series.
    """
    assert len(x) >= 2*order  # the order of the fft is limited by the number of samples
    xf = rfft(x) / len(x)
    fft_0 = [xf[0].real]  # find the 0 order coefficient
    fft_cos = 2 * xf[1:order + 1].real  # find the cosine coefficients
    fft_sin = -2 * xf[:order + 1].imag  # find the sine coefficients
    combined_fft = np.concatenate([fft_sin, fft_0, fft_cos])
    return (combined_fft)


def ifft_simsopt(x, order):
    """
    IFFT a array in real space to a sin/cos series used by sinsopt.geo.curve

    Args:
        x: 
            The array to IFFT.
        order: 
            The order of the IFFT.
    Returns:
        dof: 
            The coefficients of the sin/cos series.
    """
    assert len(x) >= 2*order  # the order of the fft is limited by the number of samples
    xf = rfft(x) / len(x)
    fft_0 = [xf[0].real]  # find the 0 order coefficient
    fft_cos = 2 * xf[1:order + 1].real  # find the cosine coefficients
    fft_sin = (-2 * xf[:order + 1].imag)[1:]  # find the sine coefficients
    dof = np.zeros(order*2+1)
    dof[0] = fft_0[0]
    dof[1::2] = fft_sin
    dof[2::2] = fft_cos
    return (dof)


def coil_xyz_from_cp(
        cp,
        coilsPerHalfPeriod=1,
        thetaShift=0,
        save=False, save_name='placeholder'):
    """
    Get coils in CurveXYZFourier format from a CurrentPotentialFourier object.

    This script assumes the contours do not zig-zag back and forth across the theta=0 line,
    after shifting the current potential by thetaShift grid points.
    The nescin file is used to provide the coil winding surface, so make sure this is consistent with the regcoil run.
    ilambda is the index in the lambda scan which you want to select.
    def cut_coil(cp, cpst):
    filename = 'regcoil_out.li383.nc' # sys.argv[1]
    TODO: these tow have a lot of duplicate code. Clean up when finalizing how QUADCOIL is integrated.

    Args:
        cp: 
            The current potential.
        coilsPerHalfPeriod: 
            The number of coils per half period.
        thetaShift: 
            The theta shift.
        save: 
            Whether to save the coils.
        save_name: 
            The save name.
    """
    contour_zeta, contour_theta = coil_zeta_theta_from_cp(
        cp=cp,
        coilsPerHalfPeriod=coilsPerHalfPeriod,
        thetaShift=thetaShift
    )
    numCoils = len(contour_zeta)
    nfp = cp.winding_surface.nfp
    net_poloidal_current_amperes = cp.net_poloidal_current_amperes

    # ------------------------
    # Load surface shape
    # ------------------------
    contour_R = []
    contour_Z = []
    for j in range(numCoils):
        contour_R.append(np.zeros_like(contour_zeta[j]))
        contour_Z.append(np.zeros_like(contour_zeta[j]))

    surf = cp.winding_surface
    for m in range(surf.mpol+1):  # 0 to mpol
        for i in range(2*surf.ntor+1):
            n = i-surf.ntor
            crc = surf.rc[m, i]
            czs = surf.zs[m, i]
            if surf.stellsym:
                crs = 0
                czc = 0
            else:  # Returns ValueError for stellsym cases
                crs = surf.get_rs(m, n)
                czc = surf.get_zc(m, n)
            for j in range(numCoils):
                angle = m*contour_theta[j] - n*contour_zeta[j]*surf.nfp
                # Was filled with zeroes.
                # Are lists because contou lengths are not uniform.
                contour_R[j] = contour_R[j] + crc*np.cos(angle) + crs*np.sin(angle)
                contour_Z[j] = contour_Z[j] + czs*np.sin(angle) + czc*np.cos(angle)

    contour_X = []
    contour_Y = []
    maxR = 0
    for j in range(numCoils):
        maxR = np.max((maxR, np.max(contour_R[j])))
        contour_X.append(contour_R[j]*np.cos(contour_zeta[j]))
        contour_Y.append(contour_R[j]*np.sin(contour_zeta[j]))

    coilCurrent = net_poloidal_current_amperes / numCoils

    # # Find the point of minimum separation
    # minSeparation2=1.0e+20
    # #for whichCoil1 in [5*nfp]:
    # #    for whichCoil2 in [4*nfp]:
    # for whichCoil1 in range(numCoils):
    #     for whichCoil2 in range(whichCoil1):
    #         for whichPoint in range(len(contour_X[whichCoil1])):
    #             dx = contour_X[whichCoil1][whichPoint] - contour_X[whichCoil2]
    #             dy = contour_Y[whichCoil1][whichPoint] - contour_Y[whichCoil2]
    #             dz = contour_Z[whichCoil1][whichPoint] - contour_Z[whichCoil2]
    #             separation2 = dx*dx+dy*dy+dz*dz
    #             this_minSeparation2 = np.min(separation2)
    #             if this_minSeparation2<minSeparation2:
    #                 minSeparation2 = this_minSeparation2

    if save:
        coilsFilename = 'coils.'+save_name
        print("coilsFilename:", coilsFilename)
        # Write coils file
        f = open(coilsFilename, 'w')
        f.write('periods '+str(nfp)+'\n')
        f.write('begin filament\n')
        f.write('mirror NIL\n')

        for j in range(numCoils):
            N = len(contour_X[j])
            for k in range(N):
                f.write('{:14.22e} {:14.22e} {:14.22e} {:14.22e}\n'.format(contour_X[j][k], contour_Y[j][k], contour_Z[j][k], coilCurrent))
            # Close the loop
            k = 0
            f.write('{:14.22e} {:14.22e} {:14.22e} {:14.22e} 1 Modular\n'.format(contour_X[j][k], contour_Y[j][k], contour_Z[j][k], 0))

        f.write('end\n')
        f.close()
    return (
        contour_X,
        contour_Y,
        contour_Z,
        coilCurrent,
        # np.sqrt(minSeparation2)
    )

# Load coils from lists of arrays containing x, y, and z.


def load_curves_from_xyz_legacy(
        contour_X,
        contour_Y,
        contour_Z,
        order=None, ppp=20):
    """
    Load CurveXYZFourier coils from contours of the current potential in x, y, and z.

    Args:
        contour_X: 
            The x contour.
        contour_Y: 
            The y contour.
        contour_Z: 
            The z contour.
        coilCurrent: 
            The current in the coil.
        order: 
            The order of the coil Fourier series.
        ppp: 
            The number of points per period.

    Returns:
        coils: 
            The coils in CurveXYZFourier format.
    """
    from simsopt.geo import CurveXYZFourier
    if not order:
        order = float('inf')
        for i in range(len(contour_X)):
            xArr = contour_X[i]
            yArr = contour_Y[i]
            zArr = contour_Z[i]
            for x in [xArr, yArr, zArr]:
                if len(x)//2 < order:
                    order = len(x)//2
    coil_data = []
    # Compute the Fourier coefficients for each coil
    for i in range(len(contour_X)):
        xArr = contour_X[i]
        yArr = contour_Y[i]
        zArr = contour_Z[i]

        curves_Fourier = []
        # Compute the Fourier coefficients
        for x in [xArr, yArr, zArr]:
            combined_fft = ifft_simsopt_legacy(x, order)
            curves_Fourier.append(combined_fft)

        coil_data.append(np.concatenate(curves_Fourier))

    coil_data = np.asarray(coil_data)
    coil_data = coil_data.reshape(6 * len(contour_X), order + 1)  # There are 6 * order coefficients per coil
    coil_data = np.transpose(coil_data)

    assert coil_data.shape[1] % 6 == 0
    assert order <= coil_data.shape[0]-1

    num_coils = coil_data.shape[1] // 6
    coils = [CurveXYZFourier(order*ppp, order) for i in range(num_coils)]
    for ic in range(num_coils):
        dofs = coils[ic].dofs_matrix
        dofs[0][0] = coil_data[0, 6*ic + 1]
        dofs[1][0] = coil_data[0, 6*ic + 3]
        dofs[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, min(order, coil_data.shape[0] - 1)):
            dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].local_x = np.concatenate(dofs)
    return (coils)


def load_curves_from_xyz(
        contour_X,
        contour_Y,
        contour_Z,
        order=None, ppp=20):
    """
    Load CurveXYZFourier curves from contours of the current potential in x, y, and z.

    Args:
        contour_X: 
            The x contour.
        contour_Y: 
            The y contour.
        contour_Z: 
            The z contour.
        coilCurrent: 
            The current in the coil.
        order: 
            The order of the coil Fourier series.
        ppp: 
            The number of points per period.

    Returns:
        coils: 
            The coil curves in CurveXYZFourier format.
    """
    from simsopt.geo import CurveXYZFourier
    num_coils = len(contour_X)
    # Calculating order
    if not order:
        order = float('inf')
        for i in range(num_coils):
            xArr = contour_X[i]
            yArr = contour_Y[i]
            zArr = contour_Z[i]
            for x in [xArr, yArr, zArr]:
                if len(x)//2 < order:
                    order = len(x)//2

    coils = [CurveXYZFourier(order*ppp, order) for i in range(num_coils)]
    # Compute the Fourier coefficients for each coil
    for ic in range(num_coils):
        xArr = contour_X[ic]
        yArr = contour_Y[ic]
        zArr = contour_Z[ic]

        # Compute the Fourier coefficients
        dofs = []
        for x in [xArr, yArr, zArr]:
            dof_i = ifft_simsopt(x, order)
            dofs.append(dof_i)

        coils[ic].local_x = np.concatenate(dofs)
    return (coils)


def load_coils_from_xyz(
        contour_X,
        contour_Y,
        contour_Z,
        coilCurrent,
        order=None, ppp=20):
    """
    Load CurveXYZFourier coils from contours of the current potential in x, y, and z.

    Args:
        contour_X: 
            The x contour.
        contour_Y: 
            The y contour.
        contour_Z: 
            The z contour.
        coilCurrent: 
            The current in the coil.
        order: 
            The order of the coil Fourier series.
        ppp: 
            The number of points per period.

    Returns:
        coils: 
            The coils in CurveXYZFourier format.
    """
    from simsopt.field import Current, Coil
    curves = load_curves_from_xyz(
        contour_X=contour_X,
        contour_Y=contour_Y,
        contour_Z=contour_Z,
        order=order,
        ppp=ppp
    )
    coils = []
    for curve in curves:
        coils.append(Coil(curve, Current(coilCurrent)))
    return (coils)


def cut_coil_from_cp(
        cp, coilsPerHalfPeriod, thetaShift,
        method=coil_xyz_from_cp,
        order=10, ppp=40):
    """
    Cut coils from a current potential and its winding surface.

    Args:
        cp: 
            The current potential.
        coilsPerHalfPeriod: 
            The number of coils per half period.
        thetaShift: 
            The shift in theta.
        method: 
            The method to use to cut the coils.
        order: 
            The order of the coil Fourier series.
        ppp: 
            The number of points per period.

    Returns:
        coils: 
            The coils in CurveXYZFourier format.
    """
    (
        contour_X,
        contour_Y,
        contour_Z,
        coilCurrent,
        # min_separation
    ) = method(
        cp=cp,
        coilsPerHalfPeriod=coilsPerHalfPeriod,
        thetaShift=thetaShift,
        save=False
    )
    coils = load_coils_from_xyz(
        contour_X,
        contour_Y,
        contour_Z,
        coilCurrent,
        order=order,
        ppp=ppp)
    return (coils)

def staircase(
    x:np.float64,
    x_step=2,
    x_width=0.5,
    x_phase=1):
    """
    Performs a staircase interpolation on x.

    The staircase always average to y=x so 
    that phi is not scaled.
    x_width must be within (0, x_step).
    Setting x_width of 0 will make the grad zero,
    while it should've been a series of delta funcs.

    Args:
        x: 
            The "period" of the staircase
        x_step: 
            The width of slopes.
        x_width: 
            The width of the slopes.
        x_phase: 
            The location of the center of the first plateau.

    Returns:
        x_base + stair_residual + x_phase
    """
    x_base = x_step*np.floor((x-x_phase)/x_step)
    x_residual = x-x_base-x_phase
    stair_residual = np.interp(
        x=x_residual,
        xp=np.array([0, x_step/2-x_width/2, x_step/2+x_width/2, x_step]),
        fp=np.array([0, 0, x_step, x_step])
    )
    return(x_base+stair_residual+x_phase)