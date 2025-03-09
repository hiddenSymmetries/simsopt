import sys
sys.path.insert(1, '..')
import numpy as np
from simsopt.util import avg_order_of_magnitude, sin_or_cos, project_arr_cylindrical
from functools import partial
from jax import jit

__all__ = ['norm_helper', 'Kdash_helper', 'diff_helper',
           'grid_curvature_operator', 'grid_curvature_operator_cylindrical',
           'grid_curvature', 'grid_curvature_cylindrical',
           'f_B_operator_and_current_scale', 'K_operator_cylindrical',
           'AK_helper', 'AK_helper', 'K_helper', 'K', 'f_B']

# Copied from the grid_curvature_operator.py file


def grid_curvature_operator(
        cp,
        single_value_only: bool = True,
        L2_unit=False,
        current_scale=1):
    """
    Generates a (n_phi, n_theta, 3(xyz), dof, dof) bilinear 
    operator that calculates K cdot grad K on grid points
    specified by 
    cp.winding_surface.quadpoints_phi
    cp.winding_surface.quadpoints_theta
    from 2 sin/sin-cos Fourier current potentials 
    sin or cos(m theta - n phi) with m, n given by
    cp.m
    cp.n.

    Args: 
        cp: CurrentPotential class instantiation
            CurrentPotential object (Fourier representation)
            to optimize

        single_value_only: bool
            When `True`, treat I,G as free quantities as well. 
            the operator acts on the vector (Phi, I, G) rather than (Phi)
        L2_unit: bool
            When L2_unit=True, the resulting matrices 
            contains the surface element and jacobian for integrating K^2
            over the winding surface.
            When L2_unit=False, the resulting matrices calculates
            the actual components of K.
        current_scale: float
            Scaling factor to make the current magnitudes of order one.

    Returns: 
        A (n_phi, n_theta, 3(xyz), n_dof_phi, n_dof_phi) operator 
        that evaluates K dot grad K on grid points. When free_GI is True,
        the operator has shape (n_phi, n_theta, 3(xyz), n_dof_phi+2, n_dof_phi+2) 
    """

    # Initialize a simsopt winding surface
    winding_surface = cp.winding_surface
    (
        Kdash1_sv_op,
        Kdash2_sv_op,
        Kdash1_const,
        Kdash2_const
    ) = Kdash_helper(
        normal=winding_surface.normal(),
        gammadash1=winding_surface.gammadash1(),
        gammadash2=winding_surface.gammadash2(),
        gammadash1dash1=winding_surface.gammadash1dash1(),
        gammadash1dash2=winding_surface.gammadash1dash2(),
        gammadash2dash2=winding_surface.gammadash2dash2(),
        nfp=cp.nfp,
        cp_m=cp.m,
        cp_n=cp.n,
        net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
        quadpoints_phi=winding_surface.quadpoints_phi,
        quadpoints_theta=winding_surface.quadpoints_theta,
        stellsym=winding_surface.stellsym,
        current_scale=current_scale
    )
    (
        _,  # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        _,  # partial_phi_phi,
        _,  # partial_phi_theta,
        _,  # partial_theta_theta,
    ) = diff_helper(
        nfp=cp.nfp, cp_m=cp.m, cp_n=cp.n,
        quadpoints_phi=winding_surface.quadpoints_phi,
        quadpoints_theta=winding_surface.quadpoints_theta,
        stellsym=winding_surface.stellsym
    )

    # Pointwise product with partial r/partial phi or theta
    normN_prime_2d, inv_normN_prime_2d = norm_helper(winding_surface.normal())

    term_a_op = (trig_diff_m_i_n_i@partial_phi)[:, :, None, :, None]\
        * Kdash2_sv_op[:, :, :, None, :]
    term_b_op = (trig_diff_m_i_n_i@partial_theta)[:, :, None, :, None]\
        * Kdash1_sv_op[:, :, :, None, :]
    # Includes only the single-valued contributions
    K_dot_grad_K_operator_sv = (term_a_op-term_b_op) * inv_normN_prime_2d[:, :, None, None, None]

    if single_value_only:
        K_dot_grad_K_operator = K_dot_grad_K_operator_sv
        K_dot_grad_K_const = np.nan
    else:
        # NOTE: direction phi or 1 is poloidal in simsopt.
        G = cp.net_poloidal_current_amperes * current_scale
        I = cp.net_toroidal_current_amperes * current_scale

        # Constant component of K dot grad K
        # Shape: (n_phi, n_theta, 3(xyz))
        K_dot_grad_K_const = inv_normN_prime_2d[:, :, None] * (G * Kdash2_const - I * Kdash1_const)

        # Component of K dot grad K linear in Phi
        # Shape: (n_phi, n_theta, 3(xyz), n_dof)
        K_dot_grad_K_linear = inv_normN_prime_2d[:, :, None, None]*(
            (trig_diff_m_i_n_i@partial_phi)[:, :, None, :]*Kdash2_const[:, :, :, None]
            + G*Kdash2_sv_op
            - (trig_diff_m_i_n_i@partial_theta)[:, :, None, :]*Kdash1_const[:, :, :, None]
            - I*Kdash1_sv_op
        )

        K_dot_grad_K_operator = np.zeros((
            K_dot_grad_K_operator_sv.shape[0],
            K_dot_grad_K_operator_sv.shape[1],
            K_dot_grad_K_operator_sv.shape[2],
            K_dot_grad_K_operator_sv.shape[3]+1,
            K_dot_grad_K_operator_sv.shape[4]+1
        ))

        K_dot_grad_K_operator[:, :, :, :-1, :-1] = K_dot_grad_K_operator_sv
        K_dot_grad_K_operator[:, :, :, -1, :-1] = K_dot_grad_K_linear
        K_dot_grad_K_operator[:, :, :, -1, -1] = K_dot_grad_K_const

    # K cdot grad K is (term_a_op-term_b_op)/|N|.
    # Shape: (n_phi, n_theta, 3(xyz), dof, dof)
    # Multiply by sqrt grid spacing, so that it's L2 norm^2 is K's surface integral.
    if L2_unit:
        # Here, 1/sqrt(|N|) cancels with the Jacobian |N|
        # whenever the square of K dot grad K is integrated.
        dtheta_coil = (winding_surface.quadpoints_theta[1] - winding_surface.quadpoints_theta[0])
        dphi_coil = (winding_surface.quadpoints_phi[1] - winding_surface.quadpoints_phi[0])
        L2_scale = np.sqrt(
            dphi_coil * dtheta_coil * normN_prime_2d[:, :, None, None, None]
        )
        K_dot_grad_K_operator *= L2_scale

    # Symmetrize:
    # We only care about symmetric Phi bar bar.
    K_dot_grad_K_operator = (K_dot_grad_K_operator+np.swapaxes(K_dot_grad_K_operator, 3, 4))/2

    return (K_dot_grad_K_operator/current_scale)


def grid_curvature_operator_cylindrical(
    cp,
    current_scale,
    single_value_only: bool = False,
    normalize=True
):
    """
    Creates the K dot nabla K components in R, Phi, Z. Only 1 field period.

    Args:
        cp: CurrentPotentialFourier
            CurrentPotential object (Fourier representation)
            to optimize
        current_scale: float
            Scaling factor to make the current magnitudes of order one.
        single_value_only: bool
            When `True`, treat I,G as free quantities as well.
        normalize: bool
            When `True`, normalize the output by the average order of magnitude.
    """
    K_dot_grad_K = grid_curvature_operator(
        cp=cp,
        single_value_only=single_value_only,
        current_scale=current_scale,
        L2_unit=False
    )
    out = project_arr_cylindrical(
        gamma=cp.winding_surface.gamma(),
        operator=K_dot_grad_K,
    )
    # Keep only 1 fp
    out = out[:out.shape[0]//cp.nfp]
    if normalize:
        out_scale = avg_order_of_magnitude(out)
        out /= out_scale
    else:
        out_scale = 1
    return (out, out_scale)


@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
    'L2_unit',
])
def grid_curvature(
        normal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
        L2_unit=False
):
    """
    Generates a (n_phi, n_theta, 3(xyz), dof, dof) bilinear 
    operator that calculates K cdot grad K on grid points
    specified by 
    cp.winding_surface.quadpoints_phi
    cp.winding_surface.quadpoints_theta
    from 2 sin/sin-cos Fourier current potentials 
    sin or cos(m theta - n phi) with m, n given by
    cp_m
    cp_n.

    Main difference with grid_curvature_operator is that this
    function uses jit and does not rely on the simsopt winding
    surface object.

    Args: 
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
        gammadash1dash1: (n_phi, n_theta, 3) array
            Second partial phi to the winding surface
        gammadash1dash2: (n_phi, n_theta, 3) array
            Mixed partial phi-theta to the winding surface
        gammadash2dash2: (n_phi, n_theta, 3) array
            Second partial theta to the winding surface
        net_poloidal_current_amperes: float
            Poloidal current in amperes
        net_toroidal_current_amperes: float
            Toroidal current in amperes
        quadpoints_phi: (n_phi,) array
            Quadrature points in phi
        quadpoints_theta: (n_theta,) array
            Quadrature points in theta
        nfp: int
            Number of field periods
        cp_m: (n,) array
            Fourier modes m of the current potential 
        cp_n: (n,) array
            Fourier modes n of the current potential 
        stellsym: bool
            Whether the stellarator is stellarator symmetric
        L2_unit: bool
            When L2_unit=True, the resulting matrices
            contains the surface element and jacobian for integrating K^2
            over the winding surface.
            When L2_unit=False, the resulting matrices calculates
            the actual components of K.

    Returns:    
        A (n_phi, n_theta, 3(xyz), n_dof_phi, n_dof_phi) operator 
        that evaluates K dot grad K on grid points. When free_GI is True,
        the operator has shape (n_phi, n_theta, 3(xyz), n_dof_phi+2, n_dof_phi+2) 
    """

    # No longer calling the simsopt winding surface
    # which was not coded up with JAX functionality
    (
        Kdash1_sv_op,
        Kdash2_sv_op,
        Kdash1_const,
        Kdash2_const
    ) = Kdash_helper(
        normal=normal,
        gammadash1=gammadash1,
        gammadash2=gammadash2,
        gammadash1dash1=gammadash1dash1,
        gammadash1dash2=gammadash1dash2,
        gammadash2dash2=gammadash2dash2,
        nfp=nfp,
        cp_m=cp_m,
        cp_n=cp_n,
        net_poloidal_current_amperes=net_poloidal_current_amperes,
        net_toroidal_current_amperes=net_toroidal_current_amperes,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        stellsym=stellsym
    )
    (
        _,  # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        _,  # partial_phi_phi,
        _,  # partial_phi_theta,
        _,  # partial_theta_theta,
    ) = diff_helper(
        nfp=nfp, cp_m=cp_m, cp_n=cp_n,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        stellsym=stellsym
    )

    # Pointwise product with partial r/partial phi or theta
    normN_prime_2d, inv_normN_prime_2d = norm_helper(normal)

    term_a_op = (trig_diff_m_i_n_i@partial_phi)[:, :, None, :, None]\
        * Kdash2_sv_op[:, :, :, None, :]
    term_b_op = (trig_diff_m_i_n_i@partial_theta)[:, :, None, :, None]\
        * Kdash1_sv_op[:, :, :, None, :]
    # Includes only the single-valued contributions
    K_dot_grad_K_operator_sv = (term_a_op-term_b_op) * inv_normN_prime_2d[:, :, None, None, None]

    # NOTE: direction phi or 1 is poloidal in simsopt.
    G = net_poloidal_current_amperes
    I = net_toroidal_current_amperes

    # Constant component of K dot grad K
    # Shape: (n_phi, n_theta, 3(xyz))
    K_dot_grad_K_const = inv_normN_prime_2d[:, :, None]*(G*Kdash2_const - I*Kdash1_const)

    # Component of K dot grad K linear in Phi
    # Shape: (n_phi, n_theta, 3(xyz), n_dof)
    K_dot_grad_K_linear = inv_normN_prime_2d[:, :, None, None]*(
        (trig_diff_m_i_n_i@partial_phi)[:, :, None, :]*Kdash2_const[:, :, :, None]
        + G*Kdash2_sv_op
        - (trig_diff_m_i_n_i@partial_theta)[:, :, None, :]*Kdash1_const[:, :, :, None]
        - I*Kdash1_sv_op
    )

    # K cdot grad K is (term_a_op-term_b_op)/|N|.
    # Shape: (n_phi, n_theta, 3(xyz), dof, dof)
    # Multiply by sqrt grid spacing, so that it's L2 norm^2 is K's surface integral.
    if L2_unit:
        # Here, 1/sqrt(|N|) cancels with the Jacobian |N|
        # whenever the square of K dot grad K is integrated.
        dtheta_coil = (quadpoints_theta[1] - quadpoints_theta[0])
        dphi_coil = (quadpoints_phi[1] - quadpoints_phi[0])
        L2_scale = jnp.sqrt(
            dphi_coil * dtheta_coil * normN_prime_2d[:, :, None, None, None]
        )
        K_dot_grad_K_operator_sv *= L2_scale
        K_dot_grad_K_linear *= L2_scale
        K_dot_grad_K_const *= L2_scale

    return (
        K_dot_grad_K_operator_sv,
        K_dot_grad_K_linear,
        K_dot_grad_K_const,
    )


@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
])
def grid_curvature_cylindrical(
        normal,
        gamma,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    """
    Main difference with grid_curvature_cylindrical_operator is that this
    function uses jit and does not rely on the simsopt winding
    surface object.

    Args: 
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
        gammadash1dash1: (n_phi, n_theta, 3) array
            Second partial phi to the winding surface
        gammadash1dash2: (n_phi, n_theta, 3) array
            Mixed partial phi-theta to the winding surface
        gammadash2dash2: (n_phi, n_theta, 3) array
            Second partial theta to the winding surface
        net_poloidal_current_amperes: float
            Poloidal current in amperes
        net_toroidal_current_amperes: float
            Toroidal current in amperes
        quadpoints_phi: (n_phi,) array
            Quadrature points in phi
        quadpoints_theta: (n_theta,) array
            Quadrature points in theta
        nfp: int
            Number of field periods
        cp_m: (n,) array
            Fourier modes m of the current potential 
        cp_n: (n,) array
            Fourier modes n of the current potential 
        stellsym: bool
            Whether the stellarator is stellarator symmetric
        L2_unit: bool
            When L2_unit=True, the resulting matrices
            contains the surface element and jacobian for integrating K^2
            over the winding surface.
            When L2_unit=False, the resulting matrices calculates
            the actual components of K.
    """
    (
        A_KK,
        b_KK,
        c_KK,
    ) = grid_curvature(
        normal=normal,
        gammadash1=gammadash1,
        gammadash2=gammadash2,
        gammadash1dash1=gammadash1dash1,
        gammadash1dash2=gammadash1dash2,
        gammadash2dash2=gammadash2dash2,
        net_poloidal_current_amperes=net_poloidal_current_amperes,
        net_toroidal_current_amperes=net_toroidal_current_amperes,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        nfp=nfp,
        cp_m=cp_m,
        cp_n=cp_n,
        stellsym=stellsym,
    )
    A_KK_cyl = project_arr_cylindrical(gamma=gamma, operator=A_KK)
    b_KK_cyl = project_arr_cylindrical(gamma=gamma, operator=b_KK)
    c_KK_cyl = project_arr_cylindrical(gamma=gamma, operator=c_KK)
    return (
        A_KK_cyl,
        b_KK_cyl,
        c_KK_cyl,
    )


def f_B_operator_and_current_scale(cpst, normalize=True, current_scale=None):
    """
    Produces a dimensionless f_B and K operator that act on X by 
    tr(AX). Also produces a scaling factor current_scale that is an 
    estimate for the order of magnitude of Phi.

    Args: 
        A: CurrentPotentialSolve class instantiation
            CurrentPotentialSolve object
    Returns:
        f_B_x_operator:
            An f_B operator with shape (nfod+1, ndof+1). Produces a 
            scalar when acted on X by tr(AX).
        Bnormal:
            The normal component of B on the plasma surface.
        current_scale:
            A scalar scaling factor that estimates the order of magnitude
            of Phi. Used when constructing other operators, 
            such as grid_curvature_operator_cylindrical
            and K_operator_cylindrical.
        f_B_scale:
            A scalar scaling factor that normalizes the f_B operator.
    """
    # Equivalent to A in regcoil.
    normN = np.linalg.norm(cpst.plasma_surface.normal().reshape(-1, 3), axis=-1)
    # The matrices may not have been precomputed
    try:
        B_normal = cpst.gj/np.sqrt(normN[:, None])
    except AttributeError:
        cpst.B_matrix_and_rhs()
        B_normal = cpst.gj/np.sqrt(normN[:, None])
    # Scaling factor to make X matrix dimensionless.
    if current_scale is None:
        current_scale = avg_order_of_magnitude(B_normal)/avg_order_of_magnitude(cpst.b_e)
    ''' f_B operator '''
    # Scaling blocks of the operator
    B_normal_scaled = B_normal/current_scale
    ATA_scaled = B_normal_scaled.T@B_normal_scaled
    ATb_scaled = B_normal_scaled.T@cpst.b_e
    bTb_scaled = np.dot(cpst.b_e, cpst.b_e)
    # Concatenating blocks into the operator
    f_B_x_operator = np.block([
        [ATA_scaled, -ATb_scaled[:, None]],
        [-ATb_scaled[None, :], bTb_scaled[None, None]]
    ])/2*cpst.current_potential.nfp
    if normalize:
        f_B_scale = avg_order_of_magnitude(f_B_x_operator)
        f_B_x_operator /= f_B_scale
    else:
        f_B_scale = 1
    return (f_B_x_operator, B_normal, current_scale, f_B_scale)


def K_operator_cylindrical(cp, current_scale, normalize=True):
    """
    Produces a dimensionless K operator that act on X by 
    tr(AX). Note that this operator is linear in Phi, rather
    than X.

    The K operator has shape (#grid per period, 3, nfod+1, ndof+1). 
    tr(A[i, j, :, :]X) cannot gives the grid value of a K component
    in (R, phi, Z). 

    It cannot directly produce a scalar objective, but can be used 
    for constructing norms (L1, L2). 

    Args:
        cp: CurrentPotentialFourier
            CurrentPotential object (Fourier representation)
            to optimize
        current_scale: float
            Scaling factor to make the current magnitudes of order one.
        normalize: bool
            When `True`, normalize the output by the average order of magnitude.
    """
    AK_operator, _ = K_operator(
        cp=cp,
        current_scale=current_scale,
        normalize=False
    )
    # Keep only 1 fp
    AK_operator_cylindrical = project_arr_cylindrical(
        gamma=cp.winding_surface.gamma(),
        operator=AK_operator
    )
    AK_operator_cylindrical = AK_operator_cylindrical[:AK_operator_cylindrical.shape[0]//cp.nfp]
    if normalize:
        AK_scale = avg_order_of_magnitude(AK_operator_cylindrical)
        AK_operator_cylindrical /= AK_scale
    return (AK_operator_cylindrical, AK_scale)


def AK_helper(cp):
    """
    We take advantage of the fj matrix already 
    implemented in CurrentPotentialSolve to calculate K.
    This is a helper method that applies the necessary units 
    and scaling factors. 

    When L2_unit=True, the resulting matrices 
    contains the surface element and jacobian for integrating K^2
    over the winding surface.

    When L2_unit=False, the resulting matrices calculates
    the actual components of K.

    Args:
        cp: CurrentPotentialFourier
            CurrentPotential object (Fourier representation)
            to optimize
    """
    winding_surface = cp.winding_surface
    (
        _,  # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        _,  # partial_phi_phi,
        _,  # partial_phi_theta,
        _,  # partial_theta_theta,
    ) = diff_helper(
        nfp=cp.nfp, cp_m=cp.m, cp_n=cp.n,
        quadpoints_phi=winding_surface.quadpoints_phi,
        quadpoints_theta=winding_surface.quadpoints_theta,
        stellsym=winding_surface.stellsym,
    )
    inv_normN_prime_2d = 1/np.linalg.norm(winding_surface.normal(), axis=-1)
    dg1 = winding_surface.gammadash1()
    dg2 = winding_surface.gammadash2()
    G = cp.net_poloidal_current_amperes
    I = cp.net_toroidal_current_amperes
    AK = inv_normN_prime_2d[:, :, None, None] * (
        dg2[:, :, :, None] * (trig_diff_m_i_n_i @ partial_phi)[:, :, None, :]
        - dg1[:, :, :, None] * (trig_diff_m_i_n_i @ partial_theta)[:, :, None, :]
    )
    bK = inv_normN_prime_2d[:, :, None] * (
        dg2 * G
        - dg1 * I
    )
    return (AK, bK)


def K_operator(cp, current_scale, normalize=True):
    """
    Produces a dimensionless K operator that act on X by 
    tr(AX). Note that this operator is linear in Phi, rather
    than X.

    The K operator has shape:
    (n_phi, n_theta, 3, ndof+1, ndof+1). 
    tr(A[i, j, :, :]X) cannot gives the grid value of a K component
    in (R, phi, Z). 

    It cannot directly produce a scalar objective, but can be used 
    for constructing norms (L1, L2). 

    Args:
        cp: CurrentPotentialFourier
            CurrentPotential object (Fourier representation)
            to optimize
        current_scale: float
            Scaling factor to make the current magnitudes of order one.
        normalize: bool
            When `True`, normalize the output by the average order of magnitude.
    """
    AK, bK = AK_helper(cp)
    # To fill the part of ther operator representing
    # 2nd order coefficients
    AK_blank_square = np.zeros(
        list(AK.shape)+[AK.shape[-1]]
    )
    AK_operator = np.block([
        [AK_blank_square, AK[:, :, :, :, None]/current_scale],
        [np.zeros_like(AK)[:, :, :, None, :], bK[:, :, :, None, None]]
    ])
    AK_operator = (AK_operator+np.swapaxes(AK_operator, -2, -1))/2
    if normalize:
        AK_scale = avg_order_of_magnitude(AK_operator)
        AK_operator /= AK_scale
    return (AK_operator, AK_scale)


def K_l2_operator(cp, current_scale, normalize=True):
    """
    An operator that calculates the L2 norm of K.
    Shape: (n_phi (1 field period) x n_theta, n_dof+1, n_dof+1)

    Args:
        cp: CurrentPotentialFourier
            CurrentPotential object (Fourier representation)
            to optimize
        current_scale: float
            Scaling factor to make the current magnitudes of order one.
        normalize: bool
            When `True`, normalize the output by the average order of magnitude.
    """
    AK, bK = AK_helper(cp)
    AK = AK[
        :AK.shape[0]//cp.nfp,
        :,
        :,
        :,
    ]
    # Take only one field period
    bK = bK[
        :bK.shape[0]//cp.nfp,
        :,
        :,
    ]
    # To fill the part of ther operator representing
    # 2nd order coefficients
    AK_scaled = (AK/current_scale)
    ATA_K_scaled = np.matmul(np.swapaxes(AK_scaled, -1, -2), AK_scaled)
    ATb_K_scaled = np.sum(AK_scaled*bK[:, :, :, None], axis=-2)
    bTb_K_scaled = np.sum(bK*bK, axis=-1)
    AK_l2_operator = np.block([
        [ATA_K_scaled, ATb_K_scaled[:, :, :, None]],
        [ATb_K_scaled[:, :, None, :], bTb_K_scaled[:, :, None, None]]
    ])
    AK_l2_operator = AK_l2_operator.reshape((
        -1,
        AK_l2_operator.shape[-2],
        AK_l2_operator.shape[-1]
    ))
    if normalize:
        AK_l2_scale = avg_order_of_magnitude(AK_l2_operator)
        AK_l2_operator /= AK_l2_scale
    else:
        AK_l2_scale = 1
    return (
        AK_l2_operator, AK_l2_scale,
    )


def K_theta(cp, current_scale, normalize=True):
    """
    K in the theta direction. Used to eliminate windowpane coils.  
    by K_theta >= -G. (Prevents current from flowing against G).

    Args:  
        cp: CurrentPotentialFourier
            CurrentPotential object (Fourier representation)
            to optimize
        current_scale: float
            Scaling factor to make the current magnitudes of order one.
        normalize: bool
            When `True`, normalize the output by the average order of magnitude.
    """
    n_harmonic = len(cp.m)
    iden = np.identity(n_harmonic)
    winding_surface = cp.winding_surface
    # When stellsym is enabled, Phi is a sin fourier series.
    # After a derivative, it becomes a cos fourier series.
    if winding_surface.stellsym:
        trig_choice = 1
    # Otherwise, it's a sin-cos series. After a derivative,
    # it becomes a cos-sin series.
    else:
        trig_choice = np.repeat([1, -1], n_harmonic//2)
    partial_phi = -cp.n*trig_choice*iden*cp.nfp*2*np.pi
    phi_grid = np.pi*2*winding_surface.quadpoints_phi[:, None]
    theta_grid = np.pi*2*winding_surface.quadpoints_theta[None, :]
    trig_diff_m_i_n_i = sin_or_cos(
        (cp.m)[None, None, :]*theta_grid[:, :, None]
        - (cp.n*cp.nfp)[None, None, :]*phi_grid[:, :, None],
        -trig_choice
    )
    K_theta_shaped = (trig_diff_m_i_n_i@partial_phi)
    # Take 1 field period
    K_theta_shaped = K_theta_shaped[:K_theta_shaped.shape[0]//cp.nfp, :]
    K_theta = K_theta_shaped.reshape((-1, K_theta_shaped.shape[-1]))
    K_theta_scaled = K_theta/current_scale
    ATA_scaled = np.zeros((K_theta_scaled.shape[0], K_theta_scaled.shape[1], K_theta_scaled.shape[1]))
    ATb_scaled = K_theta_scaled/2
    bTb_scaled = cp.net_poloidal_current_amperes*np.ones((K_theta_scaled.shape[0], 1, 1))
    # Concatenating blocks into the operator
    K_theta_operator = np.block([
        [ATA_scaled, ATb_scaled[:, :, None]],
        [ATb_scaled[:, None, :], bTb_scaled]
    ])
    if normalize:
        K_theta_scale = avg_order_of_magnitude(K_theta_operator)
        K_theta_operator /= K_theta_scale
    else:
        K_theta_scale = 1
    return (K_theta_operator, current_scale, K_theta_scale)


@partial(jit, static_argnames=[
    'nfp',
])
def f_B(
        gj, b_e,  # Quantities in CurrentPotential
        plasma_normal,
        nfp,
):
    """
    Produces a dimensionless f_B and K operator that act on X by 
    tr(AX). Also produces a scaling factor current_scale that is an 
    estimate for the order of magnitude of Phi.

    Args: 
        A: CurrentPotentialSolve class instantiation
    Returns:
        A_f_B: 
            The A matrix for the f_B operator.
        b_f_B:
            The b matrix for the f_B operator.
        c_f_B:
            The A matrix for the f_B operator.
        B_normal:
            The normal component of B on the plasma surface.
    """
    # Equivalent to A in regcoil.
    # normN = np.linalg.norm(cpst.plasma_surface.normal().reshape(-1, 3), axis=-1)
    normN = jnp.linalg.norm(plasma_normal.reshape(-1, 3), axis=-1)
    # The matrices may not have been precomputed
    B_normal = gj / jnp.sqrt(normN[:, None])
    # f_B operator
    ATA = B_normal.T@B_normal
    ATb = B_normal.T@b_e
    bTb = jnp.dot(b_e, b_e)
    A_f_B = ATA/2*nfp
    b_f_B = -ATb*nfp  # the factor of 2 cancelled
    c_f_B = bTb/2*nfp
    return (A_f_B, b_f_B, c_f_B, B_normal)


@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
])
def K_helper(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    """
    We take advantage of the fj matrix already 
    implemented in CurrentPotentialSolve to calculate K.
    This is a helper method that applies the necessary units 
    and scaling factors. 

    When L2_unit=True, the resulting matrices 
    contains the surface element and jacobian for integrating K^2
    over the winding surface.

    When L2_unit=False, the resulting matrices calculates
    the actual components of K.

    Args: 
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
        net_poloidal_current_amperes: float
            Poloidal current in amperes
        net_toroidal_current_amperes: float
            Toroidal current in amperes
        quadpoints_phi: (n_phi,) array
            Quadrature points in phi
        quadpoints_theta: (n_theta,) array
            Quadrature points in theta
        nfp: int
            Number of field periods
        cp_m: (n,) array
            Fourier modes m of the current potential 
        cp_n: (n,) array
            Fourier modes n of the current potential 
        stellsym: bool
            Whether the stellarator is stellarator symmetric
        L2_unit: bool
            When L2_unit=True, the resulting matrices
            contains the surface element and jacobian for integrating K^2
            over the winding surface.
            When L2_unit=False, the resulting matrices calculates
            the actual components of K.

    Returns:
        b_K: 
            The jax array for b in the K operator.
        c_K: 
            The jax array for c in the K operator.
    """
    (
        _,  # trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        _,  # partial_phi_phi,
        _,  # partial_phi_theta,
        _,  # partial_theta_theta,
    ) = diff_helper(
        nfp=nfp, cp_m=cp_m, cp_n=cp_n,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        stellsym=stellsym,
    )
    inv_normN_prime_2d = 1/jnp.linalg.norm(normal, axis=-1)
    dg1 = gammadash1
    dg2 = gammadash2
    G = net_poloidal_current_amperes
    I = net_toroidal_current_amperes
    b_K = inv_normN_prime_2d[:, :, None, None] * (
        dg2[:, :, :, None] * (trig_diff_m_i_n_i @ partial_phi)[:, :, None, :]
        - dg1[:, :, :, None] * (trig_diff_m_i_n_i @ partial_theta)[:, :, None, :]
    )
    c_K = inv_normN_prime_2d[:, :, None] * (
        dg2 * G
        - dg1 * I
    )
    return (b_K, c_K)


@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
])
def K(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    """
    Produces the A, b, c for K(phi). Because K is linear in phi, A is blank.

    It cannot directly produce a scalar objective, but can be used 
    for constructing norms (L1, L2). 

    Args: 
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
        net_poloidal_current_amperes: float
            Poloidal current in amperes
        net_toroidal_current_amperes: float
            Toroidal current in amperes
        quadpoints_phi: (n_phi,) array
            Quadrature points in phi
        quadpoints_theta: (n_theta,) array
            Quadrature points in theta
        nfp: int
            Number of field periods
        cp_m: (n,) array
            Fourier modes m of the current potential 
        cp_n: (n,) array
            Fourier modes n of the current potential 
        stellsym: bool
            Whether the stellarator is stellarator symmetric

    Returns:
        A_K: 
            The jax array for A in the K operator.
        b_K: 
            The jax array for b in the K operator.
        c_K: 
            The jax array for c in the K operator.
    """
    b_K, c_K = K_helper(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
    )
    # To fill the part of ther operator representing
    # 2nd order coefficients
    A_K = jnp.zeros(
        list(b_K.shape)+[b_K.shape[-1]]
    )
    return (A_K, b_K, c_K)


@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
])
def K2(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    """
    An operator that calculates the L2 norm of K.
    Shape: (n_phi (1 field period) x n_theta, n_dof+1, n_dof+1)

    Args: 
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
        net_poloidal_current_amperes: float
            Poloidal current in amperes
        net_toroidal_current_amperes: float
            Toroidal current in amperes
        quadpoints_phi: (n_phi,) array
            Quadrature points in phi
        quadpoints_theta: (n_theta,) array
            Quadrature points in theta
        nfp: int
            Number of field periods
        cp_m: (n,) array
            Fourier modes m of the current potential 
        cp_n: (n,) array
            Fourier modes n of the current potential 
        stellsym: bool
            Whether the stellarator is stellarator symmetric

    Returns:
        A_K2: 
            The jax array for A in the K^2 operator.
        b_K2:
            The jax array for b in the K^2 operator.
        c_K2: 
            The jax array for c in the K^2 operator.
    """
    AK, bK = K_helper(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
    )
    # To fill the part of ther operator representing
    # 2nd order coefficients
    A_K2 = jnp.matmul(jnp.swapaxes(AK, -1, -2), AK)
    b_K2 = 2*jnp.sum(AK * bK[:, :, :, None], axis=-2)
    c_K2 = jnp.sum(bK * bK, axis=-1)
    return (A_K2, b_K2, c_K2)


@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
])
def K_theta(
        net_poloidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    """
    K in the theta direction. Used to eliminate windowpane coils.  
    by K_theta >= -G. (Prevents current from flowing against G).


    Args: 
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        quadpoints_phi: (n_phi,) array
            Quadrature points in phi
        quadpoints_theta: (n_theta,) array
            Quadrature points in theta
        nfp: int
            Number of field periods
        cp_m: (n,) array
            Fourier modes m of the current potential 
        cp_n: (n,) array
            Fourier modes n of the current potential 
        stellsym: bool
            Whether the stellarator is stellarator symmetric

    Returns:
        A_K_theta:
            The jax array for A in the K_theta operator.
        b_K_theta:
            The jax array for b in the K_theta operator.
        c_K_theta:
            The jax array for c in the K_theta operator.
    """
    n_harmonic = len(cp_m)
    iden = jnp.identity(n_harmonic)
    # When stellsym is enabled, Phi is a sin fourier series.
    # After a derivative, it becomes a cos fourier series.
    if stellsym:
        trig_choice = 1
    # Otherwise, it's a sin-cos series. After a derivative,
    # it becomes a cos-sin series.
    else:
        trig_choice = jnp.repeat([1, -1], n_harmonic//2)
    partial_phi = -cp_n*trig_choice*iden*nfp*2*jnp.pi
    phi_grid = jnp.pi*2*quadpoints_phi[:, None]
    theta_grid = jnp.pi*2*quadpoints_theta[None, :]
    trig_diff_m_i_n_i = sin_or_cos(
        (cp_m)[None, None, :]*theta_grid[:, :, None]
        - (cp_n*nfp)[None, None, :]*phi_grid[:, :, None],
        -trig_choice
    )
    K_theta_shaped = (trig_diff_m_i_n_i@partial_phi)
    # Take 1 field period
    # K_theta_shaped = K_theta_shaped[:K_theta_shaped.shape[0]//nfp, :]
    # K_theta = K_theta_shaped.reshape((-1, K_theta_shaped.shape[-1]))
    K_theta = K_theta_shaped
    A_K_theta = jnp.zeros((K_theta.shape[0], K_theta.shape[1], K_theta.shape[2], K_theta.shape[2]))
    b_K_theta = K_theta
    c_K_theta = net_poloidal_current_amperes*jnp.ones((K_theta.shape[0], K_theta.shape[1]))
    return (A_K_theta, b_K_theta, c_K_theta)


def self_force_integrands_xyz(
        normal,
        unitnormal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        nfp, cp_m, cp_n,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        stellsym,
        AK_x_trimmed, bK_x_trimmed,
        current_scale):
    """
    Calculates the integrands in Robin, Volpe from a number of arrays.
    The arrays needs trimming compared to the outputs
    with a standard cp.
    The inputs are array properties of a surface object
    containing only one field period so that the code is easy to port 
    into c++.

    Calculates the nominators of the sheet current self-force in Robin, Volpe 2022.
    The K_y dependence is lifted outside the integrals. Therefore, the nominator 
    this function calculates are operators that acts on the QUADCOIL vector
    (scaled Phi, 1). The operator produces a 
    (n_phi_x, n_theta_x, 3(xyz, to act on Ky), 3(xyz), n_dof+1)
    After the integral, this will become a (n_phi_y, n_theta_y, 3, 3, n_dof+1)
    tensor that acts on K(y) to produce a vector with shape (n_phi_y, n_theta_y, 3, n_dof+1, n_dof+1)
    Shape: (n_phi_x, n_theta_x, 3(xyz), 3(xyz), n_dof+1(x)).

    Reminder: Do not use this with BIEST, because the x, y, z components of the vector field 
    has only one period, however many field periods that vector field has.

    Args: 
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
        gammadash1dash1: (n_phi, n_theta, 3) array
            Second partial phi to the winding surface
        gammadash1dash2: (n_phi, n_theta, 3) array
            Mixed partial phi-theta to the winding surface
        gammadash2dash2: (n_phi, n_theta, 3) array
            Second partial theta to the winding surface
        net_poloidal_current_amperes: float
            Poloidal current in amperes
        net_toroidal_current_amperes: float
            Toroidal current in amperes
        quadpoints_phi: (n_phi,) array
            Quadrature points in phi
        quadpoints_theta: (n_theta,) array
            Quadrature points in theta
        nfp: int
            Number of field periods
        cp_m: (n,) array
            Fourier modes m of the current potential 
        cp_n: (n,) array
            Fourier modes n of the current potential 
        stellsym: bool
            Whether the stellarator is stellarator symmetric
        Ak_x_trimmed: 
            The Kx operator, acts on the current potential harmonics (Phi).
            Shape: (n_phi_x, n_theta_x, 3, n_dof), (n_phi_x, n_theta_x, 3)
        bK_x_trimmed:
            The Kx operator, acts on the QUADCOIL vector (Phi/current_scale, 1).

    Returns:    
        K_x_op:
            The Kx operator, acts on the QUADCOIL vector (Phi/current_scale, 1).
            Shape: (n_phi_x, n_theta_x, 3, n_dof+1)
        integrand_single:
            The integrand for the single layer potential.
        integrand_double:
            The integrand for the double layer potential.
    """
    # Add a subscript to remind that this is the unit normal
    # as a function of x in the formula.
    unitnormal_x = unitnormal
    unitnormaldash1_x, unitnormaldash2_x = unitnormaldash(
        normal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
    )
    # An assortment of useful quantities related to K
    (
        Kdash1_sv_op_x,
        Kdash2_sv_op_x,
        Kdash1_const_x,
        Kdash2_const_x
    ) = Kdash_helper(
        normal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        nfp, cp_m, cp_n,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        stellsym,
        current_scale)
    # The Kx operator, acts on the current potential harmonics (Phi).
    # Shape: (n_phi_x, n_theta_x, 3, n_dof), (n_phi_x, n_theta_x, 3)
    # Contravariant basis of x
    grad1_x, grad2_x = grad_helper(gammadash1, gammadash2)
    # Unit normal for x

    # Operators that acts on the quadcoil vector.
    # Shape: (n_phi_x, n_theta_x, 3, n_dof+1)
    Kdash1_op_x = np.concatenate([
        Kdash1_sv_op_x,
        Kdash1_const_x[:, :, :, None]
    ], axis=-1)
    Kdash2_op_x = np.concatenate([
        Kdash2_sv_op_x,
        Kdash2_const_x[:, :, :, None]
    ], axis=-1)

    # The Kx operator, acts on the QUADCOIL vector (Phi/current_scale, 1).
    AK_x = AK_x_trimmed/current_scale
    K_x_op = np.concatenate([
        AK_x, bK_x_trimmed[:, :, :, None]
    ], axis=-1)

    ''' nabla_x cdot [pi_x K(y)] K(x) '''

    # divergence of the unit normal
    # Shape: (n_phi_x, n_theta_x)
    div_n_x = (
        np.sum(grad1_x * unitnormaldash1_x, axis=-1)
        + np.sum(grad2_x * unitnormaldash2_x, axis=-1)
    )

    ''' div_x pi_x '''
    # Shape: (n_phi_x, n_theta_x, 3)
    n_x_dot_grad_n_x = (
        np.sum(unitnormal_x * grad1_x, axis=-1)[:, :, None] * unitnormaldash1_x
        + np.sum(unitnormal_x * grad2_x, axis=-1)[:, :, None] * unitnormaldash2_x
    )
    # Shape: (n_phi_x, n_theta_x, 3)
    div_pi_x = -(
        div_n_x[:, :, None] * unitnormal_x
        + n_x_dot_grad_n_x
    )

    # Functions to integrate using the single and double layer
    # Laplacian kernels
    # 1e-7 is mu0/4pi
    integrand_single = 1e-7 * (
        # Term 1
        # n(x) div n K(x)
        # - (
        #     grad phi partial_phi
        #     + grad theta partial_theta
        # ) K(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        (
            unitnormal_x[:, :, :, None, None] * div_n_x[:, :, None, None, None] * K_x_op[:, :, None, :, :]
        )
        - (
            grad1_x[:, :, :, None, None] * Kdash1_op_x[:, :, None, :, :]
            + grad2_x[:, :, :, None, None] * Kdash2_op_x[:, :, None, :, :]
        )
        # Term 3
        # K(x) div pi_x
        # + partial_phi K(x) grad phi
        # + partial_theta K(x) grad theta
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        + (K_x_op[:, :, :, None, :] * div_pi_x[:, :, None, :, None])
        + (
            Kdash1_op_x[:, :, :, None, :] * grad1_x[:, :, None, :, None]
            + Kdash2_op_x[:, :, :, None, :] * grad2_x[:, :, None, :, None]
        )
    )

    integrand_double = 1e-7 * (
        # Term 2
        # n(x) K(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        (unitnormal_x[:, :, :, None, None] * K_x_op[:, :, None, :, :])
        # Term 4
        # K(x) n(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        - (K_x_op[:, :, :, None, :] * unitnormal_x[:, :, None, :, None])
    )

    return (
        K_x_op,
        integrand_single,
        integrand_double
    )


def self_force_cylindrical(cp, current_scale, normalize=True, skip_integral=False):
    """
    Calculates the self-force operator in cylindrical coordinates.

    Args:
        cp: CurrentPotentialFourier
            CurrentPotential object (Fourier representation)
            to optimize
        current_scale: float
            Scaling factor to make the current magnitudes of order one.
        normalize: bool
            When `True`, normalize the output by the average order of magnitude.
        skip_integral: bool
            When `True`, skip the integral calculation and return the integrands.

    Returns:
        self_force_cylindrical_operator:
            The self-force operator in cylindrical coordinates.
        self_force_scale:
            The scaling factor for the self-force operator.
    """
    winding_surface = cp.winding_surface
    nfp = cp.nfp
    len_phi_1fp = len(winding_surface.quadpoints_phi)//nfp
    AK, bK = AK_helper(cp)
    AK_trimmed = AK[:len_phi_1fp]
    bK_trimmed = bK[:len_phi_1fp]
    (
        K_x_op_xyz,
        integrand_single_xyz,
        integrand_double_xyz
    ) = self_force_integrands_xyz(
        normal=winding_surface.normal()[:len_phi_1fp],
        unitnormal=winding_surface.unitnormal()[:len_phi_1fp],
        gammadash1=winding_surface.gammadash1()[:len_phi_1fp],
        gammadash2=winding_surface.gammadash2()[:len_phi_1fp],
        gammadash1dash1=winding_surface.gammadash1dash1()[:len_phi_1fp],
        gammadash1dash2=winding_surface.gammadash1dash2()[:len_phi_1fp],
        gammadash2dash2=winding_surface.gammadash2dash2()[:len_phi_1fp],
        nfp=nfp,
        cp_m=cp.m,
        cp_n=cp.n,
        net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
        quadpoints_phi=winding_surface.quadpoints_phi[:len_phi_1fp],
        quadpoints_theta=winding_surface.quadpoints_theta,
        stellsym=winding_surface.stellsym,
        AK_x_trimmed=AK_trimmed,
        bK_x_trimmed=bK_trimmed,
        current_scale=current_scale
    )

    gamma_1fp = winding_surface.gamma()[:len_phi_1fp]
    # We must perform the xyz -> R,Phi,Z coordinate change twice for both
    # axis 2 and 3. Otherwise the operator will not have
    # the same nfp-fold discrete symmetry as the equilibrium.
    integrand_single_cylindrical = project_arr_cylindrical(
        gamma_1fp,
        integrand_single_xyz
    )
    integrand_double_cylindrical = project_arr_cylindrical(
        gamma_1fp,
        integrand_double_xyz
    )
    # The projection function assumes that the first 3 components of the array represents the
    # phi, theta grid and resulting components of the array. Hence the swapaxes.
    integrand_single_cylindrical = project_arr_cylindrical(
        gamma_1fp,
        integrand_single_cylindrical.swapaxes(2, 3)
    ).swapaxes(2, 3)
    integrand_double_cylindrical = project_arr_cylindrical(
        gamma_1fp,
        integrand_double_cylindrical.swapaxes(2, 3)
    ).swapaxes(2, 3)

    K_op_cylindrical = project_arr_cylindrical(
        gamma_1fp,
        K_x_op_xyz
    )

    if skip_integral:
        return (K_op_cylindrical, integrand_single_cylindrical, integrand_double_cylindrical)

    # Performing the singular integral using BIEST
    integrand_single_cylindrical_reshaped = integrand_single_cylindrical.reshape((
        integrand_single_cylindrical.shape[0],
        integrand_single_cylindrical.shape[1],
        -1
    ))
    integrand_double_cylindrical_reshaped = integrand_double_cylindrical.reshape((
        integrand_double_cylindrical.shape[0],
        integrand_double_cylindrical.shape[1],
        -1
    ))
    result_single = np.zeros_like(integrand_single_cylindrical_reshaped)
    result_double = np.zeros_like(integrand_double_cylindrical_reshaped)
    biest_call.integrate_multi(
        gamma_1fp,  # xt::pyarray<double> &gamma,
        integrand_single_cylindrical_reshaped,  # xt::pyarray<double> &func_in_single,
        result_single,  # xt::pyarray<double> &result,
        True,
        10,  # int digits,
        nfp,  # int nfp
    )
    biest_call.integrate_multi(
        gamma_1fp,  # xt::pyarray<double> &gamma,
        integrand_double_cylindrical_reshaped,  # xt::pyarray<double> &func_in_single,
        result_double,  # xt::pyarray<double> &result,
        False,
        10,  # int digits,
        nfp,  # int nfp
    )
    # BIEST's convention has an extra 1/4pi.
    # We remove it now, and reshape the output
    # into [n_phi(y), n_theta(y), 3(operates on K_y), 3, ndof].
    result_single = 4 * np.pi * result_single.reshape(
        result_single.shape[0],
        result_single.shape[1],
        3, 3, -1
    )
    # BIEST's convention has an extra 1/4pi.
    # We remove it now, and reshape the output
    # into [n_phi(y), n_theta(y), 3(operates on K_y), 3, ndof].
    result_double = 4 * np.pi * result_double.reshape(
        result_double.shape[0],
        result_double.shape[1],
        3, 3, -1
    )
    # The final operator
    # [n_phi(y), n_theta(y), 3, ndof]
    # The negative sign is added by BIEST (NEED CONFIRMATION)
    self_force_cylindrical_operator = np.sum(
        K_op_cylindrical[:, :, :, None, :, None]
        * (result_single - result_double)[:, :, :, :, None, :],
        axis=2
    )

    if normalize:
        self_force_scale = avg_order_of_magnitude(self_force_cylindrical_operator)
        self_force_cylindrical_operator /= self_force_scale
    else:
        self_force_scale = 1
    return (self_force_cylindrical_operator, self_force_scale)


@partial(jit, static_argnames=[
    'quadpoints_phi',
    'quadpoints_theta',
    'nfp', 'cp_m', 'cp_n',
    'stellsym',
])
def self_force_integrands_xyz(
        normal,
        unitnormal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
):
    """
    Calculates the integrands in Robin, Volpe from a number of arrays.
    The arrays needs trimming compared to the outputs
    with a standard cp.
    The inputs are array properties of a surface object
    containing only one field period so that the code is easy to port 
    into c++.

    Calculates the nominators of the sheet current self-force in Robin, Volpe 2022.
    The K_y dependence is lifted outside the integrals. Therefore, the nominator 
    is a linear function of phi. This function produces two arrays with shape

    integrand[n_phi_x, n_theta_x, 3(xyz, to act on Ky), 3(xyz), n_dof + 1]

    Here, the last slice of the last axis stores c, and the rest of the last axis stores b.
    This is to merge the single(double) layer laplacian integral in b and c in one 
    BIEST call.

    and the output from K_helper in the same concatenated form.

    Args: 
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
        gammadash1dash1: (n_phi, n_theta, 3) array
            Second partial phi to the winding surface
        gammadash1dash2: (n_phi, n_theta, 3) array
            Mixed partial phi-theta to the winding surface
        gammadash2dash2: (n_phi, n_theta, 3) array
            Second partial theta to the winding surface
        net_poloidal_current_amperes: float
            Poloidal current in amperes
        net_toroidal_current_amperes: float
            Toroidal current in amperes
        quadpoints_phi: (n_phi,) array
            Quadrature points in phi
        quadpoints_theta: (n_theta,) array
            Quadrature points in theta
        nfp: int
            Number of field periods
        cp_m: (n,) array
            Fourier modes m of the current potential 
        cp_n: (n,) array
            Fourier modes n of the current potential 
        stellsym: bool
            Whether the stellarator is stellarator symmetric
        Ak_x_trimmed: 
            The Kx operator, acts on the current potential harmonics (Phi).
            Shape: (n_phi_x, n_theta_x, 3, n_dof), (n_phi_x, n_theta_x, 3)
        bK_x_trimmed:
            The Kx operator, acts on the QUADCOIL vector (Phi/current_scale, 1).

    Returns:    
        K_concat:
            The Kx operator, acts on the QUADCOIL vector (Phi/current_scale, 1).
            Shape: (n_phi_x, n_theta_x, 3, n_dof+1)
        integrand_single_concat:
            The integrand for the single layer potential.
        integrand_double_concat:
            The integrand for the double layer potential.
    """
    len_phi_1fp = len(quadpoints_phi)//nfp
    normal = normal[:len_phi_1fp]
    unitnormal = unitnormal[:len_phi_1fp]
    gammadash1 = gammadash1[:len_phi_1fp]
    gammadash2 = gammadash2[:len_phi_1fp]
    gammadash1dash1 = gammadash1dash1[:len_phi_1fp]
    gammadash1dash2 = gammadash1dash2[:len_phi_1fp]
    gammadash2dash2 = gammadash2dash2[:len_phi_1fp]
    nfp = nfp
    cp_m = cp_m
    cp_n = cp_n
    net_poloidal_current_amperes = net_poloidal_current_amperes
    net_toroidal_current_amperes = net_toroidal_current_amperes
    quadpoints_phi = quadpoints_phi[:len_phi_1fp]
    quadpoints_theta = quadpoints_theta
    stellsym = stellsym
    b_K, c_K = K_helper(
        normal,
        gammadash1,
        gammadash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
    )
    b_K_trimmed = b_K[:len_phi_1fp]
    c_K_trimmed = c_K[:len_phi_1fp]

    # Add a subscript to remind that this is the unit normal
    # as a function of x in the formula.
    unitnormal_x = unitnormal
    unitnormaldash1_x, unitnormaldash2_x = unitnormaldash(
        normal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
    )
    # An assortment of useful quantities related to K
    (
        b_Kdash1,
        b_Kdash2,
        c_Kdash1,
        c_Kdash2
    ) = Kdash_helper(
        normal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        nfp, cp_m, cp_n,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        stellsym
    )
    # The Kx operator, acts on the current potential harmonics (Phi).
    # Shape: (n_phi_x, n_theta_x, 3, n_dof), (n_phi_x, n_theta_x, 3)
    # Contravariant basis of x
    grad1_x, grad2_x = grad_helper(gammadash1, gammadash2)
    # Unit normal for x
    # We temporarily concatenate b_K, c_K and b_Kdash, c_Kdash,
    # because they need to go through the same array operations.
    # Concatenating b and c will make it easier to check for typos.
    # The concatenated array is equivalent to an operator that
    # acts on the vector (phi, 1).
    # Shape: (n_phi_x, n_theta_x, 3, n_dof+1)
    Kdash1_op_x = jnp.concatenate([
        b_Kdash1,
        c_Kdash1[:, :, :, None]
    ], axis=-1)
    Kdash2_op_x = jnp.concatenate([
        b_Kdash2,
        c_Kdash2[:, :, :, None]
    ], axis=-1)
    K_concat = jnp.concatenate([
        b_K_trimmed, c_K_trimmed[:, :, :, None]
    ], axis=-1)

    ''' nabla_x cdot [pi_x K(y)] K(x) '''

    # divergence of the unit normal
    # Shape: (n_phi_x, n_theta_x)
    div_n_x = (
        jnp.sum(grad1_x * unitnormaldash1_x, axis=-1)
        + jnp.sum(grad2_x * unitnormaldash2_x, axis=-1)
    )

    ''' div_x pi_x '''
    # Shape: (n_phi_x, n_theta_x, 3)
    n_x_dot_grad_n_x = (
        jnp.sum(unitnormal_x * grad1_x, axis=-1)[:, :, None] * unitnormaldash1_x
        + jnp.sum(unitnormal_x * grad2_x, axis=-1)[:, :, None] * unitnormaldash2_x
    )
    # Shape: (n_phi_x, n_theta_x, 3)
    div_pi_x = -(
        div_n_x[:, :, None] * unitnormal_x
        + n_x_dot_grad_n_x
    )

    # Functions to integrate using the single and double layer
    # Laplacian kernels
    # 1e-7 is mu0/4pi
    integrand_single_concat = 1e-7 * (
        # Term 1
        # n(x) div n K(x)
        # - (
        #     grad phi partial_phi
        #     + grad theta partial_theta
        # ) K(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        (
            unitnormal_x[:, :, :, None, None] * div_n_x[:, :, None, None, None] * K_concat[:, :, None, :, :]
        )
        - (
            grad1_x[:, :, :, None, None] * Kdash1_op_x[:, :, None, :, :]
            + grad2_x[:, :, :, None, None] * Kdash2_op_x[:, :, None, :, :]
        )
        # Term 3
        # K(x) div pi_x
        # + partial_phi K(x) grad phi
        # + partial_theta K(x) grad theta
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        + (K_concat[:, :, :, None, :] * div_pi_x[:, :, None, :, None])
        + (
            Kdash1_op_x[:, :, :, None, :] * grad1_x[:, :, None, :, None]
            + Kdash2_op_x[:, :, :, None, :] * grad2_x[:, :, None, :, None]
        )
    )

    integrand_double_concat = 1e-7 * (
        # Term 2
        # n(x) K(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        (unitnormal_x[:, :, :, None, None] * K_concat[:, :, None, :, :])
        # Term 4
        # K(x) n(x)
        # Shape: (n_phi_x, n_theta_x, 3(xyz, acts on K(y)), 3(xyz), n_dof+1(x))
        - (K_concat[:, :, :, None, :] * unitnormal_x[:, :, None, :, None])
    )
    return (
        integrand_single_concat,
        integrand_double_concat,
        K_concat,
    )


def self_force_cylindrical_BIEST(
        normal,
        unitnormal,
        gamma,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        nfp, cp_m, cp_n,
        stellsym,
        skip_integral=False,
):
    """
    Args: 
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
        gammadash1dash1: (n_phi, n_theta, 3) array
            Second partial phi to the winding surface
        gammadash1dash2: (n_phi, n_theta, 3) array
            Mixed partial phi-theta to the winding surface
        gammadash2dash2: (n_phi, n_theta, 3) array
            Second partial theta to the winding surface
        net_poloidal_current_amperes: float
            Poloidal current in amperes
        net_toroidal_current_amperes: float
            Toroidal current in amperes
        quadpoints_phi: (n_phi,) array
            Quadrature points in phi
        quadpoints_theta: (n_theta,) array
            Quadrature points in theta
        nfp: int
            Number of field periods
        cp_m: (n,) array
            Fourier modes m of the current potential 
        cp_n: (n,) array
            Fourier modes n of the current potential 
        stellsym: bool
            Whether the stellarator is stellarator symmetric
        skip_integral: bool
            When `True`, skip the integral calculation and return the integrands.

    Returns:    
        K_concat:
            The Kx operator, acts on the QUADCOIL vector (Phi/current_scale, 1).
            Shape: (n_phi_x, n_theta_x, 3, n_dof+1)
        integrand_single_concat:
            The integrand for the single layer potential.
        integrand_double_concat:
            The integrand for the double layer potential.
    """
    nfp = nfp
    len_phi_1fp = len(quadpoints_phi)//nfp
    (
        integrand_single_concat,
        integrand_double_concat,
        K_concat,
    ) = self_force_integrands_xyz(
        normal=normal,
        unitnormal=unitnormal,
        gammadash1=gammadash1,
        gammadash2=gammadash2,
        gammadash1dash1=gammadash1dash1,
        gammadash1dash2=gammadash1dash2,
        gammadash2dash2=gammadash2dash2,
        net_poloidal_current_amperes=net_poloidal_current_amperes,
        net_toroidal_current_amperes=net_toroidal_current_amperes,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        nfp=nfp,
        cp_m=cp_m,
        cp_n=cp_n,
        stellsym=stellsym,
    )

    gamma_1fp = gamma[:len_phi_1fp]
    # We must perform the xyz -> R,Phi,Z coordinate change twice for both
    # axis 2 and 3. Otherwise the operator will not have
    # the same nfp-fold discrete symmetry as the equilibrium.
    integrand_single_concat_cylindrical = project_arr_cylindrical(gamma_1fp, integrand_single_concat)
    integrand_double_concat_cylindrical = project_arr_cylindrical(gamma_1fp, integrand_double_concat)
    # The projection function assumes that the first 3 components of the array represents the
    # phi, theta grid and resulting components of the array. Hence the swapaxes.
    integrand_single_concat_cylindrical = project_arr_cylindrical(
        gamma_1fp,
        integrand_single_concat_cylindrical.swapaxes(2, 3)
    ).swapaxes(2, 3)
    integrand_double_concat_cylindrical = project_arr_cylindrical(
        gamma_1fp,
        integrand_double_concat_cylindrical.swapaxes(2, 3)
    ).swapaxes(2, 3)
    K_concat_cylindrical = project_arr_cylindrical(gamma_1fp, K_concat)

    if skip_integral:
        return (K_concat, integrand_single_concat_cylindrical, integrand_double_concat_cylindrical)

    # Performing the singular integral using BIEST
    integrand_single_concat_cylindrical_reshaped = integrand_single_concat_cylindrical.reshape((
        integrand_single_concat_cylindrical.shape[0],
        integrand_single_concat_cylindrical.shape[1],
        -1
    ))
    integrand_double_concat_cylindrical_reshaped = integrand_double_concat_cylindrical.reshape((
        integrand_double_concat_cylindrical.shape[0],
        integrand_double_concat_cylindrical.shape[1],
        -1
    ))
    result_single_concat = jnp.zeros_like(integrand_single_concat_cylindrical_reshaped)
    result_double_concat = jnp.zeros_like(integrand_double_concat_cylindrical_reshaped)
    biest_call.integrate_multi(
        gamma_1fp,  # xt::pyarray<double> &gamma,
        integrand_single_concat_cylindrical_reshaped,  # xt::pyarray<double> &func_in_single,
        result_single_concat,  # xt::pyarray<double> &result,
        True,
        10,  # int digits,
        nfp,  # int nfp
    )
    biest_call.integrate_multi(
        gamma_1fp,  # xt::pyarray<double> &gamma,
        integrand_double_concat_cylindrical_reshaped,  # xt::pyarray<double> &func_in_single,
        result_double_concat,  # xt::pyarray<double> &result,
        False,
        10,  # int digits,
        nfp,  # int nfp
    )
    # BIEST's convention has an extra 1/4pi.
    # We remove it now, and reshape the output
    # into [n_phi(y), n_theta(y), 3(operates on K_y), 3, ndof+1].
    result_single_concat = 4 * jnp.pi * result_single_concat.reshape(
        result_single_concat.shape[0],
        result_single_concat.shape[1],
        3, 3, -1
    )
    # BIEST's convention has an extra 1/4pi.
    # We remove it now, and reshape the output
    # into [n_phi(y), n_theta(y), 3(operates on K_y), 3, ndof+1].
    result_double_concat = 4 * jnp.pi * result_double_concat.reshape(
        result_double_concat.shape[0],
        result_double_concat.shape[1],
        3, 3, -1
    )
    # To simplify the math, we construct a symmetric [n_phi(y), n_theta(y), 3, ndof+1, ndof+1].
    # operator that acts on (phi, 1),
    # then read off the individual blocks as A, b and c.
    self_force_cylindrical_O = jnp.sum(
        K_concat_cylindrical[:, :, :, None, :, None]
        * (result_single_concat - result_double_concat)[:, :, :, :, None, :],
        axis=2
    )
    self_force_cylindrical_O = (self_force_cylindrical_O + self_force_cylindrical_O.swapaxes(-1, -2))/2
    A_sf = self_force_cylindrical_O[:, :, :, :-1, :-1]
    b_sf = self_force_cylindrical_O[:, :, :, :-1, -1]
    c_sf = self_force_cylindrical_O[:, :, :, -1, -1]
    return (A_sf, b_sf, c_sf)

#### Operator helper files ####


@jit
def grad_helper(gammadash1, gammadash2):
    """
    This is a helper method that calculates the contravariant 
    vectors, grad phi and grad theta, using the curvilinear coordinate 
    identities:
    - grad1: grad phi = (dg2 x (dg1 x dg2))/|dg1 x dg2|^2
    - grad2: grad theta = -(dg1 x (dg2 x dg1))/|dg1 x dg2|^2
    Shape: (n_phi, n_theta, 3(xyz))

    Args:
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
    """
    dg2 = gammadash2
    dg1 = gammadash1
    dg1xdg2 = jnp.cross(dg1, dg2, axis=-1)
    denom = jnp.sum(dg1xdg2**2, axis=-1)
    # grad phi
    grad1 = jnp.cross(dg2, dg1xdg2, axis=-1)/denom[:, :, None]
    # grad theta
    grad2 = jnp.cross(dg1, -dg1xdg2, axis=-1)/denom[:, :, None]
    return (grad1, grad2)


@jit
def norm_helper(vec):
    """
    This is a helper method that calculates the following quantities:
    - normN_prime_2d: The normal vector's length, |N|
    - inv_normN_prime_2d: 1/|N|
    Shape: (n_phi, n_theta, 3(xyz)); (n_phi, n_theta); (n_phi, n_theta)

    Args:
        vec: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
    """
    # Length of the non-unit WS normal vector |N|,
    # its inverse (1/|N|) and its inverse's derivatives
    # w.r.t. phi(phi) and theta
    # Not to be confused with the normN (plasma surface Jacobian)
    # in Regcoil.
    norm = jnp.linalg.norm(vec, axis=-1)  # |N|
    inv_norm = 1/norm  # 1/|N|
    return (
        norm,
        inv_norm
    )


@jit
def dga_inv_n_dashb(
    normal,
    gammadash1,
    gammadash2,
    gammadash1dash1,
    gammadash1dash2,
    gammadash2dash2,
):
    """
    This is a helper method that calculates the following quantities:
    - dg1_inv_n_dash1: d[(1/|n|)(dgamma/dphi)]/dphi
    - dg1_inv_n_dash2: d[(1/|n|)(dgamma/dphi)]/dtheta
    - dg2_inv_n_dash1: d[(1/|n|)(dgamma/dtheta)]/dphi
    - dg2_inv_n_dash2: d[(1/|n|)(dgamma/dtheta)]/dtheta
    Shape: (n_phi, n_theta, 3(xyz))

    Args:
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
        gammadash1dash1: (n_phi, n_theta, 3) array
            Second partial phi to the winding surface
        gammadash1dash2: (n_phi, n_theta, 3) array
            Mixed partial phi-theta to the winding surface
        gammadash2dash2: (n_phi, n_theta, 3) array
            Second partial theta to the winding surface
    """
    # gammadash1() calculates partial r/partial phi. Keep in mind that the angles
    # in simsopt go from 0 to 1.
    # Shape: (n_phi, n_theta, 3(xyz))
    dg1 = gammadash1
    dg2 = gammadash2
    dg11 = gammadash1dash1
    dg12 = gammadash1dash2
    dg22 = gammadash2dash2

    # Because Phi is defined around the unit normal, rather
    # than N, we need to calculate the derivative and double derivative
    # of (dr/dtheta)/|N| and (dr/dphi)/|N|.
    # phi (phi) derivative of the normal's length
    normaldash1 = (
        jnp.cross(dg11, dg2)
        + jnp.cross(dg1, dg12)
    )

    # Theta derivative of the normal's length
    normaldash2 = (
        jnp.cross(dg12, dg2)
        + jnp.cross(dg1, dg22)
    )
    normal_vec = normal
    _, inv_normN_prime_2d = norm_helper(normal_vec)

    # Derivatives of 1/|N|:
    # d/dx(1/sqrt(f(x)^2 + g(x)^2 + h(x)^2))
    # = (-f(x)f'(x) - g(x)g'(x) - h(x)h'(x))
    # /(f(x)^2 + g(x)^2 + h(x)^2)^(3/2)
    denominator = jnp.sum(normal_vec**2, axis=-1)**1.5
    nominator_inv_normN_prime_2d_dash1 = -jnp.sum(normal_vec*normaldash1, axis=-1)
    nominator_inv_normN_prime_2d_dash2 = -jnp.sum(normal_vec*normaldash2, axis=-1)
    inv_normN_prime_2d_dash1 = nominator_inv_normN_prime_2d_dash1/denominator
    inv_normN_prime_2d_dash2 = nominator_inv_normN_prime_2d_dash2/denominator

    # d[(1/|n|)(dgamma/dphi)]/dphi
    dg1_inv_n_dash1 = dg11*inv_normN_prime_2d[:, :, None] + dg1*inv_normN_prime_2d_dash1[:, :, None]
    # d[(1/|n|)(dgamma/dphi)]/dtheta
    dg1_inv_n_dash2 = dg12*inv_normN_prime_2d[:, :, None] + dg1*inv_normN_prime_2d_dash2[:, :, None]
    # d[(1/|n|)(dgamma/dtheta)]/dphi
    dg2_inv_n_dash1 = dg12*inv_normN_prime_2d[:, :, None] + dg2*inv_normN_prime_2d_dash1[:, :, None]
    # d[(1/|n|)(dgamma/dtheta)]/dtheta
    dg2_inv_n_dash2 = dg22*inv_normN_prime_2d[:, :, None] + dg2*inv_normN_prime_2d_dash2[:, :, None]
    return (
        dg1_inv_n_dash1,
        dg1_inv_n_dash2,
        dg2_inv_n_dash1,
        dg2_inv_n_dash2
    )


@jit
def unitnormaldash(
    normal,
    gammadash1,
    gammadash2,
    gammadash1dash1,
    gammadash1dash2,
    gammadash2dash2,
):
    """
    This is a helper method that calculates the following quantities:
    - unitnormaldash1: d unitnormal/dphi
    - unitnormaldash2: d unitnormal/dtheta
    Shape: (n_phi, n_theta, 3(xyz))

    Args:
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
        gammadash1dash1: (n_phi, n_theta, 3) array
            Second partial phi to the winding surface
        gammadash1dash2: (n_phi, n_theta, 3) array
            Mixed partial phi-theta to the winding surface
        gammadash2dash2: (n_phi, n_theta, 3) array
            Second partial theta to the winding surface
    """
    _, inv_normN_prime_2d = norm_helper(normal)
    (
        dg1_inv_n_dash1, dg1_inv_n_dash2,
        _, _  # dg2_inv_n_dash1, dg2_inv_n_dash2
    ) = dga_inv_n_dashb(
        normal=normal,
        gammadash1=gammadash1,
        gammadash2=gammadash2,
        gammadash1dash1=gammadash1dash1,
        gammadash1dash2=gammadash1dash2,
        gammadash2dash2=gammadash2dash2,
    )

    dg2 = gammadash2
    dg1_inv_n = gammadash1 * inv_normN_prime_2d[:, :, None]
    dg22 = gammadash2dash2
    dg12 = gammadash1dash2
    unitnormaldash1 = (
        jnp.cross(dg1_inv_n_dash1, dg2, axis=-1)
        + jnp.cross(dg1_inv_n, dg12, axis=-1)
    )
    unitnormaldash2 = (
        jnp.cross(dg1_inv_n_dash2, dg2, axis=-1)
        + jnp.cross(dg1_inv_n, dg22, axis=-1)
    )
    return (unitnormaldash1, unitnormaldash2)


@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
])
def diff_helper(
    nfp, cp_m, cp_n,
    quadpoints_phi,
    quadpoints_theta,
    stellsym,
):
    """
    Calculates the following quantity:
    - trig_m_i_n_i, trig_diff_m_i_n_i: 
    IFT operator that transforms even/odd derivatives of Phi harmonics
    produced by partial_* (see below). 
    Shape: (n_phi, n_theta, n_dof)
    - partial_theta, partial_phi, ... ,partial_phi_theta,
    A partial derivative operators that works by multiplying the harmonic
    coefficients of Phi by its harmonic number and a sign, depending whether
    the coefficient is sin or cos. DOES NOT RE-ORDER the coefficients
    into the simsopt conventions. Therefore, IFT for such derivatives 
    must be performed with trig_m_i_n_i and trig_diff_m_i_n_i (see above).

    Args:
        nfp: int
            Number of field periods
        cp_m: (n,) array
            Fourier modes m of the current potential 
        cp_n: (n,) array
            Fourier modes n of the current potential 
        quadpoints_phi: (n_phi,) array
            Quadrature points in phi
        quadpoints_theta: (n_theta,) array
            Quadrature points in theta
        stellsym: bool
            Whether the stellarator is stellarator symmetric
    """
    # The uniform index for phi contains first sin Fourier
    # coefficients, then optionally cos is stellsym=False.
    n_harmonic = len(cp_m)
    iden = jnp.identity(n_harmonic)
    # Shape: (n_phi, n_theta)
    phi_grid = jnp.pi*2*quadpoints_phi[:, None]
    theta_grid = jnp.pi*2*quadpoints_theta[None, :]
    # When stellsym is enabled, Phi is a sin fourier series.
    # After a derivative, it becomes a cos fourier series.
    if stellsym:
        trig_choice = 1
    # Otherwise, it's a sin-cos series. After a derivative,
    # it becomes a cos-sin series.
    else:
        trig_choice = jnp.repeat([1, -1], n_harmonic//2)
    # Inverse Fourier transform that transforms a dof
    # array to grid values. trig_diff_m_i_n_i acts on
    # odd-order derivatives of dof, where the sin coeffs
    # become cos coefficients, and cos coeffs become
    # sin coeffs.
    # sin or sin-cos coeffs -> grid vals
    # Shape: (n_phi, n_theta, dof)
    trig_m_i_n_i = sin_or_cos(
        (cp_m)[None, None, :]*theta_grid[:, :, None]
        - (cp_n*nfp)[None, None, :]*phi_grid[:, :, None],
        trig_choice
    )
    # cos or cos-sin coeffs -> grid vals
    # Shape: (n_phi, n_theta, dof)
    trig_diff_m_i_n_i = sin_or_cos(
        (cp_m)[None, None, :]*theta_grid[:, :, None]
        - (cp_n*nfp)[None, None, :]*phi_grid[:, :, None],
        -trig_choice
    )

    # Fourier derivatives
    partial_theta = cp_m*trig_choice*iden*2*jnp.pi
    partial_phi = -cp_n*trig_choice*iden*nfp*2*jnp.pi
    partial_theta_theta = -cp_m**2*iden*(2*jnp.pi)**2
    partial_phi_phi = -(cp_n*nfp)**2*iden*(2*jnp.pi)**2
    partial_phi_theta = cp_n*nfp*cp_m*iden*(2*jnp.pi)**2
    return (
        trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        partial_phi_phi,
        partial_phi_theta,
        partial_theta_theta,
    )


@partial(jit, static_argnames=[
    'nfp',
    'stellsym',
])
def Kdash_helper(
        normal,
        gammadash1,
        gammadash2,
        gammadash1dash1,
        gammadash1dash2,
        gammadash2dash2,
        nfp, cp_m, cp_n,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        quadpoints_phi,
        quadpoints_theta,
        stellsym):
    """
    Calculates the following quantity
    - Kdash1_sv_op, Kdash2_sv_op: 
    Partial derivatives of K in term of Phi (current potential) harmonics.
    Shape: (n_phi, n_theta, 3(xyz), n_dof)
    - Kdash1_const, Kdash2_const: 
    Partial derivatives of K due to secular terms (net poloidal/toroidal 
    currents). 
    Shape: (n_phi, n_theta, 3(xyz))

    Args: 
        normal: (n_phi, n_theta, 3) array
            Normal vector to the winding surface
        gammadash1: (n_phi, n_theta, 3) array
            Partial phi to the winding surface
        gammadash2: (n_phi, n_theta, 3) array
            Partial theta to the winding surface
        gammadash1dash1: (n_phi, n_theta, 3) array
            Second partial phi to the winding surface
        gammadash1dash2: (n_phi, n_theta, 3) array
            Mixed partial phi-theta to the winding surface
        gammadash2dash2: (n_phi, n_theta, 3) array
            Second partial theta to the winding surface
        net_poloidal_current_amperes: float
            Poloidal current in amperes
        net_toroidal_current_amperes: float
            Toroidal current in amperes
        quadpoints_phi: (n_phi,) array
            Quadrature points in phi
        quadpoints_theta: (n_theta,) array
            Quadrature points in theta
        nfp: int
            Number of field periods
        cp_m: (n,) array
            Fourier modes m of the current potential 
        cp_n: (n,) array
            Fourier modes n of the current potential 
        stellsym: bool
            Whether the stellarator is stellarator symmetric 

    Returns:
        Kdash1_sv_op:
            Singular valued part of Kdash1.
        Kdash2_sv_op:
            Singular valued part of Kdash2.
        Kdash1_const:
            Constant part of Kdash1.
        Kdash2_const:
            Full Kdash.
    """
    normN_prime_2d, _ = norm_helper(normal)
    (
        trig_m_i_n_i,
        trig_diff_m_i_n_i,
        partial_phi,
        partial_theta,
        partial_phi_phi,
        partial_phi_theta,
        partial_theta_theta,
    ) = diff_helper(
        nfp, cp_m, cp_n,
        quadpoints_phi,
        quadpoints_theta,
        stellsym,
    )
    # Some quantities
    (
        dg1_inv_n_dash1,
        dg1_inv_n_dash2,
        dg2_inv_n_dash1,
        dg2_inv_n_dash2
    ) = dga_inv_n_dashb(
        normal=normal,
        gammadash1=gammadash1,
        gammadash2=gammadash2,
        gammadash1dash1=gammadash1dash1,
        gammadash1dash2=gammadash1dash2,
        gammadash2dash2=gammadash2dash2,
    )
    # Operators that generates the derivative of K
    # Note the use of trig_diff_m_i_n_i for inverse
    # FT following odd-order derivatives.
    # Shape: (n_phi, n_theta, 3(xyz), n_dof)
    Kdash2_sv_op = (
        dg2_inv_n_dash2[:, :, None, :]
        * (trig_diff_m_i_n_i@partial_phi)[:, :, :, None]

        + gammadash2[:, :, None, :]
        * (trig_m_i_n_i@partial_phi_theta)[:, :, :, None]
        / normN_prime_2d[:, :, None, None]

        - dg1_inv_n_dash2[:, :, None, :]
        * (trig_diff_m_i_n_i@partial_theta)[:, :, :, None]

        - gammadash1[:, :, None, :]
        * (trig_m_i_n_i@partial_theta_theta)[:, :, :, None]
        / normN_prime_2d[:, :, None, None]
    )
    Kdash2_sv_op = jnp.swapaxes(Kdash2_sv_op, 2, 3)
    Kdash1_sv_op = (
        dg2_inv_n_dash1[:, :, None, :]
        * (trig_diff_m_i_n_i@partial_phi)[:, :, :, None]

        + gammadash2[:, :, None, :]
        * (trig_m_i_n_i@partial_phi_phi)[:, :, :, None]
        / normN_prime_2d[:, :, None, None]

        - dg1_inv_n_dash1[:, :, None, :]
        * (trig_diff_m_i_n_i@partial_theta)[:, :, :, None]

        - gammadash1[:, :, None, :]
        * (trig_m_i_n_i@partial_phi_theta)[:, :, :, None]
        / normN_prime_2d[:, :, None, None]
    )
    Kdash1_sv_op = jnp.swapaxes(Kdash1_sv_op, 2, 3)
    G = net_poloidal_current_amperes
    I = net_toroidal_current_amperes
    # Constant components of K's partial derivative.
    # Shape: (n_phi, n_theta, 3(xyz))
    Kdash1_const = \
        dg2_inv_n_dash1*G \
        - dg1_inv_n_dash1*I
    Kdash2_const = \
        dg2_inv_n_dash2*G \
        - dg1_inv_n_dash2*I
    return (
        Kdash1_sv_op,
        Kdash2_sv_op,
        Kdash1_const,
        Kdash2_const
    )


@partial(jit, static_argnames=[
    'normalize',
])
def A_b_c_to_block_operator(A, b, c, current_scale, normalize):
    """
    Converts a set of A, b, c that gives 
    f(p) = pTAp + bTp +c

    into a matrix consisting of 
    O = [
        [(AT+A)/2 / (current_scale**2), b/2 / current_scale],
        [bT/2     /  current_scale,     c  ]
    ]
    That satisfy 
    f(p) = tr(OX) 
    for X = (p, 1)(p, 1)^T

    Args:
        A:
            The A matrix
        b: 
            The b vector
        c: 
            The c scalar
        current_scale: float
            The scale of the current potential
        normalize: bool
            Whether to normalize the output
    """
    O = jnp.block([
        [(A+A.swapaxes(-1, -2)) / 2/(current_scale**2), jnp.expand_dims(b, axis=-1) / 2/current_scale],
        [jnp.expand_dims(b, axis=-2)/2 / current_scale, jnp.expand_dims(c, axis=(-1, -2))]
    ])
    if normalize:
        out_scale = avg_order_of_magnitude(O)
        O /= out_scale
    else:
        out_scale = 1
    return (O, out_scale)
