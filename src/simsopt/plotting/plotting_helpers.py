import numpy as np 

from .._core.util import parallel_loop_bounds

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    verbose = (comm.rank == 0)
except ImportError:
    comm = None
    verbose = True

__all__ = ['plot_trajectory_poloidal', 'plot_trajectory_overhead_cyl']

def plot_trajectory_poloidal(res_ty, helicity_M=1, helicity_N=0, ax=None):
    r"""
    Given the trajectory of a single particle in Boozer coordinates, plot
    in the pseudo-poloidal plane (x,y) where x = sqrt(s) cos(chi) 
    and y = sqrt(s) sin(chi), where chi = helicity_M * theta - helicity_N * zeta.
    If helicity_M and helicity_N correspond with the helicity of the field-strength 
    contours, then this plot will visualize the "banana" trajectories of trapped
    particles. Circles in this plane correspond with flux surfaces. The dashed circle
    is the initial flux surface, and the solid circle is the plasma boundary. 

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the trajectory of a single particle in Boozer coordinates 
                (s, theta, zeta, vpar). Tracing should be performed with forget_exact_path=False to save the trajectory
                information. 
        helicity_M : Helicity M value for the transformation (default: 1)
        helicity_N : Helicity N value for the transformation (default: 0)
        ax : Optional matplotlib Axes object to plot on. If None, a new figure and axes will be created.

    Returns:
        ax : The matplotlib Axes object containing the plot.
    """
    s = res_ty[:, 1]
    theta = res_ty[:, 2]
    zeta = res_ty[:, 3]
    chi = helicity_M * theta - helicity_N * zeta
    x = np.sqrt(s) * np.cos(chi)
    y = np.sqrt(s) * np.sin(chi)

    chi_grid = np.linspace(0, 2 * np.pi, 100)
    s0 = s[0]
    x0 = np.sqrt(s0) * np.cos(chi_grid)
    y0 = np.sqrt(s0) * np.sin(chi_grid)

    x1 = np.cos(chi_grid)
    y1 = np.sin(chi_grid)

    if verbose:
        import matplotlib 
        matplotlib.use('Agg') # Don't use interactive backend 
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(x, y)
        ax.plot(x0, y0, color='black', linestyle='--')
        ax.plot(x1, y1, color='black', linestyle='-')
        ax.set_xlabel(r'$\sqrt{s} \cos(\chi)$')
        ax.set_ylabel(r'$\sqrt{s} \sin(\chi)$')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    else:
        ax = None 

    return ax 

def plot_trajectory_overhead_cyl(res_ty, field, ax=None):
    r"""
    Given the trajectory of a single particle in Boozer coordinates, plot
    in the overhead cylindrical plane (X,Y) where X = R cos(phi) and Y = R sin(phi).
    This plot visualizes an overhead view of the particle trajectory in cylindrical coordinates.
    The dashed circle corresponds to the magnetic axis.

    Args:
        res_ty : A 2D numpy array of shape (nsteps, 5) containing the trajectory of a single particle in Boozer coordinates 
                (s, theta, zeta, vpar).
        field : The :class:`BoozerMagneticField` instance used to set the points for the field.
        ax : Optional matplotlib Axes object to plot on. If None, a new figure and axes will be created.

    Returns:
        ax : The matplotlib Axes object containing the plot.
    """
    from ..field.trajectory_helpers import compute_trajectory_cylindrical
    R, phi, Z  = compute_trajectory_cylindrical(res_ty, field)

    X = R * np.cos(phi)
    Y = R * np.sin(phi)

    zetas_grid = np.linspace(0, 2 * np.pi, 100)
    points = np.zeros((len(zetas_grid), 3))
    points[:, 2] = zetas_grid
    field.set_points(points)
    R_axis = field.R()[:,0]
    Z_axis = field.Z()[:,0]
    nu = field.nu()[:,0]
    phi_axis = zetas_grid - nu

    X_axis = R_axis * np.cos(phi_axis)
    Y_axis = R_axis * np.sin(phi_axis)

    if verbose: 
        import matplotlib 
        matplotlib.use('Agg') # Don't use interactive backend 
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.plot(X_axis, Y_axis, linestyle='--', color='black')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
    else:
        ax = None 

    return ax 