import numpy as np
from mpi4py import MPI
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def compute_poincare(field, R0, Z0, NFP=1, npoints=20, rtol=1e-6, atol=1e-9):
    """
    Generate the raw (x,y) data for a Poincare plot

    field: A magnetic field object from SIMSOPT's magneticfieldclasses list
    R0: Initial R values. Can be an array of any dimension, or a single value.
    Z0: Initial Z values
    npoints: Number of points to compute for each initial condition for the Poincare plot
    rmin, rmax, zmin, zmax: Stop tracing field line if you leave this region
    rtol, atol: tolerances for the integration
    """
    comm = MPI.COMM_WORLD
    mpi_N_procs = comm.Get_size()
    mpi_rank = comm.Get_rank()
    # print('Hello from MPI proc {:4d} of {:4d}'.format(mpi_rank+1,mpi_N_procs))

    R0 = np.array(R0).flatten()
    Z0 = np.array(Z0).flatten()

    assert R0.shape == Z0.shape

    N_field_lines = len(R0)
    Poincare_data = [0]*N_field_lines
    for j in range(N_field_lines):
        if j % mpi_N_procs != mpi_rank:
            continue
        # print('Proc {:4d} is tracing field line {:4d} of {:4d}.'.format(mpi_rank, j, N_field_lines))
        phi_range = (0, (2*np.pi*npoints)/NFP)
        # Factors of 4 below are to get data at 1/4, 1/2, 3/4 period:
        phi_to_report = np.arange(npoints * 4) * 2 * np.pi / (4 * NFP) 
        RZ_initial = [R0[j], Z0[j]]
        solution = solve_ivp(field.d_RZ_d_phi, phi_range, RZ_initial, t_eval=phi_to_report, rtol=rtol, atol=atol)
        # phi = solution.t
        # R = solution.y[0,:]
        # Z = solution.y[1,:]
        Poincare_data[j] = solution.y

    # Send results from each processor to the root:
    for j in range(N_field_lines):
        index = j % mpi_N_procs
        if index == 0:
            # Proc 0 did this field line, so no communication is needed
            pass
        else:
            if mpi_rank == 0:
                # print('Root is receiving field line {:5d} from proc {:4d}'.format(j,index))
                Poincare_data[j] = comm.recv(source=index, tag=index)
            elif index == mpi_rank:
                # print('Proc {:4d} is sending field line {:5d} to root'.format(mpi_rank, j))
                comm.send(Poincare_data[j], dest=0, tag=index)

    return np.array(Poincare_data)


def plot_poincare(data, marker_size=2, extra_str=""):
    """
    Generate a Poincare plot using precomputed (R,Z) data.

    data: an array returned by compute_poincare().
    marker_size: size of points in the plot
    extra_str: string added to plot title and filename
    """

    plt.figure(figsize=(14, 7))
    num_rows = 2
    num_cols = 2
    for j_quarter in range(4):
        plt.subplot(num_rows, num_cols, j_quarter + 1)
        for j in range(len(data)):
            plt.scatter(data[j][0, j_quarter:-1:4], data[j][1, j_quarter:-1:4], s=marker_size, edgecolors='none')
        plt.xlabel('R')
        plt.ylabel('Z')
        plt.gca().set_aspect('equal', adjustable='box')
        # Turn on minor ticks, since it is necessary to get minor grid lines
        from matplotlib.ticker import AutoMinorLocator
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10))
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator(10))
        plt.grid(which='major', linewidth=0.5)
        plt.grid(which='minor', linewidth=0.15)

    title_string = 'Poincare_' + extra_str + '_Npoints=' + str(int(len(data[0, 0])/4)) + '_Nlines=' + str(len(data))
    plt.figtext(0.5, 0.995, title_string, fontsize=10, ha='center', va='top')

    plt.tight_layout()
    plt.show()
