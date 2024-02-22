from pathlib import Path
import warnings

import numpy as np
from pyevtk.hl import pointsToVTK

from . import Surface
import simsoptpp as sopp

__all__ = ['PSCgrid']


class PSCgrid:
    r"""
    ``PSCgrid`` is a class for setting up the grid,
    plasma surface, and other objects needed to perform PSC
    optimization for stellarators. The class
    takes as input two toroidal surfaces specified as SurfaceRZFourier
    objects, and initializes a set of points (in Cartesian coordinates)
    between these surfaces.

    Args:
        plasma_boundary: Surface class object 
            Representing the plasma boundary surface. Gets converted
            into SurfaceRZFourier object for ease of use.
        Bn: 2D numpy array, shape (ntheta_quadpoints, nphi_quadpoints)
            Magnetic field (coils and plasma) at the plasma
            boundary. Typically this will be the optimized plasma
            magnetic field from a stage-1 optimization, and the
            optimized coils from a basic stage-2 optimization.
            This variable must be specified to run PSC optimization.
    """

    def __init__(self, plasma_boundary: Surface, Bn):
        Bn = np.array(Bn)
        if len(Bn.shape) != 2: 
            raise ValueError('Normal magnetic field surface data is incorrect shape.')
        self.Bn = Bn
        self.plasma_boundary = plasma_boundary.to_RZFourier()
        self.nphi = len(self.plasma_boundary.quadpoints_phi)
        self.ntheta = len(self.plasma_boundary.quadpoints_theta)

    def _setup_uniform_grid(self):
        """
        Initializes a uniform grid in cartesian coordinates and sets
        some important grid variables for later.
        """
        # Get (X, Y, Z) coordinates of the two boundaries
        self.xyz_inner = self.inner_toroidal_surface.gamma().reshape(-1, 3)
        self.xyz_outer = self.outer_toroidal_surface.gamma().reshape(-1, 3)
        x_outer = self.xyz_outer[:, 0]
        y_outer = self.xyz_outer[:, 1]
        z_outer = self.xyz_outer[:, 2]

        x_max = np.max(x_outer)
        x_min = np.min(x_outer)
        y_max = np.max(y_outer)
        y_min = np.min(y_outer)
        z_max = np.max(z_outer)
        z_min = np.min(z_outer)
        z_max = max(z_max, abs(z_min))
        print(x_min, x_max, y_min, y_max, z_min, z_max)

        # Initialize uniform grid
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        self.dx = (x_max - x_min) / (Nx - 1)
        self.dy = (y_max - y_min) / (Ny - 1)
        self.dz = 2 * z_max / (Nz - 1)
        print(Nx, Ny, Nz, self.dx, self.dy, self.dz)
        Nmin = min(self.dx, min(self.dy, self.dz))
        self.R = Nmin / 4.0
        print('Major radius of the coils is R = ', self.R)
        print('Coils are spaced so that every coil of radius R '
              ' is at least 2R away from the next coil'
        )

        # Extra work below so that the stitching with the symmetries is done in
        # such a way that the reflected cells are still dx and dy away from
        # the old cells.
        #### Note that Cartesian cells can only do nfp = 2, 4, 6, ... 
        #### and correctly be rotated to have the right symmetries
        # if (self.plasma_boundary.nfp % 2) == 0:
        #     X = np.linspace(self.dx / 2.0, (x_max - x_min) + self.dx / 2.0, Nx, endpoint=True)
        #     Y = np.linspace(self.dy / 2.0, (y_max - y_min) + self.dy / 2.0, Ny, endpoint=True)
        # else:
        #     X = np.linspace(x_min, x_max, Nx, endpoint=True)
        #     Y = np.linspace(y_min, y_max, Ny, endpoint=True)
        # Z = np.linspace(-z_max, z_max, Nz, endpoint=True)

        # # Make 3D mesh
        # X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
        # self.xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(Nx * Ny * Nz, 3)

        # # Extra work for nfp = 4 to chop off half of the originally nfp = 2 uniform grid
        # if self.plasma_boundary.nfp == 4:
        #     inds = []
        #     for i in range(Nx):
        #         for j in range(Ny):
        #             for k in range(Nz):
        #                 if X[i, j, k] < Y[i, j, k]:
        #                     inds.append(int(i * Ny * Nz + j * Nz + k))
        #     good_inds = np.setdiff1d(np.arange(Nx * Ny * Nz), inds)
        #     self.xyz_uniform = self.xyz_uniform[good_inds, :]
        # else:
        #     # Get (R, Z) coordinates of the outer boundary
        #     rphiz_outer = np.array(
        #         [np.sqrt(self.xyz_outer[:, 0] ** 2 + self.xyz_outer[:, 1] ** 2), 
        #          np.arctan2(self.xyz_outer[:, 1], self.xyz_outer[:, 0]),
        #          self.xyz_outer[:, 2]]
        #     ).T

        #     r_max = np.max(rphiz_outer[:, 0])
        #     r_min = np.min(rphiz_outer[:, 0])
        #     z_max = np.max(rphiz_outer[:, 2])
        #     z_min = np.min(rphiz_outer[:, 2])

        #     # Initialize uniform grid of curved, square bricks
        #     Nr = int((r_max - r_min) / self.dr)
        #     self.Nr = Nr
        #     self.dz = self.dr
        #     Nz = int((z_max - z_min) / self.dz)
        #     self.Nz = Nz
        #     phi = 2 * np.pi * np.copy(self.plasma_boundary.quadpoints_phi)
        #     R = np.linspace(r_min, r_max, Nr)
        #     Z = np.linspace(z_min, z_max, Nz)

        #     # Make 3D mesh
        #     R, Phi, Z = np.meshgrid(R, phi, Z, indexing='ij')
        #     X = R * np.cos(Phi)
        #     Y = R * np.sin(Phi)
        #     self.xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(-1, 3)

        # Save uniform grid before we start chopping off parts.
        contig = np.ascontiguousarray
        pointsToVTK('uniform_grid', contig(self.xyz_uniform[:, 0]),
                    contig(self.xyz_uniform[:, 1]), contig(self.xyz_uniform[:, 2]))

    @classmethod
    def geo_setup_between_toroidal_surfaces(
        cls, 
        plasma_boundary,
        Bn,
        inner_toroidal_surface: Surface, 
        outer_toroidal_surface: Surface,
        **kwargs,
    ):
        """
        Function to initialize a SIMSOPT PermanentMagnetGrid from a 
        volume defined by two toroidal surfaces. These must be specified
        directly. Often a good choice is made by extending the plasma 
        boundary by its normal vectors.

        Args
        ----------
        inner_toroidal_surface: Surface class object 
            Representing the inner toroidal surface of the volume.
            Gets converted into SurfaceRZFourier object for 
            ease of use.
        outer_toroidal_surface: Surface object representing
            the outer toroidal surface of the volume. Typically 
            want this to have same quadrature points as the inner
            surface for a functional grid setup. 
            Gets converted into SurfaceRZFourier object for 
            ease of use.
        kwargs: The following are valid keyword arguments.
            Nx: int
                Number of points in x to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. Used only if the
                coordinate_flag = cartesian, then Nx is the x-size of the
                rectangular cubes in the grid.
            Ny: int
                Number of points in y to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. Used only if the
                coordinate_flag = cartesian, then Ny is the y-size of the
                rectangular cubes in the grid.
            Nz: int
                Number of points in z to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. Used only if the
                coordinate_flag = cartesian, then Nz is the z-size of the
                rectangular cubes in the grid.
        Returns
        -------
        psc_grid: An initialized PSCgrid class object.

        """
        
        psc_grid = cls(plasma_boundary, Bn) 
        Nx = kwargs.pop("Nx", 10)
        Ny = kwargs.pop("Ny", 10)
        Nz = kwargs.pop("Nz", 10)
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError('Nx, Ny, and Nz should be positive integers')
        psc_grid.Nx = Nx
        psc_grid.Ny = Ny
        psc_grid.Nz = Nz
        psc_grid.inner_toroidal_surface = inner_toroidal_surface.to_RZFourier()
        psc_grid.outer_toroidal_surface = outer_toroidal_surface.to_RZFourier()    
        warnings.warn(
            'Plasma boundary and inner and outer toroidal surfaces should '
            'all have the same "range" parameter in order for a permanent'
            ' magnet grid to be correctly initialized.'
        )        

        # Have the uniform grid, now need to loop through and eliminate cells.
        contig = np.ascontiguousarray
        normal_inner = inner_toroidal_surface.unitnormal().reshape(-1, 3)   
        normal_outer = outer_toroidal_surface.unitnormal().reshape(-1, 3)   
        psc_grid._setup_uniform_grid()
        psc_grid.grid_xyz = sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
            contig(normal_inner), 
            contig(normal_outer), 
            contig(psc_grid.xyz_uniform), 
            contig(psc_grid.xyz_inner), 
            contig(psc_grid.xyz_outer))
        inds = np.ravel(np.logical_not(np.all(psc_grid.grid_xyz == 0.0, axis=-1)))
        psc_grid.grid_xyz = psc_grid.grid_xyz[inds, :]
        psc_grid.num_psc = psc_grid.grid_xyz.shape[0]
        psc_grid.pm_phi = np.arctan2(psc_grid.grid_xyz[:, 1], psc_grid.grid_xyz[:, 0])
        pointsToVTK('psc_grid',
                    contig(psc_grid.grid_xyz[:, 0]),
                    contig(psc_grid.grid_xyz[:, 1]),
                    contig(psc_grid.grid_xyz[:, 2]))

        # PSC coil geometry determined by its center point in grid_xyz
        # and its alpha and delta angles, which we initialize randomly here.
        psc_grid.alphas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
        psc_grid.deltas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
        
        # Initialize curve objects corresponding to each PSC coil for 
        # plotting in 3D
        psc_grid.plot_curves()
        
        psc_grid._optimization_setup()
        return psc_grid

    # def _optimization_setup(self):

    #     if self.Bn.shape != (self.nphi, self.ntheta):
    #         raise ValueError('Normal magnetic field surface data is incorrect shape.')

    #     # minus sign below because ||Ax - b||^2 term but original
    #     # term is integral(B_P + B_C + B_M)^2
    #     self.b_obj = - self.Bn.reshape(self.nphi * self.ntheta)

    #     # Compute geometric factor with the C++ routine
    #     self.A_obj = sopp.dipole_field_Bn(
    #         np.ascontiguousarray(self.plasma_boundary.gamma().reshape(-1, 3)),
    #         np.ascontiguousarray(self.grid_xyz),
    #         np.ascontiguousarray(self.plasma_boundary.unitnormal().reshape(-1, 3)),
    #         self.plasma_boundary.nfp, int(self.plasma_boundary.stellsym),
    #         np.ascontiguousarray(self.b_obj),
    #         self.coordinate_flag,  # cartesian, cylindrical, or simple toroidal
    #         self.R0
    #     )

    #     # Rescale the A matrix so that 0.5 * ||Am - b||^2 = f_b,
    #     # where f_b is the metric for Bnormal on the plasma surface
    #     Ngrid = self.nphi * self.ntheta
    #     self.A_obj = self.A_obj.reshape(self.nphi * self.ntheta, self.num_psc * 3)
    #     Nnorms = np.ravel(np.sqrt(np.sum(self.plasma_boundary.normal() ** 2, axis=-1)))
    #     for i in range(self.A_obj.shape[0]):
    #         self.A_obj[i, :] = self.A_obj[i, :] * np.sqrt(Nnorms[i] / Ngrid)
    #     self.b_obj = self.b_obj * np.sqrt(Nnorms / Ngrid)
    #     self.ATb = self.A_obj.T @ self.b_obj

    #     # Compute singular values of A, use this to determine optimal step size
    #     # for the MwPGP algorithm, with alpha ~ 2 / ATA_scale
    #     S = np.linalg.svd(self.A_obj, full_matrices=False, compute_uv=False)
    #     self.ATA_scale = S[0] ** 2

    #     # Set initial condition for the dipoles to default IC
    #     self.m0 = np.zeros(self.num_psc * 3)

    #     # Set m to zeros
    #     self.m = self.m0

    #     # Print initial f_B metric using the initial guess
    #     total_error = np.linalg.norm((self.A_obj.dot(self.m0) - self.b_obj), ord=2) ** 2 / 2.0
    #     print('f_B (total with initial SIMSOPT guess) = ', total_error)

    def plot_curves(self):
        
        from . import CurvePlanarFourier, curves_to_vtk
        # r(\phi) = \sum_{m=0}^{\text{order}} r_{c,m}\cos(m \phi) + \sum_{m=1}^{\text{order}} r_{s,m}\sin(m \phi).
        # [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, q_0, q_i, q_j, q_k, x_{\text{center}}, y_{\text{center}}, z_{\text{center}}]
    
        order = 1
        ncoils = self.num_psc
        self.I = np.zeros(ncoils)
        # xc = np.zeros((num_psc, order + 1))
        # xs = np.zeros((num_psc, order + 1))
        # yc = np.zeros((num_psc, order + 1))
        # ys = np.zeros((num_psc, order + 1))
        # zc = np.zeros((num_psc, order + 1))
        # zs = np.zeros((num_psc, order + 1))
        
        # CurveXYZFourier wants data in order sin_x, cos_x, sin_y, cos_y, ...
        # coil_data = np.zeros((order + 1, num_psc * 6))
        # for i in range(num_psc):
        #     xyz = self.grid_xyz[i, :]
        #     x = xyz[0] * 
        #     coil_data[:, i * 6 + 0] = xs[i, :]
        #     coil_data[:, i * 6 + 1] = xc[i, :]
        #     coil_data[:, i * 6 + 2] = ys[i, :]
        #     coil_data[:, i * 6 + 3] = yc[i, :]
        #     coil_data[:, i * 6 + 4] = zs[i, :]
        #     coil_data[:, i * 6 + 5] = zc[i, :]
    
        # Set the degrees of freedom in the coil objects
        # base_currents = [Current(coilcurrents[i]) for i in range(ncoils)]
        ppp = 20
        curves = [CurvePlanarFourier(order*ppp, order) for i in range(ncoils)]
        for ic in range(ncoils):
            xyz = self.grid_xyz[ic, :]
            dofs = curves[ic].dofs_matrix
            dofs[0] = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)
            dofs[1] = self.R
            dofs[2] = self.R
            # Now specify the rotation
            dofs[3] = self.alphas[ic] * self.deltas[ic]
            dofs[4] = self.alphas[ic]
            dofs[5] = self.deltas[ic]
            # Now specify the center 
            dofs[6] = xyz[0]
            dofs[7] = xyz[1]
            dofs[8] = xyz[2]
            curves[ic].local_x = dofs
        curves_to_vtk(curves, "curves_init")
        # return coils