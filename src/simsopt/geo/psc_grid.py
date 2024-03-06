from pathlib import Path
import warnings

import numpy as np
from pyevtk.hl import pointsToVTK

from . import Surface
import simsoptpp as sopp
from simsopt.field import BiotSavart
from scipy.integrate import simpson, dblquad, IntegrationWarning
from scipy.special import roots_chebyt, roots_legendre, roots_jacobi
import time

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

    def __init__(self):
        self.mu0 = 4 * np.pi * 1e-7

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
        # print(x_min, x_max, y_min, y_max, z_min, z_max)

        # Initialize uniform grid
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        self.dx = (x_max - x_min) / (Nx - 1)
        self.dy = (y_max - y_min) / (Ny - 1)
        self.dz = 2 * z_max / (Nz - 1)
        # print(Nx, Ny, Nz, self.dx, self.dy, self.dz)
        Nmin = min(self.dx, min(self.dy, self.dz))
        self.R = Nmin / 2.0
        self.a = self.R / 100.0
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
        X = np.linspace(x_min, x_max, Nx, endpoint=True)
        Y = np.linspace(y_min, y_max, Ny, endpoint=True)
        Z = np.linspace(-z_max, z_max, Nz, endpoint=True)

        # Make 3D mesh
        X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
        self.xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(Nx * Ny * Nz, 3)

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
        plasma_boundary : Surface,
        Bn,
        B_TF,
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
            Nt: int
                Number of turns of the coil. 
        Returns
        -------
        psc_grid: An initialized PSCgrid class object.

        """
        from simsopt.field import InterpolatedField
                
        psc_grid = cls() 
        Bn = np.array(Bn)
        if len(Bn.shape) != 2: 
            raise ValueError('Normal magnetic field surface data is incorrect shape.')
        psc_grid.Bn = Bn
        
        if not isinstance(B_TF, BiotSavart):
            raise ValueError('The magnetic field from the energized coils '
                ' must be passed as a BiotSavart object.'
        )
        # psc_grid.B_TF = B_TF
        # Many big calls to B_TF.B() so make an interpolated object
        psc_grid.plasma_boundary = plasma_boundary.to_RZFourier()
        psc_grid.nphi = len(psc_grid.plasma_boundary.quadpoints_phi)
        psc_grid.ntheta = len(psc_grid.plasma_boundary.quadpoints_theta)
        Ngrid = psc_grid.nphi * psc_grid.ntheta
        Nnorms = np.ravel(np.sqrt(np.sum(psc_grid.plasma_boundary.normal() ** 2, axis=-1)))
        psc_grid.grid_normalization = np.sqrt(Nnorms / Ngrid)
        # integration over phi for L
        N = 20
        psc_grid.phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
        psc_grid.dphi = psc_grid.phi[1] - psc_grid.phi[0]
        Nx = kwargs.pop("Nx", 10)
        Ny = Nx  # kwargs.pop("Ny", 10)
        Nz = Nx  # kwargs.pop("Nz", 10)
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

        Nt = kwargs.pop("Nt", 1)
        psc_grid.Nt = Nt
        # Have the uniform grid, now need to loop through and eliminate cells.
        t1 = time.time()
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
        psc_grid.grid_xyz = np.array(psc_grid.grid_xyz[inds, :], dtype=float)
        psc_grid.num_psc = psc_grid.grid_xyz.shape[0]
        psc_grid.rho = np.linspace(0, psc_grid.R, N, endpoint=False)
        psc_grid.drho = psc_grid.rho[1] - psc_grid.rho[0]

        # psc_grid.pm_phi = np.arctan2(psc_grid.grid_xyz[:, 1], psc_grid.grid_xyz[:, 0])
        pointsToVTK('psc_grid',
                    contig(psc_grid.grid_xyz[:, 0]),
                    contig(psc_grid.grid_xyz[:, 1]),
                    contig(psc_grid.grid_xyz[:, 2]))
        print('Number of PSC locations = ', len(psc_grid.grid_xyz))
        
        # Order of the coils. For unit tests, needs > 400
        psc_grid.ppp = kwargs.pop("ppp", 400)

        # PSC coil geometry determined by its center point in grid_xyz
        # and its alpha and delta angles, which we initialize randomly here.
        
        # Initialize the coil orientations parallel to the B field. 
        psc_grid.B_TF = B_TF
        # n = 20
        # degree = 2
        # R = psc_grid.R
        # gamma_outer = outer_toroidal_surface.gamma().reshape(-1, 3)
        # rs = np.linalg.norm(gamma_outer[:, :2], axis=-1)
        # zs = gamma_outer[:, 2]
        # rrange = (0, np.max(rs) + R, n)  # need to also cover the plasma
        # phirange = (0, 2 * np.pi, n * 2)
        # zrange = (np.min(zs) - R, np.max(zs) + R, n // 2)
        # psc_grid.B_TF = InterpolatedField(
        #     B_TF, degree, rrange, phirange, zrange, True, nfp=1, stellsym=False
        # )
        B_TF.set_points(psc_grid.grid_xyz)
        B = B_TF.B()
        
        # Randomly initialize the coil orientations
        random_initialization = kwargs.pop("random_initialization", False)
        if random_initialization:
            psc_grid.alphas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
            psc_grid.deltas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
            psc_grid.coil_normals = np.array(
                [np.cos(psc_grid.alphas) * np.sin(psc_grid.deltas),
                  -np.sin(psc_grid.alphas),
                  np.cos(psc_grid.alphas) * np.cos(psc_grid.deltas)]
            ).T
            # psc_grid.coil_normals = np.array(
            #     [np.cos(psc_grid.deltas) * np.sin(psc_grid.alphas),
            #      np.sin(psc_grid.deltas) * np.sin(psc_grid.alphas),
            #      np.cos(psc_grid.alphas)]
            # ).T
        else:
            # determine the alphas and deltas from these normal vectors
            psc_grid.coil_normals = (B.T / np.sqrt(np.sum(B ** 2, axis=-1))).T
            psc_grid.alphas = np.arcsin(-psc_grid.coil_normals[:, 1])
            # minus_yinds = np.ravel(np.where(psc_grid.grid_xyz[:, 1] > 0.0))
            # psc_grid.alphas[minus_yinds] = -psc_grid.alphas[minus_yinds]
            psc_grid.deltas = np.arccos(psc_grid.coil_normals[:, 2] / np.cos(psc_grid.alphas))
            # psc_grid.alphas = np.arctan2(
            #     np.sqrt(
            #         psc_grid.coil_normals[:, 0] ** 2 + psc_grid.coil_normals[:, 1] ** 2
            #         ), psc_grid.coil_normals[:, 2]
            # )
            # psc_grid.deltas = np.arctan2(
            #     psc_grid.coil_normals[:, 1], psc_grid.coil_normals[:, 0]
            # )
            
        t2 = time.time()
        print('Initialize grid time = ', t2 - t1)
        
        # Initialize curve objects corresponding to each PSC coil for 
        # plotting in 3D
        
        t1 = time.time()
        psc_grid.setup_orientations(psc_grid.alphas, psc_grid.deltas)
        t2 = time.time()
        print('Geo setup time = ', t2 - t1)
        psc_grid.plot_curves()
        psc_grid.b_vector()
        kappas = np.ravel(np.array([psc_grid.alphas, psc_grid.deltas]))
        psc_grid.A_opt = psc_grid.least_squares(kappas)

        # psc_grid._optimization_setup()
        return psc_grid
    
    @classmethod
    def geo_setup_manual(
        cls, 
        points,
        R,
        a,
        alphas,
        deltas,
        **kwargs,
    ):
        from simsopt.util.permanent_magnet_helper_functions import initialize_coils
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries
        from simsopt.field import InterpolatedField
        
        psc_grid = cls()
        psc_grid.grid_xyz = np.array(points, dtype=float)
        psc_grid.R = R
        psc_grid.a = a
        psc_grid.alphas = alphas
        psc_grid.deltas = deltas
        Nt = kwargs.pop("Nt", 1)
        psc_grid.Nt = Nt
        psc_grid.num_psc = psc_grid.grid_xyz.shape[0]
        N = 100
        psc_grid.phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
        psc_grid.dphi = psc_grid.phi[1] - psc_grid.phi[0]
        psc_grid.rho = np.linspace(0, R, N, endpoint=False)
        psc_grid.drho = psc_grid.rho[1] - psc_grid.rho[0]
        
        Bn = kwargs.pop("Bn", np.zeros((1, 3)))
        Bn = np.array(Bn)
        if len(Bn.shape) != 2: 
            raise ValueError('Normal magnetic field surface data is incorrect shape.')
        psc_grid.Bn = Bn
    
        # initialize default plasma boundary
        input_name = 'input.LandremanPaul2021_QA_lowres'
        TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        default_surf = SurfaceRZFourier.from_vmec_input(
            surface_filename, range='full torus', nphi=16, ntheta=16
        )
        psc_grid.plasma_boundary = kwargs.pop("plasma_boundary", default_surf)
        psc_grid.nphi = len(psc_grid.plasma_boundary.quadpoints_phi)
        psc_grid.ntheta = len(psc_grid.plasma_boundary.quadpoints_theta)
        # generate planar TF coils
        ncoils = 2
        R0 = 1.0
        R1 = 0.65
        order = 4
        # qa needs to be scaled to 0.1 T on-axis magnetic field strength
        total_current = 1e3
        base_curves = create_equally_spaced_curves(
            ncoils, psc_grid.plasma_boundary.nfp, 
            stellsym=psc_grid.plasma_boundary.stellsym, 
            R0=R0, R1=R1, order=order, numquadpoints=32
        )
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils-1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        default_coils = coils_via_symmetries(base_curves, base_currents, 
                                     psc_grid.plasma_boundary.nfp, 
                                     psc_grid.plasma_boundary.stellsym)
        # fix all the coil shapes so only the currents are optimized
        for i in range(ncoils):
            base_curves[i].fix_all()
            
        B_TF = kwargs.pop("B_TF", BiotSavart(default_coils))
        
        # Order of the coils. For unit tests, needs > 400
        psc_grid.ppp = kwargs.pop("ppp", 400)
        
        # Need interpolated region big enough to cover all the coils and the plasma
        n = 20
        degree = 2
        rs = np.linalg.norm(psc_grid.grid_xyz[:, :2] + R ** 2, axis=-1)
        zs = psc_grid.grid_xyz[:, 2]
        rrange = (0, np.max(rs) + R, n)  # need to also cover the plasma
        phirange = (0, 2 * np.pi, n * 2)
        zrange = (np.min(zs) - R, np.max(zs) + R, n // 2)
        psc_grid.B_TF = InterpolatedField(
            B_TF, degree, rrange, phirange, zrange, True, nfp=1, stellsym=False
        )
        # psc_grid.B_TF = B_TF
        
        contig = np.ascontiguousarray
        pointsToVTK('psc_grid',
                    contig(psc_grid.grid_xyz[:, 0]),
                    contig(psc_grid.grid_xyz[:, 1]),
                    contig(psc_grid.grid_xyz[:, 2]))
        print('Number of PSC locations = ', len(psc_grid.grid_xyz))

        # PSC coil geometry determined by its center point in grid_xyz
        # and its alpha and delta angles, which we initialize randomly here.
        # psc_grid.alphas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
        # psc_grid.deltas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
        psc_grid.coil_normals = np.array(
            [np.cos(alphas) * np.sin(deltas),
              -np.sin(alphas),
              np.cos(alphas) * np.cos(deltas)]
        ).T
        # psc_grid.coil_normals = np.array(
        #     [np.cos(deltas) * np.sin(alphas),
        #      np.sin(deltas) * np.sin(alphas),
        #      np.cos(alphas)]
        # ).T
        
        # Initialize curve objects corresponding to each PSC coil for 
        # plotting in 3D
        psc_grid.setup_orientations(psc_grid.alphas, psc_grid.deltas)
        psc_grid.plot_curves()
        
        return psc_grid
    
    def setup_currents_and_fields(self):
        """ """
        from simsopt.field import coils_via_symmetries, Current, CircularCoil

        # Make coils that can be used with BiotSavart
        # L_inv = np.linalg.inv(self.L)
        contig = np.ascontiguousarray
        print('psi = ', self.psi)
        print('min/max psi = ', np.min(abs(self.psi)), np.max(abs(self.psi)))
        print(self.L)
        # exit()
        L_inv = np.linalg.pinv(self.L, rcond=1e-12)
        print(np.linalg.cond(self.L))
        u, s, v = np.linalg.svd(self.L)
        
        import matplotlib.pyplot as plt
        plt.semilogy(s)
        plt.grid()
        plt.show()
        # exit()
        # self.I = -np.linalg.solve(self.L, self.psi)
        self.I = -L_inv @ self.psi
        currents = []
        for i in range(len(self.I)):
            currents.append(Current(self.I[i]))
        # print(self.L)
        
        self.coils = coils_via_symmetries(
            self.curves, currents, nfp=1, stellsym=False
        )
        self.B_PSC = BiotSavart(self.coils)
        self.B_PSC.set_points(self.plasma_boundary.gamma().reshape(-1, 3))
        self.Bn_PSC = np.sum(self.B_PSC.B().reshape(
            -1, 3) * self.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
        # self.coils_z = coils_via_symmetries(
        #     self.curves_z, currents, nfp=1, stellsym=False
        # )
        # self.B_PSC_z = BiotSavart(self.coils_z)
        # self.B_PSC_z.set_points(self.plasma_boundary.gamma().reshape(-1, 3) @ self.rotation_matrix)
        # self.Bn_PSC_z = np.sum(self.B_PSC_z.B().reshape(
        #     -1, 3) * self.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
        # print('here = ', self.B_PSC.B(), self.B_PSC_z.B())
        PSC_check = sopp.Bn_PSC(
            contig(self.grid_xyz),
            contig(self.plasma_boundary.gamma().reshape(-1, 3)),
            contig(self.alphas),
            contig(self.deltas),
            contig(self.plasma_boundary.unitnormal().reshape(-1, 3)),
            contig(self.coil_normals.reshape(-1, 3)),
            self.R
        )
        Bn = 0.0
        for i in range(len(self.alphas)):
            PSC = CircularCoil(self.R, self.grid_xyz[i, :], self.I[i], self.coil_normals[i, :])
            PSC.set_points(self.plasma_boundary.gamma().reshape(-1, 3))
            Bn_i = np.sum(PSC.B().reshape(
                -1, 3) * self.plasma_boundary.unitnormal().reshape(-1, 3), axis=-1)
            Bn += Bn_i
        print(self.I)
        print('BN check = ', PSC_check @ self.I, self.Bn_PSC, Bn)
        inds = np.ravel(np.where(~np.isclose(PSC_check @ self.I, self.Bn_PSC)))
        print('BN check = ', (PSC_check @ self.I)[inds], self.Bn_PSC[inds], Bn[inds])
        assert(np.allclose(PSC_check @ self.I, self.Bn_PSC, rtol=1e-2))
        # assert(np.allclose(PSC_check @ self.I, Bn, rtol=1e-2))
    
    def setup_curves(self):
        """ """
        from . import CurvePlanarFourier

        order = 1
        ncoils = self.num_psc
        curves = [CurvePlanarFourier(order*self.ppp, order, nfp=1, stellsym=False) for i in range(ncoils)]
        # curves_z = [CurvePlanarFourier(order*ppp, order, nfp=1, stellsym=False) for i in range(ncoils)]
        for ic in range(ncoils):
            # nx = self.coil_normals[ic, 0]
            # ny = self.coil_normals[ic, 0]
            # nz = self.coil_normals[ic, 0]
            xyz = self.grid_xyz[ic, :]
            dofs = np.zeros(10)
            # dofs1 = np.zeros(10)
            dofs[0] = self.R
            dofs[1] = 0.0
            dofs[2] = 0.0
            # dofs1[0] = self.R
            # dofs1[1] = 0.0
            # dofs1[2] = 0.0
            # Conversion from Euler angles in 3-2-1 body sequence to 
            # quaternions: 
            # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
            dofs[3] = np.cos(self.alphas[ic] / 2.0) * np.cos(self.deltas[ic] / 2.0)
            dofs[4] = np.sin(self.alphas[ic] / 2.0) * np.cos(self.deltas[ic] / 2.0)
            dofs[5] = np.cos(self.alphas[ic] / 2.0) * np.sin(self.deltas[ic] / 2.0)
            dofs[6] = -np.sin(self.alphas[ic] / 2.0) * np.sin(self.deltas[ic] / 2.0)
            # dofs[3] = np.cos(self.alphas[ic] / 2.0)
            # dofs[4] = -np.sin(self.alphas[ic] / 2.0) * np.sin(self.deltas[ic])
            # dofs[5] = np.sin(self.alphas[ic] / 2.0) * np.cos(self.deltas[ic])
            # dofs[6] = 0.0
            # phi_rot = np.arccos(np.cos(self.alphas[ic] / 2.0) * np.cos(self.deltas[ic] / 2.0)) * 2.0
            # dofs[3] = np.cos(phi_rot / 2.0)
            # dofs[4] = nx * np.sin(phi_rot / 2.0)
            # dofs[5] = ny * np.sin(phi_rot / 2.0)
            # dofs[6] = nz * np.sin(phi_rot / 2.0)
            # print(dofs[3:7])
            # dofs1[3] = 1.0
            # dofs1[4] = 0.0
            # dofs1[5] = 0.0
            # dofs1[6] = 0.0
            # Now specify the center 
            dofs[7] = xyz[0]
            dofs[8] = xyz[1]
            dofs[9] = xyz[2] 
            # dofs1[7] = xyz[0]
            # dofs1[8] = xyz[1]
            # dofs1[9] = xyz[2] 
            curves[ic].set_dofs(dofs)
            # curves_z[ic].set_dofs(dofs)
        self.curves = curves
        # self.curves_z = curves_z

    def plot_curves(self):
        
        from pyevtk.hl import pointsToVTK
        from . import curves_to_vtk
        from simsopt.field import InterpolatedField
        
        curves_to_vtk(self.curves, "psc_curves", close=True, scalar_data=self.I)
        contig = np.ascontiguousarray
        if isinstance(self.B_TF, InterpolatedField):
            self.B_TF.set_points(self.grid_xyz)
            B = self.B_TF.B()
            pointsToVTK('curve_centers', 
                        contig(self.grid_xyz[:, 0]),
                        contig(self.grid_xyz[:, 1]), 
                        contig(self.grid_xyz[:, 2]),
                        data={"n": (contig(self.coil_normals[:, 0]), 
                                    contig(self.coil_normals[:, 1]),
                                    contig(self.coil_normals[:, 2])),
                              "psi": contig(self.psi),
                              "I": contig(self.I),
                              "B_TF": (contig(B[:, 0]), 
                                       contig(B[:, 1]),
                                       contig(B[:, 2])),
                              },
            )
        else:
            pointsToVTK('curve_centers', 
                        contig(self.grid_xyz[:, 0]),
                        contig(self.grid_xyz[:, 1]), 
                        contig(self.grid_xyz[:, 2]),
                        data={"n": (contig(self.coil_normals[:, 0]), 
                                    contig(self.coil_normals[:, 1]),
                                    contig(self.coil_normals[:, 2])),
                              },
            )
    
    def inductances(self, alphas, deltas):
        """ Calculate the inductance matrix needed for the PSC forward problem """
        warnings.filterwarnings("error")

        points = self.grid_xyz
        nphi = 10
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        dphi = phi[1] - phi[0]
        R = self.R
        r = self.a
        ncoils = self.num_psc
        L = np.zeros((ncoils, ncoils))
        
        def integrand(yy, xx, cdi, sdi, cai, sai, xj, yj, zj, cdj, sdj, caj, saj):
            ck = np.cos(xx)
            sk = np.sin(xx)
            ckk = np.cos(yy)
            skk = np.sin(yy)
            return ((-sk * cdi + ck * sai * sdi) * (
                -skk * cdj + ckk * saj * sdj
                ) + (ck * cai) * (ckk * caj) + (
                    sk * sdi + ck * sai * cdi
                    ) * (skk * sdj + ckk * saj * cdj
                         )) / np.sqrt((
                             xj + ckk * cdj + skk * saj * sdj - ck * cdi - sk * sai * sdi
                             ) ** 2 + (
                                 yj + skk * caj - sk * cai
                                 ) ** 2 + (
                                     zj + skk * saj * cdj - ckk * sdj - sk * sai * cdi + ck * sdi
                                     ) ** 2)
        
        for i in range(ncoils):
    	    # Loop through all the PSCs, using all the symmetries
            cdi = np.cos(deltas[i])
            sdi = np.sin(deltas[i])
            cai = np.cos(alphas[i])
            sai = np.sin(alphas[i])
            for j in range(i + 1, ncoils):
                xj = (points[j, 0] - points[i, 0]) / R
                yj = (points[j, 1] - points[i, 1]) / R
                zj = (points[j, 2] - points[i, 2]) / R
                cdj = np.cos(deltas[j])
                sdj = np.sin(deltas[j])
                caj = np.cos(alphas[j])
                saj = np.sin(alphas[j])
                try:
                    integral, err = dblquad(
                        integrand, 0, 2 * np.pi, 0, 2 * np.pi, 
                        args=(cdi, sdi, cai, sai, xj, yj, zj, cdj, sdj, caj, saj)
                    )
                except IntegrationWarning:
                    print('Integral is divergent')
                L[i, j] = integral
        # symmetrize and scale
        L = (L + L.T) / (4.0 * np.pi)  # * dphi ** 2
        np.fill_diagonal(L, np.log(8.0 * R / r) - 2.0)
        self.L = L * self.mu0 * R * self.Nt ** 2
        
    def fluxes(self, alphas, deltas, plot_vtk=False):
        warnings.filterwarnings("error")
        bs = self.B_TF
        ncoils = len(alphas)
        contig = np.ascontiguousarray
        fluxes = np.zeros(ncoils)
        def quad_func(yy, xx, cd, ca, sd, sa, nj, centers):
            x0 = xx * np.cos(yy)
            y0 = xx * np.sin(yy)
            x = x0 * cd + y0 * sa * sd + centers[0]
            y = y0 * ca + centers[1]
            z = -x0 * sd + y0 * sa * cd + centers[2]
            bs.set_points(contig(np.array([x, y, z]).reshape(-1, 3)))
            return (bs.B() @ nj) * xx

        for j in range(ncoils):
            try:
                fluxes[j], _ = dblquad(
                    quad_func, 0, self.R, 0, 2 * np.pi, epsabs=1, epsrel=1,
                    args=(np.cos(deltas[j]), 
                          np.cos(alphas[j]), 
                          np.sin(deltas[j]), 
                          np.sin(alphas[j]), 
                          self.coil_normals[j, :], 
                          self.grid_xyz[j, :])
                )
            except IntegrationWarning:
                print('Integral is divergent')
        self.psi = fluxes
        
    def b_vector(self):
        Bn_target = self.Bn.reshape(-1)
        self.B_TF.set_points(self.plasma_boundary.gamma().reshape(-1, 3))
        Bn_TF = np.sum(
            self.B_TF.B().reshape(-1, 3) * self.plasma_boundary.unitnormal(
                ).reshape(-1, 3), axis=-1
        )
        self.b_opt = (Bn_TF + Bn_target) * self.grid_normalization
        
    def least_squares(self, kappas):
        alphas = kappas[:len(kappas) // 2]
        deltas = kappas[len(kappas) // 2:]
        self.setup_orientations(alphas, deltas)
        BdotN2 = 0.5 * np.sum((self.Bn_PSC.reshape(-1) * self.grid_normalization + self.b_opt) ** 2)
        outstr = f"||Bn||^2 = {BdotN2:.2e} "
        # for i in range(len(kappas)):
        #     outstr += f"kappas[{i:d}] = {kappas[i]:.2e} "
        # outstr += "\n"
        print(outstr)
        return BdotN2
        
    def setup_orientations(self, alphas, deltas):
        contig = np.ascontiguousarray
        self.coil_normals = np.array(
            [np.cos(alphas) * np.sin(deltas),
              -np.sin(alphas),
              np.cos(alphas) * np.cos(deltas)]
        ).T
        rotation_matrix = np.zeros((len(alphas), 3, 3))
        for i in range(len(alphas)):
            rotation_matrix[i, :, :] = np.array(
                [[np.cos(deltas[i]), 
                  np.sin(deltas[i]) * np.sin(alphas[i]),
                  np.sin(deltas[i]) * np.cos(alphas[i])],
                  [0.0, np.cos(alphas[i]), -np.sin(alphas[i])],
                  [-np.sin(deltas[i]), 
                   np.cos(deltas[i]) * np.sin(alphas[i]),
                   np.cos(deltas[i]) * np.cos(alphas[i])]])
        self.rotation_matrix = rotation_matrix
        # self.coil_normals = np.array(
        #     [np.cos(deltas) * np.sin(alphas),
        #      np.sin(deltas) * np.sin(alphas),
        #      np.cos(alphas)]
        # ).T
        # t1 = time.time()
        flux_grid = sopp.flux_xyz(
            contig(self.grid_xyz), 
            contig(alphas),
            contig(deltas), 
            contig(self.rho), 
            contig(self.phi), 
            contig(self.coil_normals)
        )
        # t2 = time.time()
        # print('Fluxes time = ', t2 - t1)
        # t1 = time.time()
        self.B_TF.set_points(contig(np.array(flux_grid).reshape(-1, 3)))
        print('max = ', np.max(np.abs(self.B_TF.B())))
        # t2 = time.time()
        # print('Flux set points time = ', t2 - t1)
        N = len(self.rho)
        # t1 = time.time()
        # B = self.B_TF.B()  # .reshape(len(alphas), N, N, 3)
        # t2 = time.time()
        # print('Flux call B only time = ', t2 - t1)
        # t1 = time.time()
        B = self.B_TF.B().reshape(len(alphas), N, N, 3)
        # t2 = time.time()
        # print('Flux call B time = ', t2 - t1)
        # t1 = time.time()
        self.psi = sopp.flux_integration(
            contig(B),
            contig(self.rho),
            contig(self.coil_normals)
        )
        self.psi *= self.dphi * self.drho
        # print(flux_grid)
        # t2 = time.time()
        # print('Flux integration time = ', t2 - t1)
        # t1 = time.time()
        self.L = sopp.L_matrix(
            contig(self.grid_xyz), 
            contig(alphas), 
            contig(deltas),
            contig(self.phi),
            self.R,
        ) * self.dphi ** 2 / (4 * np.pi)
        # t2 = time.time()
        # print('Inductances c++ otal time = ', t2 - t1)
        self.L = (self.L + self.L.T)
        np.fill_diagonal(self.L, np.log(8.0 * self.R / self.a) - 2.0)
        self.L = self.L * self.mu0 * self.R * self.Nt ** 2
        # t1 = time.time()
        self.setup_curves()
        # t2 = time.time()
        # print('Setup curves time = ', t2 - t1)
        # t1 = time.time()
        self.setup_currents_and_fields()
        # t2 = time.time()
        # print('Setup fields time = ', t2 - t1)
        