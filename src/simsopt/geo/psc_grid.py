from pathlib import Path
import warnings

import numpy as np
from pyevtk.hl import pointsToVTK

from . import Surface
import simsoptpp as sopp
from simsopt.field import BiotSavart
import time

__all__ = ['PSCgrid']


class PSCgrid:
    r"""
    ``PSCgrid`` is a class for setting up the grid, normal vectors,
    plasma surface, and other objects needed to perform PSC
    optimization for stellarators. The class
    takes as input either (1) two toroidal surfaces specified as 
    SurfaceRZFourier objects, and initializes a set of points 
    (in Cartesian coordinates) between these surfaces or (2) a manual
    list of xyz locations for the coils, for which the user needs to make
    sure that coils are non-intersecting even when they are arbitrarily
    rotated around. 

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
        print(Nx, Ny, Nz, self.dx, self.dy, self.dz)
        Nmin = min(self.dx, min(self.dy, self.dz))
        self.R = Nmin / 4.0
        if self.poff is not None:
            self.R = min(self.R, self.poff / 2.0)  # Need to keep coils from touching, so denominator should be > 2

        self.a = self.R / 100.0  # Hard-coded aspect ratio of 100 right now
        print('Major radius of the coils is R = ', self.R)
        print('Coils are spaced so that every coil of radius R '
              ' is at least 2R away from the next coil'
        )

        if self.plasma_boundary.nfp > 1:
            # Throw away any points not in the section phi = [0, pi / n_p] and
            # make sure all centers points are at least a distance R from the
            # sector so that all the coil points are reflected correctly. 
            X = np.linspace(
                self.dx / 2.0 + x_min, x_max - self.dx / 2.0, 
                Nx, endpoint=True
            )
            Y = np.linspace(
                self.dy / 2.0 + y_min, y_max - self.dy / 2.0, 
                Ny, endpoint=True
            )
        else:
            X = np.linspace(x_min, x_max, Nx, endpoint=True)
            Y = np.linspace(y_min, y_max, Ny, endpoint=True)
        Z = np.linspace(-z_max, z_max, Nz, endpoint=True)

        # Make 3D mesh
        X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
        self.xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(Nx * Ny * Nz, 3)
        
        # Extra work for nfp > 1 to chop off points outside sector
        if self.nfp > 1:
            inds = []
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        phi = np.arctan2(Y[i, j, k], X[i, j, k])
                        phi2 = np.arctan2(self.R, X[i, j, k])
                        # Add a little factor to avoid phi = pi / n_p degrees 
                        # exactly, which can intersect with a symmetrized
                        # coil if not careful 
                        if phi >= (np.pi / self.nfp - phi2) or phi < 0.0:
                            inds.append(int(i * Ny * Nz + j * Nz + k))
            good_inds = np.setdiff1d(np.arange(Nx * Ny * Nz), inds)
            self.xyz_uniform = self.xyz_uniform[good_inds, :]

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
        from simsopt.util import calculate_on_axis_B
                
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
        psc_grid.nfp = plasma_boundary.nfp
        psc_grid.stellsym = plasma_boundary.stellsym
        psc_grid.symmetry = plasma_boundary.nfp * (plasma_boundary.stellsym + 1)
        if psc_grid.stellsym:
            psc_grid.stell_list = [1, -1]
        else:
            psc_grid.stell_list = [1]
        psc_grid.nphi = len(psc_grid.plasma_boundary.quadpoints_phi)
        psc_grid.ntheta = len(psc_grid.plasma_boundary.quadpoints_theta)
        psc_grid.plasma_unitnormals = psc_grid.plasma_boundary.unitnormal().reshape(-1, 3)
        psc_grid.plasma_points = psc_grid.plasma_boundary.gamma().reshape(-1, 3)
        Ngrid = psc_grid.nphi * psc_grid.ntheta
        Nnorms = np.ravel(np.sqrt(np.sum(psc_grid.plasma_boundary.normal() ** 2, axis=-1)))
        psc_grid.grid_normalization = np.sqrt(Nnorms / Ngrid)
        # integration over phi for L
        N = 10
        psc_grid.phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
        psc_grid.dphi = psc_grid.phi[1] - psc_grid.phi[0]
        N = 40
        psc_grid.flux_phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
        psc_grid.flux_dphi = psc_grid.flux_phi[1] - psc_grid.flux_phi[0]
        Nx = kwargs.pop("Nx", 10)
        Ny = Nx  # kwargs.pop("Ny", 10)
        Nz = Nx  # kwargs.pop("Nz", 10)
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError('Nx, Ny, and Nz should be positive integers')
        psc_grid.Nx = Nx
        psc_grid.Ny = Ny
        psc_grid.Nz = Nz
        psc_grid.poff = kwargs.pop("poff", None)
        psc_grid.inner_toroidal_surface = inner_toroidal_surface.to_RZFourier()
        psc_grid.outer_toroidal_surface = outer_toroidal_surface.to_RZFourier()    
        warnings.warn(
            'Plasma boundary and inner and outer toroidal surfaces should '
            'all have the same "range" parameter in order for a permanent'
            ' magnet grid to be correctly initialized.'
        )        

        Nt = kwargs.pop("Nt", 1)
        psc_grid.Nt = Nt
        
        psc_grid.out_dir = kwargs.pop("out_dir", '')

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
        psc_grid.ppp = kwargs.pop("ppp", 100)

        # PSC coil geometry determined by its center point in grid_xyz
        # and its alpha and delta angles, which we initialize randomly here.
        
        # Initialize the coil orientations parallel to the B field. 
        psc_grid.B_TF = B_TF
        # n = 40  # tried 20 here and then there are small errors in f_B 
        # degree = 3
        # R = psc_grid.R
        # gamma_outer = outer_toroidal_surface.gamma().reshape(-1, 3)
        # rs = np.linalg.norm(gamma_outer[:, :2], axis=-1)
        # zs = gamma_outer[:, 2]
        # rrange = (0, np.max(rs) + R, n)  # need to also cover the plasma
        # if psc_grid.nfp > 1:
        #     phirange = (0, 2*np.pi/psc_grid.nfp, n*2)
        # else:
        #     phirange = (0, 2 * np.pi, n * 2)
        # if psc_grid.stellsym:
        #     zrange = (0, np.max(zs) + R, n // 2)
        # else:
        #     zrange = (np.min(zs) - R, np.max(zs) + R, n // 2)
        # psc_grid.B_TF = InterpolatedField(
        #     B_TF, degree, rrange, phirange, zrange, 
        #     True, nfp=psc_grid.nfp, stellsym=psc_grid.stellsym
        # )
        B_TF.set_points(psc_grid.grid_xyz)
        B = B_TF.B()
        B_axis = calculate_on_axis_B(B_TF, psc_grid.plasma_boundary, print_out=False)
        psc_grid.normalization = B_axis ** 2 * psc_grid.plasma_boundary.area()

        # Randomly initialize the coil orientations
        random_initialization = kwargs.pop("random_initialization", False)
        if random_initialization:
            psc_grid.alphas = (np.random.rand(psc_grid.num_psc) - 0.5) * 2 * np.pi
            psc_grid.deltas = (np.random.rand(psc_grid.num_psc) - 0.5) * 2 * np.pi
            psc_grid.coil_normals = np.array(
                [np.cos(psc_grid.alphas) * np.sin(psc_grid.deltas),
                  -np.sin(psc_grid.alphas),
                  np.cos(psc_grid.alphas) * np.cos(psc_grid.deltas)]
            ).T
            # deal with -0 terms in the normals, which screw up the arctan2 calculations
            psc_grid.coil_normals[
                np.logical_and(np.isclose(psc_grid.coil_normals, 0.0), 
                               np.copysign(1.0, psc_grid.coil_normals) < 0)
                ] *= -1.0
        else:
            # determine the alphas and deltas from these normal vectors
            psc_grid.coil_normals = (B.T / np.sqrt(np.sum(B ** 2, axis=-1))).T
            # deal with -0 terms in the normals, which screw up the arctan2 calculations
            psc_grid.coil_normals[
                np.logical_and(np.isclose(psc_grid.coil_normals, 0.0), 
                               np.copysign(1.0, psc_grid.coil_normals) < 0)
                ] *= -1.0
            deltas = np.arctan2(psc_grid.coil_normals[:, 0], psc_grid.coil_normals[:, 2])  #+ np.pi
            alphas = -np.arctan2(psc_grid.coil_normals[:, 1] * np.cos(deltas), psc_grid.coil_normals[:, 2])
            # alphas = -np.arcsin(psc_grid.coil_normals[:, 1])
            
        psc_grid.setup_full_grid()
        t2 = time.time()
        print('Initialize grid time = ', t2 - t1)
        
        # Initialize curve objects corresponding to each PSC coil for 
        # plotting in 3D
        
        t1 = time.time()
        psc_grid.setup_orientations(psc_grid.alphas, psc_grid.deltas)
        t2 = time.time()
        print('Geo setup time = ', t2 - t1)
        psc_grid.setup_curves()
        psc_grid.plot_curves()
        psc_grid.b_vector()
        kappas = np.ravel(np.array([psc_grid.alphas, psc_grid.deltas]))
        psc_grid.kappas = kappas
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
        from simsopt.util.permanent_magnet_helper_functions import initialize_coils, calculate_on_axis_B
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
        psc_grid.out_dir = kwargs.pop("out_dir", '')
        
        psc_grid.num_psc = psc_grid.grid_xyz.shape[0]
        N = 1000  # decrease if not doing unit tests
        psc_grid.phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
        psc_grid.dphi = psc_grid.phi[1] - psc_grid.phi[0]
        psc_grid.rho = np.linspace(0, R, N, endpoint=False)
        psc_grid.drho = psc_grid.rho[1] - psc_grid.rho[0]
        psc_grid.flux_phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
        # initialize default plasma boundary
        input_name = 'input.LandremanPaul2021_QA_lowres'
        TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        ndefault = 4
        default_surf = SurfaceRZFourier.from_vmec_input(
            surface_filename, range='full torus', nphi=ndefault, ntheta=ndefault
        )
        default_surf.nfp = 1
        default_surf.stellsym = False
        psc_grid.plasma_boundary = kwargs.pop("plasma_boundary", default_surf)
        ndefault = len(psc_grid.plasma_boundary.quadpoints_phi)
        psc_grid.nfp = psc_grid.plasma_boundary.nfp
        psc_grid.stellsym = psc_grid.plasma_boundary.stellsym
        psc_grid.nphi = len(psc_grid.plasma_boundary.quadpoints_phi)
        psc_grid.ntheta = len(psc_grid.plasma_boundary.quadpoints_theta)
        psc_grid.plasma_unitnormals = psc_grid.plasma_boundary.unitnormal().reshape(-1, 3)
        psc_grid.plasma_points = psc_grid.plasma_boundary.gamma().reshape(-1, 3)
        psc_grid.symmetry = psc_grid.plasma_boundary.nfp * (psc_grid.plasma_boundary.stellsym + 1)

        if psc_grid.stellsym:
            psc_grid.stell_list = [1, -1]
        else:
            psc_grid.stell_list = [1]
        Ngrid = psc_grid.nphi * psc_grid.ntheta
        Nnorms = np.ravel(np.sqrt(np.sum(psc_grid.plasma_boundary.normal() ** 2, axis=-1)))
        psc_grid.grid_normalization = np.sqrt(Nnorms / Ngrid)
        Bn = kwargs.pop("Bn", np.zeros((ndefault, ndefault)))
        Bn = np.array(Bn)
        if len(Bn.shape) != 2: 
            raise ValueError('Normal magnetic field surface data is incorrect shape.')
        psc_grid.Bn = Bn
        
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
        default_coils = coils_via_symmetries(
            base_curves, base_currents, 
            psc_grid.plasma_boundary.nfp, 
            psc_grid.plasma_boundary.stellsym
        )
        # fix all the coil shapes so only the currents are optimized
        for i in range(ncoils):
            base_curves[i].fix_all()
            
        B_TF = kwargs.pop("B_TF", BiotSavart(default_coils))
        
        # Order of the coils. For unit tests, needs > 400
        psc_grid.ppp = kwargs.pop("ppp", 1000)
        
        # Need interpolated region big enough to cover all the coils and the plasma
        # n = 20
        # degree = 2
        # rs = np.linalg.norm(psc_grid.grid_xyz[:, :2] + R ** 2, axis=-1)
        # zs = psc_grid.grid_xyz[:, 2]
        # rrange = (0, np.max(rs) + R, n)  # need to also cover the plasma
        # if psc_grid.nfp > 1:
        #     phirange = (0, 2*np.pi/psc_grid.nfp, n*2)
        # else:
        #     phirange = (0, 2 * np.pi, n * 2)
        # if psc_grid.stellsym:
        #     zrange = (0, np.max(zs) + R, n // 2)
        # else:
        #     zrange = (np.min(zs) - R, np.max(zs) + R, n // 2)
        # psc_grid.B_TF = InterpolatedField(
        #     B_TF, degree, rrange, phirange, zrange, 
        #     True, nfp=psc_grid.nfp, stellsym=psc_grid.stellsym
        # )
        psc_grid.B_TF = B_TF
        B_axis = calculate_on_axis_B(B_TF, psc_grid.plasma_boundary, print_out=False)
        psc_grid.normalization = B_axis ** 2 * psc_grid.plasma_boundary.area()
        
        contig = np.ascontiguousarray
        pointsToVTK('psc_grid',
                    contig(psc_grid.grid_xyz[:, 0]),
                    contig(psc_grid.grid_xyz[:, 1]),
                    contig(psc_grid.grid_xyz[:, 2]))
        print('Number of PSC locations = ', len(psc_grid.grid_xyz))

        psc_grid.coil_normals = np.array(
            [np.cos(alphas) * np.sin(deltas),
              -np.sin(alphas),
              np.cos(alphas) * np.cos(deltas)]
        ).T
        # deal with -0 terms in the normals, which screw up the arctan2 calculations
        psc_grid.coil_normals[
            np.logical_and(np.isclose(psc_grid.coil_normals, 0.0), 
                           np.copysign(1.0, psc_grid.coil_normals) < 0)
            ] *= -1.0
        
        # Initialize curve objects corresponding to each PSC coil for 
        # plotting in 3D
        psc_grid.setup_full_grid()
        psc_grid.setup_orientations(psc_grid.alphas, psc_grid.deltas)
        psc_grid.setup_curves()
        psc_grid.plot_curves()
        psc_grid.b_vector()
        kappas = np.ravel(np.array([psc_grid.alphas, psc_grid.deltas]))
        psc_grid.kappas = kappas
        psc_grid.A_opt = psc_grid.least_squares(kappas)
        
        return psc_grid
    
    def setup_currents_and_fields(self):
        """ 
        Calculate the currents and fields from all the PSCs. Note that L 
        will be well-conditioned unless you accidentally initialized
        the PSC coils touching/intersecting, in which case it will be
        VERY ill-conditioned! 
        
        """
        from simsopt.field import Current, coils_via_symmetries
        
        # Note L will be well-conditioned unless you accidentally initialized
        # the PSC coils touching/intersecting, in which case it will be
        # VERY ill-conditioned!
        self.L_inv = np.linalg.inv(self.L)
        self.psi_total = np.zeros(self.num_psc * self.symmetry)
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                self.psi_total[q * self.num_psc:(q + 1) * self.num_psc] = self.psi * stell
                q = q + 1
        self.I_all = (-self.L_inv @ self.psi_total)
        self.I = self.I_all[:self.num_psc]
        contig = np.ascontiguousarray
        # A_matrix has shape (num_plasma_points, num_coils)
        B_PSC = np.zeros((self.nphi * self.ntheta, 3))
        A_matrix = np.zeros((self.nphi * self.ntheta, self.num_psc))
        # Need to rotate and flip it
        nn = self.num_psc
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                phi0 = (2.0 * np.pi / self.nfp) * fp
                ox = self.grid_xyz_all[q * nn: (q + 1) * nn, 0]
                oy = self.grid_xyz_all[q * nn: (q + 1) * nn, 1]
                oz = self.grid_xyz_all[q * nn: (q + 1) * nn, 2]
                xyz = np.array([ox, oy, oz]).T
                normals = self.coil_normals_all[q * nn: (q + 1) * nn, :]
                A_matrix += sopp.A_matrix(
                    contig(xyz),
                    contig(self.plasma_points),
                    contig(self.alphas_total[q * nn: (q + 1) * nn]),
                    contig(self.deltas_total[q * nn: (q + 1) * nn]),
                    contig(self.plasma_unitnormals),
                    self.R,
                )  # accounts for sign change of the currents
                B_PSC += sopp.B_PSC(
                    contig(xyz),
                    contig(self.plasma_points),
                    contig(self.alphas_total[q * nn: (q + 1) * nn]),
                    contig(self.deltas_total[q * nn: (q + 1) * nn]),
                    contig(self.I),
                    self.R,
                )
                q = q + 1
        self.A_matrix = A_matrix
        self.B_PSC_direct = B_PSC
        self.Bn_PSC = (A_matrix @ self.I).reshape(-1)
        currents = []
        for i in range(self.num_psc):
            currents.append(Current(self.I[i]))
        all_coils = coils_via_symmetries(
            self.curves, currents, nfp=self.nfp, stellsym=self.stellsym
        )
        self.B_PSC = BiotSavart(all_coils)
    
    def setup_curves(self):
        """ """
        from . import CurvePlanarFourier
        from simsopt.field import apply_symmetries_to_curves

        order = 1
        ncoils = self.num_psc
        curves = [CurvePlanarFourier(order*self.ppp, order, nfp=1, stellsym=False) for i in range(ncoils)]
        for ic in range(ncoils):
            xyz = self.grid_xyz[ic, :]
            dofs = np.zeros(10)
            dofs[0] = self.R
            dofs[1] = 0.0
            dofs[2] = 0.0
            # Conversion from Euler angles in 3-2-1 body sequence to 
            # quaternions: 
            # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
            dofs[3] = np.cos(self.alphas[ic] / 2.0) * np.cos(self.deltas[ic] / 2.0)
            dofs[4] = np.sin(self.alphas[ic] / 2.0) * np.cos(self.deltas[ic] / 2.0)
            dofs[5] = np.cos(self.alphas[ic] / 2.0) * np.sin(self.deltas[ic] / 2.0)
            dofs[6] = -np.sin(self.alphas[ic] / 2.0) * np.sin(self.deltas[ic] / 2.0)
            # Now specify the center 
            dofs[7] = xyz[0]
            dofs[8] = xyz[1]
            dofs[9] = xyz[2] 
            curves[ic].set_dofs(dofs)
        self.curves = curves
        self.all_curves = apply_symmetries_to_curves(curves, self.nfp, self.stellsym)

    def plot_curves(self, filename=''):
        
        from pyevtk.hl import pointsToVTK
        from . import curves_to_vtk
        from simsopt.field import InterpolatedField, Current, coils_via_symmetries
        
        curves_to_vtk(self.curves, self.out_dir + "psc_curves", close=True, scalar_data=self.I)
        contig = np.ascontiguousarray
        if isinstance(self.B_TF, InterpolatedField):
            self.B_TF.set_points(self.grid_xyz)
            B = self.B_TF.B()
            pointsToVTK(self.out_dir + 'curve_centers', 
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
            pointsToVTK(self.out_dir + 'curve_centers', 
                        contig(self.grid_xyz[:, 0]),
                        contig(self.grid_xyz[:, 1]), 
                        contig(self.grid_xyz[:, 2]),
                        data={"n": (contig(self.coil_normals[:, 0]), 
                                    contig(self.coil_normals[:, 1]),
                                    contig(self.coil_normals[:, 2])),
                              },
            )
        curves_to_vtk(self.all_curves, self.out_dir + filename + "all_psc_curves", close=True, scalar_data=self.I_all)
        contig = np.ascontiguousarray
        if isinstance(self.B_TF, InterpolatedField):
            self.B_TF.set_points(self.grid_xyz_all)
            B = self.B_TF.B()
            pointsToVTK(self.out_dir + filename + 'all_curve_centers', 
                        contig(self.grid_xyz_all[:, 0]),
                        contig(self.grid_xyz_all[:, 1]), 
                        contig(self.grid_xyz_all[:, 2]),
                        data={"n": (contig(self.coil_normals_all[:, 0]), 
                                    contig(self.coil_normals_all[:, 1]),
                                    contig(self.coil_normals_all[:, 2])),
                              "psi": contig(self.psi_total),
                              "I": contig(self.I_all),
                              "B_TF": (contig(B[:, 0]), 
                                       contig(B[:, 1]),
                                       contig(B[:, 2])),
                              },
            )
        else:
            pointsToVTK(self.out_dir + filename + 'all_curve_centers', 
                        contig(self.grid_xyz_all[:, 0]),
                        contig(self.grid_xyz_all[:, 1]), 
                        contig(self.grid_xyz_all[:, 2]),
                        data={"n": (contig(self.coil_normals_all[:, 0]), 
                                    contig(self.coil_normals_all[:, 1]),
                                    contig(self.coil_normals_all[:, 2])),
                              },
            )
        
    def b_vector(self):
        Bn_target = self.Bn.reshape(-1)
        self.B_TF.set_points(self.plasma_points)
        Bn_TF = np.sum(
            self.B_TF.B().reshape(-1, 3) * self.plasma_unitnormals, axis=-1
        )
        self.b_opt = (Bn_TF + Bn_target) * self.grid_normalization
        
    def least_squares(self, kappas):
        alphas = kappas[:len(kappas) // 2]
        deltas = kappas[len(kappas) // 2:]
        self.setup_orientations(alphas, deltas)
        BdotN2 = 0.5 * np.sum((self.Bn_PSC * self.grid_normalization + self.b_opt) ** 2) / self.normalization
        outstr = f"Normalized f_B = {BdotN2:.3e} "
        # for i in range(len(kappas)):
        #     outstr += f"kappas[{i:d}] = {kappas[i]:.2e} "
        # outstr += "\n"
        print(outstr)
        return BdotN2
    
    def least_squares_jacobian(self, kappas):
        """Compute Jacobian of the |B*n|^2 term in the optimization for using BFGS in scipy.minimize"""
        """and should return a gradient of shape 2 * num_pscs"""
        alphas = kappas[:len(kappas) // 2]
        deltas = kappas[len(kappas) // 2:]
        self.setup_orientations(alphas, deltas)
        # Two factors of grid normalization since it is not added to the gradients
        A = (self.Bn_PSC * self.grid_normalization + self.b_opt) * self.grid_normalization / self.normalization
        t1 = time.time()
        grad_A1 = np.tensordot(self.A_deriv(), (self.L_inv @ self.psi_total)[:self.num_psc], axes=([1], [0])).T
        t2 = time.time()
        print('grad_A1 time = ', t2 - t1)
        # Should be shape (1024, 36)
        # L_deriv = self.L_deriv()
        # print(np.tensordot(
        # self.L_inv, L_deriv, axes=([-1], [1])
        # ).shape)
        t1 = time.time()
        grad_A2 = -self.A_matrix @ (np.tensordot(
            np.tensordot(
            self.L_deriv(), self.L_inv, axes=([1], [0])
            ), self.L_inv, axes=([-1], [0])
            ) @ self.psi_total)[:self.num_psc, :]
        Nsym_psc = grad_A2.shape[-1]
        grad_A2 = np.hstack((grad_A2[:, :self.num_psc], 
                             grad_A2[:, Nsym_psc // 2:Nsym_psc // 2 + self.num_psc])) 
        t2 = time.time()
        print('grad_A2 time = ', t2 - t1)
        # print('grad A2 = ', grad_A2, grad_A2.shape)
        # exit()
        t1 = time.time()
        psi_deriv = self.psi_deriv()
        grad_A3 = self.A_matrix @ (self.L_inv[:self.num_psc, :self.num_psc] @ self.psi_deriv().T)
        t2 = time.time()
        print('grad_A3 time = ', t2 - t1)
        # print('grad A3 = ', grad_A3, grad_A3.shape)
        # exit()
        # print(grad_A3.shape)
        # print('grad_A1 = ', grad_A1)
        # print('grad_A2 = ', grad_A2)
        # print('grad_A3 = ', grad_A3)
        return A @ (grad_A1)  #  + grad_A2 + grad_A3)
    
    def A_deriv(self):
        """
        Should return gradient of the A matrix that satisfies A*I = B*n for
        the PSC arrays. in shape (2 * num_psc, num_psc, num_plasma_points).
        This is currently written in slow form with for loops but can be
        vectorized.
        Returns
        -------
            grad_A: 3D numpy array, shape (2 * num_psc, num_psc, num_plasma_points) 
                The gradient of the A matrix with respect to the PSC angles
                alpha_i and delta_i. 
        """
        from simsopt.field import coils_via_symmetries, Current
        
        term0 = np.zeros((self.num_psc, self.plasma_points.shape[0]))
        term1 = np.zeros((2 * self.num_psc, self.plasma_points.shape[0]))
        term2 = np.zeros((2 * self.num_psc, self.plasma_points.shape[0]))
        for i in range(self.num_psc):
            coil_i = coils_via_symmetries(
                [self.curves[i]], [Current(self.I[i])], nfp=1, stellsym=False
            )
            # print(i, self.I[i])
            B_i = BiotSavart(coil_i)
            B_i.set_points(self.plasma_points)
            dB_i = B_i.dB_by_dX()
            BB_i = B_i.B().reshape(-1, 3)
            I_i = self.I[i]
            # print(BB_i)
            for j in range(self.plasma_points.shape[0]):
                term0[i, j] = (
                    BB_i[j, :]
                    ) @ self.plasma_unitnormals[j, :] / I_i
                # term1[i, j] = (
                #     self.dRi_dalphak[i, :, :] @ self.rotation_matrix[i, :, :].T
                #     @ BB_i[j, :]
                #     ) @ self.plasma_unitnormals[j, :] / I_i
                # term1[i + self.num_psc, j] = (
                #     self.dRi_ddeltak[i, :, :] @ BB_i[j, :]
                #     ) @ self.plasma_unitnormals[j, :] / I_i
                term2[i, j] = ((
                    dB_i[j, :, :] @ self.dRi_dalphak[i, :, :].T @ self.plasma_points[j, :]
                    )
                    ) @ self.plasma_unitnormals[j, :] / I_i
                term2[i + self.num_psc, j] = ((
                    dB_i[j, :, :] @ self.dRi_ddeltak[i, :, :].T @ self.plasma_points[j, :]
                    )
                    ) @ self.plasma_unitnormals[j, :] / I_i
                # print(i, j, term2[i, j])
                # print(dB_i[j, :, :])
                # print(self.plasma_points[j, :])
                # print(self.plasma_points[j, :] @ dB_i[j, :, :]@ self.dRi_dalphak[i, :, :].T)
                # print(self.plasma_unitnormals[j, :])
                # exit()
                # term2[i + self.num_psc, j] = (((
                #     self.rotation_matrix[i, :, :] @ dB_i[j, :, :]
                #     ) @ self.plasma_points[j, :]
                #     ) @ self.dRi_ddeltak[i, :, :].T
                #     ) @ self.plasma_unitnormals[j, :] / I_i
        # print('A = ', term0.T, self.A_matrix)
        # exit()
        # print(term1[0, :].T, term2[0, :].T)
        # exit()
        return (term1 + term2) # .T
    
    def dB_dkappa(self):
        """
        Should return gradient of the magnetic field at each point
        Returns
        -------
            dB_dkappa: 3D numpy array, shape (2 * num_psc, num_psc, num_plasma_points) 
                The gradient of the L matrix with respect to the PSC angles
                alpha_i and delta_i. 
        """
        contig = np.ascontiguousarray
        dB_dkappa = sopp.dB_dkappa(
            contig(self.grid_xyz_all), 
            contig(self.plasma_points),
            contig(self.alphas_total), 
            contig(self.deltas_total),
            contig(self.I_all),
            contig(self.phi),
            self.R,
        ) * self.dphi  # rescale because did a phi integral
        
        # symmetrize it?
        return dB_dkappa
    
    def L_deriv(self):
        """
        Should return gradient of the inductance matrix L that satisfies 
        L^(-1) * Psi = I for the PSC arrays.
        Returns
        -------
            grad_L: 3D numpy array, shape (2 * num_psc, num_psc, num_plasma_points) 
                The gradient of the L matrix with respect to the PSC angles
                alpha_i and delta_i. 
        """
        contig = np.ascontiguousarray
        L_deriv = sopp.L_deriv(
            contig(self.grid_xyz_all), 
            contig(self.alphas_total), 
            contig(self.deltas_total),
            contig(self.phi),
            self.R,
        ) * self.dphi ** 2 / (4 * np.pi)
        
        # symmetrize it
        for i in range(L_deriv.shape[0]):
            L_deriv[i, :, :] = (L_deriv[i, :, :] + L_deriv[i, :, :].T)
        return L_deriv * self.mu0 * self.R * self.Nt ** 2
        
    def psi_deriv(self):
        """
        Should return gradient of the Psi vector that satisfies 
        L^(-1) * Psi = I for the PSC arrays.
        This is currently written in slow form with for loops but can be
        vectorized.
        Returns
        -------
            grad_Psi: 3D numpy array, shape (2 * num_psc, num_psc) 
                The gradient of the Psi vector with respect to the PSC angles
                alpha_i and delta_i. 
        """
        contig = np.ascontiguousarray
        
        # Define derivatives of the normal vectors with respect to the angles
        coil_normals_dalpha = np.array(
            [-np.sin(self.alphas) * np.sin(self.deltas),
              -np.cos(self.alphas),
              -np.sin(self.alphas) * np.cos(self.deltas)]
        ).T
        coil_normals_ddelta = np.array(
            [np.cos(self.alphas) * np.cos(self.deltas),
             np.zeros(len(self.alphas)),
             -np.cos(self.alphas) * np.sin(self.deltas)]
        ).T
        
        # First gradient terms have unchanged integrand except normal vectors
        # have been differentiated with respect to the angles
        psi_deriv1 = np.zeros((2 * self.num_psc, self.num_psc))
        N = len(self.rho)
        Nflux = len(self.flux_phi)
        psi_deriv1[:len(self.alphas), :len(self.alphas)] = np.diag(sopp.flux_integration(
            contig(self.B_TF.B().reshape(
                len(self.alphas), N, Nflux, 3
            )),
            contig(self.rho),
            contig(coil_normals_dalpha)
        ))
        psi_deriv1[:len(self.alphas), :len(self.alphas)] = np.diag(sopp.flux_integration(
            contig(self.B_TF.B().reshape(
                len(self.alphas), N, Nflux, 3
            )),
            contig(self.rho),
            contig(coil_normals_ddelta)
        ))
        
        # Need to define the integration points to do the 
        # next derivative term. 
        psi_deriv2 = np.zeros((2 * self.num_psc, self.num_psc))
        Rho, Phi = np.meshgrid(self.rho, self.flux_phi, indexing='ij')
        X = np.outer(self.grid_xyz[:, 0], np.ones(Rho.shape)) + np.outer(np.ones(self.num_psc), Rho * np.cos(Phi))
        Y = np.outer(self.grid_xyz[:, 1], np.ones(Rho.shape)) + np.outer(np.ones(self.num_psc), Rho * np.sin(Phi))
        Z = np.outer(self.grid_xyz[:, 2], np.ones(Rho.shape))
        X = np.ravel(X)
        Y = np.ravel(Y)
        Z = np.ravel(Z)
        coil_points_int = np.array([X, Y, Z]).T
        coil_points_int = coil_points_int.reshape(-1, 3)
        dB = np.transpose(self.B_TF.dB_by_dX(), [0, 2, 1])
        dRi_dalphak = np.outer(np.ones(N * Nflux), self.dRi_dalphak).reshape(coil_points_int.shape[0], 3, 3)
        dBdot = np.zeros((coil_points_int.shape[0], 3))
        for i in range(coil_points_int.shape[0]):
            dBdot[i, :] = (dRi_dalphak[i, :, :] @ dB[i, :, :]) @ coil_points_int[i, :]
        psi_deriv2[:len(self.alphas), :len(self.alphas)] = np.diag(sopp.flux_integration(
            contig(dBdot.reshape(
                len(self.alphas), N, Nflux, 3
            )),
            contig(self.rho),
            contig(self.coil_normals)
        ))
        psi_deriv2[:len(self.alphas), :len(self.alphas)] = np.diag(sopp.flux_integration(
            contig(dBdot.reshape(
                len(self.alphas), N, Nflux, 3
            )),
            contig(self.rho),
            contig(self.coil_normals)
        ))
        # Rescale dPsi by the grid spacing from the integral
        return (psi_deriv1 + psi_deriv2) * self.dphi * self.drho
    
    def setup_orientations(self, alphas, deltas):
        """
        Each time that optimization changes the PSC angles, need to update
        all the fields, currents, and fluxes to be consistent with the 
        new angles. This function does this updating to get the entire class
        object the new angle values. 
        
        Args
        ----------
        alphas : 1D numpy array, shape (num_pscs)
            Rotation angles of every PSC around the x-axis.
        deltas : 1D numpy array, shape (num_pscs)
            Rotation angles of every PSC around the y-axis.
        Returns
        -------
            grad_Psi: 3D numpy array, shape (2 * num_psc, num_psc) 
                The gradient of the Psi vector with respect to the PSC angles
                alpha_i and delta_i. 
        """
        
        # Need to check is alphas and deltas are in [-pi, pi] and remap if not
        self.alphas = alphas
        self.deltas = deltas
        contig = np.ascontiguousarray
        self.coil_normals = np.array(
            [np.cos(alphas) * np.sin(deltas),
              -np.sin(alphas),
              np.cos(alphas) * np.cos(deltas)]
        ).T
        # deal with -0 terms in the normals, which screw up the arctan2 calculations
        self.coil_normals[
            np.logical_and(np.isclose(self.coil_normals, 0.0), 
                           np.copysign(1.0, self.coil_normals) < 0)
            ] *= -1.0
        # Update all the rotation matrices and their derivatives
        self.update_rotation_matrices()
        
        # Update the flux grid with the new normal vectors
        flux_grid = sopp.flux_xyz(
            contig(self.grid_xyz), 
            contig(alphas),
            contig(deltas), 
            contig(self.rho), 
            contig(self.flux_phi), 
            contig(self.coil_normals)
        )
        self.B_TF.set_points(contig(np.array(flux_grid).reshape(-1, 3)))
        # t2 = time.time()
        # print('Flux set points time = ', t2 - t1)
        N = len(self.rho)
        Nflux = len(self.flux_phi)
        # t1 = time.time()
        
        # Update the flux values through the newly rotated coils
        self.psi = sopp.flux_integration(
            contig(self.B_TF.B().reshape(len(alphas), N, Nflux, 3)),
            contig(self.rho),
            contig(self.coil_normals)
        )
        self.psi *= self.dphi * self.drho
        
        # t2 = time.time()
        # print('Flux integration time = ', t2 - t1)
        # t1 = time.time()        
        
        # Apply discrete symmetries to the alphas and deltas and coordinates
        self.update_alphas_deltas()
        # Recompute the inductance matrices with the newly rotated coils
        L_total = sopp.L_matrix(
            contig(self.grid_xyz_all), 
            contig(self.alphas_total), 
            contig(self.deltas_total),
            contig(self.phi),
            self.R,
        ) * self.dphi ** 2 / (4 * np.pi)
        L_total = (L_total + L_total.T)
        # Add in self-inductances
        np.fill_diagonal(L_total, np.log(8.0 * self.R / self.a) - 2.0)
        # rescale inductance
        self.L = L_total * self.mu0 * self.R * self.Nt ** 2
        
        # t1 = time.time()
        # Calculate the PSC currents and B fields
        self.setup_curves()
        self.setup_currents_and_fields()
        # t2 = time.time()
        # print('Setup fields time = ', t2 - t1)
        
    def setup_full_grid(self):
        """
        Initialize the field-period and stellarator symmetrized grid locations
        and normal vectors for all the PSC coils. Note that when alpha_i
        and delta_i angles change, coil_i location does not change, only
        its normal vector changes. 
        """
        nn = self.num_psc
        self.grid_xyz_all = np.zeros((nn * self.symmetry, 3))
        self.alphas_total = np.zeros(nn * self.symmetry)
        self.deltas_total = np.zeros(nn * self.symmetry)
        self.coil_normals_all = np.zeros((nn * self.symmetry, 3))
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                phi0 = (2 * np.pi / self.nfp) * fp
                # get new locations by flipping the y and z components, then rotating by phi0
                self.grid_xyz_all[nn * q: nn * (q + 1), 0] = self.grid_xyz[:, 0] * np.cos(phi0) - self.grid_xyz[:, 1] * np.sin(phi0) * stell
                self.grid_xyz_all[nn * q: nn * (q + 1), 1] = self.grid_xyz[:, 0] * np.sin(phi0) + self.grid_xyz[:, 1] * np.cos(phi0) * stell
                self.grid_xyz_all[nn * q: nn * (q + 1), 2] = self.grid_xyz[:, 2] * stell
                # get new normal vectors by flipping the x component, then rotating by phi0
                self.coil_normals_all[nn * q: nn * (q + 1), 0] = self.coil_normals[:, 0] * np.cos(phi0) * stell - self.coil_normals[:, 1] * np.sin(phi0) 
                self.coil_normals_all[nn * q: nn * (q + 1), 1] = self.coil_normals[:, 0] * np.sin(phi0) * stell + self.coil_normals[:, 1] * np.cos(phi0) 
                self.coil_normals_all[nn * q: nn * (q + 1), 2] = self.coil_normals[:, 2]
                normals = self.coil_normals_all[q * nn: (q + 1) * nn, :]
                deltas = np.arctan2(normals[:, 0], normals[:, 2]) # + np.pi
                alphas = np.arctan2(-normals[:, 1] * np.cos(deltas), normals[:, 2])
                # alphas = -np.arcsin(normals[:, 1])
                print(alphas, deltas)
                self.alphas_total[nn * q: nn * (q + 1)] = alphas
                self.deltas_total[nn * q: nn * (q + 1)] = deltas
                q = q + 1
    
    def update_alphas_deltas(self):
        """
        Initialize the field-period and stellarator symmetrized normal vectors
        for all the PSC coils. This is required whenever the alpha_i
        and delta_i angles change.
        """
        nn = self.num_psc
        self.alphas_total = np.zeros(nn * self.symmetry)
        self.deltas_total = np.zeros(nn * self.symmetry)
        self.coil_normals_all = np.zeros((nn * self.symmetry, 3))
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                phi0 = (2 * np.pi / self.nfp) * fp
                # get new normal vectors by flipping the x component, then rotating by phi0
                self.coil_normals_all[nn * q: nn * (q + 1), 0] = self.coil_normals[:, 0] * np.cos(phi0) * stell - self.coil_normals[:, 1] * np.sin(phi0) 
                self.coil_normals_all[nn * q: nn * (q + 1), 1] = self.coil_normals[:, 0] * np.sin(phi0) * stell + self.coil_normals[:, 1] * np.cos(phi0) 
                self.coil_normals_all[nn * q: nn * (q + 1), 2] = self.coil_normals[:, 2]
                normals = self.coil_normals_all[q * nn: (q + 1) * nn, :]
                deltas = np.arctan2(normals[:, 0], normals[:, 2]) # + np.pi
                alphas = -np.arctan2(normals[:, 1] * np.cos(deltas), normals[:, 2])
                # alphas = -np.arcsin(normals[:, 1])
                self.alphas_total[nn * q: nn * (q + 1)] = alphas
                self.deltas_total[nn * q: nn * (q + 1)] = deltas
                q = q + 1
                
    def update_rotation_matrices(self):
        """
        When the PSC angles change during optimization, update all the
        rotation matrices and their derivatives. 
        """
        alphas = self.alphas 
        deltas = self.deltas
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
        rotation_matrix = rotation_matrix  # .reshape(len(alphas), -1)
        self.rotation_matrix = rotation_matrix
        self.rotation_matrixT = rotation_matrix.T
        # rotation_matrix = rotation_matrix.reshape(len(alphas), -1)
        # self.rotation_matrix_full = np.vstack((rotation_matrix, rotation_matrix))
        # self.rotation_matrixT_full = np.hstack((rotation_matrix.T, rotation_matrix.T))
        dRi_dalphak = np.zeros((len(alphas), 3, 3))    
        dRi_ddeltak = np.zeros((len(alphas), 3, 3))
        # dRi_dkappak = np.zeros((2 * len(alphas), 3, 3))
        for i in range(len(alphas)):
            dRi_dalphak[i, :, :] = np.array(
                [[0.0, 
                  np.sin(deltas[i]) * np.cos(alphas[i]),
                  -np.sin(deltas[i]) * np.sin(alphas[i])],
                  [0.0, -np.sin(alphas[i]), -np.cos(alphas[i])],
                  [0.0, np.cos(deltas[i]) * np.cos(alphas[i]),
                   -np.cos(deltas[i]) * np.sin(alphas[i])]]
            )
            dRi_ddeltak[i, :, :] = np.array(
                [[-np.sin(deltas[i]), 
                  np.cos(deltas[i]) * np.sin(alphas[i]),
                  np.cos(deltas[i]) * np.cos(alphas[i])],
                  [0.0, 0.0, 0.0],
                  [-np.cos(deltas[i]), 
                   -np.sin(deltas[i]) * np.sin(alphas[i]),
                   -np.sin(deltas[i]) * np.cos(alphas[i])]]
            )
            # dRi_dkappak[2 * i, :, :] = dRi_dalphak[i, :, :]
            # dRi_dkappak[2 * i + 1, :, :] = dRi_ddeltak[i, :, :]
        self.dRi_dalphak = dRi_dalphak  # .reshape(self.num_psc, -1)
        self.dRi_ddeltak = dRi_ddeltak  # .reshape(self.num_psc, -1)
        # self.dRi_dkappak = dRi_dkappak  # .reshape(self.num_psc * 2, -1)