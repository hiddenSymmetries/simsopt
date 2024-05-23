from pathlib import Path
import warnings

import numpy as np
from pyevtk.hl import pointsToVTK

from . import Surface
import simsoptpp as sopp
from simsopt.field import BiotSavart
import time
from scipy.linalg import inv

__all__ = ['WPgrid']


class WPgrid:
    r"""
    ``WPgrid`` is a class for setting up the grid, normal vectors,
    plasma surface, and other objects needed to perform WP
    optimization for stellarators. The class
    takes as input either (1) two toroidal surfaces specified as 
    SurfaceRZFourier objects, and initializes a set of points 
    (in Cartesian coordinates) between these surfaces or (2) a manual
    list of xyz locations for the coils, for which the user needs to make
    sure that coils are non-intersecting even when they are arbitrarily
    rotated around. 
    """

    def __init__(self):
        self.mu0 = 4 * np.pi * 1e-7
        self.fac = 1e-7
        self.BdotN2_list = []
        # Define a set of quadrature points and weights for the N point
        # Gaussian quadrature rule
        num_quad = 20
        (quad_points_phi, 
          self.quad_weights_phi) = np.polynomial.legendre.leggauss(num_quad)
        self.quad_points_phi = quad_points_phi * np.pi + np.pi

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

        # Initialize uniform grid
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        self.dx = (x_max - x_min) / (Nx - 1)
        self.dy = (y_max - y_min) / (Ny - 1)
        self.dz = 2 * z_max / (Nz - 1)
        Nmin = min(self.dx, min(self.dy, self.dz))
        
        # This is not a guarantee that coils will not touch but inductance
        # matrix blows up if they do so it is easy to tell when they do
        self.R = min(Nmin / 2.0, self.poff / 4.0)
        self.a = self.R / 10.0  # Hard-coded aspect ratio of 100 right now
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
        # This is probably not robust for every stellarator but seems to work
        # reasonably well for the Landreman/Paul QA/QH in the code. 
        if self.nfp > 1:
            inds = []
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        phi = np.arctan2(Y[i, j, k], X[i, j, k])
                        if self.nfp == 4:
                            phi2 = np.arctan2(self.R, X[i, j, k])
                        elif self.nfp == 2:
                            phi2 = np.arctan2(self.R, self.plasma_boundary.get_rc(0, 0))
                        # Add a little factor to avoid phi = pi / n_p degrees 
                        # exactly, which can intersect with a symmetrized
                        # coil if not careful 
                        # print(phi2)
                        # exit()
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
        coils_TF,
        inner_toroidal_surface: Surface, 
        outer_toroidal_surface: Surface,
        **kwargs,
    ):
        """
        Function to initialize a SIMSOPT WPgrid from a 
        volume defined by two toroidal surfaces. These must be specified
        directly. Often a good choice is made by extending the plasma 
        boundary by its normal vectors.

        Args
        ----------
        plasma_boundary: Surface class object 
            Representing the plasma boundary surface. Gets converted
            into SurfaceRZFourier object for ease of use.
        coils_TF: List of SIMSOPT Coil class objects
            A list of Coil class objects representing all of the toroidal field
            coils (including those obtained by applying the discrete 
            symmetries) needed to generate the induced currents in a WP 
            array. 
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
            out_dir: string
                File name to specify an output directory for the plots and
                optimization results generated using this class object. 
            Nx: int
                Number of points in x to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. 
            Ny: int
                Number of points in y to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. 
            Nz: int
                Number of points in z to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. 
            Bn_plasma: 2D numpy array, shape (ntheta_quadpoints, nphi_quadpoints)
                Magnetic field (only plasma) at the plasma
                boundary. Typically this will be the optimized plasma
                magnetic field from finite bootstrap current. 
                Set to zero if not specified by the user.
            ppp: int
                Order of the CurvePlanarFourier object used to represent and
                plot a WP coil in real space.
            interpolated_field: bool
                Flag to indicate if the class should initialize an 
                InterpolatedField object from the TF coils, rather than a 
                true BiotSavart object. Speeds up lots of the calculations
                but comes with some accuracy reduction.
            random_initialization: bool
                Flag to indicate if the angles of each WP should be 
                randomly initialized or not. If not, they are initialized so
                that their normal vectors align with the local B field 
                generated by the toroidal fields coils, generating essentially
                a maximum induced current in each WP.
        Returns
        -------
        wp_grid: An initialized WPgrid class object.

        """
        from simsopt.field import InterpolatedField, BiotSavart
        from simsopt.util import calculate_on_axis_B
                
        wp_grid = cls() 
        wp_grid.out_dir = kwargs.pop("out_dir", '')
        
        # Get all the TF coils data
        wp_grid.I_TF = np.array([coil.current.get_value() for coil in coils_TF])
        wp_grid.dl_TF = np.array([coil.curve.gammadash() for coil in coils_TF])
        wp_grid.gamma_TF = np.array([coil.curve.gamma() for coil in coils_TF])

        # Get all the geometric data from the plasma boundary
        wp_grid.plasma_boundary = plasma_boundary.to_RZFourier()
        wp_grid.nfp = plasma_boundary.nfp
        wp_grid.stellsym = plasma_boundary.stellsym
        wp_grid.symmetry = plasma_boundary.nfp * (plasma_boundary.stellsym + 1)
        if wp_grid.stellsym:
            wp_grid.stell_list = [1, -1]
        else:
            wp_grid.stell_list = [1]
        wp_grid.nphi = len(wp_grid.plasma_boundary.quadpoints_phi)
        wp_grid.ntheta = len(wp_grid.plasma_boundary.quadpoints_theta)
        wp_grid.plasma_unitnormals = wp_grid.plasma_boundary.unitnormal().reshape(-1, 3)
        wp_grid.plasma_points = wp_grid.plasma_boundary.gamma().reshape(-1, 3)
        Ngrid = wp_grid.nphi * wp_grid.ntheta
        Nnorms = np.ravel(np.sqrt(np.sum(wp_grid.plasma_boundary.normal() ** 2, axis=-1)))
        wp_grid.grid_normalization = np.sqrt(Nnorms / Ngrid)
        Bn_plasma = kwargs.pop("Bn_plasma", 
                               np.zeros((wp_grid.nphi, wp_grid.ntheta))
        )
        if not np.allclose(Bn_plasma.shape, (wp_grid.nphi, wp_grid.ntheta)): 
            raise ValueError('Plasma magnetic field surface data is incorrect shape.')
        wp_grid.Bn_plasma = Bn_plasma
        
        # Get geometric data for initializing the WPs
        N = 200  # Number of integration points for integrals over WP coils
        wp_grid.phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
        wp_grid.dphi = wp_grid.phi[1] - wp_grid.phi[0]
        Nx = kwargs.pop("Nx", 10)
        Ny = kwargs.pop("Ny", Nx)
        Nz = kwargs.pop("Nz", Nx)
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError('Nx, Ny, and Nz should be positive integers')
        wp_grid.Nx = Nx
        wp_grid.Ny = Ny
        wp_grid.Nz = Nz
        wp_grid.poff = kwargs.pop("poff", 100)
        wp_grid.inner_toroidal_surface = inner_toroidal_surface.to_RZFourier()
        wp_grid.outer_toroidal_surface = outer_toroidal_surface.to_RZFourier()    
        warnings.warn(
            'Plasma boundary and inner and outer toroidal surfaces should '
            'all have the same "range" parameter in order for a WP'
            ' array to be correctly initialized.'
        )        
        
        t1 = time.time()
        # Use the geometric info to initialize a grid of WPs
        contig = np.ascontiguousarray
        normal_inner = inner_toroidal_surface.unitnormal().reshape(-1, 3)   
        normal_outer = outer_toroidal_surface.unitnormal().reshape(-1, 3)   
        wp_grid._setup_uniform_grid()
        # Have the uniform grid, now need to loop through and eliminate cells.
        wp_grid.grid_xyz = sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
            contig(normal_inner), 
            contig(normal_outer), 
            contig(wp_grid.xyz_uniform), 
            contig(wp_grid.xyz_inner), 
            contig(wp_grid.xyz_outer)
        )
        inds = np.ravel(np.logical_not(np.all(wp_grid.grid_xyz == 0.0, axis=-1)))
        wp_grid.grid_xyz = np.array(wp_grid.grid_xyz[inds, :], dtype=float)
        wp_grid.num_wp = wp_grid.grid_xyz.shape[0]
        wp_grid.rho = np.linspace(0, wp_grid.R, N, endpoint=False)
        wp_grid.drho = wp_grid.rho[1] - wp_grid.rho[0]
        # Initialize 2D (rho, phi) mesh for integrals over the WP "face".
        Rho, Phi = np.meshgrid(wp_grid.rho, wp_grid.phi, indexing='ij')
        wp_grid.Rho = np.ravel(Rho)
        wp_grid.Phi = np.ravel(Phi)
        
        # Order of each WP coil when they are initialized as 
        # CurvePlanarFourier objects. For unit tests, needs > 400 for 
        # convergence.
        wp_grid.ppp = kwargs.pop("ppp", 100)
        
        # Setup the B field from the TF coils
        # Many big calls to B_TF.B() so can make an interpolated object
        # if accuracy is not paramount
        interpolated_field = kwargs.pop("interpolated_field", False)
        B_TF = BiotSavart(coils_TF)
        if interpolated_field:
            n = 40  # tried 20 here and then there are small errors in f_B 
            degree = 3
            R = wp_grid.R
            gamma_outer = outer_toroidal_surface.gamma().reshape(-1, 3)
            rs = np.linalg.norm(gamma_outer[:, :2], axis=-1)
            zs = gamma_outer[:, 2]
            rrange = (0, np.max(rs) + R, n)  # need to also cover the plasma
            if wp_grid.nfp > 1:
                phirange = (0, 2*np.pi/wp_grid.nfp, n*2)
            else:
                phirange = (0, 2 * np.pi, n * 2)
            if wp_grid.stellsym:
                zrange = (0, np.max(zs) + R, n // 2)
            else:
                zrange = (np.min(zs) - R, np.max(zs) + R, n // 2)
            B_TF = InterpolatedField(
                B_TF, degree, rrange, phirange, zrange, 
                True, nfp=wp_grid.nfp, stellsym=wp_grid.stellsym
            )
        B_TF.set_points(wp_grid.grid_xyz)
        B_axis = calculate_on_axis_B(B_TF, wp_grid.plasma_boundary, print_out=False)
        # Normalization of the ||A*Linv*psi - b||^2 objective 
        # representing Bnormal errors on the plasma surface
        wp_grid.normalization = B_axis ** 2 * wp_grid.plasma_boundary.area()
        wp_grid.fac2_norm = wp_grid.fac ** 2 / wp_grid.normalization
        wp_grid.B_TF = B_TF

        # Random or B_TF aligned initialization of the coil orientations
        random_initialization = kwargs.pop("random_initialization", False)
        if random_initialization:
            # Randomly initialize the coil orientations
            wp_grid.alphas = (np.random.rand(wp_grid.num_wp) - 0.5) * 2 * np.pi
            wp_grid.deltas = (np.random.rand(wp_grid.num_wp) - 0.5) * 2 * np.pi
            wp_grid.coil_normals = np.array(
                [np.cos(wp_grid.alphas) * np.sin(wp_grid.deltas),
                  -np.sin(wp_grid.alphas),
                  np.cos(wp_grid.alphas) * np.cos(wp_grid.deltas)]
            ).T
            # deal with -0 terms in the normals, which screw up the arctan2 calculations
            wp_grid.coil_normals[
                np.logical_and(np.isclose(wp_grid.coil_normals, 0.0), 
                               np.copysign(1.0, wp_grid.coil_normals) < 0)
                ] *= -1.0
        else:
            # determine the alphas and deltas from these normal vectors
            B = B_TF.B()
            wp_grid.coil_normals = (B.T / np.sqrt(np.sum(B ** 2, axis=-1))).T
            # deal with -0 terms in the normals, which screw up the arctan2 calculations
            wp_grid.coil_normals[
                np.logical_and(np.isclose(wp_grid.coil_normals, 0.0), 
                               np.copysign(1.0, wp_grid.coil_normals) < 0)
                ] *= -1.0
            wp_grid.deltas = np.arctan2(wp_grid.coil_normals[:, 0], 
                                         wp_grid.coil_normals[:, 2])
            wp_grid.deltas[abs(wp_grid.deltas) == np.pi] = 0.0
            wp_grid.alphas = -np.arctan2(
                wp_grid.coil_normals[:, 1] * np.cos(wp_grid.deltas), 
                wp_grid.coil_normals[:, 2]
            )
            wp_grid.alphas[abs(wp_grid.alphas) == np.pi] = 0.0
            
        # Generate all the locations of the WP coils obtained by applying
        # discrete symmetries (stellarator and field-period symmetries)
        wp_grid.setup_full_grid()
        t2 = time.time()
        # print('Initialize grid time = ', t2 - t1)
        
        # Initialize curve objects corresponding to each WP coil for 
        # plotting in 3D
        # Initialize all of the fields, currents, inductances, for all the WPs
        wp_grid.I = np.zeros(len(wp_grid.alphas))
        wp_grid.setup_orientations(wp_grid.alphas, wp_grid.deltas)
        
        # Initialize CurvePlanarFourier objects for the WPs, mostly for
        # plotting purposes
        wp_grid.setup_curves()
        wp_grid.plot_curves()
        
        # Set up vector b appearing in objective ||A*Linv*psi - b||^2
        # since it only needs to be initialized once and then stays fixed.
        wp_grid.b_vector()
        
        # Initialize kappas = [alphas, deltas] which is the array of
        # optimization variables used in this work. 
        kappa_I = np.ravel(np.array([wp_grid.alphas, wp_grid.deltas, wp_grid.I]))
        wp_grid.kappa_I = kappa_I
        return wp_grid
    
    @classmethod
    def geo_setup_manual(
        cls, 
        points,
        R,
        **kwargs,
    ):
        """
        Function to manually initialize a SIMSOPT WPgrid by explicitly 
        passing: a list of points representing the center of each WP coil, 
        the major and minor radius of each coil (assumed all the same), 
        and optionally the initial orientations of all the coils. This 
        initialization generates a generic TF field if coils_TF is not passed.

        Args
        ----------
        points: 2D numpy array, shape (num_wp, 3)
            A set of points specifying the center of every WP coil in the 
            array.
        R: double
            Major radius of every WP coil.
        kwargs: The following are valid keyword arguments.
            out_dir: string
                File name to specify an output directory for the plots and
                optimization results generated using this class object. 
            Nx: int
                Number of points in x to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. 
            Ny: int
                Number of points in y to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. 
            Nz: int
                Number of points in z to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. 
            Bn_plasma: 2D numpy array, shape (ntheta_quadpoints, nphi_quadpoints)
                Magnetic field (only plasma) at the plasma
                boundary. Typically this will be the optimized plasma
                magnetic field from finite bootstrap current. 
                Set to zero if not specified by the user.
            ppp: int
                Order of the CurvePlanarFourier object used to represent and
                plot a WP coil in real space.
            interpolated_field: bool
                Flag to indicate if the class should initialize an 
                InterpolatedField object from the TF coils, rather than a 
                true BiotSavart object. Speeds up lots of the calculations
                but comes with some accuracy reduction.
            coils_TF: List of SIMSOPT Coil class objects
                A list of Coil class objects representing all of the toroidal field
                coils (including those obtained by applying the discrete 
                symmetries) needed to generate the induced currents in a WP 
                array. 
            a: double
                Minor radius of every WP coil. Defaults to R / 100. 
            plasma_boundary: Surface class object 
                Representing the plasma boundary surface. Gets converted
                into SurfaceRZFourier object for ease of use. Defaults
                to using a low resolution QA stellarator. 
        Returns
        -------
        wp_grid: An initialized WPgrid class object.

        """
        from simsopt.util.permanent_magnet_helper_functions import initialize_coils, calculate_on_axis_B
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries
        from simsopt.field import InterpolatedField
        
        wp_grid = cls()
        # Initialize geometric information of the WP array
        wp_grid.grid_xyz = np.array(points, dtype=float)
        wp_grid.R = R
        wp_grid.a = kwargs.pop("a", R / 100.0)
        wp_grid.out_dir = kwargs.pop("out_dir", '')
        wp_grid.num_wp = wp_grid.grid_xyz.shape[0]
        wp_grid.alphas = kwargs.pop("alphas", 
                                     (np.random.rand(
                                         wp_grid.num_wp) - 0.5) * 2 * np.pi
        )
        wp_grid.deltas = kwargs.pop("deltas", 
                                     (np.random.rand(
                                         wp_grid.num_wp) - 0.5) * 2 * np.pi
        )
        N = 20  # Must be larger for some unit tests to converge
        wp_grid.phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
        wp_grid.dphi = wp_grid.phi[1] - wp_grid.phi[0]
        wp_grid.rho = np.linspace(0, R, N, endpoint=False)
        wp_grid.drho = wp_grid.rho[1] - wp_grid.rho[0]
        # Initialize 2D (rho, phi) mesh for integrals over the WP "face".
        Rho, Phi = np.meshgrid(wp_grid.rho, wp_grid.phi, indexing='ij')
        wp_grid.Rho = np.ravel(Rho)
        wp_grid.Phi = np.ravel(Phi)
    
        # initialize a default plasma boundary
        input_name = 'input.LandremanPaul2021_QA_lowres'
        TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        ndefault = 4
        default_surf = SurfaceRZFourier.from_vmec_input(
            surface_filename, range='full torus', nphi=ndefault, ntheta=ndefault
        )
        default_surf.nfp = 1
        default_surf.stellsym = False
        
        # Initialize all the plasma boundary information
        wp_grid.plasma_boundary = kwargs.pop("plasma_boundary", default_surf)
        wp_grid.nfp = wp_grid.plasma_boundary.nfp
        wp_grid.stellsym = wp_grid.plasma_boundary.stellsym
        wp_grid.nphi = len(wp_grid.plasma_boundary.quadpoints_phi)
        wp_grid.ntheta = len(wp_grid.plasma_boundary.quadpoints_theta)
        wp_grid.plasma_unitnormals = wp_grid.plasma_boundary.unitnormal().reshape(-1, 3)
        wp_grid.plasma_points = wp_grid.plasma_boundary.gamma().reshape(-1, 3)
        wp_grid.symmetry = wp_grid.plasma_boundary.nfp * (wp_grid.plasma_boundary.stellsym + 1)
        if wp_grid.stellsym:
            wp_grid.stell_list = [1, -1]
        else:
            wp_grid.stell_list = [1]
        Ngrid = wp_grid.nphi * wp_grid.ntheta
        Nnorms = np.ravel(np.sqrt(np.sum(wp_grid.plasma_boundary.normal() ** 2, axis=-1)))
        wp_grid.grid_normalization = np.sqrt(Nnorms / Ngrid)
        Bn_plasma = kwargs.pop("Bn_plasma", 
                               np.zeros((wp_grid.nphi, wp_grid.ntheta))
        )
        if not np.allclose(Bn_plasma.shape, (wp_grid.nphi, wp_grid.ntheta)): 
            raise ValueError('Plasma magnetic field surface data is incorrect shape.')
        wp_grid.Bn_plasma = Bn_plasma
        
        # generate planar TF coils
        ncoils = 2
        R0 = 1.0
        R1 = 0.65
        order = 4
        # qa scaled to 0.1 T on-axis magnetic field strength?
        total_current = 1e6
        base_curves = create_equally_spaced_curves(
            ncoils, wp_grid.plasma_boundary.nfp, 
            stellsym=wp_grid.plasma_boundary.stellsym, 
            R0=R0, R1=R1, order=order, numquadpoints=32
        )
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils)]
        I_TF_total = total_current
        total_current = Current(total_current)
        total_current.fix_all()
        default_coils = coils_via_symmetries(
            base_curves, base_currents, 
            wp_grid.plasma_boundary.nfp, 
            wp_grid.plasma_boundary.stellsym
        )
        # fix all the coil shapes so only the currents are optimized
        for i in range(ncoils):
            base_curves[i].fix_all()
        coils_TF = kwargs.pop("coils_TF", default_coils)
        
        # Get all the TF coils data
        wp_grid.I_TF = np.array([coil.current.get_value() for coil in coils_TF])
        wp_grid.dl_TF = np.array([coil.curve.gammadash() for coil in coils_TF])
        wp_grid.gamma_TF = np.array([coil.curve.gamma() for coil in coils_TF])
        
        # Setup the B field from the TF coils
        # Many big calls to B_TF.B() so can make an interpolated object
        # if accuracy is not paramount
        interpolated_field = kwargs.pop("interpolated_field", False)
        B_TF = BiotSavart(coils_TF)
        if interpolated_field:
            # Need interpolated region big enough to cover all the coils and the plasma
            n = 20
            degree = 2
            rs = np.linalg.norm(wp_grid.grid_xyz[:, :2] + R ** 2, axis=-1)
            zs = wp_grid.grid_xyz[:, 2]
            rrange = (0, np.max(rs) + R, n)  # need to also cover the plasma
            if wp_grid.nfp > 1:
                phirange = (0, 2*np.pi/wp_grid.nfp, n*2)
            else:
                phirange = (0, 2 * np.pi, n * 2)
            if wp_grid.stellsym:
                zrange = (0, np.max(zs) + R, n // 2)
            else:
                zrange = (np.min(zs) - R, np.max(zs) + R, n // 2)
            wp_grid.B_TF = InterpolatedField(
                B_TF, degree, rrange, phirange, zrange, 
                True, nfp=wp_grid.nfp, stellsym=wp_grid.stellsym
            )
        B_TF.set_points(wp_grid.grid_xyz)
        B_axis = calculate_on_axis_B(B_TF, wp_grid.plasma_boundary, print_out=False)
        # Normalization of the ||A*Linv*psi - b||^2 objective 
        # representing Bnormal errors on the plasma surface
        wp_grid.normalization = B_axis ** 2 * wp_grid.plasma_boundary.area()
        wp_grid.fac2_norm = wp_grid.fac ** 2 / wp_grid.normalization
        wp_grid.B_TF = B_TF
        
        # Order of the coils. For unit tests, needs > 400
        wp_grid.ppp = kwargs.pop("ppp", 1000)

        wp_grid.coil_normals = np.array(
            [np.cos(wp_grid.alphas) * np.sin(wp_grid.deltas),
              -np.sin(wp_grid.alphas),
              np.cos(wp_grid.alphas) * np.cos(wp_grid.deltas)]
        ).T
        # deal with -0 terms in the normals, which screw up the arctan2 calculations
        wp_grid.coil_normals[
            np.logical_and(np.isclose(wp_grid.coil_normals, 0.0), 
                           np.copysign(1.0, wp_grid.coil_normals) < 0)
            ] *= -1.0
        
        # Initialize curve objects corresponding to each WP coil for 
        # plotting in 3D
        wp_grid.setup_full_grid()
        wp_grid.I = np.zeros(len(wp_grid.alphas))
        wp_grid.setup_orientations(wp_grid.alphas, wp_grid.deltas)
        
        # Initialize CurvePlanarFourier objects for the WPs, mostly for
        # plotting purposes
        wp_grid.setup_curves()
        wp_grid.plot_curves()
        
        # Set up vector b appearing in objective ||A*Linv*psi - b||^2
        # since it only needs to be initialized once and then stays fixed.
        wp_grid.b_vector()
        
        # Initialize kappas = [alphas, deltas] which is the array of
        # optimization variables used in this work. 
        kappa_I = np.ravel(np.array([wp_grid.alphas, wp_grid.deltas, wp_grid.I]))
        wp_grid.kappa_I = kappa_I
        return wp_grid
    
    def setup_currents_and_fields(self):
        """ 
        Calculate the currents and fields from all the WPs. Note that L 
        will be well-conditioned unless you accidentally initialized
        the WP coils touching/intersecting, in which case it will be
        VERY ill-conditioned! 
        """
        
        contig = np.ascontiguousarray
        # A_matrix has shape (num_plasma_points, num_coils)
        # B_WP = np.zeros((self.nphi * self.ntheta, 3))
        A_matrix = np.zeros((self.nphi * self.ntheta, self.num_wp))
        # A_matrix2 = np.zeros((self.nphi * self.ntheta, self.num_wp))
        # Need to rotate and flip it
        nn = self.num_wp
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                # A_matrix2 += sopp.A_matrix_direct(
                #     contig(self.grid_xyz_all[q * nn: (q + 1) * nn, :]),
                #     contig(self.plasma_points),
                #     contig(self.alphas_total[q * nn: (q + 1) * nn]),
                #     contig(self.deltas_total[q * nn: (q + 1) * nn]),
                #     contig(self.plasma_unitnormals),
                #     contig(self.phi),
                #     self.R,
                # ) * self.dphi # missing factor of mu0
                # A_matrix2 += self.python_A_matrix(
                #     self.grid_xyz_all[q * nn: (q + 1) * nn, :], 
                #     self.alphas_total[q * nn: (q + 1) * nn],
                #     self.deltas_total[q * nn: (q + 1) * nn])
                # t1 = time.time()
                A_matrix += sopp.A_matrix_simd(
                    contig(self.grid_xyz_all[q * nn: (q + 1) * nn, :]),
                    contig(self.plasma_points),
                    contig(self.alphas_total[q * nn: (q + 1) * nn]),
                    contig(self.deltas_total[q * nn: (q + 1) * nn]),
                    contig(self.plasma_unitnormals),
                    self.R,
                )  # missing factor of fac, factor of 2 from the analytic expression
                # t2 = time.time()
                # print('Time1 = ',t2 - t1)
                # t1 = time.time()
                # A_matrix += 2 * sopp.A_matrix_simd(
                #     contig(xyz),
                #     contig(self.plasma_points),
                #     contig(self.alphas_total[q * nn: (q + 1) * nn]),
                #     contig(self.deltas_total[q * nn: (q + 1) * nn]),
                #     contig(self.plasma_unitnormals),
                #     self.R,
                # )  # 
                # t2 = time.time()
                # print('Time2 = ',t2 - t1)
                # print(A_matrix2, A_matrix)
                # assert np.allclose(A_matrix2, A_matrix)
                q = q + 1
        self.A_matrix = 2.0 * A_matrix
        self.Bn_WP = (self.A_matrix @ self.I * 1e6).reshape(-1) # missing factor of fac
    
    def setup_curves(self):
        """ 
        Convert the (alphas, deltas) angles into the actual circular coils
        that they represent. Also generates the symmetrized coils.
        """
        from . import CurvePlanarFourier
        from simsopt.field import apply_symmetries_to_curves

        order = 1
        ncoils = self.num_wp
        curves = [CurvePlanarFourier(order*self.ppp, order, nfp=1, stellsym=False) for i in range(ncoils)]
        for ic in range(ncoils):
            alpha2 = self.alphas[ic] / 2.0
            delta2 = self.deltas[ic] / 2.0
            calpha2 = np.cos(alpha2)
            salpha2 = np.sin(alpha2)
            cdelta2 = np.cos(delta2)
            sdelta2 = np.sin(delta2)
            dofs = np.zeros(10)
            dofs[0] = self.R
            # dofs[1] = 0.0
            # dofs[2] = 0.0
            # Conversion from Euler angles in 3-2-1 body sequence to 
            # quaternions: 
            # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
            dofs[3] = calpha2 * cdelta2
            dofs[4] = salpha2 * cdelta2
            dofs[5] = calpha2 * sdelta2
            dofs[6] = -salpha2 * sdelta2
            # Now specify the center 
            dofs[7:10] = self.grid_xyz[ic, :]
            curves[ic].set_dofs(dofs)
        self.curves = curves
        self.all_curves = apply_symmetries_to_curves(curves, self.nfp, self.stellsym)

    def plot_curves(self, filename=''):
        """
        Plots the unique WP coil curves in real space and plots the normal 
        vectors, fluxes, currents, and TF fields at the center of coils. 
        Repeats the plotting for all of the WP coil curves after discrete
        symmetries are applied. If the TF field is not type InterpolatedField,
        less quantities are plotted because of computational time. 

        Parameters
        ----------
        filename : string
            File name extension for some of the saved files.

        """
        from pyevtk.hl import pointsToVTK
        from . import curves_to_vtk
        from simsopt.field import InterpolatedField, Current, coils_via_symmetries
        
        curves_to_vtk(self.curves, self.out_dir + "wp_curves", close=True, scalar_data=self.I)
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
        q = 0
        self.I_all = np.zeros(self.num_wp * self.symmetry)
        for fp in range(self.nfp):
            for stell in self.stell_list:
                self.I_all[q * self.num_wp:(q + 1) * self.num_wp] = self.I * stell
                q = q + 1
        curves_to_vtk(self.all_curves, self.out_dir + filename + "all_wp_curves", close=True, scalar_data=self.I_all)
        contig = np.ascontiguousarray
        q = 0
        nn = self.num_wp
        self.coil_normals_all = np.zeros((nn * self.symmetry, 3))
        for fp in range(self.nfp):
            for stell in self.stell_list:
                phi0 = (2 * np.pi / self.nfp) * fp
                # get new normal vectors by flipping the x component, then rotating by phi0
                self.coil_normals_all[nn * q: nn * (q + 1), 0] = self.coil_normals[:, 0] * np.cos(phi0) * stell - self.coil_normals[:, 1] * np.sin(phi0) 
                self.coil_normals_all[nn * q: nn * (q + 1), 1] = self.coil_normals[:, 0] * np.sin(phi0) * stell + self.coil_normals[:, 1] * np.cos(phi0) 
                self.coil_normals_all[nn * q: nn * (q + 1), 2] = self.coil_normals[:, 2]
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
        """
        Initialize the vector b appearing in the ||A*Linv*psi - b||^2
        objective term representing Bnormal errors on the plasma surface.

        """
        Bn_plasma = self.Bn_plasma.reshape(-1)
        self.B_TF.set_points(self.plasma_points)
        Bn_TF = np.sum(
            self.B_TF.B().reshape(-1, 3) * self.plasma_unitnormals, axis=-1
        )
        self.b_opt = (Bn_TF + Bn_plasma) / self.fac  # scale by fac 
        
    def least_squares(self, kappa_I, verbose=False):
        """
        Evaluate the 0.5 * ||A*Linv*psi - b||^2
        objective term representing Bnormal errors on the plasma surface.
        
        Parameters
        ----------
        kappas : 1D numpy array, shape 2N
            Array of [alpha_1, ..., alpha_N, delta_1, ..., delta_N] that
            represents the coil orientation degrees of freedom being used
            for optimization.
        verbose : bool
            Flag to print out objective values and current array of kappas.
            
        Returns
        -------
            BdotN2: double
                The value of 0.5 * ||A*Linv*psi - b||^2 evaluated with the
                array of kappas.

        """
        t1 = time.time()
        ind3 = len(kappa_I) // 3
        ind3_2 = 2 * ind3
        self.I = kappa_I[ind3_2:]
        self.setup_orientations(kappa_I[:ind3], kappa_I[ind3:ind3_2])
        Ax_b = (self.Bn_WP + self.b_opt) * self.grid_normalization 
        BdotN2 = 0.5 * Ax_b.T @ Ax_b * self.fac2_norm
        self.BdotN2_list.append(BdotN2)
        t2 = time.time()
        print('obj time = ', t2 - t1)
        return BdotN2
    
    def python_A_matrix(self, points, alphas, deltas):
        import scipy.special as ss
        
        num_coils = alphas.shape[0]
        num_plasma_points = self.plasma_points.shape[0]
        A = np.zeros((num_plasma_points, num_coils))
        R = self.R
        R2 = self.R * self.R
        deltas_ptr = deltas
        alphas_ptr = alphas
        points_ptr = points 
        plasma_points_ptr = self.plasma_points
        plasma_normal_ptr = self.plasma_unitnormals
        
        for j in range(num_coils):
            cdj = np.cos(deltas_ptr[j])
            caj = np.cos(alphas_ptr[j])
            sdj = np.sin(deltas_ptr[j])
            saj = np.sin(alphas_ptr[j])
            xj = points_ptr[j, 0]
            yj = points_ptr[j, 1]
            zj = points_ptr[j, 2]
            for i in range(num_plasma_points):
                xp = plasma_points_ptr[i, 0]
                yp = plasma_points_ptr[i, 1]
                zp = plasma_points_ptr[i, 2]
                nx = plasma_normal_ptr[i, 0]
                ny = plasma_normal_ptr[i, 1]
                nz = plasma_normal_ptr[i, 2]
                x0 = (xp - xj)
                y0 = (yp - yj)
                z0 = (zp - zj)
                nxx = cdj
                nxy = sdj * saj
                nxz = sdj * caj
                nyx = 0.0
                nyy = caj
                nyz = -saj
                nzx = -sdj
                nzy = cdj * saj
                nzz = cdj * caj
                x = x0 * nxx + y0 * nyx + z0 * nzx
                y = x0 * nxy + y0 * nyy + z0 * nzy
                z = x0 * nxz + y0 * nyz + z0 * nzz
                rho2 = x * x + y * y
                r2 = rho2 + z * z
                rho = np.sqrt(rho2)
                R2_r2 = R2 + r2
                gamma2 = R2_r2 - 2.0 * R * rho
                beta2 = R2_r2 + 2.0 * R * rho
                beta = np.sqrt(beta2)
                k2 = 1.0 - gamma2 / beta2
                beta_gamma2 = beta * gamma2
                k = np.sqrt(k2)
                ellipe = ss.ellipe(k ** 2)
                ellipk = ss.ellipk(k ** 2)
                Eplus = R2_r2 * ellipe - gamma2 * ellipk
                Eminus = (R2 - r2) * ellipe + gamma2 * ellipk
                Bx = x * z * Eplus / (rho2 * beta_gamma2)
                By = y * z * Eplus / (rho2 * beta_gamma2)
                Bz = Eminus / beta_gamma2
                # Need to rotate the vector
                Bx_rot = Bx * nxx + By * nxy + Bz * nxz
                By_rot = Bx * nyx + By * nyy + Bz * nyz
                Bz_rot = Bx * nzx + By * nzy + Bz * nzz
                A[i, j] = Bx_rot * nx + By_rot * ny + Bz_rot * nz
        return 2.0 * A
    
    def least_squares_jacobian(self, kappa_I, verbose=False):
        """
        Compute Jacobian of the ||A*Linv*psi - b||^2
        objective term representing Bnormal errors on the plasma surface,
        for using BFGS in scipy.minimize.
        
        Parameters
        ----------
        kappas : 1D numpy array, shape 2N
            Array of [alpha_1, ..., alpha_N, delta_1, ..., delta_N] that
            represents the coil orientation degrees of freedom being used
            for optimization.
        Returns
        -------
            jac: 1D numpy array, shape 2N
                The gradient values of 0.5 * ||A*Linv*psi - b||^2 evaluated 
                with the array of kappas.

        """
        t1 = time.time()
        ind3 = len(kappa_I) // 3
        ind3_2 = 2 * ind3
        self.I = kappa_I[ind3_2:]
        self.setup_orientations(kappa_I[:ind3], kappa_I[ind3:ind3_2])
        Ax_b = (self.Bn_WP + self.b_opt) * self.grid_normalization
        t1_grad = time.time()
        grad_alpha_delta = self.A_deriv() * np.hstack((self.I, self.I))
        t2_grad = time.time()
        print('Aderiv time = ', t2_grad - t1_grad)
        grad_I = self.A_matrix
        # Should be shape (num_plasma_points, 3 * num_wp)
        # grad_kappa = np.hstack((grad_alpha1, grad_delta1))
        grad = np.hstack((grad_alpha_delta, grad_I)) * 1.0e6
        grad = self.grid_normalization[:, None] * grad
        jac = Ax_b.T @ grad
        t2 = time.time()
        print('jac time = ', t2 - t1)
        return jac * self.fac2_norm
    
    def A_deriv(self):
        """
        Should return gradient of the A matrix evaluated at each point on the 
        plasma boundary. This is a wrapper function for a faster c++ call.
        
        Returns
        -------
            dS_dkappa: 3D numpy array, shape (2 * num_wp, num_plasma_points, 3) 
                The gradient of the A matrix evaluated on all the plasma 
                points, with respect to the WP angles alpha_i and delta_i. 
        """
        contig = np.ascontiguousarray
        nn = self.num_wp
        dA_dkappa = np.zeros((len(self.plasma_points), 2 * self.num_wp))
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                dA = sopp.dA_dkappa_simd(
                    contig(self.grid_xyz_all[q * nn: (q + 1) * nn, :]),
                    contig(self.plasma_points),
                    contig(self.alphas_total[q * nn: (q + 1) * nn]),
                    contig(self.deltas_total[q * nn: (q + 1) * nn]),
                    contig(self.plasma_unitnormals),
                    contig(self.quad_points_phi),
                    contig(self.quad_weights_phi),
                    self.R,
                )
                dA_dkappa[:, :self.num_wp] += dA[:, :self.num_wp] * (-1) ** fp
                dA_dkappa[:, self.num_wp:] += dA[:, self.num_wp:] * (-1) ** fp * stell
                q = q + 1
        return dA_dkappa * np.pi # rescale by pi for gauss-leg quadrature
    
    def setup_orientations(self, alphas, deltas):
        """
        Each time that optimization changes the WP angles, need to update
        all the fields, currents, and fluxes to be consistent with the 
        new angles. This function does this updating to get the entire class
        object the new angle values. 
        
        Args
        ----------
        alphas : 1D numpy array, shape (num_wps)
            Rotation angles of every WP around the x-axis.
        deltas : 1D numpy array, shape (num_wps)
            Rotation angles of every WP around the y-axis.
        Returns
        -------
            grad_Psi: 3D numpy array, shape (2 * num_wp, num_wp) 
                The gradient of the Psi vector with respect to the WP angles
                alpha_i and delta_i. 
        """
        
        # Need to check is alphas and deltas are in [-pi, pi] and remap if not
        self.alphas = alphas
        self.deltas = deltas
        
        # contig = np.ascontiguousarray
        # self.coil_normals = np.array(
        #     [np.cos(alphas) * np.sin(deltas),
        #       -np.sin(alphas),
        #       np.cos(alphas) * np.cos(deltas)]
        # ).T
        # deal with -0 terms in the normals, which screw up the arctan2 calculations
        # self.coil_normals[
        #     np.logical_and(np.isclose(self.coil_normals, 0.0), 
        #                    np.copysign(1.0, self.coil_normals) < 0)
        #     ] *= -1.0
                
        # Apply discrete symmetries to the alphas and deltas and coordinates
        self.update_alphas_deltas()
        
        # Calculate the WP B fields
        # t1 = time.time()
        self.setup_currents_and_fields()
        # t2 = time.time()
        # print('Setup fields time = ', t2 - t1)
        
    def setup_full_grid(self):
        """
        Initialize the field-period and stellarator symmetrized grid locations
        and normal vectors for all the WP coils. Note that when alpha_i
        and delta_i angles change, coil_i location does not change, only
        its normal vector changes. 
        """
        nn = self.num_wp
        self.grid_xyz_all = np.zeros((nn * self.symmetry, 3))
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                phi0 = (2 * np.pi / self.nfp) * fp
                # get new locations by flipping the y and z components, then rotating by phi0
                self.grid_xyz_all[nn * q: nn * (q + 1), 0] = self.grid_xyz[:, 0] * np.cos(phi0) - self.grid_xyz[:, 1] * np.sin(phi0) * stell
                self.grid_xyz_all[nn * q: nn * (q + 1), 1] = self.grid_xyz[:, 0] * np.sin(phi0) + self.grid_xyz[:, 1] * np.cos(phi0) * stell
                self.grid_xyz_all[nn * q: nn * (q + 1), 2] = self.grid_xyz[:, 2] * stell
                q += 1
                
        self.update_alphas_deltas()
    
    def update_alphas_deltas(self):
        """
        Initialize the field-period and stellarator symmetrized normal vectors
        for all the WP coils. This is required whenever the alpha_i
        and delta_i angles change.
        """
        nn = self.num_wp
        self.alphas_total = np.zeros(nn * self.symmetry)
        self.deltas_total = np.zeros(nn * self.symmetry)
        # self.coil_normals_all = np.zeros((nn * self.symmetry, 3))
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                phi0 = (2 * np.pi / self.nfp) * fp
                # get new normal vectors by flipping the x component, then rotating by phi0
                # self.coil_normals_all[nn * q: nn * (q + 1), 0] = self.coil_normals[:, 0] * np.cos(phi0) * stell - self.coil_normals[:, 1] * np.sin(phi0) 
                # self.coil_normals_all[nn * q: nn * (q + 1), 1] = self.coil_normals[:, 0] * np.sin(phi0) * stell + self.coil_normals[:, 1] * np.cos(phi0) 
                # self.coil_normals_all[nn * q: nn * (q + 1), 2] = self.coil_normals[:, 2]
                # normals = self.coil_normals_all[q * nn: (q + 1) * nn, :]
                # normals = self.coil_normals
                # get new deltas by flipping sign of deltas, then rotation alpha by phi0
                # deltas = np.arctan2(normals[:, 0], normals[:, 2]) # + np.pi
                # deltas[abs(deltas) == np.pi] = 0.0
                # if fp == 0 and stell == 1:
                #     shift_deltas = self.deltas - deltas  # +- pi
                # deltas += shift_deltas
                # alphas = -np.arctan2(normals[:, 1] * np.cos(deltas), normals[:, 2])
                # alphas = -np.arcsin(normals[:, 1])
                # alphas[abs(alphas) == np.pi] = 0.0
                # if fp == 0 and stell == 1:
                #     shift_alphas = self.alphas - alphas  # +- pi
                # alphas += shift_alphas
                alphas = self.alphas * (-1) ** fp
                deltas = self.deltas * stell * (-1) ** fp
                self.alphas_total[nn * q: nn * (q + 1)] = alphas
                self.deltas_total[nn * q: nn * (q + 1)] = deltas
                # print(fp, stell)
                # print(self.alphas, self.deltas)
                # print(alphas, deltas)
                q = q + 1
        # print(self.alphas_total)
                