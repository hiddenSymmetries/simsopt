from pathlib import Path
import warnings

import numpy as np
from pyevtk.hl import pointsToVTK

from . import Surface
import simsoptpp as sopp
import time
from scipy.linalg import inv, pinv, pinvh

__all__ = ['PSCgrid']
contig = np.ascontiguousarray

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
    """

    def __init__(self):
        self.mu0 = 4 * np.pi * 1e-7
        self.fac = 1e-7
        self.BdotN2_list = []
        self.Bn_list = []
        # Define a set of quadrature points and weights for the N point
        # Gaussian quadrature rule
        num_quad = 20  # 8 required for unit tests to pass and 20 required to do it well
        (quad_points_phi, 
         quad_weights) = np.polynomial.legendre.leggauss(num_quad)
        self.quad_points_phi = contig(quad_points_phi * np.pi + np.pi)
        self.quad_weights = contig(quad_weights)
        self.quad_points_rho = contig(quad_points_phi * 0.5 + 0.5)

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
        if self.plasma_boundary.nfp == 2:
            # self.R = Nmin / 2.5 
            self.R = Nmin / 2.5  # 2.5
        elif self.plasma_boundary.nfp == 3:
            self.R = min(Nmin / 2.0, self.poff / 1.75)
        else:
            self.R = Nmin / 2.45
            # self.R = min(Nmin / 2.3, self.poff / 2.0) # self.poff / 2.5
            
        # Note that aspect ratio of 0.1 has ~twice as large currents
        # as aspect ratio of 0.01
        self.a = self.R / 10.0  # Hard-coded aspect ratio
        
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
                            phi2 = np.arctan2(self.R / 1.4, X[i, j, k])
                        elif self.nfp == 3:
                            phi2 = np.arctan2(Y[i, j, k] + self.R / 2.0, X[i, j, k] - self.R / 2.0) - phi
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
        Function to initialize a SIMSOPT PSCgrid from a 
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
            symmetries) needed to generate the induced currents in a PSC 
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
                plot a PSC coil in real space.
            interpolated_field: bool
                Flag to indicate if the class should initialize an 
                InterpolatedField object from the TF coils, rather than a 
                true BiotSavart object. Speeds up lots of the calculations
                but comes with some accuracy reduction.
            random_initialization: bool
                Flag to indicate if the angles of each PSC should be 
                randomly initialized or not. If not, they are initialized so
                that their normal vectors align with the local B field 
                generated by the toroidal fields coils, generating essentially
                a maximum induced current in each PSC.
        Returns
        -------
        psc_grid: An initialized PSCgrid class object.

        """
        from simsopt.field import InterpolatedField, BiotSavart, Current, coils_via_symmetries
        from simsopt.util import calculate_on_axis_B
        from . import curves_to_vtk
                
        t1 = time.time()
        psc_grid = cls() 
        psc_grid.out_dir = kwargs.pop("out_dir", '')
        
        # Get all the TF coils data
        psc_grid.I_TF = np.array([coil.current.get_value() for coil in coils_TF])
        psc_grid.dl_TF = np.array([coil.curve.gammadash() for coil in coils_TF])
        psc_grid.gamma_TF = np.array([coil.curve.gamma() for coil in coils_TF])
        TF_curves = np.array([coil.curve for coil in coils_TF])
        curves_to_vtk(TF_curves, psc_grid.out_dir + "coils_TF", close=True, scalar_data=psc_grid.I_TF)

        # Get all the geometric data from the plasma boundary
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
        psc_grid.plasma_unitnormals = contig(psc_grid.plasma_boundary.unitnormal().reshape(-1, 3))
        psc_grid.plasma_points = contig(psc_grid.plasma_boundary.gamma().reshape(-1, 3))
        Ngrid = psc_grid.nphi * psc_grid.ntheta
        Nnorms = np.ravel(np.sqrt(np.sum(psc_grid.plasma_boundary.normal() ** 2, axis=-1)))
        psc_grid.grid_normalization = np.sqrt(Nnorms / Ngrid)
        Bn_plasma = kwargs.pop("Bn_plasma", 
                               np.zeros((psc_grid.nphi, psc_grid.ntheta))
        )
        if not np.allclose(Bn_plasma.shape, (psc_grid.nphi, psc_grid.ntheta)): 
            raise ValueError('Plasma magnetic field surface data is incorrect shape.')
        psc_grid.Bn_plasma = Bn_plasma
        
        # Get geometric data for initializing the PSCs
        Nx = kwargs.pop("Nx", 10)
        Ny = kwargs.pop("Ny", Nx)
        Nz = kwargs.pop("Nz", Nx)
        psc_grid.poff = kwargs.pop("poff", 100)
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError('Nx, Ny, and Nz should be positive integers')
        psc_grid.Nx = Nx
        psc_grid.Ny = Ny
        psc_grid.Nz = Nz
        psc_grid.inner_toroidal_surface = inner_toroidal_surface.to_RZFourier()
        psc_grid.outer_toroidal_surface = outer_toroidal_surface.to_RZFourier()    
        warnings.warn(
            'Plasma boundary and inner and outer toroidal surfaces should '
            'all have the same "range" parameter in order for a PSC'
            ' array to be correctly initialized.'
        )        
        
        # Use the geometric info to initialize a grid of PSCs
        normal_inner = inner_toroidal_surface.unitnormal().reshape(-1, 3)   
        normal_outer = outer_toroidal_surface.unitnormal().reshape(-1, 3)   
        psc_grid._setup_uniform_grid()
        # Have the uniform grid, now need to loop through and eliminate cells.
        psc_grid.grid_xyz = sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
            contig(normal_inner), 
            contig(normal_outer), 
            contig(psc_grid.xyz_uniform), 
            contig(psc_grid.xyz_inner), 
            contig(psc_grid.xyz_outer)
        )
        inds = np.ravel(np.logical_not(np.all(psc_grid.grid_xyz == 0.0, axis=-1)))
        psc_grid.grid_xyz = np.array(psc_grid.grid_xyz[inds, :], dtype=float)
        
        # Check if the grid intersects a symmetry plane -- oops!
        phi0 = 2 * np.pi / psc_grid.nfp * np.arange(psc_grid.nfp)
        phi_grid = np.arctan2(psc_grid.grid_xyz[:, 1], psc_grid.grid_xyz[:, 0])
        phi_dev = np.arctan2(psc_grid.R, np.sqrt(psc_grid.grid_xyz[:, 0] ** 2  + psc_grid.grid_xyz[:, 1] ** 2))
        inds = []
        eps = 1e-3
        remove_inds = []
        # for i in range(psc_grid.nfp):
        #     conflicts = np.ravel(np.where(np.abs(phi_grid - phi0[i]) < phi_dev))
        #     if len(conflicts) > 0:
        #         inds.append(conflicts[0])
        # if len(inds) > 0:
        #     print('bad indices = ', inds)
        #     raise ValueError('The PSC coils are initialized such that they may intersect with '
        #                      'a discrete symmetry plane, preventing the proper symmetrization '
        #                      'of the coils under stellarator and field-period symmetries. '
        #                      'Please reinitialize the coils.')
        for i in range(psc_grid.grid_xyz.shape[0]):
            for j in range(i + 1, psc_grid.grid_xyz.shape[0]):
                dij = np.sqrt(np.sum((psc_grid.grid_xyz[i, :] - psc_grid.grid_xyz[j, :]) ** 2))
                conflict_bool = (dij < (2.0 + eps) * psc_grid.R)
                if conflict_bool:
                    print('bad indices = ', i, j, dij)
                    raise ValueError('There is a PSC coil initialized such that it is within a diameter'
                                     'of another PSC coil. Please reinitialize the coils.')
            eps = 0.1
            for j in range(len(coils_TF)):
                for k in range(psc_grid.gamma_TF.shape[1]):
                    dij = np.sqrt(np.sum((psc_grid.grid_xyz[i, :] - psc_grid.gamma_TF[j, k, :]) ** 2))
                    conflict_bool = (dij < (1.0 + eps) * psc_grid.R)
                    if conflict_bool:
                        print('bad indices = ', i, j, dij)
                        warnings.warn(
                            'There is a PSC coil initialized such that it is within a radius'
                            'of a TF coil. Deleting these PSCs now.')
                        remove_inds.append(i)
                        break
        
        final_inds = np.setdiff1d(np.arange(psc_grid.grid_xyz.shape[0]), remove_inds)
        psc_grid.grid_xyz = psc_grid.grid_xyz[final_inds, :]
        psc_grid.num_psc = psc_grid.grid_xyz.shape[0]
        psc_grid.quad_points_rho = psc_grid.quad_points_rho * psc_grid.R
        
        # Order of each PSC coil when they are initialized as 
        # CurvePlanarFourier objects. For unit tests, needs > 400 for 
        # convergence.
        psc_grid.ppp = kwargs.pop("ppp", 100)
        
        # Setup the B field from the TF coils
        # Many big calls to B_TF.B() so can make an interpolated object
        # if accuracy is not paramount
        interpolated_field = kwargs.pop("interpolated_field", False)
        B_TF = BiotSavart(coils_TF)
        if interpolated_field:
            n = 40  # tried 20 here and then there are small errors in f_B 
            degree = 3
            R = psc_grid.R
            gamma_outer = outer_toroidal_surface.gamma().reshape(-1, 3)
            rs = np.linalg.norm(gamma_outer[:, :2], axis=-1)
            zs = gamma_outer[:, 2]
            rrange = (0, np.max(rs) + R, n)  # need to also cover the plasma
            if psc_grid.nfp > 1:
                phirange = (0, 2*np.pi/psc_grid.nfp, n*2)
            else:
                phirange = (0, 2 * np.pi, n * 2)
            if psc_grid.stellsym:
                zrange = (0, np.max(zs) + R, n // 2)
            else:
                zrange = (np.min(zs) - R, np.max(zs) + R, n // 2)
            B_TF = InterpolatedField(
                B_TF, degree, rrange, phirange, zrange, 
                True, nfp=psc_grid.nfp, stellsym=psc_grid.stellsym
            )
        # B_TF.set_points(psc_grid.grid_xyz)
        B_axis = calculate_on_axis_B(B_TF, psc_grid.plasma_boundary, print_out=False)
        # Normalization of the ||A*Linv*psi - b||^2 objective 
        # representing Bnormal errors on the plasma surface
        psc_grid.normalization = B_axis ** 2 * psc_grid.plasma_boundary.area()
        psc_grid.Baxis_A = B_axis * psc_grid.plasma_boundary.area()
        psc_grid.fac2_norm = psc_grid.fac ** 2 / psc_grid.normalization
        psc_grid.fac2_norm2 = psc_grid.fac2_norm * 0.5
        # psc_grid.fac2_norm_grid = psc_grid.fac2_norm * psc_grid.grid_normalization
        # psc_grid.fac2_norm_grid2 = psc_grid.fac2_norm * psc_grid.grid_normalization ** 2
        # psc_grid.fac2_norm2_grid2 = psc_grid.fac2_norm * psc_grid.grid_normalization ** 2 * 0.5
        psc_grid.B_TF = B_TF

        # Random or B_TF aligned initialization of the coil orientations
        initialization = kwargs.pop("initialization", "random")
        if initialization == "random":
            # Randomly initialize the coil orientations
            psc_grid.coil_normals = np.array(
                [(np.random.rand(psc_grid.num_psc) - 0.5),
                  (np.random.rand(psc_grid.num_psc) - 0.5),
                  (np.random.rand(psc_grid.num_psc) - 0.5)]
            ).T
            psc_grid.coil_normals = psc_grid.coil_normals / np.sqrt(np.sum(psc_grid.coil_normals ** 2, axis=-1))
            print('N = ', psc_grid.coil_normals)
            # psc_grid.alphas = (np.random.rand(psc_grid.num_psc) - 0.5) * 2 * np.pi
            # psc_grid.deltas = (np.random.rand(psc_grid.num_psc) - 0.5) * np.pi
            # psc_grid.alphas = (np.random.rand(psc_grid.num_psc) - 0.5) * np.pi
            # psc_grid.deltas = (np.random.rand(psc_grid.num_psc) - 0.5) * 2 * np.pi
            # psc_grid.coil_normals = np.array(
            #     [np.cos(psc_grid.alphas) * np.sin(psc_grid.deltas),
            #       -np.sin(psc_grid.alphas),
            #       np.cos(psc_grid.alphas) * np.cos(psc_grid.deltas)]
            # ).T
        elif initialization == "plasma":
            # determine the alphas and deltas from the plasma normal vectors
            psc_grid.coil_normals = np.zeros(psc_grid.grid_xyz.shape)
            for i in range(psc_grid.num_psc):
                point = psc_grid.grid_xyz[i, :]
                dists = np.sum((point - psc_grid.plasma_points) ** 2, axis=-1)
                min_ind = np.argmin(dists)
                psc_grid.coil_normals[i, :] = psc_grid.plasma_unitnormals[min_ind, :]
                
            # psc_grid.deltas = np.arctan2(psc_grid.coil_normals[:, 0], 
            #                              psc_grid.coil_normals[:, 2])
            # psc_grid.alphas = -np.arcsin(psc_grid.coil_normals[:, 1])
        elif initialization == "TF":
            # determine the alphas and deltas from these normal vectors
            B = B_TF.B()
            psc_grid.coil_normals = (B.T / np.sqrt(np.sum(B ** 2, axis=-1))).T
            # psc_grid.deltas = np.arctan2(psc_grid.coil_normals[:, 0], 
            #                              psc_grid.coil_normals[:, 2])
            # psc_grid.alphas =  -np.arcsin(psc_grid.coil_normals[:, 1]) 
        else:  # default is to initialize to zeros -- seems to work better in optimization anyways
            psc_grid.coil_normals = np.array(
                [np.zeros(psc_grid.num_psc),np.zeros(psc_grid.num_psc), np.ones(psc_grid.num_psc) ]
            ).T
        psc_grid.alphas = np.arctan2(
                -psc_grid.coil_normals[:, 1], 
                np.sqrt(psc_grid.coil_normals[:, 0] ** 2 + psc_grid.coil_normals[:, 2] ** 2))
        psc_grid.deltas = np.arcsin(psc_grid.coil_normals[:, 0] / \
                                    np.sqrt(psc_grid.coil_normals[:, 0] ** 2 + psc_grid.coil_normals[:, 2] ** 2))
            
        # deal with -0 terms in the normals, which screw up the arctan2 calculations
        # psc_grid.coil_normals[
        #     np.logical_and(np.isclose(psc_grid.coil_normals, 0.0), 
        #                    np.copysign(1.0, psc_grid.coil_normals) < 0)
        #     ] *= -1.0
            
        # Generate all the locations of the PSC coils obtained by applying
        # discrete symmetries (stellarator and field-period symmetries)
        psc_grid.setup_full_grid()
        psc_grid.update_alphas_deltas()
        
        # Initialize curve objects corresponding to each PSC coil for 
        # plotting in 3D
        # Initialize all of the fields, currents, inductances, for all the PSCs
        psc_grid.setup_orientations(psc_grid.alphas, psc_grid.deltas)
        psc_grid.update_psi()

        psc_grid.currents = []
        for i in range(psc_grid.num_psc):
            psc_grid.currents.append(Current(0.0))
        # psc_grid.currents = np.array(psc_grid.currents)
        psc_grid.all_currents = []
        for fp in range(psc_grid.nfp):
            for stell in psc_grid.stell_list:
                for i in range(psc_grid.num_psc):
                    psc_grid.all_currents.append(Current(0.0))
        
        psc_grid.setup_curves()
        psc_grid.setup_currents_and_fields()
        # for i in range(psc_grid.num_psc):
        #     psc_grid.currents[i].fix_all()
        #     names_i = psc_grid.curves[i].local_dof_names
        #     psc_grid.curves[i].fix(names_i[0])
        #     psc_grid.curves[i].fix(names_i[1])
        #     psc_grid.curves[i].fix(names_i[2])
            
        #     # Fix the center point for now
        #     psc_grid.curves[i].fix(names_i[7])
        #     psc_grid.curves[i].fix(names_i[8])
        #     psc_grid.curves[i].fix(names_i[9])
            
        psc_grid.psc_coils = coils_via_symmetries(
            psc_grid.curves, psc_grid.currents, nfp=psc_grid.nfp, stellsym=psc_grid.stellsym
        )
        psc_grid.B_PSC = BiotSavart(psc_grid.psc_coils)
        psc_grid.B_PSC.set_points(psc_grid.plasma_points)
        
        # Initialize CurvePlanarFourier objects for the PSCs, mostly for
        # plotting purposes
        # psc_grid.setup_curves()
        psc_grid.plot_curves()
        
        # Set up vector b appearing in objective ||A*Linv*psi - b||^2
        # since it only needs to be initialized once and then stays fixed.
        psc_grid.b_vector()
        
        # Initialize kappas = [alphas, deltas] which is the array of
        # optimization variables used in this work. 
        t2 = time.time()
        print('Geo setup time = ', t2 - t1)
        return psc_grid
    
    @classmethod
    def geo_setup_manual(
        cls, 
        points,
        R,
        **kwargs,
    ):
        """
        Function to manually initialize a SIMSOPT PSCgrid by explicitly 
        passing: a list of points representing the center of each PSC coil, 
        the major and minor radius of each coil (assumed all the same), 
        and optionally the initial orientations of all the coils. This 
        initialization generates a generic TF field if coils_TF is not passed.

        Args
        ----------
        points: 2D numpy array, shape (num_psc, 3)
            A set of points specifying the center of every PSC coil in the 
            array.
        R: double
            Major radius of every PSC coil.
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
                plot a PSC coil in real space.
            interpolated_field: bool
                Flag to indicate if the class should initialize an 
                InterpolatedField object from the TF coils, rather than a 
                true BiotSavart object. Speeds up lots of the calculations
                but comes with some accuracy reduction.
            coils_TF: List of SIMSOPT Coil class objects
                A list of Coil class objects representing all of the toroidal field
                coils (including those obtained by applying the discrete 
                symmetries) needed to generate the induced currents in a PSC 
                array. 
            a: double
                Minor radius of every PSC coil. Defaults to R / 100. 
            plasma_boundary: Surface class object 
                Representing the plasma boundary surface. Gets converted
                into SurfaceRZFourier object for ease of use. Defaults
                to using a low resolution QA stellarator. 
        Returns
        -------
        psc_grid: An initialized PSCgrid class object.

        """
        from simsopt.util.permanent_magnet_helper_functions import initialize_coils, calculate_on_axis_B
        from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
        from simsopt.field import InterpolatedField, Current, coils_via_symmetries, BiotSavart
        from . import curves_to_vtk
        
        t1 = time.time()
        psc_grid = cls()
        # Initialize geometric information of the PSC array
        psc_grid.grid_xyz = np.array(points, dtype=float)
        psc_grid.R = R
        psc_grid.a = kwargs.pop("a", R / 10.0)
        psc_grid.out_dir = kwargs.pop("out_dir", '')
    
        # initialize a default plasma boundary
        psc_grid.plasma_boundary = kwargs.pop("plasma_boundary", None)
        if psc_grid.plasma_boundary is None:
            input_name = 'input.LandremanPaul2021_QA_lowres'
            TEST_DIR = (Path(__file__).parent / ".." / ".." / ".." / "tests" / "test_files").resolve()
            surface_filename = TEST_DIR / input_name
            ndefault = 4
            default_surf = SurfaceRZFourier.from_vmec_input(
                surface_filename, range='full torus', nphi=ndefault, ntheta=ndefault
            )
            default_surf.nfp = 1
            default_surf.stellsym = False
            psc_grid.plasma_boundary = default_surf
        
        # Initialize all the plasma boundary information
        psc_grid.nfp = psc_grid.plasma_boundary.nfp
        psc_grid.stellsym = psc_grid.plasma_boundary.stellsym
        psc_grid.nphi = len(psc_grid.plasma_boundary.quadpoints_phi)
        psc_grid.ntheta = len(psc_grid.plasma_boundary.quadpoints_theta)
        psc_grid.plasma_unitnormals = contig(psc_grid.plasma_boundary.unitnormal().reshape(-1, 3))
        psc_grid.plasma_points = contig(psc_grid.plasma_boundary.gamma().reshape(-1, 3))
        psc_grid.symmetry = psc_grid.plasma_boundary.nfp * (psc_grid.plasma_boundary.stellsym + 1)
        if psc_grid.stellsym:
            psc_grid.stell_list = [1, -1]
        else:
            psc_grid.stell_list = [1]
        Ngrid = psc_grid.nphi * psc_grid.ntheta
        Nnorms = np.ravel(np.sqrt(np.sum(psc_grid.plasma_boundary.normal() ** 2, axis=-1)))
        psc_grid.grid_normalization = np.sqrt(Nnorms / Ngrid)
        Bn_plasma = kwargs.pop("Bn_plasma", 
                               np.zeros((psc_grid.nphi, psc_grid.ntheta))
        )
        if not np.allclose(Bn_plasma.shape, (psc_grid.nphi, psc_grid.ntheta)): 
            raise ValueError('Plasma magnetic field surface data is incorrect shape.')
        psc_grid.Bn_plasma = Bn_plasma
        
        # generate planar TF coils
        coils_TF = kwargs.pop("coils_TF", None)
        if coils_TF is None:
            ncoils = 4
            R0 = 1.4
            R1 = 0.9
            order = 4
            total_current = 1e6
            base_curves = create_equally_spaced_curves(
                ncoils, psc_grid.plasma_boundary.nfp, 
                stellsym=psc_grid.plasma_boundary.stellsym, 
                R0=R0, R1=R1, order=order, numquadpoints=256
            )
            base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils)]
            total_current = Current(total_current)
            total_current.fix_all()
            default_coils = coils_via_symmetries(
                base_curves, base_currents, 
                psc_grid.plasma_boundary.nfp, 
                psc_grid.plasma_boundary.stellsym
            )
            # fix all the coil shapes so only the currents are optimized
            for i in range(ncoils):
                base_curves[i].fix_all()
                base_currents[i].fix_all()
            coils_TF = default_coils
        
        # Get all the TF coils data
        psc_grid.I_TF = np.array([coil.current.get_value() for coil in coils_TF])
        psc_grid.dl_TF = np.array([coil.curve.gammadash() for coil in coils_TF])
        psc_grid.gamma_TF = np.array([coil.curve.gamma() for coil in coils_TF])
        
        # Check if the grid intersects a symmetry plane -- oops!
        phi0 = 2 * np.pi / psc_grid.nfp * np.arange(psc_grid.nfp)
        phi_grid = np.arctan2(psc_grid.grid_xyz[:, 1], psc_grid.grid_xyz[:, 0])
        phi_dev = np.arctan2(psc_grid.R, np.sqrt(psc_grid.grid_xyz[:, 0] ** 2  + psc_grid.grid_xyz[:, 1] ** 2))
        inds = []
        eps = 5e-2
        remove_inds = []
        # for i in range(psc_grid.nfp):
        #     conflicts = np.ravel(np.where(np.abs(phi_grid - phi0[i]) < phi_dev))
        #     if len(conflicts) > 0:
        #         inds.append(conflicts[0])
        # if len(inds) > 0:
        #     print('bad indices = ', inds)
        #     raise ValueError('The PSC coils are initialized such that they may intersect with '
        #                       'a discrete symmetry plane, preventing the proper symmetrization '
        #                       'of the coils under stellarator and field-period symmetries. '
        #                       'Please reinitialize the coils.')
        for i in range(psc_grid.grid_xyz.shape[0]):
            for j in range(i + 1, psc_grid.grid_xyz.shape[0]):
                dij = np.sqrt(np.sum((psc_grid.grid_xyz[i, :] - psc_grid.grid_xyz[j, :]) ** 2))
                conflict_bool = (dij < (2.0 + eps) * psc_grid.R)
                if conflict_bool:
                    print('bad indices = ', i, j, dij)
                    raise ValueError('There is a PSC coil initialized such that it is within a diameter'
                                      'of another PSC coil. Please reinitialize the coils.')
            eps = 0.1
            for j in range(len(coils_TF)):
                for k in range(psc_grid.gamma_TF.shape[1]):
                    dij = np.sqrt(np.sum((psc_grid.grid_xyz[i, :] - psc_grid.gamma_TF[j, k, :]) ** 2))
                    conflict_bool = (dij < (1.0 + eps) * psc_grid.R)
                    if conflict_bool:
                        print('bad indices = ', i, j, dij)
                        warnings.warn(
                            'There is a PSC coil initialized such that it is within a radius'
                            'of a TF coil. Deleting these PSCs now.')
                        remove_inds.append(i)
                        break
        
        final_inds = np.setdiff1d(np.arange(psc_grid.grid_xyz.shape[0]), remove_inds)
        psc_grid.grid_xyz = psc_grid.grid_xyz[final_inds, :]
        
        # Setup the B field from the TF coils
        # Many big calls to B_TF.B() so can make an interpolated object
        # if accuracy is not paramount
        interpolated_field = kwargs.pop("interpolated_field", False)
        B_TF = BiotSavart(coils_TF)
        if interpolated_field:
            # Need interpolated region big enough to cover all the coils and the plasma
            n = 20
            degree = 2
            rs = np.linalg.norm(psc_grid.grid_xyz[:, :2] + R ** 2, axis=-1)
            zs = psc_grid.grid_xyz[:, 2]
            rrange = (0, np.max(rs) + R, n)  # need to also cover the plasma
            if psc_grid.nfp > 1:
                phirange = (0, 2*np.pi/psc_grid.nfp, n*2)
            else:
                phirange = (0, 2 * np.pi, n * 2)
            if psc_grid.stellsym:
                zrange = (0, np.max(zs) + R, n // 2)
            else:
                zrange = (np.min(zs) - R, np.max(zs) + R, n // 2)
            psc_grid.B_TF = InterpolatedField(
                B_TF, degree, rrange, phirange, zrange, 
                True, nfp=psc_grid.nfp, stellsym=psc_grid.stellsym
            )
        # B_TF.set_points(psc_grid.grid_xyz)
        B_axis = calculate_on_axis_B(B_TF, psc_grid.plasma_boundary, print_out=False)
        
        # Normalization of the ||A*Linv*psi - b||^2 objective 
        # representing Bnormal errors on the plasma surface
        psc_grid.normalization = B_axis ** 2 * psc_grid.plasma_boundary.area()
        psc_grid.Baxis_A = B_axis * psc_grid.plasma_boundary.area()
        psc_grid.fac2_norm = psc_grid.fac ** 2 / psc_grid.normalization
        psc_grid.fac2_norm2 = psc_grid.fac2_norm * 0.5
        # psc_grid.fac2_norm_grid = psc_grid.fac2_norm * psc_grid.grid_normalization
        # psc_grid.fac2_norm_grid2 = psc_grid.fac2_norm * psc_grid.grid_normalization ** 2
        psc_grid.B_TF = B_TF
        
        # Order of the coils. For unit tests, needs > 400
        psc_grid.ppp = kwargs.pop("ppp", 100)
        
        psc_grid.num_psc = psc_grid.grid_xyz.shape[0]
        # psc_grid.coil_normals = np.array(
        #         [(np.random.rand(psc_grid.num_psc) - 0.5),
        #           (np.random.rand(psc_grid.num_psc) - 0.5),
        #           (np.random.rand(psc_grid.num_psc) - 0.5)]
        #     ).T
        # psc_grid.coil_normals = psc_grid.coil_normals / np.sqrt(np.sum(psc_grid.coil_normals ** 2, axis=-1))
        # kwargs.pop("coil_normals", psc_grid.coil_normals)
        psc_grid.alphas = kwargs.pop("alphas", 
                                     (np.random.rand(
                                         psc_grid.num_psc) - 0.5) * 2 * np.pi
        )
        psc_grid.deltas = kwargs.pop("deltas", 
                                     (np.random.rand(
                                         psc_grid.num_psc) - 0.5) * np.pi
        )
        ## Need to remove bad elements from alphas and deltas too!
        psc_grid.alphas = psc_grid.alphas[final_inds]
        psc_grid.deltas = psc_grid.deltas[final_inds]
        
        psc_grid.quad_points_rho = psc_grid.quad_points_rho * psc_grid.R

        psc_grid.coil_normals = np.array(
            [np.cos(psc_grid.alphas) * np.sin(psc_grid.deltas),
              -np.sin(psc_grid.alphas),
              np.cos(psc_grid.alphas) * np.cos(psc_grid.deltas)]
        ).T

        psc_grid.alphas = np.arctan2(
                -psc_grid.coil_normals[:, 1], 
                np.sqrt(psc_grid.coil_normals[:, 0] ** 2 + psc_grid.coil_normals[:, 2] ** 2))
        psc_grid.deltas = np.arcsin(psc_grid.coil_normals[:, 0] / \
                                    np.sqrt(psc_grid.coil_normals[:, 0] ** 2 + psc_grid.coil_normals[:, 2] ** 2))
        # Now redefine the alphas according to the normals
        # deal with -0 terms in the normals, which screw up the arctan2 calculations
        # psc_grid.coil_normals[
        #     np.logical_and(np.isclose(psc_grid.coil_normals, 0.0), 
        #                    np.copysign(1.0, psc_grid.coil_normals) < 0)
        #     ] *= -1.0
        
        # Initialize curve objects corresponding to each PSC coil for 
        # plotting in 3D
        psc_grid.setup_full_grid()
        psc_grid.update_alphas_deltas()
        psc_grid.setup_orientations(psc_grid.alphas, psc_grid.deltas)
        psc_grid.update_psi()
        psc_grid.currents = []
        for i in range(psc_grid.num_psc):
            psc_grid.currents.append(Current(0.0))
        # psc_grid.currents = np.array(psc_grid.currents)
        psc_grid.all_currents = []
        for fp in range(psc_grid.nfp):
            for stell in psc_grid.stell_list:
                for i in range(psc_grid.num_psc):
                    psc_grid.all_currents.append(Current(0.0))
        
        psc_grid.setup_curves()
        psc_grid.setup_currents_and_fields()
        # for i in range(psc_grid.num_psc):
        #     psc_grid.currents[i].fix_all()
        #     names_i = psc_grid.curves[i].local_dof_names
        #     psc_grid.curves[i].fix(names_i[0])
        #     psc_grid.curves[i].fix(names_i[1])
        #     psc_grid.curves[i].fix(names_i[2])
            
        #     # Fix the center point for now
        #     psc_grid.curves[i].fix(names_i[7])
        #     psc_grid.curves[i].fix(names_i[8])
        #     psc_grid.curves[i].fix(names_i[9])
        
        from simsopt.field import coils_via_symmetries, Current
        psc_grid.psc_coils = coils_via_symmetries(
            psc_grid.curves, 
            psc_grid.currents, 
            nfp=psc_grid.nfp, 
            stellsym=psc_grid.stellsym
        )
        psc_grid.B_PSC = BiotSavart(psc_grid.psc_coils)
        psc_grid.B_PSC.set_points(psc_grid.plasma_points)
        
        # Initialize CurvePlanarFourier objects for the PSCs
        # psc_grid.setup_curves()
        psc_grid.plot_curves()
        
        # Set up vector b appearing in objective ||A*Linv*psi - b||^2
        # since it only needs to be initialized once and then stays fixed.
        psc_grid.b_vector()
        
        # Initialize kappas = [alphas, deltas] which is the array of
        # optimization variables used in this work. 
        t2 = time.time()
        # print('Initialize grid time = ', t2 - t1)
        return psc_grid
    
    def setup_currents_and_fields(self):
        """ 
        Calculate the currents and fields from all the PSCs. Note that L 
        will be well-conditioned unless you accidentally initialized
        the PSC coils touching/intersecting, in which case it will be
        VERY ill-conditioned! 
        """
        from simsopt.field import BiotSavart, coils_via_symmetries, Current
        
        self.L_inv = np.linalg.inv(self.L)
        # if hasattr(self, 'I'):
        #     'do nothing'
        # else:
        self.I = -self.L_inv[:self.num_psc, :] @ self.psi_total / self.fac
        
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                for i in range(self.num_psc):
                    self.all_currents[i + q * self.num_psc].set('x0', self.I[i] * stell)
                q += 1
            # print(self.I)
        # self.setup_curves()
        # for i, current in enumerate(self.currents):
            # current.unfix_all()
            # current.set('x0', self.I[i])
            # current.fix_all()
        # self.psc_coils = coils_via_symmetries(
        #     self.curves, self.currents, nfp=self.nfp, stellsym=self.stellsym
        # )
        # self.B_PSC = BiotSavart(self.psc_coils)
        # self.B_PSC.set_points(self.plasma_points)
        self.setup_A_matrix()
        # print(self.I)
        self.Bn_PSC = self.A_matrix @ self.I
        # print(self.Bn_PSC, self.A_matrix)
        
    def setup_A_matrix(self):
        """
        """
        A_matrix = np.zeros((self.nphi * self.ntheta, self.num_psc))
        nn = self.num_psc
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                A_matrix += sopp.A_matrix_simd(
                    contig(self.grid_xyz_all[q * nn: (q + 1) * nn, :]),
                    self.plasma_points,
                    contig(self.alphas_total[q * nn: (q + 1) * nn]),
                    contig(self.deltas_total[q * nn: (q + 1) * nn]),
                    self.plasma_unitnormals,
                    self.R,
                ) # accounts for sign change of the currents
                q = q + 1
        # if hasattr(self, 'A_matrix'):
        #     'do nothing'
        # else:
        self.A_matrix = 2 * A_matrix  
    
    def setup_curves(self):
        """ 
        Convert the (alphas, deltas) angles into the actual circular coils
        that they represent. Also generates the symmetrized coils.
        """
        from . import PSCCurve, CurvePlanarFourier
        from simsopt.field import apply_symmetries_to_psc_curves

        order = 1
        ncoils = self.num_psc
        self.curves = [CurvePlanarFourier(order*self.ppp, order, nfp=1, stellsym=False) for i in range(ncoils)]
        for ic in range(ncoils):
            alpha2 = self.alphas[ic] / 2.0
            delta2 = self.deltas[ic] / 2.0
            calpha2 = np.cos(alpha2)
            salpha2 = np.sin(alpha2)
            cdelta2 = np.cos(delta2)
            sdelta2 = np.sin(delta2)
            dofs = np.zeros(10)
            dofs[0] = self.R
            # Conversion from Euler angles in 3-2-1 body sequence to 
            # quaternions: 
            # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
            dofs[3] = calpha2 * cdelta2
            dofs[4] = salpha2 * cdelta2
            dofs[5] = calpha2 * sdelta2
            dofs[6] = -salpha2 * sdelta2
            # Now specify the center 
            dofs[7:10] = self.grid_xyz[ic, :]
            
            for j in range(10):
                self.curves[ic].set('x' + str(j), dofs[j])
            # self.curves[ic].set_dofs(dofs)
            # names_i = self.curves[ic].local_dof_names
            # self.curves[ic].fix(names_i[0])
            # self.curves[ic].fix(names_i[0])
            # self.curves[ic].fix(names_i[0])
            
            # # Fix the center point for now
            # self.curves[ic].fix(names_i[4])
            # self.curves[ic].fix(names_i[4])
            # self.curves[ic].fix(names_i[4])
        self.all_curves = apply_symmetries_to_psc_curves(self.curves, self.nfp, self.stellsym)
        
    def update_curves(self):
        """ 
        Convert the (alphas, deltas) angles into the actual circular coils
        that they represent. Also generates the symmetrized coils.
        """
        ncoils = self.num_psc
        for ic in range(ncoils):
            alpha2 = self.alphas[ic] / 2.0
            delta2 = self.deltas[ic] / 2.0
            calpha2 = np.cos(alpha2)
            salpha2 = np.sin(alpha2)
            cdelta2 = np.cos(delta2)
            sdelta2 = np.sin(delta2)
            # print(ic, self.curves[ic].get_dofs())
            dofs = self.curves[ic].get_dofs()
            # dofs[0] = self.R
            dofs[3] = calpha2 * cdelta2
            dofs[4] = salpha2 * cdelta2
            dofs[5] = calpha2 * sdelta2
            dofs[6] = -salpha2 * sdelta2
            # self.curves[ic].unfix_all()
            # self.curves[ic].set_dofs(dofs)
            self.curves[ic].set('x3', dofs[3])
            self.curves[ic].set('x4', dofs[4])
            self.curves[ic].set('x5', dofs[5])
            self.curves[ic].set('x6', dofs[6])
            # names_i = self.curves[ic].local_dof_names
            # self.curves[ic].fix(names_i[0])
            # self.curves[ic].fix(names_i[1])
            # self.curves[ic].fix(names_i[2])
            
            # # Fix the center point for now
            # self.curves[ic].fix(names_i[7])
            # self.curves[ic].fix(names_i[8])
            # self.curves[ic].fix(names_i[9])
            # Now specify the center 
            # dofs[7:10] = self.grid_xyz[ic, :]
            # print(ic, self.curves[ic].get_dofs())
            # names_i = self.curves[ic].local_dof_names

    def plot_curves(self, filename=''):
        """
        Plots the unique PSC coil curves in real space and plots the normal 
        vectors, fluxes, currents, and TF fields at the center of coils. 
        Repeats the plotting for all of the PSC coil curves after discrete
        symmetries are applied. If the TF field is not type InterpolatedField,
        less quantities are plotted because of computational time. 

        Parameters
        ----------
        filename : string
            File name extension for some of the saved files.

        """
        # from pyevtk.hl import pointsToVTK
        from . import curves_to_vtk
        
        # curves_to_vtk(self.curves, self.out_dir + "psc_curves", close=True, scalar_data=self.I)
        # self.B_TF.set_points(self.grid_xyz)
        # B = self.B_TF.B()
        # pointsToVTK(self.out_dir + 'curve_centers', 
        #             contig(self.grid_xyz[:, 0]),
        #             contig(self.grid_xyz[:, 1]), 
        #             contig(self.grid_xyz[:, 2]),
        #             data={"n": (contig(self.coil_normals[:, 0]), 
        #                         contig(self.coil_normals[:, 1]),
        #                         contig(self.coil_normals[:, 2])),
        #                   "psi": contig(self.psi),
        #                   "I": contig(self.I),
        #                   "B_TF": (contig(B[:, 0]), 
        #                             contig(B[:, 1]),
        #                             contig(B[:, 2])),
        #                   },
        # )
        
        # Repeat for the whole torus
        self.I_all = -self.L_inv @ self.psi_total / self.fac
        self.I_all_with_sign_flips = np.zeros(self.num_psc * self.symmetry)
        # self.I_all = np.zeros(self.num_psc * self.symmetry)
        q = 0
        nn = self.num_psc
        for fp in range(self.nfp):
            for stell in self.stell_list:
                # self.I_all[q * nn: (q + 1) * nn] = self.I
                self.I_all_with_sign_flips[q * nn: (q + 1) * nn] = self.I * stell
                # print('here = ', q, self.I)
                q += 1
        curves_to_vtk(self.all_curves, self.out_dir + filename + "all_psc_curves", close=True, scalar_data=self.I_all_with_sign_flips)
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
                          "I": contig(self.I_all_with_sign_flips),
                          "B_TF": (contig(B[:, 0]), 
                                    contig(B[:, 1]),
                                    contig(B[:, 2])),
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
        self.b_opt = (Bn_TF + Bn_plasma) / self.fac
        
    def normalized_Bn(self):
        """
        """
        Bn_mag = np.sum(np.abs((self.Bn_PSC + self.b_opt) * self.grid_normalization ** 2))
        Bn_mag = Bn_mag * self.fac / self.Baxis_A
        self.Bn_list.append(Bn_mag)
        
    def least_squares(self, kappas, verbose=False):
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
        self.setup_orientations(kappas[:self.num_psc], kappas[self.num_psc:])
        # self.update_curves()
        self.update_psi()
        self.setup_currents_and_fields()
        Ax_b = (self.Bn_PSC + self.b_opt) * self.grid_normalization
        # print('bs Squared flux from psc array = ', 
        #       (self.b_opt * self.grid_normalization).T @ (self.b_opt * self.grid_normalization) * self.fac2_norm2)
        # print('PSC Squared flux from psc array = ', 
        #       (self.Bn_PSC * self.grid_normalization).T @ (self.Bn_PSC * self.grid_normalization) * self.fac2_norm2)
        BdotN2 = (Ax_b.T @ Ax_b) * self.fac2_norm2
        self.BdotN2_list.append(BdotN2)
        self.normalized_Bn()
        # Bn_check = np.sum(self.B_PSC.B().reshape(-1, 3) * self.plasma_unitnormals, axis=-1)
        # print('Bn1, Bn2 = ', Bn_check, self.Bn_PSC * self.fac)
        return BdotN2
    
    def least_squares_jacobian(self, kappas, verbose=False):
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
        #### Probably this makes it so we are really using the Jacobian from
        #### the previous iteration, but seems to work pretty well and speeds
        #### up the optimization a fair bit
        # self.setup_orientations(kappas[:self.num_psc], kappas[self.num_psc:])
        # self.update_psi()
        # self.setup_currents_and_fields()
        # Ax_b = (self.Bn_PSC + self.b_opt) * self.grid_normalization
        # grad1 = self.A_deriv() * np.hstack((self.I, self.I))
        # Linv = self.L_inv[:self.num_psc, :self.num_psc]
        # grad3 = -self.A_matrix @ (np.hstack((Linv, Linv)) * self.psi_deriv())
        # return (Ax_b.T @ (self.grid_normalization[:, None] * (grad1 + grad3))) * self.fac2_norm
    
        self.setup_orientations(kappas[:self.num_psc], kappas[self.num_psc:])
        self.update_psi()
        self.setup_currents_and_fields()
        Ax_b = (self.Bn_PSC + self.b_opt) * self.grid_normalization
        A_deriv = self.A_deriv() 
        grad_alpha1 = A_deriv[:, :self.num_psc] * self.I
        grad_delta1 = A_deriv[:, self.num_psc:] * self.I
        grad_kappa1 = self.grid_normalization[:, None] * np.hstack((grad_alpha1, grad_delta1)) 
        self.psi_deriv()
        self.psi_deriv_full()
        Linv = self.L_inv[:self.num_psc, :self.num_psc]
        I_deriv2 = -Linv * self.dpsi[:self.num_psc]
        I_deriv3 = -Linv * self.dpsi[self.num_psc:]
        grad_alpha3 = self.A_matrix @ I_deriv2
        grad_delta3 = self.A_matrix @ I_deriv3
        grad_kappa3 = self.grid_normalization[:, None] * np.hstack((grad_alpha3, grad_delta3))
        return (Ax_b.T @ (grad_kappa1 + grad_kappa3)) * self.fac2_norm 
    
    def A_deriv(self):
        """
        Should return gradient of the A matrix evaluated at each point on the 
        plasma boundary. This is a wrapper function for a faster c++ call.
        
        Returns
        -------
            A_deriv: 2D numpy array, shape (num_plasma_points, 2 * num_wp) 
                The gradient of the A matrix evaluated on all the plasma 
                points, with respect to the WP angles alpha_i and delta_i. 
        """
        nn = self.num_psc
        dA_dkappa = np.zeros((len(self.plasma_points), 2 * self.num_psc))
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                dA = sopp.dA_dkappa_simd(
                    contig(self.grid_xyz_all[q * nn: (q + 1) * nn, :]),
                    self.plasma_points,
                    contig(self.alphas_total[q * nn: (q + 1) * nn]),
                    contig(self.deltas_total[q * nn: (q + 1) * nn]),
                    self.plasma_unitnormals,
                    self.quad_points_phi,
                    self.quad_weights,
                    self.R,
                )
                dA_dkappa[:, :nn] += dA[:, :nn] * self.aaprime_aa[q * nn: (q + 1) * nn] + dA[:, nn:] * self.ddprime_aa[q * nn: (q + 1) * nn]
                dA_dkappa[:, nn:] += dA[:, nn:] * self.ddprime_dd[q * nn: (q + 1) * nn] + dA[:, :nn] * self.aaprime_dd[q * nn: (q + 1) * nn]
                q = q + 1
        return dA_dkappa * np.pi # rescale by pi for gauss-leg quadrature
    
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
        t1 = time.time()
        L_deriv = sopp.L_deriv_simd(
            self.grid_xyz_all_normalized, 
            self.alphas_total,
            self.deltas_total,
            self.quad_points_phi,
            self.quad_weights,
            self.num_psc,
            self.stellsym,  # These  last few params are defunct in the function
            self.nfp
        )  
        nsym = self.symmetry
        L_deriv_copy = np.zeros(L_deriv.shape)
        
        # Extra work required to impose the symmetries correctly
        nn = self.num_psc
        ncoils_sym = nn * nsym
        if nsym > 1:
            q = 0
            for fp in range(self.nfp):
                for stell in self.stell_list:
                    L_deriv_copy[:nn, :, :] += L_deriv[q * nn:(q + 1) * nn, :, :] * self.aaprime_aa[q * nn: (q + 1) * nn, None, None] + L_deriv[ncoils_sym + q * nn:ncoils_sym + (q + 1) * nn, :, :] * self.ddprime_aa[q * nn: (q + 1) * nn, None, None]
                    L_deriv_copy[ncoils_sym:ncoils_sym + nn, :, :] += L_deriv[q * nn:(q + 1) * nn, :, :] * self.aaprime_dd[q * nn: (q + 1) * nn, None, None] + L_deriv[ncoils_sym + q * nn:ncoils_sym + (q + 1) * nn, :, :] * self.ddprime_dd[q * nn: (q + 1) * nn, None, None]
                    q = q + 1
        else:
            L_deriv_copy = L_deriv
            
        # symmetrize it
        L_deriv = (L_deriv_copy + np.transpose(L_deriv_copy, axes=[0, 2, 1]))
        return L_deriv * self.R
    
    def L_deriv_full(self):
        """
        Should return gradient of the inductance matrix L that satisfies 
        L^(-1) * Psi = I for the PSC arrays.
        Returns
        -------
            grad_L: 3D numpy array, shape (2 * num_psc, num_psc, num_plasma_points) 
                The gradient of the L matrix with respect to the PSC angles
                alpha_i and delta_i. 
        """
        # t1 = time.time()
        nn = self.num_psc
        q = 0
        self.dL_daa = np.zeros((nn, nn * self.symmetry, nn))
        self.dL_dad = np.zeros((nn, nn * self.symmetry, nn))
        self.dL_ddd = np.zeros((nn, nn * self.symmetry, nn))
        self.dL_dda = np.zeros((nn, nn * self.symmetry, nn))
        for fp in range(self.nfp):
            for stell in self.stell_list:
                dL = sopp.L_deriv_simd(
                    self.grid_xyz_all_normalized[q * nn: (q + 1) * nn], 
                    self.alphas_total[q * nn: (q + 1) * nn],
                    self.deltas_total[q * nn: (q + 1) * nn],
                    self.quad_points_phi,
                    self.quad_weights,
                    nn,
                    self.stellsym,  # These  last few params are defunct in the function
                    self.nfp
                )  
                # self.dpsi_daa[q * nn:(q + 1) * nn] = dpsi[:nn] * self.aaprime_aa[q * nn:(q + 1) * nn]
                self.dL_daa[:, q * nn:(q + 1) * nn, :] += dL[:nn, :, :] * self.aaprime_aa[q * nn: (q + 1) * nn, None, None]
                self.dL_dad[:, q * nn:(q + 1) * nn, :] += dL[:nn, :, :] * self.aaprime_dd[q * nn: (q + 1) * nn, None, None]
                self.dL_ddd[:, q * nn:(q + 1) * nn, :] += dL[:nn, :, :] * self.ddprime_dd[q * nn: (q + 1) * nn, None, None]
                self.dL_dda[:, q * nn:(q + 1) * nn, :] += dL[:nn, :, :] * self.ddprime_aa[q * nn: (q + 1) * nn, None, None]

        self.dL_daa = (self.dL_daa + np.transpose(self.dL_daa, axes=[2, 1, 0])) * self.R
        self.dL_dad = (self.dL_daa + np.transpose(self.dL_daa, axes=[2, 1, 0])) * self.R
        self.dL_ddd = (self.dL_daa + np.transpose(self.dL_daa, axes=[2, 1, 0])) * self.R
        self.dL_dda = (self.dL_daa + np.transpose(self.dL_daa, axes=[2, 1, 0])) * self.R
        # self.dL_full = L_deriv
        # nsym = self.symmetry
        # L_deriv_copy = np.zeros(L_deriv.shape)
        
        # # Extra work required to impose the symmetries correctly
        # nn = self.num_psc
        # ncoils_sym = nn * nsym
        # if nsym > 1:
        #     q = 0
        #     for fp in range(self.nfp):
        #         for stell in self.stell_list:
        #             L_deriv_copy[:nn, :, :] += L_deriv[q * nn:(q + 1) * nn, :, :] * self.aaprime_aa[q * nn: (q + 1) * nn, None, None] + L_deriv[ncoils_sym + q * nn:ncoils_sym + (q + 1) * nn, :, :] * self.ddprime_aa[q * nn: (q + 1) * nn, None, None]
        #             L_deriv_copy[ncoils_sym:ncoils_sym + nn, :, :] += L_deriv[q * nn:(q + 1) * nn, :, :] * self.aaprime_dd[q * nn: (q + 1) * nn, None, None] + L_deriv[ncoils_sym + q * nn:ncoils_sym + (q + 1) * nn, :, :] * self.ddprime_dd[q * nn: (q + 1) * nn, None, None]
        #             q = q + 1
        # else:
        #     L_deriv_copy = L_deriv
            
        # # symmetrize it
        # L_deriv = (L_deriv_copy + np.transpose(L_deriv_copy, axes=[0, 2, 1]))
        # return L_deriv * self.R
    
    def psi_deriv(self):
        """
        Should return gradient of the inductance matrix L that satisfies 
        L^(-1) * Psi = I for the PSC arrays.
        Returns
        -------
            grad_psi: 1D numpy array, shape (2 * num_psc) 
                The gradient of the psi vector with respect to the PSC angles
                alpha_i and delta_i. 
        """
        nn = self.num_psc
        psi_deriv = np.zeros(2 * nn)
        # psi_deriv_unrotated = np.zeros(2 * nn)
        q = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                dpsi = sopp.dpsi_dkappa(
                    contig(self.I_TF),
                    contig(self.dl_TF),
                    contig(self.gamma_TF),
                    contig(self.grid_xyz_all[q * nn: (q + 1) * nn, :]),
                    contig(self.alphas_total[q * nn: (q + 1) * nn]),
                    contig(self.deltas_total[q * nn: (q + 1) * nn]),
                    contig(self.coil_normals_all[q * nn: (q + 1) * nn, :]),
                    contig(self.quad_points_rho),
                    contig(self.quad_points_phi),
                    self.quad_weights,
                    self.R,
                ) 
                # psi_deriv_unrotated[:nn] += dpsi[:nn] 
                # psi_deriv_unrotated[nn:] += dpsi[nn:]
                psi_deriv[:nn] += dpsi[:nn] * self.aaprime_aa[q * nn:(q + 1) * nn] + dpsi[nn:] * self.ddprime_aa[q * nn:(q + 1) * nn]
                psi_deriv[nn:] += dpsi[:nn] * self.aaprime_dd[q * nn:(q + 1) * nn] + dpsi[nn:] * self.ddprime_dd[q * nn:(q + 1) * nn]
                # dpsi = sopp.dpsi_dkappa(
                #     contig(self.I_TF),
                #     contig(self.dl_TF),
                #     contig(self.gamma_TF),
                #     contig(self.grid_xyz_all[q * nn: (q + 1) * nn, :]),
                #     contig(self.alphas),
                #     contig(self.deltas),
                #     contig(self.coil_normals_all[q * nn: (q + 1) * nn, :]),
                #     contig(self.quad_points_rho),
                #     contig(self.quad_points_phi),
                #     self.quad_weights,
                #     self.R,
                # ) 
                # psi_deriv_unrotated[:nn] += dpsi[:nn] 
                # psi_deriv_unrotated[nn:] += dpsi[nn:]
                q += 1
        # self.dpsi2 = psi_deriv_unrotated * (1.0 / self.gamma_TF.shape[1])  # / self.nfp / (self.stellsym + 1.0)
        self.dpsi = psi_deriv * (1.0 / self.gamma_TF.shape[1]) / self.nfp / (self.stellsym + 1.0)  # Factors because TF fields get overcounted
        return self.dpsi
    
    def psi_deriv_full(self):
        """
        Should return gradient of the inductance matrix L that satisfies 
        L^(-1) * Psi = I for the PSC arrays.
        Returns
        -------
            grad_psi: 1D numpy array, shape (2 * num_psc) 
                The gradient of the psi vector with respect to the PSC angles
                alpha_i and delta_i. 
        """
        nn = self.num_psc
        psi_deriv = np.zeros(2 * nn * self.symmetry)
        self.dpsi_daa = np.zeros(nn)
        self.dpsi_dad = np.zeros(nn)
        self.dpsi_ddd = np.zeros(nn)
        self.dpsi_dda = np.zeros(nn)
        q = 0
        qq = 0
        for fp in range(self.nfp):
            for stell in self.stell_list:
                dpsi = sopp.dpsi_dkappa(
                    contig(self.I_TF),
                    contig(self.dl_TF),
                    contig(self.gamma_TF),
                    contig(self.grid_xyz_all[q * nn: (q + 1) * nn, :]),
                    contig(self.alphas_total[q * nn: (q + 1) * nn]),
                    contig(self.deltas_total[q * nn: (q + 1) * nn]),
                    contig(self.coil_normals_all[q * nn: (q + 1) * nn, :]),
                    contig(self.quad_points_rho),
                    contig(self.quad_points_phi),
                    self.quad_weights,
                    self.R,
                ) 
                # psi_deriv[2 * q * nn:(2 * q + 2) * nn] = 
                self.dpsi_daa += dpsi[:nn] * self.aaprime_aa[q * nn:(q + 1) * nn]
                self.dpsi_dad += dpsi[:nn] * self.aaprime_dd[q * nn:(q + 1) * nn]
                self.dpsi_ddd += dpsi[nn:] * self.ddprime_dd[q * nn:(q + 1) * nn]
                self.dpsi_dda += dpsi[nn:] * self.ddprime_aa[q * nn:(q + 1) * nn]
                psi_deriv[qq * nn:(qq + 1) * nn] = dpsi[:nn] * self.aaprime_aa[q * nn:(q + 1) * nn] + dpsi[nn:] * self.ddprime_aa[q * nn:(q + 1) * nn]
                psi_deriv[(qq + 1) * nn:(qq + 2) * nn] = dpsi[:nn] * self.aaprime_dd[q * nn:(q + 1) * nn] + dpsi[nn:] * self.ddprime_dd[q * nn:(q + 1) * nn]
                q += 1
                qq += 2
        # print(psi_deriv)
        # exit()
        # dpsi = sopp.dpsi_dkappa(
        #     contig(self.I_TF),
        #     contig(self.dl_TF),
        #     contig(self.gamma_TF),
        #     contig(self.grid_xyz_all),
        #     contig(self.alphas_total),
        #     contig(self.deltas_total),
        #     contig(self.coil_normals_all),
        #     contig(self.quad_points_rho),
        #     contig(self.quad_points_phi),
        #     self.quad_weights,
        #     self.R,
        # ) 
        # psi_deriv[2 * q * nn:(2 * q + 2) * nn] = 
        self.dpsi_daa *= (1.0 / self.gamma_TF.shape[1]) / self.nfp / (self.stellsym + 1.0)
        self.dpsi_dad *= (1.0 / self.gamma_TF.shape[1]) / self.nfp / (self.stellsym + 1.0)
        self.dpsi_ddd *= (1.0 / self.gamma_TF.shape[1]) / self.nfp / (self.stellsym + 1.0)
        self.dpsi_dda *= (1.0 / self.gamma_TF.shape[1]) / self.nfp / (self.stellsym + 1.0)
        self.dpsi_full = psi_deriv * (1.0 / self.gamma_TF.shape[1])   # / self.nfp / (self.stellsym + 1.0)
    
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
        self.coil_normals = np.array(
            [np.cos(alphas) * np.sin(deltas),
              -np.sin(alphas),
              np.cos(alphas) * np.cos(deltas)]
        ).T
        alphas = np.arctan2(
                -self.coil_normals[:, 1], 
                np.sqrt(self.coil_normals[:, 0] ** 2 + self.coil_normals[:, 2] ** 2))
        deltas = np.arcsin(self.coil_normals[:, 0] / \
                                    np.sqrt(self.coil_normals[:, 0] ** 2 + self.coil_normals[:, 2] ** 2))
        self.alphas = alphas
        self.deltas = deltas
        kappas = np.ravel(np.array([self.alphas, self.deltas]))
        self.kappas = kappas
        
        self.coil_normals = np.array(
            [np.cos(self.alphas) * np.sin(self.deltas),
              -np.sin(self.alphas),
              np.cos(self.alphas) * np.cos(self.deltas)]
        ).T
        # deal with -0 terms in the normals, which screw up the arctan2 calculations
        # self.coil_normals[
        #     np.logical_and(np.isclose(self.coil_normals, 0.0), 
        #                    np.copysign(1.0, self.coil_normals) < 0)
        #     ] *= -1.0
                
        # Apply discrete symmetries to the alphas and deltas and coordinates
        self.update_alphas_deltas()
        
        # Recompute the inductance matrices with the newly rotated coils
        L_total = sopp.L_matrix(
            self.grid_xyz_all_normalized, 
            self.alphas_total, 
            self.deltas_total,
            self.quad_points_phi,
            self.quad_weights
        )
        L_total = (L_total + L_total.T)
        # Add in self-inductances
        np.fill_diagonal(L_total, (np.log(8.0 * self.R / self.a) - 2.0) * 4 * np.pi)
        # rescale inductance
        # if keeping track of the number of coil turns, need factor
        # of Nt ** 2 below as well
        if hasattr(self, 'L'):
            'do nothing'
        else:
            self.L = L_total * self.R
            # print(self.L)
            # self.L = np.diag(np.diag(self.L))
            # print('cond = ', np.linalg.cond(self.L))
        
    def update_psi(self):
        """
        Update the flux grid with the new normal vectors.
        """
        flux_grid = sopp.flux_xyz(
            contig(self.grid_xyz_all), 
            contig(self.alphas_total),
            contig(self.deltas_total), 
            contig(self.quad_points_rho), 
            contig(self.quad_points_phi), 
        )
        self.flux_grid = np.array(flux_grid).reshape(-1, 3)
        self.B_TF.set_points(contig(self.flux_grid))
        N = len(self.quad_points_rho)
        # # Update the flux values through the newly rotated coils
        self.psi_total = sopp.psi_check(
                    contig(self.I_TF),
                    contig(self.dl_TF),
                    contig(self.gamma_TF),
                    contig(self.grid_xyz_all),
                    contig(self.alphas_total),
                    contig(self.deltas_total),
                    contig(self.coil_normals_all),
                    contig(self.quad_points_rho),
                    contig(self.quad_points_phi),
                    self.quad_weights,
                    self.R,
                ) 
        self.psi_total *=  (1.0 / self.gamma_TF.shape[1]) * 1e-7 / self.nfp / (self.stellsym + 1.0)

        # print('psi1 = ', self.psi_total, self.psi_total.shape)
        # print('a, d = ', self.alphas, self.deltas, self.alphas_total, self.deltas_total)
        
        self.psi_total = sopp.flux_integration(
            contig(self.B_TF.B().reshape(len(self.alphas_total), N, N, 3)),
            contig(self.quad_points_rho),
            contig(self.coil_normals_all),
            self.quad_weights
        )
        # print('psi2 = ', self.psi_total)
        self.psi = self.psi_total[:self.num_psc]
        self.B_TF.set_points(self.plasma_points)  # reset the plasma points (update_phi changes them)
        
    def setup_full_grid(self):
        """
        Initialize the field-period and stellarator symmetrized grid locations
        and normal vectors for all the PSC coils. Note that when alpha_i
        and delta_i angles change, coil_i location does not change, only
        its normal vector changes. 
        """
        nn = self.num_psc
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
                
        self.grid_xyz_all = contig(self.grid_xyz_all)
        self.grid_xyz_all_normalized = self.grid_xyz_all / self.R
        # self.update_alphas_deltas()
    
    def update_alphas_deltas(self):
        """
        Initialize the field-period and stellarator symmetrized normal vectors
        for all the PSC coils. This is required whenever the alpha_i
        and delta_i angles change.
        """
        # Made this unnecessarily fast xsimd calculation for setting the 
        # variables from the discrete symmetries
        (self.coil_normals_all, self.alphas_total, self.deltas_total, 
          self.aaprime_aa, self.aaprime_dd, self.ddprime_dd, self.ddprime_aa
          ) = sopp.update_alphas_deltas_xsimd(
              contig(self.coil_normals), 
              self.nfp, 
              int(self.stellsym)
        )
        assert np.allclose(self.alphas, self.alphas_total[:self.num_psc])
        assert np.allclose(self.deltas, self.deltas_total[:self.num_psc])
        self.alphas_total = contig(self.alphas_total)
        self.deltas_total = contig(self.deltas_total)
        self.coil_normals_all = contig(self.coil_normals_all)
        assert np.allclose(self.coil_normals, self.coil_normals_all[:self.num_psc, :])
        
    # def grad_TF(self, JF):  # , JF):
    #     # Just return a finite difference estimate for now!
    #     print(JF.dof_names)
    #     dofs = JF.x
    #     h = np.random.uniform(size=dofs.shape)
    #     orig_dof = []
    #     eps = 1e-4
    #     x1 = dofs + eps*h
    #     x2 = dofs - eps*h
    #     for i in range(len(dofs)):
    #         name_i = JF.dof_names[i]
    #         for j in range(len(self.B_TF._coils)):
    #             cj = self.B_TF._coils[j]
    #             for k in range(len(cj.dof_names)):
    #                 orig_dof.append(cj.x[k])
    #                 name_k = cj.dof_names[k]
    #                 print(cj._names)
    #                 if name_i == name_k:
    #                     self.B_TF._coils[j].set(name_k, x1[i])
    #     self.update_TF(self.B_TF)
    #     Ax_b = (self.Bn_PSC + self.b_opt) * self.grid_normalization
    #     fB1 = (Ax_b.T @ Ax_b) * self.fac2_norm2
        
    #     for i in range(len(dofs)):
    #         name_i = JF.dof_names[i]
    #         for j in range(len(self.B_TF._coils)):
    #             cj = self.B_TF._coils[j]
    #             for k in range(len(cj.dof_names)):
    #                 name_k = cj.dof_names[k]
    #                 if name_i == name_k:
    #                     self.B_TF._coils[j].set(cj.local_dof_names[k], x2[i])
    #     self.update_TF(self.B_TF)
    #     Ax_b = (self.Bn_PSC + self.b_opt) * self.grid_normalization
    #     fB2 = (Ax_b.T @ Ax_b) * self.fac2_norm2
        
    #     # Refix things
    #     for i in range(len(dofs)):
    #         name_i = JF.dof_names[i]
    #         for j in range(len(self.B_TF._coils)):
    #             cj = self.B_TF._coils[j]
    #             for k in range(len(cj.dof_names)):
    #                 name_k = cj.dof_names[k]
    #                 if name_i == name_k:
    #                     self.B_TF._coils[j].set(cj.local_dof_names[k], orig_dof[k])
    #             cj.fix_all()
    #     self.update_TF(self.B_TF)
        
    #     # Note b_opt doesn't get updated when you do this! Only changes the fluxes in the PSCs
    #     return (fB1 - fB2) / (2 * eps)
        
        # ALinv = (self.grid_normalization[:, None] * self.A_matrix) @ self.L_inv[:self.num_psc, :self.num_psc]
        # flux_grid = sopp.flux_xyz(
        #     contig(self.grid_xyz_all), 
        #     contig(self.alphas_total),
        #     contig(self.deltas_total), 
        #     contig(self.quad_points_rho), 
        #     contig(self.quad_points_phi), 
        # )
        # self.flux_grid = np.array(flux_grid).reshape(-1, 3)
        # self.B_TF.set_points(contig(self.flux_grid))
        # N = len(self.quad_points_rho)
        # dB_by_dX = self.B_TF.dB_by_dX()  #.reshape(len(self.alphas_total), N, N, 3, 3)
        # dB_by_dFourier = np.zeros((dB_by_dX.shape[0], len(JF.x), 3))
        # for i in range(dB_by_dX.shape[0]):
        #     for j in range(3):
        #         dB_by_dFourier[i, :, j] = self.B_TF.B_vjp(dB_by_dX[i, :, j])(JF)
        # # integrate dB/dX over the loop
        # dpsi_by_dX = sopp.dB_by_dX_integration(
        #     contig(dB_by_dFourier.reshape(len(self.alphas_total), N, N, -1, 3)),
        #     contig(self.quad_points_rho),
        #     contig(self.coil_normals_all),
        #     self.quad_weights
        # )
        # dpsi_by_dX = dpsi_by_dX[:self.num_psc, :]  # @ dX_by_dFourier
        # ALinv_dpsi = ALinv @ dpsi_by_dX
        # self.B_TF.set_points(self.plasma_points)
        # return ALinv_dpsi.T @ (ALinv @ self.psi - self.b_opt) * self.fac2_norm
    
    # def update_TF(self, B_TF):
    #     from simsopt.util import calculate_on_axis_B

    #     # B_TF = BiotSavart(coils_TF)
    #     B_axis = calculate_on_axis_B(B_TF, self.plasma_boundary, print_out=False)
        
    #     # Normalization of the ||A*Linv*psi - b||^2 objective 
    #     # representing Bnormal errors on the plasma surface
    #     self.normalization = B_axis ** 2 * self.plasma_boundary.area()
    #     self.Baxis_A = B_axis * self.plasma_boundary.area()
    #     self.fac2_norm = self.fac ** 2 / self.normalization
    #     self.fac2_norm2 = self.fac2_norm * 0.5
    #     self.B_TF = B_TF
    #     # self.update_curves()
    #     self.update_psi()
    #     self.setup_currents_and_fields()
    #     self.b_vector()

    # def update_curves_and_currents(self, dofs, B_TF):
    #     """
    #     """   
    #     from simsopt.util import calculate_on_axis_B
    #     # dofs need to be converted to kappas
    #     # kappas = np.zeros(len(dofs) // 2)  # two Euler angles, 4 quaternion dofs
    #     # dofs[3] = cos(alpha / 2.0) * cos(delta / 2.0)
    #     # dofs[4] = sin(alpha / 2.0) * cos(delta / 2.0)
    #     # dofs[5] = cos(alpha / 2.0) * sin(delta / 2.0)
    #     # dofs[6] = -sin(alpha / 2.0) * sin(delta / 2.0)
    #     # dofs[3] ** 2 + dofs[5] ** 2 = cos(alpha / 2.0) ** 2
    #     # dofs[3] ** 2 + dofs[4] ** 2 = cos(delta / 2.0) ** 2
    #     # constraints on the dofs
    #     # dofs[3] ** 2 + dofs[5] ** 2 <= 1
    #     # dofs[3] ** 2 + dofs[4] ** 2 <= 1
    #     # dofs[3] + dofs[6] = cos(alpha / 2.0 + delta / 2.0)
    #     # dofs[4] + dofs[5] = sin(alpha / 2.0 + delta / 2.0)
    #     # (dofs[3] + dofs[6])^2 + (dofs[4] + dofs[5])^2 = 1 
    #     # How to apply this quadratic but convex constraint? 
    #     # Would be better to directly optimize alphas and deltas... 
    #     # Could create new variable in CurvePlanarFourier?
    #     dofs = dofs.reshape(-1, 4)
    #     print(dofs, np.sqrt(dofs[:, 0] ** 2 + dofs[:, 2] ** 2))
    #     # dofs[6] is duplicate and not needed
    #     # print(dofs)
    #     # Normalize the quaternion
    #     dofs = dofs / np.sqrt(np.sum(dofs ** 2, axis=-1))
    #     alphas = np.arctan2(2 * (dofs[:, 0] * dofs[:, 1] + dofs[:, 2] * dofs[:, 3]), 
    #                         1 - 2.0 * (dofs[:, 1] ** 2 + dofs[:, 2] ** 2))
    #     deltas = -np.pi / 2.0 + 2.0 * np.arctan2(
    #         np.sqrt(1.0 + 2 * (dofs[:, 0] * dofs[:, 2] - dofs[:, 1] * dofs[:, 3])), 
    #         np.sqrt(1.0 - 2 * (dofs[:, 0] * dofs[:, 2] - dofs[:, 1] * dofs[:, 3])))
    #     print('a, d = ', alphas, deltas)
    #     # alphas = 2.0 * np.arcsin(np.sqrt(dofs[:, 1] ** 2 + dofs[:, 3] ** 2))
    #     # deltas = 2.0 * np.arccos(np.sqrt(dofs[:, 0] ** 2 + dofs[:, 1] ** 2))
    #     # print('a, d = ', alphas, deltas)
    #     self.setup_orientations(alphas, deltas)
    #     # self.update_curves()
    #     B_axis = calculate_on_axis_B(B_TF, self.plasma_boundary, print_out=False)
    #     self.normalization = B_axis ** 2 * self.plasma_boundary.area()
    #     self.Baxis_A = B_axis * self.plasma_boundary.area()
    #     self.fac2_norm = self.fac ** 2 / self.normalization
    #     self.fac2_norm2 = self.fac2_norm * 0.5
    #     self.B_TF = B_TF
    #     self.update_psi()
    #     self.setup_currents_and_fields()
    #     self.b_vector()