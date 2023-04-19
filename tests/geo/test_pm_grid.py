from pathlib import Path
import unittest

import numpy as np
import simsoptpp as sopp

from simsopt.field import (BiotSavart, Current, DipoleField, InterpolatedField,
                           coils_via_symmetries, Coil)
from simsopt.geo import (PermanentMagnetGrid, SurfaceRZFourier,
                         create_equally_spaced_curves)
from simsopt.objectives import SquaredFlux
from simsopt.solve import GPMO, relax_and_split
from simsopt.util import *
from simsopt.util.polarization_project import *

#from . import TEST_DIR
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

# File for the desired boundary magnetic surface:
filename = TEST_DIR / 'input.LandremanPaul2021_QA'


class Testing(unittest.TestCase):

    def test_bad_params(self):
        """
            Test that the permanent magnet optimizer initialization
            correctly catches bad instances of the function arguments.
        """
        nphi = 32
        ntheta = 32
        Bn = np.zeros((nphi, ntheta))
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s1 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s2 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s1.extend_via_projected_normal(0.1)
        s2.extend_via_projected_normal(0.2)

        with self.assertRaises(ValueError):
            PermanentMagnetGrid(
                s, s1, s2, Bn, dr=-0.05 
            )
        with self.assertRaises(TypeError):
            PermanentMagnetGrid(
                s, s1, s2, 
            )
        with self.assertRaises(ValueError):
            PermanentMagnetGrid(
                s, s1, s2, Bn, Nx=-2 
            )
        with self.assertRaises(TypeError):
            PermanentMagnetGrid(
                s, s1, s2, Bn, coil_offset=0.1, dr=0.05, plasma_offset=0.0,
            )
        with self.assertRaises(TypeError):
            PermanentMagnetGrid(
                10,
            )
        with self.assertRaises(ValueError):
            PermanentMagnetGrid(
                s, 10, s2, Bn, 
            )
        with self.assertRaises(ValueError):
            PermanentMagnetGrid(
                s, s1, 10, Bn, 
            )
        with self.assertRaises(ValueError):
            PermanentMagnetGrid(
                s, s1, s2, Bn, coordinate_flag='cylindrical', pol_vectors=[10],
            )
        inner = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta-2)
        with self.assertRaises(ValueError):
            PermanentMagnetGrid(
                s, inner, outer, Bn, 
            )
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta+2)
        with self.assertRaises(ValueError):
            PermanentMagnetGrid(
                s, inner, outer, Bn, 
            )
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi-2, ntheta=ntheta)
        with self.assertRaises(ValueError):
            PermanentMagnetGrid(
                s, inner, outer, Bn, 
            )
        outer = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi+2, ntheta=ntheta)
        with self.assertRaises(ValueError):
            PermanentMagnetGrid(
                s, inner, outer, Bn, 
            )
        with self.assertRaises(ValueError):
            PermanentMagnetGrid(
                s, s1, s2, Bn=0.0 
            )

        with self.assertRaises(ValueError):
            pm = PermanentMagnetGrid(
                s, s1, s2, Bn=np.zeros((nphi, ntheta // 2))
            )
            pm.geo_setup()
        with self.assertRaises(ValueError):
            pm = PermanentMagnetGrid(
                s, s1, s2, Bn=np.zeros((nphi, ntheta))
            )
            pm.geo_setup_from_famus(TEST_DIR / 'zot80.log')
        with self.assertRaises(ValueError):
            pm = PermanentMagnetGrid(
                s, s1, s2, Bn=np.zeros((nphi, ntheta)), pol_vectors=np.ones((5, 3, 2))
            )
        with self.assertRaises(ValueError):
            pm = PermanentMagnetGrid(
                s, s1, s2, Bn=np.zeros((nphi, ntheta)), pol_vectors=np.ones((5, 3))
            )
        with self.assertRaises(ValueError):
            pm = PermanentMagnetGrid(
                s, s1, s2, Bn=np.zeros((nphi, ntheta)), pol_vectors=np.ones((5, 3, 3)), coordinate_flag='cylindrical'
            )

    def test_optimize_bad_parameters(self):
        """
            Test that the permanent magnet optimize
            correctly catches bad instances of the function arguments.
        """
        nphi = 32
        ntheta = 32
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s1 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s2 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s1.extend_via_projected_normal(0.1)
        s2.extend_via_projected_normal(0.2)

        # Create some initial coils:
        base_curves = create_equally_spaced_curves(2, s.nfp, stellsym=False, R0=1.0, R1=0.5, order=5)
        base_currents = [Current(1e5) for i in range(2)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))

        # Create PM class
        Bn = np.sum(bs.B().reshape(nphi, ntheta, 3) * s.unitnormal(), axis=-1)
        pm_opt = PermanentMagnetGrid(s, s1, s2, dr=0.15, Bn=Bn)
        pm_opt.geo_setup()

        # Note that the rest of the optimization parameters are checked
        # interactively when python permanent_magnet_optimization.py True 
        # is used on the command line. 
        with self.assertRaises(ValueError):
            _, _, _, _, = relax_and_split(pm_opt)

        with self.assertRaises(ValueError):
            _, _, _, = relax_and_split(pm_opt, m0=np.zeros(pm_opt.ndipoles * 3 // 2))

        with self.assertRaises(ValueError):
            _, _, _, = relax_and_split(pm_opt, m0=np.zeros((pm_opt.ndipoles, 3)))

        with self.assertRaises(ValueError):
            mmax = np.ravel(np.array([pm_opt.m_maxima, pm_opt.m_maxima, pm_opt.m_maxima]).T)
            _, _, _, = relax_and_split(pm_opt, m0=np.ones((pm_opt.ndipoles * 3)) * mmax)

        kwargs = {}
        GPMO(pm_opt, algorithm='baseline', **kwargs)

        kwargs = initialize_default_kwargs('GPMO')
        with self.assertRaises(ValueError):
            GPMO(pm_opt, algorithm='backtracking', **kwargs)

        with self.assertRaises(ValueError):
            GPMO(pm_opt, algorithm='multi', **kwargs)

        pm_opt = PermanentMagnetGrid(s, s1, s2, dr=0.15, Bn=Bn)

        with self.assertRaises(ValueError):
            _, _, _ = relax_and_split(pm_opt)

        with self.assertRaises(ValueError):
            GPMO(pm_opt, algorithm='multi', **kwargs)

    def test_projected_normal(self):
        """
            Make two RZFourier surfaces, extend one of them with
            the projected normal vector in the RZ plane, and check
            that both surfaces still have the same values of the
            toroidal angle. 
        """
        nphi = 32
        ntheta = 32
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        p = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s.extend_via_projected_normal(0.1)
        pgamma = p.gamma().reshape(-1, 3)
        pphi = np.arctan2(pgamma[:, 1], pgamma[:, 0])
        sgamma = s.gamma().reshape(-1, 3)
        sphi = np.arctan2(sgamma[:, 1], sgamma[:, 0])
        assert np.allclose(pphi, sphi)
        assert not np.allclose(pgamma[:, 0], sgamma[:, 0])
        assert not np.allclose(pgamma[:, 1], sgamma[:, 1])
        assert not np.allclose(pgamma[:, 2], sgamma[:, 2])

    def test_Bn(self):
        """
            Creates a realistic QA configuration, some permanent magnets, and optimizes it
            with the default parameters. Checks the the pm_opt object agrees with Bn and
            f_B with the DipoleField + SquaredFlux way of calculating Bn and f_B.
        """
        nphi = 32
        ntheta = 32
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s1 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s2 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        s1.extend_via_projected_normal(0.1)
        s2.extend_via_projected_normal(0.2)

        # Create some initial coils:
        base_curves = create_equally_spaced_curves(2, s.nfp, stellsym=False, R0=1.0, R1=0.5, order=5)
        base_currents = [Current(1e5) for i in range(2)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
        bs = BiotSavart(coils)
        bs.set_points(s.gamma().reshape((-1, 3)))

        # Create PM class
        Bn = np.sum(bs.B().reshape(nphi, ntheta, 3) * s.unitnormal(), axis=-1)
        pm_opt = PermanentMagnetGrid(s, s1, s2, dr=0.15, Bn=Bn)
        pm_opt.geo_setup()
        _, _, _, = relax_and_split(pm_opt)
        b_dipole = DipoleField(
            pm_opt.dipole_grid_xyz,
            pm_opt.m_proxy,
            nfp=s.nfp,
            coordinate_flag=pm_opt.coordinate_flag,
            m_maxima=pm_opt.m_maxima,
        )
        b_dipole.set_points(s.gamma().reshape(-1, 3))

        # check Bn
        Nnorms = np.ravel(np.sqrt(np.sum(s.normal() ** 2, axis=-1)))
        Ngrid = nphi * ntheta
        Bn_Am = (pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms) 
        assert np.allclose(Bn_Am.reshape(nphi, ntheta), np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))

        # check <Bn>
        B_opt = np.mean(np.abs(pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms))
        B_dipole_field = np.mean(np.abs(np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        assert np.isclose(B_opt, B_dipole_field)

        # check integral Bn^2
        f_B_Am = 0.5 * np.linalg.norm(pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj, ord=2) ** 2
        f_B = SquaredFlux(s, b_dipole, -Bn).J()
        assert np.isclose(f_B, f_B_Am)

        # Create PM class with cylindrical bricks
        Bn = np.sum(bs.B().reshape(nphi, ntheta, 3) * s.unitnormal(), axis=-1)
        pm_opt = PermanentMagnetGrid(s, s1, s2, dr=0.15, Bn=Bn, coordinate_flag='cylindrical')
        pm_opt.geo_setup()
        _, _, _, = relax_and_split(pm_opt)
        b_dipole = DipoleField(
            pm_opt.dipole_grid_xyz,
            pm_opt.m_proxy,
            nfp=s.nfp,
            coordinate_flag=pm_opt.coordinate_flag,
            m_maxima=pm_opt.m_maxima,
        )
        b_dipole.set_points(s.gamma().reshape(-1, 3))

        # check Bn
        Nnorms = np.ravel(np.sqrt(np.sum(s.normal() ** 2, axis=-1)))
        Ngrid = nphi * ntheta
        Bn_Am = (pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms) 
        assert np.allclose(Bn_Am.reshape(nphi, ntheta), np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))

        # check <Bn>
        B_opt = np.mean(np.abs(pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms))
        B_dipole_field = np.mean(np.abs(np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        assert np.isclose(B_opt, B_dipole_field)

        # check integral Bn^2
        f_B_Am = 0.5 * np.linalg.norm(pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj, ord=2) ** 2
        f_B = SquaredFlux(s, b_dipole, -Bn).J()
        assert np.isclose(f_B, f_B_Am)

    def test_grid_chopping(self):
        """
            Makes a tokamak, extends two toroidal surfaces from this surface, and checks
            that the grid chopping function correctly removes magnets that are not between
            the inner and outer surfaces.
        """
        filename = TEST_DIR / 'input.circular_tokamak'
        nphi = 32
        ntheta = 32
        contig = np.ascontiguousarray
        s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)
        s1 = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)
        s2 = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)
        s1.extend_via_projected_normal(1)
        s2.extend_via_projected_normal(2)

        # Generate uniform grid
        R0 = s.get_rc(0, 0)  # major radius
        r0 = s.get_rc(1, 0)  # minor radius
        r = np.linspace(0, 5)
        theta = np.linspace(0, 2 * np.pi)
        phi = [0, np.pi]
        r, phi, theta = np.meshgrid(r, phi, theta, indexing='ij')
        R = R0 + r * np.cos(theta)
        Z = r * np.sin(theta)
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        RphiZ = np.array([R, phi, Z]).T.reshape(-1, 3)
        XYZ = np.array([X, Y, Z]).T.reshape(-1, 3)
        normal_inner = s1.unitnormal().reshape(-1, 3)   
        normal_outer = s2.unitnormal().reshape(-1, 3)
        xyz_inner = s1.gamma()
        xyz_outer = s2.gamma()
        # Repeat with cartesian functionality
        final_grid = sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
            contig(normal_inner), 
            contig(normal_outer), 
            contig(XYZ), 
            contig(xyz_inner), 
            contig(xyz_outer)
        )
        inds = np.ravel(np.logical_not(np.all(final_grid == 0.0, axis=-1)))
        final_grid = final_grid[inds, :]
        final_rz_grid = np.zeros(final_grid.shape)
        final_rz_grid[:, 0] = np.sqrt(final_grid[:, 0] ** 2 + final_grid[:, 1] ** 2)
        final_rz_grid[:, 1] = np.arctan2(final_grid[:, 1], final_grid[:, 0])
        final_rz_grid[:, 2] = final_grid[:, 2] 
        r_fit = np.sqrt((final_grid[:, 0] - R0) ** 2 + final_grid[:, 2] ** 2)
        assert (np.min(r_fit) > (r0 + 1.0))
        assert (np.max(r_fit) < (r0 + 2.0))

    def test_famus_functionality(self):
        """
            Tests the FocusData class and class functions    
        """
        mag_data = FocusData(TEST_DIR / 'zot80.focus')
        for i in range(mag_data.nMagnets):
            assert np.isclose(np.dot(np.array(mag_data.perp_vector([i])).T, np.array(mag_data.unit_vector([i]))), 0.0)
        assert (np.sum(mag_data.pho < 0) > 0)
        with self.assertRaises(RuntimeError):
            mag_data.adjust_rho(4.0)
        mag_data.flip_negative_magnets()
        assert (np.sum(mag_data.pho < 0) == 0)
        mag_data.adjust_rho(4.0)
        assert mag_data.momentq == 4.0
        mag_data.print_to_file('test')
        mag_data.init_pol_vecs(3)
        assert mag_data.pol_x.shape == (mag_data.nMagnets, 3)
        assert mag_data.pol_y.shape == (mag_data.nMagnets, 3)
        assert mag_data.pol_z.shape == (mag_data.nMagnets, 3)
        nMagnets = mag_data.nMagnets
        mag_data.repeat_hp_to_fp(nfp=2, magnet_sector=1)
        assert mag_data.nMagnets == nMagnets * 2
        assert np.allclose(mag_data.oz[:mag_data.nMagnets // 2], -mag_data.oz[mag_data.nMagnets // 2:])

    def test_polarizations(self):
        """
            Tests the polarizations and related functions from the
            polarization_project file. 
        """
        theta = 0.0 
        vecs = faceedge_vectors(theta)
        assert np.allclose(np.linalg.norm(vecs, axis=-1), 1.0)
        vecs = facecorner_vectors(theta)
        assert np.allclose(np.linalg.norm(vecs, axis=-1), 1.0)
        theta = np.pi / 6.0
        assert np.allclose(pol_fe30, faceedge_vectors(theta))
        theta = np.pi / 8.0
        assert np.allclose(pol_fe23, faceedge_vectors(theta))
        theta = 17.0 * np.pi / 180.0
        assert np.allclose(pol_fe17, faceedge_vectors(theta))
        theta = 27.0 * np.pi / 180.0
        assert np.allclose(pol_fc27, facecorner_vectors(theta))
        theta = 39.0 * np.pi / 180.0
        assert np.allclose(pol_fc39, facecorner_vectors(theta))
        pol_axes, _ = polarization_axes('face')
        assert np.allclose(pol_f, pol_axes)
        pol_axes, _ = polarization_axes('edge')
        assert np.allclose(pol_e, pol_axes)
        pol_axes, _ = polarization_axes('corner')
        assert np.allclose(pol_c, pol_axes)
        pol_axes, _ = polarization_axes('faceedge')
        assert np.allclose(pol_fe, pol_axes)
        pol_axes, _ = polarization_axes('facecorner')
        assert np.allclose(pol_fc, pol_axes)
        pol_axes, _ = polarization_axes('edgecorner')
        assert np.allclose(pol_ec, pol_axes)
        pol_axes, _ = polarization_axes('fe17')
        assert np.allclose(pol_fe17, pol_axes)
        pol_axes, _ = polarization_axes('fe23')
        assert np.allclose(pol_fe23, pol_axes)
        pol_axes, _ = polarization_axes('fe30')
        assert np.allclose(pol_fe30, pol_axes)
        pol_axes, _ = polarization_axes('fc27')
        assert np.allclose(pol_fc27, pol_axes)
        pol_axes, _ = polarization_axes('fc39')
        assert np.allclose(pol_fc39, pol_axes)
        pol_axes, _ = polarization_axes('ec23')
        assert np.allclose(pol_ec23, pol_axes)
        theta = 38.12 * np.pi / 180.0 
        vectors = facecorner_vectors(theta)
        pol_axes, _ = polarization_axes('fc_ftri')
        assert np.allclose(vectors, pol_axes)
        theta = 30.35 * np.pi / 180.0 
        vectors = faceedge_vectors(theta)
        pol_axes, _ = polarization_axes('fe_ftri')
        assert np.allclose(vectors, pol_axes)
        theta = 18.42 * np.pi / 180.0 
        vectors = faceedge_vectors(theta)
        pol_axes, _ = polarization_axes('fe_etri')
        assert np.allclose(vectors, pol_axes)
        theta = 38.56 * np.pi / 180.0 
        vectors = facecorner_vectors(theta)
        pol_axes, _ = polarization_axes('fc_etri')
        assert np.allclose(vectors, pol_axes)
        vec_shape = np.shape(vectors)[0]
        pol_axes, pol_types = polarization_axes(['fc_ftri', 'fc_etri'])
        assert np.allclose(pol_types, np.ravel(np.array([np.ones(vec_shape), 2 * np.ones(vec_shape)])))
        pol_axes, pol_types = polarization_axes(['face', 'fc_ftri', 'fe_ftri'])
        mag_data = FocusData(TEST_DIR / 'zot80.focus')
        mag_data.init_pol_vecs(len(pol_axes))
        ox = mag_data.ox
        oy = mag_data.oy
        oz = mag_data.oz
        premade_dipole_grid = np.array([ox, oy, oz]).T
        ophi = np.arctan2(premade_dipole_grid[:, 1], premade_dipole_grid[:, 0])
        cyl_r = mag_data.cyl_r
        discretize_polarizations(mag_data, ophi, pol_axes, pol_types)
        assert not np.allclose(mag_data.cyl_r, cyl_r)

    def test_pm_helpers(self):
        """
            Test the helper functions in utils/permanent_magnet_helpers.py.
        """

        # Test build of the MUSE coils
        input_name = 'input.muse'
        nphi = 8
        ntheta = nphi
        surface_filename = TEST_DIR / input_name
        s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_inner = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_outer = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        base_curves, base_currents, ncoils = read_focus_coils(TEST_DIR / 'muse_tf_coils.focus')
        coils = []
        for i in range(ncoils):
            coils.append(Coil(base_curves[i], base_currents[i]))
        base_currents[0].fix_all()

        # fix all the coil shapes
        for i in range(ncoils):
            base_curves[i].fix_all()
        bs = BiotSavart(coils)

        # Calculate average, approximate on-axis B field strength
        B0avg = calculate_on_axis_B(bs, s)
        assert np.allclose(B0avg, 0.15)

        # Repeat with wrapper function
        initialize_coils('muse_famus', TEST_DIR, '', s)
        bs = BiotSavart(coils)
        B0avg = calculate_on_axis_B(bs, s)
        assert np.allclose(B0avg, 0.15)

        # Get FAMUS dipoles
        m, m0s = get_FAMUS_dipoles(TEST_DIR / 'zot80.focus')
        m00 = m0s[0]
        assert np.all(m <= m00)
        for m0 in m0s:
            assert np.isclose(m00, m0)
        pm_opt = PermanentMagnetGrid(
            s, s_inner, s_outer,  # s_inner and s_outer overwritten in next line since using a FAMUS grid 
            Bn=np.zeros(s.normal().shape[:2]), 
        )
        pm_opt.geo_setup_from_famus(TEST_DIR / 'zot80.focus')

        # Test rescaling 
        reg_l0 = 0.2
        reg_l1 = 0.1
        reg_l2 = 1e-4
        nu = 1e5
        old_reg_l0 = reg_l0
        old_ATA_scale = pm_opt.ATA_scale
        reg_l0, reg_l1, reg_l2, nu = rescale_for_opt(pm_opt, reg_l0, reg_l1, reg_l2, nu)
        assert (reg_l0 == old_reg_l0 / (2 * nu))
        assert (pm_opt.ATA_scale == old_ATA_scale + 2 * reg_l2 + 1.0 / nu)
        reg_l0 = -1
        with self.assertRaises(ValueError):
            reg_l0, reg_l1, reg_l2, nu = rescale_for_opt(pm_opt, reg_l0, reg_l1, reg_l2, nu)
        reg_l0 = 2
        with self.assertRaises(ValueError):
            reg_l0, reg_l1, reg_l2, nu = rescale_for_opt(pm_opt, reg_l0, reg_l1, reg_l2, nu)

        # Test write PermanentMagnetGrid to FAMUS file
        write_pm_optimizer_to_famus('', pm_opt)
        m, m0s = get_FAMUS_dipoles('SIMSOPT_dipole_solution.focus')
        assert np.all(m <= m00)
        for m0 in m0s:
            assert np.isclose(m00, m0)

        # Load in file we made to FocusData class and do some tests
        mag_data = FocusData('SIMSOPT_dipole_solution.focus')
        for i in range(mag_data.nMagnets):
            assert np.isclose(np.dot(np.array(mag_data.perp_vector([i])).T, np.array(mag_data.unit_vector([i]))), 0.0)
        mag_data.print_to_file('test')
        mag_data.init_pol_vecs(3)
        assert mag_data.pol_x.shape == (mag_data.nMagnets, 3)
        assert mag_data.pol_y.shape == (mag_data.nMagnets, 3)
        assert mag_data.pol_z.shape == (mag_data.nMagnets, 3)
        nMagnets = mag_data.nMagnets
        mag_data.repeat_hp_to_fp(nfp=2, magnet_sector=1)
        assert mag_data.nMagnets == nMagnets * 2
        assert np.allclose(mag_data.oz[:mag_data.nMagnets // 2], -mag_data.oz[mag_data.nMagnets // 2:])

        # Test algorithm kwarg initialization
        kwargs = initialize_default_kwargs(algorithm='RS')
        assert isinstance(kwargs, dict)
        kwargs = initialize_default_kwargs(algorithm='GPMO')
        assert isinstance(kwargs, dict)
        kwargs = initialize_default_kwargs(algorithm='ArbVec_backtracking')
        assert isinstance(kwargs, dict)
        assert kwargs['K'] == 1000

        # Test Bnormal plots
        make_Bnormal_plots(bs, s, '', 'biot_savart_test')

        # optimize pm_opt and plot optimization progress
        kwargs = initialize_default_kwargs(algorithm='GPMO')
        kwargs['K'] = 1000
        R2_history, Bn_history, m_history = GPMO(pm_opt, 'baseline', **kwargs)
        m_history = np.transpose(m_history, [2, 0, 1])
        m_history = m_history.reshape((1, 501, 75460, 3))
        make_optimization_plots(R2_history, m_history, m_history, pm_opt, '')
        kwargs['K'] = 10
        with self.assertRaises(ValueError):
            R2_history, Bn_history, m_history = GPMO(pm_opt, 'baseline', **kwargs)


if __name__ == "__main__":
    unittest.main()
