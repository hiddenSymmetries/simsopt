from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.field import (BiotSavart, Current, DipoleField, coils_via_symmetries, Coil)
from simsopt.geo import (PermanentMagnetGrid, SurfaceRZFourier, SurfaceXYZFourier,
                         create_equally_spaced_curves)
from simsopt.objectives import SquaredFlux
from simsopt.solve import GPMO, relax_and_split
from simsopt.util import *
from simsopt.util.polarization_project import (faceedge_vectors, facecorner_vectors,
                                               pol_e, pol_f, pol_fe, pol_c,
                                               pol_fc, pol_ec, pol_fc27, pol_fc39,
                                               pol_ec23, pol_fe17, pol_fe23, pol_fe30)


TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

# File for the desired boundary magnetic surface:
filename = TEST_DIR / 'input.LandremanPaul2021_QA'


class PermanentMagnetGridTesting(unittest.TestCase):

    def test_bad_params(self):
        """
        Test that the permanent magnet optimizer initialization
        correctly catches bad instances of the function arguments.
        """
        nphi = 4
        ntheta = nphi
        Bn = np.zeros((nphi, ntheta))
        with ScratchDir("."):
            s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            s1 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            s2 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            s3 = SurfaceXYZFourier(nfp=2, stellsym=True)
            s1.extend_via_projected_normal(0.1)
            s2.extend_via_projected_normal(0.2)

            PermanentMagnetGrid(s3, Bn)

            with self.assertRaises(TypeError):
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn)
            with self.assertRaises(TypeError):
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1)
            with self.assertRaises(ValueError):
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(Bn, s1, s2, s)
            with self.assertRaises(ValueError):
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s1, s, s2, Bn)
            with self.assertRaises(TypeError):
                PermanentMagnetGrid(s)
            with self.assertRaises(ValueError):
                kwargs = {"dr": -0.05}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"dz": -0.05}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"Nx": -2}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"Ny": -2}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"Nz": -2}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"coordinate_flag": "cylindrical", "pol_vectors": [10]}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"coordinate_flag": "toroidal", "pol_vectors": [10]}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, 0, s1, s2)
            with self.assertRaises(ValueError):
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, np.zeros((nphi, ntheta // 2)), s1, s2)
            with self.assertRaises(ValueError):
                kwargs = {"m_maxima": np.ones(4)}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                PermanentMagnetGrid.geo_setup_from_famus(s, np.zeros((nphi, ntheta)), TEST_DIR / 'zot80.log')
            with self.assertRaises(ValueError):
                kwargs = {"m_maxima": np.ones(4)}
                PermanentMagnetGrid.geo_setup_from_famus(
                    s, np.zeros((nphi, ntheta)), TEST_DIR / 'zot80.focus', **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"pol_vectors": np.ones((5, 3, 2))}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                    s, np.zeros((nphi, ntheta)), s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"pol_vectors": np.ones((5, 3))}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                    s, np.zeros((nphi, ntheta)), s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"coordinate_flag": "cylindrical", "pol_vectors": np.ones((5, 3, 3))}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                    s, np.zeros((nphi, ntheta)), s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"coordinate_flag": "random"}
                PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                    s, np.zeros((nphi, ntheta)), s1, s2, **kwargs)

    def test_optimize_bad_parameters(self):
        """
        Test that the permanent magnet optimize
        correctly catches bad instances of the function arguments.
        """
        nphi = 4
        ntheta = nphi
        with ScratchDir("."):
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
            kwargs = {"dr": 0.15}
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs)

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
            kwargs['verbose'] = False
            GPMO(pm_opt, algorithm='baseline', **kwargs)

            kwargs = initialize_default_kwargs('GPMO')
            kwargs['verbose'] = False
            with self.assertRaises(ValueError):
                GPMO(pm_opt, algorithm='backtracking', **kwargs)

            with self.assertRaises(ValueError):
                GPMO(pm_opt, algorithm='multi', **kwargs)

    def test_projected_normal(self):
        """
        Make two RZFourier surfaces, extend one of them with the projected
        normal vector in the RZ plane, and check that both surfaces still
        have the same values of the toroidal angle.
        """
        nphi = 8
        ntheta = nphi
        with ScratchDir("."):
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
        nphi = 8
        ntheta = nphi
        with ScratchDir("."):
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
            kwargs = {"dr": 0.15}
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs)
            _, _, _, = relax_and_split(pm_opt)
            b_dipole = DipoleField(pm_opt.dipole_grid_xyz,
                                   pm_opt.m_proxy,
                                   nfp=s.nfp,
                                   coordinate_flag=pm_opt.coordinate_flag,
                                   m_maxima=pm_opt.m_maxima)
            b_dipole.set_points(s.gamma().reshape(-1, 3))

            # check Bn
            Nnorms = np.ravel(np.sqrt(np.sum(s.normal() ** 2, axis=-1)))
            Ngrid = nphi * ntheta
            Bn_Am = (pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms)
            np.testing.assert_allclose(Bn_Am.reshape(nphi, ntheta), np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2), atol=1e-15)

            # check <Bn>
            B_opt = np.mean(np.abs(pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms))
            B_dipole_field = np.mean(np.abs(np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
            np.testing.assert_allclose(B_opt, B_dipole_field)

            # check integral Bn^2
            f_B_Am = 0.5 * np.linalg.norm(pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj, ord=2) ** 2
            f_B = SquaredFlux(s, b_dipole, -Bn).J()
            np.testing.assert_allclose(f_B, f_B_Am)

            # Create PM class with cylindrical bricks
            Bn = np.sum(bs.B().reshape(nphi, ntheta, 3) * s.unitnormal(), axis=-1)
            kwargs_geo = {"dr": 0.15, "coordinate_flag": "cylindrical"}
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs_geo)
            mmax_new = pm_opt.m_maxima / 2.0
            kwargs_geo = {"dr": 0.15, "coordinate_flag": "cylindrical", "m_maxima": mmax_new}
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs_geo)
            np.testing.assert_allclose(pm_opt.m_maxima, mmax_new)
            mmax_new = pm_opt.m_maxima[-1] / 2.0
            kwargs_geo = {"dr": 0.15, "coordinate_flag": "cylindrical", "m_maxima": mmax_new}
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2, **kwargs_geo)
            np.testing.assert_allclose(pm_opt.m_maxima, mmax_new)
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(s, Bn, s1, s2)
            _, _, _, = relax_and_split(pm_opt)
            b_dipole = DipoleField(pm_opt.dipole_grid_xyz, pm_opt.m_proxy, nfp=s.nfp,
                                   coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima)
            b_dipole.set_points(s.gamma().reshape(-1, 3))

        # check Bn
        Nnorms = np.ravel(np.sqrt(np.sum(s.normal() ** 2, axis=-1)))
        Ngrid = nphi * ntheta
        Bn_Am = (pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms)
        np.testing.assert_allclose(Bn_Am.reshape(nphi, ntheta), np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2), atol=1e-15)

        # check <Bn>
        B_opt = np.mean(np.abs(pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms))
        B_dipole_field = np.mean(np.abs(np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        np.testing.assert_allclose(B_opt, B_dipole_field)

        # check integral Bn^2
        f_B_Am = 0.5 * np.linalg.norm(pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj, ord=2) ** 2
        f_B = SquaredFlux(s, b_dipole, -Bn).J()
        np.testing.assert_allclose(f_B, f_B_Am)

    def test_grid_chopping(self):
        """
        Makes a tokamak, extends two toroidal surfaces from this surface, and checks
        that the grid chopping function correctly removes magnets that are not between
        the inner and outer surfaces.
        """
        filename = TEST_DIR / 'input.circular_tokamak'
        nphi = 16
        ntheta = nphi
        # with ScratchDir("."):
        for range_type in ["full torus", "half period", "field period"]:
            s = SurfaceRZFourier.from_vmec_input(filename, range=range_type, nphi=nphi, ntheta=ntheta)
            s1 = SurfaceRZFourier.from_vmec_input(filename, range=range_type, nphi=nphi, ntheta=ntheta)
            s2 = SurfaceRZFourier.from_vmec_input(filename, range=range_type, nphi=nphi, ntheta=ntheta)
            s1.extend_via_projected_normal(1)
            s2.extend_via_projected_normal(2)

            # Generate uniform grid
            R0 = s.get_rc(0, 0)  # major radius
            r0 = s.get_rc(1, 0)  # minor radius
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                s, np.zeros((nphi, ntheta)), s1, s2
            )
            s.to_vtk('s')
            s1.to_vtk('s1')
            s2.to_vtk('s2')
            match_tol = 0.1
            r_cartesian = np.sqrt(pm_opt.dipole_grid_xyz[:, 0] ** 2 + pm_opt.dipole_grid_xyz[:, 1] ** 2)
            r_fit = np.sqrt((r_cartesian - R0) ** 2 + pm_opt.dipole_grid_xyz[:, 2] ** 2)
            assert (np.min(r_fit) > (r0 + 1 - match_tol))
            assert (np.max(r_fit) < (r0 + 2 + match_tol))

            # Try with a uniform cylindrical grid
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                s, np.zeros((nphi, ntheta)), s1, s2, coordinate_flag='cylindrical'
            )
            r_cartesian = np.sqrt(pm_opt.dipole_grid_xyz[:, 0] ** 2 + pm_opt.dipole_grid_xyz[:, 1] ** 2)
            r_fit = np.sqrt((r_cartesian - R0) ** 2 + pm_opt.dipole_grid_xyz[:, 2] ** 2)
            assert (np.min(r_fit) > (r0 + 1 - match_tol))
            assert (np.max(r_fit) < (r0 + 2 + match_tol))

            # Check for incompatible toroidal angles between the plasma
            # surface and the inner/outer toroidal surfaces
            s1 = SurfaceRZFourier.from_vmec_input(filename, range=range_type, nphi=nphi, ntheta=2 * ntheta)
            s2 = SurfaceRZFourier.from_vmec_input(filename, range=range_type, nphi=nphi, ntheta=2 * ntheta)
            s1.extend_via_projected_normal(1)
            s2.extend_via_projected_normal(2)
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                s, np.zeros((nphi, ntheta)), s1, s2
            )
            assert (np.min(r_fit) > (r0 + 1 - match_tol))
            assert (np.max(r_fit) < (r0 + 2 + match_tol))
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                s, np.zeros((nphi, ntheta)), s1, s2, coordinate_flag='cylindrical'
            )
            assert (np.min(r_fit) > (r0 + 1 - match_tol))
            assert (np.max(r_fit) < (r0 + 2 + match_tol))

            s1 = SurfaceRZFourier.from_vmec_input(filename, range=range_type, nphi=2 * nphi, ntheta=ntheta)
            s2 = SurfaceRZFourier.from_vmec_input(filename, range=range_type, nphi=2 * nphi, ntheta=ntheta)
            s1.extend_via_projected_normal(1)
            s2.extend_via_projected_normal(2)
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                s, np.zeros((nphi, ntheta)), s1, s2
            )
            assert (np.min(r_fit) > (r0 + 1 - match_tol))
            assert (np.max(r_fit) < (r0 + 2 + match_tol))
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                s, np.zeros((nphi, ntheta)), s1, s2, coordinate_flag='cylindrical'
            )
            assert (np.min(r_fit) > (r0 + 1 - match_tol))
            assert (np.max(r_fit) < (r0 + 2 + match_tol))

    def test_famus_functionality(self):
        """
        Tests the FocusData and FocusPlasmaBnormal classes
        and class functions.
        """
        with ScratchDir("."):
            fname_plasma = TEST_DIR / 'c09r00_B_axis_half_tesla_PM4Stell.plasma'
            bnormal_obj_ncsx = FocusPlasmaBnormal(fname_plasma)
            bn_plasma = bnormal_obj_ncsx.bnormal_grid(8, 8, 'half period')
            assert bn_plasma.shape == (8, 8)
            assert np.allclose(bnormal_obj_ncsx.bnc, 0.0)
            bnormal_obj_ncsx.stellsym = False
            bnormal_obj_ncsx.bnc = bnormal_obj_ncsx.bns
            bn_plasma = bnormal_obj_ncsx.bnormal_grid(8, 8, 'field period')
            assert bn_plasma.shape == (8, 8)

            mag_data = FocusData(TEST_DIR / 'zot80.focus', downsample=100)
            for i in range(mag_data.nMagnets):
                assert np.isclose(np.dot(np.array(mag_data.perp_vector([i])).T, np.array(mag_data.unit_vector([i]))), 0.0)

            with self.assertRaises(Exception):
                mag_data.perp_vector([mag_data.nMagnets])
            with self.assertRaises(Exception):
                mag_data.unit_vector([mag_data.nMagnets])

            assert (np.sum(mag_data.pho < 0) > 0)
            with self.assertRaises(RuntimeError):
                mag_data.adjust_rho(4.0)
            mag_data.flip_negative_magnets()
            mag_data.flip_negative_magnets()
            assert (np.sum(mag_data.pho < 0) == 0)
            mag_data.adjust_rho(4.0)
            assert mag_data.momentq == 4.0
            mag_data.has_momentq = False
            mag_data.adjust_rho(3.0)
            assert mag_data.momentq == 3.0
            mag_data.print_to_file('test')
            mag_data.has_momentq = False
            mag_data.has_op = True
            mag_data.op = mag_data.oz
            mag_data.print_to_file('test')
            nMag = mag_data.nMagnets
            mag_data.nMagnets = 0
            with self.assertRaises(RuntimeError):
                mag_data.print_to_file('test')
            mag_data.nMagnets = nMag
            mag_data.init_pol_vecs(3)
            assert mag_data.pol_x.shape == (mag_data.nMagnets, 3)
            assert mag_data.pol_y.shape == (mag_data.nMagnets, 3)
            assert mag_data.pol_z.shape == (mag_data.nMagnets, 3)
            nMagnets = mag_data.nMagnets
            mag_data.repeat_hp_to_fp(nfp=2, magnet_sector=1)
            assert mag_data.nMagnets == nMagnets * 2
            assert np.allclose(mag_data.oz[:mag_data.nMagnets // 2], -mag_data.oz[mag_data.nMagnets // 2:])
            with self.assertRaises(ValueError):
                mag_data.repeat_hp_to_fp(nfp=2, magnet_sector=10)
            mag_data.symm = 1 * np.ones(len(mag_data.symm))
            with self.assertRaises(ValueError):
                mag_data.repeat_hp_to_fp(nfp=2, magnet_sector=1)
            mag_data.symm = 2 * np.ones(len(mag_data.symm))
            phi0 = np.pi / 2
            ox2, oy2, oz2 = stell_point_transform('reflect', phi0, mag_data.ox, mag_data.oy, mag_data.oz)
            assert np.allclose(mag_data.oz, -oz2)
            ox2, oy2, oz2 = stell_point_transform('translate', phi0, mag_data.ox, mag_data.oy, mag_data.oz)
            assert np.allclose(mag_data.oz, oz2)
            symm_inds = np.where(mag_data.symm == 2)[0]
            nx, ny, nz = mag_data.unit_vector(symm_inds)
            nx2, ny2, nz2 = stell_vector_transform('reflect', phi0, nx, ny, nz)
            assert np.allclose(nz, nz2)
            assert np.allclose(nx ** 2 + ny ** 2 + nz ** 2, nx2 ** 2 + ny2 ** 2 + nz2 ** 2)
            nx2, ny2, nz2 = stell_vector_transform('translate', phi0, nx, ny, nz)
            assert np.allclose(nz, nz2)
            assert np.allclose(nx ** 2 + ny ** 2 + nz ** 2, nx2 ** 2 + ny2 ** 2 + nz2 ** 2)
            with self.assertRaises(ValueError):
                nx2, ny2, nz2 = stell_vector_transform('random', phi0, nx, ny, nz)
            with self.assertRaises(ValueError):
                nx2, ny2, nz2 = stell_point_transform('random', phi0, mag_data.ox, mag_data.oy, mag_data.oz)

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
        assert np.allclose(np.concatenate((pol_f, faceedge_vectors(theta),
                                           facecorner_vectors(theta)), axis=0),
                           face_triplet(theta, theta))
        assert np.allclose(np.concatenate((pol_e, faceedge_vectors(theta),
                                           facecorner_vectors(theta)), axis=0),
                           edge_triplet(theta, theta))
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
        mag_data = FocusData(TEST_DIR / 'zot80.focus', downsample=100)
        mag_data.init_pol_vecs(len(pol_axes))
        ox = mag_data.ox
        oy = mag_data.oy
        oz = mag_data.oz
        premade_dipole_grid = np.array([ox, oy, oz]).T
        ophi = np.arctan2(premade_dipole_grid[:, 1], premade_dipole_grid[:, 0])
        cyl_r = mag_data.cyl_r
        discretize_polarizations(mag_data, ophi, pol_axes, pol_types)
        assert not np.allclose(mag_data.cyl_r, cyl_r)
        fname_corn = TEST_DIR / 'magpie_trial104b_corners_PM4Stell.csv'
        ophi = orientation_phi(fname_corn) + np.pi
        assert (np.all(ophi <= 2 * np.pi) and np.all(ophi >= 0.0))

    def test_pm_helpers(self):
        """
            Test the helper functions in utils/permanent_magnet_helpers.py.
        """

        # Test build of the MUSE coils
        with ScratchDir("."):
            input_name = 'input.muse'
            nphi = 8
            ntheta = nphi
            surface_filename = TEST_DIR / input_name
            s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
            s_inner = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
            s_outer = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
            s_inner.extend_via_projected_normal(0.1)
            s_outer.extend_via_projected_normal(0.2)
            base_curves, base_currents, ncoils = read_focus_coils(TEST_DIR / 'muse_tf_coils.focus')
            coils = []
            for i in range(ncoils):
                coils.append(Coil(base_curves[i], base_currents[i]))
            base_currents[0].fix_all()

            # fix all the coil shapes
            for i in range(ncoils):
                base_curves[i].fix_all()
            bs = BiotSavart(coils)

            # Calculate average B field strength along the major radius
            B0avg = calculate_modB_on_major_radius(bs, s)
            assert np.allclose(B0avg, 0.15)

            # Check coil initialization for some common stellarators wout_LandremanPaul2021_QA_lowres
            s = SurfaceRZFourier.from_wout(TEST_DIR / 'wout_LandremanPaul2021_QA_lowres.nc',
                                           range="half period", nphi=nphi, ntheta=ntheta)
            base_curves, curves, coils = initialize_coils('qa', TEST_DIR, s)
            bs = BiotSavart(coils)
            B0avg = calculate_modB_on_major_radius(bs, s)
            assert np.allclose(B0avg, 0.15)

            s = SurfaceRZFourier.from_wout(TEST_DIR / 'wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc',
                                           range="half period", nphi=nphi, ntheta=ntheta)
            base_curves, curves, coils = initialize_coils('qh', TEST_DIR, s)
            bs = BiotSavart(coils)
            B0avg = calculate_modB_on_major_radius(bs, s)
            assert np.allclose(B0avg, 5.7)

            # Repeat with wrapper function
            s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
            base_curves, curves, coils = initialize_coils('muse_famus', TEST_DIR, s)
            bs = BiotSavart(coils)
            B0avg = calculate_modB_on_major_radius(bs, s)
            assert np.allclose(B0avg, 0.15)

        # Test rescaling
        Bn = np.zeros((nphi, ntheta))
        with ScratchDir("."):
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                s, Bn, s_inner, s_outer)
            reg_l0 = 0.2
            reg_l1 = 0.1
            reg_l2 = 1e-4
            nu = 1e5
            old_reg_l0 = reg_l0
            old_ATA_scale = pm_opt.ATA_scale
            reg_l0, reg_l1, reg_l2, nu = pm_opt.rescale_for_opt(reg_l0, reg_l1, reg_l2, nu)
            assert np.isclose(reg_l0, old_reg_l0 / (2 * nu))
            assert np.isclose(pm_opt.ATA_scale, old_ATA_scale + 2 * reg_l2 + 1.0 / nu)
            reg_l0 = -1
            with self.assertRaises(ValueError):
                reg_l0, reg_l1, reg_l2, nu = pm_opt.rescale_for_opt(reg_l0, reg_l1, reg_l2, nu)
            reg_l0 = 2
            with self.assertRaises(ValueError):
                reg_l0, reg_l1, reg_l2, nu = pm_opt.rescale_for_opt(reg_l0, reg_l1, reg_l2, nu)

            # Test write PermanentMagnetGrid to FAMUS file
            pm_opt.write_to_famus()
            pm_opt.coordinate_flag = 'cylindrical'
            pm_opt.write_to_famus()
            pm_opt.coordinate_flag = 'toroidal'
            pm_opt.write_to_famus()

            # Load in file we made to FocusData class and do some tests
            mag_data = FocusData('SIMSOPT_dipole_solution.focus', downsample=10)
            for i in range(mag_data.nMagnets):
                assert np.isclose(np.dot(np.array(mag_data.perp_vector([i])).T,
                                         np.array(mag_data.unit_vector([i]))),
                                  0.0)
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
        kwargs['verbose'] = False
        assert isinstance(kwargs, dict)
        assert kwargs['K'] == 1000

        with ScratchDir("."):
            # Test Bnormal plots
            make_Bnormal_plots(bs, s)

            # optimize pm_opt and plot optimization progress
            kwargs = initialize_default_kwargs(algorithm='GPMO')
            kwargs['K'] = 100
            kwargs['nhistory'] = 10
            kwargs['verbose'] = False
            R2_history, Bn_history, m_history = GPMO(pm_opt, 'baseline', **kwargs)
            m_history = np.transpose(m_history, [2, 0, 1])
            m_history = m_history.reshape(1, 11, m_history.shape[1], 3)
            make_optimization_plots(R2_history, m_history, m_history, pm_opt)

            kwargs_geo = {"downsample": 100}
            pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bn, TEST_DIR / 'zot80.focus', **kwargs_geo)
            R2_history, Bn_history, m_history = GPMO(pm_opt, 'baseline', **kwargs)
            make_optimization_plots(R2_history, m_history, m_history, pm_opt)
            kwargs['K'] = 5
            with self.assertRaises(ValueError):
                R2_history, Bn_history, m_history = GPMO(pm_opt, 'baseline', **kwargs)

    def test_pm_post_processing(self):
        """
            Test the helper functions for QFM and poincare cross sections that
            are typically run after a successful PM optimization.
        """

        # Test build of the MUSE coils
        input_name = 'input.muse'
        nphi = 8
        ntheta = nphi
        surface_filename = TEST_DIR / input_name
        s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        base_curves, curves, coils = initialize_coils('muse_famus', TEST_DIR, s)
        bs = BiotSavart(coils)
        B0avg = calculate_modB_on_major_radius(bs, s)
        assert np.allclose(B0avg, 0.15)

        # drastically downsample the grid for speed here
        kwargs = {"downsample": 100}
        with ScratchDir("."):
            pm_opt = PermanentMagnetGrid.geo_setup_from_famus(
                s, Bn=np.zeros(s.normal().shape[:2]),
                famus_filename=(TEST_DIR / 'zot80.focus'), **kwargs)
            kwargs = initialize_default_kwargs(algorithm='GPMO')
            kwargs['verbose'] = False
            kwargs['K'] = 500
            GPMO(pm_opt, 'baseline', **kwargs)
            b_dipole = DipoleField(pm_opt.dipole_grid_xyz, pm_opt.m, nfp=s.nfp,
                                   coordinate_flag=pm_opt.coordinate_flag,
                                   m_maxima=pm_opt.m_maxima)
        # Make higher resolution surface for qfm
        qphi = 2 * nphi
        quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
        quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
        s_plot = SurfaceRZFourier.from_focus(surface_filename, range="full torus",
                                             quadpoints_phi=quadpoints_phi,
                                             quadpoints_theta=quadpoints_theta)

        # Make QFM surfaces
        Bfield = bs + b_dipole
        Bfield.set_points(s_plot.gamma().reshape((-1, 3)))
        qfm_surf = make_qfm(s_plot, Bfield)
        qfm_surf = qfm_surf.surface

        # Run poincare plotting
        with ScratchDir("."):
           run_Poincare_plots(s_plot, bs, b_dipole, None, 'poincare_test')

if __name__ == "__main__":
    unittest.main()
