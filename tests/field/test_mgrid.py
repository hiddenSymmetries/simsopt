#!/usr/bin/env python

from pathlib import Path
import numpy as np
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries, ScaledCurrent
from simsopt.geo.curve import curves_to_vtk
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

from simsopt.field.mgrid import MGrid
from scipy.io import netcdf as nc
import unittest

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
test_file = TEST_DIR / 'mgrid.pnas-qa-test-lowres-standard.nc'


class Testing(unittest.TestCase):

    def test_from_file(self):
        mgrid = MGrid.from_file(test_file)

        assert mgrid.rmin == 0.5
        assert mgrid.nphi == 6
        assert mgrid.bvec.shape == (11, 11, 6, 3) 

    def test_load_field(self):
        f = nc.netcdf_file(test_file, 'r')
        mgrid = MGrid()
        mgrid.load_field(f)

        names = mgrid.coil_names
        assert len(names) == 1
        assert names[0] == '________simsopt_coils_________'

        assert mgrid.br_arr.shape == (1, 6, 11, 11)
        assert mgrid.br[0, 0, 0] == -1.0633399551863771  # -0.9946816978184079

    def test_add_field_cylinder(self):

        N_points = 5
        br = np.ones(N_points)
        bp = br*2
        bz = br*3
        name = "test_coil"

        mgrid = MGrid()
        mgrid.add_field_cylindrical(br, bp, bz, name=name)
        assert mgrid.n_ext_cur == 1
        assert mgrid.coil_names[0] == '__________test_coil___________'
        assert np.allclose(mgrid.br_arr[0], br)

    def test_write(self):
        mgrid = MGrid.from_file(test_file)

        filename = '/tmp/mgrid.test'
        mgrid.write(filename)

        f = nc.netcdf_file(filename)
        zmin = f.variables['zmin'][:][0]
        assert zmin == -0.5

        nextcur = f.variables['nextcur'][:][0]
        assert nextcur == 1

        assert f.variables['br_001'][:].shape == (6, 11, 11)
        assert f.variables['mgrid_mode'][:][0].decode('ascii') == 'N'

        byte_string = f.variables['coil_group'][:][0]
        message = "".join([x.decode('ascii') for x in byte_string])
        assert message == '________simsopt_coils_________'




