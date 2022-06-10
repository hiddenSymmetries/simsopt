#!/usr/bin/env python

from pathlib import Path
import numpy as np
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries, ScaledCurrent
from simsopt.geo.curve import curves_to_vtk
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

import unittest


class Testing(unittest.TestCase):

    def test_add_field_cylinder(self):
        pass

    def test_write(self):
        pass

    def test_from_file(self):
        pass

    def test_load_field(self):
        pass



