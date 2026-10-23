import unittest
import warnings
from pathlib import Path

import numpy as np

import simsoptpp as sopp
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.boozermagneticfield import BoozerMagneticField, InterpolatedBoozerField
from simsopt.field.coil import Coil, Current
from simsopt.field.wireframefield import WireframeField
from simsopt.geo import CurveXYZFourier, SurfaceRZFourier, ToroidalWireframe

FIELD_REF_QUANTITIES = ("B", "dB_by_dX", "d2B_by_dXdX", "AbsB", "GradAbsB",
                        "B_cyl", "A_cyl", "A", "dA_by_dX", "d2A_by_dXdX")

BOOZER_REF_QUANTITIES = (
    "dKdtheta", "dKdzeta", "K_derivs", "K",
    "nu", "dnudtheta", "dnudzeta", "dnuds", "nu_derivs",
    "R", "Z", "dRdtheta", "dRdzeta", "dRds", "R_derivs",
    "dZdtheta", "dZdzeta", "dZds", "Z_derivs",
    "modB", "dmodBdtheta", "dmodBdzeta", "dmodBds", "modB_derivs",
    "d2modBdtheta2", "d2modBdzeta2", "d2modBdthetadzeta",
    "G", "I", "psip", "iota", "dGds", "dIds", "diotads",
)

FIELDCACHE_HOOKS = ("fieldcache_get_or_create", "fieldcache_get_status")

# Cartesian points in general position: inside the make_biotsavart coil and
# the make_wireframe_field torus hole, away from the wires, and off the z axis
# and the z = 0 plane, so that the field quantities have no structural zeros
# and different caches hold different values. Otherwise the values are arbitrary.
FIELD_POINTS = np.array([[0.1, 0.2, 0.3], [-0.2, 0.1, 0.4]])
FIELD_CACHE_DIMS = [len(FIELD_POINTS), 3]

# Boozer coordinates (s, theta, zeta). Only the number of points matters,
# because ConstantBoozerField ignores the values; s stays inside (0, 1), away
# from the magnetic axis at s = 0, so the points would also suit a real field.
BOOZER_POINTS = np.array([[0.5, 0.0, 0.0], [0.6, 0.0, 0.0]])


def make_biotsavart():
    # The quadrature count is arbitrary but must be at least 4: at least
    # 2*order + 1 = 3 to sample the circle, and different from the 3 spatial
    # components and len(FIELD_POINTS). The coarse coil must also keep the field
    # quantities nonzero and distinct; accuracy does not matter.
    curve = CurveXYZFourier(50, 1)
    curve.set("xc(1)", 1.0)
    curve.set("ys(1)", 1.0)
    field = BiotSavart([Coil(curve, Current(1e4))])
    field.set_points(FIELD_POINTS)
    return field


def make_wireframe_field():
    surf = SurfaceRZFourier(nfp=1, mpol=1, ntor=0)
    surf.set_rc(0, 0, 2.0)
    surf.set_rc(1, 0, 1.0)
    surf.set_zs(1, 0, 1.0)
    field = WireframeField(ToroidalWireframe(surf, 2, 4))
    field.set_points(FIELD_POINTS)
    return field


def _constant_impl(value):
    def impl(self, out):
        out[:] = value
    return impl


# Fills every Boozer quantity with its own constant (its 1-based index in
# BOOZER_REF_QUANTITIES), so a value identifies which cache produced it.
ConstantBoozerField = type(
    "ConstantBoozerField",
    (BoozerMagneticField,),
    {f"_{quantity}_impl": _constant_impl(float(index))
     for index, quantity in enumerate(BOOZER_REF_QUANTITIES, start=1)},
)


class DeprecatedRefAliasTests(unittest.TestCase):
    """
    The former public cache-reference names remain as deprecated aliases
    of the private underscored methods.
    """

    def call_alias(self, obj, name, *args):
        """Call ``obj.<name>(*args)``, check its DeprecationWarning, and return the result and message."""
        with self.assertWarns(DeprecationWarning) as caught:
            result = getattr(obj, name)(*args)
        self.assertEqual(Path(caught.filename).resolve(), Path(__file__).resolve(),
                         f"{name}: warning is not attributed to the calling line")
        return result, str(caught.warning)

    def test_old_public_names_exist(self):
        for cls in (sopp.MagneticField, BiotSavart, WireframeField):
            for quantity in FIELD_REF_QUANTITIES:
                self.assertTrue(callable(getattr(cls, f"{quantity}_ref", None)),
                                f"{cls.__name__}.{quantity}_ref is missing")
        for cls in (BiotSavart, WireframeField):
            for hook in FIELDCACHE_HOOKS:
                self.assertTrue(callable(getattr(cls, hook, None)), f"{cls.__name__}.{hook} is missing")
        for cls in (sopp.BoozerMagneticField, InterpolatedBoozerField):
            for quantity in BOOZER_REF_QUANTITIES:
                self.assertTrue(callable(getattr(cls, f"{quantity}_ref", None)),
                                f"{cls.__name__}.{quantity}_ref is missing")

    def test_field_ref_aliases_return_private_live_views(self):
        field = make_biotsavart()
        for quantity in FIELD_REF_QUANTITIES:
            alias = f"{quantity}_ref"
            view, message = self.call_alias(field, alias)
            self.assertIn(f"please use {quantity}()", message, f"{alias}: message does not name the replacement")
            self.assertTrue(np.shares_memory(view, getattr(field, f"_{alias}")()),
                            f"{alias} is not the live array returned by _{alias}")
            np.testing.assert_array_equal(view, getattr(field, quantity)(),
                                          err_msg=f"{alias} disagrees with {quantity}()")

    def test_boozer_ref_aliases_return_private_live_views(self):
        field = ConstantBoozerField(1.0)
        field.set_points(BOOZER_POINTS)
        for index, quantity in enumerate(BOOZER_REF_QUANTITIES, start=1):
            alias = f"{quantity}_ref"
            view, message = self.call_alias(field, alias)
            self.assertIn(f"please use {quantity}()", message, f"{alias}: message does not name the replacement")
            self.assertTrue(np.shares_memory(view, getattr(field, f"_{alias}")()),
                            f"{alias} is not the live array returned by _{alias}")
            np.testing.assert_array_equal(view, np.full_like(view, index),
                                          err_msg=f"{alias} returned another quantity's cache")

    def check_fieldcache_aliases(self, field, key, dims, accessor):
        status, message = self.call_alias(field, "fieldcache_get_status", key)
        self.assertIn(accessor, message, "fieldcache_get_status: message does not name the public accessor")
        self.assertTrue(status, f"fieldcache_get_status({key!r}) should report the populated cache entry")
        missing, _ = self.call_alias(field, "fieldcache_get_status", "no_such_key")
        self.assertFalse(missing, "fieldcache_get_status should report a missing cache entry")

        buffer, message = self.call_alias(field, "fieldcache_get_or_create", key, dims)
        self.assertIn(accessor, message, "fieldcache_get_or_create: message does not name the public accessor")
        self.assertTrue(np.shares_memory(buffer, field._fieldcache_get_or_create(key, dims)),
                        "fieldcache_get_or_create is not the live cache entry returned by _fieldcache_get_or_create")

    def test_biotsavart_fieldcache_aliases(self):
        field = make_biotsavart()
        field.dB_by_dcoilcurrents()
        self.check_fieldcache_aliases(field, "B_0", FIELD_CACHE_DIMS, "dB_by_dcoilcurrents()")

    def test_wireframe_fieldcache_aliases(self):
        field = make_wireframe_field()
        field.dB_by_dsegmentcurrents(0)
        self.check_fieldcache_aliases(field, "B_0", FIELD_CACHE_DIMS, "dB_by_dsegmentcurrents()")

    def test_private_names_and_copying_accessors_do_not_warn(self):
        field = make_biotsavart()
        booz = ConstantBoozerField(1.0)
        booz.set_points(BOOZER_POINTS)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            for quantity in FIELD_REF_QUANTITIES:
                getattr(field, f"_{quantity}_ref")()
                getattr(field, quantity)()
            for quantity in BOOZER_REF_QUANTITIES:
                getattr(booz, f"_{quantity}_ref")()
                getattr(booz, quantity)()
            field.dB_by_dcoilcurrents()
            field._fieldcache_get_status("B_0")
            field._fieldcache_get_or_create("B_0", FIELD_CACHE_DIMS)

    def test_warning_escalated_to_error_propagates(self):
        field = make_biotsavart()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with self.assertRaises(DeprecationWarning):
                field.B_ref()


if __name__ == "__main__":
    unittest.main()
