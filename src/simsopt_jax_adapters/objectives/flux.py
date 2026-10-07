"""JAX-backed squared-flux objective for Stage-II coil optimization."""

from __future__ import annotations

import hashlib

import jax
import numpy as np
from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt_jax.core.field import (
    grouped_biot_savart_B_from_spec,
    grouped_coil_set_spec_from_inputs,
)
from simsopt_jax.core.integral_bdotn import (
    FLUX_DEFINITIONS,
    fixed_surface_flux_integral_from_B,
)
from simsopt_jax.core.specs import FixedSurfaceFluxSpec, make_fixed_surface_flux_spec
from simsopt_jax.runtime.host_boundary import host_array

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

__all__ = ["SquaredFluxJAX"]


def _surface_dofs_fingerprint(surface) -> bytes:
    """Return a fixed-width digest of a surface's full DOF vector, fixed DOFs included."""
    dofs = np.ascontiguousarray(surface.local_full_x, dtype=np.float64)
    return hashlib.blake2b(dofs.tobytes(), digest_size=16).digest()


def _squared_flux_from_coil_arrays(flux_spec: FixedSurfaceFluxSpec, coil_arrays):
    B = grouped_biot_savart_B_from_spec(
        flux_spec.points,
        grouped_coil_set_spec_from_inputs(coil_arrays),
    )
    return fixed_surface_flux_integral_from_B(B, flux_spec)


_squared_flux_value = jax.jit(_squared_flux_from_coil_arrays)
_squared_flux_value_and_coil_cotangents = jax.jit(
    jax.value_and_grad(_squared_flux_from_coil_arrays, argnums=1)
)


class SquaredFluxJAX(Optimizable):
    r"""JAX-backed mirror of :class:`~simsopt.objectives.SquaredFlux`.

    Same definitions, constructor and ``Derivative`` as the native objective
    for a :class:`BiotSavartJAX` field. The flux integral and its gradient with
    respect to the coil geometry and currents run as one jitted program; the
    field projects that gradient onto the coil DOFs. Like the native objective,
    construction sets the field's evaluation points to the surface points.

    As in the native objective, ``target`` and ``definition`` are plain
    attributes read at every evaluation. The surface must stay fixed: its
    points and normals are captured at construction, and evaluating after its
    DOFs change raises ``RuntimeError``. :meth:`fixed_surface_flux_spec`
    returns the current contract for the fused Stage-II objective.

    Args:
        surface: the fixed :class:`~simsopt.geo.Surface`.
        field: a :class:`BiotSavartJAX` field.
        target: optional ``(nphi, ntheta)`` target normal field (default 0).
        definition: ``"quadratic flux"``, ``"normalized"`` or ``"local"``.
    """

    def __init__(self, surface, field: BiotSavartJAX, target=None, definition="quadratic flux"):
        if definition not in FLUX_DEFINITIONS:
            raise ValueError("Unrecognized option for 'definition'.")
        self.surface = surface
        self.field = field
        self.definition = definition
        points = np.ascontiguousarray(surface.gamma().reshape((-1, 3)))
        normal = np.ascontiguousarray(surface.normal())
        self.target = (
            np.zeros(normal.shape[:2]) if target is None else np.ascontiguousarray(target)
        )
        surface_geometry = make_fixed_surface_flux_spec(
            points=points,
            normal=normal,
            target=self.target,
            definition=definition,
        )
        self._surface_points = surface_geometry.points
        self._surface_normal = surface_geometry.normal
        self._surface_dofs_fingerprint = _surface_dofs_fingerprint(surface)
        field.set_points(points)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def _raise_if_surface_changed(self) -> None:
        if _surface_dofs_fingerprint(self.surface) != self._surface_dofs_fingerprint:
            raise RuntimeError(
                "SquaredFluxJAX captures the surface geometry at construction and "
                "the surface DOFs have changed since; rebuild the objective."
            )

    def fixed_surface_flux_spec(self) -> FixedSurfaceFluxSpec:
        """Return the immutable contract of the captured surface and the current
        ``target`` and ``definition``."""
        self._raise_if_surface_changed()
        return make_fixed_surface_flux_spec(
            points=self._surface_points,
            normal=self._surface_normal,
            target=self.target,
            definition=self.definition,
        )

    def J(self):
        value = _squared_flux_value(
            self.fixed_surface_flux_spec(),
            self.field.coil_set_spec().field_inputs(),
        )
        return float(host_array(value, dtype=np.float64))

    @derivative_dec
    def dJ(self):
        flux_spec = self.fixed_surface_flux_spec()
        coil_set_spec = self.field.coil_set_spec()
        _, coil_cotangents = _squared_flux_value_and_coil_cotangents(
            flux_spec,
            coil_set_spec.field_inputs(),
        )
        return self.field.coil_cotangents_to_derivative(
            coil_cotangents,
            coil_set_spec.coil_index_lists(),
        )

    return_fn_map = {"J": J, "dJ": dJ}
