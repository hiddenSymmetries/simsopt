"""
Parity tests for the JAX Biot-Savart implementation.

Validates against:
1. Analytical on-axis field of a circular current loop.
2. Maxwell's equation ∇·B = 0 (trace of dB/dX).
3. C++ reference from the installed simsoptpp extension.
"""


from jax_test_support import (
    fixture_jax_runtime_guard,  # noqa: F401
)


import numpy as np
from typing import cast
from simsopt._core.derivative import Derivative


import jax


import jax.numpy as jnp


from simsopt.configs import get_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.field.coil import Coil, Current
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt_jax_adapters.field.biotsavart_backend import (
    BiotSavartJAX,
)
from simsopt_jax_adapters.field.biotsavart_backend import (
    _per_coil_unit_field,
)




from simsopt_jax.core.specs import CoilGroupSpec, GroupedCoilSetSpec


from simsopt_jax.core import biotsavart as core_biotsavart


biot_savart_B = core_biotsavart.biot_savart_B


biot_savart_dB_by_dX = core_biotsavart.biot_savart_dB_by_dX


biot_savart_A = core_biotsavart.biot_savart_A


biot_savart_dA_by_dX = core_biotsavart.biot_savart_dA_by_dX


_DIRECT_KERNEL_TOLS = {'rtol': 1e-10, 'atol': 1e-12, 'requires_same_state': True, 'requires_direct_cpp_oracle': True, 'vector_parity_required': True}


_DERIVATIVE_HEAVY_TOLS = {'scalar_value_rtol': 1e-10, 'scalar_value_atol': 1e-12, 'first_derivative_rtol': 1e-08, 'first_derivative_atol': 1e-10, 'second_derivative_rtol': 1e-06, 'second_derivative_atol': 1e-08, 'requires_same_input': True, 'requires_direct_cpp_oracle': True, 'fd_validation_secondary': True}


def _ncsx_biotsavart_parity_fixture():

    curves, currents_objs, _, nfp, _ = get_data("ncsx")
    coils = coils_via_symmetries(curves, currents_objs, nfp, stellsym=True)
    bs = BiotSavart(coils)

    npoints = 50
    rng = np.random.default_rng(42)
    points_np = rng.standard_normal((npoints, 3)) * 0.3
    points_np[:, 0] += 1.0  # shift near torus

    bs.set_points(points_np)
    gammas_np = np.array([coil.curve.gamma() for coil in coils])
    gds_np = np.array([coil.curve.gammadash() for coil in coils])
    currents_np = np.array([coil.current.get_value() for coil in coils])
    return bs, points_np, gammas_np, gds_np, currents_np


def _cart_points_to_cyl(points):
    return np.ascontiguousarray(
        np.stack(
            (
                np.sqrt(points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]),
                np.arctan2(points[:, 1], points[:, 0]),
                points[:, 2],
            ),
            axis=1,
        )
    )


def _assert_cylindrical_points_match_cpu(jax_field, cpu_field):
    np.testing.assert_allclose(
        np.asarray(jax_field.get_points_cyl()),
        np.asarray(cpu_field.get_points_cyl()),
        rtol=0.0,
        atol=1.0e-15,
    )


def _assert_cylindrical_accessors_match_cpu(jax_field, cpu_field):
    _assert_cylindrical_points_match_cpu(jax_field, cpu_field)
    np.testing.assert_allclose(
        np.asarray(jax_field.AbsB()),
        np.asarray(cpu_field.AbsB()),
        rtol=_DIRECT_KERNEL_TOLS["rtol"],
        atol=_DIRECT_KERNEL_TOLS["atol"],
    )
    assert np.asarray(jax_field.AbsB()).shape == np.asarray(cpu_field.AbsB()).shape
    for method_name in ("B_cyl", "A_cyl"):
        np.testing.assert_allclose(
            np.asarray(getattr(jax_field, method_name)()),
            np.asarray(getattr(cpu_field, method_name)()),
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )
    np.testing.assert_allclose(
        np.asarray(jax_field.GradAbsB_cyl()),
        np.asarray(cpu_field.GradAbsB_cyl()),
        rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
    )


class TestBiotSavartJaxCppParity:
    """Compare against the installed C++ simsoptpp kernel."""


    def test_B_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        B_ref = bs.B()

        B_jax = biot_savart_B(
            jnp.array(points_np),
            jnp.array(gammas_np),
            jnp.array(gds_np),
            jnp.array(currents_np),
        )

        np.testing.assert_allclose(
            np.array(B_jax),
            B_ref,
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_A_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        A_ref = bs.A()

        A_jax = biot_savart_A(
            jnp.array(points_np),
            jnp.array(gammas_np),
            jnp.array(gds_np),
            jnp.array(currents_np),
        )

        np.testing.assert_allclose(
            np.array(A_jax),
            A_ref,
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_dB_by_dX_parity_ncsx(self):
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        dB_ref = bs.dB_by_dX()

        dB_jax = biot_savart_dB_by_dX(
            jnp.array(points_np),
            jnp.array(gammas_np),
            jnp.array(gds_np),
            jnp.array(currents_np),
        )

        np.testing.assert_allclose(
            np.array(dB_jax),
            dB_ref,
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_dA_by_dX_kernel_parity_ncsx(self):
        """``biot_savart_dA_by_dX`` matches ``BiotSavart.dA_by_dX()``."""
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        dA_ref = bs.dA_by_dX()

        dA_jax = biot_savart_dA_by_dX(
            jnp.array(points_np),
            jnp.array(gammas_np),
            jnp.array(gds_np),
            jnp.array(currents_np),
        )

        np.testing.assert_allclose(
            np.array(dA_jax),
            dA_ref,
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_cylindrical_public_accessors_parity_ncsx(self):
        """``BiotSavartJAX`` owns cylindrical public accessors on its boundary."""

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cyl = _cart_points_to_cyl(points_np)
        points_cyl[:, 1] += 2.0 * np.pi
        bs.set_points_cyl(points_cyl)

        bs_jax = BiotSavartJAX(list(bs._coils))
        assert bs_jax.set_points_cyl(points_cyl) is bs_jax
        assert bs_jax.set_points_cart(points_np) is bs_jax
        assert bs_jax.set_points_cyl(points_cyl) is bs_jax

        _assert_cylindrical_accessors_match_cpu(bs_jax, bs)

    def test_cylindrical_public_accessors_use_cached_phi_basis_ncsx(self):
        """``B_cyl`` / ``A_cyl`` / ``GradAbsB_cyl`` use cached cylindrical phi."""

        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cyl = np.ascontiguousarray(
            np.array(
                [
                    [0.0, 1.234, 0.1],
                    [0.0, -1.5, -0.2],
                    [0.7, 2.0 * np.pi + 0.4, 0.3],
                ],
                dtype=np.float64,
            )
        )
        bs.set_points_cyl(points_cyl)

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points_cyl(points_cyl)

        _assert_cylindrical_accessors_match_cpu(bs_jax, bs)

    def test_cartesian_public_accessors_normalize_cylindrical_phi_ncsx(self):
        """Cartesian point conversion follows C++ ``get_points_cyl_impl``."""

        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cart = np.ascontiguousarray(
            np.array(
                [
                    [0.5, -0.5, 0.1],
                    [-0.5, -0.25, -0.2],
                    [0.0, 0.0, 0.3],
                ],
                dtype=np.float64,
            )
        )
        bs.set_points_cart(points_cart)

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points_cart(points_cart)

        _assert_cylindrical_points_match_cpu(bs_jax, bs)


    def test_B_vjp_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""

        bs_cpu, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        coils = list(bs_cpu._coils)

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points_np)

        v = np.asarray(bs_cpu.B(), dtype=np.float64).copy()
        deriv_cpu = cast(Derivative, bs_cpu.B_vjp(v))
        deriv_jax = bs_jax.B_vjp(v)

        for coil in coils:
            np.testing.assert_allclose(
                np.asarray(deriv_jax(coil)),
                np.asarray(deriv_cpu(coil)),
                rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
                atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
                err_msg=(
                    "BiotSavartJAX.B_vjp() does not match BiotSavart.B_vjp() "
                    "on the NCSX parity fixture"
                ),
            )

    def test_dA_by_dX_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        dA_ref = bs.dA_by_dX()

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points(points_np)
        dA_jax = bs_jax.dA_by_dX()

        np.testing.assert_allclose(
            np.array(dA_jax),
            dA_ref,
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_d2B_by_dXdX_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        d2B_ref = bs.d2B_by_dXdX()

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points(points_np)
        d2B_jax = bs_jax.d2B_by_dXdX()

        np.testing.assert_allclose(
            np.array(d2B_jax),
            d2B_ref,
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )

    def test_d2A_by_dXdX_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        d2A_ref = bs.d2A_by_dXdX()

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points(points_np)
        d2A_jax = bs_jax.d2A_by_dXdX()

        np.testing.assert_allclose(
            np.array(d2A_jax),
            d2A_ref,
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )

class TestBiotSavartJaxCppCoilCurrentParity:
    """Per-coil field and spatial derivative parity against native Biot-Savart."""


    @staticmethod
    def _assert_coil_current_list_parity(cache_method, list_method, *, rtol, atol):

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        # Populate the matching C++ fieldcache entries before pulling the
        # per-coil list, so ordering is deterministic.
        getattr(bs, cache_method)()
        cpu_list = getattr(bs, list_method)()

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points(points_np)
        jax_list = getattr(bs_jax, list_method)()

        assert len(jax_list) == len(cpu_list)
        for k, (j_entry, c_entry) in enumerate(zip(jax_list, cpu_list)):
            np.testing.assert_allclose(
                np.array(j_entry),
                c_entry,
                rtol=rtol,
                atol=atol,
                err_msg=f"coil {k}",
            )

    def test_dB_by_dcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "B",
            "dB_by_dcoilcurrents",
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_per_coil_unit_field_vectorizes_within_quadrature_group(self):

        points = jnp.asarray([[0.0, 0.0, 0.0], [0.25, -0.5, 1.0]], dtype=jnp.float64)
        group0_gammas = jnp.arange(18, dtype=jnp.float64).reshape(2, 3, 3)
        group0_gammadashs = group0_gammas + 0.5
        group1_gammas = jnp.arange(9, dtype=jnp.float64).reshape(1, 3, 3) - 2.0
        group1_gammadashs = group1_gammas - 0.25
        coil_set_spec = GroupedCoilSetSpec(
            groups=(
                CoilGroupSpec(
                    gammas=group0_gammas,
                    gammadashs=group0_gammadashs,
                    currents=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
                    coil_indices=(2, 0),
                ),
                CoilGroupSpec(
                    gammas=group1_gammas,
                    gammadashs=group1_gammadashs,
                    currents=jnp.asarray([5.0], dtype=jnp.float64),
                    coil_indices=(1,),
                ),
            )
        )
        calls = []

        def kernel(kernel_points, gammas, gammadashs, currents):
            calls.append(gammas.shape)
            value = jnp.sum(gammas) + jnp.sum(gammadashs) + jnp.sum(currents)
            return jnp.broadcast_to(value, kernel_points.shape)

        results = _per_coil_unit_field(points, coil_set_spec, kernel)

        assert len(calls) == 2
        assert calls == [(1, 3, 3), (1, 3, 3)]
        assert len(results) == 3
        np.testing.assert_allclose(
            np.asarray(results[0]),
            np.broadcast_to(
                np.sum(np.asarray(group0_gammas[1]))
                + np.sum(np.asarray(group0_gammadashs[1]))
                + 1.0,
                points.shape,
            ),
        )
        np.testing.assert_allclose(
            np.asarray(results[1]),
            np.broadcast_to(
                np.sum(np.asarray(group1_gammas[0]))
                + np.sum(np.asarray(group1_gammadashs[0]))
                + 1.0,
                points.shape,
            ),
        )
        np.testing.assert_allclose(
            np.asarray(results[2]),
            np.broadcast_to(
                np.sum(np.asarray(group0_gammas[0]))
                + np.sum(np.asarray(group0_gammadashs[0]))
                + 1.0,
                points.shape,
            ),
        )

    def test_dA_by_dcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "A",
            "dA_by_dcoilcurrents",
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_d2B_by_dXdcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "dB_by_dX",
            "d2B_by_dXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_d2A_by_dXdcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "dA_by_dX",
            "d2A_by_dXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_d3B_by_dXdXdcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "d2B_by_dXdX",
            "d3B_by_dXdXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )

    def test_d3A_by_dXdXdcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "d2A_by_dXdX",
            "d3A_by_dXdXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )


class TestBiotSavartJAXCoilStateToken:
    """Traceable runtime cache invalidation is keyed to coil DOF state."""

    @staticmethod
    def _make_two_basic_coils():

        coils = []
        for current_amp in (1.0e6, -1.0e6):
            curve = CurveXYZFourier(quadpoints=16, order=1)
            curve.x = np.array(
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
            coils.append(Coil(curve, Current(current_amp)))
        return coils

    @staticmethod
    def _coil_arrays_in_original_order(coil_set_spec):
        coil_count = sum(len(group.coil_indices) for group in coil_set_spec.groups)
        gammas = [None] * coil_count
        gammadashs = [None] * coil_count
        currents = [None] * coil_count
        for group in coil_set_spec.groups:
            for position, coil_index in enumerate(group.coil_indices):
                gammas[coil_index] = group.gammas[position]
                gammadashs[coil_index] = group.gammadashs[position]
                currents[coil_index] = group.currents[position]
        return gammas, gammadashs, currents

    @staticmethod
    def _assert_per_coil_entries_equal(actual_entries, expected_entries):
        assert len(actual_entries) == len(expected_entries)
        for actual, expected in zip(actual_entries, expected_entries, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=0.0,
                atol=0.0,
            )


    def test_live_coil_set_spec_matches_explicit_dofs_and_native_coils(self):
        """The cached live spec is the explicit-DOF reconstruction of the native coils."""
        coils = self._make_two_basic_coils()
        bs_jax = BiotSavartJAX(coils)

        live = self._coil_arrays_in_original_order(bs_jax.coil_set_spec())
        explicit = self._coil_arrays_in_original_order(
            bs_jax.coil_set_spec_from_dofs(bs_jax.x)
        )
        native = (
            [coil.curve.gamma() for coil in coils],
            [coil.curve.gammadash() for coil in coils],
            [coil.current.get_value() for coil in coils],
        )

        for live_entries, explicit_entries, native_entries in zip(
            live, explicit, native, strict=True
        ):
            for actual, from_dofs, expected in zip(
                live_entries, explicit_entries, native_entries, strict=True
            ):
                np.testing.assert_allclose(
                    cast(jax.Array, actual), cast(jax.Array, from_dofs),
                    rtol=1e-14, atol=1e-15,
                )
                np.testing.assert_allclose(
                    cast(jax.Array, actual), expected, rtol=1e-13, atol=1e-14,
                )

    def test_biotsavart_jax_advances_coil_dof_state_token_on_x_update(self):

        coils = self._make_two_basic_coils()
        bs_jax = BiotSavartJAX(list(coils))
        initial_token = bs_jax._coil_dof_state_token

        bs_jax.x = np.asarray(bs_jax.x, dtype=np.float64)

        assert bs_jax._coil_dof_state_token != initial_token
        assert bs_jax._coil_dofs_generation == 1

        next_token = bs_jax._coil_dof_state_token
        bs_jax.full_x = np.asarray(bs_jax.full_x, dtype=np.float64)

        assert bs_jax._coil_dof_state_token != next_token
        assert bs_jax._coil_dofs_generation == 2

    def test_biotsavart_jax_advances_coil_dof_state_token_on_parent_update(self):

        coils = self._make_two_basic_coils()
        bs_jax = BiotSavartJAX(list(coils))
        initial_token = bs_jax._coil_dof_state_token
        curve_dofs = np.asarray(coils[0].curve.x, dtype=np.float64)
        curve_dofs[0] += 1.0e-4

        coils[0].curve.x = curve_dofs

        assert bs_jax._coil_dof_state_token != initial_token
        assert bs_jax._coil_dofs_generation == 1


    def test_biotsavart_extraction_spec_changes_only_for_captured_dof_contract(self):

        coils = self._make_two_basic_coils()
        field = BiotSavartJAX(list(coils))
        curve = coils[0].curve
        initial_spec = field.coil_dof_extraction_spec()
        initial_value = float(curve.local_full_x[0])

        curve.set(0, initial_value + 1.0e-3)
        assert field.coil_dof_extraction_spec() is initial_spec

        curve.fix(0)
        fixed_spec = field.coil_dof_extraction_spec()
        assert fixed_spec is not initial_spec

        curve.set(0, initial_value + 2.0e-3)
        assert field.coil_dof_extraction_spec() is not fixed_spec
