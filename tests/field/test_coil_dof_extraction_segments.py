"""Captured coil DOF maps preserve segmented dependencies and exact input order."""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    import jax.numpy as jnp
    from simsopt_jax.core import coil_specs_from_dof_extraction_spec
    from simsopt_jax_adapters.field.biotsavart_backend import (
        JaxBiotSavart,
        _dof_map_cotangent_to_owner_gradient,
        _owner_segments_from_free_positions,
    )


import numpy as np

from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveXYZFourier

_NQUAD = 8
_ORDER = 1
_SPLIT_CURVE_FIXED = ("yc(0)", "zc(1)")


def _bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def coil_graph():
    """Two coils: the first curve has two interior dofs fixed, the second current is fixed."""
    curves = [CurveXYZFourier(_NQUAD, _ORDER) for _ in range(2)]
    for index, curve in enumerate(curves):
        curve.x = (
            0.1 * (index + 1) * np.arange(1, int(curve.dof_size) + 1, dtype=np.float64)
        )
    for name in _SPLIT_CURVE_FIXED:
        curves[0].fix(name)
    currents = [Current(1.0e4), Current(2.0e4)]
    currents[1].fix_all()
    coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
    bs_jax = JaxBiotSavart(coils)
    oracle = BiotSavart(coils)
    assert list(oracle.dof_names) == list(bs_jax.dof_names)
    return curves, currents, bs_jax, oracle


def _owner_start(bs_jax, opt) -> int:
    return list(bs_jax.dof_names).index(f"{opt.name}:{opt.local_dof_names[0]}")


def _lowered_extraction_text(bs_jax) -> str:
    spec = bs_jax.coil_dof_extraction_spec()
    free_dofs = jnp.asarray(np.linspace(-1.0, 1.0, bs_jax.dof_size))
    lowered = jax.jit(lambda dofs: coil_specs_from_dof_extraction_spec(spec, dofs))
    return lowered.lower(free_dofs).as_text()


class TestCoilDofExtractionSegments(JaxTestCase):
    def setUp(self):
        super().setUp()
        self.coil_graph = coil_graph()
        self.addCleanup(delattr, self, "coil_graph")

    def test_segments_from_free_positions_merge_runs_and_split_at_fixed_dofs(self):
        """Extraction segments merge contiguous free positions and split correctly at
        fixed DOFs, including empty input."""
        self.assertTrue(
            _owner_segments_from_free_positions(5, [0, 1, 2]) == ((5, 8, 0, 3),),
            "_owner_segments_from_free_positions(5, [0, 1, 2]) == ((5, 8, 0, 3),)",
        )
        self.assertTrue(
            _owner_segments_from_free_positions(2, [0, 1, 2, 4, 5, 6, 7])
            == (
                (2, 5, 0, 3),
                (5, 9, 4, 8),
            ),
            "_owner_segments_from_free_positions(2, [0, 1, 2, 4, 5, 6, 7]) == ( (2, 5, 0, 3), (5, 9, 4, 8), )",
        )
        self.assertTrue(
            _owner_segments_from_free_positions(0, [1, 3, 5])
            == (
                (0, 1, 1, 2),
                (1, 2, 3, 4),
                (2, 3, 5, 6),
            ),
            "_owner_segments_from_free_positions(0, [1, 3, 5]) == ( (0, 1, 1, 2), (1, 2, 3, 4), (2, 3, 5, 6), )",
        )
        self.assertTrue(
            _owner_segments_from_free_positions(7, []) == (),
            "_owner_segments_from_free_positions(7, []) == ()",
        )

    def test_extraction_spec_carries_one_segment_per_run(self):
        """Curve and current extraction maps contain one copy segment per contiguous
        free-DOF run."""
        coil_graph = self.coil_graph
        curves, currents, bs_jax, _oracle = coil_graph
        spec = bs_jax.coil_dof_extraction_spec()

        split_start = _owner_start(bs_jax, curves[0])
        self.assertTrue(
            spec.coils[0].curve_map.owner_segments
            == (
                (split_start, split_start + 3, 0, 3),
                (split_start + 3, split_start + 7, 4, 8),
            ),
            "spec.coils[0].curve_map.owner_segments == ( (split_start, split_start + 3, 0, 3), (split_start + 3, split_start + 7, 4, 8), )",
        )
        intact_start = _owner_start(bs_jax, curves[1])
        self.assertTrue(
            spec.coils[1].curve_map.owner_segments
            == ((intact_start, intact_start + 9, 0, 9),),
            "spec.coils[1].curve_map.owner_segments == ( (intact_start, intact_start + 9, 0, 9), )",
        )
        current_start = _owner_start(bs_jax, currents[0])
        self.assertTrue(
            spec.coils[0].current_map.owner_segments
            == ((current_start, current_start + 1, 0, 1),),
            "spec.coils[0].current_map.owner_segments == ( (current_start, current_start + 1, 0, 1), )",
        )
        self.assertTrue(
            spec.coils[1].current_map.owner_segments == (),
            "spec.coils[1].current_map.owner_segments == ()",
        )

    def test_coil_specs_from_dofs_match_the_optimizable_setter_bitwise(self):
        """Compiled reconstruction matches native DOF setters bitwise under the strict
        transfer guard."""
        coil_graph = self.coil_graph
        curves, currents, bs_jax, oracle = coil_graph
        spec = bs_jax.coil_dof_extraction_spec()
        rng = np.random.default_rng(20260914)
        free_dofs = rng.normal(size=bs_jax.dof_size)

        oracle.x = free_dofs
        expected_curve_dofs = [
            np.asarray(curve.local_full_x).copy() for curve in curves
        ]
        expected_currents = [float(current.get_value()) for current in currents]

        compiled = (
            jax.jit(lambda dofs: coil_specs_from_dof_extraction_spec(spec, dofs))
            .lower(jnp.asarray(free_dofs))
            .compile()
        )
        with jax.transfer_guard("disallow"):
            coil_specs = compiled(jnp.asarray(free_dofs))

        for coil_spec, expected_dofs, expected_current in zip(
            coil_specs, expected_curve_dofs, expected_currents
        ):
            self.assertTrue(
                np.array_equal(_bits(coil_spec.curve.dofs), _bits(expected_dofs)),
                "np.array_equal(_bits(coil_spec.curve.dofs), _bits(expected_dofs))",
            )
            self.assertTrue(
                np.array_equal(
                    _bits(jnp.atleast_1d(coil_spec.current.value)),
                    _bits(np.asarray([expected_current])),
                ),
                "np.array_equal( _bits(jnp.atleast_1d(coil_spec.current.value)), _bits(np.asarray([expected_current])), )",
            )

    def test_fully_free_extraction_lowers_without_selector_matmuls(self):
        """Fully free DOF reconstruction lowers without selector matrix multiplications."""
        curves = [CurveXYZFourier(_NQUAD, _ORDER) for _ in range(2)]
        currents = [Current(1.0e4), Current(2.0e4)]
        bs_jax = JaxBiotSavart([Coil(c, i) for c, i in zip(curves, currents)])

        text = _lowered_extraction_text(bs_jax)

        self.assertTrue(
            "stablehlo.dot_general" not in text, '"stablehlo.dot_general" not in text'
        )

    def test_partially_fixed_extraction_lowers_one_placement_matmul_per_run(self):
        """Only the split curve's two runs need a placement; free maps are slices."""
        coil_graph = self.coil_graph
        _curves, _currents, bs_jax, _oracle = coil_graph

        text = _lowered_extraction_text(bs_jax)

        self.assertTrue(
            text.count("stablehlo.dot_general") == 2,
            'text.count("stablehlo.dot_general") == 2',
        )

    def test_gradient_through_the_mapping_matches_the_manual_cotangent(self):
        """Autodiff and manual extraction-map cotangents match the native setter-derived
        selection gradient bitwise."""
        coil_graph = self.coil_graph
        curves, _currents, bs_jax, oracle = coil_graph
        spec = bs_jax.coil_dof_extraction_spec()
        rng = np.random.default_rng(7)
        free_dofs = rng.normal(size=bs_jax.dof_size)
        weights = [rng.normal(size=curve.local_full_dof_size) for curve in curves]

        def weighted_curve_dofs(dofs):
            coil_specs = coil_specs_from_dof_extraction_spec(spec, dofs)
            return sum(
                jnp.dot(jnp.asarray(weight), coil_spec.curve.dofs)
                for weight, coil_spec in zip(weights, coil_specs)
            )

        autodiff_gradient = np.asarray(
            jax.grad(weighted_curve_dofs)(jnp.asarray(free_dofs))
        )

        manual_gradient = np.zeros(bs_jax.dof_size)
        for coil_spec, weight in zip(spec.coils, weights):
            manual_gradient += np.asarray(
                _dof_map_cotangent_to_owner_gradient(
                    coil_spec.curve_map,
                    jnp.asarray(weight),
                    jnp.asarray(free_dofs),
                )
            )

        selection = np.zeros((bs_jax.dof_size, sum(w.size for w in weights)))
        oracle.x = np.zeros(bs_jax.dof_size)
        baseline = np.concatenate([np.asarray(curve.local_full_x) for curve in curves])
        for column in range(bs_jax.dof_size):
            probe = np.zeros(bs_jax.dof_size)
            probe[column] = 1.0
            oracle.x = probe
            selection[column] = (
                np.concatenate([np.asarray(curve.local_full_x) for curve in curves])
                - baseline
            )
        expected_gradient = selection @ np.concatenate(weights)

        self.assertTrue(
            np.array_equal(_bits(autodiff_gradient), _bits(expected_gradient)),
            "np.array_equal(_bits(autodiff_gradient), _bits(expected_gradient))",
        )
        self.assertTrue(
            np.array_equal(_bits(manual_gradient), _bits(expected_gradient)),
            "np.array_equal(_bits(manual_gradient), _bits(expected_gradient))",
        )
