"""Public native flux derivatives on the objective's quadrature grid."""

from abc import abstractmethod
import unittest
from typing import Literal, Protocol, cast

import numpy as np

from simsopt._core.derivative import Derivative
from simsopt.field import Coil, Current, NetFluxes
from simsopt.geo import CurveXYZFourier, create_equally_spaced_curves


class _PartialDerivative(Protocol):
    @abstractmethod
    def __call__(self, *, partials: Literal[True]) -> Derivative:
        """Describe the partials keyword supplied by derivative_dec."""


class NetFluxesDownsamplingTests(unittest.TestCase):
    def test_gradient_matches_downsampled_objective(self):
        """Check target/source shape and source-current gradients by differences of J."""
        rng = np.random.default_rng(47)
        curves = create_equally_spaced_curves(3, 1, False, order=3, numquadpoints=12)
        for curve in curves:
            curve.x = curve.x + 0.025 * rng.standard_normal(curve.x.shape)
        sources = [CurveXYZFourier(24, 3, dofs=curve.dofs) for curve in curves[1:]]
        target_coil = Coil(curves[0], Current(2.1e4))
        source_coils = [
            Coil(curve, Current(current))
            for curve, current in zip(sources, [1.7e4, -1.3e4])
        ]
        for downsample in [1, 2, 3]:
            objective = NetFluxes(target_coil, source_coils, downsample=downsample)
            initial = np.array(objective.x)
            gradient = objective.dJ()
            self.assertGreater(abs(float(objective.J())), 1e-6)
            for dependency in [
                curves[0],
                *sources,
                *(coil.current for coil in source_coils),
            ]:
                with self.subTest(downsample=downsample, dependency=dependency.name):
                    direction = np.zeros_like(initial)
                    indices = objective.dof_indices[dependency]
                    direction[indices[0] : indices[1]] = rng.standard_normal(
                        indices[1] - indices[0]
                    )
                    if isinstance(dependency, Current):
                        direction *= 1e4
                    step = 1e-6
                    objective.x = initial + step * direction
                    plus = float(objective.J())
                    objective.x = initial - step * direction
                    minus = float(objective.J())
                    objective.x = initial
                    finite_difference = (plus - minus) / (2 * step)
                    np.testing.assert_allclose(
                        gradient @ direction,
                        finite_difference,
                        rtol=2e-7,
                        atol=2e-10,
                        err_msg="NetFluxes.dJ must differentiate J on its sampled target grid",
                    )

    def test_downsampled_flux_matches_explicit_target_grid(self):
        """Check free and fixed derivatives against an explicitly sampled target."""
        rng = np.random.default_rng(23)
        curves = create_equally_spaced_curves(2, 1, False, order=3, numquadpoints=12)
        for curve in curves:
            curve.x = curve.x + 0.04 * rng.standard_normal(curve.x.shape)
        curves[0].fix("xc(0)")
        source = Coil(CurveXYZFourier(24, 3, dofs=curves[1].dofs), Current(1.7e4))
        source.current.fix_all()
        target_current = Current(2.1e4)
        target = Coil(curves[0], target_current)
        for downsample in [2, 3]:
            with self.subTest(downsample=downsample):
                sampled_curve = CurveXYZFourier(
                    np.asarray(curves[0].quadpoints)[::downsample],
                    3,
                    dofs=curves[0].dofs,
                )
                sampled = NetFluxes(Coil(sampled_curve, target_current), [source])
                objective = NetFluxes(target, [source], downsample=downsample)
                np.testing.assert_allclose(
                    objective.J(), sampled.J(), rtol=1e-13, atol=1e-14
                )
                objective_partials = cast(_PartialDerivative, objective.dJ)(
                    partials=True
                )
                sampled_partials = cast(_PartialDerivative, sampled.dJ)(partials=True)
                for dependency, sampled_dependency in [
                    (curves[0], sampled_curve),
                    (source.curve, source.curve),
                    (source.current, source.current),
                ]:
                    np.testing.assert_allclose(
                        np.asarray(objective_partials(dependency)),
                        np.asarray(sampled_partials(sampled_dependency)),
                        rtol=1e-12,
                        atol=1e-13,
                    )
                    np.testing.assert_allclose(
                        objective_partials.data[dependency],
                        sampled_partials.data[sampled_dependency],
                        rtol=1e-12,
                        atol=1e-13,
                    )
