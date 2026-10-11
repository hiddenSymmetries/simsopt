"""Native precision/platform settings and C++ curve length host ownership."""

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
from tempfile import TemporaryDirectory
from pathlib import Path


import os
import subprocess
import sys


def _fresh_environment():
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("JAX_", "SIMSOPT_"))
    }


_GPU_DETERMINISM_FLAG = "--xla_gpu_exclude_nondeterministic_ops=true"


def _with_gpu_determinism(environment):
    environment["XLA_FLAGS"] = " ".join(
        flag for flag in (environment.get("XLA_FLAGS"), _GPU_DETERMINISM_FLAG) if flag
    )
    return environment


_FRESH_PROCESS = Path(__file__).resolve().with_name("fresh_process")


def _run_fresh_process(script, *arguments, environment):
    """Run ``fresh_process/<script>`` in a new interpreter; it fails on its asserts."""
    subprocess.run(
        [sys.executable, str(_FRESH_PROCESS / script), *arguments],
        env=environment,
        check=True,
    )


from simsopt.geo import CurveLength, create_equally_spaced_curves


class TestNativeJaxDevice(JaxTestCase):
    def test_removed_float32_mode_is_rejected_in_fresh_process(self):
        """Removed float32 backend modes are rejected through both explicit and
        environment selection."""
        for selector in ["set_backend", "environment"]:
            with self.subTest(selector=selector), self.case():
                self._case_removed_float32_mode_is_rejected_in_fresh_process(selector)

    def _case_removed_float32_mode_is_rejected_in_fresh_process(self, selector):
        """Run one row with case-local objects released before runtime cleanup."""
        environment = _fresh_environment()
        environment["JAX_PLATFORMS"] = "cpu"
        if selector == "environment":
            environment["SIMSOPT_BACKEND_MODE"] = "jax_cpu_float32_smoke"
        _run_fresh_process("removed_float32_mode.py", selector, environment=environment)

    def test_native_import_preserves_platform_selection_in_fresh_process(self):
        """Importing native geometry respects explicit CPU platform selection and enables
        double precision."""
        for platform_variable in [None, "JAX_PLATFORMS", "JAX_PLATFORM_NAME"]:
            with self.subTest(platform_variable=platform_variable), self.case():
                self._case_native_import_preserves_platform_selection_in_fresh_process(
                    platform_variable
                )

    def _case_native_import_preserves_platform_selection_in_fresh_process(
        self, platform_variable
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        environment = _fresh_environment()
        if platform_variable is not None:
            environment[platform_variable] = "cpu"
        _run_fresh_process("native_import_platform_selection.py", environment=environment)

    def test_native_import_enables_x64_despite_explicit_disable_in_fresh_process(self):
        """Importing native geometry restores double precision even when the environment
        explicitly disables it."""
        for x64_setting in ["false", "0"]:
            with self.subTest(x64_setting=x64_setting), self.case():
                self._case_native_import_enables_x64_despite_explicit_disable_in_fresh_process(
                    x64_setting
                )

    def _case_native_import_enables_x64_despite_explicit_disable_in_fresh_process(
        self, x64_setting
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        environment = _fresh_environment()
        environment["JAX_PLATFORMS"] = "cpu"
        environment["JAX_ENABLE_X64"] = x64_setting
        _run_fresh_process("native_import_x64.py", environment=environment)

    def test_native_jax_curve_constructed_before_parity_backend_in_fresh_process(self):
        """A helical curve constructed before backend setup still yields finite float64
        length and field values."""
        patches = self.patches
        tmp_path = Path(patches.enter_context(TemporaryDirectory()))
        environment = _fresh_environment()
        environment.update(
            SIMSOPT_BACKEND_MODE="jax_cpu_parity",
            SIMSOPT_JAX_COMPILATION_CACHE_DIR=str(tmp_path / "compilation-cache"),
        )
        _run_fresh_process("curve_before_parity_backend.py", environment=environment)

    def test_set_backend_default_platform_after_native_import_in_fresh_process(self):
        """``set_backend`` makes its platform the default after ``simsopt.geo`` set ``cpu``.

        Without JAX platform variables, ``simsopt.geo`` sets the legacy
        ``jax_platform_name="cpu"``, which JAX consults before ``jax_platforms``.
        Every row checks that ``set_backend`` clears it and that its platform is
        then the default backend. The CUDA rows (skipped without a CUDA device)
        are the failures the fix removed: ``Unknown backend cpu`` or, with
        ``cuda,cpu`` exported after the import, CPU while CUDA was requested.
        """
        for device, platforms_after_import in [
            ("cpu", None),
            ("gpu", None),
            ("gpu", "cuda,cpu"),
        ]:
            with self.subTest(
                device=device, platforms_after_import=platforms_after_import
            ), self.case():
                self._case_set_backend_default_platform_after_native_import_in_fresh_process(
                    device, platforms_after_import
                )

    def _case_set_backend_default_platform_after_native_import_in_fresh_process(
        self, device, platforms_after_import
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        if device == "gpu" and not any(d.platform == "gpu" for d in jax.devices()):
            self.skipTest("CUDA JAX device required")
        _run_fresh_process(
            "set_backend_after_native_import.py",
            device,
            platforms_after_import or "",
            environment=_with_gpu_determinism(_fresh_environment()),
        )

    def test_native_stage_two_with_explicit_platforms_in_fresh_process(self):
        """Native stage II after ``set_backend``; ``gpu-no-platform-env`` exports no
        JAX platform variable, so ``simsopt.geo`` sets ``cpu`` before ``set_backend``
        and native length kernels run on CUDA alone."""
        for case_id_0, (device, jax_platforms) in zip(
            ["cpu", "gpu-cuda-cpu-platforms", "gpu-no-platform-env"],
            [("cpu", "cpu"), ("gpu", "cuda,cpu"), ("gpu", None)],
            strict=True,
        ):
            with self.subTest(
                case_id_0=case_id_0, device=device, jax_platforms=jax_platforms
            ), self.case() as patches:
                self._case_native_stage_two_with_explicit_platforms_in_fresh_process(
                    device, jax_platforms, patches
                )

    def _case_native_stage_two_with_explicit_platforms_in_fresh_process(
        self, device, jax_platforms, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        tmp_path = Path(patches.enter_context(TemporaryDirectory()))
        if device == "gpu" and not any(d.platform == "gpu" for d in jax.devices()):
            self.skipTest("CUDA JAX device required")
        environment = _with_gpu_determinism(_fresh_environment())
        environment["SIMSOPT_JAX_COMPILATION_CACHE_DIR"] = str(
            tmp_path / "compilation-cache"
        )
        if jax_platforms is not None:
            environment["JAX_PLATFORMS"] = jax_platforms
        native_platform = "gpu" if jax_platforms is None else "cpu"
        _run_fresh_process(
            "native_stage_two.py",
            device,
            native_platform,
            jax_platforms or "",
            environment=environment,
        )

    def test_curve_length_preserves_incremental_arclength_derivative(self):
        """CurveLength retains its upstream dJ_dl callable and exact mean gradient."""
        curve = create_equally_spaced_curves(
            1, 1, False, R0=1.0, R1=0.5, order=2, numquadpoints=32,
        )[0]
        length = CurveLength(curve)
        arclength = curve.incremental_arclength()
        self.assertTrue(callable(length.dJ_dl))
        self.assertTrue(jax.numpy.array_equal(
            length.dJ_dl(arclength), jax.numpy.full_like(arclength, 1.0 / arclength.size),
        ))
