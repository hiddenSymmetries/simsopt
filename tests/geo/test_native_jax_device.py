"""Backend mode validation, native x64/platform settings and the native jit device
policy."""

from unittest_jax_support import JaxTestCase, optional_jax_imports, require_cuda

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    from simsopt_jax.backend import get_backend_config, set_backend
from tempfile import TemporaryDirectory
from pathlib import Path


import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np


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


from simsopt.geo.config import parameters
from simsopt.geo.jit import jit, native_jax_device


class TestNativeJaxDevice(JaxTestCase):
    def test_unknown_backend_mode_is_rejected(self):
        """An unknown backend mode is rejected through both explicit and environment
        selection, naming the accepted modes."""
        for selector in ["set_backend", "environment"]:
            with self.subTest(selector=selector), self.case():
                self._case_unknown_backend_mode_is_rejected(selector)

    def _case_unknown_backend_mode_is_rejected(self, selector):
        """Run one row with case-local objects released before runtime cleanup."""
        with self.assertRaisesRegex(
            ValueError,
            r"Backend mode 'jax_cpu_unknown' is not valid\. Accepted: \('native_cpu',",
        ):
            if selector == "set_backend":
                # The mode is deliberately outside the accepted ``BackendMode`` values.
                set_backend("jax_cpu_unknown")  # pyright: ignore[reportArgumentType]
            else:
                os.environ["SIMSOPT_BACKEND_MODE"] = "jax_cpu_unknown"
                get_backend_config()

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
        if device == "gpu":
            require_cuda("CUDA JAX device required")
        _run_fresh_process(
            "set_backend_after_native_import.py",
            device,
            platforms_after_import or "",
            environment=_with_gpu_determinism(_fresh_environment()),
        )

    def test_native_stage_two_with_explicit_platforms_in_fresh_process(self):
        """Native stage II after ``set_backend``; ``gpu-no-platform-env`` exports no
        JAX platform variable, so ``simsopt.geo`` sets ``cpu`` before ``set_backend``."""
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
        if device == "gpu":
            require_cuda("CUDA JAX device required")
        environment = _with_gpu_determinism(_fresh_environment())
        environment["SIMSOPT_JAX_COMPILATION_CACHE_DIR"] = str(
            tmp_path / "compilation-cache"
        )
        if jax_platforms is not None:
            environment["JAX_PLATFORMS"] = jax_platforms
        _run_fresh_process(
            "native_stage_two.py",
            device,
            jax_platforms or "",
            environment=environment,
        )

    def _run_platform_startup(self, scenario, jax_platforms, *, needs_cuda):
        """Run one ``set_backend_platform_startup.py`` scenario in a fresh process."""
        if needs_cuda:
            require_cuda("CUDA JAX device required")
        environment = _with_gpu_determinism(_fresh_environment())
        if jax_platforms is not None:
            environment["JAX_PLATFORMS"] = jax_platforms
        _run_fresh_process(
            "set_backend_platform_startup.py", scenario, environment=environment
        )

    def test_gpu_selected_at_startup_also_starts_the_cpu_in_fresh_process(self):
        """Without ``JAX_PLATFORMS``, a GPU selection starts CUDA as default and the CPU for native kernels."""
        self._run_platform_startup("gpu_at_startup", None, needs_cuda=True)

    def test_selections_after_initialization_keep_the_platforms_in_fresh_process(self):
        """Under ``cuda,cpu``, CPU-then-GPU selections after initialization keep native kernels and platforms."""
        self._run_platform_startup(
            "cpu_then_gpu_after_initialization", "cuda,cpu", needs_cuda=True
        )

    def test_explicit_cuda_platforms_start_cuda_alone_in_fresh_process(self):
        """An explicit ``JAX_PLATFORMS=cuda`` starts only CUDA, and native kernels fall back to it."""
        self._run_platform_startup("explicit_cuda_only", "cuda", needs_cuda=True)

    def test_cpu_selected_before_initialization_keeps_cuda_off_in_fresh_process(self):
        """A CPU selection before initialization starts only the CPU, also under ``cuda,cpu``."""
        for jax_platforms in (None, "cuda,cpu"):
            with self.subTest(jax_platforms=jax_platforms), self.case():
                self._run_platform_startup(
                    "cpu_before_initialization", jax_platforms, needs_cuda=False
                )

    def test_native_jit_runs_on_cpu_with_cuda_default_in_fresh_process(self):
        """Under ``JAX_PLATFORMS=cuda,cpu`` native kernels return CPU arrays,
        also with jit disabled (``SIMSGEOJIT=0``)."""
        require_cuda("CUDA JAX device required")
        for use_jit in ("1", "0"):
            with self.subTest(SIMSGEOJIT=use_jit):
                environment = _with_gpu_determinism(_fresh_environment())
                environment.update(JAX_PLATFORMS="cuda,cpu", SIMSGEOJIT=use_jit)
                _run_fresh_process("native_jit_device_policy.py", environment=environment)

    def test_native_jit_wrapper_preserves_results(self):
        """The native ``jit`` wrapper returns ``jax.jit``'s result on ``native_jax_device()``."""

        def scaled_norm(x, scale):
            """Scaled Euclidean norm."""
            return scale * jax.numpy.linalg.norm(x)

        # Upstream geo tests disable jit at import; this check needs it enabled.
        self.patches.enter_context(patch.dict(parameters, {"jit": True}))
        x = np.linspace(0.0, 1.0, 7)
        wrapped = jit(scaled_norm, static_argnums=1)
        result = wrapped(x, 2.0)
        np.testing.assert_array_equal(
            result, jax.jit(scaled_norm, static_argnums=1)(x, 2.0)  # noqa: TID251 raw jax.jit is the reference result
        )
        self.assertEqual(result.devices(), {native_jax_device()})
        self.assertIs(wrapped.__wrapped__, scaled_norm)
        self.assertEqual(wrapped.__name__, scaled_norm.__name__)
        self.assertEqual(wrapped.__doc__, scaled_norm.__doc__)
