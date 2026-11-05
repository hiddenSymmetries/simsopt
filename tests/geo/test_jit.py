import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from simsopt.geo.config import parameters
from simsopt.geo.jit import jit


# JAX on CPU uses a NumPy argument's memory in place, without a copy, only when
# the data is aligned to this many bytes. Small unaligned arguments are copied
# on the way in, which would hide the race that the wrapper's copy prevents.
ALIGNMENT_BYTES = 64


def aligned_buffer(n):
    """A float64 array of length ``n`` whose data pointer is ``ALIGNMENT_BYTES``-aligned."""
    itemsize = np.dtype(np.float64).itemsize
    raw = np.empty(n + ALIGNMENT_BYTES // itemsize, dtype=np.float64)
    start = (-raw.ctypes.data % ALIGNMENT_BYTES) // itemsize
    buf = raw[start:start + n]
    assert buf.ctypes.data % ALIGNMENT_BYTES == 0
    return buf


# Size of the slow gate computation in the race tests. The gate only has to be
# still running after the input buffer is overwritten, which each attempt
# checks; these values make it take several milliseconds on a typical CPU, far
# longer than the write.
GATE_MATRIX_SIZE = 1000
GATE_POWER = 32

# Attempts per input style before the race test treats its inability to probe
# the race as a failure. With the expected asynchronous dispatch, at least one
# attempt should find the gate still running. Five invalid attempts in a row
# indicate that asynchronous dispatch is off or that unexpected dispatch or
# timing behavior needs investigation.
MAX_ATTEMPTS = 5

# Length of the input buffer in the race tests. Any small length works, because
# JAX on CPU reads aligned buffers in place at every size; small keeps the
# wrapper's copy cheap.
ALIGNED_INPUT_BUFFER_LENGTH = 16

# The ways a NumPy input can reach the wrapper, each copied by a different
# part of it. Each entry has a name, a function that reads the input from
# where it was passed and multiplies it by the gate, and a call that passes
# the input that way.
INPUT_PASSING_STYLES = (
    ("positional", lambda x, gate: jnp.sum(x * gate[0, 0]), lambda f, x, gate: f(x, gate)),
    ("list", lambda xs, gate: jnp.sum(xs[0] * gate[0, 0]), lambda f, x, gate: f([x], gate)),
    ("tuple", lambda xs, gate: jnp.sum(xs[0] * gate[0, 0]), lambda f, x, gate: f((x,), gate)),
    ("dict", lambda d, gate: jnp.sum(d["x"] * gate[0, 0]), lambda f, x, gate: f({"x": x}, gate)),
    ("keyword", lambda gate, x: jnp.sum(x * gate[0, 0]), lambda f, x, gate: f(gate, x=x)),
)


class JitWrapperTests(unittest.TestCase):

    def probe_held_result_not_updated_by_later_write(self, use_jax_jit, start_gate, style):
        """
        Runs the race check for one entry of ``INPUT_PASSING_STYLES``, and
        returns whether the race was probed: ``False`` if the gate finished
        before the write in all ``MAX_ATTEMPTS`` attempts.
        """
        # The wrapped computation cannot start until the slow ``gate`` result
        # is ready. If the gate is still running after the input buffer is
        # overwritten, the computation reads the buffer after the write. JAX
        # on CPU reads aligned NumPy inputs in place unless they are copied.
        # The gate is a power of the identity, so ``gate[0, 0]`` is exactly 1;
        # multiplying by it, rather than by ``0.0 * gate[0, 0]``, gives a
        # dependency that the compiler cannot remove.
        name, fun, call = style
        with mock.patch.dict(parameters, {'jit': use_jax_jit}):
            f = jit(fun)
        # Compile the gate, and the wrapped function for this input shape and
        # call style, before the attempts.
        call(f, np.ones(ALIGNED_INPUT_BUFFER_LENGTH), start_gate()).block_until_ready()
        for attempt in range(MAX_ATTEMPTS):
            buf = aligned_buffer(ALIGNED_INPUT_BUFFER_LENGTH)
            buf[:] = 1.0
            expected = float(buf.sum())
            gate = start_gate()
            held = call(f, buf, gate)
            buf[:] = 2.0
            if gate.is_ready():
                # The gate finished before the write, so the computation may
                # already have read the buffer and this attempt proves nothing.
                continue
            self.assertEqual(float(held), expected,
                             f"{name} input, attempt {attempt}: result read the input after the caller "
                             f"overwrote it (jit={use_jax_jit})")
            return True
        return False

    def check_held_result_not_updated_by_later_write(self, use_jax_jit):
        slow = jax.jit(lambda m: jnp.linalg.matrix_power(m, GATE_POWER))
        m = jnp.eye(GATE_MATRIX_SIZE)
        unprobed = []
        for style in INPUT_PASSING_STYLES:
            if not self.probe_held_result_not_updated_by_later_write(use_jax_jit, lambda: slow(m), style):
                unprobed.append(style[0])
        if unprobed:
            self.fail(f"the gate finished before the input was overwritten in all {MAX_ATTEMPTS} "
                      f"attempts for the {', '.join(unprobed)} inputs, so the race was not probed. "
                      "JAX may be dispatching CPU computations synchronously "
                      "(jax_cpu_enable_async_dispatch); the copy cannot be tested in that configuration.")

    def test_held_result_not_updated_by_later_write_with_jit(self):
        self.check_held_result_not_updated_by_later_write(use_jax_jit=True)

    def test_held_result_not_updated_by_later_write_without_jit(self):
        self.check_held_result_not_updated_by_later_write(use_jax_jit=False)

    def test_containers_keywords_and_static_arguments(self):
        f = jit(lambda xs, n, scale=1.0: scale * sum(jnp.sum(x) for x in xs) * n, static_argnums=(1,))
        # The lengths are arbitrary and need not match, because the function
        # sums each array separately. Alignment does not matter here either:
        # this test never writes to an input while a computation may read it.
        xs = [aligned_buffer(4), aligned_buffer(4)]
        xs[0][:] = 1.0
        xs[1][:] = 2.0
        originals = [x.copy() for x in xs]
        n, scale = 3, 0.5
        expected = scale * (xs[0].sum() + xs[1].sum()) * n
        # These inputs happen to give an exactly representable result, but
        # exactness is not part of the contract, so only agreement to rounding
        # is required.
        np.testing.assert_allclose(f(xs, n, scale=scale), expected, rtol=1e-15, atol=0)
        for i, (x, original) in enumerate(zip(xs, originals)):
            np.testing.assert_array_equal(x, original, err_msg=f"input {i} was modified")

    def test_transformations_of_wrapped_function(self):
        g = jit(lambda x: jnp.sum(x ** 3))
        x = np.linspace(0.0, 1.0, 5)
        # Allow roundoff-level differences across JAX transformations and backends.
        np.testing.assert_allclose(jax.grad(g)(x), 3 * x ** 2, rtol=1e-14, atol=0)
        np.testing.assert_allclose(jit(lambda x: jax.grad(g)(x))(x), 3 * x ** 2, rtol=1e-14, atol=0)


if __name__ == "__main__":
    unittest.main()
