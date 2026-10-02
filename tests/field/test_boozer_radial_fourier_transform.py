"""``fourier_transform_even`` and ``fourier_transform_odd`` are thread-safe.

Their point sums run in an OpenMP ``parallel for`` and must be reductions. Each
case runs in a child process, as OpenMP reads ``OMP_NUM_THREADS`` at load
time. A race does not fail every run; with 4 threads and 2e5 points it nearly
always does.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_CHILD_PROGRAM = """
import sys
import numpy as np
import simsoptpp as sopp

data = np.load(sys.argv[1])
args = [np.ascontiguousarray(data[name]) for name in ("K", "xm", "xn", "thetas", "zetas")]
np.savez(sys.argv[2],
         even=np.asarray(sopp.fourier_transform_even(*args)),
         odd=np.asarray(sopp.fourier_transform_odd(*args)))
"""

NUM_POINTS = 200_000
RELATIVE_TOLERANCE = 1e-10


def _modes():
    xm, xn = [], []
    for m in range(4):
        for n in range(-3, 4):
            if m == 0 and n < 0:
                continue
            xm.append(m)
            xn.append(2 * n)
    return np.array(xm, dtype=float), np.array(xn, dtype=float)


def _reference(K, xm, xn, thetas, zetas, basis, first_mode):
    """Serial sums; modes before ``first_mode`` are not transformed (zero)."""
    angles = xm[first_mode:, None] * thetas[None, :] - xn[first_mode:, None] * zetas[None, :]
    values = basis(angles)
    kmns = np.zeros(xm.size)
    kmns[first_mode:] = (values @ K) / np.einsum("mp,mp->m", values, values)
    return kmns


class BoozerRadialFourierTransformTests(unittest.TestCase):

    def _run(self, workdir, inputs, num_threads):
        output = os.path.join(workdir, f"out_{num_threads}.npz")
        env = dict(os.environ, OMP_NUM_THREADS=str(num_threads))
        completed = subprocess.run(
            [sys.executable, "-c", _CHILD_PROGRAM, inputs, output],
            env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        with np.load(output) as loaded:
            return {key: np.array(loaded[key]) for key in loaded.files}

    def test_transforms_match_a_serial_reference_with_four_threads(self):
        rng = np.random.default_rng(1)
        xm, xn = _modes()
        thetas = rng.uniform(0, 2 * np.pi, NUM_POINTS)
        zetas = rng.uniform(0, 2 * np.pi, NUM_POINTS)
        K = 1.0 + rng.standard_normal(NUM_POINTS)
        expected = {
            "even": _reference(K, xm, xn, thetas, zetas, np.cos, 0),
            "odd": _reference(K, xm, xn, thetas, zetas, np.sin, 1),
        }
        with tempfile.TemporaryDirectory() as workdir:
            inputs = os.path.join(workdir, "inputs.npz")
            np.savez(inputs, K=K, xm=xm, xn=xn, thetas=thetas, zetas=zetas)
            for num_threads in (1, 4):
                result = self._run(workdir, inputs, num_threads)
                for kind in ("even", "odd"):
                    with self.subTest(threads=num_threads, transform=kind):
                        np.testing.assert_allclose(
                            result[kind], expected[kind],
                            rtol=RELATIVE_TOLERANCE, atol=RELATIVE_TOLERANCE * np.max(np.abs(expected[kind])),
                            err_msg=f"fourier_transform_{kind} with {num_threads} OpenMP "
                                    "threads differs from the serial sum")


if __name__ == "__main__":
    unittest.main()
