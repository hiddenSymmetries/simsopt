"""Child script for ``test_boozer_radial_fourier_transform``.

Run in a fresh interpreter so that OpenMP reads ``OMP_NUM_THREADS`` when
``simsoptpp`` loads. Usage: ``python fresh_process_boozer_fourier_transform.py
INPUTS.npz OUTPUT.npz``.
"""

import sys

import numpy as np
import simsoptpp as sopp


def main(inputs, output):
    """Write the even and odd transforms of the arrays in ``inputs`` to ``output``."""
    data = np.load(inputs)
    args = [np.ascontiguousarray(data[name]) for name in ("K", "xm", "xn", "thetas", "zetas")]
    np.savez(output,
             even=np.asarray(sopp.fourier_transform_even(*args)),
             odd=np.asarray(sopp.fourier_transform_odd(*args)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
