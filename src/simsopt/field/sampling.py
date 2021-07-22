import numpy as np


def draw_uniform_on_curve(curve, nsamples, safetyfactor=10):
    r"""
    Uses rejection sampling to sample points on a curve. *Warning*: assumes that
    the underlying quadrature points on the Curve are uniformly distributed.

    Args:
        curve: The :mod:`simsopt.geo.curve.Curve` to spawn the particles on.
        nsamples: number of samples.
        safetyfactor: how many more samples than ``nsamples`` to generate for
                      rejection/acceptance.
    """
    alen = curve.incremental_arclength()
    M = np.max(alen)
    nattempts = 10 * nsamples
    idxs = np.random.randint(0, alen.shape[0], size=(nattempts, ))
    accept = np.where(np.random.uniform(low=0, high=1, size=(nattempts, )) < alen[idxs]/M)[0]
    assert len(accept) > nsamples
    idxs = np.sort(idxs[accept[:nsamples]])
    xyz = curve.gamma()[idxs, :]
    return xyz, idxs


def draw_uniform_on_surface(surface, nsamples, safetyfactor=10):
    r"""
    Uses rejection sampling to sample points on a surface. *Warning*: assumes that
    the underlying quadrature points on the surface are uniformly distributed.

    Args:
        surface: The :mod:`simsopt.geo.surface.Surface` to spawn the particles
                 on.
        nsamples: number of samples.
        safetyfactor: how many more samples than ``nsamples`` to generate for
                      rejection/acceptance.
    """
    jac = np.linalg.norm(surface.normal().reshape((-1, 3)), axis=1)
    M = np.max(jac)
    nattempts = 10 * nsamples
    idxs = np.random.randint(0, jac.shape[0], size=(nattempts, ))
    accept = np.where(np.random.uniform(low=0, high=1, size=(nattempts, )) < jac[idxs]/M)[0]
    assert len(accept) > nsamples
    idxs = np.sort(idxs[accept[:nsamples]])
    gamma = surface.gamma()
    order = 'F' if np.isfortran(gamma) else 'C'
    idxs = np.unravel_index(idxs, gamma.shape[:2], order)
    xyz = gamma[idxs[0], idxs[1], :]
    return xyz, idxs
