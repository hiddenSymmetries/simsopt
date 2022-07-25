import numpy as np
from sympy import Symbol, lambdify, exp
from monty.json import MSONable, MontyDecoder

import simsoptpp as sopp
from simsopt.geo.curve import Curve

__all__ = ['GaussianSampler', 'PerturbationSample', 'CurvePerturbed']


class GaussianSampler(MSONable):

    def __init__(self, points, sigma, length_scale, n_derivs=1):
        r"""
        Generate a periodic gaussian process on the interval [0, 1] on a given list of quadrature points.
        The process has standard deviation ``sigma`` a correlation length scale ``length_scale``. 
        Large values of ``length_scale`` correspond to smooth processes, small values result in highly oscillatory
        functions.
        Also has the ability to sample the derivatives of the function.

        We consider the kernel

        .. math::

            \kappa(d) = \sigma^2 \exp(-d^2/l^2)

        and then consider a Gaussian process with covariance

        .. math::

            Cov(X(s), X(t)) = \sum_{i=-\infty}^\infty \sigma^2 \exp(-(s-t+i)^2/l^2)

        the sum is used to make the kernel periodic and in practice the infinite sum is truncated.

        Args:
            points: the quadrature points along which the perturbation should be computed.
            sigma: standard deviation of the underlying gaussian process
                   (measure for the magnitude of the perturbation).
            length_scale: length scale of the underlying gaussian process
                          (measure for the smoothness of the perturbation).
            n_derivs: number of derivatives of the gaussian process to sample.
        """
        self.points = points
        self.sigma = sigma
        self.length_scale = length_scale
        xs = self.points
        n = len(xs)
        self.n_derivs = n_derivs
        cov_mat = np.zeros((n*(n_derivs+1), n*(n_derivs+1)))

        def kernel(x, y):
            return sum((sigma**2)*exp(-(x-y+i)**2/(length_scale**2)) for i in range(-5, 6))

        XX, YY = np.meshgrid(xs, xs, indexing='ij')
        x = Symbol("x")
        y = Symbol("y")
        f = kernel(x, y)
        for ii in range(n_derivs+1):
            for jj in range(n_derivs+1):
                if ii + jj == 0:
                    lam = lambdify((x, y), f, "numpy")
                else:
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])), "numpy")
                cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = lam(XX, YY)

        from scipy.linalg import sqrtm
        self.L = np.real(sqrtm(cov_mat))

    def draw_sample(self, randomgen=None):
        """
        Returns a list of ``n_derivs+1`` arrays of size ``(len(points), 3)``, containing the
        perturbation and the derivatives.
        """
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = self.L@z
        return [curve_and_derivs[(i*n):((i+1)*n), :] for i in range(n_derivs+1)]


class PerturbationSample(MSONable):
    """
    This class represents a single sample of a perturbation.  The point of
    having a dedicated class for this is so that we can apply the same
    perturbation to multipe curves (e.g. in the case of multifilament
    approximations to finite build coils).
    The main way to interact with this class is via the overloaded ``__getitem__``
    (i.e. ``[ ]`` indexing).
    For example::

        sample = PerturbationSample(...)
        g = sample[0] # get the values of the perturbation
        gd = sample[1] # get the first derivative of the perturbation
    """

    def __init__(self, sampler, randomgen=None, sample=None):
        self.sampler = sampler
        self.randomgen = randomgen   # If not None, most likely fail with serialization
        if sample:
            self._sample = sample
        else:
            self.resample()

    def resample(self):
        self._sample = self.sampler.draw_sample(self.randomgen)

    def __getitem__(self, deriv):
        """
        Get the perturbation (if ``deriv=0``) or its ``deriv``-th derivative.
        """
        assert isinstance(deriv, int)
        if deriv >= len(self._sample):
            raise ValueError("""
The sample on has {len(self._sample)-1} derivatives.
Adjust the `n_derivs` parameter of the sampler to access higher derivatives.
""")
        return self._sample[deriv]


class CurvePerturbed(sopp.Curve, Curve):

    """A perturbed curve."""

    def __init__(self, curve, sample):
        r"""
        Perturb a underlying :mod:`simsopt.geo.curve.Curve` object by drawing a perturbation from a
        :obj:`GaussianSampler`.

        Comment:
        Doing anything involving randomness in a reproducible way requires care.
        Even more so, when doing things in parallel.
        Let's say we have a list of :mod:`simsopt.geo.curve.Curve` objects ``curves`` that represent a stellarator,
        and now we want to consider ``N`` perturbed stellarators. Let's also say we have multiple MPI ranks.
        To avoid the same thing happening on the different MPI ranks, we could pick a different seed on each rank.
        However, then we get different results depending on the number of MPI ranks that we run on. Not ideal.
        Instead, we should pick a new seed for each :math:`1\le i\le N`. e.g.

        .. code-block:: python

            from randomgen import SeedSequence, PCG64
            import numpy as np
            curves = ...
            sigma = 0.01
            length_scale = 0.2
            sampler = GaussianSampler(curves[0].quadpoints, sigma, length_scale, n_derivs=1)
            globalseed = 1
            N = 10 # number of perturbed stellarators
            seeds = SeedSequence(globalseed).spawn(N)
            idx_start, idx_end = split_range_between_mpi_rank(N) # e.g. [0, 5) on rank 0, [5, 10) on rank 1
            perturbed_curves = [] # this will be a List[List[Curve]], with perturbed_curves[i] containing the perturbed curves for the i-th stellarator
            for i in range(idx_start, idx_end):
                rg = np.random.Generator(PCG64(seeds_sys[j], inc=0))
                stell = []
                for c in curves:
                    pert = PerturbationSample(sampler_systematic, randomgen=rg)
                    stell.append(CurvePerturbed(c, pert))
                perturbed_curves.append(stell)
        """
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, x0=np.asarray([]), depends_on=[curve])
        self.sample = sample

    def resample(self):
        self.sample.resample()
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        gamma[:] = self.curve.gamma() + self.sample[0]

    def gammadash_impl(self, gammadash):
        gammadash[:] = self.curve.gammadash() + self.sample[1]

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self.curve.gammadashdash() + self.sample[2]

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self.curve.gammadashdashdash() + self.sample[3]

    def dgamma_by_dcoeff_vjp(self, v):
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdash_by_dcoeff_vjp(v)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v)

    def as_dict(self) -> dict:
        d = {}
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["curve"] = self.curve.as_dict()
        d["sample"] = self.sample.as_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        decoder = MontyDecoder()
        curve = decoder.process_decoded(d["curve"])
        sample = decoder.process_decoded(d["sample"])
        return cls(curve, sample)
