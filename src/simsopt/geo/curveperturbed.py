import numpy as np
import simsoptpp as sopp
from simsopt.geo.curve import Curve
from sympy import Symbol, lambdify, exp
import jax.numpy as jnp


class GaussianSampler():
    """
    Generate a periodic gaussian process on the interval [0, 1] on a given list of quadrature points.
    The process has standard deviation ``sigma`` a correlation length scale ``length_scale``. 
    Large values of ``length_scale`` correspond to smooth processes, small values result in highly oscillatory
    functions.
    Also has the ability to sample the derivatives of the function.
    """

    def __init__(self, points, sigma, length_scale, n_derivs=1):
        self.points = points
        xs = self.points
        n = len(xs)
        self.n_derivs = n_derivs
        cov_mat = np.zeros((n*(n_derivs+1), n*(n_derivs+1)))

        def kernel(x, y):
            return sum((sigma**2)*exp(-(x-y+i)**2/(length_scale**2)) for i in range(-4, 5))
        XX, YY = np.meshgrid(xs, xs)
        for i in range(n):
            for j in range(n):
                x = XX[i, j]
                y = YY[i, j]
                if abs(x-y) > 0.5:
                    if y > 0.5:
                        YY[i, j] -= 1
                    else:
                        XX[i, j] -= 1

        for ii in range(n_derivs+1):
            for jj in range(n_derivs+1):
                x = Symbol("x")
                y = Symbol("y")
                f = kernel(x, y)
                if ii + jj == 0:
                    lam = lambdify((x, y), f, "numpy")
                else:
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])), "numpy")
                cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = lam(XX, YY)

        from scipy.linalg import sqrtm
        self.L = np.real(sqrtm(cov_mat))

    def sample(self, randomgen=None):
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = self.L@z
        return [curve_and_derivs[(i*n):((i+1)*n), :] for i in range(n_derivs+1)]


class CurvePerturbed(sopp.Curve, Curve):

    """A perturbed curve."""

    def __init__(self, curve, sampler, randomgen=None, zero_mean=False):
        """
        Perturb a underlying :mod:`simsopt.geo.curve.Curve` object by drawing a perturbation from a 
        ``GaussianSampler``.

        Comment:
        Doing anything involving randomness in parallel is complicated.
        Let's say we have a list of :mod:`simsopt.field.curve.Curve`s ``curves`` that represent a stellarator,
        and now we want to consider ``N`` perturbed stellarators. Let's also say we have multiple MPI ranks.
        Two avoid the same thing happening on the different MPI ranks, we could pick a different seed on each rank.
        However, then we get different results depending on the number of MPI ranks that we run on. Not ideal.
        Instead, we should pick a new seed for each :math:`1\le i\le N`. e.g.

        .. code-block::
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
            for i in range(idx_start, idx_end):
                rg = np.random.Generator(PCG64(seeds[i]))
                perturbed_curves = [CurvePerturbed(c, sampler, randomgen=rg) for c in curves]


        """
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, x0=np.asarray([]), depends_on=[curve])
        self.sampler = sampler
        self.randomgen = randomgen
        self.sample = sampler.sample(self.randomgen)
        self.zero_mean = zero_mean

    def resample(self):
        self.sample = self.sampler.sample(self.randomgen)
        self.invalidate_cache()

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def get_dofs(self):
        return np.asarray([])

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        pert = self.sample[0].copy()
        gamma[:] = self.curve.gamma() + pert
        if self.zero_mean:
            mean_before = np.mean(self.curve.gamma() * self.curve.incremental_arclength()[:, None], axis=0)
            mean_after = np.mean(gamma * self.incremental_arclength()[:, None], axis=0)
            adj = (mean_after-mean_before)/np.mean(self.incremental_arclength())
            gamma -= adj[None, :]

    def gammadash_impl(self, gammadash):
        gammadash[:] = self.curve.gammadash() + self.sample[1]

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self.curve.gammadashdash() + self.sample[2]

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self.curve.gammadashdashdash() + self.sample[3]

    def dgamma_by_dcoeff_vjp(self, v):
        res = self.curve.dgamma_by_dcoeff_vjp(v)
        if self.zero_mean:
            pert = self.sample[0]
            n = len(self.quadpoints)
            denom = np.mean(self.incremental_arclength())
            mean_before = np.mean(self.curve.gamma() * self.curve.incremental_arclength()[:, None], axis=0)
            mean_after = np.mean((self.curve.gamma() + self.sample[0]) * self.incremental_arclength()[:, None], axis=0)
            adj = mean_after-mean_before

            # # derivative of pert * self.incremental_arclength()[:, None]
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 0])/n))*pert[:, 0])
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 1])/n))*pert[:, 1])
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 2])/n))*pert[:, 2])
            # # derivative of self.curve.gamma() * self.incremental_arclength()[:, None]
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 0])/n))*self.curve.gamma()[:, 0])
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 1])/n))*self.curve.gamma()[:, 1])
            # res -= (1./denom)*self.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 2])/n))*self.curve.gamma()[:, 2])
            # res -= (1./denom)*self.curve.dgamma_by_dcoeff_vjp((np.sum(v, axis=0)/n)[None, :]*self.incremental_arclength()[:, None])

            # # derivative of self.curve.gamma() * self.curve.incremental_arclength()[:, None]
            # res += (1./denom)*self.curve.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 0])/n))*self.curve.gamma()[:, 0])
            # res += (1./denom)*self.curve.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 1])/n))*self.curve.gamma()[:, 1])
            # res += (1./denom)*self.curve.dincremental_arclength_by_dcoeff_vjp(float((np.sum(v[:, 2])/n))*self.curve.gamma()[:, 2])
            # res += (1./denom)*self.curve.dgamma_by_dcoeff_vjp((np.sum(v, axis=0)/n)[None, :]*self.curve.incremental_arclength()[:, None])

            # # derivative of the denominator
            # res += (adj[0]/denom**2) * self.dincremental_arclength_by_dcoeff_vjp(np.ones((n, ))*float(np.mean(v[:, 0])))
            # res += (adj[1]/denom**2) * self.dincremental_arclength_by_dcoeff_vjp(np.ones((n, ))*float(np.mean(v[:, 1])))
            # res += (adj[2]/denom**2) * self.dincremental_arclength_by_dcoeff_vjp(np.ones((n, ))*float(np.mean(v[:, 2])))

            vmean = np.mean(v, axis=0)
            v0, v1, v2 = vmean
            lhs1 = (1./denom) * (
                - v0*pert[:, 0] - v1*pert[:, 1] - v2*pert[:, 2]
                - v0*self.curve.gamma()[:, 0] - v1*self.curve.gamma()[:, 1] - v2*self.curve.gamma()[:, 2]
            ) + (1./denom**2) * (
                v0*np.ones((n, ))*adj[0] + v1*np.ones((n, ))*adj[1] + v2*np.ones((n, ))*adj[2]
            )
            res += self.dincremental_arclength_by_dcoeff_vjp(lhs1)

            lhs2 = (1./denom) * (
                + v0*self.curve.gamma()[:, 0] + v1*self.curve.gamma()[:, 1] + v2*self.curve.gamma()[:, 2]
            )
            res += self.curve.dincremental_arclength_by_dcoeff_vjp(lhs2)

            lhs3 = (1./denom) * (
                - vmean[None, :]*self.incremental_arclength()[:, None]
                + vmean[None, :]*self.curve.incremental_arclength()[:, None]
            )
            res += self.curve.dgamma_by_dcoeff_vjp(lhs3)

        return res

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v)
