from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.sampling import draw_uniform_on_curve, draw_uniform_on_surface
import numpy as np
import unittest


class SamplingTesting(unittest.TestCase):

    """
    When sampling from a curve/surface, simply randomly sampling phi (and
    theta) won't give a uniform distribution in space, since a curve or surface
    might be stretched out in certain regions. Here we test that we are
    actually creating uniform samples on geometric objects.
    """

    def test_curve_sampling(self):
        np.random.seed(1)
        nquadpoints = int(1e5)
        curve = CurveRZFourier(nquadpoints, 1, 1, True)
        dofs = curve.get_dofs()
        dofs[0] = 1
        dofs[1] = 0.9
        curve.set_dofs(dofs)
        l = curve.incremental_arclength()
        print(np.min(l), np.max(l))

        start = int(0.1*nquadpoints)
        stop = int(0.5*nquadpoints)
        quadpoints = curve.quadpoints
        from scipy.integrate import simpson
        length_of_subset = simpson(y=l[start:stop], x=quadpoints[start:stop])
        total_length = simpson(y=l, x=quadpoints)
        print("length_of_subset/total_length", length_of_subset/total_length)

        nsamples = int(1e6)
        xyz, idxs = draw_uniform_on_curve(curve, nsamples, safetyfactor=10)
        samples_in_range = np.sum((idxs >= start) * (idxs < stop))
        print("samples_in_range/nsamples", samples_in_range/nsamples)
        print("fraction of samples if uniform", (stop-start)/(nquadpoints))
        assert abs(samples_in_range/nsamples - length_of_subset/total_length) < 1e-3

    def test_surface_sampling(self):
        np.random.seed(1)
        nquadpoints = int(4e2)
        surface = SurfaceRZFourier.from_nphi_ntheta(nfp=1, stellsym=True, mpol=1,
                                                    ntor=0, nphi=nquadpoints,
                                                    ntheta=nquadpoints)
        dofs = surface.get_dofs()
        dofs[0] = 1
        dofs[1] = 0.8
        surface.set_dofs(dofs)
        n = np.linalg.norm(surface.normal(), axis=2)
        print(np.min(n), np.max(n))

        start = int(0.2*nquadpoints)
        stop = int(0.5*nquadpoints)

        from scipy.integrate import simpson
        quadpoints_phi = surface.quadpoints_phi
        quadpoints_theta = surface.quadpoints_theta
        lineintegrals = [simpson(y=n[i, start:stop], x=quadpoints_theta[start:stop]) for i in range(start, stop)]
        area_of_subset = simpson(y=lineintegrals, x=quadpoints_phi[start:stop])
        total_area = surface.area()
        print("area_of_subset/total_area", area_of_subset/total_area)

        nsamples = int(1e6)
        xyz, idxs = draw_uniform_on_surface(surface, nsamples, safetyfactor=10)
        samples_in_range = np.sum((idxs[0] >= start) * (idxs[0] < stop)*(idxs[1] >= start) * (idxs[1] < stop))
        print("samples_in_range/nsamples", samples_in_range/nsamples)
        print("fraction of samples if uniform", (stop-start)**2/(nquadpoints**2))
        assert abs(samples_in_range/nsamples - area_of_subset/total_area) < 1e-2
