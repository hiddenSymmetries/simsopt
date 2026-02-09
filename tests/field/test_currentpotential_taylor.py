import unittest
from simsopt.field.currentpotential import CurrentPotentialFourier
import numpy as np
from simsopt.geo import SurfaceRZFourier


class CurrentPotentialTaylorTests(unittest.TestCase):
    cptypes = ["CurrentPotentialFourier"]

    def subtest_currentpotential_phi_derivative(self, cptype, stellsym):
        epss = [0.5**i for i in range(10, 15)]
        phis = np.asarray([0.6] + [0.6 + eps for eps in epss])
        cp = get_currentpotential(cptype, stellsym, phis=phis)

        f0 = cp.Phi()[0, 0]
        deriv = cp.Phidash1()[0, 0]
        err_old = 1e6
        for i in range(len(epss)):
            fh = cp.Phi()[i+1, 0]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def test_currentpotential_phi_derivative(self):
        """
        Taylor test to verify that the surface tangent in the phi direction
        """
        for cptype in self.cptypes:
            for stellsym in [True, False]:
                with self.subTest(cptype=cptype, stellsym=stellsym):
                    self.subtest_currentpotential_phi_derivative(cptype, stellsym)

    def subtest_currentpotential_theta_derivative(self, cptype, stellsym):
        epss = [0.5**i for i in range(10, 15)]
        thetas = np.asarray([0.6] + [0.6 + eps for eps in epss])
        cp = get_currentpotential(cptype, stellsym, thetas=thetas)
        f0 = cp.Phi()[0, 0]
        deriv = cp.Phidash2()[0, 0]
        err_old = 1e6
        for i in range(len(epss)):
            fh = cp.Phi()[0, i+1]
            deriv_est = (fh-f0)/epss[i]
            err = np.linalg.norm(deriv_est-deriv)
            assert err < 0.55 * err_old
            err_old = err

    def test_currentpotential_theta_derivative(self):
        """
        Taylor test to verify that the surface tangent in the theta direction
        """
        for cptype in self.cptypes:
            for stellsym in [True, False]:
                with self.subTest(cptype=cptype, stellsym=stellsym):
                    self.subtest_currentpotential_theta_derivative(cptype, stellsym)


def get_currentpotential(cptype, stellsym, phis=None, thetas=None):
    np.random.seed(2)
    mpol = 4
    ntor = 3
    nfp = 2
    phis = phis if phis is not None else np.linspace(0, 1, 31, endpoint=False)
    thetas = thetas if thetas is not None else np.linspace(0, 1, 31, endpoint=False)

    if cptype == "CurrentPotentialFourier":
        s = SurfaceRZFourier()
        cp = CurrentPotentialFourier(s, nfp=nfp, stellsym=stellsym, mpol=mpol, ntor=ntor,
                                     quadpoints_phi=phis, quadpoints_theta=thetas)
        cp.x = cp.x * 0.
        cp.phis[1, ntor + 0] = 0.3
    else:
        assert False

    dofs = cp.get_dofs()
    np.random.seed(2)
    rand_scale = 0.01
    cp.x = dofs + rand_scale * np.random.rand(len(dofs))  # .reshape(dofs.shape)
    return cp


if __name__ == "__main__":
    unittest.main()
