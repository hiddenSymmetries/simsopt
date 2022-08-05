import unittest
from simsopt.field.currentpotential import CurrentPotential, CurrentPotentialFourier
from simsopt.geo import SurfaceRZFourier
import numpy as np

def taylor_test(f, df, x, epsilons=None, direction=None, order=2):
    np.random.seed(1)
    f0 = f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape))-0.5
    dfx = df(x)@direction
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(8, 20)))
    print("###################################################################")
    err_old = 1e9
    counter = 0
    for eps in epsilons:
        if counter > 8:
            break
        fpluseps = f(x + eps * direction)
        fminuseps = f(x - eps * direction)
        if order == 2:
            fak = 0.3
            dfest = (fpluseps-fminuseps)/(2*eps)
        elif order == 4:
            fplus2eps = f(x + 2*eps * direction)
            fminus2eps = f(x - 2*eps * direction)
            fak = 0.13
            dfest = ((1/12) * fminus2eps - (2/3) * fminuseps + (2/3)*fpluseps
                     - (1/12)*fplus2eps)/eps
        else:
            raise NotImplementedError
        err = np.linalg.norm(dfest - dfx)
        print(err)
        assert err < 1e-9 or err < 0.3 * err_old
        counter += 1
        if err < 1e-9:
            break
        err_old = err
    if err > 1e-10:
        assert counter > 2
    print("###################################################################")


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
