import unittest
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.coil import coils_via_symmetries, Current
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.curveobjectives import CurveLength
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux, CoilOptObjective


from pathlib import Path
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'


def check_taylor_test(J):
    dofs = J.x
    np.random.seed(1)
    h = np.random.uniform(size=dofs.shape)
    J0, dJ0 = J.J(), J.dJ()
    dJh = sum(dJ0 * h)
    err_old = 1e10
    for i in range(12, 19):
        eps = 0.5**i
        J.x = dofs + eps*h
        J1 = J.J()
        J.x = dofs - eps*h
        J2 = J.J()
        err = np.abs((J1-J2)/(2*eps) - dJh)
        assert err < 0.6**2 * err_old
        # print("err", err)
        # print("err/err_old", err/err_old)
        err_old = err


class FluxObjectiveTests(unittest.TestCase):

    def test_flux(self):
        s = SurfaceRZFourier.from_vmec_input(filename)
        ncoils = 4
        ALPHA = 1e-5

        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=s.stellsym, R0=1.0, R1=0.5, order=6)
        base_currents = []
        for i in range(ncoils):
            curr = Current(1e5)
            if i == 0:
                curr.fix_all()
            base_currents.append(curr)

        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
        bs = BiotSavart(coils)

        Jf = SquaredFlux(s, bs)
        check_taylor_test(Jf)

        target = np.zeros(s.gamma().shape[0:2])
        Jf2 = SquaredFlux(s, bs, target)
        check_taylor_test(Jf2)
        target = np.ones(s.gamma().shape[0:2])
        Jf3 = SquaredFlux(s, bs, target)
        check_taylor_test(Jf3)

        Jls = [CurveLength(c) for c in base_curves]
        JF = CoilOptObjective(Jf, Jls, ALPHA)
        check_taylor_test(JF)

        JF_scaled_summed = Jf + ALPHA * sum(Jls)
        check_taylor_test(JF_scaled_summed)
