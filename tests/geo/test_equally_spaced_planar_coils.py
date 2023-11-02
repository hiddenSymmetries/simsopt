import unittest
from simsopt.geo.curve import create_equally_spaced_curves, create_equally_spaced_planar_curves
from simsopt.field import coils_via_symmetries, BiotSavart, Current

class Testing(unittest.TestCase):
    def test_equally_spaced_planar_coils(self):
        ncoils = 4
        nfp = 4
        stellsym = False

        curves = create_equally_spaced_curves(ncoils, nfp, False)
        currents = [Current(1e5) for i in range(ncoils)]

        curves_planar = create_equally_spaced_planar_curves(ncoils, nfp, False)
        currents_planar = [Current(1e5) for i in range(ncoils)]

        coils = coils_via_symmetries(curves, currents, nfp, stellsym)
        coils_planar = coils_via_symmetries(curves_planar, currents_planar, nfp, stellsym)
        bs = BiotSavart(coils)
        bs_planar = BiotSavart(coils_planar)
        err = bs.AbsB()[0][0] - bs_planar.AbsB()[0][0]
        print(err)
        assert(err < 3e-10)

if __name__ == "__main__":
    unittest.main()