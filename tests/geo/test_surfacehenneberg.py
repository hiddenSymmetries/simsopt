import unittest
from pathlib import Path
import numpy as np

from simsopt.geo.surfacehenneberg import SurfaceHenneberg

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

class SurfaceHennebergTests(unittest.TestCase):
    def test_names(self):
        """
        Check that the names of the dofs are set correctly.
        """
        surf = SurfaceHenneberg(nfp=1, alpha_times_2=1, mmax=2, nmax=1)
        names_correct = ['R0nH(0)', 'R0nH(1)',
                         'Z0nH(1)',
                         'bn(0)', 'bn(1)',
                         'rhomn(0,0)', 'rhomn(0,1)',
                         'rhomn(1,-1)', 'rhomn(1,0)', 'rhomn(1,1)',
                         'rhomn(2,-1)', 'rhomn(2,0)', 'rhomn(2,1)']
        self.assertEqual(surf.names, names_correct)

if __name__ == "__main__":
    unittest.main()
