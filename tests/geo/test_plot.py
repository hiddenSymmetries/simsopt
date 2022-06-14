import unittest
import logging
from pathlib import Path

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.plotting import plot

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()


class PlotTests(unittest.TestCase):

    def test_curves_and_surface(self):
        """
        Verify that a set of curves and a surface can all be plotted
        together.  The ``show`` argument is set to ``False`` so the
        tests do not require human intervention to close plot windows.
        However, if you do want to actually display the figure, you
        can change ``show`` to ``True`` in the first line of this
        function.
        """
        show = False

        engines = []
        try:
            import matplotlib
        except ImportError:
            pass
        else:
            engines.append("matplotlib")

        try:
            import mayavi
        except ImportError:
            pass
        else:
            engines.append("mayavi")

        try:
            import plotly
        except ImportError:
            pass
        else:
            engines.append("plotly")

        logger.info(f'Testing these plotting engines: {engines}')

        nphi = 32
        ntheta = 32
        filename = TEST_DIR / "input.LandremanPaul2021_QA"
        s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
        # Number of unique coil shapes:
        ncoils = 5
        # Major radius for the initial circular coils:
        R0 = 1.0
        # Minor radius for the initial circular coils:
        R1 = 0.5
        # Number of Fourier modes describing each Cartesian component of each coil:
        order = 5

        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
        base_currents = [Current(1e5) for i in range(ncoils)]
        base_currents[0].local_fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
        items_to_plot = coils + [s]  # Coils and surface together
        items_to_plot2 = [c.curve for c in coils] + [s]  # Curves and surface together

        for engine in engines:
            plot(items_to_plot, engine=engine, show=show)
            # Try adding a keyword argument:
            plot(items_to_plot2, engine=engine, show=show, close=True)


if __name__ == "__main__":
    unittest.main()
