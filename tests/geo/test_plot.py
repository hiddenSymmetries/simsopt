import unittest
import logging
from pathlib import Path
from unittest.mock import patch

from monty.tempfile import ScratchDir

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

        with ScratchDir("."):
            for engine in engines:
                plot(items_to_plot, engine=engine, show=show)
                # Try adding a keyword argument:
                plot(items_to_plot2, engine=engine, show=show, close=True)


    def test_close_full_surface(self):
        """
        Regression test for Surface.plot() with closed=True for a "full torus" surface.
        Actually asserts that the plot has not changed by accessing `_vec` in
        matplotlib's `Poly3DCollection`.
        """
        show = False
        nphi = 64
        ntheta = 32
        filename = TEST_DIR / "input.LandremanPaul2021_QA"
        s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)
        ax = s.plot(close=True, show=show)
        children = ax.get_children() # get the surface plot
        p = None
        for c in children:
            if c.__class__.__name__ == 'Poly3DCollection':
                p = c
                break
        vec = p._vec # might be dangerous, not part of public matplotlib API
        self.assertAlmostEqual(vec[0][0], 1.30042725)
        self.assertAlmostEqual(vec[2][-1], -0.0910074)

    def test_plot_without_3d_axis(self):
        """
        Make sure we log a warning if an axis that is not a 3d axis is passed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, ax = plt.subplots(1)
        show = False
        nphi = 64
        ntheta = 32
        filename = TEST_DIR / "input.LandremanPaul2021_QA"
        s = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)

        # This replaces logging.Logger.warning
        # with a mocked logger that doesn't log anything but instead  tracks if
        # it has been called and lets us assert that it was called as expected.
        with patch('logging.Logger.warning') as mocked_logger:
            ax = s.plot(close=True, show=show, ax=ax)
            mocked_logger.assert_called_with('A non-3D matplotlib axis was passed as an input. It cannot be used for plotting. Instead, a new axis was created.')
        
if __name__ == "__main__":
    unittest.main()
