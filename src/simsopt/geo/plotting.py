import numpy as np

__all__ = ['fix_matplotlib_3d', 'plot']


def fix_matplotlib_3d(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ``ax.set_aspect('equal')`` and ``ax.axis('equal')`` not working for 3D.
    This function is to be called after objects have been plotted.

    This function was taken from
    `<https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to>`_

    Args:
      ax: a matplotlib axis, e.g., as output from ``plt.gca()``.
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot(items, ax=None, show=True, **kwargs):
    """
    Plot multiple Coil, Curve, and/or Surface objects together on the
    same axes. Any keyword arguments other than the ones listed here
    are passed to the ``plot()`` method of each item. A particularly
    useful argument is ``engine``, which can be set to
    ``"matplotlib"``, ``"mayavi"``, or ``"plotly"``.

    Args:
        items: A list of objects to plot, consisting of :obj:`~simsopt.field.coil.Coil`,
            :obj:`~simsopt.geo.curve.Curve`, and :obj:`~simsopt.geo.surface.Surface` objects.
        ax: The axis object on which to plot. If equal to the default ``None``, a new axis will be created.
        show: Whether to call the ``show()`` function of the graphics engine. Should be set to
            ``False`` if more objects will be plotted on the same axes.

    Returns:
      The axis object used.
    """
    n = len(items)
    for j in range(n):
        if j == n - 1:
            this_show = show
        else:
            this_show = False
        ax = items[j].plot(ax=ax, show=this_show, **kwargs)
    return ax
