from numpy.random import uniform
from simsopt.geo import SurfaceBSpline
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('qtagg')

if __name__ == "__main__":
    surf_kwargs = {
        "axis_points": 3,
        "points_per_cs": 4,
        "n_cs": 4,
        "nfp": 2,
        "M": 8,
        "N": 4,
        "p_u": 3,
        "p_v": 3,
        "cs_equispaced": True,
        "rays_equispaced": False,
        "cs_global_angle_free": False,
        "axis_angles_fixed": False,
        "cs_basis": "polar",
        "nurbs": False,
        "use_bishop_frame": True,
    }
    surf = SurfaceBSpline(
        **surf_kwargs
    )
    random_dofs = uniform(surf.lower_bounds, surf.upper_bounds)
    surf.x = random_dofs

    surf.plot()
    plt.show()