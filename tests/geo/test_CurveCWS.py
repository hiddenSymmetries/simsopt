from simsopt.geo import CurveCWS
from simsopt.geo import SurfaceRZFourier
import matplotlib.pyplot as plt
import numpy as np

# filename = "../test_files/wout_circular_tokamak_reference.nc"
#filename = "../test_files/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
filename = "../test_files/wout_n3are_R7.75B5.7.nc"


s = SurfaceRZFourier.from_wout(
    filename, range="full torus", ntheta=64, nphi=64
)  # range = 'full torus', 'field period', 'half period'

sdofs = s.get_dofs()

cws = CurveCWS(
    mpol=s.mpol,
    ntor=s.ntor,
    idofs=sdofs,
    numquadpoints=250,
    order=1,
    nfp=s.nfp,
    stellsym=s.stellsym,
)

phi_array = np.linspace(0, 2 * np.pi, 15)

fig = plt.figure()
#Create multiple cws curves and plot them with the surface
ax = fig.add_subplot(projection="3d")
for phi in phi_array:
    cws.set_dofs([0, phi, 0, 0, 1, 0, 0, 0])
    gamma = cws.gamma()
    x = gamma[:, 0]
    y = gamma[:, 1]
    z = gamma[:, 2]
    ax.plot(x, y, z)

s.plot(ax=ax, show=False, alpha=0.2)

fig = plt.figure()
ax = fig.add_subplot(111)
for phi in phi_array:
    cws.set_dofs([1, 0, 0, 0, 0, phi, 0, 0])
    gamma = cws.gamma()
    x = gamma[:, 0]
    y = gamma[:, 1]
    z = gamma[:, 2]

    r = np.sqrt(x * x + y * y)
    ax.plot(r, z, label=f"coil at phi={phi}")

    surface_gamma = s.cross_section(phi)
    x = surface_gamma[:, 0]
    y = surface_gamma[:, 1]
    z = surface_gamma[:, 2]
    r = np.sqrt(x * x + y * y)
    ax.plot(r, z, "--", label=f"surface at phi={phi}")
# ax.legend()

plt.show()