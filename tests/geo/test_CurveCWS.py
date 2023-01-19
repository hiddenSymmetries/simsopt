from simsopt.geo import CurveCWS
from simsopt.geo import SurfaceRZFourier
import matplotlib.pyplot as plt
import numpy as np

#filename = "../test_files/wout_circular_tokamak_reference.nc"
filename = "../test_files/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"

s = SurfaceRZFourier.from_wout(
    filename, range="full torus", ntheta=32, nphi=32
)  # range = 'full torus', 'field period', 'half period'
sdofs = s.get_dofs()
cws = CurveCWS(
    mpol=s.mpol,
    ntor=s.ntor,
    idofs=sdofs,
    numquadpoints=100,
    order=1,
    nfp=s.nfp,
    stellsym=s.stellsym,
)
# cws.set_dofs([1, 0, 0, 0])

phi_array = np.linspace(0., 2 * np.pi/s.nfp/2, 7)
fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
ax = fig.add_subplot(111)
# ax.set_xlim3d(-7.5, 7.5)
# ax.set_ylim3d(-7.5, 7.5)
# ax.set_zlim3d(-7.5, 7.5)
for phi in phi_array:
    # print(phi)
    cws.set_dofs([1, 0, 0, 0, 0, phi, 0, 0])
    gamma = cws.gamma()
    x = gamma[:, 0]
    y = gamma[:, 1]
    z = gamma[:, 2]
    r = np.sqrt(x*x+y*y)
    ax.plot(r, z, label=f'coil at phi={phi}')

    surface_gamma = s.cross_section(phi)
    x = surface_gamma[:,0]
    y = surface_gamma[:,1]
    z = surface_gamma[:,2]
    r = np.sqrt(x*x+y*y)
    ax.plot(r, z, '--', label=f'surface at phi={phi}')
plt.legend()
# s.plot(show=False, ax=ax, alpha=0.2)
# cws.plot()
# print(cws.gamma())
# plt.plot(cws.gamma())
plt.show()
