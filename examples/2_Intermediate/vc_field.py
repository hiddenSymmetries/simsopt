from simsopt.field import VirtualCasingField
from simsopt.geo import SurfaceRZFourier
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Resolution on the plasma boundary surface:
# nphi is the number of grid points in 1/2 a field period.
nphi = 32
ntheta = 33

# Resolution for the virtual casing calculation:
vc_src_nphi = 80

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = 'wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc'
vmec_file = TEST_DIR / filename

vc = VirtualCasingField.from_vmec(vmec_file, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)

# Get offset surface to evaluate magnetic field on 
surf = SurfaceRZFourier.from_wout(vmec_file, nphi=nphi, ntheta=ntheta)
surf.extend_via_normal(0.1)

vc.set_points(surf.gamma().reshape((-1, 3)))
B = vc.B() 

modB = np.linalg.norm(B, axis=-1)
modB = modB.reshape((nphi, ntheta))

plt.figure()
plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, modB.T, cmap='RdBu')
plt.xlabel('phi')
plt.ylabel('theta')
plt.colorbar()
plt.title("Virtual Casing Magnetic Field Magnitude")
plt.show()