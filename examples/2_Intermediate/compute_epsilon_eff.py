import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from simsopt.mhd import Vmec
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField, BoozerAnalytic
from simsopt.field.bounce import eps_eff
import simsoptpp as sopp

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'wout_aten_v444_b586.nc'
vmec = Vmec(filename)

order = 3
bri = BoozerRadialInterpolant(vmec, order, mpol=8, ntor=8, enforce_vacuum=True)

nfp = vmec.wout.nfp
degree = 3
srange = (0, 1, 20)
thetarange = (0, np.pi, 20)
zetarange = (0, 2*np.pi/nfp, 20)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

epseff = np.loadtxt(TEST_DIR / 'neo_out.ext')[:, 1]
s_index = np.loadtxt(TEST_DIR / 'neo_out.ext')[:, 0]
s_vmec = vmec.s_half_grid
s_vmec = [s_vmec[int(index)-2] for index in s_index]

epseff_simsopt = np.zeros_like(s_vmec)

nlam = 30
nmin_tol = 1e-3
nmin = 100
nmax = 1000
nstep = 100
nzeta = 100
ntheta = 100
for i in range(len(s_vmec)):
    epseff_simsopt[i] = eps_eff(field, s_vmec[i], nfp, nlam, ntheta, nzeta=nzeta, nmin_tol=nmin_tol,
                                nstep=nstep, nmin=nmin, nmax=nmax, step_size=1e-3, tol=1e-6,
                                nmax_bounce=4, norm=2, root_tol=1e-5)
    print(i)


plt.figure()
plt.plot(s_vmec, epseff, label='NEO')
plt.plot(s_vmec, epseff_simsopt, marker='*', color='black', linestyle='none', label='simsopt')
plt.legend()
plt.xlabel(r'$s$')
plt.ylabel(r'$\epsilon_{\mathrm{eff}}^{3/2}$')
plt.savefig('epsilon_eff.png')
