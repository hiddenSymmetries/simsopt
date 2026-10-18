#!/usr/bin/env python
"""
Field of the plasma currents by virtual casing, and field lines of the coils and the plasma.

VmecVirtualCasingField evaluates the field of the plasma currents of a VMEC equilibrium at
arbitrary points outside its boundary (simsopt.mhd.VirtualCasing gives it only on the boundary).
Here it is evaluated 10 cm outside the boundary of the Landreman-Buller-Drevlak quasi-helical
equilibrium at 5% beta.

VmecPlasmaField extends it inside the plasma as B_VMEC - B_ext, with B_ext the field of the
external currents from the same virtual casing calculation. Adding the coils of Wiedman, Buller
and Landreman, J. Plasma Phys. (2024), gives the total field of the coils and the plasma,
continuous across the plasma boundary; inside it differs from B_VMEC only by the field of the
coils' error, B_coil - B_ext. The total field is interpolated on a grid covering the plasma and a
layer around it, and field lines are traced from the core to beyond the plasma boundary.

This example requires the python ``virtual_casing`` package.
"""
import time
from pathlib import Path

import numpy as np

from simsopt import load
from simsopt.field import (InterpolatedField, LevelsetStoppingCriterion, VmecPlasmaField, VmecVirtualCasingField,
                           compute_fieldlines, plot_poincare_data)
from simsopt.geo import SurfaceClassifier, SurfaceRZFourier
from simsopt.mhd import Vmec
from simsopt.util import in_github_actions

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc"
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
vmec = Vmec(filename)

# Source grid points per half field period. Within about a grid cell of the boundary the
# accuracy depends on this resolution.
nphi = 64
vc = VmecVirtualCasingField(vmec, src_nphi=nphi, src_ntheta=nphi, digits=6)

# Points 10 cm outside the plasma boundary:
n = 32
surf = vc.surf.copy(nphi=n, ntheta=n, range="half period")
surf.extend_via_normal(0.1)
vc.set_points(surf.gamma().reshape((-1, 3)))
modB = np.linalg.norm(vc.B(), axis=-1).reshape((n, n))
print(f"|B| of the plasma currents 10 cm outside the boundary: {modB.min():.4f} to {modB.max():.4f} T")

# On the boundary itself the field of the external currents, which a coil set has to supply:
print(f"max |B_external . n| on the boundary: {np.abs(vc.Bnormal_due_ext).max():.4f} T")

# Total field of the coils and the plasma, inside and outside the plasma:
coils = load(TEST_DIR / "coils_QH_nfp4_aspect6p5_beta0p05_Wiedman2024.json")
plasma = VmecPlasmaField(vmec, src_nphi=64, digits=5)
field = coils + plasma

# It is continuous across the plasma boundary:
boundary = SurfaceRZFourier.from_wout(filename, range="half period", nphi=8, ntheta=8)
x = boundary.gamma().reshape((-1, 3))
normal = np.sign(boundary.volume()) * boundary.unitnormal().reshape((-1, 3))
field.set_points(x + 1e-6 * normal)
B_out = field.B().copy()
field.set_points(x - 1e-6 * normal)
B_in = field.B().copy()
print("max |B(1 um outside) - B(1 um inside)| / |B|:",
      np.max(np.linalg.norm(B_out - B_in, axis=1) / np.linalg.norm(B_in, axis=1)))

# Region to interpolate on: the plasma and a layer of thickness `margin` around it.
margin = 0.4
plasma_surf = SurfaceRZFourier.from_wout(filename, range="full torus", nphi=128, ntheta=64)
domain = plasma_surf.copy()
domain.extend_via_normal(margin)
classifier = SurfaceClassifier(domain, h=0.1, p=2)


def skip(rs, phis, zs):
    rphiz = np.asarray([rs, phis, zs]).T.copy()
    return list((classifier.evaluate_rphiz(rphiz) < -0.2).flatten())


nfp = plasma_surf.nfp
R = np.linalg.norm(domain.gamma()[:, :, :2], axis=2)
Z = domain.gamma()[:, :, 2]
cells = 8 if in_github_actions else 16
interpolated = InterpolatedField(field, 3, (R.min(), R.max(), cells), (0, 2 * np.pi / nfp, cells),
                                 (0, Z.max(), cells // 2), True, nfp=nfp, stellsym=True, skip=skip)

# The interpolant against the direct evaluation, inside and outside the plasma:
rng = np.random.default_rng(0)
theta, phi = rng.uniform(0, 1, 100), rng.uniform(0, 1, 100)
probe = plasma_surf.copy(quadpoints_phi=phi, quadpoints_theta=theta)
x0 = np.diagonal(probe.gamma(), axis1=0, axis2=1).T
nhat = np.sign(plasma_surf.volume()) * np.diagonal(probe.unitnormal(), axis1=0, axis2=1).T
x = np.concatenate([x0 - 0.2 * nhat, x0 + 0.1 * nhat])
t0 = time.time()
interpolated.set_points(x)
B_interp = interpolated.B().copy()
print(f"interpolant built in {time.time() - t0:.0f} s")
field.set_points(x)
diff = np.linalg.norm(B_interp - field.B(), axis=1) / np.linalg.norm(field.B(), axis=1)
print(f"interpolant against direct evaluation, 0.2 m inside and 0.1 m outside the boundary: "
      f"median {np.median(diff):.1e}, max {diff.max():.1e}")

# Field lines from the core to beyond the plasma boundary, at phi = 0 and z = 0:
R_axis = np.sum(vmec.wout.rmnc[:, 0])
R_edge = np.sum(vmec.wout.rmnc[:, -1])
R0 = np.linspace(R_axis + 0.3 * (R_edge - R_axis), R_edge + 0.5 * margin, 4 if in_github_actions else 10)
phis = [2 * np.pi * i / (4 * nfp) for i in range(4)]
t0 = time.time()
tys, phi_hits = compute_fieldlines(interpolated, R0, np.zeros_like(R0), tmax=1000 if in_github_actions else 20000,
                                   tol=1e-10, phis=phis, stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)])
print(f"{len(R0)} field lines traced in {time.time() - t0:.0f} s")
plot_poincare_data(phi_hits, phis, OUT_DIR / "vc_field_poincare.png", dpi=150, surf=plasma_surf)
