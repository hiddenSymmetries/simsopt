# Background on files in this directory

    1DOF_Garabedian.sp

This SPEC input file describes a circular-cross-section torus with
some torsion of the magnetic axis, for a 1-volume vacuum field.

---

    2DOF_targetIotaAndVolume.sp

This SPEC input file describes a classical stellarator with rotating
elliptical cross-section, for a 1-volume vacuum field.

---

    QH-residues.sp

This SPEC input file describes the WISTELL-A configuration in Bader et
al, Journal of Plasma Physics 86, 905860506 (2020). It uses a 1-volume
vacuum field model.

---

    input.circular_tokamak
    boozmn_circular_tokamak.nc

This VMEC input file and BOOZ_XFORM output file are for an
axisymmetric tokamak with circular cross-section, using ITER-like
parameters (but not the ITER poloidal shaping). This example is useful
for testing simsopt functions for the case of axisymmetry.

---

    input.li383_low_res
    wout_li383_low_res_reference.nc
    boozmn_li383_low_res.nc
    
These VMEC input and output files and BOOZ_XFORM output file refer to
the LI383 configuration of NCSX. In contrast to the "official" version
of NCSX, the VMEC resolution parameters `mpol`, `ntor`, and `ns` have
been lowered so the tests run quickly.

---

    input.simsopt_nfp2_QA_20210328-01-020_000_000251
    input.20210406-01-002-nfp4_QH_000_000240

These VMEC input files come from previous simsopt optimizations for
quasi-axisymmetry and quasi-helical symmetry, respectively. These
files are useful for testing measures of quasisymmetry.

---

    input.LandremanSengupta2019_section5.4_B2_A80

This VMEC input file is for the quasi-helically-symmetric
configuration in section 5.4 of Landreman & Sengupta, Journal of
Plasma Physics 85, 815850601 (2019), except that the aspect ratio has
been raised to 80, and the mean field has been increased to 2
Tesla. This configuration is useful for comparing to analytic results
from the near-axis expansion.

---

    input.LandremanSenguptaPlunk_section5p3
    wout_LandremanSenguptaPlunk_section5p3_reference.nc

These VMEC input and output files refer to the configuration in
section 5.3 of Landreman, Sengupta, and Plunk, Journal of Plasma
Physics 85, 905850103 (2019). This configuration is not
stellarator-symmetric, so these files are useful for testing that
simsopt functions work for non-stellarator-symmetric geometries.

---

    input.NuhrenbergZille_1988_QHS

This VMEC input file is for the first quasisymmetric configuration to
be found, from Nuhrenberg and Zille (1988).

---

    input.W7-X_standard_configuration

This VMEC input file is for the W7-X standard configuration with no
plasma pressure or current.

---

    input.cfqs_2b40

This VMEC input file is for a configuration of the CFQS experiment.

---

    input.rotating_ellipse

This VMEC input file is for a classical stellarator with rotating
elliptical cross-section.

---

    tf_only_half_tesla.plasma

This FOCUS input file gives the NCSX plasma boundary, as well as the
component of B normal to this boundary due to a 0.5 Tesla purely
toroidal field.

