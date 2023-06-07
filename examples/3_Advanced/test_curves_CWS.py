import os
import numpy as np
from pathlib import Path
from simsopt import load
from simsopt.mhd import Vmec
from simsopt.geo import curves_to_vtk
from simsopt.objectives import SquaredFlux
this_path = str(Path(__file__).parent.resolve())

nphi = 128
ntheta = 64
nfp_array = [2, 3]

## This assumes that the script is run from the examples/3_Advanced/ folder
for nfp in nfp_array:
    if nfp == 2:
        coils_file = os.path.join(this_path, 'optimization_cws_singlestage_nfp2_QA_ncoils3_axiTorus', 'coils', 'biot_savart_opt_maxmode3.json')
        vmec_file = os.path.join(this_path, 'optimization_cws_singlestage_nfp2_QA_ncoils3_axiTorus', 'input.maxmode3')
    elif nfp == 3:
        coils_file = os.path.join(this_path, 'optimization_cws_singlestage_nfp3_QA_ncoils4_axiTorus', 'coils', 'biot_savart_opt_maxmode4.json')
        vmec_file = os.path.join(this_path, 'optimization_cws_singlestage_nfp3_QA_ncoils4_axiTorus', 'input.maxmode4')
    else:
        raise ValueError("nfp not supported")

    vmec = Vmec(vmec_file, nphi=nphi, ntheta=ntheta, range_surface="full torus", verbose=False)
    surf = vmec.boundary

    if os.path.exists(coils_file):
        bs = load(coils_file)
        coils = bs.coils
        curves = [coil._curve for coil in coils]
        currents = [coil._current for coil in coils]
        ncoils = int(len(bs.coils)/surf.nfp/2)
        base_curves = [bs.coils[i]._curve for i in range(ncoils)]
        base_currents = [bs.coils[i]._current for i in range(ncoils)]
    else:
        raise ValueError("Coils file not found")

    Jf = SquaredFlux(surf, bs, definition="local")
    bs.set_points(surf.gamma().reshape((-1, 3)))
    BdotN = np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)
    pointData = {"B_N": BdotN[:, :, None]}
    surf.to_vtk(os.path.join(this_path, f'surface_nfp{nfp}'), extra_data=pointData)
    curves_to_vtk(curves, os.path.join(this_path, f'coils_nfp{nfp}'))

    BdotNmean = np.mean(np.abs(BdotN))
    BdotNmax = np.max(np.abs(BdotN))
    outstr = f"nfp={nfp}: Jf={Jf.J():.1e}, ⟨B·n⟩={BdotNmean:.1e}, B·n max={BdotNmax:.1e}"
    print(outstr)