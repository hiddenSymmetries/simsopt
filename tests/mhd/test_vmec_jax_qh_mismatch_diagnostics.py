import os
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("vmec_jax")

try:
    import vmec as vmec_mod  # noqa: F401
except Exception:
    vmec_mod = None

import vmec_jax as vj
from vmec_jax.driver import _final_flux_profiles_from_state
from vmec_jax.state import VMECState
from vmec_jax.wout import state_from_wout

from simsopt.mhd import Vmec


def _input_filename():
    filename = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "examples",
        "2_Intermediate",
        "inputs",
        "input.nfp4_QH_warm_start",
    )
    return os.path.abspath(filename)


def _reference_case():
    filename = _input_filename()

    vmec = Vmec(filename, verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3
    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)")
    vmec.run()

    wref = vj.load_wout(vmec.output_file)
    state_ref = state_from_wout(wref)

    cfg, indata = vj.load_config(filename)
    cfg = replace(cfg, mpol=3, ntor=3)
    static = vj.build_static(cfg)
    boundary = vj.boundary_from_indata(indata, static.modes, apply_m1_constraint=True)
    state0 = vj.initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    geom0 = vj.eval_geom(state0, static)
    signgs0 = vj.signgs_from_sqrtg(np.asarray(geom0.sqrtg), axis_index=1)

    flux = vj.flux_profiles_from_indata(indata, static.s, signgs=int(wref.signgs))
    s_half = np.concatenate([np.asarray(static.s)[:1], 0.5 * (np.asarray(static.s)[1:] + np.asarray(static.s)[:-1])])
    prof = vj.eval_profiles(indata, s_half)
    pressure = np.asarray(prof.get("pressure", np.zeros_like(s_half)))

    return {
        "wref": wref,
        "state_ref": state_ref,
        "static": static,
        "indata": indata,
        "flux": flux,
        "prof": prof,
        "pressure": pressure,
        "state0": state0,
        "signgs0": int(signgs0),
    }


@pytest.mark.skipif(vmec_mod is None, reason="vmec not installed")
def test_reference_state_recomputes_current_driven_iota_exactly():
    case = _reference_case()
    _flux_out, prof_out = _final_flux_profiles_from_state(
        indata=case["indata"],
        static_in=case["static"],
        state=case["state_ref"],
        signgs=int(case["wref"].signgs),
        flux_local=case["flux"],
        prof_local=case["prof"],
        pressure_local=case["pressure"],
    )

    np.testing.assert_allclose(
        np.asarray(prof_out["iota"]),
        np.asarray(case["wref"].iotas),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.skipif(vmec_mod is None, reason="vmec not installed")
def test_initial_qh_mismatch_is_geometry_dominated_not_lambda_dominated():
    case = _reference_case()
    state0 = case["state0"]
    state_ref = case["state_ref"]

    initial_plus_ref_lambda = VMECState(
        layout=state0.layout,
        Rcos=state0.Rcos,
        Rsin=state0.Rsin,
        Zcos=state0.Zcos,
        Zsin=state0.Zsin,
        Lcos=state_ref.Lcos,
        Lsin=state_ref.Lsin,
    )
    ref_geom_plus_initial_lambda = VMECState(
        layout=state0.layout,
        Rcos=state_ref.Rcos,
        Rsin=state_ref.Rsin,
        Zcos=state_ref.Zcos,
        Zsin=state_ref.Zsin,
        Lcos=state0.Lcos,
        Lsin=state0.Lsin,
    )

    def iota_rel(st):
        _flux_out, prof_out = _final_flux_profiles_from_state(
            indata=case["indata"],
            static_in=case["static"],
            state=st,
            signgs=case["signgs0"],
            flux_local=case["flux"],
            prof_local=case["prof"],
            pressure_local=case["pressure"],
        )
        iota = np.asarray(prof_out["iota"])
        iota_ref = np.asarray(case["wref"].iotas)
        return np.linalg.norm(iota - iota_ref) / (np.linalg.norm(iota_ref) + 1e-30)

    rel_initial = iota_rel(state0)
    rel_ref_lambda = iota_rel(initial_plus_ref_lambda)
    rel_ref_geom = iota_rel(ref_geom_plus_initial_lambda)

    assert rel_ref_geom < 0.25
    assert rel_ref_geom < rel_initial
    assert rel_ref_geom < rel_ref_lambda


@pytest.mark.skipif(vmec_mod is None, reason="vmec not installed")
def test_initial_qh_iota_has_wrong_sign_before_residual_iterations():
    case = _reference_case()
    _flux_out, prof_out = _final_flux_profiles_from_state(
        indata=case["indata"],
        static_in=case["static"],
        state=case["state0"],
        signgs=case["signgs0"],
        flux_local=case["flux"],
        prof_local=case["prof"],
        pressure_local=case["pressure"],
    )

    iota0 = np.asarray(prof_out["iota"])
    iota_ref = np.asarray(case["wref"].iotas)

    assert iota0[1] > 0.0
    assert iota_ref[1] < 0.0
