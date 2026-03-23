import os

import numpy as np

import pytest

vj = pytest.importorskip("vmec_jax")

try:
    import vmec as vmec_mod  # noqa: F401
except Exception:
    vmec_mod = None

from simsopt.mhd import VmecJax
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


def test_vmec_jax_lazy_static_and_context():
    vmec = VmecJax(_input_filename(), verbose=False)

    assert vmec._static is None
    assert vmec._context is None
    assert vmec._static_dirty
    assert vmec._context_dirty

    vmec.indata.mpol = 3
    vmec.indata.ntor = 3

    assert vmec._static is None
    assert vmec._context is None
    assert vmec._static_dirty
    assert vmec._context_dirty

    boundary = vmec.boundary
    assert boundary is vmec._boundary
    assert vmec._static is not None
    assert vmec._context is None
    assert not vmec._static_dirty
    assert vmec._context_dirty


def test_vmec_jax_resolution_change_preserves_matching_dofs():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3

    surf = vmec.boundary
    surf.fix_all()
    surf.unfix("rc(0,1)")
    x_before = surf.x
    free_before = surf.free

    vmec.indata.mpol = 4
    vmec.indata.ntor = 4
    surf2 = vmec.boundary

    idx = surf2.dof_names.index("rc(0,1)")
    assert surf2.free[idx] == free_before[surf.dof_names.index("rc(0,1)")]
    assert surf2.x[idx] == pytest.approx(x_before[surf.dof_names.index("rc(0,1)")])


def test_vmec_jax_solver_option_change_invalidates_cache():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3
    _ = vmec.boundary

    vmec._cached_x = (1.0,)
    vmec._cached_state = object()
    vmec._cached_wout = object()
    vmec._cached_run = object()

    vmec.set_solver_options(max_iter=123, implicit_cg_tol=1.0e-8, step_size=1.0e-2)

    assert vmec._cached_x is None
    assert vmec._cached_state is None
    assert vmec._cached_wout is None
    assert vmec._cached_run is None


def test_vmec_jax_defaults_honor_staged_input_controls():
    vmec = VmecJax(_input_filename(), verbose=False)

    assert vmec._max_iter == 1500
    assert vmec._grad_tol == pytest.approx(1.0e-13)
    assert vmec._warm_start_iters == 0


def test_vmec_jax_wrapper_keeps_unconstrained_boundary_coefficients():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3

    static = vmec.get_static()
    boundary = vmec.boundary._boundary0
    boundary_expected = vj.boundary_from_indata(vmec._indata_raw, static.modes, apply_m1_constraint=False)
    boundary_wrong = vj.boundary_from_indata(vmec._indata_raw, static.modes, apply_m1_constraint=True)

    np.testing.assert_allclose(np.asarray(boundary.R_cos), np.asarray(boundary_expected.R_cos))
    np.testing.assert_allclose(np.asarray(boundary.Z_sin), np.asarray(boundary_expected.Z_sin))
    assert not np.allclose(np.asarray(boundary.R_cos), np.asarray(boundary_wrong.R_cos))
    assert not np.allclose(np.asarray(boundary.Z_sin), np.asarray(boundary_wrong.Z_sin))


@pytest.mark.skipif(vmec_mod is None, reason="vmec not installed")
def test_vmec_jax_wrapper_initial_guess_has_correct_qh_iota_sign():
    from vmec_jax.driver import _final_flux_profiles_from_state

    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3

    static = vmec.get_static()
    boundary = vmec.boundary._boundary0
    st = vj.initial_guess_from_boundary(static, boundary, vmec._indata_raw, vmec_project=True)
    geom = vj.eval_geom(st, static)
    signgs = int(vj.signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
    flux = vj.flux_profiles_from_indata(vmec._indata_raw, static.s, signgs=signgs)
    s_half = np.concatenate([np.asarray(static.s)[:1], 0.5 * (np.asarray(static.s)[1:] + np.asarray(static.s)[:-1])])
    prof = vj.eval_profiles(vmec._indata_raw, s_half)
    pressure = np.asarray(prof.get("pressure", np.zeros_like(s_half)))
    _, prof_out = _final_flux_profiles_from_state(
        indata=vmec._indata_raw,
        static_in=static,
        state=st,
        signgs=signgs,
        flux_local=flux,
        prof_local=prof,
        pressure_local=pressure,
    )

    iota = np.asarray(prof_out["iota"])
    assert iota[1] < 0.0


def test_vmec_jax_context_uses_plain_initial_guess_by_default():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3

    static = vmec.get_static()
    boundary = vmec.boundary._boundary0
    expected = vj.initial_guess_from_boundary(static, boundary, vmec._indata_raw, vmec_project=True)
    context = vmec.get_context()

    np.testing.assert_allclose(np.asarray(context.st_guess.Rcos), np.asarray(expected.Rcos))
    np.testing.assert_allclose(np.asarray(context.st_guess.Zsin), np.asarray(expected.Zsin))
    np.testing.assert_allclose(np.asarray(context.st_guess.Lsin), np.asarray(expected.Lsin))


def test_vmec_jax_wout_exposes_mode_count_metadata():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3
    vmec.set_solver_options(solver="initial", warm_start_iters=0)

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)")
    x0 = surf.get_free_params()

    wout = vmec.get_wout(x0)

    assert int(wout.mnmax) == len(np.asarray(wout.xm))
    assert int(wout.mnmax_nyq) == len(np.asarray(wout.xm_nyq))
    assert int(wout.mpol_nyq) == int(np.max(np.asarray(wout.xm_nyq)))
    assert int(wout.ntor_nyq) == int(np.max(np.abs(np.asarray(wout.xn_nyq) // int(wout.nfp))))


@pytest.mark.skipif(vmec_mod is None, reason="vmec not installed")
def test_vmec_jax_boundary_free_modes_match_vmec_for_stellsym_qh_setup():
    vmec_ref = Vmec(_input_filename(), verbose=False)
    vmec_ref.indata.mpol = 3
    vmec_ref.indata.ntor = 3
    surf_ref = vmec_ref.boundary
    surf_ref.fix_all()
    surf_ref.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf_ref.fix("rc(0,0)")
    free_ref = [name.split(":", 1)[-1] for name in surf_ref.dof_names]

    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3
    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)")
    free = [name for name, is_free in zip(surf.dof_names, surf.free) if is_free]

    assert free == free_ref
