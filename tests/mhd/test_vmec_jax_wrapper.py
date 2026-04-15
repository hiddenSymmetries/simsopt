import os
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np
import jax

try:
    import vmec_jax as vj
except Exception as exc:
    raise unittest.SkipTest(f"vmec_jax not installed: {exc}")

try:
    import vmec as vmec_mod  # noqa: F401
except Exception:
    vmec_mod = None

from vmec_jax.implicit import _pack_stellsym_feasible_state, _stellsym_feasible_indices
from vmec_jax.solve import _mask_grad_for_constraints, _mode00_index
from vmec_jax.boundary import boundary_from_input_convention

from simsopt.mhd import VmecJax
from simsopt.mhd import Vmec
from simsopt.solve import build_vmec_objective_stage


@contextmanager
def _temporary_cwd():
    """Run a test inside a temporary working directory."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            yield tmpdir
        finally:
            os.chdir(cwd)


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
    np.testing.assert_allclose(surf2.x[idx], x_before[surf.dof_names.index("rc(0,1)")])


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
    np.testing.assert_allclose(vmec._grad_tol, 1.0e-13)
    assert vmec._warm_start_iters == 0


def test_vmec_jax_qh_profile_applies_outer_optimization_defaults():
    vmec = VmecJax(_input_filename(), verbose=False)

    vmec.use_residual_autodiff_defaults(
        outer_method="gradient_descent",
        optimization_profile="qh",
    )

    assert vmec._solver == "vmec2000"
    assert vmec._max_iter == 1500
    np.testing.assert_allclose(vmec._grad_tol, 1.0e-13)
    assert vmec._residual_adjoint_mode == "chunked"
    assert vmec._residual_tangent_mode == "opaque"
    assert not vmec._stateless_evaluations


def test_vmec_jax_can_select_discrete_adjoint_backend():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.set_solver_options(residual_derivative_backend="discrete_adjoint")
    assert vmec._residual_derivative_backend == "discrete_adjoint"


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


@unittest.skipIf(vmec_mod is None, "vmec not installed")
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


def test_vmec_jax_solver_option_change_resets_warm_start_seed():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3

    context = vmec.get_context()
    seed_before = np.asarray(context.st_guess.Rcos)
    context.st_guess = vj.initial_guess_from_boundary(vmec.get_static(), vmec.boundary._boundary0, vmec._indata_raw, vmec_project=False)

    vmec.set_solver_options(max_iter=123)

    np.testing.assert_allclose(np.asarray(vmec.get_context().st_guess.Rcos), seed_before)


def test_vmec_jax_aspect_jax_is_jittable_for_qh_setup():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)")
    x0 = jax.numpy.asarray(surf.get_free_params())

    aspect = jax.jit(vmec.aspect_jax)(x0)
    assert np.isfinite(float(np.asarray(aspect)))


def test_vmec_jax_qh_start_objective_matches_converged_control_path():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)")

    stage = build_vmec_objective_stage(
        vmec,
        max_mode=1,
        objective_tuples=[("aspect", 7.0, 1.0), ("qs", 0.0, 1.0)],
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
        x_scale_alpha=1.2,
        x_scale_min=1e-9,
    )
    x0 = jax.numpy.asarray(stage.x0, dtype=jax.numpy.float64)
    residual = np.asarray(stage.residuals(x0))
    objective = float(np.sum(residual * residual))

    np.testing.assert_allclose(objective, 0.2983122217172473, rtol=0.0, atol=1.0e-12)


def test_vmec_jax_qh_qs_state_gradient_is_finite():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)")

    stage = build_vmec_objective_stage(
        vmec,
        max_mode=1,
        objective_tuples=[("aspect", 7.0, 1.0), ("qs", 0.0, 1.0)],
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
        x_scale_alpha=1.2,
        x_scale_min=1e-9,
    )
    state = vmec._solve_state(jax.numpy.asarray(stage.x0, dtype=jax.numpy.float64))
    qs = stage.extras["qs"]
    static = vmec.get_static()
    idx00 = int(_mode00_index(static.modes))
    rz_idx, lam_idx, _ns, _K = _stellsym_feasible_indices(static, idx00=idx00, mask_lambda_axis=True)

    def qs_objective(state):
        residual = jax.numpy.asarray(qs.residuals_from_state(state), dtype=jax.numpy.float64)
        return 0.5 * jax.numpy.sum(residual * residual)

    grad = jax.grad(qs_objective)(state)
    grad = _mask_grad_for_constraints(grad, static, idx00=idx00, mask_lambda_axis=True)
    packed = np.asarray(_pack_stellsym_feasible_state(grad, rz_idx=rz_idx, lam_idx=lam_idx))

    assert np.all(np.isfinite(packed))


def test_vmec_jax_discrete_backend_aspect_directional_derivative_matches_fd():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3
    vmec.set_solver_options(
        solver="vmec2000",
        max_iter=1,
        grad_tol=1.0e-13,
        residual_derivative_backend="discrete_adjoint",
    )

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)")
    x0 = jax.numpy.asarray(surf.get_free_params(), dtype=jax.numpy.float64)
    direction = jax.numpy.zeros_like(x0).at[0].set(1.0)

    def aspect_fn(x):
        return vmec.aspect_equilibrium_jax(x)

    grad = jax.grad(aspect_fn)(x0)
    ad = float(np.asarray(jax.numpy.dot(grad, direction)))
    eps = 1.0e-5
    fd = (
        float(np.asarray(aspect_fn(x0 + eps * direction)))
        - float(np.asarray(aspect_fn(x0 - eps * direction)))
    ) / (2.0 * eps)

    np.testing.assert_allclose(ad, fd, rtol=1.0e-5, atol=1.0e-8)


def test_vmec_jax_discrete_backend_reuses_exact_solve_between_residual_and_jacobian():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3
    vmec.set_solver_options(
        solver="vmec2000",
        max_iter=1,
        grad_tol=1.0e-13,
        residual_derivative_backend="discrete_adjoint",
    )

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)")

    stage = build_vmec_objective_stage(
        vmec,
        max_mode=1,
        objective_tuples=[("aspect", 7.0, 1.0), ("qs", 0.0, 1.0)],
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
        x_scale_alpha=1.2,
        x_scale_min=1e-9,
    )
    x0 = jax.numpy.asarray(stage.x0, dtype=jax.numpy.float64)

    solve_calls = {"count": 0}
    original = vmec._solve_state_discrete_adjoint_residual

    def counted_solve(*args, **kwargs):
        solve_calls["count"] += 1
        return original(*args, **kwargs)

    vmec._solve_state_discrete_adjoint_residual = counted_solve
    try:
        residual = np.asarray(stage.residuals.scipy_residuals(x0))
        jacobian = np.asarray(stage.residuals.scipy_jacobian(x0))
    finally:
        vmec._solve_state_discrete_adjoint_residual = original

    assert solve_calls["count"] == 1
    assert residual.ndim == 1
    assert jacobian.shape[0] == residual.shape[0]
    assert jacobian.shape[1] == x0.size


def test_vmec_jax_discrete_backend_qh_jacobian_matches_frozen_axis_fd():
    if os.environ.get("RUN_SLOW", "") != "1":
        raise unittest.SkipTest("Set RUN_SLOW=1 to run slow discrete-adjoint QH checks")

    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3
    vmec.set_solver_options(
        solver="vmec2000",
        max_iter=1,
        grad_tol=1.0e-13,
        residual_derivative_backend="discrete_adjoint",
    )
    vmec.use_residual_autodiff_defaults(
        outer_method="scipy",
        residual_adjoint_mode="chunked",
        stateless_evaluations=False,
        optimization_profile="qh",
    )

    stage = build_vmec_objective_stage(
        vmec,
        max_mode=1,
        objective_tuples=[("aspect", 7.0, 1.0), ("qs", 0.0, 1.0)],
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
        x_scale_alpha=1.2,
        x_scale_min=1e-9,
    )
    x0 = np.asarray(stage.x0, dtype=float)
    residual0 = np.asarray(stage.residuals.scipy_residuals(x0), dtype=float)
    jacobian = np.asarray(stage.residuals.scipy_jacobian(x0), dtype=float)

    state0, payload0 = vmec._solve_state_discrete_adjoint_residual(
        x0,
        step_size=float(vmec._indata_raw.get_float("DELT", 1.0)),
        return_payload=True,
    )
    axis_override = payload0["axis_override"]

    static = vmec._static
    boundary_wrapper = vmec.boundary
    indata = vmec._indata_raw
    base_full = np.asarray(boundary_wrapper._base)
    specs = tuple(boundary_wrapper._specs)
    modes = boundary_wrapper._modes
    lasym = bool(vmec._cfg.lasym)
    layout = state0.layout
    step_size = float(vmec._indata_raw.get_float("DELT", 1.0))
    qs = stage.extras["qs"]

    def initial_state_packed(x_free, *, axis_override_local=None):
        x_full = boundary_wrapper.expand_free(x_free)
        delta_full = np.asarray(x_full) - base_full
        boundary_input = vj.apply_boundary_params(boundary_wrapper._boundary_input0, specs, delta_full)
        boundary = boundary_from_input_convention(
            boundary_input,
            modes,
            lasym=lasym,
            apply_m1_constraint=False,
        )
        state = vj.initial_guess_from_boundary(
            static,
            boundary,
            indata,
            vmec_project=True,
            axis_override=axis_override_local,
        )
        return np.asarray(vj.pack_state(state), dtype=float)

    def solve_frozen_axis(x_free):
        packed0 = initial_state_packed(x_free, axis_override_local=axis_override)
        state_init = vj.unpack_state(packed0, layout)
        tape = vj.build_residual_checkpoint_tape_direct(
            state_init,
            static,
            max_iter=int(vmec._max_iter),
            solver_kwargs=dict(
                indata=indata,
                signgs=int(vmec._signgs),
                ftol=float(vmec._grad_tol),
                step_size=step_size,
                vmec2000_control=True,
                reference_mode=False,
                backtracking=True,
                limit_dt_from_force=True,
                limit_update_rms=True,
                verbose=False,
                verbose_vmec2000_table=False,
                jit_forces="auto",
                use_scan=False,
                light_history=True,
                resume_state_mode="full",
            ),
            indata=indata,
            signgs=int(vmec._signgs),
            ftol=float(vmec._grad_tol),
            step_size=step_size,
            light_history=True,
            store_trace=False,
        )
        return vj.unpack_state(np.asarray(tape.final_packed_state, dtype=float), layout)

    def frozen_axis_residuals(x_free):
        state = solve_frozen_axis(x_free)
        aspect_residual = np.asarray([vmec.aspect_equilibrium_from_state_jax(state) - 7.0], dtype=float)
        qs_residual = np.asarray(qs.residuals_from_state(state), dtype=float)
        return np.concatenate((aspect_residual, qs_residual))

    eps = 1.0e-6
    direction = np.zeros_like(x0)
    direction[0] = 1.0
    residual_plus = frozen_axis_residuals(x0 + eps * direction)
    residual_minus = frozen_axis_residuals(x0 - eps * direction)
    fd_column = (residual_plus - residual_minus) / (2.0 * eps)

    rel_column_error = np.linalg.norm(jacobian[:, 0] - fd_column) / max(np.linalg.norm(fd_column), 1.0e-30)
    assert rel_column_error < 5.0e-4
    assert float(np.max(np.abs(jacobian[:, 0] - fd_column))) < 1.0e-3

    ad_objective_direction = float(2.0 * residual0.dot(jacobian[:, 0]))
    fd_objective_direction = float(2.0 * residual0.dot(fd_column))
    np.testing.assert_allclose(ad_objective_direction, fd_objective_direction, rtol=5.0e-4, atol=1.0e-5)


def test_vmec_jax_aspect_matches_vmec_interface():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3
    vmec.set_solver_options(solver="vmec2000", warm_start_iters=0, max_iter=100, grad_tol=1.0e-3)

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)")
    x0 = surf.get_free_params()

    np.testing.assert_allclose(vmec.aspect(x0), vmec.aspect_equilibrium(x0))


def test_vmec_jax_equilibrium_aspect_jax_matches_wout():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3
    vmec.set_solver_options(solver="vmec2000", warm_start_iters=0, max_iter=100, grad_tol=1.0e-3)

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)")
    x0 = jax.numpy.asarray(surf.get_free_params())

    aspect_jax = vmec.aspect_equilibrium_jax(x0)
    aspect_wout = vmec.get_wout(x0).aspect
    np.testing.assert_allclose(np.asarray(aspect_jax), np.asarray(aspect_wout), rtol=1e-10, atol=1e-10)


def test_vmec_jax_mean_iota_from_state_matches_wout():
    vmec = VmecJax(_input_filename(), verbose=False)
    vmec.indata.mpol = 3
    vmec.indata.ntor = 3
    vmec.set_solver_options(solver="vmec2000", warm_start_iters=0, max_iter=100, grad_tol=1.0e-3)

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
    surf.fix("rc(0,0)")
    x0 = jax.numpy.asarray(surf.get_free_params())

    state = vmec._solve_state(x0)
    mean_iota_jax = vmec.mean_iota_from_state_jax(state)
    iotas_wout = np.asarray(vmec.get_wout(x0).iotas)
    mean_iota_wout = 0.0 if iotas_wout.size <= 1 else float(np.mean(iotas_wout[1:]))
    np.testing.assert_allclose(np.asarray(mean_iota_jax), mean_iota_wout, rtol=1e-10, atol=1e-10)



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


@unittest.skipIf(vmec_mod is None, "vmec not installed")
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
