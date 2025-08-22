from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from scipy.constants import mu_0

from simsopt.field import BiotSavart, DipoleField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.solve import GPMO
from simsopt.util import (
    FocusData, discretize_polarizations, polarization_axes, in_github_actions
)
from simsopt.util.permanent_magnet_helper_functions import (
    initialize_coils, calculate_modB_on_major_radius, initialize_default_kwargs,
)
from simsopt.solve.macromag import MacroMag, Tiles


def _write_vtp_points_with_vec_and_scalar(pts, vec, scal, vec_name, scal_name, fname: Path):
    pts   = np.asarray(pts, float)
    vec   = np.asarray(vec, float)
    scal  = np.asarray(scal, float).ravel()
    n = pts.shape[0]
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" byte_order="LittleEndian">\n')
        f.write(f' <PolyData>\n  <Piece NumberOfPoints="{n}" NumberOfVerts="{n}">\n')
        f.write('   <Verts>\n')
        f.write('    <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write(" ".join(map(str, range(n))) + "\n")
        f.write('    </DataArray>\n')
        f.write('    <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write(" ".join(str(i+1) for i in range(n)) + "\n")
        f.write('    </DataArray>\n')
        f.write('   </Verts>\n')
        f.write('   <Points>\n')
        f.write('    <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for x, y, z in pts:
            f.write(f"{x:.16e} {y:.16e} {z:.16e}\n")
        f.write('    </DataArray>\n')
        f.write('   </Points>\n')
        f.write('   <PointData>\n')
        f.write(f'    <DataArray type="Float64" Name="{vec_name}" NumberOfComponents="3" format="ascii">\n')
        for vx, vy, vz in vec:
            f.write(f"{vx:.16e} {vy:.16e} {vz:.16e}\n")
        f.write('    </DataArray>\n')
        f.write(f'    <DataArray type="Float64" Name="{scal_name}" format="ascii">\n')
        for s in scal:
            f.write(f"{s:.16e}\n")
        f.write('    </DataArray>\n')
        f.write('   </PointData>\n')
        f.write('  </Piece>\n')
        f.write(' </PolyData>\n')
        f.write('</VTKFile>\n')


def magnetization_to_moment(M, sizes):
    sizes = np.asarray(sizes)
    vol = np.prod(sizes, axis=1)
    return M * vol[:, None]


def _vec_to_sph(u):
    u = np.asarray(u, float)
    nz = np.clip(u[2], -1.0, 1.0)
    theta = float(np.arccos(nz))
    phi = float(np.arctan2(u[1], u[0]))
    return theta, phi


# ---- basis helpers ----
def _cyl_basis_vectors(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    phi = np.arctan2(y, x)
    c, s = np.cos(phi), np.sin(phi)
    eR   = np.stack([ c,  s, np.zeros_like(phi)], axis=1)
    ePHI = np.stack([-s,  c, np.zeros_like(phi)], axis=1)
    eZ   = np.tile(np.array([0.0, 0.0, 1.0]), (xyz.shape[0], 1))
    return eR, ePHI, eZ

def _to_xyz(vecs, xyz, coordinate_flag: str):
    cf = (coordinate_flag or "cartesian").lower()
    vecs = np.asarray(vecs, float)
    xyz  = np.asarray(xyz, float)
    if cf.startswith("cart"):
        return vecs
    if cf.startswith("cyl"):
        eR, ePHI, eZ = _cyl_basis_vectors(xyz)
        return vecs[:, [0]] * eR + vecs[:, [1]] * ePHI + vecs[:, [2]] * eZ
    raise ValueError(f"Unsupported coordinate_flag '{coordinate_flag}'")
# -----------------------


def build_tiles_from_gpmo_selection(pm_opt, m_gpmo, mu_par=1.05, mu_perp=1.05, B_max=1.465):
    """Easy-axis = GPMO orientation (± tried later)."""
    xyz_all = pm_opt.dipole_grid_xyz
    m_all   = m_gpmo.reshape(pm_opt.ndipoles, 3)
    m_cart  = _to_xyz(m_all, xyz_all, pm_opt.coordinate_flag)
    amp_all = np.linalg.norm(m_cart, axis=1)
    idx_active = np.where(amp_all > 0)[0]
    if len(idx_active) == 0:
        raise RuntimeError("GPMO returned zero magnets; nothing to refine.")

    u = m_cart[idx_active] / (amp_all[idx_active][:, None] + 1e-300)

    M_max = B_max / mu_0
    mmax = pm_opt.m_maxima
    if np.isscalar(mmax):
        mmax = np.full(pm_opt.ndipoles, float(mmax))
    else:
        mmax = np.asarray(mmax).ravel()
    V = np.maximum(mmax[idx_active] / M_max, 1e-18)
    edge = V ** (1/3.0)
    sizes = np.stack([edge, edge, edge], axis=1)

    n = len(idx_active)
    tiles = Tiles(n)
    tiles.tile_type = 2
    tiles.size = sizes
    tiles.offset = xyz_all[idx_active]
    tiles.rot = np.zeros((n, 3))
    tiles.mu_r_ea = mu_par
    tiles.mu_r_oa = mu_perp
    tiles.M_rem = np.minimum(amp_all[idx_active] / V, M_max)
    for i in range(n):
        tiles.set_easy_axis(val=_vec_to_sph(u[i]), idx=i)
    return tiles, idx_active, M_max, sizes


def _ok_distance(centers, active, i, min_sep):
    if min_sep <= 0 or len(active) == 0:
        return True
    d2 = np.sum((centers[active] - centers[i])**2, axis=1)
    return (np.sqrt(d2).min() >= min_sep)


def sequential_macromag_refine_monotone(
    tiles_all: Tiles,
    N_full: np.ndarray,
    order_idx: np.ndarray,
    s_plot: SurfaceRZFourier,
    Bnormal_plot: np.ndarray,
    nfp: int,
    coordinate_flag: str,
    min_sep: float = 0.0,
    verbose: bool = True,
    bs_coils: BiotSavart | None = None,
):
    centres = tiles_all.offset
    chosen = []
    chosen_u = {}
    f_curr = np.inf
    M_final = None

    def build_sub(active_idx, u_override=None):
        k = len(active_idx)
        sub = Tiles(k)
        sub.tile_type = 2
        sub.size   = tiles_all.size[active_idx]
        sub.offset = tiles_all.offset[active_idx]
        sub.rot    = tiles_all.rot[active_idx]
        sub.mu_r_ea = tiles_all.mu_r_ea[active_idx]
        sub.mu_r_oa = tiles_all.mu_r_oa[active_idx]
        sub.M_rem   = tiles_all.M_rem[active_idx]
        for j, src in enumerate(active_idx):
            u = (u_override[src] if (u_override and src in u_override) else tiles_all.u_ea[src])
            sub.set_easy_axis(val=_vec_to_sph(u), idx=j)
        return sub

    def f_of_sub(sub, demag_sub):
        mm = MacroMag(sub)
        if bs_coils is not None:
            mm._bs_coil = bs_coils
        mm.direct_solve(
            Ms=0.0, K=0.0, const_H=(0.0, 0.0, 0.0),
            use_coils=(bs_coils is not None), demag_only=False,
            krylov_tol=1e-4, krylov_it=400,
            demag_tensor=demag_sub, print_progress=False
        )
        Mtest = sub.M.copy()
        m_test = magnetization_to_moment(Mtest, sub.size)
        b = DipoleField(sub.offset, m_test.ravel(), nfp=nfp,
                        coordinate_flag=coordinate_flag)
        b.set_points(s_plot.gamma().reshape((-1, 3)))
        return SquaredFlux(s_plot, b, -Bnormal_plot).J(), Mtest

    accepted_any = False
    sep = float(min_sep)
    for i in order_idx:
        if not _ok_distance(centres, chosen, i, sep):
            if verbose:
                print(f"  skip idx {i} (closer than {sep*100:.0f} mm)")
            continue
        trial_idx = chosen + [i]
        N_sub = N_full[np.ix_(trial_idx, trial_idx)]

        u_plus = dict(chosen_u);  u_plus[i]  =  tiles_all.u_ea[i]
        u_minus = dict(chosen_u); u_minus[i] = -tiles_all.u_ea[i]

        sub_plus  = build_sub(trial_idx, u_plus)
        sub_minus = build_sub(trial_idx, u_minus)

        f_plus,  M_plus  = f_of_sub(sub_plus,  N_sub)
        f_minus, M_minus = f_of_sub(sub_minus, N_sub)

        if f_plus <= f_minus:
            f_try, M_try, u_chosen = f_plus, M_plus, u_plus
        else:
            f_try, M_try, u_chosen = f_minus, M_minus, u_minus

        if f_try <= f_curr:
            chosen = trial_idx
            chosen_u = u_chosen
            f_curr = f_try
            M_final = M_try.copy()
            accepted_any = True
            if verbose and (len(chosen) % 50 == 0 or len(chosen) <= 10):
                print(f"  accepted {len(chosen)} magnets | f_B={f_curr:.6e}")
        else:
            if verbose:
                print(f"  reject idx {i} (no decrease)")

    if (not accepted_any) and (min_sep > 0):
        if verbose:
            print(f"  no magnets accepted with min_sep={min_sep:.3f}m → retry with {0.5*min_sep:.3f}m")
        return sequential_macromag_refine_monotone(
            tiles_all, N_full, order_idx, s_plot, Bnormal_plot,
            nfp, coordinate_flag, min_sep=0.5*min_sep, verbose=verbose, bs_coils=bs_coils
        )

    return np.array(chosen, dtype=int), M_final, chosen_u


def main():
    t_start = time.time()

    if in_github_actions:
        nphi = 2; nIter_max = 100; nBacktracking = 50; max_nMagnets = 20; downsample = 100
    else:
        nphi = 16
        nIter_max = 10000
        nBacktracking = 200
        max_nMagnets = 1000
        downsample = 1

    ntheta = nphi
    dr = 0.01
    input_name = 'input.muse'

    TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
    famus_filename = TEST_DIR / 'zot80.focus'
    surface_filename = TEST_DIR / input_name

    s = SurfaceRZFourier.from_focus(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

    qphi = 2 * nphi
    s_plot = SurfaceRZFourier.from_focus(
        surface_filename,
        quadpoints_phi=np.linspace(0, 1, qphi, endpoint=True),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=True),
        range="half period"
    )

    out_dir = Path("output_permanent_magnet_GPMO_MUSE_with_macromag")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_curves, curves, coils = initialize_coils('muse_famus', TEST_DIR, s, out_dir)
    bs = BiotSavart(coils)
    calculate_modB_on_major_radius(bs, s)

    bs.set_points(s.gamma().reshape((-1, 3)))
    Bnormal_s = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal_plot = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)

    mag_data = FocusData(famus_filename, downsample=downsample)
    pol_axes = np.zeros((0, 3))
    pol_type = np.zeros(0, dtype=int)
    pol_axes_f, pol_type_f = polarization_axes(['face'])
    ntype_f = len(pol_type_f) // 2
    pol_axes_f = pol_axes_f[:ntype_f, :]
    pol_type_f = pol_type_f[:ntype_f]
    pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
    pol_type = np.concatenate((pol_type, pol_type_f))
    ophi = np.arctan2(mag_data.oy, mag_data.ox)
    discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
    pol_vectors = np.zeros((mag_data.nMagnets, len(pol_type), 3))
    pol_vectors[:, :, 0] = mag_data.pol_x
    pol_vectors[:, :, 1] = mag_data.pol_y
    pol_vectors[:, :, 2] = mag_data.pol_z
    print('pol_vectors_shape = ', pol_vectors.shape)
    kwargs = {"pol_vectors": pol_vectors, "downsample": downsample, "dr": dr}

    pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal_s, famus_filename, **kwargs)
    print('Number of available dipoles = ', pm_opt.ndipoles)
    print(f"[INFO] coordinate_flag = {pm_opt.coordinate_flag}")

    algorithm = 'ArbVec_backtracking'
    kwargs_gpmo = initialize_default_kwargs('GPMO')
    kwargs_gpmo['K'] = nIter_max
    kwargs_gpmo['nhistory'] = 20
    kwargs_gpmo['backtracking'] = nBacktracking
    kwargs_gpmo['Nadjacent'] = 1
    kwargs_gpmo['dipole_grid_xyz'] = np.ascontiguousarray(pm_opt.dipole_grid_xyz)
    kwargs_gpmo['max_nMagnets'] = max_nMagnets
    kwargs_gpmo['thresh_angle'] = np.pi

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "stage1_gpmo_cache.npz"
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        m_gpmo = data["m_gpmo"]
        if m_gpmo.size != 3 * pm_opt.ndipoles:
            print("Stage-1 cache mismatch; recomputing.")
            cache_file.unlink(missing_ok=True)
            t1 = time.time()
            R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs_gpmo)
            t2 = time.time()
            print('GPMO took t = ', t2 - t1, ' s')
            min_ind = np.argmin(R2_history)
            m_gpmo = np.ravel(m_history[:, :, min_ind])
            np.savez(cache_file, m_gpmo=m_gpmo)
        else:
            print(f"Loaded Stage-1 cache: {cache_file}")
    else:
        t1 = time.time()
        R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm, **kwargs_gpmo)
        t2 = time.time()
        print('GPMO took t = ', t2 - t1, ' s')
        min_ind = np.argmin(R2_history)
        m_gpmo = np.ravel(m_history[:, :, min_ind])
        np.savez(cache_file, m_gpmo=m_gpmo)

    b_dipole_gpmo = DipoleField(
        pm_opt.dipole_grid_xyz, m_gpmo, nfp=s.nfp,
        coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima
    )
    b_dipole_gpmo.set_points(s_plot.gamma().reshape((-1, 3)))
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    f_B_gpmo = SquaredFlux(s_plot, b_dipole_gpmo, -Bnormal_plot).J()
    print('f_B (GPMO) = ', f_B_gpmo)

    # ---- post-GPMO visualization in XYZ, and wrong-basis diagnostic
    m_pre = m_gpmo.reshape(pm_opt.ndipoles, 3)
    idx_nz_pre = np.flatnonzero(np.linalg.norm(m_pre, axis=1) > 0)
    pts_pre = pm_opt.dipole_grid_xyz[idx_nz_pre]
    vec_pre = m_pre[idx_nz_pre]

    raw_unit_wrong = vec_pre / (np.linalg.norm(vec_pre, axis=1, keepdims=True) + 1e-300)
    cosX_wrong = np.abs(raw_unit_wrong @ np.array([1.0, 0.0, 0.0]))
    print(f"[DEBUG] WRONG-basis mean|cosX|={cosX_wrong.mean():.4f}, p90={np.quantile(cosX_wrong,0.9):.4f}, max={cosX_wrong.max():.4f}")

    m_gpmo_cart = _to_xyz(vec_pre, pts_pre, pm_opt.coordinate_flag)
    unit_gpmo_cart = m_gpmo_cart / (np.linalg.norm(m_gpmo_cart, axis=1, keepdims=True) + 1e-300)
    cosX_right = np.abs(unit_gpmo_cart @ np.array([1.0, 0.0, 0.0]))
    print(f"[DEBUG] XYZ-converted mean|cosX|={cosX_right.mean():.4f}, p90={np.quantile(cosX_right,0.9):.4f}, max={cosX_right.max():.4f}")

    _write_vtp_points_with_vec_and_scalar(
        pts_pre, unit_gpmo_cart, np.linalg.norm(m_gpmo_cart, axis=1),
        vec_name="unit_m", scal_name="|m|",
        fname=out_dir / "orients_post_GPMO.vtp"
    )
    b_dipole_gpmo._toVTK(out_dir / "dipoles_post_GPMO")

    mu_par = 1.00
    mu_perp = 1.00
    min_sep = 0.00
    B_max = 1.465

    print("\n[MacroMag] Building tiles from GPMO nonzero set… (easy-axis = GPMO dir)")
    tiles, idx_active_all, M_max, sizes_all = build_tiles_from_gpmo_selection(
        pm_opt, m_gpmo, mu_par=mu_par, mu_perp=mu_perp, B_max=B_max
    )
    print(f"[MacroMag] Active (nonzero) magnets: {tiles.n}")

    print("[MacroMag] Precomputing demag tensor…")
    mm_full = MacroMag(tiles)
    N_full = mm_full.fast_get_demag_tensor(cache=True)

    amp_cart_all = np.linalg.norm(_to_xyz(m_pre[idx_active_all], pm_opt.dipole_grid_xyz[idx_active_all], pm_opt.coordinate_flag), axis=1)
    order_local = np.argsort(-amp_cart_all)

    print("[MacroMag] Monotone sequential placement (± easy-axis)…")
    chosen_local_idx, M_active_local, chosen_u = sequential_macromag_refine_monotone(
        tiles_all=tiles,
        N_full=N_full,
        order_idx=order_local,
        s_plot=s_plot,
        Bnormal_plot=Bnormal_plot,
        nfp=s.nfp,
        coordinate_flag=pm_opt.coordinate_flag,
        min_sep=min_sep,
        verbose=True,
        bs_coils=bs
    )
    chosen_global_idx = idx_active_all[chosen_local_idx]

    m_macro = np.zeros_like(m_gpmo)
    sizes_sel = tiles.size[chosen_local_idx]
    m_sel = magnetization_to_moment(M_active_local, sizes_sel)
    for j, gidx in enumerate(chosen_global_idx):
        m_macro[3*gidx:3*gidx+3] = m_sel[j]

    b_dipole_macro = DipoleField(
        pm_opt.dipole_grid_xyz, m_macro, nfp=s.nfp,
        coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima
    )
    b_dipole_macro.set_points(s_plot.gamma().reshape((-1, 3)))
    f_B_macro = SquaredFlux(s_plot, b_dipole_macro, -Bnormal_plot).J()
    print('f_B (MacroMag refined, monotone) = ', f_B_macro)
    delta = f_B_macro - f_B_gpmo
    print(f"Δf_B = {delta:.6e} ({'decrease' if delta < 0 else 'increase'})")

    pts_post = pm_opt.dipole_grid_xyz[chosen_global_idx]
    M_cart = M_active_local.copy()
    unit_M_cart = M_cart / (np.linalg.norm(M_cart, axis=1, keepdims=True) + 1e-300)
    _write_vtp_points_with_vec_and_scalar(
        pts_post, unit_M_cart, np.linalg.norm(M_cart, axis=1),
        vec_name="unit_M", scal_name="|M|",
        fname=out_dir / "moments_post_MacroMag.vtp"
    )

    ea_cart = tiles.u_ea[chosen_local_idx]
    ea_cart = ea_cart / (np.linalg.norm(ea_cart, axis=1, keepdims=True) + 1e-300)
    _write_vtp_points_with_vec_and_scalar(
        pts_post, ea_cart, np.ones(ea_cart.shape[0]),
        vec_name="easy_axis", scal_name="ones",
        fname=out_dir / "easy_axes_post_MacroMag.vtp"
    )

    align = np.clip(np.sum(ea_cart * unit_M_cart, axis=1), -1.0, 1.0)
    mean_angle_deg = float(np.rad2deg(np.arccos(np.mean(align))))
    max_angle_deg  = float(np.rad2deg(np.arccos(np.min(align))))
    print(f"[DEBUG] Alignment(M vs easy-axis): mean={mean_angle_deg:.3e} deg, worst={max_angle_deg:.3e} deg")

    b_dipole_macro._toVTK(out_dir / "dipoles_post_MacroMag")

    Bn_pre   = np.sum(b_dipole_gpmo.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    Bn_post  = np.sum(b_dipole_macro.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    dBn = (Bn_post - Bn_pre)[:, :, None]
    s_plot.to_vtk(out_dir / "surface_deltaBn_post_minus_pre", extra_data={"ΔBn": dBn})

    nz_pre = idx_active_all.size
    nz_post = chosen_global_idx.size
    print(f"Stage-2 kept {nz_post} / {nz_pre} nonzero GPMO magnets (min-sep = {min_sep*100:.0f} mm).")

    t_end = time.time()
    print('Total time = ', t_end - t_start)


if __name__ == "__main__":
    main()
