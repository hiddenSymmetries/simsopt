"""
This module contains the a number of useful functions for using 
the permanent magnets functionality in the SIMSOPT code.
"""
__all__ = [
           'initialize_coils_for_pm_optimization',
           'make_optimization_plots', 'run_Poincare_plots_with_permanent_magnets',
           'initialize_default_kwargs',
           'get_utc_timestamp_iso',
           'write_gpmo_run_history_csv',
           'write_gpmo_run_metadata_yaml',
           'build_polarization_vectors_from_focus_data',
           'compute_m_maxima_and_cube_dim_from_focus_file',
           'compute_final_squared_flux',
           'run_gpmo_optimization_on_muse_grid',
           'normalize_vectors_to_unit_length',
           'compute_angle_between_vectors_degrees',
           'compute_normal_field_component_from_dipoles',
           'check_magnet_volume_stellarator_symmetry',
           'reshape_to_vtk_field_format',
           'print_bnormal_error_summary_statistics',
           ]

import csv
import datetime
import numpy as np
from pathlib import Path

try:
    from ruamel.yaml import YAML
except ImportError:
    YAML = None


def get_utc_timestamp_iso() -> str:
    """
    Return the current UTC timestamp as an ISO 8601 formatted string.

    Returns:
        str: Timestamp string in format 'YYYY-MM-DDTHH:MM:SS+00:00'.
    """
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec="seconds")


def write_gpmo_run_history_csv(out_subdir: Path, run_id: str, pm_opt, fB_hist, absBn_hist) -> Path:
    """
    Write GPMO run history (iteration index, squared flux, mean |Bn|, active magnet count)
    to a CSV file.

    Args:
        out_subdir: Directory path for output files.
        run_id: Unique identifier for this run (used in filename).
        pm_opt: PermanentMagnetGrid optimizer object; may have k_history and num_nonzeros
            attributes. If absent or length mismatch, defaults are used.
        fB_hist: 1D array of squared flux values per iteration.
        absBn_hist: 1D array of mean |B·n| values per iteration.

    Returns:
        Path: Path to the written CSV file (runhistory_{run_id}.csv).
    """
    out_subdir.mkdir(parents=True, exist_ok=True)
    run_csv = out_subdir / f"runhistory_{run_id}.csv"

    k_hist = getattr(pm_opt, "k_history", None)
    if k_hist is None or len(k_hist) != len(fB_hist):
        k_hist = np.arange(len(fB_hist), dtype=int)

    n_active = getattr(pm_opt, "num_nonzeros", None)
    if n_active is None or len(n_active) != len(fB_hist):
        n_active = np.zeros(len(fB_hist), dtype=int)

    with run_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["k", "fB", "absBn", "n_active"])
        w.writeheader()
        for i in range(len(fB_hist)):
            w.writerow(
                {
                    "k": int(k_hist[i]),
                    "fB": float(fB_hist[i]),
                    "absBn": float(absBn_hist[i]),
                    "n_active": int(n_active[i]),
                }
            )
    return run_csv


def write_gpmo_run_metadata_yaml(
    out_subdir: Path,
    run_id: str,
    *,
    algorithm: str,
    current_scale: float,
    metrics: dict,
    material_name: str,
    material: dict,
    run_params: dict,
    paths: dict,
) -> Path:
    """
    Write run metadata (algorithm, material, parameters, paths, metrics) to a YAML file.

    Args:
        out_subdir: Directory path for output files.
        run_id: Unique identifier for this run (used in filename).
        algorithm: Algorithm name (e.g. 'GPMO', 'GPMOmr').
        current_scale: Coil current scaling factor applied.
        metrics: Dict of computed metrics (e.g. fB_last_history, fB_recomputed).
        material_name: Name of the material preset (e.g. 'N52').
        material: Dict of material properties (B_max, mu_ea, mu_oa, etc.).
        run_params: Dict of run parameters (nphi, ntheta, downsample, dr, K, etc.).
        paths: Dict of input paths (test_dir, famus_filename, surface_filename, coil_path).

    Returns:
        Path: Path to the written YAML file (run_{run_id}.yaml).
    """
    if YAML is None:
        raise ImportError("ruamel.yaml is required for write_gpmo_run_metadata_yaml")

    doc = {
        "run_id": run_id,
        "timestamp_utc": get_utc_timestamp_iso(),
        "algorithm": algorithm,
        "material": {"name": material_name, **material},
        "params": {**run_params, "current_scale": float(current_scale)},
        "paths": paths,
        "metrics": metrics,
    }

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml_path = out_subdir / f"run_{run_id}.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(doc, f)
    return yaml_path


def build_polarization_vectors_from_focus_data(mag_data) -> np.ndarray:
    """
    Build polarization vectors for a FAMUS/MUSE magnet grid using face-type orientations.

    Uses the 'face' polarization axes (6 directions: ±x, ±y, ±z) and discretizes
    them according to the magnet orientations (ophi) from the grid data.

    Args:
        mag_data: FocusData instance with magnet positions (ox, oy) and polarization
            components (pol_x, pol_y, pol_z). Must have nMagnets attribute.

    Returns:
        np.ndarray: Array of shape (nMagnets, n_pol_axes, 3) with Cartesian polarization
            vectors for each magnet and each allowed orientation.
    """
    from simsopt.util import discretize_polarizations, polarization_axes

    pol_axes = np.zeros((0, 3))
    pol_type = np.zeros(0, dtype=int)
    pol_axes_f, pol_type_f = polarization_axes(["face"])
    ntype_f = int(len(pol_type_f) / 2)
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
    return pol_vectors


def compute_m_maxima_and_cube_dim_from_focus_file(focus_path: Path, downsample: int, B_max: float):
    """
    Compute per-magnet maximum dipole moments and cube dimension from a FAMUS .focus file.

    Reads magnet data (positions, Ic, M0s) from the file, applies downsampling,
    and scales M0s by B_max/B_ref to get m_maxima. The cube dimension is derived
    from the cell volume (M0 * mu0 / B_ref) assuming cubic magnets.

    Args:
        focus_path: Path to the .focus (FAMUS) grid file.
        downsample: Downsampling factor (1 = use all magnets, 2 = every 2nd, etc.).
        B_max: Target remanence (Tesla) for scaling m_maxima. The reference B_ref
            used when generating the grid is 1.465 T.

    Returns:
        tuple: (m_maxima, cube_dim)
            - m_maxima: 1D array of maximum dipole magnitudes per magnet (A·m²).
            - cube_dim: Scalar cube edge length (m) for VTK visualization.
    """
    B_ref = 1.465  # Reference B_max used when generating the MUSE .focus grid
    mu0 = 4 * np.pi * 1e-7

    ox, oy, oz, Ic, M0s = np.loadtxt(
        focus_path, skiprows=3, usecols=[3, 4, 5, 6, 7], delimiter=",", unpack=True
    )

    inds_total = np.arange(len(ox))
    inds_downsampled = inds_total[::downsample]
    nonzero_inds = np.intersect1d(np.ravel(np.where(Ic == 1.0)), inds_downsampled)

    m_maxima = M0s[nonzero_inds] * (B_max / B_ref)

    cell_vol = M0s * mu0 / B_ref
    cube_dim = float(cell_vol[0] ** (1.0 / 3.0))
    return m_maxima, cube_dim


def compute_final_squared_flux(
    *,
    s_plot,
    bs,
    pm_opt,
    nphi: int,
    ntheta: int,
) -> float:
    """
    Compute the final squared flux (f_B) from dipoles plus coils on a plasma boundary.

    Evaluates B·n from coils and dipoles on the surface, then returns the SquaredFlux
    objective (half the L2 norm of the residual vs target -Bnormal from coils).

    Args:
        s_plot: SurfaceRZFourier plasma boundary (full or half period).
        bs: BiotSavart field from coils.
        pm_opt: PermanentMagnetGrid with optimized m and dipole_grid_xyz.
        nphi: Number of phi quadrature points on the surface.
        ntheta: Number of theta quadrature points on the surface.

    Returns:
        float: Squared flux value (0.5 * sum((B_dipole·n + B_coil·n - target)^2)).
    """
    from simsopt.field import DipoleField
    from simsopt.objectives import SquaredFlux

    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    b_dipole = DipoleField(
        pm_opt.dipole_grid_xyz,
        pm_opt.m,
        nfp=s_plot.nfp,
        coordinate_flag=pm_opt.coordinate_flag,
        m_maxima=pm_opt.m_maxima,
    )
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
    return float(SquaredFlux(s_plot, b_dipole, -Bnormal).J())


def run_gpmo_optimization_on_muse_grid(
    algorithm: str,
    *,
    subdir: str,
    s,
    bs,
    Bnormal: np.ndarray,
    pol_vectors: np.ndarray,
    m_maxima: np.ndarray,
    cube_dim: float,
    current_scale: float,
    out_dir: Path,
    famus_filename: Path,
    material_name: str,
    material: dict,
    downsample: int,
    dr: float,
    nIter_max: int,
    nHistory: int,
    history_every: int,
    nBacktracking: int,
    Nadjacent: int,
    max_nMagnets: int,
    thresh_angle: float,
    mm_refine_every: int,
    nphi: int,
    ntheta: int,
    coil_path: Path,
    test_dir: Path,
    surface_filename: Path,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    """
    Run one GPMO or GPMOmr optimization on the MUSE grid and write outputs.

    Sets up the permanent magnet grid from the FAMUS file, runs GPMO or GPMOmr,
    and writes run history CSV, run metadata YAML, final dipoles NPZ, FAMUS file,
    and VTK visualization.

    Args:
        algorithm: 'GPMO' or 'GPMOmr'.
        subdir: Subdirectory name under out_dir for this run (e.g. 'GPMO', 'GPMOmr').
        s: SurfaceRZFourier plasma boundary.
        bs: BiotSavart field from coils.
        Bnormal: 2D array (nphi, ntheta) of target -B·n from coils.
        pol_vectors: Polarization vectors from build_polarization_vectors_from_focus_data.
        m_maxima: Per-magnet maximum dipole moments from compute_m_maxima_and_cube_dim_from_focus_file.
        cube_dim: Cube dimension for GPMOmr and VTK.
        current_scale: Coil current scaling factor.
        out_dir: Base output directory.
        famus_filename: Path to .focus FAMUS grid file.
        material_name: Material preset name (e.g. 'N52').
        material: Material dict (B_max, mu_ea, mu_oa, etc.).
        downsample: Grid downsampling factor.
        dr: Radial extent of cylindrical magnet bricks (m).
        nIter_max: Maximum GPMO iterations (K).
        nHistory: Number of history snapshots.
        history_every: Save history every N iterations.
        nBacktracking: Backtracking iterations.
        Nadjacent: Number of adjacent magnets for backtracking.
        max_nMagnets: Maximum number of magnets to place.
        thresh_angle: Threshold angle (rad) for magnet removal.
        mm_refine_every: GPMOmr refinement interval (0 for GPMO).
        nphi: Surface phi resolution.
        ntheta: Surface theta resolution.
        coil_path: Path to coil .focus file (for GPMOmr).
        test_dir: Test files directory.
        surface_filename: Plasma surface input path.
        dx, dy, dz: Magnet dimensions for VTK output (m).
    """
    from simsopt.field import DipoleField
    from simsopt.geo import PermanentMagnetGrid
    from simsopt.solve import GPMO

    out_subdir = out_dir / subdir
    out_subdir.mkdir(parents=True, exist_ok=True)

    kwargs_grid = {"pol_vectors": pol_vectors, "downsample": downsample, "dr": dr, "m_maxima": m_maxima}
    pm_opt = PermanentMagnetGrid.geo_setup_from_famus(s, Bnormal, famus_filename, **kwargs_grid)
    print(f"[INFO] Number of available dipoles = {pm_opt.ndipoles}")

    kwargs = initialize_default_kwargs("GPMO")
    kwargs["K"] = int(nIter_max)
    kwargs["nhistory"] = int(nHistory)
    kwargs["backtracking"] = int(nBacktracking)
    kwargs["Nadjacent"] = int(Nadjacent)
    kwargs["dipole_grid_xyz"] = np.ascontiguousarray(pm_opt.dipole_grid_xyz)
    kwargs["max_nMagnets"] = int(max_nMagnets)
    kwargs["thresh_angle"] = float(thresh_angle)

    if algorithm == "GPMOmr":
        kwargs["cube_dim"] = float(cube_dim)
        kwargs["mu_ea"] = float(material["mu_ea"])
        kwargs["mu_oa"] = float(material["mu_oa"])
        kwargs["use_coils"] = True
        kwargs["use_demag"] = True
        kwargs["coil_path"] = coil_path
        kwargs["mm_refine_every"] = int(mm_refine_every)
        kwargs["current_scale"] = float(current_scale)

    print(
        f"\n[{get_utc_timestamp_iso()}] Run: {algorithm} (nphi={nphi}, downsample={downsample}, "
        f"K={nIter_max}, max_nMagnets={max_nMagnets})"
    )
    fB_hist, absBn_hist, _m_history = GPMO(pm_opt, algorithm=algorithm, **kwargs)

    run_id = f"lowres_nphi{nphi}_ds{downsample}_mat{material_name}_K{nIter_max}_nmax{max_nMagnets}_{algorithm}"
    run_csv = write_gpmo_run_history_csv(out_subdir, run_id, pm_opt, fB_hist, absBn_hist)

    fB_final = compute_final_squared_flux(s_plot=s, bs=bs, pm_opt=pm_opt, nphi=nphi, ntheta=ntheta)
    fB_last_hist = float(fB_hist[-1]) if len(fB_hist) else float("nan")
    print(f"[INFO] f_B (last history)   = {fB_last_hist:.16e}")
    print(f"[INFO] f_B (recomputed)    = {fB_final:.16e}")

    run_params = {
        "nphi": nphi,
        "ntheta": ntheta,
        "downsample": downsample,
        "dr": dr,
        "K": nIter_max,
        "nhistory": nHistory,
        "history_every": history_every,
        "backtracking": nBacktracking,
        "Nadjacent": Nadjacent,
        "max_nMagnets": max_nMagnets,
        "thresh_angle": float(thresh_angle),
        "mm_refine_every": mm_refine_every if algorithm == "GPMOmr" else 0,
    }
    paths = {
        "test_dir": str(test_dir),
        "famus_filename": str(famus_filename),
        "surface_filename": str(surface_filename),
        "coil_path": str(coil_path),
    }

    yaml_path = write_gpmo_run_metadata_yaml(
        out_subdir,
        run_id,
        algorithm=algorithm,
        current_scale=current_scale,
        metrics={"fB_last_history": fB_last_hist, "fB_recomputed": fB_final},
        material_name=material_name,
        material=material,
        run_params=run_params,
        paths=paths,
    )

    npz_path = out_subdir / f"dipoles_final_{run_id}.npz"
    np.savez(
        npz_path,
        xyz=pm_opt.dipole_grid_xyz,
        m=pm_opt.m.reshape(pm_opt.ndipoles, 3),
        nfp=s.nfp,
        coordinate_flag=pm_opt.coordinate_flag,
        m_maxima=pm_opt.m_maxima,
    )

    pm_opt.write_to_famus(out_subdir)
    b_final = DipoleField(
        pm_opt.dipole_grid_xyz,
        pm_opt.m,
        nfp=s.nfp,
        coordinate_flag=pm_opt.coordinate_flag,
        m_maxima=pm_opt.m_maxima,
    )
    b_final._toVTK(out_subdir / f"dipoles_final_{run_id}", dx, dy, dz)

    print(f"[INFO] Wrote {run_csv}")
    print(f"[INFO] Wrote {yaml_path}")
    print(f"[INFO] Wrote {npz_path}")


def normalize_vectors_to_unit_length(vectors: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """
    Normalize an array of vectors to unit length.

    Args:
        vectors: Array of shape (N, 3) containing vectors to normalize.
        eps: Small positive value added to the norm to prevent division by zero.

    Returns:
        Array of shape (N, 3) with unit-normalized vectors.
    """
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + eps)


def compute_angle_between_vectors_degrees(
    vectors_a: np.ndarray, vectors_b: np.ndarray, eps: float = 1e-30
) -> np.ndarray:
    """
    Compute the angle in degrees between corresponding pairs of vectors.

    Args:
        vectors_a: Array of shape (N, 3) containing first set of vectors.
        vectors_b: Array of shape (N, 3) containing second set of vectors.
        eps: Small positive value for numerical stability when normalizing.

    Returns:
        Array of shape (N,) containing angles in degrees between each pair.
    """
    a_norm = normalize_vectors_to_unit_length(vectors_a, eps)
    b_norm = normalize_vectors_to_unit_length(vectors_b, eps)
    dot_product = np.clip(np.sum(a_norm * b_norm, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product))


def compute_normal_field_component_from_dipoles(
    dipole_centers: np.ndarray,
    dipole_moments: np.ndarray,
    evaluation_points: np.ndarray,
    surface_normals: np.ndarray,
) -> np.ndarray:
    """
    Compute the normal component of magnetic field (B·n) at evaluation points due to dipoles.

    Args:
        dipole_centers: Array of shape (N, 3) with dipole center positions [m].
        dipole_moments: Array of shape (N, 3) with dipole moments m = M * V [A·m²].
        evaluation_points: Array of shape (M, 3) with evaluation point positions [m].
        surface_normals: Array of shape (M, 3) with unit normal vectors at evaluation points.

    Returns:
        Array of shape (M,) with B·n at each evaluation point [T].
    """
    from simsopt.field import DipoleField

    dip = DipoleField(
        dipole_centers,
        dipole_moments,
        nfp=1,
        stellsym=False,
        coordinate_flag="cartesian",
    )
    dip.set_points(evaluation_points)
    B = dip.B()
    return np.sum(B * surface_normals, axis=1)


def check_magnet_volume_stellarator_symmetry(
    magnet_centers: np.ndarray,
    magnet_volumes: np.ndarray,
    nfp: int = 2,
    z_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-3,
) -> None:
    """
    Check whether magnet volumes obey stellarator, midplane, and inversion symmetries.

    Prints diagnostic statistics for:
    - Toroidal (nfp) periodicity: volumes and positions under rotation by 2π/nfp
    - Midplane (Z -> -Z) symmetry: volumes of magnet pairs reflected across z=0
    - Inversion symmetry (φ+π/nfp, Z->-Z): combined rotation and reflection

    Args:
        magnet_centers: Array of shape (N, 3) with magnet center coordinates [x, y, z] in meters.
        magnet_volumes: Array of shape (N,) with magnet volumes [m³].
        nfp: Number of field periods.
        z_tolerance: Position tolerance [m] for matching midplane symmetry in z.
        relative_tolerance: Relative tolerance for volume equality (unused; for API compatibility).
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation as R

    n_magnets = len(magnet_volumes)
    phi_period = 2 * np.pi / nfp

    print("\n[Symmetry check] --- Magnet volume symmetries ---")

    # Toroidal (nfp) periodicity test
    rot = R.from_euler("z", phi_period).as_matrix()
    rotated = (rot @ magnet_centers.T).T
    tree = cKDTree(magnet_centers)
    dists, idx = tree.query(rotated, k=1)
    vol_err = np.abs(magnet_volumes - magnet_volumes[idx]) / np.maximum(magnet_volumes, 1e-20)
    print(f"  nfp={nfp} toroidal symmetry:")
    print(f"    mean |ΔV/V| = {vol_err.mean():.2e}")
    print(f"    max  |ΔV/V| = {vol_err.max():.2e}")
    print(f"    mean position mismatch = {dists.mean():.3e} m")
    print(f"    max  position mismatch = {dists.max():.3e} m")

    # Midplane (Z -> -Z) symmetry
    flipped = magnet_centers.copy()
    flipped[:, 2] *= -1
    tree_z = cKDTree(magnet_centers)
    d_z, idx_z = tree_z.query(flipped, k=1)
    mask = d_z < z_tolerance
    if np.any(mask):
        vol_err_z = (
            np.abs(magnet_volumes[mask] - magnet_volumes[idx_z[mask]])
            / np.maximum(magnet_volumes[mask], 1e-20)
        )
        print("  midplane (Z->-Z) symmetry:")
        print(f"    matched {mask.sum()} / {n_magnets} tiles within {z_tolerance:.1e} m")
        print(f"    mean |ΔV/V| = {vol_err_z.mean():.2e}")
        print(f"    max  |ΔV/V| = {vol_err_z.max():.2e}")
    else:
        print("  midplane symmetry: no matching pairs found within tolerance")

    # Inversion (R, φ, Z) -> (R, φ+π/nfp, -Z)
    rot_half = R.from_euler("z", phi_period / 2).as_matrix()
    inv = (rot_half @ magnet_centers.T).T
    inv[:, 2] *= -1
    tree_inv = cKDTree(magnet_centers)
    d_inv, idx_inv = tree_inv.query(inv, k=1)
    vol_err_inv = (
        np.abs(magnet_volumes - magnet_volumes[idx_inv])
        / np.maximum(magnet_volumes, 1e-20)
    )
    print("  inversion (φ+π/nfp, Z->-Z) symmetry:")
    print(f"    mean |ΔV/V| = {vol_err_inv.mean():.2e}")
    print(f"    max  |ΔV/V| = {vol_err_inv.max():.2e}")
    print(f"    mean position mismatch = {d_inv.mean():.3e} m")
    print(f"    max  position mismatch = {d_inv.max():.3e} m")
    print("-------------------------------------------------------------\n")


def reshape_to_vtk_field_format(field_2d: np.ndarray) -> np.ndarray:
    """
    Reshape a 2D field array to 3D format required for VTK extra_data.

    VTK surface field data expects shape (nphi, ntheta, 1) for scalar fields.

    Args:
        field_2d: 2D array of shape (nphi, ntheta).

    Returns:
        3D array of shape (nphi, ntheta, 1), contiguous and float64.
    """
    return np.ascontiguousarray(field_2d, dtype=np.float64)[:, :, None]


def print_bnormal_error_summary_statistics(
    case_name: str, bnormal_error: np.ndarray
) -> None:
    """
    Print summary statistics (mean, max, rms) for B·n error on a surface wedge.

    Args:
        case_name: Descriptive label for the coupling case (e.g. "Uncoupled", "Magnet-magnet only").
        bnormal_error: 1D array of B·n error values [T].
    """
    import math

    print(f"\n{case_name} total B·n error (unique half period wedge)")
    print(f"mean = {bnormal_error.mean():.12e}")
    print(f"max  = {np.abs(bnormal_error).max():.12e}")
    print(f"rms  = {math.sqrt(np.mean(bnormal_error ** 2)):.12e}")


def initialize_coils_for_pm_optimization(config_flag, TEST_DIR, s, out_dir=''):
    """
    Initializes coils for each of the target configurations that are
    used for permanent magnet optimization. The total current is chosen
    to achieve a desired average field strength along the major radius.

    Args:
        config_flag: String denoting the stellarator configuration 
          being initialized.
        TEST_DIR: String denoting where to find the input files.
        out_dir: Path or string for the output directory for saved files.
        s: plasma boundary surface.
    Returns:
        base_curves: List of CurveXYZ class objects.
        curves: List of Curve class objects.
        coils: List of Coil class objects.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, Coil, coils_via_symmetries
    from simsopt.geo import curves_to_vtk
    from simsopt.util.coil_optimization_helper_functions import read_focus_coils

    out_dir = Path(out_dir)
    if 'muse' in config_flag:
        # Load in pre-optimized coils
        coils_filename = TEST_DIR / 'muse_tf_coils.focus'

        # Slight difference from older MUSE initialization. 
        # Here we fix the total current summed over all coils and 
        # older MUSE opt. fixed the first coil current. 
        base_curves, base_currents0, ncoils = read_focus_coils(coils_filename)
        total_current = np.sum([curr.get_value() for curr in base_currents0])
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils - 1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = []
        for i in range(ncoils):
            coils.append(Coil(base_curves[i], base_currents[i]))

    elif config_flag == 'qh':
        # generate planar TF coils
        ncoils = 4
        R0 = s.get_rc(0, 0)
        R1 = s.get_rc(1, 0) * 4
        order = 2

        # QH reactor scale needs to be 5.7 T average magnetic field strength
        total_current = 48712698  # Amperes
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True,
                                                   R0=R0, R1=R1, order=order, numquadpoints=64)
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils - 1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

    elif config_flag == 'qa':
        # generate planar TF coils
        ncoils = 8
        R0 = 1.0
        R1 = 0.65
        order = 5

        # qa needs to be scaled to 0.1 T on-axis magnetic field strength
        total_current = 187500
        base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
        base_currents = [(Current(total_current / ncoils * 1e-5) * 1e5) for _ in range(ncoils-1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    
    # fix all the coil shapes so only the currents are optimized
    for i in range(ncoils):
        base_curves[i].fix_all()

    # Initialize the coil curves and save the data to vtk
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, out_dir / "curves_init")
    return base_curves, curves, coils


def make_optimization_plots(RS_history, m_history, m_proxy_history, pm_opt, out_dir=''):
    """
    Make line plots of the algorithm convergence and make histograms
    of m, m_proxy, and if available, the FAMUS solution.

    Args:
        RS_history: List of the relax-and-split optimization progress.
        m_history: List of the permanent magnet solutions generated
          as the relax-and-split optimization progresses.
        m_proxy_history: Same but for the 'proxy' variable in the
          relax-and-split optimization.
        pm_opt: PermanentMagnetGrid class object that was optimized.
        out_dir: Path or string for the output directory for saved files.
    """
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation
    
    out_dir = Path(out_dir)

    # Make plot of the relax-and-split convergence
    plt.figure()
    plt.plot(RS_history)
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(out_dir / 'RS_objective_history.png')

    # make histogram of the dipoles, normalized by their maximum values
    plt.figure()
    m0_abs = np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima
    mproxy_abs = np.sqrt(np.sum(pm_opt.m_proxy.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1)) / pm_opt.m_maxima

    # get FAMUS rho values for making comparison histograms
    if hasattr(pm_opt, 'famus_filename'):
        m0, p = np.loadtxt(
            pm_opt.famus_filename, skiprows=3,
            usecols=[7, 8],
            delimiter=',', unpack=True
        )
        # momentq = 4 for NCSX but always = 1 for MUSE and recent FAMUS runs
        momentq = np.loadtxt(pm_opt.famus_filename, skiprows=1, max_rows=1, usecols=[1])
        rho = p ** momentq
        rho = rho[pm_opt.Ic_inds]
        x_multi = [m0_abs, mproxy_abs, abs(rho)]
    else:
        x_multi = [m0_abs, mproxy_abs]
        rho = None
    plt.hist(x_multi, bins=np.linspace(0, 1, 40), log=True, histtype='bar')
    plt.grid(True)
    plt.legend(['m', 'w', 'FAMUS'])
    plt.xlabel('Normalized magnitudes')
    plt.ylabel('Number of dipoles')
    plt.savefig(out_dir / 'm_histograms.png')

    # If relax-and-split was used with l0 norm,
    # make a nice histogram of the algorithm progress
    if len(RS_history) != 0:

        m_history = np.array(m_history)
        m_history = m_history.reshape(m_history.shape[0] * m_history.shape[1], pm_opt.ndipoles, 3)
        m_proxy_history = np.array(m_proxy_history).reshape(m_history.shape[0], pm_opt.ndipoles, 3)

        for i, datum in enumerate([m_history, m_proxy_history]):
            # Code from https://matplotlib.org/stable/gallery/animation/animated_histogram.html
            def prepare_animation(bar_container):
                def animate(frame_number):
                    plt.title(frame_number)
                    data = np.sqrt(np.sum(datum[frame_number, :] ** 2, axis=-1)) / pm_opt.m_maxima
                    n, _ = np.histogram(data, np.linspace(0, 1, 40))
                    for count, rect in zip(n, bar_container.patches):
                        rect.set_height(count)
                    return bar_container.patches
                return animate

            # make histogram animation of the dipoles at each relax-and-split save
            fig, ax = plt.subplots()
            data = np.sqrt(np.sum(datum[0, :] ** 2, axis=-1)) / pm_opt.m_maxima
            if rho is not None:
                ax.hist(abs(rho), bins=np.linspace(0, 1, 40), log=True, alpha=0.6, color='g')
            _, _, bar_container = ax.hist(data, bins=np.linspace(0, 1, 40), log=True, alpha=0.6, color='r')
            ax.set_ylim(top=1e5)  # set safe limit to ensure that all data is visible.
            plt.grid(True)
            if rho is not None:
                plt.legend(['FAMUS', np.array([r'm$^*$', r'w$^*$'])[i]])
            else:
                plt.legend([np.array([r'm$^*$', r'w$^*$'])[i]])

            plt.xlabel('Normalized magnitudes')
            plt.ylabel('Number of dipoles')
            animation.FuncAnimation(
                fig, prepare_animation(bar_container),
                range(0, m_history.shape[0], 2),
                repeat=False, blit=True
            )


def run_Poincare_plots_with_permanent_magnets(s_plot, bs, b_dipole, comm, filename_poincare, out_dir=''):
    """
    Wrapper function for making Poincare plots.

    Args:
        s_plot: plasma boundary surface with range = 'full torus'.
        bs: MagneticField class object.
        b_dipole: DipoleField class object.
        comm: MPI COMM_WORLD object for using MPI for tracing.
        filename_poincare: Filename for the output poincare picture.
        out_dir: Path or string for the output directory for saved files.
    """
    from simsopt.field.magneticfieldclasses import InterpolatedField
    from simsopt.util.coil_optimization_helper_functions import trace_fieldlines
    # from simsopt.objectives import SquaredFlux

    out_dir = Path(out_dir)

    n = 32
    rs = np.linalg.norm(s_plot.gamma()[:, :, 0:2], axis=2)
    zs = s_plot.gamma()[:, :, 2]
    r_margin = 0.05
    rrange = (np.min(rs) - r_margin, np.max(rs) + r_margin, n)
    phirange = (0, 2 * np.pi / s_plot.nfp, n * 2)
    zrange = (0, np.max(zs), n // 2)
    degree = 2  # 2 is sufficient sometimes
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))

    bsh = InterpolatedField(
        bs + b_dipole, degree, rrange, phirange, zrange, 
        True, nfp=s_plot.nfp, stellsym=s_plot.stellsym
    )
    bsh.set_points(s_plot.gamma().reshape((-1, 3)))
    trace_fieldlines(bsh, 'bsh_PMs_' + filename_poincare, s_plot, comm, out_dir)

def initialize_default_kwargs(algorithm='RS'):
    """
    Keywords to the permanent magnet optimizers are now passed
    by the kwargs dictionary. Default dictionaries are initialized
    here.

    Args:
        algorithm: String denoting which algorithm is being used
          for permanent magnet optimization. Options are 'RS'
          (relax and split), 'GPMO' (greedy placement), 
          'GPMO_Backtracking' (GPMO with backtracking),
          'multi' (GPMO, placing multiple magnets each iteration),
          'GPMO_ArbVec' (GPMO with arbitrary dipole orientations),
          'GPMOmr' (GPMO with macromagnetic refinement),
          'GPMO_py' (testing-only pure-Python reference implementation).

    Returns:
        kwargs: Dictionary of keywords to pass to a permanent magnet 
          optimizer.
    """

    kwargs = {}
    kwargs['verbose'] = True   # print out errors every few iterations
    if algorithm == 'RS':
        kwargs['nu'] = 1e100  # Strength of the "relaxation" part of relax-and-split
        kwargs['max_iter'] = 100  # Number of iterations to take in a convex step
        kwargs['reg_l0'] = 0.0
        kwargs['reg_l1'] = 0.0
        kwargs['alpha'] = 0.0
        kwargs['min_fb'] = 0.0
        kwargs['epsilon'] = 1e-3
        kwargs['epsilon_RS'] = 1e-3
        kwargs['max_iter_RS'] = 2  # Number of total iterations of the relax-and-split algorithm
        kwargs['reg_l2'] = 0.0
    elif 'GPMO' in algorithm or 'ArbVec' in algorithm:
        kwargs['K'] = 1000
        kwargs["reg_l2"] = 0.0
        kwargs['nhistory'] = 500  # K > nhistory and nhistory must be divisor of K
    return kwargs
