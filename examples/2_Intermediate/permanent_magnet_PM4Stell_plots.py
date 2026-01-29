import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simsopt.geo import SurfaceRZFourier
from simsopt.field import DipoleField

outdir = Path("PM4Stell_angle{angle}_nb{nBacktracking)_na{nAdjacent}")

save_dir = outdir / "plots"
save_dir.mkdir(parents=True, exist_ok=True)

def parse_suffix(stem, prefix):
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return None

R2_files = sorted(outdir.glob("R2history_*.txt"))
mh_files = sorted(outdir.glob("mhistory_*.txt"))
Bn_files = sorted([p for p in outdir.glob("*history_*.txt") if "bn" in p.stem.lower()])

if R2_files:
    plt.figure()
    for R2_file in R2_files:
        R2_suffix = parse_suffix(R2_file.stem, "R2history_")
        if R2_suffix is None:
            continue

        mh_match = next((mh for mh in mh_files if mh.stem.endswith(R2_suffix)), None)
        if mh_match is None:
            continue

        Bn_match = None
        for bn in Bn_files:
            suffix = bn.stem.split("history_", 1)[-1] if "history_" in bn.stem else None
            if suffix == R2_suffix:
                Bn_match = bn
                break
        if Bn_match is None:
            continue

        try:
            R2_history = np.loadtxt(R2_file)
            Bn_history = np.loadtxt(Bn_match)
        except Exception:
            continue

        K_nom = None
        for tok in mh_match.stem.split("_"):
            if tok.startswith("K"):
                try:
                    K_nom = int(tok[1:])
                    break
                except:
                    pass
        if K_nom is None:
            K_nom = max(len(R2_history), len(Bn_history))

        H = min(len(R2_history), len(Bn_history))
        R2_plot = R2_history[:H]
        Bn_plot = Bn_history[:H]
        iterations = np.linspace(0, K_nom, H, endpoint=False)

        parts = R2_suffix.split("_")
        if len(parts) > 3:
            algo_label = "_".join(parts[3:])
        else:
            algo_label = R2_suffix

        if algo_label == "GPMO":
            plt.semilogy(iterations, R2_plot, label=r'$f_B$ (GPMO (no coupling))')
            #plt.semilogy(iterations, Bn_plot, label=fr'$<|Bn|>$ (GPMO (no coupling))')
        else:
            plt.semilogy(iterations, R2_plot, label=r'$f_B$ (MacroMag GPMO)')
            #plt.semilogy(iterations, Bn_plot, label=fr'$<|Bn|>$ (MacroMag GPMO)')

    plt.grid(True)
    plt.xlabel('K')
    plt.ylabel('Metric values')
    plt.legend(fontsize=8)
    fname = save_dir / "Combined_MSE_history.png"
    plt.savefig(fname, dpi=180)
    plt.close()
    print(f"Saved combined plot {fname}")

npz_files = [f for f in outdir.glob("dipoles_final_*.npz") if not f.name.endswith("relax-and-split.npz")]
if len(npz_files) >= 2:
    data = {}
    for f in npz_files:
        key = f.stem.split("dipoles_final_")[-1]
        arr = np.load(f)
        data[key] = arr["m"]

    if len(data) == 2:
        (alg1, M1), (alg2, M2) = data.items()
        if M1.shape != M2.shape:
            raise ValueError(f"Shape mismatch: {alg1} {M1.shape}, {alg2} {M2.shape}")

        diffs = []
        for v1, v2 in zip(M1, M2):
            if np.allclose(v1, 0) and np.allclose(v2, 0):
                continue
            elif np.allclose(v1, 0):
                diffs.append(np.linalg.norm(v2))
            elif np.allclose(v2, 0):
                diffs.append(np.linalg.norm(v1))
            else:
                diffs.append(np.linalg.norm(v1 - v2))
        diffs = np.array(diffs)
        diffs = diffs[diffs > 0]

        # Linear scale histogram 
        plt.figure()
        plt.hist(diffs, bins=200, alpha=0.7)
        plt.xlabel(r"$|\Delta M| = \|M_i - M'_i\|_2$ [A·m$^2$]")
        plt.ylabel("Number of magnets")
        plt.title("Histogram of |ΔM| (linear scale)")
        plt.grid(True)
        fname_lin = save_dir / "Histogram_DeltaM_linear.png"
        plt.savefig(fname_lin, dpi=180)
        plt.close()
        print(f"Saved linear-scale histogram to {fname_lin}")

        # Log scale histogram 
        plt.figure()
        plt.hist(diffs, bins=200, alpha=0.7)
        plt.xlabel(r"$|\Delta M| = \|M_i - M'_i\|_2$ [A·m$^2$]")
        plt.ylabel("Number of magnets (log scale)")
        plt.title("Histogram of |ΔM| (log scale)")
        plt.yscale("log")
        plt.grid(True)
        fname_log = save_dir / "Histogram_DeltaM_log.png"
        plt.savefig(fname_log, dpi=180)
        plt.close()
        print(f"Saved log-scale histogram to {fname_log}")

        # Trying to COmpute max B*n error betwen uncoupled and coupled version
        full_data = {}
        for f in npz_files:
            key = f.stem.split("dipoles_final_")[-1]
            full_data[key] = np.load(f)

        if len(full_data) == 2:
            (name1, arr1), (name2, arr2) = full_data.items()

            def is_coupled(name):
                name_l = name.lower()
                return ("macromag" in name_l) or ("gpmomr" in name_l)

            # Identify uncoupled (classical GPMO ArbVec) vs coupled (MacroMag GPMO)
            if is_coupled(name1) == is_coupled(name2):
                print("Could not uniquely identify uncoupled vs coupled solution from filenames.")
            else:
                if is_coupled(name1):
                    name_cpl, arr_cpl = name1, arr1
                    name_unc, arr_unc = name2, arr2
                else:
                    name_cpl, arr_cpl = name2, arr2
                    name_unc, arr_unc = name1, arr1

                xyz_cpl = arr_cpl["xyz"]
                xyz_unc = arr_unc["xyz"]
                if not np.allclose(xyz_cpl, xyz_unc):
                    raise ValueError("xyz grids differ between coupled and uncoupled solutions.")

                m_cpl = arr_cpl["m"]
                m_unc = arr_unc["m"]
                nfp = int(arr_cpl["nfp"])
                coordinate_flag = int(arr_cpl["coordinate_flag"])
                m_maxima = arr_cpl["m_maxima"]

                # Build a viewing surface from input.muse
                TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
                surface_filename = TEST_DIR / "input.muse"

                min_res = 164
                nphi_view = min_res
                ntheta_view = min_res
                quad_phi = np.linspace(0, 1, nphi_view, endpoint=False)
                quad_theta = np.linspace(0, 1, ntheta_view, endpoint=False)

                s_view = SurfaceRZFourier.from_focus(
                    surface_filename,
                    quadpoints_phi=quad_phi,
                    quadpoints_theta=quad_theta
                )

                pts = s_view.gamma().reshape((-1, 3))
                n_hat = s_view.unitnormal().reshape(nphi_view, ntheta_view, 3)

                # Build dipole fields for uncoupled and coupled solutions
                b_unc = DipoleField(
                    xyz_unc,
                    m_unc.reshape(-1),
                    nfp=nfp,
                    coordinate_flag=coordinate_flag,
                    m_maxima=m_maxima,
                )
                b_cpl = DipoleField(
                    xyz_cpl,
                    m_cpl.reshape(-1),
                    nfp=nfp,
                    coordinate_flag=coordinate_flag,
                    m_maxima=m_maxima,
                )

                b_unc.set_points(pts)
                b_cpl.set_points(pts)

                B_unc = b_unc.B().reshape(nphi_view, ntheta_view, 3)
                B_cpl = b_cpl.B().reshape(nphi_view, ntheta_view, 3)

                Bn_unc = np.sum(B_unc * n_hat, axis=2)
                Bn_cpl = np.sum(B_cpl * n_hat, axis=2)

                Bn_diff = Bn_unc - Bn_cpl

                ref_B = 0.15
                max_rel = float(np.max(np.abs(Bn_diff))) / ref_B
                rms_rel = float(np.sqrt(np.mean(Bn_diff**2))) / ref_B

                print(f"Algorithm uncoupled: {name_unc}")
                print(f"Algorithm coupled:   {name_cpl}")
                print(f"Max |Bn_uncoupled - Bn_coupled| / 0.15 = {max_rel:.3e}")
                print(f"RMS |Bn_uncoupled - Bn_coupled| / 0.15 = {rms_rel:.3e}")
