import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

outdir = Path("output_permanent_magnet_GPMO_MUSE")

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
    plt.figure(figsize=(5.5, 3.5))
    r2_min = None
    r2_max = None

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

        # Nominal K from filename
        K_nom = None
        for tok in mh_match.stem.split("_"):
            if tok.startswith("K"):
                try:
                    K_nom = int(tok[1:])
                    break
                except Exception:
                    pass
        if K_nom is None:
            K_nom = max(len(R2_history), len(Bn_history))
        
        K_norm = len(R2_history)
        #Parse params and algorithm from file name 
        print(K_nom)
        parts = R2_suffix.split("_")

        bt_val = None
        nadj_val = None
        nmax_val = None
        kmm_val = None

        for tok in parts:
            if tok.startswith("bt"):
                try:
                    bt_val = int(tok[2:])
                except Exception:
                    pass
            elif tok.startswith("Nadj"):
                try:
                    nadj_val = int(tok[4:])
                except Exception:
                    pass
            elif tok.startswith("nmax"):
                try:
                    nmax_val = int(tok[4:])
                except Exception:
                    pass
            elif tok.startswith("kmm"):
                try:
                    kmm_val = int(tok[3:])
                except Exception:
                    pass

        base_prefixes = ("K", "nphi", "ntheta")
        param_prefixes = ("bt", "Nadj", "nmax", "kmm")
        algo_tokens = [
            tok for tok in parts
            if not tok.startswith(base_prefixes + param_prefixes)
        ]
        algo_label = "_".join(algo_tokens) if algo_tokens else R2_suffix

        # Human-readable algorithm label
        if "macromag" in algo_label:
            algo_human = "GPMOmr"
        elif "ArbVec_backtracking" in algo_label:
            algo_human = "GPMO"
        else:
            algo_human = algo_label

        # x-axis in terms of number of nonzero magnets (cap)
        max_n_magnets = nmax_val


        R2_plot = R2_history[:]
        K_actual = len(R2_plot)
        iterations = np.linspace(0, len(R2_history), len(R2_history), endpoint=True)

        # track global min/max for y-limits
        cur_min = float(np.min(R2_plot))
        cur_max = float(np.max(R2_plot))
        if r2_min is None:
            r2_min, r2_max = cur_min, cur_max
        else:
            r2_min = min(r2_min, cur_min)
            r2_max = max(r2_max, cur_max)

        # Build label with backtracking params (and kmm if present)
        param_labels = []
        if bt_val is not None:
            param_labels.append(f"bt={bt_val}")
        if nadj_val is not None:
            param_labels.append(f"Nadj={nadj_val}")
        if nmax_val is not None:
            param_labels.append(f"nmax={nmax_val}")
        if kmm_val is not None:
            param_labels.append(f"kmm={kmm_val}")

        if param_labels:
            label = rf"$f_B$ ({algo_human}, " + ", ".join(param_labels) + ")"
        else:
            label = rf"$f_B$ ({algo_human})"

        plt.semilogy(iterations, R2_plot, label=label, linestyle="--")

    plt.grid(True)
    plt.xlabel(r'Iteration $K$')
    plt.ylabel(r'$f_B$ [T$^2$ m$^2$]')

    if r2_min is not None and r2_max is not None:
        plt.ylim(r2_min * 0.8, r2_max * 1.2)

    plt.legend(fontsize=8, frameon=True, facecolor='white', edgecolor='black')
    plt.tight_layout()
    fname = save_dir / "Combined_MSE_history.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved combined plot {fname}")

npz_files = [f for f in outdir.glob("dipoles_final_*.npz") if not f.name.endswith("relax-and-split.npz")]
print(npz_files)
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
        diffs = diffs[diffs > 0]  # clip out zeros or near-zeros

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
