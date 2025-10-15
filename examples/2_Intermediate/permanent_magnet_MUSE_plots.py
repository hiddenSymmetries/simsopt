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

        plt.semilogy(iterations, R2_plot, label=fr'$f_B$ ({algo_label})')
        plt.semilogy(iterations, Bn_plot, label=fr'$<|Bn|>$ ({algo_label})')

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
        diffs = diffs[diffs > 0]  # clip out zeros or near-zeros

        plt.figure()
        plt.hist(diffs, bins=200, alpha=0.7)
        plt.xlabel(r"$|\Delta M| = \|M_i - M'_i\|_2$ [A·m$^2$]")
        plt.ylabel("Number of magnets (log scale)")
        plt.yscale("log")    # log scale on the y-axis
        plt.grid(True)
        fname = save_dir / "Histogram_DeltaM.png"
        plt.savefig(fname, dpi=180)
        plt.close()
        print(f"Saved histogram of |ΔM| to {fname}")
        


