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

def active_counts_from_mhistory_txt(path, eps=0.0, max_loadtxt_bytes=200 * 1024 * 1024):
    """
    Returns n_active (H,) = number of nonzero dipoles at each saved history index.
    mhistory file is written as (3*ndipoles, H) text via np.savetxt(m_history.reshape(3N, H)).
    Uses loadtxt for smaller files; streams line-by-line for large files to avoid O(3N*H) memory.
    """
    try:
        fsize = path.stat().st_size
    except Exception:
        fsize = None

    def _counts_from_full_array(arr2d):
        arr2d = np.atleast_2d(arr2d)
        if arr2d.shape[0] % 3 != 0:
            raise ValueError(f"{path.name}: mhistory rows {arr2d.shape[0]} not divisible by 3.")
        ndip = arr2d.shape[0] // 3
        H = arr2d.shape[1]
        m = arr2d.reshape(ndip, 3, H)
        norm2 = np.sum(m * m, axis=1)  # (ndip, H)
        return np.count_nonzero(norm2 > eps, axis=0)

    # Small enough: load whole thing
    if fsize is not None and fsize <= max_loadtxt_bytes:
        arr = np.loadtxt(path)
        return _counts_from_full_array(arr)

    # Otherwise: stream line-by-line (memory O(H))
    with open(path, "r") as f:
        first = f.readline()
        if not first:
            return None
        v0 = np.fromstring(first, sep=" ")
        H = v0.size
        if H == 0:
            return None

        counts = np.zeros(H, dtype=int)
        buf = [v0]

        for line in f:
            v = np.fromstring(line, sep=" ")
            if v.size != H:
                # If formatting mismatch, try to be conservative and skip this file.
                raise ValueError(f"{path.name}: inconsistent column count (expected {H}, got {v.size}).")
            buf.append(v)
            if len(buf) == 3:
                mx, my, mz = buf
                norm2 = mx * mx + my * my + mz * mz
                counts += (norm2 > eps).astype(int)
                buf = []

        if len(buf) != 0:
            # leftover incomplete dipole triple; ignore
            pass

    return counts

R2_files = sorted(outdir.glob("R2history_*.txt"))
mh_files = sorted(outdir.glob("mhistory_*.txt"))
Bn_files = sorted([p for p in outdir.glob("*history_*.txt") if "bn" in p.stem.lower()])

record_every = 10  # Set to 1, 1000, etc. depending on how often f_B is recorded

if R2_files:
    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))
    ax2 = ax1.twinx()  # secondary axis for number of active magnets

    r2_min = None
    r2_max = None

    for R2_file in R2_files:
        R2_suffix = parse_suffix(R2_file.stem, "R2history_")
        if R2_suffix is None:
            continue

        Bn_match = None
        for bn in Bn_files:
            suffix = bn.stem.split("history_", 1)[-1] if "history_" in bn.stem else None
            if suffix == R2_suffix:
                Bn_match = bn
                break
        if Bn_match is None:
            continue

        # Find matching mhistory file 
        mh_match = None
        for mh in mh_files:
            mh_suffix = parse_suffix(mh.stem, "mhistory_")
            if mh_suffix == R2_suffix:
                mh_match = mh
                break

        R2_history = np.loadtxt(R2_file)
        Bn_history = np.loadtxt(Bn_match)
        print(f"{R2_file.name}: loaded {len(R2_history)} points, final iteration = {R2_history[-1]}")

        # Nominal K from filename
        K_nom = len(R2_history) * record_every

        K_norm = len(R2_history)
        # Parse params and algorithm from file name
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
        iterations = np.arange(1, len(R2_history) + 1) * record_every

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
            nlabel = rf"$N_{{\rm active}}$ ({algo_human}, " + ", ".join(param_labels) + ")"
        else:
            label = rf"$f_B$ ({algo_human})"
            nlabel = rf"$N_{{\rm active}}$ ({algo_human})"

        # Plot f_B
        (line_fb,) = ax1.semilogy(iterations, R2_plot, label=label)

        # Plot number of active magnets if mhistory exists
        if mh_match is not None:
            try:
                n_active = active_counts_from_mhistory_txt(mh_match, eps=0.0)
                if n_active is not None:
                    L = min(len(iterations), len(n_active))
                    ax2.plot(
                        iterations[:L],
                        n_active[:L],
                        linestyle="--",
                        linewidth=1.0,
                        color=line_fb.get_color(),
                        label=nlabel,
                    )
            except Exception as e:
                print(f"[WARN] Skipping N_active for {R2_suffix}: {e}")

    ax1.grid(True)
    ax1.set_xlabel(r'Iteration $K$')
    ax1.set_ylabel(r'$f_B$ [T$^2$ m$^2$]')
    ax2.set_ylabel(r'Number of active magnets')

    if r2_min is not None and r2_max is not None:
        ax1.set_ylim(r2_min * 0.8, r2_max * 1.2)

    # combined legend (both axes)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, frameon=True, facecolor='white', edgecolor='black')

    fig.tight_layout()
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

        diffs_raw = []
        one_only_vals = []
        n_both_zero = 0

        for v1, v2 in zip(M1, M2):
            z1 = np.allclose(v1, 0)
            z2 = np.allclose(v2, 0)

            if z1 and z2:
                n_both_zero += 1
                continue

            if z1 ^ z2:
                d = np.linalg.norm(v2) if z1 else np.linalg.norm(v1)
                one_only_vals.append(d)
            else:
                d = np.linalg.norm(v1 - v2)

            diffs_raw.append(d)

        diffs_raw = np.asarray(diffs_raw)

        # Estimate M_rem from the "one run only" peak when possible
        if len(one_only_vals) > 0:
            Mrem_est = float(np.median(one_only_vals))
        else:
            mags = np.r_[np.linalg.norm(M1, axis=1), np.linalg.norm(M2, axis=1)]
            mags = mags[mags > 0]
            Mrem_est = float(np.median(mags)) if len(mags) else np.nan

        # Bucket boundaries halfway between peaks at 0, Mrem, sqrt(2)Mrem, 2Mrem
        if np.isfinite(Mrem_est) and Mrem_est > 0:
            b01 = 0.5 * Mrem_est
            b12 = 0.5 * (1.0 + np.sqrt(2.0)) * Mrem_est
            b23 = 0.5 * (np.sqrt(2.0) + 2.0) * Mrem_est

            c0 = int(np.sum(diffs_raw < b01))
            c1 = int(np.sum((diffs_raw >= b01) & (diffs_raw < b12)))
            c2 = int(np.sum((diffs_raw >= b12) & (diffs_raw < b23)))
            c3 = int(np.sum(diffs_raw >= b23))

            counts = np.array([c0, c1, c2, c3], dtype=int)
            labels = ["near 0", "near M_rem", "near sqrt(2) M_rem", "near 2 M_rem"]

            N_union = len(diffs_raw)
            N_total = M1.shape[0]

            pct_union = 100.0 * counts / max(N_union, 1)
            pct_total = 100.0 * counts / max(N_total, 1)

            print("\nΔM bucket breakdown:")
            print(f"  estimated M_rem: {Mrem_est:.6e} [A·m^2]")
            print(f"  total sites: {N_total}")
            print(f"  both-zero sites: {n_both_zero} ({100.0*n_both_zero/max(N_total,1):.2f}%)")
            print(f"  union-active sites: {N_union} ({100.0*N_union/max(N_total,1):.2f}%)")

            for lab, n, pu, pt in zip(labels, counts, pct_union, pct_total):
                print(f"  {lab:>18}: {n:6d}  ({pu:6.2f}% of union-active, {pt:6.2f}% of all sites)")

            print("\nLaTeX:")
            print(
                rf"{pct_union[0]:.1f}\% near $0$, {pct_union[1]:.1f}\% near $M_{{\rm rem}}$, "
                rf"{pct_union[2]:.1f}\% near $\sqrt{{2}}M_{{\rm rem}}$, {pct_union[3]:.1f}\% near $2M_{{\rm rem}}$"
            )

        diffs = diffs_raw
        diffs = diffs[diffs > 0]  # clip out exact zeros

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
