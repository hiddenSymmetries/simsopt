import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

outdir = Path("output_permanent_magnet_GPMO_MUSE")

save_dir = outdir / "plots"
save_dir.mkdir(parents=True, exist_ok=True)

# Snapshot Delta M
# if set (e.g. 18000), compute ΔM using mhistory snapshot near this iteration K.
# If None, compute ΔM from dipoles_final_*.npz (final state).
m_history_delta_M_iter = 18000 #Iter to do snap at

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
                raise ValueError(f"{path.name}: inconsistent column count (expected {H}, got {v.size}).")
            buf.append(v)
            if len(buf) == 3:
                mx, my, mz = buf
                norm2 = mx * mx + my * my + mz * mz
                counts += (norm2 > eps).astype(int)
                buf = []

    return counts

def m_from_mhistory_txt_at_K(path, K_target, record_every):
    """
    Stream mhistory_*.txt and extract the dipole vectors at the history index
    closest to iteration K_target (given record_every).
    Returns:
        M: (ndipoles, 3) array
        K_snap: the actual iteration corresponding to the extracted history column
        h_idx: history column index used (0-based)
        H: total number of history columns
    """
    if K_target is None:
        raise ValueError("K_target is None")

    with open(path, "r") as f:
        first = f.readline()
        if not first:
            raise ValueError(f"{path.name}: empty file")

        v0 = np.fromstring(first, sep=" ")
        H = v0.size
        if H == 0:
            raise ValueError(f"{path.name}: no columns detected")

        # iterations used elsewhere are (i+1)*record_every, so map K_target -> i = round(K/record_every)-1
        h_idx = int(np.round(float(K_target) / float(record_every))) - 1
        if h_idx < 0:
            h_idx = 0
        if h_idx > H - 1:
            h_idx = H - 1

        K_snap = int((h_idx + 1) * record_every)

        def pick(v):
            if v.size != H:
                raise ValueError(f"{path.name}: inconsistent column count (expected {H}, got {v.size}).")
            return float(v[h_idx])

        # stream triples of rows -> one dipole (x,y,z)
        dipoles = []
        buf_vals = [pick(v0)]

        for line in f:
            v = np.fromstring(line, sep=" ")
            buf_vals.append(pick(v))
            if len(buf_vals) == 3:
                dipoles.append(buf_vals)
                buf_vals = []

        M = np.asarray(dipoles, dtype=float)
        if M.ndim != 2 or M.shape[1] != 3:
            raise ValueError(f"{path.name}: failed to parse (got shape {M.shape}).")

    return M, K_snap, h_idx, H

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

        if "macromag" in algo_label:
            algo_human = "GPMOmr"
        elif "ArbVec_backtracking" in algo_label:
            algo_human = "GPMO"
        else:
            algo_human = algo_label

        iterations = np.arange(1, len(R2_history) + 1) * record_every

        cur_min = float(np.min(R2_history))
        cur_max = float(np.max(R2_history))
        if r2_min is None:
            r2_min, r2_max = cur_min, cur_max
        else:
            r2_min = min(r2_min, cur_min)
            r2_max = max(r2_max, cur_max)

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

        (line_fb,) = ax1.semilogy(iterations, R2_history, label=label)

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

    # combined legend placed ABOVE the plot
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(
        h1 + h2,
        l1 + l2,
        fontsize=8,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,  # set to 2 if you want it more compact
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.81])

    fname = save_dir / "Combined_MSE_history.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined plot {fname}")

# ----------------------------
# ΔM histogram (FINAL npz OR mhistory snapshot at K=m_history_delta_M_iter)
# ----------------------------
npz_files = [f for f in outdir.glob("dipoles_final_*.npz") if not f.name.endswith("relax-and-split.npz")]
print(npz_files)

def find_r2_for_npz_suffix(R2_files, npz_suffix_tail):
    # R2history files include K/nphi/ntheta prefix, but should END with the same tail as dipoles_final_*.npz
    return next((r2 for r2 in R2_files if r2.stem.endswith(npz_suffix_tail)), None)

def find_mhistory_for_r2_suffix(mh_files, r2_suffix):
    # This is EXACTLY the same matching rule used in the MSE plot loop
    return next((mh for mh in mh_files if parse_suffix(mh.stem, "mhistory_") == r2_suffix), None)

if len(npz_files) >= 2:
    # load final dipoles (fallback)
    data = {}
    meta = {}  # store matched mhistory per key
    for f in npz_files:
        key_tail = f.stem.split("dipoles_final_")[-1]  # e.g. "bt200_..._macromag_py"
        arr = np.load(f)
        data[key_tail] = arr["m"]

        # match to the SAME mhistory used in the combined MSE plot
        r2_match = find_r2_for_npz_suffix(R2_files, key_tail)
        r2_suffix = parse_suffix(r2_match.stem, "R2history_") if r2_match is not None else None
        mh_match = find_mhistory_for_r2_suffix(mh_files, r2_suffix) if r2_suffix is not None else None

        meta[key_tail] = dict(
            npz=f,
            r2=r2_match,
            r2_suffix=r2_suffix,
            mh=mh_match,
        )

    if len(data) == 2:
        (alg1, M1_final), (alg2, M2_final) = data.items()

        M1, M2 = M1_final, M2_final
        snap_info = "final (.npz)"

        if m_history_delta_M_iter is not None:
            mh1 = meta[alg1]["mh"]
            mh2 = meta[alg2]["mh"]

            if mh1 is not None and mh2 is not None and mh1.exists() and mh2.exists():
                try:
                    M1_snap, K1_snap, idx1, H1 = m_from_mhistory_txt_at_K(mh1, m_history_delta_M_iter, record_every)
                    M2_snap, K2_snap, idx2, H2 = m_from_mhistory_txt_at_K(mh2, m_history_delta_M_iter, record_every)
                    M1, M2 = M1_snap, M2_snap
                    snap_info = f"at K(iter)={m_history_delta_M_iter}"
                    print(f"[INFO] Using ΔM from {snap_info}")
                    print(f"[INFO] mhistory files: {mh1.name}  AND  {mh2.name}")
                except Exception as e:
                    print(f"[WARN] Falling back to final .npz for ΔM (mhistory snapshot failed): {e}")
            else:
                print("[WARN] Falling back to final .npz for ΔM (missing mhistory match).")
                print(f"       {alg1}: mh={None if mh1 is None else mh1.name}")
                print(f"       {alg2}: mh={None if mh2 is None else mh2.name}")

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

        if len(one_only_vals) > 0:
            Mrem_est = float(np.median(one_only_vals))
        else:
            mags = np.r_[np.linalg.norm(M1, axis=1), np.linalg.norm(M2, axis=1)]
            mags = mags[mags > 0]
            Mrem_est = float(np.median(mags)) if len(mags) else np.nan

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
            print(f"  snapshot: {snap_info}")
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
        diffs = diffs[diffs > 0]

        plt.figure()
        plt.hist(diffs, bins=200, alpha=0.7)
        plt.xlabel(r"$|\Delta M| = \|M_i - M'_i\|_2$ [A·m$^2$]")
        plt.ylabel("Number of magnets")
        plt.title(f"Histogram of |ΔM| (linear scale) — {snap_info}")
        plt.grid(True)
        fname_lin = save_dir / "Histogram_DeltaM_linear.png"
        plt.savefig(fname_lin, dpi=180)
        plt.close()
        print(f"Saved linear-scale histogram to {fname_lin}")

        plt.figure()
        plt.hist(diffs, bins=200, alpha=0.7)
        plt.xlabel(r"$|\Delta M| = \|M_i - M'_i\|_2$ [A·m$^2$]")
        plt.ylabel("Number of magnets (log scale)")
        plt.title(f"Histogram of |ΔM| (log scale) — {snap_info}")
        plt.yscale("log")
        plt.grid(True)
        fname_log = save_dir / "Histogram_DeltaM_log.png"
        plt.savefig(fname_log, dpi=180)
        plt.close()
        print(f"Saved log-scale histogram to {fname_log}")
