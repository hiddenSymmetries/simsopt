#!/usr/bin/env python

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import YAML


def _load_yaml(path: Path) -> Dict[str, Any]:
    yaml = YAML(typ="safe")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.load(f) or {}
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected YAML mapping")
    return obj


def _load_runhistory_csv(path: Path) -> Dict[str, np.ndarray]:
    ks: List[int] = []
    fB: List[float] = []
    absBn: List[float] = []
    n_active: List[int] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ks.append(int(row["k"]))
            fB.append(float(row["fB"]))
            absBn.append(float(row["absBn"]))
            n_active.append(int(row.get("n_active", 0)))

    return {
        "k": np.asarray(ks, dtype=int),
        "fB": np.asarray(fB, dtype=float),
        "absBn": np.asarray(absBn, dtype=float),
        "n_active": np.asarray(n_active, dtype=int),
    }


def _human_algorithm(run_doc: Dict[str, Any]) -> str:
    alg = str(run_doc.get("algorithm", ""))
    if not alg:
        return "unknown"
    return alg


def _style_for_run(run_doc: Dict[str, Any]) -> Dict[str, Any]:
    material = str(run_doc.get("material", {}).get("name", ""))
    alg = _human_algorithm(run_doc)

    colors = {
        "N52": "tab:blue",
        "GB50UH": "tab:orange",
        "AlNiCo": "tab:green",
    }

    linestyle = "--" if alg == "GPMOmr" else "-"
    color = colors.get(material, None)
    return {"linestyle": linestyle, "color": color}


def _label_for_run(run_doc: Dict[str, Any]) -> str:
    alg = _human_algorithm(run_doc)
    mat = run_doc.get("material", {}).get("name", None)
    p = run_doc.get("params", {}) or {}

    parts = [alg]
    if mat:
        parts.append(f"mat={mat}")

    K_max = p.get("K", None)
    if K_max is not None:
        parts.append(rf"$K_{{\mathrm{{max}}}}={int(K_max)}$")

    bt = p.get("backtracking", None)
    if bt is not None:
        parts.append(f"bt={int(bt)}")

    Nadj = p.get("Nadjacent", None)
    if Nadj is not None:
        parts.append(f"Nadj={int(Nadj)}")

    kmm = p.get("mm_refine_every", 0) or 0
    if int(kmm) > 0:
        parts.append(rf"$k_{{\mathrm{{mm}}}}={int(kmm)}$")
    return r"$f_B$ (" + ", ".join(parts) + ")"


def _label_for_mse(run_doc: Dict[str, Any], *, compact: bool) -> str:
    """
    Label used in the MSE-history plots.

    For paper-style plots we prefer short labels (algorithm + key knobs) so the
    legend remains readable.
    """
    alg = _human_algorithm(run_doc)
    params = run_doc.get("params", {}) or {}

    if not compact:
        return _label_for_run(run_doc)

    # Compact label still includes key knobs used to identify paper runs.
    K_max = params.get("K", None)
    bt = params.get("backtracking", None)
    Nadj = params.get("Nadjacent", None)
    kmm = params.get("mm_refine_every", 0) or 0

    parts = [alg]
    if K_max is not None:
        parts.append(rf"$K_{{\mathrm{{max}}}}={int(K_max)}$")
    if bt is not None:
        parts.append(f"bt={int(bt)}")
    if Nadj is not None:
        parts.append(f"Nadj={int(Nadj)}")
    if alg == "GPMOmr" and int(kmm) > 0:
        parts.append(rf"$k_{{\mathrm{{mm}}}}={int(kmm)}$")

    return r"$f_B$ (" + ", ".join(parts) + ")"


def _color_cycle(n: int) -> List[str]:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def _color_for_algorithm(alg: str) -> str:
    # Paper-style: keep consistent colors across plots.
    if alg == "GPMO":
        return "tab:blue"
    if alg == "GPMOmr":
        return "tab:orange"
    return "tab:gray"


def _collect_runs(outdir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for csv_path in sorted(outdir.glob("runhistory_*.csv")):
        run_id = csv_path.stem.split("runhistory_", 1)[-1]
        yaml_path = outdir / f"run_{run_id}.yaml"
        run_doc = _load_yaml(yaml_path) if yaml_path.exists() else {"run_id": run_id}
        hist = _load_runhistory_csv(csv_path)
        runs.append({"run_id": run_id, "run": run_doc, "hist": hist, "csv": csv_path, "yaml": yaml_path})
    return runs


def _select_runs(runs: List[Dict[str, Any]], patterns: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not patterns:
        return runs
    out: List[Dict[str, Any]] = []
    for r in runs:
        rid = r["run_id"]
        if any(p in rid for p in patterns):
            out.append(r)
    return out


def plot_mse(
    runs: List[Dict[str, Any]],
    save_dir: Path,
    *,
    show_n_active: bool = True,
    distinct_n_active: bool = False,
) -> None:
    if not runs:
        raise SystemExit("No runs found (expected runhistory_*.csv).")

    is_scan = len(runs) > 2
    scan_colors = _color_cycle(len(runs)) if is_scan else []

    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))
    ax2 = ax1.twinx() if show_n_active else None

    for idx, r in enumerate(runs):
        run_doc = r["run"]
        hist = r["hist"]

        alg = _human_algorithm(run_doc)
        label = _label_for_mse(run_doc, compact=True)

        if is_scan:
            # Paper run 01: multiple curves (kmm sweep). Use dashed lines and distinct colors.
            line_style = "--"
            color = scan_colors[idx]
        else:
            # Paper runs 02/03/04: two curves. Use solid lines and algorithm-coded colors.
            line_style = "-"
            color = _color_for_algorithm(alg)

        (line_fb,) = ax1.semilogy(
            hist["k"],
            hist["fB"],
            label=label,
            linestyle=line_style,
            color=color,
        )
        if ax2 is not None:
            if distinct_n_active:
                # Optional: distinguish N_active curves even when they overlap heavily.
                if alg == "GPMO":
                    na_linestyle = "--"
                    na_linewidth = 1.4
                    na_marker = None
                    na_markevery = None
                    na_markersize = None
                elif alg == "GPMOmr":
                    na_linestyle = "-."
                    na_linewidth = 1.0
                    na_marker = "*"
                    na_markevery = max(1, int(len(hist["k"]) / 20))
                    na_markersize = 4
                else:
                    na_linestyle = "--"
                    na_linewidth = 1.0
                    na_marker = None
                    na_markevery = None
                    na_markersize = None
            else:
                # Default: keep N_active styling consistent across runs.
                na_linestyle = "--"
                na_linewidth = 1.0
                na_marker = None
                na_markevery = None
                na_markersize = None

            ax2.plot(
                hist["k"],
                hist["n_active"],
                linestyle=na_linestyle,
                linewidth=na_linewidth,
                marker=na_marker,
                markevery=na_markevery,
                markersize=na_markersize,
                color=line_fb.get_color(),
                label=rf"$N_{{\rm active}}$ ({_human_algorithm(run_doc)})",
            )

    ax1.grid(True)
    ax1.set_xlabel(r"Iteration $K$")
    ax1.set_ylabel(r"$f_B$ [T$^2$ m$^2$]")
    if ax2 is not None:
        ax2.set_ylabel(r"Number of active magnets")

    if ax2 is None:
        ax1.legend(fontsize=8, frameon=True, facecolor="white", edgecolor="black", loc="best")
        fig.tight_layout()
    else:
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        fig.legend(
            h1 + h2,
            l1 + l2,
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="black",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=1,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.81])

    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / "Combined_MSE_history.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def _load_final_m_from_run(outdir: Path, run_doc: Dict[str, Any]) -> Optional[np.ndarray]:
    npz_name = run_doc.get("artifacts", {}).get("dipoles_final_npz", None)
    if not npz_name:
        return None
    npz_path = outdir / str(npz_name)
    if not npz_path.exists():
        return None
    arr = np.load(npz_path)
    return arr["m"]


def _choose_two_runs_for_compare(
    runs: List[Dict[str, Any]],
    outdir: Path,
    patterns: Optional[List[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    sel = _select_runs(runs, patterns)
    if len(sel) != 2:
        raise SystemExit(f"Need exactly 2 runs to compare; found {len(sel)}.")
    r1, r2 = sel
    m1 = _load_final_m_from_run(outdir, r1["run"])
    m2 = _load_final_m_from_run(outdir, r2["run"])
    if m1 is None or m2 is None:
        raise SystemExit("Missing dipoles_final_*.npz for one of the selected runs.")
    if m1.shape != m2.shape:
        raise SystemExit(f"Shape mismatch: {m1.shape} vs {m2.shape}")
    return r1, r2, m1, m2


def plot_delta_m(runs: List[Dict[str, Any]], outdir: Path, save_dir: Path, compare: Optional[List[str]]) -> None:
    r1, r2, M1, M2 = _choose_two_runs_for_compare(runs, outdir, compare)
    tag = f"{_human_algorithm(r1['run'])}_vs_{_human_algorithm(r2['run'])}"

    diffs_raw: List[float] = []
    one_only_vals: List[float] = []
    n_both_zero = 0

    for v1, v2 in zip(M1, M2):
        z1 = np.allclose(v1, 0.0)
        z2 = np.allclose(v2, 0.0)
        if z1 and z2:
            n_both_zero += 1
            continue
        if z1 ^ z2:
            d = np.linalg.norm(v2) if z1 else np.linalg.norm(v1)
            one_only_vals.append(float(d))
        else:
            d = np.linalg.norm(v1 - v2)
        diffs_raw.append(float(d))

    diffs_raw_arr = np.asarray(diffs_raw, dtype=float)
    diffs = diffs_raw_arr[diffs_raw_arr > 0]

    Mrem_est = float(np.median(one_only_vals)) if one_only_vals else float(np.nan)
    if np.isfinite(Mrem_est) and Mrem_est > 0:
        b01 = 0.5 * Mrem_est
        b12 = 0.5 * (1.0 + np.sqrt(2.0)) * Mrem_est
        b23 = 0.5 * (np.sqrt(2.0) + 2.0) * Mrem_est

        c0 = int(np.sum(diffs_raw_arr < b01))
        c1 = int(np.sum((diffs_raw_arr >= b01) & (diffs_raw_arr < b12)))
        c2 = int(np.sum((diffs_raw_arr >= b12) & (diffs_raw_arr < b23)))
        c3 = int(np.sum(diffs_raw_arr >= b23))

        counts = np.array([c0, c1, c2, c3], dtype=int)
        N_union = len(diffs_raw_arr)
        N_total = M1.shape[0]
        pct_union = 100.0 * counts / max(N_union, 1)
        pct_total = 100.0 * counts / max(N_total, 1)

        print("\nΔM bucket breakdown:")
        print(f"  estimated M_rem: {Mrem_est:.6e} [A·m^2]")
        print(f"  total sites: {N_total}")
        print(f"  both-zero sites: {n_both_zero} ({100.0*n_both_zero/max(N_total,1):.2f}%)")
        print(f"  union-active sites: {N_union} ({100.0*N_union/max(N_total,1):.2f}%)")
        labels = ["near 0", "near M_rem", "near sqrt(2) M_rem", "near 2 M_rem"]
        for lab, n, pu, pt in zip(labels, counts, pct_union, pct_total):
            print(f"  {lab:>18}: {n:6d}  ({pu:6.2f}% of union-active, {pt:6.2f}% of all sites)")

    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(diffs, bins=200, alpha=0.7)
    plt.xlabel(r"$|\Delta M| = \|M_i - M'_i\|_2$ [A·m$^2$]")
    plt.ylabel("Number of magnets")
    plt.title(f"Histogram of |ΔM| (linear) — {tag}")
    plt.grid(True)
    fname_lin = save_dir / f"Histogram_DeltaM_linear_{tag}.png"
    plt.savefig(fname_lin, dpi=180)
    plt.close()
    print(f"Saved {fname_lin}")

    plt.figure()
    plt.hist(diffs, bins=200, alpha=0.7)
    plt.xlabel(r"$|\Delta M| = \|M_i - M'_i\|_2$ [A·m$^2$]")
    plt.ylabel("Number of magnets (log scale)")
    plt.title(f"Histogram of |ΔM| (log) — {tag}")
    plt.yscale("log")
    plt.grid(True)
    fname_log = save_dir / f"Histogram_DeltaM_log_{tag}.png"
    plt.savefig(fname_log, dpi=180)
    plt.close()
    print(f"Saved {fname_log}")


def main():
    p = argparse.ArgumentParser(description="Plot MUSE permanent-magnet run artifacts (YAML+CSV).")
    p.add_argument("--outdir", type=Path, default=Path("output_permanent_magnet_GPMO_MUSE"))
    p.add_argument("--mode", choices=["mse", "deltam", "all"], default="all")
    p.add_argument("--save-dir", type=Path, default=None)
    p.add_argument(
        "--no-n-active",
        action="store_true",
        help="Disable the secondary axis showing N_active on MSE plots.",
    )
    p.add_argument(
        "--distinct-n-active",
        action="store_true",
        help="Use distinct line styles/markers for N_active curves (useful when curves overlap).",
    )
    p.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional substrings to select runs (matched against run_id).",
    )
    p.add_argument(
        "--compare",
        nargs="*",
        default=None,
        help="For --mode deltam: 2 substrings selecting the two runs to compare.",
    )
    args = p.parse_args()

    outdir = args.outdir
    save_dir = args.save_dir if args.save_dir is not None else (outdir / "plots")

    runs = _collect_runs(outdir)
    runs = _select_runs(runs, args.runs)

    if args.mode in ("mse", "all"):
        plot_mse(
            runs,
            save_dir,
            show_n_active=not args.no_n_active,
            distinct_n_active=args.distinct_n_active,
        )
    if args.mode in ("deltam", "all"):
        if args.compare is None and len(runs) != 2:
            print(
                f"[INFO] Skipping ΔM: need exactly 2 runs (found {len(runs)}). "
                "Use `--compare <substr1> <substr2>` to select two runs."
            )
        else:
            plot_delta_m(runs, outdir, save_dir, args.compare)


if __name__ == "__main__":
    main()
