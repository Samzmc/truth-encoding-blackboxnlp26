"""Generate all paper figures into submission/figures/{draft,final}/.

draft/ : PNG with in-figure titles (for the appendix draft, written around figures)
final/ : PNG + PDF without figure-level titles (camera-ready; minimal panel
         identifiers such as model names are kept because multi-panel figures
         are unreadable without them)

Data sources: workspace/reproduction/ (read-only). No original files touched.

Run from anywhere under the blackboxnlp conda env:
    conda run -n blackboxnlp python submission/figures/make_figures.py
Optionally restrict:  --only B1 C2 D6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
# Root of the full experiment outputs (per-run checkpoints/metrics not shipped
# in this repo; see README for the data release). Override with REPRO_DIR.
REPRO = Path(os.environ.get("REPRO_DIR", HERE.parent.parent / "workspace" / "reproduction"))
ABL = REPRO / "ablation"
PH2 = REPRO / "scaling" / "phase2"
PH3 = REPRO / "scaling" / "phase3"

DRAFT = HERE / "draft"
FINAL = HERE / "final"

RELATION_ORDER = ["P19", "P103", "P101", "P159", "P176", "P138"]
RELATION_NAMES = {
    "P19": "place of birth",
    "P103": "native language",
    "P101": "field of work",
    "P159": "HQ location",
    "P176": "manufacturer",
    "P138": "named after",
}
REL_LABELS = [RELATION_NAMES[r] for r in RELATION_ORDER]
DEPTHS = [0.25, 0.5, 0.75, 1.0]
DEPTH_LABELS = ["25%", "50%", "75%", "100%"]

C_NORMAL = "#1f77b4"
C_ABLATED = "#d62728"
C_8B = "#2a78d6"
C_70B = "#1baf7a"

BLOCKS = {
    "memory":           {"sign": +1, "row": "uy", "col": "ex"},
    "neg_identity":     {"sign": -1, "row": "ex", "col": "ex"},
    "reverse_lookup":   {"sign": +1, "row": "ex", "col": "ey"},
    "self_suppression": {"sign": -1, "row": "uy", "col": "ey"},
}
BLOCK_LABELS = {
    "memory":           r"Memory ($e_x \to u_{g(x)}$)",
    "neg_identity":     r"Neg. identity ($e_x \to -e_x$)",
    "reverse_lookup":   r"Reverse lookup ($e_y \to e_{g^{-1}(y)}$)",
    "self_suppression": r"Self-suppr. ($e_y \to -u_y$)",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def save(fig, name: str, final: bool):
    """Save into draft/ (png) or final/ (png+pdf)."""
    out = FINAL if final else DRAFT
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    if final:
        fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  {'final' if final else 'draft'}/{name}.png")


# ────────────────────── shared loaders ────────────────────── #

def load_auc_curves(dirs, layer_key="L1"):
    """(steps, values[seed, step]) from auc_curve_data.json in each dir."""
    all_steps, all_vals = [], []
    for d in dirs:
        with open(Path(d) / "auc_curve_data.json") as f:
            data = json.load(f)
        steps = [c["step"] for c in data["checkpoints"]]
        vals = [c["auc_per_layer"][layer_key]["mean"] for c in data["checkpoints"]]
        all_steps.append(steps)
        all_vals.append(vals)
    common = sorted(set(all_steps[0]).intersection(*[set(s) for s in all_steps]))
    aligned = [[dict(zip(s, v))[c] for c in common]
               for s, v in zip(all_steps, all_vals)]
    return np.array(common), np.array(aligned)


def band(ax, steps, vals, color, label, ls="-"):
    mean, std = vals.mean(axis=0), vals.std(axis=0)
    ax.plot(steps, mean, ls, color=color, lw=1.5, label=f"{label} (n={len(vals)})")
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)


def toy_metadata(matrix_dir: Path):
    with open(matrix_dir / "metadata.json") as f:
        meta = json.load(f)
    meta["f_map"] = {int(k): v for k, v in meta["f_map"].items()}
    return meta


def toy_reorder(meta):
    N, V, L = meta["N"], meta["V"], meta["L"]
    ex = np.arange(N)
    ey = np.array([meta["f_map"][x] for x in range(N)])
    order = np.concatenate([ex, ey, np.arange(V, V + L), V + L + ex, V + L + ey])
    ranges = {"ex": (0, N), "ey": (N, 2 * N), "pos": (2 * N, 2 * N + L),
              "ux": (2 * N + L, 3 * N + L), "uy": (3 * N + L, 4 * N + L)}
    return order, ranges


def toy_block_metrics(matrix_dir: Path, step: int):
    """{block: (a_B, frob_norm)} at a given step."""
    meta = toy_metadata(matrix_dir)
    order, ranges = toy_reorder(meta)
    W = np.load(matrix_dir / f"W_step{step}.npy")
    W_re = W[np.ix_(order, order)]
    out = {}
    for name, spec in BLOCKS.items():
        r0, r1 = ranges[spec["row"]]
        c0, c1 = ranges[spec["col"]]
        B = W_re[r0:r1, c0:c1]
        n = B.shape[0]
        a_B = spec["sign"] * np.trace(B) / n  # <B, sI> / ||I||^2
        out[name] = (float(a_B), float(np.linalg.norm(B)))
    return out


TOY_NORMAL = [ABL / "phase0_original" / f"toy_rho08_seed{s}" / "analysis"
              for s in range(5)]
TOY_ABLATED = [ABL / "phase1_ablation" / f"toy_rho08_seed{s}" / "analysis"
               for s in range(5)]


# ────────────────────── Block A ────────────────────── #

def draw_toy_heatmap(ax, W, meta, vmin=None, vmax=None):
    im = ax.imshow(W, interpolation="nearest", vmin=vmin, vmax=vmax)
    N, V, L = meta["N"], meta["V"], meta["L"]
    bounds = [0, N, V, V + L, V + L + N, V + L + V]
    labels = [r"$e_x$", r"$e_y$", "pos", r"$u_x$", r"$u_y$"]
    mid = [(bounds[i] + bounds[i + 1]) / 2 for i in range(5)]
    ax.set_xticks(mid), ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(mid), ax.set_yticklabels(labels, fontsize=11)
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="white", lw=0.8, alpha=0.7)
        ax.axvline(b - 0.5, color="white", lw=0.8, alpha=0.7)
    ax.grid(False)
    return im


def fig_A1(final: bool):
    mdir = TOY_NORMAL[0] / "ov_matrices"
    meta = toy_metadata(mdir)
    W = np.load(mdir / "W_step2000.npy")
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    im = draw_toy_heatmap(ax, W, meta)
    fig.colorbar(im, ax=ax, shrink=0.85)
    if not final:
        ax.set_title(r"$W = W_o W_v$ at step 2000 (toy, $\rho=0.8$, seed 0)")
    save(fig, "A1_toy_ov_heatmap", final)


# ────────────────────── Block B ────────────────────── #

def fig_B1(final: bool):
    panels = [
        ("(a) fully-trained, $\\rho=0.99$",
         [ABL / "phase0_original" / f"L1_rho099_seed{s}" / "analysis" for s in range(5)],
         [ABL / "phase1_ablation" / f"L1_rho099_seed{s}" / "analysis" for s in range(5)]),
        ("(b) fully-trained, $\\rho=0.8$",
         [ABL / "phase0_original" / f"L1_rho08_seed{s}" / "analysis" for s in range(3)],
         [ABL / "phase1_ablation" / f"L1_rho08_seed{s}" / "analysis" for s in range(3)]),
        ("(c) toy, $\\rho=0.8$",
         [d for d in TOY_NORMAL], [d for d in TOY_ABLATED]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (label, nd, ad) in zip(axes, panels):
        sn, vn = load_auc_curves(nd)
        sa, va = load_auc_curves(ad)
        for steps, vals, color, lab, ls in [(sn, vn, C_NORMAL, "Normal", "-"),
                                            (sa, va, C_ABLATED, "Ablated", "--")]:
            mean, std = vals.mean(axis=0), vals.std(axis=0)
            ax.plot(steps, mean, ls, color=color, lw=1.5, label=lab)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.6)
        ax.set_xlabel("Training step")
        ax.set_ylim(0.4, 1.02)
        ax.set_title(f"{label} ($n={len(nd)}$)", fontsize=10)
    axes[0].set_ylabel("Probe AUC (pos $x'$, final layer)")
    axes[0].legend(loc="center right", fontsize=8)
    if not final:
        fig.suptitle("Linear truth decodability: shared truth vs independent truth",
                     y=1.02)
    fig.tight_layout()
    save(fig, "B1_ablation_auc_triptych", final)


def fig_B2(final: bool):
    dirs = [TOY_NORMAL[0] / "ov_matrices", TOY_ABLATED[0] / "ov_matrices"]
    labels = ["Normal (shared truth)", "Ablated (independent truth)"]
    Ws, metas = [], []
    for d in dirs:
        metas.append(toy_metadata(d))
        Ws.append(np.load(d / "W_step2000.npy"))
    vmax = max(np.abs(W).max() for W in Ws)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, W, meta, lab in zip(axes, Ws, metas, labels):
        im = draw_toy_heatmap(ax, W, meta, vmin=-vmax, vmax=vmax)
        ax.set_title(lab, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    if not final:
        fig.suptitle(r"Toy-model $W = W_o W_v$ at step 2000 (seed 0, shared color scale)",
                     y=1.0)
    save(fig, "B2_toy_ov_heatmaps_pair", final)


def fig_B3(final: bool):
    def collect(dirs):
        per_seed = []
        for d in dirs:
            with open(d / "cosine_similarity.json") as f:
                per_seed.append(json.load(f)["checkpoints"])
        step_sets = [set(r["step"] for r in res) for res in per_seed]
        common = sorted(step_sets[0].intersection(*step_sets[1:]))
        arr = {}
        for name in BLOCKS:
            arr[name] = np.array([
                [dict((r["step"], r[name]) for r in res)[s] for s in common]
                for res in per_seed])
        return common, arr

    sn, an = collect(TOY_NORMAL)
    sa, aa = collect(TOY_ABLATED)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.6), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, BLOCKS):
        band(ax, sn, an[name], C_NORMAL, "Normal")
        band(ax, sa, aa[name], C_ABLATED, "Ablated", ls="--")
        ax.set_title(BLOCK_LABELS[name], fontsize=9)
        ax.set_ylim(-0.15, 1.05)
    for ax in axes[1]:
        ax.set_xlabel("Training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cosine sim. with target")
    axes[0, 0].legend(fontsize=8)
    if not final:
        fig.suptitle("OV sub-block orientation survives the ablation "
                     "(cosine similarity with signed-identity target)", y=1.0)
    fig.tight_layout()
    save(fig, "B3_ov_cosine_4panel", final)


def fig_B4(final: bool):
    def load_prob(dirs, key):
        all_steps, all_vals = [], []
        for d in dirs:
            with open(Path(d) / "prob_curve_data.json") as f:
                data = json.load(f)
            steps = [c["step"] for c in data["checkpoints"]]
            vals = [c[key]["mean"] if isinstance(c[key], dict) else c[key]
                    for c in data["checkpoints"]]
            all_steps.append(steps)
            all_vals.append(vals)
        common = sorted(set(all_steps[0]).intersection(*[set(s) for s in all_steps]))
        aligned = [[dict(zip(s, v))[c] for c in common]
                   for s, v in zip(all_steps, all_vals)]
        return np.array(common), np.array(aligned)

    nd = [ABL / "phase0_original" / f"L1_rho08_seed{s}" / "analysis" for s in range(3)]
    ad = [ABL / "phase1_ablation" / f"L1_rho08_seed{s}" / "analysis" for s in range(3)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, key, lab in zip(
            axes, ["true", "false"],
            [r"$P(\mathrm{correct} \mid \mathrm{true\ sequence})$",
             r"$P(\mathrm{correct} \mid \mathrm{false\ sequence})$"]):
        s_n, v_n = load_prob(nd, key)
        s_a, v_a = load_prob(ad, key)
        band(ax, s_n, v_n, C_NORMAL, "Normal")
        band(ax, s_a, v_a, C_ABLATED, "Ablated", ls="--")
        ax.set_xlabel("Training step")
        ax.set_title(lab, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.axhline(0.8, color="gray", ls=":", alpha=0.6)
    axes[0].set_ylabel(r"$P(g(x') \mid x, y, x')$")
    axes[0].legend(fontsize=8)
    if not final:
        fig.suptitle("Correct-token probability — fully-trained, $\\rho=0.8$",
                     y=1.02)
    fig.tight_layout()
    save(fig, "B4_prob_comparison_rho08", final)


def fig_B5(final: bool):
    step = 2000
    metrics_n = [toy_block_metrics(d / "ov_matrices", step) for d in TOY_NORMAL]
    metrics_a = [toy_block_metrics(d / "ov_matrices", step) for d in TOY_ABLATED]

    names = list(BLOCKS)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, idx, ylab, title in zip(
            axes, [0, 1],
            [r"Signed target projection $a_B$", "Frobenius norm"],
            ["Mechanism amplitude", "Total block strength"]):
        x = np.arange(len(names))
        for off, metrics, color, lab in [(-0.2, metrics_n, C_NORMAL, "Normal"),
                                          (+0.2, metrics_a, C_ABLATED, "Ablated")]:
            vals = np.array([[m[n][idx] for n in names] for m in metrics])
            ax.bar(x + off, vals.mean(axis=0), width=0.38, color=color,
                   alpha=0.85, label=f"{lab} (n={len(metrics)})",
                   yerr=vals.std(axis=0), capsize=3,
                   error_kw={"lw": 1}, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([BLOCK_LABELS[n] for n in names],
                           fontsize=7, rotation=12)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    if not final:
        fig.suptitle("Toy OV sub-blocks at step 2000: amplitude collapses "
                     "under ablation, orientation does not", y=1.03)
    fig.tight_layout()
    save(fig, "B5_ov_amplitude", final)

    # sanity check against previously computed reference values
    print("    a_B check (normal | ablated), reference: memory 3.61|6.08, "
          "neg_id 3.59|0.52, rev 4.01|0.39, selfsup 3.88|4.55")
    for n in names:
        mn = np.mean([m[n][0] for m in metrics_n])
        ma = np.mean([m[n][0] for m in metrics_a])
        print(f"      {n:18s} {mn:6.3f} | {ma:6.3f}")


# ────────────────────── Block C ────────────────────── #

F1_ORDER = ["FF", "TT", "FFF", "TTT", "FFFF", "TTTT"]


def fig_C1(final: bool):
    df = pd.read_parquet(PH3 / "results" / "context_nll.parquet")
    rng_master = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, model in zip(axes, ("8B", "70B")):
        sub = df[(df["model"] == model) & (df["stage"] == "a")]
        rng = np.random.default_rng(0)
        data = [sub[sub["condition"] == c]["nll"].to_numpy() for c in F1_ORDER]
        bp = ax.boxplot(data, positions=range(len(F1_ORDER)), widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops={"linewidth": 1.8, "color": "black"})
        for box in bp["boxes"]:
            box.set_facecolor("#dddddd")
            box.set_alpha(0.6)
        palette = plt.cm.tab10.colors
        for x, ys in enumerate(data):
            c = palette[x % len(palette)]
            ax.scatter(x + 0.15 * (rng.random(len(ys)) - 0.5), ys,
                       s=7, alpha=0.35, color=c, zorder=3)
            ax.scatter(x, np.median(ys), marker="D", s=45, color=c,
                       edgecolor="black", linewidth=0.5, zorder=4)
        base = sub[sub["condition"] == "none"]["nll"]
        if not base.empty:
            ax.axhline(base.median(), color="gray", ls="--", lw=1.1, alpha=0.8)
        ax.set_xticks(range(len(F1_ORDER)))
        ax.set_xticklabels(F1_ORDER, fontsize=11)
        ax.set_xlabel("Context factuality", fontsize=11)
        ax.set_title(f"Llama-3.1-{model}", fontsize=11)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.xaxis.grid(False)
    axes[0].set_ylabel(r"$-\log P(\mathrm{correct\ attribute})$", fontsize=11)
    if not final:
        fig.suptitle("NLL of the correct attribute vs. context truthfulness "
                     "(CounterFact P103; dashed = no-context median)", y=1.02)
    fig.tight_layout()
    save(fig, "C1_fig6a_combined", final)


def fig_C2(final: bool):
    mats = {m: pd.read_csv(PH3 / "results" / f"grid_matrix_{m}.csv", index_col=0)
                 .reindex(index=RELATION_ORDER, columns=RELATION_ORDER)
            for m in ("70B", "8B")}
    vmax = max(m.abs().max().max() for m in mats.values())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (model, m) in zip(axes, mats.items()):
        im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(6))
        ax.set_xticklabels(REL_LABELS, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(6))
        ax.set_yticklabels(REL_LABELS, fontsize=8)
        ax.set_xlabel("Target relation", fontsize=10)
        ax.set_ylabel("Context relation", fontsize=10)
        ax.set_title(f"Llama-3.1-{model}", fontsize=11)
        ax.grid(False)
        for i in range(6):
            for j in range(6):
                v = m.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 * vmax else "black")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(r"median paired $\Delta$NLL (F $-$ T)", fontsize=8)
    if not final:
        fig.suptitle("Behavioral transfer of the context-truthfulness effect",
                     y=1.0)
    fig.tight_layout()
    save(fig, "C2_behavioral_grid", final)


# ────────────────────── Block D ────────────────────── #

def load_phase2():
    gap = pd.read_csv(PH2 / "results" / "generality_gap.csv")
    with open(PH2 / "results" / "selected_layers.json") as f:
        selected = json.load(f)
    m8 = pd.read_csv(PH2 / "results" / "stage2_matrix_8b.csv", index_col=0)
    m70 = pd.read_csv(PH2 / "results" / "stage2_matrix_70b.csv", index_col=0)
    return gap, selected, m8, m70


def fig_D1(final: bool):
    gap, selected, _, _ = load_phase2()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for key, color, label in [("llama_3_1_8b", C_8B, "8B"),
                              ("llama_3_1_70b", C_70B, "70B")]:
        m = gap[gap["model_key"] == key]
        wm, ws, lm, ls_ = [], [], [], []
        for d in DEPTHS:
            dd = m[m["normalized_depth"] == d]
            wm.append(dd["within_mean_auc"].mean())
            ws.append(dd["within_mean_auc"].std())
            lm.append(dd["loo_auc"].mean())
            ls_.append(dd["loo_auc"].std())
        x = np.arange(len(DEPTHS))
        ax.errorbar(x, wm, yerr=ws, color=color, marker="o", ls="-",
                    label=f"{label} within", capsize=3,
                    markeredgecolor="white", markeredgewidth=1)
        ax.errorbar(x, lm, yerr=ls_, color=color, marker="s", ls="--",
                    label=f"{label} leave-one-out", capsize=3,
                    markeredgecolor="white", markeredgewidth=1)
    for key, color in [("llama_3_1_8b", C_8B), ("llama_3_1_70b", C_70B)]:
        ax.axvline(DEPTHS.index(selected[key]["normalized_depth"]),
                   color=color, alpha=0.15, lw=8, zorder=0)
    ax.set_xticks(range(len(DEPTHS))), ax.set_xticklabels(DEPTH_LABELS)
    ax.set_xlabel("Normalized depth"), ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.88, 1.005)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower left", framealpha=0.9, edgecolor="none")
    if not final:
        ax.set_title("Within-relation and leave-one-out AUC by depth")
    fig.tight_layout()
    save(fig, "D1_depth_profile", final)


def fig_D2(final: bool):
    _, _, m8, m70 = load_phase2()
    blues = LinearSegmentedColormap.from_list("custom_blues", [
        "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])
    fig = plt.figure(figsize=(10.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.45)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    cax = fig.add_subplot(gs[0, 2])
    for k, (ax, mat, title) in enumerate(zip(
            axes, [m8, m70],
            ["Llama-3.1-8B (layer 7, depth 25%)",
             "Llama-3.1-70B (layer 39, depth 50%)"])):
        data = mat.loc[RELATION_ORDER, RELATION_ORDER].astype(float).values
        im = ax.imshow(data, cmap=blues, vmin=0.84, vmax=1.0, aspect="equal")
        for i in range(6):
            for j in range(6):
                v = data[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.97 else "#0b0b0b",
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(6))
        ax.set_xticklabels(REL_LABELS, fontsize=7, rotation=30, ha="right")
        ax.set_yticks(range(6))
        ax.set_yticklabels(REL_LABELS, fontsize=7)
        ax.set_xlabel("Target relation")
        if k == 0:
            ax.set_ylabel("Source relation")
        ax.set_title(title, fontsize=9, pad=8)
        ax.grid(False)
        ax.tick_params(length=0)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("ROC-AUC", fontsize=9)
    save(fig, "D2_probe_transfer_matrices", final)


def fig_D3(final: bool):
    gap, _, _, _ = load_phase2()
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    x = np.arange(len(DEPTHS))
    for i, (key, color, label) in enumerate([
            ("llama_3_1_8b", C_8B, "8B"), ("llama_3_1_70b", C_70B, "70B")]):
        m = gap[gap["model_key"] == key]
        vals = [m[m["normalized_depth"] == d]["generality_gap"].abs().mean()
                for d in DEPTHS]
        bars = ax.bar(x + (i - 0.5) * 0.35, vals, 0.31, color=color,
                      alpha=0.85, label=label, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                    color="#52514e")
    ax.set_xticks(x), ax.set_xticklabels(DEPTH_LABELS)
    ax.set_xlabel("Normalized depth"), ax.set_ylabel("Mean |generality gap|")
    ax.set_ylim(0, 0.06)
    ax.legend(framealpha=0.9, edgecolor="none")
    if not final:
        ax.set_title("Generality gap: 8B vs 70B")
    fig.tight_layout()
    save(fig, "D3_generality_gap", final)


def fig_D4(final: bool):
    gap, _, _, _ = load_phase2()
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.5), sharex=True, sharey=True)
    for idx, rel in enumerate(RELATION_ORDER):
        ax = axes[idx // 3, idx % 3]
        for key, color, label in [("llama_3_1_8b", C_8B, "8B"),
                                  ("llama_3_1_70b", C_70B, "70B")]:
            m = gap[(gap["model_key"] == key) & (gap["target_relation"] == rel)]
            m = m.sort_values("normalized_depth")
            x = np.arange(len(DEPTHS))
            ax.plot(x, m["within_mean_auc"].values, color=color, marker="o",
                    markersize=4, label=f"{label} within",
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.plot(x, m["loo_auc"].values, color=color, marker="s",
                    markersize=4, ls="--", label=f"{label} leave-one-out",
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.set_title(RELATION_NAMES[rel], fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(DEPTHS))), ax.set_xticklabels(DEPTH_LABELS, fontsize=7)
        ax.set_ylim(0.80, 1.01)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if idx == 0:
            ax.legend(fontsize=6, loc="lower left", framealpha=0.9,
                      edgecolor="none")
    fig.supxlabel("Normalized depth", fontsize=9)
    fig.supylabel("ROC-AUC", fontsize=9)
    if not final:
        fig.suptitle("Per-relation AUC profiles", y=1.0)
    fig.tight_layout()
    save(fig, "D4_relation_profiles", final)


def fig_D5(final: bool):
    gap, _, _, _ = load_phase2()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    n = len(RELATION_ORDER)
    for ax, key, color, title, depth in [
            (axes[0], "llama_3_1_8b", C_8B, "8B (layer 7)", 0.25),
            (axes[1], "llama_3_1_70b", C_70B, "70B (layer 39)", 0.5)]:
        m = gap[(gap["model_key"] == key) & (gap["normalized_depth"] == depth)]
        m = m.set_index("target_relation").loc[RELATION_ORDER]
        gaps = m["generality_gap"].values
        colors = [color if g >= 0 else "#e34948" for g in gaps]
        bars = ax.barh(np.arange(n), gaps, height=0.6, color=colors,
                       alpha=0.8, edgecolor="white")
        for b, v in zip(bars, gaps):
            ax.text(v + (0.002 if v >= 0 else -0.002),
                    b.get_y() + b.get_height() / 2,
                    f"{v:+.3f}" if abs(v) >= 0.0005 else "0.000",
                    ha="left" if v >= 0 else "right", va="center",
                    fontsize=7, color="#52514e")
        ax.set_yticks(np.arange(n)), ax.set_yticklabels(REL_LABELS, fontsize=8)
        ax.axvline(0, color="#c3c2b7", lw=0.8, zorder=0)
        ax.set_xlabel("Generality gap (within − leave-one-out)", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.035, 0.065)
        ax.set_ylim(n - 0.5, -0.5)
    if not final:
        fig.suptitle("Per-relation generality gap at selected layer", y=1.02)
    fig.tight_layout()
    save(fig, "D5_relation_gap_detail", final)


def fig_D6(final: bool):
    """Behavioral ΔNLL vs probe transfer AUC, 36 cells per model."""
    from scipy.stats import spearmanr
    _, _, m8, m70 = load_phase2()
    probe = {"8B": m8, "70B": m70}
    behav = {m: pd.read_csv(PH3 / "results" / f"grid_matrix_{m}.csv", index_col=0)
             for m in ("8B", "70B")}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, model, color in zip(axes, ("8B", "70B"), (C_8B, C_70B)):
        p = probe[model].loc[RELATION_ORDER, RELATION_ORDER].astype(float)
        b = behav[model].reindex(index=RELATION_ORDER,
                                 columns=RELATION_ORDER).astype(float)
        xs, ys, diag = [], [], []
        for i, ri in enumerate(RELATION_ORDER):
            for j, rj in enumerate(RELATION_ORDER):
                xs.append(p.iloc[i, j])
                ys.append(b.iloc[i, j])
                diag.append(i == j)
        xs, ys, diag = np.array(xs), np.array(ys), np.array(diag)
        ax.scatter(xs[~diag], ys[~diag], s=28, alpha=0.75, color=color,
                   edgecolor="white", linewidth=0.5, label="off-diagonal")
        ax.scatter(xs[diag], ys[diag], s=55, marker="D", facecolor="none",
                   edgecolor=color, linewidth=1.4, label="diagonal")
        rho_all, p_all = spearmanr(xs, ys)
        ax.annotate(
            rf"Spearman $\rho={rho_all:.2f}$ ($p={p_all:.2f}$, $n=36$)",
            xy=(0.03, 0.92), xycoords="axes fraction", fontsize=8)
        ax.set_xlabel("Probe transfer ROC-AUC (source → target)")
        ax.set_title(f"Llama-3.1-{model}", fontsize=10)
    axes[0].set_ylabel(r"Behavioral $\Delta$NLL (context → target)")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 0.88))
    if not final:
        fig.suptitle("Representational vs behavioral transfer structure: "
                     "no rank correlation", y=1.02)
    fig.tight_layout()
    save(fig, "D6_behavioral_vs_probe_scatter", final)


# ────────────────────── main ────────────────────── #

ALL_FIGS = {
    "A1": fig_A1,
    "B1": fig_B1, "B2": fig_B2, "B3": fig_B3, "B4": fig_B4, "B5": fig_B5,
    "C1": fig_C1, "C2": fig_C2,
    "D1": fig_D1, "D2": fig_D2, "D3": fig_D3, "D4": fig_D4, "D5": fig_D5,
    "D6": fig_D6,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", choices=sorted(ALL_FIGS), default=None)
    args = p.parse_args()
    setup_style()
    todo = args.only or list(ALL_FIGS)
    for name in todo:
        print(f"{name}:")
        for final in (False, True):
            ALL_FIGS[name](final)
    print("Done.")


if __name__ == "__main__":
    main()
