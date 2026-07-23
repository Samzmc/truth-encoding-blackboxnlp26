"""Generate Phase 2 publication figures.

Run from this directory under the blackboxnlp conda env:
    python generate_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

RELATION_ORDER = ["P19", "P103", "P101", "P159", "P176", "P138"]
RELATION_LABELS = {
    "P19": "P19\nplace of birth",
    "P103": "P103\nnative language",
    "P101": "P101\nfield of work",
    "P159": "P159\nHQ location",
    "P176": "P176\nmanufacturer",
    "P138": "P138\nnamed after",
}
RELATION_SHORT = {
    "P19": "P19",
    "P103": "P103",
    "P101": "P101",
    "P159": "P159",
    "P176": "P176",
    "P138": "P138",
}

C_8B = "#2a78d6"
C_70B = "#1baf7a"
C_8B_LIGHT = "#86b6ef"
C_70B_LIGHT = "#7dd4b0"

DEPTHS = [0.25, 0.5, 0.75, 1.0]
DEPTH_LABELS = ["25%", "50%", "75%", "100%"]


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    })


def load_data():
    gap = pd.read_csv(RESULTS_DIR / "generality_gap.csv")
    with open(RESULTS_DIR / "selected_layers.json") as f:
        selected = json.load(f)
    matrix_8b = pd.read_csv(RESULTS_DIR / "stage2_matrix_8b.csv", index_col=0)
    matrix_70b = pd.read_csv(RESULTS_DIR / "stage2_matrix_70b.csv", index_col=0)
    return gap, selected, matrix_8b, matrix_70b


def fig1_depth_profile(gap: pd.DataFrame, selected: dict):
    """Within-relation and LOO AUC across depths for both models."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for model_key, color, label in [
        ("llama_3_1_8b", C_8B, "8B"),
        ("llama_3_1_70b", C_70B, "70B"),
    ]:
        m = gap[gap["model_key"] == model_key]

        within_means = []
        within_stds = []
        loo_means = []
        loo_stds = []
        for d in DEPTHS:
            depth_data = m[m["normalized_depth"] == d]
            within_means.append(depth_data["within_mean_auc"].mean())
            within_stds.append(depth_data["within_mean_auc"].std())
            loo_means.append(depth_data["loo_auc"].mean())
            loo_stds.append(depth_data["loo_auc"].std())

        x = np.arange(len(DEPTHS))

        ax.errorbar(x, within_means, yerr=within_stds, color=color,
                     marker="o", linestyle="-", label=f"{label} within",
                     capsize=3, capthick=1.2, markeredgecolor="white",
                     markeredgewidth=1)
        ax.errorbar(x, loo_means, yerr=loo_stds, color=color,
                     marker="s", linestyle="--", label=f"{label} leave-one-out",
                     capsize=3, capthick=1.2, markeredgecolor="white",
                     markeredgewidth=1)

    best_8b_idx = DEPTHS.index(selected["llama_3_1_8b"]["normalized_depth"])
    best_70b_idx = DEPTHS.index(selected["llama_3_1_70b"]["normalized_depth"])
    ax.axvline(best_8b_idx, color=C_8B, alpha=0.15, linewidth=8, zorder=0)
    ax.axvline(best_70b_idx, color=C_70B, alpha=0.15, linewidth=8, zorder=0)

    ax.set_xticks(range(len(DEPTHS)))
    ax.set_xticklabels(DEPTH_LABELS)
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.88, 1.005)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower left", framealpha=0.9, edgecolor="none")
    ax.set_title("Within-relation and leave-one-out AUC by depth")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_depth_profile.pdf")
    fig.savefig(FIGURES_DIR / "fig1_depth_profile.png")
    plt.close(fig)
    print("  fig1_depth_profile.pdf")


def fig2_generality_gap(gap: pd.DataFrame, selected: dict):
    """Mean absolute generality gap by depth for both models."""
    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    bar_width = 0.35
    x = np.arange(len(DEPTHS))

    for i, (model_key, color, label) in enumerate([
        ("llama_3_1_8b", C_8B, "8B"),
        ("llama_3_1_70b", C_70B, "70B"),
    ]):
        m = gap[gap["model_key"] == model_key]
        mean_abs_gaps = []
        for d in DEPTHS:
            depth_data = m[m["normalized_depth"] == d]
            mean_abs_gaps.append(depth_data["generality_gap"].abs().mean())

        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, mean_abs_gaps, bar_width * 0.88,
                       color=color, alpha=0.85, label=label,
                       edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, mean_abs_gaps):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                    color="#52514e")

    ax.set_xticks(x)
    ax.set_xticklabels(DEPTH_LABELS)
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Mean |generality gap|")
    ax.set_ylim(0, 0.06)
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.set_title("Generality gap: 8B vs 70B")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_generality_gap.pdf")
    fig.savefig(FIGURES_DIR / "fig2_generality_gap.png")
    plt.close(fig)
    print("  fig2_generality_gap.pdf")


def _draw_heatmap(ax, matrix: pd.DataFrame, title: str, vmin: float, vmax: float,
                  cmap, show_cbar: bool = False):
    """Draw a single transfer matrix heatmap."""
    ordered = matrix.loc[RELATION_ORDER, RELATION_ORDER].astype(float)
    data = ordered.values

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    n = len(RELATION_ORDER)
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            text_color = "white" if val > 0.97 else "#0b0b0b"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=text_color, fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_labels = [RELATION_SHORT[r] for r in RELATION_ORDER]
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Target relation", fontsize=9)
    ax.set_ylabel("Source relation", fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)

    ax.spines[:].set_visible(True)
    ax.spines[:].set_linewidth(0.5)
    ax.spines[:].set_color("#c3c2b7")
    ax.tick_params(length=0)

    return im


def fig3_transfer_matrices(matrix_8b: pd.DataFrame, matrix_70b: pd.DataFrame):
    """Side-by-side 6×6 transfer matrix heatmaps."""
    blues = LinearSegmentedColormap.from_list("custom_blues", [
        "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"
    ])

    fig = plt.figure(figsize=(9.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    _draw_heatmap(ax1, matrix_8b, "Llama-3.1-8B (layer 7, depth 25%)",
                  vmin=0.84, vmax=1.0, cmap=blues)
    im = _draw_heatmap(ax2, matrix_70b, "Llama-3.1-70B (layer 39, depth 50%)",
                       vmin=0.84, vmax=1.0, cmap=blues)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("ROC-AUC", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)

    fig.savefig(FIGURES_DIR / "fig3_transfer_matrices.pdf")
    fig.savefig(FIGURES_DIR / "fig3_transfer_matrices.png")
    plt.close(fig)
    print("  fig3_transfer_matrices.pdf")


def _fmt_gap(val: float) -> str:
    if abs(val) < 0.0005:
        return "0.000"
    return f"{val:+.3f}"


def fig4_relation_gap_detail(gap: pd.DataFrame, selected: dict):
    """Per-relation generality gap at the selected layer for each model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0))

    n = len(RELATION_ORDER)
    y_positions = np.arange(n)

    for ax, model_key, color, title, layer_depth in [
        (ax1, "llama_3_1_8b", C_8B, "8B (layer 7)", 0.25),
        (ax2, "llama_3_1_70b", C_70B, "70B (layer 39)", 0.5),
    ]:
        m = gap[(gap["model_key"] == model_key) &
                (gap["normalized_depth"] == layer_depth)]
        m = m.set_index("target_relation").loc[RELATION_ORDER]

        gaps = m["generality_gap"].values

        colors = [color if g >= 0 else "#e34948" for g in gaps]
        bars = ax.barh(y_positions, gaps, height=0.6, color=colors, alpha=0.8,
                        edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, gaps):
            x_pos = val + 0.002 if val >= 0 else val - 0.002
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    _fmt_gap(val), ha=ha, va="center", fontsize=7,
                    color="#52514e")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(RELATION_ORDER, fontsize=8)
        ax.axvline(0, color="#c3c2b7", linewidth=0.8, zorder=0)
        ax.set_xlabel("Generality gap (within − leave-one-out)", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.035, 0.065)
        ax.set_ylim(n - 0.5, -0.5)

    fig.suptitle("Per-relation generality gap at selected layer",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "fig4_relation_gap_detail.pdf")
    fig.savefig(FIGURES_DIR / "fig4_relation_gap_detail.png")
    plt.close(fig)
    print("  fig4_relation_gap_detail.pdf")


def fig5_relation_depth_profiles(gap: pd.DataFrame):
    """Small multiples: each relation's within AUC trajectory across depths."""
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.5), sharex=True, sharey=True)

    for idx, rel in enumerate(RELATION_ORDER):
        ax = axes[idx // 3, idx % 3]

        for model_key, color, label in [
            ("llama_3_1_8b", C_8B, "8B"),
            ("llama_3_1_70b", C_70B, "70B"),
        ]:
            m = gap[(gap["model_key"] == model_key) &
                    (gap["target_relation"] == rel)]
            m = m.sort_values("normalized_depth")
            x = np.arange(len(DEPTHS))
            ax.plot(x, m["within_mean_auc"].values, color=color,
                    marker="o", markersize=4, label=f"{label} within",
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.plot(x, m["loo_auc"].values, color=color,
                    marker="s", markersize=4, linestyle="--",
                    label=f"{label} leave-one-out",
                    markeredgecolor="white", markeredgewidth=0.8)

        ax.set_title(f"{rel}", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(DEPTHS)))
        ax.set_xticklabels(DEPTH_LABELS, fontsize=7)
        ax.set_ylim(0.80, 1.01)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        if idx == 0:
            ax.legend(fontsize=6, loc="lower left", framealpha=0.9,
                      edgecolor="none")

    fig.supxlabel("Normalized depth", fontsize=9)
    fig.supylabel("ROC-AUC", fontsize=9)
    fig.suptitle("Per-relation AUC profiles", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_relation_profiles.pdf")
    fig.savefig(FIGURES_DIR / "fig5_relation_profiles.png")
    plt.close(fig)
    print("  fig5_relation_profiles.pdf")


def table1_relation_results(gap: pd.DataFrame, selected: dict):
    """Relation-level table at selected layers."""
    rows = []
    for rel in RELATION_ORDER:
        row = {"Relation": rel}
        for model_key, short, depth in [
            ("llama_3_1_8b", "8B", 0.25),
            ("llama_3_1_70b", "70B", 0.5),
        ]:
            m = gap[(gap["model_key"] == model_key) &
                    (gap["normalized_depth"] == depth) &
                    (gap["target_relation"] == rel)]
            row[f"{short} within"] = f"{m['within_mean_auc'].values[0]:.3f}"
            row[f"{short} std"] = f"{m['within_std_auc'].values[0]:.3f}"
            row[f"{short} leave-one-out"] = f"{m['loo_auc'].values[0]:.3f}"
            row[f"{short} gap"] = _fmt_gap(m['generality_gap'].values[0])
        rows.append(row)

    mean_row = {"Relation": "Mean"}
    for model_key, short, depth in [
        ("llama_3_1_8b", "8B", 0.25),
        ("llama_3_1_70b", "70B", 0.5),
    ]:
        m = gap[(gap["model_key"] == model_key) &
                (gap["normalized_depth"] == depth)]
        mean_row[f"{short} within"] = f"{m['within_mean_auc'].mean():.3f}"
        mean_row[f"{short} std"] = f"{m['within_std_auc'].mean():.3f}"
        mean_row[f"{short} leave-one-out"] = f"{m['loo_auc'].mean():.3f}"
        mean_row[f"{short} gap"] = _fmt_gap(m['generality_gap'].mean())
    rows.append(mean_row)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "table1_relation_results.csv", index=False)
    print("  table1_relation_results.csv")
    print(table.to_string(index=False))


def main():
    setup_style()
    gap, selected, matrix_8b, matrix_70b = load_data()

    print("Generating figures...")
    fig1_depth_profile(gap, selected)
    fig2_generality_gap(gap, selected)
    fig3_transfer_matrices(matrix_8b, matrix_70b)
    fig4_relation_gap_detail(gap, selected)
    fig5_relation_depth_profiles(gap)
    table1_relation_results(gap, selected)
    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
