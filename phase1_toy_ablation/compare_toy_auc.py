"""
Compare toy-model probe AUC over training: Normal vs Independent-Truth Ablation.

Reads auc_curve_data.json (produced by toy_auc_curve.py) from multiple seeds
and produces mean±std comparison plots for probe position 2 (x', main figure)
and position 1 (y, supplementary figure).

Usage:
    python compare_toy_auc.py \
        --normal_dirs phase0_original/toy_rho08_seed{0..4}/analysis \
        --ablated_dirs phase1_ablation/toy_rho08_seed{0..4}/analysis \
        --output_path comparison_figures_toy_rho08/ \
        --label toy_rho08
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from compare_fulltrained import load_jsons, plot_band


def extract_curves(all_data, layer_key, field="auc_per_layer"):
    """Extract (steps, values_per_seed) for a given layer and probe position."""
    all_steps = []
    all_vals = []
    for d in all_data:
        steps = [c["step"] for c in d["checkpoints"]]
        vals = [c[field][layer_key]["mean"] for c in d["checkpoints"]]
        all_steps.append(steps)
        all_vals.append(vals)

    common = sorted(set(all_steps[0]).intersection(*[set(s) for s in all_steps]))
    aligned = []
    for steps, vals in zip(all_steps, all_vals):
        s2v = dict(zip(steps, vals))
        aligned.append([s2v[s] for s in common])

    return np.array(common), np.array(aligned)


def plot_position(normal_data, ablated_data, layer_key, field, pos_label,
                  output_path, label, suffix=""):
    steps_n, vals_n = extract_curves(normal_data, layer_key, field)
    steps_a, vals_a = extract_curves(ablated_data, layer_key, field)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_band(ax, steps_n, vals_n, "#1f77b4", "Normal")
    plot_band(ax, steps_a, vals_a, "#d62728", "Ablated", linestyle="--")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Probe AUC")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title(f"Probe AUC over training — {label} "
                 f"({pos_label}, layer {layer_key})")
    fig.tight_layout()

    fname = os.path.join(output_path, f"auc_comparison_{label}{suffix}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

    return {
        "position": pos_label,
        "normal_final_mean": float(vals_n[:, -1].mean()),
        "normal_final_std": float(vals_n[:, -1].std()),
        "ablated_final_mean": float(vals_a[:, -1].mean()),
        "ablated_final_std": float(vals_a[:, -1].std()),
    }


def main():
    p = argparse.ArgumentParser(
        description="Compare toy-model probe AUC: Normal vs Ablated")
    p.add_argument("--normal_dirs", nargs="+", required=True)
    p.add_argument("--ablated_dirs", nargs="+", required=True)
    p.add_argument("--output_path", default="comparison_figures_toy_rho08")
    p.add_argument("--label", default="toy_rho08")
    p.add_argument("--num_layers", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    layer_key = f"L{args.num_layers}"

    normal_data = load_jsons(args.normal_dirs, "auc_curve_data.json")
    ablated_data = load_jsons(args.ablated_dirs, "auc_curve_data.json")

    print("Plotting AUC comparison (pos 2, x')...")
    summary_pos2 = plot_position(normal_data, ablated_data, layer_key,
                                 "auc_per_layer", "pos 2 x'",
                                 args.output_path, args.label)

    print("Plotting AUC comparison (pos 1, y)...")
    summary_pos1 = plot_position(normal_data, ablated_data, layer_key,
                                 "auc_per_layer_pos1", "pos 1 y",
                                 args.output_path, args.label, suffix="_pos1")

    summary = {
        "label": args.label,
        "layer": layer_key,
        "n_seeds_normal": len(normal_data),
        "n_seeds_ablated": len(ablated_data),
        "pos2": summary_pos2,
        "pos1": summary_pos1,
    }
    fname_json = os.path.join(args.output_path,
                              f"auc_comparison_{args.label}.json")
    with open(fname_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {fname_json}")
    print("Done.")


if __name__ == "__main__":
    main()
