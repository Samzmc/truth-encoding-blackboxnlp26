"""
Compare fully-trained model results: Normal vs Independent-Truth Ablation.

Reads auc_curve_data.json and prob_curve_data.json from multiple seeds,
produces mean±std comparison plots.

Usage:
    python compare_fulltrained.py \
        --normal_dirs phase0_original/L1_seed{0..4}/analysis \
        --ablated_dirs phase1_ablation/L1_seed{0..4}/analysis \
        --output_path comparison_figures_fulltrained/ \
        --label L1
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def load_jsons(dirs, filename):
    data = []
    for d in dirs:
        path = os.path.join(d, filename)
        with open(path) as f:
            data.append(json.load(f))
    return data


def extract_auc_curves(all_data, layer_key):
    """Extract (steps, values_per_seed) for a given layer from auc_curve_data."""
    all_steps = []
    all_vals = []
    for d in all_data:
        steps = [c["step"] for c in d["checkpoints"]]
        vals = [c["auc_per_layer"][layer_key]["mean"] for c in d["checkpoints"]]
        all_steps.append(steps)
        all_vals.append(vals)

    common = sorted(set(all_steps[0]).intersection(*[set(s) for s in all_steps]))
    aligned = []
    for steps, vals in zip(all_steps, all_vals):
        s2v = dict(zip(steps, vals))
        aligned.append([s2v[s] for s in common])

    return np.array(common), np.array(aligned)


def extract_prob_curves(all_data, key):
    """Extract (steps, values_per_seed) for 'true' or 'false' from prob_curve_data."""
    all_steps = []
    all_vals = []
    for d in all_data:
        steps = [c["step"] for c in d["checkpoints"]]
        vals = [c[key]["mean"] if isinstance(c[key], dict) else c[key]
                for c in d["checkpoints"]]
        all_steps.append(steps)
        all_vals.append(vals)

    common = sorted(set(all_steps[0]).intersection(*[set(s) for s in all_steps]))
    aligned = []
    for steps, vals in zip(all_steps, all_vals):
        s2v = dict(zip(steps, vals))
        aligned.append([s2v[s] for s in common])

    return np.array(common), np.array(aligned)


def plot_band(ax, steps, values, color, label, linestyle="-"):
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    ax.plot(steps, mean, linestyle, color=color, linewidth=1.5,
            label=f"{label} (n={len(values)})")
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)


def plot_auc_comparison(normal_dirs, ablated_dirs, output_path, label,
                        num_layers):
    """Probe AUC over training: normal vs ablated."""
    normal_data = load_jsons(normal_dirs, "auc_curve_data.json")
    ablated_data = load_jsons(ablated_dirs, "auc_curve_data.json")

    layer_key = f"L{num_layers}"

    steps_n, vals_n = extract_auc_curves(normal_data, layer_key)
    steps_a, vals_a = extract_auc_curves(ablated_data, layer_key)

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
                 f"(pos 2, final layer {layer_key})")
    fig.tight_layout()

    fname = os.path.join(output_path, f"auc_comparison_{label}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

    summary = {
        "label": label,
        "layer": layer_key,
        "normal_final_mean": float(vals_n[:, -1].mean()),
        "normal_final_std": float(vals_n[:, -1].std()),
        "ablated_final_mean": float(vals_a[:, -1].mean()),
        "ablated_final_std": float(vals_a[:, -1].std()),
    }
    fname_json = os.path.join(output_path, f"auc_comparison_{label}.json")
    with open(fname_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {fname_json}")


def plot_prob_comparison(normal_dirs, ablated_dirs, output_path, label):
    """P(correct|true) and P(correct|false) over training: normal vs ablated."""
    normal_data = load_jsons(normal_dirs, "prob_curve_data.json")
    ablated_data = load_jsons(ablated_dirs, "prob_curve_data.json")

    steps_nt, vals_nt = extract_prob_curves(normal_data, "true")
    steps_nf, vals_nf = extract_prob_curves(normal_data, "false")
    steps_at, vals_at = extract_prob_curves(ablated_data, "true")
    steps_af, vals_af = extract_prob_curves(ablated_data, "false")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ax = axes[0]
    plot_band(ax, steps_nt, vals_nt, "#1f77b4", "Normal")
    plot_band(ax, steps_at, vals_at, "#d62728", "Ablated", linestyle="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("P(f(x') | sequence)")
    ax.set_title(r"$P(\mathrm{correct} \mid \mathrm{true\ sequence})$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    plot_band(ax, steps_nf, vals_nf, "#1f77b4", "Normal")
    plot_band(ax, steps_af, vals_af, "#d62728", "Ablated", linestyle="--")
    ax.set_xlabel("Training step")
    ax.set_title(r"$P(\mathrm{correct} \mid \mathrm{false\ sequence})$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(f"Correct-token probability — {label}", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fname = os.path.join(output_path, f"prob_comparison_{label}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    p = argparse.ArgumentParser(
        description="Compare fully-trained model: Normal vs Ablated")
    p.add_argument("--normal_dirs", nargs="+", required=True,
                   help="Analysis dirs for normal condition seeds")
    p.add_argument("--ablated_dirs", nargs="+", required=True,
                   help="Analysis dirs for ablated condition seeds")
    p.add_argument("--output_path", default="comparison_figures_fulltrained")
    p.add_argument("--label", default="L1",
                   help="Label for the comparison (e.g. L1, L3)")
    p.add_argument("--num_layers", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    print(f"Plotting AUC comparison ({args.label})...")
    plot_auc_comparison(args.normal_dirs, args.ablated_dirs,
                        args.output_path, args.label, args.num_layers)

    print(f"Plotting probability comparison ({args.label})...")
    plot_prob_comparison(args.normal_dirs, args.ablated_dirs,
                         args.output_path, args.label)

    print("Done.")


if __name__ == "__main__":
    main()
