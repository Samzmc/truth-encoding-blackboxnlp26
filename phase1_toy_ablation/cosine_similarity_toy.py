"""
Per-block cosine similarity between W = W_o W_v sub-blocks and
theoretical targets (paper Eqs 6-7).

Standalone — only needs .npy matrices + metadata.json from
analyze_toy.py --plot_ov_curve.

Usage:
    # Single run
    python cosine_similarity_toy.py \
        --matrix_dir phase0_original/toy_seed0/analysis/ov_matrices/

    # Compare normal vs ablated (4-panel mean±std plot)
    python cosine_similarity_toy.py --compare \
        --normal_dirs phase0_original/toy_seed{0..4}/analysis/ov_matrices/ \
        --ablated_dirs phase1_ablation/toy_seed{0..4}/analysis/ov_matrices/ \
        --output_path comparison_figures/

Index layout (one-hot, d_model = 2V + L):
    0..N-1          entity embeddings   e_x
    N..2N-1         attribute embeddings e_y
    2N..2N+L-1      positional encodings
    2N+L..3N+L-1    entity unembeddings  u_x
    3N+L..4N+L-1    attribute unembeddings u_y

After g-reordering (attribute dims sorted by g), the four target
sub-blocks all reduce to +/-I:

    Block              Location (reordered)          Target
    memory             W[u_y(g), e_x]                +I
    neg. identity      W[e_x, e_x]                   -I
    reverse lookup     W[e_x, e_y(g)]                +I
    self-suppression   W[u_y(g), e_y(g)]             -I
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt


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
    "self_suppression": r"Self-suppression ($e_y \to -u_y$)",
}


def load_metadata(matrix_dir):
    with open(os.path.join(matrix_dir, "metadata.json")) as f:
        meta = json.load(f)
    meta["f_map"] = {int(k): v for k, v in meta["f_map"].items()}
    return meta


def reorder_indices(meta):
    N, V, L = meta["N"], meta["V"], meta["L"]
    f_map = meta["f_map"]

    ex = np.arange(N)
    ey = np.array([f_map[x] for x in range(N)])
    pos = np.arange(V, V + L)
    ux = V + L + ex
    uy = V + L + ey

    order = np.concatenate([ex, ey, pos, ux, uy])

    ranges = {
        "ex":  (0, N),
        "ey":  (N, 2 * N),
        "pos": (2 * N, 2 * N + L),
        "ux":  (2 * N + L, 3 * N + L),
        "uy":  (3 * N + L, 4 * N + L),
    }
    return order, ranges


def extract_subblock(W_reordered, ranges, row_key, col_key):
    r0, r1 = ranges[row_key]
    c0, c1 = ranges[col_key]
    return W_reordered[r0:r1, c0:c1]


def cosine_sim_with_identity(block, sign):
    """Cosine similarity between block and sign * I."""
    N = block.shape[0]
    target = sign * np.eye(N)
    b_flat = block.flatten()
    t_flat = target.flatten()
    denom = np.linalg.norm(b_flat) * np.linalg.norm(t_flat)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(b_flat, t_flat) / denom)


def compute_all(matrix_dir, meta):
    order, ranges = reorder_indices(meta)

    npy_files = glob.glob(os.path.join(matrix_dir, "W_step*.npy"))
    npy_files.sort(key=lambda f: int(re.search(r"step(\d+)", f).group(1)))

    results = []
    for path in npy_files:
        step = int(re.search(r"step(\d+)", path).group(1))
        W = np.load(path)
        W_re = W[np.ix_(order, order)]

        row = {"step": step}
        for name, spec in BLOCKS.items():
            block = extract_subblock(W_re, ranges, spec["row"], spec["col"])
            row[name] = cosine_sim_with_identity(block, spec["sign"])
        results.append(row)

    return results


def save_json(results, meta, output_path):
    data = {
        "metric": "cosine_similarity_vs_signed_identity",
        "seed": meta["seed"],
        "N": meta["N"],
        "blocks": list(BLOCKS.keys()),
        "checkpoints": results,
    }
    fname = os.path.join(output_path, "cosine_similarity.json")
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {fname}")


def plot_curves(results, output_path):
    steps = [r["step"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in BLOCKS:
        vals = [r[name] for r in results]
        ax.plot(steps, vals, "-o", markersize=3, label=BLOCK_LABELS[name])

    ax.set_xlabel("Training step")
    ax.set_ylabel("Cosine similarity with target")
    ax.set_ylim(-0.3, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("OV sub-block cosine similarity over training")
    fig.tight_layout()
    fname = os.path.join(output_path, "cosine_similarity.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ───────────────── Multi-seed comparison ───────────────── #

def load_multi_seed(dirs):
    """Load cosine similarity results from multiple seed directories."""
    all_results = []
    for d in dirs:
        meta = load_metadata(d)
        results = compute_all(d, meta)
        all_results.append(results)
    return all_results


def align_steps(all_results):
    """Find common steps across all seeds, return (steps, block_arrays).

    block_arrays: dict {block_name: ndarray shape (n_seeds, n_steps)}
    """
    step_sets = [set(r["step"] for r in res) for res in all_results]
    common = sorted(step_sets[0].intersection(*step_sets[1:]))

    block_arrays = {name: [] for name in BLOCKS}
    for res in all_results:
        step_to_row = {r["step"]: r for r in res}
        for name in BLOCKS:
            block_arrays[name].append([step_to_row[s][name] for s in common])

    block_arrays = {k: np.array(v) for k, v in block_arrays.items()}
    return common, block_arrays


def plot_comparison(normal_dirs, ablated_dirs, output_path):
    """4-panel figure: one per block, normal vs ablated with mean±std."""
    print("Loading normal condition...")
    normal_all = load_multi_seed(normal_dirs)
    print("Loading ablated condition...")
    ablated_all = load_multi_seed(ablated_dirs)

    steps_n, blocks_n = align_steps(normal_all)
    steps_a, blocks_a = align_steps(ablated_all)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = {"normal": "#1f77b4", "ablated": "#d62728"}

    for idx, name in enumerate(BLOCKS):
        ax = axes[idx]

        mean_n = blocks_n[name].mean(axis=0)
        std_n = blocks_n[name].std(axis=0)
        mean_a = blocks_a[name].mean(axis=0)
        std_a = blocks_a[name].std(axis=0)

        ax.plot(steps_n, mean_n, "-", color=colors["normal"],
                label=f"Normal (n={len(normal_all)})", linewidth=1.5)
        ax.fill_between(steps_n, mean_n - std_n, mean_n + std_n,
                        color=colors["normal"], alpha=0.2)

        ax.plot(steps_a, mean_a, "--", color=colors["ablated"],
                label=f"Ablated (n={len(ablated_all)})", linewidth=1.5)
        ax.fill_between(steps_a, mean_a - std_a, mean_a + std_a,
                        color=colors["ablated"], alpha=0.2)

        ax.set_title(BLOCK_LABELS[name], fontsize=11)
        ax.set_ylim(-0.15, 1.05)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9)

    axes[2].set_xlabel("Training step")
    axes[3].set_xlabel("Training step")
    axes[0].set_ylabel("Cosine similarity with target")
    axes[2].set_ylabel("Cosine similarity with target")

    fig.suptitle("OV sub-block convergence: Normal vs Independent-Truth Ablation",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_path, exist_ok=True)
    fname = os.path.join(output_path, "cosine_similarity_comparison.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

    fname_json = os.path.join(output_path, "cosine_similarity_comparison.json")
    summary = {}
    for name in BLOCKS:
        summary[name] = {
            "normal_final_mean": float(blocks_n[name][:, -1].mean()),
            "normal_final_std":  float(blocks_n[name][:, -1].std()),
            "ablated_final_mean": float(blocks_a[name][:, -1].mean()),
            "ablated_final_std":  float(blocks_a[name][:, -1].std()),
        }
    with open(fname_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {fname_json}")


# ───────────────── CLI ───────────────── #

def main():
    p = argparse.ArgumentParser(
        description="Cosine similarity of OV sub-blocks vs theoretical targets")

    sub = p.add_subparsers(dest="mode")

    # Single-run mode (default when no subcommand)
    p.add_argument("--matrix_dir",
                   help="Directory with W_step*.npy + metadata.json")
    p.add_argument("--output_path", default=None,
                   help="Output directory (default: parent of matrix_dir)")

    # Comparison mode
    cmp = sub.add_parser("compare",
                         help="Compare normal vs ablated (multi-seed)")
    cmp.add_argument("--normal_dirs", nargs="+", required=True)
    cmp.add_argument("--ablated_dirs", nargs="+", required=True)
    cmp.add_argument("--output_path", default="comparison_figures")

    args = p.parse_args()

    if args.mode == "compare":
        plot_comparison(args.normal_dirs, args.ablated_dirs, args.output_path)
    elif args.matrix_dir:
        meta = load_metadata(args.matrix_dir)
        out = args.output_path or os.path.join(args.matrix_dir, "..")
        os.makedirs(out, exist_ok=True)

        print(f"Computing cosine similarity (N={meta['N']}, seed={meta['seed']})...")
        results = compute_all(args.matrix_dir, meta)
        save_json(results, meta, out)
        plot_curves(results, out)

        final = results[-1]
        print(f"\n  Final step {final['step']}:")
        for name in BLOCKS:
            print(f"    {name:20s} = {final[name]:.4f}")
        print("Done.")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
