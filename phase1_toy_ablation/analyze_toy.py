"""
Post-training analysis for toy model checkpoints.

Usage:
    python analyze_toy.py --checkpoint outputs/checkpoints/ckpt_step1000.pt \
        --plot_ov --plot_probe --plot_pca --plot_attention

    python analyze_toy.py --checkpoint_dir outputs/checkpoints/ --plot_ov_curve
"""
import argparse
import glob
import math
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_toy import (
    CausalTransformer, gen_batch, device,
    INPUT_TOKENS, OUTPUT_TOKENS,
    VOCAB_SIZE, SEQ_LENGTH, EMBEDDING_DIM,
    NUM_HEADS, NUM_LAYERS, BATCH_SIZE,
    USE_FIRST_LAYER_MLP,
)


# ───────────────────── Checkpoint loading ───────────────────── #

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt['config']
    model = CausalTransformer(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        num_heads=cfg['num_heads'],
        num_layers=cfg['num_layers'],
        first_layer_mlp=cfg.get('first_layer_mlp', False),
        no_norm=cfg.get('no_norm', False),
    )
    model.load_state_dict(ckpt['state'])
    model.to(device).eval()
    return model, ckpt['f_map'], cfg, ckpt['step']


# ───────────────────── Activation collection ───────────────────── #

def collect_activations(model, f_map, token_indices, num_samples=2000,
                        batch_size=BATCH_SIZE, seed=12345):
    rng = np.random.default_rng(seed)
    acts = {tok: None for tok in token_indices}
    labels = []
    collected = 0
    while collected < num_samples:
        cur_bs = min(batch_size, num_samples - collected)
        x, is_true = gen_batch(rng, cur_bs, f_map, is_train=False)
        inp = torch.from_numpy(x[:, :-1]).to(device)
        with torch.no_grad():
            _, hids = model(inp, return_activations=True)
        for tok in token_indices:
            per_layer = [h[:, tok, :].cpu().numpy() for h in hids]
            if acts[tok] is None:
                acts[tok] = per_layer
            else:
                acts[tok] = [np.concatenate([o, n], axis=0)
                             for o, n in zip(acts[tok], per_layer)]
        labels.append(is_true)
        collected += cur_bs
    return acts, np.concatenate(labels)


# ───────────────────── 1. OV circuit heatmap (Figure 1) ───────────────────── #

def extract_ov_matrix(model, layer_idx=0):
    attn = model.transformer_blocks[layer_idx].attention
    Wo = attn.out_proj.weight.detach().cpu().numpy()
    Wv = attn.v_proj.weight.detach().cpu().numpy()
    return Wo @ Wv


def plot_ov_heatmap(model, layer_idx=0, output_path=".", step=None):
    """Raw W = W_o @ W_v (d_model × d_model), all dimensions including positional."""
    attn = model.transformer_blocks[layer_idx].attention
    Wo = attn.out_proj.weight.detach().cpu().numpy()
    Wv = attn.v_proj.weight.detach().cpu().numpy()
    W = Wo @ Wv

    N = len(INPUT_TOKENS)
    V = VOCAB_SIZE
    L = SEQ_LENGTH

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(W, interpolation='nearest')
    title = r"$W = W_o W_v$"
    if step is not None:
        title += f"  (step {step})"
    ax.set_title(title, fontsize=16)
    fig.colorbar(im, ax=ax, shrink=0.8)

    boundaries = [0, N, V, V + L, V + L + N, V + L + V]
    labels = [r"$e_x$", r"$e_y$", "pos", r"$u_x$", r"$u_y$"]
    mid = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(5)]
    ax.set_xticks(mid)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(mid)
    ax.set_yticklabels(labels, fontsize=12)

    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='white', linewidth=0.8, alpha=0.7)
        ax.axvline(b - 0.5, color='white', linewidth=0.8, alpha=0.7)

    fig.tight_layout()
    fname = os.path.join(output_path, f"ov_heatmap_step{step}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_ov_heatmap_subblocks(model, f_map, layer_idx=0, output_path=".", step=None):
    """Raw W reordered as [eₓ, eᵧ(g-ordered), pos, uₓ, uᵧ(g-ordered)] for clean diagonals."""
    attn = model.transformer_blocks[layer_idx].attention
    Wo = attn.out_proj.weight.detach().cpu().numpy()
    Wv = attn.v_proj.weight.detach().cpu().numpy()
    W = Wo @ Wv

    N = len(INPUT_TOKENS)
    V = VOCAB_SIZE
    L = SEQ_LENGTH

    x_dims = np.arange(N)
    y_dims = np.array([f_map[x] for x in range(N)])
    pos_dims = np.arange(V, V + L)
    ux_dims = V + L + x_dims
    uy_dims = V + L + y_dims

    order = np.concatenate([x_dims, y_dims, pos_dims, ux_dims, uy_dims])
    W_sub = W[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(W_sub, interpolation='nearest')
    title = r"$W = W_o W_v$ (sub-block ordered)"
    if step is not None:
        title += f"  (step {step})"
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)

    boundaries = [0, N, 2 * N, 2 * N + L, 3 * N + L, 4 * N + L]
    labels = [r"$e_x$", r"$e_y$", "pos", r"$u_x$", r"$u_y$"]
    mid = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(5)]
    ax.set_xticks(mid)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(mid)
    ax.set_yticklabels(labels, fontsize=12)

    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='white', linewidth=0.8, alpha=0.7)
        ax.axvline(b - 0.5, color='white', linewidth=0.8, alpha=0.7)

    fig.tight_layout()
    fname = os.path.join(output_path, f"ov_subblocks_step{step}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ───────────────────── 2. Linear probe AUC ───────────────────── #

def eval_linear_separability(model, f_map, token_indices=(1, 2),
                              num_eval_samples=2000, n_splits=5, seed=12345):
    acts, labels = collect_activations(model, f_map, token_indices,
                                       num_samples=num_eval_samples, seed=seed)
    auc_dict = {}
    for tok, vecs_per_layer in acts.items():
        means, stds = [], []
        for X in vecs_per_layer:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            fold_aucs = []
            for tr, dv in skf.split(X, labels):
                clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-4)
                clf.fit(X[tr], labels[tr])
                prob = clf.predict_proba(X[dv])[:, 1]
                fold_aucs.append(roc_auc_score(labels[dv], prob))
            means.append(np.mean(fold_aucs))
            stds.append(np.std(fold_aucs))
        auc_dict[tok] = (means, stds)
    return auc_dict


def plot_auc(auc_dict, output_path=".", step=None):
    layers = range(len(next(iter(auc_dict.values()))[0]))
    fig, ax = plt.subplots()
    for tok, (means, stds) in auc_dict.items():
        label = "y (pos 1)" if tok == 1 else "x' (pos 2)"
        ax.errorbar(layers, means, yerr=stds, fmt='-o', capsize=3, label=label)
    ax.set_xlabel("Layer (0 = embedding)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.set_xticks(list(layers))
    ax.legend()
    title = "Linear separability"
    if step is not None:
        title += f" (step {step})"
    ax.set_title(title)
    fig.tight_layout()
    fname = os.path.join(output_path, f"probe_auc_step{step}.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ───────────────────── 3. PCA visualisation ───────────────────── #

def plot_pca(model, f_map, token_indices=(1, 2), num_samples=2000,
             output_path=".", step=None, max_points=2000, seed=2024):
    acts, labels = collect_activations(model, f_map, token_indices,
                                       num_samples=num_samples, seed=seed)
    rng = np.random.default_rng(seed)

    n_tok = len(token_indices)
    n_layers = len(next(iter(acts.values())))
    n_cols = min(4, n_layers)
    n_rows = n_tok * math.ceil(n_layers / n_cols)

    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    fig.suptitle("PCA coloured by truthfulness")

    for r, tok in enumerate(token_indices):
        tok_str = "y" if tok == 1 else "x'"
        pcs_list = [PCA(n_components=2).fit_transform(X) for X in acts[tok]]
        for i, pcs in enumerate(pcs_list):
            row = r * math.ceil(n_layers / n_cols) + (i // n_cols)
            col = i % n_cols
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            idx = rng.choice(pcs.shape[0], min(max_points, pcs.shape[0]), replace=False)
            ax.scatter(pcs[idx, 0], pcs[idx, 1],
                       c=labels[idx], s=6, alpha=0.7, cmap="coolwarm")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"tok {tok_str} · L{i}", fontsize=8)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fname = os.path.join(output_path, f"pca_step{step}.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ───────────────────── 4. Attention maps ───────────────────── #

def plot_attention(model, f_map, output_path=".", step=None, seed=2024):
    model.eval()
    rng = np.random.default_rng(seed)
    x, _ = gen_batch(rng, 1, f_map, is_train=False)
    inp = torch.from_numpy(x[:, :-1]).to(device)
    with torch.no_grad():
        model(inp)

    n_blocks = len(model.transformer_blocks)
    fig, axes = plt.subplots(1, n_blocks, figsize=(3 * n_blocks, 3), squeeze=False)

    for i, block in enumerate(model.transformer_blocks):
        if block.attention is None:
            continue
        ax = axes[0, i]
        attn = block.attention.attn_weights[0, 0].cpu().numpy()
        ax.imshow(attn)
        ticks = ['x', 'y', "x'"]
        ax.set_xticks(range(SEQ_LENGTH))
        ax.set_xticklabels(ticks)
        ax.set_yticks(range(SEQ_LENGTH))
        ax.set_yticklabels(ticks)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        ax.set_title(f"Layer {i}")

    fig.tight_layout()
    fname = os.path.join(output_path, f"attention_step{step}.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ───────────────────── 5. Evaluate mapping accuracy ───────────────────── #

def evaluate_mapping(model, f_map):
    model.eval()
    correct = 0
    for x_tok in sorted(f_map.keys()):
        inp = torch.tensor([[x_tok]], dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(inp)[0, 0].argmax().item()
        if pred == f_map[x_tok]:
            correct += 1
    acc = correct / len(f_map)
    print(f"  Mapping accuracy: {correct}/{len(f_map)} = {acc:.2%}")
    return acc


# ───────────────────── 6. OV over all checkpoints ───────────────────── #

def plot_ov_all_checkpoints(checkpoint_dir, output_path="."):
    """For every checkpoint: save W matrix as .npy, plot OV heatmap + subblock."""
    raw = glob.glob(os.path.join(checkpoint_dir, "ckpt_step*.pt"))
    ckpt_files = sorted(raw, key=lambda f: int(re.search(r'step(\d+)', f).group(1)))
    if not ckpt_files:
        print(f"  No checkpoints found in {checkpoint_dir}")
        return

    mat_dir = os.path.join(output_path, "ov_matrices")
    heatmap_dir = os.path.join(output_path, "ov_heatmaps")
    subblock_dir = os.path.join(output_path, "ov_subblocks")
    os.makedirs(mat_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(subblock_dir, exist_ok=True)

    metadata_saved = False
    for path in ckpt_files:
        model, f_map, cfg, step = load_checkpoint(path)
        W = extract_ov_matrix(model)
        np.save(os.path.join(mat_dir, f"W_step{step}.npy"), W)
        if not metadata_saved:
            import json as _json
            meta = {
                "N": len(INPUT_TOKENS), "M": len(OUTPUT_TOKENS),
                "V": VOCAB_SIZE, "L": SEQ_LENGTH, "d_model": EMBEDDING_DIM,
                "f_map": {str(k): v for k, v in f_map.items()},
                "seed": cfg.get("seed"),
            }
            with open(os.path.join(mat_dir, "metadata.json"), "w") as mf:
                _json.dump(meta, mf, indent=2)
            metadata_saved = True
        plot_ov_heatmap(model, output_path=heatmap_dir, step=step)
        plot_ov_heatmap_subblocks(model, f_map, output_path=subblock_dir, step=step)
        print(f"  step {step}: W saved + plots done")


# ───────────────────── CLI ───────────────────── #

def main():
    p = argparse.ArgumentParser(description="Analyze toy model checkpoints")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a single checkpoint")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Directory of checkpoints (for OV curve over training)")
    p.add_argument("--output_path", type=str, default=None,
                   help="Directory for plots (default: <checkpoint>/../analysis)")
    p.add_argument("--plot_ov", action="store_true", help="OV circuit heatmap (Figure 1)")
    p.add_argument("--plot_ov_sub", action="store_true", help="OV heatmap with sub-block ordering")
    p.add_argument("--plot_ov_curve", action="store_true",
                   help="OV heatmaps + matrices for ALL checkpoints (requires --checkpoint_dir)")
    p.add_argument("--plot_probe", action="store_true", help="Linear probe AUC")
    p.add_argument("--plot_pca", action="store_true", help="PCA visualisation")
    p.add_argument("--plot_attention", action="store_true", help="Attention heatmaps")
    p.add_argument("--eval_mapping", action="store_true", help="Evaluate mapping f accuracy")
    p.add_argument("--all", action="store_true", help="Run all analyses")
    args = p.parse_args()

    if args.all:
        args.plot_ov = args.plot_ov_sub = args.plot_probe = True
        args.plot_pca = args.plot_attention = args.eval_mapping = True
        args.plot_ov_curve = True

    if args.plot_ov_curve:
        ckpt_dir = args.checkpoint_dir
        if ckpt_dir is None and args.checkpoint is not None:
            ckpt_dir = os.path.dirname(args.checkpoint)
        if ckpt_dir is None:
            print("Error: --checkpoint_dir or --checkpoint required for --plot_ov_curve")
            return
        out = args.output_path or os.path.join(ckpt_dir, "..", "analysis")
        os.makedirs(out, exist_ok=True)
        print("OV analysis over all checkpoints...")
        plot_ov_all_checkpoints(ckpt_dir, output_path=out)

    if args.checkpoint is None:
        if not args.plot_ov_curve:
            print("Error: --checkpoint required for single-checkpoint analyses")
        return

    model, f_map, cfg, step = load_checkpoint(args.checkpoint)
    out = args.output_path or os.path.join(os.path.dirname(args.checkpoint), "..", "analysis")
    os.makedirs(out, exist_ok=True)
    print(f"Checkpoint: step {step}, seed {cfg.get('seed', '?')}")

    if args.plot_ov:
        print("OV heatmap...")
        plot_ov_heatmap(model, output_path=out, step=step)

    if args.plot_ov_sub:
        print("OV sub-block heatmap...")
        plot_ov_heatmap_subblocks(model, f_map, output_path=out, step=step)

    if args.plot_probe:
        print("Linear probing...")
        auc_dict = eval_linear_separability(model, f_map)
        plot_auc(auc_dict, output_path=out, step=step)
        for tok, (m, s) in auc_dict.items():
            formatted = "  ".join(f"L{i}:{mu:.3f}+/-{sd:.3f}" for i, (mu, sd) in enumerate(zip(m, s)))
            label = "y" if tok == 1 else "x'"
            print(f"  token {label}: {formatted}")

    if args.plot_pca:
        print("PCA...")
        plot_pca(model, f_map, output_path=out, step=step)

    if args.plot_attention:
        print("Attention maps...")
        plot_attention(model, f_map, output_path=out, step=step)

    if args.eval_mapping:
        print("Mapping evaluation...")
        evaluate_mapping(model, f_map)

    print("Done.")


if __name__ == "__main__":
    main()
