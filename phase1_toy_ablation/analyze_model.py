"""
Post-training analysis for fully-trained model checkpoints.

Usage:
    python analyze_model.py --checkpoint outputs/checkpoints/ckpt_step50000.pt --all
    python analyze_model.py --checkpoint_dir outputs/checkpoints/ --plot_auc_curve
"""

import argparse
import json
import math
import os
import glob
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_model import (
    Config, CausalTransformer, gen_batch, iterate_batches,
)


# ───────────────────── Checkpoint loading ───────────────────── #

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt['cfg']
    cfg_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    known = set(Config.__dataclass_fields__)
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in known})
    cfg.finalise()

    model = CausalTransformer(cfg)
    model.load_state_dict(ckpt['state'])
    model.to(cfg.device).eval()
    return model, ckpt['f_map'], cfg, ckpt['step']


# ───────────────────── Activation collection ───────────────────── #

def collect_activations(model, cfg, f_map, token_indices,
                        num_samples=2000, seed=12345):
    rng = np.random.default_rng(seed)
    acts = {tok: None for tok in token_indices}
    labels = []
    collected = 0
    while collected < num_samples:
        cur = min(cfg.batch_size, num_samples - collected)
        full, is_true = gen_batch(rng, cur, cfg, f_map, is_train=False)
        inp = torch.as_tensor(full[:, :-1], device=cfg.device)
        with torch.no_grad():
            _, hids = model(inp, return_activations=True)
        for tok in token_indices:
            per = [lay[:, tok, :].cpu().numpy() for lay in hids]
            if acts[tok] is None:
                acts[tok] = per
            else:
                acts[tok] = [np.concatenate([o, n], axis=0)
                             for o, n in zip(acts[tok], per)]
        labels.append(is_true)
        collected += cur
    return acts, np.concatenate(labels)


# ───────────────────── 1. OV circuit heatmap ───────────────────── #

def plot_ov_heatmap(model, cfg, layer_idx=0, output_path=".", step=None):
    blk = model.blocks[layer_idx]
    if blk.attn is None:
        print(f"  Layer {layer_idx} has no attention, skipping OV heatmap.")
        return

    Wo = blk.attn.o.weight.detach().cpu().numpy()
    Wv = blk.attn.v.weight.detach().cpu().numpy()

    E = torch.vstack([
        model.tok.weight.data,
        model.out.weight.data,
    ]).detach().cpu().numpy()

    W = Wo @ Wv
    K = E @ W @ E.T

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(K, interpolation='nearest')
    title = r"$E\,W_o\,W_v\,E^\top$"
    if step is not None:
        title += f"  (step {step})"
    ax.set_title(title, fontsize=16)
    fig.colorbar(im, ax=ax, shrink=0.8)

    V = model.tok.weight.shape[0]
    for b in [V]:
        ax.axhline(b - 0.5, color='white', linewidth=0.8, alpha=0.7)
        ax.axvline(b - 0.5, color='white', linewidth=0.8, alpha=0.7)
    ax.set_xticks([V // 2, V + V // 2])
    ax.set_xticklabels(["tok embed", "out embed"], fontsize=10)
    ax.set_yticks([V // 2, V + V // 2])
    ax.set_yticklabels(["tok embed", "out embed"], fontsize=10)

    fig.tight_layout()
    fname = os.path.join(output_path, f"ov_heatmap_step{step}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ───────────────────── 2. Linear probe AUC ───────────────────── #

def eval_linear_probe_auc(model, cfg, f_map, num_samples=2000, seed=12345):
    token_indices = (1, 2, 3) if cfg.use_bos else (0, 1, 2)
    acts, labels = collect_activations(model, cfg, f_map, token_indices,
                                       num_samples=num_samples, seed=seed)
    auc_dict = {}
    for tok, vecs_per_layer in acts.items():
        means, stds = [], []
        for X in vecs_per_layer:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
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
        label = f"pos {tok}"
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

def plot_pca(model, cfg, f_map, num_samples=2000, output_path=".",
             step=None, seed=2024):
    token_indices = (2, 3) if cfg.use_bos else (1, 2)
    acts, labels = collect_activations(model, cfg, f_map, token_indices,
                                       num_samples=num_samples, seed=seed)
    rng = np.random.default_rng(seed)

    n_tok = len(token_indices)
    n_layers = len(next(iter(acts.values())))
    n_cols = min(4, n_layers)
    n_rows = n_tok * math.ceil(n_layers / n_cols)

    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    title = "PCA coloured by truthfulness"
    if step is not None:
        title += f" (step {step})"
    fig.suptitle(title)

    for r, tok in enumerate(token_indices):
        tok_str = f"pos {tok}"
        pcs_list = [PCA(n_components=2).fit_transform(X) for X in acts[tok]]
        for i, pcs in enumerate(pcs_list):
            row = r * math.ceil(n_layers / n_cols) + (i // n_cols)
            col = i % n_cols
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            n_show = min(2000, pcs.shape[0])
            idx = rng.choice(pcs.shape[0], n_show, replace=False)
            ax.scatter(pcs[idx, 0], pcs[idx, 1],
                       c=labels[idx], s=6, alpha=0.7, cmap="coolwarm")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{tok_str} · L{i}", fontsize=8)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fname = os.path.join(output_path, f"pca_step{step}.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ───────────────────── 4. Attention maps ───────────────────── #

def plot_attention(model, cfg, f_map, output_path=".", step=None, seed=2024):
    model.eval()
    rng = np.random.default_rng(seed)
    full, _ = gen_batch(rng, 1, cfg, f_map, is_train=False)
    inp = torch.as_tensor(full[:, :-1], device=cfg.device)
    with torch.no_grad():
        model(inp)

    attn_blocks = [(i, blk) for i, blk in enumerate(model.blocks)
                   if blk.attn is not None]
    n = len(attn_blocks)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
    lbl = ['bos', 'x', 'y', "x'"] if cfg.use_bos else ['x', 'y', "x'"]

    for j, (i, blk) in enumerate(attn_blocks):
        ax = axes[0, j]
        attn = blk.attn.attn_weights[0, 0].cpu().numpy()
        ax.imshow(attn)
        ax.set_xticks(range(cfg.seq_len))
        ax.set_xticklabels(lbl)
        ax.set_yticks(range(cfg.seq_len))
        ax.set_yticklabels(lbl)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        ax.set_title(f"Layer {i}")

    fig.tight_layout()
    fname = os.path.join(output_path, f"attention_step{step}.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ───────────────────── 5. Mapping accuracy ───────────────────── #

def evaluate_mapping(model, cfg, f_map):
    model.eval()
    bos = 1 if cfg.use_bos else 0
    correct = 0
    for x_tok in sorted(f_map.keys()):
        if cfg.use_bos:
            inp = torch.tensor([[0, x_tok]], dtype=torch.long, device=cfg.device)
        else:
            inp = torch.tensor([[x_tok]], dtype=torch.long, device=cfg.device)
        with torch.no_grad():
            pred = model(inp)[0, bos].argmax().item()
        if pred == f_map[x_tok]:
            correct += 1
    acc = correct / len(f_map)
    print(f"  Mapping accuracy: {correct}/{len(f_map)} = {acc:.2%}")
    return acc


# ───────────────────── 6. AUC curve over checkpoints ───────────────────── #

def plot_auc_curve(checkpoint_dir, output_path=".", token_idx=2, seed=12345):
    """Plot AUC at a single token position over all checkpoints in a directory."""
    raw = glob.glob(os.path.join(checkpoint_dir, "ckpt_step*.pt"))
    ckpt_files = sorted(raw, key=lambda f: int(re.search(r'step(\d+)', f).group(1)))
    if not ckpt_files:
        print(f"  No checkpoints found in {checkpoint_dir}")
        return

    steps, aucs_mean, aucs_std = [], [], []
    all_layer_data = []
    for f in ckpt_files:
        model, f_map, cfg, step = load_checkpoint(f)
        auc_dict = eval_linear_probe_auc(model, cfg, f_map, num_samples=1000, seed=seed)
        tok = token_idx
        if tok not in auc_dict:
            tok = list(auc_dict.keys())[0]
        means, stds = auc_dict[tok]
        steps.append(step)
        aucs_mean.append(means[-1])
        aucs_std.append(stds[-1])
        all_layer_data.append({
            "step": step,
            "auc_per_layer": {f"L{i}": {"mean": m, "std": s}
                              for i, (m, s) in enumerate(zip(means, stds))},
        })
        print(f"  step {step}: AUC={means[-1]:.4f}")

    data_out = {
        "token_idx": tok,
        "num_layers": cfg.num_layers,
        "seed": cfg.seed,
        "N": cfg.N,
        "d_model": cfg.d_model,
        "checkpoints": all_layer_data,
    }
    json_fname = os.path.join(output_path, "auc_curve_data.json")
    with open(json_fname, "w") as jf:
        json.dump(data_out, jf, indent=2)
    print(f"  Saved {json_fname}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(steps, aucs_mean, yerr=aucs_std, fmt='-o', capsize=3, markersize=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Probe AUC (final layer)")
    ax.set_ylim(0.4, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title(f"AUC over training (pos {token_idx})")
    fig.tight_layout()
    fname = os.path.join(output_path, "auc_curve.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ───────────────── 7. True-token probability ───────────────── #

@torch.no_grad()
def eval_true_token_prob(model, cfg, f_map, num_samples=1000, seed=12345):
    """Compute P(f(x') | x, y, x') separately for true and false sequences."""
    model.eval()
    rng = np.random.default_rng(seed)
    bos = 1 if cfg.use_bos else 0
    x2_pos = bos + 2

    probs_true, probs_false = [], []

    while len(probs_true) < num_samples or len(probs_false) < num_samples:
        full, labs = gen_batch(rng, cfg.batch_size, cfg, f_map, is_train=False)
        inp = torch.as_tensor(full[:, :-1], device=cfg.device)
        logits = model(inp)
        sp = torch.softmax(logits, dim=-1)

        for i in range(full.shape[0]):
            x2_tok = int(full[i, x2_pos])
            p = sp[i, x2_pos, f_map[x2_tok]].item()
            if labs[i] == 1 and len(probs_true) < num_samples:
                probs_true.append(p)
            elif labs[i] == 0 and len(probs_false) < num_samples:
                probs_false.append(p)

    return {
        'true':  {'mean': float(np.mean(probs_true)),  'std': float(np.std(probs_true))},
        'false': {'mean': float(np.mean(probs_false)), 'std': float(np.std(probs_false))},
    }


def plot_prob_curve(checkpoint_dir, output_path=".", num_samples=1000, seed=12345):
    """Plot P(f(x') | x, y, x') for true/false sequences over training."""
    raw = glob.glob(os.path.join(checkpoint_dir, "ckpt_step*.pt"))
    ckpt_files = sorted(raw, key=lambda f: int(re.search(r'step(\d+)', f).group(1)))
    if not ckpt_files:
        print(f"  No checkpoints found in {checkpoint_dir}")
        return

    steps = []
    true_means, true_stds = [], []
    false_means, false_stds = [], []
    all_data = []

    for path in ckpt_files:
        model, f_map, cfg, step = load_checkpoint(path)
        result = eval_true_token_prob(model, cfg, f_map, num_samples, seed)
        steps.append(step)
        true_means.append(result['true']['mean'])
        true_stds.append(result['true']['std'])
        false_means.append(result['false']['mean'])
        false_stds.append(result['false']['std'])
        all_data.append({'step': step, 'true': result['true'], 'false': result['false']})
        print(f"  step {step}: P(correct|true)={result['true']['mean']:.4f}  "
              f"P(correct|false)={result['false']['mean']:.4f}")

    json_data = {
        'metric': "P(f(x') | x, y, x')",
        'num_samples': num_samples,
        'seed': cfg.seed,
        'N': cfg.N,
        'd_model': cfg.d_model,
        'num_layers': cfg.num_layers,
        'checkpoints': all_data,
    }
    json_fname = os.path.join(output_path, "prob_curve_data.json")
    with open(json_fname, "w") as jf:
        json.dump(json_data, jf, indent=2)
    print(f"  Saved {json_fname}")

    steps_arr = np.array(steps)
    tm, ts = np.array(true_means), np.array(true_stds)
    fm, fs = np.array(false_means), np.array(false_stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps_arr, tm, '-o', markersize=4, label='True sequences')
    ax.fill_between(steps_arr, tm - ts, tm + ts, alpha=0.2)
    ax.plot(steps_arr, fm, '-s', markersize=4, label='False sequences')
    ax.fill_between(steps_arr, fm - fs, fm + fs, alpha=0.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("P(f(x') | x, y, x')")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Correct-token probability: true vs false sequences")
    fig.tight_layout()
    fname = os.path.join(output_path, "prob_curve.png")
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ───────────────────── CLI ───────────────────── #

def main():
    p = argparse.ArgumentParser(description="Analyze fully-trained model checkpoints")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a single checkpoint")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Directory of checkpoints (for AUC curve)")
    p.add_argument("--output_path", type=str, default=None,
                   help="Directory for plots (default: <checkpoint>/../analysis)")
    p.add_argument("--plot_ov", action="store_true", help="OV circuit heatmap")
    p.add_argument("--plot_probe", action="store_true", help="Linear probe AUC")
    p.add_argument("--plot_pca", action="store_true", help="PCA visualisation")
    p.add_argument("--plot_attention", action="store_true", help="Attention heatmaps")
    p.add_argument("--eval_mapping", action="store_true", help="Evaluate mapping accuracy")
    p.add_argument("--plot_auc_curve", action="store_true",
                   help="AUC over training steps (requires --checkpoint_dir)")
    p.add_argument("--eval_prob", action="store_true",
                   help="P(correct token) on true/false sequences (single checkpoint)")
    p.add_argument("--plot_prob_curve", action="store_true",
                   help="P(correct token) curve over training (requires --checkpoint_dir)")
    p.add_argument("--all", action="store_true", help="Run all single-checkpoint analyses")
    args = p.parse_args()

    if args.all:
        args.plot_ov = args.plot_probe = args.plot_pca = True
        args.plot_attention = args.eval_mapping = args.plot_auc_curve = True
        args.eval_prob = args.plot_prob_curve = True

    need_ckpt_dir = args.plot_auc_curve or args.plot_prob_curve
    if need_ckpt_dir:
        ckpt_dir = args.checkpoint_dir
        if ckpt_dir is None and args.checkpoint is not None:
            ckpt_dir = os.path.dirname(args.checkpoint)
        if ckpt_dir is None:
            print("Error: --checkpoint_dir or --checkpoint required for curve plots")
            return
        out = args.output_path or os.path.join(ckpt_dir, "..", "analysis")
        os.makedirs(out, exist_ok=True)
        if args.plot_auc_curve:
            print("AUC curve over checkpoints...")
            plot_auc_curve(ckpt_dir, output_path=out)
        if args.plot_prob_curve:
            print("Correct-token probability curve over checkpoints...")
            plot_prob_curve(ckpt_dir, output_path=out)

    if args.checkpoint is None:
        if not need_ckpt_dir:
            print("Error: --checkpoint required for single-checkpoint analyses")
        return

    model, f_map, cfg, step = load_checkpoint(args.checkpoint)
    out = args.output_path or os.path.join(os.path.dirname(args.checkpoint),
                                            "..", "analysis")
    os.makedirs(out, exist_ok=True)
    print(f"Checkpoint: step {step}, seed {cfg.seed}, N={cfg.N}, d={cfg.d_model}")

    if args.plot_ov:
        print("OV heatmap...")
        for i in range(cfg.num_layers):
            plot_ov_heatmap(model, cfg, layer_idx=i, output_path=out, step=step)

    if args.plot_probe:
        print("Linear probing...")
        auc_dict = eval_linear_probe_auc(model, cfg, f_map)
        plot_auc(auc_dict, output_path=out, step=step)
        probe_data = {"step": step, "seed": cfg.seed,
                      "N": cfg.N, "d_model": cfg.d_model,
                      "num_layers": cfg.num_layers}
        for tok, (m, s) in auc_dict.items():
            probe_data[f"pos{tok}"] = {f"L{i}": {"mean": mu, "std": sd}
                                       for i, (mu, sd) in enumerate(zip(m, s))}
            formatted = "  ".join(f"L{i}:{mu:.3f}+/-{sd:.3f}"
                                  for i, (mu, sd) in enumerate(zip(m, s)))
            print(f"  pos {tok}: {formatted}")
        json_fname = os.path.join(out, f"probe_data_step{step}.json")
        with open(json_fname, "w") as jf:
            json.dump(probe_data, jf, indent=2)
        print(f"  Saved {json_fname}")

    if args.plot_pca:
        print("PCA...")
        plot_pca(model, cfg, f_map, output_path=out, step=step)

    if args.plot_attention:
        print("Attention maps...")
        plot_attention(model, cfg, f_map, output_path=out, step=step)

    if args.eval_mapping:
        print("Mapping evaluation...")
        evaluate_mapping(model, cfg, f_map)

    if args.eval_prob:
        print("True-token probability...")
        result = eval_true_token_prob(model, cfg, f_map)
        print(f"  P(correct | true seq):  {result['true']['mean']:.4f} +/- {result['true']['std']:.4f}")
        print(f"  P(correct | false seq): {result['false']['mean']:.4f} +/- {result['false']['std']:.4f}")
        prob_data = {
            'step': step, 'seed': cfg.seed,
            'N': cfg.N, 'd_model': cfg.d_model, 'num_layers': cfg.num_layers,
            'true': result['true'], 'false': result['false'],
        }
        json_fname = os.path.join(out, f"prob_data_step{step}.json")
        with open(json_fname, "w") as jf:
            json.dump(prob_data, jf, indent=2)
        print(f"  Saved {json_fname}")

    print("Done.")


if __name__ == "__main__":
    main()
