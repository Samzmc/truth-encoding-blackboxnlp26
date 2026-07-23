"""
Compute linear-probe AUC over ALL checkpoints of toy runs.

For each checkpoint in <run_dir>/checkpoints/, runs the 5-fold linear probe
(eval_linear_separability from analyze_toy) and writes
<run_dir>/analysis/auc_curve_data.json in the same format as the
fully-trained runs (compatible with compare_fulltrained.py).

"auc_per_layer" holds position 2 (x'), matching the L1/L3 convention;
position 1 (y) is stored additionally under "auc_per_layer_pos1".

Usage:
    python toy_auc_curve.py --run_dirs phase0_original/toy_rho08_seed0 ...
"""
import argparse
import glob
import json
import os
import re

from analyze_toy import load_checkpoint, eval_linear_separability
from train_toy import EMBEDDING_DIM, INPUT_TOKENS


def process_run(run_dir):
    ckpt_files = sorted(
        glob.glob(os.path.join(run_dir, "checkpoints", "ckpt_step*.pt")),
        key=lambda f: int(re.search(r"step(\d+)", f).group(1)))
    if not ckpt_files:
        print(f"  No checkpoints in {run_dir}, skipping")
        return

    entries = []
    cfg = None
    for path in ckpt_files:
        model, f_map, cfg, step = load_checkpoint(path)
        auc_dict = eval_linear_separability(model, f_map)

        def fmt(tok):
            means, stds = auc_dict[tok]
            return {f"L{i}": {"mean": float(m), "std": float(s)}
                    for i, (m, s) in enumerate(zip(means, stds))}

        entries.append({
            "step": step,
            "auc_per_layer": fmt(2),
            "auc_per_layer_pos1": fmt(1),
        })
        pos2 = entries[-1]["auc_per_layer"]
        print(f"  step {step}: pos2 " +
              " ".join(f"{k}:{v['mean']:.3f}" for k, v in pos2.items()),
              flush=True)

    data = {
        "token_idx": 2,
        "num_layers": cfg["num_layers"],
        "seed": cfg.get("seed"),
        "N": len(INPUT_TOKENS),
        "d_model": EMBEDDING_DIM,
        "checkpoints": entries,
    }
    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "auc_curve_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Probe AUC over all checkpoints of toy runs")
    p.add_argument("--run_dirs", nargs="+", required=True,
                   help="Run dirs, each containing checkpoints/ and analysis/")
    args = p.parse_args()

    for run_dir in args.run_dirs:
        print(f"Processing {run_dir} ...")
        process_run(run_dir)
    print("Done.")


if __name__ == "__main__":
    main()
