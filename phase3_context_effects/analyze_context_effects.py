"""Phase 3 analysis: NLL table, F1 (Figure 6a reproduction), F2 (6x6 grid).

Reads shard metadata written by extract_context_probs.py (works before
consolidation), joins prompt metadata, and produces:

    results/context_nll.parquet      per-prompt NLL/entropy/rank, all runs
    results/grid_matrix_{model}.csv  6x6 median paired delta-NLL
    results/grid_cells_{model}.csv   per-cell median, bootstrap CI, Wilcoxon p
    figures/f1_fig6a_{model}.{pdf,png}
    figures/f2_grid_heatmaps.{pdf,png}
    results/stage_a_summary_{model}.csv

Usage (blackboxnlp env):
    python analyze_context_effects.py            # everything available
    python analyze_context_effects.py --only f1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ACT_ROOT = ROOT / "context_activations"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PROMPTS = {
    "a": ROOT / "context_prompts" / "stage_a_p103_tuples.jsonl",
    "b": ROOT / "context_prompts" / "stage_b_grid_tuples.jsonl",
}
MODEL_DIRS = {"llama_3_1_8b": "8B", "llama_3_1_70b": "70B"}
RELATION_ORDER = ["P19", "P103", "P101", "P159", "P176", "P138"]

N_BOOT = 10_000
BOOT_SEED = 20260715


def load_prompt_meta(stage: str) -> pd.DataFrame:
    rows = []
    with PROMPTS[stage].open("r", encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            rows.append(
                {
                    "prompt_id": r["prompt_id"],
                    "stage": r["stage"],
                    "condition": r["condition"],
                    "context_len": r["context_len"],
                    "target_relation": r["target_relation"],
                    "context_relation": r["context_relation"],
                    "target_true_attribute": r["target_true_attribute"],
                    "primed": r.get("primed"),
                    "leaked_subject": r.get("leaked_subject"),
                    "pair_key": r.get("pair_key"),
                }
            )
    return pd.DataFrame(rows)


def collect() -> pd.DataFrame:
    """Gather per-prompt measurements from every run's shard metadata."""
    parts = []
    for model_dir, model_label in MODEL_DIRS.items():
        for stage in ("a", "b"):
            shard_dir = ACT_ROOT / model_dir / f"stage_{stage}" / "shards"
            if not shard_dir.exists():
                continue
            metas = []
            for mf in sorted(shard_dir.glob("shard_*.meta.json")):
                metas.extend(
                    json.loads(mf.read_text(encoding="utf-8"))["prompts"]
                )
            if not metas:
                continue
            df = pd.DataFrame(
                [
                    {
                        "prompt_id": m["prompt_id"],
                        "nll": m["nll"],
                        "p_target": m["p_target"],
                        "entropy": m["entropy"],
                        "target_rank": m["target_rank"],
                        "prompt_tokens": m["prompt_tokens"],
                    }
                    for m in metas
                ]
            )
            df = df.merge(load_prompt_meta(stage), on="prompt_id", how="left")
            df["model"] = model_label
            parts.append(df)
            print(f"{model_label} stage_{stage}: {len(df)} prompts collected")
    if not parts:
        raise RuntimeError("No shard metadata found under context_activations/")
    out = pd.concat(parts, ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    out.to_parquet(RESULTS / "context_nll.parquet", index=False)
    return out


# --------------------------------------------------------------------------
# F1 — Figure 6a reproduction
# --------------------------------------------------------------------------

F1_ORDER = ["FF", "TT", "FFF", "TTT", "FFFF", "TTTT"]


def _draw_f1_ax(ax, sub: pd.DataFrame, model_label: str) -> None:
    rng = np.random.default_rng(0)
    data = [sub[sub["condition"] == c]["nll"].to_numpy() for c in F1_ORDER]
    bp = ax.boxplot(
        data,
        positions=range(len(F1_ORDER)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 2, "color": "black"},
    )
    for box in bp["boxes"]:
        box.set_facecolor("#dddddd")
        box.set_alpha(0.6)
    palette = plt.cm.tab10.colors
    for x, ys in enumerate(data):
        c = palette[x % len(palette)]
        ax.scatter(
            x + 0.15 * (rng.random(len(ys)) - 0.5), ys,
            s=10, alpha=0.4, color=c, zorder=3,
        )
        ax.scatter(
            x, np.median(ys), marker="D", s=60, color=c,
            edgecolor="black", linewidth=0.5, zorder=4,
        )
    base = sub[sub["condition"] == "none"]["nll"]
    if not base.empty:
        ax.axhline(
            base.median(), color="gray", linestyle="--", linewidth=1.2,
            alpha=0.8, zorder=1,
        )
    ax.set_xticks(range(len(F1_ORDER)))
    ax.set_xticklabels(F1_ORDER, fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Context factuality", fontsize=16)
    ax.set_title(f"Llama-3.1-{model_label}", fontsize=16)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)


def figure_f1_combined(df: pd.DataFrame) -> None:
    subs = {
        m: df[(df["model"] == m) & (df["stage"] == "a")]
        for m in ("8B", "70B")
    }
    if any(s.empty for s in subs.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, (model_label, sub) in zip(axes, subs.items()):
        _draw_f1_ax(ax, sub, model_label)
    axes[0].set_ylabel(r"$-\log P(\mathrm{correct\ attribute})$", fontsize=16)
    fig.suptitle(
        "NLL of the correct attribute vs. context truthfulness "
        "(CounterFact P103; dashed line = no-context median)",
        fontsize=15,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"f1_fig6a_combined.{ext}", dpi=300)
    plt.close(fig)
    print("F1 combined written")


def figure_f1(df: pd.DataFrame, model_label: str) -> None:
    sub = df[(df["model"] == model_label) & (df["stage"] == "a")]
    if sub.empty:
        print(f"F1: no Stage A data for {model_label}, skipping")
        return
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [sub[sub["condition"] == c]["nll"].to_numpy() for c in F1_ORDER]

    bp = ax.boxplot(
        data,
        positions=range(len(F1_ORDER)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 2, "color": "black"},
    )
    for box in bp["boxes"]:
        box.set_facecolor("#dddddd")
        box.set_alpha(0.6)

    palette = plt.cm.tab10.colors
    for x, ys in enumerate(data):
        c = palette[x % len(palette)]
        ax.scatter(
            x + 0.15 * (rng.random(len(ys)) - 0.5), ys,
            s=10, alpha=0.4, color=c, zorder=3,
        )
        ax.scatter(
            x, np.median(ys), marker="D", s=60, color=c,
            edgecolor="black", linewidth=0.5, zorder=4,
        )

    base = sub[sub["condition"] == "none"]["nll"]
    if not base.empty:
        ax.axhline(
            base.median(), color="gray", linestyle="--", linewidth=1.2,
            alpha=0.8, zorder=1,
        )
        ax.text(
            len(F1_ORDER) - 0.45, base.median() + 0.25,
            "no-context median", fontsize=11, color="gray", ha="right",
        )

    ax.set_xticks(range(len(F1_ORDER)))
    ax.set_xticklabels(F1_ORDER, fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel("Context factuality", fontsize=18)
    ax.set_ylabel(r"$-\log P(\mathrm{correct\ attribute})$", fontsize=18)
    ax.set_title(
        f"Llama-3.1-{model_label}: NLL of the correct attribute vs. "
        "context truthfulness (CounterFact P103)",
        fontsize=15, pad=12,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"f1_fig6a_{model_label}.{ext}", dpi=300)
    plt.close(fig)

    rows = []
    for cond in F1_ORDER + ["none"]:
        ys = sub[sub["condition"] == cond]["nll"]
        rows.append(
            {
                "condition": cond,
                "n": len(ys),
                "median_nll": ys.median(),
                "mean_nll": ys.mean(),
                "q25": ys.quantile(0.25),
                "q75": ys.quantile(0.75),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        RESULTS / f"stage_a_summary_{model_label}.csv", index=False
    )
    gaps = {
        f"{f}-{t}": (
            summary.set_index("condition").loc[f, "median_nll"]
            - summary.set_index("condition").loc[t, "median_nll"]
        )
        for f, t in (("FF", "TT"), ("FFF", "TTT"), ("FFFF", "TTTT"))
    }
    print(f"F1 {model_label}: median NLL gaps (false - true): "
          + ", ".join(f"{k}={v:.2f}" for k, v in gaps.items())
          + "  (paper 8B FF-TT: 1.52)")


# --------------------------------------------------------------------------
# F2 — 6x6 grid
# --------------------------------------------------------------------------

def wilcoxon_p(diffs: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon

        if np.allclose(diffs, 0):
            return 1.0
        return float(wilcoxon(diffs, alternative="greater").pvalue)
    except ImportError:
        return float("nan")


def grid_cells(df: pd.DataFrame, model_label: str) -> pd.DataFrame | None:
    sub = df[
        (df["model"] == model_label)
        & (df["stage"] == "b")
        & (df["condition"].isin(["TT", "FF"]))
    ]
    if sub.empty:
        return None
    wide = sub.pivot_table(
        index=["pair_key", "context_relation", "target_relation"],
        columns="condition",
        values="nll",
    ).reset_index()
    wide = wide.dropna(subset=["TT", "FF"])
    wide["delta"] = wide["FF"] - wide["TT"]

    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for (ctx_rel, tgt_rel), grp in wide.groupby(
        ["context_relation", "target_relation"]
    ):
        d = grp["delta"].to_numpy()
        boots = np.median(
            rng.choice(d, size=(N_BOOT, len(d)), replace=True), axis=1
        )
        rows.append(
            {
                "model": model_label,
                "context_relation": ctx_rel,
                "target_relation": tgt_rel,
                "n_pairs": len(d),
                "median_delta_nll": float(np.median(d)),
                "ci_low": float(np.percentile(boots, 2.5)),
                "ci_high": float(np.percentile(boots, 97.5)),
                "wilcoxon_p_greater": wilcoxon_p(d),
                "mean_delta_nll": float(d.mean()),
            }
        )
    cells = pd.DataFrame(rows)
    cells.to_csv(RESULTS / f"grid_cells_{model_label}.csv", index=False)
    matrix = cells.pivot(
        index="context_relation", columns="target_relation",
        values="median_delta_nll",
    ).reindex(index=RELATION_ORDER, columns=RELATION_ORDER)
    matrix.to_csv(RESULTS / f"grid_matrix_{model_label}.csv")
    return matrix


def figure_f2(matrices: dict[str, pd.DataFrame]) -> None:
    if not matrices:
        print("F2: no Stage B data, skipping")
        return
    vmax = max(m.abs().max().max() for m in matrices.values())
    fig, axes = plt.subplots(
        1, len(matrices), figsize=(7.2 * len(matrices), 6), squeeze=False
    )
    for ax, (model_label, m) in zip(axes[0], sorted(matrices.items())):
        im = ax.imshow(
            m.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        ax.set_xticks(range(len(RELATION_ORDER)))
        ax.set_xticklabels(RELATION_ORDER, fontsize=12)
        ax.set_yticks(range(len(RELATION_ORDER)))
        ax.set_yticklabels(RELATION_ORDER, fontsize=12)
        ax.set_xlabel("Target relation", fontsize=14)
        ax.set_ylabel("Context relation", fontsize=14)
        ax.set_title(f"Llama-3.1-{model_label}", fontsize=15)
        for i in range(len(RELATION_ORDER)):
            for j in range(len(RELATION_ORDER)):
                v = m.iloc[i, j]
                if pd.notna(v):
                    ax.text(
                        j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=11,
                        color="white" if abs(v) > 0.6 * vmax else "black",
                    )
        fig.colorbar(
            im, ax=ax, shrink=0.8,
            label=r"median paired $\Delta$NLL (false $-$ true context)",
        )
    fig.suptitle(
        "Behavioral transfer of the context-truthfulness effect",
        fontsize=16,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"f2_grid_heatmaps.{ext}", dpi=300)
    plt.close(fig)
    print("F2 written:", ", ".join(sorted(matrices)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["collect", "f1", "f2"], default=None)
    args = parser.parse_args()

    df = collect()
    if args.only == "collect":
        return
    if args.only in (None, "f1"):
        for model_label in ("70B", "8B"):
            figure_f1(df, model_label)
        figure_f1_combined(df)
    if args.only in (None, "f2"):
        matrices = {}
        for model_label in ("70B", "8B"):
            m = grid_cells(df, model_label)
            if m is not None:
                matrices[model_label] = m
        figure_f2(matrices)


if __name__ == "__main__":
    main()
