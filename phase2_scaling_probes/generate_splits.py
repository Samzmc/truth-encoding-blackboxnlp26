"""Generate 3-fold pair-grouped cross-validation splits for N100.

True/false examples from the same pair are always in the same fold.
Downstream code joins on example_id, never on row index.

Run from workspace/reproduction/scaling/ under the blackboxnlp conda env:
    python generate_splits.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260712_03
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "examples.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "splits.parquet"

N_FOLDS = 3


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df = df[df["subset"] == "N100"]
    print(f"Loaded {len(df)} N100 examples across "
          f"{df['relation_id'].nunique()} relations")

    rng = np.random.RandomState(SEED)

    fold_assignments: list[dict] = []

    for rel in sorted(df["relation_id"].unique()):
        rel_df = df[df["relation_id"] == rel]
        pairs = sorted(rel_df[rel_df["label"] == 1]["case_id"].unique())
        n_pairs = len(pairs)

        shuffled = rng.permutation(pairs)
        folds = np.array_split(shuffled, N_FOLDS)
        fold_sizes = [len(f) for f in folds]

        pair_to_fold = {}
        for fold_idx, fold_pairs in enumerate(folds):
            for case_id in fold_pairs:
                pair_to_fold[case_id] = fold_idx

        for _, row in rel_df.iterrows():
            fold_assignments.append({
                "example_id": row["example_id"],
                "pair_id": row["pair_id"],
                "case_id": row["case_id"],
                "relation_id": row["relation_id"],
                "within_relation_fold": pair_to_fold[row["case_id"]],
            })

        print(f"  {rel}: {n_pairs} pairs -> folds {fold_sizes}")

    splits = pd.DataFrame(fold_assignments)

    # --- Validation ---
    errors = []

    for rel in splits["relation_id"].unique():
        rel_splits = splits[splits["relation_id"] == rel]

        for case_id in rel_splits["case_id"].unique():
            pair_rows = rel_splits[rel_splits["case_id"] == case_id]
            if pair_rows["within_relation_fold"].nunique() != 1:
                errors.append(f"{rel} case_id={case_id}: pair split across folds")

        fold_counts = rel_splits.groupby("within_relation_fold").size()
        if len(fold_counts) != N_FOLDS:
            errors.append(f"{rel}: expected {N_FOLDS} folds, got {len(fold_counts)}")

    joined = splits.merge(
        df[["example_id", "label"]], on="example_id", how="left"
    )
    for rel in splits["relation_id"].unique():
        rel_j = joined[joined["relation_id"] == rel]
        for fold in range(N_FOLDS):
            fold_j = rel_j[rel_j["within_relation_fold"] == fold]
            n_true = (fold_j["label"] == 1).sum()
            n_false = (fold_j["label"] == 0).sum()
            if n_true != n_false:
                errors.append(f"{rel} fold {fold}: {n_true} true vs {n_false} false")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise RuntimeError("Split validation failed")

    print(f"\nValidation passed:")
    print(f"  All pairs have true/false in same fold")
    print(f"  All relations have {N_FOLDS} folds")
    print(f"  Class balance OK within every fold")

    splits.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"\nSaved {len(splits)} rows to {OUTPUT_PATH}")
    print(f"Split seed: {SEED}")


if __name__ == "__main__":
    main()
