"""Phase 2 probing pipeline: Stage 1A, 1B, layer selection, Stage 2.

Run from this directory under the blackboxnlp conda env:
    python run_probing.py
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "experiment_config.json"
DATA_DIR = ROOT / "data" / "processed"
ACT_DIR = ROOT / "activations"
RESULTS_DIR = ROOT / "results"
WEIGHTS_DIR = RESULTS_DIR / "probe_weights"

MODELS = {
    "llama_3_1_8b": {"model_id": "meta-llama/Llama-3.1-8B"},
    "llama_3_1_70b": {"model_id": "meta-llama/Llama-3.1-70B"},
}

RELATION_ORDER = ["P19", "P103", "P101", "P159", "P176", "P138"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg["probe"], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_data():
    examples = pd.read_parquet(DATA_DIR / "examples.parquet")
    examples = examples[examples["subset"] == "N100"]
    splits = pd.read_parquet(DATA_DIR / "splits.parquet")
    merged = examples.merge(splits[["example_id", "within_relation_fold"]],
                            on="example_id", how="inner")
    return merged


def load_activations(model_key: str) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    model_dir = ACT_DIR / model_key
    with open(model_dir / "manifest.json") as f:
        manifest = json.load(f)
    sample_index = pd.read_parquet(model_dir / "sample_index.parquet")

    acts = {}
    for layer_info in manifest["target_layers"]:
        li = layer_info["layer_index"]
        arr = np.load(model_dir / f"layer_{li}.npy").astype(np.float32)
        acts[li] = arr

    return acts, sample_index


def make_probe(cfg: dict) -> LogisticRegression:
    p = cfg["probe"]
    return LogisticRegression(
        solver=p["solver"],
        penalty=p["penalty"],
        C=p["C"],
        max_iter=p["max_iter"],
        tol=p["tol"],
        fit_intercept=p["fit_intercept"],
        class_weight=p["class_weight"],
        random_state=p["random_state"],
    )


def fit_and_predict(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    cfg: dict,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray, float, float, int]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    probe = make_probe(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe.fit(X_train_s, y_train)

    for w in caught:
        if issubclass(w.category, sklearn.exceptions.ConvergenceWarning):
            raise RuntimeError(
                f"Probe did not converge within max_iter={cfg['probe']['max_iter']}. "
                f"Training samples={len(y_train)}, features={X_train.shape[1]}"
            )

    scores = probe.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, scores)
    preds = (scores >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(y_test, preds)
    n_iter = int(probe.n_iter_[0])

    return probe, scaler, scores, auc, bal_acc, n_iter


def save_probe(
    probe: LogisticRegression, scaler: StandardScaler,
    probe_id: str, meta: dict,
) -> None:
    np.savez(
        WEIGHTS_DIR / f"{probe_id}.npz",
        coefficient=probe.coef_,
        intercept=probe.intercept_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=probe.classes_,
    )
    manifest_path = WEIGHTS_DIR / f"{probe_id}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(meta, f, indent=2)


def build_activation_index(sample_index: pd.DataFrame, data: pd.DataFrame):
    eid_to_row = {eid: i for i, eid in enumerate(sample_index["example_id"])}
    data = data.copy()
    data["act_row"] = data["example_id"].map(eid_to_row)
    assert data["act_row"].notna().all(), "example_id mismatch between data and activations"
    data["act_row"] = data["act_row"].astype(int)
    return data


def run_stage1a(data: pd.DataFrame, model_key: str, acts: dict[int, np.ndarray],
                manifest: dict, cfg: dict, cfg_h: str):
    relations = RELATION_ORDER
    metrics_rows = []
    pred_rows = []
    probe_count = 0

    for layer_info in manifest["target_layers"]:
        li = layer_info["layer_index"]
        depth = layer_info["depth"]
        X_all = acts[li]

        for rel in relations:
            rel_data = data[data["relation_id"] == rel]

            for fold in range(3):
                train_mask = rel_data["within_relation_fold"] != fold
                test_mask = rel_data["within_relation_fold"] == fold

                train_rows = rel_data[train_mask]["act_row"].values
                test_rows = rel_data[test_mask]["act_row"].values

                X_train = X_all[train_rows]
                y_train = rel_data[train_mask]["label"].values
                X_test = X_all[test_rows]
                y_test = rel_data[test_mask]["label"].values

                probe, scaler, scores, auc, bal_acc, n_iter = fit_and_predict(
                    X_train, y_train, X_test, y_test, cfg
                )

                probe_id = f"s1a_{model_key}_L{li}_{rel}_f{fold}"
                train_eids = rel_data[train_mask]["example_id"].tolist()

                save_probe(probe, scaler, probe_id, {
                    "probe_id": probe_id,
                    "stage": "1A",
                    "protocol": "within_relation",
                    "model_id": MODELS[model_key]["model_id"],
                    "model_revision": "served via NDIF",
                    "layer_index": li,
                    "normalized_depth": depth,
                    "source_relations": [rel],
                    "target_relation": rel,
                    "fold": fold,
                    "training_example_ids": train_eids,
                    "probe_configuration_hash": cfg_h,
                    "sklearn_version": sklearn.__version__,
                    "n_iter": n_iter,
                })

                metrics_rows.append({
                    "stage": "1A",
                    "model_id": MODELS[model_key]["model_id"],
                    "model_key": model_key,
                    "layer_index": li,
                    "normalized_depth": depth,
                    "protocol": "within_relation",
                    "source_relations": rel,
                    "target_relation": rel,
                    "fold": fold,
                    "auc": auc,
                    "balanced_accuracy": bal_acc,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "n_iter": n_iter,
                })

                test_eids = rel_data[test_mask]["example_id"].values
                for eid, true_label, score in zip(test_eids, y_test, scores):
                    pred_rows.append({
                        "stage": "1A",
                        "model_id": MODELS[model_key]["model_id"],
                        "model_key": model_key,
                        "layer_index": li,
                        "normalized_depth": depth,
                        "protocol": "within_relation",
                        "source_relations": rel,
                        "target_relation": rel,
                        "fold": fold,
                        "example_id": eid,
                        "true_label": true_label,
                        "prediction_score": float(score),
                    })

                probe_count += 1

    print(f"  Stage 1A: {probe_count} probes fitted for {model_key}")
    return metrics_rows, pred_rows


def run_stage1b(data: pd.DataFrame, model_key: str, acts: dict[int, np.ndarray],
                manifest: dict, cfg: dict, cfg_h: str):
    relations = RELATION_ORDER
    metrics_rows = []
    pred_rows = []
    probe_count = 0

    for layer_info in manifest["target_layers"]:
        li = layer_info["layer_index"]
        depth = layer_info["depth"]
        X_all = acts[li]

        for target_rel in relations:
            train_data = data[data["relation_id"] != target_rel]
            test_data = data[data["relation_id"] == target_rel]

            train_rows = train_data["act_row"].values
            test_rows = test_data["act_row"].values

            X_train = X_all[train_rows]
            y_train = train_data["label"].values
            X_test = X_all[test_rows]
            y_test = test_data["label"].values

            source_rels = [r for r in relations if r != target_rel]

            probe, scaler, scores, auc, bal_acc, n_iter = fit_and_predict(
                X_train, y_train, X_test, y_test, cfg
            )

            probe_id = f"s1b_{model_key}_L{li}_target_{target_rel}"
            train_eids = train_data["example_id"].tolist()

            save_probe(probe, scaler, probe_id, {
                "probe_id": probe_id,
                "stage": "1B",
                "protocol": "leave_one_out",
                "model_id": MODELS[model_key]["model_id"],
                "model_revision": "served via NDIF",
                "layer_index": li,
                "normalized_depth": depth,
                "source_relations": source_rels,
                "target_relation": target_rel,
                "fold": None,
                "training_example_ids": train_eids,
                "probe_configuration_hash": cfg_h,
                "sklearn_version": sklearn.__version__,
                "n_iter": n_iter,
            })

            metrics_rows.append({
                "stage": "1B",
                "model_id": MODELS[model_key]["model_id"],
                "model_key": model_key,
                "layer_index": li,
                "normalized_depth": depth,
                "protocol": "leave_one_out",
                "source_relations": ";".join(source_rels),
                "target_relation": target_rel,
                "fold": None,
                "auc": auc,
                "balanced_accuracy": bal_acc,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_iter": n_iter,
            })

            for eid, true_label, score in zip(test_data["example_id"].values,
                                               y_test, scores):
                pred_rows.append({
                    "stage": "1B",
                    "model_id": MODELS[model_key]["model_id"],
                    "model_key": model_key,
                    "layer_index": li,
                    "normalized_depth": depth,
                    "protocol": "leave_one_out",
                    "source_relations": ";".join(source_rels),
                    "target_relation": target_rel,
                    "fold": None,
                    "example_id": eid,
                    "true_label": true_label,
                    "prediction_score": float(score),
                })

            probe_count += 1

    print(f"  Stage 1B: {probe_count} probes fitted for {model_key}")
    return metrics_rows, pred_rows


def select_layers(metrics_1a: pd.DataFrame, cfg: dict) -> dict:
    tol = cfg["layer_selection_tie_tolerance"]
    selected = {}

    for model_key in metrics_1a["model_key"].unique():
        m = metrics_1a[metrics_1a["model_key"] == model_key]
        mean_by_depth = (
            m.groupby(["layer_index", "normalized_depth"])["auc"]
            .mean()
            .reset_index()
        )
        best_auc = mean_by_depth["auc"].max()
        candidates = mean_by_depth[mean_by_depth["auc"] >= best_auc - tol]
        chosen = candidates.loc[candidates["normalized_depth"].idxmin()]

        selected[model_key] = {
            "layer_index": int(chosen["layer_index"]),
            "normalized_depth": float(chosen["normalized_depth"]),
            "mean_auc": float(chosen["auc"]),
            "best_auc": float(best_auc),
            "all_depths": mean_by_depth.to_dict(orient="records"),
        }
        print(f"  {model_key}: selected layer {int(chosen['layer_index'])} "
              f"(depth={chosen['normalized_depth']}, "
              f"mean_auc={chosen['auc']:.4f}, best={best_auc:.4f})")

    return selected


def fit_probe_only(
    X_train: np.ndarray, y_train: np.ndarray, cfg: dict,
) -> tuple[LogisticRegression, StandardScaler, int]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    probe = make_probe(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe.fit(X_train_s, y_train)

    for w in caught:
        if issubclass(w.category, sklearn.exceptions.ConvergenceWarning):
            raise RuntimeError(
                f"Probe did not converge within max_iter={cfg['probe']['max_iter']}. "
                f"Training samples={len(y_train)}, features={X_train.shape[1]}"
            )

    return probe, scaler, int(probe.n_iter_[0])


def run_stage2(all_indexed: dict[str, pd.DataFrame], selected: dict,
               acts_cache: dict, manifests: dict,
               metrics_1a: pd.DataFrame, cfg: dict, cfg_h: str):
    relations = RELATION_ORDER
    metrics_rows = []
    pred_rows = []
    matrices = {}
    probe_count = 0

    for model_key, layer_info in selected.items():
        li = layer_info["layer_index"]
        depth = layer_info["normalized_depth"]
        X_all = acts_cache[model_key][li]
        data = all_indexed[model_key]

        matrix = pd.DataFrame(index=relations, columns=relations, dtype=float)

        s1a_at_layer = metrics_1a[
            (metrics_1a["model_key"] == model_key) &
            (metrics_1a["layer_index"] == li)
        ]
        for rel in relations:
            rel_aucs = s1a_at_layer[s1a_at_layer["target_relation"] == rel]["auc"]
            matrix.loc[rel, rel] = rel_aucs.mean()

        for source_rel in relations:
            source_data = data[data["relation_id"] == source_rel]
            X_train = X_all[source_data["act_row"].values]
            y_train = source_data["label"].values

            probe, scaler, n_iter = fit_probe_only(X_train, y_train, cfg)

            probe_id = f"s2_{model_key}_L{li}_src_{source_rel}"
            train_eids = source_data["example_id"].tolist()

            save_probe(probe, scaler, probe_id, {
                "probe_id": probe_id,
                "stage": "2",
                "protocol": "transfer",
                "model_id": MODELS[model_key]["model_id"],
                "model_revision": "served via NDIF",
                "layer_index": li,
                "normalized_depth": depth,
                "source_relations": [source_rel],
                "target_relation": "all",
                "fold": None,
                "training_example_ids": train_eids,
                "probe_configuration_hash": cfg_h,
                "sklearn_version": sklearn.__version__,
                "n_iter": n_iter,
            })
            probe_count += 1

            for target_rel in relations:
                if target_rel == source_rel:
                    continue

                target_data = data[data["relation_id"] == target_rel]
                X_test = X_all[target_data["act_row"].values]
                y_test = target_data["label"].values

                X_test_s = scaler.transform(X_test)
                scores = probe.predict_proba(X_test_s)[:, 1]
                auc = roc_auc_score(y_test, scores)
                preds = (scores >= 0.5).astype(int)
                bal_acc = balanced_accuracy_score(y_test, preds)

                matrix.loc[source_rel, target_rel] = auc

                metrics_rows.append({
                    "stage": "2",
                    "model_id": MODELS[model_key]["model_id"],
                    "model_key": model_key,
                    "layer_index": li,
                    "normalized_depth": depth,
                    "protocol": "transfer",
                    "source_relations": source_rel,
                    "target_relation": target_rel,
                    "fold": None,
                    "auc": auc,
                    "balanced_accuracy": bal_acc,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "n_iter": n_iter,
                })

                for eid, true_label, score in zip(target_data["example_id"].values,
                                                   y_test, scores):
                    pred_rows.append({
                        "stage": "2",
                        "model_id": MODELS[model_key]["model_id"],
                        "model_key": model_key,
                        "layer_index": li,
                        "normalized_depth": depth,
                        "protocol": "transfer",
                        "source_relations": source_rel,
                        "target_relation": target_rel,
                        "fold": None,
                        "example_id": eid,
                        "true_label": true_label,
                        "prediction_score": float(score),
                    })

        matrices[model_key] = matrix
        print(f"  Stage 2: {probe_count} source probes fitted for {model_key}")

    return metrics_rows, pred_rows, matrices


def main() -> None:
    cfg = load_config()
    cfg_h = config_hash(cfg)
    print(f"Config hash: {cfg_h}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    data = load_data()
    print(f"  {len(data)} N100 examples, {data['relation_id'].nunique()} relations")

    all_metrics_1a = []
    all_preds_1a = []
    all_metrics_1b = []
    all_preds_1b = []
    acts_cache = {}
    manifests = {}

    for model_key in MODELS:
        print(f"\n{'='*50}")
        print(f"Model: {model_key}")
        print(f"{'='*50}")

        acts, sample_index = load_activations(model_key)
        model_dir = ACT_DIR / model_key
        with open(model_dir / "manifest.json") as f:
            manifest = json.load(f)

        indexed_data = build_activation_index(sample_index, data)
        acts_cache[model_key] = acts
        manifests[model_key] = manifest

        m1a, p1a = run_stage1a(indexed_data, model_key, acts, manifest, cfg, cfg_h)
        all_metrics_1a.extend(m1a)
        all_preds_1a.extend(p1a)

        m1b, p1b = run_stage1b(indexed_data, model_key, acts, manifest, cfg, cfg_h)
        all_metrics_1b.extend(m1b)
        all_preds_1b.extend(p1b)

    metrics_1a_df = pd.DataFrame(all_metrics_1a)
    metrics_1b_df = pd.DataFrame(all_metrics_1b)
    preds_1a_df = pd.DataFrame(all_preds_1a)
    preds_1b_df = pd.DataFrame(all_preds_1b)

    print(f"\n{'='*50}")
    print("Layer Selection")
    print(f"{'='*50}")
    selected = select_layers(metrics_1a_df, cfg)

    with open(RESULTS_DIR / "selected_layers.json", "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\n{'='*50}")
    print("Stage 2: Transfer Matrices")
    print(f"{'='*50}")

    all_indexed = {}
    for model_key in MODELS:
        sample_index = pd.read_parquet(ACT_DIR / model_key / "sample_index.parquet")
        all_indexed[model_key] = build_activation_index(sample_index, data)

    m2, p2, matrices = run_stage2(
        all_indexed, selected, acts_cache, manifests,
        metrics_1a_df, cfg, cfg_h
    )
    metrics_2_df = pd.DataFrame(m2)
    preds_2_df = pd.DataFrame(p2)

    print(f"\n{'='*50}")
    print("Saving results")
    print(f"{'='*50}")

    stage1_metrics = pd.concat([metrics_1a_df, metrics_1b_df], ignore_index=True)
    stage1_metrics.to_csv(RESULTS_DIR / "stage1_metrics.csv", index=False)
    print(f"  stage1_metrics.csv: {len(stage1_metrics)} rows (1A + 1B)")

    metrics_2_df.to_csv(RESULTS_DIR / "stage2_metrics.csv", index=False)
    print(f"  stage2_metrics.csv: {len(metrics_2_df)} rows")

    stage1_preds = pd.concat([preds_1a_df, preds_1b_df], ignore_index=True)
    stage1_preds.to_parquet(RESULTS_DIR / "stage1_predictions.parquet", index=False)
    print(f"  Stage 1 predictions: {len(stage1_preds)} rows")

    preds_2_df.to_parquet(RESULTS_DIR / "stage2_predictions.parquet", index=False)
    print(f"  Stage 2 predictions: {len(preds_2_df)} rows (off-diagonal only)")

    for model_key, matrix in matrices.items():
        ordered_matrix = matrix.loc[RELATION_ORDER, RELATION_ORDER]
        fname = f"stage2_matrix_{model_key.split('_')[-1]}.csv"
        ordered_matrix.to_csv(RESULTS_DIR / fname)
        print(f"  Transfer matrix: {fname}")
        print(ordered_matrix.round(3).to_string())
        print()

    # --- Generality gap CSV ---
    gap_rows = []
    for model_key in MODELS:
        for layer_info in manifests[model_key]["target_layers"]:
            li = layer_info["layer_index"]
            depth = layer_info["depth"]
            for rel in RELATION_ORDER:
                within_aucs = metrics_1a_df[
                    (metrics_1a_df["model_key"] == model_key) &
                    (metrics_1a_df["layer_index"] == li) &
                    (metrics_1a_df["target_relation"] == rel)
                ]["auc"]
                mean_within = within_aucs.mean()
                std_within = within_aucs.std()

                loo_auc = metrics_1b_df[
                    (metrics_1b_df["model_key"] == model_key) &
                    (metrics_1b_df["layer_index"] == li) &
                    (metrics_1b_df["target_relation"] == rel)
                ]["auc"].values[0]

                gap_rows.append({
                    "model_key": model_key,
                    "model_id": MODELS[model_key]["model_id"],
                    "layer_index": li,
                    "normalized_depth": depth,
                    "target_relation": rel,
                    "within_mean_auc": mean_within,
                    "within_std_auc": std_within,
                    "loo_auc": loo_auc,
                    "generality_gap": mean_within - loo_auc,
                })

    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(RESULTS_DIR / "generality_gap.csv", index=False)
    print(f"  generality_gap.csv: {len(gap_df)} rows")

    print(f"\n{'='*50}")
    print("Generality Gap Summary")
    print(f"{'='*50}")
    for model_key in MODELS:
        print(f"\n  {model_key}:")
        for layer_info in manifests[model_key]["target_layers"]:
            li = layer_info["layer_index"]
            depth = layer_info["depth"]
            model_gaps = gap_df[
                (gap_df["model_key"] == model_key) &
                (gap_df["layer_index"] == li)
            ]
            mean_abs_gap = model_gaps["generality_gap"].abs().mean()
            print(f"    Layer {li} (depth={depth})  mean|gap|={mean_abs_gap:.4f}")
            for _, row in model_gaps.iterrows():
                print(f"      {row['target_relation']}: "
                      f"within={row['within_mean_auc']:.3f}  "
                      f"LOO={row['loo_auc']:.3f}  "
                      f"gap={row['generality_gap']:+.3f}")

    total_probes = len(all_metrics_1a) + len(all_metrics_1b) + len(m2)
    print(f"\nTotal probes fitted: {total_probes}")
    print(f"Probe weights saved to: {WEIGHTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
