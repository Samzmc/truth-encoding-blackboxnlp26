"""Extract activations from Llama models via NDIF for Phase 2.

Handles both pilot timing runs and full six-relation extraction.
Uses the blackboxnlp-ndif conda environment.

Usage:
    # Pilot: one relation, N50, both models, record timing
    python extract_activations.py --pilot

    # Full extraction after N_final is frozen
    python extract_activations.py --subset N50   # or N75, N100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
DATA_PATH = ROOT / "data" / "processed" / "examples.parquet"
ACTIVATIONS_DIR = ROOT / "activations"
PILOT_DIR = ROOT / "activations" / "pilot"

MODELS = {
    "llama_3_1_8b": "meta-llama/Llama-3.1-8B",
    "llama_3_1_70b": "meta-llama/Llama-3.1-70B",
}

DEPTHS = [0.25, 0.50, 0.75, 1.00]

PILOT_RELATION = "P19"

def load_env() -> str:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Missing .env: {ENV_PATH}")
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    api_key = os.environ.get("NDIF_API_KEY", "").strip()
    if not api_key or api_key.startswith("PASTE_YOUR_"):
        raise RuntimeError("Set NDIF_API_KEY in .env")
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token or hf_token.startswith("PASTE_YOUR_"):
        raise RuntimeError("Set HF_TOKEN in .env")
    return api_key


def get_layer_indices(num_layers: int) -> list[tuple[float, int]]:
    return [(d, math.ceil(d * num_layers) - 1) for d in DEPTHS]


def load_sentences(subset: str, relation: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df[df["subset"] == subset]
    if relation:
        df = df[df["relation_id"] == relation]
    return df.sort_values("example_id").reset_index(drop=True)


def extract_for_model(
    model_key: str,
    model_id: str,
    sentences: list[str],
    api_key: str,
) -> dict[str, Any]:
    """Run extraction for one model, one sentence at a time.

    Passes each sentence as a raw string to match the NNsight/NDIF
    remote trace pattern validated in the smoke test.
    """
    from nnsight import CONFIG, LanguageModel

    CONFIG.set_default_api_key(api_key)

    print(f"\n{'='*50}")
    print(f"Model: {model_id}")
    print(f"Sentences: {len(sentences)}")
    print(f"{'='*50}")

    t_init_start = time.time()
    model = LanguageModel(model_id)
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    t_init = time.time() - t_init_start
    print(f"Model init: {t_init:.1f}s  (layers={num_layers}, hidden={hidden_dim})")

    layers = get_layer_indices(num_layers)
    print(f"Target layers: {[(d, i) for d, i in layers]}")

    tokenizer = model.tokenizer

    assert len(layers) == 4, f"Expected 4 depths, got {len(layers)}"
    li0, li1, li2, li3 = [idx for _, idx in layers]

    all_activations: dict[int, list[np.ndarray]] = {
        li0: [], li1: [], li2: [], li3: [],
    }
    all_positions: list[int] = []

    t_remote_total = 0.0
    t_wall_start = time.time()

    for i, sent in enumerate(sentences):
        tok_len = len(tokenizer(sent)["input_ids"])
        last_pos = tok_len - 1
        all_positions.append(last_pos)

        t_sub = time.time()

        # NNsight 0.7 does not trace Python for-loops inside the
        # context manager — .save() calls inside a loop are silently
        # dropped.  Unroll the four depths explicitly.
        with model.trace(sent, remote=True):
            s0 = model.model.layers[li0].output[0].save()
            s1 = model.model.layers[li1].output[0].save()
            s2 = model.model.layers[li2].output[0].save()
            s3 = model.model.layers[li3].output[0].save()

        t_remote_total += time.time() - t_sub

        for li, tensor in [(li0, s0), (li1, s1), (li2, s2), (li3, s3)]:
            t = tensor[0] if tensor.dim() == 3 else tensor
            vec = t[last_pos, :].detach().cpu().float().numpy()
            all_activations[li].append(vec)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_wall_start
            print(f"  {i+1}/{len(sentences)}  "
                  f"({elapsed:.0f}s elapsed, "
                  f"~{elapsed/(i+1):.2f}s/sent)")

    t_wall_total = time.time() - t_wall_start

    activation_arrays = {}
    for layer_idx, vecs in all_activations.items():
        arr = np.stack(vecs).astype(np.float16)
        activation_arrays[layer_idx] = arr
        print(f"  Layer {layer_idx}: shape={arr.shape}, dtype={arr.dtype}")

    timing = {
        "model_id": model_id,
        "model_key": model_key,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "sentences_processed": len(sentences),
        "model_init_seconds": round(t_init, 2),
        "remote_execution_seconds": round(t_remote_total, 2),
        "total_wall_clock_seconds": round(t_wall_total, 2),
        "seconds_per_sentence": round(t_wall_total / len(sentences), 3),
    }

    return {
        "activations": activation_arrays,
        "positions": all_positions,
        "layers": layers,
        "timing": timing,
    }


def save_activations(
    result: dict[str, Any],
    output_dir: Path,
    model_key: str,
    df: pd.DataFrame,
    dataset_hash: str,
) -> None:
    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, arr in result["activations"].items():
        np.save(model_dir / f"layer_{layer_idx}.npy", arr)

    sample_index = df[["example_id", "pair_id", "case_id", "relation_id",
                        "subject", "label", "subset"]].copy()
    sample_index["token_position"] = result["positions"]
    sample_index.to_parquet(model_dir / "sample_index.parquet", index=False)

    manifest = {
        **result["timing"],
        "target_layers": [
            {"depth": d, "layer_index": i} for d, i in result["layers"]
        ],
        "activation_shapes": {
            str(idx): list(arr.shape)
            for idx, arr in result["activations"].items()
        },
        "activation_dtype": "float16",
        "dataset_hash": dataset_hash,
        "token_position_strategy": "last_real_token (= final subtoken of target attribute)",
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def run_pilot(api_key: str) -> None:
    print("=" * 60)
    print(f"PILOT TIMING: relation={PILOT_RELATION}, subset=N50")
    print("=" * 60)

    df = load_sentences("N50", PILOT_RELATION)
    sentences = df["sentence"].tolist()
    print(f"Loaded {len(sentences)} pilot sentences")

    dataset_hash = hashlib.sha256(
        "\n".join(sentences).encode()
    ).hexdigest()[:16]

    all_timing = {}

    for model_key, model_id in MODELS.items():
        result = extract_for_model(model_key, model_id, sentences, api_key)
        all_timing[model_key] = result["timing"]

        save_activations(result, PILOT_DIR, model_key, df, dataset_hash)
        print(f"\nSaved pilot activations to {PILOT_DIR / model_key}")

    print("\n" + "=" * 60)
    print("PILOT TIMING SUMMARY")
    print("=" * 60)

    for mk, t in all_timing.items():
        print(f"\n  {mk}:")
        print(f"    Wall clock:     {t['total_wall_clock_seconds']:.1f}s")
        print(f"    Remote exec:    {t['remote_execution_seconds']:.1f}s")
        print(f"    Per sentence:   {t['seconds_per_sentence']:.3f}s")
        print(f"    Sentences:      {t['sentences_processed']}")

    total_8b = all_timing["llama_3_1_8b"]["total_wall_clock_seconds"]
    total_70b = all_timing["llama_3_1_70b"]["total_wall_clock_seconds"]

    print("\n  Estimated full extraction times:")
    for label, n_pairs in [("N50", 50), ("N75", 75), ("N100", 100)]:
        multiplier = (6 * n_pairs * 2) / 100
        est_8b = total_8b * multiplier
        est_70b = total_70b * multiplier
        est_total = est_8b + est_70b
        print(f"    {label}: 8B={est_8b/60:.0f}min + 70B={est_70b/60:.0f}min "
              f"= {est_total/60:.0f}min total")

    decision_path = ROOT / "sample_size_decision.json"
    decision = {
        "pilot_relation": PILOT_RELATION,
        "pilot_subset": "N50",
        "pilot_sentences": len(sentences),
        "pilot_timing": all_timing,
        "estimated_full_extraction": {},
    }
    for label, n_pairs in [("N50", 50), ("N75", 75), ("N100", 100)]:
        mult = (6 * n_pairs * 2) / 100
        decision["estimated_full_extraction"][label] = {
            "sentences": 6 * n_pairs * 2,
            "multiplier": mult,
            "estimated_8b_seconds": round(total_8b * mult, 1),
            "estimated_70b_seconds": round(total_70b * mult, 1),
            "estimated_total_seconds": round((total_8b + total_70b) * mult, 1),
        }
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved timing decision to {decision_path}")


def run_full(api_key: str, subset: str) -> None:
    print("=" * 60)
    print(f"FULL EXTRACTION: subset={subset}, all relations")
    print("=" * 60)

    df = load_sentences(subset)
    sentences = df["sentence"].tolist()
    print(f"Loaded {len(sentences)} sentences across {df['relation_id'].nunique()} relations")

    dataset_hash = hashlib.sha256(
        "\n".join(sentences).encode()
    ).hexdigest()[:16]

    for model_key, model_id in MODELS.items():
        result = extract_for_model(model_key, model_id, sentences, api_key)
        out_dir = ACTIVATIONS_DIR / model_key
        save_activations(result, ACTIVATIONS_DIR, model_key, df, dataset_hash)
        print(f"\nSaved activations to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true",
                       help="Run pilot timing with one relation")
    group.add_argument("--subset", choices=["N50", "N75", "N100"],
                       help="Run full extraction for this subset")
    args = parser.parse_args()

    api_key = load_env()

    if args.pilot:
        run_pilot(api_key)
    else:
        run_full(api_key, args.subset)


if __name__ == "__main__":
    main()
