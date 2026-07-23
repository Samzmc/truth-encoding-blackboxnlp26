"""Phase 3 NDIF extraction: next-token logits + activations per prompt.

One remote trace per prompt (Phase 2-validated pattern).  Saves, per prompt:
  - full last-position logits row (fp16); NLL/entropy/rank computed locally
    in fp32
  - activations (fp16) at the model's Phase 2 depths, at each context
    sentence's attribute-final subtoken and at the final prompt position

Resumable: output is written in shards of 200 prompts; a rerun skips
prompt_ids already present in existing shards.

Usage (blackboxnlp-ndif conda env):
    python extract_context_probs.py --stage b --model 8b --pilot 30
    python extract_context_probs.py --stage a --model 70b
    python extract_context_probs.py --stage b --model 70b --consolidate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT.parent / "phase2" / ".env"
PROMPTS = {
    "a": ROOT / "context_prompts" / "stage_a_p103_tuples.jsonl",
    "b": ROOT / "context_prompts" / "stage_b_grid_tuples.jsonl",
}
OUT_ROOT = ROOT / "context_activations"

MODELS = {
    "8b": ("llama_3_1_8b", "meta-llama/Llama-3.1-8B"),
    "70b": ("llama_3_1_70b", "meta-llama/Llama-3.1-70B"),
}
# Phase 2 depths {0.25, 0.5, 0.75, 1.0} — keep identical so Phase 2 probes
# apply directly to these activations.
LAYERS = {"8b": [7, 15, 23, 31], "70b": [19, 39, 59, 79]}

SHARD_SIZE = 200
MAX_RETRIES = 3


def load_env() -> str:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Missing .env: {ENV_PATH}")
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    api_key = os.environ.get("NDIF_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set NDIF_API_KEY in phase2/.env")
    return api_key


def load_prompts(stage: str) -> list[dict[str, Any]]:
    rows = []
    with PROMPTS[stage].open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def char_end_to_token_pos(offsets: list[tuple[int, int]], char_end: int) -> int:
    """Last token whose span contains character char_end - 1."""
    best = None
    for i, (s, e) in enumerate(offsets):
        if s == e:  # special tokens (BOS) have empty spans
            continue
        if s < char_end <= e:
            best = i
    if best is None:
        raise ValueError(f"No token covers char_end={char_end}")
    return best


def done_prompt_ids(shard_dir: Path) -> set[str]:
    done: set[str] = set()
    for meta_path in sorted(shard_dir.glob("shard_*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        done.update(m["prompt_id"] for m in meta["prompts"])
    return done


def flush_shard(
    shard_dir: Path,
    shard_idx: int,
    layer_rows: dict[int, list[np.ndarray]],
    logits_rows: list[np.ndarray],
    metas: list[dict],
) -> None:
    arrays = {
        f"layer_{li}": np.stack(rows).astype(np.float16)
        for li, rows in layer_rows.items()
    }
    arrays["logits"] = np.stack(logits_rows).astype(np.float16)
    np.savez_compressed(shard_dir / f"shard_{shard_idx:05d}.npz", **arrays)
    (shard_dir / f"shard_{shard_idx:05d}.meta.json").write_text(
        json.dumps({"prompts": metas}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run(stage: str, model_key: str, pilot: int | None) -> None:
    import torch
    from nnsight import CONFIG, LanguageModel

    api_key = load_env()
    CONFIG.set_default_api_key(api_key)

    dir_key, model_id = MODELS[model_key]
    layers = LAYERS[model_key]
    out_dir = OUT_ROOT / dir_key / f"stage_{stage}"
    if pilot:
        out_dir = OUT_ROOT / "pilot" / f"{dir_key}_stage_{stage}"
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(stage)
    if pilot:
        # spread pilot prompts over conditions: take every k-th prompt
        step = max(1, len(prompts) // pilot)
        prompts = prompts[::step][:pilot]

    done = done_prompt_ids(shard_dir)
    todo = [p for p in prompts if p["prompt_id"] not in done]
    print(f"{model_id} stage_{stage}: {len(todo)} to do "
          f"({len(done)} already done), layers={layers}")
    if not todo:
        return

    t0 = time.time()
    model = LanguageModel(model_id)
    tokenizer = model.tokenizer
    assert len(layers) == 4
    l0, l1, l2, l3 = layers
    print(f"model init {time.time()-t0:.1f}s")

    shard_idx = 0
    existing = sorted(shard_dir.glob("shard_*.npz"))
    if existing:
        shard_idx = int(existing[-1].stem.split("_")[1]) + 1

    layer_rows: dict[int, list[np.ndarray]] = {li: [] for li in layers}
    logits_rows: list[np.ndarray] = []
    metas: list[dict] = []
    failures: list[str] = []
    t_start = time.time()

    for i, p in enumerate(todo):
        text = p["prompt_text"]
        enc = tokenizer(text, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        final_pos = len(ids) - 1
        ctx_pos = [
            char_end_to_token_pos(offsets, ce)
            for ce in p["context_attr_char_ends"]
        ]
        positions = ctx_pos + [final_pos]
        roles = [f"ctx_{k+1}" for k in range(len(ctx_pos))] + ["final"]

        tgt_ids = tokenizer(
            " " + p["target_true_attribute"], add_special_tokens=False
        )["input_ids"]
        tgt_id = tgt_ids[0]

        ok = False
        for attempt in range(MAX_RETRIES):
            try:
                with model.trace(text, remote=True):
                    s0 = model.model.layers[l0].output[0][..., positions, :].save()
                    s1 = model.model.layers[l1].output[0][..., positions, :].save()
                    s2 = model.model.layers[l2].output[0][..., positions, :].save()
                    s3 = model.model.layers[l3].output[0][..., positions, :].save()
                    lg = model.lm_head.output[..., -1, :].save()
                ok = True
                break
            except Exception as exc:  # noqa: BLE001 — NDIF hiccups; retry
                wait = 5 * (attempt + 1)
                print(f"  [{p['prompt_id']}] attempt {attempt+1} failed: "
                      f"{type(exc).__name__}: {exc}; retry in {wait}s")
                time.sleep(wait)
        if not ok:
            failures.append(p["prompt_id"])
            continue

        def to_2d(t: "torch.Tensor", n: int) -> np.ndarray:
            a = t.detach().cpu().float()
            a = a.reshape(-1, a.shape[-1])
            assert a.shape[0] == n, f"unexpected shape {tuple(t.shape)}"
            return a.numpy()

        n_pos = len(positions)
        for li, s in zip(layers, (s0, s1, s2, s3)):
            rows = to_2d(s, n_pos)
            for r in rows:
                layer_rows[li].append(r)

        logits = to_2d(lg, 1)[0]
        logits_rows.append(logits)
        logp = torch.log_softmax(torch.from_numpy(logits).float(), dim=-1)
        nll = float(-logp[tgt_id])
        entropy = float(-(logp.exp() * logp).sum())
        rank = int((logp > logp[tgt_id]).sum()) + 1
        top1_id = int(logp.argmax())

        metas.append(
            {
                "prompt_id": p["prompt_id"],
                "n_positions": n_pos,
                "roles": roles,
                "token_positions": positions,
                "prompt_tokens": len(ids),
                "target_token_id": tgt_id,
                "target_n_subtokens": len(tgt_ids),
                "nll": nll,
                "p_target": float(math.exp(-nll)),
                "entropy": entropy,
                "target_rank": rank,
                "top1_token_id": top1_id,
                "top1_token": tokenizer.decode([top1_id]),
            }
        )

        if len(metas) >= SHARD_SIZE:
            flush_shard(shard_dir, shard_idx, layer_rows, logits_rows, metas)
            shard_idx += 1
            layer_rows = {li: [] for li in layers}
            logits_rows, metas = [], []

        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t_start
            print(f"  {i+1}/{len(todo)}  {el:.0f}s  ~{el/(i+1):.2f}s/prompt")

    if metas:
        flush_shard(shard_dir, shard_idx, layer_rows, logits_rows, metas)

    wall = time.time() - t_start
    n_done = len(todo) - len(failures)
    run_manifest = {
        "model_id": model_id,
        "stage": stage,
        "layers": layers,
        "prompts_processed": n_done,
        "failures": failures,
        "wall_seconds": round(wall, 1),
        "seconds_per_prompt": round(wall / max(n_done, 1), 3),
        "prompts_file_hash_check": PROMPTS[stage].name,
        "activation_dtype": "float16",
        "logits_dtype": "float16",
        "position_convention": (
            "ctx_k = final subtoken of context sentence k's displayed "
            "attribute; final = last prompt token (prediction position)"
        ),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\ndone: {n_done} prompts, {len(failures)} failures, "
          f"{wall:.0f}s ({wall/max(n_done,1):.2f}s/prompt)")
    if failures:
        print("FAILED prompt_ids:", failures[:20],
              "... rerun the same command to retry" if len(failures) > 20 else
              "— rerun the same command to retry")

    if pilot:
        print("\n--- pilot sanity ---")
        meta_files = sorted(shard_dir.glob("shard_*.meta.json"))
        all_meta = []
        for mf in meta_files:
            all_meta.extend(json.loads(mf.read_text(encoding="utf-8"))["prompts"])
        by_id = {p["prompt_id"]: p for p in load_prompts(stage)}
        for m in all_meta[:10]:
            p = by_id[m["prompt_id"]]
            print(f"[{m['prompt_id']}] ...{p['prompt_text'][-60:]!r}")
            print(f"   target={p['target_true_attribute']!r} "
                  f"p={m['p_target']:.4f} nll={m['nll']:.2f} "
                  f"rank={m['target_rank']} top1={m['top1_token']!r}")


def consolidate(stage: str, model_key: str) -> None:
    import pandas as pd

    dir_key, model_id = MODELS[model_key]
    layers = LAYERS[model_key]
    out_dir = OUT_ROOT / dir_key / f"stage_{stage}"
    shard_dir = out_dir / "shards"
    shards = sorted(shard_dir.glob("shard_*.npz"))
    if not shards:
        raise RuntimeError(f"No shards in {shard_dir}")

    prompts = {p["prompt_id"]: p for p in load_prompts(stage)}
    layer_parts: dict[int, list[np.ndarray]] = {li: [] for li in layers}
    logits_parts: list[np.ndarray] = []
    index_rows: list[dict] = []
    logit_row = 0

    for npz_path in shards:
        meta = json.loads(
            npz_path.with_suffix("").with_suffix(".meta.json").read_text(
                encoding="utf-8"
            )
        )["prompts"]
        data = np.load(npz_path)
        for li in layers:
            layer_parts[li].append(data[f"layer_{li}"])
        logits_parts.append(data["logits"])
        act_row = sum(a.shape[0] for a in layer_parts[layers[0]][:-1])
        for m in meta:
            p = prompts[m["prompt_id"]]
            for k, role in enumerate(m["roles"]):
                index_rows.append(
                    {
                        "prompt_id": m["prompt_id"],
                        "stage": p["stage"],
                        "condition": p["condition"],
                        "context_len": p["context_len"],
                        "target_relation": p["target_relation"],
                        "context_relation": p["context_relation"],
                        "target_true_attribute": p["target_true_attribute"],
                        "pair_key": p.get("pair_key"),
                        "position_role": role,
                        "token_position": m["token_positions"][k],
                        "act_row": act_row,
                        "logit_row": logit_row,
                        "target_token_id": m["target_token_id"],
                        "nll": m["nll"],
                        "p_target": m["p_target"],
                        "entropy": m["entropy"],
                        "target_rank": m["target_rank"],
                    }
                )
                act_row += 1
            logit_row += 1

    for li in layers:
        arr = np.concatenate(layer_parts[li], axis=0)
        np.save(out_dir / f"layer_{li}.npy", arr)
        print(f"layer_{li}.npy {arr.shape} {arr.dtype}")
    logits = np.concatenate(logits_parts, axis=0)
    np.save(out_dir / "logprobs_logits.npy", logits)
    print(f"logprobs_logits.npy {logits.shape} {logits.dtype}")
    df = pd.DataFrame(index_rows)
    df.to_parquet(out_dir / "sample_index.parquet", index=False)
    print(f"sample_index.parquet {len(df)} rows "
          f"({df['prompt_id'].nunique()} prompts)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a", "b"], required=True)
    parser.add_argument("--model", choices=["8b", "70b"], required=True)
    parser.add_argument("--pilot", type=int, default=None)
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()
    if args.consolidate:
        consolidate(args.stage, args.model)
    else:
        run(args.stage, args.model, args.pilot)


if __name__ == "__main__":
    main()
