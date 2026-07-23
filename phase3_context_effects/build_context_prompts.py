"""Build and freeze Phase 3 context prompts (Stage A and Stage B).

Stage A: faithful Figure 6a reproduction data (P103, paper recipe, flaws kept).
Stage B: controlled 6x6 grid data (Phase 2 N100 set, paired contexts,
subject + anti-priming exclusions).

Outputs (frozen; both models consume the same files):
    context_prompts/stage_a_p103_tuples.jsonl
    context_prompts/stage_b_grid_tuples.jsonl
    context_prompts/prompts_manifest.json

Run in the `blackboxnlp` conda env:
    python build_context_prompts.py
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_COUNTERFACT = ROOT.parent / "phase2" / "data" / "raw" / "counterfact.json"
EXAMPLES_PARQUET = ROOT.parent / "phase2" / "data" / "processed" / "examples.parquet"
OUT_DIR = ROOT / "context_prompts"

SEED_STAGE_A = 42          # matches the original notebook's tuple seed
SEED_STAGE_B = 20260715

STAGE_A_RELATION = "P103"
STAGE_A_TUPLES_PER_CONDITION = 128
STAGE_A_BASELINE_STUBS = 128
STAGE_A_CONTEXT_LENS = (2, 3, 4)   # -> FF/TT, FFF/TTT, FFFF/TTTT

STAGE_B_SUBSET = "N100"
STAGE_B_TARGETS_PER_RELATION = 64
STAGE_B_CONTEXT_LEN = 2
SEP = ". "


def make_sentence(template: str, subject: str, attribute: str) -> str:
    return f"{template.format(subject).rstrip()} {attribute.strip()}"


def make_stub(template: str, subject: str) -> str:
    return template.format(subject).rstrip()


def build_prompt(context_sentences: list[str], stub: str) -> tuple[str, list[int]]:
    """Join context sentences with '. ', append stub, NO trailing space.

    Returns the prompt text and, for each context sentence, the character
    index (exclusive end) of its final character in the prompt — i.e. the end
    of its displayed attribute.  Used later for token-position mapping.
    """
    parts: list[str] = []
    char_ends: list[int] = []
    pos = 0
    for sent in context_sentences:
        parts.append(sent)
        pos += len(sent)
        char_ends.append(pos)
        parts.append(SEP)
        pos += len(SEP)
    parts.append(stub)
    return "".join(parts), char_ends


# --------------------------------------------------------------------------
# Stage A — faithful P103 reproduction data
# --------------------------------------------------------------------------

def load_p103_facts() -> list[dict[str, Any]]:
    with RAW_COUNTERFACT.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    records = [
        r for r in raw if r["requested_rewrite"]["relation_id"] == STAGE_A_RELATION
    ]
    records.sort(key=lambda r: r["case_id"])
    facts, seen_subjects = [], set()
    for r in records:
        rw = r["requested_rewrite"]
        if rw["subject"] in seen_subjects:  # subject dedup, keep lowest case_id
            continue
        seen_subjects.add(rw["subject"])
        if rw["target_new"]["str"].strip() == rw["target_true"]["str"].strip():
            continue
        facts.append(
            {
                "case_id": r["case_id"],
                "subject": rw["subject"],
                "template": rw["prompt"],
                "true_attribute": rw["target_true"]["str"].strip(),
                "false_attribute": rw["target_new"]["str"].strip(),
            }
        )
    return facts


def stage_a_rows(facts: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    """Faithful recipe: no subject/anti-priming exclusions, seed 42.

    False sentences use CounterFact's own target_new (= 'a different attribute
    from the same relation', exactly the paper's description).
    All-false tuples: M distinct facts, false sentences as context, last is
    the target.  Mixed tuples: M-1 distinct facts as true context + one target
    drawn independently (the original's leakage flaw, kept on purpose).
    """
    rng = random.Random(SEED_STAGE_A)
    rows: list[dict] = []
    stats: dict[str, Any] = {"conditions": {}}

    def add_row(condition: str, ctx: list[dict], ctx_factuality: str,
                target: dict, idx: int) -> None:
        ctx_sents = [
            make_sentence(
                f["template"], f["subject"],
                f["true_attribute"] if ctx_factuality == "T" else f["false_attribute"],
            )
            for f in ctx
        ]
        displayed = [
            f["true_attribute"] if ctx_factuality == "T" else f["false_attribute"]
            for f in ctx
        ]
        stub = make_stub(target["template"], target["subject"])
        prompt, char_ends = build_prompt(ctx_sents, stub)
        rows.append(
            {
                "prompt_id": f"a_{condition}_{idx:04d}",
                "stage": "a",
                "condition": condition,
                "context_len": len(ctx),
                "context_factuality": ctx_factuality * len(ctx) or None,
                "target_relation": STAGE_A_RELATION,
                "context_relation": STAGE_A_RELATION if ctx else None,
                "target_case_id": target["case_id"],
                "target_subject": target["subject"],
                "target_true_attribute": target["true_attribute"],
                "context_case_ids": [f["case_id"] for f in ctx],
                "context_displayed_attributes": displayed,
                "primed": target["true_attribute"] in displayed,
                "leaked_subject": any(
                    f["subject"] == target["subject"] for f in ctx
                ),
                "prompt_text": prompt,
                "context_attr_char_ends": char_ends,
                "pair_key": None,
            }
        )

    for m in STAGE_A_CONTEXT_LENS:
        # all-false: M distinct facts (context m, target 1)
        cond_f = "F" * m
        seen: set[tuple] = set()
        idx = 0
        while idx < STAGE_A_TUPLES_PER_CONDITION:
            tup = rng.sample(facts, m + 1)
            key = tuple(sorted(f["case_id"] for f in tup))
            if key in seen:
                continue
            seen.add(key)
            add_row(cond_f, tup[:m], "F", tup[m], idx)
            idx += 1

        # mixed: M true context + independent target (faithful: no exclusions)
        cond_t = "T" * m
        seen = set()
        idx = 0
        while idx < STAGE_A_TUPLES_PER_CONDITION:
            ctx = rng.sample(facts, m)
            target = rng.choice(facts)
            key = (tuple(sorted(f["case_id"] for f in ctx)), target["case_id"])
            if key in seen:
                continue
            seen.add(key)
            add_row(cond_t, ctx, "T", target, idx)
            idx += 1

    # no-context baseline stubs
    for idx, target in enumerate(rng.sample(facts, STAGE_A_BASELINE_STUBS)):
        stub = make_stub(target["template"], target["subject"])
        rows.append(
            {
                "prompt_id": f"a_base_{idx:04d}",
                "stage": "a",
                "condition": "none",
                "context_len": 0,
                "context_factuality": None,
                "target_relation": STAGE_A_RELATION,
                "context_relation": None,
                "target_case_id": target["case_id"],
                "target_subject": target["subject"],
                "target_true_attribute": target["true_attribute"],
                "context_case_ids": [],
                "context_displayed_attributes": [],
                "primed": False,
                "leaked_subject": False,
                "prompt_text": stub,
                "context_attr_char_ends": [],
                "pair_key": None,
            }
        )

    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond]
        stats["conditions"][cond] = {
            "prompts": len(sub),
            "primed": sum(r["primed"] for r in sub),
            "leaked_subject": sum(r["leaked_subject"] for r in sub),
        }
    stats["facts_pool"] = len(facts)
    stats["attribute_top3"] = Counter(
        f["true_attribute"] for f in facts
    ).most_common(3)
    return rows, stats


# --------------------------------------------------------------------------
# Stage B — controlled 6x6 grid
# --------------------------------------------------------------------------

def load_stage_b_pairs() -> dict[str, list[dict[str, Any]]]:
    df = pd.read_parquet(EXAMPLES_PARQUET)
    df = df[df["subset"] == STAGE_B_SUBSET]
    pairs_by_relation: dict[str, list[dict]] = {}
    for (relation, pair_id), grp in df.groupby(["relation_id", "pair_id"]):
        true_row = grp[grp["label"] == 1].iloc[0]
        false_row = grp[grp["label"] == 0].iloc[0]
        pairs_by_relation.setdefault(relation, []).append(
            {
                "pair_id": str(pair_id),
                "relation": relation,
                "subject": true_row["subject"],
                "template": true_row["template"],
                "true_attribute": true_row["used_attribute"],
                "false_attribute": false_row["used_attribute"],
                "true_sentence": true_row["sentence"],
                "false_sentence": false_row["sentence"],
                "true_example_id": true_row["example_id"],
                "false_example_id": false_row["example_id"],
            }
        )
    for relation in pairs_by_relation:
        pairs_by_relation[relation].sort(key=lambda p: p["pair_id"])
    return pairs_by_relation


def stage_b_rows(pairs_by_relation: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    rng = random.Random(SEED_STAGE_B)
    relations = sorted(pairs_by_relation)
    rows: list[dict] = []
    stats: dict[str, Any] = {"cells": {}, "min_pool_size": None}
    min_pool = None

    targets_by_relation = {
        rel: rng.sample(pairs_by_relation[rel], STAGE_B_TARGETS_PER_RELATION)
        for rel in relations
    }

    for tgt_rel in relations:
        for t_idx, target in enumerate(targets_by_relation[tgt_rel]):
            stub = make_stub(target["template"], target["subject"])
            rows.append(
                {
                    "prompt_id": f"b_base_{tgt_rel}_{t_idx:03d}",
                    "stage": "b",
                    "condition": "none",
                    "context_len": 0,
                    "context_factuality": None,
                    "target_relation": tgt_rel,
                    "context_relation": None,
                    "target_pair_id": target["pair_id"],
                    "target_subject": target["subject"],
                    "target_true_attribute": target["true_attribute"],
                    "context_pair_ids": [],
                    "context_displayed_attributes": [],
                    "prompt_text": stub,
                    "context_attr_char_ends": [],
                    "pair_key": None,
                }
            )
            for ctx_rel in relations:
                pool = [
                    p
                    for p in pairs_by_relation[ctx_rel]
                    if p["subject"] != target["subject"]
                    and p["true_attribute"] != target["true_attribute"]
                    and p["false_attribute"] != target["true_attribute"]
                    and p["pair_id"] != target["pair_id"]
                ]
                cell = f"{ctx_rel}->{tgt_rel}"
                cell_stats = stats["cells"].setdefault(
                    cell, {"pool_sizes": [], "prompts": 0}
                )
                cell_stats["pool_sizes"].append(len(pool))
                if min_pool is None or len(pool) < min_pool:
                    min_pool = len(pool)
                if len(pool) < STAGE_B_CONTEXT_LEN:
                    raise RuntimeError(
                        f"Cell {cell}, target pair {target['pair_id']}: "
                        f"context pool too small ({len(pool)})."
                    )
                ctx = rng.sample(pool, STAGE_B_CONTEXT_LEN)
                pair_key = (
                    f"{ctx_rel}->{tgt_rel}|t{target['pair_id']}|"
                    + "-".join(str(p["pair_id"]) for p in ctx)
                )
                for factuality, cond in (("T", "TT"), ("F", "FF")):
                    ctx_sents = [
                        p["true_sentence"] if factuality == "T" else p["false_sentence"]
                        for p in ctx
                    ]
                    displayed = [
                        p["true_attribute"] if factuality == "T" else p["false_attribute"]
                        for p in ctx
                    ]
                    prompt, char_ends = build_prompt(ctx_sents, stub)
                    rows.append(
                        {
                            "prompt_id": (
                                f"b_{cond}_{ctx_rel}_{tgt_rel}_{t_idx:03d}"
                            ),
                            "stage": "b",
                            "condition": cond,
                            "context_len": STAGE_B_CONTEXT_LEN,
                            "context_factuality": factuality * STAGE_B_CONTEXT_LEN,
                            "target_relation": tgt_rel,
                            "context_relation": ctx_rel,
                            "target_pair_id": target["pair_id"],
                            "target_subject": target["subject"],
                            "target_true_attribute": target["true_attribute"],
                            "context_pair_ids": [p["pair_id"] for p in ctx],
                            "context_displayed_attributes": displayed,
                            "prompt_text": prompt,
                            "context_attr_char_ends": char_ends,
                            "pair_key": pair_key,
                        }
                    )
                    cell_stats["prompts"] += 1

    for cell, cs in stats["cells"].items():
        sizes = cs.pop("pool_sizes")
        cs["pool_min"] = min(sizes)
        cs["pool_mean"] = round(sum(sizes) / len(sizes), 1)
    stats["min_pool_size"] = min_pool
    stats["relations"] = relations
    return rows, stats


# --------------------------------------------------------------------------

def validate(rows_a: list[dict], rows_b: list[dict]) -> None:
    n_a = 6 * STAGE_A_TUPLES_PER_CONDITION + STAGE_A_BASELINE_STUBS
    assert len(rows_a) == n_a, f"Stage A: {len(rows_a)} != {n_a}"
    n_b = 6 * 6 * STAGE_B_TARGETS_PER_RELATION * 2 + 6 * STAGE_B_TARGETS_PER_RELATION
    assert len(rows_b) == n_b, f"Stage B: {len(rows_b)} != {n_b}"
    for rows in (rows_a, rows_b):
        ids = [r["prompt_id"] for r in rows]
        assert len(ids) == len(set(ids)), "duplicate prompt_id"
        for r in rows:
            assert not r["prompt_text"].endswith(" "), "trailing space"
            assert r["target_true_attribute"], "missing target attribute"
            for end in r["context_attr_char_ends"]:
                displayed = r["context_displayed_attributes"]
                assert r["prompt_text"][:end].endswith(tuple(displayed)), (
                    f"char_end misaligned for {r['prompt_id']}"
                )
    # Stage B: every TT prompt has a matching FF prompt with the same facts
    by_key: dict[str, set] = {}
    for r in rows_b:
        if r["pair_key"]:
            by_key.setdefault(r["pair_key"], set()).add(r["condition"])
    assert all(v == {"TT", "FF"} for v in by_key.values()), "unpaired Stage B rows"
    # Stage B exclusions hold
    for r in rows_b:
        if r["condition"] in ("TT", "FF"):
            assert r["target_true_attribute"] not in r[
                "context_displayed_attributes"
            ], f"priming leak in {r['prompt_id']}"


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> None:
    facts = load_p103_facts()
    rows_a, stats_a = stage_a_rows(facts)
    pairs = load_stage_b_pairs()
    rows_b, stats_b = stage_b_rows(pairs)
    validate(rows_a, rows_b)

    hash_a = write_jsonl(OUT_DIR / "stage_a_p103_tuples.jsonl", rows_a)
    hash_b = write_jsonl(OUT_DIR / "stage_b_grid_tuples.jsonl", rows_b)

    manifest = {
        "built_at": "2026-07-15",
        "seeds": {"stage_a": SEED_STAGE_A, "stage_b": SEED_STAGE_B},
        "sources": {
            "counterfact_raw": str(RAW_COUNTERFACT),
            "examples_parquet": str(EXAMPLES_PARQUET),
            "stage_b_subset": STAGE_B_SUBSET,
        },
        "stage_a": {
            "prompts": len(rows_a),
            "negatives": "counterfact target_new (paper recipe)",
            **stats_a,
        },
        "stage_b": {
            "prompts": len(rows_b),
            "targets_per_relation": STAGE_B_TARGETS_PER_RELATION,
            **stats_b,
        },
        "file_hashes": {
            "stage_a_p103_tuples.jsonl": hash_a,
            "stage_b_grid_tuples.jsonl": hash_b,
        },
    }
    (OUT_DIR / "prompts_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Stage A: {len(rows_a)} prompts (pool={stats_a['facts_pool']} facts)")
    for cond, cs in stats_a["conditions"].items():
        print(f"  {cond:>5}: {cs['prompts']} prompts, primed={cs['primed']}, "
              f"leaked_subject={cs['leaked_subject']}")
    print(f"Stage B: {len(rows_b)} prompts, min context pool = "
          f"{stats_b['min_pool_size']}")
    print("\nSample prompts:")
    for r in (rows_a[0], rows_a[400], rows_b[1], rows_b[2]):
        print(f"\n[{r['prompt_id']}] target attr = {r['target_true_attribute']!r}")
        print(f"  {r['prompt_text']}")
    print("\nFrozen. Hashes:", hash_a, hash_b)


if __name__ == "__main__":
    main()
