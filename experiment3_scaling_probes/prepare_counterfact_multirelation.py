"""Prepare multi-relation CounterFact dataset for Phase 2 scaling experiment.

Processes six CounterFact relations into balanced true/false sentence pairs
with derangement-based negative construction.  Produces nested N50/N75/N100
subsets for the timing-gate sample-size selection.

Run from this directory under the blackboxnlp conda env:
    python prepare_counterfact_multirelation.py
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

SEED = 20260712

RELATIONS = ["P19", "P103", "P176", "P101", "P159", "P138"]
RELATION_NAMES = {
    "P19": "place of birth",
    "P103": "native language",
    "P176": "manufacturer",
    "P101": "field of work",
    "P159": "headquarters location",
    "P138": "named after",
}

SUBSET_SIZES = [("N50", 50), ("N75", 75), ("N100", 100)]

P103_FRENCH_CAPS = {"N50": 20, "N75": 30, "N100": 40}

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "counterfact.json"
OUTPUT_DIR = ROOT / "data" / "processed"


def make_sentence(template: str, subject: str, attribute: str) -> str:
    return f"{template.format(subject).rstrip()} {attribute.strip()}"


def load_raw(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Raw CounterFact not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def relation_seed(rel: str) -> int:
    return SEED + RELATIONS.index(rel) * 10007


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def filter_relations(records: list[dict]) -> dict[str, list[dict]]:
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        rel = r["requested_rewrite"]["relation_id"]
        if rel in RELATIONS:
            by_rel[rel].append(r)
    return dict(by_rel)


def dedup_subjects(records: list[dict]) -> list[dict]:
    """Keep lowest case_id per subject."""
    sorted_recs = sorted(records, key=lambda r: r["case_id"])
    seen: set[str] = set()
    out = []
    for r in sorted_recs:
        subj = r["requested_rewrite"]["subject"]
        if subj not in seen:
            seen.add(subj)
            out.append(r)
    return out


def find_cross_relation_subjects(
    by_rel: dict[str, list[dict]],
) -> set[str]:
    subj_rels: dict[str, set[str]] = defaultdict(set)
    for rel, recs in by_rel.items():
        for r in recs:
            subj_rels[r["requested_rewrite"]["subject"]].add(rel)
    return {s for s, rels in subj_rels.items() if len(rels) > 1}


def remove_subjects(records: list[dict], subjects: set[str]) -> list[dict]:
    return [r for r in records if r["requested_rewrite"]["subject"] not in subjects]


def dedup_sentences(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in records:
        rw = r["requested_rewrite"]
        sent = make_sentence(rw["prompt"], rw["subject"], rw["target_true"]["str"])
        if sent not in seen:
            seen.add(sent)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_nested(
    records: list[dict],
    rel: str,
) -> dict[str, list[dict]]:
    if rel == "P103":
        return _sample_p103(records)

    rng = random.Random(relation_seed(rel))
    pool = list(records)
    rng.shuffle(pool)

    subsets = {}
    for name, size in SUBSET_SIZES:
        if size > len(pool):
            raise ValueError(f"{rel}: need {size} records for {name}, have {len(pool)}")
        subsets[name] = sorted(pool[:size], key=lambda r: r["case_id"])
    return subsets


def _sample_p103(records: list[dict]) -> dict[str, list[dict]]:
    french = [r for r in records if r["requested_rewrite"]["target_true"]["str"] == "French"]
    non_french = [r for r in records if r["requested_rewrite"]["target_true"]["str"] != "French"]

    base = relation_seed("P103")
    rng_fr = random.Random(base + 1)
    rng_oth = random.Random(base + 2)
    rng_fr.shuffle(french)
    rng_oth.shuffle(non_french)

    subsets = {}
    for name, size in SUBSET_SIZES:
        cap = P103_FRENCH_CAPS[name]
        n_other = size - cap
        if cap > len(french):
            raise ValueError(f"P103: need {cap} French for {name}, have {len(french)}")
        if n_other > len(non_french):
            raise ValueError(f"P103: need {n_other} non-French for {name}, have {len(non_french)}")
        selected = french[:cap] + non_french[:n_other]
        subsets[name] = sorted(selected, key=lambda r: r["case_id"])
    return subsets


# ---------------------------------------------------------------------------
# Derangement
# ---------------------------------------------------------------------------

def construct_derangement(
    records: list[dict],
) -> list[tuple[dict, str, int]]:
    """Attribute-shifted derangement.

    Sort records by (attribute, case_id), then shift indices by max group
    size.  Since max_group <= n//2, the shifted block never overlaps the
    original block for any attribute, guaranteeing no self-match.

    Returns (record, false_attribute, source_case_id) for each record.
    """
    n = len(records)
    attrs = [r["requested_rewrite"]["target_true"]["str"] for r in records]

    sorted_idx = sorted(range(n), key=lambda i: (attrs[i], records[i]["case_id"]))
    sorted_attrs = [attrs[i] for i in sorted_idx]

    max_group = max(Counter(attrs).values())
    if max_group > n // 2:
        attr_counts = Counter(attrs).most_common(3)
        raise ValueError(
            f"Derangement impossible: max group {max_group} > n//2={n // 2}. "
            f"Top attributes: {attr_counts}"
        )

    shift = max_group
    result: list[tuple[dict, str, int]] = []
    for pos, orig_i in enumerate(sorted_idx):
        target_pos = (pos + shift) % n
        target_i = sorted_idx[target_pos]
        false_attr = attrs[target_i]
        assert false_attr != attrs[orig_i], (
            f"Self-match at pos {pos}: {false_attr}"
        )
        result.append((records[orig_i], false_attr, records[target_i]["case_id"]))

    return result


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------

def build_examples(
    deranged: list[tuple[dict, str, int]],
    subset_name: str,
) -> list[dict]:
    examples = []
    for record, false_attr, source_case_id in deranged:
        rw = record["requested_rewrite"]
        cid = record["case_id"]
        subj = rw["subject"]
        tpl = rw["prompt"]
        true_attr = rw["target_true"]["str"]
        rel = rw["relation_id"]

        common = {
            "pair_id": f"{rel}_{subset_name}_{cid}",
            "case_id": cid,
            "relation_id": rel,
            "subject": subj,
            "template": tpl,
            "true_attribute": true_attr,
            "false_attribute_source_id": source_case_id,
            "subset": subset_name,
        }
        examples.append({
            **common,
            "example_id": f"{rel}_{subset_name}_{cid}_true",
            "used_attribute": true_attr,
            "sentence": make_sentence(tpl, subj, true_attr),
            "label": 1,
        })
        examples.append({
            **common,
            "example_id": f"{rel}_{subset_name}_{cid}_false",
            "used_attribute": false_attr,
            "sentence": make_sentence(tpl, subj, false_attr),
            "label": 0,
        })
    return examples


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_subset(
    examples: list[dict],
    subset_name: str,
    rel: str,
) -> dict[str, Any]:
    true_ex = [e for e in examples if e["label"] == 1]
    false_ex = [e for e in examples if e["label"] == 0]
    errors: list[str] = []

    expected = dict(SUBSET_SIZES)[subset_name]
    if len(true_ex) != expected:
        errors.append(f"Expected {expected} pairs, got {len(true_ex)}")
    if len(true_ex) != len(false_ex):
        errors.append(f"Class imbalance: {len(true_ex)} true vs {len(false_ex)} false")

    ids = [e["example_id"] for e in examples]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate example_ids")

    sents = [e["sentence"] for e in examples]
    if len(sents) != len(set(sents)):
        n_dup = len(sents) - len(set(sents))
        errors.append(f"{n_dup} duplicate sentences")

    subjects = [e["subject"] for e in true_ex]
    if len(subjects) != len(set(subjects)):
        errors.append("Duplicate subjects")

    for e in false_ex:
        if e["used_attribute"] == e["true_attribute"]:
            errors.append(f"false==true for case_id {e['case_id']}")
            break

    true_marginals = Counter(e["used_attribute"] for e in true_ex)
    false_marginals = Counter(e["used_attribute"] for e in false_ex)
    marginals_match = true_marginals == false_marginals
    if not marginals_match:
        errors.append("Attribute marginals differ between true and false")

    if rel == "P103":
        n_french = sum(1 for e in true_ex if e["true_attribute"] == "French")
        expected_cap = P103_FRENCH_CAPS[subset_name]
        if n_french != expected_cap:
            errors.append(f"P103 French count {n_french} != cap {expected_cap}")

    return {
        "relation_id": rel,
        "subset": subset_name,
        "pairs": len(true_ex),
        "examples": len(examples),
        "unique_subjects": len(set(subjects)),
        "unique_attributes_true": len(true_marginals),
        "unique_attributes_false": len(false_marginals),
        "marginals_match": marginals_match,
        "errors": errors,
        "valid": len(errors) == 0,
    }


def validate_nesting(
    subsets: dict[str, list[dict]],
    rel: str,
) -> list[str]:
    """Check N50 ⊂ N75 ⊂ N100 by case_id."""
    errors = []
    ids = {name: {e["example_id"] for e in exs if e["label"] == 1}
           for name, exs in subsets.items()}
    case_ids = {name: {e["case_id"] for e in exs if e["label"] == 1}
                for name, exs in subsets.items()}

    if not case_ids["N50"] <= case_ids["N75"]:
        errors.append(f"{rel}: N50 not subset of N75")
    if not case_ids["N75"] <= case_ids["N100"]:
        errors.append(f"{rel}: N75 not subset of N100")
    return errors


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_outputs(
    all_examples: list[dict],
    relation_meta: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = OUTPUT_DIR / "examples.parquet"
    df = pd.DataFrame(all_examples)
    col_order = [
        "example_id", "pair_id", "case_id", "relation_id", "subject",
        "template", "true_attribute", "used_attribute", "sentence",
        "label", "false_attribute_source_id", "subset",
    ]
    df = df[col_order]
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"\nSaved {len(df)} examples to {parquet_path}")

    manifest["output_hash"] = sha256_file(parquet_path)

    rel_path = OUTPUT_DIR / "relations.json"
    rel_path.write_text(
        json.dumps(relation_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest_path = OUTPUT_DIR / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved relations.json and dataset_manifest.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("CounterFact multi-relation preprocessing")
    print("=" * 60)

    raw = load_raw(RAW_PATH)
    print(f"Loaded {len(raw)} raw CounterFact records\n")

    # --- Stage 1: filter ---
    by_rel = filter_relations(raw)
    s1 = {r: len(v) for r, v in by_rel.items()}
    print("Stage 1 — Filter to target relations:")
    for rel in RELATIONS:
        print(f"  {rel} ({RELATION_NAMES[rel]}): {s1[rel]}")

    # --- Stage 2: within-relation subject dedup ---
    for rel in RELATIONS:
        by_rel[rel] = dedup_subjects(by_rel[rel])
    s2 = {r: len(v) for r, v in by_rel.items()}
    print("\nStage 2 — Within-relation subject dedup:")
    for rel in RELATIONS:
        d = s1[rel] - s2[rel]
        print(f"  {rel}: {s1[rel]} -> {s2[rel]}  ({d} removed)")

    # --- Stage 3: cross-relation subject removal ---
    cross_subjs = find_cross_relation_subjects(by_rel)
    for rel in RELATIONS:
        by_rel[rel] = remove_subjects(by_rel[rel], cross_subjs)
    s3 = {r: len(v) for r, v in by_rel.items()}
    print(f"\nStage 3 — Cross-relation subject removal ({len(cross_subjs)} subjects):")
    for rel in RELATIONS:
        d = s2[rel] - s3[rel]
        print(f"  {rel}: {s2[rel]} -> {s3[rel]}  ({d} removed)")
    print(f"  Removed subjects: {sorted(cross_subjs)}")

    # --- Stage 4: sentence dedup ---
    for rel in RELATIONS:
        by_rel[rel] = dedup_sentences(by_rel[rel])
    s4 = {r: len(v) for r, v in by_rel.items()}
    print("\nStage 4 — Sentence dedup:")
    for rel in RELATIONS:
        d = s3[rel] - s4[rel]
        print(f"  {rel}: {s3[rel]} -> {s4[rel]}  ({d} removed)")

    # --- P103 attribute distribution after cleaning ---
    p103_attrs = Counter(
        r["requested_rewrite"]["target_true"]["str"] for r in by_rel["P103"]
    )
    n_french = p103_attrs.get("French", 0)
    print(f"\nP103 after cleaning: {s4['P103']} records, "
          f"French={n_french} ({100*n_french/s4['P103']:.1f}%)")

    # --- Sample, derange, build examples ---
    all_examples: list[dict] = []
    all_validations: list[dict] = []
    nesting_errors: list[str] = []
    relation_meta: dict[str, Any] = {}

    for rel in RELATIONS:
        print(f"\nProcessing {rel}...")
        subsets_records = sample_nested(by_rel[rel], rel)

        rel_info: dict[str, Any] = {
            "relation_id": rel,
            "relation_name": RELATION_NAMES[rel],
            "cleaning_counts": {
                "raw": s1[rel],
                "after_subject_dedup": s2[rel],
                "after_cross_relation_removal": s3[rel],
                "after_sentence_dedup": s4[rel],
            },
            "subsets": {},
        }

        subset_examples: dict[str, list[dict]] = {}

        for subset_name, size in SUBSET_SIZES:
            recs = subsets_records[subset_name]
            deranged = construct_derangement(recs)
            exs = build_examples(deranged, subset_name)

            val = validate_subset(exs, subset_name, rel)
            all_validations.append(val)
            subset_examples[subset_name] = exs
            all_examples.extend(exs)

            attr_dist = Counter(
                r["requested_rewrite"]["target_true"]["str"] for r in recs
            )
            rel_info["subsets"][subset_name] = {
                "pairs": len(recs),
                "examples": len(exs),
                "attribute_distribution": dict(attr_dist.most_common()),
                "unique_templates": len({r["requested_rewrite"]["prompt"] for r in recs}),
                "unique_subjects": len({r["requested_rewrite"]["subject"] for r in recs}),
                "validation": val,
            }

            status = "PASS" if val["valid"] else f"FAIL: {val['errors']}"
            print(f"  {subset_name}: {size} pairs -> {len(exs)} examples  {status}")

        nest_err = validate_nesting(subset_examples, rel)
        nesting_errors.extend(nest_err)
        if nest_err:
            print(f"  NESTING ERROR: {nest_err}")
        else:
            print(f"  Nesting N50 ⊂ N75 ⊂ N100: OK")

        relation_meta[rel] = rel_info

    # --- Cross-relation duplicate check on N100 ---
    n100_subjects: dict[str, set[str]] = defaultdict(set)
    for e in all_examples:
        if e["subset"] == "N100" and e["label"] == 1:
            n100_subjects[e["relation_id"]].add(e["subject"])
    cross_check_errors = []
    for i, r1 in enumerate(RELATIONS):
        for r2 in RELATIONS[i + 1:]:
            overlap = n100_subjects[r1] & n100_subjects[r2]
            if overlap:
                cross_check_errors.append(f"{r1}+{r2}: {len(overlap)} shared subjects")
    if cross_check_errors:
        print(f"\nCROSS-RELATION SUBJECT LEAK: {cross_check_errors}")
    else:
        print("\nCross-relation subject check on N100: OK (no overlap)")

    # --- Build manifest ---
    manifest: dict[str, Any] = {
        "seed": SEED,
        "relations": RELATIONS,
        "subset_sizes": {name: size for name, size in SUBSET_SIZES},
        "p103_french_caps": P103_FRENCH_CAPS,
        "cleaning_stages": {
            "stage1_raw": s1,
            "stage2_subject_dedup": s2,
            "stage3_cross_relation": s3,
            "stage4_sentence_dedup": s4,
        },
        "cross_relation_subjects": sorted(cross_subjs),
        "total_examples": len(all_examples),
        "examples_per_subset": {
            name: sum(1 for e in all_examples if e["subset"] == name)
            for name, _ in SUBSET_SIZES
        },
        "validation": {
            "all_subset_checks_passed": all(v["valid"] for v in all_validations),
            "nesting_checks_passed": len(nesting_errors) == 0,
            "cross_relation_check_passed": len(cross_check_errors) == 0,
            "checks_passed": sum(1 for v in all_validations if v["valid"]),
            "checks_total": len(all_validations),
        },
        "raw_data_hash": sha256_file(RAW_PATH),
    }

    save_outputs(all_examples, relation_meta, manifest)

    # --- Final report ---
    all_passed = (
        all(v["valid"] for v in all_validations)
        and len(nesting_errors) == 0
        and len(cross_check_errors) == 0
    )
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for v in all_validations:
        status = "PASS" if v["valid"] else f"FAIL: {v['errors']}"
        print(f"  {v['relation_id']} {v['subset']}: {status}")
    if nesting_errors:
        for err in nesting_errors:
            print(f"  NESTING: {err}")
    print(f"\n{'ALL CHECKS PASSED' if all_passed else 'FAILURES DETECTED'}")

    if not all_passed:
        raise RuntimeError("Validation failures — see output above")


if __name__ == "__main__":
    main()
