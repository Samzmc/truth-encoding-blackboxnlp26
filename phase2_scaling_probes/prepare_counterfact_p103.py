"""Create a controlled, balanced true/false P103 dataset from CounterFact.

French occurs in more than half of raw P103 records, making a frequency-
preserving derangement of all P103 examples mathematically impossible. We
therefore retain every non-French record and deterministically sample an equal
number of French records. False attributes then exchange the French and
non-French pools. Every attribute has exactly the same frequency in both
labels, and no false sentence retains its own true attribute. This prevents a
linear probe from succeeding merely because a language name is more frequent
in one label.

Run from any directory:
    python prepare_counterfact_p103.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SEED = 20260711
RELATION_ID = "P103"
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "counterfact.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "counterfact_p103_balanced_pairs.jsonl"
SUMMARY_PATH = ROOT / "data" / "processed" / "counterfact_p103_balanced_summary.json"


def make_sentence(template: str, subject: str, attribute: str) -> str:
    """Fill CounterFact's subject placeholder and add exactly one word space."""
    return f"{template.format(subject).rstrip()} {attribute.strip()}"


def load_p103_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw CounterFact file is missing: {path}\n"
            "Download it before running this preparation script."
        )

    with path.open("r", encoding="utf-8") as handle:
        raw_records = json.load(handle)

    records = [
        record
        for record in raw_records
        if record["requested_rewrite"]["relation_id"] == RELATION_ID
    ]
    if not records:
        raise RuntimeError(f"No records found for relation {RELATION_ID}.")
    return records


def select_balanced_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, int]:
    """Balance the dominant attribute against the aggregate of all others."""
    attributes = [record["requested_rewrite"]["target_true"]["str"] for record in records]
    dominant_attribute, dominant_count = Counter(attributes).most_common(1)[0]
    dominant_records = [
        record
        for record in records
        if record["requested_rewrite"]["target_true"]["str"] == dominant_attribute
    ]
    other_records = [record for record in records if record not in dominant_records]

    if not other_records:
        raise RuntimeError("P103 has no non-dominant attributes to construct negatives from.")

    rng = random.Random(SEED)
    sampled_dominant = rng.sample(dominant_records, len(other_records))
    selected = sorted(sampled_dominant + other_records, key=lambda record: record["case_id"])
    excluded = dominant_count - len(sampled_dominant)
    return selected, dominant_attribute, excluded


def build_examples(records: list[dict[str, Any]], dominant_attribute: str) -> list[dict[str, Any]]:
    dominant_records = [
        record
        for record in records
        if record["requested_rewrite"]["target_true"]["str"] == dominant_attribute
    ]
    other_records = [record for record in records if record not in dominant_records]

    false_by_case_id: dict[int, str] = {}
    other_attributes = [record["requested_rewrite"]["target_true"]["str"] for record in other_records]
    random.Random(SEED).shuffle(other_attributes)
    for record, false_attribute in zip(dominant_records, other_attributes):
        false_by_case_id[record["case_id"]] = false_attribute
    for record in other_records:
        false_by_case_id[record["case_id"]] = dominant_attribute

    examples: list[dict[str, Any]] = []
    for record in records:
        rewrite = record["requested_rewrite"]
        case_id = record["case_id"]
        subject = rewrite["subject"]
        template = rewrite["prompt"]
        true_attribute = rewrite["target_true"]["str"]
        false_attribute = false_by_case_id[case_id]

        common = {
            "pair_id": case_id,
            "case_id": case_id,
            "relation_id": rewrite["relation_id"],
            "subject": subject,
            "template": template,
            "target_true": true_attribute,
            "target_false": false_attribute,
            "construction": "P103_balanced_dominant_attribute_exchange",
        }
        examples.append(
            {
                **common,
                "example_id": f"{case_id}_true",
                "label": 1,
                "sentence": make_sentence(template, subject, true_attribute),
            }
        )
        examples.append(
            {
                **common,
                "example_id": f"{case_id}_false",
                "label": 0,
                "sentence": make_sentence(template, subject, false_attribute),
            }
        )

    return examples


def validate(
    source_records: int,
    records: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    dominant_attribute: str,
    excluded_records: int,
) -> dict[str, Any]:
    true_examples = [example for example in examples if example["label"] == 1]
    false_examples = [example for example in examples if example["label"] == 0]

    if len(examples) != 2 * len(records):
        raise AssertionError("Each source record must create exactly two examples.")
    if len(true_examples) != len(false_examples):
        raise AssertionError("The labels are not balanced.")
    if any(example["target_true"] == example["target_false"] for example in examples):
        raise AssertionError("A false example retained its true attribute.")
    if Counter(example["target_true"] for example in true_examples) != Counter(
        example["target_false"] for example in false_examples
    ):
        raise AssertionError("True and false attribute frequencies are not balanced.")

    return {
        "source": str(RAW_PATH),
        "relation_id": RELATION_ID,
        "seed": SEED,
        "source_records": source_records,
        "selected_records": len(records),
        "excluded_records": excluded_records,
        "dominant_attribute": dominant_attribute,
        "pairs": len(records),
        "examples": len(examples),
        "true_examples": len(true_examples),
        "false_examples": len(false_examples),
        "unique_attributes": len({example["target_true"] for example in true_examples}),
        "validation": {
            "labels_balanced": True,
            "no_false_attribute_equals_its_true_attribute": True,
            "true_false_attribute_frequencies_match": True,
        },
    }


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    source_records = load_p103_records(RAW_PATH)
    records, dominant_attribute, excluded_records = select_balanced_records(source_records)
    examples = build_examples(records, dominant_attribute)
    summary = validate(
        len(source_records), records, examples, dominant_attribute, excluded_records
    )
    write_jsonl(OUTPUT_PATH, examples)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("P103 CounterFact preparation completed.")
    print(
        f"Source P103 records: {summary['source_records']}; "
        f"balanced pairs: {summary['pairs']}; examples: {summary['examples']}"
    )
    print(
        f"Balanced dominant attribute: {summary['dominant_attribute']}; "
        f"excluded source records: {summary['excluded_records']}"
    )
    print(f"Output: {OUTPUT_PATH}")
    print("\nFirst five true/false pairs:")
    for index in range(0, min(10, len(examples)), 2):
        true_example, false_example = examples[index], examples[index + 1]
        print(f"\nPair {true_example['pair_id']}")
        print(f"  TRUE : {true_example['sentence']}")
        print(f"  FALSE: {false_example['sentence']}")


if __name__ == "__main__":
    main()
