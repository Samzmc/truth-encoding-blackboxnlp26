# Emergence of Linear Truth Encodings — Reproduction, Ablation, and Scaling Study

Code and key results for a BlackboxNLP 2026 reproducibility submission targeting
Ravfogel et al., *"Emergence of Linear Truth Encodings in Language Models"*
(NeurIPS 2025, [arXiv:2510.15804](https://arxiv.org/abs/2510.15804)).

The study has three experiments, in the same order as the paper:

| Experiment | Directory | Question |
|---|---|---|
| 1 | `experiment1_toy_ablation/` | Does the two-block OV structure survive when the copy circuit is ablated? (toy + fully-trained 1-layer transformers) |
| 2 | `experiment2_context_effects/` | Do behavioral context effects (ΔNLL under true/false context) transfer across relations in Llama-3-8B / 70B? |
| 3 | `experiment3_scaling_probes/` | Do linear truth probes transfer across relations at the representational level? (CounterFact, 6 relations) |

`figures/` contains all paper figures (pre-generated PNG + PDF) and the script
that regenerates them.

> **Naming note.** Inside `experiment1_toy_ablation/`, the run directories
> `phase0_original/` (unablated baseline) and `phase1_ablation/` (ablated
> variant) are internal run labels kept for consistency with the full data
> release; they are unrelated to the experiment numbering above. Likewise,
> internal script comments may refer to the historical stage names of the
> pipeline.

## Environment

Two environments are used:

- **General** (experiment 1 and figure generation): Python ≥ 3.10, install
  PyTorch for your platform first, then `pip install -r requirements.txt`.
- **NDIF/nnsight** (experiments 2–3 remote extraction): same requirements, plus
  an [NDIF](https://ndif.us/) API key. Create `experiment3_scaling_probes/.env`
  with:

  ```
  NDIF_API_KEY=...
  HF_TOKEN=...          # gated Llama access
  ```

  `.env` is git-ignored; never commit it.
  `experiment3_scaling_probes/ndif_smoke_test.py` verifies the setup.

Note: experiment 2 reads the CounterFact data prepared by the experiment 3
pipeline (`prepare_counterfact_*.py`), so run those preparation steps first
even if you only want the behavioral results.

## Experiment 1 — toy & fully-trained ablation

Training/analysis code at the top level; `phase0_original/` and
`phase1_ablation/` hold the run scripts for the unablated baseline and the
ablated variant respectively (5 seeds toy, 3–5 seeds fully-trained):

```
cd experiment1_toy_ablation
bash phase0_original/run_toy.sh          # baseline toy, 5 seeds
bash phase1_ablation/run_toy.sh          # ablated toy, 5 seeds
bash run_overnight_rho08.sh              # fully-trained ρ=0.8, normal + ablated
python compare_toy_auc.py                # summary comparisons
python compare_fulltrained.py
python cosine_similarity_toy.py
```

Per-run outputs (checkpoints, per-step metrics) are written next to the run
scripts; the shipped `comparison_figures_*/` directories contain the summary
outputs used in the paper.

## Experiment 2 — behavioral context effects

```
cd experiment2_context_effects
python build_context_prompts.py          # freeze Stage A/B prompt sets
python extract_context_probs.py --stage a --model 8b    # (and b / 70b)
python analyze_context_effects.py        # NLL table, Fig. 6a repro, 6x6 grid
```

`run_overnight.ps1` drives the full Stage B extraction (resumable). Key outputs
in `results/`: `grid_matrix_{8B,70B}.csv` (6×6 median paired ΔNLL),
`grid_cells_*.csv` (bootstrap CIs, Wilcoxon p), `stage_a_summary_*.csv`.

## Experiment 3 — scaling probes (Llama-3-8B / 70B)

Pipeline order:

```
cd experiment3_scaling_probes
python prepare_counterfact_p103.py           # balanced true/false P103 set
python prepare_counterfact_multirelation.py  # extend to 6 relations
python generate_splits.py                    # pair-grouped 3-fold CV splits
python extract_activations.py --pilot        # NDIF extraction (then full runs)
python run_probing.py                        # Stage 1A/1B, layer selection, Stage 2
python generate_figures.py
```

Key outputs in `results/`: `stage1_metrics.csv`, `stage2_matrix_{8b,70b}.csv`
(6×6 probe-transfer AUC), `generality_gap.csv`, `table1_relation_results.csv`.

## Figures

See `figures/README.md` for the figure-by-figure provenance table. Regenerating
figures requires the full per-run outputs.

## Data release

Heavy artifacts (model checkpoints, extracted activations, per-run outputs) are
too large for this repository. An archive will be linked here upon publication
(link withheld for anonymity during review).
