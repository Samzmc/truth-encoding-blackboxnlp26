# Paper figures

Pre-generated figures used in the paper. `make_figures.py` regenerates them
from the full experiment outputs (per-run checkpoints and metrics, not shipped
in this repo — see the data release note in the top-level README). Point
`REPRO_DIR` at the root of those outputs:

```
REPRO_DIR=/path/to/reproduction python make_figures.py
# or a subset:
REPRO_DIR=/path/to/reproduction python make_figures.py --only B1 D6
```

`final/` holds the versions used in the paper: PNG + PDF without figure-level
titles (panel identifiers such as model names are kept; everything else goes
into captions). The script can additionally emit titled draft versions into
`draft/`, which are not shipped here.

Blocks: A = reproduction, B = ablation (experiment 1), C = behavioral
(experiment 2), D = representational (experiment 3).

Paths in the *Data source* column refer to the full-output layout under
`REPRO_DIR` (the data release), which keeps the historical internal directory
names (`phase0_original`, `phase2`, `phase3`, …) — see the naming note in the
top-level README.

| Figure | Content | Data source |
|---|---|---|
| A1_toy_ov_heatmap | W = WoWv, toy normal seed 0, step 2000 (paper Fig. 1 repro) | ablation/phase0_original/toy_rho08_seed0 |
| B1_ablation_auc_triptych | Probe AUC vs step, normal vs ablated: full ρ=0.99 (n=5), full ρ=0.8 (n=3), toy ρ=0.8 (n=5) | auc_curve_data.json, all runs |
| B2_toy_ov_heatmaps_pair | Toy W normal vs ablated, shared color scale | ov_matrices, seed 0 |
| B3_ov_cosine_4panel | OV sub-block cosine similarity (orientation preserved) | cosine_similarity.json, 5+5 seeds |
| B4_prob_comparison_rho08 | P(correct\|true/false seq.) over training | prob_curve_data.json, ρ=0.8 |
| B5_ov_amplitude | Signed target projection a_B + Frobenius norm at step 2000 (7–10× collapse) | ov_matrices, 5+5 seeds |
| C1_fig6a_combined | Paper Fig. 6a reproduction, 8B + 70B box plots | phase3/results/context_nll.parquet |
| C2_behavioral_grid | 6×6 behavioral ΔNLL heatmaps | phase3/results/grid_matrix_*.csv |
| D1_depth_profile | Within vs leave-one-out AUC by depth | phase2/results/generality_gap.csv |
| D2_probe_transfer_matrices | 6×6 probe transfer AUC matrices | phase2/results/stage2_matrix_*.csv |
| D3_generality_gap | Mean \|generality gap\| bars by depth | phase2/results/generality_gap.csv |
| D4_relation_profiles | Per-relation AUC small multiples | phase2/results/generality_gap.csv |
| D5_relation_gap_detail | Per-relation gap at selected layer | phase2/results/generality_gap.csv |
| D6_behavioral_vs_probe_scatter | 36-cell scatter linking C2 and D2 (Spearman null) | grid_matrix_*.csv + stage2_matrix_*.csv |
