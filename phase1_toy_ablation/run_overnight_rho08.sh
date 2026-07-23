#!/bin/bash
## Overnight: ρ=0.8 fully-trained model, normal + ablated, 3 seeds, L1 only
set -e
cd "$(dirname "$0")"

PY="conda run -n blackboxnlp --no-capture-output python"

COMMON="--N 512 --d_model 256 --num_heads 1 --num_layers 1 \
  --p_train 0.8 --lr 1e-3 --weight_decay 1e-5 --batch_size 128 \
  --num_steps 50000 --save_interval 1000 \
  --early_save_interval 200 --early_save_steps 1000 \
  --print_interval 1000 --use_rms"

echo "============================================"
echo " ρ=0.8 Overnight Run — $(date)"
echo "============================================"

# ── 1. Phase 0: Normal, L1 × 3 seeds ──
for seed in 0 1 2; do
    out="phase0_original/L1_rho08_seed${seed}"
    echo ""
    echo "===== [Phase0] Training L=1 rho=0.8 seed=${seed} -> ${out} ====="
    $PY train_model.py $COMMON --seed $seed --output_path "$out"
done

# ── 2. Phase 1: Ablated, L1 × 3 seeds ──
for seed in 0 1 2; do
    out="phase1_ablation/L1_rho08_seed${seed}"
    echo ""
    echo "===== [Phase1] Training L=1 rho=0.8 ablated seed=${seed} -> ${out} ====="
    $PY train_model.py $COMMON --independent_truth --seed $seed --output_path "$out"
done

# ── 3. Analyze all 6 runs ──
for seed in 0 1 2; do
    ckpt="phase0_original/L1_rho08_seed${seed}/checkpoints/ckpt_step50000.pt"
    echo ""
    echo "===== Analyzing Phase0 rho=0.8 seed=${seed} ====="
    $PY analyze_model.py --checkpoint "$ckpt" --all
done

for seed in 0 1 2; do
    ckpt="phase1_ablation/L1_rho08_seed${seed}/checkpoints/ckpt_step50000.pt"
    echo ""
    echo "===== Analyzing Phase1 rho=0.8 ablated seed=${seed} ====="
    $PY analyze_model.py --checkpoint "$ckpt" --all
done

# ── 4. Generate comparison figures ──
OUTDIR="comparison_figures_fulltrained_rho08"
mkdir -p "$OUTDIR"

echo ""
echo "===== Generating comparison figures (ρ=0.8) ====="
$PY compare_fulltrained.py \
    --normal_dirs \
        phase0_original/L1_rho08_seed0/analysis \
        phase0_original/L1_rho08_seed1/analysis \
        phase0_original/L1_rho08_seed2/analysis \
    --ablated_dirs \
        phase1_ablation/L1_rho08_seed0/analysis \
        phase1_ablation/L1_rho08_seed1/analysis \
        phase1_ablation/L1_rho08_seed2/analysis \
    --output_path "$OUTDIR" \
    --label "L1_rho08" \
    --num_layers 1

echo ""
echo "============================================"
echo " ALL DONE — $(date)"
echo "============================================"
