#!/bin/bash
## Phase 1.3 — Fully-trained model ablation (independent truth)
set -e
cd "$(dirname "$0")/.."

PY="conda run -n blackboxnlp --no-capture-output python"
OUT="phase1_ablation"

COMMON="--N 512 --d_model 256 --num_heads 1 \
  --p_train 0.99 --lr 1e-3 --weight_decay 1e-5 --batch_size 128 \
  --num_steps 50000 --save_interval 1000 \
  --early_save_interval 200 --early_save_steps 1000 \
  --print_interval 1000 --use_rms \
  --independent_truth"

# ── 1. Train: 1 layer × 5 seeds ──
for seed in 0 1 2 3 4; do
    out="$OUT/L1_seed${seed}"
    echo ""
    echo "===== Training L=1 ablated seed=${seed} -> ${out} ====="
    $PY train_model.py $COMMON --num_layers 1 --seed $seed --output_path "$out"
done

# ── 2. Train: 3 layers × 5 seeds ──
for seed in 0 1 2 3 4; do
    out="$OUT/L3_seed${seed}"
    echo ""
    echo "===== Training L=3 ablated seed=${seed} -> ${out} ====="
    $PY train_model.py $COMMON --num_layers 3 --seed $seed --output_path "$out"
done

# ── 3. Analyze each run ──
for layers in 1 3; do
    for seed in 0 1 2 3 4; do
        ckpt="$OUT/L${layers}_seed${seed}/checkpoints/ckpt_step50000.pt"
        echo ""
        echo "===== Analyzing L=${layers} ablated seed=${seed} ====="
        $PY analyze_model.py --checkpoint "$ckpt" --all
    done
done

echo ""
echo "===== ALL DONE ====="
