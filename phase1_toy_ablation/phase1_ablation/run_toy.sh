#!/bin/bash
## Phase 1.2 — Toy model ablation (independent truth)
set -e
cd "$(dirname "$0")/.."

PY="conda run -n blackboxnlp --no-capture-output python"
OUT="phase1_ablation"

for seed in 0 1 2 3 4; do
    out="$OUT/toy_seed${seed}"
    echo ""
    echo "===== Training toy ablated seed=${seed} -> ${out} ====="
    $PY train_toy.py --one_hot --freeze_embeddings \
        --independent_truth \
        --num_steps 2000 --seed $seed --save_interval 50 --print_interval 100 \
        --output_path "$out"
done

for seed in 0 1 2 3 4; do
    echo ""
    echo "===== Analyzing toy ablated seed=${seed} ====="
    $PY analyze_toy.py \
        --checkpoint "$OUT/toy_seed${seed}/checkpoints/ckpt_step2000.pt" --all
done

echo ""
echo "===== ALL DONE ====="
