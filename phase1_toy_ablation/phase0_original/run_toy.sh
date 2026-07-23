#!/bin/bash
## Phase 0.2 — Toy model reproduction (one-hot, uniform attention, 5 seeds)
set -e
cd "$(dirname "$0")/.."

PY="conda run -n blackboxnlp --no-capture-output python"
OUT="phase0_original"

for seed in 0 1 2 3 4; do
    out="$OUT/toy_seed${seed}"
    echo ""
    echo "===== Training toy model seed=${seed} -> ${out} ====="
    $PY train_toy.py --one_hot --freeze_embeddings \
        --num_steps 2000 --seed $seed --save_interval 50 --print_interval 100 \
        --output_path "$out"
done

for seed in 0 1 2 3 4; do
    echo ""
    echo "===== Analyzing toy model seed=${seed} ====="
    $PY analyze_toy.py \
        --checkpoint "$OUT/toy_seed${seed}/checkpoints/ckpt_step2000.pt" --all
done

echo ""
echo "===== ALL DONE ====="
