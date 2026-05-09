#!/usr/bin/env bash
# ---------------------------------------------------------------
# Dr(eye)ve end-to-end training workflow
#
# Step 1: Prepare dataset  (runs YOLO offline on all sequences)
# Step 2: Train the model
# Step 3: Evaluate on same data (sanity check)
#
# Edit DATASET_ROOT and OUTPUT_JSON to match your paths.
# ---------------------------------------------------------------

set -e

DATASET_ROOT="D:/TW/dr(eye)ve"
OUTPUT_JSON="data/dreyeve_train.json"
MODEL_OUT="models/intent_dreyeve.pth"

# ── Step 1: Prepare ─────────────────────────────────────────────
echo "=== Step 1: Preparing dr(eye)ve dataset ==="
python scripts/prepare_dreyeve.py \
    --dataset   "$DATASET_ROOT" \
    --output    "$OUTPUT_JSON" \
    --frame-skip 1 \
    --min-frames 30

# ── Step 2: Train ───────────────────────────────────────────────
echo ""
echo "=== Step 2: Training intent model ==="
python scripts/train_intent.py train \
    --data      "$OUTPUT_JSON" \
    --output    "$MODEL_OUT" \
    --epochs    60 \
    --batch     32 \
    --lr        1e-3 \
    --seq-len   90 \
    --stride    15 \
    --early-stop 15

# ── Step 3: Evaluate ────────────────────────────────────────────
echo ""
echo "=== Step 3: Evaluating ==="
python scripts/train_intent.py evaluate \
    --data   "$OUTPUT_JSON" \
    --output "$MODEL_OUT" \
    --seq-len 90

echo ""
echo "Done. Model saved to $MODEL_OUT"
echo "Run inference with:"
echo "  python inference.py --driver 0 --scene 1 --model $MODEL_OUT"
