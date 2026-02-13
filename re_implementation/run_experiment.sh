#!/usr/bin/env bash
# ==================================================================
#  run_experiment.sh — Full training + evaluation pipeline
#
#  Hyperparameters chosen to match the reference paper:
#    emb_dim=128, hidden=256, 1 layer, 10 epochs, lr=0.001, clip=5
#  with our SentencePiece BPE tokenizer (32K shared vocab).
#
#  Run with:  screen -S experiment bash run_experiment.sh
# ==================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────
DATA_DIR="../data"
SAVE_DIR="./checkpoints"
OUTPUT_DIR="./testing_res"
LOG_FILE="./experiment.log"
SP_DIR="${SAVE_DIR}/sp"

# Hyperparameters (matching reference paper)
VOCAB_SIZE=32000
EMB_DIM=128
HIDDEN_SIZE=256
N_LAYERS=1
DROPOUT=0.1

# Training schedule
EPOCHS=10
BATCH_SIZE=32
ACCUM_STEPS=4          # effective batch = 32 × 4 = 128
LR=0.001
CLIP=5.0
PATIENCE=5

# Logging / checkpointing
LOG_EVERY=500
EVAL_EVERY=5000        # validate ~every 5K steps
SAVE_EVERY=10000       # numbered checkpoint every 10K steps

# Evaluation
BEAM_SIZE=5
MAX_LEN=100

SEED=42

# ── Helpers ───────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_FILE"; }

# ── Start ─────────────────────────────────────────────────────
echo "" >> "$LOG_FILE"
log "============================================================"
log "  EXPERIMENT START"
log "============================================================"
log "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "  Python: $(python3 --version 2>&1)"
log "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>&1)"
log ""

# ── Phase 1: Training ────────────────────────────────────────
log ">>> PHASE 1: Training"
log "  Hyperparameters:"
log "    emb_dim=${EMB_DIM}  hidden=${HIDDEN_SIZE}  layers=${N_LAYERS}  dropout=${DROPOUT}"
log "    epochs=${EPOCHS}  batch_size=${BATCH_SIZE}  accum=${ACCUM_STEPS}  eff_batch=$((BATCH_SIZE * ACCUM_STEPS))"
log "    lr=${LR}  clip=${CLIP}  patience=${PATIENCE}"
log "    vocab_size=${VOCAB_SIZE} (SentencePiece BPE)"
log ""

python3 train.py \
    --data-dir "$DATA_DIR" \
    --save-dir "$SAVE_DIR" \
    --sp-dir "$SP_DIR" \
    --vocab-size "$VOCAB_SIZE" \
    --emb-dim "$EMB_DIM" \
    --hidden-size "$HIDDEN_SIZE" \
    --n-layers "$N_LAYERS" \
    --dropout "$DROPOUT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --accum-steps "$ACCUM_STEPS" \
    --lr "$LR" \
    --clip "$CLIP" \
    --patience "$PATIENCE" \
    --eval-every "$EVAL_EVERY" \
    --save-every "$SAVE_EVERY" \
    --log-every "$LOG_EVERY" \
    --amp \
    --num-workers 4 \
    --seed "$SEED" \
    2>&1 | tee -a "$LOG_FILE"

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    log "!!! Training FAILED with exit code $TRAIN_EXIT"
    exit 1
fi
log ""
log ">>> Training complete."
log ""

# ── Phase 2: Evaluation ──────────────────────────────────────
log ">>> PHASE 2: Evaluation (beam search, beam_size=${BEAM_SIZE})"
log ""

python3 evaluate.py \
    --data-dir "$DATA_DIR" \
    --checkpoint "${SAVE_DIR}/best.pt" \
    --sp-dir "$SP_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --beam-size "$BEAM_SIZE" \
    --max-len "$MAX_LEN" \
    --batch-size 64 \
    --seed "$SEED" \
    2>&1 | tee -a "$LOG_FILE"

EVAL_EXIT=$?
if [ $EVAL_EXIT -ne 0 ]; then
    log "!!! Evaluation FAILED with exit code $EVAL_EXIT"
    exit 1
fi

log ""
log "============================================================"
log "  EXPERIMENT COMPLETE"
log "============================================================"
log "  Checkpoints : ${SAVE_DIR}/"
log "  Results     : ${OUTPUT_DIR}/"
log "  Log         : ${LOG_FILE}"
log "============================================================"
log ""
log "Files produced:"
ls -lh "${OUTPUT_DIR}/" 2>/dev/null | tee -a "$LOG_FILE"
log ""
log ">>> ALL DONE. Safe to close screen session."
