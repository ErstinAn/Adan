#!/usr/bin/env bash
set -euo pipefail

# ─── Usage ───────────────────────────────────────────────────────────────────
# bash train_script.sh [model] [optimizer]
#   model:     vit-s | resnet50
#   optimizer: adan | adamw | adanNC
# Example:
#   bash train_script.sh vit-s adan
#   bash train_script.sh resnet50 adamw
# ─────────────────────────────────────────────────────────────────────────────

MODEL=${1:?Usage: bash train_script.sh [vit-s|resnet50] [adan|adamw|adanNC]}
OPTIMIZER=${2:?Usage: bash train_script.sh [vit-s|resnet50] [adan|adamw|adanNC]}

# ─── Common config ───────────────────────────────────────────────────────────
DATA_DIR="/mnt/disks/local/ka56_workspace/data"
OUTPUT_DIR="/mnt/disks/local/ka56_workspace/adan"
GPUS="4,5"
NPROC=2
EPOCHS=150
WANDB_ENTITY="PINN_PtB"
WANDB_PROJECT="adan-optimizer-bench"

# ─── Optimizer-specific betas ────────────────────────────────────────────────
if [ "$OPTIMIZER" = "adan" ]; then
    OPT_FLAG="adan"
    OPT_BETAS="0.98 0.92 0.99"
elif [ "$OPTIMIZER" = "adamw" ]; then
    OPT_FLAG="adamw"
    OPT_BETAS="0.9 0.999"
elif [ "$OPTIMIZER" = "adanNC" ]; then
    OPT_FLAG="adanNC"
    OPT_BETAS="0.9 0.999 0.9"
    OPT_ALPHA="1.0"
else
    echo "Error: optimizer must be 'adan', 'adamw', or 'adanNC', got '$OPTIMIZER'"
    exit 1
fi

# ─── Model-specific hyperparams ─────────────────────────────────────────────
if [ "$MODEL" = "vit-s" ]; then
    MODEL_NAME="deit_small_patch16_224"
    RUN_NAME="vit-s-${OPTIMIZER}-ep${EPOCHS}"
    MODEL_ARGS=(
        --model "$MODEL_NAME"
        --lr 1.5e-2
        --weight-decay 0.02
        --opt-eps 1e-8
        --warmup-epochs 60
        --warmup-lr 1e-8
        --min-lr 1e-5
        --drop-path 0.1
        --mixup 0.8
        --cutmix 1.0
        --bce-loss
        --smoothing 0.1
        --reprob 0.25
        --aa rand-m7-mstd0.5-inc1
        --amp
        --batch-size 1024
    )
elif [ "$MODEL" = "resnet50" ]; then
    MODEL_NAME="resnet50"
    RUN_NAME="resnet50-${OPTIMIZER}-ep${EPOCHS}"
    MODEL_ARGS=(
        --model "$MODEL_NAME"
        --lr 1.5e-2
        --weight-decay 0.02
        --opt-eps 1e-8
        --max-grad-norm 5.0
        --warmup-epochs 60
        --warmup-lr 1e-9
        --min-lr 1e-5
        --bias-decay
        --crop-pct 0.95
        --drop-path 0.05
        --mixup 0.1
        --cutmix 1.0
        --bce-loss
        --smoothing 0.0
        --reprob 0.0
        --aa rand-m7-mstd0.5-inc1
        --amp
        --batch-size 1024
    )
else
    echo "Error: model must be 'vit-s' or 'resnet50', got '$MODEL'"
    exit 1
fi

# ─── Launch training ─────────────────────────────────────────────────────────
# ─── Optional extra args (e.g. --alpha for adanNC) ─────────────────────────
EXTRA_ARGS=()
if [ -n "${OPT_ALPHA:-}" ]; then
    EXTRA_ARGS+=(--alpha "$OPT_ALPHA")
fi

echo "==> Training ${MODEL_NAME} with ${OPT_FLAG} for ${EPOCHS} epochs"
echo "==> Run name: ${RUN_NAME}"
echo "==> GPUs: ${GPUS} (nproc=${NPROC})"

CUDA_VISIBLE_DEVICES=$GPUS \
python -m torch.distributed.launch \
    --nproc_per_node=$NPROC \
    train.py \
    "${MODEL_ARGS[@]}" \
    --opt "$OPT_FLAG" \
    --opt-betas $OPT_BETAS \
    --epochs "$EPOCHS" \
    --experiment "$RUN_NAME" \
    --output "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --log-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    "${EXTRA_ARGS[@]}"
