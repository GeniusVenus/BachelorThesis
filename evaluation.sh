#!/bin/bash

# Define arrays for all options
MODELS=("deeplabv3plus" "pspnet" "fpn" "unet" "segformer" "upernet")
LOSS_FUNCTIONS=("cross_entropy" "dice" "focal" "jaccard" "lovasz" "tversky")
DATASETS=("naver")
BATCH_SIZE=32

# Define log directory and file explicitly
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/evaluation_runs.log"

# Create log directory and file with error checking
echo "Creating log directory at: $LOG_DIR"
if ! mkdir -p "$LOG_DIR"; then
    echo "ERROR: Failed to create log directory at $LOG_DIR"
    echo "Please check permissions or create it manually"
    exit 1
fi

# Function to determine backbone based on model
get_backbone_info() {
    local model=$1

    if [[ "$model" == "segformer" ]]; then
        # For SegFormer, use NVIDIA MIT-b3, but checkpoint name shows mit_b3
        BACKBONE_PARAM="nvidia/mit-b3"
        BACKBONE_NAME="mit_b3"
    elif [[ "$model" == "upernet" ]]; then
        # For UperNet, use openmmlab/upernet-convnext-base, but checkpoint name shows convnext_base
        BACKBONE_PARAM="openmmlab/upernet-convnext-base"
        BACKBONE_NAME="convnext_base"
    else
        # For other models (deeplabv3plus, unet, pspnet, fpn, etc.), use resnet50 for both
        BACKBONE_PARAM="resnet50"
        BACKBONE_NAME="resnet50"
    fi
}

# Track total and completed runs
total_combinations=0
completed_combinations=0

# Count combinations
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for LOSS in "${LOSS_FUNCTIONS[@]}"; do
            total_combinations=$((total_combinations + 1))
        done
    done
done

echo "Planning to evaluate $total_combinations combinations" | tee -a "$LOG_FILE"

# Nested loops with dataset as outermost loop
for DATASET in "${DATASETS[@]}"; do
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Processing dataset: $DATASET" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    for MODEL in "${MODELS[@]}"; do
        # Get backbone info for this model
        get_backbone_info "$MODEL"

        echo "Using model: $MODEL with backbone: $BACKBONE_PARAM" | tee -a "$LOG_FILE"

        for LOSS in "${LOSS_FUNCTIONS[@]}"; do
            # Generate checkpoint name using the display name for the backbone
            CHECKPOINT="${DATASET}_${LOSS}_${BACKBONE_NAME}_${MODEL}"

            # Log the current combination
            echo "----------------------------------------" | tee -a "$LOG_FILE"
            echo "Evaluating $CHECKPOINT ($(($completed_combinations + 1))/$total_combinations)" | tee -a "$LOG_FILE"
            echo "Using dataset=$DATASET, model=$MODEL, backbone=$BACKBONE_PARAM, loss=$LOSS" | tee -a "$LOG_FILE"
            echo "Started at $(date)" | tee -a "$LOG_FILE"

            # Check if checkpoint exists
            if [ ! -f "checkpoints/${CHECKPOINT}.pth" ]; then
                echo "Checkpoint not found: checkpoints/${CHECKPOINT}.pth" | tee -a "$LOG_FILE"
                echo "Skipping evaluation..." | tee -a "$LOG_FILE"
                completed_combinations=$((completed_combinations + 1))
                continue
            fi

            # Run the evaluation command
            python scripts/evaluation.py \
                --model $MODEL \
                --backbone "$BACKBONE_PARAM" \
                --loss $LOSS \
                --checkpoint $CHECKPOINT \
                --dataset $DATASET | tee -a "$LOG_FILE"

            # Check if evaluation was successful
            if [ $? -eq 0 ]; then
                echo "Successfully evaluated $CHECKPOINT at $(date)" | tee -a "$LOG_FILE"
            else
                echo "Failed to evaluate $CHECKPOINT at $(date)" | tee -a "$LOG_FILE"
            fi

            completed_combinations=$((completed_combinations + 1))

            # Wait for a moment to allow system resources to stabilize
            sleep 5
        done
    done
done

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Evaluation completed at $(date)" | tee -a "$LOG_FILE"
echo "Total combinations evaluated: $completed_combinations/$total_combinations" | tee -a "$LOG_FILE"
