#!/bin/bash

# Define arrays for all options
MODELS=("upernet")
LOSS_FUNCTIONS=("cross_entropy" "dice" "focal" "jaccard" "lovasz" "tversky")
DATASETS=("open_earth_map")
EPOCHS=50
BATCH_SIZE=16

# Define log directory and file explicitly
LOG_DIR="./logs"  # Current directory/logs
LOG_FILE="$LOG_DIR/training_runs.log"

# Create log directory and file with error checking
echo "Creating log directory at: $LOG_DIR"
if ! mkdir -p "$LOG_DIR"; then
    echo "ERROR: Failed to create log directory at $LOG_DIR"
    echo "Please check permissions or create it manually"
    exit 1
fi

echo "Creating log file at: $LOG_FILE"
if ! touch "$LOG_FILE"; then
    echo "ERROR: Failed to create log file at $LOG_FILE"
    echo "Please check permissions or create it manually"
    exit 1
fi

echo "Log file created successfully at: $LOG_FILE"

# Log start time
echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Training started at $(date)" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Function to determine backbone based on model
get_backbone_info() {
    local model=$1
    local backbone_override=$2

    if [[ "$model" == "segformer" ]]; then
        # For SegFormer, use NVIDIA MIT-b3, but checkpoint name shows mit_b3
        BACKBONE_PARAM="nvidia/mit-b3"
        BACKBONE_NAME="mit_b3"
    elif [[ "$model" == "upernet" ]]; then
        # For UperNet, use openmmlab/upernet-convnext-base, but checkpoint name shows convnext_base
        BACKBONE_PARAM="openmmlab/upernet-convnext-base"
        BACKBONE_NAME="convnext_base"
    elif [[ "$model" == "unet" ]]; then
        # For UNet, use specified backbone or default to resnet50
        if [[ -n "$backbone_override" ]]; then
            BACKBONE_PARAM="$backbone_override"
            BACKBONE_NAME="$backbone_override"
        else
            BACKBONE_PARAM="resnet50"
            BACKBONE_NAME="resnet50"
        fi
    else
        # For other models (deeplabv3plus, pspnet, fpn, etc.), use resnet50 for both
        BACKBONE_PARAM="resnet50"
        BACKBONE_NAME="resnet50"
    fi
}

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Track total and completed runs
total_combinations=0
completed_combinations=0
skipped_combinations=0

# Count combinations
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        if [[ "$MODEL" == "unet" ]]; then
            # UNet has two backbones: vgg11 and resnet50
            for BACKBONE in "vgg11" "resnet50"; do
                for LOSS in "${LOSS_FUNCTIONS[@]}"; do
                    total_combinations=$((total_combinations + 1))
                done
            done
        else
            # Other models use their default backbone
            for LOSS in "${LOSS_FUNCTIONS[@]}"; do
                total_combinations=$((total_combinations + 1))
            done
        fi
    done
done

echo "Planning to run $total_combinations combinations" | tee -a "$LOG_FILE"

# Nested loops with dataset as outermost loop
for DATASET in "${DATASETS[@]}"; do
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Processing dataset: $DATASET" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    for MODEL in "${MODELS[@]}"; do
        if [[ "$MODEL" == "unet" ]]; then
            # For UNet, use both vgg11 and resnet50 backbones
            BACKBONES=("vgg11" "resnet50")
        else
            # For other models, use the default backbone
            BACKBONES=("")
        fi

        for BACKBONE_OVERRIDE in "${BACKBONES[@]}"; do
            # Get backbone info for this model
            get_backbone_info "$MODEL" "$BACKBONE_OVERRIDE"

            echo "Using model: $MODEL with backbone: $BACKBONE_PARAM" | tee -a "$LOG_FILE"

            for LOSS in "${LOSS_FUNCTIONS[@]}"; do
                # Generate checkpoint name using the display name for the backbone
                CHECKPOINT="${DATASET}_${LOSS}_${BACKBONE_NAME}_${MODEL}"
                CHECKPOINT_FILE="checkpoints/${CHECKPOINT}.pth"

                # Check if checkpoint already exists
                if [ -f "$CHECKPOINT_FILE" ]; then
                    echo "----------------------------------------" | tee -a "$LOG_FILE"
                    echo "Skipping $CHECKPOINT ($(($completed_combinations + $skipped_combinations + 1))/$total_combinations)" | tee -a "$LOG_FILE"
                    echo "Checkpoint already exists at: $CHECKPOINT_FILE" | tee -a "$LOG_FILE"
                    skipped_combinations=$((skipped_combinations + 1))
                    continue
                fi

                # Log the current combination
                echo "----------------------------------------" | tee -a "$LOG_FILE"
                echo "Training $CHECKPOINT ($(($completed_combinations + $skipped_combinations + 1))/$total_combinations)" | tee -a "$LOG_FILE"
                echo "Using dataset=$DATASET, model=$MODEL, backbone=$BACKBONE_PARAM, loss=$LOSS" | tee -a "$LOG_FILE"
                echo "Started at $(date)" | tee -a "$LOG_FILE"

                # Run the training command with the actual backbone parameter
                python scripts/train.py \
                    --model $MODEL \
                    --backbone "$BACKBONE_PARAM" \
                    --loss $LOSS \
                    --epochs $EPOCHS \
                    --batch-size $BATCH_SIZE \
                    --checkpoint $CHECKPOINT \
                    --dataset $DATASET | tee -a "$LOG_FILE"

                # Check if training was successful
                if [ $? -eq 0 ]; then
                    echo "Successfully completed $CHECKPOINT at $(date)" | tee -a "$LOG_FILE"
                else
                    echo "Failed to train $CHECKPOINT at $(date)" | tee -a "$LOG_FILE"
                fi

                completed_combinations=$((completed_combinations + 1))

                # Wait for a moment to allow system resources to stabilize
                sleep 5
            done
        done
    done
done

# Log completion
echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "All training runs completed at $(date)" | tee -a "$LOG_FILE"
echo "Completed $completed_combinations/$total_combinations combinations" | tee -a "$LOG_FILE"
echo "Skipped $skipped_combinations/$total_combinations existing combinations" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

echo "Training complete. Log file saved at: $LOG_FILE"