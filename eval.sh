#!/bin/bash
# This script runs the evaluation on all checkpoints in the checkpoints directory using the specified datasets below.

# Define datasets and their corresponding data paths
declare -A DATASETS
DATASETS["kovo_video"]="data/volleyball/2324/mp4"
DATASETS["volleyball"]="data/volleyball/volleyball_mp4"

# Define the output file for the results
RESULTS_FILE="logs/eval.txt"

# Define the main directory containing the subdirectories
CHECKPOINT_DIR="checkpoints"

# Collect all checkpoint paths
CHECKPOINTS=$(ls $CHECKPOINT_DIR/*/*.ckpt)

# Create the logs directory if it doesn't exist
mkdir -p logs

# Clear the results file if it exists
> $RESULTS_FILE

# Add the date and time to the results file
echo "Evaluation started at: $(date)" | tee -a $RESULTS_FILE
echo "===================================" | tee -a $RESULTS_FILE

# Loop through each checkpoint
for CHECKPOINT in $CHECKPOINTS; do
  echo "Processing checkpoint: $CHECKPOINT" | tee -a $RESULTS_FILE

  # Loop through each dataset
  for DATASET in "${!DATASETS[@]}"; do
    DATA_PATH=${DATASETS[$DATASET]}
    echo "  Evaluating on dataset: $DATASET with data path: $DATA_PATH" | tee -a $RESULTS_FILE

    # Run the evaluation command
    COMMAND="python main.py --mode eval --use_temporal_encodings --eval_checkpoint $CHECKPOINT --dataset $DATASET --data_path $DATA_PATH"
    echo "    Running command: $COMMAND" | tee -a $RESULTS_FILE
    $COMMAND >> $RESULTS_FILE 2>&1
  done
done

echo "===================================" | tee -a $RESULTS_FILE
echo "Evaluation completed at: $(date)" | tee -a $RESULTS_FILE
echo "Results saved in $RESULTS_FILE."
