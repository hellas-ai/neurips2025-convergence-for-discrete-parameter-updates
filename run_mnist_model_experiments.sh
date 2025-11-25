#!/bin/bash

# Run all combinations of models for MNIST with all optimizers
# This will generate multiple runs for error bar analysis

set -e  # Exit on any error

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: Model and epochs arguments are required"
    echo "Usage: $0 <model> <epochs>"
    echo "Example: $0 linear 10"
    echo "Available models: linear, hidden, convolutional, resnet"
    echo "Optimizers: sgd, ZIM_per_param, ZIM"
    exit 1
fi

model=$1
epochs=$2
repetitions=10

echo "Starting experiment runs for MNIST dataset..."
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo ""

# Define arrays
optimizers=("sgd" "ZIM")
dataset="MNIST"

echo "Running experiments with the following configuration:"
echo "Dataset: $dataset"
echo "Model: $model"
echo "Optimizers: ${optimizers[*]}"
echo "Epochs: $epochs"
echo "Repetitions per experiment: $repetitions"
echo ""

# Counter for progress tracking
total_runs=$((${#optimizers[@]} * $repetitions))
current_run=0

# Loop through all optimizers
for optimizer in "${optimizers[@]}"; do
    echo "Running $model with $optimizer optimizer (10 repetitions)..."

    # Run each experiment 10 times
    for run in $(seq 1 $repetitions); do
        current_run=$((current_run + 1))
        echo "  [$current_run/$total_runs] Run $run/10: $model with $optimizer optimizer"

        # Run the experiment
        python main.py --model "$model" --dataset "$dataset" --optimizer "$optimizer" --epochs "$epochs"
    done

    echo "Completed all runs for: $model with $optimizer"
    echo ""
done
