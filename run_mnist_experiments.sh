#!/bin/bash

# Wrapper script to run MNIST experiments for non-linear models only
# (hidden, convolutional, resnet)

set -e  # Exit on any error

# Check if epochs argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Number of epochs is required"
    echo "Usage: $0 <epochs>"
    echo "Example: $0 10"
    echo "This will run experiments for: hidden, convolutional, resnet models"
    exit 1
fi

epochs=$1

echo "Starting non-linear model experiments for MNIST dataset..."
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Models to run: hidden, convolutional, resnet"
echo "Epochs: $epochs"
echo ""

# Define models to run
models=("hidden" "convolutional" "resnet")

# Run experiments for each model
for model in "${models[@]}"; do
    echo "========================================"
    echo "Starting experiments for model: $model"
    echo "========================================"

    ./run_mnist_model_experiments.sh "$model" "$epochs"

    echo "Completed experiments for model: $model"
    echo ""
done

echo "All non-linear model experiments completed!"
echo "Generating final plots..."
python plot.py

echo "Done! Check the generated PNG files for results."
