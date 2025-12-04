# Convergence for Discrete Parameter Updates

This repository contains experiments accompanying the paper
[Convergence for Discrete Parameter Updates](https://arxiv.org/abs/2512.04051).
In particular, the *multinomial sampling optimizer* proposed.

Experiments plot convergence for four models against the MNIST dataset:

- Linear
- Hidden layer  
- Convolutional
- ResNet (modified for small datasets)

The optimizers tested are:

- SGD
- Custom multinomial sampling optimizer (per-parameter sampling)
- Global multinomial sampling optimizer (all weights simultaneously sampled)

## Usage

### Run Single Experiment
```bash
python main.py --model [model] --dataset [dataset] --optimizer [optimizer] --epochs [epochs]
```

### Run All MNIST Experiments (10 repetitions each)
```bash
./run_mnist_experiments.sh [epochs]
```

### Plot Results with Error Bars
```bash
python plot.py                    # Plot latest commit
python plot.py --commit [hash]    # Plot specific commit  
python plot.py --list-commits     # List available commits
```

# Experiment Tracking

- Running `main.py` writes experiment results to `experiment_runs` in the format described by `experiment.py`.
- Data from each run is stored in a file named `$MODEL_$DATASET_$OPTIMIZER_$COMMIT_HASH_$TIMESTAMP.json`
- Running `plot.py` will plot an experiment run for a given commit.
- This is either inferred (finds the latest commit), or can be specified

To run experiments, use

    ./run_mnist_experiments.sh

This will run each experiment 10 times - an experiment is a combination of model/optimizer.

## Results

Plots are written to files `$DATASET_results_$COMMIT.png`

Generated PNG files show loss/accuracy plots with error bars. Filenames include commit hash for traceability.

# Files

- `main.py`: Main experiment runner with commit logging
- `plot.py`: Plotting with error bars and commit-specific visualization
- `experiment.py`: Structured experiment data format
- `run_mnist_experiments.sh`: Batch experiment runner
- `optimize.py`: Custom optimizer implementations
- `models/`: Model definitions
- `experiment_runs/`: Saved experiment results (format: `model_dataset_optimizer_commit_timestamp.json`)
