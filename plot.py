import os
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import defaultdict
from experiment import Experiment

def get_all_commits():
    """Get all unique commits from experiment files."""
    data_dir = 'experiment_runs'
    if not os.path.exists(data_dir):
        return []

    commits = set()
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            parts = filename.split('_')
            if len(parts) >= 6:  # model_dataset_optimizer_commit_date_time.json
                commit = parts[3]  # commit hash is at index 3
                commits.add(commit)
    return sorted(list(commits))

def get_latest_commit():
    """Get the most recent commit based on file timestamps."""
    commits = get_all_commits()
    if not commits:
        return None

    data_dir = 'experiment_runs'
    commit_times = {}

    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            parts = filename.split('_')
            if len(parts) >= 6:
                commit = parts[3]  # commit hash is at index 3
                timestamp_str = '_'.join(parts[-2:]).split('.')[0]  # get date_time part
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                if commit not in commit_times or timestamp > commit_times[commit]:
                    commit_times[commit] = timestamp

    if not commit_times:
        return None

    return max(commit_times.keys(), key=lambda c: commit_times[c]) if commit_times else None

def load_runs_for_commit(model_name, dataset_name, optimizer, commit):
    """Load all runs for a specific commit."""
    data_dir = 'experiment_runs'
    pattern = f"{model_name}_{dataset_name}_{optimizer}_{commit}"
    files = [f for f in os.listdir(data_dir) if f.startswith(pattern) and f.endswith('.json')]

    experiments = []
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        experiment = Experiment.load(file_path)
        if experiment:
            experiments.append(experiment)

    return experiments

def aggregate_results(experiments):
    """Aggregate results from multiple experiments, computing mean and std."""
    if not experiments:
        return None, None, None, None

    # Group results by iteration number
    iter_data = defaultdict(list)
    for exp in experiments:
        for result in exp.results:
            iter_data[result.iters].append((result.test_loss, result.test_acc))

    # Calculate mean and std for each iteration
    iters = sorted(iter_data.keys())
    loss_means, loss_stds = [], []
    acc_means, acc_stds = [], []

    for iter_num in iters:
        losses = [data[0] for data in iter_data[iter_num]]
        accs = [data[1] for data in iter_data[iter_num]]

        loss_means.append(np.mean(losses))
        loss_stds.append(np.std(losses) if len(losses) > 1 else 0)
        acc_means.append(np.mean(accs))
        acc_stds.append(np.std(accs) if len(accs) > 1 else 0)

    return iters, loss_means, loss_stds, acc_means, acc_stds

def load_latest_run(model_name, dataset_name, optimizer):
    """Load latest run for backward compatibility."""
    latest_commit = get_latest_commit()
    if not latest_commit:
        return None

    experiments = load_runs_for_commit(model_name, dataset_name, optimizer, latest_commit)
    if experiments:
        return experiments[0].results
    return None

def plot_dataset_results(dataset_name, model_names, optimizers, commit=None):
    plt.rcParams.update({'font.size': 14})  # Increase base font size
    plt.figure(figsize=(12, 8))  # Slightly smaller figure for better scaling
    has_data = False
    data_summary = {f"{model}_{opt}": "No data" for model in model_names for opt in optimizers}

    if commit is None:
        commit = get_latest_commit()

    if not commit:
        print("No commits found in experiment_runs")
        return data_summary

    # Plot loss
    plt.subplot(2, 1, 1)
    for model_name in model_names:
        for optimizer in optimizers:
            experiments = load_runs_for_commit(model_name, dataset_name, optimizer, commit)
            if experiments:
                iters, loss_means, loss_stds, _, _ = aggregate_results(experiments)
                if iters and loss_means:
                    opt_label = "ZIM (ours)" if optimizer == "ZIM" else optimizer
                    plt.plot(iters, loss_means, label=f"{model_name} ({opt_label})")
                    plt.fill_between(iters, np.array(loss_means) - np.array(loss_stds),
                                   np.array(loss_means) + np.array(loss_stds), alpha=0.3)
                    plt.yscale('log')
                    has_data = True
                    data_summary[f"{model_name}_{optimizer}"] = f"Data available ({len(experiments)} runs, {len(iters)} iters)"
                else:
                    data_summary[f"{model_name}_{optimizer}"] = "Empty data file"
    plt.title(f"{dataset_name} - Test Loss") # (Commit: {commit[:8]})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    if not has_data:
        plt.text(0.5, 0.5, "No data available", ha='center', va='center')

    # Plot accuracy
    plt.subplot(2, 1, 2)
    for model_name in model_names:
        for optimizer in optimizers:
            experiments = load_runs_for_commit(model_name, dataset_name, optimizer, commit)
            if experiments:
                iters, _, _, acc_means, acc_stds = aggregate_results(experiments)
                if iters and acc_means:
                    opt_label = "ZIM (ours)" if optimizer == "ZIM" else optimizer
                    plt.plot(iters, acc_means, label=f"{model_name} ({opt_label})")
                    plt.fill_between(iters, np.array(acc_means) - np.array(acc_stds),
                                   np.array(acc_means) + np.array(acc_stds), alpha=0.3)
                    has_data = True
    plt.title(f"{dataset_name} - Test Accuracy") # (Commit: {commit[:8]})")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.ylim(95, 100)  # Set y-axis range from 80 to 100
    if not has_data:
        plt.text(0.5, 0.5, "No data available", ha='center', va='center')

    plt.tight_layout()

    if has_data:
        # Get handles and labels from the first subplot
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.figlegend(handles, labels, bbox_to_anchor=(0.5, 0.02), loc='lower center', ncol=2, fontsize=12)

    plt.subplots_adjust(bottom=0.18)  # Make room for legend
    plt.savefig(f"result_{dataset_name}_{commit[:8]}.png", bbox_inches='tight')
    plt.close()

    return data_summary

def main():
    parser = argparse.ArgumentParser(description='Plot experiment results with error bars')
    parser.add_argument('--commit', type=str, help='Specific commit hash to plot (default: latest commit)')
    parser.add_argument('--list-commits', action='store_true', help='List all available commits')
    args = parser.parse_args()

    if args.list_commits:
        commits = get_all_commits()
        print("Available commits:")
        for commit in commits:
            print(f"  {commit}")
        return

    datasets = ["MNIST"]
    model_names = ["convolutional", "resnet"]
    optimizers = ["sgd", "ZIM"]

    commit = args.commit if args.commit else get_latest_commit()

    if not commit:
        print("No experiment data found in experiment_runs/")
        return

    print(f"Plotting results for commit: {commit}")
    print("Data Availability Summary:")

    for dataset in datasets:
        print(f"\n{dataset}:")
        data_summary = plot_dataset_results(dataset, model_names, optimizers, commit)
        for model_opt, status in data_summary.items():
            model, opt = model_opt.split('_')
            print(f"  {model} ({opt}): {status}")

    print("\nPlots have been generated and saved as PNG files.")
    print("Note: Error bars show standard deviation across multiple runs for the same commit.")
    print("Use --list-commits to see all available commits.")
    print("Use --commit <hash> to plot a specific commit.")

if __name__ == "__main__":
    main()
