import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import linear, hidden, convolutional, resnet
from optimize import ZIMPerParamOptimizer, ZIMOptimizer, custom_init, is_discrete
from experiment import Experiment, ExperimentResult

def get_git_commit_hash():
    """Get the current git commit hash. Crashes if git command fails."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def max_abs_weight(model):
    a = 0.0
    for name, param in model.named_parameters():
        a = max(param.data.abs().max())
    return a



def print_weight_statistics(model, optimizer_name):
    print(f"Weight statistics:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data.min()} {param.data.max()}")
        if optimizer_name == "ZIMPerParam" and not is_discrete(param.data):
            raise ValueError("Model weights should be discrete, but aren't!")

def load_dataset(name):
    if name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

def get_model(name, input_size, num_classes, dataset_name):
    if name == "linear":
        return linear.LinearModel(input_size, num_classes)
    elif name == "hidden":
        return hidden.HiddenLayerModel(input_size, 128, num_classes)
    elif name == "convolutional":
        if dataset_name == "MNIST":
            return convolutional.ConvolutionalModel(num_classes=10, input_channels=1, input_height=28, input_width=28)
        raise ValueError(f"Unknown dataset: {dataset_name}")
    elif name == "resnet":
        if dataset_name == "MNIST":
            return resnet.ModifiedResNet(num_classes=num_classes, input_channels=1)
        raise ValueError(f"Unknown dataset: {dataset_name}")
    else:
        raise ValueError(f"Unknown model: {name}")

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f'Iteration {batch_idx + 1}: Loss: {running_loss / 100:.3f} | Acc: {100. * correct / total:.2f}%')
            running_loss = 0.0
            correct = 0
            total = 0

        if batch_idx % 20 == 0:
            yield batch_idx

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)')
    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser(description='ML Experiment')
    parser.add_argument('--model', type=str, required=True, choices=['linear', 'hidden', 'convolutional', 'resnet'])
    parser.add_argument('--dataset', type=str, required=True, choices=['MNIST'])
    parser.add_argument('--optimizer', type=str, required=True, choices=['sgd', 'ZIMPerParam', 'ZIM'])
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_dataset(args.dataset)
    input_size = 784
    num_classes = 10

    model = get_model(args.model, input_size, num_classes, args.dataset).to(device)

    if args.optimizer in ['ZIMPerParam', 'ZIM']:
        custom_init(model)

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif args.optimizer == 'ZIMPerParam':
        optimizer = ZIMPerParamOptimizer(model.parameters(), lr=0.01)
    elif args.optimizer == 'ZIM':
        optimizer = ZIMOptimizer(model.parameters(), lr=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    results = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print_weight_statistics(model, args.optimizer)

        model.train()
        for batch_idx in train(model, train_loader, optimizer, criterion, device):
            print_weight_statistics(model, args.optimizer)
            print(f"evaluating {batch_idx=}")
            model.eval()
            with torch.no_grad():
                test_loss, test_acc = test(model, test_loader, criterion, device)
                results.append(ExperimentResult(
                    epoch=epoch + 1,
                    iters=epoch * len(train_loader) + batch_idx,
                    test_loss=test_loss,
                    test_acc=test_acc
                ))
            model.train()

    print_weight_statistics(model, args.optimizer)

    # Save results
    commit_hash = get_git_commit_hash()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'experiment_runs'
    filename = f"{output_dir}/{args.model}_{args.dataset}_{args.optimizer}_{commit_hash[:8]}_{timestamp}.json"
    Path(output_dir).mkdir(exist_ok=True)

    experiment = Experiment(
        commit=commit_hash,
        timestamp=timestamp,
        model=args.model,
        dataset=args.dataset,
        optimizer=args.optimizer,
        epochs=args.epochs,
        results=results
    )

    experiment.save(filename)

    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
