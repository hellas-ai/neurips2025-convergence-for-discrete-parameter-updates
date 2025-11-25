import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalModel(nn.Module):
    def __init__(self, num_classes, input_channels, input_height, input_width):
        super(ConvolutionalModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size of the feature maps after convolutions and pooling
        self.feature_size = (input_height // 4) * (input_width // 4) * 64

        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.layer_norm(x, x.shape[1:])

        x = self.pool(F.relu(self.conv2(x)))
        x = F.layer_norm(x, x.shape[1:])

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.layer_norm(x, x.shape[1:])

        x = self.fc2(x)
        x = F.layer_norm(x, x.shape[1:])
        return x

# Function to count the number of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test the model
if __name__ == "__main__":
    torch.manual_seed(42)
    num_classes = 10
    batch_size = 64

    # Create model for MNIST
    model_mnist = ConvolutionalModel(num_classes=10, input_channels=1, input_height=28, input_width=28)

    print("MNIST Model:")
    print(model_mnist)
    print(f"Number of trainable parameters: {count_parameters(model_mnist)}")

    # Test with MNIST dimensions
    x_mnist = torch.randn(batch_size, 1, 28, 28)
    output_mnist = model_mnist(x_mnist)
    print(f"\nMNIST output shape: {output_mnist.shape}")
    assert output_mnist.shape == (batch_size, num_classes), "MNIST output shape is incorrect"

    print("Convolutional model tests passed!")

    # Optionally, print out some statistics about the output
    print("\nSample MNIST output (first 5 elements of the first instance in the batch):")
    print(output_mnist[0, :5])
    print("\nMean of the MNIST output (should be close to 0 for each instance):")
    print(output_mnist.mean(dim=1)[:5])
    print("\nStandard deviation of the MNIST output (should be close to 1 for each instance):")
    print(output_mnist.std(dim=1)[:5])
