import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18

class ModifiedResNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(ModifiedResNet, self).__init__()
        # Load the pretrained ResNet18 model
        self.resnet = resnet18(pretrained=False)
        
        # Modify the first conv layer to accept different input channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final fully connected layer for the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        # Apply F.layer_norm to the output
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

    # Create instance of the model
    model_mnist = ModifiedResNet(num_classes=num_classes, input_channels=1)

    print("MNIST Model:")
    print(model_mnist)
    print(f"Number of trainable parameters: {count_parameters(model_mnist)}")

    # Test with MNIST dimensions
    x_mnist = torch.randn(batch_size, 1, 28, 28)
    output_mnist = model_mnist(x_mnist)
    print(f"\nMNIST output shape: {output_mnist.shape}")
    assert output_mnist.shape == (batch_size, num_classes), "MNIST output shape is incorrect"

    print("Modified ResNet-18 model tests passed!")

    # Optionally, print out some statistics about the output
    print("\nSample MNIST output (first 5 elements of the first instance in the batch):")
    print(output_mnist[0, :5])
    print("\nMean of the MNIST output (should be close to 0 for each instance):")
    print(output_mnist.mean(dim=1)[:5])
    print("\nStandard deviation of the MNIST output (should be close to 1 for each instance):")
    print(output_mnist.std(dim=1)[:5])
