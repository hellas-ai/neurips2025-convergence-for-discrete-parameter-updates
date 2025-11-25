import torch
import torch.nn as nn
import torch.nn.functional as F

class HiddenLayerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(HiddenLayerModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.layer_norm = nn.LayerNorm([num_classes], elementwise_affine=False, bias=False)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x

# Function to count the number of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test the model
if __name__ == "__main__":
    # Test with MNIST dimensions
    input_size = 28 * 28  # 784 for MNIST
    hidden_size = 128
    num_classes = 10
    batch_size = 64

    # Create an instance of the model
    model = HiddenLayerModel(input_size, hidden_size, num_classes)

    # Print the model architecture
    print(model)

    # Count and print the number of trainable parameters
    print(f"Number of trainable parameters: {count_parameters(model)}")

    # Create a random input tensor
    x = torch.randn(batch_size, 1, 28, 28)

    # Forward pass
    output = model(x)

    # Print output shape
    print(f"Output shape: {output.shape}")

    # Ensure the output has the correct shape
    assert output.shape == (batch_size, num_classes), "Output shape is incorrect"

    # Check if LayerNorm is parameterless
    for name, param in model.named_parameters():
        if 'layer_norm' in name:
            raise AssertionError("LayerNorm should not have any parameters")

    print("Hidden layer model test passed!")

    # Optionally, you can print out the output to check the normalization
    print("\nSample output (first 5 elements of the first instance in the batch):")
    print(output[0, :5])
    print("\nMean of the output (should be close to 0 for each instance):")
    print(output.mean(dim=1)[:5])
    print("\nStandard deviation of the output (should be close to 1 for each instance):")
    print(output.std(dim=1)[:5])
