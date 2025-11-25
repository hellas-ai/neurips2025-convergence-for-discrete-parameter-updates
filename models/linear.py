import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.ln = nn.LayerNorm([num_classes], elementwise_affine=False, bias=False)
    
    def forward(self, x):
        # Flatten the input if it's not already flat
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Pass through the linear layer
        x = self.linear(x)
        x = self.ln(x)
        
        return x

# Function to count the number of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test the model
if __name__ == "__main__":
    # Test with MNIST dimensions
    input_size = 28 * 28  # 784 for MNIST
    num_classes = 10
    batch_size = 64

    # Create an instance of the model
    model = LinearModel(input_size, num_classes)

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

    print("Linear model test passed!")
