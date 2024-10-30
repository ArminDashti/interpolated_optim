import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define a simple 2-layer network
class TwoLayerNet(nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
model = TwoLayerNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Variables to track parameter values and loss ratios
parameter_values = []
loss_ratios = []
previous_parameters = {name: param.clone() for name, param in model.named_parameters() if 'weight' in name}
previous_loss = None
step_count = 0

# Select a single value from a parameter randomly
parameter_names = [name for name, _ in model.named_parameters() if 'weight' in name]
selected_parameter_name = random.choice(parameter_names)
selected_parameter = model.state_dict()[selected_parameter_name]
selected_index = tuple(random.randint(0, i - 1) for i in selected_parameter.shape)
print(f"Selected parameter: {selected_parameter_name}, Selected index: {selected_index}")

# Training loop
for epoch in range(20):  # 5 epochs for demonstration
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.view(-1, 784)  # Flatten the images

        # Forward pass
        outputs = model(inputs)
        current_loss = criterion(outputs, targets)
        epoch_loss += current_loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

        # Update parameters every 3 steps
        step_count += 1
        if step_count % 2 == 0:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        previous_param = previous_parameters[name]
                        averaged_param = param - previous_param
                        param.add_(averaged_param)  # Add the averaged_param to the current param

                # Update previous parameters for all parameters
        previous_parameters = {name: param.clone() for name, param in model.named_parameters() if 'weight' in name}

        # Print the total loss for the epoch
    print(f"Epoch {epoch + 1}, Total Loss: {epoch_loss}")

    # Track the current value of the selected parameter
    with torch.no_grad():
        current_value = model.state_dict()[selected_parameter_name][selected_index].item()
        parameter_values.append(current_value)

    # Compute ratio of current loss to previous loss
    if previous_loss is not None:
        loss_ratio = epoch_loss / (previous_loss + 1e-8)  # Add small value to avoid division by zero
        loss_ratios.append(loss_ratio)
    previous_loss = epoch_loss

# Compute variance of each parameter individually
parameter_variances = {}
with torch.no_grad():
    for name, param in model.named_parameters():
        if param.requires_grad:  # Compute variance for all parameters individually
            variance = torch.var(param).item()
            parameter_variances[name] = variance

print("Training complete.")
