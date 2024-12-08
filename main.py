import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
import time

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()

# Define a more complex convolutional neural network
class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
hyperparams = {
    'update_params': False,
    'optimizer_choice': 'Adam',  # Options: 'SGD', 'Adam'
    'num_epochs': 20,
    'batch_size': 64,
    'learning_rate': {'SGD': 0.01, 'Adam': 0.001}
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss, and optimizer
model = ComplexNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=hyperparams['learning_rate']['SGD']) if hyperparams['optimizer_choice'] == 'SGD' else optim.Adam(model.parameters(), lr=hyperparams['learning_rate']['Adam'])

# Load CIFAR-10 dataset
def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = get_data_loaders(hyperparams['batch_size'])

# Training loop to store parameters at epoch 0 and 5, predict next 10 epochs, replace, then resume
def train_model(model, train_loader, criterion, optimizer, hyperparams):
    epoch_params = {}
    start_time = time.time()
    step_count = 0

    for epoch in range(hyperparams['num_epochs']):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            current_loss = criterion(outputs, targets)
            epoch_loss += current_loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()

        # Save parameters at epoch 0 and epoch 5
        if epoch == 0 or epoch == 4:
            epoch_params[epoch + 1] = {name: param.clone() for name, param in model.named_parameters() if 'weight' in name}

        # Print the total loss for the epoch
        print(f"Epoch {epoch + 1}, Total Loss: {epoch_loss}")

        # Interpolate parameters to predict epochs 6-15 after epoch 5
        if epoch == 4:
            predicted_params = interpolate_params(epoch_params[1], epoch_params[5], steps=10)
            # Replace the predicted parameters in the model
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name and name in predicted_params:
                        param.copy_(predicted_params[name])

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training complete. Total training time: {training_time:.2f} seconds.")

# Interpolation function
def interpolate_params(params_epoch_1, params_epoch_5, steps=10):
    interpolated_params = {}
    for step in range(1, steps + 1):
        alpha = step / steps
        for name in params_epoch_1:
            interpolated_params[name] = (1 - alpha) * params_epoch_1[name] + alpha * params_epoch_5[name]
    return interpolated_params

# Train the model
train_model(model, train_loader, criterion, optimizer, hyperparams)

# Evaluation after training
def evaluate_model(model, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    accuracy = total_correct / total_samples * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

evaluate_model(model, test_loader)
