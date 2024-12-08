import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer, required
import torch.multiprocessing as mp

# Set fixed randomness
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Define a simple CNN for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# CosineAdjustingMomentum Optimizer
class CosineAdjustingMomentum(Optimizer):
    def __init__(self, params, lr=required, beta=0.9, epsilon=1e-8):
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon)
        super(CosineAdjustingMomentum, self).__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            beta = group['beta']
            lr = group['lr']
            epsilon = group['epsilon']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('CosineAdjustingMomentum does not support sparse gradients')
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['prev_grad'] = torch.zeros_like(p.data)
                    state['momentum'] = torch.zeros_like(p.data)
                    
                prev_grad = state['prev_grad']
                momentum = state['momentum']
                state['step'] += 1
                
                # Compute cosine similarity between current and previous gradient
                cos_sim = torch.nn.functional.cosine_similarity(
                    grad.view(-1), prev_grad.view(-1), dim=0
                )
                
                # Adjust beta based on cosine similarity
                adaptive_beta = beta * (1 + cos_sim) / 2  # Maps cos_sim (-1,1) to beta*(0,1)
                adaptive_beta = adaptive_beta.clamp(0, beta)
                
                # Update momentum
                momentum.mul_(adaptive_beta).add_(grad, alpha=1 - adaptive_beta)
                
                # Update parameters
                p.data.add_(-lr, momentum)
                
                # Update previous gradient
                state['prev_grad'].copy_(grad)
                
        return loss

# Define a function to initialize the optimizer based on a hyperparameter
optimizer_name = "adam"  # Change to "cosine" to use the CosineAdjustingMomentum optimizer

if optimizer_name == "adam":
    optimizer = optim.Adam(model.parameters(), lr=0.001)
elif optimizer_name == "cosine":
    optimizer = CosineAdjustingMomentum(model.parameters(), lr=0.001)
else:
    raise ValueError("Unsupported optimizer name. Use 'adam' or 'cosine'.")

# Train the model
if __name__ == "__main__":
    def train_model(model, optimizer, epochs=5):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.3f}")
        print('Finished Training')

    # Train with selected optimizer
    print(f"Training with {optimizer_name.capitalize()} Optimizer:")
    train_model(model, optimizer)

    # Reinitialize model weights before training with another optimizer
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
