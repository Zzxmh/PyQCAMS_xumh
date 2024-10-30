import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import datetime
import os
import glob
import re
import wandb  # Import wandb

# Initialize wandb
wandb.init(
    project="energy_prediction_project",  # Replace with your project name
    config={
        "input_dim": 3,
        "neuron": 50,
        "learning_rate": 0.0005,
        "batch_size": 32,
        "num_epochs": 1000,
        "patience": 20,
        "scaler": "MinMaxScaler",
        "optimizer": "Adam",
        "loss_function": "MSELoss",
        "data_file": "energy_data_1_1.txt",
        "process_param_l": 0.5
    },
    name=f"run_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}",  # Optional: Name your run
    reinit=True  # Allow multiple wandb.init() calls
)

# Constants
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
date = datetime.datetime.now().strftime("%y%m%d")
ev2au = 0.0367493
cur_path = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(cur_path, "results")
model_save_path = os.path.join(results_dir, 'best_model.pth')
print(f"Model will be saved to: {model_save_path}")

# Function to process data
def process_data_batch(data, l):
    result = np.zeros_like(data)
    for i, t in enumerate(data):
        r1, r2, r3 = t[0], t[1], t[2]
        y1 = np.exp(-r1 / l)
        y2 = np.exp(-r2 / l)
        y3 = np.exp(-r3 / l)
        p1 = y2 + y3
        p2 = y2**2 + y3**2
        p3 = y1
        x1 = p1
        x2 = p2**(1 / 2)
        x3 = p3
        result[i] = [x1, x2, x3]
    return result

# Model definition
class SimpleModel(nn.Module):
    def __init__(self, input_dim, neuron):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, neuron)
        self.fc2 = nn.Linear(neuron, neuron)
        self.fc3 = nn.Linear(neuron, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training process
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, patience):
    history = {'train_loss': [], 'test_loss': []}
    best_loss = float('inf')
    patience_counter = 0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        loss_train = []

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())

        mean_train_loss = np.mean(loss_train)
        history['train_loss'].append(mean_train_loss)

        # Evaluate on test set
        model.eval()
        loss_test = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_test.append(loss.item())

        mean_test_loss = np.mean(loss_test)
        history['test_loss'].append(mean_test_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {mean_train_loss:.6f}, Test Loss: {mean_test_loss:.6f}')

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "test_loss": mean_test_loss
        })

        # Save best model
        if mean_test_loss < best_loss:
            best_loss = mean_test_loss
            best_model_wts = model.state_dict()
            patience_counter = 0
            # Save the best model locally (overwrite if exists)
            torch.save(best_model_wts, model_save_path)
            # Log the best model to wandb
            wandb.save(model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return history

# Data loading and preprocessing

# Define the root directory
root_dir = 'E:/'  # Note: Do not include backslash here

# Define the subdirectories
sub_dirs = ['tasks', 'documents_in_pku', 'research', 'roaming', 'graph_3']

# Use os.path.join to construct the full path
data_dir = os.path.join(root_dir, *sub_dirs)

# Get the specific text file
data_files = os.path.join(data_dir, 'energy_data_1_1.txt')

# Initialize data list
data_list = []

# Read each file and append to data list
temp_data = pd.read_csv(data_files, delim_whitespace=True, header=None)
data_list.append(temp_data)

# Concatenate all data
data_df = pd.concat(data_list, ignore_index=True)

# Extract features and labels
X = data_df.iloc[:, :3].values  # First three columns are coordinates
y = data_df.iloc[:, 3].values.reshape(-1, 1)  # Fourth column is potential energy

# Process data
process_param_l = wandb.config.process_param_l  # Use config parameter
data = process_data_batch(X, l=process_param_l)

# Feature scaling
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Label scaling
label_scaler = MinMaxScaler()
labels = label_scaler.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)
train_loader = DataLoader(
    train_dataset, batch_size=wandb.config.batch_size, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=wandb.config.batch_size, shuffle=False
)

# Model instantiation
input_dim = wandb.config.input_dim
neuron = wandb.config.neuron
model = SimpleModel(input_dim, neuron).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
num_epochs = wandb.config.num_epochs
patience = wandb.config.patience

# Log model graph to wandb (optional)
# To log the model graph, you need a sample input
sample_input, _ = next(iter(train_loader))
wandb.watch(model, log="all", log_freq=100)

# Training
start_time = time.time()
history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, patience)
end_time = time.time()

# Finish the wandb run
wandb.finish()
