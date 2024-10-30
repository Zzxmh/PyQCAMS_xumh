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
import joblib  # For loading the scaler

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

# Define the root directory relative to the script's location
cur_path = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(cur_path, "results")
os.makedirs(results_dir, exist_ok=True)

# Define a fixed model save path
model_save_path = os.path.join(results_dir, 'best_model.pth')
print(f"Model will be saved to: {model_save_path}")

# Load scaler parameters (Assuming scaler was fitted and saved previously)
scaler_path = os.path.join(cur_path, "models", "scaler.pkl")  # Update the path as necessary
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Please ensure it exists.")
scaler = joblib.load(scaler_path)
scaler_min = scaler.min_
scaler_scale = scaler.scale_

# Model definition (from potentials.py)
class SimpleModel(nn.Module):
    def __init__(self, input_dim, neuron, scaler_min=None, scaler_scale=None, process_param_l=0.5):
        """
        Initialize the SimpleModel.

        Parameters:
        - input_dim (int): Number of input features after processing.
        - neuron (int): Number of neurons in hidden layers.
        - scaler_min (np.ndarray): Minimum values used for scaling (from MinMaxScaler).
        - scaler_scale (np.ndarray): Scale values used for scaling (from MinMaxScaler).
        - process_param_l (float): Parameter 'l' used in data processing.
        """
        super(SimpleModel, self).__init__()
        self.process_param_l = process_param_l  # Parameter 'l' from wandb.config

        # Store scaler parameters as buffers (non-trainable)
        if scaler_min is not None and scaler_scale is not None:
            self.register_buffer('scaler_min', torch.tensor(scaler_min, dtype=torch.float32))
            self.register_buffer('scaler_scale', torch.tensor(scaler_scale, dtype=torch.float32))
        else:
            # Default scaling if scaler parameters are not provided
            self.register_buffer('scaler_min', torch.zeros(input_dim))
            self.register_buffer('scaler_scale', torch.ones(input_dim))

        # Define MLP layers
        self.fc1 = nn.Linear(input_dim, neuron)
        self.fc2 = nn.Linear(neuron, neuron)
        self.fc3 = nn.Linear(neuron, 1)
        
    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Raw input tensor with shape (batch_size, 3).

        Returns:
        - out (torch.Tensor): Output tensor with shape (batch_size, 1).
        """
        # Ensure input has the correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # Process raw coordinates into polynomial features
        r1, r2, r3 = x[:, 0], x[:, 1], x[:, 2]
        y1 = torch.exp(-r1 / self.process_param_l)
        y2 = torch.exp(-r2 / self.process_param_l)
        y3 = torch.exp(-r3 / self.process_param_l)
        p1 = y2 + y3
        p2 = y2**2 + y3**2
        p3 = y1
        x1 = p1
        x2 = torch.sqrt(p2)
        x3 = p3
        processed = torch.stack((x1, x2, x3), dim=1)  # Shape: (batch_size, 3)

        # Scale the processed features
        scaled = (processed - self.scaler_min) / self.scaler_scale

        # Pass through MLP layers with ReLU activations
        out = F.relu(self.fc1(scaled))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def set_scaler_parameters(self, scaler_min, scaler_scale):
        """
        Update scaler parameters.

        Parameters:
        - scaler_min (np.ndarray): Minimum values used for scaling.
        - scaler_scale (np.ndarray): Scale values used for scaling.
        """
        self.scaler_min = torch.tensor(scaler_min, dtype=torch.float32).to(self.scaler_min.device)
        self.scaler_scale = torch.tensor(scaler_scale, dtype=torch.float32).to(self.scaler_scale.device)

# Instantiate the model with scaler parameters
input_dim = wandb.config.input_dim
neuron = wandb.config.neuron
process_param_l = wandb.config.process_param_l
model = SimpleModel(input_dim, neuron, scaler_min=scaler_min, scaler_scale=scaler_scale, process_param_l=process_param_l).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
num_epochs = wandb.config.num_epochs
patience = wandb.config.patience

# Prepare data loading
# Define the root directory
root_dir = 'E:/'  # Note: Do not include backslash here

# Define the subdirectories
sub_dirs = ['tasks', 'documents_in_pku', 'research', 'roaming', 'graph_3']

# Use os.path.join to construct the full path
data_dir = os.path.join(root_dir, *sub_dirs)

# Get the specific text file
data_files = os.path.join(data_dir, wandb.config.data_file)

# Initialize data list
data_list = []

# Read each file and append to data list
if not os.path.exists(data_files):
    raise FileNotFoundError(f"Data file not found at {data_files}. Please check the path.")
temp_data = pd.read_csv(data_files, delim_whitespace=True, header=None)
data_list.append(temp_data)

# Concatenate all data
data_df = pd.concat(data_list, ignore_index=True)

# Extract features and labels
X = data_df.iloc[:, :3].values  # First three columns are coordinates
y = data_df.iloc[:, 3].values.reshape(-1, 1)  # Fourth column is potential energy

# Note: process_data_batch is now integrated into the model's forward method
# Thus, skip processing and scaling here

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling is now handled inside the model

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

# Log model graph to wandb (optional)
# To log the model graph, you need a sample input
sample_input, _ = next(iter(train_loader))
wandb.watch(model, log="all", log_freq=100)

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

# Training
start_time = time.time()
history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, patience)
end_time = time.time()

print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Optionally, plot training history
plt.figure(figsize=(10,5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['test_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'loss_history.png'))
plt.show()

# Finish the wandb run
wandb.finish()
