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
import wandb  # 导入 wandb
import torch.nn.functional as F  # Needed for activation functions
import pandas as pd
from scipy.interpolate import interp1d

# Initialize wandb
wandb.init(
    project="energy_prediction_project_refined_data",  # Replace with your project name
    config={
        "input_dim": 3,
        "neuron": 64,
        "learning_rate": 0.0005,
        "batch_size": 512,
        "num_epochs": 2000,
        "patience": 15,
        "scaler": "MinMaxScaler",
        "optimizer": "Adam",
        "loss_function": "MSELoss",
        "data_file": "energy_surface_1_1.txt",
        "process_param_l": 2.0,
        "dropout" : 0.3,
        "unit_conversion": "au_to_eV",
    },
    name=f"run_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}",  # Optional: Name your run
    reinit=True  # Allow multiple wandb.init() calls
)

# Constants
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Using device: {device}")
date = datetime.datetime.now().strftime("%y%m%d")
ev2au = 0.0367493
# Define the root directory relative to the script's location
cur_path = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(cur_path, "results")
os.makedirs(results_dir, exist_ok=True)

# Define a fixed model save path
model_save_path = os.path.join(results_dir, 'best_model.pth')
print(f"Model will be saved to: {model_save_path}")

# Model definition with integrated MinMaxScaler
class SimpleModel(nn.Module):
    def __init__(self, input_dim, neuron, process_param_l=0.5,dropout_rate = 0):
        super(SimpleModel, self).__init__()
        self.process_param_l = process_param_l  # Parameter 'l' from wandb.config

        self.fc1 = nn.Linear(input_dim, neuron)
        self.fc2 = nn.Linear(neuron, neuron)
        self.fc3 = nn.Linear(neuron, 20)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
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

        # Apply MinMax scaling: (x - min) / scale
        scaled = processed
        # Pass through MLP layers with ReLU activations
        out = F.relu(self.fc1(scaled))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out
# Training process
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, patience):
    history = {'train_loss': [], 'test_loss': []}
    best_loss = float('inf')
    patience_counter = 0
    best_model_wts = model.state_dict()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        loss_train = []

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mask for valid (non-NaN) entries in the labels
            batch_mask = ~torch.isnan(labels)  # Mask for valid entries in the batch
            batch_mask = batch_mask.float()  # Convert mask to float for calculation

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Apply the mask: Set loss for NaN values to 0 (ignore NaN entries)
            loss = loss * batch_mask  # Apply the mask to ignore NaN entries
            loss = loss.sum() / batch_mask.sum()  # Average loss for valid entries only

            # Backward pass
            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())

        mean_train_loss = np.mean(loss_train)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {mean_train_loss:.6f}')

        # Evaluation on validation set
        model.eval()
        loss_test = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # Mask for valid (non-NaN) entries in the labels
                batch_mask = ~torch.isnan(labels)  # Mask for valid entries in the batch
                batch_mask = batch_mask.float()  # Convert mask to float for calculation

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                # Apply the mask: Set loss for NaN values to 0 (ignore NaN entries)
                loss = loss * batch_mask
                loss = loss.sum() / batch_mask.sum()  # Average loss for valid entries only

                loss_test.append(loss.item())

        mean_test_loss = np.mean(loss_test)
        print(f'Test Loss: {mean_test_loss:.6f}')

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "test_loss": mean_test_loss
        })

        # Save the best model
        if mean_test_loss < best_loss:
            best_loss = mean_test_loss
            best_model_wts = model.state_dict()
            patience_counter = 0
            torch.save(best_model_wts, model_save_path)
            print(f"Best model updated at epoch {epoch + 1} with Test Loss: {mean_test_loss:.6f}")
            wandb.save(model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')
                break

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, model_save_path)

    return history

# Define the directory containing the energy data files
data_dir = 'NN/dataset'  # Your dataset directory
data_files = sorted(glob.glob(os.path.join(data_dir, 'energy_surface_*_*.txt')))  # Get all energy surface files

features_list = []
labels_list = []

# Loop through all the files and read them
for data_file in data_files:
    try:
        # Read the file into a DataFrame (no header and space-separated)
        temp_data = pd.read_csv(data_file, delim_whitespace=True, header=0)
        
        # Extract coordinates (first 3 columns) -- Same for all files
        coordinates = temp_data.iloc[:, :3].values  # Shape (27000, 3)
        
        # Extract energy values (4th column)
        energy_values = temp_data.iloc[:, 3].values  # Shape (27000, 1)

        # Add coordinates to the features list (same coordinates for all samples)
        features_list.append(coordinates)
        
        # Add energy values as labels for the respective sample
        labels_list.append(energy_values)
        
        print(f"Successfully read file: {data_file}")
    except Exception as e:
        print(f"Error reading file {data_file}: {e}")

# Now we want to stack the feature and label data together

# Concatenate all features and labels across all files
import numpy as np

# Assuming you have the features_data and labels_data arrays
features_data = np.array(features_list[0])  # Take coordinates from the first file
labels_data = np.column_stack(labels_list)  # Concatenate the energy values from all files

# Print example data for debugging
print(labels_data[13500])
print(f"Feature data shape: {features_data.shape}")
print(f"Labels data shape: {labels_data.shape}")

# Identifying NaN entries
remove_list = []
for i in range(labels_data.shape[0]):  # Loop through each sample
    if np.isnan(labels_data[i]).any():  # Check if any element in the row is NaN
        remove_list.append(i)

# Reverse to ensure safe removal when modifying the arrays
remove_list.reverse()
print(remove_list)

# Removing rows with NaN values from features_data and labels_data
features_data = np.delete(features_data, remove_list, axis=0)
labels_data = np.delete(labels_data, remove_list, axis=0)

print(f"Cleaned feature data shape: {features_data.shape}")
print(f"Cleaned labels data shape: {labels_data.shape}")

# Now, you have (27000, 3) features and (27000, 20) labels, as expected

print(f"All data successfully concatenated. Feature data shape: {features_data.shape}, Labels data shape: {labels_data.shape}")

# Now you have features_data as a 2D array with shape (num_samples, 3)
# and labels_data as a 2D array with shape (num_samples, 20)

# Clean the data by removing any rows with NaN values (if present)
data_clean = np.hstack((features_data, labels_data))  # Combine features and labels for cleaning

# Split back into features and labels after cleaning
cleaned_features = data_clean[:, :3]
cleaned_labels = data_clean[:, 3:]

print(f"Data cleaned, number of samples: {cleaned_features.shape[0]}")

# Now you can proceed to train your model using the cleaned data
X = cleaned_features.astype(np.float32)  # Features (coordinates)
y = cleaned_labels.astype(np.float32)    # Labels (energy levels)

# Ensure the labels are 20-dimensional
y = y.reshape(-1, 20)  # Reshape to ensure each sample has 20 energy levels

# Check final shapes
print(f"Final feature shape: {X.shape}, Final label shape: {y.shape}")

# Scale the data using MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y)

# Save scaler parameters for future inverse transformation
np.save('NN/scaler_X_min.npy', scaler_X.min_)
np.save('NN/scaler_X_scale.npy', scaler_X.scale_)
np.save('NN/scaler_y_min.npy', scaler_y.min_)
np.save('NN/scaler_y_scale.npy', scaler_y.scale_)

# 2. In the training and testing data, use normalized X and y
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False)

# Instantiate the model with scaler parameters
model = SimpleModel(
    input_dim=wandb.config.input_dim,
    neuron=wandb.config.neuron,
    process_param_l=wandb.config.process_param_l,
    dropout_rate=wandb.config.dropout
).to(device)

# Set optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=wandb.config.learning_rate, weight_decay=1e-4)
criterion = nn.MSELoss()

# Train the model
history = train_model(
    model, train_loader, test_loader, criterion, optimizer,
    num_epochs=wandb.config.num_epochs, patience=wandb.config.patience
)

# Display training history
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['test_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict and inverse transform the output
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        y_pred.append(outputs.cpu().numpy())
        y_true.append(labels.cpu().numpy())

# Convert the list of predictions and true values to numpy arrays
y_pred = np.concatenate(y_pred, axis=0)
y_true = np.concatenate(y_true, axis=0)

# Inverse transform the predictions and true values
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_true_rescaled = scaler_y.inverse_transform(y_true)

# Plot the true vs predicted values
num_energy_levels = 20
energy_labels = [f"Energy Level {i+1}" for i in range(num_energy_levels)]  # Labeling energy levels

# Create a figure with subplots for each energy level
fig, axes = plt.subplots(5, 4, figsize=(20, 16))  # 5 rows, 4 columns of subplots
axes = axes.flatten()  # Flatten the axes array to make indexing easier

# Loop over each energy level and plot the histograms
for i in range(num_energy_levels):
    # Extract the i-th energy level from true and predicted values
    y_true_energy = y_true_rescaled[i]
    y_pred_energy = y_pred_rescaled[i]

    axes[i].scatter(y_true_energy, y_pred_energy, label='Predictions')
    axes[i].plot([min(y_true_energy), max(y_true_energy)], [min(y_true_energy), max(y_true_energy)], 'r--', label='Perfect Fit')
    axes[i].set_title(f"Energy Level {i+1}")
    axes[i].set_xlabel('True Values (au)')
    axes[i].set_ylabel('Predicted Values (au)')
    axes[i].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.suptitle('Distribution of True and Predicted Energy Levels', fontsize=16)
plt.subplots_adjust(top=0.95)  # Adjust top margin for suptitle
plt.show()
plt.savefig('NN/results_multi/data_distribution.png')

wandb.finish()
