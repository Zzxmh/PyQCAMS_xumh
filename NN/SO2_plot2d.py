import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.cm import ScalarMappable
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define model file path
model_file = 'NN/results/best_model.pth'  # Replace with actual model path
scaler_X_min = np.load('NN/scaler_X_min.npy')        # Shape should be (3,)
scaler_X_scale = np.load('NN/scaler_X_scale.npy')    # Shape should be (3,)
scaler_y_min = np.load('NN/scaler_y_min.npy')        # Shape should be (3,)
scaler_y_scale = np.load('NN/scaler_y_scale.npy')
# Function to compute r3
def derive_r3(r1, r2, theta):
    return np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(theta))

def plot_cross_section(ax, x, y, values, title, vmin, vmax):
    """
    Plot contour, ignore NaN values, and use consistent color mapping.
    """
    masked_values = np.ma.masked_invalid(values)
    cfill = ax.contourf(x, y, masked_values, levels=50, cmap='coolwarm', vmin=vmin, vmax=vmax)
    contour = ax.contour(x, y, masked_values, levels=50, colors='black', linewidths=0.5)
    ax.clabel(contour, inline=1, fontsize=8, colors='black')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('$r_{OO}$', fontsize=10)
    ax.set_ylabel('$r_{OS}$', fontsize=10)


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

# Instantiate and load model
input_dim = 3
neuron = 64
process_param_l = 1.4
model = SimpleModel(input_dim, neuron,process_param_l).to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

# Generate test data
r2 = np.linspace(0, 4.0, 100)
r3 = np.linspace(0, 4.0, 100)
theta_values = np.linspace(0, np.pi, 20)
fig, axes = plt.subplots(5, 4, figsize=(20, 25))
axes = axes.flatten()

# Pre-compute predictions
all_predicted_energy = []
for theta in theta_values:
    r2_grid, r3_grid = np.meshgrid(r2, r3)
    r1_grid = derive_r3(r2_grid, r3_grid, theta)
    test_input = np.vstack([r1_grid.flatten(), r2_grid.flatten(), r3_grid.flatten()]).T
    # 使用归一化后的数据生成 tensor
    test_input_tensor = torch.from_numpy(test_input).float().to(device)

    with torch.no_grad():
        predicted_energy = model(test_input_tensor).cpu().numpy()
    #predicted_energy_transformed = predicted_energy*scaler_scale + scaler_min
    predicted_energy_transformed = predicted_energy[0]
    all_predicted_energy.append(predicted_energy_transformed)

# Calculate global min and max for consistent color mapping
all_predicted_energy = np.array(all_predicted_energy)
global_min, global_max = all_predicted_energy.min(), all_predicted_energy.max()

# Plot each theta slice
for idx, theta in enumerate(theta_values):
    predicted_energy_grid = all_predicted_energy[idx].reshape(r1_grid.shape)
    ax = axes[idx]
    plot_cross_section(ax, r2_grid, r3_grid, predicted_energy_grid, f'θ = {theta:.2f} rad', vmin=global_min, vmax=global_max)

# Remove extra subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and add colorbar
plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
sm = ScalarMappable(cmap='coolwarm')
sm.set_array([])  # Empty array to prevent warning
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Potential Energy (eV)', fontsize=14)

plt.suptitle('Potential Energy Surfaces for Different Theta Values', fontsize=20)
plt.savefig('PES_sections_contour.png', dpi=300, bbox_inches='tight')
plt.show()