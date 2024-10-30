import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define model file path
model_file = 'NN/results/best_model.pth'  # Replace with actual model path

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
    def __init__(self, input_dim, neuron, scaler_min, scaler_scale, process_param_l=0.5):
        super(SimpleModel, self).__init__()
        self.process_param_l = process_param_l
        self.register_buffer('scaler_min', torch.tensor(scaler_min, dtype=torch.float32))
        self.register_buffer('scaler_scale', torch.tensor(scaler_scale, dtype=torch.float32))
        self.fc1 = nn.Linear(input_dim, neuron)
        self.fc2 = nn.Linear(neuron, neuron)
        self.fc3 = nn.Linear(neuron, 1)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        r1, r2, r3 = x[:, 0], x[:, 1], x[:, 2]
        y1 = torch.exp(-r1 / self.process_param_l)
        y2 = torch.exp(-r2 / self.process_param_l)
        y3 = torch.exp(-r3 / self.process_param_l)
        processed = torch.stack((y2 + y3, torch.sqrt(y2**2 + y3**2), y1), dim=1)
        scaled = (processed - self.scaler_min) / self.scaler_scale
        out = F.relu(self.fc1(scaled))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Instantiate and load model
input_dim = 3
neuron = 64
process_param_l = 1.5
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.array([0, 0, 0]), np.array([1, 1, 1])  # Replace with actual min and scale
model = SimpleModel(input_dim, neuron, scaler.min_, scaler.scale_, process_param_l).to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

# Generate test data
r1 = np.linspace(0, 7.0, 100)
r2 = np.linspace(0, 4.0, 100)
theta_values = np.linspace(0, np.pi, 20)
fig, axes = plt.subplots(5, 4, figsize=(24, 20))
axes = axes.flatten()

# Pre-compute predictions
all_predicted_energy = []
for theta in theta_values:
    r1_grid, r2_grid = np.meshgrid(r1, r2)
    r3_grid = derive_r3(r1_grid, r2_grid, theta)
    test_input = np.vstack([r1_grid.flatten(), r2_grid.flatten(), r3_grid.flatten()]).T
    test_input_tensor = torch.from_numpy(test_input).float().to(device)
    with torch.no_grad():
        predicted_energy = model(test_input_tensor).cpu().numpy()
    all_predicted_energy.append(predicted_energy)

# Calculate global min and max for consistent color mapping
all_predicted_energy = np.array(all_predicted_energy)
global_min, global_max = all_predicted_energy.min(), all_predicted_energy.max()

# Plot each theta slice
for idx, theta in enumerate(theta_values):
    predicted_energy_grid = all_predicted_energy[idx].reshape(r1_grid.shape)
    ax = axes[idx]
    plot_cross_section(ax, r1_grid, r2_grid, predicted_energy_grid, f'θ = {theta:.2f} rad', vmin=global_min, vmax=global_max)

# Remove extra subplots
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and add colorbar
plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
norm = Normalize(vmin=global_min, vmax=global_max)
sm = ScalarMappable(norm=norm, cmap='coolwarm')
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Potential Energy (eV)', fontsize=14)

plt.suptitle('Potential Energy Surfaces for Different Theta Values', fontsize=20)
plt.savefig('PES_sections_contour.png', dpi=300, bbox_inches='tight')
plt.show()
