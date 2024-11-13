import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
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

# 初始化 wandb
wandb.init(
    project="energy_prediction_project_refined_data",  # 替换为您的项目名称
    config={
        "input_dim": 3,
        "neuron": 64,
        "learning_rate": 0.0005,
        "batch_size": 256,
        "num_epochs": 1000,
        "patience": 20,
        "scaler": "MinMaxScaler",
        "optimizer": "Adam",
        "loss_function": "MSELoss",
        "data_file": "energy_surface_1_1.txt",
        "process_param_l": 1.5,
        "unit_conversion": "au_to_eV",
        # "noise_mean": noise_mean,
        # "noise_std": noise_std,
        # "scale_min": scale_min,
        # "scale_max": scale_max,
        # "shift_min": shift_min,
        # "shift_max": shift_max
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
    def __init__(self, input_dim, neuron, scaler_min, scaler_scale, process_param_l=0.5):
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

        # Register scaler parameters as buffers (non-trainable)
        self.register_buffer('scaler_min', torch.tensor(scaler_min, dtype=torch.float32))
        self.register_buffer('scaler_scale', torch.tensor(scaler_scale, dtype=torch.float32))

        # Define MLP layers
        self.fc1 = nn.Linear(input_dim, neuron)
        self.fc2 = nn.Linear(neuron, neuron)
        self.fc3 = nn.Linear(neuron, neuron)
        self.fc4 = nn.Linear(neuron, 1)
        
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

        # Apply MinMax scaling: (x - min) / scale
        scaled = (processed - self.scaler_min) / self.scaler_scale

        # Pass through MLP layers with ReLU activations
        out = F.relu(self.fc1(scaled))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

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
            # Removed per-batch print statements for efficiency
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
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                # Removed per-batch print statements for efficiency
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
            torch.save(best_model_wts, model_save_path)
            print(f"Best model updated at epoch {epoch + 1} with Test Loss: {mean_test_loss:.6f}")
            wandb.save(model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, model_save_path)

    return history

# Data loading and preprocessing

# Use os.path.join to construct the full path
data_dir = 'NN/dataset'

# Get the specific text file
data_files = os.path.join(data_dir, wandb.config.data_file)

# Initialize data list
data_list = []

# 读取文件并添加到数据列表
try:
    temp_data = pd.read_csv(data_files, delim_whitespace=True, header=None)
    data_list.append(temp_data)
    print(f"成功读取文件: {data_files}")
except Exception as e:
    print(f"读取文件 {data_files} 时出错: {e}")

# 检查是否有数据被读取
if not data_list:
    raise ValueError("未读取到任何数据文件。请检查文件路径和格式。")

# 连接所有数据
data_df = pd.concat(data_list, ignore_index=True)
print("所有数据已成功连接。")

# 将所有列转换为浮点数，无法转换的设置为 NaN
data_df = data_df.apply(pd.to_numeric, errors='coerce')

# 删除任何包含 NaN 的行
data_df.dropna(inplace=True)
print(f"数据清理后样本数量: {data_df.shape[0]}")

# 提取特征和标签
X = data_df.iloc[:, :3].values.astype(np.float32)  # 前三列为坐标，转换为浮点数
y = data_df.iloc[:, 3].values.reshape(-1, 1).astype(np.float32)  # 第四列为潜在能量，转换为浮点数

# 打印 X 和 y 的数据类型和部分数据以验证
print(f"X 的数据类型: {X.dtype}")
print(f"X 的前5个样本:\n{X[:5]}")
print(f"y 的数据类型: {y.dtype}")
print(f"y 的前5个值:\n{y[:5]}")

# 将标签从 au 转换为 eV
y_eV = y   # 使用定义的转换因子

# 打印转换后的 y_eV 以验证
print(f"y_eV 的数据类型: {y_eV.dtype}")
print(f"y_eV 的前5个值:\n{y_eV[:5]}")
process_param_l = wandb.config.process_param_l 
# 特征缩放
scaler = MinMaxScaler()
data = scaler.fit_transform(X)
print("特征数据已缩放。")

# 标签缩放
label_scaler = MinMaxScaler()
labels = label_scaler.fit_transform(y_eV)  # 使用转换后的 eV 标签
print("标签数据已缩放。")

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
# Obtain a sample batch and move it to the GPU

test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False)

# Instantiate the model with scaler parameters
input_dim = wandb.config.input_dim
neuron = wandb.config.neuron
model = SimpleModel(
    input_dim,
    neuron,
    scaler_min=scaler.min_,
    scaler_scale=scaler.scale_,
    process_param_l=process_param_l
).to(device)

# Verify model's device
print(f"Model is on device: {next(model.parameters()).device}")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
num_epochs = wandb.config.num_epochs
patience = wandb.config.patience
# To log the model graph, you need a sample input
sample_input, _ = next(iter(train_loader))
sample_input = sample_input.to(device)  # Move to GPU
print(f"Sample input device: {sample_input.device}")  # Should output cuda:0
 # Diagnostic
wandb.watch(model, log="all", log_freq=100)

# Training
start_time = time.time()
# Fit label_scaler on the training labels before passing to the model


# Now, pass the fitted label_scaler to the training function
history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, patience)
end_time = time.time()

# Plotting training history
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['test_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'loss_history.png'))
plt.show()

print(f'Training completed in {end_time - start_time:.2f} seconds')
print(f"Best Test Loss: {history['test_loss'][-1]:.6f}")

# Finish the wandb run
wandb.finish()