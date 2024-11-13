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
        out = F.tanh(self.fc1(scaled))
        out = F.tanh(self.fc2(out))
        out = F.tanh(self.fc2(out))
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
data_dir = 'NN/dataset'
data_files = os.path.join(data_dir, wandb.config.data_file)
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
data_df = data_df.apply(pd.to_numeric, errors='coerce')
data_df.dropna(inplace=True)
print(f"数据清理后样本数量: {data_df.shape[0]}")

# 提取特征和标签
X = data_df.iloc[:, :3].values.astype(np.float32)  # 前三列为坐标，转换为浮点数
y = data_df.iloc[:, 3].values.reshape(-1, 1).astype(np.float32)  # 第四列为潜在能量，转换为浮点数

# 将标签从 au 转换为 eV
y_eV = y   # 使用定义的转换因子

# 特征缩放
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("特征数据已缩放。")

# 标签缩放
label_scaler = MinMaxScaler()
y_scaled = label_scaler.fit_transform(y_eV)  # 使用转换后的 eV 标签
print("标签数据已缩放。")

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# Convert to PyTorch tensors
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
    scaler_min=scaler.min_,
    scaler_scale=scaler.scale_
).to(device)

# Set optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
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

wandb.finish()
