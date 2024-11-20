import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy as copy
import pandas as pd
import seaborn as sns

def derive_r3(r1, r2, theta):
    """
    根据 r1, r2 和 theta 计算 r3。
    """
    return (r1**2 + r2**2 + 2*r1*r2*np.cos(theta))**0.5

def parse_energy_data(filename):
    """
    读取文件并解析每行的能量数据。
    """
    energy_data = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # 跳过无效行
            file_path = parts[0][:-1]  # 移除末尾的冒号
            try:
                energy = float(parts[-1])  # 最后一个元素是能量值
                # 假设文件名格式为 '/home/xumh/SO2_PES/inp/SO2_202.out'
                file_index_str = file_path.split('_')[-1].split('.')[0]
                file_index = int(file_index_str)
                energy_data[file_index] = energy
            except (ValueError, IndexError) as e:
                print(f"跳过解析错误的行: {line.strip()} | 错误: {e}")
                continue
    return energy_data

def sort_by_number(energy_data):
    """
    根据文件索引对能量数据进行排序。
    返回一个包含 (file_index, energy) 元组的列表。
    """
    sorted_energies = sorted(energy_data.items())  # 按文件索引排序
    return sorted_energies

def plot_cross_section(ax, x, y, values, title):
    """
    绘制等高线图，自动忽略 NaN 值。
    """
    # 创建掩膜，忽略 NaN 值
    masked_values = np.ma.masked_invalid(values)
    
    # 绘制等高线填充
    cfill = ax.contourf(x, y, masked_values, levels=50, cmap='coolwarm')
    
    # 绘制等高线
    contour = ax.contour(x, y, masked_values, levels=50, colors='black', linewidths=0.5)
    
    # 添加等高线标签
    ax.clabel(contour, inline=1, fontsize=8, colors='black')
    
    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel('$r_{2}$')
    ax.set_ylabel('$r_{3}$')
    
    # 添加颜色条
    plt.colorbar(cfill, ax=ax)

def extract_valid_data(output, r2_array, r3_array, theta_array):
    """
    提取有效的数据点并返回特征和标签。保留 NaN 数据。
    """
    # 获取所有的索引（包括 NaN 数据）
    indices = np.indices(output.shape)
    
    # 提取 r2, r3
    r2s = r2_array[indices[2]]
    r3s = r3_array[indices[1]]
    
    # 计算 r1
    thetas = theta_array[indices[0]]
    r1s = derive_r3(r2s, r3s, thetas)
    
    # 将所有数据（包括 NaN）组合成特征和标签
    features = np.stack((r1s, r2s, r3s), axis=-1)
    labels = output.flatten()  # 保留 NaN，直接返回整个标签数组
    
    # Flatten features array to 2D
    features_flattened = features.reshape(-1, features.shape[-1])  # Flatten the first three dimensions
    
    # Ensure labels are 2D (N, 1)
    labels = labels.reshape(-1, 1)  # Reshape to (N, 1)
    
    return features_flattened, labels



def main():
    data_folder_path = 'NN/dataset'
    fig_folder_path = 'NN/dataset_visualize'
    r2_array = np.linspace(1.3, 3.5, 30)  # S-O
    r3_array = np.linspace(1.3, 3.5, 30)  # S-O
    theta_array = np.linspace(0, np.pi, 30)
    lr2 = len(r2_array)
    lr3 = len(r3_array)
    lt = len(theta_array)
    threshold = -540
    inner_threshold_list = [
        [-546.78, -546.72, -546.72, -546.70, -546.70, -546.71, -546.69, -546.67, -546.65, -546.64],
        [-546.77, -546.71, -546.71, -546.70, -546.69, -546.69, -546.67, -546.65, -546.63, -546.63]
    ]
    alpha = 0.01
    beta = 0.05
    
    # 确保输出文件夹存在
    os.makedirs(data_folder_path, exist_ok=True)
    os.makedirs(fig_folder_path, exist_ok=True)
    
    for a in range(1, 11):
        for b in range(1, 3):
            inner_threshold = inner_threshold_list[b - 1][a - 1]
            file_name = f'PLOT_{a}_{b}_contour.png'
            output = np.full((lt, lr3, lr2), np.nan)  # 初始化为 NaN
            filename = f'E:\\tasks\\documents_in_pku\\research\\roaming\\process_2\\output_{a}_{b}.txt'  # 数据文件路径
            energies = []
            try:
                energy_data = parse_energy_data(filename)
                sorted_energies = sort_by_number(energy_data)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                continue

            # 填充 output 数组
            for file_index, energy in sorted_energies:
                t_index = (file_index - 1) // (lr2 * lr3)
                r2_index = ((file_index - 1) % (lr2 * lr3)) // lr2
                r3_index = ((file_index - 1) % (lr2 * lr3)) % lr2
                # 仅当 t_index, r1_index, r2_index 在有效范围内时赋值
                if t_index < lt and r2_index < lr2 and r3_index < lr3:
                    if energy <= threshold:
                        if energy <= inner_threshold:
                            output[t_index, r3_index, r2_index] = energy
                        else:
                            output[t_index, r3_index, r2_index] = inner_threshold + 0.01*(energy - inner_threshold) /(threshold-inner_threshold)
                    else:
                        # output[t_index, r3_index, r2_index] = threshold + alpha/(1 + 10*np.exp(-beta*(energy - threshold)))
                        output[t_index, r3_index, r2_index] = np.nan
                        # output[t_index, r3_index, r2_index] = threshold
                    if t_index == 0 or t_index == lt - 1: 
                        output[t_index, r3_index, r2_index] = np.nan
                energies.append(output[t_index, r3_index, r2_index])

                # 对大于阈值的部分应用指数变化
            # plt.figure(figsize=(10, 6))
            # sns.violinplot(data=energies)
            # plt.ylabel("Adiabatic (au)")
            # plt.xlabel("u")
            # plt.show()

            # 提取有效的数据点并进行标准化处理
            features, labels = extract_valid_data(output, r2_array, r3_array, theta_array)

            # 将数据保存为 TXT 文件（不包含 theta）
            if features.size > 0:
                # Flatten the features array to 2D (flatten the first three dimensions)
                features_flattened = features.reshape(-1, features.shape[-1])  # Shape will be (N, 3) where N is the number of samples
                
                # Combine features and labels (ensure both are 2D)
                data_to_save = np.hstack((features_flattened, labels.reshape(-1, 1)))  # 合并特征和标签
                save_file_name = f'energy_surface_{a}_{b}.txt'
                save_file_path = os.path.join(data_folder_path, save_file_name)
                header = "r1 r2 r3 energy"
                
                # 使用 fmt 参数避免科学计数法，设置为保留6位小数
                np.savetxt(save_file_path, data_to_save, delimiter=' ', header=header, comments='', fmt='%.6f')
                print(f"已保存能级 {a}, 状态 {b} 的势能面数据到 {save_file_path}")
            else:
                print(f"能级 {a}, 状态 {b} 没有有效的能量数据可保存。")

if __name__ == "__main__":
    main()
