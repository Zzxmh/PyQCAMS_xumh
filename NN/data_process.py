import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy as copy
import pandas as pd

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
    ax.set_xlabel('$r_{OO}$')
    ax.set_ylabel('$r_{OS}$')
    
    # 添加颜色条
    plt.colorbar(cfill, ax=ax)

def extract_valid_data(output, r1_array, r2_array, theta_array):
    """
    从 output 数组中提取有效的数据点（非 NaN）。
    对能量进行标准化处理：
    1. 每个能量值加上 546.78
    2. 将最低能量平移至 0 点
    返回特征（r1, r2, r3）和标签（energy）的数组。
    """
    # 获取所有有效的索引
    valid_indices = np.where(~np.isnan(output))
    if valid_indices[0].size == 0:
        print("未找到有效的能量数据。")
        return np.array([]), np.array([])
    
    # 提取有效能量值
    valid_energies = output[valid_indices]
    
    # 提取 r1, r2
    r1s = r1_array[valid_indices[2]]
    r2s = r2_array[valid_indices[1]]
    
    # 计算 r3
    thetas = theta_array[valid_indices[0]]
    r3s = derive_r3(r1s, r2s, thetas)
    
    # 组合特征
    features = np.stack((r1s, r2s, r3s), axis=-1)
    labels = valid_energies
    
    return features, labels

def main():
    folder_path = 'NN/dataset_visualize'
    data_path = 'NN/dataset'
    r1_array = np.linspace(1.3, 3.5, 20)
    r2_array = np.linspace(1, 7, 60)
    theta_array = np.linspace(0, np.pi, 20)
    lr1 = len(r1_array)
    lr2 = len(r2_array)
    lt = len(theta_array)
    
    # 确保输出文件夹存在
    os.makedirs(folder_path, exist_ok=True)
    
    for a in range(1, 11):
        for b in range(1, 3):
            file_name = f'PLOT_{a}_{b}_contour.png'
            output = np.full((lt, lr2, lr1), np.nan)  # 初始化为 NaN
            filename = f'E:\\tasks\\documents_in_pku\\research\\roaming\\process_3_asymmetrical\\output_{a}_{b}.txt'  # 数据文件路径
            
            try:
                energy_data = parse_energy_data(filename)
                sorted_energies = sort_by_number(energy_data)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                continue

            # 填充 output 数组
            for file_index, energy in sorted_energies:
                t_index = (file_index - 1) // (lr1 * lr2)
                r1_index = ((file_index - 1) % (lr1 * lr2)) // lr2
                r2_index = ((file_index - 1) % (lr1 * lr2)) % lr2

                # 仅当 t_index, r1_index, r2_index 在有效范围内时赋值
                if t_index < lt and r1_index < lr1 and r2_index < lr2:
                    output[t_index, r2_index, r1_index] = energy

            # 删除能量过高的数据点（设置为 NaN）
            # 假设能量过高的阈值为 -546.78
            output[output >= -546.78] = np.nan  # 将能量 >= -546.78 的点设置为 NaN

            # 提取有效的数据点并进行标准化处理
            features, labels = extract_valid_data(output, r1_array, r2_array, theta_array)

            # 将数据保存为 TXT 文件（不包含 theta）
            if features.size > 0:
                data_to_save = np.hstack((features, labels.reshape(-1, 1)))  # 合并特征和标签
                save_file_name = f'energy_surface_{a}_{b}.txt'
                save_file_path = os.path.join(data_path, save_file_name)
                header = "r1 r2 r3 energy"
                # 使用 fmt 参数避免科学计数法，设置为保留6位小数
                np.savetxt(save_file_path, data_to_save, delimiter=' ', header=header, comments='', fmt='%.6f')
                print(f"已保存能级 {a}, 状态 {b} 的势能面数据到 {save_file_path}")
            else:
                print(f"能级 {a}, 状态 {b} 没有有效的能量数据可保存。")

            # 创建网格用于绘图
            X, Y = np.meshgrid(r2_array, r1_array, indexing='ij')
            
            # 创建图形和子图
            fig, axes = plt.subplots(5, 4, figsize=(18, 15))  # 根据需要调整网格大小
            axes = axes.flatten()  # 展平二维数组以简化循环
            
            # 绘制每个 theta 的截面
            for i, ax in enumerate(axes):
                if i < len(theta_array):
                    plot_cross_section(ax, X, Y, output[i, :, :], f'Theta = {theta_array[i]:.2f} rad')
                else:
                    ax.axis('off')  # 关闭未使用的子图
            
            # 设置图形标题
            if b == 1:
                fig.suptitle(f'Potential Energy Surface of State {a} A\'', fontsize=16)
            else:
                fig.suptitle(f'Potential Energy Surface of State {a} A\'\'', fontsize=16)
            
            # 调整布局以防止重叠
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # 保存图像
            file_path = os.path.join(folder_path, file_name)
            plt.savefig(file_path)
            plt.close(fig)
            print(f"已保存势能面等高线图到 {file_path}")

if __name__ == "__main__":
    main()
