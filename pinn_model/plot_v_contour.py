"""
绘制值函数 V 的等高线图（类似参考图风格）
"""

import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.PINN import PINN
from utils import load_yaml
from data.data_loder import calculate_parements_stock
from plot_model_results import load_model, evaluate_model


def plot_v_contour_slices(model, cfg, paraments, save_path):
    """
    绘制值函数 V 在不同时间片的等高线图（类似参考图风格）
    
    Args:
        model: 加载好的模型
        cfg: 配置
        paraments: 参数
        save_path: 保存路径
    """
    device = torch.device(cfg.device)
    n_assets = len(paraments)
    
    # 创建网格（参考图的范围）
    Nw, Nr = 50, 50
    w_grid = np.linspace(0.5, 2.0, Nw)  # 财富范围
    r_grid = np.linspace(0.01, 4, Nr)  # 利率范围（参考图是0.02-0.10）
    
    # 5个时间点 + 1个时间序列图
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 存储特定点的V值用于时间序列图
    r_fixed_idx = Nr // 2  # 中间位置的r
    w_fixed_idx = Nw // 2  # 中间位置的w
    r_fixed = r_grid[r_fixed_idx]
    w_fixed = w_grid[w_fixed_idx]
    V_vs_time = []
    
    for idx, t_val in enumerate(time_points):
        V, _, _, W, R = evaluate_model(model, t_val, w_grid, r_grid, n_assets, device)
        
        # 等高线图
        ax = axes[idx]
        levels = 20
        cs = ax.contourf(R, W, V, levels=levels, cmap='viridis')
        ax.contour(R, W, V, levels=levels, colors='k', linewidths=0.5, alpha=0.3)
        ax.set_xlabel('Interest Rate (r)')
        ax.set_ylabel('Wealth (w)')
        ax.set_title(f'Value Function V at t={t_val:.2f}')
        cbar = plt.colorbar(cs, ax=ax)
        cbar.set_label('V')
        
        # 记录特定点的V值
        V_vs_time.append(V[w_fixed_idx, r_fixed_idx])
    
    # 时间序列图：V vs Time (固定 r 和 w)
    ax_time = axes[5]
    ax_time.plot(time_points, V_vs_time, 'b-', linewidth=2)
    ax_time.set_xlabel('Time (t)')
    ax_time.set_ylabel('V')
    ax_time.set_title(f'V vs Time (r={r_fixed:.3f}, w={w_fixed:.2f})')
    ax_time.grid(True, alpha=0.3)
    
    plt.suptitle('Value Function V Contour Slices', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ V等高线图已保存: {save_path}")


if __name__ == "__main__":
    # 加载配置
    cfg = load_yaml('config.yaml')
    
    # 加载参数
    paraments = calculate_parements_stock(cfg.csv_path, cfg.result_path, save_csv=False)
    n_assets = len(paraments)
    print(f"总资产数量: {n_assets}")
    
    # 加载模型
    device = torch.device(cfg.device)
    model_path = os.path.join(cfg.model_dir, "pinn_model_test1.pth")
    model = load_model(model_path, cfg, device)
    
    if model is not None:
        # 绘制等高线图
        save_path = os.path.join(cfg.log_dir, "v_contour_slices.png")
        plot_v_contour_slices(model, cfg, paraments, save_path)
    else:
        print("❌ 模型加载失败")
