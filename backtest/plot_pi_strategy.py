"""
绘制 pinn_hjb_stock.py 中的投资组合策略 π
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn_hjb_stock import PINNHJBSolver


def plot_pi_strategy(solver, t_fixed=0.0, save_path='pi_strategy.png'):
    """
    绘制固定时间 t 下的投资组合策略 π
    
    Args:
        solver: 训练好的 PINNHJBSolver 实例
        t_fixed: 固定的时间值
        save_path: 保存路径
    """
    device = solver.device
    
    # 创建网格
    Nr, NX = 50, 50
    r_grid = np.linspace(solver.r_min, solver.r_max, Nr)
    X_grid = np.linspace(solver.X_min, solver.X_max, NX)
    
    R, X = np.meshgrid(r_grid, X_grid, indexing='ij')
    
    # 创建输入张量
    t_tensor = torch.full((Nr * NX, 1), t_fixed, device=device)
    r_tensor = torch.tensor(R.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    X_tensor = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    
    x_eval = torch.cat([t_tensor, r_tensor, X_tensor], dim=1)
    
    # 计算 π (Merton 比例)
    # π = (μ - r) / (σ² * (1-θ))
    gamma = 1.0 - solver.theta
    r_batch = x_eval[:, 1]
    
    if solver.sigma > 0:
        pi_merton = (solver.mu - r_batch) / (solver.sigma**2 * gamma)
    else:
        pi_merton = torch.zeros_like(r_batch)
    
    # 限制在 [-1, 1] 范围内
    pi = torch.clamp(pi_merton, min=-1.0, max=1.0)
    
    pi_np = pi.cpu().numpy().reshape(Nr, NX)
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 3D 表面图
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax1.plot_surface(R, X, pi_np, cmap='coolwarm', edgecolor='none')
    ax1.set_xlabel('Interest Rate (r)')
    ax1.set_ylabel('Wealth (X)')
    ax1.set_zlabel('Portfolio Strategy (π)')
    ax1.set_title(f'Portfolio Strategy π at t={t_fixed}')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='π')
    
    # 2. 2D 等高线图
    ax2 = fig.add_subplot(1, 3, 2)
    contour = ax2.contourf(R, X, pi_np, levels=20, cmap='coolwarm')
    ax2.contour(R, X, pi_np, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax2.set_xlabel('Interest Rate (r)')
    ax2.set_ylabel('Wealth (X)')
    ax2.set_title(f'Portfolio Strategy π Contour at t={t_fixed}')
    fig.colorbar(contour, ax=ax2, label='π')
    
    # 3. 单变量曲线：固定 r，π 随 X 变化
    ax3 = fig.add_subplot(1, 3, 3)
    
    # 选择几个不同的 r 值
    r_values = [0.02, 0.05, 0.08, 0.10, 0.15]
    colors = plt.cm.viridis(np.linspace(0, 1, len(r_values)))
    
    for r_val, color in zip(r_values, colors):
        # 找到最接近的 r 索引
        r_idx = np.argmin(np.abs(r_grid - r_val))
        pi_slice = pi_np[r_idx, :]
        ax3.plot(X_grid, pi_slice, color=color, linewidth=2, label=f'r={r_grid[r_idx]:.3f}')
    
    ax3.set_xlabel('Wealth (X)')
    ax3.set_ylabel('Portfolio Strategy (π)')
    ax3.set_title(f'π vs Wealth (at t={t_fixed})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 设置 y 轴范围，完整显示
    y_min, y_max = pi_np.min(), pi_np.max()
    margin = (y_max - y_min) * 0.1
    ax3.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 策略图已保存: {save_path}")
    
    # 打印统计信息
    print(f"\n=== Portfolio Strategy π Statistics (t={t_fixed}) ===")
    print(f"Mean: {np.mean(pi_np):.6f}")
    print(f"Std:  {np.std(pi_np):.6f}")
    print(f"Min:  {np.min(pi_np):.6f}")
    print(f"Max:  {np.max(pi_np):.6f}")
    
    return pi_np


def plot_pi_vs_r(solver, X_fixed=2.0, save_path='pi_vs_r.png'):
    """
    绘制固定财富 X 下，π 随利率 r 变化的曲线
    
    Args:
        solver: 训练好的 PINNHJBSolver 实例
        X_fixed: 固定的财富值
        save_path: 保存路径
    """
    device = solver.device
    
    # 创建 r 网格
    r_grid = np.linspace(solver.r_min, solver.r_max, 100)
    
    # 计算 π
    gamma = 1.0 - solver.theta
    r_tensor = torch.tensor(r_grid, dtype=torch.float32, device=device)
    
    if solver.sigma > 0:
        pi_merton = (solver.mu - r_tensor) / (solver.sigma**2 * gamma)
    else:
        pi_merton = torch.zeros_like(r_tensor)
    
    pi = torch.clamp(pi_merton, min=-1.0, max=1.0)
    pi_np = pi.cpu().numpy()
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_grid, pi_np, 'b-', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=solver.mu, color='red', linestyle='--', alpha=0.5, label=f'μ={solver.mu}')
    
    ax.set_xlabel('Interest Rate (r)', fontsize=12)
    ax.set_ylabel('Portfolio Strategy (π)', fontsize=12)
    ax.set_title(f'Portfolio Strategy vs Interest Rate (X={X_fixed})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 设置 y 轴范围，完整显示
    y_min, y_max = pi_np.min(), pi_np.max()
    margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    # plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ π-r 曲线图已保存: {save_path}")
    
    return pi_np


def main():
    """主函数"""
    params = {
        # 经济参数
        'theta': 0.7,
        'mu': 0.08,
        'sigma': 0.2,
        'sigma_r': 0.05,
        
        # 利率过程参数
        'alpha': 0.02,
        'beta': 0.1,
        
        # 财富扩散参数
        'A': 0.1,
        
        # 股票指数和贴现因子参数
        'phi0': 0.2,
        'phi1': 0.1,
        'phi2': 0.05,
        
        # 股票指数过程参数
        'S0': 1.0,
        'mu_s': 0.05,
        
        # 网格参数
        't_min': 0.0,
        't_max': 1.0,
        'r_min': 0.01,
        'r_max': 1.0,
        'w_min': 0.5,
        'w_max': 4.0,
        
        # 训练参数
        'epochs': 1000,  # 减少训练轮数以快速绘图
        'batch_size': 2048,
        'lr': 1e-3
    }
    
    print("创建求解器...")
    solver = PINNHJBSolver(params)
    
    print("开始训练...")
    solver.train()
    
    print("\n绘制策略图...")
    plot_pi_strategy(solver, t_fixed=0.0, save_path='pi_strategy_t0.png')
    plot_pi_strategy(solver, t_fixed=0.5, save_path='pi_strategy_t05.png')
    
    print("\n绘制 π-r 曲线...")
    plot_pi_vs_r(solver, X_fixed=2.0, save_path='pi_vs_r.png')
    
    print("\n完成!")


if __name__ == "__main__":
    main()
