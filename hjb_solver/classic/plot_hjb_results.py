"""
HJB求解结果可视化脚本
调用 hjb_solver_implicit.py 进行求解并绘制多种图表
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# 将当前目录加入路径，以便找到同目录的求解器
sys.path.insert(0, os.path.dirname(__file__))

# 导入求解器
from hjb_solver_implicit import HJBSolverImplicitFDM


def run_solver(params=None):
    """
    运行HJB求解器
    
    Parameters:
    -----------
    params : dict, optional
        求解参数，如果为None则使用默认参数
    
    Returns:
    --------
    solver : HJBSolverImplicitFDM
        求解器实例，包含V, c_opt, pi_opt等结果
    """
    if params is None:
        params = {
            't_min': 0.0,
            't_max': 1.0,
            'r_min': 0.01,
            'r_max': 0.10,
            'w_min': 0.5,
            'w_max': 2.0,
            'Nt': 30,
            'Nr': 20,
            'Nw': 20,
            'theta': 0.7,
            'rho': 0.5,
            'mu': 0.08,
            'sigma': 0.2,
            'sigma_r': 0.05,
            'kappa': 0.1,
            'r_bar': 0.05,
            'max_iter_nr': 30,
            'tol_nr': 1e-6,
            'damping': 0.7
        }
    
    print("=" * 70)
    print("运行 HJB 隐式求解器")
    print("=" * 70)
    
    solver = HJBSolverImplicitFDM(params)
    V, c_opt, pi_opt = solver.solve_backward()
    
    return solver


def plot_value_function_slices(solver, save_path='hjb_v_slices.png'):
    """
    绘制值函数在不同时间截面的2D等高线图
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    time_indices = [0, solver.Nt//4, solver.Nt//2, 3*solver.Nt//4, solver.Nt-1]
    
    R, W = np.meshgrid(solver.r, solver.w, indexing='ij')
    
    for idx, (ax, t_idx) in enumerate(zip(axes[:-1], time_indices)):
        t_val = solver.t[t_idx]
        V_slice = solver.V[t_idx]
        
        # 绘制等高线
        levels = np.linspace(V_slice.min(), V_slice.max(), 20)
        cs = ax.contourf(R, W, V_slice, levels=levels, cmap='viridis')
        ax.contour(R, W, V_slice, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel('Interest Rate (r)', fontsize=10)
        ax.set_ylabel('Wealth (w)', fontsize=10)
        ax.set_title(f'Value Function V at t={t_val:.2f}', fontsize=11)
        
        # 添加颜色条
        cbar = plt.colorbar(cs, ax=ax)
        cbar.set_label('V', fontsize=9)
    
    # 最后一个子图：值函数随时间变化（固定r和w）
    ax = axes[-1]
    r_fixed_idx = solver.Nr // 2
    w_fixed_idx = solver.Nw // 2
    
    ax.plot(solver.t, solver.V[:, r_fixed_idx, w_fixed_idx], 'b-', linewidth=2)
    ax.set_xlabel('Time (t)', fontsize=10)
    ax.set_ylabel('V', fontsize=10)
    ax.set_title(f'V vs Time (r={solver.r[r_fixed_idx]:.3f}, w={solver.w[w_fixed_idx]:.2f})', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"值函数截面图已保存: {save_path}")


def plot_consumption_policy(solver, save_path='hjb_c_policy.png'):
    """
    绘制最优消费策略的多种可视化
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 选择时间点
    time_indices = [0, solver.Nt//2, solver.Nt-1]
    
    for idx, t_idx in enumerate(time_indices):
        t_val = solver.t[t_idx]
        
        # 3D表面图
        ax1 = fig.add_subplot(3, 3, idx+1, projection='3d')
        R, W = np.meshgrid(solver.r, solver.w, indexing='ij')
        surf = ax1.plot_surface(R, W, solver.c_opt[t_idx], cmap='plasma', alpha=0.8)
        ax1.set_xlabel('r')
        ax1.set_ylabel('w')
        ax1.set_zlabel('c')
        ax1.set_title(f'Optimal Consumption (3D) at t={t_val:.2f}')
        
        # 等高线图
        ax2 = fig.add_subplot(3, 3, idx+4)
        c_slice = solver.c_opt[t_idx]
        levels = np.linspace(c_slice.min(), c_slice.max(), 15)
        cs = ax2.contourf(R, W, c_slice, levels=levels, cmap='plasma')
        ax2.contour(R, W, c_slice, levels=levels, colors='white', alpha=0.5, linewidths=0.5)
        ax2.set_xlabel('Interest Rate (r)')
        ax2.set_ylabel('Wealth (w)')
        ax2.set_title(f'Optimal Consumption (Contour) at t={t_val:.2f}')
        plt.colorbar(cs, ax=ax2)
        
        # 沿w方向的切片（固定r）
        ax3 = fig.add_subplot(3, 3, idx+7)
        r_fixed_idx = solver.Nr // 2
        ax3.plot(solver.w, solver.c_opt[t_idx, r_fixed_idx, :], 'r-', linewidth=2)
        ax3.set_xlabel('Wealth (w)')
        ax3.set_ylabel('c')
        ax3.set_title(f'c vs w (r={solver.r[r_fixed_idx]:.3f}, t={t_val:.2f})')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"消费策略图已保存: {save_path}")


def plot_portfolio_policy(solver, save_path='hjb_pi_policy.png'):
    """
    绘制最优投资组合策略的多种可视化
    """
    fig = plt.figure(figsize=(16, 10))
    
    time_indices = [0, solver.Nt//2, solver.Nt-1]
    
    for idx, t_idx in enumerate(time_indices):
        t_val = solver.t[t_idx]
        
        # 3D表面图
        ax1 = fig.add_subplot(2, 3, idx+1, projection='3d')
        R, W = np.meshgrid(solver.r, solver.w, indexing='ij')
        surf = ax1.plot_surface(R, W, solver.pi_opt[t_idx], cmap='coolwarm', alpha=0.8)
        ax1.set_xlabel('r')
        ax1.set_ylabel('w')
        ax1.set_zlabel('π')
        ax1.set_title(f'Portfolio Weight π at t={t_val:.2f}')
        
        # 等高线图
        ax2 = fig.add_subplot(2, 3, idx+4)
        pi_slice = solver.pi_opt[t_idx]
        # 对称的颜色范围
        vmax = max(abs(pi_slice.min()), abs(pi_slice.max()))
        levels = np.linspace(-vmax, vmax, 21)
        cs = ax2.contourf(R, W, pi_slice, levels=levels, cmap='coolwarm')
        ax2.contour(R, W, pi_slice, levels=[0], colors='black', linewidths=2)
        ax2.set_xlabel('Interest Rate (r)')
        ax2.set_ylabel('Wealth (w)')
        ax2.set_title(f'Portfolio Weight (Contour) at t={t_val:.2f}')
        plt.colorbar(cs, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"投资组合策略图已保存: {save_path}")


def plot_comparison_time_evolution(solver, save_path='hjb_time_evolution.png'):
    """
    绘制值函数和控制变量随时间的演化
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 选择固定的 (r, w) 点
    r_idx = solver.Nr // 2
    w_idx = solver.Nw // 2
    
    r_val = solver.r[r_idx]
    w_val = solver.w[w_idx]
    
    # 值函数随时间变化
    ax = axes[0, 0]
    ax.plot(solver.t, solver.V[:, r_idx, w_idx], 'b-', linewidth=2)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('V(t, r, w)')
    ax.set_title(f'Value Function Evolution (r={r_val:.3f}, w={w_val:.2f})')
    ax.grid(True, alpha=0.3)
    
    # 最优消费随时间变化
    ax = axes[0, 1]
    ax.plot(solver.t, solver.c_opt[:, r_idx, w_idx], 'r-', linewidth=2)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('c(t, r, w)')
    ax.set_title(f'Optimal Consumption Evolution (r={r_val:.3f}, w={w_val:.2f})')
    ax.grid(True, alpha=0.3)
    
    # 最优投资组合随时间变化
    ax = axes[1, 0]
    ax.plot(solver.t, solver.pi_opt[:, r_idx, w_idx], 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('π(t, r, w)')
    ax.set_title(f'Optimal Portfolio Evolution (r={r_val:.3f}, w={w_val:.2f})')
    ax.grid(True, alpha=0.3)
    
    # 多个 (r, w) 组合的比较
    ax = axes[1, 1]
    for r_i in [0, solver.Nr//2, solver.Nr-1]:
        for w_j in [0, solver.Nw//2, solver.Nw-1]:
            label = f'r={solver.r[r_i]:.2f}, w={solver.w[w_j]:.1f}'
            ax.plot(solver.t, solver.V[:, r_i, w_j], label=label, linewidth=1.5)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('V(t, r, w)')
    ax.set_title('Value Function Comparison (Multiple Points)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"时间演化图已保存: {save_path}")


def plot_wealth_slices(solver, save_path='hjb_wealth_analysis.png'):
    """
    分析不同财富水平下的策略
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t_idx = 0  # t=0时刻
    r_idx = solver.Nr // 2
    
    # 值函数 vs 财富
    ax = axes[0, 0]
    for r_i in [0, solver.Nr//2, solver.Nr-1]:
        ax.plot(solver.w, solver.V[t_idx, r_i, :], 
                label=f'r={solver.r[r_i]:.3f}', linewidth=2)
    ax.set_xlabel('Wealth (w)')
    ax.set_ylabel('V')
    ax.set_title(f'Value Function vs Wealth (t={solver.t[t_idx]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 消费 vs 财富
    ax = axes[0, 1]
    for r_i in [0, solver.Nr//2, solver.Nr-1]:
        ax.plot(solver.w, solver.c_opt[t_idx, r_i, :], 
                label=f'r={solver.r[r_i]:.3f}', linewidth=2)
    ax.set_xlabel('Wealth (w)')
    ax.set_ylabel('c')
    ax.set_title(f'Optimal Consumption vs Wealth (t={solver.t[t_idx]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 投资组合 vs 财富
    ax = axes[1, 0]
    for r_i in [0, solver.Nr//2, solver.Nr-1]:
        ax.plot(solver.w, solver.pi_opt[t_idx, r_i, :], 
                label=f'r={solver.r[r_i]:.3f}', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Wealth (w)')
    ax.set_ylabel('π')
    ax.set_title(f'Optimal Portfolio vs Wealth (t={solver.t[t_idx]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 消费/财富比率 vs 财富
    ax = axes[1, 1]
    for r_i in [0, solver.Nr//2, solver.Nr-1]:
        c_over_w = solver.c_opt[t_idx, r_i, :] / solver.w
        ax.plot(solver.w, c_over_w, 
                label=f'r={solver.r[r_i]:.3f}', linewidth=2)
    ax.set_xlabel('Wealth (w)')
    ax.set_ylabel('c/w (Consumption Rate)')
    ax.set_title(f'Consumption Rate vs Wealth (t={solver.t[t_idx]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"财富分析图已保存: {save_path}")


def plot_interest_rate_analysis(solver, save_path='hjb_rate_analysis.png'):
    """
    分析不同利率水平下的策略
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t_idx = 0
    w_idx = solver.Nw // 2
    
    # 值函数 vs 利率
    ax = axes[0, 0]
    for w_j in [0, solver.Nw//2, solver.Nw-1]:
        ax.plot(solver.r, solver.V[t_idx, :, w_j], 
                label=f'w={solver.w[w_j]:.2f}', linewidth=2)
    ax.set_xlabel('Interest Rate (r)')
    ax.set_ylabel('V')
    ax.set_title(f'Value Function vs Interest Rate (t={solver.t[t_idx]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 消费 vs 利率
    ax = axes[0, 1]
    for w_j in [0, solver.Nw//2, solver.Nw-1]:
        ax.plot(solver.r, solver.c_opt[t_idx, :, w_j], 
                label=f'w={solver.w[w_j]:.2f}', linewidth=2)
    ax.set_xlabel('Interest Rate (r)')
    ax.set_ylabel('c')
    ax.set_title(f'Optimal Consumption vs Interest Rate (t={solver.t[t_idx]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 投资组合 vs 利率
    ax = axes[1, 0]
    for w_j in [0, solver.Nw//2, solver.Nw-1]:
        ax.plot(solver.r, solver.pi_opt[t_idx, :, w_j], 
                label=f'w={solver.w[w_j]:.2f}', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Interest Rate (r)')
    ax.set_ylabel('π')
    ax.set_title(f'Optimal Portfolio vs Interest Rate (t={solver.t[t_idx]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 投资组合与风险溢价的关系
    ax = axes[1, 1]
    risk_premium = solver.mu - solver.r
    for w_j in [0, solver.Nw//2, solver.Nw-1]:
        ax.plot(risk_premium, solver.pi_opt[t_idx, :, w_j], 
                label=f'w={solver.w[w_j]:.2f}', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Risk Premium (μ - r)')
    ax.set_ylabel('π')
    ax.set_title(f'Portfolio vs Risk Premium (t={solver.t[t_idx]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"利率分析图已保存: {save_path}")


def generate_summary_statistics(solver):
    """
    生成求解结果的统计摘要
    """
    print("\n" + "=" * 70)
    print("求解结果统计摘要")
    print("=" * 70)
    
    print(f"\n网格设置:")
    print(f"  时间维度: Nt = {solver.Nt}, t ∈ [{solver.t_min}, {solver.t_max}]")
    print(f"  利率维度: Nr = {solver.Nr}, r ∈ [{solver.r_min}, {solver.r_max}]")
    print(f"  财富维度: Nw = {solver.Nw}, w ∈ [{solver.w_min}, {solver.w_max}]")
    
    print(f"\n值函数 V(t, r, w):")
    print(f"  全局范围: [{solver.V.min():.6f}, {solver.V.max():.6f}]")
    print(f"  终端时刻: [{solver.V[-1].min():.6f}, {solver.V[-1].max():.6f}]")
    print(f"  初始时刻: [{solver.V[0].min():.6f}, {solver.V[0].max():.6f}]")
    
    print(f"\n最优消费 c(t, r, w):")
    print(f"  全局范围: [{solver.c_opt.min():.6f}, {solver.c_opt.max():.6f}]")
    print(f"  平均值: {solver.c_opt.mean():.6f}")
    
    print(f"\n最优投资组合 π(t, r, w):")
    print(f"  全局范围: [{solver.pi_opt.min():.6f}, {solver.pi_opt.max():.6f}]")
    print(f"  平均值: {solver.pi_opt.mean():.6f}")
    print(f"  做多比例: {(solver.pi_opt > 0).mean()*100:.1f}%")
    print(f"  做空比例: {(solver.pi_opt < 0).mean()*100:.1f}%")
    
    # 特定点的值
    r_idx = solver.Nr // 2
    w_idx = solver.Nw // 2
    print(f"\n中心点 (r={solver.r[r_idx]:.3f}, w={solver.w[w_idx]:.2f}):")
    print(f"  V(0, r, w) = {solver.V[0, r_idx, w_idx]:.6f}")
    print(f"  c(0, r, w) = {solver.c_opt[0, r_idx, w_idx]:.6f}")
    print(f"  π(0, r, w) = {solver.pi_opt[0, r_idx, w_idx]:.6f}")
    
    print("=" * 70)


def main():
    """
    主函数：运行求解并生成所有图表
    """
    # 运行求解器
    solver = run_solver()
    
    # 生成统计摘要
    generate_summary_statistics(solver)
    
    # 生成各种可视化图表
    print("\n生成可视化图表...")
    
    plot_value_function_slices(solver)
    plot_consumption_policy(solver)
    plot_portfolio_policy(solver)
    plot_comparison_time_evolution(solver)
    plot_wealth_slices(solver)
    plot_interest_rate_analysis(solver)
    
    # 原始3D图
    solver.plot_results('hjb_solution_3d.png')
    
    print("\n所有图表生成完成!")
    
    return solver


if __name__ == "__main__":
    solver = main()
