import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.PINN import PINN
from utils import load_yaml
from model.data import generate_domain_data
from data.data_loder import calculate_parements_stock


def infer_hidden_dims(state_dict, input_dim, output_dim):
    """从 state_dict 推断隐藏层维度"""
    weights = [v for k, v in state_dict.items() if 'weight' in k]
    hidden_dims = []
    for i, w in enumerate(weights[:-1]):
        out_dim, in_dim = w.shape
        hidden_dims.append(out_dim)
    return hidden_dims


def load_model(model_path, cfg, device):
    """加载模型权重并自动推断结构"""
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    state_dict = torch.load(model_path, map_location=device)
    hidden_dims = infer_hidden_dims(state_dict, cfg.input_dim, cfg.output_dim)
    
    model = PINN(
        input_dim=cfg.input_dim,
        hidden_dims=hidden_dims,
        output_dim=cfg.output_dim,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ 成功加载模型: {model_path}")
    return model


def get_pi_predictions(model, X, n_assets, device):
    """
    从模型输出中提取 π (投资策略) 并应用 softmax
    假设输出格式: [V, raw_pi_1, raw_pi_2, ..., raw_pi_n]
    对 raw_pi 应用 softmax 得到归一化的投资比例
    """
    with torch.no_grad():
        output = model(X)
        # 提取 raw_pi (去掉第一个 V)
        raw_pi = output[:, 1:1+n_assets]
        # 应用 softmax 确保在 0-1 之间且和为 1
        # pi = F.softmax(raw_pi, dim=1)
        pi = raw_pi
    return pi.cpu().numpy()


def compare_pi_for_asset(cfg, paraments, asset_idx=0, save_path=None):
    """
    对比两个模型对某一特定资产的 π 预测
    
    Args:
        asset_idx: 要对比的资产索引 (0-based)
    """
    device = torch.device(cfg.device)
    
    # 获取资产数量
    n_assets = len(paraments)
    if asset_idx >= n_assets:
        print(f"错误: 资产索引 {asset_idx} 超出范围 (总资产数: {n_assets})")
        return
    
    # 加载两个模型
    model_normal_path = os.path.join(cfg.model_dir, "pinn_model_test.pth")
    model_smooth_path = os.path.join(cfg.model_dir, "pinn_model_smooth_test.pth")
    
    model_normal = load_model(model_normal_path, cfg, device)
    model_smooth = load_model(model_smooth_path, cfg, device)
    
    if model_normal is None or model_smooth is None:
        print("模型加载失败，无法进行对比")
        return
    
    # 生成测试数据
    n_test = 1000
    X_test = generate_domain_data(n_test, device=device)
    
    # 获取两个模型的 π 预测
    pi_normal = get_pi_predictions(model_normal, X_test, n_assets, device)
    pi_smooth = get_pi_predictions(model_smooth, X_test, n_assets, device)
    
    # 提取特定资产的 π
    pi_normal_asset = pi_normal[:, asset_idx]
    pi_smooth_asset = pi_smooth[:, asset_idx]
    
    # 提取输入维度用于绘图 (假设输入是 [t, w, ...])
    t = X_test[:, 0].cpu().numpy()
    w = X_test[:, 1].cpu().numpy()
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 原始模型 - π 随时间 t 变化
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(t, pi_normal_asset, c=w, cmap='viridis', alpha=0.6, s=10)
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel(f'π (Asset {asset_idx})')
    ax1.set_title(f'Original Model: π vs Time (colored by Wealth)')
    plt.colorbar(scatter1, ax=ax1, label='Wealth (w)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 平滑模型 - π 随时间 t 变化
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(t, pi_smooth_asset, c=w, cmap='viridis', alpha=0.6, s=10)
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel(f'π (Asset {asset_idx})')
    ax2.set_title(f'Smooth Model: π vs Time (colored by Wealth)')
    plt.colorbar(scatter2, ax=ax2, label='Wealth (w)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 差异对比
    ax3 = axes[0, 2]
    pi_diff = pi_smooth_asset - pi_normal_asset
    scatter3 = ax3.scatter(t, pi_diff, c=w, cmap='coolwarm', alpha=0.6, s=10)
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('π Difference (Smooth - Original)')
    ax3.set_title(f'Difference in π (Asset {asset_idx})')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.colorbar(scatter3, ax=ax3, label='Wealth (w)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 原始模型 - π 随财富 w 变化
    ax4 = axes[1, 0]
    scatter4 = ax4.scatter(w, pi_normal_asset, c=t, cmap='plasma', alpha=0.6, s=10)
    ax4.set_xlabel('Wealth (w)')
    ax4.set_ylabel(f'π (Asset {asset_idx})')
    ax4.set_title(f'Original Model: π vs Wealth (colored by Time)')
    plt.colorbar(scatter4, ax=ax4, label='Time (t)')
    ax4.grid(True, alpha=0.3)
    
    # 5. 平滑模型 - π 随财富 w 变化
    ax5 = axes[1, 1]
    scatter5 = ax5.scatter(w, pi_smooth_asset, c=t, cmap='plasma', alpha=0.6, s=10)
    ax5.set_xlabel('Wealth (w)')
    ax5.set_ylabel(f'π (Asset {asset_idx})')
    ax5.set_title(f'Smooth Model: π vs Wealth (colored by Time)')
    plt.colorbar(scatter5, ax=ax5, label='Time (t)')
    ax5.grid(True, alpha=0.3)
    
    # 6. 直方图对比
    ax6 = axes[1, 2]
    ax6.hist(pi_normal_asset, bins=50, alpha=0.5, label='Original', color='blue', density=True)
    ax6.hist(pi_smooth_asset, bins=50, alpha=0.5, label='Smooth', color='red', density=True)
    ax6.set_xlabel(f'π (Asset {asset_idx})')
    ax6.set_ylabel('Density')
    ax6.set_title(f'Distribution Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model Comparison for Asset {asset_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 对比图已保存: {save_path}")
    
    plt.show()
    
    # 打印统计信息
    print(f"\n=== Asset {asset_idx} π Statistics ===")
    print(f"Original Model - Mean: {np.mean(pi_normal_asset):.6f}, Std: {np.std(pi_normal_asset):.6f}")
    print(f"                 Min: {np.min(pi_normal_asset):.6f}, Max: {np.max(pi_normal_asset):.6f}")
    print(f"Smooth Model   - Mean: {np.mean(pi_smooth_asset):.6f}, Std: {np.std(pi_smooth_asset):.6f}")
    print(f"                 Min: {np.min(pi_smooth_asset):.6f}, Max: {np.max(pi_smooth_asset):.6f}")
    print(f"Difference     - Mean: {np.mean(np.abs(pi_diff)):.6f}, Max Abs: {np.max(np.abs(pi_diff)):.6f}")


if __name__ == "__main__":
    # 加载配置
    cfg = load_yaml('config.yaml')
    
    # 加载参数
    paraments = calculate_parements_stock(cfg.csv_path, cfg.result_path, save_csv=False)
    n_assets = len(paraments)
    print(f"总资产数量: {n_assets}")
    
    # 对比第 0 个资产（你可以修改这个索引来选择其他资产）
    asset_to_compare = 2
    
    save_path = os.path.join(cfg.log_dir, f"pi_comparison_asset_{asset_to_compare}.png")
    compare_pi_for_asset(cfg, paraments, asset_idx=asset_to_compare, save_path=save_path)
