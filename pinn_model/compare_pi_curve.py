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
    print(f"📐 推断的隐藏层结构: {hidden_dims}")
    
    model = PINN(
        input_dim=cfg.input_dim,
        hidden_dims=hidden_dims,
        output_dim=cfg.output_dim,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ 成功加载模型: {model_path}")
    return model


def get_pi_with_softmax(model, X, n_assets):
    """
    从模型输出中提取 π 并应用 softmax
    输出格式: [V, raw_pi_1, raw_pi_2, ..., raw_pi_n]
    对 raw_pi 应用 softmax 得到归一化的投资比例
    """
    with torch.no_grad():
        output = model(X)
        # 提取 raw_pi (去掉第一个 V)
        raw_pi = output[:, 1:1+n_assets]
        # 应用 softmax 确保在 0-1 之间且和为 1
        pi = F.softmax(raw_pi, dim=1)
    return pi.cpu().numpy()


def compare_pi_curve(cfg, paraments, asset_idx=0, t_fixed=0.0, r_fixed=0.05, save_path=None):
    """
    对比两个模型在固定 t 和 r 下，π 随财富 w 变化的曲线
    
    Args:
        asset_idx: 要对比的资产索引
        t_fixed: 固定的时间值
        r_fixed: 固定的利率值
    """
    device = torch.device(cfg.device)
    
    # 获取资产数量
    n_assets = len(paraments)
    if asset_idx >= n_assets:
        print(f"错误: 资产索引 {asset_idx} 超出范围 (总资产数: {n_assets})")
        return
    
    # 加载两个模型
    model_normal_path = os.path.join(cfg.model_dir, "pinn_model.pth")
    model_smooth_path = os.path.join(cfg.model_dir, "pinn_model_smooth.pth")
    
    model_normal = load_model(model_normal_path, cfg, device)
    model_smooth = load_model(model_smooth_path, cfg, device)
    
    if model_normal is None or model_smooth is None:
        print("模型加载失败，无法进行对比")
        return
    
    # 生成财富网格: w 从 0.5 到 2.0，100 个点
    w_grid = np.linspace(0.5, 2.0, 100)
    
    # 构建输入: [t, w, r] (假设输入维度为 3)
    t_vals = np.full_like(w_grid, t_fixed)
    r_vals = np.full_like(w_grid, r_fixed)
    
    # 组合成输入张量
    X_input = np.stack([t_vals, w_grid, r_vals], axis=1)
    X_tensor = torch.tensor(X_input, dtype=torch.float32, device=device)
    
    # 获取两个模型的 π 预测 (带 softmax)
    pi_normal = get_pi_with_softmax(model_normal, X_tensor, n_assets)
    pi_smooth = get_pi_with_softmax(model_smooth, X_tensor, n_assets)
    
    # 提取特定资产的 π
    pi_normal_asset = pi_normal[:, asset_idx]
    pi_smooth_asset = pi_smooth[:, asset_idx]
    
    # 创建对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制两条曲线
    ax.plot(w_grid, pi_normal_asset, 'b-', linewidth=2, label='Original Model', alpha=0.8)
    ax.plot(w_grid, pi_smooth_asset, 'r-', linewidth=2, label='Smooth Model', alpha=0.8)
    
    # 设置图表属性
    ax.set_xlabel('Wealth (w)', fontsize=12)
    ax.set_ylabel(f'π (Asset {asset_idx})', fontsize=12)
    ax.set_title(f'Investment Strategy Comparison\n(t={t_fixed}, r={r_fixed})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置 y 轴范围，确保完整显示 (根据用户偏好)
    y_min = min(np.min(pi_normal_asset), np.min(pi_smooth_asset))
    y_max = max(np.max(pi_normal_asset), np.max(pi_smooth_asset))
    margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin, y_max + margin)
    
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
    
    # 计算平滑度指标 (方差变化)
    diff_normal = np.diff(pi_normal_asset)
    diff_smooth = np.diff(pi_smooth_asset)
    print(f"\n=== Smoothness (Variance of differences) ===")
    print(f"Original Model: {np.var(diff_normal):.8f}")
    print(f"Smooth Model:   {np.var(diff_smooth):.8f}")
    print(f"Improvement:    {np.var(diff_normal) / np.var(diff_smooth):.2f}x")


if __name__ == "__main__":
    # 加载配置
    cfg = load_yaml('config.yaml')
    
    # 加载参数
    paraments = calculate_parements_stock(cfg.csv_path, cfg.result_path, save_csv=False)
    n_assets = len(paraments)
    print(f"总资产数量: {n_assets}")
    
    # 对比第 0 个资产
    asset_to_compare = 0
    
    save_path = os.path.join(cfg.log_dir, f"pi_curve_comparison_asset_{asset_to_compare}.png")
    compare_pi_curve(cfg, paraments, asset_idx=asset_to_compare, t_fixed=0.0, r_fixed=0.05, save_path=save_path)
