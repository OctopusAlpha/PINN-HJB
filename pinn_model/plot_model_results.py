"""
加载训练好的模型并绘制 V、C、π 的 3D 图
类似 pinn_hjb_stock.py 中的可视化风格
"""

import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.PINN import PINN
from utils import load_yaml
from data.data_loder import calculate_parements_stock


def infer_hidden_dims(state_dict):
    """从 state_dict 推断隐藏层维度"""
    # 按层数排序获取权重
    weight_keys = sorted([k for k in state_dict.keys() if 'weight' in k], 
                        key=lambda x: int(x.split('.')[1]))
    
    hidden_dims = []
    for i, key in enumerate(weight_keys[:-1]):  # 除了最后一层
        w = state_dict[key]
        out_dim, in_dim = w.shape
        hidden_dims.append(out_dim)
    
    # 最后一层输出维度
    last_weight = state_dict[weight_keys[-1]]
    actual_output_dim = last_weight.shape[0]
    
    return hidden_dims, actual_output_dim


def load_model(model_path, cfg, device):
    """加载模型权重并自动推断结构"""
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    state_dict = torch.load(model_path, map_location=device)
    hidden_dims, actual_output_dim = infer_hidden_dims(state_dict)
    print(f"📐 推断的隐藏层结构: {hidden_dims}, 输出维度: {actual_output_dim}")
    
    model = PINN(
        input_dim=cfg.input_dim,
        hidden_dims=hidden_dims,
        output_dim=actual_output_dim,  # 使用实际的输出维度
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ 成功加载模型: {model_path}")
    return model


def evaluate_model(model, t_val, w_grid, r_grid, n_assets, device):
    """
    在指定网格上评估模型
    
    Returns:
        V, C, pi, W, R: 值函数、消费、投资组合、财富网格、利率网格
    """
    Nw, Nr = len(w_grid), len(r_grid)
    W, R = np.meshgrid(w_grid, r_grid, indexing='ij')
    
    # 创建输入张量 [t, r, w]（与 loss.py 中训练时的索引顺序一致：x[:,0]=t, x[:,1]=r, x[:,2]=w）
    t_tensor = torch.full((Nw * Nr, 1), t_val, device=device)
    w_tensor = torch.tensor(W.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    r_tensor = torch.tensor(R.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    
    x_eval = torch.cat([t_tensor, r_tensor, w_tensor], dim=1)  # 顺序: [t, r, w]
    
    # 常量定义（与 loss.py 保持一致）
    THETA = 0.7
    
    # 计算 V 对 w 的导数
    x_eval.requires_grad_(True)
    output = model(x_eval)
    
    # 从输出计算 V = (1/THETA) * w^THETA * exp(h)
    h = output[:, 0:1]
    w_col = x_eval[:, 2:3]  # 财富 w 列（x[:,2]=w，与 loss.py 一致）
    w_safe = torch.clamp(w_col, min=1e-3) 
    
    # 【安全锁】：限制 h 的范围，防止 torch.exp(h) 数值爆炸
    h_safe = torch.clamp(h, min=-20.0, max=10.0)
    V_tensor = (1.0 / THETA) * (w_safe ** THETA) * torch.exp(h_safe)
    
    V = V_tensor.detach().cpu().numpy().reshape(Nw, Nr)
    
    # 计算 V 对 w 的偏导数（x[:,2]=w，与 loss.py 中 V_w = grads[:,2] 一致）
    V_w = torch.autograd.grad(V_tensor.sum(), x_eval, create_graph=False)[0][:, 2]
    # 使用 softplus 确保 V_w 为正数，然后计算消费 C = V_w^(THETA-1)
    # V_w_positive = torch.nn.functional.softplus(V_w) + 1e-6

    C = torch.pow(V_w, THETA - 1).detach().cpu().numpy().reshape(Nw, Nr)
    
    # 提取 raw_pi
    raw_pi = output[:, 1:1+n_assets]
    pi_prob = raw_pi
    
    # 取第一个资产的投资比例作为示例
    # pi = pi_prob[:, 0].detach().cpu().numpy().reshape(Nw, Nr)
    
    # 前40个资产的平均投资比例
    pi = pi_prob[:, :40].sum(dim=1).detach().cpu().numpy().reshape(Nw, Nr)
    
    return V, C, pi, W, R


def plot_model_results(model, cfg, paraments, model_name, save_path):
    """
    绘制模型的 V、C、π 3D 图
    
    Args:
        model: 加载好的模型
        cfg: 配置
        paraments: 参数
        model_name: 模型名称（用于标题）
        save_path: 保存路径
    """
    device = torch.device(cfg.device)
    n_assets = len(paraments)
    
    # 创建网格
    Nw, Nr = 50, 50
    w_grid = np.linspace(0.5, 6.0, Nw)  # 财富范围
    r_grid = np.linspace(0.001, 4, Nr)  # 利率范围
    
    # 选择时间点
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    fig = plt.figure(figsize=(20, 12))
    
    for idx, t_val in enumerate(time_points):
        V, C, pi, W, R = evaluate_model(model, t_val, w_grid, r_grid, n_assets, device)
        
        # 值函数 V
        ax1 = fig.add_subplot(3, 5, idx+1, projection='3d')
        surf1 = ax1.plot_surface(R, W, V, cmap='viridis', edgecolor='none')
        ax1.set_xlabel('r')
        ax1.set_ylabel('w')
        ax1.set_zlabel('V')
        ax1.set_title(f'V at t={t_val:.2f}')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
        
        # 消费 C - 旋转180度视角
        ax2 = fig.add_subplot(3, 5, idx+6, projection='3d')
        surf2 = ax2.plot_surface(R, W, C, cmap='plasma', edgecolor='none')
        ax2.set_xlabel('r')
        ax2.set_ylabel('w')
        ax2.set_zlabel('C')
        ax2.set_title(f'C at t={t_val:.2f}')
        # ax2.view_init(elev=30, azim=120)  # 逆时针旋转180度
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        
        # 投资组合 π (第一个资产) - 旋转180度视角
        ax3 = fig.add_subplot(3, 5, idx+11, projection='3d')
        surf3 = ax3.plot_surface(R, W, pi, cmap='coolwarm', edgecolor='none')
        ax3.set_xlabel('r')
        ax3.set_ylabel('w')
        ax3.set_zlabel('π')
        ax3.set_title(f'π at t={t_val:.2f}')
        # ax3.view_init(elev=30, azim=120)  # 逆时针旋转180度
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    plt.suptitle(f'{model_name} Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    # plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    # print(f"✅ 结果图已保存: {save_path}")


def compare_models(cfg, paraments):
    """对比两个模型的结果"""
    device = torch.device(cfg.device)
    n_assets = len(paraments)
    
    # 加载两个模型
    model_normal_path = os.path.join(cfg.model_dir, "pinn_model_test_104.pth")
    # model_smooth_path = os.path.join(cfg.model_dir, "pinn_model_smooth_test.pth")
    
    model_normal = load_model(model_normal_path, cfg, device)
    # model_smooth = load_model(model_smooth_path, cfg, device)
    
    if model_normal is None and model_smooth is None:
        print("❌ 两个模型都加载失败")
        return
    
    # 绘制原始模型
    if model_normal is not None:
        save_path_normal = os.path.join(cfg.log_dir, "model_normal_results.png")
        plot_model_results(model_normal, cfg, paraments, "Original Model", save_path_normal)
    
    # 绘制平滑模型
    # if model_smooth is not None:
    #     save_path_smooth = os.path.join(cfg.log_dir, "model_smooth_results.png")
    #     plot_model_results(model_smooth, cfg, paraments, "Smooth Model", save_path_smooth)


if __name__ == "__main__":
    # 加载配置
    cfg = load_yaml('config.yaml')
    
    # 加载参数
    paraments = calculate_parements_stock(cfg.csv_path, cfg.result_path, save_csv=False)
    n_assets = len(paraments)
    print(f"总资产数量: {n_assets}")
    
    # 对比两个模型
    compare_models(cfg, paraments)
