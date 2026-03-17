
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from model.PINN import PINN
import logging
from tqdm import tqdm
from model.loss import pinn_loss, bc_loss, smooth_loss, consistency_loss
from model.data import *
import os
import matplotlib.pyplot as plt



def _prepare_params(paraments, device):
    """将参数列表/数据框转为 tensor."""
    if isinstance(paraments, list):
        df = pd.DataFrame(paraments)
    else:
        df = paraments
    mu = torch.tensor(df["mu"].values, dtype=torch.float32, device=device)
    sigma = torch.tensor(df["sigma"].values, dtype=torch.float32, device=device)
    return mu, sigma

def _generate_S(batch_size, n_assets, device):
    """生成股票价格张量S"""
    S = torch.randn(1, n_assets, device=device)
    return S.repeat(batch_size, 1)  # 扩展成 batch

def _plot_loss_curves(history_df, log_dir):
    """绘制训练损失曲线并保存"""
    os.makedirs(log_dir, exist_ok=True)
    png_path = os.path.join(log_dir, "loss_curve.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["total_loss"], label="total_loss")
    plt.plot(history_df["epoch"], history_df["constraint_loss"], label="constraint_loss")
    plt.plot(history_df["epoch"], history_df["loss"], label="HJB_residual")
    plt.plot(history_df["epoch"], history_df["boundary_loss"], label="boundary_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path


def train(cfg, paraments):
    # 设备已在load_yaml中处理，直接使用
    device = torch.device(cfg.device)
    logger = logging.getLogger(cfg.log_name)

    mu, sigma = _prepare_params(paraments, device)
    n_assets = mu.shape[0]
    model = PINN(
        input_dim=cfg.input_dim,
        hidden_dims=cfg.hidden_dims,
        output_dim=cfg.output_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 学习率衰减：每200个epoch学习率乘以0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    # 各损失项权重（便于调节）
    W_HJB        = 1.0   # HJB 残差权重
    W_CONSTRAINT = 1.0   # pi 约束权重
    W_BOUNDARY   = 1.0   # 边界条件权重
    W_SMOOTH     = 0.5   # 平滑性权重
    W_CONSIST    = 1.0   # 一致性权重
    history = {
        "epoch": [],
        "total_loss": [],
        "constraint_loss": [],
        "loss": [],
        "boundary_loss": [],
    }
    for epoch in tqdm(range(cfg.epochs)):
        optimizer.zero_grad()
        # 构造样本
        domain_X = generate_domain_data(cfg.batch_size, device=device)
        boundary_x = generate_boundary_data(cfg.batch_size, device=device)
        S = _generate_S(cfg.batch_size, n_assets, device)
        x_scalar = domain_X[:, [1]]  # x_scalar = x[:, 1]，用于约束损失
        
        constraint_loss, M, v, pi = pinn_loss(model, domain_X, sigma, mu, x_scalar, S, device)
        boundary_loss,tt = bc_loss(model, boundary_x, S, sigma, mu, device)
        smooth_loss1 = smooth_loss(model, domain_X, S, sigma, mu, device)
        consistency_loss1 = consistency_loss(model, domain_X, S, sigma, mu, device)
        # loss_total = loss + boundary_loss + smooth_loss1
        loss_total = (W_HJB * M
                      + W_CONSTRAINT * constraint_loss
                      + W_BOUNDARY * boundary_loss
                    #   + W_SMOOTH * smooth_loss1
                      + W_CONSIST * consistency_loss1)
        # loss_total = loss + boundary_loss
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # 记录每个 epoch 的指标
        history["epoch"].append(epoch)
        history["total_loss"].append(loss_total.item())
        history["constraint_loss"].append(constraint_loss.item())
        history["loss"].append(M.item())
        history["boundary_loss"].append(boundary_loss.item())

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss_total.item():.6f}")
            logger.info(f"  Constraint Loss: {constraint_loss.item():.6f}, M: {M.item():.6f}, Boundary Loss: {boundary_loss.item():.6f}, Smooth Loss: {smooth_loss1.item():.6f}, Consistency Loss: {consistency_loss1.item():.6f}")
            # logger.info(f"  M: {M.item():.6f}, v: {v.mean().item():.6f},tt: {tt:.6f}")
    
    os.makedirs(cfg.model_dir, exist_ok=True)
    model_path = os.path.join(cfg.model_dir, "pinn_model_test_72.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"✅ 模型参数已保存到: {model_path}")
    
    # 保存并绘制损失
    history_df = pd.DataFrame(history)
    csv_path = os.path.join(cfg.log_dir, "loss_history.csv")
    history_df.to_csv(csv_path, index=False)
    try:
        png_path = _plot_loss_curves(history_df, cfg.log_dir)
        logger.info(f"✅ 损失曲线已保存: {png_path}")
    except Exception as e:
        logger.warning(f"绘制损失曲线失败: {e}")

    return "success"
