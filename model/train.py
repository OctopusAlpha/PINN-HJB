import torch
import torch.autograd as autograd
import pandas as pd
from model.PINN import PINN
import logging
from tqdm import tqdm
from model.loss import pinn_loss,bc_loss
from model.data import *
def train(cfg):

    # 假设你有输入变量 x、sigma、x_scalar
    df = pd.read_csv("./stock_ga_parameters_new.csv")
    mu = torch.tensor(df["mu"].values, dtype=torch.float32, device=device)  # shape: (40,)
    sigma = torch.tensor(df["sigma"].values, dtype=torch.float32, device=device)
    model = PINN(input_dim = cfg.input_dim, hidden_dims = hidden_dims, output_dim = output_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    constraint_losses = []
    Ms = []
    boundary_losses = []
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        # 构造样本
        domain_X = generate_domain_data(batch_size,device=device)
        boundary_x = generate_boundary_data(batch_size,device=device)

        S = torch.randn(1, 40).to(device)  # 假设 S 是一个随机向量
        S = S.repeat(domain_X.shape[0], 1)  # 扩展成 batch

        # sigma = torch.randn(1, 40).to("cuda")  #
        # 扩展成 batch
        x_scalar = domain_X[:, [1]]  # 假设 x_scalar = x[:, 1]，比如时间或其他
        
        loss, constraint_loss, M, v,pi= pinn_loss(model, domain_X, sigma, mu,x_scalar,S,device)
        boundary_loss = bc_loss(model,boundary_x,S,sigma,mu,device)
        loss_total = loss +  boundary_loss

        loss_total.backward()
        optimizer.step()


        if epoch % 100 == 0:
            losses.append(loss.item())
            constraint_losses.append(constraint_loss.item())
            Ms.append(M.item())
            boundary_losses.append(boundary_loss.item())
            # print(v, pi.max(),pi.min())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            print(f"  Constraint Loss: {constraint_loss.item():.6f}, M: {M.item():.6f}, Boundary Loss: {boundary_loss.item():.6f}")
    
    model_path = "pinn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ 模型参数已保存到: {model_path}")
    return '1'
