"""
基于PINN的时变股票指数HJB方程求解器

HJB方程:
0 = V_t + (α-βr)V_r + 0.5σ_r²V_rr 
    + max_{π,C} { [X(πμ+(1-π)r)-C]V_x + 0.5[(πσ)²+(1-π)²A²]X²V_xx
                  + (1/θ)ψ^(1-θ)C^θ + 0.5V_xr X(1-π) }

状态变量: t (时间), r (利率), X (财富)
控制变量: π (投资组合), C (消费)
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PINN(nn.Module):
    """物理信息神经网络"""
    
    def __init__(self, input_dim=3, hidden_dims=[128, 128, 128, 64], output_dim=1):
        """
        Args:
            input_dim: 输入维度 (t, r, X)
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度 (V - 值函数)
        """
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(dims[-2], dims[-1]))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, 3] - (t, r, X)
        
        Returns:
            V: [batch_size, 1] - 值函数
        """
        # 使用softplus确保值函数为正
        raw_output = self.network(x)
        V = torch.nn.functional.softplus(raw_output)
        return V


class PINNHJBSolver:
    """PINN HJB求解器"""
    
    def __init__(self, params):
        """
        初始化求解器
        
        Args:
            params: 参数字典
        """
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 经济参数
        self.theta = params.get('theta', 0.7)
        self.mu = params.get('mu', 0.08)
        self.sigma = params.get('sigma', 0.2)
        self.sigma_r = params.get('sigma_r', 0.05)
        
        # 利率过程参数
        self.alpha = params.get('alpha', 0.02)
        self.beta = params.get('beta', 0.1)
        
        # 财富扩散参数
        self.A = params.get('A', 0.1)
        
        # 股票指数和贴现因子参数
        self.phi0 = params.get('phi0', 0.2)
        self.phi1 = params.get('phi1', 0.1)
        self.phi2 = params.get('phi2', 0.05)
        
        # 网格参数
        self.t_min = params.get('t_min', 0.0)
        self.t_max = params.get('t_max', 1.0)
        self.r_min = params.get('r_min', 0.01)
        self.r_max = params.get('r_max', 1.0)
        self.X_min = params.get('w_min', 0.5)
        self.X_max = params.get('w_max', 4.0)
        
        # 股票指数过程
        self.S0 = params.get('S0', 1.0)
        self.mu_s = params.get('mu_s', 0.05)
        
        # 训练参数
        self.epochs = params.get('epochs', 5000)
        self.batch_size = params.get('batch_size', 2048)
        self.lr = params.get('lr', 1e-3)
        
        # 初始化模型
        self.model = PINN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)
        
        # 损失历史
        self.loss_history = {
            'total': [],
            'pde': [],
            'boundary': [],
            'terminal': []
        }
    
    def stock_process(self, t):
        """股票指数过程 S(t) = S0 * exp(μ_s * t)"""
        return self.S0 * torch.exp(self.mu_s * t)
    
    def compute_psi(self, t):
        """
        计算贴现因子 ψ(t)
        ψ(t) = exp{ -1/(1-θ) * [φ₀t + φ₁·S(t)·t + φ₂·S(t)²·t] }
        """
        S = self.stock_process(t)
        integral_term = self.phi0 * t + self.phi1 * S * t + self.phi2 * S**2 * t
        gamma = 1.0 - self.theta
        psi = torch.exp(-integral_term / gamma)
        return psi
    
    def compute_terminal_value(self, X, psi_T):
        """
        终端条件: V(T) = (1/θ) * ψ(T)^(1-θ) * X^θ
        """
        gamma = 1.0 - self.theta
        X_safe = torch.clamp(X, min=1e-6)
        terminal_value = (1.0 / self.theta) * (psi_T ** gamma) * (X_safe ** self.theta)
        return terminal_value
    
    def compute_optimal_controls(self, V_x, V_xx, V_xr, X, r, t):
        """
        计算最优控制变量 (C, π)
        
        最优消费: C = [ψ^(1-θ) / V_x]^(1/(1-θ))
        最优投资组合: π = [V_x * X * (r - μ) + V_xx * X² * A² + 0.5 * V_xr * X] 
                         / [V_xx * X² * (σ² + A²)]
        """
        psi = self.compute_psi(t)
        gamma = 1.0 - self.theta
        
        # 确保导数有效
        V_x_safe = torch.clamp(V_x, min=1e-6)
        
        # 最优消费
        psi_gamma = psi ** gamma
        C_optimal = (psi_gamma / V_x_safe) ** (1.0 / gamma)
        C = torch.clamp(C_optimal, min=1e-6)
        C = torch.min(C, X)  # 消费不能超过财富
        
        # 最优投资组合 - 使用HJB一阶条件的解析解
        # 【修复：强制 V_xx 保持负数（凹性），防止策略爆炸】
        V_xx_safe = torch.minimum(V_xx, torch.tensor(-1e-6, device=V_xx.device))
        
        # 分子: V_x * X * (r - μ) + V_xx * X² * A² + 0.5 * V_xr * X
        numerator = (V_x_safe * X * (r - self.mu) + 
                     V_xx_safe * (X**2) * (self.A**2) + 
                     0.5 * V_xr * X)
        
        # 分母: V_xx * X² * (σ² + A²)
        denominator = V_xx_safe * (X**2) * (self.sigma**2 + self.A**2)
        
        # 避免除零
        pi = torch.where(
            torch.abs(denominator) > 1e-10,
            numerator / denominator,
            torch.zeros_like(denominator)
        )
        
        # 限制投资组合范围 [-1, 1]
        pi = torch.clamp(pi, min=0.0, max=1.0)
        
        return C, pi
    
    def compute_derivatives(self, x, V):
        """
        计算值函数的各阶导数
        
        Returns:
            V_t, V_r, V_X, V_rr, V_XX, V_Xr
        """
        grads = autograd.grad(
            V, x, 
            grad_outputs=torch.ones_like(V),
            create_graph=True,
            retain_graph=True
        )[0]
        
        V_t = grads[:, 0]
        V_r = grads[:, 1]
        V_X = grads[:, 2]
        
        # 二阶导数
        V_rr = autograd.grad(
            V_r, x, 
            grad_outputs=torch.ones_like(V_r),
            create_graph=True,
            retain_graph=True
        )[0][:, 1]
        
        V_XX = autograd.grad(
            V_X, x, 
            grad_outputs=torch.ones_like(V_X),
            create_graph=True,
            retain_graph=True
        )[0][:, 2]
        
        V_Xr = autograd.grad(
            V_X, x, 
            grad_outputs=torch.ones_like(V_X),
            create_graph=True,
            retain_graph=True
        )[0][:, 1]
        
        return V_t, V_r, V_X, V_rr, V_XX, V_Xr
    
    def pde_residual(self, x):
        """
        计算HJB方程残差
        
        HJB方程:
        0 = V_t + (α-βr)V_r + 0.5σ_r²V_rr 
            + [X(πμ+(1-π)r)-C]V_x + 0.5[(πσ)²+(1-π)²A²]X²V_xx
            + (1/θ)ψ^(1-θ)C^θ + 0.5V_xr X(1-π)
        """
        x.requires_grad_(True)
        
        # 前向传播
        V = self.model(x).squeeze()
        
        # 提取变量
        t = x[:, 0]
        r = x[:, 1]
        X = x[:, 2]
        
        # 计算导数
        V_t, V_r, V_X, V_rr, V_XX, V_Xr = self.compute_derivatives(x, V)
        
        # 计算贴现因子
        psi = self.compute_psi(t)
        gamma = 1.0 - self.theta
        
        # 计算最优控制
        C, pi = self.compute_optimal_controls(V_X, V_XX, V_Xr, X, r, t)
        
        # HJB方程各项
        # 时间导数项
        time_term = V_t
        
        # 利率漂移项
        r_drift = self.alpha - self.beta * r
        r_drift_term = r_drift * V_r
        
        # 利率扩散项
        r_diffusion = 0.5 * self.sigma_r**2 * V_rr
        
        # 财富漂移项
        portfolio_return = pi * self.mu + (1 - pi) * r
        wealth_drift = X * portfolio_return - C
        wealth_drift_term = wealth_drift * V_X
        
        # 财富扩散项
        wealth_diffusion = 0.5 * ((pi * self.sigma)**2 + ((1 - pi) * self.A)**2) * X**2
        wealth_diffusion_term = wealth_diffusion * V_XX
        
        # 效用项
        utility_term = (1.0 / self.theta) * (psi ** gamma) * (C ** self.theta)
        
        # 交叉导数项
        cross_term = 0.5 * V_Xr * X * (1 - pi)
        
        # HJB残差
        residual = (time_term + r_drift_term + r_diffusion + 
                   wealth_drift_term + wealth_diffusion_term + 
                   utility_term + cross_term)
        
        return residual, V, C, pi
    
    def generate_training_data(self):
        """生成训练数据"""
        # 域内数据
        t = torch.rand(self.batch_size, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
        r = torch.rand(self.batch_size, 1, device=self.device) * (self.r_max - self.r_min) + self.r_min
        X = torch.rand(self.batch_size, 1, device=self.device) * (self.X_max - self.X_min) + self.X_min
        
        domain_data = torch.cat([t, r, X], dim=1)
        
        # 终端边界数据 (t = T)
        t_terminal = torch.full((self.batch_size, 1), self.t_max, device=self.device)
        r_terminal = torch.rand(self.batch_size, 1, device=self.device) * (self.r_max - self.r_min) + self.r_min
        X_terminal = torch.rand(self.batch_size, 1, device=self.device) * (self.X_max - self.X_min) + self.X_min
        
        terminal_data = torch.cat([t_terminal, r_terminal, X_terminal], dim=1)
        
        return domain_data, terminal_data
    
    def train(self):
        """训练PINN模型"""
        print("=" * 60)
        print("开始训练PINN HJB求解器")
        print(f"设备: {self.device}")
        print("=" * 60)
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            
            # 生成数据
            domain_data, terminal_data = self.generate_training_data()
            
            # 计算PDE残差
            residual, V_domain, C_domain, pi_domain = self.pde_residual(domain_data)
            pde_loss = torch.mean(residual**2)
            
            # 计算终端边界损失
            V_terminal = self.model(terminal_data).squeeze()
            t_term = terminal_data[:, 0]
            X_term = terminal_data[:, 2]
            psi_T = self.compute_psi(t_term)
            V_terminal_target = self.compute_terminal_value(X_term, psi_T)
            terminal_loss = torch.mean((V_terminal - V_terminal_target)**2)
            
            # 总损失
            total_loss = pde_loss + terminal_loss
            
            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录损失
            self.loss_history['total'].append(total_loss.item())
            self.loss_history['pde'].append(pde_loss.item())
            self.loss_history['terminal'].append(terminal_loss.item())
            
            # 打印进度
            if epoch % 500 == 0:
                print(f"Epoch {epoch}/{self.epochs}")
                print(f"  Total Loss: {total_loss.item():.6e}")
                print(f"  PDE Loss: {pde_loss.item():.6e}")
                print(f"  Terminal Loss: {terminal_loss.item():.6e}")
                print(f"  V range: [{V_domain.min().item():.4f}, {V_domain.max().item():.4f}]")
                print(f"  C range: [{C_domain.min().item():.4f}, {C_domain.max().item():.4f}]")
                print(f"  π range: [{pi_domain.min().item():.4f}, {pi_domain.max().item():.4f}]")
                print("-" * 40)
        
        print("=" * 60)
        print("训练完成!")
        print("=" * 60)
    
    def evaluate(self, t_val, r_grid, X_grid):
        """
        在指定网格上评估模型
        
        Args:
            t_val: 时间值
            r_grid: 利率网格
            X_grid: 财富网格
        
        Returns:
            V, C, pi: 值函数和最优控制
        """
        Nr, NX = len(r_grid), len(X_grid)
        R, X = np.meshgrid(r_grid, X_grid, indexing='ij')
        
        # 创建输入张量
        t_tensor = torch.full((Nr * NX, 1), t_val, device=self.device)
        r_tensor = torch.tensor(R.flatten(), dtype=torch.float32, device=self.device).unsqueeze(1)
        X_tensor = torch.tensor(X.flatten(), dtype=torch.float32, device=self.device).unsqueeze(1)
        
        x_eval = torch.cat([t_tensor, r_tensor, X_tensor], dim=1)
        
        # 【修复：避免在torch.no_grad()中计算梯度，分两步进行】
        # 第一步：计算值函数（不需要梯度）
        with torch.no_grad():
            V_no_grad = self.model(x_eval).squeeze()
            V = V_no_grad.cpu().numpy().reshape(Nr, NX)
        
        # 第二步：重新创建输入张量并计算控制变量（需要梯度）
        x_grad = x_eval.clone().detach().requires_grad_(True)
        V_grad = self.model(x_grad).squeeze()
        
        # 计算一阶导数
        grads = autograd.grad(
            V_grad, x_grad,
            grad_outputs=torch.ones_like(V_grad),
            create_graph=True,  # 需要创建计算图以计算二阶导数
            retain_graph=True
        )[0]
        
        V_t = grads[:, 0]
        V_r = grads[:, 1]
        V_X = grads[:, 2]
        
        t_batch = x_grad[:, 0]
        r_batch = x_grad[:, 1]
        X_batch = x_grad[:, 2]
        
        psi = self.compute_psi(t_batch)
        gamma = 1.0 - self.theta
        
        # 最优消费
        V_X_safe = torch.clamp(V_X, min=1e-6)
        psi_gamma = psi ** gamma
        C_optimal = (psi_gamma / V_X_safe) ** (1.0 / gamma)
        C = torch.clamp(C_optimal, min=1e-6)
        C = torch.min(C, X_batch)  # 消费不能超过财富
        
        # 计算二阶导数和交叉导数
        V_rr = autograd.grad(
            V_r, x_grad,
            grad_outputs=torch.ones_like(V_r),
            create_graph=False,
            retain_graph=False
        )[0][:, 1]
        
        V_XX = autograd.grad(
            V_X, x_grad,
            grad_outputs=torch.ones_like(V_X),
            create_graph=False,
            retain_graph=False
        )[0][:, 2]
        
        V_Xr = autograd.grad(
            V_X, x_grad,
            grad_outputs=torch.ones_like(V_X),
            create_graph=False,
            retain_graph=False
        )[0][:, 1]
        
        # 最优投资组合 - 使用HJB一阶条件的解析解
        V_XX_safe = torch.minimum(V_XX, torch.tensor(-1e-6, device=V_XX.device))
        
        numerator = (V_X_safe * X_batch * (r_batch - self.mu) + 
                     V_XX_safe * (X_batch**2) * (self.A**2) + 
                     0.5 * V_Xr * X_batch)
        
        denominator = V_XX_safe * (X_batch**2) * (self.sigma**2 + self.A**2)
        
        pi = torch.where(
            torch.abs(denominator) > 1e-10,
            numerator / denominator,
            torch.zeros_like(denominator)
        )
        
        pi = torch.clamp(pi, min=0.0, max=1.0)
        
        C = C.detach().cpu().numpy().reshape(Nr, NX)
        pi = pi.detach().cpu().numpy().reshape(Nr, NX)
        
        return V, C, pi, R, X
    
    def plot_results(self, save_path='pinn_hjb_solution.png'):
        """绘制结果"""
        # 创建网格
        Nr, NX = 50, 50
        r_grid = np.linspace(self.r_min, self.r_max, Nr)
        X_grid = np.linspace(self.X_min, self.X_max, NX)
        
        # 选择时间点
        time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        fig = plt.figure(figsize=(18, 12))
        
        for idx, t_val in enumerate(time_points):
            V, C, pi, R, X = self.evaluate(t_val, r_grid, X_grid)
            
            # 值函数
            ax1 = fig.add_subplot(3, 5, idx+1, projection='3d')
            ax1.plot_surface(R, X, V, cmap='viridis')
            ax1.set_xlabel('r')
            ax1.set_ylabel('X')
            ax1.set_zlabel('V')
            ax1.set_title(f'V at t={t_val:.2f}')
            
            # 最优消费
            ax2 = fig.add_subplot(3, 5, idx+6, projection='3d')
            ax2.plot_surface(R, X, C, cmap='plasma')
            ax2.set_xlabel('r')
            ax2.set_ylabel('X')
            ax2.set_zlabel('C')
            ax2.set_title(f'C at t={t_val:.2f}')
            
            # 最优投资组合
            ax3 = fig.add_subplot(3, 5, idx+11, projection='3d')
            ax3.plot_surface(R, X, pi, cmap='coolwarm')
            ax3.set_xlabel('r')
            ax3.set_ylabel('X')
            ax3.set_zlabel('π')
            ax3.set_title(f'π at t={t_val:.2f}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"结果已保存至: {save_path}")
        
        # 绘制损失曲线
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].semilogy(self.loss_history['total'])
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        axes[1].semilogy(self.loss_history['pde'])
        axes[1].set_title('PDE Residual Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
        
        axes[2].semilogy(self.loss_history['terminal'])
        axes[2].set_title('Terminal Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('pinn_loss_curves.png', dpi=150)
        plt.show()
        print("损失曲线已保存至: pinn_loss_curves.png")


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
        'epochs': 10000,
        'batch_size': 2048,
        'lr': 1e-3
    }
    
    # 创建求解器并训练
    solver = PINNHJBSolver(params)
    solver.train()
    
    # 绘制结果
    solver.plot_results('pinn_hjb_solution.png')
    
    print("\n最终结果:")
    r_grid = np.linspace(params['r_min'], params['r_max'], 50)
    X_grid = np.linspace(params['w_min'], params['w_max'], 50)
    V, C, pi, _, _ = solver.evaluate(0.0, r_grid, X_grid)
    print(f"值函数范围 (t=0): [{V.min():.4f}, {V.max():.4f}]")
    print(f"最优消费范围 (t=0): [{C.min():.4f}, {C.max():.4f}]")
    print(f"最优投资组合范围 (t=0): [{pi.min():.4f}, {pi.max():.4f}]")


if __name__ == "__main__":
    main()
