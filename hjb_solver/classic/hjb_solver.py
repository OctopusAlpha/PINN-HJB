"""
HJB方程有限差分法求解器 (简化版本)
三维状态变量: t (时间), r (利率), w (财富)
控制变量: c (消费), π (单一风险资产投资比例)

基于用户PINN-HJB项目中的HJB方程简化:
- 多资产 -> 单一风险资产
- 保留关键特征: 随机贴现因子、消费、投资组合选择
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HJBSolverFDM:
    """
    HJB方程求解器 - 简化版单一风险资产
    
    方程形式 (基于loss.py中的方程简化):
    0 = V_t + V_w * w * (π*μ + (1-π)*r - c) 
        + 0.5 * σ_r² * V_rr
        + V_ww * [0.5*(π*σ)² + 0.5*((1-π)*w)² + (1-π)*w²]
        + 0.5 * V_wr * ρ * σ * (1-π)*w
        
    终端条件: V(T, r, w) = boundary_value
    
    其中:
    - c = (V_w)^(θ-1) * φ  (最优消费)
    - φ: 随机贴现因子
    - θ = 0.7: 风险厌恶系数
    - ρ = 0.5: 相关系数
    """
    
    def __init__(self, params):
        """
        初始化求解器
        
        params: 参数字典
        """
        self.params = params
        
        # 网格设置
        self.t_min = params.get('t_min', 0.0)
        self.t_max = params.get('t_max', 1.0)
        self.r_min = params.get('r_min', 0.01)
        self.r_max = params.get('r_max', 0.10)
        self.w_min = params.get('w_min', 0.1)
        self.w_max = params.get('w_max', 10.0)
        
        self.Nt = params.get('Nt', 50)
        self.Nr = params.get('Nr', 30)
        self.Nw = params.get('Nw', 30)
        
        # 生成网格
        self.t = np.linspace(self.t_min, self.t_max, self.Nt)
        self.r = np.linspace(self.r_min, self.r_max, self.Nr)
        self.w = np.linspace(self.w_min, self.w_max, self.Nw)
        
        self.dt = self.t[1] - self.t[0]
        self.dr = self.r[1] - self.r[0]
        self.dw = self.w[1] - self.w[0]
        
        # HJB方程参数 (来自loss.py)
        self.theta = params.get('theta', 0.7)  # 风险厌恶系数
        self.rho = params.get('rho', 0.5)      # 相关系数
        self.mu = params.get('mu', 0.08)       # 风险资产期望收益
        self.sigma = params.get('sigma', 0.2)  # 风险资产波动率
        self.sigma_r = params.get('sigma_r', 0.05)  # 利率波动率
        self.kappa = params.get('kappa', 0.1)  # 利率均值回归速度
        self.r_bar = params.get('r_bar', 0.05) # 利率长期均值
        
        # 值函数初始化 [Nt, Nr, Nw]
        self.V = np.zeros((self.Nt, self.Nr, self.Nw))
        self.c_opt = np.zeros((self.Nt, self.Nr, self.Nw))
        self.pi_opt = np.zeros((self.Nt, self.Nr, self.Nw))
        
        # 迭代参数
        self.max_iter = params.get('max_iter', 100)
        self.tol = params.get('tol', 1e-6)
        
    def compute_phi(self, t):
        """
        计算随机贴现因子 φ (简化版本)
        原代码使用蒙特卡洛，这里用简化公式
        """
        # 简化: φ = exp(-(1-θ)^(-1) * (0.2 + 0.2*t))
        phi = np.exp(-(1-self.theta)**(-1) * (0.2 + 0.2*t))
        return phi
    
    def compute_terminal_value(self, r, w):
        """
        计算终端时刻的边界值
        基于loss.py中的_compute_boundary_value
        """
        phi_T = self.compute_phi(self.t_max)
        # tt = θ^(-1) * φ^(1-θ) * w^θ * (-6.676 - 0.8*1.198*r² + 5.705*r)
        tt = (self.theta**(-1) * 
              phi_T**(1-self.theta) * 
              w**self.theta * 
              (-6.676 - 0.8*1.198*r**2 + 5.705*r))
        return tt
    
    def compute_optimal_consumption(self, V_w, phi):
        """
        计算最优消费 c = (V_w)^(θ-1) * φ
        """
        V_w_clipped = np.maximum(V_w, 1e-6)  # 保证正数
        c = V_w_clipped**(self.theta - 1) * phi
        return c
    
    def compute_derivatives(self, V, idx_t, idx_r, idx_w):
        """
        计算指定点的偏导数
        使用中心差分，边界使用单侧差分
        """
        Nt, Nr, Nw = V.shape
        
        # 初始化导数
        V_t = 0.0
        V_w = 0.0
        V_r = 0.0
        V_ww = 0.0
        V_rr = 0.0
        V_wr = 0.0
        
        it, ir, iw = idx_t, idx_r, idx_w
        
        # 时间导数 (后向差分，因为是终端值问题)
        if it < Nt - 1:
            V_t = (V[it+1, ir, iw] - V[it, ir, iw]) / self.dt
        else:
            V_t = 0.0
        
        # w方向导数
        if iw > 0 and iw < Nw - 1:
            V_w = (V[it, ir, iw+1] - V[it, ir, iw-1]) / (2 * self.dw)
            V_ww = (V[it, ir, iw+1] - 2*V[it, ir, iw] + V[it, ir, iw-1]) / (self.dw**2)
        elif iw == 0:
            V_w = (V[it, ir, iw+1] - V[it, ir, iw]) / self.dw
            V_ww = (V[it, ir, iw+2] - 2*V[it, ir, iw+1] + V[it, ir, iw]) / (self.dw**2)
        else:  # iw == Nw - 1
            V_w = (V[it, ir, iw] - V[it, ir, iw-1]) / self.dw
            V_ww = (V[it, ir, iw] - 2*V[it, ir, iw-1] + V[it, ir, iw-2]) / (self.dw**2)
        
        # r方向导数
        if ir > 0 and ir < Nr - 1:
            V_r = (V[it, ir+1, iw] - V[it, ir-1, iw]) / (2 * self.dr)
            V_rr = (V[it, ir+1, iw] - 2*V[it, ir, iw] + V[it, ir-1, iw]) / (self.dr**2)
        elif ir == 0:
            V_r = (V[it, ir+1, iw] - V[it, ir, iw]) / self.dr
            V_rr = (V[it, ir+2, iw] - 2*V[it, ir+1, iw] + V[it, ir, iw]) / (self.dr**2)
        else:  # ir == Nr - 1
            V_r = (V[it, ir, iw] - V[it, ir-1, iw]) / self.dr
            V_rr = (V[it, ir, iw] - 2*V[it, ir-1, iw] + V[it, ir-2, iw]) / (self.dr**2)
        
        # 交叉导数 V_wr
        if (ir > 0 and ir < Nr - 1 and iw > 0 and iw < Nw - 1):
            V_wr = (V[it, ir+1, iw+1] - V[it, ir+1, iw-1] - 
                    V[it, ir-1, iw+1] + V[it, ir-1, iw-1]) / (4 * self.dr * self.dw)
        
        return V_t, V_w, V_r, V_ww, V_rr, V_wr
    
    def optimize_portfolio(self, V_w, V_ww, V_wr, w, r_val):
        """
        优化投资组合比例 π
        通过求解一阶条件
        """
        # 离散化搜索
        pi_grid = np.linspace(-2, 2, 100)
        best_pi = 0.0
        best_value = -np.inf
        
        phi = self.compute_phi(0)  # 简化
        c = self.compute_optimal_consumption(abs(V_w), phi)
        
        for pi in pi_grid:
            # 计算HJB方程的右边 (不含V_t)
            drift_w = pi * self.mu + (1 - pi) * r_val - c
            
            # 扩散项
            diffusion = (0.5 * (pi * self.sigma)**2 + 
                        0.5 * ((1 - pi) * w)**2 + 
                        (1 - pi) * w**2)
            
            # 交叉项
            cross = 0.5 * V_wr * self.rho * self.sigma * (1 - pi) * w
            
            # HJB残差 (我们希望最大化这个)
            hjb_value = (V_w * w * drift_w + 
                        V_ww * diffusion + 
                        cross)
            
            if hjb_value > best_value:
                best_value = hjb_value
                best_pi = pi
        
        return best_pi
    
    def solve_backward(self):
        """
        反向时间步进求解
        从终端时刻倒推
        """
        print("开始求解HJB方程...")
        
        # 步骤1: 设置终端条件
        for ir in range(self.Nr):
            for iw in range(self.Nw):
                self.V[-1, ir, iw] = self.compute_terminal_value(
                    self.r[ir], self.w[iw]
                )
        
        print(f"终端条件已设置，V(T)范围: [{self.V[-1].min():.4f}, {self.V[-1].max():.4f}]")
        
        # 步骤2: 反向时间步进
        for it in range(self.Nt - 2, -1, -1):
            t_val = self.t[it]
            phi = self.compute_phi(t_val)
            
            # 在每个时间层进行策略迭代
            for iteration in range(self.max_iter):
                V_old_layer = self.V[it].copy()
                
                for ir in range(self.Nr):
                    for iw in range(self.Nw):
                        # 计算导数
                        V_t, V_w, V_r, V_ww, V_rr, V_wr = self.compute_derivatives(
                            self.V, it, ir, iw
                        )
                        
                        # 保证V_w为正 (值函数关于财富递增)
                        V_w = max(V_w, 1e-6)
                        
                        # 计算最优消费
                        c = self.compute_optimal_consumption(V_w, phi)
                        self.c_opt[it, ir, iw] = c
                        
                        # 优化投资组合
                        pi = self.optimize_portfolio(V_w, V_ww, V_wr, 
                                                     self.w[iw], self.r[ir])
                        self.pi_opt[it, ir, iw] = pi
                        
                        # 计算漂移项
                        drift_w = pi * self.mu + (1 - pi) * self.r[ir] - c
                        
                        # 利率漂移 (均值回归)
                        drift_r = self.kappa * (self.r_bar - self.r[ir])
                        
                        # 计算HJB方程右边
                        hjb_rhs = (
                            V_w * self.w[iw] * drift_w +
                            V_r * drift_r +
                            0.5 * self.sigma_r**2 * V_rr +
                            V_ww * (0.5 * (pi * self.sigma)**2 + 
                                   0.5 * ((1 - pi) * self.w[iw])**2 +
                                   (1 - pi) * self.w[iw]**2) +
                            0.5 * V_wr * self.rho * self.sigma * (1 - pi) * self.w[iw]
                        )
                        
                        # 更新值函数 (显式欧拉)
                        self.V[it, ir, iw] = self.V[it+1, ir, iw] - self.dt * hjb_rhs
                
                # 检查收敛
                error = np.max(np.abs(self.V[it] - V_old_layer))
                if error < self.tol:
                    break
            
            if it % 10 == 0:
                print(f"时间步 {it}/{self.Nt}, t={t_val:.3f}, V范围: [{self.V[it].min():.4f}, {self.V[it].max():.4f}]")
        
        print("求解完成!")
        return self.V, self.c_opt, self.pi_opt
    
    def plot_results(self):
        """可视化结果"""
        # 选择几个时间点展示
        time_indices = [0, self.Nt//4, self.Nt//2, 3*self.Nt//4, self.Nt-1]
        
        fig = plt.figure(figsize=(18, 12))
        
        for idx, it in enumerate(time_indices):
            t_val = self.t[it]
            
            # 值函数
            ax1 = fig.add_subplot(3, 5, idx+1, projection='3d')
            R, W = np.meshgrid(self.r, self.w, indexing='ij')
            surf1 = ax1.plot_surface(R, W, self.V[it], cmap='viridis')
            ax1.set_xlabel('r')
            ax1.set_ylabel('w')
            ax1.set_zlabel('V')
            ax1.set_title(f'V at t={t_val:.2f}')
            
            # 最优消费
            ax2 = fig.add_subplot(3, 5, idx+6, projection='3d')
            surf2 = ax2.plot_surface(R, W, self.c_opt[it], cmap='plasma')
            ax2.set_xlabel('r')
            ax2.set_ylabel('w')
            ax2.set_zlabel('c')
            ax2.set_title(f'c at t={t_val:.2f}')
            
            # 最优投资组合
            ax3 = fig.add_subplot(3, 5, idx+11, projection='3d')
            surf3 = ax3.plot_surface(R, W, self.pi_opt[it], cmap='coolwarm')
            ax3.set_xlabel('r')
            ax3.set_ylabel('w')
            ax3.set_zlabel('π')
            ax3.set_title(f'π at t={t_val:.2f}')
        
        plt.tight_layout()
        plt.savefig('hjb_solution_fdm.png', dpi=150)
        plt.show()


def main():
    """主函数示例"""
    # 参数设置
    params = {
        't_min': 0.0,
        't_max': 1.0,
        'r_min': 0.01,
        'r_max': 0.10,
        'w_min': 0.5,
        'w_max': 2.0,
        'Nt': 50,
        'Nr': 30,
        'Nw': 30,
        'theta': 0.7,      # 风险厌恶系数 (来自loss.py)
        'rho': 0.5,        # 相关系数 (来自loss.py)
        'mu': 0.08,        # 风险资产期望收益
        'sigma': 0.2,      # 风险资产波动率
        'sigma_r': 0.05,   # 利率波动率
        'kappa': 0.1,      # 利率均值回归速度
        'r_bar': 0.05,     # 利率长期均值
        'max_iter': 50,
        'tol': 1e-6
    }
    
    # 创建求解器
    solver = HJBSolverFDM(params)
    
    # 求解
    V, c_opt, pi_opt = solver.solve_backward()
    
    # 可视化
    solver.plot_results()
    
    print("\n求解完成!")
    print(f"值函数范围: [{V.min():.4f}, {V.max():.4f}]")
    print(f"最优消费范围: [{c_opt.min():.4f}, {c_opt.max():.4f}]")
    print(f"最优投资组合范围: [{pi_opt.min():.4f}, {pi_opt.max():.4f}]")


if __name__ == "__main__":
    main()
