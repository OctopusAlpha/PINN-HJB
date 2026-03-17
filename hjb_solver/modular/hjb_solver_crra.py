"""
CRRA效用函数下的HJB方程求解器
基于通用框架的具体实现
"""

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from typing import Tuple

from hjb_solver_base import HJBSolverBase, UtilityFunction


class HJBSolverCRRA(HJBSolverBase):
    """
    CRRA效用函数下的HJB方程求解器
    
    特点:
    - CRRA效用: u(c) = c^θ / θ
    - 随机贴现因子 φ(t)
    - 两个状态变量: r (利率), w (财富)
    - 两个控制变量: c (消费), π (投资组合)
    """
    
    def compute_phi(self, t: float) -> float:
        """计算随机贴现因子"""
        return np.exp(-(1 - self.theta)**(-1) * (0.2 + 0.2 * t))
    
    def compute_terminal_value(self, r: float, w: float) -> float:
        """
        终端时刻的边界值
        
        基于CRRA效用的解析近似
        """
        phi_T = self.compute_phi(self.t_max)
        # 终端值函数形式
        terminal_value = (
            self.theta**(-1) *
            phi_T**(1 - self.theta) *
            w**self.theta *
            (-6.676 - 0.8 * 1.198 * r**2 + 5.705 * r)
        )
        return terminal_value
    
    def compute_optimal_consumption(self, V_w: float, t: float = None, **kwargs) -> float:
        """
        计算最优消费
        
        一阶条件: u'(c) = V_w * φ
        对于CRRA: c^(θ-1) = V_w * φ
        因此: c = (V_w * φ)^(1/(θ-1))
        """
        if t is None:
            t = self.t_max
        
        phi = self.compute_phi(t)
        
        # 确保V_w为正
        V_w_safe = max(V_w, 1e-10)
        
        # 最优消费
        c = (V_w_safe * phi) ** (1.0 / (self.theta - 1))
        
        return max(c, 1e-10)
    
    def compute_optimal_portfolio(self, V_w: float, V_ww: float, V_wr: float,
                                   w: float, r: float, t: float = None, **kwargs) -> float:
        """
        计算最优投资组合
        
        通过求解Hamiltonian对π的一阶条件
        """
        if t is None:
            t = self.t_max
        
        phi = self.compute_phi(t)
        c = self.compute_optimal_consumption(V_w, t)
        
        # 初始猜测
        if V_ww < 0:
            pi_init = (self.mu - r) / (self.sigma**2)
        else:
            pi_init = 0.0
        pi_init = np.clip(pi_init, -5, 5)
        
        # Newton-Raphson迭代
        pi = pi_init
        for _ in range(20):
            # Hamiltonian对π的导数
            dH = (V_w * w * (self.mu - r) +
                  V_ww * (pi * self.sigma**2 - (1 - pi) * w**2 - w**2) -
                  0.5 * V_wr * self.rho * self.sigma * w)
            
            # 二阶导数
            ddH = V_ww * (self.sigma**2 + w**2)
            
            if abs(ddH) < 1e-10:
                break
            
            delta = dH / ddH
            pi_new = pi - delta
            pi_new = np.clip(pi_new, -10, 10)
            
            if abs(delta) < 1e-10:
                break
            
            pi = pi_new
        
        return np.clip(pi, -5, 5)
    
    def solve_time_step(self, V_next: np.ndarray, t_idx: int) -> np.ndarray:
        """
        使用策略迭代求解单个时间步
        """
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        
        # 初始猜测
        V_current = V_next.copy()
        c_current = np.ones((Nr, Nw)) * 0.5
        pi_current = np.zeros((Nr, Nw))
        
        t_val = self.t[t_idx]
        phi = self.compute_phi(t_val)
        
        for iter_pi in range(self.max_iter):
            # 步骤1: 构建并求解线性系统
            A, b = self._build_linear_system(V_current, V_next, t_idx, 
                                             c_current, pi_current, phi)
            
            try:
                V_new_flat = spsolve(A, b)
                V_new = V_new_flat.reshape(Nr, Nw)
            except Exception as e:
                print(f"  线性求解失败: {e}")
                V_new = 0.5 * V_current + 0.5 * V_next
            
            # 约束值函数
            V_new = np.maximum(V_new, 1e-10)
            V_new = np.minimum(V_new, 10 * np.abs(V_next).max())
            
            # 步骤2: 更新控制变量
            c_new, pi_new = self._update_controls(V_new, t_idx, phi)
            
            # 检查收敛
            V_diff = np.linalg.norm(V_new - V_current) / (np.linalg.norm(V_current) + 1e-10)
            c_diff = np.linalg.norm(c_new - c_current) / (np.linalg.norm(c_current) + 1e-10)
            
            if iter_pi % 5 == 0:
                print(f"    迭代 {iter_pi}: V变化={V_diff:.6e}, c变化={c_diff:.6e}")
            
            V_current = V_new
            c_current = c_new
            pi_current = pi_new
            
            self.c_opt[t_idx] = c_current
            self.pi_opt[t_idx] = pi_current
            
            if max(V_diff, c_diff) < self.tol:
                print(f"  策略迭代收敛于迭代 {iter_pi}")
                break
        
        return V_current
    
    def _build_linear_system(self, V_current: np.ndarray, V_next: np.ndarray,
                             t_idx: int, c: np.ndarray, pi: np.ndarray, 
                             phi: float) -> Tuple:
        """构建线性系统 A * V = b"""
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        
        # 系数数组
        drift_w = np.zeros(N_total)
        drift_r = np.zeros(N_total)
        diff_w = np.zeros(N_total)
        diff_r = np.zeros(N_total)
        utility = np.zeros(N_total)
        
        for i in range(Nr):
            for j in range(Nw):
                idx = i * Nw + j
                r_val = self.r[i]
                w_val = self.w[j]
                
                pi_ij = pi[i, j]
                c_ij = max(c[i, j], 1e-10)
                
                # 漂移系数
                drift_w[idx] = w_val * (pi_ij * self.mu + (1 - pi_ij) * r_val - c_ij)
                drift_r[idx] = self.kappa * (self.r_bar - r_val)
                
                # 扩散系数
                diff_w[idx] = (0.5 * (pi_ij * self.sigma)**2 + 
                              0.5 * ((1 - pi_ij) * w_val)**2 +
                              (1 - pi_ij) * w_val**2)
                diff_r[idx] = 0.5 * self.sigma_r**2
                
                # 效用项
                utility[idx] = UtilityFunction.crra(np.array([c_ij]), self.theta)[0]
        
        # 构建线性算子
        L = (diags(drift_w) @ self.Dw_2d + 
             diags(drift_r) @ self.Dr_2d +
             diags(diff_w) @ self.Dww_2d +
             diags(diff_r) @ self.Drr_2d)
        
        # 左端矩阵和右端向量
        A = eye(N_total) / self.dt - L
        b = V_next.reshape(N_total) / self.dt + utility
        
        return A, b
    
    def _update_controls(self, V: np.ndarray, t_idx: int, phi: float) -> Tuple[np.ndarray, np.ndarray]:
        """更新控制变量"""
        Nr, Nw = self.Nr, self.Nw
        
        V_flat = V.reshape(Nr * Nw)
        V_w = (self.Dw_2d @ V_flat).reshape(Nr, Nw)
        V_ww = (self.Dww_2d @ V_flat).reshape(Nr, Nw)
        
        # 交叉导数
        V_wr = np.zeros((Nr, Nw))
        for i in range(1, Nr-1):
            for j in range(1, Nw-1):
                V_wr[i, j] = (V[i+1, j+1] - V[i+1, j-1] - 
                              V[i-1, j+1] + V[i-1, j-1]) / (4 * self.dr * self.dw)
        
        c_new = np.zeros((Nr, Nw))
        pi_new = np.zeros((Nr, Nw))
        
        for i in range(Nr):
            for j in range(Nw):
                r_val = self.r[i]
                w_val = self.w[j]
                
                V_w_ij = max(V_w[i, j], 1e-10)
                
                c_new[i, j] = self.compute_optimal_consumption(V_w_ij, self.t[t_idx])
                pi_new[i, j] = self.compute_optimal_portfolio(
                    V_w_ij, V_ww[i, j], V_wr[i, j], w_val, r_val, self.t[t_idx]
                )
        
        return c_new, pi_new


def main():
    """主函数示例"""
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
        'max_iter': 30,
        'tol': 1e-6,
        'damping': 0.7
    }
    
    solver = HJBSolverCRRA(params)
    
    V, c_opt, pi_opt = solver.solve_backward()
    
    solver.plot_results('hjb_solutio n_crra.png')
    
    print("\n最终结果:")
    print(f"值函数范围: [{V.min():.4f}, {V.max():.4f}]")
    print(f"最优消费范围: [{c_opt.min():.4f}, {c_opt.max():.4f}]")
    print(f"最优投资组合范围: [{pi_opt.min():.4f}, {pi_opt.max():.4f}]")


if __name__ == "__main__":
    main()
