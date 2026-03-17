"""
对数效用函数下的HJB方程求解器
展示如何基于通用框架快速实现不同的效用函数
"""

import numpy as np
from typing import Tuple

from hjb_solver_base import HJBSolverBase, UtilityFunction
from hjb_solver_crra import HJBSolverCRRA


class HJBSolverLog(HJBSolverBase):
    """
    对数效用函数下的HJB方程求解器
    
    与CRRA版本的主要区别:
    - 效用函数: u(c) = log(c) 而不是 c^θ/θ
    - 边际效用: u'(c) = 1/c
    - 最优消费: c = 1/(V_w * φ)
    """
    
    def __init__(self, params: dict):
        """初始化 - 对数效用的theta参数不需要"""
        # 对数效用相当于theta=0的极限情况
        params = params.copy()
        params['theta'] = 0.0  # 标记为对数效用
        super().__init__(params)
    
    def compute_phi(self, t: float) -> float:
        """随机贴现因子 - 对数效用形式不同"""
        # 对数效用的贴现因子
        return np.exp(-0.05 * t)  # 简化的贴现因子
    
    def compute_terminal_value(self, r: float, w: float) -> float:
        """终端值函数 - 对数效用形式"""
        phi_T = self.compute_phi(self.t_max)
        # 对数效用的终端值近似
        return phi_T * np.log(max(w, 1e-10)) * (1 + 0.1 * r)
    
    def compute_optimal_consumption(self, V_w: float, t: float = None, **kwargs) -> float:
        """
        对数效用的最优消费
        
        一阶条件: u'(c) = V_w * φ
        对于对数效用: 1/c = V_w * φ
        因此: c = 1/(V_w * φ)
        """
        if t is None:
            t = self.t_max
        
        phi = self.compute_phi(t)
        
        # 确保V_w为正
        V_w_safe = max(V_w, 1e-10)
        
        # 最优消费
        c = 1.0 / (V_w_safe * phi)
        
        return max(c, 1e-10)
    
    def compute_optimal_portfolio(self, V_w: float, V_ww: float, V_wr: float,
                                   w: float, r: float, t: float = None, **kwargs) -> float:
        """
        最优投资组合 - 与CRRA类似，但使用对数效用的值函数导数
        """
        if t is None:
            t = self.t_max
        
        # 对数效用的投资组合公式与CRRA类似
        # 主要区别在于值函数的形态
        
        if V_ww < 0:
            pi_init = (self.mu - r) / (self.sigma**2)
        else:
            pi_init = 0.0
        
        pi_init = np.clip(pi_init, -5, 5)
        
        # 简化的Newton迭代
        pi = pi_init
        for _ in range(20):
            dH = (V_w * w * (self.mu - r) +
                  V_ww * (pi * self.sigma**2 - (1 - pi) * w**2 - w**2) -
                  0.5 * V_wr * self.rho * self.sigma * w)
            
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
        求解单个时间步 - 复用CRRA版本的逻辑，但使用对数效用的效用函数
        """
        # 为了代码复用，可以创建一个内部方法来构建线性系统
        # 这里简化处理，直接调用父类的策略迭代框架
        
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        
        V_current = V_next.copy()
        c_current = np.ones((Nr, Nw)) * 0.5
        pi_current = np.zeros((Nr, Nw))
        
        t_val = self.t[t_idx]
        phi = self.compute_phi(t_val)
        
        for iter_pi in range(self.max_iter):
            # 构建线性系统
            A, b = self._build_linear_system_log(V_current, V_next, t_idx,
                                                  c_current, pi_current, phi)
            
            try:
                from scipy.sparse.linalg import spsolve
                V_new_flat = spsolve(A, b)
                V_new = V_new_flat.reshape(Nr, Nw)
            except Exception as e:
                print(f"  线性求解失败: {e}")
                V_new = 0.5 * V_current + 0.5 * V_next
            
            V_new = np.maximum(V_new, 1e-10)
            
            # 更新控制
            c_new, pi_new = self._update_controls_log(V_new, t_idx, phi)
            
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
    
    def _build_linear_system_log(self, V_current, V_next, t_idx, c, pi, phi):
        """为对数效用构建线性系统"""
        from scipy.sparse import diags, eye
        
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        
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
                
                drift_w[idx] = w_val * (pi_ij * self.mu + (1 - pi_ij) * r_val - c_ij)
                drift_r[idx] = self.kappa * (self.r_bar - r_val)
                
                diff_w[idx] = (0.5 * (pi_ij * self.sigma)**2 + 
                              0.5 * ((1 - pi_ij) * w_val)**2 +
                              (1 - pi_ij) * w_val**2)
                diff_r[idx] = 0.5 * self.sigma_r**2
                
                # 对数效用
                utility[idx] = UtilityFunction.log(np.array([c_ij]))[0]
        
        L = (diags(drift_w) @ self.Dw_2d + 
             diags(drift_r) @ self.Dr_2d +
             diags(diff_w) @ self.Dww_2d +
             diags(diff_r) @ self.Drr_2d)
        
        A = eye(N_total) / self.dt - L
        b = V_next.reshape(N_total) / self.dt + utility
        
        return A, b
    
    def _update_controls_log(self, V, t_idx, phi):
        """更新控制变量 - 对数效用版本"""
        Nr, Nw = self.Nr, self.Nw
        
        V_flat = V.reshape(Nr * Nw)
        V_w = (self.Dw_2d @ V_flat).reshape(Nr, Nw)
        V_ww = (self.Dww_2d @ V_flat).reshape(Nr, Nw)
        
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
    """主函数示例 - 对数效用"""
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
    
    print("=" * 60)
    print("对数效用下的HJB方程求解")
    print("=" * 60)
    
    solver = HJBSolverLog(params)
    
    V, c_opt, pi_opt = solver.solve_backward()
    
    solver.plot_results('hjb_solution_log.png')
    
    print("\n最终结果:")
    print(f"值函数范围: [{V.min():.4f}, {V.max():.4f}]")
    print(f"最优消费范围: [{c_opt.min():.4f}, {c_opt.max():.4f}]")
    print(f"最优投资组合范围: [{pi_opt.min():.4f}, {pi_opt.max():.4f}]")


if __name__ == "__main__":
    main()
