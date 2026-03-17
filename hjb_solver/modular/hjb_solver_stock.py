"""
时变股票指数影响贴现因子的 HJB 方程求解器（支持并行计算）

特点:
- 状态变量: t (时间), r (利率), X (财富)
- 控制变量: π (投资组合), C (消费)
- 股票指数 S(t) 作为时变参数影响贴现因子 ψ(t)
- 效用函数: U = (1/θ) * ψ^(1-θ) * C^θ
- 支持多进程并行加速
"""

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

from hjb_solver_base import HJBSolverBase, UtilityFunction


class HJBSolverStockIndex(HJBSolverBase):
    """
    股票指数影响贴现因子的 HJB 求解器
    
    贴现因子:
    ψ(t) = exp{ -1/(1-θ) * [φ₀t + φ₁·S(t)·t + φ₂·S(t)²·t] }
    
    HJB方程:
    0 = V_t + (α-βr)V_r + 0.5σ_r²V_rr 
        + max_{π,C} { [X(πμ+(1-π)r)-C]V_x + 0.5[(πσ)²+(1-π)²A²]X²V_xx
                      + (1/θ)ψ^(1-θ)C^θ + 0.5V_xr X(1-π) }
    """
    
    def __init__(self, params: dict):
        """
        初始化求解器
        
        额外参数:
        - stock_process: 股票指数过程 S(t) 的函数或数组
        - phi0, phi1, phi2: 贴现因子参数
        - alpha, beta: 利率过程参数
        - A: 财富扩散的额外波动率参数
        """
        # 股票相关参数
        self.stock_process = params.get('stock_process', None)
        self.phi0 = params.get('phi0', 0.2)
        self.phi1 = params.get('phi1', 0.1)
        self.phi2 = params.get('phi2', 0.05)
        
        # 利率过程参数
        self.alpha = params.get('alpha', 0.02)
        self.beta = params.get('beta', 0.1)
        
        # 财富扩散参数
        self.A = params.get('A', 0.1)
        
        super().__init__(params)
        
        # 预计算股票指数和贴现因子
        self._compute_stock_and_psi()
    
    def _compute_stock_and_psi(self):
        """计算股票指数 S(t) 和贴现因子 ψ(t)"""
        if self.stock_process is None:
            # 默认股票指数: 简单的均值回归或趋势过程
            # S(t) = S0 * exp(μ_s * t) 或常数
            S0 = self.params.get('S0', 1.0)
            mu_s = self.params.get('mu_s', 0.05)
            self.S = S0 * np.exp(mu_s * self.t)
        elif callable(self.stock_process):
            # 如果传入的是函数
            self.S = np.array([self.stock_process(ti) for ti in self.t])
        else:
            # 如果传入的是数组
            self.S = np.array(self.stock_process)
            if len(self.S) != self.Nt:
                raise ValueError(f"stock_process长度({len(self.S)})必须与Nt({self.Nt})一致")
        
        # 计算贴现因子 ψ(t)
        # ψ(t) = exp{ -1/(1-θ) * [φ₀t + φ₁·S(t)·t + φ₂·S(t)²·t] }
        integral_term = (self.phi0 * self.t + 
                        self.phi1 * self.S * self.t + 
                        self.phi2 * self.S**2 * self.t)
        self.psi = np.exp(-integral_term / (1 - self.theta))
        
        print(f"股票指数范围: [{self.S.min():.4f}, {self.S.max():.4f}]")
        print(f"贴现因子范围: [{self.psi.min():.4f}, {self.psi.max():.4f}]")
    
    def compute_terminal_value(self, r: float, X: float) -> float:
        """
        终端时刻的边界值
        
        基于效用函数的猜测形式
        V(T) = (1/θ) * ψ(T)^(1-θ) * X^θ
        """
        psi_T = self.psi[-1]
        X_safe = max(X, 1e-10)
        
        # 简化的终端条件（不依赖r，使问题更稳定）
        terminal_value = (1.0 / self.theta) * (psi_T ** (1 - self.theta)) * (X_safe ** self.theta)
        return terminal_value
    
    def compute_optimal_consumption(self, V_x: float, t_idx: int, **kwargs) -> float:
        """
        计算最优消费（绝对消费 C，不是消费率 c）
        
        一阶条件: ∂H/∂C = 0
        H中的消费项: -C*V_x + (1/θ)*ψ^(1-θ)*C^θ
        一阶条件: -V_x + ψ^(1-θ)*C^(θ-1) = 0
        因此: C = [ψ^(1-θ) / V_x]^(1/(1-θ))
        
        经济学约束: C ≤ X（不能消费超过财富）
        """
        psi_t = self.psi[t_idx]
        X = kwargs.get('X', 1.0)
        
        # 确保 V_x 为正
        V_x_safe = max(V_x, 1e-10)
        
        # 计算理论最优消费
        gamma = 1.0 - self.theta  # γ > 0
        psi_gamma = psi_t ** gamma
        
        C_optimal = (psi_gamma / V_x_safe) ** (1.0 / gamma)
        
        # 经济学约束: 消费不能超过财富（不能借贷）
        # 同时消费不能为负
        C = np.clip(C_optimal, 1e-6, X)
        
        return C
    
    def compute_optimal_portfolio(self, V_x: float, V_xx: float, V_xr: float,
                                   X: float, r: float, t_idx: int, **kwargs) -> float:
        """
        计算最优投资组合
        
        Hamiltonian对π的导数:
        dH/dπ = V_x * X * (r - μ) 
                + V_xx * X² * [πσ² - (1-π)A²] 
                - 0.5 * V_xr * X
        = 0
        
        其中交叉导数项 0.5 * V_xr * X * (1-π) 对π求导得到 -0.5 * V_xr * X
        
        解出:
        π = [V_x * X * (r - μ) + V_xx * X² * A² + 0.5 * V_xr * X] 
            / [V_xx * X² * (σ² + A²)]
        """
        # 安全检查
        if not np.isfinite(V_x) or not np.isfinite(V_xx):
            return 0.0
        
        # 确保二阶导为负（凹性）
        if V_xx >= -1e-10:
            # 如果V_xx不够负，使用Merton比例
            if self.sigma > 0:
                pi_merton = (self.mu - r) / (self.sigma**2 * (1 - self.theta))
                return np.clip(pi_merton, -1, 1)
            return 0.0
        
        # 确保X为正
        X_safe = max(X, 1e-10)
        
        # 分子: V_x * X * (r - μ) + V_xx * X² * A² + 0.5 * V_xr * X
        # 注意: 交叉导数项 0.5 * V_xr * X * (1-π) 对π求导得到 -0.5 * V_xr * X，移到等式另一边为 +0.5 * V_xr * X
        numerator = (V_x * X_safe * (r - self.mu) + 
                     V_xx * X_safe**2 * self.A**2 + 
                     0.5 * V_xr * X_safe)
        
        # 分母: V_xx * X² * (σ² + A²)
        denominator = V_xx * X_safe**2 * (self.sigma**2 + self.A**2)
        
        if abs(denominator) < 1e-10:
            return 0.0
        
        pi = numerator / denominator
        
        # 限制投资组合范围
        return np.clip(pi, -1, 1)
    
    def solve_time_step(self, V_next: np.ndarray, t_idx: int) -> np.ndarray:
        """
        使用策略迭代求解单个时间步
        """
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        
        # 初始猜测
        V_current = V_next.copy()
        C_current = np.ones((Nr, Nw)) * 0.5
        pi_current = np.zeros((Nr, Nw))
        
        for iter_pi in range(self.max_iter):
            # 步骤1: 构建并求解线性系统
            A, b = self._build_linear_system(V_current, V_next, t_idx,
                                             C_current, pi_current)
            
            try:
                V_new_flat = spsolve(A, b)
                V_new = V_new_flat.reshape(Nr, Nw)
                # 清理数值问题
                V_new = np.nan_to_num(V_new, nan=1e-10, posinf=1e10, neginf=1e-10)
            except Exception as e:
                print(f"  线性求解失败: {e}")
                V_new = 0.5 * V_current + 0.5 * V_next
            
            # 约束值函数
            V_new = np.clip(V_new, 1e-10, 1e8)
            
            # 步骤2: 更新控制变量
            C_calc, pi_calc = self._update_controls(V_new, t_idx)
            
            # 【修复：引入阻尼 Damping，防止策略迭代来回震荡发散】
            # 使用 params 里的 damping 参数（0.7意味着保留70%的新值，30%的旧值）
            damping = self.params.get('damping', 0.7)
            C_new = damping * C_calc + (1 - damping) * C_current
            pi_new = damping * pi_calc + (1 - damping) * pi_current
            
            # 检查收敛
            V_diff = np.linalg.norm(V_new - V_current) / (np.linalg.norm(V_current) + 1e-10)
            C_diff = np.linalg.norm(C_new - C_current) / (np.linalg.norm(C_current) + 1e-10)
            
            if iter_pi % 5 == 0:
                print(f"    迭代 {iter_pi}: V变化={V_diff:.6e}, C变化={C_diff:.6e}")
            
            V_current = V_new
            C_current = C_new
            pi_current = pi_new
            
            self.c_opt[t_idx] = C_current
            self.pi_opt[t_idx] = pi_current
            
            if max(V_diff, C_diff) < self.tol:
                print(f"  策略迭代收敛于迭代 {iter_pi}")
                break
        
        return V_current
    
    def _build_linear_system(self, V_current: np.ndarray, V_next: np.ndarray,
                             t_idx: int, C: np.ndarray, pi: np.ndarray) -> Tuple:
        """构建线性系统 A * V = b（向量化加速版本）"""
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        
        psi_t = self.psi[t_idx]
        
        # 使用向量化操作替代双重循环
        # 创建网格
        R_grid, X_grid = np.meshgrid(self.r, self.w, indexing='ij')
        
        # 确保 C 为正
        C_safe = np.maximum(C, 1e-10)
        
        # 向量化计算所有系数
        # X的漂移: X(πμ+(1-π)r) - C
        drift_X = (X_grid * (pi * self.mu + (1 - pi) * R_grid) - C_safe).flatten()
        
        # r的漂移: α - βr
        drift_r = (self.alpha - self.beta * R_grid).flatten()
        
        # X的扩散: 0.5 * [(πσ)² + (1-π)²A²] * X²
        diff_X = (0.5 * ((pi * self.sigma)**2 + ((1 - pi) * self.A)**2) * X_grid**2).flatten()
        
        # r的扩散: 常数
        diff_r = np.full(N_total, 0.5 * self.sigma_r**2)
        
        # 交叉导数项系数: 0.5 * X * (1-π) (来自HJB方程中的 0.5 * V_xr * X * (1-π))
        cross_term = (0.5 * X_grid * (1 - pi)).flatten()
        
        # 效用项: (1/θ) * ψ^(1-θ) * C^θ
        utility = ((1.0 / self.theta) * (psi_t ** (1 - self.theta)) * (C_safe ** self.theta)).flatten()
        
        # 构建线性算子
        L = (diags(drift_X) @ self.Dw_2d + 
             diags(drift_r) @ self.Dr_2d +
             diags(diff_X) @ self.Dww_2d +
             diags(diff_r) @ self.Drr_2d +
             diags(cross_term) @ self.Dwr_2d)
        
        # 左端矩阵和右端向量
        A = eye(N_total) / self.dt - L
        b = V_next.reshape(N_total) / self.dt + utility
        
        return A, b
    
    def _update_controls(self, V: np.ndarray, t_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """更新控制变量"""
        Nr, Nw = self.Nr, self.Nw
        
        # 确保V有限
        V = np.nan_to_num(V, nan=1e-10, posinf=1e10, neginf=1e-10)
        V = np.maximum(V, 1e-10)
        
        V_flat = V.reshape(Nr * Nw)
        V_x = (self.Dw_2d @ V_flat).reshape(Nr, Nw)  # 对X的导数
        V_xx = (self.Dww_2d @ V_flat).reshape(Nr, Nw)  # 对X的二阶导
        
        # 清理导数
        V_x = np.nan_to_num(V_x, nan=1e-10, posinf=1e10, neginf=1e-10)
        V_xx = np.nan_to_num(V_xx, nan=-1.0, posinf=-1e-10, neginf=-1e10)
        
        # 交叉导数 V_xr（使用向量化卷积）
        V_xr = np.zeros((Nr, Nw))
        if Nr > 2 and Nw > 2:
            # 中心区域使用向量化计算
            V_xr[1:-1, 1:-1] = (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2]) / (4 * self.dr * self.dw)
            
            # 【修复：给四周边缘做外推平滑，防止边界休克】
            V_xr[0, :] = V_xr[1, :]
            V_xr[-1, :] = V_xr[-2, :]
            V_xr[:, 0] = V_xr[:, 1]
            V_xr[:, -1] = V_xr[:, -2]
            
        V_xr = np.nan_to_num(V_xr, nan=0.0)
        
        # 向量化计算控制变量
        R_grid, X_grid = np.meshgrid(self.r, self.w, indexing='ij')
        
        # 确保导数有效
        V_x_safe = np.clip(V_x, 1e-10, 1e10)
        
        # 计算最优消费（向量化）
        psi_t = self.psi[t_idx]
        gamma = 1.0 - self.theta
        psi_gamma = psi_t ** gamma
        C_new = (psi_gamma / V_x_safe) ** (1.0 / gamma)
        C_new = np.clip(C_new, 1e-6, X_grid)
        
        # 计算最优投资组合（向量化）
        # 【修复：强制 V_xx 保持负数（凹性），防止策略爆炸】
        # 这里不仅是为了防除0，更是强制执行经济学的严格风险厌恶
        V_xx_safe = np.minimum(V_xx, -1e-6)
        
        # 分子: V_x * X * (r - μ) + V_xx * X² * A² + 0.5 * V_xr * X
        # 注意: 交叉导数项 0.5 * V_xr * X * (1-π) 对π求导得到 -0.5 * V_xr * X，移到等式另一边为 +0.5 * V_xr * X
        numerator = (V_x_safe * X_grid * (R_grid - self.mu) + 
             V_xx_safe * (X_grid**2) * (self.A**2) + 
             0.5 * V_xr * X_grid)

        # 分母: 使用安全的 V_xx_safe
        denominator = V_xx_safe * (X_grid**2) * (self.sigma**2 + self.A**2)

        pi_new = numerator / denominator
        pi_new = np.clip(pi_new, -1, 1) # 依据你的设定截断
        
        return C_new, pi_new


def main():
    """主函数示例"""
    # 定义时变股票指数过程
    def stock_process(t):
        """股票指数: S(t) = 1 + 0.1*sin(2πt) + 0.05*t"""
        return 1.0 + 0.1 * np.sin(2 * np.pi * t) + 0.05 * t
    
    params = {
        # 网格参数（向量化后可以使用更细的网格）
        't_min': 0.0,
        't_max': 1.0,
        'Nt': 100,  # 时间步数
        'r_min': 0.01,
        'r_max': 1,
        'Nr': 50,   # 利率网格
        'w_min': 0.5,  # X_min
        'w_max': 4.0,  # X_max
        'Nw': 50,   # 财富网格
        
        # 经济参数
        'theta': 0.7,
        'mu': 0.08,
        'sigma': 0.2,
        'sigma_r': 0.05,
        'r_bar': 0.05,
        
        # 股票指数和贴现因子参数
        'stock_process': stock_process,
        'phi0': 0.2,
        'phi1': 0.1,
        'phi2': 0.05,
        
        # 利率过程参数
        'alpha': 0.02,
        'beta': 0.1,
        
        # 财富扩散参数
        'A': 0.1,
        
        # 数值参数
        'max_iter': 30,
        'tol': 1e-6,
        'damping': 0.7
    }
    
    print("=" * 60)
    print("时变股票指数影响贴现因子的HJB方程求解")
    print("=" * 60)
    
    solver = HJBSolverStockIndex(params)
    
    V, C_opt, pi_opt = solver.solve_backward()
    
    solver.plot_results('hjb_solution_stock.png')
    
    print("\n最终结果:")
    print(f"值函数范围: [{V.min():.4f}, {V.max():.4f}]")
    print(f"最优消费范围: [{C_opt.min():.4f}, {C_opt.max():.4f}]")
    print(f"最优投资组合范围: [{pi_opt.min():.4f}, {pi_opt.max():.4f}]")


if __name__ == "__main__":
    main()
