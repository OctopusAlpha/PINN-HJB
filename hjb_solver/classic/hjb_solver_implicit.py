"""
HJB方程隐式有限差分法求解器
使用 Newton-Raphson 方法求解非线性方程组

三维状态变量: t (时间), r (利率), w (财富)
控制变量: c (消费), π (单一风险资产投资比例)
"""

import numpy as np
from scipy.sparse import csr_matrix, eye, kron, diags
from scipy.sparse.linalg import spsolve, gmres, bicgstab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import newton
import warnings
warnings.filterwarnings('ignore')


class HJBSolverImplicitFDM:
    """
    HJB方程求解器 - 隐式FDM + Newton-Raphson
    
    方程形式:
    0 = V_t + H(t, r, w, V, V_r, V_w, V_rr, V_ww, V_wr, c, π)
    
    其中 Hamiltonian H 包含:
    - 漂移项: V_w * w * (π*μ + (1-π)*r - c) + V_r * κ*(r̄-r)
    - 扩散项: 0.5*σ_r²*V_rr + V_ww*[0.5*(π*σ)² + 0.5*((1-π)*w)² + (1-π)*w²]
    - 交叉项: 0.5*V_wr*ρ*σ*(1-π)*w
    
    终端条件: V(T, r, w) = boundary_value
    """
    
    def __init__(self, params):
        """初始化求解器"""
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
        
        # HJB方程参数
        self.theta = params.get('theta', 0.7)
        self.rho = params.get('rho', 0.5)
        self.mu = params.get('mu', 0.08)
        self.sigma = params.get('sigma', 0.2)
        self.sigma_r = params.get('sigma_r', 0.05)
        self.kappa = params.get('kappa', 0.1)
        self.r_bar = params.get('r_bar', 0.05)
        
        # Newton-Raphson 参数
        self.max_iter_nr = params.get('max_iter_nr', 50)
        self.tol_nr = params.get('tol_nr', 1e-8)
        self.damping = params.get('damping', 0.8)  # 阻尼因子
        
        # 值函数和控制变量
        self.V = np.zeros((self.Nt, self.Nr, self.Nw))
        self.c_opt = np.zeros((self.Nt, self.Nr, self.Nw))
        self.pi_opt = np.zeros((self.Nt, self.Nr, self.Nw))
        
        # 预计算微分算子矩阵
        self._build_differential_operators()
        
    def _build_differential_operators(self):
        """
        构建空间微分算子的稀疏矩阵表示
        用于隐式格式中的线性部分
        """
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        
        # r方向的微分算子 (一阶和二阶)
        Dr = np.zeros((Nr, Nr))
        Drr = np.zeros((Nr, Nr))
        
        for i in range(Nr):
            if i == 0:  # 左边界 (前向差分)
                Dr[i, i] = -1.0 / self.dr
                Dr[i, i+1] = 1.0 / self.dr
                Drr[i, i] = 2.0 / (self.dr**2)
                Drr[i, i+1] = -5.0 / (self.dr**2)
                Drr[i, i+2] = 4.0 / (self.dr**2)
                Drr[i, i+3] = -1.0 / (self.dr**2)
            elif i == Nr - 1:  # 右边界 (后向差分)
                Dr[i, i-1] = -1.0 / self.dr
                Dr[i, i] = 1.0 / self.dr
                Drr[i, i] = 2.0 / (self.dr**2)
                Drr[i, i-1] = -5.0 / (self.dr**2)
                Drr[i, i-2] = 4.0 / (self.dr**2)
                Drr[i, i-3] = -1.0 / (self.dr**2)
            else:  # 内部 (中心差分)
                Dr[i, i-1] = -0.5 / self.dr
                Dr[i, i+1] = 0.5 / self.dr
                Drr[i, i-1] = 1.0 / (self.dr**2)
                Drr[i, i] = -2.0 / (self.dr**2)
                Drr[i, i+1] = 1.0 / (self.dr**2)
        
        # w方向的微分算子
        Dw = np.zeros((Nw, Nw))
        Dww = np.zeros((Nw, Nw))
        
        for j in range(Nw):
            if j == 0:  # 左边界
                Dw[j, j] = -1.0 / self.dw
                Dw[j, j+1] = 1.0 / self.dw
                Dww[j, j] = 2.0 / (self.dw**2)
                Dww[j, j+1] = -5.0 / (self.dw**2)
                Dww[j, j+2] = 4.0 / (self.dw**2)
                Dww[j, j+3] = -1.0 / (self.dw**2)
            elif j == Nw - 1:  # 右边界
                Dw[j, j-1] = -1.0 / self.dw
                Dw[j, j] = 1.0 / self.dw
                Dww[j, j] = 2.0 / (self.dw**2)
                Dww[j, j-1] = -5.0 / (self.dw**2)
                Dww[j, j-2] = 4.0 / (self.dw**2)
                Dww[j, j-3] = -1.0 / (self.dw**2)
            else:  # 内部
                Dw[j, j-1] = -0.5 / self.dw
                Dw[j, j+1] = 0.5 / self.dw
                Dww[j, j-1] = 1.0 / (self.dw**2)
                Dww[j, j] = -2.0 / (self.dw**2)
                Dww[j, j+1] = 1.0 / (self.dw**2)
        
        # 转换为稀疏矩阵
        self.Dr = csr_matrix(Dr)
        self.Drr = csr_matrix(Drr)
        self.Dw = csr_matrix(Dw)
        self.Dww = csr_matrix(Dww)
        
        # 单位矩阵
        I_r = eye(Nr)
        I_w = eye(Nw)
        
        # 2D微分算子 (Kronecker积)
        self.Dr_2d = kron(self.Dr, I_w)  # ∂/∂r
        self.Dw_2d = kron(I_r, self.Dw)  # ∂/∂w
        self.Drr_2d = kron(self.Drr, I_w)  # ∂²/∂r²
        self.Dww_2d = kron(I_r, self.Dww)  # ∂²/∂w²
        
        # 交叉导数算子 (使用中心差分)
        self.Drw_2d = self.Dr_2d @ self.Dw_2d  # 近似 ∂²/∂r∂w
        
    def compute_phi(self, t):
        """计算随机贴现因子"""
        phi = np.exp(-(1-self.theta)**(-1) * (0.2 + 0.2*t))
        return phi
    
    def compute_terminal_value(self, r, w):
        """计算终端时刻的边界值"""
        phi_T = self.compute_phi(self.t_max)
        tt = (self.theta**(-1) * 
              phi_T**(1-self.theta) * 
              w**self.theta * 
              (-6.676 - 0.8*1.198*r**2 + 5.705*r))
        return tt
    
    def compute_optimal_consumption(self, V_w, phi):
        """
        计算最优消费 c = (V_w)^(θ-1) * φ
        使用 Newton-Raphson 求解非线性方程
        """
        V_w_clipped = np.maximum(V_w, 1e-10)
        c = V_w_clipped**(self.theta - 1) * phi
        return np.maximum(c, 1e-10)
    
    def compute_optimal_portfolio(self, V_w, V_ww, V_wr, w, r_val, phi):
        """
        使用 Newton-Raphson 求解最优投资组合 π
        
        一阶条件: ∂H/∂π = 0
        H = V_w * w * (π*μ + (1-π)*r - c) + V_ww * [0.5*(π*σ)² + 0.5*((1-π)*w)² + (1-π)*w²]
            + 0.5*V_wr*ρ*σ*(1-π)*w
        """
        # 初始猜测
        pi_init = (self.mu - r_val) / (self.sigma**2) if V_ww < 0 else 0.0
        pi_init = np.clip(pi_init, -5, 5)
        
        def hamiltonian_derivative(pi):
            """Hamiltonian 对 π 的导数"""
            # 消费对 π 的导数 (通过包络定理)
            c = self.compute_optimal_consumption(V_w, phi)
            
            # dH/dπ
            dH = (V_w * w * (self.mu - r_val) +
                  V_ww * (pi * self.sigma**2 - (1-pi) * w**2 - w**2) -
                  0.5 * V_wr * self.rho * self.sigma * w)
            return dH
        
        def hamiltonian_second_derivative(pi):
            """Hamiltonian 对 π 的二阶导数"""
            return V_ww * (self.sigma**2 + w**2)
        
        # Newton-Raphson 迭代
        pi = pi_init
        for _ in range(20):
            dH = hamiltonian_derivative(pi)
            ddH = hamiltonian_second_derivative(pi)
            
            if abs(ddH) < 1e-10:
                break
                
            delta = dH / ddH
            pi_new = pi - delta
            
            # 阻尼和边界处理
            pi_new = np.clip(pi_new, -10, 10)
            
            if abs(delta) < 1e-10:
                break
                
            pi = pi_new
            
        return np.clip(pi, -5, 5)
    
    def compute_residual(self, V_current, V_next, t_idx):
        """
        计算 HJB 方程的残差 (用于 Newton-Raphson)
        
        隐式格式: (V^{n} - V^{n+1})/dt + H(V^{n}) = 0
        即: F(V^{n}) = V^{n} - V^{n+1} + dt * H(V^{n}) = 0
        """
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        
        # 将 V_current 和 V_next 展平
        V_curr_flat = V_current.reshape(N_total)
        V_next_flat = V_next.reshape(N_total)
        
        # 计算导数 (使用矩阵乘法)
        V_r = (self.Dr_2d @ V_curr_flat).reshape(Nr, Nw)
        V_w = (self.Dw_2d @ V_curr_flat).reshape(Nr, Nw)
        V_rr = (self.Drr_2d @ V_curr_flat).reshape(Nr, Nw)
        V_ww = (self.Dww_2d @ V_curr_flat).reshape(Nr, Nw)
        
        # 交叉导数 (使用有限差分近似)
        V_wr = np.zeros((Nr, Nw))
        for i in range(1, Nr-1):
            for j in range(1, Nw-1):
                V_wr[i, j] = (V_current[i+1, j+1] - V_current[i+1, j-1] - 
                              V_current[i-1, j+1] + V_current[i-1, j-1]) / (4 * self.dr * self.dw)
        
        # 当前时间的参数
        t_val = self.t[t_idx]
        phi = self.compute_phi(t_val)
        
        # 计算最优控制和 Hamiltonian
        residual = np.zeros((Nr, Nw))
        
        for i in range(Nr):
            for j in range(Nw):
                r_val = self.r[i]
                w_val = self.w[j]
                
                # 保证 V_w 为正
                V_w_ij = max(V_w[i, j], 1e-10)
                
                # 计算最优消费
                c = self.compute_optimal_consumption(V_w_ij, phi)
                
                # 计算最优投资组合
                pi = self.compute_optimal_portfolio(
                    V_w_ij, V_ww[i, j], V_wr[i, j], w_val, r_val, phi
                )
                
                # 存储控制变量
                self.c_opt[t_idx, i, j] = c
                self.pi_opt[t_idx, i, j] = pi
                
                # 计算 Hamiltonian 的各项
                # 漂移项
                drift_w = pi * self.mu + (1 - pi) * r_val - c
                drift_r = self.kappa * (self.r_bar - r_val)
                
                # Hamiltonian
                H = (V_w_ij * w_val * drift_w +
                     V_r[i, j] * drift_r +
                     0.5 * self.sigma_r**2 * V_rr[i, j] +
                     V_ww[i, j] * (0.5 * (pi * self.sigma)**2 + 
                                   0.5 * ((1 - pi) * w_val)**2 +
                                   (1 - pi) * w_val**2) +
                     0.5 * V_wr[i, j] * self.rho * self.sigma * (1 - pi) * w_val)
                
                # 隐式格式的残差: F(V) = V - V_next + dt * H
                residual[i, j] = V_current[i, j] - V_next[i, j] + self.dt * H
        
        return residual
    
    def compute_jacobian_approximation(self, V_current, t_idx):
        """
        计算残差对 V 的 Jacobian 矩阵的近似
        使用有限差分法近似
        """
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        eps = 1e-8
        
        # 基础残差
        V_next = V_current  # 在迭代中，V_next 是固定的
        F_base = self.compute_residual(V_current, V_next, t_idx).flatten()
        
        # 构建 Jacobian (稀疏矩阵)
        rows, cols, data = [], [], []
        
        for idx in range(min(N_total, 500)):  # 限制计算量
            i, j = idx // Nw, idx % Nw
            
            V_perturbed = V_current.copy()
            V_perturbed[i, j] += eps
            
            F_perturbed = self.compute_residual(V_perturbed, V_next, t_idx).flatten()
            
            # 有限差分近似导数
            dF = (F_perturbed - F_base) / eps
            
            # 只保留显著的元素
            for k in range(N_total):
                if abs(dF[k]) > 1e-12:
                    rows.append(k)
                    cols.append(idx)
                    data.append(dF[k])
        
        J = csr_matrix((data, (rows, cols)), shape=(N_total, N_total))
        return J, F_base
    
    def solve_time_step_policy_iteration(self, V_next, t_idx):
        """
        使用 Policy Iteration（策略迭代）求解单个时间步
        
        策略迭代对于 HJB 方程更稳定：
        1. 固定控制变量 (c, π)，求解线性 PDE 得到 V
        2. 更新控制变量
        3. 重复直到收敛
        
        线性系统: (I - dt * L^{k}) V^{n} = V^{n+1} - dt * f^{k}
        其中 L^{k} 是固定控制下的线性算子，f^{k} 是相应的源项
        """
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        
        # 初始猜测
        V_current = V_next.copy()
        
        # 初始化控制变量（使用上一时间步的值或默认值）
        c_current = np.ones((Nr, Nw)) * 0.5
        pi_current = np.zeros((Nr, Nw))
        
        print(f"  策略迭代开始")
        
        for iter_pi in range(self.max_iter_nr):
            # 步骤1: 构建并求解线性系统（固定控制变量）
            A, b = self._build_linear_system(V_current, V_next, t_idx, c_current, pi_current)
            
            try:
                V_new_flat = spsolve(A, b)
                V_new = V_new_flat.reshape(Nr, Nw)
            except Exception as e:
                print(f"  线性求解失败: {e}，使用松弛更新")
                V_new = 0.5 * V_current + 0.5 * V_next
            
            # 确保值函数合理（单调性和有界性约束）
            V_new = np.maximum(V_new, 1e-10)  # 正值约束
            V_new = np.minimum(V_new, 10 * np.abs(V_next).max())  # 上界约束
            
            # 步骤2: 更新控制变量
            c_new, pi_new = self._update_controls(V_new, t_idx)
            
            # 计算收敛性
            V_diff = np.linalg.norm(V_new - V_current) / (np.linalg.norm(V_current) + 1e-10)
            c_diff = np.linalg.norm(c_new - c_current) / (np.linalg.norm(c_current) + 1e-10)
            pi_diff = np.linalg.norm(pi_new - pi_current) / (np.linalg.norm(pi_current) + 1e-10)
            
            max_diff = max(V_diff, c_diff, pi_diff)
            
            if iter_pi % 5 == 0 or max_diff < self.tol_nr:
                print(f"    迭代 {iter_pi}: V变化={V_diff:.6e}, c变化={c_diff:.6e}, π变化={pi_diff:.6e}")
            
            # 更新
            V_current = V_new
            c_current = c_new
            pi_current = pi_new
            
            # 存储控制变量
            self.c_opt[t_idx] = c_current
            self.pi_opt[t_idx] = pi_current
            
            if max_diff < self.tol_nr:
                print(f"  策略迭代收敛于迭代 {iter_pi}")
                break
        
        return V_current
    
    def _build_linear_system(self, V_current, V_next, t_idx, c, pi):
        """
        构建固定控制变量下的线性系统: A * V = b
        
        离散格式: (V^n - V^{n+1})/dt + H(V^n; c, π) = 0
        即: [I/dt - L(c,π)] V^n = V^{n+1}/dt + u(c)
        
        其中 H 包含:
        - 效用项: u(c) = c^θ / θ  (CRRA效用)
        - 漂移项: V_w * w * (π*μ + (1-π)*r - c) + V_r * κ*(r̄-r)
        - 扩散项: 0.5*σ_r²*V_rr + V_ww*[0.5*(π*σ)² + 0.5*((1-π)*w)² + (1-π)*w²]
        - 交叉项: 0.5*V_wr*ρ*σ*(1-π)*w
        """
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        
        t_val = self.t[t_idx]
        
        # 构建对角系数矩阵（漂移和扩散系数）
        drift_w_diag = np.zeros(N_total)  # w * (π*μ + (1-π)*r - c)
        drift_r_diag = np.zeros(N_total)  # κ*(r̄-r)
        diff_w_diag = np.zeros(N_total)   # 0.5*(π*σ)² + 0.5*((1-π)*w)² + (1-π)*w²
        diff_r_diag = np.zeros(N_total)   # 0.5*σ_r²
        cross_coef = np.zeros(N_total)    # 0.5*ρ*σ*(1-π)*w
        
        # 效用项源项 u(c) = c^θ / θ
        utility_source = np.zeros(N_total)
        
        for i in range(Nr):
            for j in range(Nw):
                idx = i * Nw + j
                r_val = self.r[i]
                w_val = self.w[j]
                
                pi_ij = pi[i, j]
                c_ij = max(c[i, j], 1e-10)  # 确保消费为正
                
                # 漂移系数
                drift_w_diag[idx] = w_val * (pi_ij * self.mu + (1 - pi_ij) * r_val - c_ij)
                drift_r_diag[idx] = self.kappa * (self.r_bar - r_val)
                
                # 扩散系数
                diff_w_diag[idx] = (0.5 * (pi_ij * self.sigma)**2 + 
                                   0.5 * ((1 - pi_ij) * w_val)**2 +
                                   (1 - pi_ij) * w_val**2)
                diff_r_diag[idx] = 0.5 * self.sigma_r**2
                
                # 交叉项系数
                cross_coef[idx] = 0.5 * self.rho * self.sigma * (1 - pi_ij) * w_val
                
                # 效用项: u(c) = c^θ / θ
                utility_source[idx] = (c_ij ** self.theta) / self.theta
        
        # 构建线性算子 L = drift_w * Dw + drift_r * Dr + diff_w * Dww + diff_r * Drr
        L = (diags(drift_w_diag) @ self.Dw_2d + 
             diags(drift_r_diag) @ self.Dr_2d +
             diags(diff_w_diag) @ self.Dww_2d +
             diags(diff_r_diag) @ self.Drr_2d)
        
        # 左端矩阵: A = I/dt - L
        A = eye(N_total) / self.dt - L
        
        # 右端向量: b = V^{n+1}/dt + u(c)  (包含效用项！)
        b = V_next.reshape(N_total) / self.dt + utility_source
        
        return A, b
    
    def _update_controls(self, V, t_idx):
        """
        根据当前值函数更新最优控制变量
        """
        Nr, Nw = self.Nr, self.Nw
        t_val = self.t[t_idx]
        phi = self.compute_phi(t_val)
        
        # 计算导数
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
                
                # 最优消费
                c_new[i, j] = self.compute_optimal_consumption(V_w_ij, phi)
                
                # 最优投资组合
                pi_new[i, j] = self.compute_optimal_portfolio(
                    V_w_ij, V_ww[i, j], V_wr[i, j], w_val, r_val, phi
                )
        
        return c_new, pi_new
    
    def _build_simplified_jacobian(self, V_current, t_idx):
        """
        构建简化的 Jacobian 矩阵
        忽略控制变量对 V 的依赖，只保留线性部分
        """
        Nr, Nw = self.Nr, self.Nw
        N_total = Nr * Nw
        
        # 基础矩阵: I + dt * (线性算子)
        # 线性部分包括: 扩散项和利率漂移项
        
        I = eye(N_total)
        
        # 利率扩散项: 0.5 * sigma_r^2 * V_rr
        L_r = 0.5 * self.sigma_r**2 * self.Drr_2d
        
        # 利率漂移项: kappa * (r_bar - r) * V_r (近似为线性)
        # 使用对角近似
        drift_r_diag = np.zeros(N_total)
        for i in range(Nr):
            for j in range(Nw):
                idx = i * Nw + j
                drift_r_diag[idx] = self.kappa * (self.r_bar - self.r[i])
        L_r_drift = diags(drift_r_diag) @ self.Dr_2d
        
        # 财富扩散项 (近似): 使用当前 pi 的估计
        # 这里使用一个平均的扩散系数
        sigma_w_eff = 0.5 * self.sigma**2 + 0.5 * self.w_max**2 + self.w_max**2
        L_w = sigma_w_eff * self.Dww_2d
        
        # 组合 Jacobian
        J = I + self.dt * (L_r + L_r_drift + L_w)
        
        return J
    
    def solve_backward(self):
        """
        反向时间步进求解 (隐式格式)
        """
        print("=" * 60)
        print("开始隐式FDM求解HJB方程 (Policy Iteration)")
        print("=" * 60)
        
        # 步骤1: 设置终端条件
        for i in range(self.Nr):
            for j in range(self.Nw):
                self.V[-1, i, j] = self.compute_terminal_value(
                    self.r[i], self.w[j]
                )
        
        print(f"终端条件已设置，V(T)范围: [{self.V[-1].min():.4f}, {self.V[-1].max():.4f}]")
        
        # 步骤2: 反向时间步进
        for t_idx in range(self.Nt - 2, -1, -1):
            t_val = self.t[t_idx]
            V_next = self.V[t_idx + 1]
            
            print(f"\n时间步 {t_idx}/{self.Nt}, t={t_val:.3f}")
            
            # 使用策略迭代求解当前时间步
            V_current = self.solve_time_step_policy_iteration(V_next, t_idx)
            self.V[t_idx] = V_current
            
            if t_idx % 5 == 0:
                print(f"  V范围: [{V_current.min():.4f}, {V_current.max():.4f}]")
                print(f"  c范围: [{self.c_opt[t_idx].min():.4f}, {self.c_opt[t_idx].max():.4f}]")
                print(f"  π范围: [{self.pi_opt[t_idx].min():.4f}, {self.pi_opt[t_idx].max():.4f}]")
        
        print("\n" + "=" * 60)
        print("求解完成!")
        print("=" * 60)
        
        return self.V, self.c_opt, self.pi_opt
    
    def plot_results(self, save_path='hjb_solution_implicit.png'):
        """可视化结果"""
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
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"结果已保存至: {save_path}")


def main():
    """主函数示例"""
    params = {
        't_min': 0.0,
        't_max': 1.0,
        'r_min': 0.01,
        'r_max': 0.10,
        'w_min': 0.5,
        'w_max': 2.0,
        'Nt': 30,  # 减少时间步数以加速
        'Nr': 20,
        'Nw': 20,
        'theta': 0.7,
        'rho': 0.5,
        'mu': 0.08,
        'sigma': 0.2,
        'sigma_r': 0.05,
        'kappa': 0.1,
        'r_bar': 0.05,
        'max_iter_nr': 30,
        'tol_nr': 1e-6,
        'damping': 0.7
    }
    
    solver = HJBSolverImplicitFDM(params)
    
    V, c_opt, pi_opt = solver.solve_backward()
    
    solver.plot_results()
    
    print("\n最终结果:")
    print(f"值函数范围: [{V.min():.4f}, {V.max():.4f}]")
    print(f"最优消费范围: [{c_opt.min():.4f}, {c_opt.max():.4f}]")
    print(f"最优投资组合范围: [{pi_opt.min():.4f}, {pi_opt.max():.4f}]")


if __name__ == "__main__":
    main()
