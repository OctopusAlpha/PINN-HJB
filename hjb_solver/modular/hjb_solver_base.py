"""
HJB方程求解器 - 通用基础框架
支持自定义效用函数、生产函数、约束条件等
"""

import numpy as np
from scipy.sparse import csr_matrix, eye, kron, diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class UtilityFunction:
    """效用函数基类 - 支持自定义效用"""
    
    @staticmethod
    def crra(c: np.ndarray, theta: float) -> np.ndarray:
        """CRRA效用: u(c) = c^θ / θ"""
        c_safe = np.maximum(c, 1e-10)
        return (c_safe ** theta) / theta
    
    @staticmethod
    def crra_marginal(c: np.ndarray, theta: float) -> np.ndarray:
        """CRRA边际效用: u'(c) = c^(θ-1)"""
        c_safe = np.maximum(c, 1e-10)
        return c_safe ** (theta - 1)
    
    @staticmethod
    def crra_inverse_marginal(u_prime: np.ndarray, theta: float) -> np.ndarray:
        """CRRA逆边际效用: (u')^(-1) = u'^(1/(θ-1))"""
        u_prime_safe = np.maximum(u_prime, 1e-10)
        return u_prime_safe ** (1.0 / (theta - 1))
    
    @staticmethod
    def log(c: np.ndarray) -> np.ndarray:
        """对数效用: u(c) = log(c)"""
        return np.log(np.maximum(c, 1e-10))
    
    @staticmethod
    def log_marginal(c: np.ndarray) -> np.ndarray:
        """对数边际效用: u'(c) = 1/c"""
        return 1.0 / np.maximum(c, 1e-10)


class ProductionFunction:
    """生产/收益函数 - 定义资产收益结构"""
    
    @staticmethod
    def risky_asset_return(pi: float, mu: float, r: float) -> float:
        """风险资产组合收益: π*μ + (1-π)*r"""
        return pi * mu + (1 - pi) * r
    
    @staticmethod
    def wealth_drift(w: float, pi: float, mu: float, r: float, c: float) -> float:
        """财富漂移: w * (组合收益 - 消费率)"""
        portfolio_return = ProductionFunction.risky_asset_return(pi, mu, r)
        return w * (portfolio_return - c)
    
    @staticmethod
    def wealth_diffusion_squared(w: float, pi: float, sigma: float) -> float:
        """财富扩散项平方: (w*π*σ)^2"""
        return (w * pi * sigma) ** 2


class HJBSolverBase(ABC):
    """
    HJB方程求解器基类
    
    提供通用框架，支持:
    - 自定义效用函数
    - 自定义状态变量维度
    - 自定义控制变量
    - 自定义约束条件
    """
    
    def __init__(self, params: Dict):
        """
        初始化求解器
        
        Parameters:
        -----------
        params : dict
            包含所有参数的字典
        """
        self.params = params
        
        # 网格参数
        self._setup_grids()
        
        # 经济参数
        self._setup_economic_params()
        
        # 数值参数
        self._setup_numerical_params()
        
        # 初始化存储
        self._initialize_storage()
        
        # 构建微分算子
        self._build_differential_operators()
    
    def _setup_grids(self):
        """设置计算网格 - 可重写以支持更多状态变量"""
        # 时间网格
        self.t_min = self.params.get('t_min', 0.0)
        self.t_max = self.params.get('t_max', 1.0)
        self.Nt = self.params.get('Nt', 50)
        self.t = np.linspace(self.t_min, self.t_max, self.Nt)
        self.dt = self.t[1] - self.t[0]
        
        # 利率网格 (状态变量1)
        self.r_min = self.params.get('r_min', 0.01)
        self.r_max = self.params.get('r_max', 0.10)
        self.Nr = self.params.get('Nr', 30)
        self.r = np.linspace(self.r_min, self.r_max, self.Nr)
        self.dr = self.r[1] - self.r[0]
        
        # 财富网格 (状态变量2)
        self.w_min = self.params.get('w_min', 0.1)
        self.w_max = self.params.get('w_max', 10.0)
        self.Nw = self.params.get('Nw', 30)
        self.w = np.linspace(self.w_min, self.w_max, self.Nw)
        self.dw = self.w[1] - self.w[0]
    
    def _setup_economic_params(self):
        """设置经济参数"""
        self.theta = self.params.get('theta', 0.7)  # 风险厌恶系数
        self.rho = self.params.get('rho', 0.5)      # 相关系数
        self.mu = self.params.get('mu', 0.08)       # 风险资产收益率
        self.sigma = self.params.get('sigma', 0.2)  # 风险资产波动率
        self.sigma_r = self.params.get('sigma_r', 0.05)  # 利率波动率
        self.kappa = self.params.get('kappa', 0.1)  # 利率均值回归速度
        self.r_bar = self.params.get('r_bar', 0.05)  # 利率长期均值
    
    def _setup_numerical_params(self):
        """设置数值计算参数"""
        self.max_iter = self.params.get('max_iter', 50)
        self.tol = self.params.get('tol', 1e-6)
        self.damping = self.params.get('damping', 0.8)
    
    def _initialize_storage(self):
        """初始化存储数组"""
        self.V = np.zeros((self.Nt, self.Nr, self.Nw))
        self.c_opt = np.zeros((self.Nt, self.Nr, self.Nw))
        self.pi_opt = np.zeros((self.Nt, self.Nr, self.Nw))
    
    def _build_differential_operators(self):
        """构建微分算子矩阵"""
        self.Dr, self.Drr = self._build_1d_operators(self.Nr, self.dr)
        self.Dw, self.Dww = self._build_1d_operators(self.Nw, self.dw)
        
        # 转换为稀疏矩阵
        self.Dr = csr_matrix(self.Dr)
        self.Drr = csr_matrix(self.Drr)
        self.Dw = csr_matrix(self.Dw)
        self.Dww = csr_matrix(self.Dww)
        
        # 2D算子
        I_r = eye(self.Nr)
        I_w = eye(self.Nw)
        
        self.Dr_2d = kron(self.Dr, I_w)
        self.Dw_2d = kron(I_r, self.Dw)
        self.Drr_2d = kron(self.Drr, I_w)
        self.Dww_2d = kron(I_r, self.Dww)
        
        # 交叉导数算子 Dwr = D_r ⊗ D_w (先对w求导，再对r求导)
        self.Dwr_2d = kron(self.Dr, self.Dw)
    
    def _build_1d_operators(self, N: int, dh: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建一维微分算子
        
        Parameters:
        -----------
        N : int
            网格点数
        dh : float
            网格步长
            
        Returns:
        --------
        D, D2 : 一阶和二阶微分算子矩阵
        """
        D = np.zeros((N, N))
        D2 = np.zeros((N, N))
        
        for i in range(N):
            if i == 0:  # 左边界 - 前向差分
                D[i, i] = -1.0 / dh
                D[i, i+1] = 1.0 / dh
                D2[i, i] = 2.0 / (dh**2)
                D2[i, i+1] = -5.0 / (dh**2)
                D2[i, i+2] = 4.0 / (dh**2)
                D2[i, i+3] = -1.0 / (dh**2)
            elif i == N - 1:  # 右边界 - 后向差分
                D[i, i-1] = -1.0 / dh
                D[i, i] = 1.0 / dh
                D2[i, i] = 2.0 / (dh**2)
                D2[i, i-1] = -5.0 / (dh**2)
                D2[i, i-2] = 4.0 / (dh**2)
                D2[i, i-3] = -1.0 / (dh**2)
            else:  # 内部 - 中心差分
                D[i, i-1] = -0.5 / dh
                D[i, i+1] = 0.5 / dh
                D2[i, i-1] = 1.0 / (dh**2)
                D2[i, i] = -2.0 / (dh**2)
                D2[i, i+1] = 1.0 / (dh**2)
        
        return D, D2
    
    @abstractmethod
    def compute_terminal_value(self, r: float, w: float) -> float:
        """
        计算终端时刻的边界值
        
        Parameters:
        -----------
        r : float
            利率
        w : float
            财富
            
        Returns:
        --------
        float
            终端值函数 V(T, r, w)
        """
        pass
    
    @abstractmethod
    def compute_optimal_consumption(self, V_w: float, **kwargs) -> float:
        """
        计算最优消费
        
        Parameters:
        -----------
        V_w : float
            值函数对财富的导数
        **kwargs : 其他可能需要的参数
            
        Returns:
        --------
        float
            最优消费率 c
        """
        pass
    
    @abstractmethod
    def compute_optimal_portfolio(self, **kwargs) -> float:
        """
        计算最优投资组合
        
        Parameters:
        -----------
        **kwargs : 需要的参数（如V_w, V_ww, w, r等）
            
        Returns:
        --------
        float
            最优投资组合比例 π
        """
        pass
    
    @abstractmethod
    def solve_time_step(self, V_next: np.ndarray, t_idx: int) -> np.ndarray:
        """
        求解单个时间步
        
        Parameters:
        -----------
        V_next : np.ndarray
            下一时间步的值函数 (Nr, Nw)
        t_idx : int
            当前时间步索引
            
        Returns:
        --------
        np.ndarray
            当前时间步的值函数 (Nr, Nw)
        """
        pass
    
    def solve_backward(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        反向时间步进求解
        
        Returns:
        --------
        V, c_opt, pi_opt : 值函数和最优控制
        """
        print("=" * 60)
        print("开始求解HJB方程")
        print("=" * 60)
        
        # 设置终端条件
        self._set_terminal_condition()
        
        print(f"终端条件已设置，V(T)范围: [{self.V[-1].min():.4f}, {self.V[-1].max():.4f}]")
        
        # 反向时间步进
        for t_idx in range(self.Nt - 2, -1, -1):
            t_val = self.t[t_idx]
            V_next = self.V[t_idx + 1]
            
            print(f"\n时间步 {t_idx}/{self.Nt}, t={t_val:.3f}")
            
            # 求解当前时间步
            V_current = self.solve_time_step(V_next, t_idx)
            self.V[t_idx] = V_current
            
            if t_idx % 5 == 0:
                print(f"  V范围: [{V_current.min():.4f}, {V_current.max():.4f}]")
                print(f"  c范围: [{self.c_opt[t_idx].min():.4f}, {self.c_opt[t_idx].max():.4f}]")
                print(f"  π范围: [{self.pi_opt[t_idx].min():.4f}, {self.pi_opt[t_idx].max():.4f}]")
        
        print("\n" + "=" * 60)
        print("求解完成!")
        print("=" * 60)
        
        return self.V, self.c_opt, self.pi_opt
    
    def _set_terminal_condition(self):
        """设置终端条件"""
        for i in range(self.Nr):
            for j in range(self.Nw):
                self.V[-1, i, j] = self.compute_terminal_value(
                    self.r[i], self.w[j]
                )
    
    def plot_results(self, save_path: str = 'hjb_solution.png'):
        """可视化结果"""
        time_indices = [0, self.Nt//4, self.Nt//2, 3*self.Nt//4, self.Nt-1]
        
        fig = plt.figure(figsize=(18, 12))
        
        for idx, it in enumerate(time_indices):
            t_val = self.t[it]
            R, W = np.meshgrid(self.r, self.w, indexing='ij')
            
            # 值函数
            ax1 = fig.add_subplot(3, 5, idx+1, projection='3d')
            ax1.plot_surface(R, W, self.V[it], cmap='viridis')
            ax1.set_xlabel('r')
            ax1.set_ylabel('w')
            ax1.set_zlabel('V')
            ax1.set_title(f'V at t={t_val:.2f}')
            
            # 最优消费
            ax2 = fig.add_subplot(3, 5, idx+6, projection='3d')
            ax2.plot_surface(R, W, self.c_opt[it], cmap='plasma')
            ax2.set_xlabel('r')
            ax2.set_ylabel('w')
            ax2.set_zlabel('c')
            ax2.set_title(f'c at t={t_val:.2f}')
            
            # 最优投资组合
            ax3 = fig.add_subplot(3, 5, idx+11, projection='3d')
            ax3.plot_surface(R, W, self.pi_opt[it], cmap='coolwarm')
            ax3.set_xlabel('r')
            ax3.set_ylabel('w')
            ax3.set_zlabel('π')
            ax3.set_title(f'π at t={t_val:.2f}')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"结果已保存至: {save_path}")
