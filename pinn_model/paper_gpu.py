import numpy as np
import math
import torch
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_V_and_g(t, r, w, x):
    """
    计算值函数 V 和 g。
    g(t, r) = x[0]*t^2 + x[1]*t + x[2]*r + x[3]*r^2 + x[4]*r*t
    g_formula = exp(g)  即公式中的 g(t,r)
    V = (1/theta) * phi^(1-theta) * exp(g)^(1-theta) * w^theta
    """
    theta = 0.8
    phi = 0.5
    g = x[0] * t * t + x[1] * t + x[2] * r + x[3] * r * r + x[4] * r * t
    V = theta ** (-1) * phi ** (1 - theta) * torch.exp(g) ** (1 - theta) * w ** theta
    return V, g


def fun_loss(t, r, w, pi, pop):
    """
    最大化解析最优 Hamiltonian：

    H* = V * { theta * [ r + 0.5*A^2*(theta-1) + 0.5*(x3*r+x4)
                          - 1/(2*(theta-1)) * K^T @ Sigma^{-1} @ K ]
               + (1-theta) * g^{1/(theta-1)} }

    其中：
      Sigma = sigma @ sigma.T + A^2 * pi @ pi.T        shape: (N, n, n)
      K     = 0.5*pi*(x3*r+x4) - (mu - r*pi) + A^2*pi*(theta-1)  shape: (N, n)
      g_formula = exp(g_code)  即公式中的 g(t,r)

    WOA 最大化 H* 均值，返回均值（越大越好）。
    """
    df = pd.read_csv("./data/raw/stock_ga_parameters_seed72.csv")
    mu_np    = df["mu"].values.astype(np.float32)     # shape: (n,)
    sigma_np = df["sigma"].values.astype(np.float32)  # shape: (n,)
    n = len(mu_np)
    N = t.shape[0]

    mu_t    = torch.tensor(mu_np,    device=device)  # (n,)
    sigma_v = torch.tensor(sigma_np, device=device)  # (n,)  各资产标准差

    theta = 0.8
    Aa    = 1.0  # 无风险资产波动率系数 A

    # 参数对应：g = x[0]*t^2 + x[1]*t + x[2]*r + x[3]*r^2 + x[4]*r*t
    # 公式里 x3=pop[2]（r^2项系数），x4=pop[1]（t项系数）
    x3 = pop[2]  # r^2 项系数
    x4 = pop[3]  # r 项系数

    # 计算 V 和 g_code
    V, g_code = compute_V_and_g(t, r, w, pop)  # (N,)
    # g_formula^{1/(theta-1)} = exp(g_code / (theta-1))
    # theta-1 = -0.2，指数放大 5 倍，需要 clip 防溢出
    g_exp_arg = torch.clamp(g_code / (theta - 1), min=-50.0, max=50.0)
    g_pow = torch.exp(g_exp_arg)    # (N,)

    # V 中 exp(g)^(1-theta) = exp(g*(1-theta))，同样需要防溢出
    V = torch.clamp(V, min=-1e6, max=1e6)

    # ---- 构造 Sigma: (N, n, n) ----
    # sigma_v: (n,)  ->  对角矩阵 diag(sigma_v^2)
    # pi: (N, n)
    sigma_diag = torch.diag(sigma_v ** 2)                    # (n, n)
    Sigma = sigma_diag.unsqueeze(0) \
            + Aa ** 2 * torch.bmm(pi.unsqueeze(2), pi.unsqueeze(1))  # (N, n, n)

    # ---- 构造 K: (N, n) ----
    # K = 0.5 * pi * (x3*r + x4) - (mu - r*pi) + A^2 * pi * (theta-1)
    x3r_x4 = x3 * r + x4  # (N,)
    K = (0.5 * pi * x3r_x4.unsqueeze(1)
         - (mu_t.unsqueeze(0) - r.unsqueeze(1) * pi)
         + Aa ** 2 * pi * (theta - 1))  # (N, n)

    # ---- K^T Sigma^{-1} K: (N,) ----
    # 用 torch.linalg.solve 避免显式求逆
    Sigma_inv_K = torch.linalg.solve(Sigma, K.unsqueeze(2)).squeeze(2)  # (N, n)
    KSK = (K * Sigma_inv_K).sum(dim=1)  # (N,)
    KSK = torch.clamp(KSK, min=-1e6, max=1e6)  # 防溢出

    # ---- H* 各项 ----
    # 注意: 1/(2*(theta-1)) = 1/(2*(-0.2)) = -2.5
    inner = (r
             + 0.5 * Aa ** 2 * (theta - 1)
             + 0.5 * x3r_x4
             - 1.0 / (2.0 * (theta - 1)) * KSK)  # (N,)

    H_star = V * (theta * inner + (1 - theta) * g_pow)  # (N,)

    # 过滤 nan/inf，只对有限值取均值
    H_np = H_star.detach().cpu().numpy()
    valid = np.isfinite(H_np)
    if valid.sum() == 0:
        return -1e9  # 全部溢出时返回极小值
    H_valid = H_np[valid]
    # 取对数尺度：sign(H)*log(1+|H|)，保留符号且压缩量级
    H_scaled = np.sign(H_valid) * np.log1p(np.abs(H_valid))
    average_value = float(H_scaled.mean())
    return average_value


def generate_grid(nt=10, nr=10, nw=10, n_assets=40):
    """
    自动生成 (t, r, w) 三维网格点及均匀分配的投资策略 pi。
    t ∈ [0.1, 1.0]，r ∈ [0.01, 0.15]，w ∈ [0.1, 10.0]
    pi: 每资产均匀分配，pi_i = 0.5 / n_assets，合计持有风险资产比例为 0.5，
        无风险资产比例 pi_comp = 0.5。
    返回: t, r, w (各 shape (N,) 的 requires_grad tensor)，pi (shape (N, n_assets) tensor)
    """
    t_vals = np.linspace(0.1, 1.0, nt)
    r_vals = np.linspace(1, 3, nr)
    w_vals = np.linspace(0.1, 4.0, nw)

    # 生成三维网格，展平为 (N, 3)
    grid = np.array(np.meshgrid(t_vals, r_vals, w_vals, indexing='ij')).reshape(3, -1).T  # (N, 3)
    N = grid.shape[0]

    coords = torch.tensor(grid, dtype=torch.float32, device=device)
    t = coords[:, 0]  # (N,)
    r = coords[:, 1]  # (N,)
    w = coords[:, 2]  # (N,)

    # 均匀分配：每资产占比 0.5/n_assets，总风险资产占 0.5
    pi_val = 0.9 / n_assets
    pi = torch.full((N, n_assets), pi_val, dtype=torch.float32, device=device)

    return t, r, w, pi


def fitness(pop):
    """
    WOA 适应度函数：自动生成三维网格 (t, r, w) 和均匀 pi，
    最大化 H*，WOA 最小化取负的 H*。
    """
    n_assets = 40
    t, r, w, pi = generate_grid(nt=10, nr=10, nw=10, n_assets=n_assets)

    fit = -fun_loss(t, r, w, pi, pop)  # WOA 最小化，故取负
    return fit


def boundary(pop, lb, ub):
    """将粒子位置限制在搜索边界内。"""
    for j in range(len(lb)):
        if pop[j] > ub[j] or pop[j] < lb[j]:
            pop[j] = (ub[j] - lb[j]) * np.random.rand() + lb[j]
    return pop


def WOA(max_iterations=100, noposs=50):
    """
    鲸鱼优化算法（WOA）：寻优 g(t,r) = x[0]*t*t + x[1]*t + x[2]*r + x[3]*r*r + x[4]*r*t 的系数。
    搜索维度 5，对应 x[0]~x[4]，x[5] 为范围扩大的常数项（备用）。
    """
    lb = [-100, -100, -100, -100, -100]
    ub = [ 100,  100,  100,  100,  100]
    noclus = len(lb)

    poss_sols = np.zeros((noposs, noclus))
    gbest = np.zeros((noclus,))
    b = 2.0

    # 种群初始化
    for i in range(noposs):
        for j in range(noclus):
            poss_sols[i][j] = (ub[j] - lb[j]) * np.random.rand() + lb[j]

    # 初始全局最优
    global_fitness = np.inf
    for i in range(noposs):
        cur_fitness = fitness(poss_sols[i, :])
        if cur_fitness < global_fitness:
            global_fitness = cur_fitness
            gbest = poss_sols[i].copy()

    trace = []
    trace_pop = []

    for it in range(max_iterations):
        for i in range(noposs):
            a = 2.0 - (2.0 * it) / (1.0 * max_iterations)
            r_rand = np.random.random_sample()
            A = 2.0 * a * r_rand - a
            C = 2.0 * r_rand
            l = 2.0 * np.random.random_sample() - 1.0
            p = np.random.random_sample()

            for j in range(noclus):
                x = poss_sols[i][j]
                if p < 0.5:
                    if abs(A) < 1:
                        _x = gbest[j].copy()
                    else:
                        rand = np.random.randint(noposs)
                        _x = poss_sols[rand][j]
                    D = abs(C * _x - x)
                    updatedx = _x - A * D
                else:
                    _x = gbest[j].copy()
                    D = abs(_x - x)
                    updatedx = D * math.exp(b * l) * math.cos(2.0 * math.pi * l) + _x

                poss_sols[i][j] = updatedx

            poss_sols[i, :] = boundary(poss_sols[i, :], lb, ub)
            fitnessi = fitness(poss_sols[i])
            if fitnessi < global_fitness:
                global_fitness = fitnessi
                gbest = poss_sols[i].copy()

        trace.append(global_fitness)
        print(f"Iteration {it + 1}: global_fitness = {global_fitness:.6f}, "
              f"gbest = {[round(gbest[i], 4) for i in range(noclus)]}")
        trace_pop.append(gbest.copy())

    return gbest, trace, trace_pop


if __name__ == '__main__':
    gbest, trace, trace_pop = WOA(max_iterations=100, noposs=50)
    print(f"\n优化完成！最优参数: {[round(v, 4) for v in gbest]}")
    print(f"最优适应度: {trace[-1]:.6f}")
