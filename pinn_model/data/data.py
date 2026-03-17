"""
data.py —— 市场参数估计模块
============================
提供两类参数估计功能：

1. 股票参数（几何布朗运动 GBM）
   dS = mu * S * dt + sigma * S * dW
   使用最大似然估计（MLE）从日频收盘价序列估计 mu（漂移率）和 sigma（波动率）。
   方法与 data_loder.py 中 mle_gbm 保持一致。

2. 利率参数（OU / Vasicek 过程）
   dr = kappa * (mu_r - r) * dt + sigma_r * dW
   使用最大似然估计（MLE）从 SHIBOR 月度/日频利率序列估计：
     kappa  —— 均值回归速率（年化）
     mu_r   —— 长期利率均值（小数，如 0.03）
     sigma_r —— 利率波动率（年化）

对外接口：
    estimate_stock_params(stocks)             -> DataFrame[stock_id, mu, sigma]
    estimate_rate_params(shibor_df)           -> dict(kappa, mu_r, sigma_r)
    load_or_estimate_stock_params(param_path, stocks, recalculate) -> DataFrame
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


# ===========================================================
# 1. GBM 参数估计（仿照 data_loder.py）
# ===========================================================

def mle_gbm(prices, dt=1 / 252):
    """
    几何布朗运动（GBM）最大似然估计
    与 data_loder.py 中 mle_gbm 逻辑完全一致。

    Args:
        prices: 价格序列（array-like），长度至少为 2，且均为正数
        dt    : 时间步长，日频默认 1/252

    Returns:
        mu_hat   : 漂移率（年化连续复利）
        sigma_hat: 波动率（年化）
    """
    prices = np.asarray(prices, dtype=float)
    prices = prices[prices > 0]          # 过滤无效价格
    if len(prices) < 2:
        return 0.0, 0.3                  # 数据不足时返回保守默认值

    log_returns = np.diff(np.log(prices))
    n = len(log_returns)

    # 波动率 MLE
    sigma_hat = np.sqrt(np.sum((log_returns - np.mean(log_returns)) ** 2) / (n * dt))

    # 漂移率 MLE（含 Itô 修正项）
    mu_hat = (np.mean(log_returns) / dt) + 0.5 * sigma_hat ** 2

    return float(mu_hat), float(sigma_hat)


def estimate_stock_params(stocks, lookback_days=252):
    """
    对多只股票批量估计 GBM 参数（mu, sigma）。

    Args:
        stocks      : list of (stock_id, df)，df 含列 ['日期', 'close']
        lookback_days: 用于估计的最近交易日数，默认 252（约 1 年）

    Returns:
        DataFrame[stock_id, mu, sigma]
    """
    results = []
    for stock_id, df in tqdm(stocks, desc="估计股票GBM参数"):
        prices = df['close'].dropna().values
        if lookback_days and len(prices) > lookback_days:
            prices = prices[-lookback_days:]
        mu, sigma = mle_gbm(prices, dt=1 / 252)
        results.append({'stock_id': stock_id, 'mu': mu, 'sigma': sigma})
        print(f"  {stock_id}: mu={mu:.4f}, sigma={sigma:.4f}")
    return pd.DataFrame(results)


def load_or_estimate_stock_params(param_path, stocks=None, recalculate=False,
                                  lookback_days=252):
    """
    加载或重新估计股票 GBM 参数。

    Args:
        param_path   : CSV 文件路径（含列 stock_id, mu, sigma）
        stocks       : list of (stock_id, df)，recalculate=True 时必须提供
        recalculate  : 是否重新从数据估计（True）或直接读 CSV（False）
        lookback_days: GBM 估计使用的最近 N 个交易日

    Returns:
        DataFrame[stock_id, mu, sigma]
    """
    if not recalculate:
        print(f"直接加载参数文件: {param_path}")
        if not os.path.exists(param_path):
            raise FileNotFoundError(f"参数文件不存在: {param_path}")
        return pd.read_csv(param_path, encoding='utf-8')

    print("从真实日频收盘价重新估计 GBM 参数...")
    if stocks is None:
        raise ValueError("recalculate=True 时必须传入 stocks 列表")
    params_df = estimate_stock_params(stocks, lookback_days=lookback_days)
    if param_path:
        os.makedirs(os.path.dirname(param_path), exist_ok=True)
        params_df.to_csv(param_path, index=False, encoding='utf-8')
        print(f"参数已保存至: {param_path}")
    return params_df


# ===========================================================
# 2. OU 过程参数估计（Vasicek 利率模型）
# ===========================================================

def mle_ou(rates, dt=1 / 12):
    """
    Ornstein-Uhlenbeck（OU / Vasicek）过程最大似然估计
    模型: dr = kappa*(mu_r - r)*dt + sigma_r*dW

    使用离散时间精确 MLE（条件正态似然）：
        r(t+dt) | r(t) ~ N(r(t)*exp(-kappa*dt) + mu_r*(1-exp(-kappa*dt)),
                            sigma_r^2/(2*kappa)*(1-exp(-2*kappa*dt)))

    通过 OLS 回归 r(t+1) ~ a + b*r(t) 快速估计，再反解 kappa、mu_r、sigma_r。

    Args:
        rates: 利率序列（小数形式，如 0.03），array-like
        dt   : 时间步长，月度为 1/12，日频为 1/252

    Returns:
        kappa  : 均值回归速率（年化）
        mu_r   : 长期利率均值（小数）
        sigma_r: 利率波动率（年化）
    """
    rates = np.asarray(rates, dtype=float)
    rates = rates[~np.isnan(rates)]
    if len(rates) < 3:
        return 1.0, 0.03, 0.01   # 数据不足返回默认值

    r_t   = rates[:-1]   # r(t)
    r_tp1 = rates[1:]    # r(t + dt)
    n     = len(r_t)

    # OLS 回归: r(t+dt) = a + b * r(t) + eps
    # 最小二乘解
    X = np.column_stack([np.ones(n), r_t])
    beta, _, _, _ = np.linalg.lstsq(X, r_tp1, rcond=None)
    a_hat, b_hat = beta[0], beta[1]

    # 从 OLS 系数反解 OU 参数
    # b = exp(-kappa * dt)  =>  kappa = -ln(b) / dt
    b_hat = np.clip(b_hat, 1e-8, 1 - 1e-8)   # 防止数值问题
    kappa = -np.log(b_hat) / dt               # 年化均值回归速率

    # a = mu_r * (1 - exp(-kappa*dt))  =>  mu_r = a / (1 - b)
    mu_r = a_hat / (1 - b_hat)

    # 残差标准差 => sigma_r
    residuals = r_tp1 - (a_hat + b_hat * r_t)
    var_eps   = np.var(residuals, ddof=0)
    # var(eps) = sigma_r^2 / (2*kappa) * (1 - exp(-2*kappa*dt))
    denom = (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
    sigma_r = np.sqrt(var_eps / denom) if denom > 1e-10 else np.std(residuals) * np.sqrt(1 / dt)

    return float(kappa), float(mu_r), float(sigma_r)


def estimate_rate_params(shibor_df, rate_col='SHIBOR_7D', freq='monthly'):
    """
    从 SHIBOR 数据估计 OU 过程参数。

    Args:
        shibor_df: DataFrame，含 '日期' 列和利率列
        rate_col : 利率列名，默认 'SHIBOR_7D'（单位：%）
        freq     : 数据频率，'monthly'（dt=1/12）或 'daily'（dt=1/252）

    Returns:
        dict: {
            'kappa'  : 均值回归速率（年化）,
            'mu_r'   : 长期利率均值（小数）,
            'sigma_r': 利率波动率（年化）,
            'r_mean' : 样本均值（小数）,
            'r_std'  : 样本标准差（小数）,
        }
    """
    df = shibor_df.copy()
    df = df.dropna(subset=[rate_col]).sort_values('日期').reset_index(drop=True)

    # 转换为小数（SHIBOR 原始单位为 %）
    rates = df[rate_col].values / 100.0

    dt = 1 / 12 if freq == 'monthly' else 1 / 252
    kappa, mu_r, sigma_r = mle_ou(rates, dt=dt)

    # 合理性约束（防止估计异常）
    kappa   = max(kappa,   0.01)
    mu_r    = np.clip(mu_r, 0.0, 0.5)
    sigma_r = max(sigma_r, 1e-4)

    params = {
        'kappa'  : kappa,
        'mu_r'   : mu_r,
        'sigma_r': sigma_r,
        'r_mean' : float(np.mean(rates)),
        'r_std'  : float(np.std(rates)),
    }
    return params


# ===========================================================
# 3. 便捷汇总函数
# ===========================================================

def print_rate_params(params):
    """打印利率 OU 参数摘要"""
    print("\n" + "=" * 50)
    print("利率 OU 过程参数估计结果（Vasicek 模型）")
    print("=" * 50)
    print(f"  均值回归速率 kappa  = {params['kappa']:.4f}  （年化）")
    print(f"  长期利率均值 mu_r   = {params['mu_r']:.4f}  （{params['mu_r']*100:.2f}%）")
    print(f"  利率波动率   sigma_r = {params['sigma_r']:.4f}  （年化）")
    print(f"  样本均值            = {params['r_mean']:.4f}  （{params['r_mean']*100:.2f}%）")
    print(f"  样本标准差          = {params['r_std']:.4f}")
    print("=" * 50)


def print_stock_params(params_df):
    """打印股票 GBM 参数摘要"""
    print("\n" + "=" * 50)
    print(f"股票 GBM 参数估计结果（共 {len(params_df)} 只）")
    print("=" * 50)
    print(f"  mu    —— 均值: {params_df['mu'].mean():.4f}, "
          f"范围: [{params_df['mu'].min():.4f}, {params_df['mu'].max():.4f}]")
    print(f"  sigma —— 均值: {params_df['sigma'].mean():.4f}, "
          f"范围: [{params_df['sigma'].min():.4f}, {params_df['sigma'].max():.4f}]")
    print("=" * 50)
