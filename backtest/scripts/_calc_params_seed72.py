"""
对 seed=72 随机选出的 40 只股票，用 GBM 极大似然估计 (mle_gbm) 计算 mu/sigma，
保存为与 stock_ga_parameters_new.csv 相同格式的新参数文件。
"""
import os, sys, random
import numpy as np
import pandas as pd

# ===== 路径 =====
STOCK_DIR  = r'c:/Users/zzy/Desktop/毕业论文2.23/code/data/A股5000+股票历史日数据2026227'
SAVE_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pinn_model', 'data', 'raw', 'stock_ga_parameters_seed72.csv'))
N_ASSETS   = 40
RANDOM_SEED = 72

# ===== GBM 极大似然估计（与 data_loder.py 完全一致）=====
def mle_gbm(prices, dt=1/252):
    log_returns = np.diff(np.log(prices))
    n = len(log_returns)
    sigma_hat = np.sqrt(np.sum((log_returns - np.mean(log_returns))**2) / (n * dt))
    mu_hat = (np.mean(log_returns) / dt) + 0.5 * sigma_hat**2
    return mu_hat, sigma_hat

# ===== 按 seed=72 顺序选出 40 只股票（与 wealth_backtest copy.py 逻辑完全一致）=====
all_files = sorted([f for f in os.listdir(STOCK_DIR) if f.endswith('.csv')])
random.seed(RANDOM_SEED)
random.shuffle(all_files)

stocks = []
for fname in all_files:
    if len(stocks) >= N_ASSETS:
        break
    fpath = os.path.join(STOCK_DIR, fname)
    try:
        df = pd.read_csv(fpath, encoding='utf-8')
        df.columns = ['日期', '证券代码', '开盘价', '最高价', '最低价', '收盘价',
                      '前收盘价', '成交量', '成交额', '复权状态', '换手率',
                      '交易状态', '涨跌幅', '滚动市盈率', '市净率', '滚动市销率',
                      '滚动市现率', '是否ST']
        df['日期']  = pd.to_datetime(df['日期'], errors='coerce')
        df['收盘价'] = pd.to_numeric(df['收盘价'], errors='coerce')
        df = df.dropna(subset=['日期', '收盘价']).sort_values('日期').reset_index(drop=True)
        if len(df) == 0:
            continue
        stocks.append((fname.replace('.csv', ''), df))
    except:
        pass

print(f"共选出 {len(stocks)} 只股票，开始计算 GBM 参数...\n")

# ===== 计算每只股票的 mu/sigma =====
results = []
for code, df in stocks:
    prices = df['收盘价'].values
    # 用最近 252 个交易日（与 data_loder.py 一致：group[-252:]）
    prices_used = prices[-252:] if len(prices) >= 252 else prices
    if len(prices_used) < 2:
        print(f"  跳过 {code}（数据不足）")
        continue
    mu, sigma = mle_gbm(prices_used)
    results.append({'stock_id': code, 'mu': mu, 'sigma': sigma})
    print(f"  {code:<20}  mu={mu:>10.6f}  sigma={sigma:>10.6f}  (用{len(prices_used)}条)")

# ===== 保存 =====
results_df = pd.DataFrame(results)
results_df.to_csv(SAVE_PATH, index=False, encoding='utf-8')
print(f"\n参数已保存至: {SAVE_PATH}")
print(results_df.to_string(index=False))
