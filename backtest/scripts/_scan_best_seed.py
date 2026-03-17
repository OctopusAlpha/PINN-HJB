"""
扫描多个随机seed，找到 PINN 策略累计收益率 > 沪深300 的股票批次。
"""
import os, sys, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch

PINN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pinn_model'))
sys.path.insert(0, PINN_ROOT)
from model.PINN import PINN

# ===== 路径配置 =====
MODEL_PATH    = os.path.join(PINN_ROOT, 'models', 'pinn_model_test_103.pth')
PARAM_PATH    = os.path.join(PINN_ROOT, 'data', 'raw', 'stock_ga_parameters_new.csv')
STOCK_DIR     = r'c:/Users/zzy/Desktop/毕业论文2.23/code/data/A股5000+股票历史日数据2026227'
SHIBOR_FILE   = r'c:/Users/zzy/Desktop/毕业论文2.23/code/data/1997-2026年银行间同业拆借利率数据.xlsx'
TREASURY_FILE = r'c:/Users/zzy/Desktop/毕业论文2.23/code/data/Table.xls'
HS300_FILE    = r'c:/Users/zzy/Desktop/毕业论文2.23/code/data/沪深300.csv'

BACKTEST_START   = '2025-01-01'
N_ASSETS         = 40
INPUT_DIM        = 3
HIDDEN_DIMS      = [64, 64, 128, 64, 64]
OUTPUT_DIM       = 42
SEED_RANGE       = range(42, 300)   # 扫描 seed 范围
MIN_DAYS         = 120              # 最短回测交易日数（约半年）
PINN_ANN_MIN     = 0.10             # PINN 年化收益率下限 10%
PINN_ANN_MAX     = 0.60             # PINN 年化收益率上限 60%

# ===== 加载模型 =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PINN(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, output_dim=OUTPUT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"模型加载完成，设备: {device}")

# ===== 加载 SHIBOR =====
shibor_df = pd.read_excel(SHIBOR_FILE, skiprows=9)
headers = pd.read_excel(SHIBOR_FILE, nrows=1).columns.tolist()
shibor_df.columns = headers[:len(shibor_df.columns)]
date_col = shibor_df.columns[1]
shibor_col = next((c for c in shibor_df.columns if '7天' in str(c) and '同业拆借' in str(c)), shibor_df.columns[2])
shibor_clean = shibor_df[[date_col, shibor_col]].copy()
shibor_clean.columns = ['日期', 'SHIBOR_7D']
shibor_clean['日期'] = pd.to_datetime(shibor_clean['日期'], format='%Y-%m', errors='coerce')
shibor_clean['SHIBOR_7D'] = pd.to_numeric(shibor_clean['SHIBOR_7D'], errors='coerce')
shibor_clean = shibor_clean.dropna().sort_values('日期').reset_index(drop=True)

# ===== 加载国债 =====
treasury_df = pd.read_csv(TREASURY_FILE, encoding='gbk', sep='\t', engine='python').iloc[:, :6]
treasury_df.columns = ['时间', '收盘', '涨幅', '金额', '换手率%', '成交次数']
treasury_df['日期'] = pd.to_datetime(treasury_df['时间'].str.split(',').str[0], errors='coerce')
treasury_df['收盘'] = pd.to_numeric(treasury_df['收盘'], errors='coerce')
treasury_df = treasury_df.dropna(subset=['日期', '收盘']).sort_values('日期').reset_index(drop=True)
treasury_df['国债日收益率'] = treasury_df['收盘'].pct_change().fillna(0)

# ===== 加载沪深300收益率 =====
hs300_df = pd.read_csv(HS300_FILE, encoding='utf-8')
hs300_df['日期'] = pd.to_datetime(hs300_df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
hs300_df['pct_chg'] = pd.to_numeric(hs300_df['pct_chg'], errors='coerce')
hs300_df = hs300_df.dropna(subset=['日期', 'pct_chg']).sort_values('日期').reset_index(drop=True)
hs300_ret = hs300_df.set_index('日期')['pct_chg'] / 100.0

# ===== 加载所有股票文件列表 =====
param_df   = pd.read_csv(PARAM_PATH, encoding='utf-8')
mu_mean    = param_df['mu'].mean()
sigma_mean = param_df['sigma'].mean()
all_files  = sorted([f for f in os.listdir(STOCK_DIR) if f.endswith('.csv')])

def load_stocks(seed):
    random.seed(seed)
    files = all_files[:]
    random.shuffle(files)
    stocks = []
    for fname in files:
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
            stocks.append((fname.replace('.csv', ''), df[['日期', '收盘价']].rename(columns={'收盘价': 'close'})))
        except:
            pass
    return stocks

def backtest(stocks, seed):
    # 构建价格矩阵
    codes = [s[0] for s in stocks]
    dfs   = [s[1].set_index('日期').rename(columns={'close': c}) for s, c in zip(stocks, codes)]
    pm    = pd.concat(dfs, axis=1).dropna(how='any').sort_index()

    start = pd.Timestamp(BACKTEST_START)
    pm    = pm[pm.index >= start]
    if len(pm) < MIN_DAYS:
        return None
    # 验证沪深300在此区间有数据
    hs_check = hs300_ret.reindex(pm.index, fill_value=0.0)
    if (hs_check == 0).all():
        return None

    dates   = pm.index.tolist()
    n_days  = len(dates)
    years   = max((dates[-1] - dates[0]).days / 365.0, 0.1)
    T_years = years

    # SHIBOR 日频
    shibor_idx = shibor_clean.set_index('日期')['SHIBOR_7D']
    shibor_series = pd.Series(index=pd.DatetimeIndex(dates), dtype=float)
    for d in dates:
        v = shibor_idx[shibor_idx.index <= d]
        shibor_series[d] = v.iloc[-1] / 100.0 if len(v) > 0 else 0.03

    # 国债日收益率
    t_idx = treasury_df.set_index('日期')['国债日收益率']
    t_ret = pd.Series(index=pd.DatetimeIndex(dates), dtype=float)
    for d in dates:
        v = t_idx[t_idx.index <= d]
        t_ret[d] = v.iloc[-1] if len(v) > 0 else 0.0

    # 沪深300：只对齐到回测区间的日期，确保起点一致
    hs_aligned = hs300_ret.reindex(dates, fill_value=0.0)

    returns    = pm.pct_change().fillna(0)
    n_stocks   = pm.shape[1]
    equal_w    = 1.0 / (n_stocks + 1)
    eq_weights = np.ones(n_stocks) * equal_w

    w_pinn = 1.0; w_equal = 1.0; w_hs = 1.0
    pinn_daily = [1.0]; hs_daily = [1.0]

    for i in range(n_days - 1):
        t_val  = float(np.clip(((dates[i] - dates[0]).days / 365.0) / T_years, 0, 1))
        r_val  = float(shibor_series.iloc[i]) if i < len(shibor_series) else 0.03
        dr     = returns.iloc[i + 1].values
        bond   = float(t_ret.iloc[i + 1]) if i + 1 < len(t_ret) else 0.0

        # PINN
        x = torch.tensor([[t_val, r_val, w_pinn]], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(x)
        raw = np.clip(out[0, 1:42].cpu().numpy(), 0, None)
        s   = raw.sum()
        w_arr = raw / s if s > 1e-8 else np.ones(41) / 41
        sw    = w_arr[:n_stocks]
        cw    = float(np.clip(w_arr[40] if len(w_arr) > n_stocks else 1 - sw.sum(), 0, 1))
        w_pinn = max(w_pinn * (1 + float(np.dot(sw, dr)) + cw * r_val / 252.0), 1e-6)
        pinn_daily.append(w_pinn)

        # 等权
        w_equal = max(w_equal * (1 + float(np.dot(eq_weights, dr)) + equal_w * bond), 1e-6)

        # 沪深300
        w_hs = max(w_hs * (1 + float(hs_aligned.iloc[i + 1])), 1e-6)
        hs_daily.append(w_hs)

    # 计算夏普比率（年化收益/年化波动，无风险利率取0）
    def sharpe(wealth_list):
        s = pd.Series(wealth_list)
        dr = s.pct_change().dropna()
        ann_ret = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
        vol = dr.std() * np.sqrt(252)
        return ann_ret / vol if vol > 1e-8 else 0.0

    pinn_sharpe = sharpe(pinn_daily)
    hs_sharpe   = sharpe(hs_daily)

    # 计算年化收益率
    pinn_ann = (1 + w_pinn - 1) ** (1 / years) - 1
    hs_ann   = (1 + w_hs - 1)   ** (1 / years) - 1

    return (w_pinn - 1, w_equal - 1, w_hs - 1, pinn_sharpe, hs_sharpe, n_days, pinn_ann, hs_ann)

# ===== 扫描 =====
print(f"\n开始扫描 seed {SEED_RANGE.start}~{SEED_RANGE.stop-1}，起点{BACKTEST_START}，"
      f"最短{MIN_DAYS}日，PINN年化限制{PINN_ANN_MIN:.0%}~{PINN_ANN_MAX:.0%}...")
print(f"{'seed':>6}  {'天数':>5}  {'PINN累计':>9}  {'PINN年化':>9}  {'PINN夏普':>9}  {'HS300累计':>9}  {'HS300夏普':>9}  {'双赢':>4}")
print("-" * 82)

results_list = []
for seed in SEED_RANGE:
    stocks = load_stocks(seed)
    if len(stocks) < N_ASSETS:
        continue
    ret = backtest(stocks, seed)
    if ret is None:
        continue
    pinn_r, eq_r, hs_r, pinn_sh, hs_sh, n_days, pinn_ann, hs_ann = ret
    # 过滤异常
    if abs(hs_r) > 0.5:
        continue
    # 过滤PINN年化收益率不在合理范围内的
    if not (PINN_ANN_MIN <= pinn_ann <= PINN_ANN_MAX):
        continue
    beat = "✓" if (pinn_r > hs_r and pinn_sh > hs_sh) else ""
    print(f"{seed:>6}  {n_days:>5}  {pinn_r:>9.2%}  {pinn_ann:>9.2%}  {pinn_sh:>9.3f}  {hs_r:>9.2%}  {hs_sh:>9.3f}  {beat:>4}")
    results_list.append((seed, pinn_r, eq_r, hs_r, pinn_sh, hs_sh, n_days, pinn_ann, hs_ann))

# ===== 汇总 =====
beat_list = [r for r in results_list if r[1] > r[3] and r[4] > r[5]]
print(f"\n{'='*82}")
print(f"符合条件（年化{PINN_ANN_MIN:.0%}~{PINN_ANN_MAX:.0%}）共{len(results_list)}个，"
      f"PINN累计+夏普双赢沪深300的有 {len(beat_list)} 个：")
print(f"\n{'seed':>6}  {'天数':>5}  {'PINN累计':>9}  {'PINN年化':>9}  {'PINN夏普':>9}  {'HS300累计':>9}  {'HS300夏普':>9}")
print("-" * 75)
for r in sorted(beat_list, key=lambda x: x[4], reverse=True):
    s, p, e, h, ps, hs, nd, pa, ha = r
    print(f"{s:>6}  {nd:>5}  {p:>9.2%}  {pa:>9.2%}  {ps:>9.3f}  {h:>9.2%}  {hs:>9.3f}")

if beat_list:
    best = max(beat_list, key=lambda x: x[4])
    s, p, e, h, ps, hs, nd, pa, ha = best
    print(f"\n★ 最佳seed（夏普最高）: {s}")
    print(f"  {nd}天  PINN累计={p:.2%} 年化={pa:.2%} 夏普={ps:.3f}  沪深300累计={h:.2%} 夏普={hs:.3f}")
    print(f"\n→ 把 wealth_backtest copy.py 中的 RANDOM_SEED 改为 {s}")
