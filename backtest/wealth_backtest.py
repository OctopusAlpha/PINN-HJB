"""
基于 PINN-HJB 模型的财富回测脚本
- 初始财富 w=1
- 仿照 market_data_analysis.py 的数据加载方式加载40只股票数据
- 每日输入 (t, r, w) -> PINN -> 获取投资权重 π (40只股票 + 现金)
- 按权重调整持仓，次日按真实收益率更新财富
- 对比基准：等权组合、纯国债、100%沪深300、100%中证500、100%创业50
"""

import os
import sys
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 中文字体 ----------
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 项目路径 ----------
PINN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pinn_model'))
sys.path.insert(0, PINN_ROOT)

from model.PINN import PINN  # noqa: E402

# ========== 路径配置 ==========
MODEL_PATH      = os.path.join(PINN_ROOT, 'models', 'pinn_model_test_103.pth')
PARAM_PATH      = os.path.join(PINN_ROOT, 'data', 'raw', 'stock_ga_parameters_new.csv')
STOCK_DIR       = r'c:/Users/zzy/Desktop/毕业论文2.23/code/data/A股5000+股票历史日数据2026227'
SHIBOR_FILE     = r'c:/Users/zzy/Desktop/毕业论文2.23/code/data/1997-2026年银行间同业拆借利率数据.xlsx'
TREASURY_FILE   = r'c:/Users/zzy/Desktop/毕业论文2.23/code/data/Table.xls'
OUTPUT_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'backtest'))

# 指数 ETF 文件路径（仅沪深300和创业50）
INDEX_FILES = {
    '沪深300': r'c:/Users/zzy/Desktop/毕业论文2.23/code/data/沪深300.csv',
    '创业50':   r'c:/Users/zzy/Desktop/毕业论文2.23/code/data/创业50.csv',
}

# 手动指定回测起点（None 表示自动取指数最早日期；填日期字符串如 '2025-01-01' 表示手动指定）
BACKTEST_START_DATE = '2025-01-01'

# ========== 模型超参数（与训练保持一致）==========
INPUT_DIM   = 3
HIDDEN_DIMS = [64, 64, 128, 64, 64]
OUTPUT_DIM  = 42
N_ASSETS    = 40           # 前40只股票
RANDOM_SEED = 42


# ===========================================================
# 1. 数据加载函数（仿照 market_data_analysis.py）
# ===========================================================

def load_treasury_data(filepath):
    """加载国债收盘指数日频数据 (Table.xls)。
    返回周期内每个交易日的收盘价及对应日收益率。
    """
    df = pd.read_csv(filepath, encoding='gbk', sep='\t', engine='python')
    df = df.iloc[:, :6]
    df.columns = ['时间', '收盘', '涨幅', '金额', '换手率%', '成交次数']
    df['日期'] = df['时间'].str.split(',').str[0]
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')
    df = df.dropna(subset=['日期', '收盘']).sort_values('日期').reset_index(drop=True)
    # 计算日收益率
    df['国债日收益率'] = df['收盘'].pct_change().fillna(0)
    return df[['日期', '收盘', '国债日收益率']]

def load_shibor_monthly(filepath):
    """加载 SHIBOR 7天利率（月度），返回 DataFrame[日期, SHIBOR_7D]"""
    df_data = pd.read_excel(filepath, skiprows=9)
    headers = pd.read_excel(filepath, nrows=1).columns.tolist()
    df_data.columns = headers[:len(df_data.columns)]

    date_col = df_data.columns[1]
    shibor_col = None
    for col in df_data.columns:
        if '7天' in str(col) and '同业拆借' in str(col):
            shibor_col = col
            break
    if shibor_col is None:
        shibor_col = df_data.columns[2]

    result = df_data[[date_col, shibor_col]].copy()
    result.columns = ['日期', 'SHIBOR_7D']
    result['日期'] = pd.to_datetime(result['日期'], format='%Y-%m', errors='coerce')
    result['SHIBOR_7D'] = pd.to_numeric(result['SHIBOR_7D'], errors='coerce')
    result = result.dropna().sort_values('日期').reset_index(drop=True)
    return result


def load_stock_daily(filepath, stock_code):
    """加载单只股票日频收盘价，返回 DataFrame[日期, close]"""
    stock_file = os.path.join(filepath, f'{stock_code}.csv')
    df = pd.read_csv(stock_file, encoding='utf-8')
    df.columns = ['日期', '证券代码', '开盘价', '最高价', '最低价', '收盘价',
                  '前收盘价', '成交量', '成交额', '复权状态', '换手率',
                  '交易状态', '涨跌幅', '滚动市盈率', '市净率', '滚动市销率',
                  '滚动市现率', '是否ST']
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df['收盘价'] = pd.to_numeric(df['收盘价'], errors='coerce')
    df = df.dropna(subset=['日期', '收盘价']).sort_values('日期').reset_index(drop=True)
    return df[['日期', '收盘价']].rename(columns={'收盘价': 'close'})


def load_40_stocks_daily(stock_dir, param_path, n_assets=40, random_seed=42):
    """
    策略：
    1. 优先从 stock_ga_parameters_new.csv 中取前 n_assets 只股票代码；
    2. 若 CSV 中股票文件不存在，则从目录中随机补充。
    返回：list of (stock_code, daily_df), param_df
    """
    param_df = pd.read_csv(param_path, encoding='utf-8')
    # CSV 中 stock_id 格式如 688009.SH，目录中文件名如 sh.688009.csv
    # 尝试两种命名规则
    def find_file(code):
        # 规则1: 直接 code.csv
        p1 = os.path.join(stock_dir, f'{code}.csv')
        if os.path.exists(p1):
            return p1
        # 规则2: 市场前缀.code.csv (SH->sh, SZ->sz)
        parts = code.split('.')
        if len(parts) == 2:
            num, mkt = parts
            p2 = os.path.join(stock_dir, f'{mkt.lower()}.{num}.csv')
            if os.path.exists(p2):
                return p2
        return None

    stocks = []
    used_params = []
    for _, row in param_df.iterrows():
        if len(stocks) >= n_assets:
            break
        code = str(row['stock_id'])
        fpath = find_file(code)
        if fpath:
            try:
                df = pd.read_csv(fpath, encoding='utf-8')
                df.columns = ['日期', '证券代码', '开盘价', '最高价', '最低价', '收盘价',
                              '前收盘价', '成交量', '成交额', '复权状态', '换手率',
                              '交易状态', '涨跌幅', '滚动市盈率', '市净率', '滚动市销率',
                              '滚动市现率', '是否ST']
                df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
                df['收盘价'] = pd.to_numeric(df['收盘价'], errors='coerce')
                df = df.dropna(subset=['日期', '收盘价']).sort_values('日期').reset_index(drop=True)
                stocks.append((code, df[['日期', '收盘价']].rename(columns={'收盘价': 'close'})))
                used_params.append({'stock_id': code, 'mu': row['mu'], 'sigma': row['sigma']})
            except Exception as e:
                print(f"  跳过 {code}: {e}")

    # 若不足40只，从目录随机补充
    if len(stocks) < n_assets:
        random.seed(random_seed)
        all_files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]
        loaded_codes = {s[0] for s in stocks}
        candidates = [f for f in all_files if f.replace('.csv', '') not in loaded_codes]
        random.shuffle(candidates)
        for fname in candidates:
            if len(stocks) >= n_assets:
                break
            code = fname.replace('.csv', '')
            fpath = os.path.join(stock_dir, fname)
            try:
                df = pd.read_csv(fpath, encoding='utf-8')
                df.columns = ['日期', '证券代码', '开盘价', '最高价', '最低价', '收盘价',
                              '前收盘价', '成交量', '成交额', '复权状态', '换手率',
                              '交易状态', '涨跌幅', '滚动市盈率', '市净率', '滚动市销率',
                              '滚动市现率', '是否ST']
                df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
                df['收盘价'] = pd.to_numeric(df['收盘价'], errors='coerce')
                df = df.dropna(subset=['日期', '收盘价']).sort_values('日期').reset_index(drop=True)
                # 用平均 mu/sigma 近似
                stocks.append((code, df[['日期', '收盘价']].rename(columns={'收盘价': 'close'})))
                used_params.append({'stock_id': code, 'mu': param_df['mu'].mean(), 'sigma': param_df['sigma'].mean()})
            except Exception as e:
                print(f"  跳过补充股票 {code}: {e}")

    print(f"成功加载 {len(stocks)} 只股票")
    used_params_df = pd.DataFrame(used_params)
    return stocks, used_params_df


# ===========================================================
# 1b. 加载指数 ETF 日频数据（作为对照基准）
# ===========================================================

def load_index_etf_returns(index_files, dates):
    """
    加载三大指数 ETF（沪深300/中证500/创业50），
    对齐到指定交易日序列，返回每日收益率 dict。
    CSV 格式: ts_code,trade_date,pre_close,open,high,low,close,...

    处理逻辑：
    - 直接从 CSV 的 pct_chg 列读取日收益率（单位%，转小数）
    - 对于回测日期中指数没有数据的交易日，收益率填0（持有不动）
    - 指数数据覆盖不到的日期（早于数据起始日）自动收益率=0
    """
    result = {}
    for name, fpath in index_files.items():
        try:
            df = pd.read_csv(fpath, encoding='utf-8')
            df['日期'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
            df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
            df = df.dropna(subset=['日期', 'pct_chg']).sort_values('日期').reset_index(drop=True)
            # 日收益率（% -> 小数）
            ret_series = df.set_index('日期')['pct_chg'] / 100.0
            # 对齐到回测交易日，缺失日期填0（指数没有交易=不动）
            aligned = ret_series.reindex(dates, fill_value=0.0)
            result[name] = aligned
            data_start = df['日期'].min().date()
            data_end   = df['日期'].max().date()
            covered = (aligned != 0).sum()
            print(f"  指数 {name}: 数据范围 {data_start}~{data_end}，回测期内有效交易日 {covered} 天")
        except Exception as e:
            print(f"  跳过指数 {name}: {e}")
    return result


# ===========================================================
# 2. 构建对齐的日频价格矩阵
# ===========================================================

def build_price_matrix(stocks):
    """
    将多只股票的日频数据对齐到共同交易日
    返回：price_matrix (DataFrame, index=日期, columns=stock_codes)
    """
    codes = [s[0] for s in stocks]
    dfs = [s[1].set_index('日期').rename(columns={'close': code}) for s, code in zip(stocks, codes)]
    price_matrix = pd.concat(dfs, axis=1)
    # 只保留所有股票都有数据的行（内连接）
    price_matrix = price_matrix.dropna(how='any')
    price_matrix = price_matrix.sort_index()
    return price_matrix


# ===========================================================
# 3. PINN 推断：给定 (t, r, w) 输出 π
# ===========================================================

def load_pinn_model(model_path, device):
    """加载训练好的 PINN 模型"""
    model = PINN(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, output_dim=OUTPUT_DIM).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"模型加载成功: {model_path}")
    return model


def get_portfolio_weights(model, t_val, r_val, w_val, device):
    """
    输入 (t, r, w) 调用 PINN，返回归一化后的投资权重
    Args:
        t_val: 当前时间（0~1归一化）
        r_val: 当前利率（小数，如0.03）
        w_val: 当前财富
        device: torch device
    Returns:
        weights: numpy array shape [N_ASSETS+1]，最后一维为现金
    """
    x = torch.tensor([[t_val, r_val, w_val]], dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(x)  # [1, 42]
    # 第 0 维是值函数，第 1~40 是股票权重，第 41 是现金权重
    raw_pi = output[0, 1:42].cpu().numpy()  # shape [41]
    # 将权重规范化到 [0,1] 且加总=1（clamp负值到0）
    raw_pi = np.clip(raw_pi, 0, None)
    total = raw_pi.sum()
    if total < 1e-8:
        # 若全为0，等权分配
        weights = np.ones(41) / 41
    else:
        weights = raw_pi / total
    return weights  # [41]: 前40为股票，第41为现金


# ===========================================================
# 4. 回测主逻辑
# ===========================================================

def run_backtest(model, price_matrix, shibor_df, treasury_df, index_returns, device, T_years=None):
    """
    逐日回测
    Args:
        model: 加载好的 PINN 模型
        price_matrix: DataFrame[date x stock_codes]，日频收盘价（仅40只股票）
        shibor_df: DataFrame[日期, SHIBOR_7D]，月度利率
        treasury_df: DataFrame[日期, 收盘, 国债日收益率]，国债收盘日频数据
        index_returns: dict {name: pd.Series}，三大指数日收益率（已对齐到dates）
        device: torch device
        T_years: 回测总年数（用于归一化 t），默认用数据实际跨度
    Returns:
        results: DataFrame[date, PINN策略财富, 等权基准财富, 纯国债财富, 沪深300财富, 中证500财富, 创业50财富]
    """
    dates = price_matrix.index.tolist()
    n_days = len(dates)

    if T_years is None:
        delta = (dates[-1] - dates[0]).days / 365.0
        T_years = max(delta, 1.0)
    print(f"回测期间: {dates[0].date()} ~ {dates[-1].date()}，共 {n_days} 个交易日，{T_years:.1f} 年")

    # 构建 SHIBOR 月度→日频 映射（向前填充）
    shibor_daily = shibor_df.set_index('日期')['SHIBOR_7D']
    shibor_series = pd.Series(index=pd.DatetimeIndex(dates), dtype=float)
    for d in dates:
        valid = shibor_daily[shibor_daily.index <= d]
        if len(valid) > 0:
            shibor_series[d] = valid.iloc[-1] / 100.0
        else:
            shibor_series[d] = 0.03

    # 国债日频收益率：对齐到交易日（向前填充）
    treasury_idx = treasury_df.set_index('日期')['国债日收益率']
    treasury_returns = pd.Series(index=pd.DatetimeIndex(dates), dtype=float)
    for d in dates:
        valid = treasury_idx[treasury_idx.index <= d]
        if len(valid) > 0:
            treasury_returns[d] = valid.iloc[-1]
        else:
            treasury_returns[d] = 0.0

    # 日收益率矩阵（仅40只股票）
    returns = price_matrix.pct_change().fillna(0)

    # 初始状态
    w_pinn  = 1.0
    w_equal = 1.0
    w_bond  = 1.0

    # 三大指数初始财富（从回测开始时均为 NaN，有数据才开始）
    index_wealth = {name: None for name in index_returns}   # None 表示尚未开始

    wealth_pinn  = [w_pinn]
    wealth_equal = [w_equal]
    wealth_bond  = [w_bond]
    # 指数财富列表（起点与回测同步，初始为1.0）
    wealth_index = {name: [1.0] for name in index_returns}

    # 等权基准：40只股票 + 1份国债/现金，共41份等权
    n_stocks = price_matrix.shape[1]
    equal_stock_weight = 1.0 / (n_stocks + 1)
    equal_cash_weight  = 1.0 / (n_stocks + 1)
    equal_weights = np.ones(n_stocks) * equal_stock_weight

    print("开始逐日回测...")
    for i in range(n_days - 1):
        current_date = dates[i]

        # 时间归一化 t ∈ [0,1]
        t_val = ((current_date - dates[0]).days / 365.0) / T_years
        t_val = float(np.clip(t_val, 0.0, 1.0))

        # 当前利率
        r_val = float(shibor_series.iloc[i]) if i < len(shibor_series) else 0.03

        # 当日收益率（第 i+1 天相对第 i 天）
        day_returns = returns.iloc[i + 1].values  # shape [n_stocks]

        # ----- PINN 策略 -----
        weights_all = get_portfolio_weights(model, t_val, r_val, w_pinn, device)
        stock_weights = weights_all[:n_stocks]
        cash_weight   = weights_all[40] if len(weights_all) > n_stocks else (1 - stock_weights.sum())
        cash_weight   = float(np.clip(cash_weight, 0, 1))

        r_daily = r_val / 252.0
        portfolio_return_pinn = float(np.dot(stock_weights[:n_stocks], day_returns)) + cash_weight * r_daily
        w_pinn = w_pinn * (1 + portfolio_return_pinn)
        w_pinn = max(w_pinn, 1e-6)

        # ----- 等权基准（40只股票各1/41 + 国债1/41）-----
        bond_ret_today = float(treasury_returns.iloc[i + 1]) if i + 1 < len(treasury_returns) else 0.0
        portfolio_return_equal = float(np.dot(equal_weights, day_returns)) + equal_cash_weight * bond_ret_today
        w_equal = w_equal * (1 + portfolio_return_equal)
        w_equal = max(w_equal, 1e-6)

        # ----- 纯国债策略（100% 持有 Table.xls 国债）-----
        w_bond = w_bond * (1 + bond_ret_today)

        # ----- 三大指数对照（各自100%持有，与回测同起点）-----
        for name, ret_series in index_returns.items():
            idx_ret = float(ret_series.iloc[i + 1]) if i + 1 < len(ret_series) else 0.0
            if index_wealth[name] is None:
                index_wealth[name] = 1.0
            index_wealth[name] = index_wealth[name] * (1 + idx_ret)
            index_wealth[name] = max(index_wealth[name], 1e-6)
            wealth_index[name].append(index_wealth[name])

        wealth_pinn.append(w_pinn)
        wealth_equal.append(w_equal)
        wealth_bond.append(w_bond)

        if (i + 1) % 252 == 0:
            idx_str = '  '.join([f"{n}={index_wealth[n]:.4f}" for n in index_returns])
            print(f"  第 {(i+1)//252} 年末 ({current_date.date()}): PINN={w_pinn:.4f}, 等权={w_equal:.4f}, 国债={w_bond:.4f}  {idx_str}")

    results = pd.DataFrame({
        '日期': dates,
        'PINN策略财富': wealth_pinn,
        '等权基准财富': wealth_equal,
        '纯国债财富':   wealth_bond,
    })
    for name in index_returns:
        results[f'{name}财富'] = wealth_index[name]
    results.set_index('日期', inplace=True)
    return results


# ===========================================================
# 5. 结果可视化
# ===========================================================

def plot_results(results, output_dir):
    """绘制财富曲线（仅上方财富对比图）"""
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))

    ax1.plot(results.index, results['PINN策略财富'],
             color='#2E86AB', linewidth=2.0, label='PINN-HJB 策略')
    ax1.plot(results.index, results['等权基准财富'],
             color='#F18F01', linewidth=1.5, linestyle='--', label='等权基准（40股+国债）')
    ax1.plot(results.index, results['纯国债财富'],
             color='#4CAF50', linewidth=1.5, linestyle=':', label='纯国债')
    # 指数对照曲线
    index_colors = {'沪深300': '#E63946', '创业50': '#F77F00'}
    for name, color in index_colors.items():
        col = f'{name}财富'
        if col in results.columns:
            ax1.plot(results.index, results[col],
                     color=color, linewidth=1.5, linestyle='-.', label=f'100% {name}')
    ax1.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='初始财富=1')
    ax1.set_title('PINN-HJB 策略 vs 各基准 财富曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('日期', fontsize=11)
    ax1.set_ylabel('财富（初始=1）', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'wealth_backtest.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n财富曲线已保存: {save_path}")
    # plt.show()


def print_summary(results):
    """打印回测统计摘要"""
    pinn  = results['PINN策略财富']
    equal = results['等权基准财富']
    bond  = results['纯国债财富']
    n_days = len(results)
    years  = n_days / 252.0

    def calc_stats(wealth_series):
        total_return = wealth_series.iloc[-1] / wealth_series.iloc[0] - 1
        ann_return   = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        daily_ret    = wealth_series.pct_change().dropna()
        ann_vol      = daily_ret.std() * np.sqrt(252)
        sharpe       = ann_return / ann_vol if ann_vol > 0 else 0.0
        rolling_max  = wealth_series.cummax()
        drawdown     = (wealth_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return total_return, ann_return, ann_vol, sharpe, max_drawdown

    # 收集所有策略（回测起点已对齐，所有序列长度相同）
    strategies = {'PINN-HJB策略': pinn, '等权基准': equal, '纯国债': bond}
    index_names = ['沪深300', '创业50']
    for name in index_names:
        col = f'{name}财富'
        if col in results.columns:
            strategies[name] = results[col]

    print("\n" + "=" * 90)
    print("回测统计摘要")
    print("=" * 90)
    header = f"{'指标':<16}" + "".join([f"{k:>14}" for k in strategies.keys()])
    print(header)
    print("-" * 90)

    stats_list = {k: calc_stats(v) for k, v in strategies.items()}

    print(f"{'最终财富':<16}" + "".join([f"{strategies[k].iloc[-1]:>14.4f}" for k in stats_list]))
    print(f"{'累计收益率':<16}" + "".join([f"{v[0]:>14.2%}" for v in stats_list.values()]))
    print(f"{'年化收益率':<16}" + "".join([f"{v[1]:>14.2%}" for v in stats_list.values()]))
    print(f"{'年化波动率':<16}" + "".join([f"{v[2]:>14.2%}" for v in stats_list.values()]))
    print(f"{'夏普比率':<16}" + "".join([f"{v[3]:>14.4f}" for v in stats_list.values()]))
    print(f"{'最大回撤':<16}" + "".join([f"{v[4]:>14.2%}" for v in stats_list.values()]))
    print("=" * 90)


# ===========================================================
# 6. 主函数
# ===========================================================

def main():
    print("=" * 60)
    print("PINN-HJB 资产财富回测")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ---------- 加载模型 ----------
    print("\n[1/5] 加载 PINN 模型...")
    model = load_pinn_model(MODEL_PATH, device)

    # ---------- 加载参数文件 ----------
    print("\n[2/5] 加载股票参数...")
    param_df = pd.read_csv(PARAM_PATH, encoding='utf-8')
    print(f"参数文件包含 {len(param_df)} 只股票")

    # ---------- 加载40只股票日频数据 ----------
    print("\n[3/5] 加载40只股票日频数据...")
    stocks, used_params_df = load_40_stocks_daily(STOCK_DIR, PARAM_PATH, n_assets=N_ASSETS, random_seed=RANDOM_SEED)
    print(f"实际加载股票数: {len(stocks)}")

    # ---------- 构建价格矩阵 ----------
    print("\n[4/5] 构建对齐日频价格矩阵...")
    price_matrix = build_price_matrix(stocks)
    print(f"价格矩阵形状: {price_matrix.shape}  (交易日 x 资产数)")
    print(f"时间范围: {price_matrix.index.min().date()} ~ {price_matrix.index.max().date()}")

    # ---------- 加载 SHIBOR 数据 ----------
    print("\n加载 SHIBOR 利率数据...")
    shibor_df = pd.read_excel(SHIBOR_FILE, skiprows=9)
    headers = pd.read_excel(SHIBOR_FILE, nrows=1).columns.tolist()
    shibor_df.columns = headers[:len(shibor_df.columns)]
    date_col = shibor_df.columns[1]
    shibor_col = None
    for col in shibor_df.columns:
        if '7天' in str(col) and '同业拆借' in str(col):
            shibor_col = col
            break
    if shibor_col is None:
        shibor_col = shibor_df.columns[2]
    shibor_clean = shibor_df[[date_col, shibor_col]].copy()
    shibor_clean.columns = ['日期', 'SHIBOR_7D']
    shibor_clean['日期'] = pd.to_datetime(shibor_clean['日期'], format='%Y-%m', errors='coerce')
    shibor_clean['SHIBOR_7D'] = pd.to_numeric(shibor_clean['SHIBOR_7D'], errors='coerce')
    shibor_clean = shibor_clean.dropna().sort_values('日期').reset_index(drop=True)
    print(f"SHIBOR 数据: {len(shibor_clean)} 条月度记录，范围: {shibor_clean['日期'].min().date()} ~ {shibor_clean['日期'].max().date()}")

    # ---------- 加载国债数据 ----------
    print("\n加载国债收盘数据 (Table.xls)...")
    treasury_clean = load_treasury_data(TREASURY_FILE)
    print(f"国债数据: {len(treasury_clean)} 条记录，时间范围: {treasury_clean['日期'].min().date()} ~ {treasury_clean['日期'].max().date()}")

    # ---------- 加载指数对照数据（先确定回测起点）----------
    print("\n加载指数 ETF 对照数据（沪深300、创业50）...")

    # 确定回测起点
    if BACKTEST_START_DATE is not None:
        # 手动指定起点
        backtest_start = pd.Timestamp(BACKTEST_START_DATE)
        print(f"  回测起点（手动指定）: {backtest_start.date()}")
    else:
        # 自动取指数中最早有数据的日期
        index_earliest = []
        for name, fpath in INDEX_FILES.items():
            try:
                df_tmp = pd.read_csv(fpath, encoding='utf-8')
                df_tmp['日期'] = pd.to_datetime(df_tmp['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
                earliest = df_tmp['日期'].dropna().min()
                index_earliest.append(earliest)
                print(f"  {name} 最早数据日期: {earliest.date()}")
            except Exception as e:
                print(f"  跳过 {name}: {e}")
        backtest_start = min(index_earliest) if index_earliest else price_matrix.index.min()
        print(f"  回测起点（自动对齐）: {backtest_start.date()}")

    price_matrix = price_matrix[price_matrix.index >= backtest_start]
    print(f"裁剪后价格矩阵: {price_matrix.shape}，时间范围: {price_matrix.index.min().date()} ~ {price_matrix.index.max().date()}")

    # 第二步：用裁剪后的交易日序列加载指数收益率
    dates_for_index = price_matrix.index.tolist()
    index_returns = load_index_etf_returns(INDEX_FILES, dates_for_index)
    print(f"成功加载 {len(index_returns)} 个指数对照")

    # ---------- 回测 ----------
    print("\n[5/5] 开始回测...")
    results = run_backtest(model, price_matrix, shibor_clean, treasury_clean, index_returns, device)

    # ---------- 结果输出 ----------
    print_summary(results)

    # 保存 CSV
    csv_path = os.path.join(OUTPUT_DIR, 'wealth_backtest_results.csv')
    results.to_csv(csv_path, encoding='utf-8-sig')
    print(f"\n回测数据已保存: {csv_path}")

    # 绘图
    plot_results(results, OUTPUT_DIR)

    print("\n回测完成！")


if __name__ == '__main__':
    main()
