from utils import load_yaml
from model.train import train
import torch
from utils import get_logger
import logging
import time
import os
import pandas as pd
from data.data_loder import calculate_parements_stock
from data.data import (
    load_or_estimate_stock_params,
    estimate_rate_params,
    print_rate_params,
    print_stock_params,
)


def load_shibor(shibor_path):
    """加载 SHIBOR 数据，返回 DataFrame[\u65e5\u671f, SHIBOR_7D]"""
    df_data = pd.read_excel(shibor_path, skiprows=9)
    headers = pd.read_excel(shibor_path, nrows=1).columns.tolist()
    df_data.columns = headers[:len(df_data.columns)]
    date_col = df_data.columns[1]
    shibor_col = None
    for col in df_data.columns:
        if '7\u5929' in str(col) and '\u540c\u4e1a\u62c6\u501f' in str(col):
            shibor_col = col
            break
    if shibor_col is None:
        shibor_col = df_data.columns[2]
    result = df_data[[date_col, shibor_col]].copy()
    result.columns = ['\u65e5\u671f', 'SHIBOR_7D']
    result['\u65e5\u671f'] = pd.to_datetime(result['\u65e5\u671f'], format='%Y-%m', errors='coerce')
    result['SHIBOR_7D'] = pd.to_numeric(result['SHIBOR_7D'], errors='coerce')
    return result.dropna().sort_values('\u65e5\u671f').reset_index(drop=True)


def load_stocks_for_gbm(stock_dir, csv_path, n_assets, lookback_days):
    """加载日频股票数据用于 GBM 参数估计"""
    import random
    param_df = pd.read_csv(csv_path, encoding='utf-8')

    def find_file(code):
        p1 = os.path.join(stock_dir, f'{code}.csv')
        if os.path.exists(p1):
            return p1
        parts = code.split('.')
        if len(parts) == 2:
            num, mkt = parts
            p2 = os.path.join(stock_dir, f'{mkt.lower()}.{num}.csv')
            if os.path.exists(p2):
                return p2
        return None

    stocks = []
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
                if lookback_days and len(df) > lookback_days:
                    df = df.tail(lookback_days)
                stocks.append((code, df[['日期', '收盘价']].rename(columns={'收盘价': 'close'})))
            except Exception as e:
                print(f'  跳过 {code}: {e}')
    print(f'共加载 {len(stocks)} 只股票用于 GBM 参数估计')
    return stocks


if __name__ == "__main__":
    cfg = load_yaml('config.yaml')

    # 设置日志
    os.makedirs(cfg.log_dir, exist_ok=True)
    get_logger(cfg.log_name, log_file=f"{cfg.log_dir}/loss_{time.strftime('%d')}.log")
    logger = logging.getLogger(cfg.log_name)

    # ----------------------------------------------------------
    # 股票 GBM 参数：根据 config.yaml 中 needing_calculate 决定
    # ----------------------------------------------------------
    if cfg.needing_calculate:
        logger.info("从真实日频数据重新估计股票 GBM 参数...")
        stocks = load_stocks_for_gbm(
            stock_dir=cfg.stock_dir,
            csv_path=cfg.csv_path,
            n_assets=cfg.n_assets,
            lookback_days=cfg.gbm_lookback_days,
        )
        params_df = load_or_estimate_stock_params(
            param_path=cfg.result_path,
            stocks=stocks,
            recalculate=True,
            lookback_days=cfg.gbm_lookback_days,
        )
    else:
        logger.info("直接加载股票 GBM 参数文件...")
        params_df = load_or_estimate_stock_params(
            param_path=cfg.csv_path,
            recalculate=False,
        )

    print_stock_params(params_df)
    # 转为字典列表格式（与原 train 接口兼容）
    paraments = params_df.to_dict('records')
    logger.info(f"股票参数已加载，共 {len(paraments)} 只股票")

    # ----------------------------------------------------------
    # 利率 OU 参数：根据 config.yaml 中 needing_calculate_rate 决定
    # ----------------------------------------------------------
    if cfg.needing_calculate_rate:
        logger.info("从 SHIBOR 数据估计利率 OU 过程参数...")
        shibor_df = load_shibor(cfg.shibor_path)
        logger.info(f"SHIBOR 数据已加载，{len(shibor_df)} 条月度记录")
        rate_params = estimate_rate_params(shibor_df, rate_col='SHIBOR_7D', freq='monthly')
        print_rate_params(rate_params)
        # 保存 OU 参数到 CSV
        os.makedirs(os.path.dirname(cfg.rate_result_path), exist_ok=True)
        pd.DataFrame([rate_params]).to_csv(cfg.rate_result_path, index=False, encoding='utf-8')
        logger.info(f"利率 OU 参数已保存至: {cfg.rate_result_path}")
    else:
        logger.info("跳过利率 OU 参数估计（needing_calculate_rate=false）")

    # ----------------------------------------------------------
    # 训练模型
    # ----------------------------------------------------------
    if cfg.train:
        train(cfg, paraments)
    else:
        model_path = os.path.join(cfg.model_dir, "pinn_model_test.pth")
        if os.path.exists(model_path):
            logger.info(f"加载模型: {model_path}")
        else:
            logger.warning(f"模型文件不存在: {model_path}")