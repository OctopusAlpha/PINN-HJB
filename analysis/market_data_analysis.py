"""
市场数据分析脚本
- 绘制国债、SHIBOR、股票趋势图
- 计算相关性热力图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib
from matplotlib import font_manager

# 1. 先设置全局样式
sns.set_style("whitegrid")

# 2. 再设置中文字体（防止被样式覆盖）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_treasury_data(filepath):
    """加载国债数据 (Table.xls)"""
    # 读取GBK编码的文本文件
    df = pd.read_csv(filepath, encoding='gbk', sep='\t', engine='python')
    
    # 处理列名（可能有Unnamed列）
    df = df.iloc[:, :6]  # 只取前6列
    df.columns = ['时间', '收盘', '涨幅', '金额', '换手率%', '成交次数']
    
    # 处理日期
    df['日期'] = df['时间'].str.split(',').str[0]
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    
    # 处理收盘价
    df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')
    
    # 删除无效数据
    df = df.dropna(subset=['日期', '收盘'])
    df = df.sort_values('日期')
    
    return df[['日期', '收盘']].rename(columns={'收盘': '国债收盘'})


def load_shibor_data(filepath):
    """加载SHIBOR数据 (银行间同业拆借利率)"""
    # 读取Excel，跳过前9行元数据
    df = pd.read_excel(filepath, skiprows=9)
    
    # 获取日期列和7天利率列
    df = df.iloc[:, :2]  # 只取前两列
    df.columns = ['序号', '日期']
    
    # 读取完整的利率数据
    df_full = pd.read_excel(filepath, skiprows=9)
    
    # 获取列名（指标名称）
    headers = pd.read_excel(filepath, nrows=1).columns.tolist()
    
    # 重新读取，使用正确的列名
    df_data = pd.read_excel(filepath, skiprows=9)
    df_data.columns = headers[:len(df_data.columns)]
    
    # 找到日期列和7天SHIBOR列
    date_col = df_data.columns[1]  # 第二列是日期
    
    # 查找7天银行间同业拆借利率列
    shibor_col = None
    for col in df_data.columns:
        if '7天' in str(col) and '同业拆借' in str(col):
            shibor_col = col
            break
    
    if shibor_col is None:
        # 如果没有找到，使用第三列（通常是第一个利率数据）
        shibor_col = df_data.columns[2]
    
    result = df_data[[date_col, shibor_col]].copy()
    result.columns = ['日期', 'SHIBOR_7D']
    
    # 处理日期
    result['日期'] = pd.to_datetime(result['日期'], format='%Y-%m', errors='coerce')
    result['SHIBOR_7D'] = pd.to_numeric(result['SHIBOR_7D'], errors='coerce')
    
    # 删除无效数据
    result = result.dropna()
    result = result.sort_values('日期')
    
    return result


def load_stock_data(filepath, stock_code='sh.600000'):
    """加载股票数据"""
    stock_file = os.path.join(filepath, f'{stock_code}.csv')
    
    if not os.path.exists(stock_file):
        # 尝试查找第一个可用的股票文件
        files = [f for f in os.listdir(filepath) if f.endswith('.csv')]
        if files:
            stock_file = os.path.join(filepath, files[0])
            stock_code = files[0].replace('.csv', '')
    
    df = pd.read_csv(stock_file, encoding='utf-8')
    
    # 重命名列
    df.columns = ['日期', '证券代码', '开盘价', '最高价', '最低价', '收盘价', 
                  '前收盘价', '成交量', '成交额', '复权状态', '换手率', 
                  '交易状态', '涨跌幅', '滚动市盈率', '市净率', '滚动市销率', 
                  '滚动市现率', '是否ST']
    
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df['收盘价'] = pd.to_numeric(df['收盘价'], errors='coerce')
    
    # 删除无效数据
    df = df.dropna(subset=['日期', '收盘价'])
    df = df.sort_values('日期')
    
    return df[['日期', '收盘价']].rename(columns={'收盘价': f'{stock_code}_收盘'}), stock_code


def load_multiple_stocks(filepath, n_stocks=40, random_seed=42):
    """加载多只股票数据（随机选择）"""
    import random
    files = [f for f in os.listdir(filepath) if f.endswith('.csv')]
    random.seed(random_seed)
    files = random.sample(files, min(n_stocks, len(files)))
    
    all_data = []
    stock_codes = []
    
    for file in files:
        try:
            stock_code = file.replace('.csv', '')
            stock_file = os.path.join(filepath, file)
            
            df = pd.read_csv(stock_file, encoding='utf-8')
            df.columns = ['日期', '证券代码', '开盘价', '最高价', '最低价', '收盘价', 
                          '前收盘价', '成交量', '成交额', '复权状态', '换手率', 
                          '交易状态', '涨跌幅', '滚动市盈率', '市净率', '滚动市销率', 
                          '滚动市现率', '是否ST']
            
            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            df['收盘价'] = pd.to_numeric(df['收盘价'], errors='coerce')
            df = df.dropna(subset=['日期', '收盘价'])
            df = df.sort_values('日期')
            
            # 按月重采样
            df['年月'] = df['日期'].dt.to_period('M')
            monthly = df.groupby('年月')['收盘价'].last().reset_index()
            monthly['日期'] = monthly['年月'].dt.to_timestamp()
            monthly = monthly.rename(columns={'收盘价': stock_code})
            
            all_data.append(monthly[['日期', stock_code]])
            stock_codes.append(stock_code)
        except Exception as e:
            print(f"加载 {file} 失败: {e}")
            continue
    
    return all_data, stock_codes


def plot_trends(treasury_df, shibor_df, stock_df, stock_code, output_dir='c:/Users/zzy/Desktop/毕业论文2.23/code/new'):
    """绘制趋势图（合成一张，时间范围以国债数据为准）"""
    
    # 获取国债数据的时间范围
    treasury_start = treasury_df['日期'].min()
    treasury_end = treasury_df['日期'].max()
    print(f"国债数据时间范围: {treasury_start} ~ {treasury_end}")
    
    # 统一为月度数据
    treasury_df['年月'] = treasury_df['日期'].dt.to_period('M')
    treasury_monthly = treasury_df.groupby('年月')['国债收盘'].last().reset_index()
    treasury_monthly['日期'] = treasury_monthly['年月'].dt.to_timestamp()
    
    # 股票数据也按国债时间范围过滤
    stock_df_filtered = stock_df[(stock_df['日期'] >= treasury_start) & (stock_df['日期'] <= treasury_end)]
    stock_df_filtered['年月'] = stock_df_filtered['日期'].dt.to_period('M')
    stock_monthly = stock_df_filtered.groupby('年月')[f'{stock_code}_收盘'].last().reset_index()
    stock_monthly['日期'] = stock_monthly['年月'].dt.to_timestamp()
    
    # SHIBOR数据按国债时间范围过滤
    shibor_filtered = shibor_df[(shibor_df['日期'] >= treasury_start) & (shibor_df['日期'] <= treasury_end)]
    
    # 创建合成图表（3行1列）
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 国债趋势
    axes[0].plot(treasury_monthly['日期'], treasury_monthly['国债收盘'], 
                 color='#2E86AB', linewidth=1.5, label='国债收盘指数')
    axes[0].set_title('国债收盘指数趋势', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('日期', fontsize=11)
    axes[0].set_ylabel('收盘指数', fontsize=11)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # 2. SHIBOR趋势
    axes[1].plot(shibor_filtered['日期'], shibor_filtered['SHIBOR_7D'], 
                 color='#A23B72', linewidth=1.5, label='SHIBOR 7天利率')
    axes[1].set_title('银行间同业拆借利率(7天)趋势', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('日期', fontsize=11)
    axes[1].set_ylabel('利率 (%)', fontsize=11)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 股票趋势
    axes[2].plot(stock_monthly['日期'], stock_monthly[f'{stock_code}_收盘'], 
                 color='#F18F01', linewidth=1.5, label=f'{stock_code} 收盘价')
    axes[2].set_title(f'{stock_code} 股票收盘价趋势', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('日期', fontsize=11)
    axes[2].set_ylabel('收盘价 (元)', fontsize=11)
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trends_analysis.png', dpi=300, bbox_inches='tight')
    print(f"趋势图已保存: {output_dir}/trends_analysis.png")
    plt.show()
    
    return treasury_monthly, stock_monthly


def calculate_correlation(treasury_df, shibor_df, stock_data_list, stock_codes, output_dir='c:/Users/zzy/Desktop/毕业论文2.23/code/new'):
    """计算并绘制相关性热力图"""
    
    # 准备国债月度数据
    treasury_df['年月'] = treasury_df['日期'].dt.to_period('M')
    treasury_monthly = treasury_df.groupby('年月')['国债收盘'].last().reset_index()
    treasury_monthly['日期'] = treasury_monthly['年月'].dt.to_timestamp()
    
    # 合并所有数据
    merged_data = treasury_monthly[['日期', '国债收盘']].copy()
    merged_data = merged_data.rename(columns={'国债收盘': '国债'})
    
    # 添加SHIBOR数据（按月匹配）
    shibor_df['年月'] = shibor_df['日期'].dt.to_period('M')
    shibor_monthly = shibor_df.groupby('年月')['SHIBOR_7D'].last().reset_index()
    shibor_monthly['日期'] = shibor_monthly['年月'].dt.to_timestamp()
    
    merged_data = merged_data.merge(shibor_monthly[['日期', 'SHIBOR_7D']], on='日期', how='outer')
    merged_data = merged_data.rename(columns={'SHIBOR_7D': 'SHIBOR'})
    
    # 添加股票数据
    for i, (stock_df, code) in enumerate(zip(stock_data_list, stock_codes)):
        merged_data = merged_data.merge(stock_df[['日期', code]], on='日期', how='outer')
    
    # 删除日期列，只保留数值列用于计算相关性
    merged_data = merged_data.drop('日期', axis=1)
    
    # 删除全为NaN的列
    merged_data = merged_data.dropna(axis=1, how='all')
    
    # 计算相关性（使用成对完整观测值）
    corr_matrix = merged_data.corr(method='pearson', min_periods=10)
    
    # 绘制热力图
    plt.figure(figsize=(16, 14))
    
    # 只显示国债、SHIBOR和前40只股票
    display_cols = ['国债', 'SHIBOR'] + stock_codes[:40]
    display_cols = [c for c in display_cols if c in corr_matrix.columns]
    
    corr_display = corr_matrix.loc[display_cols, display_cols]
    
    sns.heatmap(corr_display, 
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={'size': 7})
    
    plt.title('市场数据相关性热力图\n(国债、SHIBOR与40只股票)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"相关性热力图已保存: {output_dir}/correlation_heatmap.png")
    plt.show()
    
    return corr_matrix


def main():
    """主函数"""
    # 文件路径
    treasury_file = 'c:/Users/zzy/Desktop/毕业论文2.23/code/data/Table.xls'
    shibor_file = 'c:/Users/zzy/Desktop/毕业论文2.23/code/data/1997-2026年银行间同业拆借利率数据.xlsx'
    stock_dir = 'c:/Users/zzy/Desktop/毕业论文2.23/code/data/A股5000+股票历史日数据2026227'
    output_dir = 'c:/Users/zzy/Desktop/毕业论文2.23/code/new'
    
    print("=" * 60)
    print("市场数据分析")
    print("=" * 60)
    
    # 1. 加载国债数据
    print("\n[1/4] 加载国债数据...")
    treasury_df = load_treasury_data(treasury_file)
    print(f"国债数据: {len(treasury_df)} 条记录, 时间范围: {treasury_df['日期'].min()} ~ {treasury_df['日期'].max()}")
    
    # 2. 加载SHIBOR数据
    print("\n[2/4] 加载SHIBOR数据...")
    shibor_df = load_shibor_data(shibor_file)
    print(f"SHIBOR数据: {len(shibor_df)} 条记录, 时间范围: {shibor_df['日期'].min()} ~ {shibor_df['日期'].max()}")
    
    # 3. 先加载多只股票数据（用于相关性分析），随机选择40只
    print("\n[3/4] 加载多只股票数据用于相关性分析...")
    stock_data_list, stock_codes = load_multiple_stocks(stock_dir, n_stocks=40)
    print(f"成功加载 {len(stock_codes)} 只股票数据")
    print(f"随机选取的股票代码: {stock_codes}")
    
    # 4. 从随机40只中选择第一只作为趋势图展示的股票
    print("\n[4/4] 加载趋势图展示的股票数据...")
    selected_stock_code = stock_codes[0]  # 使用随机40只中的第一只
    stock_df = stock_data_list[0]
    # 转换为与原来相同的格式
    stock_df = stock_df.rename(columns={selected_stock_code: f'{selected_stock_code}_收盘'})
    stock_df['日期'] = pd.to_datetime(stock_df['日期'])
    print(f"趋势图股票 ({selected_stock_code}): {len(stock_df)} 条记录, 时间范围: {stock_df['日期'].min()} ~ {stock_df['日期'].max()}")
    stock_code = selected_stock_code
    
    # 5. 绘制趋势图
    print("\n" + "=" * 60)
    print("绘制趋势图...")
    print("=" * 60)
    treasury_monthly, stock_monthly = plot_trends(treasury_df, shibor_df, stock_df, stock_code, output_dir)
    
    # 6. 计算并绘制相关性热力图
    print("\n" + "=" * 60)
    print("计算相关性并绘制热力图...")
    print("=" * 60)
    corr_matrix = calculate_correlation(treasury_df, shibor_df, stock_data_list, stock_codes, output_dir)
    
    # 输出一些相关性统计
    print("\n" + "=" * 60)
    print("相关性统计")
    print("=" * 60)
    
    if '国债' in corr_matrix.columns and 'SHIBOR' in corr_matrix.columns:
        corr_treasury_shibor = corr_matrix.loc['国债', 'SHIBOR']
        print(f"国债与SHIBOR相关性: {corr_treasury_shibor:.4f}")
    
    print("\n分析完成!")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
