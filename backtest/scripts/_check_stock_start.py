import pandas as pd, os

param_df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pinn_model', 'data', 'raw', 'stock_ga_parameters_new.csv')))
stock_dir = r'c:/Users/zzy/Desktop/毕业论文2.23/code/data/A股5000+股票历史日数据2026227'

results = []
for _, row in param_df.iterrows():
    code = str(row['stock_id'])
    parts = code.split('.')
    p1 = os.path.join(stock_dir, f'{code}.csv')
    p2 = os.path.join(stock_dir, f'{parts[1].lower()}.{parts[0]}.csv') if len(parts) == 2 else None
    fpath = p1 if os.path.exists(p1) else (p2 if p2 and os.path.exists(p2) else None)
    if fpath:
        df = pd.read_csv(fpath, encoding='utf-8', usecols=[0])
        df.columns = ['日期']
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        start = df['日期'].dropna().min()
        results.append((code, start))

results.sort(key=lambda x: x[1], reverse=True)
print('数据最晚开始的股票（前15名）:')
for code, dt in results[:15]:
    print(f'  {code}: {dt.date()}')

print(f'\n共扫描 {len(results)} 只股票')
print(f'最早共同起点（内连接）将是: {max(dt for _, dt in results).date()}')

# ---------- 查看排除 688729.SH 后，实际加载的40只是哪些 ----------
EXCLUDE = {'688729.SH'}
selected = []
for _, row in param_df.iterrows():
    if len(selected) >= 40:
        break
    code = str(row['stock_id'])
    if code in EXCLUDE:
        print(f'\n[排除] {code}')
        continue
    parts = code.split('.')
    p1 = os.path.join(stock_dir, f'{code}.csv')
    p2 = os.path.join(stock_dir, f'{parts[1].lower()}.{parts[0]}.csv') if len(parts) == 2 else None
    fpath = p1 if os.path.exists(p1) else (p2 if p2 and os.path.exists(p2) else None)
    if fpath:
        df = pd.read_csv(fpath, encoding='utf-8', usecols=[0])
        df.columns = ['日期']
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        start = df['日期'].dropna().min()
        selected.append((code, start, '参数文件'))

# 若不足40，从目录随机补充（random_seed=42）
import random
random.seed(42)
all_files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]
loaded_codes = {s[0] for s in selected} | EXCLUDE
candidates = [f for f in all_files if f.replace('.csv', '') not in loaded_codes]
random.shuffle(candidates)
for fname in candidates:
    if len(selected) >= 40:
        break
    code = fname.replace('.csv', '')
    fpath = os.path.join(stock_dir, fname)
    try:
        df = pd.read_csv(fpath, encoding='utf-8', usecols=[0])
        df.columns = ['日期']
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        start = df['日期'].dropna().min()
        selected.append((code, start, '随机补充'))
    except:
        pass

print(f'\n实际加载的40只股票（共{len(selected)}只）:')
for i, (code, dt, src) in enumerate(selected, 1):
    tag = f'[{src}]' if src == '随机补充' else ''
    print(f'  {i:2d}. {code}  起始:{dt.date()}  {tag}')
print(f'\n最终共同起点: {max(dt for _, dt, _ in selected).date()}')
