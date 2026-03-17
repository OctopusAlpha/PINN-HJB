# PINN-HJB 量化投资系统

基于物理信息神经网络（PINN）求解 Hamilton-Jacobi-Bellman（HJB）方程的最优投资组合与消费策略研究系统。

## 项目结构

```
code/
├── data/                          # 原始金融数据（不纳入 git）
│   ├── A股5000+股票历史日数据.../  # 个股日频数据
│   ├── 1997-2026年银行间同业拆借利率数据.xlsx
│   ├── Table.xls                  # 国债收盘指数
│   ├── 沪深300.csv / 创业50.csv / 中证500.csv
├── pinn_model/                    # PINN 模型训练模块
│   ├── model/                     # 网络结构 (PINN.py / loss.py / train.py)
│   ├── data/raw/                  # GA 估计参数 CSV
│   ├── models/                    # 训练好的模型权重 *.pth
│   ├── logs/                      # 训练日志与可视化图表
│   ├── config.yaml                # 训练超参数配置
│   └── main.py                    # 训练入口
├── hjb_solver/
│   ├── classic/                   # 经典有限差分法 (FDM) 求解器
│   │   ├── hjb_solver.py          # 显式 FDM
│   │   ├── hjb_solver_implicit.py # 隐式 FDM + Newton-Raphson
│   │   └── plot_hjb_results.py    # 求解结果可视化
│   └── modular/                   # 通用模块化框架
│       ├── hjb_solver_base.py     # 抽象基类 + 效用函数工具
│       ├── hjb_solver_crra.py     # CRRA 效用实现
│       ├── hjb_solver_log.py      # 对数效用实现
│       ├── hjb_solver_stock.py    # 股票指数贴现因子扩展
│       ├── pinn_hjb_stock.py      # 单资产 PINN 求解器
│       └── README.py              # 框架修改与扩展说明
├── backtest/                      # 策略回测模块
│   ├── wealth_backtest.py         # 主回测脚本（PINN vs 多基准）
│   ├── plot_pi_strategy.py        # 投资组合策略 π 可视化
│   └── scripts/                   # 辅助工具（一次性使用）
│       ├── _scan_best_seed.py     # 扫描最优 seed
│       ├── _calc_params_seed72.py # 重新估计指定 seed 的 GBM 参数
│       └── _check_stock_start.py  # 检查股票数据起始日期
├── analysis/
│   └── market_data_analysis.py    # 金融数据趋势与相关性分析
├── outputs/                       # 所有生成图片/CSV（不纳入 git）
│   ├── hjb/                       # HJB 求解结果图
│   └── backtest/                  # 回测结果图与数据
├── docs/
│   └── formula.txt                # 核心 HJB 公式 (LaTeX)
├── pyproject.toml                 # uv 项目依赖配置
├── requirements.txt               # pip 依赖备用
└── .gitignore
```

## 快速开始

### 环境安装（推荐 uv）

```bash
# 安装 uv（若未安装）
pip install uv

# 创建虚拟环境并安装依赖
uv sync

# 激活环境
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
```

### 备用方式（pip）

```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 训练 PINN 模型

```bash
cd pinn_model
python main.py
```

训练参数通过 `pinn_model/config.yaml` 配置，主要参数：

| 参数 | 说明 |
|------|------|
| `needing_calculate` | 是否重新估计 GBM 参数 |
| `csv_path` | 股票参数 CSV 路径 |
| `hidden_dims` | 神经网络隐藏层维度 |
| `epochs` | 训练轮数 |
| `output_dim` | 输出维度（= N_assets + 2） |

### 2. HJB 有限差分求解

```bash
cd hjb_solver/classic
python plot_hjb_results.py
```

输出图片保存至 `outputs/hjb/`。

### 3. 策略回测

```bash
cd backtest
python wealth_backtest.py
```

回测对比基准：PINN-HJB 策略 vs 等权组合 vs 纯国债 vs 沪深300 vs 创业50。

结果保存至 `outputs/backtest/`。

### 4. 金融数据分析

```bash
cd analysis
python market_data_analysis.py
```

生成国债、SHIBOR、股票趋势图及相关性热力图。

## HJB 方程

$$V_t + (\alpha - \beta r_t)V_r + \frac{1}{2}\sigma_r^2 V_{rr} + \max_{\pi, C} \left\{ [X_t\pi_t\mu_t + (1-\pi_t)X_t r_t - C_t]V_x + \frac{1}{2}[(\pi_t\sigma_t)^2 + (1-\pi_t)^2 A^2]X_t^2 V_{xx} + \frac{1}{\theta}\psi^{1-\theta}C_t^{\theta} \right\} = 0$$

效用函数：$U(t, x) = \frac{1}{\theta}\psi(t)^{1-\theta} x^{\theta}$

贴现因子：$\psi(t) = \exp\left\{-\frac{1}{1-\theta}\int_0^t \left[\phi_0 + \phi_1 \bar{S}(u) + \phi_2 \bar{S}(u)^2\right] du\right\}$

## 数据说明

| 文件 | 说明 | 来源 |
|------|------|------|
| `data/A股5000+股票历史日数据.../` | 个股日频收盘价 | Tushare |
| `data/1997-2026年银行间同业拆借利率数据.xlsx` | SHIBOR 7天月度利率 | Wind |
| `data/Table.xls` | 国债收盘指数日频数据 | Wind |
| `data/沪深300.csv` 等 | 指数日收益率 | Tushare |

> 数据文件不纳入 git 版本控制，需自行准备并放置到 `data/` 目录下。

## 依赖环境

- Python >= 3.10
- PyTorch >= 2.0
- NumPy / SciPy / Pandas / Matplotlib / Seaborn
