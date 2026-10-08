# 量化交易系统

> 项目原则：简单、干净、整洁，用最少的工作量、最清晰的逻辑实现最终目的。

---

## 1. 项目概述

构建一个简洁、专注的量化交易系统：
- 通用框架（股票 + 商品期货）
- 短线选股（每天1-5只）
- 多指标组合确认选股（可配置）
- 信号强度 → 仓位分配
- 完整回测引擎 + 可视化报告

---

## 2. 目录结构

```
quantitative_trading_system/
├── .env                    # 环境变量配置
├── run_backtest.py         # 回测入口脚本（--rolling 支持滚动窗口）
├── run_scan.py             # 每日选股扫描入口
├── run_factor_eval.py      # 因子评价入口（IC/分层/换手率）
├── trading.db              # SQLite 数据库
│
├── config/
│   └── system_config.py    # 系统配置（唯一配置入口）
│
├── research/               # 研究基础设施
│   └── factor_eval.py      # 因子评价（IC/RankIC/分层回测/换手率）
│
├── data/                   # 数据接口
│   ├── __init__.py
│   ├── base/                # 抽象层（Protocol 接口 + DataLoader）
│   │   ├── __init__.py
│   │   ├── interfaces.py    # IKlineProvider, IQuoteProvider,
│   │   │                    #   ISymbolProvider, IIndustryProvider
│   │   └── data_loader.py  # CachedDataSource + DataLoader 工厂
│   ├── wind.py             # Wind 数据接口
│   ├── akshare.py          # AKShare 数据接口
│   └── wind_db.py          # Wind Oracle DB 数据源（统一四接口）
│
├── indicators/             # 技术指标库
│   ├── __init__.py
│   ├── base.py             # 指标注册表（无 ABC）
│   ├── utils.py           # 工具函数（detect_cross, validate_ohlcv）
│   ├── registry.py          # 指标注册表
│   ├── trend_indicators.py  # 趋势指标 (SMA, EMA, MACD, DMI)
│   ├── momentum_indicators.py # 动量指标 (RSI, Stochastic, CCI)
│   ├── volatility_indicators.py # 波动率指标 (ATR, BollingerBands)
│   ├── volume_indicators.py # 成交量指标 (OBV, MFI, VWAP)
│   └── pattern_indicators.py # 形态指标（支撑/阻力/斐波那契）
│
├── selection/              # 选股模块
│   ├── __init__.py
│   ├── config.py           # 选股策略配置
│   ├── scanner.py          # 多指标扫描器
│   └── strategies.yml      # 策略配置文件
│
├── strategy/               # 策略框架
│   ├── __init__.py
│   └── momentum.py         # 动量选股策略
│
├── backtesting/            # 回测引擎
│   ├── __init__.py
│   ├── engine.py           # 核心回测逻辑
│   ├── runner.py           # 回测运行器（流程编排）
│   ├── reporter.py         # 回测报告生成
│   └── templates/          # HTML 报告模板
│       └── report.html     # 回测HTML报告模板
│
├── risk/                   # 风险管理
│   ├── __init__.py
│   ├── constants.py        # 风险参数常量（止损/止盈/仓位权重）
│   └── manager.py          # 仓位计算、止损止盈
│
├── webapp/                 # Web 可视化
│   ├── app.py              # Flask 应用
│   ├── database.py         # 数据库访问层
│   └── templates/         # HTML 模板
│
└── results/                # 回测报告输出目录
```

---

## 3. 数据架构

### 3.1 数据源（均实现 Protocol 接口）

| 数据源 | 实现接口 | 用途 |
|--------|----------|------|
| Wind（WindPy 终端） | IKlineProvider, IQuoteProvider, ISymbolProvider | **PIT 股票池（含退市股）**、批量K线、实时行情 |
| AKShare | IKlineProvider, IQuoteProvider, ISymbolProvider | K线（东财优先、新浪回退）、指数行情、免费备用 |
| Wind Oracle DB | IKlineProvider, IQuoteProvider, ISymbolProvider, IIndustryProvider | K线 + 申万行业 |

Wind 数据源说明（`DATA_SOURCE_TYPE=wind`，需本机 Wind 终端 + WindPy）：
- `list_symbols(as_of=date)`：**PIT 全A股票池**——按历史日期取当日成分，
  包含当时存续、后来退市的股票（消除幸存者偏差的关键能力）
- `preload_batch()`：批量K线预加载（多代码单字段限制下按字段聚合调用，50只/次）
- 回测 runner 自动用回测起始日的 PIT 成分作为股票池

AKShare 数据源说明（免费备用，`DATA_SOURCE_TYPE=akshare`）：
- 个股K线先走东财接口，失败自动回退新浪接口
- 常用指数代码（000300.SH 等）自动路由到指数接口
- **不含退市股票（幸存者偏差）**，仅用于无 Wind 环境的快速验证
- `CachedDataSource.get_trading_dates()` 从缓存日期并集推导真实交易日历

### 3.2 Protocol 接口（data/base/interfaces.py）

```
IKlineProvider      get_kline(symbol, start, end) → pd.DataFrame
IQuoteProvider      get_realtime(symbol), get_close(symbol, date)
ISymbolProvider     list_symbols(), get_tradable_symbols(date)
IIndustryProvider   get_sw_industry_constituents(), get_industry_momentum(code)
```

### 3.3 数据加载流程

```
DataLoader (工厂)
  ├── from_winddb(dsn, user, password)  → WindDBDataSource
  ├── from_wind()                       → WindDataSource
  └── from_akshare()                   → AKShareDataSource

CachedDataSource   — 带内存缓存的 DataLoader 包装
  └── preload_klines(symbols, start, end)  预加载所有 K线
```

### 3.4 数据库表

| 表名 | 用途 |
|------|------|
| `backtests` | 回测结果记录 |
| `trades` | 交易明细 |
| `kline_data` | K线数据缓存 |
| `data_load_log` | 数据加载历史（可追溯） |

### 3.5 数据加载日志

每次数据加载记录到 `data_load_log` 表：
- symbol、load_date、record_count、start_date、end_date、source、backtest_id

---

## 4. 选股系统

### 4.1 板块筛选（申万二级行业）

每日收盘后筛选强势板块：
- 条件：5日涨幅 > 0 AND MA5 > MA20 AND 成交量比 > 1.0
- 取前10名板块
- 仅扫描板块内成分股

### 4.2 选股配置

```python
@dataclass
class IndicatorRule:
    name: str        # "RSI", "MACD", "MA"
    operator: str    # ">", "<", "cross_up", "cross_down"
    threshold: float
    weight: float = 1.0

@dataclass
class SelectionConfig:
    name: str
    indicators: List[IndicatorRule]
    logic: str = "AND"       # AND/OR 组合逻辑
    min_score: int = 60
    max_positions: int = 5
    sw_industry_filter: bool = False   # 是否开启板块筛选
    sw_industry_top_n: int = 10         # 取前N名板块
```

### 4.3 策略配置示例 (strategies.yml)

```yaml
momentum_with_sector_filter:
  name: 动量策略+板块筛选
  indicators:
    - name: RSI
      operator: "<"
      threshold: 40
      weight: 1.0
    - name: MACD
      operator: "cross_up"
      threshold: 0
      weight: 1.0
  logic: AND
  min_score: 60
  max_positions: 5
  sw_industry_filter: true
  sw_industry_top_n: 10
```

特殊操作符：
- `rps_rank_gt`：横截面相对强度（欧奈尔 RPS）——20日收益强于全市场 X% 的股票。
  排名由引擎每日从缓存计算注入，回测/扫描（run_scan 传 constraint 时）均可用。

---

## 5. 回测流程

```
run_backtest.py
├── 数据加载（Wind/AKShare/WindDB）
│   ├── K线预加载带暖机窗口（回溯200自然日，保证指标有回看数据）
│   └── 一并预加载基准指数 + 市场状态指数
├── 交易日历（从缓存K线日期并集推导，自动排除节假日；缓存为空回退工作日历）
├── 板块筛选（可选，Wind Oracle DB）
│   └── 得到 allowed_symbols
├── 选股（StockScanner.scan）
│   └── 连续强度评分：score = 60 + 40 × 加权平均强度
├── 回测（BacktestEngine）
│   ├── 买入：评分最高的前N只（执行价含滑点）
│   ├── T+1：当天买入不能当天卖出
│   ├── 退出优先级：时间止损 → 移动止损 → 均线破位 → 固定/ATR止损止盈
│   └── 止损冷却：止损后N个交易日禁止回买同标的
└── 生成报告（Markdown + HTML，含基准/超额收益）并持久化到 SQLite
```

---

## 6. 风险管理

### 6.1 仓位分配

| 信号评分 | 仓位权重 |
|----------|----------|
| ≥80 | 20% |
| ≥70 | 15% |
| ≥60 | 10% |
| <60 | 5% |

最大持仓数：5只

### 6.2 退出系统（多层）

| 机制 | 默认参数 | 说明 |
|------|----------|------|
| 固定止损 | -3% | 与 ATR 止损取更严格者 |
| ATR 动态止损 | 入场价 - 2×ATR | Wilder EMA 口径，与固定止损取更严格者 |
| 移动止损 | 回撤 5% | 从持仓期最高收盘价回撤触发 |
| 时间止损 | 20 个交易日 | 强制平仓，避免"亏损死扛" |
| 均线破位退出 | 跌破 MA20（默认关闭） | 信号恶化退出 |
| 止盈 | +6% | 固定 |
| 止损冷却 | 5 个交易日 | 止损后禁止回买同标的，防 churn |

**自适应退出**（`adaptive_exits=True`，默认开）：市场处于趋势状态（指数 ADX>30）时
自动放宽移动止损（×1.6）与时间止损（×2），避免趋势股被紧止损打断。

所有参数集中在 `risk/constants.py`（BacktestParams / RiskParams），
可通过 `.env` 覆盖（见 `.env.example`），禁止在业务代码中硬编码。

### 6.3 执行假设（A股交易现实建模）

- 信号与成交均基于当日收盘价（收盘执行模型）
- **涨跌停不可成交**：收盘封涨停不买入、封跌停不卖出（板块差异化：主板10%/创业科创20%/北交30%）
- **交易成本**：佣金（双边）+ 印花税 0.05%（卖出）+ 过户费 0.01‰（双边）+ 可配滑点
- 手续费模型常量见 `risk/constants.py`（BacktestParams）
- T+1：当天买入不能当天卖出

### 6.4 市场环境（四维评分 + 季节性）

| 维度 | 数据源 | 说明 |
|------|--------|------|
| 市场宽度 | 引擎从缓存K线计算涨跌家数 | 时点正确、零网络 |
| 估值分位 | 中证指数历史PE（5年窗口，≤当日） | 全量缓存+内存切片 |
| 宏观增长 | PMI/GDP/CPI（≤当日） | MacroDB 自动装配 |
| 先行指数 | 社融+M2+PMI 合成 | 同上 |
| 季节性 | **A股实证月度统计**（沪深300 2002-2026 中位数+胜率） | 1/2/12月有利，8月最弱；可用 refresh_monthly_stats() 重算 |

仓位系数 position_factor = base(市场评分) × 季节性因子，直接缩配买入资金。

---

## 7. 开发规范

### 7.1 代码风格

- 遵循 PEP 8
- 使用 4 空格缩进
- 函数和类之间用两个空行分隔
- 注释使用中文

### 7.2 命名约定

| 类型 | 风格 | 示例 |
|------|------|------|
| 变量/函数 | snake_case | `initial_capital` |
| 常量 | UPPER_SNAKE_CASE | `MAX_POSITIONS` |
| 类 | CamelCase | `BacktestEngine` |
| 模块 | snake_case | `scanner.py` |

### 7.3 私有成员

- 私有变量：`_private`
- 强私有变量：`__private`

### 7.4 函数设计

- 函数长度不超过 50 行
- 每个函数只负责一个明确任务
- 超过 20 行的逻辑应抽象为函数

### 7.5 错误处理

- 使用 try-except 捕获具体异常
- 异常处理包含具体错误信息
- 使用 logging 模块记录日志
- 生产环境避免使用 print 调试

### 7.6 配置管理

- 所有配置通过 `SystemConfig` 读取 `.env`
- 不在代码中硬编码配置值
- 不使用 config.json（已删除）

### 7.7 数据库操作

- 使用 SQLite，轻量无需配置
- 通过 `webapp/database.py` 的 `BacktestDB` 类操作数据库

---

## 8. 报告说明

### 8.1 回测报告

- **Markdown 报告**：文本格式，便于存档
- **HTML 报告**：交互式权益曲线（以交易日为轴），带基准/超额收益对比

### 8.2 HTML 报告内容

- 整体绩效指标（收益率、基准收益、超额收益、夏普、最大回撤等）
- 权益曲线（交易日索引）
- 交易明细表格（含退出原因：stop_loss / take_profit / trailing_stop / time_stop / signal_exit_ma）

---

## 9. 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10 |
| 数据处理 | pandas, numpy |
| 数据获取 | WindPy, akshare |
| 数据库 | SQLite |
| Web框架 | Flask |
| 图表 | lightweight-charts |
| Oracle | oracledb (thin mode) |

### 9.1 辅助目录

| 目录 | 内容 |
|------|------|
| `utils/` | 日志、WeChat/飞书推送、图片分析 |
| `docs/base/` | 参考文档 PDF、数据库 schema 图片 |

---

## 10. 运行方式

```bash
# 每日选股扫描（短线选股入口）
python run_scan.py                                  # 沪深300池 + 纯趋势跟随
python run_scan.py --strategy oversold_bounce --top 10 --market-env
python run_scan.py --universe zz500                 # 中证500池（zz500/zz1000/all 可选）

# 因子评价（IC/分层/换手率；--source wind 使用PIT池含退市股）
python run_factor_eval.py --factor rsi --period 1 5 10 --universe 100 --source wind

# 基本面因子评价（价值/红利/市值，Wind 日频字段）
python run_fundamental_eval.py --universe 500 --years 3 --period 10 20 60

# 因子驱动组合回测（Phase 3）
python run_portfolio_backtest.py --universe 500 --top 100 --rebalance 20 --fundamental bp,size
python run_portfolio_backtest.py --universe 500 --top 50 --rebalance 5 --factors mfi,rsi,cci --direction negative

# 批量因子筛选（一次预加载，全部指标排名）
python ai-created/factor_screen.py 100 3

# 运行回测（--rolling 支持滚动窗口）
python run_backtest.py

# 启动 Web 服务（项目根目录执行）
python -m webapp.app
# 回测记录: http://localhost:5000
# 研究结果浏览器: http://localhost:5000/results
```