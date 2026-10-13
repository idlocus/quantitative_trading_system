# 量化交易系统

> 项目原则：简单、干净、整洁，用最少的工作量、最清晰的逻辑实现最终目的。
> 文档索引：[docs/README.md](README.md)（按场景导航）

---

## 1. 项目概述

构建一个简洁、专注的量化交易系统：
- **主链**（生产/回测/实盘）：5因子Top100等权组合，T-1信号+T日执行
- **执行风控**：L2数据停机/L3组合熔断/L4个股止损（指令改写层）
- **账实核对**：每日对账、差异报告、超额卖出防线
- **每日重放回测**：逐日复刻生产语义（组件同源，非近似）

---

## 2. 目录结构

```
quant-project/
├── .env / .env.example     # 环境变量（Oracle凭据等，.env不入库）
├── run_daily.py            # ★ 每日主流程 CLI（AM日志+PM执行，Paper/QMT）
├── run_live.py             # 实盘信号驱动（--signal --execute paper/qmt）
├── run_daily_replay.py     # ★ 每日重放回测（严格生产语义，--compare A/B）
├── run_portfolio_backtest.py # 因子组合回测（--fundamental/--band/--all-equal）
├── run_portfolio_walkforward.py # 组合滚动样本外验证
├── run_scan.py             # 每日选股扫描
├── run_backtest.py         # 逐股策略回测（旧引擎，已弃用）
├── run_factor_eval.py      # 因子评价（--source wind/akshare）
├── run_fundamental_eval.py # 基本面因子评价
│
├── trading/                # ★ 主链共享层（config+rules，回测与实盘同源）
├── data/                   # 三层数据源（layered: 缓存→DB→客户端）
│   ├── layered.py          # K线归一化（volume=股/amount=元/adjclose双轨）
│   ├── st_provider.py      # ST状态（ASHAREST PIT）
│   ├── turnover.py         # 日成交额统一口径
│   └── board_params.py     # 申报数量（LOT_SIZE表）
├── live/                   # ★ 实盘模块
│   ├── signals.py          # 生产信号唯一入口 generate_production_signal
│   ├── risk_gate.py        # 执行风控闸门（L2/L3/L4）
│   ├── broker.py           # PaperBroker 模拟撮合
│   ├── qmt_broker.py       # QMT实盘通道（等待券商环境验证）
│   ├── ledger.py           # 持仓账本（去重/零股/超额卖出防线）
│   ├── reconcile.py        # 账实核对
│   └── journal.py          # 每日交易日志
├── portfolio/              # 组合回测
│   ├── backtester.py       # 近似口径回测器
│   ├── daily_replay.py     # ★ 每日重放回测器（严格生产语义）
│   ├── tradable_benchmark.py # 可交易等权对照
│   └── risk_manager.py     # 研究风控（回撤/波动率/行业）
├── selection/ strategy/ backtesting/  # ⚠️ DEPRECATED 研究遗留
├── research/               # 因子评价/风险模型/工件
├── indicators/             # 技术指标库
├── market_env/ sector/     # 市场环境评分/板块
├── webapp/                 # 研究仪表盘
├── experiments/            # 可复现实验脚本
├── tests/                  # 295项单元测试
├── docs/                   # 文档（README.md为索引）
├── runtime/                # 运行时状态（gitignored：risk_state/qmt_orders）
├── results/                # 报告与结构化工件
└── data_cache/             # 本地parquet缓存（可再生，不入库）
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