# Progress Log

## Session: 2026-06-01

### Phase 1: 需求确认与数据源探索
- **Status:** complete
- **Started:** 2026-06-01
- Actions taken:
  - 读取现有 market_env 代码（scorer.py, macro_db.py, wind_provider.py）
  - 读取 backtesting/engine.py 了解市场环境集成方式
  - 读取 selection/scanner.py 了解选股约束机制
  - 测试 akshare 宏观数据函数可用性
  - 创建 task_plan.md, findings.md, progress.md 三个规划文件
- Files created/modified:
  - task_plan.md (created)
  - findings.md (created)
  - progress.md (created)

### Phase 2: 设计并实现 AkShareMarketEnvProvider
- **Status:** complete
- **Started:** 2026-06-01
- Actions taken:
  - 创建 market_env/akshare_provider.py (AkShareMarketEnvProvider)
  - 实现 _AkShareFetcher 封装 akshare 函数调用
  - 实现日期解析（支持 "2026年05月份", "YYYYMM", "YYYYMMDD" 等格式）
  - 实现 27 个指标到 akshare 函数的映射（17 可用，10 需替代）
  - 特殊处理：国债收益率过滤"中债国债收益率曲线"、融资融券按日汇总
- Files created/modified:
  - market_env/akshare_provider.py (created)
  - market_env/__init__.py (updated)

### Phase 3: 实现 MacroDB 基于 AkShareMarketEnvProvider
- **Status:** complete
- **Started:** 2026-06-01
- Actions taken:
  - 重写 market_env/macro_db.py 使用 AkShareMarketEnvProvider
  - 实现 get_macro_score() — PMI/GDP/CPI 加权评分
  - 实现 get_leading_score() — 领先指标扩散指数
  - 实现 get_syzm() — 先行合成指数（社融+M2+PMI）
  - 实现 get_current_values() — 当前指标快照
- Files created/modified:
  - market_env/macro_db.py (rewritten)

### Phase 4: 集成 AkShareMarketEnvProvider 到 BacktestRunner
- **Status:** complete
- **Started:** 2026-06-01
- Actions taken:
  - 修改 backtesting/runner.py 在 run() 中实例化 AkShareMarketEnvProvider
  - 将 provider 传入 BacktestEngine(backtest_config, market_env_provider=provider)
  - 验证 MarketEnvScorer 四维评分正确计算
- Files created/modified:
  - backtesting/runner.py (updated)

### Phase 5: 测试与验证
- **Status:** complete
- **Started:** 2026-06-01
- Actions taken:
  - 端到端测试四维评分（市场综合分=57, position_factor=0.6）
  - 验证宏观指标获取（15 个可用）
  - 四维均值=58 与综合分=57 基本一致

### Phase 6: ATR 动态止损与季节性因子
- **Status:** complete
- **Started:** 2026-06-01
- Actions taken:
  - 在 Position 添加 atr 字段，RiskManager.allocate 计算 ATR
  - RiskManager.check_exit 支持 ATR 动态止损（entry - 2×ATR）
  - 取固定止损和 ATR 止损中更严格者
  - 创建 market_env/seasonal.py（总统周期 + 月度规律 + Sell in May）
  - MarketEnvScore 添加 seasonal_signal / seasonal_factor 字段
  - MarketEnvScorer.score 集成季节性因子（base_factor × seasonal_factor）
  - 验证 ATR 动态止损逻辑、季节性因子计算
- Files created/modified:
  - risk/manager.py (updated — ATR 动态止损)
  - market_env/seasonal.py (created)
  - market_env/scorer.py (updated — 季节性因子集成)
- Test Results:
  | Test | Input | Expected | Actual | Status |
  |------|-------|----------|--------|--------|
  | ATR stop (atr=3, price=96, -4%) | entry=100, atr=3 | None (ATR stop=-6%) | None | ✓ |
  | ATR stop (atr=3, price=94, -6%) | entry=100, atr=3 | stop_loss | stop_loss | ✓ |
  | ATR stop (atr=1, price=97, -3%) | entry=100, atr=1 | stop_loss (fixed tighter) | stop_loss | ✓ |
  | Seasonal 2026-06 | 年2+弱月 | unfavorable, ~0.83 | unfavorable, 0.832 | ✓ |
  | Seasonal 2026-11 | 年2+强月 | favorable, 1.2 | favorable, 1.200 | ✓ |
  | Seasonal 2024-11 | 年4+强月 | favorable, 1.2 | favorable, 1.200 | ✓ |
  | Scorer with seasonal | 20260529 | factor × seasonal | 52分→0.72 | ✓ |

### Phase 7: DMI 策略集成 + 自适应策略
- **Status:** complete
- **Started:** 2026-06-01
- Actions taken:
  - DMI+ADX 已在 indicators/trend_indicators.py 实现
  - scanner.py 添加 dmi_golden_cross / dmi_death_cross 操作符
  - strategies.yml 添加 trend_following_with_dmi 和 atr_trend_strategy
  - ScanConstraint 新增市场信号字段（breadth/valuation/macro/leading/seasonal）
  - BacktestEngine._build_constraint 填充市场信号到 ScanConstraint
  - 新建 strategy/adaptive.py（AdaptiveStrategy 根据市场信号自动选策略）
  - runner.py 支持 strategy_name="adaptive" 创建自适应策略
- Files created/modified:
  - selection/scanner.py (updated — DMI 操作符)
  - selection/strategies.yml (updated — DMI 策略 + adaptive 条目)
  - selection/config.py (updated — ScanConstraint 市场信号字段)
  - backtesting/engine.py (updated — _build_constraint 填充信号)
  - strategy/adaptive.py (created — AdaptiveStrategy)
  - strategy/__init__.py (updated — 导出 AdaptiveStrategy)
- Strategy selection rules:
  | 市场状态 | 仓位系数 | 选用策略 |
  |---------|---------|---------|
  | trending + cheap | ≥0.8 | atr_trend |
  | trending | ≥0.6 | trend_following |
  | ranging + cheap | ≥0.5 | momentum |
  | ranging + expensive | 0.3-0.5 | breakout |
  | pf < 0.3 | <0.3 | atr_trend（保守） |

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 27 指标 fetch | akshare | 17 可用 | 17 可用 | ✓ |
| 四维评分计算 | 20260529 | 0-100 | 综合57 | ✓ |
| MacroDB.get_macro_score | 20260529 | score+signal | 55/neutral | ✓ |
| MacroDB.get_leading_score | 20260529 | score+signal | 70/expanding | ✓ |
| MacroDB.get_syzm | 20260529 | score+signal | 35/contracting | ✓ |
| MarketEnvScorer.score | 20260529 | 四维评分 | 综合57/因子0.6 | ✓ |
| BacktestRunner集成 | provider注入 | engine接收 | 成功 | ✓ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-06-01 | akshare 函数签名错误（stock_board_change_em 不接受 symbol） | 1 | 移除参数，使用无参版本 |
| 2026-06-01 | GDP 日期格式 "2025年第1-4季度" 无法解析 | 2 | 添加 YYYYMM 和季度格式解析 |
| 2026-06-01 | 国债收益率数据只到 2021-01（akshare 数据限制） | 1 | 用宏观指标推导估算 |
| 2026-06-01 | 融资融券余额返回空（groupby 后日期未解析） | 2 | 添加 agg="sum" 参数支持按日汇总 |
| 2026-06-01 | 北向资金当日净买额最新几天为 NaN | 1 | 接受 NaN（数据源限制） |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 7 全部完成 |
| Where am I going? | 框架完成，可用 |
| What's the goal? | 自适应策略自动选择（无需手工切换）+ ATR 动态止损 + 季节性因子 |
| What have I learned? | AdaptiveStrategy 读取 ScanConstraint 市场信号自动选策略；engine 每日填充 breadth/valuation/macro/leading/seasonal 信号 |
| What have I done? | Phase 7: AdaptiveStrategy + ScanConstraint 市场信号 + engine._build_constraint |

## Session: 2026-06-01 (afternoon)

### Task 1: 多策略组合 ✅
- **Status:** complete
- Actions taken:
  - 创建 `strategy/multi.py`（MultiStrategy + SubStrategy）
  - 注册 `MultiStrategy` 到 `strategy/__init__.py`
  - 添加 `multi_strategy` 到 `strategies.yml`（4个子策略，权重0.35/0.30/0.20/0.15）
  - 完全重写 `backtesting/runner.py` 添加 `_create_strategy()` 工厂方法
  - 验证 MultiStrategy 正确加载子策略和权重
- Files modified: strategy/multi.py, strategy/__init__.py, selection/strategies.yml, backtesting/runner.py

### Task 2: 滚动窗口回测 + 样本外验证 ✅
- **Status:** complete
- Actions taken:
  - 给 `Trade` 添加 `phase` 字段（"IS"/"OOS"）
  - 创建 `BacktestPhaseResult` 数据类（用于单阶段结果）
  - 给 `BacktestResult` 添加 `phase_results` 列表字段
  - 添加 `_current_phase` 实例变量到 `BacktestEngine`
  - 在 `_execute_buy`/`_execute_sell` 自动记录 phase 标签
  - 实现 `BacktestEngine.run_rolling()`：每年 IS（80%）+ OOS（20%）滚动前进
  - 实现 `_reset_state()` 重置各阶段状态
  - 实现 `_run_single_day()` 供滚动窗口调用
  - `BacktestRunner` 添加 `run_rolling()` 方法支持 `--rolling-window 252 --oos-ratio 0.2`
  - `_generate_reports_rolling()` 生成各阶段对比表（rolling_phases_*.md）
- Files modified: backtesting/engine.py, backtesting/runner.py
- Rolling window logic:
  - step_days = 252 × 0.8 = 202（每年IS前进202天）
  - OOS窗口 = 252 × 0.2 = 50天
  - 各阶段 phase_results 记录：phase_name, start/end_date, 总收益/年化/夏普/回撤/胜率/交易次数

### Task 3: 每日监控报告 ✅
- **Status:** complete
- Actions taken:
  - `BacktestReporter` 添加 `generate_daily_monitor(engine, latest_date)` 方法
  - 报告包含：市场环境6维信号、当前持仓、今日交易、账户概览
  - 修复 reporter.py 多次编辑损坏问题，完整重写
  - 修复 reporter.py trade.phase 字段显示
- Files modified: backtesting/reporter.py

### Session: 2026-06-02 (上午) - 数据验证

**WindDB 数据源修复:**
- wind_db.py 原来使用的 `WIND_CODE` 列名不存在，实际列为 `S_INFO_WINDCODE`
- `S_INFO_LISTED_STATE` 不存在，改用 `S_INFO_DELISTDATE IS NULL` 筛选上市股票
- oracledb thin mode 有参数绑定 bug，改用 `oracledb.init_oracle_client()` thick mode
- K线查询改用直接 cursor.execute + pd.DataFrame（绕过 pandas read_sql 的 named param 问题）
- cursor.description 返回大写列名，添加 `.lower()` 处理

**滚动窗口验证结果（300只股票，2023-01 至 2024-12）:**

| Phase | Return | Sharpe | MaxDD | Trades |
|-------|--------|--------|-------|--------|
| 2023_IS | +2.00% | +0.17 | -4.67% | 61 |
| 2023_OOS | +0.80% | -0.20 | -4.40% | 29 |
| 2023_IS | -2.09% | -0.20 | -12.54% | 112 |
| 2023_OOS | +10.21% | +4.93 | -2.88% | 30 |

**结论:** OOS 表现波动大（-0.20 ~ +4.93），样本量不足（仅2个完整窗口），需更长历史（2020-起）才能可靠评估策略。

### All Tasks Complete
| Task | Status |
|------|--------|
| 多策略组合 | ✅ |
| 滚动窗口回测 + 样本外验证 | ✅ |
| 每日监控报告 | ✅ |

## Session: 2026-06-03 - 港股数据支持与 9868.HK 分析

### Task 1: HKSHAREEODPRICES 港股行情支持 ✅
- **Status:** complete
- 添加 `_get_hkshare_kline()` 方法支持 `.HK` 结尾股票代码
- `WindDBDataSource.get_kline()` 自动识别 A 股或港股
- 验证：9868.HK 数据 1204 行（2021-07-07 至 2026-06-02）

### Task 2: DMI 指标 Bug 修复 ✅
- **Status:** complete
- **Bug:** `DMI_ADX()` 中 `plus_dm`/`minus_dm` 使用 `np.where()` 默认索引，与 data 索引不匹配，导致所有 rolling mean 计算 NaN
- **Fix:** `pd.Series(np.where(...), index=data.index)` 确保索引对齐
- 同时修复了 `tr = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)` 缺少 `index=data.index`

### Task 3: 9868.HK 完整分析 ✅
- **Status:** complete
- `wind_db.py` 已支持港股查询（HKSHAREEODPRICES）
- 完成 9868.HK 回测分析（报告保存至 `results/9868HK_analysis_20260603.md`）

| 策略 | 收益率 | vs 买入持有 |
|------|--------|------------|
| 增强动量 (RSI<30+ADX>20+MACD金叉) | 0% | -120% |
| 趋势跟随 (RSI>50+ADX>25+MACD>0) | -96% | -220% |
| **买入持有** | **+120.5%** | - |

- **结论：** 所有技术策略大幅跑输买入持有；RSI 超卖策略在港股趋势股上完全不适用

## Session: 2026-06-04 - Top-Down 市场状态检测 + 纯趋势跟随策略

### Task 1: ADX 市场状态注入 ✅
- **Status:** complete
- 在 `ScanConstraint` 添加 `adx_signal` 字段（'trending'/'ranging'/'neutral'）
- 添加 `is_adx_trending()` / `is_adx_ranging()` 判断方法
- 在 `engine._build_constraint()` 中从市场指数 K线计算 ADX 并注入 constraint
  - ADX > 30 → trending
  - ADX < 20 → ranging
  - 20 ≤ ADX ≤ 30 → neutral
- 使用 `000985.SH` 中证全指作为市场代表

### Task 2: price_above_ma 操作符 ✅
- **Status:** complete
- 在 `scanner._check_condition()` 添加 `price_above_ma` 操作符
- 在 `config.py` 的 `Operator` 类型中添加 `"price_above_ma"`
- 用于纯趋势跟随策略判断价格是否站稳均线上方

### Task 3: pure_trend_follow 策略 ✅
- **Status:** complete
- 在 `strategies.yml` 添加 `pure_trend_follow` 策略
- 指标：ADX>25 + price_above_ma(20) + RSI>40 + RSI<80
- 逻辑：AND，min_score=60，max_positions=5

### Task 4: AdaptiveStrategy 重写 ✅
- **Status:** complete
- 重写 `_select_strategy()` 以 ADX 信号为核心判断依据
- 新策略池添加 `'pure_trend': 'pure_trend_follow'`
- 策略选择规则：
  - ADX>30 (trending) → pure_trend
  - ADX<20 (ranging) + cheap → momentum
  - ADX<20 (ranging) + not cheap → breakout
  - 20≤ADX≤30 (neutral) + cheap + favorable → momentum
  - 20≤ADX≤30 (neutral) + other → momentum_sector
  - position_factor<0.3 → atr_trend

### Task 5: 验证 ✅
- **Status:** complete
- 9868.HK 回测结果（2024-06-03 至 2026-06-02，买入持有 +120.5%）：

| 策略 | 止损 | 收益率 | 交易次数 | 最大回撤 | 说明 |
|------|------|--------|----------|----------|------|
| pure_trend_follow | 无 | +27.5% | 1 | - | 买入后一直持有，止损=999 |
| pure_trend_follow | 5% | -77.3% | 58 | -83.1% | 频繁止损出局 |
| pure_trend_follow | 10% | -93.8% | 35 | -94.0% | 止损太紧 |
| enhanced_momentum | 5% | 0% | 0 | 0% | ADX过滤严格，无信号 |
| short_term_momentum | 5% | 0% | 0 | 0% | 无信号 |

- **关键发现**: ADX过滤过于严格导致RSI超卖策略无法触发；纯趋势跟随在大波动港股上被5%止损频繁打断；buy-hold是港股最优策略

### Known Issue
- **WindDB 无 000985.SH 指数数据**：`get_kline('000985.SH')` 返回空，ADX 市场状态注入失败（constraint.adx_signal = None）
- 当前 fallback：adx_signal=None → is_adx_trending()=False → 回退到原有 breadth/leading 信号逻辑
- 未来需要通过 AkShare 获取指数数据来计算 ADX

### Files Modified
- `selection/config.py` — adx_signal 字段 + is_adx_trending/is_adx_ranging 方法
- `backtesting/engine.py` — _build_constraint 添加 ADX 计算和注入
- `selection/strategies.yml` — 添加 pure_trend_follow 策略
- `selection/scanner.py` — 添加 price_above_ma 操作符
- `strategy/adaptive.py` — 重写 _select_strategy，添加 pure_trend 池
## Session: 2026-09-23 - 引擎正确性修复 + 多层退出系统

### Task 1: 致命 Bug 修复 ✅
- **Bug A（致命）:** `BacktestEngine.__init__` 调用 `RiskManager(stop_loss=..., take_profit=...)`，
  但签名是 `(params, data_source)` → 引擎无法实例化，回测全链路断裂（run_9868_backtest.py 用
  `__new__` 绕过即此因）。修复：传入 `RiskParams(STOP_LOSS=..., TAKE_PROFIT=...)`
- **Bug B（估值）:** `_generate_result` 期末持仓按入场日估值（`get_close(symbol, pos.entry_date)`），
  持浮盈持仓时 final_capital 系统性偏差。修复：按最后交易日收盘估值
- **Bug C（日历）:** 交易日历用 `pd.bdate_range`（含A股节假日，节假日重复信号）。
  修复：`CachedDataSource.get_trading_dates()` 从缓存日期并集推导真实日历
- **Bug D（性能）:** T+1 检查 `trading_dates.index()` 每次全表扫描（O(n²)）。
  修复：MarketData 惰性构建 date→idx 字典，O(1) 查询；胜率/持仓天数改单遍配对
- **Bug E:** 指标暖机缺失——K线只从回测开始日预加载，前期指标计算数据不足。
  修复：预加载回溯 200 自然日（PRELOAD_LOOKBACK_DAYS），交易日历仍从开始日切齐
- **Bug F:** run_9868_backtest.py 传入 Timestamp 而非 date。修复：add_trading_date 统一归一化

### Task 2: 多层退出系统 ✅
- 移动止损（trailing_stop=5%）：从持仓期最高收盘回撤触发
- 时间止损（max_holding_days=20）：强制平仓
- 均线破位退出（exit_on_ma_break，默认关闭）：跌破 MA20
- 止损冷却（cooldown_days=5）：止损后禁止回买同标的，防 churn（修 BBB 止损次日回买问题）
- 退出优先级：时间 → 移动 → 均线破位 → 固定/ATR止损止盈
- 期末持仓当日无价格（停牌）跳过退出检查

### Task 3: 执行真实性 ✅
- 滑点落地：买 close×(1+s)，卖 close×(1-s)，成本基准同步更新

### Task 4: 连续强度评分 ✅
- 原评分退化：AND 通过者恒 100 分（无区分度，仓位权重恒 20%）
- 新评分：score = 60 + 40 × 加权平均强度；强度 = 满足程度（尺度无关相对距离）
- 条件记录附带 strength 字段

### Task 5: 基准对比 ✅
- BacktestResult 新增 benchmark_return / benchmark_symbol
- 引擎每日跟踪基准指数收盘（默认 000300.SH，可配，无数据优雅降级）
- Markdown/HTML 报告显示基准收益与超额收益

### Task 6: AKShare 数据源加固 ✅
- 个股K线东财→新浪自动回退（当前网络东财被断连，新浪可用）
- 指数代码自动路由（INDEX_SYMBOLS），修复新浪 date 列类型问题
- 预加载纳入基准+市场状态指数（连带修复 progress.md 2026-06-04 记录的
  "000985.SH 无数据 → adx_signal=None" 问题——根因是指数未进缓存）

### Task 7: 清理 ✅
- strategies.yml 去重（institutional_accumulation/retail_hot 各重复4次 → 1次）
- 根目录杂物（mcp_out.txt/mcp_val.txt/空trading.db）→ ai-created/
- runner.py 去重（run/run_rolling 共用 _prepare），硬编码参数改常量+SystemConfig
- reporter 每日监控持仓浮动盈亏修复（原恒为占位符）
- 新增 .env.example；.gitignore 补充运行产物

### 测试
- tests/test_engine.py（13用例）：构造回归、期末估值、T+1、移动/时间/均线退出、冷却、滑点、基准、日历
- tests/test_scanner.py（5用例）：连续评分区间/区分度、AND/OR门、强度尺度无关
- 全量 42 passed；akshare 真实数据 E2E 通过（ai-created/e2e_test.py）

### E2E 结果（10只白马股，2024-01~2025-12，沪深300基准 +36.7%）
| 策略 | 收益 | 超额 | 夏普 | 交易 | 卖出原因分布 |
|------|------|------|------|------|--------------|
| pure_trend_follow | +12.85% | -23.9% | 0.73 | 235 | time_stop 71 / stop 21 / trail 13 / tp 10 |
| short_term_momentum | +0.63% | -36.1% | 0.10 | 33 | time_stop 10 / trail 4 |
| oversold_bounce | +3.97% | -32.8% | 0.68 | 14 | 均匀分布 |

**关键发现:** 暖机修复后 pure_trend_follow 从 +3.82% 提升到 +12.85%（前期信号正确性提升）。
牛市中紧止损（5%/8%/20天）导致所有策略跑输买入持有——与 9868.HK 结论一致。
下一步方向：按市场状态（ADX/市场环境分）自适应调节退出参数。

### Known Issues（下一步）
- 退出参数目前全局静态，未随市场状态自适应（牛市紧止损是主要回撤来源）
- 板块筛选依赖 Oracle 内网（10.1.33.123），当前环境不可达，akshare 无申万行业替代
- webapp 未与新的 benchmark/excess 字段联动

### Task 8: 每日选股入口 run_scan.py ✅（新增）
- 补齐"短线选股每天1-5只"的产品闭环（此前只有回测入口）
- 股票池：沪深300/中证500/中证1000 成分股（csindex接口）或全市场
- 输出：控制台表格 + results/daily_scan_*.md（含条件强度明细）
- --market-env 附带当日市场环境评分（综合分/仓位系数/六维信号）
- 实测：pure_trend_follow 在沪深300池扫出 49 只通过，连续评分 95-100 有区分度

## Session: 2026-09-24 - 全项目 Review：市场环境层修复（前视偏差+接线断点）

### Review 发现（按严重度）

**P0 前视偏差（回测结果不可信的根源）:**
- `scorer.score(date)` 中 `get_macro_score()`/`get_syzm()` 不传日期 → 回测任意历史日
  用的都是最新宏观数据。修复：评分全链路传入 date，时点过滤到 ≤ date
- `get_market_valuation` 的 `iloc[-1]` 取全历史最新 PE → 历史日估值=今日估值。
  修复：先过滤 ≤ end_dt 再取当日值
- `position_factor` 算出后从未作用于仓位（仅 AdaptiveStrategy 调 min_score 用到）。
  修复：引擎 `allocate(cash × position_factor)`，市场环境真正调节敞口

**P0 接线断点（功能存在但从未生效）:**
- 引擎构造 Scorer 不传 MacroDB → 宏观/领先两维恒 50 分。修复：provider 支持
  fetch_indicator 时自动装配 MacroDB
- 市场宽度 provider 恒返回 None → 宽度维恒中性。修复：引擎从缓存K线向量化计算
  涨跌家数（时点正确+零网络），注入 scorer
- 验证：2024-03-15 评分（宏观扩张55/先行扩张65/季节有利）vs 2026-09-20
  （宏观收缩35/先行收缩35/季节不利）——不同时点给出不同读数 ✓

**P1 性能:**
- `_AkShareFetcher` 无缓存，逐日评分会触发 天数×27 次网络请求。
  重构：全量序列缓存一次 + 内存时点切片；估值 csindex 同样处理。
  首日评分 0.7s，之后 0.0s

**P1 正确性杂项:**
- `get_logger` 全局单例 bug：首个调用者霸占名字，全系统日志记在同一模块名下。
  修复：按模块名缓存
- `_parse_akshare_date` YYYYMMDD 分支 month 取的是日期字符 → 日频指标解析崩。
  修复：s[4:6]
- MultiStrategy max 归一化缺陷：唯一弱信号也被放大到 100 分拿满仓。
  修复：改用绝对加权和（封顶100）
- scanner 读 DataFrame 指标取"最后一列"（dict 顺序依赖）。修复：注册表新增
  primary_column 元数据（macd→histogram, bb→percent_b, dmi→adx 等），行为不变但显式

**P1 安全:**
- Oracle 凭据明文硬编码于 6 个文件（system_config 默认值/run_full_backtest/
  run_9868/fund_flow_provider/futures_db/report_server/visual_report）。
  全部改为环境变量读取，.env.example 提供模板

**P2 工程清理:**
- webapp DB 增加 benchmark_return/benchmark_symbol 列（含存量库迁移），runner 传入
- run_backtest.py 暴露 --rolling/--rolling-window/--oos-ratio（run_rolling 此前无 CLI 入口）
- pyproject 清理未用重型依赖（TA-Lib/binance/ccxt/dash/numba/python-docx→dev 组）

### 测试
- 新增 tests/test_market_env.py（11 用例）：logger 按名缓存、时点评分、自动装配
  MacroDB、宽度注入/计算、MultiStrategy 绝对评分、fetcher 内存切片
- 全量 53 passed；真实数据 E2E：仓位系数生效后 pure_trend_follow +6.6%
  （回撤 -8.6%→-7.7%），弱环境降敞口符合设计意图

### Review 中发现但未动（需要用户决策/更大改动）
1. **季节性因子用美股数据**（S&P月度收益+美国总统周期）作用于 A 股系统——建议换
   A 股自身月度统计（可用中证全指历史从 akshare 拉取计算），或降低该因子权重
2. **DMI/ATR 用 rolling mean 而非标准 Wilder 平滑**（ewm alpha=1/period）——改为
   标准算法会改变所有 ADX 阈值信号，需重跑验证
3. **RPS 是时序自排名非横截面排名**——与欧奈尔 RPS 语义不同且 O(n²)，应移到
   scanner 层做全市场横截面
4. **Ichimoku chikou 列 shift(-26) 引用未来数据**——当前无策略使用，引入时须注意
5. **退出参数静态**——牛市紧止损是跑输主因（两轮 E2E 一致结论），建议按
   ADX/position_factor 自适应调节（trailing/时间止损随趋势状态放宽）
6. detect_divergence 注册签名与 registry.compute 不兼容（传入 data 会崩）——未使用，暂留

## Session: 2026-09-24 (下午) - 五项策略层迭代

### Task 1: 季节性因子A股化 ✅
- 移除美国总统周期与 S&P 月度表（对A股无因果基础）
- 用沪深300 2002-2026 共296个月度样本实证重算：**中位数**+胜率
  （均值被 2006/2014 极端年拉偏：12月均值+3.05% vs 中位数+1.22%）
- 实证规律：1月1.137/2月1.200/12月1.123 有利（春季躁动+年末行情），
  8月0.945最弱，其余中性
- 修复旧公式量纲 bug：胜率项误乘100导致因子退化为二值（>50%→1.2，<50%→0.8）
- refresh_monthly_stats() 可从 akshare 重算更新（验证：重算值与内置表一致）
- 旧 get_presidential_cycle_info 无引用，已删除

### Task 2: DMI/ATR 标准 Wilder 平滑 ✅
- DMI_ADX：atr/plus_dm/minus_dm/adx 平滑从 rolling mean 改为 ewm(alpha=1/period)
- ATR 指标与 risk/manager._get_atr 同步（口径一致，ATR止损使用同一算法）

### Task 3: 自适应退出参数 ✅
- 趋势市场（ADX>30）自动放宽：移动止损 ×1.6、时间止损天数 ×2
- BacktestConfig.adaptive_exits 开关（默认开）；常量在 BacktestParams
- 引擎存储当日 constraint，退出检查用有效参数

### Task 4: 横截面 RPS ✅
- 引擎每日从缓存价格矩阵向量化计算全市场 N 日收益百分位（BacktestParams.RPS_PERIOD=20）
- 注入 ScanConstraint.rps_rank；指数（基准/市场状态）不参与排名
- 新操作符 rps_rank_gt（欧奈尔语义：强于全市场 X% 的股票）
- 收盘价矩阵抽取为 _get_close_matrix（宽度/RPS 共用，一次构建）
- 新策略 trend_rps_follow（趋势+ADX+RPS>70）；旧时序自排名 RPS 标注弃用

### Task 5: Ichimoku 时点安全 ✅
- 移除 chikou_span=shift(-26)（引用未来数据），替换为 chikou_above
  （收盘价>26日前价格，多头确认，时点安全）
- 截断一致性测试验证无未来依赖

### 测试与验证
- 新增 tests/test_iteration5.py（18用例），全量 71 passed
- 真实数据 E2E（10只白马股 2024-2025，沪深300基准+36.7%）：
  | 策略 | 收益 | 夏普 | 回撤 | vs上一轮 |
  |------|------|------|------|----------|
  | pure_trend_follow | +10.07% | 0.76 | -6.8% | 夏普0.52→0.76，回撤-7.7%→-6.8%，收益+6.6%→+10.1% |
  | trend_rps_follow（新） | +9.65% | 0.70 | -7.2% | — |
  | short_term_momentum | +0.31% | 0.07 | -3.1% | 基本持平 |
  | oversold_bounce | +3.02% | 0.70 | -3.1% | 略降 |
- 结论：自适应退出+A股季节性+Wilder 对趋势类策略的风险调整后收益改善显著；
  绝对收益仍跑输牛市基准（仓位系数+止损的代价），符合预期

## Session: 2026-09-24 (晚) - Phase 1 回测诚实性 + 因子评价（方法论对齐）

### 背景
对标头部量化私募（幻方/九坤/明汯/衍复）的差距分析：不比超算，在方法论和工程上对齐。

### Task 1: 涨跌停不可成交 ✅
- 板块差异化限制：主板10% / 创业板(30x/301x)与科创板(688) 20% / 北交所 30% / 港股无限制
- 收盘距涨/跌停价 0.2% 以内视为封板：涨停日不买入、跌停日不卖出（延后退出）
- ST ±5% 需证券名称数据，按板块基数近似（保守）；enforce_price_limits 开关

### Task 2: 真实成本模型 ✅
- 印花税 0.05%（卖出单边）+ 过户费 0.01‰（双边），Trade 新增 taxes 字段
- 胜率计算与 DB 持久化盈亏均计入税费

### Task 3: 股票池无偏化 ✅
- akshare 无历史成分接口（PIT 指数成分不可得），改为：回测用全市场+上市时长过滤（近似PIT）
- E2E 从手选10只白马股（后见之明偏差）改为全市场代码序前12只（无偏）
- 回测报告自动标注幸存者偏差 caveat

### Task 4: 因子评价模块（research/factor_eval.py + run_factor_eval.py）✅
- IC/RankIC 序列及均值/IR/正向率；5分位分层收益与年化；多空年化；多头组换手率；分层单调性检验
- 时点正确：指标全历史一次计算（rolling/ewm 只依赖历史），按日期对齐截面
- CLI: python run_factor_eval.py --factor rsi --period 1 5 10 --universe 60

### 首个真实研究结论（RSI 因子评价，60只无偏池 2023-09~2026-09）
| 前瞻期 | RankIC | 多空年化 | 关键发现 |
|--------|--------|----------|----------|
| 1日 | -0.036 | — | 弱反转 |
| 5日 | -0.059 | -20.7% | Q5(超买) -0.09%/期，Q1-Q4 全正 |
| 10日 | -0.075 | -22.4% | Q5 -0.29%/期 vs Q3 +0.67%/期 |
**结论：A股短期反转效应显著强于动量（与学术共识一致）；换手率37%/期，成本敏感**

### 诚实化前后 E2E 对比（残酷但必要）
| 策略 | 白马池+无约束 | 无偏池+涨跌停+印花税 |
|------|--------------|---------------------|
| pure_trend_follow | +10.1% (夏普0.76) | **-8.4% (夏普-0.39)** |
| short_term_momentum | +0.3% | -4.2% (夏普-0.63) |
| oversold_bounce | +3.0% | **+4.6% (夏普0.98, 回撤-1%)** |
| trend_rps_follow | +9.7% | -3.9% |
**解读：白马池收益大半是选择偏差幻觉；趋势策略打深市低波动老股是风格错配；
oversold_bounce 与因子评价的反转发现内部一致——唯一经得起诚实检验的策略**

### 测试
- 新增 test_market_realism.py（7用例）+ test_factor_eval.py（5用例），全量 83 passed

## Session: 2026-09-24 (夜) - Wind 接口接入：PIT 数据升级

### Wind 连通
- 本机 Wind 终端 D:/program_files/Wind，官方 installWindPy.py 装入 .venv
- 接口规律（本机 WindPy 版本）：
  - 多代码+多字段 → -40522018 拒绝；多代码只能单字段
  - usedf=true 被忽略，返回经典结构：单代码 Data[field][date]，多代码 Data[code][date]
  - wsd 日期必须用标准 yyyy-MM-dd（相对日期如 -5days 报错）

### PIT 能力验证（核心成果）
- sectorconstituent + 历史日期 = 时点全A：**2024-01-02 全A 5335 只，其中 102 只现已退市**
  （akshare 全部缺失 → 幸存者偏差规模被直接量化）
- 退市股 K线可取（000004.SZ 2024全年242条）✓
- indexmembers PIT 参数 -40522012 失败（未解决，用全A池+RPS横截面替代——头部机构量化选股同样全市场选）

### 代码升级
- data/wind.py 重写：list_symbols(as_of) PIT / preload_batch 按字段聚合批量 / get_kline 单代码
- DataLoader.load_symbols(as_of) + preload_klines 自动走批量路径
- runner 回测股票池改为回测起始日 PIT 成分
- run_factor_eval.py --source wind：Wind PIT 池因子评价

### 幸存者偏差定量实验（RSI 因子，代码序前100只，2023-2026，5日前瞻）
| 指标 | akshare 幸存者池 | Wind PIT 池 | 差异 |
|------|-----------------|-------------|------|
| Q1(超卖)年化 | +17.3% | **-0.08%** | **~17pp 高估** |
| Q5(超买)年化 | -0.33% | -2.53% | |
| 多空年化 | -17.6% | -2.45% | |
**结论："超卖反弹"收益大半是幸存者偏差幻觉——akshare 只剩活下来的股票，
超卖后反弹的留下、继续暴跌退市的消失。PIT 数据下反转效应仍在（Q2-Q4最好）
但极端超卖组无超额收益 → oversold_bounce 策略结论需要重检**

### Task（续）：E2E 重跑 + 扫描器优化 + 批量因子筛选

**扫描器性能优化（selection/scanner.py）:**
- 新增 _history_cache：每只股票K线+指标全历史只算一次，逐日内存切片
  （rolling/ewm 均为因果算子，切片=时点正确；窗口 3×LOOKBACK 暖机 + 400天余量）
- E2E 耗时 35分钟 → 4分53秒（约7×）；Wind 批量预加载 102只仅6秒
- 全市场(5335只)预估仍需小时级 → Phase 3 需向量化扫描器（已记录）

**Wind PIT 池 E2E（100只含退市，2024-2025，基准+36.7%）:**
| 策略 | akshare幸存者池 | Wind PIT池 | 结论 |
|------|----------------|-----------|------|
| pure_trend_follow | -8.4% | **-29.2%** | 小盘老股+紧止损=灾难 |
| short_term_momentum | -4.2% | **+11.45%**（夏普0.64） | 唯一正收益 |
| oversold_bounce | +4.6% | **-4.0%**（胜率17%） | 无确认抄底被退市股摧毁 |
| trend_rps_follow | -3.9% | **-31.9%** | 同趋势类 |
**核心洞见：short_term_momentum 买超卖但要求 MACD 金叉右侧确认 → 避开继续崩塌的
退市股；与因子评价"Q2-Q4中间区最好、极端Q1无收益"完全自洽**

**批量因子筛选（17个指标，PIT池100只，5日前瞻）:**
- 动量/摆动类全线负 RankIC：mfi -0.048 > rsi -0.046 > cci -0.043 > bb -0.042 > roc -0.040
- 均线/通道类接近零：sma/ema/keltner/donchian ≈ 0
- cmf 多空年化 -24.8%（资金流反向指标最强）
- **结论：该池子5日维度是明确的反转市——动量类指标反向使用才可能有alpha**

### 测试：全量 83 passed（Wind 依赖终端不进单测，经真实E2E验证）

## Session: 2026-09-24 (Phase 3) - 组合构造升级：向量化组合回测器 + 四组对照实验

### 新增 portfolio/ 模块
- construction.py: 因子合成（日频横截面 rank 加权）、TopN 选股（含分数带过滤）、
  等权/波动率倒数加权
- backtester.py: TopN 组合回测器（周期调仓、差额交易、印花税/过户费/滑点、
  涨停不买/跌停顺延、逐日MTM、换手率统计、基准对比）——全向量化
- run_portfolio_backtest.py CLI；tests/test_portfolio.py 10用例；全量 93 passed
- 性能：500只×3年 全流程 28 秒（含 Wind 批量预加载与因子合成）

### 四组对照实验（同 PIT 池500只含退市，2023-09~2026-09，基准000300 +21.61%）
| 组合 | 收益 | vs 等权对照 | 成本占本金 | 换手/次 |
|------|------|------------|-----------|---------|
| A 极端反转 Top50（MFI/RSI/CCI反向） | -21.57% | **-35.9pp** | 8.0% | 57% |
| B 中间带反转 Top50（band 0.2-0.8） | -6.32% | -20.6pp | 12.1% | 77% |
| C 全池等权对照（Top500） | **+14.31%** | — | 1.8% | 7% |
| D 低波动 Top50（ATR/STD反向） | +2.92% | -11.4pp | 4.1% | 21% |

### 研究结论（诚实且重要）
1. **负 RankIC ≠ 多头alpha**：反转信号的预测力来自"高动量差"（空头腿），
   纯多头组合无法收割——这是市场中性策略需要对冲融券的根本原因（教科书级验证）
2. 极端超卖尾部在组合层面再次确认为"接飞刀"（A vs B/C）
3. **量价技术因子（日线）在该池/该期无多头选股alpha**：所有 Top50 集中策略
   跑输全池等权；分散化击败了每一种集中化
4. 含义：继续前进需要 (i) 基本面/另类因子（Wind wss 财务数据可得）
   (ii) 对冲后的中性形态（IF/IM期货）(iii) 全市场宽度（向量化已就绪）

### 下一步选项
- 基本面因子库：Wind wss 取 PE/PB/ROE/增速，构建价值/质量因子 → 组合
- 市场中性形态：股票多头 + IF/IM 对冲（空头腿收割反转信号）
- 全市场验证：5335只跑低频（月调仓）等权/因子组合

## Session: 2026-09-24 (基本面因子库)

### 探测结果
- 可用日频字段：pe_ttm / pb_lf / ps_ttm / dividendyield2 / mkt_cap_ard（批量wsd OK）
- ROE等报告期字段无数据（需公告日PIT工程，留待下一步）

### 新增
- data/fundamentals.py: WindFundamentalProvider（字段矩阵批量+缓存）、
  因子注册表 ep/bp/sp/dp/size（1/PE、1/PB、1/PS、股息率、log市值）
- FactorEvaluator.evaluate_matrix（矩阵型因子直评，与函数型共用核心）
- portfolio.construction.composite_from_matrices；组合CLI --fundamental 模式
- run_fundamental_eval.py CLI；全量 93+ 测试通过

### 基本面因子评价（PIT 500只，2023-09~2026-09）
| 因子 | 20日RankIC | 60日RankIC | 60日IR | 换手/期 |
|------|-----------|-----------|--------|---------|
| bp | +0.034 | **+0.048** | **+0.47** | 3% |
| size | **-0.055** | -0.039 | -0.15 | 2% |
| ep/sp | +0.02~0.03 | +0.03~0.05 | 0.25~0.30 | 3% |
| dp | +0.01 | +0.02 | 0.14 | 3% |
小市值组60日+3.26% vs 大市值+1.59%（小盘行情主导期）；高股息跑输（成长行情）

### 🏆 首个正向选股alpha：基本面组合 bp+size（月调仓Top100）
| 指标 | bp+size | 等权对照 | 基准300 |
|------|---------|---------|---------|
| 总收益 | **+22.62%** | +14.31% | +21.61% |
| vs 等权 | **+8.3pp** | — | — |
| 夏普 | 0.41 | 0.32 | — |
| 成本 | 1.91% | 1.82% | — |
| 回撤 | -34.1% | -26.5% | — |
此前所有量价集中策略均跑输等权；基本面月频tilt首次跑赢分散化本身

### 诚实 caveats
- 因子选择与组合检验同一样本（in-sample风险），需滚动样本外验证
- 回撤-34%深于对照（含2024初小盘危机），size因子是行情依赖的
- Wind日频估值字段有少量回填风险

### 下一步
- 滚动样本外验证（run_rolling 思路移植到组合回测器）
- 报告期财务因子（ROE/增速）+ 公告日PIT
- 全市场5335只验证宽度

## Session: 2026-09-24 (滚动样本外验证) — in-sample alpha 未存活

### 设计（无后视）
- 每年用过去3年数据选因子（60日RankIC按|IR|取前二，方向=符号），只交易下一年
- 2022-2026 五折滚动；同窗等权对照+沪深300基准；PIT池500只

### 结果
| 年 | 选中因子 | 策略 | 等权 | 基准 | vs等权 |
|----|---------|------|------|------|--------|
| 2022 | sp+,bp+ | -2.0% | -6.7% | -21.3% | +4.7pp |
| 2023 | bp+,size- | -7.0% | -6.0% | -11.7% | -1.0pp |
| 2024 | bp+,size- | +4.3% | +2.9% | +16.2% | +1.4pp |
| 2025 | bp+,sp+ | +8.6% | **+17.6%** | +21.2% | **-9.0pp** |
| 2026 | bp+,sp+ | -6.7% | -7.3% | -4.2% | +0.6pp |
| **累计** | | **-3.7%** | **-1.6%** | -6.3% | **-2.1pp** |

### 结论（诚实）
1. **in-sample 的 +8.3pp 是过拟合**：样本外累计 -2.1pp vs 等权，未存活
2. 驱动 in-sample 结果的 2025 年恰是样本外跑输最重的一年
3. bp 每年都被选中（最持久），但追 trailing IR 的 top-2 选择规则本身在震荡
4. 该池（代码序前500=深市老股）等权5年-1.6%，池本身质量也有限

### 幸存的认知
- 样本外纪律救了我们一次——这就是它存在的意义
- 因子评价(IR 0.47) ≠ 组合alpha：选择规则/调仓/池共同决定结果
- 下一步最有希望的三个方向（按信息含量）：
  a) 全市场5335只重跑（池的广度可能改变一切——小盘暴露与分散质量）
  b) 静态全因子等权合成（避免追trailing IR的规则震荡）
  c) 报告期财务因子+公告日PIT（ROE/增速，新的因子族）

### 新增：run_portfolio_walkforward.py（可复用的验证框架）

## Session: 2026-09-24 (全市场验证) — 因 Wind 流量配额阻塞

- 全市场脚本就绪（ai-created/fullmarket_test.py）：PIT 4685只@2022-01-01、
  静态全因子合成（先验方向）、Top100/Top500/等权对照/基准、年度分解
- 运行被阻塞：所有 wsd 调用返回 -40522017（含最小请求，会话重启无效）
  → 账户级量化API流量配额超限（当日累计 ~1500+ 批量调用）
- 待配额恢复（通常次日）后直接重跑即可，无需改代码
- 经验教训：全市场大规模取数应加流量预算意识（5335只×5字段=535调用/字段组），
  且结果应本地缓存复用，避免重复拉取

## Session: 2026-09-24 (本地缓存 + 三层数据源)

### 用量审计（配额超限复盘）
- 当日成功提取 ~3150万条数据点（~760次调用），根因：同池重复拉取8次
- 教训：数据必须本地持久化复用

### 三层数据源 data/layered.py（新）
- ① 本地parquet缓存（data_cache/，按symbol/field，增量append+缺口检测）
  ② Wind Oracle DB（无配额）：K线用不复权列 S_DQ_OPEN/CLOSE（与WindPy一致，
  避免复权接缝）；估值字段映射 ASHAREEODDERIVATIVEINDICATOR（PE_TTM/PB_NEW/
  PS_TTM/S_VAL_MV，数据到当日）；PIT池=当日有行情的股票（与wset计数一致5335只）
  ③ WindPy客户端只补DB缺口（配额消耗仅限差额）
- 三层用量统计 stats；DataLoader支持 DATA_SOURCE_TYPE=layered（.env已配置）
- 所有CLI已切换；.gitignore补.env

### 集成验证（配额锁定状态下全走DB层）
- PIT@2024-01-02 = 5335只（与wset一致）；退市股数据完整（000004到退市日）
- 600519 close=1251.24 与WindPy交叉一致；pe_ttm=19.2 一致
- 二次调用0.3s命中parquet；93测试通过
- 全市场实验（4685只）已通过DB层重新启动，不再受配额限制

## Session: 2026-09-24 (全市场验证完成) — 窄池之疑问裁决：是窄池

### 全市场静态全因子组合（PIT 4685只含退市，2022-2026，月调仓）
| 组合 | 总收益 | 年化 | 夏普 | 回撤 | 成本 |
|------|--------|------|------|------|------|
| 策略Top100 | **+76.7%** | +12.8% | **0.65** | -28.2% | 4.8% |
| 策略Top500 | +66.2% | +11.4% | 0.62 | **-22.3%** | 4.0% |
| 全池等权 | +3.4% | +0.7% | 0.15 | -40.2% | 2.6% |

年度：Top100 vs 等权 vs 基准300
2022: +3.8/-18.0/-21.3 | 2023: +18.5/-0.4/-11.7 | 2024: +8.9/-1.9/+16.2
2025: +27.3/+32.2/+21.2 | 2026YTD: -13.9/-4.3/-4.2

### 裁决与解读
1. **因子组合 4/5 年跑赢等权**，累计 +73pp，夏普 4 倍于等权，回撤还更浅
2. 之前 walk-forward 的失败主因是**窄池**（500只深市老股）：横截面因子需要宽度
   才能分散个股风险；全市场 Top100 才是真筛选
3. 2022 熊市 +3.8% vs 等权 -18%（价值+小盘的熊市防守是真样本外惊喜）
4. 2026YTD 风格切换三者皆亏、Top100 最差——行情依赖仍在，非圣杯
5. dp 因子因配额缺席（客户端only），优雅降级，不碍主结论

### 技术备注
- 分层源新增指数表支持（AIndexEODPrices）
- 待优化：尾部1天缺口触发DB全区间重取（应按缺口段精确查询）
- 真样本外验证方向：DB有20年历史，可测2017-2021

## Session: 2026-09-24 (研究结果浏览器 + 缺口精确补取)

- 分层源优化：DB 补取按头部/尾部缺口各一次精确批量查询（不再全区间重取）；
  修复缓存与请求区间零交集时的越界 bug（IndexError）
- webapp 新增研究结果浏览器（/results）：results/ 下因子评价/组合回测/样本外
  验证/每日选股等报告分类列表 + Markdown 表格渲染 + HTML 报告直链；
  路径穿越防护；原有回测记录功能不受影响
- 用法: cd webapp && python app.py → http://localhost:5000/results

## Session: 2026-09-24 (2017-2021 真样本外验证通过)

### 结果（PIT 3034只含退市，因子方向完全未见的数据）
| 组合 | 总收益 | 年化 | 夏普 | 回撤 |
|------|--------|------|------|------|
| 策略Top100 | **+28.6%** | +5.2% | 0.35 | -42.2% |
| 全池等权 | **-10.2%** | -2.2% | -0.05 | -45.1% |

年度（Top100/等权/基准300）: 2017: -5.2/-18.0/+20.6 | 2018: -27.8/-28.5/-26.3
2019: +16.4/+17.1/+38.0 | 2020: +12.1/+8.4/+25.5 | 2021: +19.0/+17.2/-6.2

### 结论（两段样本证据链闭合）
1. **相对等权的选股alpha在两个独立十年段均成立**：
   2017-2021 +38.8pp（5/5年不输等权）、2022-2026 +73pp（4/5年赢）→ 选股能力真实
2. **绝对收益高度依赖风格行情**：2017-2021 大盘蓝筹牛 → 年化5.2%大幅跑输沪深300(+45%累计)；
   2022-2026 小盘价值牛 → 年化12.8%大幅跑赢。裸因子组合是风格暴露工具，不是绝对收益工具
3. 2018 熊市 -27.8% 无防守（vs 2022 熊市 +3.8%）——防守不稳定
4. → 下一步方向明确：基准约束的真指增形态（行业/市值中性化，跟踪误差约束），
   把"相对等权的选股能力"转化为"相对基准的稳定超额"

### 技术
- 三层统计：cache 13万行 + DB 365万行 + client 0（精确缺口补取生效，零配额）
- 报告已入 results/，webapp /results 可见

## Session: 2026-09-24 (指数增强验证) — 假设被否证，风格暴露本质确认

### 结果（300成分内因子Top100，PIT成员，月调仓）
| 年代 | 策略 | 基准300 | 超额 | 跟踪误差 | IR |
|------|------|---------|------|---------|-----|
| 2017-2021 | +9.3% | +47.1% | **-38.8pp** | 10.9% | **-0.56** |
| 2022-2026 | +7.7% | -9.2% | **+16.1pp** | 10.3% | +0.34 |

### 裁决（本因子集的边界至此完全清楚）
1. **指增假设被否证**：IR 符号完全随风格行情切换（-0.56 / +0.34）
2. 本质确认：**"价值+小盘"组合是风格暴露（smart beta），不是市场无关 alpha**。
   之前"赢等权"很大程度上=小盘因子赢（等权即满仓小盘倾斜）；成分域限制
   无法保留小盘暴露后优势即消失
3. 真指增需要：风格/行业/市值中性化后的残差 alpha + 更宽因子面
   （报告期财务 ROE/增速、分析师预期、另类数据）——这是与头部机构
   剩余的方法论差距，也是下一步的真实方向

### 技术沉淀（本轮新增基础设施，全部可复用）
- PIT 指数成分查询（AINDEXMEMBERS 入场/剔除日期）
- 指增 KPI（超额/跟踪误差/信息比率）计算
- webapp /results 研究结果浏览器
- 三层数据源（缓存→DB→客户端）+ 精确缺口补取，零配额全流程
- 两段十年样本的标准验证框架（因子评价→组合→样本外→指增检验）

## Session: 2026-09-24 (研究仪表盘)

- research/artifacts.py: 因子/组合结果 → 结构化 JSON 工件（results/data/，含净值/基准/IC序列/分位收益）
- ai-created/dash_artifacts.py: 一键重建全部实验工件（16个：4段×3组合+4因子，全缓存零配额）
- webapp/dashboard.py + ECharts 模板：
  - /research 研究总览（时代×组合对比矩阵、因子表、基准对照）
  - /research/view/portfolio/* 净值vs基准曲线+年度柱状+KPI磁贴（深色机构风）
  - /research/view/factor/* IC时序柱图+分位收益
- 修复 webapp 启动（python -m webapp.app）与 dashboard 路径
- BP@60d RankIC 复核: enhanced_2017_2021 段 +0.1386（成分内价值因子显著）

## Session: 2026-09-24 (因子贡献消融) — 单因子皆弱，合成增益才是alpha来源

### 结果（Top100月调仓，vs 等权对照）
| 组合 | 2017-2021 (等权-10.2%) | 2022-2026 (等权+2.2%) |
|------|----------------------|----------------------|
| ep 单用 | -16.5% | -13.5% |
| bp 单用 | -6.3% | **-47.7%（灾难）** |
| sp 单用 | **+15.7%** | -16.9% |
| size 单用 | +9.2% | -21.5% |
| **4因子合成** | **+28.6%** | **+76.7%** |

### 归因结论（预注册预期被否证）
1. **预注册预期"size单因子大幅跑赢等权"被数据否证**——2022-2026 size单用-21.5%
2. **每个单因子单独使用都很弱甚至为负，但合成+76.7%**：
   alpha 不来自任何单一因子，来自**多因子rank合成的互补去噪**
   ——单因子Top100=单维度极端集中（价值陷阱/壳股尾部）；
   多因子合成=各维度都还不错的股票，避开所有单一维度的极端尾部
3. 合成增益 vs 最佳单因子：2017-2021 +12.9pp、2022-2026 >+90pp
   ——实证了机构用多因子模型而非单因子的原因
4. 修正指增失败的归因：不是"size暴露"单一原因，成分域限制破坏了
   多因子互补所需的选股宽度
（注：等权对照两轮差异1.2pp，源于矩阵切片细节，非确定性差异，待查）

### 仪表盘新增：因子贡献板块（ablation artifact 自动展示）

## Session: 2026-09-24 (Phase A 启动) — 过拟合防御 + 公告日PIT财务因子

### Task 1: 过拟合防御（research/overfitting.py）✅
- Deflated Sharpe Ratio（Bailey & López de Prado 完整公式：偏度/峰度校正 +
  零假设期望最大夏普 Euler-Mascheroni 近似）
- TrialRegistry 实验登记（JSON 持久化）——诚实计数策略尝试次数作为 n_trials，
  防"只报告最好那次"
- 用法：dsr = deflated_sharpe_ratio(returns, n_trials=registry.count)

### Task 2: 报告期财务因子 + 公告日PIT（data/fundamental_pit.py）✅
- 因子注册表：roe / np_growth(归母净利同比) / rev_growth(营收同比) / gmargin
- PIT 规则：ANN_DT 公告日可见、同报告期更正公告按公告序覆盖、
  同日多报告期取最新、区间前回看带入、报告期间前向填充
- DB验证：AShareFinancialIndicator 茅台ROE 10.6→38.4 累计值+公告日真实
- LayeredDataSource 新增 get_kline_calendar（缓存日历）

### 测试：test_phase_a.py 8用例（DSR噪声/真信号/登记持久化 + PIT公告时点/
更正覆盖/回看带入/无公告NaN），全量 101 passed
### 集成首测（进行中）：4个财务因子 @ 2000只全市场池 2022-2026

### Task 3 首测结果：财务因子 @ 2000只全市场池 2022-2026（RankIC@20d/60d）
| 因子 | 20d | 60d | 换手/期 |
|------|-----|-----|---------|
| roe | -0.009 | -0.026 | 2% |
| np_growth | +0.001 | -0.006 | 1% |
| rev_growth | +0.005 | -0.002 | 1% |
| gmargin | -0.006 | -0.011 | 0% |

**结论（诚实）**：质量/成长因子在 2022-2026 的 A股全市场、日频前瞻下
无多头预测力（ROE/毛利率甚至小幅为负——与该年代小盘题材风格一致）。
管道本身可信（同库的价值因子同法测出 IR 0.47；PIT 测试全过）。
**含义**：印证机构经验——单因子原始值少有 alpha，alpha 在中性化残差/
因子组合/择时里。下一步简化 Barra 的动机更强：市值中性化后质量因子
行为如何，正是要检验的问题。

## Session: 2026-09-24 (简化Barra V1) — 中性化后质量因子符号翻转，首个残差alpha

### research/risk_model.py（新）
- style_exposures: 市值/动量/波动率暴露（PIT因果，逐日zscore+3σ截尾）
- neutralize: 每日横截面回归取残差（合成验证：size相关0.622→-0.000，
  alpha成分保留99.1%）
- factor_returns: 日频因子收益估计（归因备用）
- V1不含行业（现有SW分类非PIT，引入时标注）

### 关键检验：ROE 中性化前后（2000只全市场，2022-2026，60日前瞻）
| 版本 | RankIC | IR | 多空年化 |
|------|--------|-----|---------|
| 原始ROE | -0.0258 | -0.14 | -8.2% |
| **市值/动量/波动率中性化后** | **+0.0335** | **+0.22** | **+6.5%** |

**符号翻转**：质量因子的失效确认为风格纠缠（高ROE=大盘=逆风）；
剔除风格后"真质量"有正向预测力——正是机构"alpha在中性化残差里"的实证。
至此有两个正交正向信号：bp（价值，IR 0.47）+ roe_neutral（残差质量，IR 0.22）

### 下一步（Phase A 收尾）
- roe_neutral 作为第五因子进合成 → 消融+滚动样本外复检
- cvxpy 优化器（风格暴露硬约束 + TE预算）
- 行业中性（需PIT行业分类工程）

## Session: 2026-09-24 (Phase A 收官) — 五因子复检否决 + cvxpy优化器

### 五因子合成复检（全市场池，Top100月调仓）
| 年代 | 4因子基线 | +中性化ROE残差 |
|------|----------|---------------|
| 2017-2021 | (见oos基线) | +22.7% |
| 2022-2026 | +76.7% | **+69.0%（-7.7pp）** |
**裁决：残差质量因子组合层面被否决**——尽管独立IC+0.034/IR0.22，
进组合后两段均稀释收益（-6~-8pp）。教训（重要）：因子级IC≠组合级增值，
组合消融是唯一裁判。质量因子不进合成（其IC疑似与本时代样本重叠）。

### portfolio/optimizer.py（cvxpy 凸优化器）✅
- 目标: max α'w - λ·w'Σw - τ·||w-w_prev||₁（风险厌恶+换手惩罚）
- 约束: 满仓/单票上限/风格暴露偏离预算（相对基准锚点）
- 协方差: 样本+对角收缩；ECOS→SCS→等权三级回退
- tests/test_optimizer.py 5用例全过（约束满足/alpha倾斜/风格预算/回退/收缩）

### Phase A 最终清单
✅ DSR+实验登记 ✅ 公告日PIT财务因子 ✅ 简化Barra中性化（残差alpha发现）
✅ cvxpy优化器 ✅ 五因子复检（否决，纪律生效）
全量测试 106 passed

## Session: 2026-09-24 (Phase B 启动) — 对冲形态

### portfolio/hedged.py（新）✅
- beta_hedge: 滚动β对冲（trailing 60日，时点正确）+ 贴水成本（默认年化6%）
- 合成验证: 1.3β组合对冲后市场相关<0.25、β收敛1.3、成本扣减正确（tests 2用例）
- V1简化声明: 期货以指数现货近似，保证金忽略；V2接CFutureEODPrices主力连续

### Phase B 实验（运行中）
两年代 × {等权, 优化器加权(风格预算0.3, 单票3%上限)} × {多头, β对冲中性}
- 优化器接入回测器 weight_func（含prev_weights换手惩罚状态）
- 中性KPI: 绝对收益/夏普/市场相关性（应≈0）
- 工件组 phaseB_*/phaseB中性_* 已注册仪表盘

## Session: 2026-09-24 (Phase B 完成) — 收益来源彻底分解，策略性质最终确认

### 8组实验结果（Top100月调仓，β对冲=trailing60日，贴水6%/年）
| 年代 | 构造 | 多头 | 中性（对冲后） |
|------|------|------|---------------|
| 2017-2021 | 等权 | +28.6% | **-31.3%**（夏普-0.46）|
| 2017-2021 | 优化器风格约束 | +3.4% | -45.0% |
| 2022-2026 | 等权 | +76.7% | **+25.0%**（夏普0.36）|
| 2022-2026 | 优化器风格约束 | +43.3% | +0.9% |
市场相关性全部≈0（对冲数学正确）

### 三个答案
1. **优化器+风格约束不能修复指增**：约束正确生效，但杀死了收益来源
   （等权 vs 优化器差 -25~-33pp）
2. **中性形态拆解了收益来源**：
   - 2022-2026: +76.7% = 残差+25pp + 风格β+52pp（小盘牛市双重奖励）
   - 2017-2021: +28.6% = 残差**-31pp** + β+60pp（蓝筹β大赚，残差为负！）
3. **最终结论（全项目研究终点）**：该多因子组合的收益≈2/3来自风格β暴露
   （小盘+价值），残差alpha仅一个年代为正——是smart beta，不是全天候alpha。
   与消融实验（单因子皆弱）、指增失败、五因子否决全部互相印证

### 产品含义
- 定位为smart beta（风格工具）：成立，但需向持有人明示风格暴露
- 定位为市场中性：残差不稳定，当前因子集不足以支撑
- 通往真中性：需要跨期稳定的残差因子（财务深度/预期/另类）——回到
  "更宽因子面"的结论，Phase B把"为什么"量化清楚了

### 测试：全量108 passed（新增test_hedged 2用例）

## Session: 2026-10-08 (深度因子族：质量+预期) — 寻找跨期稳定残差alpha

### 目标（Phase B 结论驱动）
找到中性化残差 IC 两年代同号且 |IR|>0.10 的因子——通往真中性的唯一缺口

### 新增
1. PIT_FINANCIAL_FACTORS 扩展4个深度质量因子：
   salecash(销售收现比,+)/asset_g(资产增速,-,asset growth异象)/
   ocf_to_rev(经营现金流/营收,+)/leverage(资产负债率,-)
2. data/consensus_pit.py: 分析师一致预期EPS修正因子
   - ASHARECONSENSUSDATA, EST_DT公告日PIT，目标=下一个年度报告期
   - 修正=20日变化率；EPS≤0期为NaN；覆盖约半数（偏大市值）
   - 合成验证精确（0.1/0.1364，窗口衰减语义正确）

### 运行中
- ai-created/deep_quality_screen.py: 4质量因子 × 两年代 × 原始/中性化 + 验收表
- 一致预期修正因子两年代评价（并行）

### 验收标准（预注册，防事后挑因子）
中性化残差 IC：2017-2021 与 2022-2026 同号，且两段 |IR|>0.10

### 深度质量因子筛选结果（方向校正后中性化残差IC，验收标准预注册）
| 因子 | 2017-21 | 2022-26 | 验收 |
|------|---------|---------|------|
| **leverage(低杠杆)** | +0.0151(IR0.18) | +0.0117(IR0.14) | **通过** |
| ocf_to_rev | +0.0524(IR0.50) | +0.0090(IR0.06) | 同号差一步 |
| salecash | -0.0047 | -0.0025 | 太弱 |
| asset_g | +0.0090 | -0.0204 | 异号否决 |
**leverage 是首个跨年代稳定残差因子**；ocf_to_rev 候补（一段强一段弱）
注：GBK控制台不能打✓✗（脚本汇总表崩了，数字完整）

### 一致预期修正因子验收（预注册标准）
| 年代 | 覆盖率 | 原始IC(IR) | 中性化IC(IR) |
|------|--------|-----------|--------------|
| 2017-2021 | 71% | +0.0165(+0.42) | **+0.0214(+0.34)** |
| 2022-2026 | 18% | -0.0817(-0.78) | **+0.0102(+0.18)** |
中性化两段同号、双IR>0.10 → **通过预注册验收**。
原始IC在2022-26大负（覆盖偏大盘+小盘行情），中性化后转正——
又一次证明"裸因子IC会被风格污染"。覆盖衰减（71%→18%）为运营警示。

### leverage 组合级消融（composite_v3，Top100月调仓）
| 年代 | 4因子基线 | +低杠杆 | 裁决 |
|------|----------|--------|------|
| 2017-2021 | +31.1% | **+31.9%** | ✓ 改善 |
| 2022-2026 | +83.9% | **+84.5%** | ✓ 改善 |
**leverage 通过消融，正式进入生产合成（6因子候选之一）**——
与roe_n（-6~-8pp稀释）对比：稳定小幅增量+零换手成本=合格拼图

### 6因子终审（NaN语义修复后的有效判定，Top100月调仓）
| 年代 | 5因子生产 | +预期修正 | 裁决 |
|------|----------|----------|------|
| 2017-2021 | +40.5% | +30.8% | **-9.7pp 稀释** |
| 2022-2026 | +65.4% | +74.6% | +9.2pp 改善 |
**预期修正因子组合级否决**（分裂判决：一段改善一段稀释）——与roe_n同类档案
（因子级验收通过、组合级不通过），第三次印证该模式。

注意：NaN语义修复同时改变了基线本身（5因子从+84.5%→+65.4%，
因覆盖扩大改变成分）——所有v4数字为新语义下的有效版本。
待办一致性项：4因子vs5因子(leverage)需在新语义下复检。

## 深度因子阶段总结（2026-10-08）
1. 生产合成定格5因子：ep+bp+sp+size+leverage（四族：估值/规模/质量）
2. 通过验收的残差拼图：leverage（组合级两段均改善）
3. 否决档案：roe_neutral/五因子+roe/crev——全部"因子级有效、组合级稀释"，
   该模式已是稳定经验规律（我们的组合alpha主要来自估值+规模的互补去噪，
   追加弱残差因子会稀释该机制）
4. 工程沉淀：稀疏因子NaN合成语义（重大bug修复）、验收标准预注册制度、
   一致预期PIT管道（consensus_pit）、深度质量因子注册
5. 下一步方向：新语义下4v5复检 → 6因子组合优化器版（cvxpy+风格约束）
   → 真中性重检（残差因子增加后2017-2021负残差是否修复）

## Session: 2026-10-08 (阶段复盘)

- docs/STAGE_REPORT.md 正式阶段报告（摘要/架构/9假设时间线含全部否决/
  收益分解/生产策略证据链/诚实边界清单/机构差距重估/路线图）
- 4v5一致性复检（composite_v5）后台运行中

## Session: 2026-10-08 (实盘准备 V1)

### live/ 模块（新）
- broker.py: Broker抽象 + PaperBroker（收盘±滑点撮合、印花税/过户费、
  涨跌停拒单、零股拒单）+ QMTBroker预留（券商环境前置条件已文档化）
- ledger.py: 持仓账本（JSON持久化：现金/持仓/成本/成交/净值序列，
  移动平均成本，防重复入金）
- signals.py: 生产信号（5因子Top30小资金版，PIT到当日）+ 调仓指令
  （20%缓冲带降换手——机构标准做法）
- run_live.py CLI: --init/--signal/--execute paper/--nav/--status

### 测试与验证
- tests/test_live.py 7用例（撮合成本/涨跌停拒单/往返盈亏精确到分/
  净值标记/防重入金/缓冲带换手控制），全量 115 passed
- 端到端 Paper 演示运行中（真实信号全市场合成）

### 待用户侧完成的实盘前置
1. 券商开通量化权限（资金门槛）+ miniQMT 安装
2. 程序化交易报备（券商合规）
3. 之后在 QMTBroker._submit_qmt 实现委托（接口已留位）

### 建议验证流程
Paper 模式跟跑 2-4 周（每日收盘后 run_live --signal --execute paper --nav）
→ 净值 vs 回测预期偏差 < 1% → 小资金实盘

### Paper 端到端首跑成功（2026-10-08）
- 修复 initialize 未落盘 bug（Paper演练第一天即抓到——流程价值实证）
- 真实信号全链路：全市场5因子合成 → Top30 → 30笔Paper成交（含费用/滑点）
  → 账本入账 → 净值 998,692（成本拖累 0.13%，符合预期：印花税+滑点+佣金≈0.14%单边）
- 账本含持仓成本（如 688279.SH 167股@199.51）
- 至此实盘准备代码侧全部就位，待QMT券商环境

## Session: 2026-10-08 (每日交易日志)

### live/journal.py + run_daily.py（新）
- 每日一份 journal/YYYY-MM-DD.md，六章节：
  盘前市场展望（基准/ADX/环境评分，昨收PIT）→ 盘前决策依据（信号/指令/
  新入选因子画像）→ 盘前风控检查 → 成交记录 → 持仓与归因（当日贡献
  top/bottom）→ 收盘点评（自动事实性观察+明日计划）
- CLI: --phase am|pm|full，--execute paper；am/pm 分时追加同一文件
- webapp /results 新增"每日交易日志"分类（journal/ 目录，git跟踪）
- tests/test_journal.py 3用例，全量 115 passed

### 每日操作流程（Paper 验证期）
  早盘:  python run_daily.py --phase am
  收盘后: python run_daily.py --phase pm --execute paper
  或收盘后一次: python run_daily.py --execute paper
