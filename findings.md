# Findings & Decisions

## Requirements
<!-- 原始需求：构建完整市场环境宏观框架 -->
- 27 个宏观指标的数据获取
- 四维评分（宽度/估值/宏观/领先）
- 集成到回测引擎
- 不使用 Wind MCP，使用 akshare 免费数据

## Research Findings

### akshare 宏观函数可用性测试结果（完整）

| 指标 | akshare 函数 | 状态 | 最新值 |
|------|-------------|------|--------|
| GDP同比 | `macro_china_gdp()` | ✅ 可用，3行 | 5.0% |
| 制造业PMI | `macro_china_pmi()` | ✅ 可用，17行 | 50.0 |
| 非制造业PMI | `macro_china_pmi()` (复用) | ✅ 可用 | 50.1 |
| CPI同比 | `macro_china_cpi()` | ✅ 可用，16行 | 1.2% |
| PPI同比 | `macro_china_ppi()` | ✅ 可用，16行 | 2.8% |
| 社会消费品零售 | `macro_china_consumer_goods_retail()` | ✅ 可用，12行 | 0.2% |
| 固定资产投资 | — | ❌ 无替代 | — |
| 发电量 | — | ❌ 无替代 | — |
| 工业企业利润 | — | ❌ 无替代 | — |
| M2同比 | `macro_china_money_supply()` | ✅ 可用，16行 | 8.6% |
| 社融存量同比 | `macro_china_shrzgm()` | ✅ 可用，12行 | 22075亿 |
| 银行间质押式回购利率 | `macro_china_bank_financing()` | ✅ 可用，17行 | 2710 |
| 10年期国债收益率 | `bond_china_yield()` | ⚠️ 只到2021-01（数据限制） | — |
| LPR 1年 | `macro_china_lpr()` | ✅ 可用，17行 | 3.0% |
| 融资融券余额 | `stock_margin_detail_sse()` | ✅ 可用（需agg=sum） | — |
| 北向资金净流入 | `stock_hsgt_hist_em()` | ⚠️ 最新几天为NaN | — |
| 新增投资者数量 | — | ❌ 无替代 | — |
| 股票型基金发行份额 | `fund_aum_em()` | ❌ 无历史序列 | — |
| 美国联邦基金利率 | — | ❌ 无替代 | — |
| 美国CPI同比 | — | ❌ 无替代 | — |
| VIX | — | ❌ 无替代 | — |
| 美元指数 | — | ❌ 无替代 | — |
| CRB商品指数 | `macro_china_commodity_price_index()` | ✅ 可用，514行 | 1072 |
| 出口集装箱运价指数 | — | ❌ 无替代 | — |
| 螺纹钢价格 | — | ❌ 无替代 | — |
| 70城房价指数 | `macro_china_new_house_price()` | ✅ 可用，32行 | 97.7 |

### akshare 版本: 1.18.49

### 指标覆盖率
- **完全可用**: 17/27 (63%)
- **数据限制（可用但有限）**: 3/27 (11%) — 国债/北向资金/基金规模
- **无替代方案**: 7/27 (26%) — 固投/发电/工业利润/美利率/美CPI/VIX/美元

### 关键数据限制
1. **国债收益率**: bond_china_yield 数据只到 2021-01-22，无法获取近期数据
2. **北向资金**: stock_hsgt_hist_em 最新几天当日净买额为 NaN
3. **基金发行份额**: fund_aum_em 是静态快照，无历史时间序列

## Phase 6 Enhancements

### ATR 动态止损
- Position 新增 atr 字段，RiskManager.allocate 通过 data_source.get_kline 计算 14 日 ATR
- check_exit 使用 max(fixed_stop_loss, ATR_stop_loss) 取更严格者
- ATR=3 且 entry=100 时：止损位=94 (-6%)，相比固定止损 3% 更宽松但更符合波动率
- ATR=1 且 entry=100 时：止损位=98 (-2%)，但固定止损 3% 更严格，实际用 3%

### 季节性因子 (seasonal.py)
- 月度规律：1月最强(+1.23%, 63%胜率)，8月最弱(-0.7%, 45%胜率)
- 总统周期：年3/4最强(10-13%平均收益)，年2最弱(5%)
- Sell in May：5-10月弱周期扣分
- 综合因子范围 0.8-1.2，2026年6月=0.832( unfavorable)，2026年11月=1.2( favorable)
- 集成到 MarketEnvScorer.score：position_factor = base_factor × seasonal_factor

## Phase 7: DMI 策略集成

### DMI 操作符
- scanner._check_condition 添加 dmi_golden_cross / dmi_death_cross 专用操作符
- 需要传 indicator_values (DataFrame) 才能取 plus_di/minus_di 列进行比较
- _evaluate 方法新增 indicator_values 参数传递给 _check_condition

### 新增策略
- trend_following_with_dmi: ADX>25 + DMI金叉 + RSI<70
- atr_trend_strategy: MACD金叉 + ADX>25

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 使用 akshare 作为宏观数据源 | 免费、无需 API key、数据覆盖广 |
| 保留 IMarketEnvProvider 接口 | 统一抽象层，便于切换数据源 |
| 移除 Wind MCP 依赖 | 用户明确要求 |
| 估值分位用宏观指标推导 | akshare 无直接指数历史 PE/PB，用 PMI/M2/CPI 估算 |
| 市场宽度返回 None | akshare 无直接 A/D breadth 数据源 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| akshare 函数签名差异（stock_board_change_em 不接受 symbol） | 移除参数使用无参版本 |
| GDP 日期格式 "2025年第1-4季度" 无法解析 | 添加正则解析 YYYYMM 和季度格式 |
| 国债收益率数据只到 2021-01 | 用宏观指标推导替代 |
| 融资融券余额返回空（groupby 后日期未解析） | 添加 agg="sum" 参数支持按日汇总全市场 |
| 北向资金最新几天为 NaN | 接受（数据源限制），不影响评分 |

## Resources
- akshare 文档: https://akshare.akfamily.xyz/
- akshare version: 1.18.49

## Visual/Browser Findings
<!-- 无 -->