# Task Plan: 市场环境宏观框架 (market_env)

## Goal
构建完整的市场环境宏观评分框架，基于 akshare 免费数据源，实现 27 个宏观指标 + 四维评分（宽度/估值/宏观/领先），并集成到回测引擎。

## Current Phase
Phase 7

## Phases

### Phase 1: 需求确认与数据源探索
- [x] 确认 27 个宏观指标的数据来源（akshare 可用性测试）
- [x] 验证 akshare 宏观函数的稳定性和数据质量
- [x] 文档化 findings.md
- **Status:** complete

### Phase 2: 设计并实现 AkShareMarketEnvProvider
- [ ] 设计 IMarketEnvProvider 的 akshare 实现
- [ ] 实现市场宽度数据获取（涨跌家数 via AShareEODPrices）
- [ ] 实现估值分位获取（PE/PB 历史百分位）
- [ ] 实现 27 个宏观指标的 akshare 适配（16个可用+11个需替代）
- **Status:** pending

### Phase 3: 实现宏观数据库 (MacroDB with akshare)
- [ ] 将 27 个指标映射到 akshare 函数
- [ ] 实现 get_macro_score() — PMI/GDP/CPI 加权评分
- [ ] 实现 get_leading_score() — 领先指标扩散指数
- [ ] 实现 get_syzm() — 先行合成指数（社融+M2+PMI）
- [ ] 实现 get_current_values() — 当前指标快照
- **Status:** pending

### Phase 4: 集成 AkShareMarketEnvProvider 到 BacktestRunner
- [x] 在 BacktestRunner.run() 中实例化 AkShareMarketEnvProvider
- [x] 将 provider 传入 BacktestEngine
- [x] 验证 MarketEnvScorer 四维评分正确计算
- **Status:** complete

### Phase 5: 测试与验证
- [x] 验证 17 个可用宏观指标的获取
- [x] 验证四维评分的市场环境集成
- [x] 确认回测引擎中的 position_factor 生效
- **Status:** complete

### Phase 6: ATR 动态止损与季节性分析
- [x] 用 ATR 替代 BacktestRunner 的固定止损 (stop_loss=0.03)
- [x] 实现季节性因子计算（总统周期 + 月度规律）
- [x] 将季节性因子集成到 MarketEnvScorer 的四维评分中
- [ ] 尝试用 stock_zt_pool_em（涨跌停数据）估算市场宽度
- **Status:** complete

### Phase 7: DMI 策略集成
- [x] DMI+ADX 已在 indicators/trend_indicators.py 实现
- [x] 添加 dmi_golden_cross / dmi_death_cross 操作符到 scanner
- [x] 添加 trend_following_with_dmi / atr_trend_strategy 到 strategies.yml
- **Status:** complete

## Key Questions
1. akshare 宏观数据覆盖 27 个指标中的多少？
2. 哪些指标缺失或不可用？如何替代？
3. 数据更新频率是否满足回测需求（历史 + 实时）？
4. 如何确保 T+0 日的市场环境评分在回测中使用？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 使用 akshare 作为宏观数据源 | 免费、无需 API key、数据全面 |
| 保留 IMarketEnvProvider 接口 | 统一抽象，便于切换数据源 |
| 不使用 Wind MCP | 用户明确要求不用 Wind MCP |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
|       | 1       |            |