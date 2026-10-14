# 开发约定（必须遵守，违反 = 分叉缺陷）

以下规则由测试强制执行（`pytest -m release_gate`），不是建议。

## R1 策略只有一个实现

因子、质量过滤、动量、选股、权重由同一套函数生成。
- 生产 `live/signals.py` 和重放 `run_daily_replay.py` 只传不同日期和数据，不维护各自的因子列表
- 改策略 = 改 `trading/config.py` + 共享函数，两侧自动生效
- **禁止**在任何文件中硬编码因子列表或阈值（如 `['ep','bp','sp']` 或 `0.08`）
- 一致性测试：`test_release_gates.py::TestConsistencyGate`

## R2 数据必须带口径

每个因子明确单位、可见日期、覆盖率；价格明确 raw/adjusted。
- 新因子必须在 `data/contracts.py::FACTOR_CONTRACTS` 注册（unit + range + coverage）
- 价格访问必须通过 `data/semantics.py::get_price_column(df, basis=...)`
- 成交额计算必须通过 `data/semantics.py::daily_turnover()` 或 `turnover_matrix()`
- **禁止**在业务代码中直接 `df['close'] * df['volume']` 或 `df['close']` 不声明基准
- 契约测试：`test_contracts.py`

## R3 无法确认就停止新交易

以下情况统一返回 HALT/NO_BUY，不能只打 warning 后继续：
- 任一必需因子覆盖率 < 30%
- 目标持仓数 < min(20, top_n)
- 有持仓但净值基线异常 (peak_nav ≤ 0)
- 上次对账失败或距上次对账 > 3 天
- QMT 在途订单状态查询失败
- ST 数据不可用

## R4 订单和账本按状态流转

请求序号(seq) ≠ 真实订单号 ≠ 成交编号(trade_id)，分开存储。
- 成交入账成功后才清退在途状态
- 恢复流程(sync_pending)与正常流程(_book_order_fill)共用入账函数
- 超额卖出阻止入账并记异常队列，不产生负持仓

## R5 测试业务结果，不测试代码形状

- 同一数据下断言生产与重放目标一致（不是断言"调用了某函数"）
- 覆盖缺因子、跨日成交、只有现金差异的 --fix 等故障场景
- E2E 安全网（`test_e2e_pipeline.py`）验证系统不变量

## R6 策略变更必须重新验收

改因子/阈值/口径/成交规则时，必须同时更新：
- [ ] `experiments/mainchain_v5.py` 重放结果
- [ ] 一致性测试通过（生产 = 重放）
- [ ] 日报说明更新
- [ ] `trading/config.py` 注释中的验证数据更新

## 发布门槛

三个门槛全部通过才能合并到 master：

```bash
pytest -m release_gate    # 一致性 + 安全 + QMT集成
pytest                    # 全量测试
```

## 文件输出位置

- 正式文档 → `docs/`
- 中间文件（临时脚本/调试/草稿）→ `ai-created/`
- 运行时状态 → `runtime/`（gitignored）
