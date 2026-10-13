# 项目文档索引

> 快速导航：按使用场景找文档，而非按文件名。

## 你想做什么？

| 场景 | 文档 | 说明 |
|---|---|---|
| 理解系统架构 | [唯一主链架构.md](唯一主链架构.md) | **主文档**：主链定义、历次修复、收益结论 |
| 日常运行 | [SYSTEM.md](SYSTEM.md) | 系统总览、模块地图、CLI 命令 |
| 了解风控 | [风险分层与执行闸门.md](风险分层与执行闸门.md) | L2/L3/L4 分层、闸门配置 |
| 账实核对 | [账实核对机制.md](账实核对机制.md) | 对账流程、差异处理 |
| 数据库结构 | [wind_db_schema.md](wind_db_schema.md) | Wind Oracle 表结构、单位归一化 |
| 技术框架 | [TECHNICAL_FRAMEWORK.md](TECHNICAL_FRAMEWORK.md) | 技术选型、分层设计 |
| 阶段成果 | [STAGE_REPORT.md](STAGE_REPORT.md) | 各阶段验收记录 |
| 开发进度 | [PROGRESS_LOG.md](PROGRESS_LOG.md) | 开发日志 |

## 当前系统状态（2026-10-13）

- **Paper 全链路可用**：信号→闸门→撮合→账本→对账
- **QMT 代码就绪**，等待券商环境验证（首次实盘清单见 qmt_broker.py docstring）
- **策略收益结论**：可交易口径下因子增益≈0，策略无盈利证据（详见主链文档 §2.8）
- **关键数据修复**：复权口径断裂已修复（close=不复权/adjclose=前复权双轨）

## 主链代码地图

```
数据层  data/layered.py → data/wind.py → data/st_provider.py → data/turnover.py
          ↓
信号    live/signals.py (generate_production_signal — 唯一入口)
          ↓
订单    live/signals.py (build_orders — 缓冲带)
          ↓
风控    live/risk_gate.py (L2数据/L3熔断/L4止损)
          ↓
执行    live/broker.py (Paper) / live/qmt_broker.py (QMT)
          ↓
账本    live/ledger.py → live/reconcile.py (对账)
```

## 弃用代码（禁止主链引用）

`backtesting/` + `strategy/` + `selection/` 是旧的个股止损型策略引擎（研究遗留），
与生产因子链不共享语义。已加 DEPRECATED 标记，由 `tests/test_legacy_isolation.py` 隔离。
