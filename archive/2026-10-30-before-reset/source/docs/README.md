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
| 本轮修复与验收边界 | [REVIEW_REPAIRS_2026-10-30.md](REVIEW_REPAIRS_2026-10-30.md) | 12 项问题、版本变化、旧结果适用边界 |

## 当前系统状态（2026-10-30）

- 工程：修复本轮审查确认的数据、风控、执行、回放和测试问题，策略规格升级至 **3.0.0**。
- 研究：遵循 [研究重置协议](研究重置协议.md)。旧版本与当前版本的收益结论不能直接沿用；新版本尚未完成历史收益验收。
- 运行：本次没有部署、操作真实账户或修正历史账本；QMT 仍需券商环境验收。
- 数据：`close` 为不复权价；Wind DB 的 `adjclose` 为后复权价，用于收益计算。
- 下一阶段方向尚待讨论，本轮修复不表示采纳新策略或开启实盘。

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
