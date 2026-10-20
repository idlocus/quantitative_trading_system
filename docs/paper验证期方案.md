# Paper 验证期方案（预注册，2026-10-15 起）

> 目的：真策略 + 真行情 + 模拟成交，验证**系统与运维的执行保真**。
> 这不是收益验证——2-3 个月的收益噪声主导，无法证实或证伪策略
>（十年重放已回答收益问题：组合预期 4-6%/年）。
>
> **首日即兑现价值（2026-10-15）**：验证期第一次真实运行抓到两个
> 生产 bug——signals.py 质量过滤分支 NameError（依赖重接线误删定义，
> 404 项测试全绿未覆盖该真数据分支）与 run_daily 健康报告异常分支的
> 未定义 logger（触发时会掩盖原始错误）。修复后新增 pyflakes F821
> 未定义名门槛（tests/test_no_undefined_names.py），静态兜住
> "测试形状与 bug 形状不匹配"的盲区（评审根因 3.3）。

## 期间与基线

- **起算**：2026-10-15（本方案提交日）
- **最短期限**：至 2026-12-31 ——覆盖 11 月、12 月**两次股票月度调仓** +
  ETF sleeve **首次建仓**（11 月首个交易日）+ 跨假期周
- **基线**：股票 sleeve NAV 998,746（100 只持仓，10-14 建仓）；
  ETF sleeve 300,000（全现金）

## 运行机制（持续性保障）

| 组件 | 内容 | 时间 |
|---|---|---|
| `scripts/paper_day.py` | 股票全流程 + ETF + 心跳留痕 | 每交易日 14:45（计划任务） |
| `scripts/paper_watchdog.py` | 心跳缺失/失败 → webhook 告警 | 每交易日 16:00（计划任务） |
| 手动补跑 | 当日漏跑时：`python scripts/paper_day.py` | 任意时刻（收盘价撮合语义不变） |

计划任务注册（管理员 PowerShell/cmd 执行一次）：
```
schtasks /Create /TN "QuantPaperDay" /TR "D:\workspace\python_home\quant-project\.venv\Scripts\python.exe D:\workspace\python_home\quant-project\scripts\paper_day.py" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 14:45 /F
schtasks /Create /TN "QuantPaperWatchdog" /TR "D:\workspace\python_home\quant-project\.venv\Scripts\python.exe D:\workspace\python_home\quant-project\scripts\paper_watchdog.py" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 16:00 /F
```
（工作目录须为项目根：如需可在 /TR 内包 `cmd /c "cd /d D:\... && python ..."`）

## 通过判据（期末逐项核对，预注册）

| # | 判据 | 阈值 |
|---|---|---|
| 1 | **连续性** | ≥95% 交易日在 `paper_heartbeat.json` 留痕（计划内停机如长假除外，需在 journal 说明） |
| 2 | **执行保真** | paper 成交成本（佣金+滑点合计）与模型假设偏差 <0.1%（每笔可从账本 trades 复核） |
| 3 | **一致性抽样** | 人工抽 2-3 个交易日：当日真数据喂重放引擎，目标持仓与 paper 实际一致（release_gate 在合成数据上的保证在真实数据上抽样确认） |
| 4 | **运维卫生** | 无未处理告警；账本异常队列清零；月度对账模板可用 |
| 5 | **风控事件可解释** | 若触发止损/熔断/重入：每次事件后 journal 有完整留痕与归因 |

## 明确不作为判据的（防自欺）

- **期间收益率本身**——无论正负都不构成对策略的证实/证伪
- 短期相对基准偏离——2 个月内的偏离是噪声
- ETF sleeve 首月表现——它首个调仓周期只是建仓

## 检查节奏

- **每日（自动）**：paper_day + watchdog（无需人工）
- **每周一（10 分钟）**：`runtime/paper_heartbeat.json` 周完整性；
  `monitoring/webhook_dead.jsonl` 无死信；journal 浏览一遍
- **每月调仓日次日**：journal 的调仓明细 + 换手/成本 vs `compare_targets`
  式预期
- **期末**：按上表 5 项判据出结论

## 期末决策规则

- **5 项全过** → 进入 CANARY 小资金实盘（go_live.md 清单）
- **有不过项** → 定位修复后延长 2-4 周重验（不推倒重来——滚动状态延续）
- **期间发生重大改动**（策略/规则）→ 按 runbook 五关流程走，
  验证期顺延一个调仓周期
