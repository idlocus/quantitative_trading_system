# 运维手册（Runbook）—— 出问题时怎么办

> 个人资金管理，简洁实用。每个场景一个问题、一个动作。

## 策略/框架变更上线流程（实盘在跑时改任何东西都走这里）

实盘环境=滚动状态（账本/持仓/净值历史），不是可替换的软件包。
代码与策略随时可换，状态永远延续——变更通过**下一个交易日的正常决策**
生效，存量持仓由 diff 订单渐进迁移（缓冲带控制换手），无"清仓重来"。

**五关 checklist（任何策略/参数/规则改动上实盘前）：**

```
1. 验证关（离线）：R6 全套
   □ mainchain/十年重放工件重新生成，结果可解释
   □ pytest -m release_gate 通过（生产=重放一致性）
   □ 全量 pytest 通过
   □ docs 更新（口径变更留档）
2. 影子关（1-2 周，可与新信号的研究验证并行）：
   □ 版本A下: python scripts/compare_targets.py --save A
   □ 版本B下: python scripts/compare_targets.py --save B
   □ python scripts/compare_targets.py --compare A B
   □ diff 可解释（预期换手/迁移成本）；paper 并行跑新版对照
3. 灰度关：CLOSE_ONLY 部署（python scripts/trading_switch.py close）
   □ 部署新代码；旧目标自然退出（只卖不买一个调仓周期）
   □ 确认无异常 → python scripts/trading_switch.py normal
4. 监控关：每日实盘 vs paper vs 回测三线偏差；周报对比
5. 回退路：config/代码在 git——revert 即回滚未来决策；
   已成交的迁移交易不可撤销（沉没成本）——所以第 1、2 关在前
```

**状态文件迁移**：`risk_state.json` / 账本带 `schema_version`，语义变更
加载时自动迁移并留痕（`live/risk_gate.py::_migrate_state`）。升级后首日
检查日志无迁移告警以外的异常。

**最坏情况**：`HALT_ALL` → 人工审查 → 按下方"硬限清仓与重入"一节处理。

## 紧急停止

```bash
# 激活（停止一切交易）
python scripts/kill_switch.py "原因"

# 解除
python scripts/kill_switch.py off
```

激活后所有交易停止（Paper/QMT 均受控），RiskGate 返回 halt。
状态读取 fail-safe：`runtime/switch_state.json` 损坏时按 HALT_ALL 处理
（读不出状态=停；2026-10-15 定稿）。

## 硬限清仓与重入（2026-10-15 定稿）

**触发**：组合回撤（自全程峰值）≥ 15% → 硬限清仓。清仓完成后
**立即以现金为新锚满仓重入**（十年实证：慢重入被数据否决——
-1.5%/-27.6% vs 立即重入 +59.0%/-24.9%），同时滚动计数：
365 日历日内 >3 次硬限清仓 → **永久停机**（防 2008 式连续崩盘循环）。
参数见 `trading/config.py`（REENTRY_MAX_LIQUIDATIONS / REENTRY_CIRCUIT_WINDOW_DAYS）。

**日常无需干预**：单次清仓后系统自动重入。协议事件在风控理由留痕
（日报/日志可见 `[L3] 硬限清仓完成…立即重入` / `[L3] 重入熔断` 字样）。

**需要人工操作的情形**：

1. **重入熔断永久停机**（`[L2.7] 重入协议永久停机`）——一年内 4 次
   硬限清仓，市场持续恶化或策略失效。处置：
   ```bash
   # 1) 先诊断：日报回撤归因、因子覆盖率、是否系统性下跌
   # 2) 理解成因后决定重启（不要无脑重置）：
   #    清除 runtime/risk_state.json 中 protocol_halted / liq_dates 字段
   #    （回撤锚以当前现金为新起点）
   ```
2. **想临时加严**——用三态开关（`scripts/trading_switch.py`），
   不要手改协议状态。


## 常见故障处理

### 1. 数据源故障（Oracle 不可达）
- **症状**：日志 `[layered] DB K线批量失败` 或 `DB 不可用`
- **动作**：系统自动降级到缓存+Wind客户端，不影响交易
- **升级**：如缓存也过期（>5天），闸门自动 halt

### 2. 因子覆盖率崩塌
- **症状**：日志 `[signal] 因子覆盖率崩塌` 或 `信号质量闸门`
- **动作**：系统自动 halt 保留持仓。检查 `data_cache/fundamental/` 更新时间
- **修复**：等数据供应商更新或手动触发更新

### 3. 账本与券商不一致
- **症状**：`python run_live.py --statement` 显示差异
- **动作**：
  ```bash
  # 查看差异
  python run_live.py --statement
  # 差异小（费用差）：自动修正
  python run_live.py --statement --fix
  # 差异大（持仓差）：人工核查后再修正
  python scripts/kill_switch.py "账实不一致"  # 如需先停交易
  ```

### 4. 策略异常行为（如突然大量交易）
- **动作**：
  ```bash
  # 1. 立即停止
  python scripts/kill_switch.py "策略异常"
  # 2. 查看日志
  cat journal/$(date +%Y-%m-%d).md
  # 3. 检查最近变更
  git log --oneline -5
  # 4. 如需回滚
  git checkout <stable-commit>
  ```

### 5. QMT 实盘异常
- **动作**：同上立即停止。QMT 已知有未修复的阻断项（见 CLAUDE.md），不建议实盘。

## 每日运维清单

```bash
# 收盘后
python run_daily.py --phase full --execute paper   # Paper 交易
python run_live.py --nav                            # 更新净值
python run_live.py --status                        # 检查状态

# 对账（如有券商对账单）
python run_live.py --statement

# 每周检查
python scripts/preflight_check.py                   # 实盘前检查
pytest -m release_gate                              # 发布门槛
```

## 异常事件记录

重要事件记录在两个地方：
- `live/ledger.json` 的 `exceptions` 列表（超额卖出等，需人工 resolve_exception）
- `journal/YYYY-MM-DD.md`（每日交易日志，含风控闸门决策）

## 版本回滚

```bash
git log --oneline -10                  # 查看最近版本
git checkout <commit-hash>            # 回滚到指定版本
pytest tests                           # 验证回滚后测试通过
python scripts/preflight_check.py      # 验证配置正确
```

## 数据供应商故障切换

当前三层：缓存 → Oracle DB → Wind 客户端。故障自动降级。
如需手动切换：
- 缓存故障：删除 `data_cache/kline/*.parquet` 强制重新拉取
- DB 故障：自动跳过，用缓存+客户端
- 客户端故障：检查 WindPy 终端是否运行
