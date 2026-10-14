# 运维手册（Runbook）—— 出问题时怎么办

> 个人资金管理，简洁实用。每个场景一个问题、一个动作。

## 紧急停止

```bash
# 激活（停止一切交易）
python scripts/kill_switch.py "原因"

# 解除
python scripts/kill_switch.py off
```

激活后所有交易停止（Paper/QMT 均受控），RiskGate 返回 halt。

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
