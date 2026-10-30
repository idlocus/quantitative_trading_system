# 灰度上线流程（Paper → 小额实盘 → 正常）

> 前提：仅管理自有资金。每一步通过后再进下一步，不可跳级。

## 阶段一：Paper 验证（当前）

**持续时间**：至少 2 个月（覆盖不同市场状态）

**通过条件**：
- [ ] 每日信号正常生成（覆盖率>80%）
- [ ] 换手率符合预期（月度调仓，日均<3%）
- [ ] 实际成本与回测假设偏差<50%
- [ ] 止损触发频率合理（每月<20笔）
- [ ] 净值曲线与回测方向一致（不要求完全一致）
- [ ] 无未处理异常
- [ ] `python scripts/preflight_check.py` 全部通过

**退出条件**：如果Paper期间出现不可解释的偏差→停下排查，不进入下一阶段。

## 阶段二：小额实盘灰度

**资金**：总资金的 10-20%（如总100万→10-20万）

**前置**：
- [x] QMT 订单生命周期重写（2026-10-15：①客户端引用↔真实订单号映射；
      ②成交→入账原子化（R4）；③台账原子写；④8参数官方签名；⑤查询-1阻断）
- [x] QMT 后续阻断项修复（2026-10-30 审计二项：
      ⑥ booked=-1 污染——查询失败哨兵不再写入台账基数（此前恢复后
        delta=traded+1 多入账 1 股+）；测试 test_query_failure_does_not_poison_booked；
      ⑦ 对账 --fix 对 QMT 硬拦截——账本键 id:{oid}#{qty} vs 对账单键
        id:{order_id} 命名空间分裂，自动补记=BUY双记；ID 规范化统一
        并验证匹配率前拒绝修正（RuntimeError），人工核对不受影响；
        测试 TestQmtFixHardBlock。
      验证：tests/test_qmt_lifecycle_v2.py 10 项（已纳入 release_gate）+
      既有生命周期测试。
      **首实盘日仍需人工验证引用映射与成交回报日志**——模拟无法替代真实券商回报）
- [ ] 对账 ID 规范化统一（B8 收尾：统一账本/对账单成交键格式并验证
      匹配率，之后解除 --fix 对 QMT 的硬拦截）
- [ ] 完成券商程序化交易报备
- [ ] `python scripts/preflight_check.py` 通过
- [ ] `pytest -m release_gate` 通过
- [ ] Kill Switch 测试通过（激活→确认停止→解除）

**运行**：
- Paper 继续跑（作为对照）
- QMT 每日运行（收盘竞价窗口内）
- 每日对账必须成功
- 每周对比 Paper vs 实盘收益差

**退出条件**：
- 实盘 vs Paper 偏差 > 2%（成本/滑点/执行差异）→ 停下分析
- 任何对账不一致 → 立即停止，人工核查

**持续时间**：至少 1 个月

## 阶段三：逐步放大

- 10-20% → 50% → 100%
- 每次放大间隔至少 2 周
- 每次放大前重新运行 preflight check

## 随时可做的事

```bash
# 一键停止
python scripts/kill_switch.py "原因"

# 检查状态
python run_live.py --status

# 实盘前检查
python scripts/preflight_check.py

# 发布门槛
pytest -m release_gate
```
