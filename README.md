# 量化研究底座

当前版本 **0.2.1**。10-30 重置为无策略研究底座后，研究线已从股票扩展到
商品期货（2026-11-16 预注册扩围决定，见 [CTA_PREREG](docs/CTA_PREREG_2026-11-16.md)）。
**不执行真实委托**：所有入口只产生研究结论与每日信号建议，人工执行。

## 研究范围与策略状态

| 线 | 范围 | 当前状态 | 入口 |
|---|---|---|---|
| G1 | A股选股（每日决策） | 样本外验证通过（+0.52%/月，合格池内排名，双边换手 0.39x/月） | 研究闭环，无自动入口 |
| CTA Z10 | 商品期货 6 品种·活跃等权 | 连续行情代理回测 PASS（次日开盘按实际名义成交量计费，年化 +13.41%、夏普 0.863、最大回撤 -24.58%。真实合约换月尚待验证），每日信号运行中 | `scripts/run_cta_daily.py` |

原始研究约束（双15%红线等）见 [研究约束](docs/RESEARCH_MANDATE.md)；
商品期货线的预注册与判据见 [CTA_PREREG](docs/CTA_PREREG_2026-11-16.md)。

## 可运行入口

```powershell
# 每日 CTA 信号报告（收盘后运行，输出 runtime/cta_daily/日期.md）
.\.venv\Scripts\python.exe scripts\run_cta_daily.py

# 策略面板（G1/Z10 状态、K线、成交记录），启动后访问 http://localhost:5000
.\start.bat        # 停止用 .\stop.bat（按 PID 记录，不误杀他人）

# 底座管理命令（只读、离线）
.\.venv\Scripts\python.exe main.py status
.\.venv\Scripts\python.exe main.py data summary
.\.venv\Scripts\python.exe main.py archive --verify
```

## 运维边界（2026-11-19 审计补齐：入口按副作用分级）

"所有入口只产生研究结论与信号"指**研究/信号链路**；保留的基础设施模块中
仍有少量会写本地状态或尝试连接券商的入口，操作前须认清级别：

| 级别 | 入口 | 副作用与前置条件 |
|---|---|---|
| 离线只读 | `main.py status/data summary/archive --verify` | 读取本地缓存与归档 |
| 研究与行情 | `scripts/run_*`、CTA 日报、网页行情与刷新 | 依入口读取本地数据或连接外部行情服务，可能写入缓存、研究证据、日报、文档及试验登记 |
| 写本地状态 | 展示开关、交易控制开关、手工对账 | 写入本地设置、对账快照或账本。交易总闸的撤单行为见下行 |
| 连接券商 | `scripts/auto_reconcile.py`（默认路径）、`live/qmt_broker.py` | 连接 QMT 终端读取对账单/在途订单；需本机 QMT 客户端与 `QMT_PATH` 配置 |
| 可能撤单/写账 | `scripts/trading_switch.py halt`（配置了 QMT 时自动撤在途订单）、`auto_reconcile.py --fix`（QMT 来源已被硬拦截，仅 manual 来源可修正账本） | `halt` 是总闸：撤不掉会明确要求人工去券商 App 处理；`--fix` 会改账本留痕 |

当前 G1 与 Z10 日报均无自动下单入口；上述券商相关路径是保留待验收的基础设施，
不是已启用的功能。QMT 适配器的验收状态见 `live/qmt_broker.py` 文档与登记册。

## 代码边界

| 目录 | 职责 |
|---|---|
| `application/` | 命令编排、工作区路径、归档校验 |
| `data/` | 数据源、缓存读取、PIT、价格与单位契约 |
| `strategies/` | 策略接口与显式注册表，当前没有策略实现 |
| `portfolio/` | 显式目标权重到订单、内存回放、组合优化工具 |
| `trading/` | 交易规则、费用及通用执行假设，无因子或 TopN 默认值 |
| `live/` | 模拟撮合、账本、风控、对账；QMT 适配器仅保留待后续验收 |
| `research/`、`indicators/` | 统计验证与指标计算工具，不选择策略 |
| `archive/2026-10-30-before-reset/` | 包含上一轮修复的完整旧源码快照、SHA256 清单与恢复说明 |

数据仍在原有 `data_cache/`，`runtime/`、`results/`、`journal/`、真实账本和 SQLite 数据库也留在原路径。
旧的 `run_*.py`、Paper 启动器以及旧 `main.py daily/am/pm` 等命令明确拒绝执行，返回非零退出码。
Windows 计划任务定义未修改；即使仍调用旧入口，也不会再运行旧交易逻辑。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m release_gate
```

旧策略测试随旧源码保存在归档内，当前测试验证通用底座和停用边界。
历史策略收益、旧测试数量与新底座的测试数量不可混为一次策略验收。

研究全程（规则→数据→信号→组合→结论）从 [研究流水线总览](docs/RESEARCH_LOG.md) 一个入口查看。

详见 [系统架构](docs/SYSTEM.md)、[重构记录](docs/RESET_2026-10-30.md)、[归档恢复说明](archive/2026-10-30-before-reset/README.md)。
