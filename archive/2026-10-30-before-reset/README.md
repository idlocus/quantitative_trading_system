# 重构前源码快照

`source/` 保存重构前的源码、策略、实验、测试和文档，包含上一轮未提交的 12 项修复。
`manifest.json` 记录快照时的 Git 基线、每个源码文件的 SHA256，以及原位保留数据的元信息。
这是历史存档，不是可自动加载的策略目录，不属于当前 pytest 收集范围。

## 主要内容

- 旧主链：`source/trading/`、`source/live/signals.py`、`source/portfolio/`。
- 旧策略与选股引擎：`source/strategy/`、`source/selection/`、`source/backtesting/`。
- 全部旧实验：`source/experiments/`。
- 历史研究裁决：`source/docs/研究日志/` 与原重置协议、架构文档。
- 旧应用和入口：`source/webapp/`、`source/main.py`、`source/run_*.py`、`source/scripts/`。
- 旧测试、依赖清单和锁文件，便于完整核对恢复。

不包含 .env、真实账户快照、原始数据或 SQLite 数据库；这些文件仍在原项目路径。
参考 PDF/图片等二进制文档也留在原 `docs/base/`，不重复复制。

## 校验和恢复

在当前项目根运行 `python main.py archive --verify` 可检查所有归档源码哈希及数据保留清单。

恢复旧源码时先另行备份当前活动源码，再按 manifest 的源码清单把 `source/` 文件恢复到原相对路径。
不得把恢复操作扩展成对 data_cache、runtime、results、journal、.env 或账本的覆盖/清空。
恢复后需在原项目根、原数据目录及匹配依赖环境下重新测试；不要直接在 `source/` 中运行旧入口，原脚本可能按相对路径创建新的状态文件。
重新启用旧策略或实盘是另一项显式决策，本轮归档不代表允许自动恢复运行。
