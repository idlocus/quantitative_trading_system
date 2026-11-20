# 当前开发约定

本项目已重置为无内置策略的量化底座，以 README.md 和 docs/SYSTEM.md 为当前架构说明。
旧规则、策略及实验完整保存在 archive/2026-10-30-before-reset/source，不能当作当前默认实现。

- application 只做命令编排，不计算策略分数。
- data 不依赖策略或执行对象，仅允许复用 trading.rules 的纯规则；本地缓存命令只读、离线。
- strategies 只定义接口和显式注册；默认注册表为空，不自动导入插件。
- portfolio 接收显式目标权重，不包含因子、股票池和 TopN 默认值。
- trading/live 保留交易规则、模拟执行、风控、账本和对账，不配置默认策略。
- archived 源码不能被当前代码或测试导入；旧自动交易入口必须明确拒绝执行。
- data_cache、runtime、results、journal、真实账本与数据库不因重构删除、迁移或重置。
- 不读取或提交 .env，不运行真实委托，不安装新依赖。
- 新增代码必须有业务行为测试，提交前全量 pytest 和 release_gate 全部通过。

历史研究裁决和策略收益证据只作历史记录；后续方向讨论后再预注册新实验。
