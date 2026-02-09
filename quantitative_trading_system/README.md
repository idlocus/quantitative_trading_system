# 量化交易系统

## 项目简介

本项目是一个完整的量化交易系统，用于自动化分析市场数据、执行交易策略、管理风险并评估绩效。系统采用模块化设计，便于扩展和维护，支持多种数据源和交易策略。

## 核心功能

- **数据获取**：支持从多个数据源获取市场数据
- **数据处理**：清洗、转换和存储数据
- **策略管理**：实现和管理多种交易策略
- **回测系统**：评估策略在历史数据上的表现
- **交易执行**：对接交易所API，执行交易订单
- **风险管理**：监控和控制交易风险
- **绩效评估**：分析策略的表现和收益
- **可视化**：展示市场数据、策略表现和风险指标

## 项目结构

```
quantitative_trading_system/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── config/            # 配置管理
├── data/              # 数据模块
│   ├── acquisition/   # 数据获取
│   ├── processing/    # 数据处理
│   └── storage/       # 数据存储
├── strategy/          # 策略模块
├── backtesting/       # 回测模块
├── execution/         # 执行模块
├── risk/              # 风险管理模块
├── performance/       # 绩效评估模块
├── utils/             # 工具模块
├── visualization/     # 可视化模块
├── tests/             # 测试模块
└── main.py            # 主入口
```

## 技术栈

- **编程语言**：Python 3.8+
- **数据处理**：pandas, numpy, ta-lib
- **数据库**：SQLite/PostgreSQL
- **Web框架**：Flask/FastAPI（可选）
- **可视化**：matplotlib, plotly, dash
- **API对接**：requests, websocket-client
- **测试**：pytest

## 安装说明

1. 克隆项目
2. 安装依赖
3. 配置数据源和交易接口
4. 运行系统

## 使用方法

1. **配置系统**：修改 `config/` 目录下的配置文件
2. **添加策略**：在 `strategy/` 目录下实现自定义策略
3. **运行回测**：使用 `backtesting/` 模块测试策略
4. **实盘交易**：配置交易接口后运行 `main.py`

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 许可证

本项目采用 MIT 许可证。