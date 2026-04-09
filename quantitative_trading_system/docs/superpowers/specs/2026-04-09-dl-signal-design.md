# 深度学习信号模块设计方案

## 1. 目标

在现有技术分析框架基础上，集成 LSTM 深度学习模型，预测 A 股次日涨跌（涨/跌/震荡三分类），输出作为信号增强层的输入，与现有技术指标信号、市场情绪信号进行融合。

## 2. 架构设计

### 2.1 整体架构

```
数据输入层
├── 价格/行情数据 (OHLCV, 60日历史窗口)
├── 技术指标 (RSI, MACD, ADX, Bollinger Bands, ATR...)
├── 基本面因子 (PE, PB, 营收增速, 净利润增速...)
├── 市场宽度 (A/D Line, McClellan Osc...)
├── 资金流因子 (OBV, CMF, MFI...)
└── 新闻情感 (NewsSignalGenerator 输出)

         ↓
    LSTM 特征提取
    (60日 × 特征维度)
         ↓
    二层 LSTM + Dropout
         ↓
    全连接层 → Softmax (涨/跌/震荡)
         ↓
    预测结果 + 置信度
         ↓
    信号融合层
    (与现有 TechnicalFramework 信号加权组合)
         ↓
    最终交易信号 + 仓位建议
```

### 2.2 模型规格

| 参数 | 值 |
|------|-----|
| 输入序列长度 | 60 个交易日 |
| 分类数 | 3 (涨/跌/震荡) |
| LSTM 层数 | 2 |
| 隐藏层维度 | 128 |
| Dropout | 0.3 |
| 优化器 | Adam (lr=0.001) |
| 损失函数 | CrossEntropyLoss |
| 类别权重 | 震荡类降权（市场大部分时候震荡） |

### 2.3 数据规格

| 数据类型 | 来源 | 特征数 |
|---------|------|--------|
| OHLCV 价格 | Wind/AKShare | 5 × 60 = 300 |
| 技术指标 | TechnicalFramework | ~20 |
| 基本面因子 | Wind/AKShare | ~15 |
| 市场宽度 | breadth_indicators | ~5 |
| 资金流 | volume_indicators | ~5 |
| 新闻情感 | NewsSignalGenerator | 3 |

**总特征维度**: ~358

## 3. 模块结构

```
quantitative_trading_system/
├── ml/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # Dataset 定义，数据对齐与填充
│   │   └── preprocessor.py   # 数据预处理，标准化
│   ├── models/
│   │   ├── __init__.py
│   │   └── lstm_predictor.py # LSTM 模型定义
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # 训练循环，早停，保存最优
│   │   └── evaluator.py       # 评估指标：准确率/召回率/F1
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py       # 推理接口，返回预测 + 置信度
│   └── signal_fusion/
│       ├── __init__.py
│       └── fusion.py          # 与现有信号系统融合
├── signals/
│   └── models.py              # Signal 数据类（已有，扩展DL信号类型）
└── jobs/
    └── scheduler_app.py       # 现有调度器（新增每日模型推理任务）
```

## 4. 信号融合设计

### 4.1 融合流程

```
DL_Model_Output:
  - prediction: 涨/跌/震荡 (0, 1, 2)
  - confidence: 0~1 (softmax 最大值)
  - proba: [p_down, p_flat, p_up] 三类概率

Technical_Score (0~100, 来自 TechnicalFramework):
  - composite_score
  - trend_score
  - momentum_score
  - volatility_score

News_Signal (来自 NewsSignalGenerator):
  - sentiment_score: -1~1
  - confidence: 0~1

融合计算:
  dl_score = (proba[2] - proba[0]) * 100  # 转成 -100~100

  combined = (
      dl_score * 0.30 +
      technical_composite * 0.45 +
      news_score * 0.25
  )

  if combined >= 20: final_signal = STRONG_BUY
  elif combined >= 5: final_signal = BUY
  elif combined <= -20: final_signal = STRONG_SELL
  elif combined <= -5: final_signal = SELL
  else: final_signal = HOLD
```

### 4.2 权重配置

| 信号源 | 权重 | 说明 |
|-------|------|------|
| DL 模型 | 30% | 新增，可根据回测效果调整 |
| 技术框架 | 45% | 现有成熟系统 |
| 新闻情感 | 25% | 现有系统 |

权重可通过配置文件调整。

## 5. 训练设计

### 5.1 训练触发

- **每日增量训练**：每天收盘后，用最新数据增量更新模型（可选）
- **定期全量重训练**：每周/每月全量重训练一次，防止分布漂移

### 5.2 训练数据

- 股票池：沪深 300 成分股（~300 只）或其他自定义池
- 时间范围：最近 2~3 年数据
- 标签定义：
  - **涨**：次日收盘价涨幅 > 0.5%
  - **跌**：次日收盘价跌幅 > 0.5%
  - **震荡**：|涨幅| <= 0.5%

### 5.3 早停与验证

- 验证集：最近 60 个交易日
- 早停：验证集 loss 连续 5 个 epoch 不下降
- 最优模型保存：按验证集 accuracy 保留最优

## 6. 与现有系统集成

### 6.1 调度集成

现有 `scheduler_app.py` 新增：
- 每日 16:30（收盘后）触发模型推理，输入当日数据，输出明日信号
- 每周日触发模型全量重训练

### 6.2 信号接口

```python
from ml.signal_fusion import DLSignalFusion

fusion = DLSignalFusion(model_path='models/lstm_v1.pt')
result = fusion.generate_signal(
    symbol='300750.SZ',
    price_data=ohlcv_df,        # 60日OHLCV
    technical_indicators={},     # TechnicalFramework输出
    news_signal=None            # NewsSignalGenerator输出
)

# result:
# {
#     'symbol': '300750.SZ',
#     'prediction': 'buy',       # 最终信号
#     'dl_prediction': 'up',     # DL模型预测
#     'dl_confidence': 0.82,
#     'combined_score': 65.3,
#     'dl_contribution': 18.5,
#     'technical_contribution': 28.8,
#     'news_contribution': 8.0,
#     'timestamp': '2026-04-09 16:30:00'
# }
```

## 7. 依赖

```
torch>=2.0
pandas
numpy
scikit-learn
tushare  # 如果用 tushare 补充数据
akshare  # 现有数据源
```

## 8. 实施顺序

1. **数据管道** — `ml/data/dataset.py` + `preprocessor.py`
2. **LSTM 模型** — `ml/models/lstm_predictor.py`
3. **训练循环** — `ml/training/trainer.py`
4. **推理接口** — `ml/inference/predictor.py`
5. **信号融合** — `ml/signal_fusion/fusion.py`
6. **集成调度** — 修改 `scheduler_app.py`
7. **回测验证** — 用现有 `backtester.py` 验证 DL 信号效果

## 9. 风险与限制

- **过拟合风险**：金融数据噪声大，LSTM 容易过拟合。需严格使用早停 + 验证集
- **预测偏向震荡**：A股市场大部分时候震荡，模型可能偏向预测震荡类，需类别权重调整
- **分布漂移**：市场机制变化（政策、情绪）可能导致历史模式失效，需定期重训练
- **计算资源**：CPU 可运行，但每日重训练建议有 GPU 支持
