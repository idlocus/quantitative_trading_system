# 技术指标权重回测优化 - 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过网格搜索找出使累计收益率最大化的技术指标权重组合

**Architecture:** 新增 `ml/optimization/` 模块，实现参数化的 composite 计算函数和回测函数。修改 `TechnicalFramework._calculate_composite` 的权重从硬编码改为可配置，支持动态传入权重。

**Tech Stack:** Python 3, numpy, pandas, Wind DB

---

## 文件结构

- **Create:** `ml/optimization/__init__.py`
- **Create:** `ml/optimization/weight_optimizer.py` — 核心优化逻辑
- **Modify:** `signals/technical_framework.py:321-466` — `_calculate_composite` 改为可配置权重
- **Modify:** `signals/technical_framework.py:23-81` — `analyze` 方法支持传入自定义权重

---

## Task 1: 创建优化模块目录

**Files:**
- Create: `ml/optimization/__init__.py`

- [ ] **Step 1: 创建目录和 __init__.py**

```bash
mkdir -p ml/optimization
```

```python
# ml/optimization/__init__.py
"""技术指标权重优化模块"""
```

---

## Task 2: 创建参数化的 composite 计算函数

**Files:**
- Create: `ml/optimization/weight_optimizer.py`

- [ ] **Step 1: 创建 weight_optimizer.py，包含参数化的 _calculate_composite 函数**

核心思路：从 `TechnicalFramework._calculate_composite` 复制逻辑，但把硬编码的权重改为函数参数。

权重列表顺序：`['rsi', 'macd', 'kdj', 'boll', 'ma', 'vol']`，对应索引 `[0, 1, 2, 3, 4, 5]`。

```python
# ml/optimization/weight_optimizer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标权重优化器

核心逻辑：从 TechnicalFramework._calculate_composite 复制，
权重改为函数参数，支持网格搜索。
"""

import numpy as np
from typing import Dict, List


def calculate_composite_with_weights(
    rsi: float,
    macd: Dict,
    kdj: Dict,
    boll: Dict,
    ma: Dict,
    volume_ratio: float,
    weights: List[float]  # [rsi_w, macd_w, kdj_w, boll_w, ma_w, vol_w]
) -> float:
    """
    计算给定权重下的综合评分 (0-100)

    Args:
        rsi: RSI值 (0-100)
        macd: MACD dict with keys: direction, histogram, value, signal
        kdj: KDJ dict with keys: k, d, j
        boll: 布林带 dict with keys: position
        ma: 均线 dict with keys: trend, ma5, ma10, ma20, ma60
        volume_ratio: 量比
        weights: 权重列表 [rsi_w, macd_w, kdj_w, boll_w, ma_w, vol_w]，之和为1.0

    Returns:
        composite score (0-100)
    """
    w = weights  # 简写

    score = 50.0  # 基准分

    # RSI评分 (0-100 -> -50 to +50)
    rsi_score = (rsi - 50) * 1.0
    score += rsi_score * w[0]

    # MACD评分
    if macd['direction'] == 'bullish':
        macd_score = min(50, abs(macd['histogram']) * 100)
    elif macd['direction'] == 'bearish':
        macd_score = -min(50, abs(macd['histogram']) * 100)
    else:
        macd_score = 0
    score += macd_score * w[1]

    # KDJ评分
    if kdj['j'] > 100:
        kdj_score = -30
    elif kdj['k'] > 80 or kdj['j'] > 80:
        kdj_score = -20
    elif kdj['k'] < 20 or kdj['j'] < 0:
        kdj_score = 25
    else:
        kdj_score = (kdj['k'] - 50) * 0.6
    score += kdj_score * w[2]

    # 布林带评分
    if boll['position'] > 80:
        boll_score = -25
    elif boll['position'] < 20:
        boll_score = 25
    else:
        boll_score = (boll['position'] - 50) * 0.5
    score += boll_score * w[3]

    # 均线评分
    ma_trend = ma.get('trend', 'neutral')
    if ma_trend == 'strong_bullish':
        ma_score = 40
    elif ma_trend == 'strong_bearish':
        ma_score = -40
    elif ma_trend == 'bullish':
        ma_score = 20
    elif ma_trend == 'bearish':
        ma_score = -20
    else:
        ma_score = 0
    score += ma_score * w[4]

    # 量比评分
    if volume_ratio > 2:
        vol_score = 15
    elif volume_ratio < 0.5:
        vol_score = -10
    else:
        vol_score = 0
    score += vol_score * w[5]

    return max(0, min(100, score))
```

---

## Task 3: 实现回测函数

**Files:**
- Modify: `ml/optimization/weight_optimizer.py`

- [ ] **Step 1: 添加 backtest 函数**

需要从 Wind 数据库加载历史数据，然后逐日计算 composite 信号并模拟交易。

```python
import pandas as pd
from typing import Dict, Tuple


def compute_indicators(price_data: pd.DataFrame) -> Dict:
    """
    计算某日的技术指标（供 composite 计算用）
    price_data: 包含 open/high/low/close/volume 的 DataFrame
    返回: {rsi, macd, kdj, boll, ma, volume_ratio}
    """
    from signals.technical_framework import TechnicalFramework
    tf = TechnicalFramework()
    return tf.analyze(price_data)['indicators']


def backtest(
    weights: List[float],
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 1000000.0
) -> Tuple[float, float, float]:
    """
    对单只股票回测

    Args:
        weights: 权重列表 [rsi, macd, kdj, boll, ma, vol]
        symbol: 股票代码
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        initial_capital: 初始资金

    Returns:
        (累计收益率, 总交易次数, 盈利交易次数)
    """
    from jobs.run_dl_wind_all_a import load_stock_data_from_wind

    price_data = load_stock_data_from_wind(symbol, days=1250)
    if price_data is None or len(price_data) < 60:
        return 0.0, 0, 0

    # 过滤日期
    price_data = price_data[start_date:end_date]

    capital = initial_capital
    position = 0  # 持仓股数
    trades = 0
    wins = 0

    daily_returns = []

    for i in range(20, len(price_data) - 1):
        window = price_data.iloc[:i+1]
        indicators = compute_indicators(window)

        composite = calculate_composite_with_weights(
            rsi=indicators.get('rsi', 50),
            macd=indicators.get('macd', {'direction': 'neutral', 'histogram': 0}),
            kdj=indicators.get('kdj', {'k': 50, 'd': 50, 'j': 50}),
            boll=indicators.get('bollinger', {'position': 50}),
            ma=indicators.get('ma', {'trend': 'neutral'}),
            volume_ratio=indicators.get('volume_ratio', 1.0),
            weights=weights
        )

        # 交易逻辑：composite >= 70 买入，< 70 卖出
        next_open = price_data.iloc[i+1]['open']
        next_close = price_data.iloc[i+1]['close']

        if composite >= 70 and position == 0:
            # 买入
            position = capital / next_open
            capital = 0
            trades += 1
        elif composite < 70 and position > 0:
            # 卖出
            capital = position * next_open
            position = 0
            if capital > initial_capital:
                wins += 1

        # 当日收益（无论是否持仓）
        if position > 0:
            daily_ret = (next_close - next_open) / next_open
            daily_returns.append(daily_ret)

    # 计算累计收益率
    final_value = capital + position * price_data.iloc[-1]['close']
    total_return = (final_value - initial_capital) / initial_capital

    return total_return, trades, wins
```

---

## Task 4: 实现网格搜索函数

**Files:**
- Modify: `ml/optimization/weight_optimizer.py`

- [ ] **Step 1: 添加 generate_weight_combinations 函数**

权重每档 5%，和为 100%。用整数权重 (0-20) 表示百分比。

```python
from itertools import combinations


def generate_weight_combinations():
    """
    生成所有可能的权重组合（每档5%，和为100%）
    返回: List[List[float]] - 每个元素是6个权重的小数形式列表
    """
    # 用整数 0-20 表示百分比 0%-100%，步长5%
    # 6个指标，每个取值 0-20，共21^6 ≈ 85M组合太多
    # 用组合公式：选5个分割点分成6份，C(16,5)=4368种
    # 20个点选5个分割点: C(20,5) = 15504，太多了
    # 实际用: 权重档位为 0, 5, 10, 15, 20 (5档)
    # 6个权重，每个0-20，共21^6 太大
    # 简化：每个权重的档位数减少
    # 方案：每个权重取值 0, 5, 10, 15, 20, 25 (6档)
    # 但要求和=100，所以实际是 C(15,5)=3003种

    # 更简洁的实现：用整数权重的组合
    # 每个权重 0-20，步长1，和=20（对应百分比0-100%，步长5%）
    # 先枚举所有和为20的6元组
    combos = []
    for a in range(21):
        for b in range(21 - a):
            for c in range(21 - a - b):
                for d in range(21 - a - b - c):
                    for e in range(21 - a - b - c - d):
                        f = 20 - a - b - c - d - e
                        combos.append([a, b, c, d, e, f])

    # 转换为小数
    return [[x/20.0 for x in combo] for combo in combos]


def grid_search_weights(
    train_stocks: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 1000000.0
) -> Dict[str, float]:
    """
    网格搜索最优权重

    Returns:
        最优权重字典 {'rsi': 0.15, 'macd': 0.20, ...}
    """
    import logging
    logger = logging.getLogger(__name__)

    combos = generate_weight_combinations()
    logger.info(f"共 {len(combos)} 种权重组合待搜索")

    best_return = -float('inf')
    best_weights = None
    results = []

    for i, weights in enumerate(combos):
        total_return = 0.0
        valid_stocks = 0

        for symbol in train_stocks:
            ret, _, _ = backtest(weights, symbol, start_date, end_date, initial_capital)
            if ret != 0.0:  # 有效结果
                total_return += ret
                valid_stocks += 1

        if valid_stocks > 0:
            avg_return = total_return / valid_stocks
            results.append((avg_return, weights))

        if avg_return > best_return:
            best_return = avg_return
            best_weights = weights
            logger.info(f"第{i+1}/{len(combos)}组合: avg_return={avg_return:.4f}, 当前最优")

    # 转为字典
    names = ['rsi', 'macd', 'kdj', 'boll', 'ma', 'vol']
    return {names[i]: best_weights[i] for i in range(6)}
```

---

## Task 5: 修改 TechnicalFramework 支持自定义权重

**Files:**
- Modify: `signals/technical_framework.py:321-466` — `_calculate_composite` 加 `weights` 参数
- Modify: `signals/technical_framework.py:23-81` — `analyze` 支持传入自定义权重

- [ ] **Step 1: 修改 `_calculate_composite` 签名和实现**

把原来的硬编码权重 `[0.15, 0.20, 0.15, 0.15, 0.25, 0.10]` 改为默认参数：

```python
def _calculate_composite(
    self,
    rsi: float, macd: Dict, kdj: Dict, boll: Dict,
    ma: Dict, volume_ratio: float,
    weights: List[float] = None  # 新增参数
) -> Dict:
    if weights is None:
        weights = [0.15, 0.20, 0.15, 0.15, 0.25, 0.10]
    # 然后把所有 `* 0.15` 改为 `* weights[0]`，以此类推
```

- [ ] **Step 2: 修改 `analyze` 支持自定义权重**

在 `analyze` 方法末尾，`_calculate_composite` 调用处传入 `weights` 参数：

```python
composite_result = self._calculate_composite(
    rsi, macd, kdj, boll, ma, volume_ratio,
    weights=kwargs.get('weights')  # 新增
)
```

---

## Task 6: 添加命令行入口并运行

**Files:**
- Create: `ml/optimization/run_optimization.py`

- [ ] **Step 1: 创建运行脚本**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行技术指标权重优化
"""
import logging
import sys
sys.path.insert(0, '.')

from ml.optimization.weight_optimizer import grid_search_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # 从 Wind 获取股票列表（用已有的批次）
    from ml.data.batch_loader import WindBatchLoader
    loader = WindBatchLoader(days=1250)
    all_stocks = loader.get_all_stocks()[:500]  # 取500只

    # 3年训练期，2年测试期
    # 数据从现在开始往前1250个交易日
    import pandas as pd
    end_date = pd.Timestamp.now().strftime('%Y%m%d')

    # 训练期: 最近3年
    train_end = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime('%Y%m%d')
    train_start = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y%m%d')

    logger.info(f"开始网格搜索，训练期: {train_start} ~ {train_end}")
    best = grid_search_weights(all_stocks, train_start, train_end)
    logger.info(f"最优权重: {best}")
```

- [ ] **Step 2: 运行优化**

```bash
cd D:/python_work/Trae/quantitative_trading_system
python ml/optimization/run_optimization.py
```

预期输出：最优权重字典

---

## Task 7: 更新 TechnicalFramework 默认权重

**Files:**
- Modify: `signals/technical_framework.py:321-466` — 将找到的最优权重更新为默认值

- [ ] **Step 1: 运行后用最优权重更新代码中的默认值**

用 Task 6 的输出，替换 `_calculate_composite` 中的默认权重。

---

## 验证清单

1. `calculate_composite_with_weights` 逻辑与原 `_calculate_composite` 一致（用相同输入验证输出相同）
2. `backtest` 能正常返回收益率
3. `grid_search_weights` 能在合理时间内完成（~500只股票，4000组合可能需要数小时，考虑限制样本数）
4. 最优权重在测试期相比基准有提升
