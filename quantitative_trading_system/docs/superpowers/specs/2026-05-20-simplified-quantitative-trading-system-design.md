# 简化量化交易系统设计

## 1. 目标

构建一个简洁、专注的量化交易系统，目标是赚钱：
- 通用框架（股票 + 商品期货）
- 短线选股（每天1-5只）
- 多指标组合确认选股（可配置）
- 信号强度 → 仓位分配
- 完整回测引擎 + 可视化报告

## 2. 核心架构

```
quantitative_trading_system/
├── indicators/           # 技术指标库（精简核心）
│   ├── __init__.py
│   ├── base.py          # 指标基类
│   ├── registry.py      # 指标注册表
│   ├── trend.py         # 趋势指标 (SMA, EMA, MACD, DMI)
│   ├── momentum.py      # 动量指标 (RSI, Stochastic, CCI)
│   ├── volatility.py    # 波动率指标 (ATR, BollingerBands)
│   ├── volume.py        # 成交量指标 (OBV, MFI, VWAP)
│   └── pattern.py       # 形态识别
│
├── selection/           # 【新建】选股模块
│   ├── __init__.py
│   ├── config.py       # 选股策略配置
│   ├── scanner.py      # 多指标扫描器
│   └── strategies.yml  # 策略配置文件
│
├── strategy/            # 【简化】策略框架
│   ├── __init__.py
│   ├── base.py         # 策略基类
│   └── momentum.py     # 动量选股策略
│
├── backtesting/        # 【简化】回测引擎
│   ├── __init__.py
│   ├── engine.py       # 核心回测逻辑
│   ├── reporter.py     # 回测报告生成
│   └── templates/      # HTML报告模板
│       ├── report.html # 整体收益报告模板
│       └── kline.html  # K线交易标记模板
│
├── risk/               # 【精简】风险管理
│   ├── __init__.py
│   └── manager.py     # 仓位计算、止损止盈
│
├── data/               # 数据接口
│   ├── __init__.py
│   ├── base.py         # 数据源基类
│   ├── wind.py         # Wind数据接口
│   └── akshare.py      # AKShare数据接口
│
├── utils/              # 工具函数
│   ├── __init__.py
│   ├── logger.py
│   └── date.py
│
└── config/             # 配置
    └── system.yaml
```

## 3. 待清理删除

| 目录/文件 | 操作 |
|-----------|------|
| `ml/` | 删除 |
| `signals/` | 删除 |
| `execution/` | 删除 |
| `factors/` | 删除 |
| `performance/` | 合并到 backtesting/ |
| `visualization/` | 简化，只保留基础图表模板 |
| `tools/` | 删除 |
| `jobs/` | 精简到1-2个核心脚本 |
| `indicators/breadth_*.py` | 删除 |
| `indicators/ichimoku.py` | 删除 |
| `indicators/market_regime.py` | 删除 |

## 4. 选股模块设计

### 4.1 配置结构

```python
@dataclass
class IndicatorRule:
    name: str           # "RSI", "MACD", "MA"
    operator: str       # ">", "<", ">=", "<=", "cross_up", "cross_down"
    threshold: float    # 阈值
    weight: float = 1.0 # 评分权重

@dataclass
class SelectionConfig:
    name: str
    indicators: List[IndicatorRule]  # 指标条件列表
    logic: str = "AND"               # AND/OR 组合逻辑
    min_score: int = 60              # 最低信号评分
    max_positions: int = 5           # 最大持仓数
```

### 4.2 策略配置示例 (YAML)

```yaml
short_term_momentum:
  name: "短线动量策略"
  logic: "AND"
  indicators:
    - name: "RSI"
      operator: "<"
      threshold: 40
      weight: 1.5
    - name: "MACD"
      operator: "cross_up"
      threshold: 0
      weight: 2.0
    - name: "MA5"
      operator: ">"
      threshold: "MA20"
      weight: 1.0
  min_score: 65
  max_positions: 3

breakout:
  name: "突破策略"
  logic: "AND"
  indicators:
    - name: "BollingerBand"
      operator: "break_upper"
      weight: 2.0
    - name: "Volume"
      operator: ">"
      threshold: 1.5
      weight: 1.0
  min_score: 70
  max_positions: 2
```

### 4.3 扫描器接口

```python
class StockScanner:
    def scan(self, symbols: List[str], data: MarketData,
             config: SelectionConfig) -> List[ScoredStock]:
        """扫描股票，返回评分列表"""
        results = []
        for symbol in symbols:
            stock_data = data.get_symbol_data(symbol)
            score, conditions = self._calculate_score(stock_data, config)
            if score >= config.min_score:
                results.append(ScoredStock(symbol, score, conditions))
        return sorted(results, key=lambda x: x.score, reverse=True)

@dataclass
class ScoredStock:
    symbol: str
    score: int                    # 0-100
    conditions: List[Condition]  # 满足的条件
    indicators: dict             # 当前指标值
```

## 5. 指标模块设计

### 5.1 指标注册

```python
# 使用装饰器注册指标
_indicators = {}

def register(name):
    def decorator(cls):
        _indicators[name.upper()] = cls
        return cls
    return decorator

@register("RSI")
class RSI(Indicator):
    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        # 计算RSI
        ...

    def evaluate(self, value: float, operator: str, threshold: float) -> bool:
        # 判断条件是否满足
        ...

@register("MACD")
class MACD(Indicator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        ...

    def get_cross_signal(self, data: pd.DataFrame) -> str:
        # 返回 "gold_cross", "death_cross", 或 None
        ...
```

### 5.2 支持的指标

**趋势指标**: SMA, EMA, MACD, DMI
**动量指标**: RSI, Stochastic, CCI, ROC
**波动率指标**: ATR, BollingerBands, StdDev
**成交量指标**: OBV, MFI, VWAP
**形态识别**: K线形态、支撑阻力位

## 6. 回测引擎设计

### 6.1 配置

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100_000   # 初始资金
    commission: float = 0.0003          # 手续费万三
    slippage: float = 0.001            # 滑点千一
    position_limit: int = 5            # 最大持仓数
    stop_loss: float = 0.05            # 止损5%
    take_profit: float = 0.10         # 止盈10%
```

### 6.2 核心逻辑

```python
class BacktestEngine:
    def run(self, strategy: Strategy, data: MarketData) -> BacktestResult:
        self.portfolio = Portfolio(self.config.initial_capital)
        self.trades = []

        for date in data.trading_dates:
            # 1. 选股
            signals = strategy.select(date, data)

            # 2. 仓位分配
            positions = self.risk_manager.allocate(signals, self.portfolio.cash)

            # 3. 执行买入
            for pos in positions:
                self._execute_buy(date, pos)

            # 4. 检查止损止盈
            self._check_stop_loss_take_profit(date)

            # 5. 更新组合
            self.portfolio.update(date, data)

        return self._generate_result()

@dataclass
class Trade:
    symbol: str
    action: str              # "BUY" / "SELL"
    date: date
    price: float
    quantity: int
    commission: float
    reason: str              # 触发原因
    conditions: dict         # 当时的指标状态
    signal_score: int        # 信号评分
```

## 7. 回测报告设计

### 7.1 三层报告结构

```
回测报告
├── 1. 整体概览（绩效摘要 + 累积收益曲线）
├── 2. 个股收益贡献（表格 + 柱状图）
└── 3. K线交易标记图（每只股票的买卖点可视化）
```

### 7.2 整体概览指标

| 指标 | 说明 |
|------|------|
| 回测期间 | 起止日期 |
| 初始资金 | ¥100,000 |
| 最终市值 | 最终总资产 |
| 总收益率 | (最终-初始)/初始 |
| 年化收益率 | 年化收益 |
| 夏普比率 | 风险调整收益 |
| 最大回撤 | 最大回撤比例 |
| 胜率 | 盈利交易/总交易 |
| 平均持仓天数 | 平均持有时间 |
| 总交易次数 | 买入次数 |

### 7.3 K线交易标记图

**图表特性**:
- K线图 + 成交量
- 买入点：绿色向上三角形 ▲（标注在K线下方）
- 卖出点：红色向下三角形 ▼（标注在K线上方）
- 悬停显示：日期、开高低收、成交量
- 点击买卖点：弹出详情（买入原因、卖出原因、收益率）

**HTML实现**: 使用 lightweight-charts 库

```html
<script>
    const chart = LightweightCharts.createChart(container, {...});
    const candleSeries = chart.addCandlestickSeries({...});
    candleSeries.setData(klineData);

    // 买入标记
    const buyMarkers = trades.filter(t => t.action === 'BUY').map(t => ({
        time: t.date,
        position: 'belowBar',
        color: '#26a69a',
        shape: 'arrowUp',
        text: `买 ${t.signal_score}分`
    }));

    // 卖出标记
    const sellMarkers = trades.filter(t => t.action === 'SELL').map(t => ({
        time: t.date,
        position: 'aboveBar',
        color: '#ef5350',
        shape: 'arrowDown',
        text: `${t.returnRate > 0 ? '+' : ''}${t.returnRate}%`
    }));

    candleSeries.setMarkers([...buyMarkers, ...sellMarkers]);
</script>
```

### 7.4 买卖点详情

```python
{
    "symbol": "600519",
    "name": "贵州茅台",
    "trades": [
        {
            "action": "BUY",
            "date": "2024-01-15",
            "price": 1680.0,
            "signal_score": 85,
            "reason": "多指标买入信号",
            "conditions": {
                "RSI": {"value": 35, "threshold": 40, "met": True},
                "MACD": {"signal": "金叉", "met": True},
                "MA5_MA20": {"value": "MA5 > MA20", "met": True}
            }
        },
        {
            "action": "SELL",
            "date": "2024-01-18",
            "price": 1720.0,
            "return": "+2.4%",
            "holding_days": 3,
            "reason": "RSI>60 触发止盈",
            "conditions": {
                "RSI": {"value": 65, "threshold": 60, "met": True}
            }
        }
    ]
}
```

## 8. 风险管理设计

### 8.1 仓位分配

```python
def _score_to_weight(score: int) -> float:
    """评分转仓位 (0-100 -> 5%-20%)"""
    if score >= 80:
        return 0.20   # 强信号，满仓20%
    elif score >= 70:
        return 0.15   # 较强信号，15%
    elif score >= 60:
        return 0.10   # 中等信号，10%
    else:
        return 0.05   # 弱信号，5%

def allocate(self, signals: List[ScoredStock], capital: float) -> List[Position]:
    """根据信号强度分配仓位"""
    positions = []
    for signal in signals[:self.config.position_limit]:
        weight = self._score_to_weight(signal.score)
        amount = capital * weight
        positions.append(Position(signal.symbol, amount, signal))
    return positions
```

### 8.2 止损止盈

```python
class RiskManager:
    def __init__(self, stop_loss: float = 0.05, take_profit: float = 0.10):
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def check_exit(self, position: Position, current_price: float) -> str:
        """检查是否需要止损止盈"""
        ret = (current_price - position.entry_price) / position.entry_price

        if ret <= -self.stop_loss:
            return "stop_loss"
        elif ret >= self.take_profit:
            return "take_profit"
        return None
```

## 9. 数据接口设计

### 9.1 统一接口

```python
class DataSource(ABC):
    @abstractmethod
    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """获取K线数据"""

    @abstractmethod
    def get_realtime(self, symbol: str) -> Quote:
        """获取实时行情"""

    @abstractmethod
    def list_symbols(self, market: str = "A股") -> List[str]:
        """获取股票列表"""
```

### 9.2 Wind实现

```python
class WindDataSource(DataSource):
    def __init__(self, ...):
        from WindPy import w
        w.start()

    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        # Wind接口获取数据
        ...
```

### 9.3 AKShare实现

```python
class AKShareDataSource(DataSource):
    def __init__(self):
        import akshare as ak

    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        # AKShare接口获取数据
        ...
```

## 10. 实施计划

### 阶段一：基础设施
1. 清理旧代码，保留核心指标
2. 实现数据接口（Wind + AKShare）
3. 实现指标基类和注册表

### 阶段二：选股模块
4. 实现选股配置和扫描器
5. 实现基础策略
6. 实现仓位管理

### 阶段三：回测引擎
7. 实现回测核心逻辑
8. 实现回测报告生成器
9. 实现HTML可视化报告（含K线交易标记）

### 阶段四：测试验证
10. 用历史数据回测验证
11. 优化策略参数