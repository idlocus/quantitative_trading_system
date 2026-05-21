# 简化量化交易系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个简洁的量化交易系统，包含多指标选股、回测引擎和可视化报告

**Architecture:** 配置驱动的选股 + 事件驱动回测 + HTML可视化报告。K线图直接标注买卖点，悬停显示指标状态。

**Tech Stack:** Python 3.8+, pandas, numpy, lightweight-charts (HTML图表库)

---

## 文件结构规划

```
quantitative_trading_system/
├── indicators/                    # 【保留/精简】技术指标库
│   ├── __init__.py
│   ├── base.py                    # 指标基类 Indicator
│   ├── registry.py                # 指标注册表
│   ├── trend.py                   # SMA, EMA, MACD, DMI
│   ├── momentum.py                # RSI, Stochastic, CCI
│   ├── volatility.py              # ATR, BollingerBands
│   ├── volume.py                  # OBV, MFI, VWAP
│   └── pattern.py                 # 形态识别
│
├── selection/                     # 【新建】选股模块
│   ├── __init__.py
│   ├── config.py                  # SelectionConfig, IndicatorRule
│   ├── scanner.py                 # StockScanner
│   └── strategies.yml            # 策略配置
│
├── strategy/                      # 【简化】策略框架
│   ├── __init__.py
│   └── momentum.py               # MomentumStrategy
│
├── backtesting/                   # 【重构】回测引擎
│   ├── __init__.py
│   ├── engine.py                  # BacktestEngine, Trade, BacktestResult
│   ├── reporter.py                # BacktestReporter (Markdown + HTML)
│   └── templates/                 # HTML模板
│       └── report.html            # K线交易标记报告
│
├── risk/                          # 【精简】风险管理
│   ├── __init__.py
│   └── manager.py                 # RiskManager
│
├── data/                          # 【整合】数据接口
│   ├── __init__.py
│   ├── base.py                    # DataSource 基类
│   ├── wind.py                    # WindDataSource
│   └── akshare.py                 # AKShareDataSource
│
├── utils/                         # 【保留】工具
│   ├── __init__.py
│   └── logger.py
│
└── config/                        # 【新建】系统配置
    └── strategies.yml             # 选股策略配置
```

---

## 待清理删除的目录/文件

| 路径 | 操作 |
|------|------|
| `ml/` | 删除（整个目录） |
| `signals/` | 删除（整个目录） |
| `execution/` | 删除（整个目录） |
| `factors/` | 删除（整个目录） |
| `performance/` | 删除 |
| `visualization/` | 删除 |
| `tools/` | 删除 |
| `jobs/` | 删除（除必要的外） |
| `indicators/breadth_indicators.py` | 删除 |
| `indicators/ichimoku.py` | 删除 |
| `indicators/market_regime.py` | 删除 |
| `indicators/technical_framework.py` | 删除 |
| `backtesting/backtester.py` | 删除（旧版） |
| `data/market/cache.py` | 删除 |
| `data/market/mock.py` | 删除 |
| `data/fundamental/` | 删除 |
| `data/news/` | 删除 |

---

## 任务分解

### Phase 1: 指标模块整理

#### Task 1: 指标基类和注册表
**Files:**
- Modify: `indicators/base.py` - 保持现有Indicator基类，添加evaluate方法
- Modify: `indicators/registry.py` - 添加cross信号检测能力

```python
# indicators/base.py 新增方法
class Indicator(ABC):
    @abstractmethod
    def evaluate(self, data: pd.Series, operator: str, threshold) -> bool:
        """判断条件是否满足"""
        pass

    def get_cross_signal(self, data: pd.DataFrame) -> str:
        """检测金叉/死叉，可被子类重写"""
        return None
```

#### Task 2: 实现核心指标
**Files:**
- Modify: `indicators/trend.py` - SMA, EMA, MACD, DMI
- Modify: `indicators/momentum.py` - RSI, Stochastic, CCI
- Modify: `indicators/volatility.py` - ATR, BollingerBands
- Modify: `indicators/volume.py` - OBV, MFI, VWAP
- Modify: `indicators/pattern.py` - 支撑阻力、简单K线形态
- Modify: `indicators/__init__.py` - 导出

```python
# indicators/trend.py
@registry.register('MACD', category='trend')
def MACD(data, fast=12, slow=26, signal=9):
    ema_fast = EMA(data['close'], fast)
    ema_slow = EMA(data['close'], slow)
    macd = ema_fast - ema_slow
    signal_line = EMA(macd, signal)
    histogram = macd - signal_line
    return pd.DataFrame({'macd': macd, 'signal': signal_line, 'histogram': histogram})

# 支持金叉检测
def get_cross_signal(macd, signal):
    if len(macd) < 2:
        return None
    if macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        return 'gold_cross'
    if macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
        return 'death_cross'
    return None
```

#### Task 3: 指标模块测试
**Files:**
- Create: `tests/test_indicators.py`

```python
def test_rsi_calculation():
    from indicators.momentum import RSI
    rsi = RSI(period=14)
    result = rsi.compute(sample_ohlcv)
    assert 0 <= result.iloc[-1] <= 100

def test_macd_cross():
    from indicators.trend import MACD
    macd = MACD(sample_ohlcv)
    signal = macd.get_cross_signal()
    assert signal in [None, 'gold_cross', 'death_cross']
```

---

### Phase 2: 选股模块

#### Task 4: 选股配置
**Files:**
- Create: `selection/config.py`

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class IndicatorRule:
    name: str                    # "RSI", "MACD"
    operator: str                # ">", "<", ">=", "<=", "cross_up", "cross_down", "break_upper", "break_lower"
    threshold: float             # 阈值
    weight: float = 1.0          # 评分权重

@dataclass
class SelectionConfig:
    name: str
    indicators: List[IndicatorRule] = field(default_factory=list)
    logic: str = "AND"           # "AND" 或 "OR"
    min_score: int = 60          # 最低信号评分 (0-100)
    max_positions: int = 5       # 最大持仓数
```

#### Task 5: 选股扫描器
**Files:**
- Create: `selection/scanner.py`
- Create: `selection/strategies.yml`

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ScoredStock:
    symbol: str
    name: str = ""
    score: int = 0                # 0-100
    conditions: List[Dict] = field(default_factory=list)  # 满足的条件
    indicators: Dict[str, float] = field(default_factory=dict)  # 当前指标值

class StockScanner:
    def __init__(self, data_source):
        self.data_source = data_source
        self.registry = IndicatorRegistry

    def scan(self, symbols: List[str], config: SelectionConfig) -> List[ScoredStock]:
        results = []
        for symbol in symbols:
            score, conditions, indicators = self._evaluate(symbol, config)
            if score >= config.min_score:
                results.append(ScoredStock(symbol=symbol, score=score,
                                          conditions=conditions, indicators=indicators))
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _evaluate(self, symbol: str, config: SelectionConfig):
        data = self.data_source.get_kline(symbol)
        if data is None or len(data) < 60:
            return 0, [], {}

        scores = []
        conditions = []
        indicators = {}

        for rule in config.indicators:
            indicator_func = self.registry.get(rule.name)
            values = indicator_func(data)

            # 获取当前值和前一个值（用于cross检测）
            current = values.iloc[-1]
            prev = values.iloc[-2] if len(values) > 1 else current
            indicators[rule.name] = current

            # 评估条件
            met = self._check_condition(current, prev, rule.operator, rule.threshold)
            if met:
                scores.append(rule.weight * 100)
                conditions.append({
                    'indicator': rule.name,
                    'operator': rule.operator,
                    'threshold': rule.threshold,
                    'value': current
                })

        # 计算总分
        if config.logic == "AND" and len(scores) != len(config.indicators):
            return 0, [], {}
        elif config.logic == "OR" and len(scores) == 0:
            return 0, [], {}

        total_score = min(100, int(sum(scores) / len(config.indicators)))
        return total_score, conditions, indicators

    def _check_condition(self, current: float, prev: float, operator: str, threshold) -> bool:
        if operator == ">":
            return current > threshold
        elif operator == "<":
            return current < threshold
        elif operator == ">=":
            return current >= threshold
        elif operator == "<=":
            return current <= threshold
        elif operator == "cross_up":
            return prev < threshold <= current
        elif operator == "cross_down":
            return prev > threshold >= current
        return False
```

#### Task 6: 策略配置YAML
**Files:**
- Create: `selection/strategies.yml`

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
    - name: "EMA5"
      operator: ">"
      threshold: "EMA20"
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

---

### Phase 3: 策略和风控

#### Task 7: 策略框架
**Files:**
- Modify: `strategy/__init__.py`
- Create: `strategy/momentum.py`

```python
# strategy/base.py
class Strategy(ABC):
    @abstractmethod
    def select(self, date, data) -> List[ScoredStock]:
        """每日选股"""
        pass

# strategy/momentum.py
class MomentumStrategy(Strategy):
    def __init__(self, config_name: str):
        self.config = load_config(config_name)
        self.scanner = StockScanner(data_source)

    def select(self, date, data) -> List[ScoredStock]:
        symbols = data.get_tradable_symbols(date)
        return self.scanner.scan(symbols, self.config)
```

#### Task 8: 风控模块
**Files:**
- Modify: `risk/manager.py`

```python
@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: int
    entry_date: date
    signal_score: int
    conditions: Dict  # 买入时的指标状态

class RiskManager:
    def __init__(self, stop_loss: float = 0.05, take_profit: float = 0.10,
                 max_position_size: float = 0.2):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_position_size = max_position_size

    def allocate(self, signals: List[ScoredStock], capital: float) -> List[Position]:
        positions = []
        for signal in signals[:5]:  # max 5 positions
            weight = self._score_to_weight(signal.score)
            amount = capital * weight
            price = self.data_source.get_realtime(signal.symbol)
            quantity = int(amount / price)
            if quantity > 0:
                positions.append(Position(
                    symbol=signal.symbol,
                    entry_price=price,
                    quantity=quantity,
                    entry_date=date.today(),
                    signal_score=signal.score,
                    conditions=signal.conditions
                ))
        return positions

    def _score_to_weight(self, score: int) -> float:
        if score >= 80: return 0.20
        elif score >= 70: return 0.15
        elif score >= 60: return 0.10
        else: return 0.05

    def check_exit(self, position: Position, current_price: float) -> Optional[str]:
        ret = (current_price - position.entry_price) / position.entry_price
        if ret <= -self.stop_loss:
            return "stop_loss"
        elif ret >= self.take_profit:
            return "take_profit"
        return None
```

---

### Phase 4: 回测引擎

#### Task 9: 回测核心
**Files:**
- Create: `backtesting/engine.py`
- Delete: `backtesting/backtester.py` (旧版)

```python
@dataclass
class Trade:
    id: int
    symbol: str
    action: str              # "BUY" / "SELL"
    date: date
    price: float
    quantity: int
    commission: float
    reason: str              # 触发原因
    conditions: Dict         # 当时的指标状态
    signal_score: int        # 信号评分

@dataclass
class BacktestResult:
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    avg_holding_days: float
    equity_curve: pd.Series
    trades: List[Trade]
    positions: Dict[str, Position]  # 当前持仓

class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades = []
        self.positions = {}
        self.cash = config.initial_capital
        self.equity_curve = []
        self._trade_id = 0

    def run(self, strategy: Strategy, data: MarketData) -> BacktestResult:
        for date in data.trading_dates:
            # 1. 选股
            signals = strategy.select(date, data)

            # 2. 仓位分配
            new_positions = self.risk_manager.allocate(signals, self.cash)

            # 3. 执行买入
            for pos in new_positions:
                if pos.symbol not in self.positions:
                    self._execute_buy(pos, date)

            # 4. 检查止损止盈
            self._check_exit_conditions(date, data)

            # 5. 更新权益
            self._update_equity(date, data)

        return self._generate_result()

    def _execute_buy(self, position: Position, date):
        cost = position.entry_price * position.quantity
        commission = cost * self.config.commission
        total_cost = cost + commission

        if total_cost > self.cash:
            # 资金不足，按比例买入
            position.quantity = int(self.cash / (position.entry_price * (1 + self.config.commission)))
            cost = position.entry_price * position.quantity
            commission = cost * self.config.commission
            total_cost = cost + commission

        self.cash -= total_cost
        self.positions[position.symbol] = position

        self.trades.append(Trade(
            id=self._trade_id,
            symbol=position.symbol,
            action="BUY",
            date=date,
            price=position.entry_price,
            quantity=position.quantity,
            commission=commission,
            reason="买入信号",
            conditions=position.conditions,
            signal_score=position.signal_score
        ))
        self._trade_id += 1

    def _execute_sell(self, symbol: str, date: date, price: float, reason: str):
        position = self.positions[symbol]
        sell_value = price * position.quantity
        commission = sell_value * self.config.commission
        net_value = sell_value - commission

        profit = (price - position.entry_price) * position.quantity - commission

        self.trades.append(Trade(
            id=self._trade_id,
            symbol=symbol,
            action="SELL",
            date=date,
            price=price,
            quantity=position.quantity,
            commission=commission,
            reason=reason,
            conditions=position.conditions,
            signal_score=position.signal_score
        ))
        self._trade_id += 1

        self.cash += net_value
        del self.positions[symbol]

    def _check_exit_conditions(self, date, data):
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            current_price = data.get_close(symbol, date)
            exit_reason = self.risk_manager.check_exit(position, current_price)
            if exit_reason:
                self._execute_sell(symbol, date, current_price, exit_reason)

    def _generate_result(self) -> BacktestResult:
        # 计算各项指标
        ...
```

#### Task 10: 回测报告生成器
**Files:**
- Create: `backtesting/reporter.py`

```python
class BacktestReporter:
    def __init__(self, template_dir: str = None):
        self.template_dir = template_dir or "backtesting/templates"

    def generate_markdown(self, result: BacktestResult) -> str:
        """生成Markdown格式报告"""
        trades_by_symbol = self._group_trades_by_symbol(result.trades)

        lines = [
            f"# 策略回测报告\n",
            f"## 绩效摘要\n",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 回测期间 | {result.start_date} ~ {result.end_date} |",
            f"| 初始资金 | ¥{result.initial_capital:,.0f} |",
            f"| 最终市值 | ¥{result.final_capital:,.0f} |",
            f"| 总收益率 | {result.total_return*100:.1f}% |",
            f"| 年化收益率 | {result.annual_return*100:.1f}% |",
            f"| 夏普比率 | {result.sharpe_ratio:.2f} |",
            f"| 最大回撤 | {result.max_drawdown*100:.1f}% |",
            f"| 胜率 | {result.win_rate*100:.1f}% |",
            f"| 平均持仓天数 | {result.avg_holding_days:.1f} |",
            f"| 总交易次数 | {result.trade_count} |",
            f"\n## 交易记录\n",
        ]

        for symbol, trades in trades_by_symbol.items():
            buy_trade = next(t for t in trades if t.action == "BUY")
            sell_trade = next((t for t in trades if t.action == "SELL"), None)
            if sell_trade:
                ret = (sell_trade.price - buy_trade.price) / buy_trade.price
                lines.append(f"### {symbol}\n")
                lines.append(f"买入: {buy_trade.date} @ ¥{buy_trade.price}\n")
                lines.append(f"卖出: {sell_trade.date} @ ¥{sell_trade.price} ({ret*100:+.1f}%)\n")
                lines.append(f"触发条件:\n")
                for cond in buy_trade.conditions:
                    lines.append(f"- {cond['indicator']}: {cond['operator']} {cond['threshold']}\n")

        return "".join(lines)

    def generate_html(self, result: BacktestResult) -> str:
        """生成HTML可视化报告"""
        template = self._load_template("report.html")
        return template.render(
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_capital=result.final_capital,
            total_return=result.total_return * 100,
            annual_return=result.annual_return * 100,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown * 100,
            win_rate=result.win_rate * 100,
            trade_count=result.trade_count,
            trades=result.trades,
            equity_curve=result.equity_curve.to_dict()
        )
```

#### Task 11: HTML报告模板
**Files:**
- Create: `backtesting/templates/report.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>策略回测报告</title>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.0.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; }
        .header { margin-bottom: 30px; }
        .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }
        .metric { background: #f5f5f5; padding: 20px; border-radius: 8px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #333; }
        .metric-label { color: #666; font-size: 14px; }
        .profit { color: #ef5350; }
        .loss { color: #26a69a; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #fafafa; font-weight: 600; }
        .buy { color: #26a69a; }
        .sell { color: #ef5350; }
        #equityChart { height: 300px; margin: 20px 0; }
        #stockList { margin-top: 40px; }
        .stock-section { border: 1px solid #eee; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }
        .stock-header { background: #fafafa; padding: 15px 20px; font-weight: bold; cursor: pointer; }
        .stock-chart { height: 400px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>策略回测报告</h1>
        <p>{{ start_date }} ~ {{ end_date }}</p>
    </div>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value {% if total_return >= 0 %}profit{% else %}loss{% endif %}">{{ total_return|floatformat:1 }}%</div>
            <div class="metric-label">总收益率</div>
        </div>
        <div class="metric">
            <div class="metric-value {% if annual_return >= 0 %}profit{% else %}loss{% endif %}">{{ annual_return|floatformat:1 }}%</div>
            <div class="metric-label">年化收益率</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ sharpe_ratio|floatformat:2 }}</div>
            <div class="metric-label">夏普比率</div>
        </div>
        <div class="metric">
            <div class="metric-value loss">{{ max_drawdown|floatformat:1 }}%</div>
            <div class="metric-label">最大回撤</div>
        </div>
    </div>

    <h2>累积收益曲线</h2>
    <div id="equityChart"></div>

    <h2>交易记录</h2>
    <table>
        <thead>
            <tr>
                <th>日期</th>
                <th>股票</th>
                <th>操作</th>
                <th>价格</th>
                <th>数量</th>
                <th>评分</th>
                <th>触发条件</th>
            </tr>
        </thead>
        <tbody>
            {% for trade in trades %}
            <tr>
                <td>{{ trade.date }}</td>
                <td>{{ trade.symbol }}</td>
                <td class="{% if trade.action == 'BUY' %}buy{% else %}sell{% endif %}">{{ trade.action }}</td>
                <td>¥{{ trade.price|floatformat:2 }}</td>
                <td>{{ trade.quantity }}</td>
                <td>{{ trade.signal_score }}</td>
                <td>{{ trade.conditions|join:", " }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <div id="stockList">
        <h2>个股K线图</h2>
        {% for symbol, stock_trades in trades_by_symbol.items %}
        <div class="stock-section">
            <div class="stock-header" onclick="toggleChart('{{ symbol }}')">
                {{ symbol }} - {{ stock_trades.0.name }} ({{ stock_trades|length|add:"-1" }} 笔交易)
            </div>
            <div id="chart-{{ symbol }}" class="stock-chart"></div>
        </div>
        {% endfor %}
    </div>

    <script>
        // 渲染累积收益曲线
        const chart = LightweightCharts.createChart(
            document.getElementById('equityChart'),
            { width: 1000, height: 300 }
        );
        chart.addLineSeries().setData(equityCurve.map((v, i) => ({
            time: i,
            value: v
        })));

        // 渲染个股K线图
        function renderStockChart(symbol, klineData, trades) {
            const container = document.getElementById('chart-' + symbol);
            const stockChart = LightweightCharts.createChart(container, { width: 1000, height: 400 });

            stockChart.addCandlestickSeries().setData(klineData);

            // 买入标记
            const buyTrades = trades.filter(t => t.action === 'BUY');
            const sellTrades = trades.filter(t => t.action === 'SELL');

            const buyMarkers = buyTrades.map(t => ({
                time: t.date,
                position: 'belowBar',
                color: '#26a69a',
                shape: 'arrowUp',
                text: `买 ${t.signal_score}分`
            }));

            const sellMarkers = sellTrades.map(t => ({
                time: t.date,
                position: 'aboveBar',
                color: '#ef5350',
                shape: 'arrowDown',
                text: `${((t.price - t.entry_price) / t.entry_price * 100).toFixed(1)}%`
            }));

            stockChart.addCandlestickSeries().setMarkers([...buyMarkers, ...sellMarkers]);
        }

        function toggleChart(symbol) {
            const chartDiv = document.getElementById('chart-' + symbol);
            chartDiv.style.display = chartDiv.style.display === 'none' ? 'block' : 'none';
        }
    </script>
</body>
</html>
```

---

### Phase 5: 数据接口

#### Task 12: 数据接口整合
**Files:**
- Create: `data/base.py` - DataSource 基类
- Modify: `data/wind.py` - WindDataSource
- Modify: `data/akshare.py` - AKShareDataSource

```python
# data/base.py
class DataSource(ABC):
    @abstractmethod
    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """获取K线数据，columns: date, open, high, low, close, volume"""

    @abstractmethod
    def get_realtime(self, symbol: str) -> float:
        """获取实时价格"""

    @abstractmethod
    def list_symbols(self, market: str = "A股") -> List[str]:
        """获取股票列表"""

    def get_tradable_symbols(self, date: date) -> List[str]:
        """获取当日可交易的股票列表"""
        return self.list_symbols()

# data/wind.py
class WindDataSource(DataSource):
    def __init__(self, ...):
        from WindPy import w
        w.start()

    def get_kline(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        # Wind接口获取数据
        # 返回: date, open, high, low, close, volume
        ...
```

---

### Phase 6: 清理旧代码

#### Task 13: 删除冗余代码
**Files:**
- Delete: `ml/` (整个目录)
- Delete: `signals/` (整个目录)
- Delete: `execution/` (整个目录)
- Delete: `factors/` (整个目录)
- Delete: `performance/`
- Delete: `visualization/`
- Delete: `tools/`
- Delete: `jobs/` (除必要的外)
- Delete: `indicators/breadth_indicators.py`
- Delete: `indicators/ichimoku.py`
- Delete: `indicators/market_regime.py`
- Delete: `indicators/technical_framework.py`
- Delete: `backtesting/backtester.py` (旧版)
- Delete: `data/market/cache.py`
- Delete: `data/market/mock.py`
- Delete: `data/fundamental/`
- Delete: `data/news/`

---

## 实施顺序

1. **Task 1-3**: 指标模块整理（基础工作）
2. **Task 12**: 数据接口整合
3. **Task 4-6**: 选股模块
4. **Task 7-8**: 策略和风控
5. **Task 9-11**: 回测引擎和报告
6. **Task 13**: 清理旧代码

---

## 验证方法

1. 用历史数据跑回测，验证累积收益曲线正确
2. 检查买卖点是否正确标注在K线图上
3. 验证止损止盈逻辑正确执行
4. 生成报告，检查所有指标计算正确