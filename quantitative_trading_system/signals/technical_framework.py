#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术框架信号生成器

基于多个技术指标综合评分生成交易信号
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TechnicalFramework:
    """技术框架分析器"""

    def __init__(self):
        pass

    def analyze(self, price_data: pd.DataFrame, **kwargs) -> Dict:
        """
        分析价格数据，生成技术框架信号

        Args:
            price_data: OHLCV数据，index为日期，columns为['open', 'high', 'low', 'close', 'volume']

        Returns:
            技术框架输出字典
        """
        if price_data.empty or len(price_data) < 20:
            return self._default_output()

        try:
            close = price_data['close'].values
            high = price_data['high'].values
            low = price_data['low'].values
            volume = price_data['volume'].values if 'volume' in price_data.columns else np.ones(len(close))

            # 计算各技术指标
            rsi = self._calculate_rsi(close, period=14)
            macd = self._calculate_macd(close, use_ma30=kwargs.get('use_ma30', False))
            kdj = self._calculate_kdj(high, low, close)
            boll = self._calculate_bollinger(close)
            ma = self._calculate_ma(close)
            volume_ratio = self._calculate_volume_ratio(volume)
            ma9_crossover = self._calculate_ma9_crossover(close)
            adx = self._calculate_adx(high, low, close)
            supertrend = self._calculate_supertrend(high, low, close)
            vwap = self._calculate_vwap(high, low, close, volume)
            ichimoku = self._calculate_ichimoku(high, low, close)
            nine_turns = self._calculate_nine_turns(close)

            # 综合评分 (0-100)
            composite_result = self._calculate_composite(
                rsi, macd, kdj, boll, ma, volume_ratio, close,
                weights=kwargs.get('weights'),
                ma9_crossover=ma9_crossover,
                adx=adx,
                supertrend=supertrend,
                vwap=vwap,
                ichimoku=ichimoku,
                nine_turns=nine_turns,
                use_dynamic_weights=kwargs.get('use_dynamic_weights', True)
            )
            composite = composite_result['score']

            # 趋势判断
            trend = self._calculate_trend(close, ma)

            # 动量判断
            momentum = self._calculate_momentum(close, macd)

            # 波动性判断
            volatility = self._calculate_volatility(price_data)

            return {
                'signals': {
                    'composite': composite,
                    'recommendation': self._composite_to_signal(composite),
                    'breakdown': composite_result['breakdown']
                },
                'trend': trend,
                'momentum': momentum,
                'volatility': volatility,
                'indicators': {
                    'rsi': rsi,
                    'macd': macd,
                    'kdj': kdj,
                    'bollinger': boll,
                    'ma': ma,
                    'volume_ratio': volume_ratio,
                    'ma9_crossover': ma9_crossover,
                    'adx': adx,
                    'supertrend': supertrend,
                    'vwap': vwap,
                    'ichimoku': ichimoku,
                    'nine_turns': nine_turns
                }
            }

        except Exception as e:
            logger.warning(f"技术框架分析失败: {e}")
            return self._default_output()

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算指数移动平均线 (EMA)"""
        if len(data) < period:
            return np.zeros(len(data))

        ema = np.zeros(len(data))
        alpha = 2.0 / (period + 1)

        # 初始EMA为SMA
        ema[period - 1] = np.mean(data[:period])

        # 后续EMA使用平滑公式
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _default_output(self) -> Dict:
        """默认输出"""
        return {
            'signals': {'composite': 50, 'recommendation': 'hold'},
            'trend': {'strength': 50, 'direction': 'neutral'},
            'momentum': {'value': 0, 'direction': 'neutral'},
            'volatility': {'value': 50, 'level': 'normal'},
            'indicators': {}
        }

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """计算RSI"""
        if len(close) < period + 1:
            return 50.0

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    def _calculate_macd(self, close: np.ndarray, use_ma30: bool = False) -> Dict:
        """计算MACD

        Args:
            close: 收盘价数组
            use_ma30: True则用价格相对30日均线偏离量，False则用标准EMA12/26
        """
        if len(close) < 30:
            return {'value': 0, 'signal': 0, 'histogram': 0, 'direction': 'neutral'}

        if use_ma30:
            # 方案：MACD = 价格 - 30日均线（偏离量），信号线 = 9日EMA
            ma30 = self._ema(close, 30)  # 用EMA近似MA30
            macd_line = close - ma30
            signal_line = self._ema(macd_line, 9)
            histogram = macd_line - signal_line
        else:
            # 标准MACD: EMA12 - EMA26
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd_line = ema12 - ema26
            signal_line = self._ema(macd_line, 9)
            histogram = macd_line - signal_line

        direction = 'bullish' if np.all(histogram > 0) else 'bearish' if np.all(histogram < 0) else 'neutral'

        return {
            'value': float(macd_line[-1]) if len(macd_line) > 0 else 0,
            'signal': float(signal_line[-1]) if len(signal_line) > 0 else 0,
            'histogram': float(histogram[-1]) if len(histogram) > 0 else 0,
            'direction': direction
        }

    def _calculate_kdj(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 9) -> Dict:
        """计算KDJ"""
        if len(close) < period:
            return {'k': 50, 'd': 50, 'j': 50, 'direction': 'neutral'}

        lowest_low = pd.Series(low).rolling(window=period).min()
        highest_high = pd.Series(high).rolling(window=period).max()

        rsv = (close[-1] - lowest_low.iloc[-1]) / (highest_high.iloc[-1] - lowest_low.iloc[-1] + 1e-10) * 100

        k = 50.0
        d = 50.0
        k = (2/3) * k + (1/3) * rsv
        d = (2/3) * d + (1/3) * k
        j = 3 * k - 2 * d

        direction = 'bullish' if j > 80 else 'bearish' if j < 20 else 'neutral'

        return {
            'k': float(k),
            'd': float(d),
            'j': float(j),
            'direction': direction
        }

    def _calculate_bollinger(self, close: np.ndarray, period: int = 20, std_dev: int = 2) -> Dict:
        """计算布林带"""
        if len(close) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 50, 'width': 0}

        middle = pd.Series(close).rolling(window=period).mean()
        std = pd.Series(close).rolling(window=period).std()

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        position = (close[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-10) * 100

        return {
            'upper': float(upper.iloc[-1]),
            'middle': float(middle.iloc[-1]),
            'lower': float(lower.iloc[-1]),
            'position': float(position),
            'width': float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1] * 100) if middle.iloc[-1] != 0 else 0
        }

    def _calculate_ma(self, close: np.ndarray) -> Dict:
        """计算移动平均线"""
        if len(close) < 60:
            return {'ma5': 0, 'ma10': 0, 'ma20': 0, 'ma60': 0, 'trend': 'neutral'}

        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:])

        # 趋势判断：多头排列、空头排列
        if ma5 > ma10 > ma20 > ma60:
            trend = 'strong_bullish'
        elif ma5 < ma10 < ma20 < ma60:
            trend = 'strong_bearish'
        elif ma5 > ma20:
            trend = 'bullish'
        elif ma5 < ma20:
            trend = 'bearish'
        else:
            trend = 'neutral'

        return {
            'ma5': float(ma5),
            'ma10': float(ma10),
            'ma20': float(ma20),
            'ma60': float(ma60),
            'trend': trend
        }

    def _calculate_volume_ratio(self, volume: np.ndarray) -> float:
        """计算量比"""
        if len(volume) < 5:
            return 1.0

        avg_volume = np.mean(volume[-20:-1]) if len(volume) > 20 else np.mean(volume[:-1])
        if avg_volume == 0:
            return 1.0

        return float(volume[-1] / avg_volume)

    def _calculate_ma9_crossover(self, close: np.ndarray) -> Dict:
        """计算9日均线转向"""
        if len(close) < 9:
            return {'ma9': 0, 'direction': 'neutral', 'cross': 'none'}

        ma9_current = np.mean(close[-9:])
        ma9_previous = np.mean(close[-10:-1])

        # 金叉：前一日MA9 <= 前二日MA9，今日MA9 > 前一日MA9 （下降转上升）
        # 死叉：前一日MA9 >= 前二日MA9，今日MA9 < 前一日MA9 （上升转下降）
        if len(close) >= 11:
            ma9_prev2 = np.mean(close[-11:-2])
            # 金叉条件：均线由降转升
            if ma9_previous <= ma9_prev2 and ma9_current > ma9_previous:
                cross = 'golden'  # 金叉，看多
            # 死叉条件：均线由升转降
            elif ma9_previous >= ma9_prev2 and ma9_current < ma9_previous:
                cross = 'death'   # 死叉，看空
            else:
                cross = 'none'
        else:
            cross = 'none'

        # 判断方向：当前MA9相对于前期均值
        ma9_avg = np.mean(close[-20:-1]) if len(close) > 20 else np.mean(close[:-1])
        if ma9_current > ma9_avg * 1.02:
            direction = 'bullish'
        elif ma9_current < ma9_avg * 0.98:
            direction = 'bearish'
        else:
            direction = 'neutral'

        return {
            'ma9': float(ma9_current),
            'direction': direction,
            'cross': cross
        }

    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Dict:
        """
        计算ADX (Average Directional Index) 趋势强度指标

        ADX > 25: 趋势明确
        ADX < 20: 趋势不明显/盘整
        +DI > -DI: 看多
        -DI > +DI: 看空
        """
        if len(close) < period + 2:
            return {'adx': 0, 'plus_di': 0, 'minus_di': 0, 'trend_strength': 'weak'}

        # 计算 True Range (TR)
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []

        for i in range(1, len(close)):
            high_curr = high[i]
            low_curr = low[i]
            high_prev = high[i-1]
            low_prev = low[i-1]
            close_prev = close[i-1]

            # True Range
            tr1 = high_curr - low_curr
            tr2 = abs(high_curr - close_prev)
            tr3 = abs(low_curr - close_prev)
            tr = max(tr1, tr2, tr3)
            tr_list.append(tr)

            # Directional Movement
            up_move = high_curr - high_prev
            down_move = low_prev - low_curr

            # +DM: 上升方向变动
            if up_move > down_move and up_move > 0:
                plus_dm = up_move
            else:
                plus_dm = 0

            # -DM: 下降方向变动
            if down_move > up_move and down_move > 0:
                minus_dm = down_move
            else:
                minus_dm = 0

            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)

        tr = np.array(tr_list)
        plus_dm = np.array(plus_dm_list)
        minus_dm = np.array(minus_dm_list)

        if len(tr) < period:
            return {'adx': 0, 'plus_di': 0, 'minus_di': 0, 'trend_strength': 'weak'}

        # 计算 Smoothed TR, +DM, -DM
        smoothed_tr = np.zeros(len(tr))
        smoothed_plus_dm = np.zeros(len(plus_dm))
        smoothed_minus_dm = np.zeros(len(minus_dm))

        # 前period个TR的简单平均作为初始值
        smoothed_tr[period-1] = np.sum(tr[:period])
        smoothed_plus_dm[period-1] = np.sum(plus_dm[:period])
        smoothed_minus_dm[period-1] = np.sum(minus_dm[:period])

        # Wilder平滑
        for i in range(period, len(tr)):
            smoothed_tr[i] = smoothed_tr[i-1] - smoothed_tr[i-1]/period + tr[i]
            smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - smoothed_plus_dm[i-1]/period + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - smoothed_minus_dm[i-1]/period + minus_dm[i]

        # 计算 +DI 和 -DI
        plus_di = np.zeros(len(close))
        minus_di = np.zeros(len(close))
        dx = np.zeros(len(close))

        for i in range(period, len(close)):
            idx = i - period

            if smoothed_tr[idx] > 0:
                plus_di[i] = 100 * smoothed_plus_dm[idx] / smoothed_tr[idx]
                minus_di[i] = 100 * smoothed_minus_dm[idx] / smoothed_tr[idx]

            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        # 计算 ADX (DX的平滑均值)
        adx = np.zeros(len(close))
        adx[period*2-1] = np.mean(dx[period:period*2])

        for i in range(period*2, len(close)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

        adx_val = adx[-1]
        plus_di_val = plus_di[-1]
        minus_di_val = minus_di[-1]

        # 趋势强度判断
        if adx_val >= 25:
            if plus_di_val > minus_di_val:
                trend_direction = 'strong_bullish'
            elif minus_di_val > plus_di_val:
                trend_direction = 'strong_bearish'
            else:
                trend_direction = 'strong_neutral'
        elif adx_val >= 20:
            if plus_di_val > minus_di_val:
                trend_direction = 'bullish'
            elif minus_di_val > plus_di_val:
                trend_direction = 'bearish'
            else:
                trend_direction = 'neutral'
        else:
            trend_direction = 'weak'

        return {
            'adx': float(adx_val),
            'plus_di': float(plus_di_val),
            'minus_di': float(minus_di_val),
            'trend_direction': trend_direction,
            'trend_strength': 'strong' if adx_val >= 25 else ('moderate' if adx_val >= 20 else 'weak')
        }

    def _calculate_supertrend(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                              period: int = 10, multiplier: float = 3.0) -> Dict:
        """
        计算 Supertrend 指标

        Supertrend: 基于ATR的趋势跟踪指标
        -价格上涨时Supertrend在价格下方（绿色）
        -价格下跌时Supertrend在价格上方（红色）

        返回:
            'trend': 'up'/'down'/'neutral'
            'supertrend_value': 当前的Supertrend线值
            'atr': ATR值
            'signal': 'buy'/'sell'/'none'
        """
        if len(close) < period + 1:
            return {'trend': 'neutral', 'supertrend_value': 0.0, 'atr': 0.0, 'signal': 'none'}

        # 计算 ATR (Average True Range)
        tr_list = []
        for i in range(1, len(close)):
            high_curr = high[i]
            low_curr = low[i]
            high_prev = high[i-1]
            low_prev = low[i-1]
            close_prev = close[i-1]

            tr1 = high_curr - low_curr
            tr2 = abs(high_curr - close_prev)
            tr3 = abs(low_curr - close_prev)
            tr_list.append(max(tr1, tr2, tr3))

        tr = np.array(tr_list)
        atr = np.mean(tr[-period:])

        if atr == 0:
            return {'trend': 'neutral', 'supertrend_value': float(close[-1]), 'atr': 0.0, 'signal': 'none'}

        # 计算 Upper Band 和 Lower Band
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        # 计算 Supertrend
        supertrend = np.zeros(len(close))
        trend = np.zeros(len(close))  # 1=上涨, -1=下跌

        supertrend[0] = upper_band[0]
        trend[0] = 1

        for i in range(1, len(close)):
            prev_close = close[i-1]
            prev_st = supertrend[i-1]
            prev_trend = trend[i-1]

            # 上涨趋势
            if prev_trend == 1:
                if close[i] < prev_st:
                    trend[i] = -1
                    supertrend[i] = lower_band[i]
                else:
                    trend[i] = 1
                    supertrend[i] = max(prev_st, lower_band[i])
            # 下跌趋势
            else:  # prev_trend == -1
                if close[i] > prev_st:
                    trend[i] = 1
                    supertrend[i] = upper_band[i]
                else:
                    trend[i] = -1
                    supertrend[i] = min(prev_st, upper_band[i])

        current_trend = 'up' if trend[-1] == 1 else 'down'

        # 生成交易信号
        signal = 'none'
        if len(trend) >= 2:
            if trend[-1] == 1 and trend[-2] == -1:
                signal = 'buy'   # 趋势由跌转涨
            elif trend[-1] == -1 and trend[-2] == 1:
                signal = 'sell'  # 趋势由涨转跌

        return {
            'trend': current_trend,
            'supertrend_value': float(supertrend[-1]),
            'atr': float(atr),
            'signal': signal
        }

    def _calculate_nine_turns(self, close: np.ndarray) -> Dict:
        """
        计算九转指标（TD Sequential）

        TD Sequential识别9根连续K线：
        - 买入计数：收盘价高于前4根K线收盘价
        - 卖出计数：收盘价低于前4根K线收盘价

        第9根K线是潜在的转折点

        Returns:
            'count': 当前计数（1-9）
            'phase': 'buy'（上涨计数）或 'sell'（下跌计数）或 'neutral'
            'signal': 'buy'（第9根买入完成）、'sell'（第9根卖出完成）、'none'
            'is_complete': 是否达到第9根完成
        """
        if len(close) < 9:
            return {'count': 0, 'phase': 'neutral', 'signal': 'none', 'is_complete': False}

        # 计算TD计数
        buy_count = 0
        sell_count = 0

        # 从最近一根开始向前计算
        for i in range(9):
            idx = len(close) - 1 - i
            if idx < 4:
                break
            # 比较当前收盘价与前4根K线的收盘价
            if close[idx] > close[idx - 4]:
                buy_count += 1
            elif close[idx] < close[idx - 4]:
                sell_count += 1

        # 确定phase和signal
        phase = 'neutral'
        signal = 'none'
        is_complete = False

        # 优先判断完成的计数（9根都满足条件）
        if buy_count == 9:
            phase = 'buy'
            signal = 'buy'
            is_complete = True
        elif sell_count == 9:
            phase = 'sell'
            signal = 'sell'
            is_complete = True
        elif buy_count > sell_count and buy_count >= 5:
            phase = 'buy'
            signal = 'none'
        elif sell_count > buy_count and sell_count >= 5:
            phase = 'sell'
            signal = 'none'

        return {
            'count': max(buy_count, sell_count),
            'phase': phase,
            'signal': signal,
            'is_complete': is_complete
        }

    def _calculate_vwap(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> Dict:
        """
        计算VWAP (Volume Weighted Average Price) 当日成交量加权均价

        价格在VWAP上方：偏多
        价格在VWAP下方：偏空
        偏离VWAP过多：可能有回调

        Returns:
            'vwap': float, 'price_vs_vwap': float (百分比偏离), 'signal': 'above'/'below'/'far_above'/'far_below'
        """
        if len(close) < 2 or len(volume) < 2:
            return {'vwap': float(close[-1]) if len(close) > 0 else 0, 'price_vs_vwap': 0.0, 'signal': 'neutral'}

        # 使用最近一天的数据计算VWAP
        typical_price = (high[-1] + low[-1] + close[-1]) / 3
        volume_sum = np.sum(volume)
        if volume_sum == 0:
            return {'vwap': float(typical_price), 'price_vs_vwap': 0.0, 'signal': 'neutral'}

        # 简化：VWAP ≈ 当日均价
        vwap = typical_price

        # 计算价格偏离VWAP的百分比
        price_vs_vwap = (close[-1] - vwap) / vwap * 100 if vwap != 0 else 0

        # 信号判断
        if price_vs_vwap > 2:
            signal = 'far_above'  # 价格远高于VWAP，可能回调
        elif price_vs_vwap > 0.5:
            signal = 'above'      # 价格略高于VWAP，偏多
        elif price_vs_vwap < -2:
            signal = 'far_below' # 价格远低于VWAP，可能反弹
        elif price_vs_vwap < -0.5:
            signal = 'below'     # 价格略低于VWAP，偏空
        else:
            signal = 'neutral'

        return {
            'vwap': float(vwap),
            'price_vs_vwap': float(price_vs_vwap),
            'signal': signal
        }

    def _calculate_ichimoku(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict:
        """
        计算Ichimoku Cloud (一目均衡表)

        组成部分：
        - Tenkan-sen (转换线): (9日最高+9日最低)/2
        - Kijun-sen (基准线): (26日最高+26日最低)/2
        - Senkou Span A (先行上线): (转换线+基准线)/2，向前26日
        - Senkou Span B (先行下线): (52日最高+52日最低)/2，向前26日
        - Chikou Span (延迟线): 当前收盘价，向后26日

        云带：Senkou A与Senkou B之间的区域
        - 价格在云带上方的：偏多
        - 价格在云带下方的：偏空
        - 价格在云带内的：中性

        Returns:
            'tenkan': float, 'kijun': float, 'senkou_a': float, 'senkou_b': float,
            'chikou': float, 'cloud_signal': 'bullish'/'bearish'/'neutral'
        """
        if len(close) < 52:
            return {
                'tenkan': 0, 'kijun': 0, 'senkou_a': 0, 'senkou_b': 0,
                'chikou': 0, 'cloud_signal': 'neutral'
            }

        # Tenkan-sen (9日)
        tenkan_high = np.max(high[-9:])
        tenkan_low = np.min(low[-9:])
        tenkan = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (26日)
        kijun_high = np.max(high[-26:])
        kijun_low = np.min(low[-26:])
        kijun = (kijun_high + kijun_low) / 2

        # Senkou Span A (先行上线)
        senkou_a = (tenkan + kijun) / 2

        # Senkou Span B (先行下线) - 52日
        span52_high = np.max(high[-52:])
        span52_low = np.min(low[-52:])
        senkou_b = (span52_high + span52_low) / 2

        # Chikou Span (延迟线) - 当前价格与26日前价格对比
        chikou = close[-1]
        # 26日前的价格（用于判断chikou是否在价格上方）
        if len(close) >= 26:
            chikou_vs_price = chikou - close[-27] if len(close) > 26 else 0
        else:
            chikou_vs_price = 0

        # 云带信号判断
        # 价格与云带比较
        current_price = close[-1]

        # 判断价格相对于云带的位置
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        if current_price > cloud_top:
            cloud_signal = 'bullish'
        elif current_price < cloud_bottom:
            cloud_signal = 'bearish'
        else:
            cloud_signal = 'neutral'

        # 另外检查转换线与基准线交叉
        if tenkan > kijun:
            cross_signal = 'bullish'
        elif tenkan < kijun:
            cross_signal = 'bearish'
        else:
            cross_signal = 'neutral'

        # 综合信号
        if cloud_signal == cross_signal:
            final_signal = cloud_signal
        elif cloud_signal == 'neutral':
            final_signal = cross_signal
        elif cross_signal == 'neutral':
            final_signal = cloud_signal
        else:
            final_signal = 'neutral'

        return {
            'tenkan': float(tenkan),
            'kijun': float(kijun),
            'senkou_a': float(senkou_a),
            'senkou_b': float(senkou_b),
            'chikou': float(chikou),
            'cloud_signal': final_signal,
            'tenkan_kijun_cross': cross_signal
        }

    def _calculate_trend(self, close: np.ndarray, ma: Dict) -> Dict:
        """计算趋势"""
        if len(close) < 20 or not ma:
            return {'strength': 50, 'direction': 'neutral'}

        # 价格相对于均线的位置
        price_position = (close[-1] - ma.get('ma20', close[-1])) / (ma.get('ma20', close[-1]) + 1e-10) * 100

        # 均线排列
        ma_trend = ma.get('trend', 'neutral')

        if ma_trend == 'strong_bullish':
            strength = min(100, 50 + abs(price_position) * 2)
            direction = 'up'
        elif ma_trend == 'strong_bearish':
            strength = min(100, 50 + abs(price_position) * 2)
            direction = 'down'
        elif ma_trend == 'bullish':
            strength = min(80, 50 + price_position)
            direction = 'up'
        elif ma_trend == 'bearish':
            strength = min(80, 50 - price_position)
            direction = 'down'
        else:
            strength = 50
            direction = 'neutral'

        return {'strength': float(strength), 'direction': direction}

    def _calculate_momentum(self, close: np.ndarray, macd: Dict) -> Dict:
        """计算动量"""
        if len(close) < 20:
            return {'value': 0, 'direction': 'neutral'}

        # 价格变化率
        price_change = (close[-1] - close[-20]) / (close[-20] + 1e-10) * 100

        # MACD方向
        macd_direction = macd.get('direction', 'neutral')

        if macd_direction == 'bullish' and price_change > 0:
            value = min(100, 50 + price_change * 2)
            direction = 'up'
        elif macd_direction == 'bearish' and price_change < 0:
            value = min(100, 50 + abs(price_change) * 2)
            direction = 'down'
        elif price_change > 5:
            value = 50 + price_change
            direction = 'up'
        elif price_change < -5:
            value = 50 - abs(price_change)
            direction = 'down'
        else:
            value = 50
            direction = 'neutral'

        return {'value': float(value), 'direction': direction}

    def _calculate_volatility(self, price_data: pd.DataFrame) -> Dict:
        """计算波动性"""
        if len(price_data) < 20:
            return {'value': 50, 'level': 'normal'}

        returns = price_data['close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率

        if volatility < 15:
            level = 'low'
        elif volatility < 30:
            level = 'normal'
        elif volatility < 50:
            level = 'high'
        else:
            level = 'very_high'

        # 标准化到0-100
        value = min(100, max(0, volatility * 2))

        return {'value': float(value), 'level': level}

    def _calculate_composite(
        self, rsi: float, macd: Dict, kdj: Dict, boll: Dict,
        ma: Dict, volume_ratio: float, close: np.ndarray,
        weights: List[float] = None,
        ma9_crossover: Dict = None,
        adx: Dict = None,
        supertrend: Dict = None,
        vwap: Dict = None,
        ichimoku: Dict = None,
        nine_turns: Dict = None,
        use_dynamic_weights: bool = True
    ) -> Dict:
        """
        计算综合评分 (0-100)
        返回: {'score': float, 'breakdown': [{'name': str, 'value': float, 'contribution': float, 'reason': str}, ...]}
        """
        # 动态权重：根据市场状态选择
        if use_dynamic_weights and adx is not None:
            adx_value = adx.get('adx', 0)
            if adx_value >= 30:
                # 强趋势
                weights = [0.10, 0.25, 0.10, 0.05, 0.10, 0.00, 0.10, 0.15, 0.15, 0.05, 0.05, 0.05]
                regime = 'strong_trend'
            elif adx_value >= 25:
                # 弱趋势
                weights = [0.15, 0.20, 0.10, 0.10, 0.05, 0.00, 0.05, 0.10, 0.10, 0.05, 0.05, 0.05]
                regime = 'weak_trend'
            else:
                # 震荡
                weights = [0.20, 0.15, 0.05, 0.15, 0.05, 0.00, 0.05, 0.10, 0.10, 0.10, 0.10, 0.05]
                regime = 'range_bound'
        elif weights is None:
            # 默认趋势追踪权重（优化后：MACD 30% + ADX 25% + Supertrend 15%）
            weights = [0.05, 0.30, 0.10, 0.00, 0.05, 0.00, 0.05, 0.20, 0.15, 0.05, 0.05, 0.05]
            regime = 'unknown'

        score = 50.0  # 基准分
        breakdown = []

        # RSI评分：趋势追踪逻辑
        # RSI > 60 强势区间，越高越偏多（趋势强）
        # RSI < 40 弱势区间，越低越偏空（趋势弱）
        # RSI 40-60 中性区间，50为中性
        if rsi > 70:
            rsi_score = 50  # 强势，持续看涨
            rsi_reason = f'RSI={rsi:.1f}强势，趋势延续'
        elif rsi > 60:
            rsi_score = 30  # 偏强
            rsi_reason = f'RSI={rsi:.1f}偏强，趋势偏多'
        elif rsi < 30:
            rsi_score = -50  # 弱势，持续看跌
            rsi_reason = f'RSI={rsi:.1f}弱势，趋势走弱'
        elif rsi < 40:
            rsi_score = -30  # 偏弱
            rsi_reason = f'RSI={rsi:.1f}偏弱，趋势偏空'
        else:
            rsi_score = (rsi - 50) * 1.5  # 中性区间
            rsi_reason = f'RSI={rsi:.1f}，中性区间'
        rsi_contribution = rsi_score * weights[0]
        score += rsi_contribution
        breakdown.append({
            'name': 'RSI',
            'value': round(rsi, 2),
            'score': round(rsi_score, 2),
            'contribution': round(rsi_contribution, 2),
            'weight': f'{int(weights[0]*100)}%',
            'reason': rsi_reason
        })

        # MACD评分：基于histogram强度，不依赖方向标签
        # histogram > 0 且值越大越强，< 0 且值越小越弱
        # 用 *2 替代 *100，避免大值直接截断到±50
        hist = macd['histogram']
        if hist > 0:
            macd_score = min(50, hist * 2)
            macd_reason = f'MACD histogram={hist:.4f}，看多'
        elif hist < 0:
            macd_score = -min(50, abs(hist) * 2)
            macd_reason = f'MACD histogram={hist:.4f}，看空'
        else:
            macd_score = 0
            macd_reason = 'MACD histogram=0，中性'
        macd_contribution = macd_score * weights[1]
        score += macd_contribution
        breakdown.append({
            'name': 'MACD',
            'value': round(macd['value'], 4),
            'score': round(macd_score, 2),
            'contribution': round(macd_contribution, 2),
            'weight': f'{int(weights[1]*100)}%',
            'reason': macd_reason
        })

        # KDJ评分：趋势追踪逻辑
        # K > 80 强势，趋势强；K < 20 弱势，趋势弱
        kdj_reason = ''
        if kdj['j'] > 100:
            kdj_score = 40  # 强势延续
            kdj_reason = f'J值强势({kdj["j"]:.1f}>100)，趋势强劲'
        elif kdj['k'] > 80 or kdj['j'] > 80:
            kdj_score = 25  # 偏强
            kdj_reason = f'KDJ强势区域(K={kdj["k"]:.1f}, J={kdj["j"]:.1f}>80)，趋势偏多'
        elif kdj['k'] < 20 or kdj['j'] < 0:
            kdj_score = -30  # 偏弱
            kdj_reason = f'KDJ弱势区域(K={kdj["k"]:.1f}, J={kdj["j"]:.1f})，趋势偏空'
        else:
            kdj_score = (kdj['k'] - 50) * 0.8
            kdj_reason = f'KDJ中性区域(K={kdj["k"]:.1f}, J={kdj["j"]:.1f})'
        kdj_contribution = kdj_score * weights[2]
        score += kdj_contribution
        breakdown.append({
            'name': 'KDJ',
            'value': round(kdj['k'], 2),
            'score': round(kdj_score, 2),
            'contribution': round(kdj_contribution, 2),
            'weight': f'{int(weights[2]*100)}%',
            'reason': kdj_reason
        })

        # 布林带评分：趋势追踪逻辑
        # position > 80 强势突破（看多），< 20 弱势（看空）
        if boll['position'] > 90:
            boll_score = 35  # 强势突破
            boll_reason = f'股价突破上轨(位置={boll["position"]:.1f}%)，强势信号'
        elif boll['position'] > 80:
            boll_score = 20  # 偏强
            boll_reason = f'股价在上轨附近(位置={boll["position"]:.1f}%)，偏多'
        elif boll['position'] < 10:
            boll_score = -35  # 弱势
            boll_reason = f'股价跌破下轨(位置={boll["position"]:.1f}%)，弱势信号'
        elif boll['position'] < 20:
            boll_score = -20  # 偏弱
            boll_reason = f'股价在下轨附近(位置={boll["position"]:.1f}%)，偏空'
        else:
            boll_score = (boll['position'] - 50) * 0.4
            boll_reason = f'股价在布林带中部(位置={boll["position"]:.1f}%)'
        boll_contribution = boll_score * weights[3]
        score += boll_contribution
        breakdown.append({
            'name': '布林带',
            'value': round(boll['position'], 2),
            'score': round(boll_score, 2),
            'contribution': round(boll_contribution, 2),
            'weight': f'{int(weights[3]*100)}%',
            'reason': boll_reason
        })

        # 均线趋势评分
        ma_trend = ma.get('trend', 'neutral')
        if ma_trend == 'strong_bullish':
            ma_score = 40
            ma_reason = '均线多头排列(5>10>20>60)，强烈看多'
        elif ma_trend == 'strong_bearish':
            ma_score = -40
            ma_reason = '均线空头排列(5<10<20<60)，强烈看空'
        elif ma_trend == 'bullish':
            ma_score = 20
            ma_reason = '均线偏多(5>20)'
        elif ma_trend == 'bearish':
            ma_score = -20
            ma_reason = '均线偏空(5<20)'
        else:
            ma_score = 0
            ma_reason = '均线混乱，无明显趋势'
        ma_contribution = ma_score * weights[4]
        score += ma_contribution
        breakdown.append({
            'name': '均线',
            'value': ma.get('ma5', 0),
            'score': round(ma_score, 2),
            'contribution': round(ma_contribution, 2),
            'weight': f'{int(weights[4]*100)}%',
            'reason': ma_reason
        })

        # 量比评分
        if volume_ratio > 2:
            vol_score = 15
            vol_reason = f'量能放大(比={volume_ratio:.2f})，资金积极'
        elif volume_ratio < 0.5:
            vol_score = -10
            vol_reason = f'量能萎缩(比={volume_ratio:.2f})，观望情绪'
        else:
            vol_score = 0
            vol_reason = f'量能正常(比={volume_ratio:.2f})'
        vol_contribution = vol_score * weights[5]
        score += vol_contribution
        breakdown.append({
            'name': '量比',
            'value': round(volume_ratio, 2),
            'score': round(vol_score, 2),
            'contribution': round(vol_contribution, 2),
            'weight': f'{int(weights[5]*100)}%',
            'reason': vol_reason
        })

        # 9日均线转向评分
        ma9_cross = ma9_crossover if ma9_crossover else {'cross': 'none', 'direction': 'neutral'}
        if ma9_cross['cross'] == 'golden':
            ma9_score = 30
            ma9_reason = 'MA9金叉，均线由降转升，看多'
        elif ma9_cross['cross'] == 'death':
            ma9_score = -30
            ma9_reason = 'MA9死叉，均线由升转降，看空'
        else:
            ma9_score = 0
            ma9_reason = 'MA9无转向信号'
        ma9_contribution = ma9_score * weights[6]
        score += ma9_contribution
        breakdown.append({
            'name': 'MA9转向',
            'value': ma9_cross.get('ma9', 0),
            'score': round(ma9_score, 2),
            'contribution': round(ma9_contribution, 2),
            'weight': f'{int(weights[6]*100)}%',
            'reason': ma9_reason
        })

        # ADX评分：趋势强度指标
        # ADX > 25 确认有趋势，配合方向给出评分
        # ADX < 20 趋势不明显，减小其他指标权重效果
        adx_data = adx if adx else {'adx': 0, 'plus_di': 0, 'minus_di': 0, 'trend_strength': 'weak'}
        adx_val = adx_data.get('adx', 0)
        plus_di = adx_data.get('plus_di', 0)
        minus_di = adx_data.get('minus_di', 0)
        trend_dir = adx_data.get('trend_direction', 'weak')

        if adx_val >= 30:
            # 强趋势
            if trend_dir in ['strong_bullish', 'bullish']:
                adx_score = 40 if trend_dir == 'strong_bullish' else 25
                adx_reason = f'ADX={adx_val:.1f}强势+DI>{minus_di:.1f}，趋势明确看多'
            elif trend_dir in ['strong_bearish', 'bearish']:
                adx_score = -40 if trend_dir == 'strong_bearish' else -25
                adx_reason = f'ADX={adx_val:.1f}强势-DI>{plus_di:.1f}，趋势明确看空'
            else:
                adx_score = 0
                adx_reason = f'ADX={adx_val:.1f}，趋势方向不明'
        elif adx_val >= 20:
            # 中等趋势
            if trend_dir in ['strong_bullish', 'bullish']:
                adx_score = 15
                adx_reason = f'ADX={adx_val:.1f}+DI>{minus_di:.1f}，偏多'
            elif trend_dir in ['strong_bearish', 'bearish']:
                adx_score = -15
                adx_reason = f'ADX={adx_val:.1f}-DI>{plus_di:.1f}，偏空'
            else:
                adx_score = 0
                adx_reason = f'ADX={adx_val:.1f}，趋势不明显'
        else:
            adx_score = -10
            adx_reason = f'ADX={adx_val:.1f}<20，趋势极弱/盘整，谨慎操作'
        adx_contribution = adx_score * weights[7]
        score += adx_contribution
        breakdown.append({
            'name': 'ADX',
            'value': round(adx_val, 2),
            'score': round(adx_score, 2),
            'contribution': round(adx_contribution, 2),
            'weight': f'{int(weights[7]*100)}%',
            'reason': adx_reason
        })

        # Supertrend评分：趋势跟踪指标
        st_data = supertrend if supertrend else {'trend': 'neutral', 'signal': 'none'}
        st_trend = st_data.get('trend', 'neutral')
        st_signal = st_data.get('signal', 'none')

        if st_signal == 'buy':
            st_score = 40
            st_reason = 'Supertrend由跌转涨，买入信号'
        elif st_signal == 'sell':
            st_score = -40
            st_reason = 'Supertrend由涨转跌，卖出信号'
        elif st_trend == 'up':
            st_score = 20
            st_reason = f'Supertrend上涨趋势，ST={st_data.get("supertrend_value", 0):.2f}'
        elif st_trend == 'down':
            st_score = -20
            st_reason = f'Supertrend下跌趋势，ST={st_data.get("supertrend_value", 0):.2f}'
        else:
            st_score = 0
            st_reason = 'Supertrend中性'
        st_contribution = st_score * weights[8]
        score += st_contribution
        breakdown.append({
            'name': 'Supertrend',
            'value': round(st_data.get('supertrend_value', 0), 2),
            'score': round(st_score, 2),
            'contribution': round(st_contribution, 2),
            'weight': f'{int(weights[8]*100)}%',
            'reason': st_reason
        })

        # VWAP评分：趋势追踪逻辑
        # price > VWAP 看多（趋势多），price < VWAP 看空（趋势空）
        vwap_data = vwap if vwap else {'vwap': 0, 'price_vs_vwap': 0, 'signal': 'neutral'}
        vwap_signal = vwap_data.get('signal', 'neutral')
        price_vs_vwap = vwap_data.get('price_vs_vwap', 0)

        if vwap_signal == 'far_above':
            vwap_score = 25  # 价格远高于VWAP，强势多头信号
            vwap_reason = f'价格远高于VWAP+{price_vs_vwap:.1f}%，强势看多'
        elif vwap_signal == 'far_below':
            vwap_score = -25  # 价格远低于VWAP，弱势空头信号
            vwap_reason = f'价格远低于VWAP{price_vs_vwap:.1f}%，弱势看空'
        elif vwap_signal == 'above':
            vwap_score = 15  # 价格略高于VWAP，偏多
            vwap_reason = f'价格高于VWAP+{price_vs_vwap:.1f}%，偏多'
        elif vwap_signal == 'below':
            vwap_score = -15  # 价格略低于VWAP，偏空
            vwap_reason = f'价格低于VWAP{price_vs_vwap:.1f}%，偏空'
        else:
            vwap_score = 0
            vwap_reason = f'价格接近VWAP，中性'
        vwap_contribution = vwap_score * weights[9]
        score += vwap_contribution
        breakdown.append({
            'name': 'VWAP',
            'value': round(vwap_data.get('vwap', 0), 2),
            'score': round(vwap_score, 2),
            'contribution': round(vwap_contribution, 2),
            'weight': f'{int(weights[9]*100)}%',
            'reason': vwap_reason
        })

        # Ichimoku Cloud评分：综合云带和转换线/基准线交叉信号
        ichi_data = ichimoku if ichimoku else {'cloud_signal': 'neutral', 'tenkan_kijun_cross': 'neutral'}
        cloud_signal = ichi_data.get('cloud_signal', 'neutral')
        tk_cross = ichi_data.get('tenkan_kijun_cross', 'neutral')

        # 综合判断
        if cloud_signal == 'bullish' and tk_cross == 'bullish':
            ichi_score = 35
            ichi_reason = 'Ichimoku云带上移+转换线/基准线金叉，强烈看多'
        elif cloud_signal == 'bearish' and tk_cross == 'bearish':
            ichi_score = -35
            ichi_reason = 'Ichimoku云带下移+转换线/基准线死叉，强烈看空'
        elif cloud_signal == 'bullish':
            ichi_score = 20
            ichi_reason = 'Ichimoku价格在云带上，偏多'
        elif cloud_signal == 'bearish':
            ichi_score = -20
            ichi_reason = 'Ichimoku价格在云带下，偏空'
        elif tk_cross == 'bullish':
            ichi_score = 15
            ichi_reason = 'Ichimoku转换线/基准线金叉，短期偏多'
        elif tk_cross == 'bearish':
            ichi_score = -15
            ichi_reason = 'Ichimoku转换线/基准线死叉，短期偏空'
        else:
            ichi_score = 0
            ichi_reason = 'Ichimoku信号混乱，中性'
        ichi_contribution = ichi_score * weights[10]
        score += ichi_contribution
        breakdown.append({
            'name': 'Ichimoku',
            'value': round(ichi_data.get('tenkan', 0), 2),
            'score': round(ichi_score, 2),
            'contribution': round(ichi_contribution, 2),
            'weight': f'{int(weights[10]*100)}%',
            'reason': ichi_reason
        })

        # 九转指标（TD Sequential）评分
        # 九转序列达到第9根是潜在的转折点信号
        nt_data = nine_turns if nine_turns else {'count': 0, 'phase': 'neutral', 'signal': 'none', 'is_complete': False}
        nt_signal = nt_data.get('signal', 'none')
        nt_phase = nt_data.get('phase', 'neutral')
        nt_count = nt_data.get('count', 0)

        if nt_signal == 'buy':
            nt_score = 40
            nt_reason = f'TD买入计数9完成，第9根K线买入信号'
        elif nt_signal == 'sell':
            nt_score = -40
            nt_reason = f'TD卖出计数9完成，第9根K线卖出信号'
        elif nt_phase == 'buy' and nt_count >= 7:
            nt_score = 20
            nt_reason = f'TD买入计数{nt_count}，接近完成，可能反转'
        elif nt_phase == 'sell' and nt_count >= 7:
            nt_score = -20
            nt_reason = f'TD卖出计数{nt_count}，接近完成，可能反转'
        elif nt_phase == 'buy':
            nt_score = 10
            nt_reason = f'TD买入计数{nt_count}，偏多'
        elif nt_phase == 'sell':
            nt_score = -10
            nt_reason = f'TD卖出计数{nt_count}，偏空'
        else:
            nt_score = 0
            nt_reason = f'九转无信号，计数{nt_count}'
        nt_contribution = nt_score * weights[11]
        score += nt_contribution
        breakdown.append({
            'name': '九转',
            'value': nt_count,
            'score': round(nt_score, 2),
            'contribution': round(nt_contribution, 2),
            'weight': f'{int(weights[11]*100)}%',
            'reason': nt_reason
        })

        final_score = max(0, min(100, score))

        return {
            'score': final_score,
            'breakdown': breakdown
        }

    def _composite_to_signal(self, composite: float) -> str:
        """综合评分转换为信号"""
        if composite >= 70:
            return 'strong_buy'
        elif composite >= 55:
            return 'buy'
        elif composite <= 30:
            return 'strong_sell'
        elif composite <= 45:
            return 'sell'
        else:
            return 'hold'
