"""第二轮因子筛选的信号构造（预注册：docs/FACTOR_PREREG_2026-11-06.md）。

输入为日期×个股矩阵（全部复权基底）：close/open/high/low/volume/amount。
所有信号定义为"值大 = 相对看多"；超卖类（KDJ/%B/WR 等）取负号对齐。
纯函数、无 IO；口径与 indicators/ 库单票交叉验证（tests/test_technical_signals.py）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd



def _ema(df: pd.DataFrame, span: int) -> pd.DataFrame:
    return df.ewm(span=span, adjust=False).mean()


def _wilder(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.ewm(alpha=1.0 / n, adjust=False).mean()


def _rolling_mad(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """逐列滑窗平均绝对偏差（CCI 标准口径；矩阵化实现避免逐窗 Python 调用）"""
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    values = df.to_numpy(dtype=float)
    for j in range(values.shape[1]):
        col = values[:, j]
        valid = np.where(np.isfinite(col))[0]
        if len(valid) < n:
            continue
        # 滑窗视图后向量化 MAD
        idx = valid[np.isin(valid, np.arange(n - 1, len(col)))]
        if len(idx) == 0:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(col, n)
        means = windows.mean(axis=1)
        out.iloc[n - 1:, j] = np.abs(windows - means[:, None]).mean(axis=1)
    return out



def _true_range(high: pd.DataFrame, low: pd.DataFrame,
                close: pd.DataFrame) -> pd.DataFrame:
    """逐元素三向最大 TR（矩阵场景；单票惯用的 concat(axis=1).max(axis=1)
    在 dates×symbols 输入下会退化为每日全市场单一最大值——2026-11-06
    由交叉验证测试抓出，D9/D10/D16 受影响一并修复）"""
    prev_close = close.shift(1)
    a = (high - low).to_numpy(dtype=float)
    b = (high - prev_close).abs().to_numpy(dtype=float)
    c = (low - prev_close).abs().to_numpy(dtype=float)
    # fmax 跳过 NaN（与单票 max(axis=1) 语义一致：prev_close 缺失日取 high-low）
    return pd.DataFrame(np.fmax(np.fmax(a, b), c),
                        index=high.index, columns=high.columns)

def build_base_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)


# ---------------- A 组：改进价格构造 ----------------

def a1_risk_adj_reversal(close: pd.DataFrame) -> pd.DataFrame:
    r5 = close / close.shift(5) - 1
    sd5 = build_base_returns(close).rolling(5, min_periods=5).std()
    return -(r5 / sd5)


def a2_idio_reversal(close: pd.DataFrame) -> pd.DataFrame:
    """特质反转：个股日收益对等权市场收益的滚动60日β回归残差，取5日和的负值"""
    r = build_base_returns(close)
    rm = r.mean(axis=1)
    w = 60
    mean_xy = r.mul(rm, axis=0).rolling(w, min_periods=w).mean()
    mean_x = r.rolling(w, min_periods=w).mean()
    mean_m = rm.rolling(w, min_periods=w).mean()
    var_m = rm.rolling(w, min_periods=w).var()
    beta = ((mean_xy - mean_x.mul(mean_m, axis=0))
            .div(var_m, axis=0)).where(var_m > 0)
    resid = r.sub(beta.mul(rm, axis=0))
    return -resid.rolling(5, min_periods=5).sum()


def a3_quiet_reversal(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    r5 = close / close.shift(5) - 1
    vr = volume.rolling(5, min_periods=5).mean() / volume.rolling(20, min_periods=20).mean()
    return -(r5.where(vr <= 1.15))          # 缩量桶；放量桶为对照不入检验


def a4_near_52w_high(close: pd.DataFrame) -> pd.DataFrame:
    return close / close.rolling(250, min_periods=250).max()


def a5_overnight_share(close: pd.DataFrame, open_px: pd.DataFrame) -> pd.DataFrame:
    """隔夜占比的稳定形式：on/(|on|+|intra|) ∈ [-1,1]（预注册"占比"的除零安全版，
    保持方向语义：隔夜贡献越占优值越大）"""
    overnight = (open_px / close.shift(1) - 1).rolling(20, min_periods=20).sum()
    intraday = (close / open_px - 1).rolling(20, min_periods=20).sum()
    denom = overnight.abs() + intraday.abs()
    return (overnight / denom).where(denom > 0)


def a6_vol_term_structure(close: pd.DataFrame) -> pd.DataFrame:
    r = build_base_returns(close)
    sd5 = r.rolling(5, min_periods=5).std()
    sd60 = r.rolling(60, min_periods=60).std()
    return -(sd5 / sd60)                    # 低短期恐慌（相对基线）= 看多


# ---------------- RPS ----------------

def rps(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """欧奈尔 RPS：N 日收益的横截面百分位（0-1，逐日 rank）"""
    ret = close / close.shift(window) - 1
    return ret.rank(axis=1, pct=True)


# ---------------- D 组：经典技术指标（20 个，横截面化=原始值，rank 由统计层统一做） ----------------

def d1_macd_level(close: pd.DataFrame) -> pd.DataFrame:
    dif = _ema(close, 12) - _ema(close, 26)
    dea = _ema(dif, 9)
    return (dif - dea) / close


def d2_macd_slope(close: pd.DataFrame) -> pd.DataFrame:
    dif = _ema(close, 12) - _ema(close, 26)
    hist = dif - _ema(dif, 9)
    return hist.diff(5)                     # 柱的5日变化量


def _kdj_j(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    h9 = high.rolling(9, min_periods=9).max()
    l9 = low.rolling(9, min_periods=9).min()
    rsv = ((close - l9) / (h9 - l9) * 100).where(h9 > l9)
    k = rsv.ewm(alpha=1.0 / 3, adjust=False).mean()
    d = k.ewm(alpha=1.0 / 3, adjust=False).mean()
    return 3 * k - 2 * d


def d3_kdj_j(high, low, close) -> pd.DataFrame:
    return -_kdj_j(high, low, close)        # 低 J = 超卖 = 看多


def _boll(close: pd.DataFrame, n: int = 20, k: float = 2.0):
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std()
    return mid, mid + k * sd, mid - k * sd


def d4_boll_pctb(close: pd.DataFrame) -> pd.DataFrame:
    _, upper, lower = _boll(close)
    width = upper - lower
    return -((close - lower) / width).where(width > 0)   # 低 %B = 超卖


def d5_boll_bandwidth(close: pd.DataFrame) -> pd.DataFrame:
    mid, upper, lower = _boll(close)
    return -((upper - lower) / mid)        # 低带宽（Squeeze 蓄势）= 看多


def d6_rsi14(close: pd.DataFrame) -> pd.DataFrame:
    """原始 RSI（方向待实证：高值=强动量解释 / 高值=超买解释，报告期决定）"""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    up.iloc[0, :] = 0.0        # 首日未知涨跌按 0（与 indicators 库口径一致）
    down.iloc[0, :] = 0.0
    avg_up = _wilder(up, 14)
    avg_down = _wilder(down, 14)
    return (100 * avg_up / (avg_up + avg_down)).where((avg_up + avg_down) > 0)


def d7_expma_gap(close: pd.DataFrame) -> pd.DataFrame:
    return (_ema(close, 12) - _ema(close, 50)) / close


def d8_bbi(close: pd.DataFrame) -> pd.DataFrame:
    bbi = sum(close.rolling(n, min_periods=n).mean() for n in (3, 6, 12, 24)) / 4
    return (close - bbi) / close


def _dmi(high, low, close):
    up = high.diff()
    dn = -low.diff()
    tr = _true_range(high, low, close)
    plus_dm = pd.DataFrame(np.where((up > dn) & (up > 0), up, np.nan),
                           index=high.index, columns=high.columns)
    minus_dm = pd.DataFrame(np.where((dn > up) & (dn > 0), dn, np.nan),
                            index=high.index, columns=high.columns)
    atr = _wilder(tr, 14)
    plus_di = 100 * _wilder(plus_dm.fillna(0), 14) / atr.where(atr > 0)
    minus_di = 100 * _wilder(minus_dm.fillna(0), 14) / atr.where(atr > 0)
    return plus_di, minus_di


def d9_dmi_diff(high, low, close) -> pd.DataFrame:
    plus_di, minus_di = _dmi(high, low, close)
    return plus_di - minus_di


def d10_adx(high, low, close) -> pd.DataFrame:
    plus_di, minus_di = _dmi(high, low, close)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).where((plus_di + minus_di) > 0) * 100
    return _wilder(dx, 14)


def d11_bias20(close: pd.DataFrame) -> pd.DataFrame:
    ma = close.rolling(20, min_periods=20).mean()
    return (close - ma) / ma


def d12_ma_alignment(close: pd.DataFrame) -> pd.DataFrame:
    ma5 = close.rolling(5).mean(); ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
    return ((ma5 > ma10).astype(float) + (ma10 > ma20) + (ma20 > ma60) + (ma5 > ma60))


def d13_wr14(high, low, close) -> pd.DataFrame:
    """威廉 %R 口径：值高=收盘贴近14日最低=超卖=看多（方向经交叉验证修正）"""
    h14 = high.rolling(14, min_periods=14).max()
    l14 = low.rolling(14, min_periods=14).min()
    wr = (h14 - close) / (h14 - l14) * 100
    return wr.where(h14 > l14)              # 高 %R = 超卖 = 看多


def d14_cci14(high, low, close) -> pd.DataFrame:
    tp = (high + low + close) / 3
    sma = tp.rolling(14, min_periods=14).mean()
    mad = _rolling_mad(tp, 14)
    return (tp - sma) / (0.015 * mad).where(mad > 0)


def d15_psy12(close: pd.DataFrame) -> pd.DataFrame:
    return -(close.diff() > 0).astype(float).rolling(12, min_periods=12).mean() * 100


def d16_rel_atr(high, low, close) -> pd.DataFrame:
    tr = _true_range(high, low, close)
    return -(_wilder(tr, 14) / close)       # 低相对 ATR = 低波动


def d17_obv_slope(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    obv = (np.sign(close.diff()) * volume).cumsum()
    vol20 = volume.rolling(20, min_periods=20).mean()
    return (obv.diff(5) / vol20).where(vol20 > 0)


def d18_volume_ratio(volume: pd.DataFrame) -> pd.DataFrame:
    return volume.rolling(5, min_periods=5).mean() / volume.rolling(20, min_periods=20).mean()


def d19_mfi14(high, low, close, volume) -> pd.DataFrame:
    tp = (high + low + close) / 3
    rmf = tp * volume
    direction = tp.diff()
    pos = rmf.where(direction > 0, 0.0).rolling(14, min_periods=14).sum()
    neg = rmf.where(direction < 0, 0.0).rolling(14, min_periods=14).sum()
    return (100 * pos / (pos + neg)).where((pos + neg) > 0)


def d20_vwap_dev(close, high, low, volume) -> pd.DataFrame:
    tp = (high + low + close) / 3
    pv = (tp * volume).rolling(20, min_periods=20).sum()
    vv = volume.rolling(20, min_periods=20).sum()
    return (close / (pv / vv).where(vv > 0) - 1)


# ---------------- 汇总入口 ----------------

def build_all_signals(m: dict) -> dict[str, pd.DataFrame]:
    """m = {'close','open','high','low','volume','amount'}（复权基底矩阵）。
    返回 34 个信号（A6×2归一化在统计层处理，构造层只出原始值）。"""
    c, o, h, l, v = m['close'], m['open'], m['high'], m['low'], m['volume']
    r120, r250 = rps(c, 120), rps(c, 250)
    j_raw = _kdj_j(h, l, c)
    hist = d1_macd_level(c) * c
    _, upper, lower = _boll(c)
    pctb = ((c - lower) / (upper - lower)).where((upper - lower) > 0)
    bw = ((upper - lower) / _boll(c)[0])
    r5 = c / c.shift(5) - 1

    signals: dict[str, pd.DataFrame] = {
        'A1_risk_adj_reversal': a1_risk_adj_reversal(c),
        'A2_idio_reversal': a2_idio_reversal(c),
        'A3_quiet_reversal': a3_quiet_reversal(c, v),
        'A4_near_52w_high': a4_near_52w_high(c),
        'A5_overnight_share': a5_overnight_share(c, o),
        'A6_vol_term': a6_vol_term_structure(c),
        'RPS_120': r120,
        'RPS_250': r250,
        'D1_macd_level': d1_macd_level(c),
        'D2_macd_slope': d2_macd_slope(c),
        'D3_kdj_j': -j_raw,
        'D4_boll_pctb': -pctb,
        'D5_boll_bandwidth': -bw,
        'D6_rsi14': d6_rsi14(c),
        'D7_expma_gap': d7_expma_gap(c),
        'D8_bbi': d8_bbi(c),
        'D9_dmi_diff': d9_dmi_diff(h, l, c),
        'D10_adx': d10_adx(h, l, c),
        'D11_bias20': d11_bias20(c),
        'D12_ma_align': d12_ma_alignment(c),
        'D13_wr14': d13_wr14(h, l, c),
        'D14_cci14': d14_cci14(h, l, c),
        'D15_psy12': d15_psy12(c),
        'D16_rel_atr': d16_rel_atr(h, l, c),
        'D17_obv_slope': d17_obv_slope(c, v),
        'D18_volume_ratio': d18_volume_ratio(v),
        'D19_mfi14': d19_mfi14(h, l, c, v),
        'D20_vwap_dev': d20_vwap_dev(c, h, l, v),
        # 联合（条件池内信号；池外 NaN）
        'J1_rps_kdj_oversold': (-r5).where((r120 >= 0.7) & (j_raw <= 20)),
        'J2_rps_macd_up': r120.where((r120 >= 0.7) & (hist > 0)),
        'J3_rps_lower_band': (-r5).where((r120 >= 0.7) & (pctb <= 0.1)),
        'J4_kdj_squeeze': (-j_raw).where((j_raw <= 20) & (bw.rank(axis=1, pct=True) <= 0.3)),
    }
    return signals


def zscore(mx: pd.DataFrame) -> pd.DataFrame:
    return mx.sub(mx.mean(axis=1), axis=0).div(mx.std(axis=1) + 1e-12, axis=0)


def rank_pct(mx: pd.DataFrame) -> pd.DataFrame:
    return mx.rank(axis=1, pct=True)
