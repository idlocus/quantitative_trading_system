"""第二轮因子筛选主流程（预注册：docs/FACTOR_PREREG_2026-11-06.md，32 个检验）。

复用第一轮 signal_screen 的面板加载/IC/HAC 框架（同数据同口径），
扩展读取原始 OHLC×复权因子构造复权 OHLC；新增 FDR 分层校正、
冗余矩阵、D 组残差增量检验与联合信号。证据输出到
ai-created/research/factor_screen2_2026-11-06/。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from data.preparation import atomic_json
from research import technical_signals as ts
from research.signal_screen import (daily_statistics, forward_labels,
                                    hac_summary, load_panel, mean_or_none,
                                    verified_frame)

PROTOCOL = {
    'start': '2016-01-01', 'train_end': '2020-12-31', 'validation_end': '2023-12-31',
    'horizon_sessions': 5, 'hac_lags': 5, 'minimum_cross_section': 30,
    'minimum_history_sessions': 120, 'liquidity_window': 20,
    'minimum_median_daily_amount_local_currency': 20_000_000,
    'exclude_a_st_types': ['S', 'X', 'Y', 'P'],
    'fdr_q': 0.10, 'cluster_corr': 0.7,
}
OUTPUT = Path('data_cache/preparation')
DEST = Path('ai-created/research/factor_screen2_2026-11-06')


def valuation_input(path, evidence):
    import hashlib
    with path.open('rb') as stream:
        evidence[path.stem + '_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0)
        return pd.read_parquet(stream)


def load_ohlc_panel(protocol: dict) -> tuple[dict, dict, dict]:
    """在第一轮 panel（adjclose/amount/usable）之上加载复权 OHLC + volume"""
    base, evidence = load_panel(OUTPUT, 'a', protocol)
    start, end = pd.Timestamp(protocol['start']), pd.Timestamp(protocol['validation_end'])
    oracle = OUTPUT / 'oracle'
    source_manifest = json.loads((oracle / 'normalized_a/manifest.json').read_text(encoding='utf-8'))
    dates = base['price'].index
    symbols = base['price'].columns
    opens = np.full((len(dates), len(symbols)), np.nan)
    highs = np.full_like(opens, np.nan)
    lows = np.full_like(opens, np.nan)
    volumes = np.full_like(opens, np.nan)
    for name, source in sorted(source_manifest['files'].items()):
        year, month = map(int, name.split('/'))
        if pd.Timestamp(year=year, month=month, day=1) > end or year < start.year:
            continue
        if source.get('sha256') != evidence['source_files'].get(name):
            raise ValueError('Bar inputs changed between research passes')
        path = oracle / 'normalized_a' / (name + '.parquet')
        frame = verified_frame(path, source, columns=['date', 'symbol', 'open', 'high',
                                                      'low', 'volume', 'adjfactor'])
        ri, ci = dates.get_indexer(frame['date']), symbols.get_indexer(frame['symbol'])
        sel = (ri >= 0) & (ci >= 0)
        factor = frame['adjfactor'].to_numpy(dtype=float)
        opens[ri[sel], ci[sel]] = frame['open'].to_numpy(dtype=float)[sel] * factor[sel]
        highs[ri[sel], ci[sel]] = frame['high'].to_numpy(dtype=float)[sel] * factor[sel]
        lows[ri[sel], ci[sel]] = frame['low'].to_numpy(dtype=float)[sel] * factor[sel]
        volumes[ri[sel], ci[sel]] = frame['volume'].to_numpy(dtype=float)[sel]
    adj = base['price']
    panel = {'close': adj,
             'open': pd.DataFrame(opens, index=dates, columns=symbols),
             'high': pd.DataFrame(highs, index=dates, columns=symbols),
             'low': pd.DataFrame(lows, index=dates, columns=symbols),
             'volume': pd.DataFrame(volumes, index=dates, columns=symbols)}
    return panel, base, evidence


def run_partitions(daily: pd.DataFrame, protocol: dict) -> dict:
    label_end = daily.index.to_series().shift(-(protocol['horizon_sessions'] + 1))
    parts = {}
    for split, lower, upper in [
            ('train', protocol['start'], protocol['train_end']),
            ('validation', (pd.Timestamp(protocol['train_end']) + pd.Timedelta(days=1)).date().isoformat(),
             protocol['validation_end'])]:
        mask = (daily.index >= lower) & (daily.index <= upper) & label_end.le(pd.Timestamp(upper))
        sel = daily.loc[mask]
        parts[split] = {**hac_summary(sel['ic'], protocol['hac_lags']),
                        'mean_paired_stocks': mean_or_none(sel['paired_stocks']),
                        'missing_forward_labels': int(sel['missing_forward_labels'].sum())}
    return parts


def cluster_by_corr(ic_corr: pd.DataFrame, threshold: float) -> list[list[str]]:
    """验证期 IC 序列相关 ≥ threshold 的连通分量聚类"""
    names = list(ic_corr.columns)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if abs(ic_corr.loc[a, b]) >= threshold:
                parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted(groups.values(), key=len, reverse=True)


def residual_ic(signal: pd.DataFrame, benchmarks: list[pd.DataFrame],
                labels: pd.DataFrame, eligible: pd.DataFrame,
                protocol: dict) -> pd.Series:
    """信号对基准集逐日横截面 OLS 正交化后的残差 IC（D 组增量检验）"""
    daily_ic = {}
    B = [b.reindex_like(signal) for b in benchmarks]
    paired_all = eligible & signal.notna() & labels.notna()
    for t in signal.index[::1]:
        mask = paired_all.loc[t]
        for b in B:
            mask &= b.loc[t].notna()
        n = int(mask.sum())
        if n < protocol['minimum_cross_section'] + 3:
            continue
        y = signal.loc[t][mask].to_numpy()
        X = np.column_stack([np.ones(n)] + [b.loc[t][mask].to_numpy() for b in B])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = pd.Series(y - X @ beta).rank(pct=True)
        lab = labels.loc[t][mask].reset_index(drop=True).rank(pct=True)
        daily_ic[t] = float(np.corrcoef(resid, lab)[0, 1])
    return pd.Series(daily_ic).sort_index()


def main() -> dict:
    protocol = PROTOCOL
    panel, base, evidence = load_ohlc_panel(protocol)
    DEST.mkdir(parents=True, exist_ok=True)
    atomic_json(DEST / 'inputs.json', evidence)
    price = base['price']
    eligible = base['usable']
    history = price.notna().rolling(protocol['minimum_history_sessions']).sum() \
        .eq(protocol['minimum_history_sessions'])
    liquid = base['amount'].rolling(protocol['liquidity_window'], min_periods=protocol['liquidity_window']) \
        .median().ge(protocol['minimum_median_daily_amount_local_currency'])
    eligible = eligible & history & liquid
    labels = forward_labels(price, protocol['horizon_sessions'])
    print(f"面板: {price.shape[0]} 日 × {price.shape[1]} 只 | 合格标的日均 {int(eligible.sum(axis=1).mean())}", flush=True)

    signals = ts.build_all_signals(panel)
    results, daily_ics = {}, {}
    for name, sig in signals.items():
        daily = daily_statistics(sig, labels, eligible, protocol['minimum_cross_section'])
        daily_ics[name] = daily['ic']
        daily.to_csv(DEST / f'{name}_daily.csv', index_label='signal_date')
        results[name] = run_partitions(daily, protocol)
        del sig
    print(f"信号构造与 IC 完成: {len(results)} 个", flush=True)

    # ---- FDR（严格口径：32 检验按验证期 HAC p 近似——用 IC 序列的 t 检验 p）----
    # p 值从验证期 HAC t 换算（正态近似双尾）——与存活判据1同口径
    from scipy import stats as sps
    pvals = {}
    for name in results:
        t_stat = results[name]['validation']['hac_t_stat']
        pvals[name] = 1.0 if t_stat is None else float(2 * sps.norm.sf(abs(t_stat)))
    names_sorted = sorted(pvals, key=pvals.get)
    p = np.array([pvals[n] for n in names_sorted])
    rejected = np.zeros(len(p), dtype=bool)
    try:
        rejected = false_discovery_control(p, method='bh') <= protocol['fdr_q']
    except Exception:
        m, k = len(p), np.arange(1, len(p) + 1)
        thresh = k * protocol['fdr_q'] / m
        passed = p <= thresh
        rejected = np.cumsum(passed[::-1])[::-1] > 0
    fdr_strict = {n: bool(r) for n, r in zip(names_sorted, rejected)}

    # ---- 冗余矩阵 + 簇代表分层视图 ----
    val = pd.DataFrame({n: daily_ics[n] for n in results})
    val = val[(val.index > protocol['train_end']) & (val.index <= protocol['validation_end'])]
    ic_corr = val.corr(min_periods=120)
    ic_corr.to_csv(DEST / 'ic_correlation_validation.csv', encoding='utf-8-sig')
    clusters = cluster_by_corr(ic_corr, protocol['cluster_corr'])
    cluster_json = [{'representative': max(c, key=lambda n: abs(results[n]['validation']['mean_ic'] or 0)),
                     'members': c} for c in clusters]
    atomic_json(DEST / 'clusters.json', cluster_json)
    reps = [c['representative'] for c in cluster_json]
    p_rep = np.array([pvals[n] for n in sorted(reps, key=lambda n_: pvals[n_])])
    rep_names = sorted(reps, key=lambda n_: pvals[n_])
    try:
        rej_rep = false_discovery_control(p_rep, method='bh') <= protocol['fdr_q']
    except Exception:
        k = np.arange(1, len(p_rep) + 1)
        passed = p_rep <= k * protocol['fdr_q'] / len(p_rep)
        rej_rep = np.cumsum(passed[::-1])[::-1] > 0
    fdr_cluster = {n: bool(r) for n, r in zip(rep_names, rej_rep)}

    # ---- D 组残差增量（对基准集：5日反转/20日低波动/RPS_120）----
    benchmarks = [-(price / price.shift(5) - 1),
                  -price.pct_change(fill_method=None).rolling(20, min_periods=20).std(),
                  ts.rps(price, 120)]
    incremental = {}
    d_names = [n for n in list(signals) if n.startswith('D')]
    keep = {n: signals.pop(n) for n in d_names}
    signals.clear()
    for name in d_names:
        ric = residual_ic(keep[name], benchmarks, labels, eligible, protocol)
        keep.pop(name, None)
        v = ric[(ric.index > protocol['train_end']) & (ric.index <= protocol['validation_end'])]
        incremental[name] = {'validation_mean_ic': float(v.mean()) if len(v) else None,
                             'validation_t': hac_summary(v, protocol['hac_lags'])['hac_t_stat'],
                             'days': int(len(v))}
    atomic_json(DEST / 'd_residual_incremental.json', incremental)

    # ---- 存活判定 ----
    survivors_strict = [n for n, ok in fdr_strict.items() if ok]
    cluster_survivors = [n for n, ok in fdr_cluster.items() if ok]
    summary = {
        'protocol': protocol, 'n_tests': len(pvals),
        'signals': results,
        'fdr_strict': fdr_strict, 'fdr_strict_survivors': survivors_strict,
        'clusters': cluster_json, 'fdr_cluster_representatives': fdr_cluster,
        'cluster_view_survivors': cluster_survivors,
        'd_residual_incremental': incremental,
        'downgrade_list': sorted(set(cluster_survivors) - set(survivors_strict)),
        'portfolio_backtest': False, 'risk_limits_evaluated': False, 'holdout_evaluated': False,
    }
    atomic_json(DEST / 'summary.json', summary)
    pd.DataFrame({'p_value': pvals}).loc[names_sorted].to_csv(
        DEST / 'p_values.csv', encoding='utf-8-sig')
    return summary


def register(verdict_note: str, label: str = 'factor_screen2_32检验') -> None:
    """登记一条试验结论（统一入口=research.overfitting.TrialRegistry）。

    审计 2026-11-19 四-a：本函数曾自带第二套写入逻辑且 label 硬编码
    'factor_screen2_32检验'——G1/P3/W/I/X 系等全部研究共用该标签（47 条
    被误标），且损坏时不拒绝写入。现在委托 TrialRegistry（事务式追加、
    损坏拒绝写入），调用方必须显式给出研究标识 label。"""
    from research.overfitting import TrialRegistry
    TrialRegistry().record(label, extra={'verdict': verdict_note})
