"""阶段二：多因子共同作用研究（预注册见 FACTOR_PREREG_2026-11-06.md 阶段二节）。

信号层：4 种合成（S1全因子定向/S2簇代表/S3正向存活/S4 IC加权）vs 最优单因子 D16。
组合层：等权 N=50 月调仓（rebalance_sessions=21）、单边25bp、双15%红线、
budget_guard/uncontrolled 两版，对照=D16 同形态。复用第一轮组合仿真组件。
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from data.preparation import atomic_json
from research import technical_signals as ts
from research.factor_screen2 import PROTOCOL, load_ohlc_panel, run_partitions
from research.portfolio_simulation import PortfolioSimulation, distributions
from research.portfolio_study import publish_study, selection_lists, snapshot_code, suspension_days
from research.signal_screen import daily_statistics, forward_labels, verified_frame

OUTPUT = Path('data_cache/preparation')
DEST = Path('ai-created/research/composite_study_2026-11-06')

POSITIVE_SURVIVORS = ['D16_rel_atr', 'D5_boll_bandwidth', 'D4_boll_pctb', 'A6_vol_term',
                      'A2_idio_reversal', 'D13_wr14', 'D15_psy12', 'A1_risk_adj_reversal',
                      'D3_kdj_j', 'A5_overnight_share']
CLUSTER_REPS = ['D7_expma_gap', 'D16_rel_atr', 'A5_overnight_share', 'A6_vol_term',
                'D2_macd_slope', 'D10_adx', 'J1_rps_kdj_oversold', 'J2_rps_macd_up',
                'J3_rps_lower_band', 'J4_kdj_squeeze']

PORTFOLIO_PROTOCOL = {
    'study': 'composite_joint_action_phase2',
    'start': '2016-01-01', 'train_end': '2020-12-31', 'validation_end': '2023-12-31',
    'holdout_start': '2024-01-01',
    'signals': [],   # 运行时填充：S1-S4 + D16_benchmark
    'initial_paper_capital': 1000000, 'capital_is_user_account_value': False,
    'positions': 50, 'gross_target': 0.95, 'rebalance_sessions': 21,
    'one_way_cost_scenarios': [0.0025], 'zero_cost_is_diagnostic_only': True,
    'costs_are_actual_broker_schedule': False, 'cash_dividend_haircut': 0.2,
    'haircut_is_statutory_tax_model': False, 'volume_participation_cap': 0.01,
    'risk_variants': ['budget_guard', 'uncontrolled'],
    'monthly_loss_limit': 0.15, 'peak_drawdown_limit': 0.15,
    'exposure_budget_adverse_shock': 0.25,
    'risk_limit_interpretation': 'fixed percentage limits confirmed 2026-11-03',
    'minimum_history_sessions': 120, 'liquidity_window': 20,
    'minimum_median_daily_amount_local_currency': 20000000,
    'horizon_sessions': 5, 'hac_lags': 5, 'minimum_cross_section': 30,
    'exclude_a_st_types': ['S', 'X', 'Y', 'P'],
    'revision': 'phase2 composite: N=50 monthly, S1-S4 vs D16 benchmark',
}


def _z(mx: pd.DataFrame) -> pd.DataFrame:
    return mx.sub(mx.mean(axis=1), axis=0).div(mx.std(axis=1) + 1e-12, axis=0)


def _weighted_mean(pairs: list):
    if not pairs:
        raise ValueError('composite member list is empty')
    """skipna 加权合成：每股用覆盖它的因子共同投票（成员缺失不传染）。
    pairs = [(z_matrix, weight), ...]；返回 sum(w*z, skipna)/sum(w, skipna)。"""
    total = None
    weight_sum = None
    for z, w in pairs:
        zw = z * w
        total = zw if total is None else total.add(zw, fill_value=0.0)
        ones = z.notna().astype(float) * abs(w)
        weight_sum = ones if weight_sum is None else weight_sum.add(ones, fill_value=0.0)
    return total / weight_sum.where(weight_sum > 0)


def compose(z_signals: dict, train_ic: dict) -> dict:
    """4 种预注册合成（纯函数可测）。z_signals 已横截面标准化；train_ic 为
    训练期锁定权重。合成语义=每股对覆盖它的因子取加权均值（2026-11-06 修正：
    初版 add 链被联合信号池外的 NaN 传染成近全空矩阵）。"""
    pairs_all, pairs_rep, pairs_pos = [], [], []
    for name, z in z_signals.items():
        w = train_ic.get(name, 0.0)
        pairs_all.append((z, 1.0 if w >= 0 else -1.0))
        if name in CLUSTER_REPS:
            pairs_rep.append((z, 1.0 if w >= 0 else -1.0))
        if name in POSITIVE_SURVIVORS:
            pairs_pos.append((z, 1.0))
    ic_pairs = [(z_signals[n], train_ic.get(n, 0.0)) for n in z_signals]
    return {'S1_all_dir_ew': _weighted_mean(pairs_all),
            'S2_cluster_reps': _weighted_mean(pairs_rep),
            'S3_positive_ew': _weighted_mean(pairs_pos),
            'S4_ic_weighted': _weighted_mean(ic_pairs)}


def signal_layer(protocol: dict) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """构造 32 信号 → IC（train 定向/加权）→ 4 合成 → 合成 IC。
    返回 (composite_scores, ic_report, eligible, d16_score)"""
    panel, base, evidence = load_ohlc_panel(protocol)
    price = base['price']
    eligible = (base['usable']
                & price.notna().rolling(protocol['minimum_history_sessions']).sum()
                .eq(protocol['minimum_history_sessions'])
                & base['amount'].rolling(protocol['liquidity_window'], min_periods=protocol['liquidity_window'])
                .median().ge(protocol['minimum_median_daily_amount_local_currency']))
    labels = forward_labels(price, protocol['horizon_sessions'])
    signals = ts.build_all_signals(panel)

    train_ic, z_signals, d16_score = {}, {}, None
    for name, sig in signals.items():
        daily = daily_statistics(sig, labels, eligible, protocol['minimum_cross_section'])
        parts = run_partitions(daily, protocol)
        tic = parts['train']['mean_ic']
        train_ic[name] = 0.0 if tic is None else float(tic)
        z_signals[name] = _z(sig)
        if name == 'D16_rel_atr':
            d16_score = sig
        del sig
    composite = compose(z_signals, train_ic)
    ic_report = {'single_train_ic': train_ic}
    for name, score in composite.items():
        parts = run_partitions(daily_statistics(score, labels, eligible,
                                                protocol['minimum_cross_section']),
                               protocol)
        ic_report[name] = parts
    d16_parts = run_partitions(daily_statistics(d16_score, labels, eligible,
                                                protocol['minimum_cross_section']),
                               protocol)
    ic_report['D16_benchmark'] = d16_parts
    return composite, ic_report, eligible, d16_score, base, evidence


def _versioned_protocol(dest: Path, payload: dict) -> None:
    """写入 protocol.json；已存在且内容不同时递增版本号另存（不静默覆盖）。

    审计 2026-11-19 一-9：重跑时若参数与已存档不同而原地覆盖，旧目录会
    留下"同名文件、不同实验含义"的结果；变化必须显式成新版本。"""
    target = dest / 'protocol.json'
    if not target.exists():
        atomic_json(target, payload)
        return
    existing = json.loads(target.read_text(encoding='utf-8'))
    if existing == payload:
        return
    raise ValueError('Protocol changed; use a new experiment destination')


def portfolio_layer(composite: dict, eligible: pd.DataFrame, d16_score: pd.DataFrame,
                    base: dict, evidence: dict, protocol: dict,
                    dest: Path = DEST, cost: float = 0.0025, availability=None) -> dict:
    with publish_study(dest) as stage:
        return _portfolio_layer(composite, eligible, d16_score, base, evidence, protocol,
                                stage, cost, availability)


def _portfolio_layer(composite, eligible, d16_score, base, evidence, protocol, dest, cost, availability):
    """组合仿真（复用第一轮组件；dest 可注入=预注册试验的独立证据目录）"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    inputs = dest / 'inputs.json'
    if inputs.exists() and json.loads(inputs.read_text(encoding='utf8')) != evidence:
        raise ValueError('Inputs changed; use a new experiment destination')
    full = dict(composite)
    full['D16_benchmark'] = d16_score
    protocol = dict(protocol)
    protocol['signals'] = list(full)
    ext = protocol.pop('external_scales', None)   # Series 不可序列化，单独存档
    dw = protocol.pop('daily_weights', None)      # 增厚②：权重矩阵同样单独存档
    hb = protocol.pop('hold_buffer_ranks', None)    # G系缓冲带（int可序列化但统一pop）
    hx = protocol.pop('hot_exit_sets', None)        # I系热卖出（set不可序列化）
    dbo = protocol.pop('daily_buffer_overrides', None)  # W系日度缓冲带
    dpo = protocol.pop('daily_position_overrides', None) # X系日度N
    # 审计 2026-11-19 一-9：被 pop 的动态参数不再"无处存档"——可序列化的
    # 进协议，不可序列化的写摘要（键数/首末日期），重跑可核对输入是否漂移
    for name, extra in (('hold_buffer_ranks', hb),
                        ('daily_position_overrides', dpo)):
        if extra:
            protocol[name] = {k: {str(d): n for d, n in v.items()}
                              if isinstance(v, dict) else v
                              for k, v in extra.items()}

    def _meta(mapping):
        return {k: {'dates': len(v), 'first': str(min(v)), 'last': str(max(v))}
                for k, v in (mapping or {}).items()}

    def canonical(value):
        if isinstance(value, pd.Series):
            value = value.to_dict()
        if isinstance(value, dict):
            return {str(k): canonical(v) for k, v in value.items()}
        if isinstance(value, set):
            return sorted(value)
        return value.item() if hasattr(value, 'item') else value

    import hashlib
    dynamic = canonical(dict(weights=dw, exits=hx, buffers=dbo, counts=dpo, scales=ext))
    protocol['dynamic_parameters_sha256'] = hashlib.sha256(
        json.dumps(dynamic, sort_keys=True, allow_nan=False).encode()).hexdigest()
    protocol['one_way_cost'] = cost
    registered = dest / 'protocol.json'
    if registered.exists() and json.loads(registered.read_text(encoding='utf8')) != protocol:
        raise ValueError('Protocol changed; use a new experiment destination')
    snapshot_code(dest)
    _versioned_protocol(dest, protocol)
    atomic_json(inputs, evidence)
    atomic_json(dest / 'params_meta.json',
                {'daily_weights': _meta(dw), 'hot_exit_sets': _meta(hx),
                 'daily_buffer_overrides': _meta(dbo),
                 'daily_position_overrides': _meta(dpo)})
    if any(value is not None for value in dynamic.values()):
        atomic_json(dest / 'dynamic_parameters.json', dynamic)
    if ext is not None:
        atomic_json(dest / 'external_scales_meta.json',
                    {k: {'len': int(s.size), 'mean': float(s.mean()),
                         'first': str(s.first_valid_index()),
                         'last': str(s.last_valid_index())} for k, s in ext.items()})
        for k, s in ext.items():
            s.rename('target_exposure').to_csv(dest / f'external_scale_{k}.csv',
                                               index_label='date')
    price = base['price']
    market_observed = price.notna().any(axis=1).to_numpy()
    dates, symbols = price.index, set(price.columns)
    hb = hb or {}   # 外层已pop的局部变量——禁止再从protocol取（第三次同款bug根治）
    dbo = dbo or {}
    dpo = dpo or {}

    def _ranking_depth(name: str) -> int:
        """候选列表深度 = 所有可能消费方的最大深度。

        审计 2026-11-19 一-4（W/X系）：此前只看 max(缓冲带, 持仓数)——W 系
        日度缓冲带 K=200/300/99999 时上游仍只保留前 100 名，K100 与 K200
        拿到同一份名单（三份净值逐日相同，放宽卖出线的实验等于没做）。"""
        depth = max(hb.get(name, 0), protocol['positions'])
        for overrides in (dbo.get(name), dpo.get(name)):
            if overrides:
                depth = max(depth, max(overrides.values()))
        return depth

    rankings = {name: selection_lists(score, eligible,
                                      _ranking_depth(name), (availability or {}).get(name))
                for name, score in full.items()}
    suspended = suspension_days(OUTPUT, dates, symbols)
    oracle = OUTPUT / 'oracle'
    master = verified_frame(oracle / 'reference/ASHAREDESCRIPTION.parquet',
                            json.loads((oracle / 'reference/ASHAREDESCRIPTION.parquet').with_suffix('.json').read_text(encoding='utf-8')))
    delist = dict(zip(master.S_INFO_WINDCODE,
                      pd.to_datetime(master.S_INFO_DELISTDATE, format='%Y%m%d', errors='coerce')))
    action_record = json.loads((oracle / 'reference/ASHAREDIVIDEND.parquet').with_suffix('.json').read_text(encoding='utf-8'))
    actions, action_quality = distributions(verified_frame(oracle / 'reference/ASHAREDIVIDEND.parquet', action_record),
                                            symbols, dates)
    atomic_json(dest / 'corporate_action_inputs.json',
                {'source_sha256': action_record['sha256'], 'quality': action_quality})
    simulations = []
    ext_map = ext or {}   # 141行已 pop——用局部变量，勿再从 protocol 取
    for signal, variant in product(protocol['signals'], protocol['risk_variants']):
        name = f'{signal}_{variant}_{round(cost * 10000)}bps'
        simulations.append((signal, PortfolioSimulation(
            name, protocol, cost, variant == 'budget_guard',
            actions, dates, delist, vol_target=(variant == 'vol_target'),
            vol_target_pure=(variant == 'vol_target_pure'),
            external_scales=ext_map.get(signal),
            daily_weights=(dw or {}).get(signal),
            hold_buffer_rank=hb.get(signal, 0),
            hot_exit_sets=(hx or {}).get(signal),
            daily_buffer_overrides=(dbo or {}).get(signal),
            daily_position_overrides=(dpo or {}).get(signal))))  # X系日度N
    manifest = json.loads((oracle / 'normalized_a/manifest.json').read_text(encoding='utf-8'))
    steps = 0
    for month, record in sorted(manifest['files'].items()):
        if month not in evidence['source_files']:
            continue
        if record.get('sha256') != evidence['source_files'].get(month):
            raise ValueError('Bar inputs changed between research passes')
        frame = verified_frame(oracle / 'normalized_a' / (month + '.parquet'), record)
        frame = frame.loc[frame.symbol.isin(symbols) & frame.date.isin(dates)]
        grouped = {day: g for day, g in frame.groupby('date', sort=True)}
        for day in dates[dates.strftime('%Y/%m') == month]:
            index = dates.get_loc(day)
            day_frame = grouped.get(day)
            bars = {} if day_frame is None else day_frame.set_index('symbol').to_dict(orient='index')
            for symbol in suspended.get(day, set()):
                if symbol not in bars:
                    bars[symbol] = {'numeric_bar_valid': False, 'source_suspended': True}
            previous_eligible = set(eligible.columns[eligible.iloc[index - 1]]) if index else set()
            for signal, simulation in simulations:
                ranking = rankings[signal][index - 1] if index else []
                if index and not market_observed[index - 1]:
                    ranking = None
                simulation.step(index, day, bars, ranking, previous_eligible)
            steps += 1
    if steps != len(dates):
        raise ValueError('replay coverage mismatch')
    reports = {}
    for signal, simulation in simulations:
        reports[simulation.name] = simulation.result(protocol['train_end'])
        pd.DataFrame(simulation.nav_rows).to_csv(dest / f'{simulation.name}_nav.csv', index=False)
        pd.DataFrame(simulation.trades).to_csv(dest / f'{simulation.name}_trades.csv', index=False)
        pd.DataFrame(simulation.issue_rows, columns=['date', 'symbol', 'issue']).to_csv(
            dest / f'{simulation.name}_issues.csv', index=False)
    return reports
