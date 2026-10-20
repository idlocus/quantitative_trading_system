#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据健康集中视图 —— runtime/health/latest.json

每次运行日报后自动生成，快速回答：
- 哪个数据源在降级？
- 哪个因子覆盖率异常？
- 当前信号是否可交易？
- 最近一次对账是否成功？
"""
import json
import os
from datetime import datetime, date

import logging

logger = logging.getLogger(__name__)

RUNTIME_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runtime')
HEALTH_DIR = os.path.join(RUNTIME_DIR, 'health')
LATEST_FILE = os.path.join(HEALTH_DIR, 'latest.json')


def generate_health_report(ledger, signal_info: dict = None,
                          factor_coverages: dict = None) -> dict:
    """
    生成数据健康报告。

    Args:
        ledger: Ledger 实例
        signal_info: 信号生成结果 {'target_count': int, 'signal_date': str}
        factor_coverages: {'ep': 0.96, 'bp': 0.95, ...}
    """
    report = {
        'as_of': str(date.today()),
        'generated_at': datetime.now().isoformat(),
        'overall': 'ok',  # ok / degraded / halt
        'providers': {},
        'factors': {},
        'signal': {},
        'reconciliation': {},
        'ledger': {},
        'style': {},
    }

    issues = []

    # ---- 数据源状态 ----
    report['providers'] = _check_providers()

    # ---- 风格暴露观测（B3：观测不拦截，区分风格逆风与系统故障） ----
    report['style'] = _style_exposure(ledger)

    # ---- 因子覆盖率 ----
    if factor_coverages:
        for fname, cov in factor_coverages.items():
            status = 'ok' if cov >= 0.30 else ('halt' if cov < 0.10 else 'degraded')
            report['factors'][fname] = {'coverage': round(cov, 3), 'status': status}
            if status != 'ok':
                issues.append(f"因子{fname}覆盖率{cov:.0%}")

    # ---- 信号状态 ----
    if signal_info:
        n_targets = signal_info.get('target_count', 0)
        sig_status = 'ok' if n_targets >= 20 else ('halt' if n_targets == 0 else 'degraded')
        report['signal'] = {
            'target_count': n_targets,
            'signal_date': signal_info.get('signal_date', ''),
            'status': sig_status,
        }
        if sig_status != 'ok':
            issues.append(f"目标数{n_targets}")

    # ---- 对账状态 ----
    last_recon = ledger._d.get('meta', {}).get('last_recon', {})
    if last_recon:
        recon_ok = last_recon.get('ok', True)
        recon_date = last_recon.get('date', '')
        # 检查是否过期
        stale = False
        if recon_date:
            try:
                from datetime import datetime as _dt
                rd = _dt.strptime(recon_date[:10], '%Y-%m-%d').date()
                stale = (date.today() - rd).days > 3
            except Exception:
                pass
        recon_status = 'halt' if (not recon_ok or stale) else 'ok'
        report['reconciliation'] = {
            'last_date': recon_date,
            'ok': recon_ok,
            'stale': stale,
            'status': recon_status,
        }
        if not recon_ok:
            issues.append("对账失败")
        elif stale:
            issues.append("对账过期")
    else:
        report['reconciliation'] = {'status': 'unknown', 'note': '从未对账'}

    # ---- 账本健康 ----
    n_exceptions = len(ledger.open_exceptions()) if hasattr(ledger, 'open_exceptions') else 0
    nav = ledger.last_nav() or ledger.cash
    report['ledger'] = {
        'positions': len(ledger.positions),
        'nav': round(nav, 0),
        'open_exceptions': n_exceptions,
        'status': 'halt' if n_exceptions > 0 else 'ok',
    }
    if n_exceptions:
        issues.append(f"{n_exceptions}项未处理异常")

    # ---- 交易控制状态 ----
    try:
        from scripts.trading_switch import effective_state
        report['switch'] = effective_state()
        if effective_state() != 'NORMAL':
            issues.append(f"交易控制={effective_state()}")
    except Exception:
        report['switch'] = 'NORMAL'

    # ---- 总体状态 ----
    if any('halt' in str(v.get('status', '')) for v in
           list(report['factors'].values()) + [report['signal'], report.get('reconciliation', {})]):
        report['overall'] = 'halt'
    elif issues:
        report['overall'] = 'degraded'
    else:
        report['overall'] = 'ok'

    report['issues'] = issues

    # 写入文件
    os.makedirs(HEALTH_DIR, exist_ok=True)
    with open(LATEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    logger.info(f"[Health] {report['overall']} ({'; '.join(issues) if issues else '无异常'})")

    return report


def _check_providers() -> dict:
    """检查数据源状态 + 最新装载来源构成（降级可见性，评审 §2.8/#13）"""
    providers = {}
    # Oracle DB
    try:
        from data.layered import LayeredDataSource
        ds = LayeredDataSource()
        db = ds._db_src()
        providers['oracle'] = {'status': 'ok' if db else 'degraded'}
    except Exception:
        providers['oracle'] = {'status': 'degraded'}
    # Wind 客户端
    try:
        import windpy  # noqa: F401
        providers['wind'] = {'status': 'ok'}
    except ImportError:
        providers['wind'] = {'status': 'unavailable'}
    # 最新装载的来源构成（data/layered 每次装载后落盘）
    try:
        import json as _json
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'runtime', 'layered_stats.json')
        if os.path.isfile(stats_path):
            with open(stats_path, encoding='utf-8') as f:
                s = _json.load(f)
            total_rows = max(s.get('cache_rows', 0) + s.get('db_rows', 0)
                             + s.get('client_rows', 0), 1)
            client_share = s.get('client_rows', 0) / total_rows
            db_share = s.get('db_rows', 0) / total_rows
            providers['source_mix'] = {
                'status': 'degraded' if client_share > 0.30 else 'ok',
                'at': s.get('at'),
                'symbols': s.get('symbols'), 'hit': s.get('hit'),
                'rows': {'cache': s.get('cache_rows', 0),
                         'db': s.get('db_rows', 0),
                         'client_windpy': s.get('client_rows', 0)},
                'shares': {'cache': round(1 - client_share - db_share, 3),
                           'db': round(db_share, 3),
                           'client_windpy': round(client_share, 3)},
                'note': ('WindPy兜底占比>30%——DB缺口扩大，配额承压'
                         if client_share > 0.30 else '正常分层（缓存→DB→客户端兜底）'),
            }
    except Exception:
        pass
    return providers


# ================= CLI =================

if __name__ == '__main__':
    from live import Ledger
    led_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'live', 'ledger.json')
    led = Ledger(led_path)
    report = generate_health_report(led)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _style_exposure(ledger) -> dict:
    """风格暴露观测（B3，2026-10-20，评审P1-1方案1采纳）。

    观测不拦截——目的是让"风格逆风期"与"系统故障"可区分。
    指标：行业偏离度（组合 vs 全宇宙等权的总变差）、规模分位
    （持仓中位市值在宇宙的分位）、超配/低配头部行业。
    数据：PIT 万得 L2 行业 + S_VAL_MV（万元）。"""
    try:
        from data.layered import LayeredDataSource
        from data.industry_pit import industry_classification, industry_names
        ds = LayeredDataSource()
        holds = dict(ledger.positions)
        if not holds:
            return {'status': 'empty', 'note': '无持仓'}
        universe = sorted(ds.list_symbols())
        cls = industry_classification(ds, universe, date.today())
        names = industry_names(ds, level=2)
        from collections import Counter
        port_cnt = Counter()
        uni_cnt = Counter()
        for s in holds:
            if s in cls:
                port_cnt[cls[s]] += 1
        for s in universe:
            if s in cls:
                uni_cnt[cls[s]] += 1
        tot_p, tot_u = sum(port_cnt.values()), sum(uni_cnt.values())
        if not tot_p or not tot_u:
            return {'status': 'n/a', 'note': '行业分类不可用'}
        codes = set(port_cnt) | set(uni_cnt)
        wp = {c: port_cnt.get(c, 0) / tot_p for c in codes}
        wu = {c: uni_cnt.get(c, 0) / tot_u for c in codes}
        tv = sum(abs(wp[c] - wu[c]) for c in codes) / 2      # 总变差 0~1
        over = sorted(codes, key=lambda c: wp[c] - wu[c], reverse=True)[:3]
        under = sorted(codes, key=lambda c: wp[c] - wu[c])[:3]

        # 规模分位（S_VAL_MV 万元）
        try:
            cur = ds._db_src()._cursor()
            cur.execute(
                "SELECT S_INFO_WINDCODE, S_VAL_MV FROM AShareEODDerivativeIndicator "
                "WHERE TRADE_DT=(SELECT MAX(TRADE_DT) FROM AShareEODDerivativeIndicator) "
                "AND S_VAL_MV IS NOT NULL")
            caps = dict(cur.fetchall())
            if caps:
                import numpy as np
                vals = sorted(caps.values())
                med_uni = vals[len(vals) // 2]
                hcap = [caps[s] for s in holds if s in caps]
                pct = float(np.mean([c < med_uni for c in hcap])) if hcap else None
                size_note = (f"持仓中位市值分位 {pct:.0%}（<50%=偏大盘）"
                             if pct is not None else '规模数据不足')
            else:
                size_note = '市值数据不可用'
        except Exception:
            size_note = '市值查询失败'

        return {
            'status': 'ok',
            'industry_deviation': round(tv, 3),
            'industry_deviation_note': f"总变差 {tv:.0%}（组合 vs 宇宙等权）",
            'top_overweight': [{'industry': names.get(c, c),
                                'w_port': round(wp[c], 3),
                                'w_universe': round(wu[c], 3)} for c in over],
            'top_underweight': [{'industry': names.get(c, c),
                                 'w_port': round(wp[c], 3),
                                 'w_universe': round(wu[c], 3)} for c in under],
            'size_percentile': size_note,
        }
    except Exception as e:
        return {'status': 'error', 'note': str(e)[:80]}
