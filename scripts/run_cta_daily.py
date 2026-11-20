#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTA Z10 活跃品种等权日报（策略详见 docs/CTA_Z10_STRATEGY.md）。

信号语义、排序与成交口径由 research/cta_signal.py 单一提供（登记册180）：
- 每日跟随方向（压缩段内 MA20 穿越即翻向，登记版语义）
- 活跃品种等权，次日开盘执行
- 数据覆盖不完整时组合结论降级为"无法判定"，不输出"持仓不动"
"""
import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import time

from research import cta_signal as cs

OUTPUT_DIR = Path('runtime/cta_daily')


def _expected_last_trading_day():
    """预期的最近已收盘交易日（无外部日历，按工作日近似；节假日由
    --allow-stale 人工确认）。收盘时刻按 15:30 计。"""
    now = pd.Timestamp.now()
    d = now.normalize()
    if now.hour < 15 or (now.hour == 15 and now.minute < 30):
        d = d - pd.Timedelta(days=1)      # 今日未收盘 → 预期昨日
    while d.weekday() >= 5:                # 跳过周末
        d = d - pd.Timedelta(days=1)
    return d.date()

SYMBOLS = {
    'AU0': ('黄金', 'AU', '上期所'), 'CU0': ('沪铜', 'CU', '上期所'),
    'RB0': ('螺纹钢', 'RB', '上期所'), 'I0': ('铁矿石', 'I', '大商所'),
    'P0': ('棕榈油', 'P', '大商所'), 'HC0': ('热卷', 'HC', '上期所'),
}


def fetch_frame(sym: str, days: int = 900) -> pd.DataFrame:
    """拉取连续主力日线，返回含 open/close 的 DataFrame（失败抛异常）。"""
    import akshare as ak
    end = date.today().strftime('%Y%m%d')
    start = (date.today() - pd.Timedelta(days=days)).strftime('%Y%m%d')
    df = ak.futures_main_sina(symbol=sym, start_date=start, end_date=end)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期').sort_index()
    if len(df) < cs.VOL_LOOKBACK + 10:
        raise ValueError(f'历史不足: {len(df)} 根')
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    report_file = OUTPUT_DIR / f'{today}.md'

    print(f'CTA Z10 {cs.DEPLOY_LABEL} 每日报告 @ {today}')
    print('数据拉取中...\n')

    raw, errors = {}, {}
    for sym, (name, _, _) in SYMBOLS.items():
        try:
            raw[sym] = fetch_frame(sym)
            time.sleep(0.3)
        except Exception as e:
            errors[sym] = str(e)[:80]
            print(f'  {name}: 拉取失败 ({errors[sym]})')

    frames = {s: cs.signal_frame(df['收盘价'].astype(float),
                                 open_=df['开盘价'].astype(float))
              for s, df in raw.items()}
    # 盘中防护：只保留最近已收盘交易日（含）之前的K线——数据源盘中可能
    # 已返回当日未定稿日线，信号/排名/杠杆必须基于已收盘数据
    expected = _expected_last_trading_day()
    frames = cs.truncate_to_closed(frames, expected)
    cov = cs.coverage_check(frames, expected=list(SYMBOLS),
                            as_of=pd.Timestamp.today())
    last_date = cov['last_date']

    lines = []
    lines.append(f'# CTA Z10 每日报告 — {today}')
    lines.append('')
    lines.append('> 策略：波动率压缩×活跃品种等权（登记册181 定案）| '
                 '规则见 [CTA_Z10_STRATEGY.md](../../docs/CTA_Z10_STRATEGY.md)')
    lines.append(f'> 信号日期：{last_date.date().isoformat() if last_date is not None else "—"}'
                 f'（信号收盘后生成，**次日开盘执行**）')
    lines.append('')

    # ── 数据覆盖核验（覆盖不完整 → 结论降级，防残缺数据出"持仓不动"）──
    if not cov['ok']:
        lines.append('## 🛑 数据覆盖不完整 —— 组合结论：无法判定')
        lines.append('')
        if cov['missing']:
            names = [SYMBOLS[s][0] for s in cov['missing']]
            lines.append(f'- **缺失品种**：{", ".join(names)}'
                         f'（共{len(cov["missing"])}/{len(SYMBOLS)}）')
        if cov['insufficient_history']:
            names = [SYMBOLS[s][0] for s in cov['insufficient_history']]
            empty_note = '（其中空表=截断后无已收盘行情）' if cov.get('empty') else ''
            lines.append(f'- **历史不足**（<{cs.MIN_BARS}根，无法形成波动分位）'
                         f'{empty_note}：{", ".join(names)}')
        if cov['no_signal_on_decision_day']:
            names = [SYMBOLS[s][0] for s in cov['no_signal_on_decision_day']]
            lines.append(f'- **决策日指标无效**（收盘价或波动分位非有限值）：'
                         f'{", ".join(names)}')
        if not cov['aligned']:
            ld = {SYMBOLS[s][0]: d.date().isoformat()
                  for s, d in cov['last_dates'].items()}
            lines.append(f'- **交易日不一致**：{ld}')
        lines.append('')
        lines.append('**禁止在数据补齐前按本报告执行任何持仓操作。**'
                     '已知持仓的处理以最近一次覆盖完整的日子为准，'
                     '或人工核验缺失品种行情后再决策。')
        lines.append('')
        if cov['missing'] and len(cov['missing']) < len(SYMBOLS):
            lines.append('## 📋 可用品种明细（仅供参考，非组合结论）')
            lines.append('')
            _append_symbol_table(lines, frames, SYMBOLS)
        report = '\n'.join(lines)
        report_file.write_text(report, encoding='utf-8')
        print(report)
        print(f'\n报告已保存: {report_file}')
        return

    # ── 强度排序与目标持仓 ──
    ranked = cs.rank_active(frames)
    book_today = cs.target_book(frames, k=cs.DEPLOY_K)

    # 昨日目标持仓：先判定真"昨日"（决策日前一已收盘交易日），再核验
    # 全部品种在该日有数据——任一缺席即"昨日未知"（防跳日漏换仓）
    yesterday_known = False
    book_yesterday = {}
    try:
        prev_date, all_have = cs.previous_book_date(frames, last_date)
        if prev_date is not None and all_have:
            if all(np.isfinite(f.loc[prev_date, 'vol_pct'])
                   for f in frames.values()):
                book_yesterday = cs.target_book(frames, k=cs.DEPLOY_K,
                                                at=prev_date)
                yesterday_known = True
    except Exception:
        yesterday_known = False

    # 目标名义敞口：T日收盘决策、T+1开盘生效的杠杆（含T日收益），
    # 与回测 vol_target_leverage 在 T+1 行的取值一致（防滞后一天）
    try:
        net_hist = cs.portfolio_daily(frames, k=cs.DEPLOY_K,
                                      execution='next_open', charge_cost=True)
        lev_next = cs.next_leverage(net_hist)      # 明日订单用
        lev_held = cs.held_leverage(net_hist)      # 今日实际持有（对照）
    except Exception:
        lev_next = lev_held = None

    lines.append('## 📊 强度排序（压缩深度 = 波动分位，越低越强；'
                 '本形态全部活跃品种等权，排序仅作观察）')
    lines.append('')
    if ranked:
        lines.append('| 排名 | 品种 | 波动分位 | 方向 | 清晰度|close/MA-1| |')
        lines.append('|---:|---|---:|---|---:|')
        for i, sym in enumerate(ranked, 1):
            row = frames[sym].iloc[-1]
            name = SYMBOLS[sym][0]
            d = '多' if row['direction'] > 0 else '空'
            lines.append(f'| {i} | {name}({SYMBOLS[sym][1]}) | '
                         f'{row["vol_pct"]:.0%} | {d} | {row["clarity"]:.1%} |')
        lines.append('')
    else:
        lines.append('当前无压缩品种，全部空仓等待。')
        lines.append('')

    lines.append('## 🎯 目标持仓（明日开盘生效，含登记口径杠杆）')
    lines.append('')
    if book_today:
        for sym, w in book_today.items():
            d = '做多' if w > 0 else '做空'
            expo = f'{w * lev_next:+.2f}' if (lev_next is not None and w) else '—'
            if yesterday_known and sym in book_yesterday and lev_held is not None:
                held = f'{book_yesterday[sym] * lev_held:+.2f}'
            else:
                held = '—'
            lines.append(f'- **{SYMBOLS[sym][0]}({SYMBOLS[sym][1]})**：{d}，'
                         f'目标名义敞口 **{expo} 倍**（昨日模型目标 {held} 倍，实际持仓以成交回报为准）')
        if lev_next is not None and lev_held is not None \
                and abs(lev_next - lev_held) > 0.05:
            lines.append(f'- ⚠️ 波动率目标杠杆变化（今日 {lev_held:.2f} → '
                         f'明日 {lev_next:.2f}）：即使品种与方向不变也需调整手数')
        lines.append('- 敞口为 sleeve 名义资金倍数；保证金约 11%/品种，'
                     'sleeve 内其余资金留缓冲/货基')
    else:
        lines.append('- 空仓（无活跃信号）')
    lines.append('')

    # ┠ 停更拦截：数据日期落后于预期已收盘交易日 → 不生成操作清单
    allow_stale = os.environ.get('QUANT_ALLOW_STALE') == '1' or '--allow-stale' in sys.argv
    data_lagging = (last_date is not None and expected is not None
                    and pd.Timestamp(last_date).date() < expected)

    # ── 操作清单（昨日未知 或 数据停更 时跳过，只展示今日目标）──
    actions, note = cs.compose_actions(book_yesterday, book_today,
                                       yesterday_known)
    if actions is not None and lev_held is not None and lev_next is not None and lev_held != lev_next:
        for item in cs.book_transition(book_yesterday, book_today):
            if item['action'] == '不动':
                actions.append(dict(item, action='调仓'))
    if note:
        lines.append(f'> ⚠️ {note}')
        lines.append('')
    if data_lagging and not allow_stale:
        lines.append(f'> 🛑 **数据停更**：数据日期 {last_date.date()} 落后于'
                     f'预期已收盘交易日 {expected}，操作清单已停发。'
                     f'若今日确为休市日（长假），加 `--allow-stale` 确认后重跑。')
        lines.append('')
        actions = None
    if actions:
        lines.append('## ⚡ 操作清单（次日开盘执行）')
        lines.append('')
        lines.append('以下差额假定昨日模型目标已全部成交。请先以成交回报核对实际数量。')
        lines.extend(cs.render_actions(actions, lev_held, lev_next,
                                         yesterday_known))
        lines.append('')
    elif yesterday_known and not data_lagging:
        if book_today and (lev_next is None or lev_held is None):
            lines.append('## 杠杆数据不足，无法判断调仓数量')
            lines.append('目标品种和方向不变，请核对实际持仓及目标杠杆。')
        else:
            lines.append('## ✅ 今日无品种级操作（模型目标不变）')
        lines.append('')

    _append_symbol_table(lines, frames, SYMBOLS)

    near = [(s, frames[s].iloc[-1]) for s in frames
            if not bool(frames[s].iloc[-1]['compressed'])
            and np.isfinite(frames[s].iloc[-1]['vol_pct'])
            and frames[s].iloc[-1]['vol_pct'] < 0.30]
    if near:
        lines.append('## 👀 压缩观察名单（分位<30%，接近触发）')
        lines.append('')
        for s, row in sorted(near, key=lambda x: x[1]['vol_pct']):
            lines.append(f'- {SYMBOLS[s][0]}: 波动分位 {row["vol_pct"]:.0%}'
                         f'（触发线 20%），还差 {(row["vol_pct"] - 0.20):.0%}')
        lines.append('')

    lines.append('## ⚠️ 风险提示')
    lines.append('')
    lines.append('- 本报告使用主力连续行情。缺少逐日合约与换月成交记录，无法验证可成交收益。')
    lines.append('- 执行口径：次日开盘成交（跳空归属修正后指标见登记册181，'
                 '本报告口径与登记回测一致）')
    lines.append('- 每日跟随语义：压缩段内价格穿越 MA20 会**翻向**，'
                 '不是"入场后持有到压缩结束"')
    lines.append('- 成本敏感：单边滑点超 2.5bp 会侵蚀收益')
    lines.append('- 期货有杠杆：建议 sleeve 内保留≥50%现金缓冲（保证金+变动保证金）')
    lines.append('- 六品种宇宙为样本内筛出（继承性偏差），两期同号是现有缓解')
    lines.append('')

    report = '\n'.join(lines)
    report_file.write_text(report, encoding='utf-8')
    print(report)
    print(f'\n报告已保存: {report_file}')
    if actions:
        print('\n' + '=' * 50)
        print('⚡ 有操作变化！请查看操作清单（次日开盘执行）')
        print('=' * 50)


def _append_symbol_table(lines, frames, symbols):
    lines.append('## 📋 品种明细')
    lines.append('')
    lines.append('| 品种 | 现价 | 20日波动(年化) | 波动分位 | MA20 | 价格vs MA20 | 状态 | 方向 |')
    lines.append('|---|---:|---:|---:|---:|---:|---|---|')
    for sym, f in sorted(frames.items(), key=lambda kv: symbols[kv[0]][0]):
        if f.empty:
            continue
        row = f.iloc[-1]
        name = symbols[sym][0]
        vol_ann = row['vol20'] * np.sqrt(252) if np.isfinite(row['vol20']) else np.nan
        status = '🟢 压缩' if bool(row['compressed']) else '⚪ 正常'
        # 审计 2026-11-19 二-3：方向为零（收盘=均线）显示 —，不得写"空"
        d = '多' if row['direction'] > 0 else ('空' if row['direction'] < 0 else '—')
        ma = row['ma'] if np.isfinite(row['ma']) else row['close']
        lines.append(
            f'| {name} | {row["close"]:.0f} | {vol_ann:.1%} | '
            f'{row["vol_pct"]:.0%} | {ma:.0f} | '
            f'{(row["close"]/ma-1):+.1%} | {status} | {d} |')
    lines.append('')


if __name__ == '__main__':
    main()
