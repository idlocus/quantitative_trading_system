#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日交易流程 CLI —— 生成完整交易日志（journal/YYYY-MM-DD.md）

用法:
    python run_daily.py --phase am           # 早盘：市场展望+决策依据（T-1信号）
    python run_daily.py --phase pm --execute paper   # 收盘后：Paper 模拟成交+日志
    python run_daily.py --phase pm --execute qmt --yes  # QMT 实盘委托（环境就绪+显式确认）
    python run_daily.py                      # 收盘后一次跑完整天（am+pm）

执行器：make_broker(execute) 统一构造——qmt 环境未就绪时打印修复指引并跳过执行，
不会静默丢弃指令；无人值守实盘必须 --yes。

时序口径（与回测 score_mx.shift(1) 同构，无前视）：
    信号 = T-1 收盘数据（PIT）｜执行 = T 日（Paper 以 T 收盘价+滑点撮合，QMT 盘中下单）
    AM 预览与 PM 执行使用同一 T-1 信号（generate_production_signal 唯一入口）
"""

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from live import Ledger, make_broker
from live.journal import JournalWriter
from live.risk_gate import RiskGate

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'live', 'ledger.json')


def run_am(ds, ledger, writer):
    ctx = writer.write_am(ds)
    print(f"盘前日志已写入: {writer.path}")
    print(f"目标持仓 {len(ctx['targets'])} 只 | 调仓指令 {len(ctx['orders'])} 条")
    return ctx


def run_pm(ds, ledger, writer, ctx, execute=None, top_n=100, yes=False,
           outside_window=False):
    """top_n 默认=主链 TOP_N（此前默认30与CLI默认100口径分裂）"""
    from live.signals import generate_production_signal, build_orders
    # QMT 在途订单补记：先补账本（部分成交/撤单残量），再用最新持仓生成信号与风控
    qmt_broker = None
    if execute == 'qmt':
        try:
            from live import make_broker as _mk, load_lot_map
            qmt_broker = _mk('qmt', lot_map=load_lot_map(ds),
                             enforce_window=not outside_window)
            synced = qmt_broker.sync_pending(ledger)
            if synced:
                print(f"QMT 在途补记 {len(synced)} 笔（已入账本）")
            if getattr(qmt_broker, 'query_failed', False):
                print("⛔ QMT 查询接口异常——在途订单状态不可确认，"
                      "阻断本次实盘执行（防止重复委托），人工检查后重试")
                return []
        except RuntimeError as e:
            print(f"QMT 不可用，将不执行实盘：\n{e}")
    # 时序（与回测 shift(1) 同构，无前视）：T-1收盘信号 + T日收盘价撮合
    psig = generate_production_signal(ds, top_n=top_n)
    targets, signal_date = psig.targets, psig.signal_date

    # 信号为空≠无目标股：可能是数据源故障/因子矩阵为空/池加载失败。
    # 必须进入停机流程（不继续build_orders——空targets会把所有持仓当作
    # "不在目标中"从而生成全部卖出订单=数据故障被误判为清仓信号）。
    if not targets:
        print("⛔ 信号为空（数据链路故障/因子矩阵为空/池加载失败）——进入停机，不执行任何交易")
        # 仍做 mark-to-market 记录当日状态（不交易）
        symbols = set(ledger.positions)
        if symbols:
            cache_halt = ds.preload_batch(list(symbols), date.today() - timedelta(days=15), date.today())
            prices_halt = {}
            for sym, df in cache_halt.items():
                if df is not None and not df.empty:
                    d = df[df['date'] <= pd.Timestamp(date.today())]
                    if not d.empty:
                        prices_halt[sym] = float(d['close'].iloc[-1])
            ledger.mark_to_market(prices_halt, date.today())
        return []

    if not psig.st_ok:
        print(f"警告：ST数据不可用（信号日{signal_date}），撮合按非ST涨跌幅处理并禁止新买入")
    symbols = set(targets) | set(ledger.positions)
    cache = ds.preload_batch(list(symbols), date.today() - timedelta(days=15), date.today())
    prices, prev = {}, {}
    stale_syms = set()
    for sym, df in cache.items():
        if df is None or df.empty:
            continue
        d = df[df['date'] <= pd.Timestamp(date.today())]
        if d.empty:
            stale_syms.add(sym)
            continue
        last_dt = d['date'].iloc[-1]
        # 行情过期检测：最后bar距执行日超过5个自然日视为停牌/延迟
        if (pd.Timestamp(date.today()) - last_dt).days > 5:
            stale_syms.add(sym)
            continue
        prices[sym] = float(d['close'].iloc[-1])
        if len(d) > 1:
            prev[sym] = float(d['close'].iloc[-2])
    print(f"信号日 {signal_date}（T-1）| 执行日 {date.today()}（收盘价撮合）")

    # 风控前净值快照：用当日价格先行 mark-to-market，确保闸门读到当日实际净值。
    # 但如果闸门因数据异常触发 halt，不把可能错误的净值写入历史（恢复快照）。
    _nav_snapshot = list(ledger._d['nav_history'])   # 快照
    if ledger.positions:
        ledger.mark_to_market(prices, date.today())

    nav = ledger.last_nav() or ledger.cash
    cur_w = ledger.current_weights(prices) if ledger.positions else {}
    orders = build_orders(targets, cur_w, nav, top_n=top_n)

    # ST 集合由生产信号入口统一加载（psig.st_symbols/st_ok，信号/撮合/闸门同源）

    # 执行风控闸门（L2停机/L3熔断/L4强制卖出）——改写后的指令流才可执行
    bench_cache = ds.preload_batch(['000300.SH'], date.today() - timedelta(days=20), date.today())
    gate = RiskGate()
    # gate.as_of=执行日（今日）：T+1锁仓/数据新鲜度均以成交日为准
    decision = gate.review(orders, ledger, prices, prev, targets,
                           bench_df=bench_cache.get('000300.SH'), as_of=date.today(),
                           st_data_ok=psig.st_ok, stale_symbols=stale_syms)
    print(f"风控闸门：{decision.verdict_plain}｜放行{len(decision.orders)} 拦截{len(decision.suppressed)} "
          f"强制卖出{len(decision.forced_sells)}")

    # 闸门 halt 时恢复净值快照：错误价格不写入历史（否则次日恢复时被误判为
    # 大幅回撤并可能触发清仓）
    if decision.action == 'halt':
        ledger._d['nav_history'] = _nav_snapshot
        ledger.save()

    fills = []
    if decision.action == 'halt':
        print(f"数据停机，今日不执行任何指令：{decision.data_issues}")
    elif decision.orders and execute in ('paper', 'qmt'):
        broker = None
        try:
            from live import load_lot_map
            broker = qmt_broker if (execute == 'qmt' and qmt_broker is not None) \
                else make_broker(execute, lot_map=load_lot_map(ds))
        except RuntimeError as e:
            print(f"执行器不可用，今日不执行任何指令：\n{e}")
        if broker is not None:
            if execute == 'qmt' and not yes:
                # 无人值守批处理提交实盘委托必须显式 --yes
                print("QMT 实盘执行需要 --yes 显式确认，本次跳过委托（指令已生成，见日志）。")
            else:
                fills = broker.execute(decision.orders, prices, date.today(), prev_prices=prev,
                                       available_cash=ledger.cash,
                                       st_symbols=psig.exec_st_symbols,  # 执行日(T)ST集合：撮合按当日戴帽/摘帽状态
                                       positions=dict(ledger.positions))
                booked, dupes, blocked = ledger.apply_fills(fills)
                if dupes:
                    print(f"成交去重：{len(dupes)} 笔重复回报/重复执行已跳过")
                if blocked:
                    print(f"⛔ 超额卖出 {len(blocked)} 笔已阻止入账（异常队列待人工核对，--status 查看）")

    # 账实核对：当日对账单存在则自动核对（live/statements/今日.json）
    recon = None
    from live.reconcile import load_statement, statement_path_for, Reconciler
    stmt_file = statement_path_for()
    if os.path.isfile(stmt_file):
        try:
            stmt = load_statement(stmt_file)
            recon = Reconciler().reconcile(ledger, stmt)
            print(f"账实核对：{recon.verdict_plain}"
                  + (f"（{'；'.join(d.detail for d in recon.diffs[:3])}）" if recon.diffs else ""))
            # 对账结果写回账本（闸门 L2.6 依赖此状态禁止/放行下次交易）
            ledger._d['meta']['last_recon'] = {
                'date': stmt.as_of, 'ok': recon.is_ok,
                'n_diffs': len(recon.diffs), 'source': stmt.source}
            ledger.save()
        except Exception as e:
            print(f"对账单解析失败，跳过核对：{e}")
            ledger._d['meta']['last_recon'] = {
                'date': str(date.today()), 'ok': False, 'n_diffs': -1,
                'source': 'parse_error', 'note': str(e)[:100]}
            ledger.save()

    # ---- F2 告警 + F6 健康视图（每次PM运行后自动生成） ----
    try:
        from monitoring.alerts import alert, P0, P1
        from monitoring.health import generate_health_report

        # 信号覆盖率告警
        if len(targets) == 0:
            alert('SIGNAL_COVERAGE_COLLAPSE', P0,
                  f"目标持仓0只（数据链路故障），闸门={decision.action}")
        elif len(targets) < 20:
            alert('SIGNAL_LOW_TARGETS', P1,
                  f"目标持仓仅{len(targets)}只（正常~100），数据可能不完整")

        # 对账失败告警
        if recon is not None and not recon.is_ok:
            alert('RECONCILIATION_FAILED', P0,
                  f"对账不一致：{len(recon.diffs)}项差异",
                  n_diffs=len(recon.diffs))

        # 闸门 halt 告警
        if decision.action == 'halt':
            alert('RISK_GATE_HALT', P0,
                  f"闸门halt：{decision.data_issues[:2]}")

        # 生成健康报告
        health = generate_health_report(
            ledger,
            signal_info={'target_count': len(targets),
                         'signal_date': str(signal_date)})
        logger.info(f"[Health] 状态: {health['overall']}")
    except Exception as e:
        logger.warning(f"[Health] 告警/健康报告生成失败: {e}")

    path = writer.write_pm(ds, fills, ctx, gate=decision, recon=recon)
    print(f"盘后日志已追加: {path}")
    if fills:
        print(f"成交 {sum(1 for f in fills if f.status=='filled')}/{len(decision.orders)} 笔（Paper）")
    return fills


def main():
    parser = argparse.ArgumentParser(description='Daily trading journal')
    parser.add_argument('--phase', choices=['am', 'pm', 'full'], default='full')
    parser.add_argument('--execute', choices=['paper', 'qmt'], default=None)
    parser.add_argument('--yes', action='store_true',
                        help='QMT 实盘委托免交互确认（无人值守批处理用）')
    parser.add_argument('--outside-window', action='store_true',
                        help='QMT 允许非收盘竞价窗口委托（默认窗口外拒单；后果自担）')
    parser.add_argument('--top', type=int, default=100)
    parser.add_argument('--ledger', default=LEDGER_PATH)
    args = parser.parse_args()

    from data.layered import LayeredDataSource
    ds = LayeredDataSource()
    ledger = Ledger(args.ledger)
    writer = JournalWriter(ledger, top_n=args.top)

    if args.phase in ('am', 'full'):
        ctx = run_am(ds, ledger, writer)
    else:
        ctx = {'targets': [], 'scores': None, 'orders': [], 'prev_trading': None}

    if args.phase in ('pm', 'full'):
        run_pm(ds, ledger, writer, ctx, execute=args.execute, top_n=args.top,
               yes=args.yes, outside_window=args.outside_window)

    print(f"\n完整日志: journal/{date.today().isoformat()}.md")


if __name__ == '__main__':
    main()
