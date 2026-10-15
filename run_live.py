#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘驱动 CLI —— 每日收盘后运行

用法:
    python run_live.py --init 1000000              # 初始入金
    python run_live.py --signal                    # 生成目标持仓+调仓指令（仅打印）
    python run_live.py --signal --execute paper    # Paper 模式模拟成交并入账
    python run_live.py --signal --execute qmt --yes # QMT 实盘委托（环境就绪才执行）
    python run_live.py --check-qmt                 # 检查 QMT 环境（路径/账号/xtquant）
    python run_live.py --nav                       # 按最新收盘更新净值
    python run_live.py --status                    # 持仓/净值/累计收益概览
    python run_live.py --deposit 50000             # 入金（资金流水留痕）
    python run_live.py --withdraw 30000            # 出金
    python run_live.py --dividend 600527.SH,1200.50 # 现金分红入账（幂等）
    python run_live.py --split 600527.SH,1.3       # 送转（10送3→1.3，数量×/成本÷）
    python run_live.py --statement                 # 对账：用当日对账单核对账本
    python run_live.py --statement --fix           # 对账并以券商为准修正账本
    python run_live.py --statement-template        # 生成对账单模板（手工誊录券商数据）

QMT 就位后: --execute qmt --account XXXX（见 live/broker.py 前置条件）
对账单目录: live/statements/YYYY-MM-DD.json（含敏感信息，勿提交git）
"""

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'live', 'ledger.json')


def get_prices(ds, symbols, as_of):
    """最新收盘价+前收+过期标的检测（与 run_daily 同套逻辑）"""
    import pandas as pd
    cache = ds.preload_batch(list(symbols), as_of - timedelta(days=10), as_of)
    prices = {}
    prev = {}
    stale = set()
    for sym, df in cache.items():
        if df is None or df.empty:
            stale.add(sym)
            continue
        d = df[df['date'] <= pd.Timestamp(as_of)]
        if d.empty:
            stale.add(sym)
            continue
        last_dt = d['date'].iloc[-1]
        # 行情过期检测：最后bar距执行日超过5个自然日视为停牌/延迟
        if (pd.Timestamp(as_of) - last_dt).days > 5:
            stale.add(sym)
            continue
        prices[sym] = float(d['close'].iloc[-1])
        if len(d) > 1:
            prev[sym] = float(d['close'].iloc[-2])
    return prices, prev, stale


def main():
    parser = argparse.ArgumentParser(description='Live trading driver (paper/QMT)')
    parser.add_argument('--init', type=float, default=None, help='初始入金')
    parser.add_argument('--signal', action='store_true', help='生成目标持仓与指令')
    parser.add_argument('--execute', choices=['paper', 'qmt'], default=None)
    parser.add_argument('--top', type=int, default=100)
    parser.add_argument('--nav', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--ledger', default=LEDGER_PATH)
    # 资金流水 / 公司行动
    parser.add_argument('--deposit', type=float, default=None, help='入金')
    parser.add_argument('--withdraw', type=float, default=None, help='出金')
    parser.add_argument('--dividend', default=None, help='现金分红: 代码,税后金额[,除息日]')
    parser.add_argument('--split', default=None, help='送转: 代码,比例(10送3=1.3)[,除权日]')
    # 账实核对
    parser.add_argument('--statement', nargs='?', const='__today__', default=None,
                        help='对账（默认 live/statements/今日.json，可指定路径）')
    parser.add_argument('--fix', action='store_true', help='对账后以券商为准修正账本')
    parser.add_argument('--statement-template', action='store_true',
                        help='生成对账单模板JSON')
    parser.add_argument('--check-qmt', action='store_true',
                        help='检查 QMT 环境是否就绪（路径/账号/xtquant），不做交易')
    parser.add_argument('--yes', action='store_true',
                        help='实盘(qmt)下单免交互确认（用于已核对的批处理）')
    parser.add_argument('--outside-window', action='store_true',
                        help='QMT 允许非收盘竞价窗口委托（默认窗口外拒单；后果自担）')
    args = parser.parse_args()

    from live import Ledger, make_broker
    ledger = Ledger(args.ledger)

    if args.check_qmt:
        from live import QMTBroker
        env = QMTBroker.check_env()
        print("QMT 环境检查：")
        for k, v in env.items():
            if k != 'ready':
                print(f"  {k}: {v}")
        print(f"结论：{'✅ 就绪（可 --execute qmt）' if env['ready'] else '❌ 未就绪（按上面 issues 逐项修复）'}"
              .replace('✅', '[OK]').replace('❌', '[X]'))
        return

    if args.init is not None:
        ledger.initialize(args.init)
        print(f"账本初始化：入金 {args.init:,.0f} → {args.ledger}")
        return

    if args.status:
        nav = ledger.last_nav()
        n_trades = len(ledger._d['trades'])
        recon = ledger._d['meta'].get('last_recon')
        print(f"持仓 {len(ledger.positions)} 只 | 现金 {ledger.cash:,.0f} | "
              f"成交 {n_trades} 笔 | 最新净值 {nav if nav else '未记录'}")
        if recon:
            outcome = '一致' if recon['ok'] else f"{recon['n_diffs']} 项差异"
            print(f"上次对账: {recon['date']} {outcome}（来源 {recon.get('source', '?')}）")
        else:
            print("上次对账: 从未（建议 python run_live.py --statement-template 后每日核对）")
        open_ex = ledger.open_exceptions()
        if open_ex:
            print(f"待核对异常: {len(open_ex)} 项（超额卖出等，人工核对后 resolve_exception）")
            for e in open_ex[:5]:
                print(f"  [{e.get('date')}] {e.get('type')} {e.get('symbol')} "
                      f"卖{e.get('quantity')}>持{e.get('holding')}")
        for s, qty in list(ledger.positions.items())[:10]:
            print(f"  {s}: {qty} 股 @成本 {ledger._d['cost'].get(s, 0):.2f}")
        return

    # ---- 资金流水 / 公司行动 ----
    if args.deposit is not None:
        cash = ledger.deposit(args.deposit)
        print(f"入金 {args.deposit:,.0f} → 现金 {cash:,.2f}（已记资金流水）")
        return
    if args.withdraw is not None:
        cash = ledger.withdraw(args.withdraw)
        print(f"出金 {args.withdraw:,.0f} → 现金 {cash:,.2f}（已记资金流水）")
        return
    if args.dividend:
        parts = args.dividend.split(',')
        sym, amt = parts[0], float(parts[1])
        exd = parts[2] if len(parts) > 2 else ''
        if ledger.apply_dividend(sym, amt, ex_date=exd):
            print(f"分红入账：{sym} +{amt:,.2f}（事件幂等，重复执行不会重复入账）")
        else:
            print(f"分红事件已存在，跳过：{sym} {amt:,.2f}")
        return
    if args.split:
        parts = args.split.split(',')
        sym, ratio = parts[0], float(parts[1])
        exd = parts[2] if len(parts) > 2 else ''
        if ledger.apply_split(sym, ratio, ex_date=exd):
            qty = ledger.positions.get(sym, 0)
            print(f"送转入账：{sym} ×{ratio} → 现持有 {qty} 股（成本已摊薄）")
        else:
            print(f"送转事件已存在，跳过：{sym} ×{ratio}")
        return

    # ---- 账实核对 ----
    if args.statement_template:
        from live.reconcile import write_statement_template
        p = write_statement_template()
        print(f"对账单模板已生成: {p}\n请按券商App/交割单填写 cash / positions / trades 后运行:\n"
              f"  python run_live.py --statement")
        return
    if args.statement is not None:
        from live.reconcile import load_statement, statement_path_for, Reconciler
        path = statement_path_for() if args.statement == '__today__' else args.statement
        if not os.path.isfile(path):
            print(f"对账单不存在: {path}\n先运行 python run_live.py --statement-template 生成模板并填写")
            return
        stmt = load_statement(path)
        rep = Reconciler().reconcile(ledger, stmt)
        print(f"\n账实核对 @ {stmt.as_of}（来源 {stmt.source}）：{rep.verdict_plain}")
        for d in rep.diffs:
            print(f"  [{d.kind_cn}] {d.symbol or ''} {d.detail}")
        if not rep.diffs:
            print("  现金/持仓/成交流水/费用 全部一致")
        if args.fix:
            actions = Reconciler().apply_adjustments(ledger, stmt, rep)
            print("\n修正动作：")
            for a in actions:
                print(f"  - {a}")
        else:
            ledger._d['meta']['last_recon'] = {'date': stmt.as_of, 'ok': rep.is_ok,
                                               'n_diffs': len(rep.diffs), 'source': stmt.source}
            ledger.save()
            if rep.diffs:
                print("\n（仅报告未修正；加 --fix 以券商为准修正账本）")
        return

    if args.nav:
        from data.layered import LayeredDataSource
        ds = LayeredDataSource()
        as_of = date.today()
        prices, _, _ = get_prices(ds, list(ledger.positions), as_of)
        nav = ledger.mark_to_market(prices, as_of)
        init = ledger._d['cash'] + sum(
            ledger._d['cost'].get(s, 0) * q for s, q in ledger._d['positions'].items())
        print(f"净值 @{as_of}: {nav:,.2f}")
        if init > 0:
            print(f"累计投入约 {init:,.0f}，盈亏 {nav - init:+,.0f} ({(nav/init-1):+.2%})")
        return

    if args.signal:
        from data.layered import LayeredDataSource
        from live.signals import generate_production_signal, build_orders
        ds = LayeredDataSource()
        as_of = date.today()

        # QMT 在途订单补记：先补账本（部分成交/撤单残量），再用最新持仓生成订单
        if args.execute == 'qmt':
            try:
                from live import make_broker as _mk, load_lot_map
                _qmt = _mk('qmt', lot_map=load_lot_map(ds),
                           enforce_window=not args.outside_window)
                synced = _qmt.sync_pending(ledger)
                if synced:
                    print(f"QMT 在途补记 {len(synced)} 笔（已入账本）")
                if getattr(_qmt, 'query_failed', False):
                    print("⛔ QMT 查询接口异常——在途订单状态不可确认，"
                          "阻断本次实盘执行（防止重复委托），人工检查后重试")
                    return
            except RuntimeError as e:
                print(f"QMT 不可用，将不执行实盘：\n{e}")

        # 时序（与回测 shift(1) 同构，无前视）：T-1收盘信号 + T日成交
        print(f"生成生产信号 @ 执行日 {as_of}（Top{args.top}，5因子）...")
        psig = generate_production_signal(ds, as_of, top_n=args.top)
        targets, signal_date = psig.targets, psig.signal_date
        if not targets:
            # 信号为空≠无目标股：可能是数据源故障/因子矩阵为空/池加载失败。
            # 必须走 halt 流程留痕（不静默退出——已有持仓仍需风控管理）。
            print("⛔ 信号为空（数据源故障/因子矩阵为空/池加载失败）——进入停机流程")
            from live.risk_gate import RiskGate, render_gate_markdown
            from live.risk_gate import RiskDecision
            halt_d = RiskDecision(action='halt', exposure=0.0)
            halt_d.data_issues.append('生产信号为空：数据链路故障或因子矩阵异常')
            halt_d.reasons.append('[L2] 信号生成失败——禁止一切交易，人工检查数据链路')
            print("\n".join(render_gate_markdown(halt_d)))
            # 仍然 mark_to-market 以记录当日状态
            prices_now, _, _ = get_prices(ds, set(ledger.positions), as_of)
            if ledger.positions:
                ledger.mark_to_market(prices_now, as_of)
            return
        if not psig.st_ok:
            print(f"警告：ST数据不可用（信号日{signal_date}），撮合按非ST涨跌幅处理并禁止新买入")
        print(f"信号日 {signal_date}（T-1收盘，PIT）| 目标持仓 {len(targets)} 只，评分区间 "
              f"[{psig.scores[targets[-1]]:.2f}, {psig.scores[targets[0]]:.2f}]")

        prices, prev, stale_syms = get_prices(ds, set(targets) | set(ledger.positions), as_of)

        # 风控前净值快照（halt时恢复——错误价格不写入历史）
        _nav_snapshot = list(ledger._d['nav_history'])
        if ledger.positions:
            ledger.mark_to_market(prices, as_of)

        nav = ledger.last_nav() or ledger.cash
        cur_w = ledger.current_weights(prices)
        orders = build_orders(targets, cur_w, nav, top_n=args.top)

        # ST 集合由生产信号入口统一加载（psig.st_symbols/st_ok，信号/撮合/闸门同源）

        # 执行风控闸门（L2停机/L3熔断/L4强制卖出）——改写后才可执行
        from live.risk_gate import RiskGate, GateConfig
        from scripts.trading_switch import effective_state
        bench_cache = ds.preload_batch(['000300.SH'], as_of - timedelta(days=20), as_of)
        gate = RiskGate(GateConfig(trading_switch=effective_state()))
        decision = gate.review(orders, ledger, prices, prev, targets,
                               bench_df=bench_cache.get('000300.SH'), as_of=as_of,
                               st_data_ok=psig.st_ok, stale_symbols=stale_syms)
        print(f"\n风控闸门：{decision.verdict_plain}｜仓位上限 {decision.exposure:.0%}｜"
              f"放行 {len(decision.orders)} / 拦截 {len(decision.suppressed)} / "
              f"强制卖出 {len(decision.forced_sells)}")
        for i in decision.data_issues:
            print(f"  [L2] {i}")
        for r in decision.reasons:
            print(f"  {r}")
        for o in decision.forced_sells:
            print(f"  [L4] 强制卖出 {o.symbol} {o.quantity}股（{o.note}）")

        if decision.action == 'halt':
            # 恢复净值快照：错误价格不写入历史
            ledger._d['nav_history'] = _nav_snapshot
            ledger.save()
            print("\n数据停机，今日不执行任何指令。")
            return
        orders = decision.orders

        print(f"\n调仓指令 {len(orders)} 条：")
        for o in orders:
            amt = o.amount or 0
            px = prices.get(o.symbol, 0)
            print(f"  {o.side:4s} {o.symbol:10s} ~{amt:,.0f}元 @{px:.2f} [{o.note}]")

        if args.execute in ('paper', 'qmt'):
            try:
                from live import load_lot_map
                from scripts.trading_switch import effective_state as _es
                broker = make_broker(args.execute, lot_map=load_lot_map(ds),
                                     enforce_window=not args.outside_window,
                                     switch_state=_es())
            except RuntimeError as e:
                print(f"\n{e}")
                return
            # 实盘二次确认（--yes 免确认，用于已核对的批处理）
            if args.execute == 'qmt' and not args.yes:
                buy_total = sum(o.amount or 0 for o in orders if o.side == 'BUY')
                try:
                    tip = input(f"\n实盘委托 {len(orders)} 条（买入合计约 {buy_total:,.0f} 元）"
                                f"将提交至 QMT 账户，确认执行？[y/N] ")
                except EOFError:
                    tip = 'n'
                if tip.strip().lower() != 'y':
                    print("已取消，未提交任何委托。")
                    return
            # 撮合用执行日(T)ST集合（当日戴帽/摘帽以执行日为准）
            fills = broker.execute(orders, prices, as_of, prev_prices=prev,
                                   available_cash=ledger.cash,
                                   st_symbols=psig.exec_st_symbols,
                                   positions=dict(ledger.positions),
                                   **({'ledger': ledger}
                                      if getattr(broker, 'ATOMIC_LEDGER_BOOKING', False)
                                      else {}))
            # QMT 已在 execute 内原子入账（ATOMIC_LEDGER_BOOKING）；此处
            # 再跑一遍 apply_fills 幂等去重（trade_id 相同不重复入账），
            # Paper 则在此首次入账。
            booked, dupes, blocked = ledger.apply_fills(fills)
            filled = [f for f in fills if f.quantity > 0]
            print(f"\n{args.execute.upper()} 成交 {len(filled)}/{len(orders)}"
                  f"（入账 {len(booked)}，去重 {len(dupes)}）：")
            for f in fills:
                mark = 'OK' if f.quantity > 0 else f.status
                print(f"  {mark:20s} {f.order.symbol} {f.quantity}股 @ {f.price:.2f}")
            nav_now = ledger.mark_to_market(prices, as_of)
            print(f"\n账本已更新，净值 {nav_now:,.2f}")
            if args.execute == 'qmt':
                print("建议立即对账：QMTBroker.fetch_statement() 生成对账单后 "
                      "python run_live.py --statement")
        else:
            print("\n（仅生成指令未执行；--execute paper 模拟入账 / qmt 实盘委托）")


if __name__ == '__main__':
    main()
