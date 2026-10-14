#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动对账 —— QMT fetch_statement → 保存原始快照 → reconcile → 写入 last_recon → 告警

用法:
  python scripts/auto_reconcile.py          # QMT自动对账
  python scripts/auto_reconcile.py --manual live/statements/2026-10-14.json
"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def save_raw_snapshot(stmt_data: dict):
    """保存原始券商回报（审计用）"""
    raw_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'runtime', 'reconciliation')
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{stmt_data.get('as_of', date.today())}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stmt_data, f, ensure_ascii=False, indent=1)
    return path


def run_reconciliation(ledger, stmt_data: dict) -> dict:
    """执行对账并写回状态"""
    from live.reconcile import BrokerStatement, Reconciler
    from monitoring.alerts import alert, P0

    stmt = BrokerStatement.from_dict(stmt_data)

    # 保存原始快照
    raw_path = save_raw_snapshot(stmt_data)

    # 对账
    rec = Reconciler()
    rep = rec.reconcile(ledger, stmt)
    result = {
        'date': stmt.as_of,
        'ok': rep.is_ok,
        'n_diffs': len(rep.diffs),
        'source': stmt.source,
        'raw_snapshot': raw_path,
        'matched_trades': rep.matched_trades,
        'diffs': [{'kind': d.kind, 'detail': d.detail} for d in rep.diffs],
    }

    # 写入 last_recon
    ledger.record_recon(stmt.as_of, rep.is_ok, len(rep.diffs),
                        source=stmt.source, fixed=False)

    # 告警
    if not rep.is_ok:
        alert('RECONCILIATION_FAILED', P0,
              f"对账不一致：{len(rep.diffs)}项差异 @ {stmt.as_of}",
              n_diffs=len(rep.diffs), date=stmt.as_of)
        print(f"[X] 对账不一致：{len(rep.diffs)}项差异")
        for d in rep.diffs[:5]:
            print(f"    [{d.kind_cn}] {d.symbol or ''} {d.detail}")
    else:
        print(f"[OK] 对账一致（成交匹配{rep.matched_trades}笔）")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Auto reconciliation')
    parser.add_argument('--manual', default=None,
                        help='手工对账单JSON路径（备用方式）')
    parser.add_argument('--ledger', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'live', 'ledger.json'))
    parser.add_argument('--fix', action='store_true', help='自动修正小差异')
    args = parser.parse_args()

    from live import Ledger
    ledger = Ledger(args.ledger)

    if args.manual:
        # 手工对账单（备用，标记 confidence=low）
        with open(args.manual, encoding='utf-8') as f:
            stmt_data = json.load(f)
        stmt_data.setdefault('source', 'manual')
        result = run_reconciliation(ledger, stmt_data)
        print(f"    来源: manual (confidence: low)")
        if args.fix and not result['ok']:
            from live.reconcile import BrokerStatement, Reconciler
            stmt = BrokerStatement.from_dict(stmt_data)
            rec = Reconciler()
            rep = rec.reconcile(ledger, stmt)
            actions = rec.apply_adjustments(ledger, stmt, rep)
            print(f"    修正: {'; '.join(actions) if actions else '无需修正'}")
        return

    # QMT 自动对账
    print("尝试QMT自动对账...")
    try:
        from live import QMTBroker
        broker = QMTBroker()
        broker.connect()
        stmt_data = broker.fetch_statement()
        stmt_data['source'] = 'qmt'
        result = run_reconciliation(ledger, stmt_data)
        print(f"    来源: qmt (confidence: high)")
        print(f"    原始快照: {result['raw_snapshot']}")

        if args.fix and not result['ok']:
            from live.reconcile import Reconciler, BrokerStatement
            stmt = BrokerStatement.from_dict(stmt_data)
            rec = Reconciler()
            rep = rec.reconcile(ledger, stmt)
            actions = rec.apply_adjustments(ledger, stmt, rep)
            print(f"    修正: {'; '.join(actions) if actions else '无需修正'}")
        broker.disconnect()

    except Exception as e:
        print(f"[!] QMT对账失败: {e}")
        print("    使用手工方式: python scripts/auto_reconcile.py --manual live/statements/今日.json")
        sys.exit(1)


if __name__ == '__main__':
    main()
