#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动对账 —— QMT fetch_statement → 保存原始快照 → reconcile → 写入 last_recon → 告警

用法:
  python scripts/auto_reconcile.py          # QMT自动对账
  python scripts/auto_reconcile.py --manual live/statements/2026-10-14.json
"""
import json
from dataclasses import asdict
import os
import sys
import zlib
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2026-11-04 收敛：移除模块级全量 load_dotenv()（.env 读取仅限授权的
# Oracle 连接参数，且只允许经由 data/oracle_preparation 的受限键读取）。
# 本脚本所需环境变量由调用方显式提供。

RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'runtime', 'reconciliation')


def save_raw_snapshot(stmt_data: dict) -> str:
    """保存原始券商回报（审计用）。

    审计 2026-11-19 三-8：①日期严格解析——外部提供的 as_of 不再直接
    进文件名/路径；②同日多次对账不再互相覆盖：文件名含微秒时间+内容
    摘要，快照路径写进对账记录，事后可追溯每次修正依据的原始单据。"""
    as_of = stmt_data.get('as_of')
    try:
        d = date.fromisoformat(str(as_of)[:10])
    except (TypeError, ValueError):
        raise ValueError(f"对账快照 as_of 非法，拒绝保存: {as_of!r}") from None
    os.makedirs(RAW_DIR, exist_ok=True)
    payload = json.dumps(stmt_data, ensure_ascii=False, sort_keys=True, indent=1)
    digest = f"{datetime.now():%H%M%S%f}_{zlib.crc32(payload.encode('utf-8')):08x}"
    path = os.path.join(RAW_DIR, f"{d.isoformat()}_{digest}.json")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(payload)
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

    # 写入 last_recon（含快照标识，审计 2026-11-19 三-8）
    ledger.record_recon(stmt.as_of, rep.is_ok, len(rep.diffs),
                        source=stmt.source, fixed=False, snapshot=raw_path)

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
            result['ok'] = rec.reconcile(ledger, stmt).is_ok
        return 0 if result['ok'] else 2

    # QMT 自动对账
    print("尝试QMT自动对账...")
    broker = None
    try:
        from live import QMTBroker
        broker = QMTBroker()
        broker.connect()
        stmt_data = asdict(broker.fetch_statement())
        stmt_data['source'] = 'qmt'
        result = run_reconciliation(ledger, stmt_data)
        print(f"    来源: qmt (confidence: high)")
        print(f"    原始快照: {result['raw_snapshot']}")

        if args.fix and not result['ok']:
            # B8硬拦截（10-22第六轮审查#2）：QMT对账的成交ID格式与账本
            # 不一致（order_id vs order_id#cumvol），--fix会重复补记。
            # 在ID匹配验证前，QMT来源的自动修正一律禁止。
            print("⛔ QMT对账的 --fix 已禁用（成交ID映射未验证，重复补记风险）")
            print("   差异请人工核对后手动修正，或等B8（ID规范化）修复后启用")
            return 2
        return 0 if result['ok'] else 2
    except Exception as e:
        print(f"[!] QMT对账失败: {e}")
        print("    使用手工方式: python scripts/auto_reconcile.py --manual live/statements/今日.json")
        sys.exit(1)
    finally:
        if broker is not None:
            broker.disconnect()


if __name__ == '__main__':
    sys.exit(main())
