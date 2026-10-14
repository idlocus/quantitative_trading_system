#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
账实核对（Reconciliation）—— JSON账本 vs 券商账户 的每日对账

覆盖八类问题：
1. 券商实际成交回报：BrokerStatement.trades 与账本 trades 逐笔匹配
2. 部分成交：quantity>0 即入账（ledger侧）；对账侧按实际成交回报匹配
3. 撤单：券商无成交 → 账本若有对应记录则报 trade_missing_broker
4. 手续费差异：匹配上的成交比较 费用合计，超容差报 fee_diff
5. 分红和送转：CLI 显式入账（apply_dividend/apply_split，幂等）
6. 资金流水：ledger.cash_log 全量留痕，对账修正也入流水
7. 持仓数量不一致：逐票比对 → position_diff（--fix 时以券商为准修正）
8. 重复执行和重复入账：dedup_key 去重 + broker侧同key重复检测

对账文件：live/statements/YYYY-MM-DD.json（手工/QMT导出均可），
        run_daily pm 阶段发现当日对账文件自动核对并写入日志。
"""

import json
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

STATEMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'statements')

# 差异类型
DIFF_CASH = 'cash_diff'                    # 现金不一致
DIFF_POSITION = 'position_diff'            # 持仓数量不一致
DIFF_TRADE_MISSING_LEDGER = 'trade_missing_ledger'   # 券商有、账本无（漏记/部分成交未入账）
DIFF_TRADE_MISSING_BROKER = 'trade_missing_broker'   # 账本有、券商无（撤单后误记/重复入账嫌疑）
DIFF_TRADE_DUP = 'trade_duplicate'         # 券商回报自身重复
DIFF_FEE = 'fee_diff'                      # 手续费不一致
DIFF_QTY = 'qty_diff'                      # 同ID成交数量不一致
DIFF_PRICE = 'price_diff'                  # 同ID成交价格不一致
DIFF_COST_BASIS = 'cost_basis_unknown'     # 持仓多出但无成交依据


@dataclass
class BrokerStatement:
    """券商账户快照（对账基准，券商为准）"""
    as_of: str
    cash: float
    positions: Dict[str, int]
    trades: List[dict] = field(default_factory=list)   # 券商成交流水
    source: str = 'manual'                              # manual / qmt / csv
    note: str = ''

    @classmethod
    def from_dict(cls, d: dict) -> 'BrokerStatement':
        for k in ('as_of', 'cash', 'positions'):
            if k not in d:
                raise ValueError(f"对账文件缺少必填字段: {k}")
        return cls(as_of=str(d['as_of'])[:10], cash=float(d['cash']),
                   positions={s: int(q) for s, q in d['positions'].items()},
                   trades=d.get('trades', []), source=d.get('source', 'manual'),
                   note=d.get('note', ''))

    def __post_init__(self):
        """成交记录规范化（无论从JSON还是代码构造，字段口径统一）"""
        self.trades = self._normalize_trades(self.trades)

    @staticmethod
    def _normalize_trades(trades: List[dict]) -> List[dict]:
        out = []
        for t in trades or []:
            for k in ('date', 'symbol', 'side', 'price', 'quantity'):
                if k not in t:
                    raise ValueError(f"券商成交缺少字段 {k}: {t}")
            if int(t['quantity']) <= 0:
                raise ValueError(f"券商成交数量≤0（非法对账数据）: {t}")
            out.append({
                'date': str(t['date'])[:10], 'symbol': t['symbol'],
                'side': str(t['side']).upper(), 'price': float(t['price']),
                'quantity': int(t['quantity']),
                'commission': float(t.get('commission', 0) or 0),
                'taxes': float(t.get('taxes', 0) or 0),
                'trade_id': str(t.get('trade_id', '') or ''),
            })
        return out


def load_statement(path: str) -> BrokerStatement:
    with open(path, encoding='utf-8') as f:
        return BrokerStatement.from_dict(json.load(f))


def statement_path_for(as_of=None) -> str:
    as_of = as_of or date.today()
    return os.path.join(STATEMENTS_DIR, f"{as_of.isoformat()}.json")


def write_statement_template(path: str = None) -> str:
    """生成对账文件模板（手工填券商数据用）"""
    path = path or statement_path_for()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tpl = {
        'as_of': date.today().isoformat(),
        'source': 'manual',
        'note': '从券商App截图/交割单手工誊录；QMT就位后可自动生成',
        'cash': 0.0,
        'positions': {},
        'trades': [],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(tpl, f, ensure_ascii=False, indent=2)
    return path


@dataclass
class ReconDiff:
    kind: str
    detail: str
    symbol: str = ''
    ledger_value: float = 0.0
    broker_value: float = 0.0

    @property
    def kind_cn(self) -> str:
        return {DIFF_CASH: '现金不一致', DIFF_POSITION: '持仓数量不一致',
                DIFF_TRADE_MISSING_LEDGER: '券商有成交·账本漏记',
                DIFF_TRADE_MISSING_BROKER: '账本有成交·券商无（误记/重复入账嫌疑）',
                DIFF_TRADE_DUP: '券商回报重复', DIFF_FEE: '手续费差异',
                DIFF_QTY: '成交数量不一致', DIFF_PRICE: '成交价格不一致',
                DIFF_COST_BASIS: '成本基准不可确认'}.get(self.kind, self.kind)


@dataclass
class ReconReport:
    as_of: str = ''
    source: str = ''
    diffs: List[ReconDiff] = field(default_factory=list)
    matched_trades: int = 0
    ledger_trades_n: int = 0
    broker_trades_n: int = 0

    @property
    def is_ok(self) -> bool:
        return not self.diffs

    @property
    def verdict_cn(self) -> str:
        return '✅ 账实一致' if self.is_ok else f'❌ {len(self.diffs)} 项差异'

    @property
    def verdict_plain(self) -> str:
        """无emoji版（GBK控制台/print安全）"""
        return '账实一致' if self.is_ok else f'{len(self.diffs)} 项差异'

    def render_markdown(self) -> List[str]:
        lines = ["## 六、账实核对", "",
                 f"- 对账基准：`{self.source}` @ {self.as_of}",
                 f"- 结论：**{self.verdict_cn}**",
                 f"- 成交匹配：{self.matched_trades} 笔（账本 {self.ledger_trades_n} / 券商 {self.broker_trades_n}）"]
        if self.diffs:
            lines += ["", "| 类型 | 代码 | 账本 | 券商 | 说明 |",
                      "|------|------|------|------|------|"]
            for x in self.diffs:
                lv = f"{x.ledger_value:,.0f}" if x.kind == DIFF_CASH else \
                    (f"{int(x.ledger_value)}" if x.kind == DIFF_POSITION else '—')
                bv = f"{x.broker_value:,.0f}" if x.kind == DIFF_CASH else \
                    (f"{int(x.broker_value)}" if x.kind == DIFF_POSITION else '—')
                lines.append(f"| {x.kind_cn} | {x.symbol or '—'} | {lv} | {bv} | {x.detail} |")
        lines.append("")
        return lines


class Reconciler:
    """每日账实核对：账本 vs 券商对账单（券商为准）"""

    def __init__(self, cash_tol: float = 1.0, fee_tol: float = 1.0, qty_tol: int = 0):
        self.cash_tol = cash_tol
        self.fee_tol = fee_tol
        self.qty_tol = qty_tol

    # ================= 核对 =================

    def reconcile(self, ledger, stmt: BrokerStatement) -> ReconReport:
        """全量核对：现金 / 持仓 / 成交流水 / 费用。返回差异报告（不修改账本）。"""
        rep = ReconReport(as_of=stmt.as_of, source=stmt.source)
        self._check_cash(ledger, stmt, rep)
        self._check_positions(ledger, stmt, rep)
        self._check_trades(ledger, stmt, rep)
        level = logger.info if rep.is_ok else logger.warning
        level(f"[Recon] {rep.verdict_plain} @ {stmt.as_of}（来源:{stmt.source}）"
              + (f"：{'；'.join(d.detail for d in rep.diffs[:5])}" if rep.diffs else ""))
        return rep

    def _check_cash(self, ledger, stmt, rep):
        diff = ledger.cash - stmt.cash
        if abs(diff) > self.cash_tol:
            rep.diffs.append(ReconDiff(
                DIFF_CASH, f"现金差 {diff:+,.2f}（超出容差{self.cash_tol}元）",
                ledger_value=ledger.cash, broker_value=stmt.cash))

    def _check_positions(self, ledger, stmt, rep):
        all_syms = set(ledger.positions) | set(stmt.positions)
        for s in sorted(all_syms):
            lq, bq = ledger.positions.get(s, 0), stmt.positions.get(s, 0)
            if abs(lq - bq) > self.qty_tol:
                tag = '（账本多出：疑重复入账/送转未记）' if lq > bq else '（券商多出：疑漏记成交/送转未记）'
                rep.diffs.append(ReconDiff(
                    DIFF_POSITION, f"数量差 {lq - bq:+d}{tag}", symbol=s,
                    ledger_value=lq, broker_value=bq))

    def _check_trades(self, ledger, stmt, rep):
        """成交逐笔匹配：trade_id 优先，否则要素指纹。
        对账单未提供成交流水（trades为空）时跳过本项——手工单可只填现金+持仓。"""
        as_of = stmt.as_of
        if not stmt.trades:
            return
        ledger_trades = [t for t in ledger._d['trades'] if str(t['date'])[:10] == as_of]
        rep.ledger_trades_n = len(ledger_trades)
        rep.broker_trades_n = len(stmt.trades)

        def key_of(t):
            tid = t.get('trade_id') or ''
            if tid:
                return f'id:{tid}'
            return '|'.join([str(t['date'])[:10], t['symbol'], t['side'],
                             str(int(t['quantity'])), f"{float(t['price']):.3f}"])

        # 券商侧自身重复检测
        seen, broker_map = {}, {}
        for t in stmt.trades:
            k = key_of(t)
            if k in broker_map:
                rep.diffs.append(ReconDiff(
                    DIFF_TRADE_DUP, f"券商回报重复：{t['symbol']} {t['side']} "
                    f"{t['quantity']}股@{t['price']:.3f}（key={k}）", symbol=t['symbol']))
                continue
            broker_map[k] = t

        ledger_map = {key_of(t): t for t in ledger_trades}

        # 券商有、账本无 → 漏记（含部分成交只入账一部分的场景：数量不同也在此报）
        for k, bt in broker_map.items():
            lt = ledger_map.get(k)
            if lt is None:
                # 数量不同的弱匹配（同日同代码同方向）提示部分成交未足额入账
                weak = [t for t in ledger_trades
                        if t['symbol'] == bt['symbol'] and t['side'] == bt['side']]
                hint = f"（账本当日同方向仅有 {len(weak)} 笔，疑部分成交漏记）" if weak else ""
                rep.diffs.append(ReconDiff(
                    DIFF_TRADE_MISSING_LEDGER,
                    f"{bt['side']} {bt['quantity']}股@{bt['price']:.3f} "
                    f"费{bt['commission'] + bt['taxes']:.2f}{hint}", symbol=bt['symbol'],
                    broker_value=bt['quantity']))
            else:
                rep.matched_trades += 1
                # 同 ID 仍需核对数量与价格（此前只查费用——同ID不同数量/价格会被判"一致"）
                qty_diff = abs(int(lt['quantity']) - int(bt['quantity']))
                px_diff = abs(float(lt['price']) - float(bt['price']))
                if qty_diff > 0:
                    rep.diffs.append(ReconDiff(
                        DIFF_QTY, f"数量差 {qty_diff:+d}（账本{lt['quantity']}/券商{bt['quantity']}）",
                        symbol=bt['symbol'], ledger_value=lt['quantity'],
                        broker_value=bt['quantity']))
                if px_diff > 0.005:  # 超1分钱价差
                    rep.diffs.append(ReconDiff(
                        DIFF_PRICE, f"价格差 {px_diff:+.4f}（账本{lt['price']:.4f}/券商{bt['price']:.4f}）",
                        symbol=bt['symbol'], ledger_value=float(lt['price']),
                        broker_value=float(bt['price'])))
                fee_l = float(lt['commission']) + float(lt['taxes'])
                fee_b = bt['commission'] + bt['taxes']
                if abs(fee_l - fee_b) > self.fee_tol:
                    rep.diffs.append(ReconDiff(
                        DIFF_FEE, f"费用差 {fee_l - fee_b:+.2f}（账本{fee_l:.2f}/券商{fee_b:.2f}）",
                        symbol=bt['symbol'], ledger_value=fee_l, broker_value=fee_b))

        # 账本有、券商无 → 撤单后误记 / 重复入账嫌疑
        for k, lt in ledger_map.items():
            if k not in broker_map:
                rep.diffs.append(ReconDiff(
                    DIFF_TRADE_MISSING_BROKER,
                    f"{lt['side']} {lt['quantity']}股@{float(lt['price']):.3f}（key={k}）",
                    symbol=lt['symbol'], ledger_value=lt['quantity']))

    # ================= 修正（--fix，券商为准） =================

    def apply_adjustments(self, ledger, stmt: BrokerStatement, rep: ReconReport) -> List[str]:
        """
        按对账差异修正账本（全部留痕）。仅处理：
        - 持仓数量 → 券商口径
        - 现金 → 券商口径
        - 券商有账本无的成交 → 按券商回报补记
        - 费用差 → 现金修正
        不自动处理：分红送转（需显式 CLI 指令）、trade_missing_broker（人工核查）
        """
        actions = []
        fills = []  # 必须在开头初始化（此前仅在有漏记成交时赋值，无成交差异时UnboundLocalError）
        as_of = stmt.as_of

        # 1) 补记券商成交（先补成交，再对齐数量/现金，避免双重修正）
        from live.broker import Order, Fill
        missing = [d for d in rep.diffs if d.kind == DIFF_TRADE_MISSING_LEDGER]
        if missing:
            keys = {self._broker_key(bt) for bt in stmt.trades}
            booked_keys = {t.get('dedup_key') for t in ledger._d['trades']}
            fills = []
            for bt in stmt.trades:
                if self._broker_key(bt) in booked_keys:
                    continue
                fills.append(Fill(Order(bt['symbol'], bt['side'], 0.0),
                                  price=bt['price'], quantity=bt['quantity'],
                                  commission=bt['commission'], taxes=bt['taxes'],
                                  trade_date=bt['date'], status='filled',
                                  trade_id=bt['trade_id']))
            if fills:
                booked, _, _blocked = ledger.apply_fills(fills)
                actions.append(f"补记券商成交 {len(booked)} 笔")

        # 2) 持仓数量对齐——区分成本基准处理
        all_syms = set(ledger.positions) | set(stmt.positions)
        n_pos = 0
        n_cost_unknown = 0
        booked_syms = {f.order.symbol for f in fills if f.status == 'filled'} if fills else set()
        for s in sorted(all_syms):
            lq, bq = ledger.positions.get(s, 0), stmt.positions.get(s, 0)
            if lq == bq:
                continue
            if bq > lq and s not in booked_syms:
                rep.diffs.append(ReconDiff(
                    DIFF_COST_BASIS,
                    f"券商多出{bq - lq:+d}股但无成交依据（疑送转/未入账交易）",
                    symbol=s, ledger_value=lq, broker_value=bq))
                n_cost_unknown += 1
                continue
            ledger.adjust_position(s, bq, f'账实核对修正@{as_of}')
            n_pos += 1
        if n_pos:
            actions.append(f"修正持仓 {n_pos} 只")
        if n_cost_unknown:
            actions.append(f"⛔ {n_cost_unknown} 只成本基准不可确认")

        # 3) 现金对齐（含费用差净额，统一以券商现金为准）
        if abs(ledger.cash - stmt.cash) > 0.005:
            diff = ledger.adjust_cash(stmt.cash, f'账实核对修正@{as_of}')
            actions.append(f"现金修正 {diff:+,.2f}")

        # 4) 记录对账结果（修正后重新核对；未自动处理项仍会体现为差异）
        rep_after = self.reconcile(ledger, stmt)
        ledger.record_recon(as_of, rep_after.is_ok, len(rep_after.diffs),
                            source=stmt.source, fixed=True)
        ledger.save()
        if any(d.kind == DIFF_TRADE_MISSING_BROKER for d in rep.diffs):
            actions.append("提示：账本有·券商无的成交未自动处理，需人工核查（撤单误记/重复入账）")
        logger.info(f"[Recon] 修正完成：{'；'.join(actions) if actions else '无需修正'}")
        return actions

    def _broker_key(self, bt: dict) -> str:
        tid = bt.get('trade_id') or ''
        if tid:
            return f'id:{tid}'
        return '|'.join([str(bt['date'])[:10], bt['symbol'], bt['side'],
                         str(int(bt['quantity'])), f"{float(bt['price']):.3f}"])
