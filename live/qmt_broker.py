#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
miniQMT 实盘交易接口

依赖：xtquant（QMT 客户端自带，或 pip install xtquant）
前提：QMT 客户端已启动并登录

配置（.env 或环境变量）：
    QMT_PATH=C:\\国金QMT交易端\\userdata_mini   # QMT 客户端数据目录
    QMT_ACCOUNT_ID=您的资金账号                 # 股票账户ID
    QMT_SESSION_ID=1                           # 会话ID（任意正整数，多策略需不同）

使用前检查清单：
    □ 券商已开通量化交易权限
    □ QMT 客户端已安装并登录
    □ 完成程序化交易报备（券商合规）
    □ .env 中填入 QMT_PATH 和 QMT_ACCOUNT_ID
    □ 先用 simulate 模式验证，再切实盘

首次实盘验证清单（撮合假设成立性，单元测试无法覆盖）：
    □ 运行时间在收盘竞价窗口内（14:50-15:05，execute窗口外会告警）
    □ price_type=LATEST_PRICE(5)（最新价，接近市价）在当前账户/各板块可用（含北交所）
    □ 部分成交/撤单/隔夜在途订单与券商回报一致（对照ledger台账核对）
    □ 小额单跑通全链路：提交→回报→入账→对账（--statement）
"""

import json
import math
import uuid
import copy
import threading
from functools import wraps
from utils.state_file import exclusive_file, replace_json
import os
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .broker import Order, Fill, Broker, valid_order

if TYPE_CHECKING:      # 注解引用（防循环导入；pyflakes 可解析）
    from live.reconcile import BrokerStatement

from utils.logger import get_logger

logger = get_logger(__name__)

from trading import config as chain_cfg


def _est_fees(side: str, value: float):
    """预估费用（唯一实现=trading/rules.estimate_fees，费率=trading/config；
    实际以券商回报为准，账实核对修正差异）"""
    from trading.rules import estimate_fees
    return estimate_fees(side, value)


def check_execution_window(now=None) -> bool:
    """
    执行时间窗口检查：策略撮合假设=T日收盘价（对应收盘竞价时段成交）。
    A股收盘竞价 14:57-15:00（沪深），留缓冲取 14:50-15:05。

    窗口外默认拒单（enforce_window=True，见 QMTBroker.execute）；
    显式传 enforce_window=False（CLI: --outside-window）方可窗口外执行。
    """
    from datetime import datetime, time as _t
    n = now or datetime.now()
    return _t(14, 50) <= n.time() <= _t(15, 5)

# QMT 配置从环境变量读取（用户填充）
QMT_PATH = os.getenv('QMT_PATH', '')           # QMT 客户端数据目录
QMT_ACCOUNT_ID = os.getenv('QMT_ACCOUNT_ID', '')  # 资金账号
QMT_SESSION_ID = int(os.getenv('QMT_SESSION_ID', '1'))

DEFAULT_ORDER_STATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'runtime', 'qmt_orders.json')   # 运行时状态独立于源码（runtime/，gitignored）


def _load_xt_constants():
    """优先从 xtquant.xtconstant 加载真实常量；未安装时用官方文档值并告警。
    此前项目内自定义了一套错误常量（FILLED=51等），单元测试用同一套错值所以
    全绿——真实环境会永久等待/错撤/丢失台账。"""
    _defaults = {
        'STOCK_BUY': 23, 'STOCK_SELL': 24,
        'LATEST_PRICE': 5, 'FIX_PRICE': 11,
        'ORDER_PENDING': 48, 'ORDER_SUBMITTED': 49,
        'ORDER_PARTIAL': 55, 'ORDER_FILLED': 56,
        'ORDER_CANCELLED': 54, 'ORDER_REJECTED': 57,
    }
    try:
        from xtquant import xtconstant
        out = {}
        for k, dv in _defaults.items():
            out[k] = getattr(xtconstant, k, dv)
        return out
    except ImportError:
        logger.warning("[QMT] xtquant 未安装，使用官方文档常量值（实盘前必须安装并验证）")
        return _defaults


XT = _load_xt_constants()


class QMTOrderState:
    """QMT 订单状态跟踪（从 xtconstant 加载或按官方文档值；禁止在项目内重定义）"""
    PENDING = XT['ORDER_PENDING']      # 48: 待报
    SUBMITTED = XT['ORDER_SUBMITTED']  # 49: 已报
    PARTIAL = XT['ORDER_PARTIAL']      # 55: 部分成交
    FILLED = XT['ORDER_FILLED']        # 56: 全部成交
    CANCELLED = XT['ORDER_CANCELLED']  # 54: 已撤
    REJECTED = XT['ORDER_REJECTED']    # 57: 废单


def _serialized(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._mutex:
            return method(self, *args, **kwargs)
    return locked


class QMTBroker(Broker):
    # 生产入账模式标记（run_daily/run_live 依此传 ledger 实现原子入账）
    ATOMIC_LEDGER_BOOKING = True
    """
    miniQMT 实盘交易通道

    接口与 PaperBroker 一致（execute），可直接替换。
    前置条件：QMT 客户端运行中，.env 已配置。
    """

    def __init__(self, account_id: str = None, qmt_path: str = None,
                 state_path: str = DEFAULT_ORDER_STATE,
                 order_timeout: float = 30,
                 enforce_window: bool = True,
                 switch_state=None):
        """enforce_window=True（默认）：非收盘竞价窗口直接拒单（策略撮合假设=T日
        收盘价，窗口外成交价将系统性偏离假设）。需窗口外执行时显式传 False
        （CLI: run_daily/run_live 的 --outside-window）。
        switch_state：交易控制三态（显式注入；2026-10-15 前此处每次 execute
        读全局状态文件且失败静默放行——热路径依赖文件系统且 fail-open）。
        生产由 make_broker → run_daily/run_live 启动时读一次注入。"""
        self.account_id = account_id or QMT_ACCOUNT_ID
        self.qmt_path = qmt_path or QMT_PATH
        self.order_timeout = order_timeout       # 单笔等待终态时长（秒）
        self.enforce_window = enforce_window
        if switch_state is None:
            from scripts.trading_switch import effective_state
            switch_state = effective_state
        self.switch_state = switch_state
        self._xt_trader = None
        self._account = None
        self._connected = False
        self._order_callbacks = {}  # order_id -> callback info
        # 在途订单台账（订单生命周期状态，崩溃/重启后可续）：{oid: {symbol, side, qty, booked, status}}
        self.state_path = state_path
        self._mutex = threading.RLock()
        self._pending = self._load_state()
        self._persisted = copy.deepcopy(self._pending)

    def name(self):
        return 'QMTBroker'

    # ==================== 连接管理 ====================

    def connect(self) -> bool:
        """连接 QMT 客户端（客户端必须已在运行）"""
        if self._connected:
            return True

        if not self.qmt_path:
            raise RuntimeError(
                "QMT_PATH 未配置。请在 .env 中设置 QMT_PATH=<QMT客户端数据目录>")

        if not self.account_id:
            raise RuntimeError(
                "QMT_ACCOUNT_ID 未配置。请在 .env 中设置 QMT_ACCOUNT_ID=<资金账号>")

        try:
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
        except ImportError:
            raise RuntimeError(
                "xtquant 未安装。请安装 QMT 客户端后确认 Python 环境可导入 xtquant")

        try:
            # 创建交易实例
            self._xt_trader = XtQuantTrader(self.qmt_path, QMT_SESSION_ID)
            self._xt_trader.start()

            result = self._xt_trader.connect()
            if result != 0:
                raise RuntimeError(f"QMT 连接失败（错误码 {result}）。请确认 QMT 客户端已启动并登录")

            # 订阅回调
            self._xt_trader.register_callback(self)

            # 创建账户对象
            self._account = StockAccount(self.account_id)
            result = self._xt_trader.subscribe(self._account)
            if result != 0:
                raise RuntimeError(f"QMT 账户订阅失败（错误码 {result}）")

            self._connected = True
            self._requires_ledger = True
            logger.info(f"[QMT] 已连接 | 账户 {self.account_id}")
            return True

        except Exception as e:
            self._connected = False
            logger.error(f"[QMT] 连接失败: {e}")
            raise

    def disconnect(self):
        """断开连接"""
        if self._xt_trader:
            self._xt_trader.stop()
            self._connected = False
            logger.info("[QMT] 已断开")

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected

    # ==================== 账户查询 ====================

    def get_cash(self) -> float:
        """查询可用资金"""
        self._ensure_connected()
        asset = self._xt_trader.query_stock_asset(self._account)
        if asset:
            return float(asset.cash)
        return 0.0

    def get_positions(self) -> Dict[str, int]:
        """查询当前持仓 {symbol: quantity}"""
        self._ensure_connected()
        positions = self._xt_trader.query_stock_positions(self._account)
        result = {}
        if positions:
            for pos in positions:
                if pos.volume > 0:
                    result[pos.stock_code] = int(pos.volume)
        return result

    def get_total_asset(self) -> float:
        """查询总资产"""
        self._ensure_connected()
        asset = self._xt_trader.query_stock_asset(self._account)
        if asset:
            return float(asset.total_asset)
        return 0.0

    def fetch_statement(self, as_of=None) -> 'BrokerStatement':
        """
        拉取券商对账单（账实核对基准）：
        资金 + 持仓 + 当日成交流水（含成交回报ID，供对账去重）。
        （返回类型 live.reconcile.BrokerStatement——模块底部运行时导入，
        此处字符串注解防循环依赖；pyflakes 白名单见 conftest 之外说明）

        需要 QMT 客户端运行中；未连接时抛 RuntimeError。
        """
        from live.reconcile import BrokerStatement
        as_of = as_of or date.today()
        if str(as_of) != date.today().isoformat():
            raise ValueError("QMT provides current-day statements only")
        self._ensure_connected()

        asset = self._xt_trader.query_stock_asset(self._account)
        # B8补强：None≠零值——查询失败时拒绝对账
        if asset is None:
            raise RuntimeError("QMT资产查询None——数据不可用，拒绝对账")
        cash = float(asset.cash)

        raw_pos = self._xt_trader.query_stock_positions(self._account)
        if raw_pos is None:
            raise RuntimeError("QMT持仓查询None——数据不可用（非空仓）")
        positions = {}
        for pos in raw_pos:
            if not float(pos.volume).is_integer() or pos.volume < 0:
                raise ValueError('Invalid broker position')
            if pos.volume > 0:
                positions[pos.stock_code] = int(pos.volume)

        raw_trades = self._xt_trader.query_stock_trades(self._account)
        if raw_trades is None:
            raise RuntimeError("QMT成交查询None——流水不可用，不能判账实一致")
        trades = []
        for t in raw_trades:
            trades.append(self._execution_row(t))

        return BrokerStatement(as_of=str(as_of)[:10], cash=cash, positions=positions,
                               trades=trades, source='qmt')

    @staticmethod
    def _execution_row(t):
        if not getattr(t, 'traded_time', 0):
            raise ValueError("Missing broker execution time")
        side = {XT['STOCK_BUY']: 'BUY', XT['STOCK_SELL']: 'SELL'}.get(getattr(t, 'order_type', None))
        if side is None:
            side = {48: 'BUY', 49: 'SELL'}.get(getattr(t, 'offset_flag', None))
        tid = str(getattr(t, 'traded_id', '') or '')
        if side is None or not tid:
            raise ValueError("Unidentified broker execution")
        qty = float(t.traded_volume)
        price = float(t.traded_price)
        fee = float(getattr(t, 'traded_fee', 0) or 0)
        if not (qty.is_integer() and qty > 0 and math.isfinite(price) and price > 0
                and math.isfinite(fee) and fee >= 0):
            raise ValueError("Invalid execution quantities or fees")
        return dict(date=datetime.fromtimestamp(t.traded_time).date().isoformat(),
                    symbol=t.stock_code, side=side, price=price, quantity=int(qty),
                    commission=fee, taxes=0., trade_id=tid)

    def _book_executions(self, order, ref, traded, ledger, status):
        rec = self._pending[ref]
        real_oid = self._resolve_oid(ref)
        if not real_oid or ledger is None:
            raise ValueError('Execution booking requires broker order ID and ledger')
        if rec.get('booked') and 'execution_ids' not in rec:
            raise ValueError('Legacy cumulative fills require reconciliation')
        rows = [self._execution_row(t) for t in self._xt_trader.query_stock_trades(self._account)
                if str(t.order_id) == str(real_oid)]
        ids = [row['trade_id'] for row in rows]
        if (len(set(ids)) != len(ids) or sum(r['quantity'] for r in rows) != traded
                or not set(rec.get('execution_ids', [])).issubset(ids)
                or any(r['symbol'] != order.symbol or r['side'] != order.side for r in rows)):
            raise ValueError('Order and execution records disagree')
        fills = [Fill(order, r['price'], r['quantity'], r['commission'], r['taxes'],
                      r['date'], status, r['trade_id']) for r in rows]
        booked, _, blocked = ledger.apply_fills(fills)
        if blocked:
            raise ValueError('Execution blocked by ledger')
        rec['execution_ids'] = ids
        quantity = sum(f.quantity for f in booked)
        value = sum(f.quantity * f.price for f in booked)
        return Fill(order, value / quantity if quantity else 0., quantity,
                    sum(f.commission for f in booked), sum(f.taxes for f in booked),
                    rows[-1]['date'], status)

    # ==================== 下单接口 ====================

    def execute(self, orders: List[Order], prices: Dict[str, float],
                trade_date, prev_prices=None, available_cash=None,
                st_symbols=None, positions: Dict[str, int] = None,
                ledger=None) -> List[Fill]:
        """
        执行订单列表（与 PaperBroker 接口一致；st_symbols 仅为接口对齐——
        实盘的涨跌停校验由券商端执行。positions 用于卖出数量合法化：
        零股一次性规则需要知道实际持仓）

        ledger（生产必传，ATOMIC_LEDGER_BOOKING=True）：成交先入账、成功后
        才清退在途台账（R4 原子性）；不传=非原子模式（仅测试用）。

        QMT 实盘流程：
        1. 交易控制状态检查（HALT_ALL→拒单；CLOSE_ONLY→过滤买入）
        2. 检查执行窗口（撮合假设=T日收盘价→收盘竞价时段；窗口外大声警告）
        3. 按顺序提交订单（先卖后买，释放资金）
        4. 等待订单回报（超时撤单，部分成交按增量入账）
        5. 返回 Fill 列表
        """
        self._ensure_connected()

        # F3: 交易控制状态在 QMT 侧二次检查（RiskGate 之后的最后防线；
        # 状态来自构造注入，热路径不读文件系统）
        _sw = self.switch_state() if callable(self.switch_state) else self.switch_state
        if getattr(self, '_requires_ledger', False) and not callable(self.switch_state):
            from scripts.trading_switch import effective_state
            live_state = effective_state()
            if live_state == 'HALT_ALL' or _sw == 'NORMAL':
                _sw = live_state
        if _sw not in ('NORMAL', 'CLOSE_ONLY', 'HALT_ALL'):
            _sw = 'HALT_ALL'
        if _sw == 'HALT_ALL':
            logger.error("[QMT] 交易控制=HALT_ALL，拒单+撤所有在途订单")
            # B7（2026-10-22）：紧急停止不仅拦新单——已挂出的旧买单若不撤
            # 仍会成交、继续扩大敞口。撤单尽力而为+留痕（撤不掉的告警）
            try:
                remaining = self.cancel_all_pending()
                if remaining:
                    logger.critical(f"[QMT] HALT_ALL: {remaining} orders still unresolved")
            except Exception as e:
                logger.critical(f"[QMT] HALT_ALL 撤单异常——人工核查券商App: {e}")
            return [Fill(o, 0, 0, 0, 0, trade_date,
                         status='rejected(HALT_ALL)') for o in orders]
        if getattr(self, '_requires_ledger', False) and ledger is None:
            raise ValueError('Live execution requires a ledger')
        if _sw == 'CLOSE_ONLY':
            orders = [o for o in orders if o.side != 'BUY']
            if not orders:
                logger.info("[QMT] CLOSE_ONLY，买入全部过滤")
        if not check_execution_window():
            if self.enforce_window:
                logger.error("[QMT] 当前时间不在收盘竞价窗口(14:50-15:05)——"
                             "撮合假设为T日收盘价，窗口外委托被拒绝。"
                             "确认必须执行时显式传 enforce_window=False"
                             "（CLI: --outside-window）")
                return [Fill(o, 0, 0, 0, 0, trade_date,
                             status='rejected(非收盘竞价窗口)') for o in orders]
            logger.warning("[QMT] 窗口外强制执行（enforce_window=False）——"
                           "成交价将偏离收盘价撮合假设，后果自担")
        fills = []

        # 二轮复审#4：任何券商调用前校验委托要素——非法方向此前被
        # `XT['STOCK_BUY'] if side=='BUY' else SELL` 映射成真实卖单。
        valid_orders, rejected = [], []
        for o in orders:
            if o.side not in ('BUY', 'SELL'):
                logger.error(f"[QMT] 拒绝非法方向委托 {o.symbol} side={o.side!r}")
                rejected.append(Fill(o, 0, 0, 0, 0, trade_date,
                                     status='rejected(invalid side)'))
            elif not valid_order(o, prices.get(o.symbol)):
                rejected.append(Fill(o, 0, 0, 0, 0, trade_date,
                                     status='rejected(invalid quantity)'))
            else:
                valid_orders.append(o)
        fills.extend(rejected)
        if not valid_orders:
            return fills

        # 排序：卖出在前（释放资金供买入）
        sorted_orders = sorted(valid_orders, key=lambda o: 0 if o.side == 'SELL' else 1)

        running_cash = available_cash if available_cash is not None else self.get_cash()

        # 二轮复审#6：execute 开头不再盲目重置 query_failed——存在未确认
        # 终态的在途/待确认记录时保持阻断（防旧单未核实就提交新批）。
        unresolved = self._pending
        if unresolved:
            self.query_failed = True
            logger.error(f"[QMT] {len(unresolved)} 条在途记录未确认终态，"
                         "本批全部拒绝（先 sync_pending 或人工核对）")
            return fills + [Fill(o, 0, 0, 0, 0, trade_date,
                                 status='rejected(unresolved pending records)')
                            for o in sorted_orders]
        self.query_failed = False
        # 二轮复审#4：同批逐笔扣减可卖数量；已知持仓表缺符号按 0（不得
        # 用申报数顶替持仓）。
        sellable = dict(positions) if positions is not None else None
        if getattr(self, '_requires_ledger', False) and any(o.side == 'SELL' for o in sorted_orders):
            holdings = self._xt_trader.query_stock_positions(self._account)
            if holdings is None:
                raise RuntimeError('Broker sellable quantities unavailable')
            broker_sellable = {p.stock_code: max(0, int(p.can_use_volume)) for p in holdings}
            sellable = broker_sellable if sellable is None else {
                sym: min(qty, broker_sellable.get(sym, 0)) for sym, qty in sellable.items()}
        for index, order in enumerate(sorted_orders):
            if sellable is not None and order.side == 'SELL':
                holding = sellable.get(order.symbol, 0)
                if order.quantity > 0 and order.quantity > holding:
                    order = Order(order.symbol, order.side, order.target_weight,
                                  quantity=max(0, holding), note=order.note)
                    if order.quantity <= 0:
                        fills.append(Fill(order, 0, 0, 0, 0, trade_date,
                                          status='rejected(no holding)'))
                        continue
            fill = self._execute_single(order, prices, trade_date, running_cash,
                                        positions=sellable, ledger=ledger)
            fills.append(fill)
            if self._pending or fill.status.startswith('query_failed') or \
                    fill.status.startswith('open'):
                # 审计 2026-11-19 三-4：订单结果未确认（查询失败，或超时
                # 撤单后仍在途/非终态部分成交）→ 停止本批后续委托。
                self.query_failed = True
                fills.extend(Fill(o, 0, 0, 0, 0, trade_date,
                                  status='rejected(previous order unconfirmed)')
                             for o in sorted_orders[index + 1:])
                break

            # 更新运行时现金与可卖数量（含费用；部分成交同样占用/释放资金）
            if fill.quantity > 0 and sellable is not None and order.side == 'SELL':
                sellable[order.symbol] = max(0, sellable.get(order.symbol, 0) - fill.quantity)
            if fill.quantity > 0:
                if order.side == 'BUY':
                    running_cash -= fill.price * fill.quantity + fill.commission + fill.taxes
                else:
                    running_cash += fill.price * fill.quantity - fill.commission - fill.taxes

        return fills

    def _execute_single(self, order: Order, prices: Dict[str, float],
                        trade_date, running_cash: float,
                        positions: Dict[str, int] = None,
                        ledger=None) -> Fill:
        """
        单笔订单全生命周期：提交 → 等待（超时撤单）→ 增量入账。

        - 部分成交不丢失：超时前已成交量按 delta 入账，trade_id=oid#累计量（幂等）
        - 超时撤单：收盘执行策略不留隔夜在途单，撤后取最终成交量入账
        - 未终态残量由 sync_pending 在下次执行前继续补记
        """
        price = prices.get(order.symbol, 0)
        if not valid_order(order, price) or not math.isfinite(running_cash):
            return Fill(order, 0, 0, 0, 0, trade_date, status='rejected(无价格)')

        # 申报数量（主板/创业板100整手；科创板≥200且1股递增；北交所≥100且1股递增）
        from trading.rules import round_buy_qty, round_sell_qty, qty_step, min_order_qty
        qty = order.quantity
        if qty <= 0 and order.amount > 0:
            qty = int(order.amount / (price * 1.001)) if order.side == 'BUY' \
                else int(order.amount / price)
        if order.side == 'BUY':
            qty = round_buy_qty(order.symbol, qty)
        elif positions is not None:
            # 卖出合法化（与 PaperBroker 同规则：零股一次性=全仓申报恒合法）
            qty = round_sell_qty(order.symbol, qty,
                                 positions.get(order.symbol, 0))
        else:
            step = qty_step(order.symbol)
            fixed = (qty // step) * step
            if fixed >= min_order_qty(order.symbol):
                qty = fixed   # 无持仓上下文时按步长/最低申报对齐，券商端最终校验
        if qty <= 0:
            return Fill(order, price, 0, 0, 0, trade_date, status='rejected(数量为零)')

        # 买入资金检查（缩减到可承受申报数量）
        if order.side == 'BUY':
            if running_cash < price * qty * 1.002:
                affordable = round_buy_qty(order.symbol, int(running_cash / (price * 1.002)))
                if affordable <= 0:
                    return Fill(order, price, 0, 0, 0, trade_date, status='rejected(资金不足)')
                qty = affordable

        oid = self._submit_order(order, qty, price)
        if not oid:
            return Fill(order, price, 0, 0, 0, trade_date, status='rejected(下单失败)')

        if self._pending.get(oid, {}).get('status') == 'unknown':
            return Fill(order, price, 0, 0, 0, trade_date, 'query_failed(unknown acknowledgement)')
        status, traded, avg_px = self._await_order(oid, timeout=self.order_timeout,
                                                   cancel_on_timeout=True)
        return self._book_order_fill(order, oid, status, traded, avg_px,
                                     price, qty, trade_date, ledger=ledger)

    # ---- 订单生命周期原语（真实实现走 xtquant；测试注入假实现） ----

    @_serialized
    def _submit_order(self, order: Order, qty: int, price: float) -> Optional[str]:
        """提交委托，返回跟踪键（客户端引用）并登记在途台账。

        XtQuant 正确参数（官方文档，8 参数）：
        - direction: STOCK_BUY=23 / STOCK_SELL=24
        - price_type: LATEST_PRICE=5（最新价，接近市价；收盘竞价适用）
        - order_remark: 预生成的客户端引用（cr…）——order_stock_async 返回的
          是请求序号 seq，非真实订单号；真实订单号由 on_stock_order 回调
          提供，回调对象的 order_remark 携带我们的引用完成关联
          （2026-10-15 重写：旧版把 seq 当订单号去查询，永远匹配不上
          真实订单 → 永远"已报未成交"→ 超时撤错单 → 成交丢失）。
        """
        if self._pending:
            raise RuntimeError('Resolve prior orders before submitting a new intent')
        direction = XT['STOCK_BUY'] if order.side == 'BUY' else XT['STOCK_SELL']
        price_type = XT['LATEST_PRICE']   # 最新价（接近市价；收盘竞价适用）
        price_val = 0                     # LATEST_PRICE 时 price=0
        ref = 'cr' + uuid.uuid4().hex
        # 审计 2026-11-19 三-4：先持久化下单意图，再向券商提交——此前先
        # 提交后落盘，ack 与 _save_state 之间退出 = 券商在途单在本地无任何
        # 记录，重启后 sync_pending 无从核对（丢单窗口）。意图先落盘后，
        # 中途退出也能按 order_remark 引用与券商核对；若提交从未到达券商，
        # sync_pending 查无此单 → 持续阻断，等待人工处理（fail-safe）。
        self._pending[ref] = {'symbol': order.symbol, 'side': order.side,
                              'qty': int(qty), 'booked': 0,
                              'status': 'submitting',
                              'submitted_at': str(datetime.now()),
                              'ref': ref, 'oid': None, 'seq': None}
        self._save_state()
        try:
            ack = self._xt_trader.order_stock_async(
                self._account, order.symbol, direction, qty, price_type,
                price_val, 'quant', ref)   # strategy_name, order_remark=ref
        except Exception as e:
            # 二轮复审#5：回执丢失（超时/连接断）≠ 被拒——柜台可能已收单。
            # 保留意图为 unknown（sync_pending 按引用核对、execute 阻断新批），
            # 只在明确拒绝（ack None/-1）时清退。
            self._pending[ref]['status'] = 'unknown'
            self._save_state()
            logger.error(f"[QMT] 下单回执异常 {order.symbol}（意图保留为 unknown）: {e}")
            return ref
        if type(ack) is not int or ack < -1:
            self._pending[ref]['status'] = 'unknown'
            self._save_state()
            return ref
        if type(ack) is int and ack == -1:
            self._pending.pop(ref, None)
            self._save_state()
            return None              # -1 = 下单请求被拒
        self._pending[ref]['seq'] = ack
        self._pending[ref]['status'] = XT['ORDER_PENDING']
        self._save_state()
        return ref

    # ---- 回调实现（xtquant 单参签名；见文件末"xtquant 回调"区，勿重复定义） ----

    @_serialized
    def _resolve_oid(self, tracking_key: str) -> Optional[str]:
        """跟踪键（客户端引用）→ 真实订单号。

        路径1：回调已落账（rec['oid']）；路径2：回调未达（断线/竞争）时
        按 order_remark 扫描券商订单表兜底关联。解析失败返回 None
        （调用方视为不可确认——绝不能拿引用号当订单号查询）。"""
        rec = self._pending.get(tracking_key)
        if rec is None:
            return str(tracking_key)          # 本就是真实订单号（外部调用）
        if rec.get('oid'):
            return rec['oid']
        try:
            for o in (self._xt_trader.query_stock_orders(self._account) or []):
                if str(getattr(o, 'order_remark', '') or '') == str(tracking_key):
                    rec['oid'] = str(o.order_id)
                    self._save_state()
                    return rec['oid']
        except Exception as e:
            logger.warning(f"[QMT] 引用号兜底关联查询失败 {tracking_key}: {e}")
        return None

    def _query_order(self, order_id) -> Tuple[int, int, float]:
        """查询订单：(状态码, 已成交量, 成交均价)；不可确认返回 (-1, -1, 0)。

        -1 状态=不可确认（引用未映射到真实订单号/接口异常/券商订单表查无）——
        调用方应视为 unknown 并重试或阻断，绝不能当作"未成交"继续。"""
        oid = self._resolve_oid(order_id)
        if oid is None:
            return -1, -1, 0.0
        try:
            for o in (self._xt_trader.query_stock_orders(self._account) or []):
                if (str(o.order_id) == str(oid) and
                        str(getattr(o, 'order_remark', '') or '') in ('', str(order_id))):
                    status = QMTOrderState.CANCELLED if o.order_status == 53 else int(o.order_status)
                    return status, o.traded_volume or 0, \
                        float(o.traded_price or 0)
        except Exception as e:
            logger.warning(f"[QMT] 查询订单异常（不可确认状态）: {e}")
            return -1, -1, 0.0
        return -1, -1, 0.0     # 查无此单：可能尚未报到（回调竞争）→ 不可确认

    def _cancel_order(self, order_id) -> bool:
        oid = self._resolve_oid(order_id)
        if oid is None:
            logger.error(f"[QMT] 撤单失败：引用 {order_id} 未映射到真实订单号")
            return False
        try:
            ack = self._xt_trader.cancel_order_stock_async(self._account, oid)
            return type(ack) is int and ack >= 0
        except Exception as e:
            logger.warning(f"[QMT] 撤单失败 {oid}: {e}")
            return False

    def _await_order(self, order_id, timeout: float = 30, cancel_on_timeout=True):
        """等待终态；超时撤单并返回最终真实成交（不丢弃部分成交量）。
        查询接口持续异常 → 返回(-1,-1,0)，调用方应阻断新委托。"""
        start = time.time()
        status, traded, px = 49, 0, 0.0
        while time.time() - start < timeout:
            status, traded, px = self._query_order(order_id)
            if status == -1:
                time.sleep(1)          # 查询异常：重试而非当作未成交
                continue
            if status in (QMTOrderState.FILLED, QMTOrderState.CANCELLED,
                          QMTOrderState.REJECTED):
                return status, traded, px
            time.sleep(1)
        if cancel_on_timeout:
            self._cancel_order(order_id)
            time.sleep(1)
            status, traded, px = self._query_order(order_id)
        return status, traded, px

    @_serialized
    def _book_order_fill(self, order: Order, oid, status, traded, avg_px,
                         ref_price, qty, trade_date, ledger=None) -> Fill:
        """增量入账：本次已成交 − 台账已入账；trade_id=oid#累计量（重放幂等）。

        原子性（R4，2026-10-15 重写）：传入 ledger 时**先入账、成功后**才推进
        台账 booked/booked_cost 并清退终态记录——入账前崩溃则台账仍是旧值，
        sync_pending 重启后按未入账差额补记（旧版先清退后入账，中间崩溃
        =成交永久丢失）。入账被阻（超额卖出防线等）→ 台账不推进、记录保留，
        由账本异常队列 + L2.6 闸门接管。"""
        rec = self._pending.get(oid) or {'symbol': order.symbol,
                                         'side': order.side, 'qty': int(qty),
                                         'booked': 0}
        # 查询失败哨兵（traded=-1）不得污染台账（2026-10-30，P0#8）：
        # 旧逻辑 booked=-1 → 恢复后 delta=traded-(-1)=traded+1 多入账。
        # 查询失败=成交不可确认 → 台账原地保留，交 sync_pending 复核。
        if (not isinstance(traded, (int, float)) or not math.isfinite(traded)
                or not float(traded).is_integer() or traded < int(rec.get('booked', 0))
                or traded > qty or (status == QMTOrderState.FILLED and traded != qty)
                or (traded > 0 and (not isinstance(avg_px, (float, int))
                                    or not math.isfinite(avg_px) or avg_px <= 0))):
            self.query_failed = True
            return Fill(order, 0, 0, 0, 0, trade_date,
                        f'query_failed(contradictory or unavailable execution: status={status})')
        # 二轮复审#7：无账本且有未入账增量 → 返回成交供调用方入账，但
        # **不推进/不清退台账**（此前允许推进=成交可能永久丢失）
        no_ledger_delta = int(traded) > int(rec.get('booked', 0)) and ledger is None
        prev = int(rec.get('booked', 0))
        delta = int(traded) - prev

        if int(traded) >= int(qty) > 0 or status == QMTOrderState.FILLED:
            status_str = 'filled'
        elif int(traded) > 0:
            if status in (QMTOrderState.CANCELLED, QMTOrderState.REJECTED):
                status_str = f'partial({traded}/{qty})'   # 终态部分成交（撤单/废单）
            else:
                # 非终态部分成交：明确标 open（审计 2026-11-19 三-4——此前
                # 与终态部分成交同写 partial(n/m)，批处理无从区分，在途残量
                # 之后仍可能成交，后续买单复用已被占用的预算）
                status_str = f'open({status}, partial {traded}/{qty})'
        elif status == QMTOrderState.CANCELLED:
            status_str = 'cancelled'
        elif status == QMTOrderState.REJECTED:
            status_str = 'rejected(废单)'
        else:
            status_str = f'open({status})'    # 撤单中/在途，残量由 sync_pending 补记

        px = float(avg_px) if avg_px and avg_px > 0 else float(ref_price)
        if delta < 0 or (delta > 0 and px <= 0):
            self.query_failed = True
            return Fill(order, 0, 0, 0, 0, trade_date,
                        'query_failed(inconsistent execution quantity or price)')
        booking_ok = True
        if delta > 0 and hasattr(self._xt_trader, 'query_stock_trades'):
            try:
                fill = self._book_executions(order, oid, traded, ledger, status_str)
            except (AttributeError, RuntimeError, TypeError, OSError, ValueError) as exc:
                self.query_failed = True
                return Fill(order, 0, 0, 0, 0, trade_date, f'query_failed({exc})')
            no_ledger_delta = False
        elif delta > 0 and px > 0:
            # 增量价格：用累计成本差除以增量数量（券商返回的是累计均价，
            # 直接用于增量段会高估/低估部分成交的实际成交价）
            total_cost = float(traded) * px
            if prev > 0 and 'booked_cost' not in rec:
                # Older recovery records did not persist cumulative cost.
                # Reconstruct it from confirmed ledger fills or stop safely.
                prior_fills = [] if ledger is None else [
                    t for t in ledger._d['trades']
                    if str(t.get('trade_id', '')).startswith(f'{oid}#')
                    and str(t['trade_id']).rsplit('#', 1)[-1].isdigit()
                    and int(str(t['trade_id']).rsplit('#', 1)[-1]) <= prev]
                if sum(t['quantity'] for t in prior_fills) != prev:
                    self.query_failed = True
                    return Fill(order, px, 0, 0, 0, trade_date,
                                'query_failed(previous execution cost unavailable)')
                rec['booked_cost'] = sum(t['quantity'] * t['price'] for t in prior_fills)
            prev_cost = float(rec.get('booked_cost', 0.0))
            delta_cost = total_cost - prev_cost
            if delta > 0 and delta_cost > 0:
                incr_px = delta_cost / delta     # 增量段的实际均价
            else:
                self.query_failed = True
                return Fill(order, px, 0, 0, 0, trade_date,
                            'query_failed(inconsistent cumulative execution cost)')
            value = incr_px * delta
            commission, taxes = _est_fees(order.side, value)
            fill = Fill(order, incr_px, delta, commission, taxes, trade_date,
                        status_str, trade_id=f'{oid}#{int(traded)}')
            if ledger is not None:
                booked, _dupes, _blocked = ledger.apply_fills([fill])
                if _dupes:
                    fill = Fill(order, incr_px, 0, 0, 0, trade_date,
                                status_str + '(already booked)')
                if not booked and not _dupes:
                    booking_ok = False
                    fill = Fill(order, incr_px, 0, 0, 0, trade_date,
                                status_str + '(入账被阻，台账保留待复核)')
                    logger.error(f"[QMT] 成交入账被阻 {oid} delta={delta}——"
                                 f"台账不推进，账本异常队列接管")
        else:
            fill = Fill(order, px, 0, 0, 0, trade_date, status_str)

        if booking_ok and not no_ledger_delta:
            # 入账成功（或无新增量）才推进台账与清退终态
            rec.update({'booked': int(traded), 'status': int(status),
                        'qty': int(qty), 'symbol': order.symbol,
                        'side': order.side})
            if delta > 0 and px > 0:
                rec['booked_cost'] = float(traded) * px
            # 台账清退：FILLED/REJECTED 即清；CANCELLED 保留至 sync_pending
            # 复核一次（撤单瞬间可能仍有成交残量；TTL 3天兜底）
            if status in (QMTOrderState.FILLED, QMTOrderState.REJECTED):
                self._pending.pop(oid, None)
        else:
            rec['status'] = int(status)   # booked/booked_cost 不推进
        self._save_state()
        return fill

    # ---- 在途订单补记（崩溃/次日恢复；每次执行前调用） ----

    @_serialized
    def sync_pending(self, ledger) -> List[Fill]:
        """同步在途订单成交到账本：查询台账所有未清退订单，补记新增成交量。
        trade_id=oid#累计量 → 幂等（回报重放/重复同步不重复入账）。
        撤单后的残量在此复核补记；超3天的僵尸记录直接清退（对账兜底）。
        查询接口异常 → 返回空且标记 query_failed=True（调用方应阻断新委托）。"""
        booked_fills = []
        today = date.today()
        self.query_failed = False     # 查询接口健康标记
        for oid in list(self._pending.keys()):
            rec = self._pending[oid]
            # TTL 兜底：老记录券商已查不到（如跨日），交给账实核对流程
            try:
                submitted = datetime.fromisoformat(rec.get('submitted_at', '')
                                                   or str(today)).date()
            except ValueError:
                submitted = today
            if (today - submitted).days > 3:
                # 二轮复审#6：超期记录必须先查询再清退——此前按自然日直接
                # 删除，四天前已成交的委托未经一次查询就消失（账本漏记）。
                status, traded, px = self._query_order(oid)
                if status == -1:
                    rec['ttl_expired_unresolved'] = True
                    self.query_failed = True
                    logger.error(f"[QMT] sync_pending 超期且查询失败 {oid}："
                                 "转入待对账（不清退，人工/账实核对处理）")
                    continue
            else:
                status, traded, px = self._query_order(oid)
            if status == -1:
                # 查询接口异常：无法确认旧订单状态——阻断新委托直到恢复
                self.query_failed = True
                logger.error(f"[QMT] sync_pending 查询失败 {oid}：无法确认在途订单状态，"
                             "阻断新委托（防止在未知旧订单上叠加重复委托）")
                continue
            # 二轮复审#6：跨日补记的成交日期用**提交日**（此前用恢复当天，
            # 成交日期被系统性后移）
            trade_day = today
            if (submitted != today and traded > rec.get('booked', 0)
                    and not hasattr(self._xt_trader, 'query_stock_trades')):
                self.query_failed = True
                continue
            fill = self._book_order_fill(
                Order(rec['symbol'], rec['side'], 0.0), oid, status,
                traded, px, px, rec['qty'], trade_day, ledger=ledger)
            if fill.status.startswith('query_failed'):
                self.query_failed = True
            if fill.quantity > 0:
                booked_fills.append(fill)
            if not fill.status.startswith('query_failed') and status == QMTOrderState.CANCELLED and \
                    int(rec.get('booked', 0)) >= int(traded):
                self._pending.pop(oid, None)
        self._save_state()
        if booked_fills:
            logger.info(f"[QMT] 在途订单补记 {len(booked_fills)} 笔")
        self.query_failed = bool(self._pending)
        return booked_fills

    # ---- 台账持久化 ----

    def _load_state(self) -> dict:
        if os.path.isfile(self.state_path):
            try:
                with open(self.state_path, encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                # 在途台账损坏=不知道券商侧有什么活动订单——禁止交易而非
                # 静默重建空状态（空状态会导致重复下单/漏记成交，评审P1-4）
                raise RuntimeError(
                    f"QMT在途台账损坏，禁止交易（人工核对券商活动订单后"
                    f"重建 {self.state_path} 再恢复）: {e}")
        return {}

    def _save_state(self):
        """原子持久化：先写临时文件再 os.replace（同卷原子）。

        直接 open('w') 写一半崩溃会截断 JSON → _load_state 静默重建空台账
        = 在途订单全部遗忘 = 重复下单/成交丢失（评审阻断项#5，R4）。
        """
        try:
            with self._mutex, exclusive_file(self.state_path):
                if self._load_state() != self._persisted:
                    raise RuntimeError('Order state changed in another instance; reload first')
                replace_json(self.state_path, self._pending)
                self._persisted = copy.deepcopy(self._pending)
        except Exception as e:
            self._pending = copy.deepcopy(self._persisted)
            # 实盘台账写入失败=丢失在途订单跟踪 → 重复下单风险，必须阻断
            logger.error(f"[QMT] 在途台账保存失败（实盘安全风险）: {e}")
            raise RuntimeError(f"QMT在途台账持久化失败，阻断执行以防止重复下单: {e}")

    def cancel_order(self, order_id) -> bool:
        """撤销未成交订单"""
        self._ensure_connected()
        try:
            result = self._xt_trader.cancel_order_stock_async(self._account, order_id)
            return type(result) is int and result >= 0
        except Exception as e:
            logger.error(f"[QMT] 撤单失败: {e}")
            return False

    def cancel_all_pending(self) -> int:
        """撤销所有未成交订单。查询失败→RuntimeError（非返回0=假无单）"""
        self._ensure_connected()
        count = 0
        try:
            orders = self._xt_trader.query_stock_orders(self._account)
        except Exception as e:
            raise RuntimeError(
                f"QMT订单查询失败——无法确认是否有在途订单，"
                f"必须人工核查券商App: {e}")
        if orders is None:
            raise RuntimeError("QMT订单查询返回None——数据不可用，人工核查")
        terminal = {QMTOrderState.CANCELLED, QMTOrderState.FILLED, QMTOrderState.REJECTED, 53}
        active = {o.order_id for o in orders if o.order_status not in terminal}
        for oid in active:
            self.cancel_order(oid)
        if not active:
            return 0
        observed = self._xt_trader.query_stock_orders(self._account)
        if observed is None:
            raise RuntimeError('Unable to verify cancellation')
        confirmed = {o.order_id for o in observed if o.order_status in terminal}
        return len(active - confirmed) + sum(
            o.order_id not in active and o.order_status not in terminal for o in observed)

    # ==================== xtquant 回调（单参签名，register_callback(self) 注入） ====================

    def on_disconnected(self):
        """连接断开回调"""
        self._connected = False
        logger.warning("[QMT] 连接断开")

    @_serialized
    def on_stock_order(self, order):
        """委托回报：order_remark(客户端引用) → 真实订单号 映射落账。

        2026-10-15 重写：旧版按 symbol 猜测关联 seq_ 记录（同标的并发单
        会串单）。现以 order_remark=客户端引用精确关联；remark 缺失时
        退化按 symbol+无oid 记录匹配（兼容旧台账）。"""
        try:
            logger.info(f"[QMT] 订单推送: {getattr(order, 'stock_code', '')} "
                        f"状态={getattr(order, 'order_status', '?')} "
                        f"成交量={getattr(order, 'traded_volume', '?')} "
                        f"订单号={getattr(order, 'order_id', '?')}")
            oid = str(getattr(order, 'order_id', '') or '')
            if not oid:
                return
            remark = str(getattr(order, 'order_remark', '') or '')
            rec = self._pending.get(remark)
            if rec is not None:
                if rec.get('oid') and str(rec['oid']) != oid:
                    raise ValueError('Conflicting broker order identity')
                if rec.get('oid') != oid:
                    rec['oid'] = oid
                    self._save_state()
                    logger.info(f"[QMT] 订单映射 {remark} → {oid}")
                return
            if remark:
                return
            # Legacy references with no matching broker remark require manual reconciliation.
        except Exception as e:
            logger.error(f"[QMT] on_stock_order 回调处理失败: {e}")

    def on_stock_trade(self, trade):
        """成交回报：留痕（成交量真相以查询接口为准，避免双计数）。"""
        logger.info(f"[QMT] 成交推送: {getattr(trade, 'stock_code', '')} "
                    f"价格={getattr(trade, 'traded_price', '?')} "
                    f"数量={getattr(trade, 'traded_volume', '?')} "
                    f"订单={getattr(trade, 'order_id', '?')}")

    @_serialized
    def on_order_error(self, order_error):
        """订单错误回调：标记对应台账记录，交由 sync_pending/人工复核。

        order_error.order_id 可能是请求序号（下单失败）或真实订单号
        （报价被拒）——两者都尝试关联。"""
        try:
            eid = str(getattr(order_error, 'order_id', ''))
            msg = str(getattr(order_error, 'error_msg', getattr(order_error, 'error', '')))
            logger.error(f"[QMT] 订单错误: {eid} {msg}")
            for key, rec in self._pending.items():
                if str(rec.get('seq')) == eid or str(rec.get('oid')) == eid:
                    rec['status'] = QMTOrderState.REJECTED
                    rec['error'] = msg
                    self._save_state()
                    return
        except Exception as e:
            logger.error(f"[QMT] on_order_error 回调处理失败: {e}")

    def on_cancel_error(self, cancel_error):
        """撤单错误回调"""
        logger.error(f"[QMT] 撤单错误: {cancel_error.order_id} {cancel_error.error_msg}")

    def on_order_stock_ack(self, order_id, error_msg):
        """下单确认回调"""
        if error_msg:
            logger.error(f"[QMT] 下单被拒: {order_id} {error_msg}")
        else:
            logger.info(f"[QMT] 下单确认: {order_id}")

    # ==================== 工具 ====================

    def _ensure_connected(self):
        if not self._connected:
            self.connect()

    @staticmethod
    def check_env() -> dict:
        """检查 QMT 环境配置是否完整"""
        issues = []
        if not QMT_PATH:
            issues.append('QMT_PATH 未设置（.env 中添加 QMT_PATH=<QMT客户端数据目录>）')
        if not QMT_ACCOUNT_ID:
            issues.append('QMT_ACCOUNT_ID 未设置（.env 中添加 QMT_ACCOUNT_ID=<资金账号>）')
        try:
            import xtquant
        except ImportError:
            issues.append('xtquant 未安装（需 QMT 客户端运行中）')
        return {
            'ready': len(issues) == 0,
            'issues': issues,
            'qmt_path': QMT_PATH,
            'account_id': QMT_ACCOUNT_ID[:4] + '****' if QMT_ACCOUNT_ID else '未配置',
        }
