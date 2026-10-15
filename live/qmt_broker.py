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
import os
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from .broker import Order, Fill, Broker

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
                 switch_state: str = 'NORMAL'):
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
        self.switch_state = switch_state
        self._xt_trader = None
        self._account = None
        self._connected = False
        self._order_callbacks = {}  # order_id -> callback info
        # 在途订单台账（订单生命周期状态，崩溃/重启后可续）：{oid: {symbol, side, qty, booked, status}}
        self.state_path = state_path
        self._pending = self._load_state()

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

        需要 QMT 客户端运行中；未连接时抛 RuntimeError。
        """
        from live.reconcile import BrokerStatement
        self._ensure_connected()
        as_of = as_of or date.today()

        asset = self._xt_trader.query_stock_asset(self._account)
        cash = float(asset.cash) if asset else 0.0

        positions = {}
        for pos in (self._xt_trader.query_stock_positions(self._account) or []):
            if pos.volume > 0:
                positions[pos.stock_code] = int(pos.volume)

        trades = []
        for t in (self._xt_trader.query_stock_trades(self._account) or []):
            d = datetime.fromtimestamp(t.traded_time).date() if getattr(t, 'traded_time', 0) else as_of
            trades.append({
                'date': str(d)[:10], 'symbol': t.stock_code,
                'side': 'BUY' if t.offset_flag == 48 else 'SELL',  # xtquant: 48买/49卖
                'price': float(t.traded_price or 0),
                'quantity': int(t.traded_volume or 0),
                'commission': float(getattr(t, 'traded_fee', 0) or 0),
                'taxes': 0.0,  # QMT成交回报费用合计，税费拆分以交割单为准
                'trade_id': str(getattr(t, 'order_id', '') or ''),
            })

        return BrokerStatement(as_of=str(as_of)[:10], cash=cash, positions=positions,
                               trades=trades, source='qmt')

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
        _sw = self.switch_state
        if _sw == 'HALT_ALL':
            logger.error("[QMT] 交易控制=HALT_ALL，拒单")
            return [Fill(o, 0, 0, 0, 0, trade_date,
                         status='rejected(HALT_ALL)') for o in orders]
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

        # 排序：卖出在前（释放资金供买入）
        sorted_orders = sorted(orders, key=lambda o: 0 if o.side == 'SELL' else 1)

        running_cash = available_cash if available_cash is not None else self.get_cash()

        for order in sorted_orders:
            fill = self._execute_single(order, prices, trade_date, running_cash,
                                        positions=positions, ledger=ledger)
            fills.append(fill)

            # 更新运行时现金（含费用；部分成交同样占用/释放资金）
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
        if price <= 0:
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
                                 positions.get(order.symbol, qty))
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

        status, traded, avg_px = self._await_order(oid, timeout=self.order_timeout,
                                                   cancel_on_timeout=True)
        return self._book_order_fill(order, oid, status, traded, avg_px,
                                     price, qty, trade_date, ledger=ledger)

    # ---- 订单生命周期原语（真实实现走 xtquant；测试注入假实现） ----

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
        direction = XT['STOCK_BUY'] if order.side == 'BUY' else XT['STOCK_SELL']
        price_type = XT['LATEST_PRICE']   # 最新价（接近市价；收盘竞价适用）
        price_val = 0                     # LATEST_PRICE 时 price=0
        self._ref_counter = getattr(self, '_ref_counter', 0) + 1
        ref = f'cr{int(time.time() * 1000) % 10**8:08d}{self._ref_counter:04d}'
        try:
            ack = self._xt_trader.order_stock_async(
                self._account, order.symbol, direction, qty, price_type,
                price_val, 'quant', ref)   # strategy_name, order_remark=ref
        except Exception as e:
            logger.error(f"[QMT] 下单异常 {order.symbol}: {e}")
            return None
        if ack is None or ack == -1:
            return None              # -1 = 下单请求被拒
        self._pending[ref] = {'symbol': order.symbol, 'side': order.side,
                              'qty': int(qty), 'booked': 0,
                              'status': XT['ORDER_PENDING'],
                              'submitted_at': str(datetime.now()),
                              'ref': ref, 'oid': None, 'seq': ack}
        self._save_state()
        return ref

    # ---- 回调实现（xtquant 单参签名；见文件末"xtquant 回调"区，勿重复定义） ----

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
                if str(o.order_id) == str(oid):
                    return int(o.order_status), int(o.traded_volume or 0), \
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
            return self._xt_trader.cancel_order_stock_async(self._account, oid) == 0
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
        prev = int(rec.get('booked', 0))
        delta = int(traded) - prev

        if int(traded) >= int(qty) > 0 or status == QMTOrderState.FILLED:
            status_str = 'filled'
        elif int(traded) > 0:
            status_str = f'partial({traded}/{qty})'
        elif status == QMTOrderState.CANCELLED:
            status_str = 'cancelled'
        elif status == QMTOrderState.REJECTED:
            status_str = 'rejected(废单)'
        else:
            status_str = f'open({status})'    # 撤单中/在途，残量由 sync_pending 补记

        px = float(avg_px) if avg_px and avg_px > 0 else float(ref_price)
        booking_ok = True
        if delta > 0 and px > 0:
            # 增量价格：用累计成本差除以增量数量（券商返回的是累计均价，
            # 直接用于增量段会高估/低估部分成交的实际成交价）
            prev_cost = float(rec.get('booked_cost', 0.0))
            total_cost = float(traded) * px
            delta_cost = total_cost - prev_cost
            if delta > 0 and delta_cost > 0:
                incr_px = delta_cost / delta     # 增量段的实际均价
            else:
                incr_px = px                      # 退化：首笔或成本差异常
            value = incr_px * delta
            commission, taxes = _est_fees(order.side, value)
            fill = Fill(order, incr_px, delta, commission, taxes, trade_date,
                        status_str, trade_id=f'{oid}#{int(traded)}')
            if ledger is not None:
                booked, _dupes, _blocked = ledger.apply_fills([fill])
                if not booked:
                    booking_ok = False
                    fill = Fill(order, incr_px, 0, 0, 0, trade_date,
                                status_str + '(入账被阻，台账保留待复核)')
                    logger.error(f"[QMT] 成交入账被阻 {oid} delta={delta}——"
                                 f"台账不推进，账本异常队列接管")
        else:
            fill = Fill(order, px, 0, 0, 0, trade_date, status_str)

        if booking_ok:
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
                self._pending.pop(oid, None)
                continue

            status, traded, px = self._query_order(oid)
            if status == -1:
                # 查询接口异常：无法确认旧订单状态——阻断新委托直到恢复
                self.query_failed = True
                logger.error(f"[QMT] sync_pending 查询失败 {oid}：无法确认在途订单状态，"
                             "阻断新委托（防止在未知旧订单上叠加重复委托）")
                continue
            delta = int(traded) - int(rec.get('booked', 0))
            if delta > 0 and px > 0:
                value = px * delta
                commission, taxes = _est_fees(rec['side'], value)
                fill = Fill(Order(rec['symbol'], rec['side'], 0.0), px, delta,
                            commission, taxes, today, 'filled(在途补记)',
                            trade_id=f'{oid}#{int(traded)}')
                booked, _dupes, _blocked = ledger.apply_fills([fill])
                if booked:
                    booked_fills.extend(booked)
                    rec['booked'] = int(traded)   # 入账成功才推进（R4 原子性）
                # 入账被阻：booked 不推进，记录保留待下次同步/人工复核
            rec['status'] = int(status)
            if status in (QMTOrderState.FILLED, QMTOrderState.CANCELLED,
                          QMTOrderState.REJECTED) and \
                    int(rec.get('booked', 0)) >= int(traded):
                self._pending.pop(oid, None)
        self._save_state()
        if booked_fills:
            logger.info(f"[QMT] 在途订单补记 {len(booked_fills)} 笔")
        return booked_fills

    # ---- 台账持久化 ----

    def _load_state(self) -> dict:
        if os.path.isfile(self.state_path):
            try:
                with open(self.state_path, encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                logger.warning("[QMT] qmt_orders.json 损坏，重建")
        return {}

    def _save_state(self):
        """原子持久化：先写临时文件再 os.replace（同卷原子）。

        直接 open('w') 写一半崩溃会截断 JSON → _load_state 静默重建空台账
        = 在途订单全部遗忘 = 重复下单/成交丢失（评审阻断项#5，R4）。
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
            tmp = self.state_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self._pending, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.state_path)
        except Exception as e:
            # 实盘台账写入失败=丢失在途订单跟踪 → 重复下单风险，必须阻断
            logger.error(f"[QMT] 在途台账保存失败（实盘安全风险）: {e}")
            raise RuntimeError(f"QMT在途台账持久化失败，阻断执行以防止重复下单: {e}")

    def cancel_order(self, order_id) -> bool:
        """撤销未成交订单"""
        self._ensure_connected()
        try:
            result = self._xt_trader.cancel_order_stock_async(self._account, order_id)
            return result == 0
        except Exception as e:
            logger.error(f"[QMT] 撤单失败: {e}")
            return False

    def cancel_all_pending(self) -> int:
        """撤销所有未成交订单"""
        self._ensure_connected()
        count = 0
        try:
            orders = self._xt_trader.query_stock_orders(self._account)
            if orders:
                for o in orders:
                    if o.order_status in (QMTOrderState.PENDING, QMTOrderState.SUBMITTED,
                                          QMTOrderState.PARTIAL):
                        if self.cancel_order(o.order_id):
                            count += 1
        except Exception as e:
            logger.error(f"[QMT] 批量撤单失败: {e}")
        return count

    # ==================== xtquant 回调（单参签名，register_callback(self) 注入） ====================

    def on_disconnected(self):
        """连接断开回调"""
        self._connected = False
        logger.warning("[QMT] 连接断开")

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
                if rec.get('oid') != oid:
                    rec['oid'] = oid
                    self._save_state()
                    logger.info(f"[QMT] 订单映射 {remark} → {oid}")
                return
            # 兼容路径：无 remark（旧台账/外部单）——仅对尚无 oid 的同标的记录
            for key, r in self._pending.items():
                if r.get('oid') is None and r.get('symbol') == getattr(order, 'stock_code', ''):
                    r['oid'] = oid
                    self._save_state()
                    logger.warning(f"[QMT] 无remark退化匹配 {key} → {oid}（按symbol）")
                    return
        except Exception as e:
            logger.error(f"[QMT] on_stock_order 回调处理失败: {e}")

    def on_stock_trade(self, trade):
        """成交回报：留痕（成交量真相以查询接口为准，避免双计数）。"""
        logger.info(f"[QMT] 成交推送: {getattr(trade, 'stock_code', '')} "
                    f"价格={getattr(trade, 'traded_price', '?')} "
                    f"数量={getattr(trade, 'traded_volume', '?')} "
                    f"订单={getattr(trade, 'order_id', '?')}")

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
