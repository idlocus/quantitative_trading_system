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
"""

import os
import time
from datetime import date, datetime
from typing import Dict, List, Optional

from .broker import Order, Fill, Broker

from utils.logger import get_logger

logger = get_logger(__name__)

# QMT 配置从环境变量读取（用户填充）
QMT_PATH = os.getenv('QMT_PATH', '')           # QMT 客户端数据目录
QMT_ACCOUNT_ID = os.getenv('QMT_ACCOUNT_ID', '')  # 资金账号
QMT_SESSION_ID = int(os.getenv('QMT_SESSION_ID', '1'))


class QMTOrderState:
    """QMT 订单状态跟踪"""
    PENDING = 48    # 待报
    SUBMITTED = 49  # 已报
    PARTIAL = 50    # 部分成交
    FILLED = 51     # 全部成交
    CANCELLED = 52  # 已撤
    REJECTED = 53   # 废单


class QMTBroker(Broker):
    """
    miniQMT 实盘交易通道

    接口与 PaperBroker 一致（execute），可直接替换。
    前置条件：QMT 客户端运行中，.env 已配置。
    """

    def __init__(self, account_id: str = None, qmt_path: str = None):
        self.account_id = account_id or QMT_ACCOUNT_ID
        self.qmt_path = qmt_path or QMT_PATH
        self._xt_trader = None
        self._account = None
        self._connected = False
        self._order_callbacks = {}  # order_id -> callback info

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

    # ==================== 下单接口 ====================

    def execute(self, orders: List[Order], prices: Dict[str, float],
                trade_date, prev_prices=None, available_cash=None) -> List[Fill]:
        """
        执行订单列表（与 PaperBroker 接口一致）

        QMT 实盘流程：
        1. 按顺序提交订单（先卖后买，释放资金）
        2. 等待订单回报（最多30秒）
        3. 查询最终成交状态
        4. 返回 Fill 列表
        """
        self._ensure_connected()
        fills = []

        # 排序：卖出在前（释放资金供买入）
        sorted_orders = sorted(orders, key=lambda o: 0 if o.side == 'SELL' else 1)

        running_cash = available_cash if available_cash is not None else self.get_cash()

        for order in sorted_orders:
            fill = self._execute_single(order, prices, trade_date, running_cash)
            fills.append(fill)

            # 更新运行时现金
            if fill.status == 'filled':
                if order.side == 'BUY':
                    running_cash -= fill.price * fill.quantity
                else:
                    running_cash += fill.price * fill.quantity

        return fills

    def _execute_single(self, order: Order, prices: Dict[str, float],
                        trade_date, running_cash: float) -> Fill:
        """执行单笔订单"""
        from xtquant.xttrader import XtQuantTrader

        price = prices.get(order.symbol, 0)
        if price <= 0:
            return Fill(order, 0, 0, 0, 0, trade_date, status='rejected(无价格)')

        # 计算数量
        qty = order.quantity
        if qty <= 0 and order.amount > 0:
            if order.side == 'BUY':
                fee_est = 0.001  # 预估费用
                qty = int(order.amount / (price * (1 + fee_est)) // 100) * 100
            else:
                qty = int(order.amount / price)

        if qty <= 0:
            return Fill(order, price, 0, 0, 0, trade_date, status='rejected(数量为零)')

        # 买入时检查资金
        if order.side == 'BUY':
            est_cost = price * qty * 1.002
            if running_cash < est_cost:
                # 缩减数量
                affordable = int(running_cash / (price * 1.002) // 100) * 100
                if affordable <= 0:
                    return Fill(order, price, 0, 0, 0, trade_date, status='rejected(资金不足)')
                qty = affordable

        # 提交 QMT 订单
        try:
            # xtquant 用 0=买入, 1=卖出
            direction = 0 if order.side == 'BUY' else 1
            # 11=市价单（最优五档）, 42=限价
            price_type = 11  # 市价单（收盘竞价时用）
            price_val = 0 if price_type == 11 else price  # 市价单价格为0

            ack = self._xt_trader.order_stock_async(
                self._account,
                order.symbol,
                direction,
                qty,
                price_type,
                price_val,
                '', '', '', '', ''
            )

            if ack is None:
                return Fill(order, price, qty, 0, 0, trade_date, status='rejected(下单失败)')

            # 等待成交回报（最多30秒）
            order_id = ack
            final_status, filled_qty, avg_price = self._wait_for_fill(order_id, timeout=30)

            # 计算费用
            if filled_qty > 0:
                value = avg_price * filled_qty
                commission = value * 0.0003  # 万3
                taxes = value * 0.00001 + (value * 0.0005 if order.side == 'SELL' else 0)
                status = 'filled' if final_status == QMTOrderState.FILLED else f'partial({filled_qty}/{qty})'
                return Fill(order, avg_price, filled_qty, commission, taxes, trade_date, status)
            else:
                status_map = {
                    QMTOrderState.CANCELLED: 'cancelled',
                    QMTOrderState.REJECTED: 'rejected(废单)',
                    QMTOrderState.PENDING: 'timeout(待报)',
                    QMTOrderState.SUBMITTED: 'timeout(已报未成交)',
                }
                return Fill(order, price, 0, 0, 0, trade_date,
                           status=status_map.get(final_status, f'unknown({final_status})'))

        except Exception as e:
            logger.error(f"[QMT] 下单异常 {order.symbol}: {e}")
            return Fill(order, price, qty, 0, 0, trade_date, status=f'error({str(e)[:50]})')

    def _wait_for_fill(self, order_id, timeout: int = 30) -> tuple:
        """等待订单成交（轮询查询，返回 (最终状态, 成交量, 均价)"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                orders = self._xt_trader.query_stock_orders(self._account)
                if orders:
                    for o in orders:
                        if o.order_id == order_id:
                            if o.order_status in (QMTOrderState.FILLED,
                                                  QMTOrderState.CANCELLED,
                                                  QMTOrderState.REJECTED):
                                return o.order_status, int(o.traded_volume), float(o.traded_price or 0)
                            elif o.order_status == QMTOrderState.PARTIAL:
                                # 部分成交，继续等待
                                pass
            except Exception as e:
                logger.warning(f"[QMT] 查询订单状态异常: {e}")

            time.sleep(1)

        # 超时，返回当前状态
        return QMTOrderState.SUBMITTED, 0, 0.0

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

    # ==================== xtquant 回调 ====================

    def on_disconnected(self):
        """连接断开回调"""
        self._connected = False
        logger.warning("[QMT] 连接断开")

    def on_stock_order(self, order):
        """订单状态推送回调"""
        logger.info(f"[QMT] 订单推送: {order.stock_code} 状态={order.order_status} "
                    f"成交量={order.traded_volume}")

    def on_stock_trade(self, trade):
        """成交推送回调"""
        logger.info(f"[QMT] 成交推送: {trade.stock_code} 价格={trade.traded_price} "
                    f"数量={trade.traded_volume}")

    def on_order_error(self, order_error):
        """订单错误回调"""
        logger.error(f"[QMT] 订单错误: {order_error.order_id} {order_error.error_msg}")

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
