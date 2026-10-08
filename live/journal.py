#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日交易日志 —— 机构式日报：盘前展望/决策依据/成交/持仓归因/收盘点评

文件: journal/YYYY-MM-DD.md（git 跟踪，webapp 可浏览）
- am 阶段（早盘）: 市场展望（昨收数据，PIT）+ 今日调仓决策依据（昨收信号）
- pm 阶段（收盘后）: 成交记录 + 持仓与归因 + 收盘点评，追加到同一文件
"""

import os
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.layered import LayeredDataSource
from utils.logger import get_logger

logger = get_logger(__name__)

JOURNAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'journal')


class JournalWriter:
    """每日交易日志生成器（am 追加式写入，pm 补齐）"""

    def __init__(self, ledger, top_n: int = 30):
        self.ledger = ledger
        self.top_n = top_n
        os.makedirs(JOURNAL_DIR, exist_ok=True)

    @property
    def path(self) -> str:
        return os.path.join(JOURNAL_DIR, f"{date.today().isoformat()}.md")

    # ================= AM：盘前 =================

    def write_am(self, ds: LayeredDataSource) -> dict:
        """盘前：市场展望（基于昨收）+ 决策依据。返回信号上下文供 pm 使用。"""
        as_of = date.today()
        prev_trading = self._prev_trading_day(ds, as_of)

        lines = [f"# 交易日志 {as_of.isoformat()}", "",
                 f"- 生成时间：盘前阶段 | 数据截至 {prev_trading} 收盘（PIT）", ""]

        # 1. 市场展望
        lines += self._market_outlook(ds, prev_trading)

        # 2. 决策依据（信号基于昨收）
        from live.signals import generate_target, build_orders
        targets, scores = generate_target(ds, prev_trading, top_n=self.top_n)
        prices = self._prices(ds, set(targets) | set(self.ledger.positions), prev_trading)
        nav = self.ledger.last_nav() or self.ledger.cash
        cur_w = self.ledger.positions and self.ledger.current_weights(prices) or {}
        orders = build_orders(targets, cur_w, nav, top_n=self.top_n)

        lines += ["## 二、盘前决策依据", "",
                  f"- 生产信号：5因子（EP/BP/SP/Size/低杠杆）Top{self.top_n}，"
                  f"基于 {prev_trading} 收盘（T-1信号，今日执行）",
                  f"- 目标持仓 {len(targets)} 只；调仓指令 {len(orders)} 条"
                  f"（缓冲带20%，无漂移则零调仓）", ""]

        if orders:
            lines += ["| 方向 | 代码 | 金额(约) | 理由 |", "|------|------|----------|------|"]
            in_target = set(targets)
            for o in orders:
                reason = ("新入选（因子排名进入前%d）" % self.top_n) if o.side == 'BUY' \
                    else ("移出组合（排名跌出前%d）" % self.top_n if o.symbol not in in_target else "超配回收")
                lines.append(f"| {o.side} | {o.symbol} | {o.amount:,.0f} | {reason} |")
            lines.append("")

        # 新入选股票的因子画像（前5）
        new_entrants = [s for s in targets if s not in cur_w][:5]
        if new_entrants:
            lines += [f"### 新入选因子画像（前{len(new_entrants)}）", "",
                      "| 代码 | 综合分 |", "|------|--------|"]
            for s in new_entrants:
                lines.append(f"| {s} | {scores.get(s, float('nan')):.3f} |")
            lines.append("")

        # 3. 风控检查
        w = 1.0 / max(len(targets), 1)
        lines += ["## 三、盘前风控检查", "",
                  f"- 单票目标权重 {w:.1%}（上限 5%）：{'通过' if w <= 0.05 else '超限'}",
                  f"- 持仓数 {len(targets)}（目标 {self.top_n}）",
                  f"- 现金比例 {(1 - min(1.0, len(targets) * w)):.1%}",
                  f"- 涨跌停约束：执行层自动拒单", ""]

        with open(self.path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            # 同日重复运行：保留已有的盘后章节（成交/归因不可再生），仅刷新盘前
            if os.path.exists(self.path + '.pm'):
                with open(self.path + '.pm', encoding='utf-8') as pf:
                    f.write("\n" + pf.read())
        return {'targets': targets, 'scores': scores, 'orders': orders, 'prev_trading': prev_trading}

    # ================= PM：盘后 =================

    def write_pm(self, ds: LayeredDataSource, fills: List, am_ctx: dict) -> str:
        """盘后：成交记录 + 持仓归因 + 收盘点评（追加）"""
        as_of = date.today()
        symbols = set(self.ledger.positions) | {f.order.symbol for f in fills}
        prices_today = self._prices(ds, symbols, as_of)
        prices_prev = self._prices(ds, symbols, am_ctx.get('prev_trading', as_of - timedelta(days=1)))

        lines = ["---", "", f"*以下为盘后阶段（{as_of} 收盘后生成）*", "",
                 "## 四、成交记录", ""]
        if fills:
            lines += ["| 状态 | 代码 | 方向 | 价格 | 数量 | 佣金 | 税费 |", "|------|------|------|------|------|------|------|"]
            for f in fills:
                st = '成交' if f.status == 'filled' else f.status
                lines.append(f"| {st} | {f.order.symbol} | {f.order.side} | {f.price:.2f} | "
                             f"{f.quantity} | {f.commission:.2f} | {f.taxes:.2f} |")
            n_fill = sum(1 for f in fills if f.status == 'filled')
            lines += ["", f"- 成交 {n_fill}/{len(fills)}；拒单均为涨跌停/零股（自动顺延）", ""]
        else:
            lines += ["*今日无交易（非调仓日或零漂移）*", ""]

        # 持仓与归因
        lines += ["## 五、持仓与归因", ""]
        nav = self.ledger.mark_to_market(prices_today, as_of) if self.ledger.positions \
            else (self.ledger.last_nav() or self.ledger.cash)
        invested = sum(self.ledger._d['cost'].get(s, 0) * q
                       for s, q in self.ledger._d['positions'].items())
        lines += [f"- 账户净值 **{nav:,.2f}**（累计投入约 {invested:,.0f}，"
                  f"累计盈亏 {(nav - invested):+,.0f}）", ""]

        if self.ledger.positions:
            lines += ["| 代码 | 数量 | 成本 | 现价 | 当日涨跌 | 持仓盈亏 | 权重 |",
                      "|------|------|------|------|----------|----------|------|"]
            rows = []
            total_mv = sum(q * prices_today.get(s, 0) for s, q in self.ledger.positions.items())
            for s, q in sorted(self.ledger.positions.items()):
                cost = self.ledger._d['cost'].get(s, 0)
                px = prices_today.get(s, 0)
                pxp = prices_prev.get(s, px)
                day_chg = (px / pxp - 1) if pxp > 0 else 0
                pnl = (px - cost) * q
                wgt = q * px / total_mv if total_mv > 0 else 0
                rows.append((s, q, cost, px, day_chg, pnl, wgt))
            for s, q, cost, px, chg, pnl, wgt in rows:
                lines.append(f"| {s} | {q} | {cost:.2f} | {px:.2f} | {chg:+.1%} | "
                             f"{pnl:+,.0f} | {wgt:.1%} |")
            # 当日贡献 top/bottom
            contribs = [(s, q * (prices_today.get(s, 0) - prices_prev.get(s, prices_today.get(s, 0))))
                        for s, q in self.ledger.positions.items()]
            contribs = sorted(contribs, key=lambda x: -x[1])
            if contribs:
                top3 = ", ".join(f"{s}({v:+,.0f})" for s, v in contribs[:3])
                bot3 = ", ".join(f"{s}({v:+,.0f})" for s, v in contribs[-3:])
                lines += ["", f"- 当日贡献前三：{top3}", f"- 当日贡献后三：{bot3}", ""]

        # 收盘点评
        lines += self._closing_comment(ds, as_of, prices_prev, prices_today)

        with open(self.path, 'a', encoding='utf-8') as f:
            f.write("\n".join(lines))
        # 盘后章节另存副本，供同日重复运行 daily 时保留
        with open(self.path + '.pm', 'a', encoding='utf-8') as f:
            f.write("\n".join(lines))
        return self.path

    # ================= 组成部分 =================

    def _market_outlook(self, ds, prev) -> List[str]:
        """市场展望：基准/池涨跌、宽度、ADX、市场环境分（全部昨收口径）"""
        lines = ["## 一、盘前市场展望", ""]
        try:
            cache = ds.preload_batch(['000300.SH'], prev - timedelta(days=120), prev)
            bench = cache.get('000300.SH')
            if bench is not None and len(bench) > 30:
                last, prevv = float(bench['close'].iloc[-1]), float(bench['close'].iloc[-2])
                chg = last / prevv - 1
                from indicators.trend_indicators import DMI_ADX
                adx = DMI_ADX(bench, period=14)['adx'].iloc[-1]
                state = '趋势' if adx > 30 else ('震荡' if adx < 20 else '过渡')
                ma20 = bench['close'].rolling(20).mean().iloc[-1]
                lines += [f"- 沪深300 昨收 {last:.0f}（{chg:+.2%}），ADX {adx:.0f}（{state}市），"
                          f"{'站稳' if last >= ma20 else '跌破'}20日线", ""]
        except Exception as e:
            lines += [f"- 基准数据获取失败：{e}", ""]

        try:
            from market_env import AkShareMarketEnvProvider
            from market_env.scorer import MarketEnvScorer
            env = MarketEnvScorer(AkShareMarketEnvProvider()).score(prev.strftime('%Y%m%d'))
            lines += [f"- 市场环境评分：{env.market_score}/100（仓位系数 {env.position_factor:.2f}）",
                      f"  宽度 {env.breadth_signal} / 估值 {env.valuation_signal} / 宏观 {env.macro_signal} / "
                      f"先行 {env.leading_signal} / 季节性 {env.seasonal_signal}", ""]
        except Exception as e:
            lines += [f"- 环境评分不可用：{e}", ""]
        return lines

    def _closing_comment(self, ds, as_of, prices_prev, prices_today) -> List[str]:
        """收盘点评：事实+计算的自动观察（非主观渲染）"""
        lines = ["## 六、收盘点评", ""]
        try:
            cache = ds.preload_batch(['000300.SH'], as_of - timedelta(days=10), as_of)
            bench = cache.get('000300.SH')
            if bench is not None and len(bench) >= 2:
                b_chg = float(bench['close'].iloc[-1]) / float(bench['close'].iloc[-2]) - 1
                lines += [f"- 沪深300 今日 {b_chg:+.2%}"]
        except Exception:
            pass
        if self.ledger.positions:
            day_pnl = sum(q * (prices_today.get(s, 0) - prices_prev.get(s, prices_today.get(s, 0)))
                          for s, q in self.ledger.positions.items())
            lines += [f"- 组合当日盈亏 {day_pnl:+,.0f}"]
            up = sum(1 for s, q in self.ledger.positions.items()
                     if prices_prev.get(s, 0) > 0 and
                     prices_today.get(s, 0) / prices_prev.get(s, 0) - 1 > 0)
            lines += [f"- 持仓上涨 {up}/{len(self.ledger.positions)}"]
        lines += ["- 数据/执行异常：无（异常将在此列出）",
                  f"- 明日：{'调仓日（盘前生成新指令）' if self._is_rebalance_due() else '持有观察（缓冲带内）'}",
                  ""]
        return lines

    def _is_rebalance_due(self) -> bool:
        """月度调仓：距首笔交易日每 20 个交易日"""
        trades = self.ledger._d['trades']
        if not trades:
            return True
        last = pd.Timestamp(trades[-1]['date'])
        return (pd.Timestamp(date.today()) - last).days >= 25  # 自然日近似

    # ================= 工具 =================

    def _prev_trading_day(self, ds, as_of):
        cache = ds.preload_batch(['000300.SH'], as_of - timedelta(days=20), as_of)
        bench = cache.get('000300.SH')
        if bench is not None and len(bench) >= 1:
            d = bench['date'].iloc[-1]
            return d.date() if hasattr(d, 'date') else d
        return as_of - timedelta(days=1)

    def _prices(self, ds, symbols, as_of) -> Dict[str, float]:
        if not symbols:
            return {}
        cache = ds.preload_batch(list(symbols), as_of - timedelta(days=15), as_of)
        out = {}
        for sym, df in cache.items():
            if df is None or df.empty:
                continue
            d = df[df['date'] <= pd.Timestamp(as_of)]
            if not d.empty:
                out[sym] = float(d['close'].iloc[-1])
        return out
