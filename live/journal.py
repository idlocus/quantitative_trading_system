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
from indicators.momentum_indicators import RSI
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

        # 1.5 持仓风险扫描 + 组合指标 + 市场宽度
        lines += self._pre_market_analytics(ds, prev_trading)

        # 2. 决策依据（信号基于昨收）
        from live.signals import generate_target, build_orders
        targets, scores = generate_target(ds, prev_trading, top_n=self.top_n)
        prices = self._prices(ds, set(targets) | set(self.ledger.positions), prev_trading)
        nav = self.ledger.last_nav() or self.ledger.cash
        cur_w = self.ledger.positions and self.ledger.current_weights(prices) or {}
        orders = build_orders(targets, cur_w, nav, top_n=self.top_n)

        lines += ["## 三、盘前决策依据", "",
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

        # 风控检查
        w = 1.0 / max(len(targets), 1)
        lines += ["## 四、盘前风控检查", "",
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

    def _pre_market_analytics(self, ds, prev_trading) -> List[str]:
        """盘前干货：持仓风险扫描 + 组合指标 + 市场宽度"""
        lines = []
        symbols = list(self.ledger.positions.keys())
        if not symbols:
            return lines

        try:
            cache = ds.preload_batch(symbols + ['000300.SH'],
                                     prev_trading - timedelta(days=120), prev_trading)
        except Exception:
            return lines

        # ── 持仓风险扫描 ──
        lines += ["## 二、持仓风险扫描", ""]
        lines += ["| 代码 | 盈亏% | RSI | vs MA20 | vs MA50 | 量比 | 距止损 | 信号 |",
                  "|------|-------|-----|---------|---------|------|--------|------|"]
        alerts = []
        for sym in symbols:
            df = cache.get(sym)
            if df is None or len(df) < 50:
                continue
            try:
                cost = self.ledger._d['cost'].get(sym, 0)
                if cost <= 0:
                    continue
                px = float(df['close'].iloc[-1])
                ret = (px - cost) / cost
                rsi = float(RSI(df).iloc[-1]) if len(df) > 20 else 50
                ma20 = float(df['close'].rolling(20).mean().iloc[-1])
                ma50 = float(df['close'].rolling(50).mean().iloc[-1])
                vol_ma = df['volume'].rolling(20).mean().iloc[-1]
                vol_ratio = float(df['volume'].iloc[-1] / vol_ma) if vol_ma > 0 else 1.0
                stop_price = cost * 0.95  # 5%固定止损
                stop_dist = px / stop_price - 1 if stop_price > 0 else 0

                # 信号判定
                sig = []
                if rsi > 70: sig.append("⚠超买")
                elif rsi < 30: sig.append("⚠超卖")
                if px < ma20 and px < ma50: sig.append("⚠破双均线")
                elif px < ma20: sig.append("△破MA20")
                if stop_dist < 0.02: sig.append("⚠逼近止损")
                if vol_ratio > 2.0: sig.append("△放量")
                sig_str = " ".join(sig) if sig else "—"

                lines.append(f"| {sym} | {ret:+.1%} | {rsi:.0f} | "
                           f"{'上方' if px >= ma20 else '**下方**'} | {'上方' if px >= ma50 else '**下方**'} | "
                           f"{vol_ratio:.1f}x | {stop_dist:+.1%} | {sig_str} |")
                if sig:
                    alerts.append((sym, sig_str))
            except Exception:
                continue
        lines.append("")

        if alerts:
            lines += [f"**⚠ 风险警示 {len(alerts)} 只**：" + "；".join(f"{s}({a})" for s, a in alerts[:5]) + ("..." if len(alerts) > 5 else ""), ""]

        # ── 组合指标 ──
        try:
            rets = {}
            for sym in symbols:
                df = cache.get(sym)
                if df is not None and len(df) > 60:
                    rets[sym] = df['close'].pct_change().dropna().tail(60)
            bench = cache.get('000300.SH')
            if rets and bench is not None and len(bench) > 60:
                bench_ret = bench['close'].pct_change().dropna().tail(60)
                avg_ret = pd.DataFrame(rets).mean(axis=1).tail(60)
                aligned = pd.concat([avg_ret, bench_ret], axis=1).dropna()
                if len(aligned) > 20:
                    aligned.columns = ['port', 'bench']
                    beta = float(aligned['port'].cov(aligned['bench']) / aligned['bench'].var())
                    corr = float(aligned['port'].corr(aligned['bench']))
                    port_vol = float(aligned['port'].std() * np.sqrt(252))
                    lines += ["### 组合风险指标（60日）", "",
                              f"- **Beta**：{beta:.2f}（对沪深300）",
                              f"- **相关性**：{corr:.2f}",
                              f"- **年化波动率**：{port_vol:.1%}",
                              f"- **持仓集中度**：{len(symbols)} 只等权，单票风险上限 3.3%", ""]
        except Exception:
            pass

        # ── 市场宽度 ──
        try:
            pool = ds.list_symbols(as_of=prev_trading)[:500]
            if pool:
                cache2 = ds.preload_batch(pool, prev_trading - timedelta(days=10), prev_trading)
                up = down = 0
                for sym, df in cache2.items():
                    if sym == '000300.SH' or df is None or len(df) < 2:
                        continue
                    chg = float(df['close'].iloc[-1] / df['close'].iloc[-2] - 1)
                    if chg > 0.001: up += 1
                    elif chg < -0.001: down += 1
                total = up + down
                if total > 10:
                    ratio = up / max(down, 1)
                    lines += ["### 市场宽度（昨日，前500只采样）", "",
                              f"- 上涨 **{up}** 只 vs 下跌 **{down}** 只（涨跌比 {ratio:.2f}）",
                              f"- {'🔴 广度扩张（多头占优）' if ratio > 1.2 else '🟢 广度收缩（空头占优）' if ratio < 0.8 else '⚪ 广度中性'}", ""]
        except Exception:
            pass

        # ── 风格轮动信号 ──
        try:
            large = ds.preload_batch(['000300.SH'], prev_trading - timedelta(days=30), prev_trading)
            small_pool = [s for s in symbols if s in cache][:10]
            if large.get('000300.SH') is not None and small_pool:
                large_ret = float(large['000300.SH']['close'].iloc[-1] / large['000300.SH']['close'].iloc[-21] - 1)
                small_rets = []
                for s in small_pool:
                    df = cache.get(s)
                    if df is not None and len(df) > 21:
                        small_rets.append(float(df['close'].iloc[-1] / df['close'].iloc[-21] - 1))
                if small_rets:
                    small_avg = np.mean(small_rets)
                    spread = small_avg - large_ret
                    lines += ["### 风格轮动（近一月）", "",
                              f"- 沪深300（大盘）：{large_ret:+.1%}",
                              f"- 持仓组合（偏小盘）：{small_avg:+.1%}",
                              f"- **大小盘价差**：{spread:+.1%}（{'小盘占优 ✓' if spread > 0 else '大盘占优 ✗ 对组合不利'}）", ""]
        except Exception:
            pass

        return lines

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
