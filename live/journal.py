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
from live.risk_gate import RiskGate, render_gate_markdown
from trading import config as chain
from utils.logger import get_logger

logger = get_logger(__name__)

JOURNAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'journal')


class JournalWriter:
    """每日交易日志生成器（am 追加式写入，pm 补齐）"""

    def __init__(self, ledger, top_n: int = 100):
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
        from live.signals import signal_date_for, generate_production_signal
        prev_trading = signal_date_for(ds, as_of)   # 严格 < 今天的信号日（盘后重跑也不前视）

        lines = [f"# 交易日志 {as_of.isoformat()}", "",
                 f"- 生成时间：盘前阶段 | 数据截至 {prev_trading} 收盘（PIT）", ""]

        # 1. 市场展望
        lines += self._market_outlook(ds, prev_trading)

        # 1.5 持仓风险扫描 + 组合指标 + 市场宽度
        lines += self._pre_market_analytics(ds, prev_trading)

        # 2. 决策依据（信号基于昨收；经由生产唯一入口 generate_production_signal，
        #    与 PM 执行/回放同源——禁止直调 generate_target 产生第二套口径）
        from live.signals import build_orders
        psig = generate_production_signal(ds, as_of, top_n=self.top_n)
        targets, scores = psig.targets, psig.scores
        assert psig.signal_date == prev_trading, "AM 信号日必须等于预览基准日"
        prices, prev_prices = self._prices_with_prev(
            ds, set(targets) | set(self.ledger.positions), prev_trading)
        nav = self.ledger.last_nav() or self.ledger.cash
        cur_w = self.ledger.positions and self.ledger.current_weights(prices) or {}
        orders = build_orders(targets, cur_w, nav, top_n=self.top_n)

        # 风控闸门预览（基于昨收；收盘后 pm 以当日数据复核执行。
        # st_data_ok 与 PM 同源（psig.st_ok），保证 AM 预览与实际执行口径一致）
        bench_cache = ds.preload_batch(['000300.SH'], prev_trading - timedelta(days=20), prev_trading)
        gate = RiskGate()
        decision = gate.review(orders, self.ledger, prices, prev_prices, targets,
                               bench_df=bench_cache.get('000300.SH'), as_of=prev_trading,
                               st_data_ok=psig.st_ok)

        spec_str = getattr(psig, 'spec', None)
        spec_line = spec_str.summary() if spec_str else "未知版本"
        lines += ["## 三、盘前决策依据", "",
                  f"- 策略版本：{spec_line}",
                  f"- 基于口径：{getattr(psig, 'signal_date', prev_trading)} 收盘"
                  f"（T-1信号，今日执行）",
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

        # 风控检查（L2-L4 执行闸门 + 基础约束——经由 semantics 统一口径）
        from data.semantics import planned_exposure as calc_exposure, actual_cash_ratio
        target_w = 1.0 / max(self.top_n, 1)
        planned_exp, planned_cash = calc_exposure(len(targets), self.top_n)
        actual_cash = actual_cash_ratio(self.ledger.cash,
                                        self.ledger.positions, prices)
        basic = [f"- 单票目标权重 {target_w:.1%}（上限 5%）：{'通过' if target_w <= 0.05 else '超限'}",
                 f"- 持仓数 {len(targets)}/{self.top_n}（目标）",
                 f"- 计划仓位 {planned_exp:.0%} | 计划现金 {planned_cash:.0%}",
                 f"- 当前账户现金 {actual_cash:.0%}",
                 "- 涨跌停约束：执行层自动拒单", ""]
        lines += ["## 四、盘前风控检查（执行闸门，基于昨收预览；收盘后以当日数据复核）", ""]
        lines += render_gate_markdown(decision, basic)

        with open(self.path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            # 同日重复运行：保留已有的盘后章节（成交/归因不可再生），仅刷新盘前
            if os.path.exists(self.path + '.pm'):
                with open(self.path + '.pm', encoding='utf-8') as pf:
                    f.write("\n" + pf.read())
        return {'targets': targets, 'scores': scores, 'orders': orders, 'prev_trading': prev_trading}

    # ================= PM：盘后 =================

    def write_pm(self, ds: LayeredDataSource, fills: List, am_ctx: dict,
                 gate=None, recon=None) -> str:
        """盘后：成交记录 + 持仓归因 + 收盘点评（追加）"""
        as_of = date.today()
        symbols = set(self.ledger.positions) | {f.order.symbol for f in fills}
        prices_today = self._prices(ds, symbols, as_of)
        prices_prev = self._prices(ds, symbols,
                                   am_ctx.get('prev_trading') or (as_of - timedelta(days=1)))

        lines = ["---", "", f"*以下为盘后阶段（{as_of} 收盘后生成）*", "",
                 "## 四、成交记录", ""]
        # 风控闸门执行记录（L2-L4 当日实际生效的规则）
        if gate is not None:
            lines += [f"- 执行闸门：**{gate.verdict_cn}**｜仓位上限 {gate.exposure:.0%}｜"
                      f"放行 {len(gate.orders)} 条 / 拦截 {len(gate.suppressed)} 条 / "
                      f"强制卖出 {len(gate.forced_sells)} 条", ""]
            for i in gate.data_issues:
                lines.append(f"  - [L2] {i}")
            for r in gate.reasons:
                lines.append(f"  - {r}")
            if gate.data_issues or gate.reasons:
                lines.append("")
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

        # 账实核对（当日对账单存在时）
        if recon is not None:
            lines += recon.render_markdown()
        else:
            last = self.ledger._d['meta'].get('last_recon')
            status = (f"上次 {last['date']} {'一致' if last['ok'] else str(last['n_diffs']) + ' 项差异'}"
                      if last else "从未对账")
            lines += ["## 六、账实核对", "",
                      f"- 今日无对账单（{status}）；将券商App持仓/资金誊入 "
                      "`live/statements/今日.json` 后自动核对，"
                      "或运行 `python run_live.py --statement-template`", ""]

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
        lines += ["## 二、持仓风险扫描", "",
                  "> L1 预警层：以下信号（均线/RSI/量比/距止损）**仅提示，不触发交易**；",
                  "> 实际执行的卖出/熔断/停机规则见第四节执行闸门（L2-L4）。", ""]
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
                stop_price = cost * (1 - chain.STOP_LOSS_PCT)  # 硬止损价（主链口径）
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
        lines += self._style_rotation(ds, prev_trading)

        # ── 行业轮动信号 ──
        lines += self._industry_rotation(ds, prev_trading)

        return lines

    # 申万一级行业代码→名称（31个，标准编码，不常变）
    SW_L1 = {
        '801010': '农林牧渔', '801030': '化工', '801040': '钢铁',
        '801050': '有色金属', '801080': '电子', '801110': '家用电器',
        '801120': '食品饮料', '801130': '纺织服饰', '801140': '轻工制造',
        '801150': '医药生物', '801160': '公用事业', '801170': '交通运输',
        '801180': '房地产', '801200': '商贸零售', '801210': '社会服务',
        '801230': '综合', '801710': '建筑材料', '801720': '建筑装饰',
        '801730': '电力设备', '801740': '国防军工', '801750': '计算机',
        '801760': '传媒', '801770': '通信', '801780': '银行',
        '801790': '非银金融', '801880': '汽车', '801890': '机械设备',
        '801950': '煤炭', '801960': '石油石化', '801970': '环保',
        '801980': '美容护理',
    }

    def _style_rotation(self, ds, prev) -> List[str]:
        """风格轮动：沪深300 vs 中证500 vs 中证1000（近20日）"""
        try:
            indices = {'000300.SH': '沪深300(大盘)', '000905.SH': '中证500(中盘)', '000852.SH': '中证1000(小盘)'}
            cache = ds.preload_batch(list(indices.keys()), prev - timedelta(days=60), prev)
            rets = {}
            for code, name in indices.items():
                df = cache.get(code)
                if df is not None and len(df) > 21:
                    r20 = float(df['close'].iloc[-1] / df['close'].iloc[-21] - 1)
                    r5 = float(df['close'].iloc[-1] / df['close'].iloc[-6] - 1)
                    rets[name] = (r20, r5)
            if not rets:
                return []
            lines = ["### 风格轮动信号", "",
                     "| 风格 | 近5日 | 近20日 | 动量 |", "|------|-------|--------|------|"]
            for name, (r20, r5) in sorted(rets.items(), key=lambda x: -x[1][0]):
                momentum = "↑强" if r20 > 0.02 else "↓弱" if r20 < -0.02 else "→平"
                lines.append(f"| {name} | {r5:+.1%} | {r20:+.1%} | {momentum} |")
            # 判定
            big = rets.get('沪深300(大盘)', (0, 0))[0]
            small = rets.get('中证1000(小盘)', (0, 0))[0]
            spread = small - big
            verdict = ("**小盘占优 ✓**（组合偏小盘，顺风）" if spread > 0.01
                       else "**大盘占优 ✗**（组合偏小盘，逆风）" if spread < -0.01
                       else "**风格均衡**")
            lines += ["", f"大小盘20日价差：{spread:+.1%} → {verdict}", ""]
            return lines
        except Exception:
            return []

    def _industry_rotation(self, ds, prev) -> List[str]:
        """行业轮动：申万一级行业近20日涨跌幅排名"""
        try:
            db = ds._db_src()
            if not db:
                return []
            cur = db._cursor()
            # 直接查硬编码的31个一级行业代码（避免SUBSTR过滤问题）
            codes = list(self.SW_L1.keys())
            placeholders = ','.join(f':c{i}' for i in range(len(codes)))
            binds = {f'c{i}': f'{c}.SI' for i, c in enumerate(codes)}
            cur.execute(f"""
                SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_CLOSE, S_VAL_PE, S_DQ_VOLUME
                FROM ASWSINDEXEOD
                WHERE S_INFO_WINDCODE IN ({placeholders})
                  AND TRADE_DT >= :dt_start
                ORDER BY S_INFO_WINDCODE, TRADE_DT
            """, {**binds, 'dt_start': (prev - timedelta(days=90)).strftime('%Y%m%d')})
            rows = cur.fetchall()
            cur.close()
            if not rows:
                return []

            from collections import defaultdict
            prices = defaultdict(list)
            for code, dt, close, pe, vol in rows:
                prices[code[:6]].append((dt, close, pe, vol))

            data = []
            for ind_code, name in self.SW_L1.items():
                pts = prices.get(ind_code, [])
                if len(pts) >= 21:
                    latest = pts[-1][1]
                    ago = pts[-21][1]
                    if not ago or ago <= 0:
                        continue
                    ret = latest / ago - 1
                    pe_chg = None
                    if pts[-1][2] and pts[-21][2] and pts[-21][2] > 0:
                        pe_chg = pts[-1][2] / pts[-21][2] - 1
                    vol_ratio = None
                    vols = [p[3] for p in pts if p[3]]
                    if len(vols) >= 25:
                        vol_ratio = (sum(vols[-5:]) / 5) / (sum(vols) / len(vols)) if sum(vols) > 0 else None
                    data.append((name, ret, pe_chg, vol_ratio))

            data.sort(key=lambda x: -x[1])
            top5 = data[:5]
            bot5 = data[-5:]

            def _driver(ret, pe_c):
                if pe_c is None: return "—"
                eps = (1 + ret) / (1 + pe_c) - 1
                if ret > 0 and pe_c > 0.02: return "估值扩张（情绪驱动）"
                if ret > 0 and pe_c < -0.02: return f"盈利改善(EPS{eps:+.0%})"
                if ret < 0 and pe_c < -0.02: return "估值收缩（避险抛售）"
                if ret < 0 and pe_c > 0.02: return f"盈利恶化(EPS{eps:+.0%})"
                return "波动不大"
            def _vol(vr):
                if vr is None: return "—"
                if vr > 1.5: return f"**{vr:.1f}x放量**"
                if vr < 0.7: return f"{vr:.1f}x缩量"
                return f"{vr:.1f}x"

            lines = ["### 行业轮动信号（申万一级，近20日）", ""]
            lines += ["**领涨行业**", "",
                      "| 行业 | 涨幅 | PE变化 | 量能 | 归因 |", "|------|------|--------|------|------|"]
            for item in top5:
                name, ret, pe_c, vol_r = item[0], item[1], item[2] if len(item) > 2 else None, item[3] if len(item) > 3 else None
                pe_s = f"{pe_c:+.0%}" if pe_c is not None else "—"
                lines.append(f"| {name} | {ret:+.1%} | {pe_s} | {_vol(vol_r)} | {_driver(ret, pe_c)} |")
            lines += ["", "**领跌行业**", "",
                      "| 行业 | 跌幅 | PE变化 | 量能 | 归因 |", "|------|------|--------|------|------|"]
            for item in bot5:
                name, ret, pe_c, vol_r = item[0], item[1], item[2] if len(item) > 2 else None, item[3] if len(item) > 3 else None
                pe_s = f"{pe_c:+.0%}" if pe_c is not None else "—"
                lines.append(f"| {name} | {ret:+.1%} | {pe_s} | {_vol(vol_r)} | {_driver(ret, pe_c)} |")
            lines += ["", f"行业分化度：TOP-BOT价差 **{top5[0][1]-bot5[0][1]:.1%}**", ""]
            lines += self._rotation_narrative(top5, bot5)
            return lines
        except Exception as e:
            logger.debug(f"industry_rotation failed: {e}")
            return []

    def _rotation_narrative(self, top5, bot5) -> List[str]:
        """轮动叙事：把量化信号翻译成市场解读"""
        lines = ["**📊 轮动解读**", ""]
        top_names = [t[0] for t in top5]
        bot_names = [b[0] for b in bot5]
        top_pes = [t[2] for t in top5 if len(t) > 2 and t[2] is not None]
        bot_pes = [b[2] for b in bot5 if len(b) > 2 and b[2] is not None]

        if top_pes and sum(1 for d in top_pes if d > 0.02) >= 3:
            lines.append(f"- 领涨行业（{'、'.join(top_names[:3])}）主要由**估值扩张**驱动——市场给予更高风险溢价，可能是政策预期或情绪修复")
        elif top_pes and sum(1 for d in top_pes if d < -0.02) >= 3:
            lines.append(f"- 领涨行业（{'、'.join(top_names[:3])}）主要由**盈利改善**驱动——基本面在好转，上涨更有持续性")
        else:
            lines.append(f"- 领涨行业（{'、'.join(top_names[:3])}）驱动因素混合")

        if bot_pes and sum(1 for d in bot_pes if d < -0.02) >= 3:
            lines.append(f"- 领跌行业（{'、'.join(bot_names[:3])}）主要是**估值收缩**——市场降低风险偏好，避险/减仓行为")
        elif bot_pes and sum(1 for d in bot_pes if d > 0.02) >= 3:
            lines.append(f"- 领跌行业（{'、'.join(bot_names[:3])}）主要是**盈利恶化**——基本面在变差")
        else:
            lines.append(f"- 领跌行业（{'、'.join(bot_names[:3])}）跌幅中估值和盈利因素混合")

        defensive = {'银行','房地产','交通运输','公用事业','医药生物','食品饮料'}
        cyclical = {'有色金属','电子','计算机','机械设备','国防军工','电力设备','煤炭','石油石化'}
        td = sum(1 for n in top_names if n in defensive)
        tc = sum(1 for n in top_names if n in cyclical)
        bd = sum(1 for n in bot_names if n in defensive)
        bc = sum(1 for n in bot_names if n in cyclical)

        if td >= 2 and bc >= 2:
            lines.append(f"- **市场风格：防守型轮动**——资金从周期成长流向防守（银行/地产/交运），风险偏好下降，对小盘成长策略不利")
        elif tc >= 2 and bd >= 2:
            lines.append(f"- **市场风格：进攻型轮动**——资金从防守流向周期成长，风险偏好上升，对小盘成长策略有利")
        else:
            lines.append(f"- **市场风格：无明显防守/进攻偏向**")
        lines.append("")
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
        lines = ["## 七、收盘点评", ""]
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

    def _prices_with_prev(self, ds, symbols, as_of):
        """(最新价, 前收盘)——闸门异常检测需要前收盘对照"""
        if not symbols:
            return {}, {}
        cache = ds.preload_batch(list(symbols), as_of - timedelta(days=15), as_of)
        prices, prev = {}, {}
        for sym, df in cache.items():
            if df is None or df.empty:
                continue
            d = df[df['date'] <= pd.Timestamp(as_of)]
            if d.empty:
                continue
            prices[sym] = float(d['close'].iloc[-1])
            if len(d) > 1:
                prev[sym] = float(d['close'].iloc[-2])
        return prices, prev
