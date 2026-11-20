"""Long-only research simulation with explicit costs, blocked exits and receivables."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import floor, isfinite

import pandas as pd

from trading.rules import round_buy_qty, round_sell_qty
from .portfolio_metrics import DailyRisk, performance


@dataclass(frozen=True)
class Distribution:
    key: str
    symbol: str
    record: pd.Timestamp
    ex_date: pd.Timestamp | None
    payout: pd.Timestamp | None
    listing: pd.Timestamp | None
    cash: float
    stock: float
    ambiguous: bool = False


def distributions(source: pd.DataFrame, symbols: set[str], dates: pd.DatetimeIndex) -> tuple[list, dict]:
    frame = source.loc[source['S_DIV_PROGRESS'].astype('string').eq('3') & source['S_INFO_WINDCODE'].isin(symbols)].copy()
    date_columns = ['EQY_RECORD_DT', 'EX_DT', 'DVD_PAYOUT_DT', 'LISTING_DT_OF_DVD_SHR']
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column], format='%Y%m%d', errors='coerce')
    for column in ('CASH_DVD_PER_SH_PRE_TAX', 'STK_DVD_PER_SH'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce').fillna(0.)
    frame = frame.loc[((frame['CASH_DVD_PER_SH_PRE_TAX'] > 0) | (frame['STK_DVD_PER_SH'] > 0)) &
        (frame['EQY_RECORD_DT'].between(dates[0], dates[-1]) | frame['EX_DT'].between(dates[0], dates[-1]))]
    economic = ['S_INFO_WINDCODE', *date_columns, 'CASH_DVD_PER_SH_PRE_TAX', 'STK_DVD_PER_SH']
    before = len(frame)
    frame = frame.drop_duplicates(economic)
    identical_removed = before - len(frame)
    # 2026-11-09 修复：同 (symbol, 登记日, 除权日) 的多笔分红=同日到账的真实
    # 多次分配（实证：601088 2017-07-10 年报2.51+特别0.46=2.97，与除权参考价
    # 19.32=昨收22.29-2.97 分毫不差），合并金额而非标记歧义；歧义仅留给
    # 真病态（缺登记日/顺序无效）。
    # 2026-11-19 审计一-8 修订：合并键必须包含**支付日/红股上市日**——
    # 同除权日但不同结算日的多笔分配合并到第一个有效支付日后，现金或
    # 红股会提前/延后可用。除权参考价的调整按各自事件顺序进行（现金
    # 分红精确 = 总额扣减；送股近似逐笔比例调整）；应收现金与锁定股份
    # 按各自到期日分别记账。
    key = ['S_INFO_WINDCODE', 'EQY_RECORD_DT', 'EX_DT',
           'DVD_PAYOUT_DT', 'LISTING_DT_OF_DVD_SHR']
    merged_count = int(frame.duplicated(key).sum())
    if merged_count:
        frame = frame.groupby(key, as_index=False, dropna=False).agg({
            'CASH_DVD_PER_SH_PRE_TAX': 'sum',
            'STK_DVD_PER_SH': 'sum',
            'OBJECT_ID': 'first'})
    duplicate = frame.duplicated(['S_INFO_WINDCODE', 'EQY_RECORD_DT', 'EX_DT',
                                  'DVD_PAYOUT_DT', 'LISTING_DT_OF_DVD_SHR'], keep=False)
    actions = []
    unknown_records = 0
    for (_, row), ambiguous in zip(frame.iterrows(), duplicate):
        if pd.isna(row.EQY_RECORD_DT):
            unknown_records += 1
            record = row.EX_DT
            ambiguous = True
        else:
            record = row.EQY_RECORD_DT
        if pd.isna(record):
            continue
        ex_date = None if pd.isna(row.EX_DT) else row.EX_DT
        invalid_order = ex_date is not None and record >= ex_date
        actions.append(Distribution(str(row.OBJECT_ID), row.S_INFO_WINDCODE, record, ex_date,
            None if pd.isna(row.DVD_PAYOUT_DT) else row.DVD_PAYOUT_DT,
            None if pd.isna(row.LISTING_DT_OF_DVD_SHR) else row.LISTING_DT_OF_DVD_SHR,
            float(row.CASH_DVD_PER_SH_PRE_TAX), float(row.STK_DVD_PER_SH), bool(ambiguous or invalid_order)))
    return actions, {'events': len(actions), 'identical_rows_deduplicated': identical_removed,
                     'same_day_distributions_merged': merged_count,
                     'ambiguous_events': sum(action.ambiguous for action in actions), 'unknown_record_dates': unknown_records}


class PortfolioSimulation:
    def __init__(self, name: str, config: dict, cost: float, guarded: bool, actions: list[Distribution],
                 dates: pd.DatetimeIndex, delist_dates: dict, vol_target: bool = False,
                 vol_target_pure: bool = False, external_scales=None,
                 daily_weights=None, hold_buffer_rank: int = 0,
                 hot_exit_sets=None, daily_buffer_overrides=None,
                 daily_position_overrides=None):
        self.name, self.config, self.cost, self.guarded = name, config, cost, guarded
        # vol targeting 风控变体（2026-11-09 预注册试验C；默认关闭=行为不变）
        self.vol_targeted = bool(vol_target)
        # 纯 vol targeting（试验C修订）：无预算floor，只做波动率缩仓+
        # 红线halt（对照 uncontrolled 的超标回撤场景）
        self.vol_pure = bool(vol_target_pure)
        # E系仓位层试验：外部目标仓位序列（date→0..1，PIT 由调用方保证）
        self.external_scales = external_scales
        # 审计 2026-11-19 一-3（前视收口）：所有"下单输入"统一按**决策日**
        # 键传入（与排名同口径），内部平移到下一交易日生效——此前权重/
        # 热卖名单/日度缓冲带/日度持仓数按执行日直接读取，T 日开盘用到了
        # T 日收盘才能算出的数据。键平移规则：决策日 D 的条目在 D 之后
        # 的第一个交易日生效；决策日在最后一日或非交易日 → 顺延到下一个
        # 存在的交易日，不存在则丢弃。
        def _lag_to_next_day(mapping):
            if mapping is None:
                return None
            shifted = {}
            for key, value in mapping.items():
                pos = dates.searchsorted(pd.Timestamp(key), side='right')
                if pos < len(dates):
                    shifted[dates[pos]] = value
            return shifted
        # 增厚②：逐票权重 {decision_date: {symbol: w}}（None=等权，行为不变）
        self.daily_weights = _lag_to_next_day(daily_weights)
        # G系：缓冲带持有——跌出 top-K 才卖（0=排名强换，行为不变）
        self.hold_buffer = int(hold_buffer_rank)
        # I系：热卖出——{decision_date: set(symbol)} 持仓中触发即强制卖
        self.hot_exit_sets = _lag_to_next_day(hot_exit_sets) or {}
        # W系：日度缓冲带覆盖 {decision_date: K}——启动日收盘决策、次日生效
        self.daily_buffer_overrides = _lag_to_next_day(daily_buffer_overrides) or {}
        # X系：日度持仓数覆盖 {decision_date: N}——启动日收盘决策、次日生效
        self.daily_position_overrides = _lag_to_next_day(daily_position_overrides) or {}
        self.cash = float(config['initial_paper_capital'])
        self.risk = DailyRisk(self.cash, config['monthly_loss_limit'], config['peak_drawdown_limit'], config['exposure_budget_adverse_shock'],
                              config.get('absolute_monthly_loss_cap'), config.get('absolute_peak_loss_cap'))
        self.positions, self.marks, self.factors, self.entitlements = {}, {}, {}, {}
        self.cash_claims, self.locked_shares, self.nav_rows, self.trades = [], [], [], []
        self.issues, self.blocked = Counter(), Counter()
        self.issue_rows = []
        self.total_cost = 0.
        self.targets = []
        self.last_rebalance = None
        self.delist_dates = delist_dates
        self.by_record, self.by_ex = defaultdict(list), defaultdict(list)
        for event in actions:
            ri = dates.searchsorted(event.record, side='right') - 1
            if ri >= 0 and event.record <= dates[-1]:
                self.by_record[dates[ri]].append(event)
            if event.ex_date is not None and dates[0] <= event.ex_date <= dates[-1]:
                self.by_ex[dates[dates.searchsorted(event.ex_date)]].append(event)

    def quantity(self, symbol):
        return self.positions.get(symbol, 0) + sum(item['qty'] for item in self.locked_shares if item['symbol'] == symbol)

    def issue(self, kind, day, symbol):
        self.issues[kind] += 1
        self.issue_rows.append({'date': day.date().isoformat(), 'symbol': symbol, 'issue': kind})

    def nav(self):
        return self.cash + sum(item['value'] for item in self.cash_claims) + sum(
            self.quantity(symbol) * self.marks.get(symbol, 0.)
            for symbol in set(self.positions) | {item['symbol'] for item in self.locked_shares})

    def apply_events(self, day):
        terms = defaultdict(lambda: [0.0, 0.0])
        for event in self.by_ex.get(day, []):
            qty = self.entitlements.pop(event.key, 0)
            if qty and event.ambiguous:
                self.issue('ambiguous_held_corporate_action', day, event.symbol)
                continue
            if qty:
                cash_value = qty * event.cash * (1 - self.config['cash_dividend_haircut'])
                if cash_value:
                    self.cash_claims.append({'value': cash_value, 'pay': event.payout})
                    if event.payout is None:
                        self.issue('unknown_cash_payment_locked', day, event.symbol)
                bonus = floor(qty * event.stock + 1e-9)
                if bonus:
                    self.locked_shares.append({'symbol': event.symbol, 'qty': bonus, 'listing': event.listing})
                    if event.listing is None:
                        self.issue('unknown_bonus_listing_locked', day, event.symbol)
            if self.quantity(event.symbol) and event.symbol in self.marks and not event.ambiguous:
                terms[event.symbol][0] += event.cash
                terms[event.symbol][1] += event.stock
        for symbol, (cash, stock) in terms.items():
            self.marks[symbol] = max(0., (self.marks[symbol] - cash) / (1 + stock))
        remaining = []
        for item in self.cash_claims:
            if item['pay'] is not None and item['pay'] < day:
                self.cash += item['value']
            else:
                remaining.append(item)
        self.cash_claims = remaining
        remaining = []
        for item in self.locked_shares:
            if item['listing'] is not None and item['listing'] <= day:
                self.positions[item['symbol']] = self.positions.get(item['symbol'], 0) + item['qty']
            else:
                remaining.append(item)
        self.locked_shares = remaining

    def mark(self, day, bars, field):
        held = set(self.positions) | {item['symbol'] for item in self.locked_shares}
        for symbol in held:
            if pd.notna(self.delist_dates.get(symbol)) and day >= self.delist_dates[symbol]:
                self.positions.pop(symbol, None)
                self.locked_shares = [item for item in self.locked_shares if item['symbol'] != symbol]
                self.marks[symbol] = 0.
                self.issue('unmodeled_delisting_zero_recovery', day, symbol)
                continue
            bar = bars.get(symbol)
            value = bar.get(field) if bar else None
            if value is not None and isfinite(value) and value > 0:
                self.marks[symbol] = value
            elif field == 'close' and (bar is None or not bool(bar.get('source_suspended', False))):
                self.issue('unexplained_held_valuation_gap', day, symbol)
            elif field == 'close' and (value is None or not isfinite(value) or value <= 0):
                self.issue('documented_suspension_valuation_carry', day, symbol)
            if symbol not in self.marks:
                raise ValueError('Held asset has no valuation mark')

    def fill(self, day, symbol, side, requested, bars):
        bar = bars.get(symbol)
        if bar is None or not bool(bar.get('numeric_bar_valid', False)) or bool(bar.get('source_suspended', False)):
            self.blocked[side + '_no_trading_bar'] += 1
            if bar is not None and not bool(bar.get('source_suspended', False)):
                self.issue('invalid_unsuspended_execution_bar', day, symbol)
            return
        price, volume = bar['open'], bar['volume']
        upper, lower = bar['source_upper_limit'], bar['source_lower_limit']
        if not all(isfinite(value) and value > 0 for value in (price, volume, upper, lower)):
            self.blocked[side + '_unknown_limits_or_liquidity'] += 1
            return
        if (side == 'BUY' and price >= upper - .005) or (side == 'SELL' and price <= lower + .005):
            self.blocked[side + '_opening_limit'] += 1
            return
        capacity = int(volume * self.config['volume_participation_cap'])
        qty = min(requested, capacity)
        if side == 'BUY':
            qty = min(qty, int(self.cash / (price * (1 + self.cost))))
            qty = round_buy_qty(symbol, qty)
        else:
            qty = round_sell_qty(symbol, qty, self.positions.get(symbol, 0))
        if qty <= 0:
            self.blocked[side + '_quantity_or_cash'] += 1
            return
        value, expense = price * qty, price * qty * self.cost
        if side == 'BUY':
            self.cash -= value + expense
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
            self.marks[symbol] = price
            self.factors[symbol] = bar['adjfactor']
        else:
            self.cash += value - expense
            self.positions[symbol] -= qty
            if not self.positions[symbol]:
                del self.positions[symbol]
        if self.cash < -1e-6:
            raise ValueError('Research execution created negative cash')
        self.cash = max(0., self.cash)
        self.total_cost += expense
        self.trades.append({'date': day.date().isoformat(), 'symbol': symbol, 'side': side,
                            'price': price, 'quantity': qty, 'scenario_cost': expense})

    def _vol_target_cap(self) -> float:
        """波动率目标敞口：min(1, 15%年化 / 60日实现年化波动)×gross（预注册
        试验C；不足60个净值观测=满仓起步；只用已收盘数据，无前视）"""
        navs = [float(row['nav']) for row in self.nav_rows[-61:]]
        if len(navs) < 61:
            return self.config['gross_target']
        rets = [navs[i + 1] / navs[i] - 1.0 for i in range(len(navs) - 1)]
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / (n - 1)
        realized_ann = var ** 0.5 * (252 ** 0.5)
        if realized_ann <= 0:
            return self.config['gross_target']
        scale = min(1.0, 0.15 / realized_ann)
        return scale * self.config['gross_target']

    def step(self, index, day, bars, ranking, eligible_previous):
        self.risk.begin_day(day)
        self.apply_events(day)
        explained = {event.symbol for event in self.by_ex.get(day, []) if not event.ambiguous}
        for symbol in set(self.positions) | {item['symbol'] for item in self.locked_shares}:
            factor = bars.get(symbol, {}).get('adjfactor')
            previous = self.factors.get(symbol)
            if factor is not None and isfinite(factor) and factor > 0:
                if previous and abs(factor / previous - 1) > 1e-7 and symbol not in explained:
                    self.issue('unexplained_held_adjustment_change', day, symbol)
                self.factors[symbol] = factor
        self.mark(day, bars, 'open')
        opening_nav = self.nav()
        opening_risk = self.risk.observe(opening_nav)
        if self.guarded:
            cap = self.risk.exposure_cap(opening_nav, self.config['gross_target'])
        elif self.vol_targeted:
            # 试验C：常态敞口=波动率目标，红线仍由 exposure_cap 的 floor 裁剪
            cap = self.risk.exposure_cap(opening_nav, self._vol_target_cap())
        elif self.vol_pure:
            # 试验C修订：纯波动率目标（无预算floor）；红线突破即halt=0
            cap = 0.0 if self.risk.halted else self._vol_target_cap()
        elif self.external_scales is not None:
            # E系：外部仓位因子决定常态敞口；红线突破仍halt=0
            cap = 0.0 if self.risk.halted else min(
                float(self.external_scales.get(day, self.config['gross_target'])),
                self.config['gross_target'])
        else:
            cap = self.config['gross_target']
        rebalance = self.last_rebalance is None or index - self.last_rebalance >= self.config['rebalance_sessions']
        if ranking is None:
            rebalance = False
            self.targets = list(dict.fromkeys([*self.targets, *self.positions]))
            eligible_previous = set(self.targets)
        if rebalance and not ranking:
            # 二轮复审#3：有效空榜必须清空目标（此前沿用旧目标，新排名
            # 已剔除的证券继续被持有）。空榜不消耗调仓节奏（数据缺口日
            # 不应顺延整个周期）。
            self.targets = []
        elif rebalance and ranking:
            n_today = self.daily_position_overrides.get(day, self.config['positions'])
            if self.hold_buffer > self.config['positions'] or self.daily_buffer_overrides:
                k_today = self.daily_buffer_overrides.get(day, self.hold_buffer)
                band = ranking[:k_today]
                hot = set((self.hot_exit_sets or {}).get(day, ()))
                # 审计 2026-11-19 一-5：缓冲带锚定**真实持仓**而非昨日目标——
                # 昨日涨停未买到时旧目标里的票不算老仓（今日跌出窄带应正常
                # 轮出）；受阻未卖的真实持仓仍在宽带内则继续保留。
                # 热卖剔除同时约束保留与新增：被热卖规则剔出的票不得立即
                # 从新增名单回补（I 系漏洞）。
                holdings = set(self.positions) | {item['symbol'] for item in self.locked_shares}
                keep = [s for s in band if s in holdings and s not in hot]
                # 审计 2026-11-19 一-4（X系）：新增候选也按今日目标持仓数取
                # （此前恒按 config['positions']，N=70 时新增仍只有前50只）
                fresh = [s for s in ranking[:n_today] if s not in keep and s not in hot]
                self.targets = (keep + fresh)[:n_today]
            else:
                self.targets = ranking[:n_today]
            self.last_rebalance = index
        self.targets = [symbol for symbol in self.targets if symbol in eligible_previous
                        and (pd.isna(self.delist_dates.get(symbol)) or day < self.delist_dates[symbol])]
        selected = set(self.targets)
        exposed_value = sum(self.quantity(symbol) * self.marks.get(symbol, 0.) for symbol in set(self.positions) | {item['symbol'] for item in self.locked_shares})
        reduce_risk = (self.guarded or self.vol_targeted or self.vol_pure or self.external_scales is not None) and exposed_value > cap * opening_nav + 1e-6
        target_quantities = {}
        if rebalance or reduce_risk:
            wmap = (self.daily_weights or {}).get(day, {})
            # 二轮复审#3：等权分母=策略规定的目标只数（当日 N 覆盖值，
            # 无覆盖=config positions）——此前用 len(targets)，候选不足时
            # 单票权重被放大（50 只目标只剩 1 只合格时买到 60% 而非 1.2%）。
            n_intended = self.daily_position_overrides.get(day, self.config['positions'])
            if any(not isfinite(w) or w < 0 for w in wmap.values()):
                raise ValueError('Weights must be finite and nonnegative')
            wsum = sum(wmap.get(s, 0.) for s in self.targets) or 1.
            for symbol in self.targets:
                w = wmap.get(symbol, 0.0) / wsum if wmap else 1.0 / max(1, n_intended)
                per_name = cap * opening_nav * w
                price = bars.get(symbol, {}).get('open')
                if price is not None and isfinite(price) and price > 0:
                    target_quantities[symbol] = max(0, int(per_name / price) - sum(item['qty'] for item in self.locked_shares if item['symbol'] == symbol))
        else:
            target_quantities = {symbol: qty for symbol, qty in self.positions.items() if symbol in selected}
        for symbol, qty in list(self.positions.items()):
            sell = qty - target_quantities.get(symbol, 0)
            if sell > 0:
                self.fill(day, symbol, 'SELL', sell, bars)
        if rebalance and cap > 0:
            # 二轮复审#2：先卖后买，买入预算=风险上限−**卖单执行后**的实际
            # 持仓市值——此前目标数量按名义预算直算，原持仓跌停/停牌卖不出
            # 时新仓仍按剩余现金买满，总仓位可从上限 60% 顶到 100%。
            held_value_after_sells = sum(
                self.quantity(symbol) * self.marks.get(symbol, 0.)
                for symbol in set(self.positions) | {item['symbol'] for item in self.locked_shares})
            held = set(self.positions) | {item['symbol'] for item in self.locked_shares}
            budget_left = max(0., cap * self.nav() - held_value_after_sells)
            for symbol in self.targets:
                if symbol not in held and len(held) >= n_intended:
                    self.blocked['position_limit'] += 1
                    continue
                buy = target_quantities.get(symbol, 0) - self.positions.get(symbol, 0)
                if buy > 0:
                    price = bars.get(symbol, {}).get('open')
                    if price is not None and isfinite(price) and price > 0:
                        buy = min(buy, int(budget_left / (price * (1 + self.cost))))
                    if buy > 0:
                        before = self.quantity(symbol) * self.marks.get(symbol, 0.)
                        self.fill(day, symbol, 'BUY', buy, bars)
                        bought_value = self.quantity(symbol) * self.marks.get(symbol, 0.) - before
                        budget_left -= bought_value * (1 + cap * self.cost)
                        if self.quantity(symbol):
                            held.add(symbol)
        self.mark(day, bars, 'close')
        closing_nav = self.nav()
        closing_risk = self.risk.observe(closing_nav, closing=True)
        for event in self.by_record.get(day, []):
            qty = self.quantity(event.symbol)
            self.entitlements[event.key] = qty
            if qty and (event.ambiguous or event.ex_date is None):
                self.issue('ambiguous_or_missing_ex_date_held', day, event.symbol)
        self.nav_rows.append({'date': day, 'nav': closing_nav, 'cash': self.cash,
            'cash_receivable': sum(item['value'] for item in self.cash_claims),
            'market_value': closing_nav - self.cash - sum(item['value'] for item in self.cash_claims),
            'position_count': len(set(self.positions) | {item['symbol'] for item in self.locked_shares}),
            'drawdown': closing_risk['drawdown'], 'month_to_date': closing_risk['month_to_date'],
            'drawdown_amount': closing_risk['drawdown_amount'], 'month_loss_amount': closing_risk['month_loss_amount'],
            'opening_breach': opening_risk['breached'], 'closing_breach': closing_risk['breached'],
            'guard_halted': self.risk.halted if self.guarded else False,
            'cumulative_scenario_cost': self.total_cost})

    def result(self, train_end):
        frame = pd.DataFrame(self.nav_rows).set_index('date')
        metrics = performance(frame['nav'], self.config['initial_paper_capital'])
        train = frame.loc[frame.index <= pd.Timestamp(train_end), 'nav']
        validation = frame.loc[frame.index > pd.Timestamp(train_end), 'nav']
        splits = {'train': performance(train, self.config['initial_paper_capital']),
                  'validation': performance(validation, float(train.iloc[-1]))}
        splits['validation']['drawdown_from_lifetime_peak'] = float(frame.loc[validation.index, 'drawdown'].min())
        splits['validation']['drawdown_amount_from_lifetime_peak'] = float(frame.loc[validation.index, 'drawdown_amount'].max())
        informational = {'unknown_cash_payment_locked', 'unknown_bonus_listing_locked', 'documented_suspension_valuation_carry',
                         'adjustment_change_no_distribution_recorded'}
        critical = {key: value for key, value in self.issues.items() if key not in informational}
        passed = (metrics['max_drawdown'] >= -self.config['peak_drawdown_limit'] - 1e-10
            and metrics['worst_daily_month_to_date'] >= -self.config['monthly_loss_limit'] - 1e-10
            and metrics['max_drawdown_amount'] <= self.config.get('absolute_peak_loss_cap', float('inf')) + 1e-6
            and metrics['max_daily_month_loss_amount'] <= self.config.get('absolute_monthly_loss_cap', float('inf')) + 1e-6
            and not frame['opening_breach'].any() and not critical)
        halt_dates = frame.index[frame['guard_halted']]
        return {'name': self.name, 'metrics': metrics, 'splits': splits, 'trades': len(self.trades),
            'one_way_cost': self.cost, 'zero_cost_diagnostic': self.cost == 0,
            'first_trade_date': self.trades[0]['date'] if self.trades else None,
            'last_trade_date': self.trades[-1]['date'] if self.trades else None,
            'guard_halt_date': halt_dates[0].date().isoformat() if len(halt_dates) else None,
            'mean_gross_exposure': float((frame['market_value'] / frame['nav']).mean()),
            'maximum_positions': int(frame['position_count'].max()),
            'scenario_cost_paid': self.total_cost, 'blocked': dict(self.blocked), 'accounting_issues': dict(self.issues),
            'accounting_valid': not critical, 'passes_observed_research_risk_limits': bool(passed),
            'qualifies_research_candidate': bool(passed and self.cost > 0 and metrics['total_return'] > 0
                                                and splits['validation']['geometric_monthly_return'] > 0),
            'live_or_statutory_cost_validation': False, 'holdout_evaluated': False}
