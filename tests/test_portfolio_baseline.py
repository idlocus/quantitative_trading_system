import numpy as np
import pandas as pd

from research.portfolio_metrics import DailyRisk, performance
from research.portfolio_simulation import Distribution, PortfolioSimulation, distributions
from research.portfolio_study import selection_lists


def config():
    return {'initial_paper_capital': 10000., 'monthly_loss_limit': .15, 'peak_drawdown_limit': .15,
        'exposure_budget_adverse_shock': .25, 'gross_target': .95, 'positions': 1,
        'rebalance_sessions': 5, 'volume_participation_cap': .01, 'cash_dividend_haircut': .2}


def bar(price=10., lower=9., upper=11.):
    return {'open': price, 'close': price, 'volume': 100000., 'source_upper_limit': upper,
            'source_lower_limit': lower, 'adjfactor': 1., 'numeric_bar_valid': True, 'source_suspended': False}


def test_month_boundary_keeps_previous_close_and_lifetime_peak():
    risk = DailyRisk(100.)
    risk.begin_day(pd.Timestamp('2020-01-31'))
    risk.observe(120., closing=True)
    risk.begin_day(pd.Timestamp('2020-02-03'))
    result = risk.observe(100.)
    assert np.isclose(result['month_to_date'], -1 / 6)
    assert risk.peak == 120.
    assert risk.halted
    risk.observe(100., closing=True)
    risk.begin_day(pd.Timestamp('2020-03-02'))
    assert risk.peak == 120.
    assert risk.exposure_cap(100., .95) == 0.


def test_metrics_include_initial_loss_and_intramonth_loss():
    nav = pd.Series([90., 100., 80., 95.], index=pd.to_datetime(['2020-01-02', '2020-01-31', '2020-02-03', '2020-02-28']))
    result = performance(nav, 100.)
    assert np.isclose(result['max_drawdown'], -.2)
    assert np.isclose(result['worst_daily_month_to_date'], -.2)
    assert np.isclose(result['worst_calendar_month'], -.05)
    assert np.isclose(result['geometric_monthly_return'], .95 ** .5 - 1)


def test_fixed_amount_cap_does_not_expand_when_account_grows():
    risk = DailyRisk(1000000., monthly_amount_limit=150000., peak_amount_limit=150000.)
    risk.begin_day(pd.Timestamp('2020-01-02'))
    risk.observe(2000000., closing=True)
    risk.begin_day(pd.Timestamp('2020-02-03'))
    result = risk.observe(1800000.)
    assert np.isclose(result['drawdown'], -.1)
    assert result['drawdown_amount'] == 200000.
    assert result['breached']


def test_cash_and_costs_are_self_financing_and_no_same_day_round_trip():
    dates = pd.bdate_range('2020-01-02', periods=3)
    simulation = PortfolioSimulation('test', config(), .0025, False, [], dates, {})
    simulation.step(0, dates[0], {'600000.SH': bar()}, [], set())
    simulation.step(1, dates[1], {'600000.SH': bar()}, ['600000.SH'], {'600000.SH'})
    assert simulation.positions == {'600000.SH': 900}
    assert len(simulation.trades) == 1
    simulation.step(2, dates[2], {'600000.SH': bar()}, [], set())
    assert simulation.positions == {}
    assert simulation.cash == 9955.
    assert simulation.total_cost == 45.


def test_blocked_limit_exit_does_not_erase_breach_or_restart_peak():
    dates = pd.to_datetime(['2020-01-31', '2020-02-03'])
    settings = {**config(), 'initial_paper_capital': 1000.}
    simulation = PortfolioSimulation('test', settings, 0., True, [], dates, {})
    simulation.cash = 0.
    simulation.positions = {'600000.SH': 100}
    simulation.marks = {'600000.SH': 10.}
    simulation.step(0, dates[0], {'600000.SH': bar(8., lower=8.)}, [], set())
    assert simulation.positions['600000.SH'] == 100
    assert simulation.risk.halted
    assert simulation.nav_rows[-1]['closing_breach']
    simulation.step(1, dates[1], {'600000.SH': bar(7.8, lower=7.)}, ['600000.SH'], {'600000.SH'})
    assert not simulation.positions
    assert simulation.risk.peak == 1000.
    assert all(trade['side'] == 'SELL' for trade in simulation.trades)


def test_dividend_right_survives_sale_and_is_not_spendable_at_ex_date():
    dates = pd.bdate_range('2020-01-02', periods=5)
    event = Distribution('event', '600000.SH', dates[0], dates[1], dates[2], None, 1., 0.)
    simulation = PortfolioSimulation('test', config(), 0., False, [event], dates, {})
    simulation.positions = {'600000.SH': 100}
    simulation.cash = 0.
    simulation.marks = {'600000.SH': 10.}
    simulation.targets = ['600000.SH']
    simulation.last_rebalance = 0
    simulation.step(0, dates[0], {'600000.SH': bar()}, [], {'600000.SH'})
    simulation.step(1, dates[1], {'600000.SH': bar(9., lower=8.)}, [], set())
    assert not simulation.positions
    assert simulation.cash == 900.
    assert simulation.cash_claims[0]['value'] == 80.
    assert simulation.nav() == 980.
    simulation.apply_events(dates[2])
    assert simulation.cash == 900.
    simulation.apply_events(dates[3])
    assert simulation.cash == 980.


def test_bonus_shares_remain_locked_until_listing():
    dates = pd.bdate_range('2020-01-02', periods=5)
    event = Distribution('event', '600000.SH', dates[0], dates[1], None, dates[3], 0., 1.)
    simulation = PortfolioSimulation('test', config(), 0., False, [event], dates, {})
    simulation.positions = {'600000.SH': 100}
    simulation.marks = {'600000.SH': 10.}
    simulation.cash = 0.
    simulation.entitlements['event'] = 100
    simulation.apply_events(dates[1])
    assert simulation.positions['600000.SH'] == 100
    assert simulation.quantity('600000.SH') == 200
    assert simulation.nav() == 1000.
    simulation.apply_events(dates[3])
    assert simulation.positions['600000.SH'] == 200


def test_missing_quote_is_flagged_and_delisting_is_not_a_fictitious_sale():
    dates = pd.bdate_range('2020-01-02', periods=3)
    simulation = PortfolioSimulation('test', config(), 0., False, [], dates, {'600000.SH': dates[2]})
    simulation.positions = {'600000.SH': 100}
    simulation.marks = {'600000.SH': 10.}
    simulation.mark(dates[1], {}, 'close')
    assert simulation.issues['unexplained_held_valuation_gap'] == 1
    before = simulation.cash
    simulation.mark(dates[2], {}, 'close')
    assert simulation.cash == before
    assert not simulation.trades
    assert simulation.issues['unmodeled_delisting_zero_recovery'] == 1


def test_documented_suspension_carries_valuation_without_inventing_a_fill():
    dates = pd.bdate_range('2020-01-02', periods=2)
    simulation = PortfolioSimulation('test', config(), 0., False, [], dates, {})
    simulation.positions = {'600000.SH': 100}
    simulation.marks = {'600000.SH': 10.}
    bars = {'600000.SH': {'numeric_bar_valid': False, 'source_suspended': True}}
    simulation.mark(dates[1], bars, 'close')
    simulation.fill(dates[1], '600000.SH', 'SELL', 100, bars)
    assert not simulation.trades
    assert simulation.positions['600000.SH'] == 100
    assert simulation.issues['documented_suspension_valuation_carry'] == 1
    assert simulation.issues['unexplained_held_valuation_gap'] == 0


def test_selection_uses_only_each_signal_day_and_eligible_names():
    score = pd.DataFrame([[1., 2.], [100., -100.]], columns=['first', 'second'])
    eligible = pd.DataFrame([[True, False], [True, True]], columns=score.columns)
    assert selection_lists(score, eligible, 1) == [['first'], ['first']]


def test_distribution_deduplication_and_same_day_merge():
    """2026-11-09 语义更新：同 (symbol,登记日,除权日) 金额不同的多笔分红
    = 同日到账的真实多次分配 → 合并金额（实证 601088：2.51+0.46=2.97 与
    除权参考价分毫不差）；歧义仅留给缺登记日/顺序无效的真病态。"""
    row = {'OBJECT_ID': 'one', 'S_DIV_PROGRESS': '3', 'S_INFO_WINDCODE': '600000.SH',
        'EQY_RECORD_DT': '20200102', 'EX_DT': '20200103', 'DVD_PAYOUT_DT': '20200106',
        'LISTING_DT_OF_DVD_SHR': None, 'CASH_DVD_PER_SH_PRE_TAX': 1., 'STK_DVD_PER_SH': 0.}
    source = pd.DataFrame([row, {**row, 'OBJECT_ID': 'copy'}])
    actions, quality = distributions(source, {'600000.SH'}, pd.bdate_range('2020-01-02', periods=5))
    assert len(actions) == 1
    assert quality['identical_rows_deduplicated'] == 1
    # 同日两笔不同金额：合并为一笔、金额相加、不再标歧义
    source = pd.DataFrame([row, {**row, 'OBJECT_ID': 'extra', 'CASH_DVD_PER_SH_PRE_TAX': 2.}])
    actions, quality = distributions(source, {'600000.SH'}, pd.bdate_range('2020-01-02', periods=5))
    assert len(actions) == 1 and not actions[0].ambiguous
    assert abs(actions[0].cash - 3.0) < 1e-9
    assert quality['same_day_distributions_merged'] == 1


def test_same_ex_date_different_payout_dates_stay_separate():
    """审计 2026-11-19 一-8：同登记/除权日但支付日不同的两笔分红不得
    合并到第一个支付日——现金按各自到期日分别可用。"""
    row = {'OBJECT_ID': 'one', 'S_DIV_PROGRESS': '3', 'S_INFO_WINDCODE': '600000.SH',
        'EQY_RECORD_DT': '20200102', 'EX_DT': '20200103', 'DVD_PAYOUT_DT': '20200106',
        'LISTING_DT_OF_DVD_SHR': None, 'CASH_DVD_PER_SH_PRE_TAX': 1., 'STK_DVD_PER_SH': 0.}
    source = pd.DataFrame([row, {**row, 'OBJECT_ID': 'two',
                                 'DVD_PAYOUT_DT': '20200108',
                                 'CASH_DVD_PER_SH_PRE_TAX': 2.}])
    actions, quality = distributions(source, {'600000.SH'}, pd.bdate_range('2020-01-02', periods=5))
    assert len(actions) == 2, '不同支付日的分配必须保持独立事件'
    payouts = sorted(str(a.payout.date()) for a in actions)
    assert payouts == ['2020-01-06', '2020-01-08']
    assert all(not a.ambiguous for a in actions)
    assert quality['same_day_distributions_merged'] == 0


def test_separate_payout_claims_settle_on_their_own_dates():
    """两笔分红各自到期：日3 只有第一笔可支配，第二笔仍在途。"""
    dates = pd.bdate_range('2020-01-02', periods=6)
    event1 = Distribution('e1', '600000.SH', dates[0], dates[1], dates[3], None, 1., 0.)
    event2 = Distribution('e2', '600000.SH', dates[0], dates[1], dates[4], None, 2., 0.)
    simulation = PortfolioSimulation('t', config(), 0., False, [event1, event2], dates, {})
    simulation.positions = {'600000.SH': 100}
    simulation.cash = 0.
    simulation.marks = {'600000.SH': 10.}
    simulation.targets = ['600000.SH']
    simulation.last_rebalance = 0
    simulation.step(0, dates[0], {'600000.SH': bar()}, [], {'600000.SH'})
    simulation.apply_events(dates[1])          # 除权日：两笔各自挂账
    assert len(simulation.cash_claims) == 2
    simulation.apply_events(dates[3])          # 支付日当日尚未可支配（次日可用）
    assert simulation.cash == 0.
    simulation.apply_events(dates[4])          # 第一笔（pay=dates[3]）可用
    assert simulation.cash == 80.
    assert len(simulation.cash_claims) == 1, '第二笔未到期仍在途'
    simulation.apply_events(dates[5])          # 第二笔（pay=dates[4]）可用
    assert simulation.cash == 240. and not simulation.cash_claims


class TestOrderInputCausality:
    """审计 2026-11-19 一-3：所有下单输入（权重/热卖/日度覆盖）按决策日
    键传入、内部平移到下一交易日生效——T 日开盘订单不得看到 T 日收盘。"""

    @staticmethod
    def _run(dates, syms, **kw):
        sim = PortfolioSimulation('t', {**config(), 'positions': len(syms),
                                        'rebalance_sessions': 1}, 0., False,
                                  [], dates, {}, **kw)
        for i, day in enumerate(dates):
            ranking = [] if i == 0 else syms
            sim.step(i, day, {s: bar() for s in syms}, ranking, set(syms))
        return sim

    def test_day_t_close_weights_do_not_change_day_t_orders(self):
        dates = pd.bdate_range('2020-01-02', periods=4)
        syms = ['A.SH', 'B.SH']
        base = {dates[1]: {'A.SH': 1., 'B.SH': 3.},
                dates[2]: {'A.SH': 1., 'B.SH': 3.}}
        variant = {dates[1]: {'A.SH': 1., 'B.SH': 3.},
                   dates[2]: {'A.SH': 9., 'B.SH': 1.}}   # 只改决策日2
        t_base = self._run(dates, syms, daily_weights=base).trades
        t_var = self._run(dates, syms, daily_weights=variant).trades
        day2 = str(dates[2].date())
        assert [t for t in t_base if t['date'] == day2] == \
               [t for t in t_var if t['date'] == day2], \
            'T 日开盘订单必须与 T 日收盘数据无关'
        day3 = str(dates[3].date())
        q_base = {t['symbol']: t['quantity'] for t in t_base if t['date'] == day3}
        q_var = {t['symbol']: t['quantity'] for t in t_var if t['date'] == day3}
        assert q_var and q_base != q_var, '决策日2 的不同权重应在生效日3体现'

    def test_hot_exit_decided_day_t_executes_day_t_plus_1(self):
        dates = pd.bdate_range('2020-01-02', periods=4)
        sim = self._run(dates, ['A.SH', 'B.SH'], hold_buffer_rank=4,
                        hot_exit_sets={dates[1]: {'A.SH'}})
        trades = [(t['date'], t['symbol'], t['side']) for t in sim.trades]
        day1, day2 = str(dates[1].date()), str(dates[2].date())
        assert (day1, 'A.SH', 'BUY') in trades, '决策日1尚未触发热卖，日1正常买入'
        assert (day2, 'A.SH', 'SELL') in trades, '决策日1的热卖名单在日2生效'
        # 热卖剔除的票当日不得从新增名单回补（I 系漏洞：卖出与回补同日）
        assert (day2, 'A.SH', 'BUY') not in trades

    def test_daily_position_override_lagged(self):
        dates = pd.bdate_range('2020-01-02', periods=3)
        syms = ['A.SH', 'B.SH', 'C.SH']
        sim = self._run(dates, syms, daily_position_overrides={dates[1]: 3})
        # 决策日1 N=3 → 生效日2 持仓3只（等权分母见 X 专测）
        assert len(sim.positions) == 3


class TestHoldBufferAnchorsActualHoldings:
    """审计 2026-11-19 一-5：缓冲带锚定真实持仓而非昨日目标。"""

    @staticmethod
    def _sim(dates, positions=2, buffer=100):
        return PortfolioSimulation(
            't', {**config(), 'positions': positions,
                  'rebalance_sessions': 1}, 0., False, [], dates, {},
            hold_buffer_rank=buffer)

    def test_failed_buy_is_not_treated_as_old_position(self):
        """昨日涨停未买到、今日跌到第55名：不算老仓（窄带前50外不买）。"""
        dates = pd.bdate_range('2020-01-02', periods=3)
        many = [f'S{i:03d}.SH' for i in range(60)]
        bars_ok = {s: bar() for s in many}
        bars_limit = dict(bars_ok)
        bars_limit['S000.SH'] = bar(11., lower=11., upper=11.)  # 一字涨停
        sim = self._sim(dates)
        sim.step(0, dates[0], bars_limit, [], set(many))
        sim.step(1, dates[1], bars_limit, ['S000.SH', 'S001.SH'], set(many))
        assert 'S000.SH' not in sim.positions and 'S001.SH' in sim.positions
        # 日2：S000 排名第55（宽带100内、窄带50外）
        rank2 = many[2:55] + ['S000.SH'] + ['S001.Sh'.replace('Sh', 'SH')] + many[55:57]
        rank2 = many[2:55] + ['S000.SH', 'S001.SH'] + many[55:57]
        sim.step(2, dates[2], bars_ok, rank2, set(many))
        assert 'S000.SH' not in sim.positions, \
            '昨日未买到的票跌出窄带后不得按老仓逻辑回补'

    def test_blocked_sell_keeps_buffer_protection(self):
        """昨日跌停卖出受阻的真实持仓，今日回到宽带内 → 必须保留。"""
        dates = pd.bdate_range('2020-01-02', periods=4)
        many = [f'S{i:03d}.SH' for i in range(60)]
        bars_ok = {s: bar() for s in many}
        bars_down = dict(bars_ok)
        bars_down['S000.SH'] = bar(9., lower=9., upper=9.)     # 开盘跌停不可卖
        sim = self._sim(dates)
        sim.step(0, dates[0], bars_ok, [], set(many))
        sim.step(1, dates[1], bars_ok, many[:2], set(many))
        assert set(sim.positions) == {'S000.SH', 'S001.SH'}
        # 日2：S000 跌出宽带（第59名，宽带=2+100 仍在内？——构造排到59名）
        rank2 = many[2:59] + ['S000.SH', 'S001.SH']
        sim.step(2, dates[2], bars_down, rank2, set(many))
        assert 'S000.SH' in sim.positions, '跌停卖出受阻的真实持仓仍在场'
        # 日3：S000 回到宽带内第55名 → 按真实持仓保留（旧口径按昨日
        # 目标已剔除该票，不再享受缓冲）
        rank3 = many[2:55] + ['S000.SH', 'S001.SH'] + many[55:57]
        sim.step(3, dates[3], bars_ok, rank3, set(many))
        assert 'S000.SH' in sim.positions, \
            '受阻未卖的真实持仓在宽带内必须享受缓冲保护'


class TestCandidateDepthAndDenominator:
    """审计 2026-11-19 一-4：候选深度必须覆盖最大 K/N；等权分母=今日目标数。"""

    def test_k100_vs_k200_behave_differently(self):
        """已持仓排第149：K100 卖出、K200 保留（修复前三份净值完全相同）。"""
        dates = pd.bdate_range('2020-01-02', periods=3)
        many = [f'S{i:03d}.SH' for i in range(200)]
        bars = {s: bar() for s in many}

        def run(k):
            sim = PortfolioSimulation(
                't', {**config(), 'positions': 2,
                      'rebalance_sessions': 1}, 0., False, [], dates, {},
                hold_buffer_rank=k)
            sim.step(0, dates[0], bars, [], set(many))
            sim.step(1, dates[1], bars, many[:2], set(many))
            rank2 = many[2:149] + many[:2]      # 已持仓跌到第148-149名
            sim.step(2, dates[2], bars, rank2, set(many))
            return sim

        assert 'S000.SH' not in run(100).positions, 'K100：排149 → 卖出'
        assert 'S000.SH' in run(200).positions, 'K200：排149 在宽带内 → 保留'

    def test_x_dynamic_n_uses_n_as_denominator(self):
        """X系：日度 N=3 生效日等权分母=3（旧口径恒为 config positions=2，
        N 只按 1/2 配置造成结构性超配/欠配）。"""
        dates = pd.bdate_range('2020-01-02', periods=3)
        syms = ['A.SH', 'B.SH', 'C.SH']
        sim = PortfolioSimulation(
            't', {**config(), 'initial_paper_capital': 12000., 'positions': 2,
                  'rebalance_sessions': 1},
            0., False, [], dates, {},
            daily_position_overrides={dates[1]: 3})
        for i, day in enumerate(dates):
            ranking = [] if i == 0 else syms
            sim.step(i, day, {s: bar() for s in syms}, ranking, set(syms))
        # 生效日2：三分母 → 目标各 3800/10=380 股，整手后 400/400/300
        # （近等权，极差 ≤ 一个整手）；旧口径二分母 → 500/500/200（极差300）
        assert set(sim.positions) == set(syms)
        assert max(sim.positions.values()) - min(sim.positions.values()) <= 100
        assert sim.positions['C.SH'] == 300, '第三只按三分母足额买入，不被现金挤压'


class TestRoundTwoPositionControl:
    """二轮复审 #2/#3：卖出受阻后按实际持仓重算买入预算；候选不足时
    等权分母=策略规定 N；有效空榜清空目标。"""

    def test_blocked_sell_caps_total_exposure(self, tmp_path):
        """原持仓跌停卖不出：新仓买入后总仓位不得越过风险上限。"""
        dates = pd.bdate_range('2020-01-02', periods=3)
        syms = ['A.SH', 'B.SH', 'C.SH', 'D.SH']
        bars = {s: bar() for s in syms}
        sim = PortfolioSimulation(
            't', {**config(), 'initial_paper_capital': 12000., 'positions': 2,
                  'rebalance_sessions': 1}, .0025, False, [], dates, {})
        sim.step(0, dates[0], bars, [], set(syms))
        sim.step(1, dates[1], bars, ['A.SH', 'B.SH'], set(syms))
        assert set(sim.positions) == {'A.SH', 'B.SH'}
        nav_open = sim.nav()
        # 日2：目标换成 C/D；A 一字跌停卖不出，B 正常卖出
        bars_down = dict(bars)
        bars_down['A.SH'] = bar(9., lower=9., upper=9.)
        sim.step(2, dates[2], bars_down, ['C.SH', 'D.SH'], set(syms))
        held = sum(q * sim.marks.get(s, 0.) for s, q in sim.positions.items())
        assert held <= 0.95 * nav_open * 1.001, \
            '卖出受阻后总仓位必须压回风险上限内（而非用回笼现金买满）'
        assert 'A.SH' in sim.positions and ('C.SH' in sim.positions or 'D.SH' in sim.positions)

    def test_candidate_shortfall_keeps_weight_at_1_over_n(self, tmp_path):
        """50 只目标只剩 1 只合格：单票权重=1/50（分母=规定 N 而非剩余数）。"""
        dates = pd.bdate_range('2020-01-02', periods=2)
        syms = [f'S{i:02d}.SH' for i in range(3)]
        bars = {s: {**bar(), 'volume': 1_000_000.} for s in syms}
        sim = PortfolioSimulation(
            't', {**config(), 'initial_paper_capital': 1_000_000., 'positions': 50,
                  'rebalance_sessions': 1}, 0., False, [], dates, {})
        sim.step(0, dates[0], bars, [], set(syms))
        sim.step(1, dates[1], bars, syms, {'S00.SH'})   # 仅 S00 合格
        # 权重=1/50：0.95×1e6×(1/50)=19000 → 1900 股（旧口径 1/1 → 95000 股）
        assert sim.positions == {'S00.SH': 1900}

    def test_empty_ranking_clears_targets(self, tmp_path):
        """有效空榜必须清空目标并卖出（不得沿用旧目标）。"""
        dates = pd.bdate_range('2020-01-02', periods=3)
        bars = {'A.SH': bar(), 'B.SH': bar()}
        sim = PortfolioSimulation(
            't', {**config(), 'positions': 1, 'rebalance_sessions': 1},
            0., False, [], dates, {})
        sim.step(0, dates[0], bars, [], set(bars))
        sim.step(1, dates[1], bars, ['A.SH'], set(bars))
        assert sim.positions == {'A.SH': 900}   # 9500/10=950 → 整手 900
        sim.step(2, dates[2], bars, [], set(bars))      # 空榜
        assert sim.positions == {}, '空榜日必须清仓'
