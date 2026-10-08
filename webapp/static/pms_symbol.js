// 个股钻取：Wind风格日K（双击某天看分时）
let SYM_DATA = null, SYM_CHART = null;

async function showSymbol(sym) {
  document.getElementById('symModal').style.display = 'block';
  document.getElementById('symTitle').textContent = sym + ' · 日K（双击某天看分时）';
  const r = await fetch('/pms/api/symbol/' + sym);
  const d = await r.json();
  if (d.error) { document.getElementById('symTitle').textContent = sym + ' · ' + d.error; return; }
  SYM_DATA = d;
  renderDailyK(sym);
  document.getElementById('symTrades').innerHTML = '<tr><th class="l">时间</th><th class="l">方向</th>' +
    '<th>价格</th><th>数量</th><th>佣金</th><th>税费</th><th class="l">交易逻辑</th></tr>' +
    (d.trades.map(t => '<tr><td class="l">' + (t.time || t.date).replace('T', ' ') + '</td>' +
      '<td class="l"><span class="badge" style="color:' + (t.side === 'BUY' ? '#f87171' : '#34d399') + '">' +
      (t.side === 'BUY' ? '买入' : '卖出') + '</span></td>' +
      '<td>' + t.price.toFixed(2) + '</td><td>' + fmt(t.quantity) + '</td>' +
      '<td>' + t.commission.toFixed(2) + '</td><td>' + t.taxes.toFixed(2) + '</td>' +
      '<td class="l" style="color:#94a3b8">' + (t.note || '建仓买入（5因子Top30入选）') + '</td></tr>').join('') ||
      '<tr><td class="empty">无交易记录</td></tr>');
}

function renderDailyK(sym) {
  const d = SYM_DATA, k = d.kline;
  document.getElementById('kInfo').innerHTML = '<b style="color:#fff">' + sym +
    '</b>　<span style="color:#94a3b8">MA5/10/20/30 · 双击K线某天 → 当日分时</span>';
  if (!SYM_CHART) {
    SYM_CHART = echarts.init(document.getElementById('kChart'));
    window.addEventListener('resize', () => SYM_CHART && SYM_CHART.resize());
  } else { SYM_CHART.clear(); }
  const buys = d.trades.filter(t => t.side === 'BUY'), sells = d.trades.filter(t => t.side === 'SELL');
  const mk = (arr, label, color) => arr.map(t => {
    const i = k.dates.indexOf(t.date);
    return i < 0 ? null : { coord: [i, k.ohlc[i] ? k.ohlc[i][3] : 0], symbol: 'triangle', symbolSize: 11,
      itemStyle: { color: color }, label: { show: true, formatter: label, position: 'top', color: color } };
  }).filter(Boolean);
  const maColors = { ma5: '#fbbf24', ma10: '#60a5fa', ma20: '#c084fc', ma30: '#34d399' };
  SYM_CHART.setOption({
    backgroundColor: '#0b0f1a', animation: false,
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross', label: { backgroundColor: '#1e293b' } },
      backgroundColor: '#111827', borderColor: '#263143', textStyle: { color: '#e5e7eb', fontSize: 12 } },
    legend: { data: ['MA5', 'MA10', 'MA20', 'MA30'], textStyle: { color: '#94a3b8', fontSize: 11 }, top: 2 },
    grid: [{ left: 10, right: 70, top: 26, height: 250, containLabel: true },
           { left: 10, right: 70, top: 292, height: 70, containLabel: true }],
    xAxis: [{ type: 'category', data: k.dates, axisLine: { lineStyle: { color: '#263143' } }, axisLabel: { color: '#64748b' } },
            { type: 'category', gridIndex: 1, data: k.dates, axisLabel: { show: false }, axisLine: { lineStyle: { color: '#263143' } } }],
    yAxis: [{ scale: true, position: 'right', axisLabel: { color: '#64748b' },
              splitLine: { lineStyle: { color: '#151b2b', type: 'dashed' } } },
            { gridIndex: 1, axisLabel: { show: false }, splitLine: { show: false } }],
    dataZoom: [{ type: 'inside', xAxisIndex: [0, 1], start: 55, end: 100 },
               { type: 'slider', top: 372, height: 18, xAxisIndex: [0, 1], borderColor: '#263143',
                 backgroundColor: '#111827', fillerColor: 'rgba(96,165,250,.15)', textStyle: { color: '#64748b' } }],
    series: [
      { name: sym, type: 'candlestick', data: k.ohlc, xAxisIndex: 0, yAxisIndex: 0,
        itemStyle: { color: '#ef4444', color0: '#22c55e', borderColor: '#ef4444', borderColor0: '#22c55e' },
        markPoint: { data: mk(buys, '买', '#f59e0b').concat(mk(sells, '卖', '#38bdf8')) },
        markLine: d.position.cost > 0 ? { symbol: 'none', lineStyle: { color: '#f59e0b', type: 'dashed' }, silent: true,
          label: { color: '#f59e0b', position: 'insideStartTop', formatter: '成本 ' + d.position.cost.toFixed(2) },
          data: [{ yAxis: d.position.cost }] } : undefined },
      { name: 'MA5', type: 'line', data: k.ma5, showSymbol: false, lineStyle: { width: 1, color: maColors.ma5 } },
      { name: 'MA10', type: 'line', data: k.ma10, showSymbol: false, lineStyle: { width: 1, color: maColors.ma10 } },
      { name: 'MA20', type: 'line', data: k.ma20, showSymbol: false, lineStyle: { width: 1, color: maColors.ma20 } },
      { name: 'MA30', type: 'line', data: k.ma30, showSymbol: false, lineStyle: { width: 1, color: maColors.ma30 } },
      { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1,
        data: k.volume.map((v, i) => ({ value: v,
          itemStyle: { color: (k.ohlc[i] && k.ohlc[i][1] >= k.ohlc[i][0]) ? '#ef4444' : '#22c55e', opacity: 0.7 } })) }]
  });
  SYM_CHART.off('dblclick');
  SYM_CHART.on('dblclick', e => {
    if (e.seriesType === 'candlestick' && k.dates[e.dataIndex]) showIntraday(sym, k.dates[e.dataIndex]);
  });
}

async function showIntraday(sym, day) {
  document.getElementById('symTitle').textContent = sym + ' · ' + day + ' 分时';
  document.getElementById('kInfo').innerHTML = '<span style="color:#38bdf8;cursor:pointer" onclick="renderDailyK(\'' + sym + '\')">← 返回日K</span>　<b style="color:#fff">' + sym + ' ' + day + '</b>';
  const r = await fetch('/pms/api/intraday/' + sym + '/' + day);
  const d = await r.json();
  if (d.error) { document.getElementById('kInfo').innerHTML += '　<span style="color:#f87171">' + d.error + '</span>'; return; }
  const pc = d.prev_close;
  SYM_CHART.clear();
  const lo = Math.min.apply(null, d.price.concat(pc != null ? [pc] : [Infinity]));
  const hi = Math.max.apply(null, d.price.concat(pc != null ? [pc] : [-Infinity]));
  const pad = (hi - lo) * 0.1 || 0.1;
  SYM_CHART.setOption({
    backgroundColor: '#0b0f1a', animation: false,
    tooltip: { trigger: 'axis', backgroundColor: '#111827', borderColor: '#263143', textStyle: { color: '#e5e7eb' } },
    grid: [{ left: 10, right: 70, top: 26, height: 250, containLabel: true },
           { left: 10, right: 70, top: 292, height: 70, containLabel: true }],
    xAxis: [{ type: 'category', data: d.times, axisLabel: { color: '#64748b' }, axisLine: { lineStyle: { color: '#263143' } } },
            { type: 'category', gridIndex: 1, data: d.times, axisLabel: { show: false }, axisLine: { lineStyle: { color: '#263143' } } }],
    yAxis: [{ min: lo - pad, max: hi + pad, position: 'right',
              axisLabel: { color: '#64748b', formatter: v => pc != null ? ((v / pc - 1) * 100).toFixed(1) + '%' : v },
              splitLine: { lineStyle: { color: '#151b2b', type: 'dashed' } } },
            { gridIndex: 1, axisLabel: { show: false }, splitLine: { show: false } }],
    series: [
      { name: '价格', type: 'line', data: d.price, showSymbol: false, lineStyle: { width: 1.4, color: '#f8fafc' },
        areaStyle: { color: 'rgba(148,163,184,.12)' },
        markLine: pc != null ? { symbol: 'none', silent: true, lineStyle: { color: '#64748b', type: 'dashed' },
          label: { show: false }, data: [{ yAxis: pc }] } : undefined },
      { name: '均价', type: 'line', data: d.avg, showSymbol: false, lineStyle: { width: 1.2, color: '#facc15' } },
      { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1,
        data: d.volume.map((v, i) => ({ value: v,
          itemStyle: { color: (pc == null || d.price[i] >= pc) ? '#ef4444' : '#22c55e', opacity: 0.7 } })) }]
  });
}
