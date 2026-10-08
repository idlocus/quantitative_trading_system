// 个股钻取：Wind风格日K（双击某天看分时）
let SYM_DATA = null, SYM_CHART = null, SYM_LOGIC = '';

async function showSymbol(sym) {
  document.getElementById('symModal').style.display = 'block';
  document.getElementById('symTitle').textContent = sym + ' · 日K（双击某天看分时）';
  const r = await fetch('/pms/api/symbol/' + sym);
  const d = await r.json();
  if (d.error) { document.getElementById('symTitle').textContent = sym + ' · ' + d.error; return; }
  SYM_DATA = d;
  renderDailyK(sym);
  renderDailyK(sym);
  SYM_LOGIC = '';
  renderSymTrades(d.trades);
  // 推荐逻辑卡片（异步加载，不阻塞K线）
    '<th>价格</th><th>数量</th><th>佣金</th><th>税费</th><th class="l">交易逻辑</th></tr>' +
    (d.trades.map(t => '<tr><td class="l">' + (t.time || t.date).replace('T', ' ') + '</td>' +
      '<td class="l"><span class="badge" style="color:' + (t.side === 'BUY' ? '#f87171' : '#34d399') + '">' +
      (t.side === 'BUY' ? '买入' : '卖出') + '</span></td>' +
      '<td>' + t.price.toFixed(2) + '</td><td>' + fmt(t.quantity) + '</td>' +
      '<td>' + t.commission.toFixed(2) + '</td><td>' + t.taxes.toFixed(2) + '</td>' +
      '<td class="l" style="color:#94a3b8">' + (t.note || '建仓买入（5因子Top30入选）') + '</td></tr>').join('') ||
      '<tr><td class="empty">无交易记录</td></tr>');
  // 推荐逻辑卡片（异步加载，不阻塞K线）
  const raBox = document.getElementById('symRationale');
  raBox.style.display = 'block';
  raBox.innerHTML = '<span style="color:#64748b;font-size:13px">推荐逻辑计算中…（首次约30-40秒）</span>';
  fetch('/pms/api/rationale/' + sym).then(rr => rr.json()).then(ra => { fillRationale(ra); })
    .catch(() => { raBox.style.display = 'none'; });
}

function fillRationale(ra) {
  const raBox = document.getElementById('symRationale');
  if (ra && ra.points && ra.points.length) {
    SYM_LOGIC = ra.points.map(p => p.replace(/\*\*/g, '')).join('；').slice(0, 160);
    if (SYM_DATA && SYM_DATA.trades) renderSymTrades(SYM_DATA.trades);
  }
  if (ra && ra.points && ra.points.length) {
    let html = '<h4 style="color:#cbd5e1;font-size:13px;margin:0 0 8px">推荐逻辑（' + ra.day + ' 买入口径）'
      + (ra.score != null ? '　综合评分 <b style="color:#f59e0b">' + (ra.score * 100).toFixed(0) + '</b>/100' : '')
      + (ra.market_total ? '　全市场约 ' + ra.market_total + ' 只' : '') + '</h4><ul style="margin:0 0 8px 18px;padding:0">';
    ra.points.forEach(p => {
      html += '<li style="font-size:13px;color:#cbd5e1;line-height:1.8;margin:2px 0">'
        + p.replace(/\*\*(.+?)\*\*/g, '<b style="color:#93c5fd">$1</b>') + '</li>';
    });
    html += '</ul><details style="margin-top:6px"><summary style="font-size:12px;color:#94a3b8;cursor:pointer">风险提示</summary><ul style="margin:6px 0 0 18px">';
    (ra.risks || []).forEach(rk => { html += '<li style="font-size:12px;color:#f59e0b;line-height:1.7">' + rk + '</li>'; });
    html += '</ul></details>';
    raBox.innerHTML = html;
    raBox.style.display = 'block';
  } else { raBox.innerHTML = '<span style="color:#64748b;font-size:13px">暂无推荐逻辑数据</span>'; }
}

function renderSymTrades(trades) {
  document.getElementById('symTrades').innerHTML = '<tr><th class="l">时间</th><th class="l">方向</th>' +
    '<th>价格</th><th>数量</th><th>佣金</th><th>税费</th><th class="l">交易逻辑</th></tr>' +
    (trades.map(t => '<tr><td class="l">' + (t.time || t.date).replace('T', ' ') + '</td>' +
      '<td class="l"><span class="badge" style="color:' + (t.side === 'BUY' ? '#f87171' : '#34d399') + '">' +
      (t.side === 'BUY' ? '买入' : '卖出') + '</span></td>' +
      '<td>' + t.price.toFixed(2) + '</td><td>' + fmt(t.quantity) + '</td>' +
      '<td>' + t.commission.toFixed(2) + '</td><td>' + t.taxes.toFixed(2) + '</td>' +
      '<td class="l" style="color:#cbd5e1;max-width:360px">' + (t.note || SYM_LOGIC || '建仓买入（5因子Top30入选）') + '</td></tr>').join('') ||
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
  // 双击K线区域任意位置 → 最近交易日的分时（不要求精确点中蜡烛体）
  SYM_CHART.getZr().off('dblclick');
  SYM_CHART.getZr().on('dblclick', e => {
    const pt = [e.offsetX, e.offsetY];
    try {
      if (!SYM_CHART.containPixel({ seriesIndex: 0 }, pt)) return;
      const idx = Math.round(SYM_CHART.convertFromPixel({ seriesIndex: 0 }, pt)[0]);
      if (idx >= 0 && idx < k.dates.length) showIntraday(sym, k.dates[idx]);
    } catch (err) { /* 转换失败忽略 */ }
  });
}

async function showIntraday(sym, day) {
  document.getElementById('symTitle').textContent = sym + ' · ' + day + ' 分时';
  document.getElementById('kInfo').innerHTML = '<span style="color:#38bdf8;cursor:pointer" onclick="renderDailyK(\'' + sym + '\')">← 返回日K</span>　<b style="color:#fff">' + sym + ' ' + day + '</b>';
  const r = await fetch('/pms/api/intraday/' + sym + '/' + day);
  const d = await r.json();
  if (d.error) {
    document.getElementById('symTitle').textContent = sym + ' · ' + d.error;
    document.getElementById('kInfo').innerHTML += '　<span style="color:#f87171">' + d.error + '</span>';
    return;
  }
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
