#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quant Strategy System - Unified Web Application.

Usage:
    python -m webapp.app
    Open http://localhost:5000
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pickle
import tempfile
from utils.state_file import exclusive_file, replace_json
from datetime import date, datetime

from flask import Flask, render_template, jsonify, g, has_request_context

from research import cta_signal

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# ── Strategy Registry ──
STATE_FILE = Path('runtime/strategy_state/config.json')

DEFAULT_CONFIG = {
    'g1': {'enabled': True, 'name': 'G1 A股选股', 'type': '股票·多头',
           'universe': 'A股 Top50', 'sharpe': '—', 'ann_ret': '+6.8%', 'max_dd': '-7.1%',
           'current_status': '研究闭环，无自动入口', 'detail_url': '/strategy/g1'},
    'z8': {'enabled': True, 'name': 'CTA Z10 等权活跃', 'type': '商品期货·多空',
           'universe': '6品种·活跃等权', 'sharpe': '0.86', 'ann_ret': '+13.4%',
           'max_dd': '≈-24.5%', 'current_status': '（按最新信号展示）',
           'detail_url': '/strategy/z8'},
}

def load_config():
    """配置=默认展示信息 + 用户设置（仅 enabled 开关）。

    审计 2026-11-19 二-6：此前整份配置（含名称/指标等展示性字段）原样
    持久化——升级部署后老用户仍看旧名称与旧指标。现在状态文件只保存
    用户设置，展示字段一律取当前代码里的 DEFAULT_CONFIG。"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    if STATE_FILE.exists():
        try:
            user = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        except Exception:
            user = {}
        for sid, settings in (user.items() if isinstance(user, dict) else []):
            if sid in merged and isinstance(settings, dict) and isinstance(settings.get('enabled'), bool):
                merged[sid]['enabled'] = bool(settings.get('enabled',
                                                           merged[sid]['enabled']))
    for sid, relative in (('g1', 'g1_holdout_pool_v3'), ('z8', 'z10_units_v1')):
        try:
            evidence = json.loads((PROJECT_ROOT / 'ai-created/research' / relative / 'verdict.json').read_text(encoding='utf8'))
            annual = (1 + evidence['monthly']) ** 12 - 1 if sid == 'g1' else evidence['annual_return']
            merged[sid].update(ann_ret=f'{annual:+.1%}', max_dd=f"{evidence['max_dd']:.1%}")
            if sid == 'z8':
                merged[sid]['sharpe'] = f"{evidence['sharpe']:.2f}"
        except (OSError, KeyError, ValueError):
            merged[sid].update(ann_ret='—', max_dd='—', sharpe='—')
    return merged

def save_config(cfg):
    """只持久化用户设置（enabled 开关）；展示字段不落盘。"""
    slim = {sid: {'enabled': bool(v.get('enabled', False))}
            for sid, v in cfg.items() if sid in DEFAULT_CONFIG}
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    replace_json(STATE_FILE, slim)

# ── Z8 Data Provider ──
SYMS = {
    'AU0': ('黄金', 'AU'), 'CU0': ('沪铜', 'CU'), 'RB0': ('螺纹钢', 'RB'),
    'I0': ('铁矿石', 'I'), 'P0': ('棕榈油', 'P'), 'HC0': ('热卷', 'HC'),
}
CACHE = Path('data_cache/cta_z8_cache')

REQUIRED_COLS = ('开盘价', '最高价', '最低价', '收盘价')


def _frames_valid(data) -> bool:
    if not isinstance(data, dict) or set(data) != set(SYMS):
        return False
    for df in data.values():
        if (not isinstance(df, pd.DataFrame) or len(df) < cta_signal.MIN_BARS
                or not set(REQUIRED_COLS).issubset(df.columns)
                or not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans
                or not df.index.is_monotonic_increasing or not df.index.is_unique):
            return False
        prices = df[list(REQUIRED_COLS)].apply(pd.to_numeric, errors='coerce')
        if (not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()
                or (prices['最高价'] < prices.max(axis=1)).any()
                or (prices['最低价'] > prices.min(axis=1)).any()):
            return False
    return True


def load_z8_data(force: bool = False):
    """拉取/缓存六品种日线。

    审计 2026-11-19 二-4：①手动刷新不再先删除旧缓存——先拉取、通过
    完整性验收后原子替换；失败时保留旧快照，绝不让"刷新失败=丢失最后
    一份可用数据"。②写入前验收（六品种/非空/字段/日期序），残缺结果
    不落盘。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / 'daily_data.pkl'
    import time as t
    previous = {}
    if has_request_context():
        g.refresh_status = 'unavailable'
    if cf.exists():
        try:
            with cf.open('rb') as stream:
                candidate = pickle.load(stream)
            if _frames_valid(candidate):
                previous = candidate
                if has_request_context():
                    g.refresh_status = 'kept_previous'
        except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError):
            pass
    if not force and previous and t.time() - cf.stat().st_mtime < 3600 * 4:
        return previous
    try:
        import akshare as ak
    except ImportError:
        return previous
    import time as tm
    data = {}
    for sym in SYMS:
        try:
            df = ak.futures_main_sina(symbol=sym, start_date='20150101', end_date=date.today().strftime('%Y%m%d'))
            df['日期'] = pd.to_datetime(df['日期'])
            data[sym] = df.set_index('日期').sort_index()
            tm.sleep(0.2)
        except Exception:
            pass
    if _frames_valid(data):
        # 原子替换：先写临时文件再 os.replace，读方永远看到完整快照
        expected = _expected_last_closed()
        latest = [df.index[df.index.date <= expected].max() for df in data.values()]
        if any(pd.isna(day) or day.date() < expected for day in latest):
            return previous or data
        with exclusive_file(cf):
            fd, tmp = tempfile.mkstemp(dir=CACHE, suffix='.tmp')
            try:
                with os.fdopen(fd, 'wb') as stream:
                    pickle.dump(data, stream)
                os.replace(tmp, cf)
            finally:
                Path(tmp).unlink(missing_ok=True)
        if has_request_context():
            g.refresh_status = 'ok'
        return data
    return previous


def _z8_frames():
    """六品种信号框架（research.cta_signal 单一实现，每日跟随语义）。

    审计 2026-11-19 二-4：同一 HTTP 请求内复用一份快照（flask.g 缓存）——
    此前首页 z8_status/z8_meta 各自加载并重算，仓位与质量提示可能来自
    不同快照。"""
    from flask import g
    cached = getattr(g, '_z8_frames_cache', None)
    if cached is not None:
        return cached
    data = load_z8_data()
    frames = {}
    for sym in SYMS:
        if sym not in data:
            continue
        df = data[sym]
        frames[sym] = cta_signal.signal_frame(
            df['收盘价'].astype(float), open_=df['开盘价'].astype(float))
    g._z8_frames_cache = (data, frames)
    return data, frames


def z8_meta():
    """数据覆盖与新鲜度（覆盖不完整/停更时页面必须提示，不得静默降级）。"""
    data, frames = _z8_frames()
    expected = _expected_last_closed()
    frames = cta_signal.truncate_to_closed(frames, expected)
    cov = cta_signal.coverage_check(frames, expected=list(SYMS),
                                    as_of=pd.Timestamp.today())
    last = max((f.index[-1] for f in frames.values() if len(f) > 0),
               default=None)
    lagging = (last is not None and last.date() < expected)
    cf = CACHE / 'daily_data.pkl'
    import time as t
    cache_age_h = (t.time() - cf.stat().st_mtime) / 3600 if cf.exists() else None
    return {'ok': cov['ok'], 'missing': cov['missing'],
            'n_symbols': len(frames),
            'last_date': last.date().isoformat() if last is not None else '—',
            'expected_date': expected.isoformat(),
            'stale': bool(cov.get('stale')) or lagging,
            'cache_age_h': round(cache_age_h, 1) if cache_age_h is not None else None}


def z8_status():
    data, frames = _z8_frames()
    # 盘中防护与日报一致：只用最近已收盘交易日
    expected = _expected_last_closed()
    frames = cta_signal.truncate_to_closed(frames, expected)
    ranked = cta_signal.rank_active(frames)
    book = cta_signal.target_book(frames, k=cta_signal.DEPLOY_K)
    n_book = len(book)
    result = []
    for sym, (name, code) in SYMS.items():
        if sym not in frames or len(frames[sym]) == 0:
            continue  # 缺失/空表品种由 meta 提示，这里不渲染（防iloc越界）
        row = frames[sym].iloc[-1]
        vol_ann = float(row['vol20']) * np.sqrt(252)
        compressed = bool(row['compressed'])
        d = float(row['direction'])
        direction = '多' if d > 0 else ('空' if d < 0 else '—')  # 收平线=无方向
        result.append({
            'sym': sym, 'name': name, 'code': code,
            'price': round(float(row['close']), 1),
            'vol20': round(vol_ann, 3) if np.isfinite(vol_ann) else 0,
            'vol_pct': round(float(row['vol_pct']), 4),
            'ma20': round(float(row['ma']), 1) if np.isfinite(row['ma']) else 0,
            'compressed': compressed,
            'direction': direction if compressed else '—',
            # 持仓读统一目标组合：方向为零的压缩品种不持仓，也不会出现 1/0
            'position': (f'持仓 {direction} 1/{n_book}') if sym in book else '空仓',
            'rank': ranked.index(sym) + 1 if sym in ranked else None,
            'distance': round(float(row['vol_pct'] - 0.20), 4),
        })
    return result


def _expected_last_closed():
    """与日报一致的最近已收盘交易日（工作日近似，15:30 为收盘线）。"""
    now = pd.Timestamp.now()
    d = now.normalize()
    if now.hour < 15 or (now.hour == 15 and now.minute < 30):
        d = d - pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d = d - pd.Timedelta(days=1)
    return d.date()

def _z8_data_cached():
    """同一 HTTP 请求内复用同一份行情快照（审计 2026-11-19 二-4：
    详情页汇总按品种重复加载、且与首页可能取到不同快照）。"""
    return _z8_frames()[0]


def z8_chart(sym):
    data = _z8_data_cached()
    if sym not in data:
        return {'candles': [], 'ma20': [], 'volPct': [], 'trades': []}
    df = data[sym].loc[data[sym].index.date <= _expected_last_closed()]
    frame = cta_signal.signal_frame(df['收盘价'].astype(float),
                                    open_=df['开盘价'].astype(float))

    candles, ma_line, vol_pcts = [], [], []
    for i in range(len(df)):
        d = df.index[i]
        ts = int(d.timestamp())
        candles.append({'time': ts, 'open': float(df.loc[d, '开盘价']),
                        'high': float(df.loc[d, '最高价']),
                        'low': float(df.loc[d, '最低价']),
                        'close': float(df.loc[d, '收盘价'])})
        if np.isfinite(frame['ma'].iloc[i]):
            ma_line.append({'time': ts, 'value': round(float(frame['ma'].iloc[i]), 2)})
        if np.isfinite(frame['vol_pct'].iloc[i]):
            vol_pcts.append({'time': ts,
                             'value': round(float(frame['vol_pct'].iloc[i]), 3)})

    # 成交口径：每日跟随语义 + 次日开盘成交（登记册180，禁止同bar归因）
    trades = []
    for t in cta_signal.trades_from_frame(frame, sym):
        trades.append({
            'entryDate': t['entryDate'],
            'entryPrice': round(t['entryPrice'], 1),
            'exitDate': t['exitDate'],
            'exitPrice': round(t['exitPrice'], 1) if t['exitPrice'] is not None else None,
            'direction': t['direction'],
            'netRet': t['netRet'],
            'holdDays': t['holdDays'],
            'pending': t['pending'],
        })

    return {'candles': candles, 'ma20': ma_line, 'volPct': vol_pcts, 'trades': trades}

def z8_trade_summary():
    summaries = []
    for sym, (name, _) in SYMS.items():
        td = z8_chart(sym)['trades']
        closed = [t for t in td if not t['pending']]
        pending = sum(1 for t in td if t['pending'])
        if not closed and not pending:
            continue
        rets = [t['netRet'] for t in closed]
        summaries.append({
            'name': name, 'n': len(rets), 'pending': pending,
            'wr': sum(1 for r in rets if r > 0) / len(rets) if rets else 0,
            'avg': sum(rets) / len(rets) if rets else 0,
            'best': max(rets) if rets else 0,
            'worst': min(rets) if rets else 0,
            'total': float(np.prod(1 + np.array(rets)) - 1) if rets else 0})
    return summaries


# ── Routes ──

@app.route('/')
def index():
    cfg = load_config()
    if cfg.get('z8', {}).get('enabled'):
        status, meta = z8_status(), z8_meta()
        cfg = {k: dict(v) for k, v in cfg.items()}
        # 二轮复审#14：覆盖不合格/停更时组合目标=无法判定，不得显示
        # "目标持仓 0 品种"（会被误读为清仓信号）
        if not meta or not meta.get('ok') or meta.get('stale'):
            cfg['z8']['current_status'] = '无法判定（数据缺失或停更，先刷新）'
        else:
            n_book = sum(1 for s in (status or []) if s['position'].startswith('持仓'))
            cfg['z8']['current_status'] = f'目标持仓 {n_book} 品种（次日开盘生效）'
    else:
        status, meta = None, None
    return render_template('index.html', page='home', strategies=cfg,
                           z8_status=status, z8_meta=meta)


@app.route('/system')
def system_health():
    import platform
    issues = []

    # 1) Workspace status
    ws = {'mode': 'unconfigured', 'active_strategies': [],
          'automatic_trading': False, 'live_execution': False,
          'cache_exists': Path('data_cache').exists(),
          'archive_exists': Path('archive').exists()}
    if not ws['cache_exists']:
        issues.append('数据缓存目录不存在')
    if not ws['archive_exists']:
        issues.append('归档备份目录不存在')
    if ws['automatic_trading']:
        issues.append('自动交易处于开启状态（确认是否有意）')
    if ws['live_execution']:
        issues.append('实盘执行处于开启状态')

    # 2) Archive verification
    archive_result = None
    try:
        archive_result = _verify_archive()
        if not archive_result:
            issues.append('归档校验未完成')
        if archive_result and not archive_result.get('source_verified', False):
            issues.append(f"归档源码校验失败: {len(archive_result.get('source_failures', []))} 个文件异常")
        if archive_result and not archive_result.get('retained_files_unchanged', True):
            issues.append(f"受保护文件有变更: {len(archive_result.get('retained_file_changes', []))} 个")
    except Exception:
        issues.append('归档校验未完成')
        archive_result = None

    # 3) Data summary
    data_summary = None
    try:
        data_summary = _data_summary()
        if not data_summary or not data_summary.get('files'):
            issues.append('数据目录为空或无法读取')
    except Exception:
        issues.append('数据目录统计失败')

    # 4) Environment
    env = {
        'python_version': platform.python_version(),
        'deps': f"pandas={_pkg_ver('pandas')}, numpy={_pkg_ver('numpy')}",
        'check_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    return render_template('system_health.html', page='system',
                           strategies=load_config(),
                           issues=issues, workspace=ws,
                           archive=archive_result,
                           data_summary=data_summary, env=env)


def _pkg_ver(pkg):
    try:
        return __import__(pkg).__version__
    except Exception:
        return '?'


def _verify_archive():
    """归档核验——与 CLI（application.archive）同一实现。

    审计 2026-11-19 二-7：此前网页自行实现且在**项目根目录**解析清单
    路径——归档完好时也报 304 个源码中的 217 个损坏；受保护文件的检查
    口径也与 CLI 不同（缺 mtime_ns 与登记册计数语义）。"""
    try:
        from application.archive import verify_snapshot
        from application.workspace import Workspace
        return verify_snapshot(Workspace.current())
    except Exception:
        return None


def _data_summary():
    cache = Path('data_cache')
    if not cache.exists():
        return None
    from collections import defaultdict
    cats = defaultdict(lambda: {'files': 0, 'bytes': 0})
    total_files, total_bytes = 0, 0
    for f in cache.rglob('*'):
        if f.is_file():
            cat = f.parent.relative_to(cache).parts[0] if f.parent != cache else 'root'
            size = f.stat().st_size
            cats[cat]['files'] += 1
            cats[cat]['bytes'] += size
            total_files += 1
            total_bytes += size

    def human(b):
        for u in ['B', 'KB', 'MB', 'GB']:
            if b < 1024:
                return f'{b:.1f}{u}'
            b /= 1024
        return f'{b:.1f}TB'

    return {
        'files': total_files, 'bytes': total_bytes,
        'size_human': human(total_bytes),
        'categories': {k: {**v, 'size_human': human(v['bytes'])} for k, v in sorted(cats.items())},
    }


@app.route('/strategy/g1')
def strategy_g1():
    cfg = load_config()
    return render_template('strategy_g1.html', page='strategy_g1',
                           strategies=cfg,
                           enabled=cfg.get('g1', {}).get('enabled', False))


@app.route('/strategy/z8')
def strategy_z8():
    cfg = load_config()
    status = z8_status()
    # 审计 2026-11-19 二-3：汇总按**目标组合**计数（此前按 compressed 计数，
    # 六品种均压缩但方向为零时会显示"6 品种各 1/6"而目标组合实际为空）
    actives = [s for s in status if s['position'].startswith('持仓')]
    n_active = len(actives)
    summary = z8_trade_summary()
    return render_template('strategy_z8.html', page='strategy_z8',
                           strategies=cfg,
                           enabled=cfg.get('z8', {}).get('enabled', False),
                           status=status, n_active=n_active, actives=actives,
                           meta=z8_meta(),
                           trade_summary=summary)


@app.route('/chart/<sym>')
def chart_detail(sym):
    if sym not in SYMS:
        return 'Not found', 404
    name, code = SYMS[sym]
    data = z8_chart(sym)
    return render_template('chart_detail.html', page='chart_z8',
                           strategies=load_config(),
                           sym=sym, name=name, code=code,
                           trades=data['trades'])


@app.route('/api/strategy/<sid>/toggle', methods=['POST'])
def toggle_strategy(sid):
    with exclusive_file(STATE_FILE):
        cfg = load_config()
        if sid in cfg:
            cfg[sid]['enabled'] = not cfg[sid]['enabled']
            save_config(cfg)
            return jsonify({'enabled': cfg[sid]['enabled']})
    return jsonify({'error': 'not found'}), 404


@app.route('/api/chart/<sym>')
def api_chart(sym):
    return jsonify(z8_chart(sym))


@app.route('/api/refresh')
def api_refresh():
    """手动刷新：同步拉取并通过完整性验收后**原子替换**缓存。

    审计 2026-11-19 二-4 + 二轮复审#31：失败时保留旧快照并**如实**返回
    kept_previous——此前回退旧缓存也算 ok，用户无从辨别刷新失败。"""
    load_z8_data(force=True)
    status = getattr(g, 'refresh_status', 'unavailable')
    return jsonify(status=status, previous_cache_kept=status == 'kept_previous')


def _pid_alive_python(pid: int) -> bool:
    """记录的 PID 是否仍是存活的 python 进程（防重复启动/误判）。"""
    import subprocess
    try:
        out = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV'],
            capture_output=True, text=True, timeout=10).stdout
        return 'python' in out.lower()
    except Exception:
        return True


def _record_alive(record: dict) -> bool:
    """PID 记录是否仍指向**同一个** webapp 实例。

    审计 2026-11-19 四-f：此前启动侧只查"PID 对应 python"——PID 被复用
    成别的 python 进程时会误判存活（或误放行第二实例）。与 stop.bat 同
    口径：pid + python + 进程创建时间与记录 start_unix 匹配（±2 秒）。
    身份不完整（旧格式记录/查询失败）按保守处理=视为存活，人工清理。"""
    try:
        pid = int((record or {}).get('pid', 0) or 0)
    except (TypeError, ValueError):
        return False
    if not pid:
        return False
    if not _pid_alive_python(pid):
        return False
    rec_start = (record or {}).get('start_unix')
    if rec_start is None:
        return True      # 旧记录无创建时间：身份不完整，保守视为存活
    try:
        import ctypes
        import ctypes.wintypes as wt
        k32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return True  # 打不开：保守视为存活
        try:
            creation, exit_, kernel, user = (wt.FILETIME() for _ in range(4))
            ok = k32.GetProcessTimes(handle, ctypes.byref(creation),
                                     ctypes.byref(exit_), ctypes.byref(kernel),
                                     ctypes.byref(user))
            if not ok:
                return True
            raw = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            start_unix = raw / 1e7 - 11644473600
            return abs(start_unix - float(rec_start)) <= 2
        finally:
            k32.CloseHandle(handle)
    except Exception:
        return True


if __name__ == '__main__':
    import atexit

    PID_FILE = Path('runtime/webapp.pid')
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 防重复启动：已有存活记录则拒绝启动，绝不覆盖有效记录
    # （否则第二实例覆盖 PID 文件后启动失败退出，会把第一实例的停止记录一起删掉）
    with exclusive_file(PID_FILE):
        if PID_FILE.exists():
            try:
                import json as _json
                old = _json.loads(PID_FILE.read_text(encoding='utf-8'))
            except Exception:
                try:
                    old = {'pid': int(PID_FILE.read_text(encoding='utf-8').strip())}
                except ValueError:
                    old = {}
            if _record_alive(old):
                print(f"已有一个 webapp 实例在运行（PID {old.get('pid')}，"
                      f"创建时间已比对）。")
                print('如确认它已失效，请先删除 runtime/webapp.pid 再启动。')
                sys.exit(1)
            # 记录过期（进程不存在/非同一进程）→ 清理后正常启动
            PID_FILE.unlink(missing_ok=True)

        # 记录 PID + 真实进程创建时间(unix秒) + 完整入口路径：stop.bat 据此
        # 逐项比对，关闭"PID 被复用"的误杀窗口（复用进程的创建时间 ≠ 记录值）
        import json as _json

        def _proc_start_unix():
            try:
                import ctypes
                import ctypes.wintypes as wt
                k32 = ctypes.windll.kernel32
                creation, exit_, kernel, user = (wt.FILETIME() for _ in range(4))
                if k32.GetProcessTimes(k32.GetCurrentProcess(),
                                       ctypes.byref(creation), ctypes.byref(exit_),
                                       ctypes.byref(kernel), ctypes.byref(user)):
                    raw = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                    return raw / 1e7 - 11644473600  # FILETIME → unix 秒
            except Exception:
                pass
            return None

        _start = _proc_start_unix()
        _entry = str(Path(__file__).resolve())
        replace_json(PID_FILE, {
            'pid': os.getpid(),
            'start_unix': _start,
            'entry': _entry,
        })

    def _cleanup_pid():
        """退出时解析 JSON 记录，确认属于当前实例后才清理。

        （旧实现拿整个 JSON 文本与纯 PID 字符串比较，永远不相等，
        记录残留后若 PID 被复用会导致下次启动误报已有实例。）
        """
        try:
            with exclusive_file(PID_FILE):
                if not PID_FILE.exists():
                    return
                rec = _json.loads(PID_FILE.read_text(encoding='utf-8'))
                if isinstance(rec, dict) and int(rec.get('pid', -1)) == os.getpid():
                    PID_FILE.unlink()
        except (OSError, ValueError):
            # 记录损坏时不删——宁可残留由启动逻辑清理，也不误删他人记录
            pass

    atexit.register(_cleanup_pid)
    print('=' * 50)
    print('  Quant Strategy System')
    print('  http://localhost:5000')
    print(f'  PID {os.getpid()} -> {PID_FILE}')
    print('=' * 50)
    app.run(host='127.0.0.1', port=5000, debug=False)
