#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
quant-agent UAT 部署脚本（参照 jcs-agent-be/deploy/build_kit.py 模式，同机变体）

目标：D:\uat_software\quant_agent —— 每日批处理系统（非常驻服务），
计划任务触发 paper_day（14:45）与 watchdog（16:00），工作日。

与 build_kit 的差异：同机部署 → 直接建原生 venv（无需 embeddable），
WindPy 经 .pth 挂载复制即可。

安全设计（沿用 build_kit）：
  - 状态目录（live/ journal/ results/ runtime/）**重部署永不覆盖**
    （首次部署时携带，延续 paper 验证期状态）；
  - 代码目录 MIR 前检测目标无 python 进程占用（计划任务运行中拒绝部署）；
  - 依赖清单从当前 venv freeze（deploy/.cache 跨次复用）。

用法（项目根目录）：
  .venv/Scripts/python.exe scripts/deploy_uat.py             # 完整部署
  ... --skip-cache    # 跳过 data_cache（目标已有时）
  ... --skip-venv     # 仅更新代码（依赖无变化时，秒级）
  ... --force         # 目标有 python 进程时强制（慎用）
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==================== 配置区 ====================
DEPLOY_DIR = Path(r'D:\uat_software\quant_agent')
# 状态目录：重部署保留（账本/日报/工件/运行时）；首次部署携带
STATE_DIRS = ['live', 'journal', 'results', 'runtime']
# 代码镜像排除项（MIR 生效域=排除状态目录后的全部）
EXCLUDE = ['.git', '.venv', 'data_cache', '__pycache__', '.pytest_cache',
           'ai-created', '*.pyc', '.idea']
# ===============================================

CACHE = PROJECT_ROOT / 'deploy' / '.cache'


def run(cmd, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def check_no_target_python(force: bool):
    """目标目录有 python 进程运行（计划任务正在执行）→ 拒绝部署。"""
    try:
        out = subprocess.check_output(
            ['wmic', 'process', 'where', "name='python.exe'", 'get',
             'ExecutablePath'], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return
    hits = [ln.strip() for ln in out.splitlines()
            if ln.strip() and str(DEPLOY_DIR).lower() in ln.lower()]
    if hits:
        if not force:
            raise SystemExit(
                f"⛔ 目标目录有 python 进程运行（计划任务执行中？）——\n"
                f"   先停任务再部署，或 --force：\n   " + "\n   ".join(hits))
        print("  ⚠️ --force：目标进程仍在运行，继续（风险自担）")


def export_requirements() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    req = CACHE / 'requirements.freeze.txt'
    r = run([sys.executable, '-m', 'pip', 'freeze', '--all'],
            capture_output=True, text=True)
    req.write_text(r.stdout, encoding='utf-8')
    print(f"  依赖清单: {req}（{len(r.stdout.splitlines())} 项）")
    return req


def sync_code():
    """代码 MIR（排除状态目录与垃圾）；robocopy 退出码 <8 视为成功。"""
    print("\n[2/5] 同步代码 …")
    xf = [f'/XF'] if False else []
    cmd = ['robocopy', str(PROJECT_ROOT), str(DEPLOY_DIR), '/MIR',
           '/R:2', '/W:2', '/NFL', '/NDL', '/NP']
    for e in EXCLUDE:
        cmd += ['/XD', e] if not any(c in e for c in '*?') else ['/XF', e]
    for d in STATE_DIRS:
        cmd += ['/XD', d]
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode >= 8:
        raise SystemExit(f"⛔ robocopy 失败 rc={r.returncode}")
    print(f"  代码同步完成（robocopy rc={r.returncode}）")


def first_deploy_state():
    """首次部署：携带状态目录（延续 paper 验证期）。重部署跳过。"""
    print("\n[3/5] 状态目录 …")
    for d in STATE_DIRS:
        src, dst = PROJECT_ROOT / d, DEPLOY_DIR / d
        if dst.exists() and any(dst.iterdir()):
            print(f"  {d}/: 目标已有 → 保留不动（状态延续）")
            continue
        if not src.exists():
            continue
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
        print(f"  {d}/: 首次携带 ✓")


def build_venv(req: Path):
    print("\n[4/5] 目标 venv …")
    venv = DEPLOY_DIR / '.venv'
    tp = venv / 'Scripts' / 'python.exe'
    if tp.exists():
        print(f"  已存在（如需重建先删 {venv}）")
    else:
        r = run([sys.executable, '-m', 'venv', str(venv)])
        if r.returncode != 0:
            raise SystemExit("⛔ venv 创建失败")
    r = run([tp, '-m', 'pip', 'install', '-q', '-r', str(req)])
    if r.returncode != 0:
        raise SystemExit("⛔ 依赖安装失败")
    # WindPy：.pth 挂载（非 pip 包，同机指向 Wind 终端 site-packages）
    for pth in (PROJECT_ROOT / '.venv' / 'Lib' / 'site-packages').glob('WindPy*.pth'):
        shutil.copy2(pth, venv / 'Lib' / 'site-packages' / pth.name)
        print(f"  WindPy 挂载复制: {pth.name}")
    print("  venv 就绪")


def sync_cache():
    print("\n[5/5] data_cache（增量，首拷 ~1.1G） …")
    r = subprocess.run(['robocopy', str(PROJECT_ROOT / 'data_cache'),
                        str(DEPLOY_DIR / 'data_cache'), '/E', '/XO',
                        '/R:1', '/W:1', '/NFL', '/NDL', '/NP'])
    print(f"  缓存同步完成（rc={r.returncode}）")


def write_launchers():
    """计划任务启动器（cd 兜底 + 绝对路径 python）。"""
    py = r'%~dp0.venv\Scripts\python.exe'
    for name, script in [('paper_day.bat', 'scripts\\paper_day.py'),
                         ('paper_watchdog.bat', 'scripts\\paper_watchdog.py')]:
        (DEPLOY_DIR / name).write_text(
            f'@echo off\r\ncd /d %~dp0\r\n{py} {script}\r\n',
            encoding='ascii')
    (DEPLOY_DIR / 'START.txt').write_text(
        f"""quant_agent UAT 部署说明
========================
本目录由 scripts/deploy_uat.py 生成（源: {PROJECT_ROOT}）

计划任务注册（管理员 cmd，一次性）：
  schtasks /Create /TN "QuantPaperDay" /TR "{DEPLOY_DIR}\\paper_day.bat" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 14:45 /F
  schtasks /Create /TN "QuantPaperWatchdog" /TR "{DEPLOY_DIR}\\paper_watchdog.bat" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 16:00 /F

注意：
  - 这是每日批处理，非常驻服务：无需"重启"，14:45 自动跑当日会话后退出
  - 状态在 live/ journal/ runtime/（重部署脚本永不覆盖）
  - .env 已随包（含数据库凭据，勿外传）
  - 更新代码：源目录跑 .venv/Scripts/python.exe scripts/deploy_uat.py --skip-venv --skip-cache
""", encoding='utf-8')
    print("  启动器: paper_day.bat / paper_watchdog.bat / START.txt")


def smoke():
    print("\n[冒烟] 目标环境导入检查 …")
    tp = DEPLOY_DIR / '.venv' / 'Scripts' / 'python.exe'
    r = run([tp, '-c',
             'import sys; sys.path.insert(0, r"%s"); '
             'import live, trading, data, portfolio, monitoring; '
             'from trading.etf_strategy import build_etf_signal; '
             'print("imports OK")' % DEPLOY_DIR],
            capture_output=True, text=True, cwd=str(DEPLOY_DIR))
    print(' ', (r.stdout or r.stderr).strip().splitlines()[-1])
    if 'OK' not in (r.stdout or ''):
        raise SystemExit("⛔ 目标环境导入失败——检查依赖/WindPy")


def main():
    args = sys.argv[1:]
    force = '--force' in args
    print(f"部署目标: {DEPLOY_DIR}\n")
    print("[1/5] 前置检查 …")
    check_no_target_python(force)
    req = export_requirements()
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    sync_code()
    first_deploy_state()
    if '--skip-venv' not in args:
        build_venv(req)
    if '--skip-cache' not in args:
        sync_cache()
    write_launchers()
    if '--skip-venv' not in args:
        smoke()
    print(f"""
✅ 部署完成 → {DEPLOY_DIR}
   注册计划任务（管理员 cmd，见 START.txt）：
     schtasks /Create /TN "QuantPaperDay" /TR "{DEPLOY_DIR}\\paper_day.bat" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 14:45 /F
     schtasks /Create /TN "QuantPaperWatchdog" /TR "{DEPLOY_DIR}\\paper_watchdog.bat" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 16:00 /F""")


if __name__ == '__main__':
    main()
