#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股智能选股器 V2 - Flask调度版

每天早上7点运行，发送企业微信消息
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from datetime import datetime

from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# 配置日志 - 输出到文件而不是stdout
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scheduler.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        # logging.StreamHandler(sys.stderr)  # 注释掉避免后台运行时窗口闪烁
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def run_screening_job():
    """执行每日统一任务"""
    import threading

    def job_in_thread():
        """在后台线程中执行每日统一任务"""
        import subprocess
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(base_dir, 'daily_unified_job.py')

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=base_dir,
                timeout=1800  # 30分钟超时
            )
            if result.returncode == 0:
                logger.info("每日统一任务执行完成")
            else:
                logger.error(f"每日统一任务执行失败，返回码: {result.returncode}")
        except subprocess.TimeoutExpired:
            logger.error("每日统一任务执行超时（30分钟）")

    # 启动后台线程执行任务
    thread = threading.Thread(target=job_in_thread, daemon=True)
    thread.start()
    logger.info("每日统一任务已在后台启动")


def run_market_summary_job():
    """执行每日市场总结任务"""
    import threading

    def job_in_thread():
        """在后台线程中执行市场总结任务"""
        import subprocess
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(base_dir, 'daily_market_summary.py')

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=base_dir,
                timeout=600  # 10分钟超时
            )
            if result.returncode == 0:
                logger.info("市场总结任务执行完成")
            else:
                logger.error(f"市场总结任务执行失败，返回码: {result.returncode}")
        except subprocess.TimeoutExpired:
            logger.error("市场总结任务执行超时（10分钟）")

    # 启动后台线程执行任务
    thread = threading.Thread(target=job_in_thread, daemon=True)
    thread.start()
    logger.info("市场总结任务已在后台启动")


def run_dl_prediction_job():
    """执行深度学习预测任务"""
    import threading

    def job_in_thread():
        """在后台线程中执行深度学习预测"""
        import subprocess
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(base_dir, 'run_dl_prediction.py')

        # 如果预测脚本不存在，跳过
        if not os.path.exists(script_path):
            logger.warning(f"深度学习预测脚本不存在: {script_path}")
            return

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=base_dir,
                timeout=600  # 10分钟超时
            )
            if result.returncode == 0:
                logger.info("深度学习预测任务执行完成")
            else:
                logger.error(f"深度学习预测任务执行失败，返回码: {result.returncode}")
        except subprocess.TimeoutExpired:
            logger.error("深度学习预测任务执行超时（10分钟）")

    # 启动后台线程执行任务
    thread = threading.Thread(target=job_in_thread, daemon=True)
    thread.start()
    logger.info("深度学习预测任务已在后台启动")


def init_scheduler():
    """初始化调度器"""
    scheduler = BackgroundScheduler()

    # 创建定时触发器：每天早上7点
    trigger = CronTrigger(
        hour=7,
        minute=0,
        timezone='Asia/Shanghai'
    )

    scheduler.add_job(
        run_screening_job,
        trigger=trigger,
        id='daily_unified_job',
        name='每日投资建议',
        replace_existing=True
    )

    # 每天下午4点执行市场总结
    trigger_summary = CronTrigger(
        hour=16,
        minute=0,
        timezone='Asia/Shanghai'
    )
    scheduler.add_job(
        run_market_summary_job,
        trigger=trigger_summary,
        id='daily_market_summary',
        name='每日市场总结',
        replace_existing=True
    )

    # 每天下午4点30分执行深度学习预测
    trigger_dl = CronTrigger(
        hour=16,
        minute=30,
        timezone='Asia/Shanghai'
    )
    scheduler.add_job(
        run_dl_prediction_job,
        trigger=trigger_dl,
        id='dl_prediction',
        name='深度学习信号预测',
        replace_existing=True
    )

    scheduler.start()
    logger.info("调度器已启动")
    logger.info("  - 每天早上7点: 每日投资建议")
    logger.info("  - 每天下午16:00: 市场总结")
    logger.info("  - 每天下午16:30: 深度学习预测")
    return scheduler


@app.route('/')
def index():
    """首页"""
    return {
        'name': 'A股投资调度器',
        'status': 'running',
        'schedule': [
            {'time': '每天07:00', 'job': 'daily_unified_job', 'name': '每日投资建议', 'features': ['市场状态分析', '大盘方向预测', '个股推荐']},
            {'time': '每天16:00', 'job': 'daily_market_summary', 'name': '每日市场总结', 'features': ['今日行情', '市场宽度', '重大消息', '操作建议']}
        ],
        'next_run': '请查看调度器日志'
    }


@app.route('/run')
def manual_run():
    """手动触发每日投资建议任务"""
    run_screening_job()
    return {'status': '每日投资建议任务已触发', 'time': datetime.now().isoformat()}


@app.route('/summary')
def manual_summary():
    """手动触发市场总结任务"""
    run_market_summary_job()
    return {'status': '市场总结任务已触发', 'time': datetime.now().isoformat()}


@app.route('/dl-predict')
def manual_dl_predict():
    """手动触发深度学习预测任务"""
    run_dl_prediction_job()
    return {'status': '深度学习预测任务已触发', 'time': datetime.now().isoformat()}


@app.route('/health')
def health():
    """健康检查"""
    return {'status': 'healthy', 'time': datetime.now().isoformat()}


if __name__ == '__main__':
    # 初始化调度器
    scheduler = init_scheduler()

    try:
        # 启动Flask应用
        app.run(host='0.0.0.0', port=5000, debug=False)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
