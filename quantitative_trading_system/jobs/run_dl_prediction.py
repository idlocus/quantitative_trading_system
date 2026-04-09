#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习预测脚本

每日收盘后运行，使用LSTM模型对股票池进行次日走势预测，
并将结果通过企业微信/飞书发送。

用法:
    python run_dl_prediction.py
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from ml.signal_fusion.fusion import DLSignalFusion
from data.market import AKShareMarketData  # 或 WindMarketData

# 配置日志
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dl_prediction.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


# 默认股票池
DEFAULT_STOCKS = [
    '000001.SZ', '000002.SZ', '000858.SZ',
    '600000.SH', '600519.SH', '600036.SH',
    '601318.SH', '601398.SH', '000333.SZ', '300750.SZ'
]


def load_latest_price_data(symbol: str, days: int = 80) -> pd.DataFrame:
    """
    加载最新价格数据

    Args:
        symbol: 股票代码
        days: 加载天数（需要比sequence_length多）

    Returns:
        OHLCV DataFrame
    """
    try:
        market = AKShareMarketData()
        data = market.get_historical_data(symbol, days=days)
        if data is not None and not data.empty:
            logger.info(f"{symbol}: 获取到 {len(data)} 条数据")
            return data
    except Exception as e:
        logger.warning(f"{symbol}: 数据获取失败 - {e}")

    return pd.DataFrame()


def run_prediction(
    model_path: str,
    stock_list: List[str] = None,
    output_path: str = None
) -> Dict:
    """
    运行预测

    Args:
        model_path: 模型文件路径
        stock_list: 股票列表
        output_path: 结果输出路径

    Returns:
        预测结果字典
    """
    if stock_list is None:
        stock_list = DEFAULT_STOCKS

    # 检查模型是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        logger.info("请先运行 python ml/train_model.py 训练模型")
        return {'error': '模型文件不存在'}

    # 初始化融合器
    try:
        fusion = DLSignalFusion(model_path)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return {'error': str(e)}

    results = []
    summary = {'total': 0, 'strong_buy': 0, 'buy': 0, 'hold': 0, 'sell': 0, 'strong_sell': 0}

    logger.info(f"开始预测 {len(stock_list)} 只股票...")

    for symbol in stock_list:
        try:
            # 加载数据
            price_data = load_latest_price_data(symbol, days=80)
            if price_data.empty:
                logger.warning(f"{symbol}: 无数据")
                continue

            # 生成融合信号
            signal = fusion.generate_signal(
                symbol=symbol,
                price_data=price_data,
                technical_framework_output=None,  # 可接入技术框架
                news_signal=None  # 可接入新闻信号
            )

            results.append(signal)
            summary['total'] += 1

            # 统计
            pred = signal['prediction']
            if pred == 'strong_buy':
                summary['strong_buy'] += 1
            elif pred == 'buy':
                summary['buy'] += 1
            elif pred == 'sell':
                summary['sell'] += 1
            elif pred == 'strong_sell':
                summary['strong_sell'] += 1
            else:
                summary['hold'] += 1

            # 日志
            logger.info(
                f"{symbol}: {signal['prediction']} "
                f"(置信度: {signal['dl_confidence']:.2f}, "
                f"综合得分: {signal['combined_score']:.1f})"
            )

        except Exception as e:
            logger.error(f"{symbol} 预测失败: {e}")
            continue

    # 按综合得分排序
    results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

    # 构建输出
    output = {
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'top_picks': results[:5] if len(results) >= 5 else results,
        'all_results': results,
        'model_info': {
            'path': model_path,
            'stocks_count': len(stock_list)
        }
    }

    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_path}")

    # 发送通知
    try:
        send_notification(output)
    except Exception as e:
        logger.error(f"发送通知失败: {e}")

    return output


def send_notification(result: Dict):
    """发送预测结果通知"""
    if 'error' in result:
        logger.warning(f"跳过通知: {result['error']}")
        return

    summary = result['summary']

    # 构建消息
    message = f"""📊 深度学习走势预测
⏰ {result['timestamp'][:10]}

📈 股票池: {summary['total']} 只
━━━━━━━━━━━━━━━
强烈买入: {summary['strong_buy']} 只
买入: {summary['buy']} 只
持有: {summary['hold']} 只
卖出: {summary['sell']} 只
强烈卖出: {summary['strong_sell']} 只
━━━━━━━━━━━━━━━

🏆 重点关注:"""

    top_picks = result.get('top_picks', [])
    for i, pick in enumerate(top_picks[:3], 1):
        signal_emoji = {
            'strong_buy': '🟢',
            'buy': '🟢',
            'hold': '🟡',
            'sell': '🔴',
            'strong_sell': '🔴'
        }.get(pick['prediction'], '⚪')

        message += f"\n{i}. {pick['symbol']} {signal_emoji}{pick['prediction'].replace('_', ' ')}"
        message += f"\n   置信度: {pick['dl_confidence']:.1%} | 得分: {pick['combined_score']:.1f}"

    # 发送到企业微信/飞书
    try:
        from utils.wechat_utils import send_message
        send_message(message)
        logger.info("企业微信通知已发送")
    except ImportError:
        pass

    try:
        from utils.feishu_utils import send_message
        send_message(message)
        logger.info("飞书通知已发送")
    except ImportError:
        pass


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("深度学习预测任务开始")
    logger.info("=" * 50)

    # 模型路径
    model_path = os.path.join(project_root, 'ml', 'models', 'best_model.pt')
    output_path = os.path.join(project_root, 'results', 'dl_prediction.json')

    # 运行预测
    result = run_prediction(
        model_path=model_path,
        stock_list=DEFAULT_STOCKS,
        output_path=output_path
    )

    if 'error' in result:
        logger.error(f"预测失败: {result['error']}")
    else:
        logger.info("预测任务完成")
        logger.info(f"强烈买入: {result['summary']['strong_buy']} 只")
        logger.info(f"买入: {result['summary']['buy']} 只")
        logger.info(f"持有: {result['summary']['hold']} 只")
        logger.info(f"卖出: {result['summary']['sell']} 只")
        logger.info(f"强烈卖出: {result['summary']['strong_sell']} 只")


if __name__ == '__main__':
    main()
