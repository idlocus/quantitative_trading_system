#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻信号分析脚本

自动获取新闻、生成情感信号、绘制分析图表
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from signals import NewsSignalGenerator
from data.news import AKShareNewsSource

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_stock_news(symbol: str):
    """分析单只股票的新闻信号"""
    logger.info(f"=" * 60)
    logger.info(f"新闻信号分析 - {symbol}")
    logger.info(f"=" * 60)

    # 初始化新闻信号生成器
    news_source = AKShareNewsSource()
    signal_gen = NewsSignalGenerator(news_source)

    # 生成信号
    signal = signal_gen.generate_signal(symbol)

    # 打印结果
    print(f"\n{'=' * 50}")
    print(f"股票: {signal.symbol}")
    print(f"信号类型: {signal.signal_type.value}")
    print(f"情感评分: {signal.sentiment_score:.3f}")
    print(f"置信度: {signal.confidence:.3f}")
    print(f"新闻数量: {signal.news_count}")
    print(f"  - 正面: {signal.positive_count}")
    print(f"  - 负面: {signal.negative_count}")
    print(f"  - 中性: {signal.neutral_count}")
    print(f"\n信号描述: {signal_gen.get_signal_description(signal)}")

    if signal.key_news:
        print(f"\n关键新闻:")
        for i, news in enumerate(signal.key_news, 1):
            print(f"  {i}. {news}")

    return signal


def create_sentiment_chart(symbol: str, output_dir: str = 'results/charts'):
    """创建情感分析图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 获取新闻
    news_source = AKShareNewsSource()
    news = news_source.get_stock_news(symbol, days=7)

    if not news:
        logger.warning(f"没有获取到 {symbol} 的新闻")
        return

    # 创建图表 - 使用英文避免字体问题
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'{symbol} News Sentiment Analysis', fontsize=14, fontweight='bold')

    # 情感关键词统计
    positive_words = [
        '涨停', '增长', '盈利', '突破', '买入', '推荐', '看好', '上调',
        '收益', '利润', '业绩', '超预期', '加速', '扩张', '合作', '中标'
    ]
    negative_words = [
        '跌停', '亏损', '减持', '卖出', '下调', '风险', '警示', '预警',
        '下滑', '下降', '不及预期', '放缓', '收缩', '终止', '诉讼', '处罚'
    ]

    # 统计每条新闻的情感
    sentiments = []
    for item in news:
        text = (item.get('title', '') + item.get('content', ''))
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        sentiments.append({'title': str(item.get('title', ''))[:50], 'pos': pos_count, 'neg': neg_count})

    df = pd.DataFrame(sentiments)

    # 绘制情感统计
    ax1 = axes[0]
    x = range(len(df))
    width = 0.35
    ax1.bar([i - width/2 for i in x], df['pos'], width, label='Positive', color='green', alpha=0.7)
    ax1.bar([i + width/2 for i in x], df['neg'], width, label='Negative', color='red', alpha=0.7)
    ax1.set_xlabel('News Index')
    ax1.set_ylabel('Keyword Count')
    ax1.set_title('News Sentiment Statistics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制情感时间线
    ax2 = axes[1]
    scores = []
    for item in news:
        text = (item.get('title', '') + item.get('content', ''))
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        score = (pos_count - neg_count) / max(pos_count + neg_count, 1)
        scores.append(score)

    colors = ['green' if s > 0 else 'red' for s in scores]
    ax2.bar(range(len(scores)), scores, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('News Index')
    ax2.set_ylabel('Sentiment Score')
    ax2.set_title('Sentiment Score Timeline')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(output_dir, f'{symbol}_news_sentiment_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"图表已保存到: {save_path}")
    plt.close()

    return save_path


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='新闻信号分析')
    parser.add_argument('symbol', nargs='?', default='002602.SZ', help='股票代码')
    args = parser.parse_args()

    symbol = args.symbol.upper()

    # 分析新闻信号
    signal = analyze_stock_news(symbol)

    # 生成图表
    chart_path = create_sentiment_chart(symbol)

    print(f"\n图表路径: {chart_path}")

    # 返回信号结果供其他模块使用
    return signal


if __name__ == '__main__':
    main()
