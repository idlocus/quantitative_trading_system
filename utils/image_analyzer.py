#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片分析工具 - 读取K线图参考图片并提取关键信息
"""

import sys
from PIL import Image
import io

def analyze_kline_image(image_path):
    """分析K线图截图，提取布局和元素信息"""
    img = Image.open(image_path)

    print(f"图片路径: {image_path}")
    print(f"图片尺寸: {img.size[0]}x{img.size[1]}")
    print(f"图片格式: {img.format}")
    print(f"颜色模式: {img.mode}")

    # 分析图片基本信息
    width, height = img.size

    # 典型K线图布局分析
    # 通常1920x813 -> 主图大约上方60%，下方是副图
    main_chart_height = int(height * 0.55)
    subplot_height = int(height * 0.15)

    print(f"\n=== 布局分析 ===")
    print(f"主图区域: 0 ~ {main_chart_height} ({main_chart_height/height*100:.0f}%)")
    print(f"副图区域: {main_chart_height} ~ {main_chart_height + subplot_height}")
    print(f"底部区域: {main_chart_height + subplot_height} ~ {height}")

    # 分析颜色分布（粗略判断颜色主题）
    # 采样一些关键位置的颜色
    pixels = []
    for x in range(0, width, width//10):
        for y in [main_chart_height//2, main_chart_height + subplot_height//2]:
            try:
                pixels.append(img.getpixel((x, y)))
            except:
                pass

    red_count = 0
    green_count = 0
    white_count = 0

    for p in pixels:
        if isinstance(p, tuple) and len(p) >= 3:
            r, g, b = p[:3]
            if r > 150 and g < 100 and b < 100:
                red_count += 1
            elif g > 150 and r < 100 and b < 100:
                green_count += 1
            elif r > 200 and g > 200 and b > 200:
                white_count += 1

    print(f"\n=== 颜色分析 ===")
    print(f"红色像素: {red_count} (可能为上涨K线)")
    print(f"绿色像素: {green_count} (可能为下跌K线)")
    print(f"白色像素: {white_count}")

    # 输出图片作为base64供进一步分析
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    print(f"\n=== 图片Base64 ===")
    print(f"长度: {len(img_b64)} 字符")

    return {
        'path': image_path,
        'size': img.size,
        'format': img.format,
        'mode': img.mode,
        'layout': {
            'main_chart_height': main_chart_height,
            'subplot_height': subplot_height,
            'total_height': height
        }
    }

def describe_wind_kline_features():
    """描述典型Wind风格K线图的特征"""
    print("""
=== Wind终端K线图典型特征 ===

【主图区域】(约60%高度)
- K线蜡烛图（红涨绿跌）
- MA均线: MA5(白色), MA10(黄色), MA20(紫色), MA60(蓝色)
- 布林带: 中轨(MA20) ± 2倍标准差
- 买卖信号标记: 买入=红色向上箭头, 卖出=绿色向下箭头

【成交量副图】(约15%高度)
- 柱状图显示成交量
- 颜色与K线对应(红涨绿跌)
- 可能有均量线(MA5, MA10)

【MACD副图】(约15%高度)
- DIF线(白色)
- DEA线(黄色)
- MACD柱状图(红涨绿跌)

【KDJ副图】(约15%高度)
- K线(白色)
- D线(黄色)
- J线(紫色)
- 超买超卖区域标记

【右侧信息面板】
- 股票名称和代码
- 当前价格、涨跌额、涨跌幅
- 成交量、成交额、换手率
- 最高价、最低价、开盘价、昨收价

【顶部工具栏】
- 周期切换: 分时/5分钟/15分钟/30分钟/60分钟/日K/周K/月K
- 指标切换: MA/BOLL/KDJ/RSI/MACD等

【特色功能】
- 鼠标悬停显示十字光标和详细信息
- 缩放和平移功能
- 买卖点标注(可显示盈利/亏损数字)
""")

if __name__ == "__main__":
    import os
    import base64

    docs_dir = r"D:\python_work\Trae\quantitative_trading_system\docs"

    print("=" * 60)
    print("K线图参考图片分析")
    print("=" * 60)

    # 分析Wind K线图
    wind_img = os.path.join(docs_dir, "wind_kline.jpg")
    if os.path.exists(wind_img):
        print(f"\n分析 Wind K线图...")
        analyze_kline_image(wind_img)
    else:
        print(f"文件不存在: {wind_img}")

    # 分析东方财富K线图
    dfcf_img = os.path.join(docs_dir, "dfcf_kline.jpg")
    if os.path.exists(dfcf_img):
        print(f"\n分析 东方财富K线图...")
        analyze_kline_image(dfcf_img)
    else:
        print(f"文件不存在: {dfcf_img}")

    # 描述典型K线图功能
    describe_wind_kline_features()