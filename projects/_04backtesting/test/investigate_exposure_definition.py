"""
深入调查Max Gross Exposure的真正定义
"""

import pandas as pd
import numpy as np
import vectorbt as vbt

# 创建一个极简单的测试来理解Exposure的定义
dates = pd.date_range('2023-01-01', periods=5, freq='D')
price = pd.Series([100, 110, 120, 130, 140], index=dates, name='STOCK')

print("=== 调查Max Gross Exposure的真正定义 ===")
print(f"价格序列: {price.tolist()}")

# 测试1: 单次买入100%资金
print("\n【测试1】单次买入100%资金，持有到底")
entry1 = pd.Series([True, False, False, False, False], index=dates, name='STOCK')
size1 = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=dates, name='STOCK')

portfolio1 = vbt.Portfolio.from_signals(
    close=price,
    entries=entry1,
    exits=False,
    size=size1,
    size_type='percent',
    init_cash=100000,
    fees=0.001,
    freq='D'
)

# 手动计算各项指标
cash1 = portfolio1.cash()
value1 = portfolio1.value()
holdings_value1 = value1 - cash1

print(f"现金变化: {cash1.tolist()}")
print(f"总价值变化: {value1.tolist()}")
print(f"持仓价值: {holdings_value1.tolist()}")

# 计算暴露度 (持仓价值/总价值)
manual_exposure1 = holdings_value1 / value1 * 100
print(f"手动计算暴露度: {manual_exposure1.tolist()}")

stats1 = portfolio1.stats()
print(f"vectorBT Max Gross Exposure: {stats1['Max Gross Exposure [%]']:.6f}%")

# 尝试直接调用exposure方法
try:
    gross_exp1 = portfolio1.gross_exposure()
    print(f"直接调用gross_exposure(): {gross_exp1.tolist()}")
except Exception as e:
    print(f"无法调用gross_exposure(): {e}")

print("\n" + "="*50)

# 测试2: 每天重新买入
print("\n【测试2】每天重新买入100%")
entry2 = pd.Series([True, True, True, True, True], index=dates, name='STOCK')
size2 = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=dates, name='STOCK')

portfolio2 = vbt.Portfolio.from_signals(
    close=price,
    entries=entry2,
    exits=False,
    size=size2,
    size_type='percent',
    init_cash=100000,
    fees=0.001,
    freq='D'
)

cash2 = portfolio2.cash()
value2 = portfolio2.value()
holdings_value2 = value2 - cash2
manual_exposure2 = holdings_value2 / value2 * 100

print(f"现金变化: {cash2.tolist()}")
print(f"手动计算暴露度: {manual_exposure2.tolist()}")

stats2 = portfolio2.stats()
print(f"vectorBT Max Gross Exposure: {stats2['Max Gross Exposure [%]']:.6f}%")

print("\n" + "="*50)

# 测试3: 检查vectorBT源码中的定义
print("\n【测试3】尝试理解vectorBT的Exposure计算方式")

# 查看portfolio的内部属性
print(f"Portfolio可用方法: {[method for method in dir(portfolio1) if 'exposure' in method.lower()]}")

# 检查是否有其他相关方法
print(f"Portfolio统计相关方法: {[method for method in dir(portfolio1) if method.startswith('get_') or 'stat' in method.lower()]}")

# 测试4: 对比不同的size_type
print("\n【测试4】测试不同的size_type")
try:
    # 使用value类型
    portfolio3 = vbt.Portfolio.from_signals(
        close=price,
        entries=entry1,
        exits=False,
        size=99000,  # 99%的初始资金
        size_type='value',
        init_cash=100000,
        fees=0.001,
        freq='D'
    )
    
    stats3 = portfolio3.stats()
    print(f"size_type='value' Max Gross Exposure: {stats3['Max Gross Exposure [%]']:.6f}%")
    
except Exception as e:
    print(f"size_type='value'测试失败: {e}")

print("\n=== 结论推测 ===")
print("基于以上测试，Max Gross Exposure可能的定义：")
print("1. 不是简单的持仓价值/总价值比例")
print("2. 可能与每日新增买入的金额相关")
print("3. 可能与交易频率或换手率相关")
print("4. 需要进一步查看vectorBT的源码或文档")