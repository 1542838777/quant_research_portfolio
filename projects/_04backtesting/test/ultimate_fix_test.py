"""
终极修复测试：确保达到100% Max Gross Exposure
"""

import pandas as pd
import numpy as np
import vectorbt as vbt

# 创建简单但有效的测试
dates = pd.date_range('2023-01-01', periods=30, freq='D')
stocks = ['A', 'B', 'C']

# 稳定的价格数据（避免价格波动影响）
prices = pd.DataFrame({
    'A': [100] * 30,  # 恒定价格
    'B': [200] * 30,  # 恒定价格  
    'C': [150] * 30,  # 恒定价格
}, index=dates)

print("=== 终极修复：确保100% Max Gross Exposure ===")

# 测试各种方案
scenarios = [
    ("单股100%一次性买入", {
        'entries': pd.DataFrame({'A': [True] + [False]*29, 'B': [False]*30, 'C': [False]*30}, index=dates),
        'size': pd.DataFrame({'A': [1.0] + [0.0]*29, 'B': [0.0]*30, 'C': [0.0]*30}, index=dates)
    }),
    ("三股等权重一次性买入", {
        'entries': pd.DataFrame({'A': [True] + [False]*29, 'B': [True] + [False]*29, 'C': [True] + [False]*29}, index=dates),
        'size': pd.DataFrame({'A': [0.33] + [0.0]*29, 'B': [0.33] + [0.0]*29, 'C': [0.34] + [0.0]*29}, index=dates)
    }),
    ("每天轮换单股100%", {
        'entries': pd.DataFrame({
            'A': [True if i % 3 == 0 else False for i in range(30)],
            'B': [True if i % 3 == 1 else False for i in range(30)], 
            'C': [True if i % 3 == 2 else False for i in range(30)]
        }, index=dates),
        'size': pd.DataFrame({
            'A': [1.0 if i % 3 == 0 else 0.0 for i in range(30)],
            'B': [1.0 if i % 3 == 1 else 0.0 for i in range(30)],
            'C': [1.0 if i % 3 == 2 else 0.0 for i in range(30)]
        }, index=dates)
    })
]

for scenario_name, config in scenarios:
    print(f"\n【{scenario_name}】")
    
    try:
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=config['entries'],
            exits=False,  # 永不卖出
            size=config['size'],
            size_type='percent',
            init_cash=100000,
            fees=0.001,
            freq='D'
        )
        
        stats = portfolio.stats()
        max_exposure = stats['Max Gross Exposure [%]']
        
        # 验证资金使用情况
        initial_cash = portfolio.cash().iloc[0]
        final_cash = portfolio.cash().iloc[-1] 
        cash_used_pct = (1 - final_cash/initial_cash) * 100
        
        print(f"  Max Gross Exposure: {max_exposure:.2f}%")
        print(f"  资金使用率: {cash_used_pct:.2f}%")
        print(f"  总交易数: {stats['Total Trades']}")
        
        if max_exposure > 95:
            print(f"  ✅ 成功达到高暴露度！")
            
            # 如果成功，显示详细信息用于分析
            print(f"  详细信息：")
            trades = portfolio.trades.records_readable
            if len(trades) > 0:
                print(f"    交易记录: {len(trades)}笔")
                print(f"    首笔交易: {trades.iloc[0]['Size']:.0f}股")
        else:
            print(f"  ❌ 暴露度仍然不足")
            
    except Exception as e:
        print(f"  错误: {e}")

print("\n=== 结论 ===")
print("如果上述任何一个场景达到了>95%的Max Gross Exposure，")
print("说明回测器的逻辑本身是正确的，问题可能在于复杂场景下的信号处理。")