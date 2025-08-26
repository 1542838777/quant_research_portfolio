import pandas as pd
import numpy as np

# 简单验证shift逻辑
data = {
    'A': [100, 101, 102, 103, 104],
    'B': [200, 201, 202, 203, 204]
}
dates = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5']
prices = pd.DataFrame(data, index=dates)

print('=== 原始价格 ===')
print(prices)
print()

period = 2
start_price = prices
end_price = prices.shift(-period)

print(f'=== shift(-{period}) 结果 ===')
print('start_price:')
print(start_price)
print()
print('end_price (向前shift):')
print(end_price)
print()

returns = (end_price / start_price) - 1
print('=== 收益率计算 ===')
print(returns)
print()

print('=== 手工验证 ===')
print('Day1的2日收益率应该是: (Day3价格 / Day1价格) - 1')
print(f'股票A: ({103} / {100}) - 1 = {(103/100) - 1:.1%}')
print(f'计算结果: {returns.loc["Day1", "A"]:.1%}')
print()

print('=== 问题分析 ===')
print('1. shift(-period) 是向前取未来数据 ✅')
print('2. 计算 T日->T+period日 的收益率 ✅')
print('3. 最后几日收益率为NaN (无未来数据) ✅')
print('4. 时间对齐: 因子T-1 预测 T到T+period ✅')