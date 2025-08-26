import pandas as pd
import numpy as np

# 模拟一只股票的除权场景
dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']

print('=== 除权事件模拟 ===')
print('2024-01-03: 股票除权，10派5元 (除权因子=0.5)')
print()

# 原始价格
raw_prices = [20, 22, 10, 11, 12]  # 1月3日除权后价格腰斩
print('原始价格:', raw_prices)

# 前复权（qfq）- 以最早日期为基准
qfq_prices = [10, 11, 10, 11, 12]  # 历史价格不变，现在价格调整
print('前复权:', qfq_prices)

# 后复权（hfq）- 以最新日期为基准  
hfq_prices = [40, 44, 20, 22, 24]  # 历史价格被放大，现在价格不变
print('后复权:', hfq_prices)

print()
print('=== 收益率计算对比 ===')

# 使用前复权计算收益率
qfq_returns = [np.nan] + [qfq_prices[i]/qfq_prices[i-1]-1 for i in range(1,5)]
print('前复权收益率:', [f'{r:.1%}' if not pd.isna(r) else 'NaN' for r in qfq_returns])

# 使用后复权计算收益率
hfq_returns = [np.nan] + [hfq_prices[i]/hfq_prices[i-1]-1 for i in range(1,5)]
print('后复权收益率:', [f'{r:.1%}' if not pd.isna(r) else 'NaN' for r in hfq_returns])

print()
print('关键问题：')
print('前复权在1月2日的收益率: 10%')
print('后复权在1月2日的收益率: 10%') 
print('看起来一样？但是...')

print()
print('致命缺陷暴露：')
print('如果我们在1月4日重新计算1月2日的收益率：')
print()

print('=== 时间旅行问题 ===')
print('假设我们在1月2日做回测:')
print('- 前复权: 只能看到1月1日和1月2日的数据')
print('- 后复权: 1月2日的44这个价格是基于未来除权事件计算的！')
print()
print('这就是为什么后复权在量化研究中是致命的:')
print('1. 它让我们在历史时点拥有了未来信息')  
print('2. 回测结果会严重失真')
print('3. 实盘表现与回测差异巨大')