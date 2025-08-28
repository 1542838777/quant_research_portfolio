import numpy as np
import pandas as pd

# 1. 初始状态：600只股票，排名从1到500
n_stocks = 500
bask_skip_quantile_change_stocks = 250
bask_tiny_change_stocks = n_stocks-bask_skip_quantile_change_stocks
before_ranking = np.arange(1, n_stocks + 1)

# 2. 模拟排名变动：让bask_tiny_change_stocks只股票稳定，bask_skip_quantile_change_stocks只股票剧变
# 目标：让平均绝对变动值恰好等于 9
changes = np.zeros(n_stocks)
# bask_tiny_change_stocks只股票，在[-3, 3]之间随机变动
changes[:bask_tiny_change_stocks] = np.random.randint(-3, 4, bask_tiny_change_stocks)
# bask_skip_quantile_change_stocks只股票，在[-30, 30]之间随机变动 (为了凑够平均9)
# 设这bask_skip_quantile_change_stocks只股票平均变动x，(bask_tiny_change_stocks*mean(abs(change_stable)) + bask_skip_quantile_change_stocks*x)/500 = 9
# 假设稳定组平均变动为2，(bask_tiny_change_stocks*2 + bask_skip_quantile_change_stocks*x)/500 = 9 -> 800 + bask_skip_quantile_change_stocksx = 4500 -> bask_skip_quantile_change_stocksx = 3700 -> x=37
changes[bask_tiny_change_stocks:] = np.random.randint(-80, 80, bask_skip_quantile_change_stocks)

# 确保总变动为0，排名才能重新分配
changes -= int(changes.mean())
np.random.shuffle(changes)

# 3. 计算新的排名
after_ranking = before_ranking + changes
# 重新排序以获得新的名次
after_ranking = pd.Series(after_ranking).rank().values

# 4. 分析结果
avg_abs_change = np.mean(np.abs(after_ranking - before_ranking))

# 计算有多少股票跨越了分组边界（每组bask_skip_quantile_change_stocks只）
before_quintile = (before_ranking - 1) // bask_skip_quantile_change_stocks
after_quintile = (after_ranking - 1) // bask_skip_quantile_change_stocks
stocks_that_crossed_quintiles = np.sum(before_quintile != after_quintile)
turnover_percentage = stocks_that_crossed_quintiles / n_stocks

print(f"模拟的平均绝对排名变动: {avg_abs_change:.2f} 名")
print(f"跨越分组边界的股票数量: {stocks_that_crossed_quintiles} 只")
print(f"等同于真实组合换手率: {turnover_percentage:.2%}")