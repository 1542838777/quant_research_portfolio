import pandas as pd

from data.data_loader import LOCAL_PARQUET_DATA_DIR

#当天存活状态的股票 todo debug 看看每一步骤
# 1. 正常加载所有需要的数据
stock_basic_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'stock_basic.parquet')
close_df = ...  # 加载处理好的价格宽表

# --- 【核心修正逻辑】 ---
# 2. 创建一个“每日存活股票”的布尔矩阵
print("正在构建每日动态股票池...")

# 确保日期格式正确
stock_basic_df['list_date'] = pd.to_datetime(stock_basic_df['list_date'])
stock_basic_df['delist_date'] = pd.to_datetime(stock_basic_df['delist_date'])

# 创建一个空的DataFrame，index是交易日，columns是所有出现过的股票代码
# close_df 已经是对齐好的，可以直接使用它的结构
universe_df = pd.DataFrame(index=close_df.index, columns=close_df.columns)

# 循环每一只股票，标记其存活的日期区间
for stock_code, row in stock_basic_df.iterrows():
    # 获取上市和退市日期
    list_date = row['list_date']
    # 如果退市日期为空，则认为它至今存活
    delist_date = row['delist_date'] if pd.notna(row['delist_date']) else pd.to_datetime('2200-01-01')

    # 在universe_df中，将该股票在存活区间的标记设为True
    # 仅当股票代码存在于我们的列中时才操作
    if row['ts_code'] in universe_df.columns:
        universe_df.loc[(universe_df.index >= list_date) & (universe_df.index < delist_date), row['ts_code']] = True

# 将所有未被标记的（NaN）填充为False
universe_df.fillna(False, inplace=True)
print("每日动态可用股票池构建完毕！")

# --- 后续所有操作都必须基于这个universe_df ---
#
# # 3. 在因子计算和回测时，屏蔽掉“不存活”的股票
# # 例如，在处理因子时
# processed_factor = process_factor(raw_factor)
# # 只保留当天存活的股票的因子值，其余设为NaN
# processed_factor[~universe_df] = np.nan
#
# # 在回测引擎中，也应该使用这个universe来过滤信号
# # (vectorbt的很多操作会自动忽略NaN，所以上面的操作已经很有效)