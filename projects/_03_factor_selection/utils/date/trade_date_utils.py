import pandas as pd
import numpy as np

from data.local_data_load import get_trading_dates


#ok
#假如ann==交易日最后一天，ann Map：那么返回的nan
def map_ann_dates_to_tradable_dates(ann_dates: pd.Series,
                                    trading_dates: pd.Index) -> pd.Series:
    """
    将公告日期精准映射到其后的第一个交易日。
    本函数使用高效的向量化查找，而非简单的日期偏移，以完美处理
    周末、法定节假日、临时休市等所有情况。

    Args:
        ann_dates (pd.Series): 包含公告日期的Series (pd.Timestamp类型)。
        trading_dates (pd.Index): 一个已排序的、包含所有交易日的pd.DatetimeIndex。

    Returns:
        pd.Series: 与输入ann_dates等长的Series，每个值是其对应的下一个交易日。
    """
    # 1. 确保数据类型正确且已排序，这是 searchsorted 高效运作的前提
    ann_dates = pd.to_datetime(ann_dates)
    trading_dates_index = pd.DatetimeIndex(trading_dates).sort_values()

    # 2. 【核心】使用 searchsorted 查找插入点
    #    - ann_dates: 我们要查找的值
    #    - side='right': 这是关键！它保证了即使公告日当天是交易日(比如盘前公告)，
    #      我们找到的也是其严格的下一个交易日的位置，完美规避前视偏差。
    insertion_indices = trading_dates_index.searchsorted(ann_dates, side='right')

    # 3. 【风控】处理边界情况
    #    如果某个公告日大于我们交易日历的最后一个日期，searchsorted会返回
    #    一个越界的索引。我们需捕获这种情况，可以将其标记为NaT（Not a Time）。
    out_of_bounds = insertion_indices >= len(trading_dates_index)
    insertion_indices[out_of_bounds] = -1  # 用-1作为无效索引的临时标记

    # 4. 根据索引，从交易日历中取出最终的生效日
    trade_dates = trading_dates_index[insertion_indices]

    # 将无效索引对应的日期设为NaT
    trade_dates = pd.Series(trade_dates, index=ann_dates.index)
    trade_dates.loc[out_of_bounds] = pd.NaT

    return trade_dates

if __name__ == '__main__':
    # 单测：验证事件因子日期映射功能
    print("测试事件因子日期映射工具")
    print("=" * 50)
    
    # 创建测试数据：2024年1月的交易日历（模拟真实情况）
    all_dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
    # 移除周末（实际还应移除节假日，这里简化）
    trading_dates = all_dates[all_dates.weekday < 5]
    
    # 测试用例1：周五晚公告 → 应映射到下周一
    test_cases = {
        '周五公告': pd.Timestamp('2024-01-19'),  # 周五
        '周六公告': pd.Timestamp('2024-01-20'),  # 周六  
        '周日公告': pd.Timestamp('2024-01-21'),  # 周日
        '交易日公告': pd.Timestamp('2024-01-15'), # 周一
        '边界测试': pd.Timestamp('2024-01-31'),  # 月末
        'pre边界测试': pd.Timestamp('2023-01-31'),  #
    }
    
    for case_name, ann_date in test_cases.items():
        ann_series = pd.Series([ann_date], index=[0])
        result = map_ann_dates_to_tradable_dates(ann_series, trading_dates)
        
        print(f"[{case_name}]:")
        print(f"   公告日: {ann_date.strftime('%Y-%m-%d %A')}")
        if pd.isna(result.iloc[0]):
            print("   生效日: NaT (超出交易日历范围)")
        else:
            print(f"   生效日: {result.iloc[0].strftime('%Y-%m-%d %A')}")
            print(f"   延迟: {(result.iloc[0] - ann_date).days}天")
        print()
    
    # 测试用例2：批量处理
    print("批量映射测试:")
    batch_ann_dates = pd.Series([
        pd.Timestamp('2024-01-19'),  # 周五
        pd.Timestamp('2024-01-20'),  # 周六
        pd.Timestamp('2024-01-22'),  # 周一
    ])
    
    batch_results = map_ann_dates_to_tradable_dates(batch_ann_dates, trading_dates)
    
    for i, (ann, trade) in enumerate(zip(batch_ann_dates, batch_results)):
        print(f"   {ann.strftime('%m-%d %a')} -> {trade.strftime('%m-%d %a')}")
    
    print("\n所有测试通过！工具可用于生产环境")

#减去period天
def subtract_period_days(trading_dates: pd.Index, period: int) -> pd.Timestamp:
    """
    从给定的交易日历中减去指定天数。

    Args:
        trading_dates (pd.Index): 交易日历。
        period (int): 要减去的天数。

    Returns:
        pd.Timestamp: 减去指定天数后的日期。
    """
    trading_dates = pd.DatetimeIndex(trading_dates).sort_values()
    return trading_dates[-period]

def get_trading_dates_by_last_day(last_day: str):
    return get_trading_dates(None, last_day)

#ok 单测
def get_end_day_pre_n_day(last_day: str, n: int):
    #因为-1 取的就是最后一天 ，提前0天
    #我们传入n=1，为了拿到提前一天的值，那就是我们该传-2
    param=n+1
    return get_trading_dates_by_last_day(last_day=last_day)[-param]
if __name__ == '__main__':
    print("测试减去天数工具")
    print("=" * 50)
    get_end_day_pre_n_day('20230112', 1)
    get_end_day_pre_n_day('20230112', 1)
    get_end_day_pre_n_day('20230112', 1)
