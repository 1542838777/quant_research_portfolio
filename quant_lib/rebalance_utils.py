"""
调仓日期工具模块

提供基于交易日历的调仓日期生成功能
"""

import pandas as pd
import numpy as np
from typing import Union
import logging
import re

logger = logging.getLogger(__name__)

#ok
def generate_rebalance_dates(trading_dates: pd.DatetimeIndex, 
                           rebalancing_freq: str) -> pd.DatetimeIndex:
    """
    基于交易日历生成调仓日期
    
    Args:
        trading_dates: 严格按照交易日的日期索引 (pd.DatetimeIndex)
        rebalancing_freq: 调仓频率
            - 'D' 或 'day': 每日调仓  
            - 'W' 或 'week': 每周调仓 (每周最后一个交易日)
            - 'M' 或 'month': 每月调仓 (每月最后一个交易日)  
            - 'Q' 或 'quarter': 每季度调仓 (每季度最后一个交易日)
            - 'Y' 或 'year': 每年调仓 (每年最后一个交易日)
            - '2D', '3D', 'nD': 每n个交易日调仓一次 (如 '2d', '5D' 等)
            
    Returns:
        pd.DatetimeIndex: 调仓日期列表
        
    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')  # 工作日
        >>> rebalance_dates = generate_rebalance_dates(dates, 'M')
        >>> print(len(rebalance_dates))  # 12个月度调仓日期
        >>> 
        >>> # 每3个交易日调仓一次
        >>> rebalance_dates_3d = generate_rebalance_dates(dates, '3D')
        >>> print(len(rebalance_dates_3d))  # 约87个调仓日期 (261工作日/3)
    """
    
    if not isinstance(trading_dates, pd.DatetimeIndex):
        trading_dates = pd.DatetimeIndex(trading_dates)
        
    if len(trading_dates) == 0:
        return pd.DatetimeIndex([])
    
    # 标准化调仓频率
    freq_mapping = {
        'd': 'D', 'day': 'D',
        'w': 'W', 'week': 'W', 
        'm': 'M', 'month': 'M',
        'q': 'Q', 'quarter': 'Q',
        'y': 'Y', 'year': 'Y'
    }
    
    # 检查是否为nD格式 (如 '2D', '3d', '5D')
    n_days_pattern = re.match(r'^(\d+)[dD]$', rebalancing_freq)
    if n_days_pattern:
        n_days = int(n_days_pattern.group(1))
        if n_days <= 0:
            raise ValueError(f"Invalid interval: {rebalancing_freq}. Interval must be positive.")
        freq = f'{n_days}D'  # 标准化为 '2D', '3D' 等格式
    else:
        freq = freq_mapping.get(rebalancing_freq.lower())
        if freq is None:
            supported_freqs = list(freq_mapping.keys()) + ["'nD' (如 '2D', '3d', '5D')"]
            raise ValueError(f"Unsupported rebalancing frequency: {rebalancing_freq}. "
                           f"Supported frequencies: {supported_freqs}")
    
    # 每日调仓：返回所有交易日
    if freq == 'D':
        return trading_dates
    
    # nD调仓：每n个交易日调仓一次
    if freq.endswith('D') and freq != 'D':
        n_days = int(freq[:-1])  # 提取间隔天数
        # 从第一个交易日开始，每隔n个交易日选择一个调仓日
        rebalance_indices = list(range(0, len(trading_dates), n_days))
        rebalance_dates = trading_dates[rebalance_indices]
        
        logger.info(f"Generated {len(rebalance_dates)} rebalance dates with {n_days}-day interval")
        return rebalance_dates
    
    # 创建临时DataFrame用于分组
    df = pd.DataFrame({'date': trading_dates, 'dummy': 1})
    df.set_index('date', inplace=True)
    
    try:
        # 步骤1: 使用pandas的resample功能，获取理想的"日历调仓日"
        if freq == 'W':
            # 每周调仓：每周的周五（理想情况）
            ideal_rebalance_dates = pd.date_range(start=trading_dates[0], 
                                                end=trading_dates[-1], 
                                                freq='W-FRI')
        elif freq == 'M':
            # 每月调仓：每月最后一天（理想情况）
            ideal_rebalance_dates = pd.date_range(start=trading_dates[0], 
                                                 end=trading_dates[-1],
                                                freq='ME')
        elif freq == 'Q':
            # 每季度调仓：每季度最后一天（理想情况）
            ideal_rebalance_dates = pd.date_range(start=trading_dates[0], 
                                                end=trading_dates[-1], 
                                                freq='QE')
        elif freq == 'Y':
            # 每年调仓：每年最后一天（理想情况）
            ideal_rebalance_dates = pd.date_range(start=trading_dates[0], 
                                                end=trading_dates[-1], 
                                                freq='YE')
        else:
            raise ValueError(f"Internal error: unhandled frequency {freq}")
            
    except Exception as e:
        logger.error(f"Error generating ideal rebalance dates: {e}")
        raise
    
    # 步骤2&3: 将"日历调仓日"映射到实际交易日
    # 对每个理想调仓日，找到对应的当天或前一个实际交易日
    actual_rebalance_dates = []
    trading_dates_series = pd.Series(trading_dates, index=trading_dates)
    
    for ideal_date in ideal_rebalance_dates:
        # 如果理想日期本身就是交易日，直接使用
        if ideal_date in trading_dates:
            actual_rebalance_dates.append(ideal_date)
        else:
            # 否则找到该日期之前最近的一个交易日
            previous_trading_days = trading_dates[trading_dates <= ideal_date]
            if len(previous_trading_days) > 0:
                actual_rebalance_dates.append(previous_trading_days[-1])
            # 如果没有找到之前的交易日（极端情况），跳过该调仓日
    
    rebalance_dates = pd.DatetimeIndex(actual_rebalance_dates).drop_duplicates().sort_values()
    
    logger.info(f"Generated rebalance dates: freq={rebalancing_freq}, "
               f"trading_days={len(trading_dates)}, rebalance_days={len(rebalance_dates)}")
    
    return rebalance_dates


def validate_rebalance_dates(rebalance_dates: pd.DatetimeIndex, 
                           trading_dates: pd.DatetimeIndex) -> bool:
    """
    验证调仓日期的有效性
    
    Args:
        rebalance_dates: 调仓日期
        trading_dates: 交易日期
        
    Returns:
        bool: 是否有效
    """
    if len(rebalance_dates) == 0:
        logger.warning("Rebalance dates is empty")
        return False
        
    # 检查所有调仓日期都在交易日范围内
    if not rebalance_dates.isin(trading_dates).all():
        invalid_dates = rebalance_dates[~rebalance_dates.isin(trading_dates)]
        logger.error(f"Found non-trading days in rebalance dates: {invalid_dates}")
        return False
        
    # 检查调仓日期是否有序
    if not rebalance_dates.is_monotonic_increasing:
        logger.error("Rebalance dates are not in chronological order")
        return False
        
    logger.info("Rebalance dates validation passed")
    return True


# 兼容旧版本的函数名
def get_rebalance_dates(trading_dates: pd.DatetimeIndex, 
                       rebalancing_freq: str) -> pd.DatetimeIndex:
    """
    兼容旧版本的函数名，功能同generate_rebalance_dates
    """
    return generate_rebalance_dates(trading_dates, rebalancing_freq)


if __name__ == "__main__":
    # 测试代码 测通过！
    import pandas as pd
    
    # # 创建测试数据：2020年的工作日
    test_dates = pd.date_range('2019-01-01', '2020-12-31', freq='B')
    #
    # print("Testing rebalance date generation:")
    # print(f"Original trading days: {len(test_dates)}")
    # print(f"Date range: {test_dates[0]} to {test_dates[-1]}")
    
    # # 测试不同频率
    # for freq in ['D', 'W', 'M', 'Q', 'Y']:
    #     rebalance_dates = generate_rebalance_dates(test_dates, freq)
    #     print(f"\nFrequency {freq}: {len(rebalance_dates)} rebalance days")
    #     if len(rebalance_dates) <= 10:
    #         print(f"  Rebalance dates: {rebalance_dates.strftime('%Y-%m-%d').tolist()}")
    #     else:
    #         print(f"  ALL: {rebalance_dates.strftime('%Y-%m-%d').tolist()}")
    #         # print(f"  First 5: {rebalance_dates[:5].strftime('%Y-%m-%d').tolist()}")
    #         # print(f"  Last 5: {rebalance_dates[-5:].strftime('%Y-%m-%d').tolist()}")
    #
    #     # 验证结果
    #     is_valid = validate_rebalance_dates(rebalance_dates, test_dates)
    #     print(f"  Validation: {'PASS' if is_valid else 'FAIL'}")
    #
    # 测试nD频率
    print("\n" + "="*50)
    print("Testing nD frequencies:")
    for freq in ['2D', '3d', '5D', '11d']:
        rebalance_dates = generate_rebalance_dates(test_dates, freq)
        expected_count = (len(test_dates) + int(freq[:-1]) - 1) // int(freq[:-1])  # 向上取整
        print(f"  ALL: {rebalance_dates.strftime('%Y-%m-%d').tolist()}")

        # print(f"\nFrequency {freq}: {len(rebalance_dates)} rebalance days (expected ~{expected_count})")
        
        if len(rebalance_dates) <= 10:
            print(f"  Rebalance dates: {rebalance_dates.strftime('%Y-%m-%d').tolist()}")
        else:
            print(f"  First 5: {rebalance_dates[:5].strftime('%Y-%m-%d').tolist()}")
            print(f"  Last 5: {rebalance_dates[-5:].strftime('%Y-%m-%d').tolist()}")
        
        # 验证结果
        is_valid = validate_rebalance_dates(rebalance_dates, test_dates)
        print(f"  Validation: {'PASS' if is_valid else 'FAIL'}")
        
        # 验证间隔是否正确（仅测试前几个日期）
        if len(rebalance_dates) >= 2:
            actual_intervals = []
            for i in range(1, min(4, len(rebalance_dates))):  # 检查前3个间隔
                start_idx = test_dates.get_loc(rebalance_dates[i-1])
                end_idx = test_dates.get_loc(rebalance_dates[i])
                interval = end_idx - start_idx
                actual_intervals.append(interval)
            print(f"  Actual intervals: {actual_intervals} (expected: {int(freq[:-1])})")
    
    # 测试错误情况
    print("\n" + "="*50)
    print("Testing error cases:")
    error_cases = ['0D', '-2D', 'XD', '2X', '']
    for freq in error_cases:
        try:
            generate_rebalance_dates(test_dates, freq)
            print(f"  {freq}: UNEXPECTED SUCCESS")
        except ValueError as e:
            print(f"  {freq}: CORRECTLY FAILED - {e}")