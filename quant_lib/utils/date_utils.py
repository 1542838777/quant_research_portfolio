"""
日期工具模块

提供交易日期相关的工具函数。
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
import datetime
import os
from pathlib import Path
import logging

from quant_lib.config.constant_config import DATA_DIR
from quant_lib.tushare.api_wrapper import call_pro_tushare_api

# 获取模块级别的logger
logger = logging.getLogger(__name__)

# 交易日历缓存
_TRADING_CALENDAR = None


def _load_or_update_trading_calendar() -> pd.DataFrame:
    """
    加载或更新交易日历
    
    Returns:
        交易日历DataFrame
    """
    global _TRADING_CALENDAR
    
    # 检查是否已加载
    if _TRADING_CALENDAR is not None:
        return _TRADING_CALENDAR
    
    # 缓存文件路径
    calendar_file = DATA_DIR / 'trading_calendar.pkl'
    
    # 检查缓存文件是否存在
    if os.path.exists(calendar_file):
        try:
            calendar_df = pd.read_pickle(calendar_file)
            
            # 检查是否需要更新
            today = datetime.datetime.now().strftime('%Y%m%d')
            max_date = calendar_df['cal_date'].max()
            
            if max_date >= today:
                _TRADING_CALENDAR = calendar_df
                return _TRADING_CALENDAR
        except Exception as e:
            raise ValueError(f"读取交易日历缓存失败: {e}")
    
    # 从Tushare获取交易日历
    try:
        start_date = '19900101'
        end_date = (datetime.datetime.now() + datetime.timedelta(days=365)).strftime('%Y%m%d')
        
        calendar_df = call_pro_tushare_api(
            'trade_cal',
            exchange='SSE',
            start_date=start_date,
            end_date=end_date,
            fields='cal_date,is_open'
        )
        
        # 保存到缓存
        os.makedirs(os.path.dirname(calendar_file), exist_ok=True)
        calendar_df.to_pickle(calendar_file)
        
        _TRADING_CALENDAR = calendar_df
        return _TRADING_CALENDAR
    except Exception as e:
        logger.error(f"获取交易日历失败: {e}")
        
        # 如果缓存文件存在，尝试使用缓存
        if os.path.exists(calendar_file):
            try:
                calendar_df = pd.read_pickle(calendar_file)
                _TRADING_CALENDAR = calendar_df
                return _TRADING_CALENDAR
            except:
                pass
        
        # 创建一个空的交易日历
        return pd.DataFrame(columns=['cal_date', 'is_open'])


def get_trading_dates(start_date: str, end_date: str) -> List[str]:
    """
    获取指定日期范围内的交易日列表
    
    Args:
        start_date: 开始日期，格式为'YYYYMMDD'或'YYYY-MM-DD'
        end_date: 结束日期，格式为'YYYYMMDD'或'YYYY-MM-DD'
        
    Returns:
        交易日列表
    """
    # 标准化日期格式
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')
    
    # 加载交易日历
    calendar = _load_or_update_trading_calendar()
    
    # 筛选交易日
    trading_days = calendar[
        (calendar['cal_date'] >= start_date) & 
        (calendar['cal_date'] <= end_date) & 
        (calendar['is_open'] == 1)
    ]['cal_date'].tolist()
    
    # 转换为'YYYY-MM-DD'格式
    trading_days = [f"{day[:4]}-{day[4:6]}-{day[6:]}" for day in trading_days]
    
    return trading_days




def get_previous_trading_day(date: str, n: int = 1) -> str:
    """
    获取指定日期前n个交易日
    
    Args:
        date: 日期，格式为'YYYYMMDD'或'YYYY-MM-DD'
        n: 前n个交易日
        
    Returns:
        前n个交易日，格式为'YYYY-MM-DD'
    """
    # 标准化日期格式
    date = date.replace('-', '')
    
    # 加载交易日历
    calendar = _load_or_update_trading_calendar()
    
    # 筛选交易日
    trading_days = calendar[calendar['is_open'] == 1]['cal_date'].tolist()
    trading_days.sort()
    
    # 找到指定日期在交易日列表中的位置
    try:
        idx = trading_days.index(date)
    except ValueError:
        # 如果指定日期不是交易日，找到下一个交易日
        for i, day in enumerate(trading_days):
            if day > date:
                idx = i
                break
        else:
            return None
    
    # 获取前n个交易日
    if idx - n >= 0:
        prev_day = trading_days[idx - n]
        return f"{prev_day[:4]}-{prev_day[4:6]}-{prev_day[6:]}"
    else:
        return None


def get_next_trading_day(date: str, n: int = 1) -> str:
    """
    获取指定日期后n个交易日
    
    Args:
        date: 日期，格式为'YYYYMMDD'或'YYYY-MM-DD'
        n: 后n个交易日
        
    Returns:
        后n个交易日，格式为'YYYY-MM-DD'
    """
    # 标准化日期格式
    date = date.replace('-', '')
    
    # 加载交易日历
    calendar = _load_or_update_trading_calendar()
    
    # 筛选交易日
    trading_days = calendar[calendar['is_open'] == 1]['cal_date'].tolist()
    trading_days.sort()
    
    # 找到指定日期在交易日列表中的位置
    try:
        idx = trading_days.index(date)
    except ValueError:
        # 如果指定日期不是交易日，找到下一个交易日
        for i, day in enumerate(trading_days):
            if day > date:
                idx = i - 1
                break
        else:
            return None
    
    # 获取后n个交易日
    if idx + n < len(trading_days):
        next_day = trading_days[idx + n]
        return f"{next_day[:4]}-{next_day[4:6]}-{next_day[6:]}"
    else:
        return None


def get_month_end_dates(start_date: str, end_date: str) -> List[str]:
    """
    获取指定日期范围内的月末交易日列表
    
    Args:
        start_date: 开始日期，格式为'YYYYMMDD'或'YYYY-MM-DD'
        end_date: 结束日期，格式为'YYYYMMDD'或'YYYY-MM-DD'
        
    Returns:
        月末交易日列表，格式为'YYYY-MM-DD'
    """
    # 获取交易日列表
    trading_dates = get_trading_dates(start_date, end_date)
    
    # 转换为datetime对象
    dates = pd.to_datetime(trading_dates)
    
    # 获取月末日期
    month_end_dates = []
    for i in range(len(dates)):
        # 如果是最后一个日期或者当前日期的月份不等于下一个日期的月份，则为月末
        if i == len(dates) - 1 or dates[i].month != dates[i+1].month:
            month_end_dates.append(trading_dates[i])
    
    return month_end_dates


