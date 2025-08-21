import random

import pandas as pd
from vectorbt.utils.docs import to_doc

from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR, permanent__day
from quant_lib.utils.test import check_step

def load_index_daily(index_code):
    index_daily = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'index_daily.parquet')
    index_daily = index_daily[index_daily['ts_code'] == index_code]
    index_daily['trade_date'] = pd.to_datetime(index_daily['trade_date'])
    # set_index() 也将创建一个 DatetimeIndex，这对于时间序列分析至关重要
    index_df = index_daily.sort_values(by='trade_date').set_index('trade_date')
    return index_df
def load_trading_lists(start,end):
    # 获取该年度所有交易日
    trade_cal = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'trade_cal.parquet')
    trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'])
    trade_dates = trade_cal[
        (trade_cal['cal_date'] >= pd.to_datetime(start)) &
        (trade_cal['cal_date'] <= pd.to_datetime(end)) &
        (trade_cal['is_open'] == 1)
        ]['cal_date'].tolist()
    return sorted(trade_dates)


def load_all_stock_codes():
    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'stock_basic.parquet')
    return list(df['ts_code'].unique())


def load_cashflow_df():
    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'cashflow.parquet')
    df['ann_date'] = pd.to_datetime(df['ann_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df = df.sort_values(by=['ts_code', 'end_date', 'update_flag'], ascending=[True, True, False]).drop_duplicates(
        subset=['ts_code', 'end_date'], keep='first')
    # 随机取5个股票的df
    # df = df[df['ts_code']. isin (['000001.SZ',
    #                           '600439.SH',
    #                           '600461.SH',
    #                           '600610.SH'])]
    df = df[['ann_date', 'ts_code', 'end_date', 'n_cashflow_act']]
    return df


def load_income_df():
    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'income.parquet')
    df['ann_date'] = pd.to_datetime(df['ann_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df = df.sort_values(by=['ts_code', 'end_date', 'update_flag'], ascending=[True, True, False]).drop_duplicates(
        subset=['ts_code', 'end_date'], keep='first')
    #                                              确认过:单位:元         元
    df = df[['ann_date', 'ts_code', 'end_date', 'n_income_attr_p','total_revenue','oper_cost']]
    return df

def load_balancesheet_df():
    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'balancesheet.parquet')
    df['ann_date'] = pd.to_datetime(df['ann_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df = df.sort_values(by=['ts_code', 'end_date', 'update_flag'], ascending=[True, True, False]).drop_duplicates(
        subset=['ts_code', 'end_date'], keep='first')

    # df =  df[df['ts_code'].isin(['000001.SZ','000002.SZ','000003.SZ'])]
    df = df[['ann_date', 'ts_code', 'end_date', 'total_hldr_eqy_exc_min_int','total_assets','total_liab']]
    return df


def load_dividend_events_long():
    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'dividend.parquet')
    df['ex_date'] = pd.to_datetime(df['ex_date'])
    df['ann_date'] = pd.to_datetime(df['ann_date'])
    df = df.drop_duplicates()
    df = df.sort_values(by=['ex_date'], inplace=False)
    # df =  df[df['ts_code'].isin(['000001.SZ','000002.SZ','000003.SZ'])]
    return df


def load_suspend_d_df():
    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'suspend_d.parquet')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df.sort_values(by=['trade_date'], ascending=[True], inplace=False)


def get_trading_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    根据起止日期，从交易日历中获取交易日序列。
    此版本通过将所有日期转换为datetime对象，避免了因字符串格式不匹配导致的错误。
    Args:
        start_date (str): 开始日期, 接受 'YYYYMMDD' 或 'YYYY-MM-DD' 等常见格式.
        end_date (str): 结束日期, 接受 'YYYYMMDD' 或 'YYYY-MM-DD' 等常见格式.
    Returns:
        pd.DatetimeIndex: 一个包含了所有在指定范围内的交易日的DatetimeIndex.
    """
    try:
        # --- 步骤1：读取并转换日历数据 ---
        # 假设 trade_cal.parquet 存在且包含 'cal_date' 和 'is_open' 列
        trade_cal = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'trade_cal.parquet')  # 请替换为你的实际路径

        # 【核心修正 1】: 立即将文件中的日期列转换为datetime对象
        trade_cal['cal_date_dt'] = pd.to_datetime(trade_cal['cal_date'], errors='raise')

        # 删除转换失败的行，增加健壮性
        trade_cal= trade_cal.dropna(subset=['cal_date_dt'], inplace=False)

        # --- 步骤2：转换输入的起止日期 ---
        # 【核心修正 2】: 同样将输入的字符串参数转换为datetime对象
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # --- 步骤3：使用datetime对象进行筛选 ---
        mask = (trade_cal['cal_date_dt'] >= start_dt) & \
               (trade_cal['cal_date_dt'] <= end_dt) & \
               (trade_cal['is_open'] == 1)

        # 从筛选后的结果中提取原始的日期列（或转换后的dt列），并确保唯一性
        # 返回一个标准的DatetimeIndex，这是后续时间序列操作的最佳格式
        return pd.DatetimeIndex(sorted(trade_cal[mask]['cal_date_dt'].unique()))

    except FileNotFoundError:
        raise ValueError(f"错误: 交易日历文件 trade_cal.parquet 未找到。")
    except Exception as e:
        raise ValueError(f"获取交易日时发生未知错误: {e}")

