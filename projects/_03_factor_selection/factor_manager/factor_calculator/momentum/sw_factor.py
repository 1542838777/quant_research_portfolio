import pandas as pd

from data.local_data_load import load_sw_l_n_daily
from projects._03_factor_selection.utils.IndustryMap import PointInTimeIndustryMap


# 假设这些是你已经有的模块/函数
# from your_modules import PointInTimeIndustryMap, load_sw_daily, get_trade_dates

class IndustryMomentumFactor:
    """
    计算并生成行业动量因子（申万一级，21日）。

    这个类将整个计算流程封装起来，包括：
    1. 计算申万一级行业的原始动量。
    2. 使用Point-in-Time正确的行业成分数据，将行业动量映射到个股上。
    """

    def __init__(self, point_in_time_mapper: PointInTimeIndustryMap, period: int = 21):
        """
        初始化因子计算器。

        Args:
            point_in_time_mapper (PointInTimeIndustryMap): 你已经实现的、包含历史行业信息的查询工具。
            period (int, optional): 计算动量的时间窗口。默认为 21。
        """
        if not hasattr(point_in_time_mapper, 'get_map_for_date'):
            raise TypeError("传入的'point_in_time_mapper'对象必须包含'get_map_for_date'方法。")

        self.mapper = point_in_time_mapper
        self.period = period

    def _compute_sw_momentum(self, l_n_code) -> pd.DataFrame:
        """
        私有方法：获取并计算所有申万l级行业在指定时间段内的动量因子。
        """
        print("步骤1：正在计算所有申万一级行业的历史动量因子...")

        # 假设 load_sw_daily() 会加载所需日期的全部数据
        industry_daily_df = load_sw_l_n_daily(l_n_code)

        # industry_daily_df['trade_date'] = pd.to_datetime(industry_daily_df['trade_date'])

        # 计算动量
        momentum_series = industry_daily_df.groupby(l_n_code)['close'].pct_change(periods=self.period)
        factor_name = f'sw_{l_n_code}_momentum_{self.period}d'

        industry_daily_df[factor_name] = momentum_series

        result_df = industry_daily_df.set_index(['trade_date', l_n_code])[[factor_name]].dropna()

        return result_df

    def _map_to_stocks(self, industry_momentum_df: pd.DataFrame, l_n_code: str, trade_dates: pd.Series) -> pd.DataFrame:
        """
        私有方法：将计算好的行业动量，逐日映射到对应的成分股上。

        Args:
            industry_momentum_df (pd.DataFrame): 行业动量因子数据。
            trade_dates (pd.Series): 需要计算因子的交易日序列。

        Returns:
            pd.DataFrame: 最终的个股因子数据，索引为 (trade_date, ts_code)。
        """
        print("步骤2：开始将行业动量逐日映射到个股...")
        final_factor_data_list = []

        for date in trade_dates:
            try:
                # 1. 获取当天的行业动量
                daily_industry_momentum = industry_momentum_df.loc[date]
            except KeyError:
                # 如果当天没有行业行情数据，则跳过
                continue

            # 2. 获取当天的股票->行业映射
            daily_stock_map = self.mapper.get_map_for_date(date)
            if daily_stock_map.empty:
                continue

            # 3. 关键的合并步骤
            merged_df = pd.merge(
                left=daily_stock_map.reset_index(),
                right=daily_industry_momentum,
                on=l_n_code,
                how='left'
            )

            # 4. 整理格式并加入列表
            merged_df['trade_date'] = date
            final_factor_data_list.append(merged_df)

        if not final_factor_data_list:
            raise ValueError("警告：在指定日期范围内未能生成任何因子数据。")

        final_factor_df = pd.concat(final_factor_data_list)
        factor_name = f'sw_{l_n_code}_momentum_{self.period}d'

        ret = pd.pivot_table(final_factor_df,index='trade_date',columns='ts_code',values= factor_name)
        return ret

    def compute(self, dates: pd.DatetimeIndex, l_n_code) -> pd.DataFrame:
        """
        公开方法：执行完整的因子计算流程。

        Args:
            start_date (str): 回测开始日期 (YYYYMMDD)。
            end_date (str): 回测结束日期 (YYYYMMDD)。

        Returns:
            pd.DataFrame: 最终的个股因子数据。
        """

        # 获取回测区间的交易日历
        trade_dates = dates

        # 步骤1：计算行业动量
        industry_momentum_df = self._compute_sw_momentum(l_n_code)

        # 步骤2：映射到个股
        stock_factor_df = self._map_to_stocks(industry_momentum_df, l_n_code, trade_dates)

        print("所有计算流程完成。")
        return stock_factor_df
