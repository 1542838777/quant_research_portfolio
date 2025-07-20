"""
数据加载模块

该模块提供了数据加载、处理和对齐的功能。
支持从本地文件、数据库和API加载数据。
"""

import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from collections import defaultdict
import logging

from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR
from quant_lib.utils import get_trading_dates

# 获取模块级别的logger
logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载器类
    
    负责从各种数据源加载数据，并进行预处理、对齐等操作。
    支持本地Parquet文件、数据库和API数据源。
    
    Attributes:
        data_path (Path): 数据存储路径
        cache (Dict): 数据缓存
        field_map (Dict): 字段到数据源的映射
    """

    def __init__(self, data_path: Optional[Path] = None, use_cache: bool = True):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据存储路径，如果为None则使用默认路径
            use_cache: 是否使用内存缓存
        """
        self.data_path = data_path or LOCAL_PARQUET_DATA_DIR
        self.use_cache = use_cache
        self.cache = {}

        if not self.data_path.exists():
            logger.warning(f"数据路径不存在: {self.data_path}")
            os.makedirs(self.data_path, exist_ok=True)
            logger.info(f"已创建数据路径: {self.data_path}")

        self.field_map = self._build_field_map()
        logger.info(f"字段->所在文件Name--映射构建完毕，共发现 {len(self.field_map)} 个字段")
        # 在初始化时就加载交易日历，因为它是后续操作的基础(此处还没区分是否open，是全部
        self.trade_cal = self._load_trade_cal()

    def _load_trade_cal(self) -> pd.DataFrame:
        """加载交易日历"""
        try:
            trade_cal_df = pd.read_parquet(self.data_path / 'trade_cal.parquet')
            trade_cal_df['cal_date'] = pd.to_datetime(trade_cal_df['cal_date'])
            return trade_cal_df
        except Exception as e:
            logger.error(f"加载交易日历失败: {e}")
            raise

    def get_trading_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """根据起止日期，从交易日历中获取交易日序列。"""
        mask = (self.trade_cal['cal_date'] >= start_date) & \
               (self.trade_cal['cal_date'] <= end_date) & \
               (self.trade_cal['is_open'] == 1)
        return pd.to_datetime(self.trade_cal[mask]['cal_date'].unique())

    def _build_field_map(self) -> Dict[str, str]:
        """
        构建字段到数据源的映射
        
        Returns:
            字段到数据源的映射字典
        """
        field_to_file_map = {}

        # 递归查找所有parquet文件
        for file_path in self.data_path.rglob('*.parquet'):
            try:
                # 只读取schema以获取列名
                columns = pq.read_schema(file_path).names

                # data.xxx 就是逻辑数据集名称（即：按年份分区的数据
                if file_path.stem == 'data':
                    # 分区数据
                    logical_name = file_path.parent.parent.name
                else:
                    # 单文件
                    logical_name = file_path.stem + '.parquet'

                # 构建字段映射
                for col in columns:
                    if (col == 'name') & (
                            logical_name == 'stock_basic.parquet'):  # 就是不要这里面的name ，我们需要namechange表里面的name 目前场景：用于过滤st开头的name股票
                        continue
                    if (col in ['close', 'open', 'high', 'low']) & (
                            logical_name != 'daily_hfq'):  # 就是不要这里面的close ，我们需要daily_hfq(后复权的数据)表里面的close
                        continue
                    if col not in field_to_file_map:
                        field_to_file_map[col] = logical_name
            except Exception as e:
                logger.error(f"读取文件 {file_path} 的元数据失败: {e}")

        return field_to_file_map

    def get_raw_dfs_by_require_fields(self,
                                      fields: List[str],
                                      start_date: str,
                                      end_date: str,
                                      ts_codes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        加载数据
        
        Args:
            fields: 需要加载的字段列表
            start_date: 开始日期
            end_date: 结束日期
            ts_codes: 股票代码列表，如果为None则加载所有股票
            
        Returns:
            字段到DataFrame的映射字典
        """
        logger.info(f"开始加载数据: 字段={fields}, 时间范围={start_date}至{end_date}")

        # 检查缓存
        cache_key = f"{','.join(sorted(fields))}-{start_date}-{end_date}"
        if self.use_cache and cache_key in self.cache:
            logger.info("从缓存加载数据")
            return self.cache[cache_key]

        # 确定需要加载的数据集和字段
        file_to_fields = defaultdict(list)
        base_fields = ['ts_code', 'trade_date']

        for field in list(set(fields + base_fields)):
            logical_name = self.field_map.get(field)
            if logical_name:
                file_to_fields[logical_name].append(field)
            else:
                logger.warning(f"未找到字段 {field} 的数据源")

        # 加载和处理数据
        raw_wide_dfs = {}  # 装 宽化的df
        raw_long_dfs = {}  # 原生的 从本地拿到的 key :文件，value：df（所有列！）
        for logical_name, columns_to_load in file_to_fields.items():
            try:
                file_path = self.data_path / logical_name

                # 检查文件中实际存在的字段
                available_columns = pd.read_parquet(file_path).columns

                # 只读取实际存在的字段
                columns_can_read = []
                for col in list(set(columns_to_load + base_fields)):
                    if col in available_columns:
                        columns_can_read.append(col)

                if not columns_can_read:
                    logger.warning(f"文件 {logical_name} 中没有找到任何需要的字段")
                    continue

                # 加载数据
                long_df = pd.read_parquet(
                    file_path,
                    columns=list(set(columns_can_read))
                )

                # 时间筛选
                long_df = self.extract_during_period(long_df, logical_name, start_date, end_date)

                # 股票池筛选
                if ts_codes is not None and 'ts_code' in long_df.columns:
                    long_df = long_df[long_df['ts_code'].isin(ts_codes)]

                raw_long_dfs[logical_name] = long_df

            except Exception as e:
                logger.error(f"处理数据集 {logical_name} 失败: {e}")
                raise ValueError(f"处理数据集 {logical_name} 失败: {e}")

        # --- 3. 将所有数据处理成统一的面板宽表格式 ---
        trading_dates = self.get_trading_dates(start_date, end_date)
        for field in sorted(fields):
            logical_name = self.field_map.get(field)
            if not logical_name or logical_name not in raw_long_dfs:
                logger.warning(f"未找到或加载失败: 字段 '{field}' 的数据源 '{logical_name}'")
                continue
            source_df = raw_long_dfs[logical_name]
            if 'trade_date' in source_df.columns:
                # a) 对于本身就是每日更新的面板数据
                df = source_df.copy()
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df[df['trade_date'].isin(trading_dates)]

                # 明确地定义重复的键
                duplicate_keys = ['trade_date', 'ts_code']

                # 在转换前，先使用 drop_duplicates 进行清洗
                # keep='last' 是一个重要的选择：我们假定文件末尾的记录是最新的、最准确的
                unique_long_df = df.drop_duplicates(subset=duplicate_keys, keep='last')

                # 确认没有重复项后，可以安全地进行转换
                #  此时可以直接使用 pivot()，它比 pivot_table() 略快，且能再次验证唯一性
                wide_df = unique_long_df.pivot(index='trade_date', columns='ts_code', values=field)
            else:
                # b) 对于需要“广播”到每日的静态属性数据 (如name, industry)
                logger.info(f"  正在将静态字段 '{field}' 广播到每日面板...")
                static_series = source_df.drop_duplicates(subset=['ts_code']).set_index('ts_code')[field]

                # #  方式（1）：直接广播
                # for ts_code in wide_df.columns:
                #     # 构造空 DataFrame，行是日期，列是股票代码
                #     wide_df = pd.DataFrame(index=pd.DatetimeIndex(trading_dates), columns=static_series.index)
                #     wide_df[ts_code] = static_series[ts_code]
                # 方式2 更高效 类似铺砖
                ##
                # np.tile(A, (M, 1)) = 把一行数组 A，重复 M 行，不重复列」
                #
                # 也就是说：
                #
                # M 控制的是“你有多少行”（行方向“铺砖”）
                #
                # 1 表示“列不要扩展”（只保留原来的股票维度）#
                wide_df = pd.DataFrame(
                    data=np.tile(static_series.values, (len(trading_dates), 1)),  # 使用numpy.tile高效复制数据
                    index=trading_dates,
                    columns=static_series.index
                )
            raw_wide_dfs[field] = wide_df

        # 对齐数据
        aligned_data = self._align_dataframes(raw_wide_dfs)

        # 更新缓存
        if self.use_cache:
            self.cache[cache_key] = aligned_data

        return aligned_data



    def _align_dataframes(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        对齐多个DataFrame
        
        Args:
            dfs: 字段到DataFrame的映射字典
            
        Returns:
            对齐后的DataFrame字典
        """
        if not dfs:
            logger.warning("没有数据需要对齐")
            return {}

        # 找出共同的日期和股票
        common_dates = None
        common_stocks = None

        for name, df in dfs.items():
            if common_dates is None:
                common_dates = df.index
                common_stocks = df.columns
            else:
                common_dates = common_dates.intersection(df.index)
                common_stocks = common_stocks.intersection(df.columns)

        # 对齐数据（不进行填充，保持原始缺失值）
        aligned_data = {}
        for name, df in dfs.items():
            aligned_df = df.reindex(index=common_dates, columns=common_stocks)
            aligned_df = aligned_df.sort_index()
            # 不进行填充，保持原始缺失值，上层DataManager配合universe决定填充策略
            aligned_data[name] = aligned_df

        logger.info(f"数据对齐完成: {len(common_dates)}个交易日, {len(common_stocks)}只股票")
        logger.info("注意：未进行缺失值填充，保持原始数据状态")
        return aligned_data

    def clear_cache(self):
        """清除缓存"""
        self.cache = {}
        logger.info("数据缓存已清除")

    def extract_during_period(self, long_df, logical_name, start_date, end_date):
        """
        根据时间范围筛选数据

        Args:
            long_df: 输入的DataFrame
            logical_name: 数据文件的逻辑名称
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            筛选后的DataFrame
        """
        if 'trade_date' in long_df.columns:
            long_df['trade_date'] = pd.to_datetime(long_df['trade_date'])
            long_df = long_df[
                (long_df['trade_date'] >= pd.Timestamp(start_date)) &
                (long_df['trade_date'] <= pd.Timestamp(end_date))
                ]
            return long_df
        # elif logical_name == 'stock_basic.parquet':
        #     # 对于股票基本信息，筛选上市日期早于开始日期的股票
        #     long_df['list_date'] = pd.to_datetime(long_df['list_date'])
        #     long_df = long_df[long_df['list_date'] < pd.Timestamp(start_date)]
        #
        #     # 添加交易日期列，便于数据统一处理
        #     trading_dates = get_trading_dates(start_date, end_date)#  待确认到底是 需要start_date end_date期间的交易日 ，还是连续的每日 确实需要这样！
        #     # 为每个股票创建所有交易日的记录
        #     stocks = long_df['ts_code'].unique()
        #     dates_df = pd.DataFrame(
        #         [(date, code) for date in trading_dates for code in stocks],
        #         columns=['trade_date', 'ts_code']
        #     )
        #     dates_df['trade_date'] = pd.to_datetime(dates_df['trade_date'])
        #
        #     # 合并基本信息到所有交易日
        #     result_df = pd.merge(dates_df, long_df, on='ts_code', how='left')
        #     return result_df

        return long_df  # 如果没有日期列，返回原始数据 反正后面有 对齐！


class DataProcessor:
    """
    数据处理器类
    
    提供各种数据处理功能，如去极值、标准化、中性化等。
    """

    @staticmethod
    def winsorize(df: pd.DataFrame,
                  lower_quantile: float = 0.025,
                  upper_quantile: float = 0.975) -> pd.DataFrame:
        """
        去极值处理
        
        Args:
            df: 输入DataFrame
            lower_quantile: 下限分位数
            upper_quantile: 上限分位数
            
        Returns:
            去极值后的DataFrame
        """
        result = df.copy()
        for date in result.index:
            row = result.loc[date]
            lower = row.quantile(lower_quantile)
            upper = row.quantile(upper_quantile)
            result.loc[date] = row.clip(lower, upper)
        return result

    @staticmethod
    def standardize(df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化处理
        
        Args:
            df: 输入DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        result = df.copy()
        for date in result.index:
            row = result.loc[date]
            result.loc[date] = (row - row.mean()) / row.std()
        return result

    @staticmethod
    def neutralize(factor_df: pd.DataFrame,
                   industry_df: pd.DataFrame,
                   market_cap_df: pd.DataFrame) -> pd.DataFrame:
        """
        行业市值中性化
        
        Args:
            factor_df: 因子DataFrame
            industry_df: 行业DataFrame
            market_cap_df: 市值DataFrame
            
        Returns:
            中性化后的因子DataFrame
        """
        # 实现中性化逻辑
        # 这里只是一个简单的示例，实际中可能需要更复杂的实现
        result = factor_df.copy()

        # 对每一个日期进行处理
        for date in result.index:
            # 获取当天的数据
            factor = result.loc[date].dropna()
            stocks = factor.index

            # 获取对应的行业和市值数据
            ind = industry_df.loc[date, stocks].dropna()
            mcap = market_cap_df.loc[date, stocks].dropna()

            # 找出三者都有数据的股票
            common_stocks = ind.index.intersection(mcap.index).intersection(factor.index)

            if len(common_stocks) > 0:
                # 构建回归矩阵
                # 实际实现中应该使用更高效的方法
                pass

        return result


# 工厂函数，方便创建DataLoader实例
def create_data_loader(data_path: Optional[Path] = None, use_cache: bool = True) -> DataLoader:
    """
    创建DataLoader实例
    
    Args:
        data_path: 数据路径
        use_cache: 是否使用缓存
        
    Returns:
        DataLoader实例
    """
    return DataLoader(data_path=data_path, use_cache=use_cache)
