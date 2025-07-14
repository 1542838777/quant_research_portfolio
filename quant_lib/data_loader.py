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
        
        logger.info("正在初始化DataLoader，开始扫描数据文件...")
        self.field_map = self._build_field_map()
        logger.info(f"字段映射构建完毕，共发现 {len(self.field_map)} 个字段")
    
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
                
                # 确定逻辑数据集名称
                if file_path.stem == 'data':
                    # 分区数据
                    logical_name = file_path.parent.parent.name
                else:
                    # 单文件
                    logical_name = file_path.stem
                
                # 构建字段映射
                for col in columns:
                    if col not in field_to_file_map:
                        field_to_file_map[col] = logical_name
            except Exception as e:
                logger.error(f"读取文件 {file_path} 的元数据失败: {e}")
        
        return field_to_file_map
    
    def load_data(self, 
                  fields: List[str], 
                  start_date: str, 
                  end_date: str,
                  universe: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        加载数据
        
        Args:
            fields: 需要加载的字段列表
            start_date: 开始日期
            end_date: 结束日期
            universe: 股票代码列表，如果为None则加载所有股票
            
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
        raw_dfs = {}
        for logical_name, columns_to_load in file_to_fields.items():
            try:
                file_path = self.data_path / logical_name
                
                # 加载数据
                long_df = pd.read_parquet(
                    file_path, 
                    columns=list(set(columns_to_load + base_fields))
                )
                
                # 时间筛选
                long_df['trade_date'] = pd.to_datetime(long_df['trade_date'])
                long_df = long_df[
                    (long_df['trade_date'] >= pd.Timestamp(start_date)) & 
                    (long_df['trade_date'] <= pd.Timestamp(end_date))
                ]
                
                # 股票池筛选
                if universe is not None:
                    long_df = long_df[long_df['ts_code'].isin(universe)]
                
                # 转换为宽表
                for col in columns_to_load:
                    if col not in base_fields:
                        wide_df = long_df.pivot_table(
                            index='trade_date', 
                            columns='ts_code', 
                            values=col
                        )
                        raw_dfs[col] = wide_df
            except Exception as e:
                logger.error(f"处理数据集 {logical_name} 失败: {e}")
        
        # 对齐数据
        aligned_data = self._align_dataframes(raw_dfs)
        
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
        
        # 对齐并填充缺失值
        aligned_data = {}
        for name, df in dfs.items():
            aligned_df = df.reindex(index=common_dates, columns=common_stocks)
            aligned_df = aligned_df.sort_index()
            aligned_df = aligned_df.ffill()  # 前向填充
            aligned_data[name] = aligned_df
        
        logger.info(f"数据对齐完成: {len(common_dates)}个交易日, {len(common_stocks)}只股票")
        return aligned_data
    
    def clear_cache(self):
        """清除缓存"""
        self.cache = {}
        logger.info("数据缓存已清除")


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