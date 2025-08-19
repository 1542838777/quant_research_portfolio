"""
因子工厂模块

提供各类因子的计算、处理和组合功能。
支持技术因子、基本面因子、情绪因子等多种因子类型。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
import logging
from abc import ABC, abstractmethod

from quant_lib.config.logger_config import log_warning

# 获取模块级别的logger
logger = logging.getLogger(__name__)


class BaseFactor(ABC):
    """因子基类"""
    
    def __init__(self, name: str):
        """
        初始化因子
        
        Args:
            name: 因子名称
        """
        self.name = name
    
    @abstractmethod
    def compute(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算因子值
        
        Args:
            data_dict: 输入数据字典，键为数据名称，值为DataFrame
            
        Returns:
            因子值DataFrame
        """
        pass


class ValueFactor(BaseFactor):
    """价值因子"""
    
    def __init__(self, name: str = 'value'):
        """初始化价值因子"""
        super().__init__(name)
    
    def compute(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算价值因子
        
        Args:
            data_dict: 包含 'pe_ttm', 'pb', 'ps_ttm' 等数据的字典
            
        Returns:
            价值因子值
        """
        logger.info("计算价值因子...")
        
        # 检查必要的数据是否存在
        required_fields = ['pe_ttm', 'pb']
        for field in required_fields:
            if field not in data_dict:
                logger.error(f"缺少计算价值因子所需的数据: {field}")
                return pd.DataFrame()
        
        # 倒数处理
        pe_inv = 1 / data_dict['pe_ttm']
        pb_inv = 1 / data_dict['pb']
        
        # 去除异常值
        pe_inv = pe_inv.replace([np.inf, -np.inf], np.nan)
        pb_inv = pb_inv.replace([np.inf, -np.inf], np.nan)
        
        # 标准化处理
        pe_inv_std = self._standardize_by_date(pe_inv)
        pb_inv_std = self._standardize_by_date(pb_inv)
        
        # 合成因子
        value_factor = (pe_inv_std + pb_inv_std) / 2
        
        logger.info("价值因子计算完成")
        return value_factor
    
    def _standardize_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """按日期标准化"""
        result = df.copy()
        for date in result.index:
            row = result.loc[date]
            mean = row.mean()
            std = row.std()
            if std > 0:
                result.loc[date] = (row - mean) / std
        return result


class MomentumFactor(BaseFactor):
    """动量因子"""
    
    def __init__(self, 
                name: str = 'momentum',
                lookback_periods: List[int] = [20, 60, 120]):
        """
        初始化动量因子
        
        Args:
            name: 因子名称
            lookback_periods: 回溯期列表，单位为交易日
        """
        super().__init__(name)
        self.lookback_periods = lookback_periods

    def _standardize_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """按日期标准化"""
        result = df.copy()
        for date in result.index:
            row = result.loc[date]
            mean = row.mean()
            std = row.std()
            if std > 0:
                result.loc[date] = (row - mean) / std
        return result


class VolatilityFactor(BaseFactor):
    """波动率因子"""
    
    def __init__(self, 
                name: str = 'volatility',
                window: int = 20):
        """
        初始化波动率因子
        
        Args:
            name: 因子名称
            window: 计算窗口大小
        """
        super().__init__(name)
        self.window = window
    
    def compute(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算波动率因子
        
        Args:
            data_dict: 包含 'close' 数据的字典
            
        Returns:
            波动率因子值
        """
        logger.info("计算波动率因子...")
        
        # 检查必要的数据是否存在
        if 'close_raw' not in data_dict:
            logger.error("缺少计算波动率因子所需的数据: close")
            return pd.DataFrame()
        
        close = data_dict['close_raw']
        
        # 计算日收益率
        daily_returns = close / close.shift(1) - 1
        
        # 计算滚动波动率
        volatility = daily_returns.rolling(window=self.window).std()
        
        # 标准化处理，并取负值（低波动率更好）
        volatility_factor = -self._standardize_by_date(volatility)
        
        logger.info("波动率因子计算完成")
        return volatility_factor
    
    def _standardize_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """按日期标准化"""
        result = df.copy()
        for date in result.index:
            row = result.loc[date]
            mean = row.mean()
            std = row.std()
            if std > 0:
                result.loc[date] = (row - mean) / std
        return result


class GrowthFactor(BaseFactor):
    """成长因子"""
    
    def __init__(self, name: str = 'growth'):
        """初始化成长因子"""
        super().__init__(name)
    
    def compute(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算成长因子
        
        Args:
            data_dict: 包含 'netprofit_yoy', 'revenue_yoy' 等数据的字典
            
        Returns:
            成长因子值
        """
        logger.info("计算成长因子...")
        
        # 检查必要的数据是否存在
        required_fields = ['netprofit_yoy', 'revenue_yoy']
        for field in required_fields:
            if field not in data_dict:
                logger.error(f"缺少计算成长因子所需的数据: {field}")
                return pd.DataFrame()
        
        # 标准化处理
        profit_growth_std = self._standardize_by_date(data_dict['netprofit_yoy'])
        revenue_growth_std = self._standardize_by_date(data_dict['revenue_yoy'])
        
        # 合成因子
        growth_factor = (profit_growth_std + revenue_growth_std) / 2
        
        logger.info("成长因子计算完成")
        return growth_factor
    
    def _standardize_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """按日期标准化"""
        result = df.copy()
        for date in result.index:
            row = result.loc[date]
            mean = row.mean()
            std = row.std()
            if std > 0:
                result.loc[date] = (row - mean) / std
        return result


class QualityFactor(BaseFactor):
    """质量因子"""
    
    def __init__(self, name: str = 'quality'):
        """初始化质量因子"""
        super().__init__(name)
    
    def compute(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算质量因子
        
        Args:
            data_dict: 包含 'roe', 'grossprofit_margin', 'debt_to_assets' 等数据的字典
            
        Returns:
            质量因子值
        """
        logger.info("计算质量因子...")
        
        # 检查必要的数据是否存在
        required_fields = ['roe', 'grossprofit_margin', 'debt_to_assets']
        for field in required_fields:
            if field not in data_dict:
                logger.error(f"缺少计算质量因子所需的数据: {field}")
                return pd.DataFrame()
        
        # 标准化处理
        roe_std = self._standardize_by_date(data_dict['roe'])
        margin_std = self._standardize_by_date(data_dict['grossprofit_margin'])
        debt_std = -self._standardize_by_date(data_dict['debt_to_assets'])  # 负债率越低越好
        
        # 合成因子
        quality_factor = (roe_std + margin_std + debt_std) / 3
        
        logger.info("质量因子计算完成")
        return quality_factor
    
    def _standardize_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """按日期标准化"""
        result = df.copy()
        for date in result.index:
            row = result.loc[date]
            mean = row.mean()
            std = row.std()
            if std > 0:
                result.loc[date] = (row - mean) / std
        return result


class FactorCombiner:
    """因子组合器"""
    
    def __init__(self, factors: Dict[str, BaseFactor], weights: Dict[str, float]):
        """
        初始化因子组合器
        
        Args:
            factors: 因子字典，键为因子名称，值为因子对象
            weights: 权重字典，键为因子名称，值为权重
        """
        self.factors = factors
        self.weights = weights
        
        # 检查权重是否合法
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            log_warning(f"权重和不为1: {total_weight}，将进行归一化处理")
            for factor_name in weights:
                self.weights[factor_name] /= total_weight
    
    def compute(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算组合因子
        
        Args:
            data_dict: 输入数据字典
            
        Returns:
            组合因子值
        """
        logger.info("计算组合因子...")
        
        # 计算各个因子
        factor_values = {}
        for factor_name, factor in self.factors.items():
            factor_values[factor_name] = factor.compute(data_dict)
        
        # 检查是否有因子计算失败
        for factor_name, factor_df in factor_values.items():
            if factor_df.empty:
                logger.error(f"因子 {factor_name} 计算失败")
                return pd.DataFrame()
        
        # 组合因子
        combined_factor = None
        for factor_name, factor_df in factor_values.items():
            weight = self.weights.get(factor_name, 0)
            if weight > 0:
                if combined_factor is None:
                    combined_factor = factor_df * weight
                else:
                    combined_factor += factor_df * weight
        
        logger.info("组合因子计算完成")
        return combined_factor


# 工厂函数，方便创建各类因子
def create_factor(factor_type: str, **kwargs) -> BaseFactor:
    """
    创建因子对象
    
    Args:
        factor_type: 因子类型，'value', 'momentum', 'volatility', 'growth', 'quality'
        **kwargs: 其他参数
        
    Returns:
        因子对象
    """
    if factor_type == 'value':
        return ValueFactor(**kwargs)
    elif factor_type == 'momentum':
        return MomentumFactor(**kwargs)
    elif factor_type == 'volatility':
        return VolatilityFactor(**kwargs)
    elif factor_type == 'growth':
        return GrowthFactor(**kwargs)
    elif factor_type == 'quality':
        return QualityFactor(**kwargs)
    else:
        raise ValueError(f"不支持的因子类型: {factor_type}")


def create_factor_combiner(factor_types: List[str], 
                          weights: List[float], 
                          **kwargs) -> FactorCombiner:
    """
    创建因子组合器
    
    Args:
        factor_types: 因子类型列表
        weights: 权重列表
        **kwargs: 其他参数
        
    Returns:
        因子组合器对象
    """
    if len(factor_types) != len(weights):
        raise ValueError("因子类型列表和权重列表长度不一致")
    
    factors = {}
    weights_dict = {}
    
    for i, factor_type in enumerate(factor_types):
        factor_name = factor_type
        factors[factor_name] = create_factor(factor_type, **kwargs)
        weights_dict[factor_name] = weights[i]
    
    return FactorCombiner(factors, weights_dict)