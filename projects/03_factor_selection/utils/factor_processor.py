"""
因子预处理流水线 - 单因子测试终极作战手册
第三阶段：因子预处理

实现完整的因子预处理流水线：
1. 去极值 (Winsorization)
2. 中性化 (Neutralization) 
3. 标准化 (Standardization)
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Optional, Tuple, Any
import warnings
from sklearn.linear_model import LinearRegression
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

class FactorProcessor:
    """
    因子预处理器 - 专业级因子预处理流水线
    
    按照华泰证券标准实现：
    1. 去极值：中位数绝对偏差法(MAD) / 分位数法
    2. 中性化：行业中性化 + 市值中性化
    3. 标准化：Z-Score标准化 / 排序标准化
    """
    
    def __init__(self, config: Dict):
        """
        初始化因子预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        
    def process_factor(self, 
                      factor_data: pd.DataFrame,
                      universe_df: pd.DataFrame,
                      auxiliary_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        完整的因子预处理流水线
        
        Args:
            factor_data: 原始因子数据
            universe_df: 股票池数据
            auxiliary_data: 辅助数据（市值、行业等）
            
        Returns:
            预处理后的因子数据
        """
        print("\n" + "="*60)
        print("第三阶段：因子预处理流水线")
        print("="*60)
        
        # 应用股票池过滤
        print("1. 应用股票池过滤...")
        processed_factor = factor_data.where(universe_df)
        
        # 步骤1：去极值
        print("2. 去极值处理...")
        processed_factor = self._winsorize(processed_factor)
        
        # 步骤2：中性化
        if self.preprocessing_config.get('neutralization', {}).get('enable', False):
            print("3. 中性化处理...")
            processed_factor = self._neutralize(processed_factor, auxiliary_data)
        else:
            print("3. 跳过中性化处理...")
        
        # 步骤3：标准化
        print("4. 标准化处理...")
        processed_factor = self._standardize(processed_factor)
        
        # 统计处理结果
        self._print_processing_stats(factor_data, processed_factor, universe_df)
        
        return processed_factor
    
    def _winsorize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        去极值处理
        
        Args:
            factor_data: 因子数据
            
        Returns:
            去极值后的因子数据
        """
        winsorization_config = self.preprocessing_config.get('winsorization', {})
        method = winsorization_config.get('method', 'mad')
        
        processed_factor = factor_data.copy()
        
        if method == 'mad':
            # 中位数绝对偏差法 (Median Absolute Deviation)
            threshold = winsorization_config.get('mad_threshold', 5)
            print(f"  使用MAD方法，阈值倍数: {threshold}")
            
            for date in processed_factor.index:
                values = processed_factor.loc[date].dropna()
                if len(values) > 10:
                    # 计算中位数和中位数绝对偏差
                    median = values.median()
                    mad = (values - median).abs().median()
                    
                    # 计算上下界
                    upper_bound = median + threshold * mad
                    lower_bound = median - threshold * mad
                    
                    # 应用去极值
                    processed_factor.loc[date] = values.clip(lower_bound, upper_bound)
                    
        elif method == 'quantile':
            # 分位数法
            quantile_range = winsorization_config.get('quantile_range', [0.01, 0.99])
            print(f"  使用分位数方法，范围: {quantile_range}")
            
            for date in processed_factor.index:
                values = processed_factor.loc[date].dropna()
                if len(values) > 10:
                    lower_bound = values.quantile(quantile_range[0])
                    upper_bound = values.quantile(quantile_range[1])
                    processed_factor.loc[date] = values.clip(lower_bound, upper_bound)
        
        return processed_factor
    
    def _neutralize(self, 
                   factor_data: pd.DataFrame,
                   auxiliary_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        中性化处理
        
        Args:
            factor_data: 因子数据
            auxiliary_data: 辅助数据
            
        Returns:
            中性化后的因子数据
        """
        neutralization_config = self.preprocessing_config.get('neutralization', {})
        factors_to_neutralize = neutralization_config.get('factors', [])
        
        if not auxiliary_data:
            print("  警告: 缺少辅助数据，跳过中性化处理")
            return factor_data
        
        processed_factor = factor_data.copy()
        
        for date in processed_factor.index:
            # 获取当天的因子值
            y = processed_factor.loc[date].dropna()
            if len(y) < 20:  # 至少需要20个样本
                continue
            
            # 构建回归变量
            X_list = []
            feature_names = []
            
            # 市值中性化
            if 'market_cap' in factors_to_neutralize and 'total_mv' in auxiliary_data:
                mv_data = auxiliary_data['total_mv']
                if date in mv_data.index:
                    mv_values = mv_data.loc[date, y.index].dropna()
                    if len(mv_values) > 0:
                        # 取对数
                        log_mv = np.log(mv_values)
                        X_list.append(log_mv)
                        feature_names.append('log_market_cap')
            
            # 行业中性化
            if 'industry' in factors_to_neutralize and 'industry' in auxiliary_data:
                industry_data = auxiliary_data['industry']
                if date in industry_data.index:
                    industry_values = industry_data.loc[date, y.index].dropna()
                    if len(industry_values) > 0:
                        # 创建行业哑变量
                        industry_dummies = pd.get_dummies(industry_values, prefix='industry')
                        # 去掉一个行业避免共线性
                        if industry_dummies.shape[1] > 1:
                            industry_dummies = industry_dummies.iloc[:, :-1]
                        
                        for col in industry_dummies.columns:
                            X_list.append(industry_dummies[col])
                            feature_names.append(col)
            
            # 执行回归
            if X_list:
                try:
                    # 对齐数据
                    common_index = y.index
                    for x in X_list:
                        common_index = common_index.intersection(x.index)
                    
                    if len(common_index) > 10:
                        # 构建回归矩阵
                        X_matrix = pd.DataFrame(index=common_index)
                        for i, x in enumerate(X_list):
                            X_matrix[feature_names[i]] = x.loc[common_index]
                        
                        y_aligned = y.loc[common_index]
                        
                        # 去除缺失值
                        valid_mask = X_matrix.notna().all(axis=1) & y_aligned.notna()
                        if valid_mask.sum() > 10:
                            X_clean = X_matrix[valid_mask]
                            y_clean = y_aligned[valid_mask]
                            
                            # 线性回归
                            reg = LinearRegression()
                            reg.fit(X_clean, y_clean)
                            
                            # 计算残差
                            y_pred = reg.predict(X_clean)
                            residuals = y_clean - y_pred
                            
                            # 更新因子值为残差
                            processed_factor.loc[date, residuals.index] = residuals
                            
                except Exception as e:
                    print(f"  警告: 日期 {date} 中性化失败: {e}")
                    continue
        
        print(f"  中性化完成，处理因子: {factors_to_neutralize}")
        return processed_factor
    
    def _standardize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化处理
        
        Args:
            factor_data: 因子数据
            
        Returns:
            标准化后的因子数据
        """
        standardization_config = self.preprocessing_config.get('standardization', {})
        method = standardization_config.get('method', 'zscore')
        
        processed_factor = factor_data.copy()
        
        if method == 'zscore':
            print("  使用Z-Score标准化")
            for date in processed_factor.index:
                values = processed_factor.loc[date].dropna()
                if len(values) > 1 and values.std() > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    processed_factor.loc[date, values.index] = (values - mean_val) / std_val
                    
        elif method == 'rank':
            print("  使用排序标准化")
            for date in processed_factor.index:
                values = processed_factor.loc[date].dropna()
                if len(values) > 1:
                    # 排序并标准化到[-1, 1]
                    ranks = values.rank()
                    normalized_ranks = 2 * (ranks - 1) / (len(ranks) - 1) - 1
                    processed_factor.loc[date, values.index] = normalized_ranks
        
        return processed_factor
    
    def _print_processing_stats(self, 
                              original_factor: pd.DataFrame,
                              processed_factor: pd.DataFrame,
                              universe_df: pd.DataFrame):
        """打印处理统计信息"""
        print("\n因子预处理统计:")
        
        # 原始因子统计
        orig_valid = original_factor.notna().sum().sum()
        orig_total = original_factor.shape[0] * original_factor.shape[1]
        
        # 处理后因子统计
        proc_valid = processed_factor.notna().sum().sum()
        
        # 股票池统计
        universe_valid = universe_df.sum().sum()
        
        print(f"  原始因子有效值: {orig_valid:,} / {orig_total:,} ({orig_valid/orig_total:.1%})")
        print(f"  处理后有效值: {proc_valid:,} / {universe_valid:,} ({proc_valid/universe_valid:.1%})")
        
        # 分布统计
        all_values = processed_factor.values.flatten()
        all_values = all_values[~np.isnan(all_values)]
        
        if len(all_values) > 0:
            print(f"  处理后分布: 均值={all_values.mean():.3f}, 标准差={all_values.std():.3f}")
            print(f"  分位数: 1%={np.percentile(all_values, 1):.3f}, "
                  f"99%={np.percentile(all_values, 99):.3f}")


class FactorCalculator:
    """
    因子计算器 - 根据配置计算目标因子
    """
    
    def __init__(self, config: Dict):
        """
        初始化因子计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.target_factor_config = config.get('target_factor', {})
    
    def calculate_factor(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算目标因子
        
        Args:
            data_dict: 数据字典
            
        Returns:
            因子数据
        """
        factor_name = self.target_factor_config.get('name', 'unknown_factor')
        fields = self.target_factor_config.get('fields', [])
        
        print(f"\n计算目标因子: {factor_name}")
        print(f"使用字段: {fields}")
        
        if factor_name == 'pe_inv' and 'pe_ttm' in fields:
            # PE倒数因子
            pe_data = data_dict['pe_ttm']
            factor_data = 1 / pe_data
            factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
            
        elif factor_name == 'pb_inv' and 'pb' in fields:
            # PB倒数因子
            pb_data = data_dict['pb']
            factor_data = 1 / pb_data
            factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
            
        elif factor_name == 'roe' and 'roe' in fields:
            # ROE因子
            factor_data = data_dict['roe']
            
        else:
            # 默认使用第一个字段
            if fields and fields[0] in data_dict:
                factor_data = data_dict[fields[0]]
            else:
                raise ValueError(f"无法计算因子 {factor_name}，缺少必要字段")
        
        print(f"因子计算完成，数据形状: {factor_data.shape}")
        return factor_data


def create_factor_processor(config: Dict) -> FactorProcessor:
    """
    创建因子预处理器
    
    Args:
        config: 配置字典
        
    Returns:
        FactorProcessor实例
    """
    return FactorProcessor(config)


def create_factor_calculator(config: Dict) -> FactorCalculator:
    """
    创建因子计算器
    
    Args:
        config: 配置字典
        
    Returns:
        FactorCalculator实例
    """
    return FactorCalculator(config)


# 使用示例
if __name__ == "__main__":
    # 加载配置
    with open("config.yml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建模拟数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    stocks = [f'stock_{i:03d}' for i in range(50)]
    
    # 模拟因子数据
    np.random.seed(42)
    factor_data = pd.DataFrame(
        index=dates,
        columns=stocks,
        data=np.random.randn(100, 50)
    )
    
    # 添加一些极值
    factor_data.iloc[10:15, 5:10] *= 10
    
    # 模拟股票池
    universe_df = pd.DataFrame(
        index=dates,
        columns=stocks,
        data=True
    )
    
    # 模拟辅助数据
    auxiliary_data = {
        'total_mv': pd.DataFrame(
            index=dates,
            columns=stocks,
            data=np.random.lognormal(10, 1, (100, 50))
        )
    }
    
    # 创建处理器并处理
    processor = create_factor_processor(config)
    processed_factor = processor.process_factor(
        factor_data, 
        universe_df, 
        auxiliary_data
    )
    
    print(f"\n处理完成！")
    print(f"原始因子形状: {factor_data.shape}")
    print(f"处理后形状: {processed_factor.shape}")
