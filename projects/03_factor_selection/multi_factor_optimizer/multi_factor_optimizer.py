"""
多因子优化器 - 实现类别内和类别间的因子优化

支持多种优化方法：
1. 类别内优化：相关性去重、IC加权、等权重等
2. 类别间优化：最大化分散化、风险平价、最优化组合等
3. 动态权重调整和回测验证

Author: Quantitative Research Team
Date: 2024-12-19
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntraCategoryOptimizer:
    """类别内优化器 - 在同一类别内选择和组合因子"""
    
    def __init__(self, correlation_threshold: float = 0.7):
        """
        初始化类别内优化器
        
        Args:
            correlation_threshold: 相关性阈值，超过此值的因子将被去重
        """
        self.correlation_threshold = correlation_threshold
    
    def remove_correlated_factors(self, 
                                 factor_data_dict: Dict[str, pd.DataFrame],
                                 factor_scores: Dict[str, float]) -> List[str]:
        """
        基于相关性去除冗余因子
        
        Args:
            factor_data_dict: 因子数据字典
            factor_scores: 因子评分字典
            
        Returns:
            去重后的因子名称列表
        """
        logger.info("开始相关性去重...")
        
        # 计算因子相关性矩阵
        factor_names = list(factor_data_dict.keys())
        correlation_matrix = self._calculate_factor_correlation(factor_data_dict)
        
        # 贪心算法去重
        selected_factors = []
        remaining_factors = factor_names.copy()
        
        # 按评分排序
        sorted_factors = sorted(remaining_factors, 
                              key=lambda x: factor_scores.get(x, 0), 
                              reverse=True)
        
        for factor in sorted_factors:
            if factor not in remaining_factors:
                continue
                
            # 检查与已选因子的相关性
            can_add = True
            for selected_factor in selected_factors:
                if abs(correlation_matrix.loc[factor, selected_factor]) > self.correlation_threshold:
                    can_add = False
                    break
            
            if can_add:
                selected_factors.append(factor)
                remaining_factors.remove(factor)
        
        logger.info(f"相关性去重完成：{len(factor_names)} -> {len(selected_factors)}")
        return selected_factors
    
    def _calculate_factor_correlation(self, 
                                    factor_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算因子相关性矩阵"""
        # 对齐所有因子数据
        aligned_data = {}
        common_index = None
        
        for name, data in factor_data_dict.items():
            if common_index is None:
                common_index = data.index
            else:
                common_index = common_index.intersection(data.index)
        
        for name, data in factor_data_dict.items():
            aligned_data[name] = data.reindex(common_index).stack().dropna()
        
        # 构建DataFrame并计算相关性
        factor_df = pd.DataFrame(aligned_data)
        correlation_matrix = factor_df.corr()
        
        return correlation_matrix
    
    def ic_weighted_combination(self, 
                               factor_data_dict: Dict[str, pd.DataFrame],
                               ic_scores: Dict[str, float]) -> pd.DataFrame:
        """
        基于IC值的加权组合
        
        Args:
            factor_data_dict: 因子数据字典
            ic_scores: IC评分字典
            
        Returns:
            组合后的因子
        """
        logger.info("开始IC加权组合...")
        
        # 标准化权重
        total_ic = sum(abs(score) for score in ic_scores.values())
        if total_ic == 0:
            weights = {name: 1/len(ic_scores) for name in ic_scores.keys()}
        else:
            weights = {name: abs(score)/total_ic for name, score in ic_scores.items()}
        
        # 加权组合
        combined_factor = None
        for name, data in factor_data_dict.items():
            weight = weights.get(name, 0)
            if weight > 0:
                if combined_factor is None:
                    combined_factor = data * weight
                else:
                    combined_factor = combined_factor.add(data * weight, fill_value=0)
        
        logger.info("IC加权组合完成")
        return combined_factor
    
    def equal_weight_combination(self, 
                                factor_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """等权重组合"""
        logger.info("开始等权重组合...")
        
        weight = 1.0 / len(factor_data_dict)
        combined_factor = None
        
        for name, data in factor_data_dict.items():
            if combined_factor is None:
                combined_factor = data * weight
            else:
                combined_factor = combined_factor.add(data * weight, fill_value=0)
        
        logger.info("等权重组合完成")
        return combined_factor
    
    def pca_combination(self, 
                       factor_data_dict: Dict[str, pd.DataFrame],
                       n_components: int = 1) -> pd.DataFrame:
        """
        PCA降维组合
        
        Args:
            factor_data_dict: 因子数据字典
            n_components: 主成分数量
            
        Returns:
            PCA组合因子
        """
        logger.info(f"开始PCA组合，主成分数量: {n_components}")
        
        # 对齐数据
        aligned_data = {}
        common_index = None
        
        for name, data in factor_data_dict.items():
            if common_index is None:
                common_index = data.index
            else:
                common_index = common_index.intersection(data.index)
        
        # 构建特征矩阵
        feature_matrix = []
        for date in common_index:
            row_data = []
            for name, data in factor_data_dict.items():
                if date in data.index:
                    row_data.append(data.loc[date].values)
                else:
                    row_data.append(np.full(data.shape[1], np.nan))
            
            # 拼接所有因子的横截面数据
            concatenated = np.concatenate(row_data)
            if not np.all(np.isnan(concatenated)):
                feature_matrix.append(concatenated)
        
        feature_matrix = np.array(feature_matrix)
        
        # 执行PCA
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_features)
        
        # 构建结果DataFrame
        if n_components == 1:
            pca_factor = pd.DataFrame(
                pca_result.flatten(),
                index=common_index[:len(pca_result)],
                columns=['PCA_Factor']
            )
        else:
            pca_factor = pd.DataFrame(
                pca_result,
                index=common_index[:len(pca_result)],
                columns=[f'PCA_Factor_{i+1}' for i in range(n_components)]
            )
        
        logger.info(f"PCA组合完成，解释方差比例: {pca.explained_variance_ratio_}")
        return pca_factor


class CrossCategoryOptimizer:
    """类别间优化器 - 在不同类别间进行因子配置"""
    
    def __init__(self, risk_aversion: float = 1.0):
        """
        初始化类别间优化器
        
        Args:
            risk_aversion: 风险厌恶系数
        """
        self.risk_aversion = risk_aversion
    
    def max_diversification_weights(self, 
                                   category_factors: Dict[str, pd.DataFrame],
                                   category_scores: Dict[str, float]) -> Dict[str, float]:
        """
        最大分散化权重配置
        
        Args:
            category_factors: 各类别的组合因子
            category_scores: 各类别的评分
            
        Returns:
            各类别的权重
        """
        logger.info("开始最大分散化权重优化...")
        
        # 计算类别间相关性
        correlation_matrix = self._calculate_category_correlation(category_factors)
        
        # 计算分散化比率
        n_categories = len(category_factors)
        equal_weights = np.ones(n_categories) / n_categories
        
        # 目标函数：最大化分散化比率
        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ correlation_matrix @ weights)
            weighted_avg_vol = np.sum(weights * np.diag(correlation_matrix))
            diversification_ratio = weighted_avg_vol / portfolio_vol
            return -diversification_ratio  # 最大化
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
        ]
        bounds = [(0, 1) for _ in range(n_categories)]
        
        # 优化
        result = minimize(
            objective, 
            equal_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
        else:
            logger.warning("优化失败，使用等权重")
            weights = equal_weights
        
        # 转换为字典
        category_names = list(category_factors.keys())
        weight_dict = {name: weight for name, weight in zip(category_names, weights)}
        
        logger.info("最大分散化权重优化完成")
        return weight_dict
    
    def risk_parity_weights(self, 
                           category_factors: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        风险平价权重配置
        
        Args:
            category_factors: 各类别的组合因子
            
        Returns:
            各类别的权重
        """
        logger.info("开始风险平价权重优化...")
        
        # 计算协方差矩阵
        covariance_matrix = self._calculate_category_covariance(category_factors)
        
        n_categories = len(category_factors)
        equal_weights = np.ones(n_categories) / n_categories
        
        # 目标函数：最小化风险贡献的差异
        def objective(weights):
            portfolio_var = weights.T @ covariance_matrix @ weights
            marginal_contrib = covariance_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_var
            target_contrib = 1.0 / n_categories
            return np.sum((contrib - target_contrib) ** 2)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        bounds = [(0.01, 0.99) for _ in range(n_categories)]
        
        # 优化
        result = minimize(
            objective,
            equal_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
        else:
            logger.warning("风险平价优化失败，使用等权重")
            weights = equal_weights
        
        # 转换为字典
        category_names = list(category_factors.keys())
        weight_dict = {name: weight for name, weight in zip(category_names, weights)}
        
        logger.info("风险平价权重优化完成")
        return weight_dict
    
    def _calculate_category_correlation(self, 
                                      category_factors: Dict[str, pd.DataFrame]) -> np.ndarray:
        """计算类别间相关性矩阵"""
        # 对齐数据并计算相关性
        aligned_data = {}
        common_index = None
        
        for name, data in category_factors.items():
            if common_index is None:
                common_index = data.index
            else:
                common_index = common_index.intersection(data.index)
        
        for name, data in category_factors.items():
            if len(data.columns) == 1:
                aligned_data[name] = data.reindex(common_index).iloc[:, 0]
            else:
                # 如果有多列，取第一列或平均值
                aligned_data[name] = data.reindex(common_index).mean(axis=1)
        
        factor_df = pd.DataFrame(aligned_data).dropna()
        correlation_matrix = factor_df.corr().values
        
        return correlation_matrix
    
    def _calculate_category_covariance(self, 
                                     category_factors: Dict[str, pd.DataFrame]) -> np.ndarray:
        """计算类别间协方差矩阵"""
        # 对齐数据并计算协方差
        aligned_data = {}
        common_index = None
        
        for name, data in category_factors.items():
            if common_index is None:
                common_index = data.index
            else:
                common_index = common_index.intersection(data.index)
        
        for name, data in category_factors.items():
            if len(data.columns) == 1:
                aligned_data[name] = data.reindex(common_index).iloc[:, 0]
            else:
                aligned_data[name] = data.reindex(common_index).mean(axis=1)
        
        factor_df = pd.DataFrame(aligned_data).dropna()
        covariance_matrix = factor_df.cov().values
        
        return covariance_matrix


class MultiFactorOptimizer:
    """
    多因子优化器 - 整合类别内和类别间优化
    
    完整的优化流程：
    1. 类别内优化：去重、组合
    2. 类别间优化：权重配置
    3. 最终因子构建和验证
    """
    
    def __init__(self, 
                 correlation_threshold: float = 0.7,
                 risk_aversion: float = 1.0):
        """
        初始化多因子优化器
        
        Args:
            correlation_threshold: 类别内相关性阈值
            risk_aversion: 风险厌恶系数
        """
        self.intra_optimizer = IntraCategoryOptimizer(correlation_threshold)
        self.cross_optimizer = CrossCategoryOptimizer(risk_aversion)
        
        logger.info("多因子优化器初始化完成")
    
    def optimize_factors(self, 
                        factors_by_category: Dict[str, Dict[str, pd.DataFrame]],
                        factor_scores: Dict[str, Dict[str, float]],
                        intra_method: str = 'ic_weighted',
                        cross_method: str = 'max_diversification') -> pd.DataFrame:
        """
        完整的多因子优化流程
        
        Args:
            factors_by_category: 按类别分组的因子数据
            factor_scores: 按类别分组的因子评分
            intra_method: 类别内优化方法
            cross_method: 类别间优化方法
            
        Returns:
            最终的多因子组合
        """
        logger.info("开始多因子优化流程...")
        
        # 第一步：类别内优化
        category_factors = {}
        for category, factors in factors_by_category.items():
            logger.info(f"优化类别: {category}")
            
            # 相关性去重
            scores = factor_scores.get(category, {})
            selected_factors = self.intra_optimizer.remove_correlated_factors(factors, scores)
            selected_factor_data = {name: factors[name] for name in selected_factors}
            
            # 类别内组合
            if intra_method == 'ic_weighted':
                selected_scores = {name: scores.get(name, 0) for name in selected_factors}
                combined_factor = self.intra_optimizer.ic_weighted_combination(
                    selected_factor_data, selected_scores
                )
            elif intra_method == 'equal_weight':
                combined_factor = self.intra_optimizer.equal_weight_combination(selected_factor_data)
            elif intra_method == 'pca':
                combined_factor = self.intra_optimizer.pca_combination(selected_factor_data)
            else:
                raise ValueError(f"不支持的类别内优化方法: {intra_method}")
            
            category_factors[category] = combined_factor
        
        # 第二步：类别间优化
        category_scores = {
            category: np.mean(list(scores.values())) 
            for category, scores in factor_scores.items()
        }
        
        if cross_method == 'max_diversification':
            category_weights = self.cross_optimizer.max_diversification_weights(
                category_factors, category_scores
            )
        elif cross_method == 'risk_parity':
            category_weights = self.cross_optimizer.risk_parity_weights(category_factors)
        elif cross_method == 'equal_weight':
            category_weights = {cat: 1.0/len(category_factors) for cat in category_factors.keys()}
        else:
            raise ValueError(f"不支持的类别间优化方法: {cross_method}")
        
        # 第三步：构建最终因子
        final_factor = None
        for category, factor_data in category_factors.items():
            weight = category_weights.get(category, 0)
            if weight > 0:
                if final_factor is None:
                    final_factor = factor_data * weight
                else:
                    final_factor = final_factor.add(factor_data * weight, fill_value=0)
        
        logger.info("多因子优化完成")
        logger.info(f"类别权重: {category_weights}")
        
        return final_factor
