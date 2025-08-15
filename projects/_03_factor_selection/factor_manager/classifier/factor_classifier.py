"""
因子分类器模块

负责对因子进行自动分类、聚类和特征提取。
"""
import re

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 导入因子注册表
from ..registry.factor_registry import FactorCategory

# 配置日志
logger = logging.getLogger(__name__)


class FactorClassifier:
    """因子分类器类"""
    
    def __init__(self):
        """初始化因子分类器"""
        self.feature_cache = {}  # 缓存因子特征
    
    def extract_factor_features(self, 
                               factor_data: pd.DataFrame,
                               returns_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        提取因子特征
        
        Args:
            factor_data: 因子数据
            returns_data: 收益率数据，用于计算与收益相关的特征
            
        Returns:
            特征字典
        """
        features = {}
        
        # 基本统计特征
        features['mean'] = factor_data.mean().mean()
        features['std'] = factor_data.std().mean()
        features['skew'] = factor_data.skew().mean()
        features['kurtosis'] = factor_data.kurtosis().mean()
        
        # 时间序列特征
        if len(factor_data) > 1:
            # 自相关性
            autocorr = factor_data.apply(lambda x: x.autocorr(lag=1)).mean()
            features['autocorr'] = autocorr if not pd.isna(autocorr) else 0
            
            # 趋势性
            factor_means = factor_data.mean(axis=1)
            features['trend'] = np.polyfit(np.arange(len(factor_means)), factor_means, 1)[0]
            
            # 波动性
            features['volatility'] = factor_data.pct_change().std().mean()
        
        # 横截面特征
        features['cross_dispersion'] = factor_data.std(axis=1).mean()
        
        # 与收益率相关的特征
        if returns_data is not None:
            # 对齐数据
            common_dates = factor_data.index.intersection(returns_data.index)
            common_stocks = factor_data.columns.intersection(returns_data.columns)
            
            if len(common_dates) > 0 and len(common_stocks) > 0:
                aligned_factor = factor_data.loc[common_dates, common_stocks]
                aligned_returns = returns_data.loc[common_dates, common_stocks]
                
                # 计算IC
                ic_values = []
                for date in common_dates:
                    f = aligned_factor.loc[date].dropna()
                    r = aligned_returns.loc[date].dropna()
                    common = f.index.intersection(r.index)
                    if len(common) > 10:  # 至少需要10只股票
                        corr = f[common].corr(r[common])
                        if not pd.isna(corr):
                            ic_values.append(corr)
                
                if ic_values:
                    features['ic_mean'] = np.mean(ic_values)
                    features['ic_std'] = np.std(ic_values)
                    features['ic_ir'] = features['ic_mean'] / features['ic_std'] if features['ic_std'] > 0 else 0
        
        return features
    
    def classify_factor(self, 
                       factor_data: pd.DataFrame,
                       returns_data: pd.DataFrame = None) -> FactorCategory:
        """
        自动分类因子
        
        Args:
            factor_data: 因子数据
            returns_data: 收益率数据
            
        Returns:
            因子类别
        """
        # 提取特征
        features = self.extract_factor_features(factor_data, returns_data)
        
        # 基于规则的分类
        # 这里是一个简化的分类逻辑，实际应用中可能需要更复杂的规则或机器学习模型
        
        # 检查趋势性和自相关性 -> 动量类
        if features.get('trend', 0) > 0.01 and features.get('autocorr', 0) > 0.7:
            return FactorCategory.MOMENTUM
        
        # 检查波动性 -> 波动率类
        if features.get('volatility', 0) > 0.05:
            return FactorCategory.VOLATILITY
        
        # 检查偏度 -> 情绪类
        if abs(features.get('skew', 0)) > 1.0:
            return FactorCategory.SENTIMENT
        
        # 默认分类
        return FactorCategory.CUSTOM
    
    def cluster_factors(self, 
                       factor_data_dict: Dict[str, pd.DataFrame],
                       n_clusters: int = 5) -> Dict[str, int]:
        """
        聚类因子
        
        Args:
            factor_data_dict: 因子数据字典
            n_clusters: 聚类数量
            
        Returns:
            因子聚类结果字典
        """
        # 提取所有因子的特征
        feature_matrix = []
        factor_names = []
        
        for name, data in factor_data_dict.items():
            if name not in self.feature_cache:
                self.feature_cache[name] = self.extract_factor_features(data)
            
            # 将特征转换为向量
            feature_vector = [v for k, v in sorted(self.feature_cache[name].items())]
            feature_matrix.append(feature_vector)
            factor_names.append(name)
        
        if not feature_matrix:
            raise  ValueError("没有可用的因子特征进行聚类")

        # 标准化特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=min(n_clusters, len(feature_matrix)), random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # 返回聚类结果
        return {name: int(cluster) for name, cluster in zip(factor_names, clusters)}
    
    def visualize_factor_clusters(self, 
                                factor_data_dict: Dict[str, pd.DataFrame],
                                n_clusters: int = 5,
                                method: str = 'pca',
                                figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        可视化因子聚类
        
        Args:
            factor_data_dict: 因子数据字典
            n_clusters: 聚类数量
            method: 降维方法，'pca'或'tsne'
            figsize: 图表大小
            
        Returns:
            图表对象
        """
        # 提取特征并聚类
        clusters = self.cluster_factors(factor_data_dict, n_clusters)
        if not clusters:
            raise ValueError("没有可用的因子进行可视化")

        
        # 准备数据
        feature_matrix = []
        factor_names = []
        
        for name in clusters.keys():
            feature_vector = [v for k, v in sorted(self.feature_cache[name].items())]
            feature_matrix.append(feature_vector)
            factor_names.append(name)
        
        # 标准化特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # 降维
        if method == 'pca':
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(scaled_features)
            explained_var = pca.explained_variance_ratio_
            
            # 创建图表
            fig, ax = plt.subplots(figsize=figsize)
            
            # 绘制散点图
            scatter = ax.scatter(
                reduced_data[:, 0], 
                reduced_data[:, 1], 
                c=[clusters[name] for name in factor_names],
                cmap='viridis', 
                alpha=0.8,
                s=100
            )
            
            # 添加标签
            for i, name in enumerate(factor_names):
                ax.annotate(
                    name, 
                    (reduced_data[i, 0], reduced_data[i, 1]),
                    fontsize=9,
                    alpha=0.8
                )
            
            # 添加图例
            legend = ax.legend(
                *scatter.legend_elements(),
                title="聚类",
                loc="upper right"
            )
            
            # 设置标题和标签
            ax.set_title('因子聚类可视化 (PCA降维)', fontsize=14)
            ax.set_xlabel(f'主成分1 ({explained_var[0]:.2%} 方差)', fontsize=12)
            ax.set_ylabel(f'主成分2 ({explained_var[1]:.2%} 方差)', fontsize=12)
            
            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.7)
            
            return fig
        else:
            # 可以添加其他降维方法，如t-SNE
            raise ValueError(f"不支持的降维方法: {method}")

    
    def analyze_factor_correlation(self, 
                                 factor_data_dict: Dict[str, pd.DataFrame],
                                 figsize: Tuple[int, int] = (12, 10)) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        分析因子相关性
        
        Args:
            factor_data_dict: 因子数据字典
            figsize: 图表大小
            
        Returns:
            (相关性矩阵, 热力图)
        """
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
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(
            correlation_matrix, 
            mask=mask,
            cmap='coolwarm', 
            annot=True, 
            fmt='.2f',
            linewidths=0.5,
            ax=ax,
            vmin=-1, 
            vmax=1,
            center=0
        )
        
        ax.set_title('因子相关性热力图', fontsize=14)
        
        return correlation_matrix, fig

    """
       【V2版】因子分类器
       根据因子名称的通用模式和特定关键字，判断其所属的风险类别。
       """

    """
       一个用于根据名称对因子进行分类的工具类。
       """

    @staticmethod
    def _match(name: str,
               exact: list[str] | None = None,
               prefixes: list[str] | None = None,
               suffixes: list[str] | None = None,
               keywords: list[str] | None = None,
               patterns: list[str] | None = None) -> bool:
        """
        【通用匹配引擎】根据多种规则判断因子名称是否匹配。

        Args:
            name (str): 待检查的因子名称。
            exact (list): 精确匹配列表。
            prefixes (list): 前缀匹配列表。
            suffixes (list): 后缀匹配列表。
            keywords (list): 包含的关键字列表。
            patterns (list): 正则表达式模式列表。

        Returns:
            bool: 如果匹配任何规则，则返回True。
        """
        # 使用空列表作为默认值，避免处理None的情况
        exact = exact or []
        prefixes = prefixes or []
        suffixes = suffixes or []
        keywords = keywords or []
        patterns = patterns or []

        # 1. 精确匹配
        if name in exact:
            return True
        # 2. 前缀/后缀匹配
        if any(name.startswith(p) for p in prefixes) or \
                any(name.endswith(s) for s in suffixes):
            return True
        # 3. 关键字匹配
        if any(k in name for k in keywords):
            return True
        # 4. 正则表达式匹配 (最强大、最精确)
        #    使用 \bbeta\b 匹配独立的单词 "beta"
        if any(re.search(pat, name) for pat in patterns):
            return True

        return False

    @classmethod
    def is_size_factor(cls, factor_name: str) -> bool:
        """判断是否为市值类因子 (更安全)"""
        return cls._match(
            name=factor_name,
            exact=['size', 'log_circ_mv','log_circ_mv','log_market_cap','log_total_mv'],
            keywords=['market_cap'],
            suffixes=['_mv', '_log_mv']  # 使用后缀匹配替代关键字'_mv'，更安全
        )

    @classmethod
    def is_industry_factor(cls, factor_name: str) -> bool:
        """判断是否为行业类因子 (逻辑不变，统一模式)"""
        return cls._match(
            name=factor_name,
            exact=['industry'],
            prefixes=['industry_']
        )

    @classmethod
    def is_beta_factor(cls, factor_name: str) -> bool:
        """判断是否为Beta类因子 (使用正则，更精确)"""
        # \b 是单词边界，确保只匹配独立的 "beta" 单词
        # 例如：匹配 'beta', 'equity_beta', 'beta_60d'
        # 不会匹配：'alphabet'
        return cls._match(
            name=factor_name,
            patterns=[r'\bbeta\b']
        )
