"""
因子管理器模块

整合因子注册表、分类器和存储功能，提供统一的因子管理接口。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from pathlib import Path
import json
import os
from datetime import datetime

from quant_lib import setup_logger
# 导入子模块
from .registry.factor_registry import FactorRegistry, FactorCategory, FactorMetadata
from .classifier.factor_classifier import FactorClassifier
from .storage.single_storage import add_single_factor_test_result

logger = setup_logger(__name__)


class FactorTestResult:
    """因子测试结果类"""
    
    def __init__(self,
                 **kwargs):

        # 添加其他测试结果
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactorTestResult':
        """从字典创建"""
        return cls(**data)


class FactorManager:
    """因子管理器类"""
    
    def __init__(self, 
                 results_dir: str = "factor_results",
                 registry_path: str = "factor_registry.json"):
        """
        初始化因子管理器
        
        Args:
            results_dir: 测试结果保存目录
            registry_path: 注册表文件路径
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.registry = FactorRegistry(registry_path)
        self.classifier = FactorClassifier()
        
        # 测试结果缓存
        self.test_results = {}
        
        logger.info("因子管理器初始化完成")
    
    def register_factor(self, 
                       name: str,
                       category: Union[str, FactorCategory],
                       description: str = "",
                       data_requirements: List[str] = None,
                       **kwargs) -> bool:
        """
        注册因子
        
        Args:
            name: 因子名称
            category: 因子类别
            description: 因子描述
            data_requirements: 数据需求
            **kwargs: 其他元数据
            
        Returns:
            是否注册成功
        """
        return self.registry.register_factor(
            name=name,
            category=category,
            description=description,
            data_requirements=data_requirements,
            **kwargs
        )
    
    def get_factor_metadata(self, name: str) -> Optional[FactorMetadata]:
        """获取因子元数据"""
        return self.registry.get_factor(name)
    
    def list_factors(self, 
                    category: Union[str, FactorCategory] = None) -> List[str]:
        """
        列出因子
        
        Args:
            category: 筛选的因子类别
            
        Returns:
            因子名称列表
        """
        return self.registry.list_factors(category)
    
    def get_factor_summary(self) -> pd.DataFrame:
        """获取因子摘要"""
        return self.registry.get_factor_summary()
    

    
    def get_test_result(self, factor_name: str) -> Optional[FactorTestResult]:
        """
        获取测试结果
        
        Args:
            factor_name: 因子名称
            
        Returns:
            测试结果对象
        """
        # 先从缓存中获取
        if factor_name in self.test_results:
            return self.test_results[factor_name]
        
        # 从文件中获取最新的测试结果
        factor_dir = self.results_dir / factor_name
        if not factor_dir.exists():
            return None
        
        result_files = list(factor_dir.glob("test_result_*.json"))
        if not result_files:
            return None
        
        # 按文件名排序，获取最新的测试结果
        latest_file = sorted(result_files)[-1]
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = FactorTestResult.from_dict(data)
            
            # 更新缓存
            self.test_results[factor_name] = result
            
            return result
        except Exception as e:
            logger.error(f"加载测试结果失败: {e}")
            return None
    
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
        return self.classifier.classify_factor(factor_data, returns_data)
    
    def analyze_factor_correlation(self, 
                                 factor_data_dict: Dict[str, pd.DataFrame],
                                 figsize: Tuple[int, int] = (12, 10)) -> Tuple[pd.DataFrame, Any]:
        """
        分析因子相关性
        
        Args:
            factor_data_dict: 因子数据字典
            figsize: 图表大小
            
        Returns:
            (相关性矩阵, 热力图)
        """
        return self.classifier.analyze_factor_correlation(factor_data_dict, figsize)
    
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
        return self.classifier.cluster_factors(factor_data_dict, n_clusters)
    
    def visualize_factor_clusters(self, 
                                factor_data_dict: Dict[str, pd.DataFrame],
                                n_clusters: int = 5,
                                method: str = 'pca',
                                figsize: Tuple[int, int] = (12, 10)) -> Any:
        """
        可视化因子聚类
        
        Args:
            factor_data_dict: 因子数据字典
            n_clusters: 聚类数量
            method: 降维方法
            figsize: 图表大小
            
        Returns:
            图表对象
        """
        return self.classifier.visualize_factor_clusters(
            factor_data_dict, n_clusters, method, figsize
        )
    


    
    def get_top_factors(self, 
                       category: Union[str, FactorCategory] = None,
                       top_n: int = 10,
                       min_score: float = 2.0) -> List[str]:
        """
        获取顶级因子
        
        Args:
            category: 因子类别
            top_n: 返回的因子数量
            min_score: 最低评分
            
        Returns:
            因子名称列表
        """
        # 获取因子摘要
        summary = self.get_factor_summary()
        
        # 筛选类别
        if category is not None:
            if isinstance(category, FactorCategory):
                category_value = category.value
            else:
                try:
                    category_value = FactorCategory[category.upper()].value
                except KeyError:
                    logger.warning(f"未知的因子类别: {category}")
                    return []
            
            summary = summary[summary['category'] == category_value]
        
        # 筛选评分
        if 'test_overall_score' in summary.columns:
            summary = summary[summary['test_overall_score'] >= min_score]
        
        # 排序并返回顶级因子
        if 'test_overall_score' in summary.columns:
            top_factors = summary.sort_values('test_overall_score', ascending=False)['name'].tolist()[:top_n]
        else:
            top_factors = summary['name'].tolist()[:top_n]
        
        return top_factors
    def _make_serializable(self, obj):
        """将结果转换为可序列化格式"""
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, pd.Series):
            # 将索引转换为字符串
            series_dict = {}
            for k, v in obj.items():
                key = str(k) if hasattr(k, '__str__') else k
                series_dict[key] = self._make_serializable(v)
            return series_dict
        elif isinstance(obj, pd.DataFrame):
            # 将索引和列名都转换为字符串
            df_dict = {}
            for idx, row in obj.iterrows():
                row_dict = {}
                for col, val in row.items():
                    col_key = str(col) if hasattr(col, '__str__') else col
                    row_dict[col_key] = self._make_serializable(val)
                idx_key = str(idx) if hasattr(idx, '__str__') else idx
                df_dict[idx_key] = row_dict
            return df_dict
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif pd.isna(obj):
            return None
        else:
            try:
                # 尝试转换为基本Python类型
                if hasattr(obj, 'item'):  # numpy标量
                    return obj.item()
                return obj
            except:
                return str(obj)

    def _save_results(self, results: Dict[str, Any], factor_name: str):
        """保存测试结果"""
        # 准备可序列化的结果
        serializable_results = self._make_serializable(results)

        # 保存JSON格式
        json_path = os.path.join(self.results_dir, f'all_single_factor_test_results.json')
        add_single_factor_test_result(json_path, serializable_results)