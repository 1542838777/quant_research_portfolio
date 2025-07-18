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

# 导入子模块
from .registry.factor_registry import FactorRegistry, FactorCategory, FactorMetadata
from .classifier.factor_classifier import FactorClassifier

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorTestResult:
    """因子测试结果类"""
    
    def __init__(self, 
                 factor_name: str,
                 test_date: str = None,
                 ic_mean: float = None,
                 ic_ir: float = None,
                 rank_ic_mean: float = None,
                 rank_ic_ir: float = None,
                 annualized_return: float = None,
                 sharpe_ratio: float = None,
                 max_drawdown: float = None,
                 turnover: float = None,
                 t_stat: float = None,
                 p_value: float = None,
                 overall_score: float = None,
                 grade: str = None,
                 **kwargs):
        """
        初始化因子测试结果
        
        Args:
            factor_name: 因子名称
            test_date: 测试日期
            ic_mean: IC均值
            ic_ir: IC信息比率
            rank_ic_mean: Rank IC均值
            rank_ic_ir: Rank IC信息比率
            annualized_return: 年化收益率
            sharpe_ratio: 夏普比率
            max_drawdown: 最大回撤
            turnover: 换手率
            t_stat: t统计量
            p_value: p值
            overall_score: 综合评分
            grade: 评级
            **kwargs: 其他测试结果
        """
        self.factor_name = factor_name
        self.test_date = test_date or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.ic_mean = ic_mean
        self.ic_ir = ic_ir
        self.rank_ic_mean = rank_ic_mean
        self.rank_ic_ir = rank_ic_ir
        self.annualized_return = annualized_return
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.turnover = turnover
        self.t_stat = t_stat
        self.p_value = p_value
        self.overall_score = overall_score
        self.grade = grade
        
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
    
    def save_test_result(self, result: Union[FactorTestResult, Dict[str, Any]]) -> bool:
        """
        保存测试结果
        
        Args:
            result: 测试结果对象或字典
            
        Returns:
            是否保存成功
        """
        if isinstance(result, dict):
            result = FactorTestResult(**result)
        
        factor_name = result.factor_name
        
        # 更新缓存
        self.test_results[factor_name] = result
        
        # 更新注册表
        self.registry.update_factor_test_result(
            factor_name,
            {
                'ic_mean': result.ic_mean,
                'ic_ir': result.ic_ir,
                'sharpe_ratio': result.sharpe_ratio,
                'overall_score': result.overall_score,
                'grade': result.grade,
                'test_date': result.test_date
            }
        )
        
        # 保存到文件
        factor_dir = self.results_dir / factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = factor_dir / f"test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"测试结果已保存到 {result_file}")
            return True
        except Exception as e:
            logger.error(f"保存测试结果失败: {e}")
            return False
    
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
    
    def batch_test_factors(self, 
                          factor_data_dict: Dict[str, pd.DataFrame],
                          test_func: callable,
                          **test_kwargs) -> Dict[str, Any]:
        """
        批量测试因子
        
        Args:
            factor_data_dict: 因子数据字典
            test_func: 测试函数
            **test_kwargs: 测试参数
            
        Returns:
            测试结果字典
        """
        results = {}
        
        for factor_name, factor_data in factor_data_dict.items():
            logger.info(f"测试因子: {factor_name}")
            
            try:
                # 执行测试
                test_result = test_func(
                    factor_data=factor_data,
                    factor_name=factor_name,
                    **test_kwargs
                )
                
                # 保存结果
                self.save_test_result(self._convert_test_result(factor_name, test_result, test_kwargs))
                
                # 添加到结果字典
                results[factor_name] = test_result
                
                logger.info(f"因子 {factor_name} 测试完成")
            except Exception as e:
                logger.error(f"因子 {factor_name} 测试失败: {e}")
        
        return results
    
    def _convert_test_result(self, 
                           factor_name: str, 
                           test_result: Dict[str, Any],
                           test_kwargs: Dict[str, Any] = None) -> FactorTestResult:
        """
        转换测试结果
        
        Args:
            factor_name: 因子名称
            test_result: 测试结果字典
            test_kwargs: 测试参数
            
        Returns:
            测试结果对象
        """
        # 提取关键指标
        ic_analysis = test_result.get('ic_analysis', {})
        layered_test = test_result.get('layered_test', {})
        regression = test_result.get('regression', {})
        evaluation = test_result.get('evaluation', {})
        
        # 创建测试结果对象
        result = FactorTestResult(
            factor_name=factor_name,
            test_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ic_mean=ic_analysis.get('ic_mean'),
            ic_ir=ic_analysis.get('ic_ir'),
            rank_ic_mean=ic_analysis.get('rank_ic_mean'),
            rank_ic_ir=ic_analysis.get('rank_ic_ir'),
            annualized_return=layered_test.get('annualized_return'),
            sharpe_ratio=layered_test.get('sharpe_ratio'),
            max_drawdown=layered_test.get('max_drawdown'),
            turnover=layered_test.get('turnover'),
            t_stat=regression.get('t_statistic'),
            p_value=regression.get('p_value'),
            overall_score=evaluation.get('score'),
            grade=evaluation.get('grade'),
            # 添加原始测试结果
            raw_result=test_result,
            test_params=test_kwargs or {}
        )
        
        return result
    
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
