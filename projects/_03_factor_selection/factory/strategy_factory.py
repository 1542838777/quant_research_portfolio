"""
策略工厂模块

整合因子管理器、单因子测试器、多因子优化器和可视化管理器，
提供一站式量化研究解决方案。
"""
import sys

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
import json

from ..config.config_file.load_config_file import _load_local_config
from ..factor_manager.factor_manager import FactorManager
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer
from ..factor_manager.registry.factor_registry import FactorCategory
from ..multi_factor_optimizer.multi_factor_optimizer import MultiFactorOptimizer
from ..data_manager.data_manager import DataManager, fill_and_align_by_stock_pool
from ..utils.factor_processor import FactorProcessor
from ..visualization_manager.visualization_manager import VisualizationManager

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.config.logger_config import setup_logger

# 配置日志
logger = setup_logger(__name__)


class StrategyFactory:
    """
    策略工厂 - 量化研究的核心引擎
    
    功能：
    1. 统一管理数据、因子、测试、优化等组件
    2. 提供标准化的研究流程
    3. 支持单因子测试到多因子策略的完整链路
    4. 便于团队协作和成果复用
    """

    def __init__(self,
                 config_path: str = "config.yaml",
                 workspace_dir: str = "strategy_workspace"):
        """
        初始化策略工厂
        
        Args:
            config_path: 配置文件路径
            workspace_dir: 工作空间目录
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self.config = _load_local_config(config_path)
        self.config_path = config_path

        # 初始化核心组件
        self._initialize_components()

        logger.info("策略工厂初始化完成（config读取，工作区间准备）")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2024-12-31',
                'stack_pool': 'hs300',
                'benchmark': '000300.SH'
            },
            'factor_test': {
                'forward_periods': [1, 5, 20],
                'quantiles': 5,
                'preprocessing': {
                    'winsorization': {'enable': True, 'method': 'mad', 'mad_threshold': 5},
                    'neutralization': {'enable': True, 'factors': ['market_cap', 'industry']},
                    'standardization': {'enable': True, 'method': 'zscore'}
                }
            },
            'optimization': {
                'method': 'equal_weight',
                'max_factors_per_category': 3,
                'correlation_threshold': 0.7
            },
            'output': {
                'save_plots': True,
                'generate_report': True,
                'export_excel': True
            }
        }

    def _initialize_components(self):
        """初始化各个组件"""
        # 创建组件目录
        components_dirs = {
            'factor_results': self.workspace_dir / "factor_results",
            'test_results': self.workspace_dir / "test_results",
            'optimization_results': self.workspace_dir / "optimization_results",
            'visualizations': self.workspace_dir / "visualizations",
            'data': self.workspace_dir / "data"
        }

        for dir_name, dir_path in components_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)

        # 数据管理器
        self.data_manager = DataManager(self.config_path)

        # 因子管理器
        registry_path = self.workspace_dir / "factor_registry.json"
        self.factor_manager = FactorManager(
            results_dir=str(components_dirs['factor_results']),
            registry_path=str(registry_path)
        )

        # 因子预处理器
        self.factor_processor = FactorProcessor(
            self.config.get('factor_test', {}).get('preprocessing', {})
        )

        # 单因子测试器
        self.single_factor_tester = None  # 延迟初始化

        # 多因子优化器
        self.multi_factor_optimizer = MultiFactorOptimizer(
            correlation_threshold=self.config.get('optimization', {}).get('correlation_threshold', 0.7)
        )

        # 可视化管理器
        self.visualization_manager = VisualizationManager(
            output_dir=str(components_dirs['visualizations'])
        )

        logger.info("核心组件初始化完成")

    def register_factor(self,
                        name: str,
                        category: Union[str, FactorCategory],
                        description: str = "",
                        data_requirements: List[str] = None,
                        **kwargs) -> bool:
        """
        注册新因子
        
        Args:
            name: 因子名称
            category: 因子类别
            description: 因子描述
            data_requirements: 数据需求
            **kwargs: 其他元数据
        """
        return self.factor_manager.register_factor(
            name=name,
            category=category,
            description=description,
            data_requirements=data_requirements,
            **kwargs
        )

    #really

    # def batch_test_factors(self,
    #                        target_factors_dict: Dict[str, pd.DataFrame],
    #                        target_factors_category_dict: Dict[str, str],
    #                        target_factor_school_type_dict: Dict[str, str],
    #                        **test_kwargs) -> Dict[str, Any]:
    #     """
    #     批量测试因子
    #     """
    #     # 初始化单因子测试器
    #     if self.single_factor_tester is None:
    #         self.single_factor_tester = FactorAnalyzer(
    #             raw_dfs=self.data_manager.raw_dfs,
    #             target_factors_dict=target_factors_dict,
    #             target_factors_category_dict=target_factors_category_dict,
    #             target_factor_school_type_dict=target_factor_school_type_dict,
    #             stock_pools_dict=self.data_manager.stock_pools_dict,
    #             config=self.config
    #         )
    #
    #     # 批量测试
    #     results = {}
    #     for factor_name, factor_data in target_factors_dict.items():
    #         try:
    #             # 执行测试
    #             ic_series_periods_dict,quantile_returns_series_periods_dict,factor_returns_series_periods_dict,summary_stats = (self.single_factor_tester.
    #             test_single_factor_entity_service(
    #                 target_factor_name=factor_name,
    #                 factor_manager=self.factor_manager,
    #                 visualization_manager = self.visualization_manager
    #             ))
    #             results[factor_name] = summary_stats
    #         except Exception as e:
    #             # traceback.print_exc()
    #             raise ValueError(f"✗ 因子{factor_name}测试失败: {e}") from e
    #
    #     return results

    def optimize_factors(self,
                         factor_data_dict: Dict[str, pd.DataFrame] = None,
                         selected_factors: List[str] = None,
                         intra_method: str = 'ic_weighted',
                         cross_method: str = 'max_diversification') -> pd.DataFrame:
        """
        优化因子组合
        
        Args:
            factor_data_dict: 因子数据字典，如果为None则使用selected_factors加载数据
            selected_factors: 选定的因子列表
            intra_method: 类别内优化方法
            cross_method: 类别间优化方法
        """
        if factor_data_dict is None and selected_factors is None:
            raise ValueError("必须提供factor_data_dict或selected_factors")

        # 如果提供了selected_factors，加载因子数据
        if factor_data_dict is None:
            factor_data_dict = {}

        # 获取因子评分
        factor_scores = {}
        for factor_name in factor_data_dict.keys():
            result = self.factor_manager.get_test_result(factor_name)
            if result:
                factor_scores[factor_name] = result.ic_ir or 0

        # 按类别分组因子
        factors_by_category = {}
        scores_by_category = {}

        # 获取所有因子的元数据
        for factor_name, factor_data in factor_data_dict.items():
            metadata = self.factor_manager.get_factor_metadata(factor_name)
            if metadata:
                category = metadata.category.value

                # 添加到类别分组
                if category not in factors_by_category:
                    factors_by_category[category] = {}
                    scores_by_category[category] = {}

                factors_by_category[category][factor_name] = factor_data
                scores_by_category[category][factor_name] = factor_scores.get(factor_name, 0)

        # 执行优化
        optimized_factor = self.multi_factor_optimizer.optimize_factors(
            factors_by_category=factors_by_category,
            factor_scores=scores_by_category,
            intra_method=intra_method,
            cross_method=cross_method
        )

        # 保存优化结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.workspace_dir / "optimization_results" / f"opt_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        config_file = output_dir / "optimization_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'intra_method': intra_method,
                'cross_method': cross_method,
                'factors': list(factor_data_dict.keys()),
                'categories': list(factors_by_category.keys())
            }, f, indent=2, ensure_ascii=False)

        # 保存优化后的因子
        factor_file = output_dir / "optimized_factor.csv"
        optimized_factor.to_csv(factor_file)

        logger.info(f"因子优化完成，结果保存到: {output_dir}")
        return optimized_factor

    def analyze_factor_correlation(self,
                                   factor_data_dict: Dict[str, pd.DataFrame],
                                   figsize: Tuple[int, int] = (12, 10)) -> Tuple[pd.DataFrame, Any]:
        """
        分析因子相关性
        
        Args:
            factor_data_dict: 因子数据字典
            figsize: 图表大小
        """
        return self.factor_manager.analyze_factor_correlation(factor_data_dict, figsize)

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
        """
        return self.factor_manager.visualize_factor_clusters(
            factor_data_dict, n_clusters, method, figsize
        )

