"""
策略工厂模块

整合因子管理器、单因子测试器、多因子优化器和可视化管理器，
提供一站式量化研究解决方案。
"""
import sys

import pandas as pd
from typing import Dict, List, Any, Union, Tuple
from pathlib import Path
import yaml
from datetime import datetime
import json
from ..factor_manager.factor_manager import FactorManager
from ..single_factor_tester.single_factor_tester import SingleFactorTester
from ..factor_manager.registry.factor_registry import FactorCategory
from ..multi_factor_optimizer.multi_factor_optimizer import MultiFactorOptimizer
from ..data_manager.data_manager import DataManager
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
        self.config = self._load_config(config_path)
        self.config_path = config_path

        # 初始化核心组件
        self._initialize_components()

        logger.info("策略工厂初始化完成（config读取，工作区间准备）")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
        else:
            raise RuntimeError("未找到config文件")

        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2024-12-31',
                'universe': 'hs300',
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

    # return with动态股票池处理好的数据
    def load_all_data_be_universe(self) -> Dict[str, pd.DataFrame]:
        """加载数据"""
        logger.info("开始加载基础必要数据...")
        data_dict = self.data_manager.load_all_data()
        logger.info(f"数据加载完成，包含 {len(data_dict)} 个数据集（data_dict）")
        return data_dict

    def test_single_factor(self,
                           factor_name: str,
                           factor_data: pd.DataFrame,
                           factor_category_type: str,
                           **test_kwargs) -> Dict[str, Any]:
        """
        测试单个因子
        
        Args:
            factor_data: 因子数据
            factor_name: 因子名称
            category: 因子类别
            **test_kwargs: 测试参数
        """
        # 初始化单因子测试器（如果还没有）
        if self.single_factor_tester is None:
            self.single_factor_tester = SingleFactorTester(
                data_dict=self.data_manager.raw_data,
                config=self.config
            )

        if factor_category_type is None:
            # 寻找
            raise ValueError("请指定因子是什么类型的")
        # # 自动分类因子
        # 先注释下面的，问题：自动识别因子类型 函数有待补充！ 暂且不用 不是很要紧
        # if category is None and auto_register:
        #
        #
        #     # returns_data = self.single_factor_tester.get_returns_data()#todo！！！
        #     # category = self.factor_manager.classify_factor(factor_data, returns_data)
        #
        # # 自动注册因子
        # if auto_register:
        #     self.register_factor(
        #         name=factor_name,
        #         category=category or FactorCategory.CUSTOM,
        #         description=f"自动注册的{category.value if isinstance(category, FactorCategory) else category or '自定义'}因子",
        #         data_requirements=["price", "returns"]
        #     )

        # 执行测试
        test_result = self.single_factor_tester.comprehensive_test(
            factor_data=factor_data,
            factor_name=factor_name,
            **test_kwargs
        )
        self.factor_manager._save_results(test_result, factor_name)
        return test_result

    # 批量测试！起始。先配置 基础底层target因子，比如价格，。。 然后自己换算出目标因子，然后为给这个factor_data_dict todo
    def batch_test_factors(self,
                           factor_data_dict: Dict[str, pd.DataFrame],
                           factor_category_type_dict: Dict[str, str],
                           **test_kwargs) -> Dict[str, Any]:
        """
        批量测试因子
        
        Args:
            factor_data_dict: 因子数据字典
            category_mapping: 因子类别映射
            **test_kwargs: 测试参数
        """

        # 初始化单因子测试器
        if self.single_factor_tester is None:
            self.single_factor_tester = SingleFactorTester(
                data_dict=self.data_manager.raw_data,
                config=self.config

            )

        # 批量测试
        results = {}
        for factor_name, factor_data in factor_data_dict.items():
            try:
                # 执行测试
                result = self.test_single_factor(
                    factor_name=factor_name,
                    factor_data=factor_data,
                    factor_category_type=factor_category_type_dict[factor_name],
                    **test_kwargs
                )
                self.factor_manager._save_results(results[factor_name], factor_name)
                results[factor_name] = result
            except Exception as e:
                logger.error(f"因子 {factor_name} 测试失败: {e}")

        return results

    def get_factor_performance_summary(self,
                                       category: Union[str, FactorCategory] = None) -> pd.DataFrame:
        """获取因子性能汇总"""
        summary = self.factor_manager.get_factor_summary()

        # 筛选类别
        if category is not None:
            if isinstance(category, FactorCategory):
                category_value = category.value
            else:
                try:
                    category_value = FactorCategory[category.upper()].value
                except KeyError:
                    raise ValueError(f"未知的因子类别: {category}")


            summary = summary[summary['category'] == category_value]

        # 排序
        if 'test_overall_score' in summary.columns:
            summary = summary.sort_values('test_overall_score', ascending=False)

        return summary

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
        """
        return self.factor_manager.get_top_factors(
            category=category,
            top_n=top_n,
            min_score=min_score
        )

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
            # TODO: 实现从存储中加载因子数据

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

    def export_results(self, output_dir: str = None) -> Dict[str, str]:
        """
        导出结果
        
        Args:
            output_dir: 输出目录
            
        Returns:
            导出文件路径字典
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.workspace_dir / "exports" / f"export_{timestamp}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # 导出因子摘要
        summary = self.get_factor_performance_summary()
        if not summary.empty:
            summary_file = output_dir / "factor_summary.xlsx"
            summary.to_excel(summary_file, index=False)
            exported_files['summary'] = str(summary_file)

        # 导出配置
        config_file = output_dir / "factory_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        exported_files['config'] = str(config_file)

        # 导出测试结果
        results_file = output_dir / "test_results.json"
        test_results = {}
        for factor_name in self.factor_manager.list_factors():
            result = self.factor_manager.get_test_result(factor_name)
            if result:
                test_results[factor_name] = result.to_dict()

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        exported_files['results'] = str(results_file)

        logger.info(f"结果导出完成: {output_dir}")
        return exported_files

    def create_factor_pipeline(self,
                               factor_configs: List[Dict[str, Any]]) -> 'FactorPipeline':
        """创建因子流水线"""
        return FactorPipeline(self, factor_configs)

    # 返回目标学术因子 （通过计算base Factor
    def get_config_target_factor_dict_by_cal_base_factor_batch(self, raw_data_dict: Dict[str, pd.DataFrame]) -> Tuple[
        Dict[
            str, pd.DataFrame], Dict[str,str]]:
        ret_data_dict = {}
        factor_category_dict = {}
        # 拿到目标学术因子
        aca_target_factors = self.config['target_factor']['fields']
        for target_factor in aca_target_factors:
            target_data_df, category_type = self.get_config_target_factor_dict_by_cal_base_factor(target_factor,
                                                                                                  raw_data_dict)
            ret_data_dict.update({target_factor: target_data_df})
            factor_category_dict.update({target_factor: category_type})
        return ret_data_dict, factor_category_dict

    # 返回目标学术因子 （通过计算base Factor
    def get_config_target_factor_dict_by_cal_base_factor(self, target_factor_name,
                                                         raw_data_dict: Dict[str, pd.DataFrame]) -> Tuple[
        pd.DataFrame, str]:
        factor_definition = pd.DataFrame(self.config['factor_definition'])
        # 价值因子
        if 'pe_ttm_inv' == target_factor_name:
            # PE因子
            pe_data = raw_data_dict['pe_ttm'].copy()
            pe_data = pe_data.where(pe_data > 0)  # 只过滤<=0的异常值
            category_type = factor_definition[factor_definition['name'] == target_factor_name]['category_type']
            return 1 / pe_data, category_type
        raise ValueError("请定义学术因子计算逻辑！")


class FactorPipeline:
    """因子研究流水线"""

    def __init__(self, factory: StrategyFactory, factor_configs: List[Dict[str, Any]]):
        """初始化因子流水线"""
        self.factory = factory
        self.factor_configs = factor_configs

    def run(self) -> Dict[str, Any]:
        """运行流水线"""
        results = {}

        # 1. 加载数据
        data_dict = self.factory.load_all_data_be_universe()

        # 2. 创建因子
        factor_dict = {}
        for config in self.factor_configs:
            factor_name = config.get('name')
            factor_type = config.get('type')
            factor_params = config.get('params', {})

            # TODO: 实现因子创建逻辑

        # 3. 批量测试
        test_results = self.factory.batch_test_factors(factor_dict)
        results['test_results'] = test_results

        # 4. 优化组合
        optimized_factor = self.factory.optimize_factors(factor_dict)
        results['optimized_factor'] = optimized_factor

        # 5. 导出结果
        exported_files = self.factory.export_results()
        results['exported_files'] = exported_files

        return results
