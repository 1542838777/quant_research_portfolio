"""
策略工厂模块

整合因子管理器、单因子测试器、多因子优化器和可视化管理器，
提供一站式量化研究解决方案。
"""
import sys
import traceback

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Tuple
from pathlib import Path
import yaml
from datetime import datetime
import json

from ..config.config_file.load_config_file import _load_local_config
from ..factor_manager.factor_manager import FactorManager
from ..factor_manager.factor_technical_cal.factor_technical_cal import calculate_rolling_beta
from ..single_factor_tester.single_factor_tester import SingleFactorTester
from ..factor_manager.registry.factor_registry import FactorCategory
from ..multi_factor_optimizer.multi_factor_optimizer import MultiFactorOptimizer
from ..data_manager.data_manager import DataManager, align_one_df_by_stock_pool_and_fill
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

    def get_category_type(self, factor_name):
        factor_definition = self.data_manager.config['factor_definition']
        return factor_definition[factor_definition['name'] == factor_name]['category_type']

    def get_school(self, factor_name):
        factor_definition = self.data_manager.config['factor_definition']
        return factor_definition[factor_definition['name'] == factor_name]['school']


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

    def batch_test_factors(self,
                           target_factors_dict: Dict[str, pd.DataFrame],
                           target_factors_category_dict: Dict[str, str],
                           target_factor_school_type_dict: Dict[str, str],
                           **test_kwargs) -> Dict[str, Any]:
        """
        批量测试因子
        """

        # 初始化单因子测试器
        if self.single_factor_tester is None:
            self.single_factor_tester = SingleFactorTester(
                raw_dfs=self.data_manager.raw_dfs,
                target_factors_dict=target_factors_dict,
                target_factors_category_dict=target_factors_category_dict,
                target_factor_school_type_dict=target_factor_school_type_dict,
                stock_pools_dict=self.data_manager.stock_pools_dict,
                config=self.config
            )

        # 批量测试
        results = {}
        for factor_name, factor_data in target_factors_dict.items():
            try:
                # 执行测试
                ic_series_periods_dict,quantile_returns_series_periods_dict,factor_returns_series_periods_dict,summary_stats = (self.single_factor_tester.
                test_single_factor_entity_service(
                    target_factor_name=factor_name,
                    factor_manager=self.factor_manager,
                    visualization_manager = self.visualization_manager
                ))
                results[factor_name] = summary_stats
            except Exception as e:
                # traceback.print_exc()
                raise ValueError(f"✗ 因子{factor_name}测试失败: {e}") from e

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

    def create_factor_pipeline(self,
                               factor_configs: List[Dict[str, Any]]) -> 'FactorPipeline':
        """创建因子流水线"""
        return FactorPipeline(self, factor_configs)


    # 返回目标学术因子 （通过计算base Factor
    def get_done_cal_factor_and_category_and_school(
            self,
            target_factor_name: str,
            raw_data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, str, str]:
        """
        【专业版】因子计算工厂。
        根据因子名称，调用对应的计算逻辑，并返回处理好的因子DataFrame及其元数据。
        """
        # --- 在函数开头一次性查找因子定义 ---
        factor_definitions = pd.DataFrame(self.config['factor_definition'])
        factor_info = factor_definitions[factor_definitions['name'] == target_factor_name]

        if factor_info.empty:
            raise ValueError(f"在配置文件中未找到因子 '{target_factor_name}' 的定义！")

        # 将Series转换为单个值，方便调用
        category_type = factor_info['category_type'].iloc[0]
        school = factor_info['school'].iloc[0]

        logger.info(f"开始计算学术因子: '{target_factor_name}' (门派: {school})")

        # --- 因子计算逻辑的分发 ---

        if 'bm_ratio' == target_factor_name:
            # 账面市值比 (B/M), 即市净率倒数
            pb_df = raw_data_dict['pb'].copy()
            # PB值必须为正才有意义 (负的净资产，B/M失去意义)
            pb_df = pb_df.where(pb_df > 0)
            factor_df = 1 / pb_df
            return factor_df, category_type, school

        elif 'momentum_12_1' == target_factor_name:
            # 经典12-1月动量
            close_df = raw_data_dict['close'].copy()
            # 计算 T-1月 / T-12月 的价格比
            # 假设每月21个交易日，每年252个交易日
            price_1m_ago = close_df.shift(21)
            price_12m_ago = close_df.shift(252)

            # 确保分母不为0或负（虽然股价基本不会）
            price_12m_ago = price_12m_ago.where(price_12m_ago > 0)

            factor_df = (price_1m_ago / price_12m_ago) - 1
            return factor_df, category_type, school

        elif 'momentum_2_1' == target_factor_name:
            # 经典2-1月动量
            close_df = raw_data_dict['close'].copy()
            # 计算 T-1月 / T-12月 的价格比
            # 假设每月21个交易日，每年252个交易日
            price_1m_ago = close_df.shift(21)
            price_2m_ago = close_df.shift(21*2)

            # 确保分母不为0或负（虽然股价基本不会）
            price_2m_ago = price_2m_ago.where(price_2m_ago > 0)

            factor_df = (price_1m_ago / price_2m_ago) - 1
            return factor_df, category_type, school

        elif 'turnover_rate_abnormal_20d' == target_factor_name:
            # 异常换手率（20日窗口）
            turnover_df = raw_data_dict['turnover_rate'].copy()

            # 计算20日滚动均值，min_periods=10确保在数据初期也能尽快产出信号
            turnover_mean_20d = turnover_df.rolling(window=20, min_periods=10).mean()

            # 用当日值减去均值，得到“超预期”的异动信号
            factor_df = turnover_df - turnover_mean_20d
            return factor_df, category_type, school

        # --- 如果有更多因子，在这里继续添加 elif 分支 ---

        # --- 毕业考题：规模因子 ---
        elif 'market_cap_log' == target_factor_name:
            # 获取市值数据
            total_mv_df = raw_data_dict['circ_mv'].copy()
            # 保证为正数，避免log报错
            total_mv_df = total_mv_df.where(total_mv_df > 0)
            # 使用 pandas 自带 log 函数，保持类型一致
            factor_df = total_mv_df.apply(np.log)
            # 反向处理因子（仅为了视觉更好看）
            factor_df = -1 * factor_df
            return factor_df, category_type, school

        # --- 毕业考题：价值因子 ---
        elif 'pe_ttm_inv' == target_factor_name:
            pe_df = raw_data_dict['pe_ttm'].copy()
            # PE为负或0时，其倒数无意义，设为NaN
            pe_df = pe_df.where(pe_df > 0)
            factor_df = 1 / pe_df
            return factor_df, category_type, school
        elif 'ps_ttm_inv' == target_factor_name:
            pe_df = raw_data_dict['ps_ttm'].copy()
            # PE为负或0时，其倒数无意义，设为NaN
            pe_df = pe_df.where(pe_df > 0)
            factor_df = 1 / pe_df
            return factor_df, category_type, school
        elif 'pb_inv' == target_factor_name:
            pe_df = raw_data_dict['pb'].copy()
            # PE为负或0时，其倒数无意义，设为NaN
            pe_df = pe_df.where(pe_df > 0)
            factor_df = 1 / pe_df
            return factor_df, category_type, school

        elif 'bm_ratio' == target_factor_name:
            pb_df = raw_data_dict['pb'].copy()
            # PB为负或0时（公司净资产为负），其倒数无意义
            pb_df = pb_df.where(pb_df > 0)
            factor_df = 1 / pb_df
            return factor_df, category_type, school

        # --- 毕业考题：动量因子 ---
        elif 'momentum_20d' == target_factor_name:
            # 小白解释：动量 = (T-1日价格) / (T-21日价格) - 1
            # 因为传入的close_df已经是T-1的价格，所以我们只需要再shift(20)即可得到T-21的价格
            close_df = raw_data_dict['close'].copy()  # 这是T-1价格

            # 获取约20个交易日前的价格 (T-1-20 = T-21)
            price_20d_ago = close_df.shift(20)

            price_20d_ago = price_20d_ago.where(price_20d_ago > 0)
            factor_df = (close_df / price_20d_ago) - 1
            return factor_df, category_type, school

        # --- 其他量价因子 ---
        elif 'turnover_rate_abnormal_20d' == target_factor_name:
            turnover_df = raw_data_dict['turnover_rate'].copy()  # T-1日的换手率
            # 计算过去20日的滚动均值 (T-20 到 T-1)
            turnover_mean_20d = turnover_df.rolling(window=20, min_periods=10).mean()
            # 用T-1日的值减去均值
            factor_df = turnover_df - turnover_mean_20d
            return factor_df, category_type, school
        elif 'beta' == target_factor_name:
            beta_df = calculate_rolling_beta(
               self.config['backtest']['start_date'],
                self.config['backtest']['end_date'],
                self.data_manager.get_pool_of_factor_name_of_stock_codes(target_factor_name)
            )
            beta_df = beta_df * -1
            return beta_df.shift(1), category_type, school
        #todo remind 注意 ，自己找的数据（不在raw——df之内的，都要自行shift1）

        # --- 如果有更多因子，在这里继续添加 elif 分支 ---

        raise ValueError(f"因子 '{target_factor_name}' 的计算逻辑尚未在本工厂中定义！")

    def get_target_factors_entity(self):

        technical_df_dict = {}
        technical_category_dict = {}
        technical_school_dict = {}

        # 找出所有目标target 因子。
        # 通过config的标识 找出需要学术计算的因子
        # 自生的门派，重新align Require的因子，参与计算，返回学术_df
        target_factors_for_evaluation = self.data_manager.config['target_factors_for_evaluation']['fields']

        for target_factor_name in target_factors_for_evaluation:
            need_technical_cal = self.get_one_factor_denifition(target_factor_name)['need_technical_cal']
            if need_technical_cal:
                # 根据门派，找出所需股票池
                # 自行计算！
                target_data_df, category_type, school = self.build_technical_factor_entity_base_on_shift_and_align_stock_pools(target_factor_name)

            else:
                # 不需要额外学术计算
                target_data_df, category_type, school = self.build_base_factor_entity_base_on_shift_and_align_stock_pools(target_factor_name)
            technical_df_dict.update({target_factor_name: target_data_df})
            technical_category_dict.update({target_factor_name: category_type})
            technical_school_dict.update({target_factor_name: school})

        return technical_df_dict, technical_category_dict, technical_school_dict

    def get_one_factor_denifition(self, target_factor_name):
        factor_definition = self.data_manager.config['factor_definition']
        factor_definition_dict = {item['name']: item for item in factor_definition}
        return factor_definition_dict.get(target_factor_name)

    # 注意 后续不要重复shift1了
    def build_technical_factor_entity_base_on_shift_and_align_stock_pools(self, target_factor_name):
        cal_require_base_fields = self.get_one_factor_denifition(target_factor_name)['cal_require_base_fields']
        stock_pool = self.data_manager.get_stock_pool_by_factor_name(target_factor_name)

        # 拿出require的原生df 基于同股票池维度对齐
        require_dfs_shifted = {field:self.data_manager.raw_dfs[field].shift(1) for field in cal_require_base_fields}
        # 自行计算！_align_many_raw_dfs_by_stock_pool_and_fill
        target_df,category,school =  self.get_done_cal_factor_and_category_and_school(
            target_factor_name,
            require_dfs_shifted)
        return align_one_df_by_stock_pool_and_fill(factor_name = target_factor_name,
                                                                      raw_df_param  = target_df, stock_pool_df= stock_pool),category,school
    def build_base_factor_entity_base_on_shift_and_align_stock_pools(self, target_factor_name):
        df = self.data_manager.raw_dfs[target_factor_name].shift(1)
        # category
        category = self.get_category_type(target_factor_name)
        school = self.get_school(target_factor_name)

        return df, category, school

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
        data_dict = None # self.factory.processed_raw_df_dict_by_stock_pool_processed()

        # 2. 创建因子
        factor_dict = {}
        for config in self.factor_configs:
            factor_name = config.get('name')
            factor_type = config.get('type')
            factor_params = config.get('params', {})


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
