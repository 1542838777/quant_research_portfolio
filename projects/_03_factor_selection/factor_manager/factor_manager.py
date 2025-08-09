"""
因子管理器模块

整合因子注册表、分类器和存储功能，提供统一的因子管理接口。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np
import pandas as pd

from quant_lib import setup_logger
from quant_lib.config.logger_config import log_warning
from .classifier.factor_calculator.factor_calculator import FactorCalculator
from .classifier.factor_classifier import FactorClassifier
from .factor_technical_cal.factor_technical_cal import calculate_rolling_beta
# 导入子模块
from .registry.factor_registry import FactorRegistry, FactorCategory, FactorMetadata
from .storage.single_storage import add_single_factor_test_result
from ..config.base_config import INDEX_CODES
from ..data_manager.data_manager import DataManager, align_one_df_by_stock_pool_and_fill

logger = setup_logger(__name__)


class FactorTestResult:
    """因子测试结果类"""

    def __init__(self,
                 **kwargs):

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
                 data_manager: DataManager=None,
                 results_dir: str = Path(__file__).parent.parent/"workspace/factor_results",
                 registry_path: str = "factor_registry.json",
                 config: Dict[str, Any] = None):


        """
        初始化因子管理器

        Args:
            results_dir: 测试结果保存目录
            registry_path: 注册表文件路径
        """
        # 因子缓存字典，用于存储已经计算好的因子，避免重复计算
        self.factors_cache: Dict[str, pd.DataFrame] = {}  # 添加其他测试结果
        self.calculator = FactorCalculator(self)
        self.data_manager = data_manager
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.registry = FactorRegistry(registry_path)
        self.classifier = FactorClassifier()

        # 测试结果缓存
        self.test_results = {}

        logger.info("因子管理器初始化完成")
    # 返回未经过任何shift的数据 来自于raw_df的数据 ，以及自己手动计算的数据比如beta
    def get_factor(self, factor_name: str) -> pd.DataFrame:
        """
        【核心】获取因子的统一接口。
        """
        if factor_name in self.factors_cache:
            print(f"  > 从缓存中加载因子: {factor_name}")
            return self.factors_cache[factor_name].copy()

        # 【核心修改】: 在 self.calculator 中查找计算方法
        calculation_method_name = f"_calculate_{factor_name}"

        if hasattr(self.calculator, calculation_method_name):
            logger.info(f"  > 发现有专门的因子计算函数，因子计算哈函数开始执行: {calculation_method_name}...")
            method_to_call = getattr(self.calculator, calculation_method_name)

            factor_df = method_to_call()

            self.factors_cache[factor_name] = factor_df
            print(f"  > 因子 {factor_name} 计算完成并已存入缓存。")

            return factor_df.copy()
        else:
            if factor_name in self.data_manager.raw_dfs:
                logger.info(f"  >  没有找到专属计算函数，大概因子 {factor_name} 是raw因子，直接从raw_dfs加载。")
                factor_df = self.data_manager.raw_dfs[factor_name]
                self.factors_cache[factor_name] = factor_df #这也需要？感觉好占内存
                return factor_df.copy()
            else:
                raise ValueError(f"失败错误：没有找到专属计算函数，也没有在原始数据中找到因子 '{factor_name}'。")
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



    def get_top_factors(self):

         return None

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

    def _save_results(self, results: Dict[str, Any],file_name_prefix: str) -> None:
        """保存测试结果"""
        # 准备可序列化的结果
        serializable_results = self._make_serializable(results)

        # 保存JSON格式
        json_path = os.path.join(self.results_dir, f'all_single_factor_test_{file_name_prefix}_results.json')
        add_single_factor_test_result(json_path, serializable_results)
    #ok 保存 精简简要的测试结果
    def update_and_save_factor_purify_summary(self, all_summary_rows: list, file_name_prefix: str):
        """
           更新或创建因子排行榜，支持增量更新。
           如果文件已存在，则删除本次测试涉及的因子和周期的旧记录，并追加新记录。
           如果文件不存在，则创建新文件。
           """
        if not all_summary_rows:
            print("警告：没有新的测试结果可供更新。")
            return

        # 1. 准备新数据
        new_results_df = pd.DataFrame(all_summary_rows)
        # (推荐) 在合并前先排序，保持数据条理性
        new_results_df.sort_values(by=['factor_name', 'period'], inplace=True)

        # 2. 【修正Bug 3】正确构建文件路径
        output_dir = Path(f'{self.results_dir}')
        output_dir.mkdir(exist_ok=True)
        parquet_path = output_dir / f'all_single_factor_test_{file_name_prefix}.parquet'
        csv_path = output_dir / f'all_single_factor_test_{file_name_prefix}.csv'

        # 3. 【修正Bug 1】安全地读取旧数据
        try:
            existing_leaderboard = pd.read_parquet(parquet_path)
        except FileNotFoundError:
            log_warning(f"信息：未找到现有的排行榜文件 at {parquet_path}。将创建新文件。")
            existing_leaderboard = pd.DataFrame()

        # 4. 【修正逻辑风险 4】从新数据中提取所有待更新的“主键”
        #    这样即使一次传入多个因子的结果也能正确处理
        keys_to_update = new_results_df[['factor_name', 'backtest_period','backtest_base_on_index']].drop_duplicates()

        # 5. 删除旧记录
        if not existing_leaderboard.empty:
            # 使用 merge + indicator 来找到并排除需要删除的行
            ##
            # indicator=True
            # 结果就是：新生成_merger列，值要么是both 要么是left_only#
            merged = existing_leaderboard.merge(keys_to_update, on=['factor_name', 'backtest_period','backtest_base_on_index'], how='left',
                                                indicator=True)
            leaderboard_to_keep = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
        else:
            leaderboard_to_keep = existing_leaderboard

        # 6. 【修正Bug 2】合并旧的“保留”数据和所有新数据
        final_leaderboard = pd.concat([leaderboard_to_keep, new_results_df], ignore_index=True)

        # 7. 保存最终的排行榜
        try:
            final_leaderboard.to_parquet(parquet_path, index=False)
            print(f"✅ 因子排行榜已成功更新并保存至: {parquet_path}")

            final_leaderboard.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 因子排行榜已成功更新并保存至: {csv_path}")

        except Exception as e:
            print(f"❌ 保存结果时发生错误: {e}")
            raise e  # 重新抛出异常，让上层知道发生了错误


    def update_and_save_fm_factor_return_matrix(self, new_fm_factor_returns_dict: dict, file_name_prefix: str):
        """
        【新】更新或创建统一的因子收益矩阵文件。
        如果文件已存在，则用新的收益序列覆盖掉同名的旧序列。

        Args:
            new_fm_factor_returns_dict (dict): 本次测试产出的新收益序列字典。
                                          键为 'factor_name_period' (如 'momentum_2_1_20d')，
                                          值为 pd.Series。
            file_name_prefix (str): 文件名前缀。
        """
        if not new_fm_factor_returns_dict:
            print("警告：没有新的因子收益序列可供更新。")
            return

        # 1. 准备新数据：将输入的字典转换为一个“宽格式”的DataFrame
        new_returns_df = pd.DataFrame(new_fm_factor_returns_dict)

        # 2. 正确构建文件路径
        output_dir = Path(f'{self.results_dir}')
        output_dir.mkdir(exist_ok=True)
        parquet_path = output_dir / f'all_single_factor_fm_returns_{file_name_prefix}.parquet'
        csv_path = output_dir / f'all_factor_returns_{file_name_prefix}.csv'

        # 3. 安全地读取旧的收益矩阵
        try:
            existing_matrix = pd.read_parquet(parquet_path)
        except FileNotFoundError:
            print(f"信息：未找到现有的收益矩阵文件 at {parquet_path}。将创建新文件。")
            existing_matrix = pd.DataFrame()

        # 4. 识别需要被替换的旧列
        #    这些列的名字，就是新数据 new_returns_df 的列名
        cols_to_update = new_returns_df.columns

        # 找出在旧矩阵中确实存在的、需要被删除的列
        cols_to_drop = [col for col in cols_to_update if col in existing_matrix.columns]

        # 5. 删除旧列，得到需要保留的旧矩阵部分
        matrix_to_keep = existing_matrix.drop(columns=cols_to_drop)

        # 6. 合并“保留的旧矩阵”和“所有新数据”
        #    axis=1 表示按列进行合并。Pandas会自动按索引（日期）对齐。
        final_matrix = pd.concat([matrix_to_keep, new_returns_df], axis=1)

        # 7. (推荐) 按列名排序，让文件结构更清晰
        final_matrix.sort_index(axis=1, inplace=True)

        # 8. 保存最终的、更新后的收益矩阵
        try:
            final_matrix.to_parquet(parquet_path, index=True)  # 收益序列，索引(日期)需要保存
            print(f"✅ 因子收益矩阵已成功更新并保存至: {parquet_path}")

            final_matrix.to_csv(csv_path, index=True, encoding='utf-8-sig')
            print(f"✅ 因子收益矩阵已成功更新并保存至: {csv_path}")

        except Exception as e:
            print(f"❌ 保存因子收益矩阵时发生错误: {e}")
            raise e

    def get_backtest_ready_factor_entity(self):

        technical_df_dict = {}
        technical_category_dict = {}
        technical_school_dict = {}

        # 找出所有目标target 因子。
        # 通过config的标识 找出需要学术计算的因子
        # 自生的门派，重新align Require的因子，参与计算，返回学术_df
        target_factors_for_evaluation = self.data_manager.config['target_factors_for_evaluation']['fields']

        for target_factor_name in target_factors_for_evaluation:
            # category
            category = self.get_style_category(target_factor_name)
            school = self.get_school_code_by_factor_name(target_factor_name)
            target_data_df = self.get_backtest_ready_factor(target_factor_name)
            technical_df_dict.update({target_factor_name: target_data_df})
            technical_category_dict.update({target_factor_name: category})
            technical_school_dict.update({target_factor_name: school})

        return technical_df_dict, technical_category_dict, technical_school_dict

    def get_backtest_ready_factor(self, factor_name):
       df = self.get_factor(factor_name)#这是纯净的因子，
       pool = self.get_stock_pool_by_factor_name(factor_name)#拿到之前基于t-1信息 构建的动态股票池
       # 对整个因子矩阵进行.shift(1)，用昨天的数据 t-1
       df=df.shift(1)
       return align_one_df_by_stock_pool_and_fill(factor_name=factor_name,
                                                  df=df, stock_pool_df=pool,_existence_matrix = self.data_manager._existence_matrix)


    def get_school_code_by_factor_name(self, factor_name):
        return self.get_school_by_style_category(self.get_style_category(factor_name))

    def get_stock_pool_by_factor_name(self, factor_name):
        school_code = self.get_school_code_by_factor_name(factor_name)
        pool_name = self.get_stock_pool_name_by_factor_school(school_code)
        return self.data_manager.stock_pools_dict[pool_name]

    def get_stock_pool_name_by_factor_school(self, factor_school):
        if factor_school in ['fundamentals', 'trend']:
            return 'institutional_stock_pool'
        if factor_school in ['microstructure']:
            return 'microstructure_stock_pool'
        raise ValueError(f'{factor_school}没有定义因子属于哪一门派')
    @staticmethod
    def get_school_by_style_category(style_category: str) -> str:
        """
        根据因子风格(style_category)，返回其所属的投资门派(school)。
        这是连接因子定义与业务逻辑（如选择股票池）的核心枢纽。

        Args:
            style_category (str): 因子定义中的风格类别。

        Returns:
            str: 'fundamentals', 'trend', or 'microstructure'.

        Raises:
            ValueError: 如果输入的style_category未被定义。
        """
        # 定义从“风格”到“门派”的映射关系
        # 这是我们系统的“唯一真实来源”
        SCHOOL_MAP = {
            # === 基本面派 (fundamentals) ===
            # 源于公司财务报表或其内在属性，反映了公司的长期价值。
            'value': 'fundamentals',
            'quality': 'fundamentals',
            'growth': 'fundamentals',
            'size': 'fundamentals',
            'sector': 'fundamentals',

            # === 趋势派 (trend) ===
            # 反映了价格在历史序列中的行为模式和风险特征。
            'momentum': 'trend',
            'risk': 'trend',  # Beta和波动率描述了股票在趋势中的行为特征和敏感度

            # === 微观派 (microstructure) ===
            # 直接来源于市场的实际交易行为（价格、成交量、换手率）。
            'liquidity': 'microstructure',
            'price': 'microstructure',
            'return': 'microstructure'
        }
        school = SCHOOL_MAP.get(style_category)

        if school is None:
            raise ValueError(
                f"无法识别的因子风格 '{style_category}'，请在 get_school_by_style_category 函数中定义其门派。")
        return school

    def get_stock_pool_index_by_factor_name(self, factor_name):
        # 拿到对应pool_name
        pool_name = self.get_stock_pool_name_by_factor_name(factor_name)

        index_filter_config = self.data_manager.config['stock_pool_profiles'][pool_name]['index_filter']
        if  not index_filter_config['enable']:
            return INDEX_CODES['ALL_A']
        return index_filter_config['index_code']

    def get_stock_pool_name_by_factor_name(self, factor_name):
        school_code = self.get_school_code_by_factor_name(factor_name)
        return self.get_stock_pool_name_by_factor_school(school_code)

    # 获取 因子所对应股票池 股票池所有的stock_codes
    def get_pool_of_factor_name_of_stock_codes(self, target_factor_name):
        pool = self.get_stock_pool_by_factor_name(factor_name=target_factor_name)
        return list(pool.columns)



    def get_style_category(self, factor_name):
        factor_definition = pd.DataFrame(self.data_manager.config['factor_definition'])
        return factor_definition[factor_definition['name'] == factor_name]['style_category'].iloc[0]



    # ok
    def build_auxiliary_dfs_shift_diff_stock_pools_dict(self):
        dfs_dict = self.build_dfs_dict_base_on_diff_pool_name(['total_mv', 'industry'])
        dfs_dict = self.build_df_dict_base_on_diff_pool_can_set_shift(
            base_dict=dfs_dict, need_shift=True)
        pct_chg_beta_dict = self.build_df_dict_base_on_diff_pool_can_set_shift(
            base_dict=self.get_pct_chg_beta_dict(), factor_name='pct_chg', need_shift=True)

        for stock_poll_name, df in pct_chg_beta_dict.items():
            # 补充beta
            dfs_dict[stock_poll_name].update({'pct_chg_beta': df})
        return dfs_dict

    def build_dfs_dict_base_on_diff_pool_name(self, factor_name_data: Union[str, list]):
        if isinstance(factor_name_data, str):
            return self.build_one_df_dict_base_on_diff_pool_name(factor_name_data)
        if isinstance(factor_name_data, list):
            dicts = []
            for factor_name in factor_name_data:
                pool_df_dict = self.build_one_df_dict_base_on_diff_pool_name(factor_name)
                dicts.append((factor_name, pool_df_dict))  # 保存因子名和对应的dict

            merged = {}
            for factor_name, pool_df_dict in dicts:
                for pool, df in pool_df_dict.items():
                    if pool not in merged:
                        merged[pool] = {}
                    merged[pool][factor_name] = df

            return merged

        raise TypeError("build_df_dict_base_on_diff_pool 入参类似有误")
    # 仅仅只是构造一个dict 不做任何处理!
    def build_one_df_dict_base_on_diff_pool_name(self, factor_name):
        df_dict_base_on_diff_pool = {}
        df = self.data_manager.raw_dfs[factor_name]
        for pool_name, stock_pool_df in self.data_manager.stock_pools_dict.items():
            df_dict_base_on_diff_pool[pool_name] = df
        return df_dict_base_on_diff_pool

    def build_df_dict_base_on_diff_pool_can_set_shift(self, base_dict=None, factor_name=None, need_shift=True):
        if base_dict is None:
            base_dict = self.build_one_df_dict_base_on_diff_pool_name(factor_name)
        if need_shift:
            ret = self.do_shift_and_align_for_dict(factor_name =factor_name , data_dict = base_dict,  _existence_matrix = self.data_manager._existence_matrix)
            return ret
        else:
            ret = self.do_align_for_dict(factor_name, base_dict)
            return ret

    def do_shift_and_align_for_dict(self, factor_name=None, data_dict=None,_existence_matrix:pd.DataFrame=None):
        result = {}
        for stock_name, stock_pool in self.data_manager.stock_pools_dict.items():
            ret = self.do_shift_and_align_where_stock_pool(factor_name, data_dict[stock_name], stock_pool,_existence_matrix = _existence_matrix)
            result[stock_name] = ret
        return result

    def do_align_for_dict(self, factor_name, data_dict):
        result = {}
        for stock_name, stock_pool in self.data_manager.stock_pools_dict.items():
            ret = self.do_align(factor_name, data_dict[stock_name], stock_pool)
            result[stock_name] = ret
        return result

    def do_shift_and_align_where_stock_pool(self, factor_name, data_to_deal, stock_pool,_existence_matrix:pd.DataFrame=None):
        # 率先shift
        data_to_deal_by_shifted = self.do_shift(data_to_deal)
        # 对齐
        result = self.do_align(factor_name, data_to_deal_by_shifted, stock_pool,_existence_matrix = _existence_matrix)
        return result

    def do_shift(
            self,
            data_to_shift: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        对输入的数据执行 .shift(1) 操作，智能处理单个DataFrame或DataFrame字典。
        Args:
            data_to_shift: 需要进行滞后处理的数据，
                           可以是一个 pandas DataFrame，
                           也可以是一个 key为字符串, value为pandas DataFrame的字典。
        Returns:
            一个与输入类型相同的新对象，其中所有的DataFrame都已被 .shift(1) 处理。
        """
        # --- 情况一：输入是字典 ---
        if isinstance(data_to_shift, dict):
            shifted_dict = {}
            for key, df in data_to_shift.items():
                if not isinstance(df, pd.DataFrame):
                    raise ValueError("do_shift失败,dict内部不是df结构")
                # 对字典中的每个DataFrame执行shift操作
                shifted_dict[key] = df.shift(1)
            return shifted_dict

        # --- 情况二：输入是单个DataFrame ---
        elif isinstance(data_to_shift, pd.DataFrame):
            return data_to_shift.shift(1)

        # --- 其他情况：输入类型错误，主动报错 ---
        else:
            raise TypeError(
                f"输入类型不支持，期望是DataFrame或Dict[str, DataFrame]，"
                f"但收到的是 {type(data_to_shift).__name__}"
            )

    def do_align(self, factor_name, data_to_align: Union[pd.DataFrame, Dict[str, pd.DataFrame]], stock_pool, _existence_matrix:pd.DataFrame=None
                 ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        # --- 情况一：输入是字典 ---
        if isinstance(data_to_align, dict):
            shifted_dict = {}
            for key, df in data_to_align.items():
                if not isinstance(df, pd.DataFrame):
                    raise ValueError("do_shift失败,dict内部不是df结构")
                # 对字典中的每个DataFrame执行shift操作
                shifted_dict[key] = align_one_df_by_stock_pool_and_fill(factor_name=key, df=df,
                                                                        stock_pool_df=stock_pool,_existence_matrix = _existence_matrix)
            return shifted_dict

        # --- 情况二：输入是单个DataFrame ---
        elif isinstance(data_to_align, pd.DataFrame):
            return align_one_df_by_stock_pool_and_fill(factor_name=factor_name, df=data_to_align,
                                                       stock_pool_df=stock_pool,_existence_matrix = _existence_matrix)

        # --- 其他情况：输入类型错误，主动报错 ---
        else:
            raise TypeError(
                f"输入类型不支持，期望是DataFrame或Dict[str, DataFrame]，"
                f"但收到的是 {type(data_to_align).__name__}"
            )


    # ok 因为需要滚动计算，所以不依赖股票池的index（trade） 只要对齐股票列就好
    def get_pct_chg_beta_dict(self):
        dict = {}
        for pool_name, _ in self.data_manager.stock_pools_dict.items():
            beta_df = self.get_pct_chg_beta_data_for_pool(pool_name)
            dict[pool_name] = beta_df
        return dict

    def get_pct_chg_beta_data_for_pool(self, pool_name):
        pool_stocks = self.data_manager.stock_pools_dict[pool_name].columns

        # 直接从主Beta矩阵中按需选取，无需重新计算
        beta_for_this_pool = self.prepare_master_pct_chg_beta_dataframe()[pool_stocks]

        return beta_for_this_pool
    def prepare_master_pct_chg_beta_dataframe(self):
        """
        一个在系统初始化时调用的方法，用于生成一份统一的、覆盖所有股票的Beta矩阵。
        """
        logger.info("开始准备主Beta矩阵...")

        # 1. 整合所有股票池的股票代码，形成一个总的股票列表
        all_unique_stocks = set()
        for stock_pool in self.data_manager.stock_pools_dict.values():
            all_unique_stocks.update(stock_pool.columns)

        master_stock_list = sorted(list(all_unique_stocks))

        # 2. 只调用一次 calculate_rolling_beta，计算所有股票的Beta
        logger.info(f"开始为总计 {len(master_stock_list)} 只股票计算统一的Beta...")
        return calculate_rolling_beta(
            self.data_manager.config['backtest']['start_date'],
            self.data_manager.config['backtest']['end_date'],
            master_stock_list
        )
