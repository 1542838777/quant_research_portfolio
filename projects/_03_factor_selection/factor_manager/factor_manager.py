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
from numpyencoder import NumpyEncoder

from quant_lib import setup_logger
from quant_lib.config.logger_config import log_warning
from .classifier.factor_calculator.factor_calculator import FactorCalculator
from .classifier.factor_classifier import FactorClassifier
# 导入子模块
from .registry.factor_registry import FactorRegistry, FactorCategory, FactorMetadata
from .storage.single_storage import add_single_factor_test_result
from ..config.factor_direction_config import FACTOR_DIRECTIONS
from ..data_manager.data_manager import DataManager, fill_and_align_by_stock_pool

logger = setup_logger(__name__)


class FactorResultsManager:
    """因子测试结果类"""

    def __init__(self,
                 **kwargs):

        self.results_dir = Path(__file__).parent.parent / 'workspace' / 'result'

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactorResultsManager':
        """从字典创建"""
        return cls(**data)

    def _save_factor_results(self,
                             factor_name: str,
                             stock_index: str,  # 比如中证800
                             start_date: str, end_date: str,
                             returns_calculator_func_name: str,  # 新增参数，用于区分 'c2c' 或 'o2c'
                             results: Dict):
        """
        将单次因子测试的所有成果，保存到结构化的目录中。
        """
        # 1. 创建一个以日期范围命名的、唯一的版本文件夹
        run_version = f"{start_date.replace('-', '')}_{end_date.replace('-', '')}"
        # 1. 创建清晰的存储路径
        output_path = Path(self.results_dir) / stock_index / factor_name / returns_calculator_func_name / run_version
        output_path.mkdir(parents=True, exist_ok=True)

        # 2. 分解并保存不同的“成果”
        # a) 保存总结性统计数据
        ic_stats_periods_dict_raw = results.get("ic_stats_periods_dict_raw", {})
        ic_stats_periods_dict_processed = results.get("ic_stats_periods_dict_processed", {})
        quantile_stats_periods_dict_raw = results.get("quantile_stats_periods_dict_raw", {})
        quantile_stats_periods_dict_processed = results.get("quantile_stats_periods_dict_processed", {})
        fm_stat_results_periods_dict = results.get("fm_stat_results_periods_dict", {})
        turnover_stats_periods_dict = results.get("turnover_stats_periods_dict", {})
        style_correlation_dict = results.get("style_correlation_dict", {})


        summary_stats = {
            'ic_analysis_raw': ic_stats_periods_dict_raw,
            'ic_analysis_processed': ic_stats_periods_dict_processed,
            'quantile_backtest_raw': quantile_stats_periods_dict_raw,
            'quantile_backtest_processed': quantile_stats_periods_dict_processed,
            'fama_macbeth': fm_stat_results_periods_dict,
            'turnover': turnover_stats_periods_dict,
            'style_correlation':style_correlation_dict
        }
        with open(output_path / 'summary_stats.json', 'w') as f:
            # 使用自定义的Encoder来处理numpy类型
            json.dump(self._make_serializable(summary_stats), f, indent=4, cls=NumpyEncoder)

        if "processed_factor_df" in results:
            results["processed_factor_df"].to_parquet(output_path / 'processed_factor.parquet')

        ic_series_periods_dict_raw = results.get("ic_series_periods_dict_raw", {})
        ic_series_periods_dict_processed = results.get("ic_series_periods_dict_processed", {})



        q_daily_returns_df_raw = results.get("q_daily_returns_df_raw", pd.DataFrame())
        q_daily_returns_df_processed = results.get("q_daily_returns_df_processed", pd.DataFrame())

        quantile_returns_series_periods_dict_raw = results.get("quantile_returns_series_periods_dict_raw", {})
        quantile_returns_series_periods_dict_processed = results.get("quantile_returns_series_periods_dict_processed", {})
        fm_returns_series_periods_dict = results.get("fm_returns_series_periods_dict", {})

        # b) 保存时间序列数据 (以 Parquet 格式，更高效)
        for period, series in ic_series_periods_dict_raw.items():
            df = series.to_frame(name='ic_series_raw')  # 给一列起名，比如 'ic'
            df.to_parquet(output_path / f'ic_series_raw_{period}.parquet')
        for period, series in ic_series_periods_dict_processed.items():
            df = series.to_frame(name='ic_series_processed')  # 给一列起名，比如 'ic'
            df.to_parquet(output_path / f'ic_series_processed_{period}.parquet')

        for period, df in quantile_returns_series_periods_dict_raw.items():
            df.to_parquet(output_path / f'quantile_returns_raw_{period}.parquet')
        for period, df in quantile_returns_series_periods_dict_processed.items():
            df.to_parquet(output_path / f'quantile_returns_processed_{period}.parquet')

        q_daily_returns_df_raw.to_parquet(output_path / f'q_daily_returns_df_raw.parquet')
        q_daily_returns_df_processed.to_parquet(output_path / f'q_daily_returns_df_processed.parquet')



        for period, series in fm_returns_series_periods_dict.items():
            df = series.to_frame(name='fm_returns_series')
            df.to_parquet(output_path / f'fm_returns_series_{period}.parquet')

        logger.info(f"✓ 因子'{factor_name}'在配置'{returns_calculator_func_name}'下的所有结果已保存至: {output_path}")

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


class FactorManager:
    """因子管理器类"""

    def __init__(self,
                 data_manager: DataManager = None,
                 results_dir: str = Path(__file__).parent.parent / "workspace/factor_results",
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
    #带着规则！ 注意用的时候 这个方向 会不会对你有影响 注意2：没有对齐股票池噢，需要对齐 可以调用  get_prepare_aligned_factor_for_analysis
    def get_factor_by_rule(self, factor_request: Union[str, tuple]) -> pd.DataFrame:
        """
        【核心】获取因子的统一接口。
        """
        # 1. 调用最底层函数，获取纯净的原始因子
        #    直接将 factor_request 透传下去
        raw_factor_df = self.get_raw_factor(factor_request)

        # 2. 应用方向性调整
        #    我们需要从请求中解析出因子的基础名字
        factor_name_str = factor_request[0] if isinstance(factor_request, tuple) else factor_request
        direction = FACTOR_DIRECTIONS.get(factor_name_str, 1)

        if direction == -1:
            final_factor_df = raw_factor_df * -1
        else:
            final_factor_df = raw_factor_df

        return final_factor_df.copy()
    #最原始的因子获取，未经过任何处理，目前被使用于 因子计算
    def get_raw_factor(self, factor_request: Union[str, tuple]) -> pd.DataFrame:
        """
        【V3.0 - 参数化版】获取纯净的原始因子。
        能处理简单的字符串请求，也能处理带参数的元组请求。
        """
        # 1. 缓存键就是请求本身，元组是可哈希的，可以直接做键
        if factor_request in self.factors_cache:
            return self.factors_cache[factor_request]

        # 2. 解析请求
        if isinstance(factor_request, str):
            factor_name = factor_request
            params = {}  # 无参数
        elif isinstance(factor_request, tuple):
            factor_name = factor_request[0]
            # 将参数打包成字典，传递给计算函数
            # 例如: ('beta', '000300.SH') -> {'benchmark_index': '000300.SH'}
            # 你需要根据因子定义，约定好参数名
            if factor_name == 'beta':
                params = {'benchmark_index': factor_request[1]}
            elif factor_name in ['close_adj_filled','open_adj_filled','high_adj_filled','low_adj_filled']: #元凶 这里 if 导致 命中下一个else
                params = {'limit': factor_request[1]}
            # 未来可以扩展到其他因子，如 'momentum'
            # elif factor_name == 'momentum':
            #     params = {'window': factor_request[1]}
            else:
                params = {}
        else:
            raise TypeError("factor_request 必须是字符串或元组")

        # 3. 调度计算
        calculation_method_name = f"_calculate_{factor_name}"
        if hasattr(self.calculator, calculation_method_name):
            method_to_call = getattr(self.calculator, calculation_method_name)
            # 【关键】将解析出的参数传递给计算函数
            raw_factor_df = method_to_call(**params)
        elif factor_name in self.data_manager.raw_dfs and not params:
            log_warning("高度重视---这是宽表 index为全交易日，所以：停牌期的行全是nan，请思考这突如其来的nan对下面公式计算是否有影响，有影响是否ffill解决，参考adj_close计算")
            raw_factor_df = self.data_manager.raw_dfs[factor_name]
        else:
            raise ValueError(f"获取因子失败：{factor_request}")

        # 4. 存入缓存并返回
        self.factors_cache[factor_request] = raw_factor_df
        return raw_factor_df

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

    def get_test_result(self, factor_name: str) -> Optional[FactorResultsManager]:
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

            result = FactorResultsManager.from_dict(data)

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

    def _save_results(self, results: Dict[str, Any], file_name_prefix: str) -> None:
        """保存测试结果"""
        # 准备可序列化的结果
        serializable_results = self._make_serializable(results)

        # 保存JSON格式
        json_path = os.path.join(self.results_dir, f'all_single_factor_test_{file_name_prefix}_results.json')
        add_single_factor_test_result(json_path, serializable_results)

    # ok 保存 精简简要的测试结果
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
        keys_to_update = new_results_df[['factor_name', 'backtest_period', 'backtest_base_on_index']].drop_duplicates()

        # 5. 删除旧记录
        if not existing_leaderboard.empty:
            # 使用 merge + indicator 来找到并排除需要删除的行
            ##
            # indicator=True
            # 结果就是：新生成_merger列，值要么是both 要么是left_only#
            merged = existing_leaderboard.merge(keys_to_update,
                                                on=['factor_name', 'backtest_period', 'backtest_base_on_index'],
                                                how='left',
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

    # def get_backtest_ready_factor_entity(self):
    #
    #     technical_df_dict = {}
    #     technical_category_dict = {}
    #     technical_school_dict = {}
    #
    #     # 找出所有目标target 因子。
    #     # 通过config的标识 找出需要学术计算的因子
    #     # 自生的门派，重新align Require的因子，参与计算，返回学术_df
    #     target_factors_for_evaluation = self.data_manager.config['target_factors_for_evaluation']['fields']
    #
    #     for target_factor_name in target_factors_for_evaluation:
    #         logger.info(f"get_backtest_ready_factor_entity加载{target_factor_name}")
    #         # category
    #         category = self.get_style_category(target_factor_name)
    #         school = self.get_school_code_by_factor_name(target_factor_name)
    #         target_data_df = self.get_prepare_aligned_factor_for_analysis(target_factor_name,True)
    #         technical_df_dict.update({target_factor_name: target_data_df})
    #         technical_category_dict.update({target_factor_name: category})
    #         technical_school_dict.update({target_factor_name: school})
    #
    #     return technical_df_dict, technical_category_dict, technical_school_dict
    #跟股票池对齐，在股票池里面马上进行测试 处于快要到分析阶段，可以调用，因为理解确实需要对齐股票池。目前没发现什么场景不需要对其的，所i无脑掉 没错
    def get_prepare_aligned_factor_for_analysis(self, factor_request: Union[str, tuple],stock_pool_index_name, for_test):
        if not for_test:
            raise ValueError('必须是用于测试前做的数据提取 因为这里的填充就在专门只给测试自身因子做的填充策略')


        factor_with_direction = self.get_factor_by_rule(factor_request)#本质就是get_raw_factor 带了规则而已 目前就规则：比如*-1 or *1

        # 2. 获取股票池DataFrame
        pool = self.data_manager.stock_pools_dict[stock_pool_index_name]

        # 3. 执行最终的对齐和填充
        #    注意：factor_request[0] 可以确保我们拿到'beta'这样的基础名字用于日志或调试
        factor_name_str = factor_request[0] if isinstance(factor_request, tuple) else factor_request

        return fill_and_align_by_stock_pool(
            factor_name=factor_name_str,
            df=factor_with_direction,
            stock_pool_df=pool,
            _existence_matrix=self.data_manager._existence_matrix
        )

    def get_school_code_by_factor_name(self, factor_name):
        return self.get_school_by_style_category(self.get_style_category(factor_name))

    #
    #
    # def get_stock_pool_name_by_factor_school(self, factor_school):
    #     if factor_school in ['fundamentals', 'trend']:
    #         return 'institutional_stock_pool'#中证800股票池
    #     if factor_school in ['microstructure']:
    #         return 'microstructure_stock_pool' #全大A 股票池
    #     raise ValueError(f'{factor_school}没有定义因子属于哪一门派')

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
            'event': 'fundamentals',


            # === 趋势派 (trend) ===
            # 反映了价格在历史序列中的行为模式和风险特征。
            'momentum': 'trend',
            'risk': 'trend',  # Beta和波动率描述了股票在趋势中的行为特征和敏感度

            # === 微观派 (microstructure) ===
            # 直接来源于市场的实际交易行为（价格、成交量、换手率）。
            'liquidity': 'microstructure',
            'price': 'microstructure',
            'return': 'microstructure',
            'sentiment': 'microstructure'

        }
        school = SCHOOL_MAP.get(style_category)

        if school is None:
            raise ValueError(
                f"无法识别的因子风格 '{style_category}'，请在 get_school_by_style_category 函数中定义其门派。")
        return school

    # def get_stock_pool_index_by_factor_name(self, factor_name):
    #     # 拿到对应pool_name
    #     pool_name = self.get_stock_pool_name_by_factor_name(factor_name)
    #
    #     index_filter_config = self.data_manager.config['stock_pool_profiles'][pool_name]['index_filter']
    #     if not index_filter_config['enable']:
    #         return INDEX_CODES['ALL_A']
    #     return index_filter_config['index_code']


    def get_style_category(self, factor_name):
        return self.data_manager.get_factor_definition(factor_name)['style_category'].iloc[0]
    #
    #
    # def generate_structuer_base_on_diff_pool_name(self, factor_name_data: Union[str, list]):
    #     if isinstance(factor_name_data, str):
    #         return self.generate_structure_dict_base_on_diff_pool_name(factor_name_data)
    #     if isinstance(factor_name_data, list):
    #         dicts = []
    #         for factor_name in factor_name_data:
    #             pool_df_dict = self.generate_structure_dict_base_on_diff_pool_name(factor_name)
    #             dicts.append((factor_name, pool_df_dict))  # 保存因子名和对应的dict
    #
    #         merged = {}
    #         for factor_name, pool_df_dict in dicts:
    #             for pool, df in pool_df_dict.items():
    #                 if pool not in merged:
    #                     merged[pool] = {}
    #                 merged[pool][factor_name] = df
    #
    #         return merged
    #
    #     raise TypeError("build_df_dict_base_on_diff_pool 入参类似有误")

    #
    # def do_shift_and_align_for_dict(self, factor_name=None, data_dict=None, _existence_matrix: pd.DataFrame = None):
    #     result = {}
    #     for stock_name, stock_pool in self.data_manager.stock_pools_dict.items():
    #         ret = self.do_shift_and_align_where_stock_pool(factor_name, data_dict[stock_name], stock_pool,
    #                                                        _existence_matrix=_existence_matrix)
    #         result[stock_name] = ret
    #     return result
    #
    # def do_align_for_dict(self, factor_name, data_dict):
    #     result = {}
    #     for stock_name, stock_pool in self.data_manager.stock_pools_dict.items():
    #         ret = self.do_align(factor_name, data_dict[stock_name], stock_pool)
    #         result[stock_name] = {factor_name: ret}
    #     return result
    #
    # def do_shift_and_align_where_stock_pool(self, factor_name, data_to_deal, stock_pool,
    #                                         _existence_matrix: pd.DataFrame = None):
    #     # 率先shift
    #     data_to_deal_by_shifted = self.do_shift(data_to_deal)
    #     # 对齐
    #     result = self.do_align(factor_name, data_to_deal_by_shifted, stock_pool, _existence_matrix=_existence_matrix)
    #     return result

    # def do_shift(
    #         self,
    #         data_to_shift: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    # ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    #     """
    #     对输入的数据执行 .shift(1) 操作，智能处理单个DataFrame或DataFrame字典。
    #     Args:
    #         data_to_shift: 需要进行滞后处理的数据，
    #                        可以是一个 pandas DataFrame，
    #                        也可以是一个 key为字符串, value为pandas DataFrame的字典。
    #     Returns:
    #         一个与输入类型相同的新对象，其中所有的DataFrame都已被 .shift(1) 处理。
    #     """
    #     # --- 情况一：输入是字典 ---
    #     if isinstance(data_to_shift, dict):
    #         shifted_dict = {}
    #         for key, df in data_to_shift.items():
    #             if not isinstance(df, pd.DataFrame):
    #                 raise ValueError("do_shift失败,dict内部不是df结构")
    #             # 对字典中的每个DataFrame执行shift操作
    #             shifted_dict[key] = df.shift(1)
    #         return shifted_dict
    #
    #     # --- 情况二：输入是单个DataFrame ---
    #     elif isinstance(data_to_shift, pd.DataFrame):
    #         return data_to_shift.shift(1)
    #
    #     # --- 其他情况：输入类型错误，主动报错 ---
    #     else:
    #         raise TypeError(
    #             f"输入类型不支持，期望是DataFrame或Dict[str, DataFrame]，"
    #             f"但收到的是 {type(data_to_shift).__name__}"
    #         )
    #
    # def do_align(self, factor_name, data_to_align: Union[pd.DataFrame, Dict[str, pd.DataFrame]], stock_pool,
    #              _existence_matrix: pd.DataFrame = None
    #              ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    #     # --- 情况一：输入是字典 ---
    #     if isinstance(data_to_align, dict):
    #         shifted_dict = {}
    #         for key, df in data_to_align.items():
    #             if not isinstance(df, pd.DataFrame):
    #                 raise ValueError("do_align失败,dict内部不是df结构")
    #             # 对字典中的每个DataFrame执行shift操作
    #             shifted_dict[key] = fill_and_align_by_stock_pool(factor_name=key, df=df,
    #                                                              stock_pool_df=stock_pool,
    #                                                              _existence_matrix=_existence_matrix)
    #         return shifted_dict
    #
    #     # --- 情况二：输入是单个DataFrame ---
    #     elif isinstance(data_to_align, pd.DataFrame):
    #         return fill_and_align_by_stock_pool(factor_name=factor_name, df=data_to_align,
    #                                             stock_pool_df=stock_pool, _existence_matrix=_existence_matrix)
    #
    #     # --- 其他情况：输入类型错误，主动报错 ---
    #     else:
    #         raise TypeError(
    #             f"输入类型不支持，期望是DataFrame或Dict[str, DataFrame]，"
    #             f"但收到的是 {type(data_to_align).__name__}"
    #         )

    # # ok 因为需要滚动计算，所以不依赖股票池的index（trade） 只要对齐股票列就好
    # def get_pct_chg_beta_dict(self):
    #     dict = {}
    #     for pool_name, _ in self.data_manager.stock_pools_dict.items():
    #         beta_df = self.get_pct_chg_beta_data_for_pool(pool_name)
    #         dict[pool_name] = beta_df
    #     return dict

    # def get_pct_chg_beta_data_for_pool(self, pool_name):
    #     pool_stocks = self.data_manager.stock_pools_dict[pool_name].columns
    #
    #     # 直接从主Beta矩阵中按需选取，无需重新计算
    #     beta_for_this_pool = self.prepare_master_pct_chg_beta_dataframe()[pool_stocks]  # todo后面考虑设计一下，取自get_Factor()
    #
    #     return beta_for_this_pool
    #
    # def prepare_master_pct_chg_beta_dataframe(self):
    #     """
    #     用于生成一份统一的、覆盖所有股票的Beta矩阵。
    #     """
    #     logger.info("开始准备主Beta矩阵...")
    #
    #     # 1. 整合所有股票池的股票代码，形成一个总的股票列表
    #     all_unique_stocks = set()
    #     for stock_pool in self.data_manager.stock_pools_dict.values():
    #         all_unique_stocks.update(stock_pool.columns)
    #
    #     master_stock_list = sorted(list(all_unique_stocks))
    #
    #     # 2. 只调用一次 calculate_rolling_beta，计算所有股票的Beta
    #     logger.info(f"开始为总计 {len(master_stock_list)} 只股票计算统一的Beta...")
    #     return calculate_rolling_beta(
    #         self.data_manager.config['backtest']['start_date'],
    #         self.data_manager.config['backtest']['end_date'],
    #         master_stock_list
    #     )


if __name__ == '__main__':
    s = FactorResultsManager()
