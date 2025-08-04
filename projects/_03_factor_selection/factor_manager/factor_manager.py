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
from quant_lib.config.logger_config import log_success, log_warning
from .factor_technical_cal.factor_technical_cal import calculate_rolling_beta
# 导入子模块
from .registry.factor_registry import FactorRegistry, FactorCategory, FactorMetadata
from .classifier.factor_classifier import FactorClassifier
from .storage.single_storage import add_single_factor_test_result
from ..config.base_config import INDEX_CODES
from ..data_manager.data_manager import DataManager, align_one_df_by_stock_pool_and_fill

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
        self.data_manager = data_manager
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

    def get_target_factors_entity(self):

        technical_df_dict = {}
        technical_category_dict = {}
        technical_school_dict = {}

        # 找出所有目标target 因子。
        # 通过config的标识 找出需要学术计算的因子
        # 自生的门派，重新align Require的因子，参与计算，返回学术_df
        target_factors_for_evaluation = self.data_manager.config['target_factors_for_evaluation']['fields']

        for target_factor_name in target_factors_for_evaluation:
            target_data_df  ,category_type ,school = self.get_factor_df_by_action(target_factor_name)
            technical_df_dict.update({target_factor_name: target_data_df})
            technical_category_dict.update({target_factor_name: category_type})
            technical_school_dict.update({target_factor_name: school})

        return technical_df_dict, technical_category_dict, technical_school_dict
  # 注意 后续不要重复shift1了
    def build_technical_factor_entity_base_on_shift_and_align_stock_pools(self, target_factor_name):
        cal_require_base_fields = self.get_one_factor_denifition(target_factor_name)['cal_require_base_fields']
        stock_pool = self.get_stock_pool_by_factor_name(target_factor_name)

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

    def get_one_factor_denifition(self, target_factor_name):
        factor_definition = self.data_manager.config['factor_definition']
        factor_definition_dict = {item['name']: item for item in factor_definition}
        return factor_definition_dict.get(target_factor_name)

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
        factor_definitions = pd.DataFrame(self.data_manager.config['factor_definition'])
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
               self.data_manager.config['backtest']['start_date'],
                self.data_manager.config['backtest']['end_date'],
                self.get_pool_of_factor_name_of_stock_codes(target_factor_name)
            )
            beta_df = beta_df * -1
            return beta_df.shift(1), category_type, school
        #todo remind 注意 ，自己找的数据（不在raw——df之内的，都要自行shift1）

        # --- 如果有更多因子，在这里继续添加 elif 分支 ---

        raise ValueError(f"因子 '{target_factor_name}' 的计算逻辑尚未在本工厂中定义！")

    def get_factor_df_by_action(self, target_factor_name):
        technical_calcu = self.get_one_factor_denifition(target_factor_name)['action']
        if technical_calcu is   None :
            # 不需要任何操作，是最基础的数据
            return self.build_base_factor_entity_base_on_shift_and_align_stock_pools(
                target_factor_name)
        elif   technical_calcu == 'technical_calcu':
            # 根据门派，找出所需股票池
            # 自行计算！
            return  self.build_technical_factor_entity_base_on_shift_and_align_stock_pools(
                target_factor_name)

        else:#属于待合成因子 上游分为合成因子测试 、单因子测试 单因子测试的target_factor不会配置合成因子，所以不会命中这行，无需担心返回none。
             return None,None,None


    def get_school_code_by_factor_name(self, factor_name):
        factor_dict = {item['name']: item for item in self.data_manager.config['factor_definition']}
        return factor_dict[factor_name]['school']

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



    def get_category_type(self, factor_name):
        factor_definition = self.data_manager.config['factor_definition']
        return factor_definition[factor_definition['name'] == factor_name]['category_type']

    def get_school(self, factor_name):
        factor_definition = self.data_manager.config['factor_definition']
        return factor_definition[factor_definition['name'] == factor_name]['school']

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
            ret = self.do_shift_and_align_for_dict(factor_name, base_dict)
            return ret
        else:
            ret = self.do_align_for_dict(factor_name, base_dict)
            return ret

    def do_shift_and_align_for_dict(self, factor_name, data_dict):
        result = {}
        for stock_name, stock_pool in self.data_manager.stock_pools_dict.items():
            ret = self.do_shift_and_align_where_stock_pool(factor_name, data_dict[stock_name], stock_pool)
            result[stock_name] = ret
        return result

    def do_align_for_dict(self, factor_name, data_dict):
        result = {}
        for stock_name, stock_pool in self.data_manager.stock_pools_dict.items():
            ret = self.do_align(factor_name, data_dict[stock_name], stock_pool)
            result[stock_name] = ret
        return result

    def do_shift_and_align_where_stock_pool(self, factor_name, data_to_deal, stock_pool):
        # 率先shift
        data_to_deal_by_shifted = self.do_shift(data_to_deal)
        # 对齐
        result = self.do_align(factor_name, data_to_deal_by_shifted, stock_pool)
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

    def do_align(self, factor_name, data_to_align: Union[pd.DataFrame, Dict[str, pd.DataFrame]], stock_pool
                 ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        # --- 情况一：输入是字典 ---
        if isinstance(data_to_align, dict):
            shifted_dict = {}
            for key, df in data_to_align.items():
                if not isinstance(df, pd.DataFrame):
                    raise ValueError("do_shift失败,dict内部不是df结构")
                # 对字典中的每个DataFrame执行shift操作
                shifted_dict[key] = align_one_df_by_stock_pool_and_fill(factor_name=key, raw_df_param=df,
                                                                        stock_pool_df=stock_pool)
            return shifted_dict

        # --- 情况二：输入是单个DataFrame ---
        elif isinstance(data_to_align, pd.DataFrame):
            return align_one_df_by_stock_pool_and_fill(factor_name=factor_name, raw_df_param=data_to_align,
                                                       stock_pool_df=stock_pool)

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
