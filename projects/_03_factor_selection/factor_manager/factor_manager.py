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
                 registry_path: str = "factor_registry.json",
                 config: Dict[str, Any] = None):


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

    ##

    # 全局排序 对所有因子进行综合排序，选出一个比如Top 50的大名单。 ，保证所有因子的个体质量都是顶尖的。
    #
    # 在“Top 50的大名单”   中进行分类和相关性分析:
    #
    # --对这Top 50的因子进行分类（价值、动量等）。
    #
    # --计算这50个因子之间的相关性矩阵。
    #
    # --从这50个最优秀的因子中，挑出比如10个，要求这10个因子彼此不相关，并且尽可能覆盖不同的风格类别。
    #
    # 例如，发现在动量类里，排名前5的因子相关性都高达0.8，那么你只保留其中综合排名最高的那一个。然后你再去价值类、质量类里做同样的操作。
    #
    # 这个混合策略，保证没有错过任何一个在全市场范围内表现优异的因子（质量），又通过后续的步骤保证了最终入选因子的多样性。稳健多因子模型。#

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
    #ok
    def update_and_save_factor_leaderboard(self, all_summary_rows: list, file_name_prefix: str):
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
        keys_to_update = new_results_df[['factor_name', 'backtest_period']].drop_duplicates()

        # 5. 删除旧记录
        if not existing_leaderboard.empty:
            # 使用 merge + indicator 来找到并排除需要删除的行
            ##
            # indicator=True
            # 结果就是：新生成_merger列，值要么是both 要么是left_only#
            merged = existing_leaderboard.merge(keys_to_update, on=['factor_name', 'backtest_period'], how='left',
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



