
##
# 为什么合成时必须“先中性化，再标准化”？我们再次回顾这个核心问题，因为它至关重要。目标: 等权合并。我们希望每个细分因子（如
# bm_ratio, ep_ratio）在最终的复合价值因子中贡献相等的影响力。问题: 不同的因子在经过中性化后，其残差的波动率（标准差）是不相等的。一个与风险因子相关性高的因子，其中性化后的残差波动会很小。解决方案: 必须在合并之前，对每一个中性化后的残差进行标准化，强行将它们的波动率都统一到1。只有这样，后续的“等权相加”才是真正意义上的“等权”#
import pandas as pd
from typing import List, Dict
import pandas as pd

from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
from projects._03_factor_selection.utils.factor_processor import FactorProcessor
from quant_lib import logger


class FactorSynthesizer:
    def __init__(self, factor_manager,factor_analyzer):
        """
        初始化因子合成器。
        Args:
            factor_processor: 传入你现有的、包含了预处理方法的对象实例。
                              我们假设这个对象有 .winsorize(), ._neutralize(), ._standardize() 等方法。
        """
        self.factor_manager = factor_manager
        self.factor_analyzer = factor_analyzer
        self.processor = FactorProcessor(factor_manager.data_manager.config)
        self.raw_dfs = factor_manager.data_manager.raw_dfs
        # 因子方向配置，1为正向，-1为反向。需要根据单因子测试的结果来手动配置。
        self.FACTOR_DIRECTIONS = {
            'ep_ratio': 1,
            'bm_ratio': 1,
            'sp_ratio': 1,
            # ... 其他因子
        }

    def process_sub_factor(self, factor_name: str) -> pd.DataFrame:
        """
        【核心】对单个细分因子，
        从raw 拿到
        该计算的做计算

        执行“去极值 -> 中性化 -> 标准化”的完整流程。
        return 最终的df
        """
        print(f"\n--- 正在处理细分因子: {factor_name} ---")

        factor_df = self.factor_manager.get_backtest_ready_factor(factor_name)
        stock_pool_name = self.factor_manager.get_stock_pool_name_by_factor_name(factor_name)
        style_category = self.factor_manager.get_style_category(factor_name)

        #加载必要数据

        auxiliary_shift_dfs_base_own_stock_pools = \
        self.factor_manager.build_auxiliary_dfs_shift_diff_stock_pools_dict()[
            stock_pool_name]
        prepare_for_neutral_shift_base_own_stock_pools_dfs = \
            self.factor_analyzer.prepare_for_neutral_data_dict_shift_diff_stock_pools()[
                stock_pool_name]
         

        processed_df = self.processor.process_factor(
            target_factor_df=factor_df,
            target_factor_name=factor_name,
            auxiliary_dfs=auxiliary_shift_dfs_base_own_stock_pools,
            neutral_dfs=prepare_for_neutral_shift_base_own_stock_pools_dfs,
            style_category=style_category,neutralize_after_standardize=False)
        return processed_df

    def synthesize_composite_factor(self,
                                    composite_factor_name: str,
                                    sub_factor_names: List[str]) -> pd.DataFrame:
        """
        将一组细分因子合成为一个复合因子。

        Args:
            composite_factor_name (str): 最终合成的复合因子的名称 (e.g., 'Value_Composite').
            sub_factor_names (List[str]): 用于合成的细分因子名称列表.

        Returns:
            pd.DataFrame: 合成后的复合因子矩阵。
        """
        print(f"\n==============================================")
        print(f"开始合成复合因子: {composite_factor_name}")
        print(f"使用 {len(sub_factor_names)} 个细分因子: {sub_factor_names}")
        print(f"==============================================")

        processed_factors = []
        for factor_name in sub_factor_names:
            # 对每个子因子，都走一遍“去极值->中性化->标准化”的流程
            processed_df = self.process_sub_factor(factor_name)

            # 乘以因子方向
            direction = self.FACTOR_DIRECTIONS.get(factor_name)
            if direction == -1:
                print(f"  > 因子 '{factor_name}' 是反向因子，已乘以-1。")
                processed_df *= -1

            processed_factors.append(processed_df)

        # 最终合并：等权相加
        # 由于每个 processed_df 都已经是标准化（std=1）的，直接相加就是等波动率贡献
        if not processed_factors:
            raise ValueError("没有任何细分因子被成功处理，无法合成。")

        # 使用 reduce 和 add 来优雅地合并所有DataFrame
        from functools import reduce
        composite_factor_df = reduce(lambda left, right: left.add(right, fill_value=0), processed_factors)

        # 对最终结果再做一次标准化，使其成为一个标准的风格因子暴露
        composite_factor_df = self.processor._standardize(composite_factor_df)

        print(f"\n复合因子 '{composite_factor_name}' 合成成功！")

        return composite_factor_df
from pathlib import Path

if __name__ == '__main__':

    # --- 如何在你的主流程中使用 ---
    # 1. 实例化你的因子处理器 (假设它叫 'fp')

    # 2. 实例化因子合成器

    logger.info("1. 加载底层原始因子raw_dict数据...")
    config_path = Path(__file__).parent.parent.parent / 'factory' / 'config.yaml'

    data_manager = DataManager(config_path)
    data_manager.prepare_basic_data()

    factor_manager = FactorManager(data_manager)
    factor_analyzer = FactorAnalyzer(factor_manager= factor_manager)

    synthesizer = FactorSynthesizer(factor_manager,factor_analyzer)

    # 3. 定义你要合成的因子列表
    value_factors = list(synthesizer.FACTOR_DIRECTIONS.keys())

    config_path = "factory/config.yaml",

    # 4. 调用合成方法
    factor_name = factor_manager.data_manager.config['target_factors_for_evaluation']['fields'][0]
    value_composite_df = synthesizer.synthesize_composite_factor(factor_name, value_factors)
    # 5. 拿到合成后的复合因子，你就可以对它进行单因子测试了！
    #准备数据
    stock_pool_name = factor_analyzer.factor_manager.get_stock_pool_name_by_factor_name(factor_name)
    close_df = factor_analyzer.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='close',
                                                                                 need_shift=False)[
        stock_pool_name]  # 传入ic 、分组、回归的 close 必须是原始的  用于t日评测结果的
    prepare_for_neutral_shift_base_own_stock_pools_dfs = factor_analyzer.prepare_for_neutral_data_dict_shift_diff_stock_pools()[
            stock_pool_name]
    circ_mv_shift_df = factor_analyzer.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(
        factor_name='circ_mv',
        need_shift=True)[stock_pool_name]
    ic_series_periods_dict, ic_stats_periods_dict,quantile_daily_returns_for_plot_dict, quantile_stats_periods_dict,factor_returns_series_periods_dict, fm_stat_results_periods_dict =     factor_analyzer.core_three_test(value_composite_df, factor_name,close_df,
                                prepare_for_neutral_shift_base_own_stock_pools_dfs, circ_mv_shift_df)
    #landing 存储宝贵的测试结果
    category = data_manager.get_which_field_of_factor_definition_by_factor_name(factor_name,'style_category').iloc[0]
    overrall_summary_stats = factor_analyzer.landing_for_core_three_analyzer_result(factor_name,category, "standard",
                                                                     ic_series_periods_dict, ic_stats_periods_dict,
                                                                     quantile_daily_returns_for_plot_dict,
                                                                     quantile_stats_periods_dict,
                                                                     factor_returns_series_periods_dict,
                                                                     fm_stat_results_periods_dict)
