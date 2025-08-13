##
# 为什么合成时必须“先中性化，再标准化”？我们再次回顾这个核心问题，因为它至关重要。目标: 等权合并。我们希望每个细分因子（如
# bm_ratio, ep_ratio）在最终的复合价值因子中贡献相等的影响力。问题: 不同的因子在经过中性化后，其残差的波动率（标准差）是不相等的。一个与风险因子相关性高的因子，其中性化后的残差波动会很小。解决方案: 必须在合并之前，对每一个中性化后的残差进行标准化，强行将它们的波动率都统一到1。只有这样，后续的“等权相加”才是真正意义上的“等权”#
from typing import List

import pandas as pd

from projects._03_factor_selection.utils.factor_processor import FactorProcessor


class FactorSynthesizer:
    def __init__(self, factor_manager, factor_analyzer,factor_processor):
        """
        初始化因子合成器。
        Args:
            factor_processor: 传入你现有的、包含了预处理方法的对象实例。
                              我们假设这个对象有 .winsorize(), ._neutralize(), ._standardize() 等方法。
        """
        self.factor_manager = factor_manager
        self.factor_analyzer = factor_analyzer
        self.processor = factor_processor
        if factor_processor is None:
            self.processor = FactorProcessor(factor_manager.data_manager.config)
        self.sub_factors = {
            'ep_ratio',
            'bm_ratio',
            'sp_ratio',
        }

    def process_sub_factor(self, factor_name: str,stock_pool_index_name:str) -> pd.DataFrame:
        """
        【核心】对单个细分因子，
        从raw 拿到
        该计算的做计算

        执行“去极值 -> 中性化 -> 标准化”的完整流程。
        return 最终的df
        """
        print(f"\n--- 正在处理细分因子: {factor_name} ---")

        factor_df = self.factor_manager.get_prepare_aligned_factor_for_analysis(factor_name,stock_pool_index_name, True)
        factor_df_shifted = factor_df
        (final_neutral_dfs, style_category, pit_map
         ) = self.factor_analyzer.prepare_date_for_process_factor(factor_name, factor_df,stock_pool_index_name)

        processed_df = self.processor.process_factor(
            factor_df_shifted=factor_df_shifted,
            target_factor_name=factor_name,
            neutral_dfs=final_neutral_dfs,
            style_category=style_category, need_standardize=True)
        return processed_df

    def synthesize_composite_factor(self,
                                    composite_factor_name: str,
                                    stock_pool_index_name: str,
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
            processed_df = self.process_sub_factor(factor_name,stock_pool_index_name)

            processed_factors.append(processed_df)

        # 最终合并：等权相加
        # 由于每个 processed_df 都已经是标准化（std=1）的，直接相加就是等波动率贡献
        if not processed_factors:
            raise ValueError("没有任何细分因子被成功处理，无法合成。")

        # 使用 reduce 和 add 来优雅地合并所有DataFrame
        from functools import reduce
        composite_factor_df = reduce(lambda left, right: left.add(right, fill_value=0), processed_factors)

        # 对最终结果再做一次标准化，使其成为一个标准的风格因子暴露 全市场
        ##
        # 子因子层面 (Stage 1)：分行业处理，目的是深入到每个行业内部，剔除噪音，挖掘纯粹的相对强弱。
        #
        # 复合因子层面 (Stage 2)：全市场处理，目的是将这些纯粹的信号整合后，进行全局定标，使其成为一个可以跨行业直接比较、并用于最终投资组合构建的标准化风格暴露。#
        composite_factor_df = self.processor._standardize_robust(composite_factor_df)

        print(f"\n复合因子 '{composite_factor_name}' 合成成功！")

        return composite_factor_df


    def do_composite(self,factor_name,stock_pool_index_name):
        # 3. 定义你要合成的因子列表

        sub_factor_names =  self.factor_manager.data_manager.get_cal_require_base_fields_for_composite(factor_name) # 改成 从config 里面读取
        # 4. 调用合成方法
        value_composite_df = self.synthesize_composite_factor(factor_name, stock_pool_index_name,sub_factor_names)
        return value_composite_df
        # 5. 拿到合成后的复合因子，你就可以对它进行单因子测试了！
        # # 准备数据
        #
        # stock_pool_name = factor_analyzer.factor_manager.get_stock_pool_name_by_factor_name(factor_name)
        # close_df = factor_analyzer.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='close',
        #                                                                                         need_shift=False)[
        #     stock_pool_name]  # 传入ic 、分组、回归的 close 必须是原始的  用于t日评测结果的
        # prepare_for_neutral_shift_base_own_stock_pools_dfs = \
        # factor_analyzer.prepare_for_neutral_data_dict_shift_diff_stock_pools()prepare_for_neutral_dfs_shift_diff_stock_pools_dict[
        #     stock_pool_name]
        #
        # ic_series_periods_dict, ic_stats_periods_dict, quantile_daily_returns_for_plot_dict, quantile_stats_periods_dict, factor_returns_series_periods_dict, fm_stat_results_periods_dict, \
        #     turnover_stats_periods_dict,style_correlation_dict = factor_analyzer.comprehensive_test(target_factor_name = factor_name
        #                                    , target_factor_df= value_composite_df,
        #                                    need_process_factor = False)
        # todo 读取实验
        # factor_analyzer.test_factor_entity_service(factor_name, value_composite_df, need_process_factor=False,
        #                                            is_composite_factor=True)



# if __name__ == '__main__':
#     # --- 如何在你的主流程中使用 ---
#     # 1. 实例化你的因子处理器 (假设它叫 'fp')
#
#     # 2. 实例化因子合成器
#
#     logger.info("1. 加载底层原始因子raw_dict数据...")
#     config_path = Path(__file__).parent.parent.parent / 'factory' / 'config.yaml'
#
#     data_manager = DataManager(config_path)
#     data_manager.prepare_basic_data()
#
#     factor_manager = FactorManager(data_manager)
#     factor_analyzer = FactorAnalyzer(factor_manager=factor_manager)
#
#     synthesizer = FactorSynthesizer(factor_manager, factor_analyzer)
#
#     # 3. 定义你要合成的因子列表
#     sub_factors = list(synthesizer.sub_factors)  # 改成 从config 里面读取
#
#     config_path = "factory/config.yaml",
#
#     # 4. 调用合成方法
#     factor_name = factor_manager.data_manager.config['target_factors_for_evaluation']['fields'][0]
#     value_composite_df = synthesizer.synthesize_composite_factor(factor_name, sub_factors)
#     # 5. 拿到合成后的复合因子，你就可以对它进行单因子测试了！
#     # # 准备数据
#     #
#     #
#     # stock_pool_name = factor_analyzer.factor_manager.get_stock_pool_name_by_factor_name(factor_name)
#     # close_df = factor_analyzer.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='close',
#     #                                                                                         need_shift=False)[
#     #     stock_pool_name]  # 传入ic 、分组、回归的 close 必须是原始的  用于t日评测结果的
#     # prepare_for_neutral_shift_base_own_stock_pools_dfs = \
#     # factor_analyzer.prepare_for_neutral_data_dict_shift_diff_stock_pools()prepare_for_neutral_dfs_shift_diff_stock_pools_dict[
#     #     stock_pool_name]
#     #
#     # ic_series_periods_dict, ic_stats_periods_dict, quantile_daily_returns_for_plot_dict, quantile_stats_periods_dict, factor_returns_series_periods_dict, fm_stat_results_periods_dict, \
#     #     turnover_stats_periods_dict,style_correlation_dict = factor_analyzer.comprehensive_test(target_factor_name = factor_name
#     #                                    , target_factor_df= value_composite_df,
#     #                                    need_process_factor = False)
#     factor_analyzer.test_factor_entity_service(factor_name, value_composite_df, need_process_factor=False,
#                                                is_composite_factor=True)
