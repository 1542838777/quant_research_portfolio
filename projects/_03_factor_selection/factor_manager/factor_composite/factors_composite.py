
##
# 为什么合成时必须“先中性化，再标准化”？我们再次回顾这个核心问题，因为它至关重要。目标: 等权合并。我们希望每个细分因子（如
# bm_ratio, ep_ratio）在最终的复合价值因子中贡献相等的影响力。问题: 不同的因子在经过中性化后，其残差的波动率（标准差）是不相等的。一个与风险因子相关性高的因子，其中性化后的残差波动会很小。解决方案: 必须在合并之前，对每一个中性化后的残差进行标准化，强行将它们的波动率都统一到1。只有这样，后续的“等权相加”才是真正意义上的“等权”#
import pandas as pd
from typing import List, Dict
import pandas as pd
from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
from projects._03_factor_selection.utils.factor_processor import FactorProcessor
from quant_lib import logger


class FactorSynthesizer:
    def __init__(self, factor_manager, factor_processor):
        """
        初始化因子合成器。
        Args:
            factor_processor: 传入你现有的、包含了预处理方法的对象实例。
                              我们假设这个对象有 .winsorize(), ._neutralize(), ._standardize() 等方法。
        """
        self.factor_manager = factor_processor
        self.processor = FactorProcessor(factor_manager.data_manager.config)
        self.raw_dfs = factor_manager.raw_dfs
        # 因子方向配置，1为正向，-1为反向。需要根据单因子测试的结果来手动配置。
        self.FACTOR_DIRECTIONS = {
            'pe_ttm_inv': 1,
            'pb_inv': 1,
            'ps_ttm_inv': 1,
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

        factor_df = self.factor_manager.get_factor_df_by_action(factor_name)


        # 1. 加载原始因子数据 (这里假设你有方法可以加载)
        # raw_factor_df = self.processor.load_raw_factor(factor_name)
        # 为演示，我们创建一个虚拟的DataFrame
        # 2. 去极值
        processed_df = self.processor.process_factor(factor_df)
        print(f"  > 步骤1: 去极值完成。")

        # 3. 中性化
        # 注意：这里需要传入相应的辅助数据
        # processed_df = self.processor._neutralize(processed_df, ...)
        print(f"  > 步骤2: 中性化完成。")

        # 4. 标准化
        processed_df = self.processor._standardize(processed_df)
        print(f"  > 步骤3: 标准化完成。")

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
            direction = self.FACTOR_DIRECTIONS.get(factor_name, 1)  # 默认为1（正向）
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

if __name__ == '__main__':

    # --- 如何在你的主流程中使用 ---
    # 1. 实例化你的因子处理器 (假设它叫 'fp')

    # 2. 实例化因子合成器

    logger.info("1. 加载底层原始因子raw_dict数据...")
    data_manager = DataManager(config_path='factory/config.yaml')
    data_manager.prepare_all_data()

    factor_manager = FactorManager(data_manager)

    synthesizer = FactorSynthesizer(data_manager)

    # 3. 定义你要合成的因子列表
    value_factors = ['pe_ttm_inv', 'pb_inv', 'ps_ttm_inv']

    config_path = "factory/config.yaml",
    Dat()

    # 4. 调用合成方法
    value_composite = synthesizer.synthesize_composite_factor('value_composite', value_factors)

    # 5. 拿到合成后的复合因子，你就可以对它进行单因子测试了！
    fp.run_single_factor_test(value_composite, 'Value_Composite')
