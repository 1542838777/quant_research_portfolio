"""
重构后的策略工厂使用示例

本脚本演示如何使用重构后的策略工厂进行完整的因子研究流程：
1. 单因子测试
2. 类别内优化
3. 类别间优化
4. 可视化和报告生成

"""
import traceback

import pandas as pd
from typing import Dict

from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
from projects._03_factor_selection.factor_manager.registry.factor_registry import FactorCategory

# 添加项目根目录到路径
# project_root = Path(__file__).parent.parent.parent.parent
# sys.path.append(str(project_root))
#
# # 添加当前项目目录到路径，以支持相对导入
# current_project = Path(__file__).parent
# sys.path.insert(0, str(current_project))

# 导入重构后的模块 - 使用绝对导入
from projects._03_factor_selection.factory.strategy_factory import StrategyFactory
from quant_lib.config.logger_config import setup_logger, log_success

# 配置日志
logger = setup_logger(__name__)

def main():


    # 2. 初始化数据仓库
    logger.info("1. 加载底层原始因子raw_dict数据...")
    data_manager  = DataManager(config_path='factory/config.yaml')
    data_manager.prepare_basic_data()

    factor_manager  = FactorManager(data_manager)
    # 3. 创建示例因子
    logger.info("3. 创建目标学术因子...")

    target_factors_dict,target_factors_category_dict,target_factors_school_dict  = factor_manager.get_backtest_ready_factor_entity()
    factor_analyzer = FactorAnalyzer(factor_manager=factor_manager,

                                     target_factors_dict = target_factors_dict,
                                     target_factors_category_dict = target_factors_category_dict,
                                     target_factor_school_type_dict= target_factors_school_dict)

    # 6. 批量测试因子
    logger.info("5. 批量测试因子...")
    batch_results = factor_analyzer.batch_test_factors(
        target_factors_dict=target_factors_dict
    )
    log_success(f"✓ 批量测试完成，成功测试 {len(batch_results)} 个因子")

    #
    # # 11. 多因子优化
    # print("\n10. 多因子优化...")
    # try:
    #     optimized_factor = factory.optimize_factors(
    #         factor_data_dict=target_factors_dict,
    #         intra_method='ic_weighted',
    #         cross_method='max_diversification'
    #     )
    #     print(f"✓ 多因子优化完成，生成最终因子: {optimized_factor.shape}")
    # except Exception as e:
    #     print(f"✗ 多因子优化失败: {e}")
    #
    # # 12. 导出结果
    # print("\n11. 导出结果...")
    # try:
    #     exported_files = factory.export_results()
    #     print("✓ 结果导出完成:")
    #     for file_type, file_path in exported_files.items():
    #         print(f"  {file_type}: {file_path}")
    # except Exception as e:
    #     print(f"✗ 结果导出失败: {e}")

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
