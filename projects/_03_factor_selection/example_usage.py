"""
重构后的策略工厂使用示例

本脚本演示如何使用重构后的策略工厂进行完整的因子研究流程：
1. 单因子测试
2. 类别内优化
3. 类别间优化
4. 可视化和报告生成

"""

import pandas as pd
from typing import Dict

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
from quant_lib.config.logger_config import setup_logger

# 配置日志
logger = setup_logger(__name__)

def main():
    """主函数 - 演示策略工厂的完整使用流程"""

    logger.info("重构后的策略工厂演示 - 完整因子研究流程")

    # 1. 初始化策略工厂
    factory = StrategyFactory(
        config_path="factory/config.yaml",
        workspace_dir="workspace"
    )
    # 2. 加载数据
    logger.info("2. 加载底层原始因子raw_dict数据...")
    data_dict = factory.load_all_data_be_universe()
    logger.info(f"✓ 数据加载成功，包含 {len(data_dict)} 个数据集")

    # 3. 创建示例因子
    logger.info("3. 创建目标学术因子...")
    factor_data_dict,factor_category_dict =  factory.get_config_target_factor_dict_by_cal_base_factor_batch(data_dict)

    # 4. 定义因子类别映射
    factor_category_mapping = {
        'pe_ttm_inv': FactorCategory.VALUE,
        'momentum_20d': FactorCategory.MOMENTUM,
        'momentum_60d': FactorCategory.MOMENTUM,
        'ROE_factor': FactorCategory.QUALITY,
        'ROA_factor': FactorCategory.QUALITY,
        'revenue_growth': FactorCategory.GROWTH,
        'profit_growth': FactorCategory.GROWTH,
        'volatility_20d': FactorCategory.VOLATILITY,
        'volatility_60d': FactorCategory.VOLATILITY
    }

    # # 5. 测试单个因子
    # # print("\n4. 测试单个因子...")
    # try:
    #     single_result = factory.test_single_factor(
    #         factor_data=factor_data_dict['PE_factor'],
    #         factor_name='PE_factor',
    #         category=FactorCategory.VALUE
    #     )
    #     logger.info(f"✓ 单因子测试完成: PE_factor")
    # except Exception as e:
    #     logger.error(f"✗ 单因子测试失败: {e}")
    #     raise RuntimeError(f"处理失败: {e}")  # 抛出新异常

    # 6. 批量测试因子
    logger.info("5. 批量测试因子...")
    try:
        batch_results = factory.batch_test_factors(
            factor_data_dict=factor_data_dict,
            factor_category_type_dict=factor_category_dict
        )
        logger.info(f"✓ 批量测试完成，成功测试 {len(batch_results)} 个因子")
    except Exception as e:
        logger.error(f"✗ 批量测试失败: {e}")

    # 7. 查看因子性能汇总
    logger.info("6. 因子性能汇总...")
    performance_summary = factory.get_factor_performance_summary()
    logger.info("Top 5 因子:")
    logger.info(f"\n{performance_summary.head()}")

    # 8. 按类别查看表现
    logger.info("7. 各类别因子表现...")
    for category in FactorCategory:
        try:
            top_factors = factory.get_top_factors(category=category, top_n=2)
            if top_factors:
                logger.info(f"{category.value}类: {', '.join(top_factors)}")
        except:
            continue

    # 9. 分析因子相关性
    print("\n8. 分析因子相关性...")
    try:
        corr_matrix, corr_fig = factory.analyze_factor_correlation(factor_data_dict)
        print(f"✓ 因子相关性分析完成")

        # 保存相关性热力图
        corr_fig.savefig("workspace/factor_correlation.png")
        print(f"  相关性热力图已保存到: workspace/factor_correlation.png")
    except Exception as e:
        print(f"✗ 因子相关性分析失败: {e}")

    # 10. 因子聚类可视化
    print("\n9. 因子聚类可视化...")
    try:
        cluster_fig = factory.visualize_factor_clusters(factor_data_dict, n_clusters=3)
        if cluster_fig:
            cluster_fig.savefig("workspace/factor_clusters.png")
            print(f"✓ 因子聚类可视化已保存到: workspace/factor_clusters.png")
    except Exception as e:
        print(f"✗ 因子聚类可视化失败: {e}")

    # 11. 多因子优化
    print("\n10. 多因子优化...")
    try:
        optimized_factor = factory.optimize_factors(
            factor_data_dict=factor_data_dict,
            intra_method='ic_weighted',
            cross_method='max_diversification'
        )
        print(f"✓ 多因子优化完成，生成最终因子: {optimized_factor.shape}")
    except Exception as e:
        print(f"✗ 多因子优化失败: {e}")

    # 12. 导出结果
    print("\n11. 导出结果...")
    try:
        exported_files = factory.export_results()
        print("✓ 结果导出完成:")
        for file_type, file_path in exported_files.items():
            print(f"  {file_type}: {file_path}")
    except Exception as e:
        print(f"✗ 结果导出失败: {e}")

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
