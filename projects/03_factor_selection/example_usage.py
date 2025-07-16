"""
重构后的策略工厂使用示例

本脚本演示如何使用重构后的策略工厂进行完整的因子研究流程：
1. 单因子测试
2. 类别内优化
3. 类别间优化
4. 可视化和报告生成

"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
import datetime
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 添加当前项目目录到路径，以支持相对导入
current_project = Path(__file__).parent
sys.path.insert(0, str(current_project))

# 导入重构后的模块 - 使用绝对导入
try:
    from factory.strategy_factory import StrategyFactory
    from factor_manager.registry.factor_registry import FactorCategory
    from factor_manager.factor_manager import FactorManager
    from single_factor_tester.single_factor_tester import SingleFactorTester
    from multi_factor_optimizer.multi_factor_optimizer import MultiFactorOptimizer
    from visualization_manager.visualization_manager import VisualizationManager
    from data_manager.data_manager import DataManager
    from utils.factor_processor import FactorProcessor
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试使用模块方式运行此脚本")
    print("请在项目根目录下运行: python -m projects.03_factor_selection.example_usage")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_factors(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """创建示例因子用于测试"""
    factors = {}
    
    # 价值因子
    if 'pb' in data_dict and 'pe_ttm' in data_dict:
        factors['PB_factor'] = 1 / data_dict['pb']  # 市净率倒数
        factors['PE_factor'] = 1 / data_dict['pe_ttm']  # 市盈率倒数
    
    # 动量因子
    if 'close_price' in data_dict:
        price = data_dict['close_price']
        factors['momentum_20d'] = price / price.shift(20) - 1  # 20日动量
        factors['momentum_60d'] = price / price.shift(60) - 1  # 60日动量
    
    # 质量因子
    if 'roe' in data_dict and 'roa' in data_dict:
        factors['ROE_factor'] = data_dict['roe']
        factors['ROA_factor'] = data_dict['roa']
    
    # 成长因子
    if 'revenue_yoy' in data_dict and 'netprofit_yoy' in data_dict:
        factors['revenue_growth'] = data_dict['revenue_yoy']
        factors['profit_growth'] = data_dict['netprofit_yoy']
    
    # 波动率因子
    if 'close_price' in data_dict:
        returns = data_dict['close_price'].pct_change()
        factors['volatility_20d'] = returns.rolling(20).std()
        factors['volatility_60d'] = returns.rolling(60).std()
    
    return factors


def main():
    """主函数 - 演示策略工厂的完整使用流程"""
    
    print("="*80)
    print("重构后的策略工厂演示 - 完整因子研究流程")
    print("="*80)
    
    # 1. 初始化策略工厂
    print("\n1. 初始化策略工厂...")
    factory = StrategyFactory(
        config_path="factory/config.yaml",
        workspace_dir="workspace"
    )
    
    # 2. 加载数据
    print("\n2. 加载数据...")
    data_dict = factory.load_data()
    # print(f"✓ 数据加载成功，包含 {len(data_dict)} 个数据集")

    
    # 3. 创建示例因子
    print("\n3. 创建示例因子...")
    factor_data_dict = create_sample_factors(data_dict)
    print(f"✓ 创建了 {len(factor_data_dict)} 个示例因子")
    
    # 4. 定义因子类别映射
    factor_category_mapping = {
        'PB_factor': FactorCategory.VALUE,
        'PE_factor': FactorCategory.VALUE,
        'momentum_20d': FactorCategory.MOMENTUM,
        'momentum_60d': FactorCategory.MOMENTUM,
        'ROE_factor': FactorCategory.QUALITY,
        'ROA_factor': FactorCategory.QUALITY,
        'revenue_growth': FactorCategory.GROWTH,
        'profit_growth': FactorCategory.GROWTH,
        'volatility_20d': FactorCategory.VOLATILITY,
        'volatility_60d': FactorCategory.VOLATILITY
    }
    
    # 5. 测试单个因子
    print("\n4. 测试单个因子...")
    try:
        single_result = factory.test_single_factor(
            factor_data=factor_data_dict['PE_factor'],
            factor_name='PE_factor',
            category=FactorCategory.VALUE
        )
        print(f"✓ 单因子测试完成: PE_factor")
        
        # 获取测试结果
        test_result = factory.factor_manager.get_test_result('PE_factor')
        if test_result:
            print(f"  - IC均值: {test_result.ic_mean:.4f}")
            print(f"  - IC IR: {test_result.ic_ir:.4f}")
            print(f"  - 评分: {test_result.overall_score:.2f} ({test_result.grade})")
    except Exception as e:
        print(f"✗ 单因子测试失败: {e}")
    
    # 6. 批量测试因子
    print("\n5. 批量测试因子...")
    try:
        batch_results = factory.batch_test_factors(
            factor_data_dict=factor_data_dict,
            auto_register=True,
            category_mapping=factor_category_mapping
        )
        print(f"✓ 批量测试完成，成功测试 {len(batch_results)} 个因子")
    except Exception as e:
        print(f"✗ 批量测试失败: {e}")
    
    # 7. 查看因子性能汇总
    print("\n6. 因子性能汇总...")
    performance_summary = factory.get_factor_performance_summary()
    print("Top 5 因子:")
    print(performance_summary.head())
    
    # 8. 按类别查看表现
    print("\n7. 各类别因子表现...")
    for category in FactorCategory:
        try:
            top_factors = factory.get_top_factors(category=category, top_n=2)
            if top_factors:
                print(f"{category.value}类: {', '.join(top_factors)}")
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
    
    print("\n" + "="*80)
    print("演示完成!")
    print("="*80)


if __name__ == "__main__":
    main()
