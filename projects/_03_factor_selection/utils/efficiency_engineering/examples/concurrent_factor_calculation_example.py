"""
并发因子计算示例

展示如何使用 ConcurrentExecutor 进行高效的批量因子计算
"""

import pandas as pd
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from projects._03_factor_selection.utils.efficiency_engineering.concurrent_executor import (
    run_concurrent_factors,
    FactorCalculationExecutor,
    ConcurrentConfig
)
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


def example_basic_concurrent_execution():
    """基础并发执行示例"""
    logger.info("=== 基础并发执行示例 ===")
    
    # 读取因子列表
    factor_file = r'/projects/_03_factor_selection/factor_manager/selector/v3未经过残差化版本.csv'
    df = pd.read_csv(factor_file)
    factor_names = df['factor_name'].unique().tolist()[:5]  # 测试前5个因子
    
    snapshot_config_id = '20250825_091622_98ed2d08'
    
    # 使用便捷函数进行并发执行
    successful_results, failed_factors = run_concurrent_factors(
        factor_names=factor_names,
        snapshot_config_id=snapshot_config_id,
        max_workers=3,
        execution_mode="single"
    )
    
    logger.info(f"成功计算: {len(successful_results)} 个因子")
    logger.info(f"失败因子: {len(failed_factors)} 个")


def example_custom_config_execution():
    """自定义配置并发执行示例"""
    logger.info("=== 自定义配置并发执行示例 ===")
    
    # 读取因子列表
    factor_file = r'/projects/_03_factor_selection/factor_manager/selector/v3未经过残差化版本.csv'
    df = pd.read_csv(factor_file)
    factor_names = df['factor_name'].unique().tolist()[:8]
    
    snapshot_config_id = '20250825_091622_98ed2d08'
    
    # 自定义并发配置
    config = ConcurrentConfig(
        max_workers=6,      # 更多并发线程
        timeout=7200,       # 30分钟超时
        retry_count=3,      # 更多重试次数
        log_interval=5      # 更频繁的进度日志
    )
    
    # 使用自定义配置的执行器
    executor = FactorCalculationExecutor(config)
    
    successful_results, failed_factors = executor.execute_factor_batch(
        factor_names=factor_names,
        snapshot_config_id=snapshot_config_id
    )
    
    logger.info(f"自定义配置执行完成")
    logger.info(f"成功: {len(successful_results)}, 失败: {len(failed_factors)}")


def example_chunked_execution():
    """分组并发执行示例"""
    logger.info("=== 分组并发执行示例 ===")
    
    # 读取因子列表
    factor_file = r'/projects/_03_factor_selection/factor_manager/selector/v3未经过残差化版本.csv'
    df = pd.read_csv(factor_file)
    factor_names = df['factor_name'].unique().tolist()[:12]  # 测试12个因子
    
    snapshot_config_id = '20250825_091622_98ed2d08'
    
    # 分组并发执行(适合内存受限的情况)
    successful_results, failed_chunks = run_concurrent_factors(
        factor_names=factor_names,
        snapshot_config_id=snapshot_config_id,
        max_workers=2,
        execution_mode="chunked"
    )
    
    logger.info(f"分组执行完成")
    logger.info(f"成功分组: {len(successful_results)}")
    logger.info(f"失败分组: {len(failed_chunks)}")


def example_production_batch():
    """生产环境批量计算示例"""
    logger.info("=== 生产环境批量计算示例 ===")
    
    # 读取完整因子列表
    factor_file = r'/projects/_03_factor_selection/factor_manager/selector/v3未经过残差化版本.csv'
    df = pd.read_csv(factor_file)
    factor_names = df['factor_name'].unique().tolist()
    
    snapshot_config_id = '20250825_091622_98ed2d08'
    
    logger.info(f"准备计算 {len(factor_names)} 个因子的滚动IC")
    
    # 分批处理,避免内存占用过高
    batch_size = 10
    total_successful = 0
    total_failed = 0
    
    for i in range(0, len(factor_names), batch_size):
        batch_factors = factor_names[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(factor_names) + batch_size - 1) // batch_size
        
        logger.info(f"🚀 开始第 {batch_num}/{total_batches} 批次计算 ({len(batch_factors)} 个因子)")
        
        try:
            successful_results, failed_factors = run_concurrent_factors(
                factor_names=batch_factors,
                snapshot_config_id=snapshot_config_id,
                max_workers=3,
                execution_mode="single"
            )
            
            total_successful += len(successful_results)
            total_failed += len(failed_factors)
            
            logger.info(f"✅ 第 {batch_num} 批次完成: 成功 {len(successful_results)}, 失败 {len(failed_factors)}")
            
        except Exception as e:
            logger.error(f"❌ 第 {batch_num} 批次执行异常: {e}")
            total_failed += len(batch_factors)
    
    logger.info(f"🎉 全部批次计算完成!")
    logger.info(f"📊 总计: 成功 {total_successful}, 失败 {total_failed}")
    logger.info(f"📈 成功率: {(total_successful/(total_successful+total_failed)*100):.1f}%")


if __name__ == "__main__":
    # 根据需要选择运行的示例
    
    # 1. 基础示例
    example_basic_concurrent_execution()
    
    # 2. 自定义配置示例
    # example_custom_config_execution()
    
    # 3. 分组执行示例
    # example_chunked_execution()
    
    # 4. 生产环境批量计算示例
    # example_production_batch()