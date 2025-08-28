"""
批量清理rolling_ic目录脚本

快速清理工具 - 删除不同因子目录下的rolling_ic文件夹
"""

import sys
from pathlib import Path
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from projects._03_factor_selection.factor_manager.ic_manager.ic_clean.rolling_ic_cleanup import RollingICCleaner
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


def quick_cleanup_all_factors():
    """快速清理所有因子的rolling_ic目录"""
    logger.info("🚀 快速清理所有因子的rolling_ic目录")
    
    # 读取因子列表
    try:
        factor_file = r'/projects/_03_factor_selection/factor_manager/selector/v3未经过残差化版本.csv'
        df = pd.read_csv(factor_file)
        all_factors = df['factor_name'].unique().tolist()
        
        logger.info(f"📊 从CSV读取到 {len(all_factors)} 个因子")
        
    except Exception as e:
        logger.warning(f"读取CSV失败，使用默认因子列表: {e}")
        all_factors = ["volatility_40d", "momentum_60d", "amihud_liquidity"]
    
    # 创建清理器
    cleaner = RollingICCleaner()
    
    # 先试运行看看会删除哪些目录
    logger.info("\n=== 试运行模式 - 预览将要删除的目录 ===")
    success, failed, errors = cleaner.cleanup_by_factor_list(
        factor_names=all_factors,
        calcu_type="c2c",
        version="20190328_20231231", 
        stock_pool="000906",
        dry_run=True  # 试运行
    )
    
    if success == 0:
        logger.info("❌ 未找到任何rolling_ic目录需要删除")
        return
    
    # 询问用户确认
    logger.info(f"\n⚠️  试运行发现 {success} 个rolling_ic目录将被删除")
    
    # 自动执行模式（生产环境）
    confirm = True  # 在脚本中直接设为True，跳过交互
    
    if confirm:
        logger.info("\n=== 开始实际删除操作 ===")
        success, failed, errors = cleaner.cleanup_by_factor_list(
            factor_names=all_factors,
            calcu_type="c2c", 
            version="20190328_20231231",
            stock_pool="000906",
            dry_run=False  # 实际删除
        )
        
        logger.info(f"✅ 清理完成: 成功删除 {success} 个目录")
        
        if failed > 0:
            logger.warning(f"⚠️  {failed} 个目录删除失败")
            
        if errors:
            logger.error("错误详情:")
            for error in errors:
                logger.error(f"  - {error}")
    else:
        logger.info("❌ 用户取消操作")


def cleanup_specific_factors():
    """清理指定因子的rolling_ic目录"""
    logger.info("🎯 清理指定因子的rolling_ic目录")
    
    # 指定要清理的因子
    target_factors = [
        "volatility_40d",
        "momentum_60d", 
        "amihud_liquidity",
        "reversal_1d",
        "momentum_120d"
    ]
    
    logger.info(f"🔍 目标因子: {target_factors}")
    
    cleaner = RollingICCleaner()
    
    # 执行清理
    success, failed, errors = cleaner.cleanup_by_factor_list(
        factor_names=target_factors,
        dry_run=False  # 直接执行删除
    )
    
    logger.info(f"✅ 指定因子清理完成: 成功 {success}, 失败 {failed}")


def cleanup_by_pattern():
    """按模式清理rolling_ic目录"""
    logger.info("🔍 按模式清理rolling_ic目录")
    
    cleaner = RollingICCleaner()
    
    # 清理特定股票池和版本下的所有因子
    success, failed, errors = cleaner.cleanup_by_criteria(
        stock_pools=["000906"],        # 只清理000906股票池
        calcu_types=["c2c"],           # 只清理c2c计算类型
        versions=["20190328_20231231"], # 只清理特定版本
        dry_run=False
    )
    
    logger.info(f"✅ 模式清理完成: 成功 {success}, 失败 {failed}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🗑️  批量rolling_ic目录清理脚本")
    logger.info("=" * 60)
    
    # 选择清理模式
    mode = 1  # 1: 全部清理, 2: 指定因子, 3: 按模式
    
    if mode == 1:
        quick_cleanup_all_factors()
    elif mode == 2:
        cleanup_specific_factors()
    elif mode == 3:
        cleanup_by_pattern()
    else:
        logger.error("无效的清理模式")
    
    logger.info("=" * 60)
    logger.info("🎉 清理脚本执行完成")
    logger.info("=" * 60)