"""
滚动IC目录清理工具

功能：批量删除不同因子目录下的rolling_ic文件夹
路径模式：{base_path}/{stock_pool}/{factor_name}/{calcu_type}/{version}/rolling_ic
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


class RollingICCleaner:
    """滚动IC清理器"""
    
    def __init__(self, base_path: str = None):
        """
        初始化清理器
        
        Args:
            base_path: 基础路径，默认为项目工作路径
        """
        if base_path is None:
            self.base_path = Path(r"/projects/_03_factor_selection/workspace/result")
        else:
            self.base_path = Path(base_path)
        
        if not self.base_path.exists():
            raise ValueError(f"基础路径不存在: {self.base_path}")
    
    def scan_rolling_ic_directories(
        self,
        stock_pools: Optional[List[str]] = None,
        factor_names: Optional[List[str]] = None,
        calcu_types: Optional[List[str]] = None,
        versions: Optional[List[str]] = None
    ) -> List[Path]:
        """
        扫描符合条件的rolling_ic目录
        
        Args:
            stock_pools: 股票池列表，None表示所有
            factor_names: 因子名称列表，None表示所有  
            calcu_types: 计算类型列表，None表示所有
            versions: 版本列表，None表示所有
            
        Returns:
            List[Path]: 找到的rolling_ic目录路径列表
        """
        rolling_ic_dirs = []
        
        logger.info(f"🔍 开始扫描rolling_ic目录: {self.base_path}")
        
        # 遍历股票池目录
        for stock_pool_dir in self.base_path.iterdir():
            if not stock_pool_dir.is_dir():
                continue
                
            stock_pool = stock_pool_dir.name
            if stock_pools and stock_pool not in stock_pools:
                continue
                
            logger.debug(f"📂 扫描股票池: {stock_pool}")
            
            # 遍历因子目录
            for factor_dir in stock_pool_dir.iterdir():
                if not factor_dir.is_dir():
                    continue
                    
                factor_name = factor_dir.name
                if factor_names and factor_name not in factor_names:
                    continue
                    
                # 遍历计算类型目录
                for calcu_type_dir in factor_dir.iterdir():
                    if not calcu_type_dir.is_dir():
                        continue
                        
                    calcu_type = calcu_type_dir.name
                    if calcu_types and calcu_type not in calcu_types:
                        continue
                        
                    # 遍历版本目录
                    for version_dir in calcu_type_dir.iterdir():
                        if not version_dir.is_dir():
                            continue
                            
                        version = version_dir.name
                        if versions and version not in versions:
                            continue
                            
                        # 检查rolling_ic目录
                        rolling_ic_dir = version_dir / "rolling_ic"
                        if rolling_ic_dir.exists() and rolling_ic_dir.is_dir():
                            rolling_ic_dirs.append(rolling_ic_dir)
                            logger.debug(f"  ✓ 找到: {rolling_ic_dir}")
        
        logger.info(f"📊 扫描完成，共找到 {len(rolling_ic_dirs)} 个rolling_ic目录")
        return rolling_ic_dirs
    
    def delete_rolling_ic_directories(
        self,
        rolling_ic_dirs: List[Path],
        dry_run: bool = True
    ) -> Tuple[int, int, List[str]]:
        """
        删除rolling_ic目录
        
        Args:
            rolling_ic_dirs: 要删除的目录列表
            dry_run: 是否为试运行模式
            
        Returns:
            (成功数量, 失败数量, 错误信息列表)
        """
        success_count = 0
        failed_count = 0
        errors = []
        
        mode_text = "试运行" if dry_run else "实际删除"
        logger.info(f"🗑️ 开始{mode_text}，共 {len(rolling_ic_dirs)} 个目录")
        
        for i, rolling_ic_dir in enumerate(rolling_ic_dirs, 1):
            try:
                if dry_run:
                    # 试运行模式，只记录但不实际删除
                    file_count = sum(1 for _ in rolling_ic_dir.rglob('*') if _.is_file())
                    logger.info(f"  [{i:3d}] [试运行] {rolling_ic_dir} (包含 {file_count} 个文件)")
                    success_count += 1
                else:
                    # 实际删除模式
                    if rolling_ic_dir.exists():
                        file_count = sum(1 for _ in rolling_ic_dir.rglob('*') if _.is_file())
                        shutil.rmtree(rolling_ic_dir)
                        logger.info(f"  [{i:3d}] [已删除] {rolling_ic_dir} (删除了 {file_count} 个文件)")
                        success_count += 1
                    else:
                        logger.warning(f"  [{i:3d}] [跳过] {rolling_ic_dir} (目录不存在)")
                        
            except Exception as e:
                error_msg = f"删除失败 {rolling_ic_dir}: {e}"
                errors.append(error_msg)
                logger.error(f"  [{i:3d}] [失败] {error_msg}")
                failed_count += 1
        
        logger.info(f"✅ {mode_text}完成: 成功 {success_count}, 失败 {failed_count}")
        return success_count, failed_count, errors
    
    def cleanup_by_criteria(
        self,
        stock_pools: Optional[List[str]] = None,
        factor_names: Optional[List[str]] = None,
        calcu_types: Optional[List[str]] = None,
        versions: Optional[List[str]] = None,
        dry_run: bool = True
    ) -> Tuple[int, int, List[str]]:
        """
        按条件清理rolling_ic目录
        
        Args:
            stock_pools: 股票池筛选
            factor_names: 因子名称筛选
            calcu_types: 计算类型筛选
            versions: 版本筛选
            dry_run: 是否为试运行
            
        Returns:
            (成功数量, 失败数量, 错误信息列表)
        """
        # 1. 扫描目录
        rolling_ic_dirs = self.scan_rolling_ic_directories(
            stock_pools=stock_pools,
            factor_names=factor_names, 
            calcu_types=calcu_types,
            versions=versions
        )
        
        if not rolling_ic_dirs:
            logger.info("未找到符合条件的rolling_ic目录")
            return 0, 0, []
        
        # 2. 执行删除
        return self.delete_rolling_ic_directories(rolling_ic_dirs, dry_run=dry_run)
    
    def cleanup_all_rolling_ic(self, dry_run: bool = True) -> Tuple[int, int, List[str]]:
        """
        清理所有rolling_ic目录
        
        Args:
            dry_run: 是否为试运行
            
        Returns:
            (成功数量, 失败数量, 错误信息列表)
        """
        return self.cleanup_by_criteria(dry_run=dry_run)
    
    def cleanup_by_factor_list(
        self,
        factor_names: List[str],
        calcu_type: str = "o2o",
        version: str = "20190328_20231231",
        stock_pool: str = "000906",
        dry_run: bool = True
    ) -> Tuple[int, int, List[str]]:
        """
        按因子列表清理rolling_ic目录
        
        Args:
            factor_names: 因子名称列表
            calcu_type: 计算类型
            version: 版本
            stock_pool: 股票池
            dry_run: 是否为试运行
            
        Returns:
            (成功数量, 失败数量, 错误信息列表)
        """
        return self.cleanup_by_criteria(
            stock_pools=[stock_pool],
            factor_names=factor_names,
            calcu_types=[calcu_type],
            versions=[version],
            dry_run=dry_run
        )


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description="滚动IC目录清理工具")
    parser.add_argument("--base-path", type=str, help="基础路径")
    parser.add_argument("--stock-pools", nargs="+", help="股票池列表")
    parser.add_argument("--factors", nargs="+", help="因子名称列表")
    parser.add_argument("--calcu-types", nargs="+", help="计算类型列表")
    parser.add_argument("--versions", nargs="+", help="版本列表")
    parser.add_argument("--dry-run", action="store_true", default=True, help="试运行模式")
    parser.add_argument("--execute", action="store_true", help="实际执行删除")
    
    args = parser.parse_args()
    
    # 创建清理器
    cleaner = RollingICCleaner(base_path=args.base_path)
    
    # 执行清理
    dry_run = not args.execute
    success, failed, errors = cleaner.cleanup_by_criteria(
        stock_pools=args.stock_pools,
        factor_names=args.factors,
        calcu_types=args.calcu_types,
        versions=args.versions,
        dry_run=dry_run
    )
    
    if errors:
        logger.error("错误详情:")
        for error in errors:
            logger.error(f"  - {error}")


if __name__ == "__main__":
    # 示例用法
    import pandas as pd
    
    logger.info("=== 滚动IC清理工具示例 ===")
    
    # 创建清理器
    cleaner = RollingICCleaner()
    
    # 示例1: 清理指定因子
    # logger.info("\n--- 示例1: 清理指定因子 ---")
    # test_factors = ["volatility_40d", "momentum_60d", "amihud_liquidity"]
    # success, failed, errors = cleaner.cleanup_by_factor_list(
    #     factor_names=test_factors,
    #     dry_run=False  # 试运行
    # )
    #
    # 示例2: 清理所有rolling_ic (试运行)
    logger.info("\n--- 示例2: 清理所有rolling_ic (试运行) ---")
    success, failed, errors = cleaner.cleanup_all_rolling_ic(dry_run=False)
    #
    # # 示例3: 从CSV读取因子并清理
    # logger.info("\n--- 示例3: 从CSV读取因子并清理 ---")
    # try:
    #     df = pd.read_csv(r'D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\factor_manager\selector\v3未经过残差化版本.csv')
    #     all_factors = df['factor_name'].unique().tolist()
    #
    #     logger.info(f"从CSV读取到 {len(all_factors)} 个因子")
    #
    #     # 试运行模式清理所有因子
    #     success, failed, errors = cleaner.cleanup_by_factor_list(
    #         factor_names=all_factors[:5],  # 先测试前5个
    #         dry_run=True
    #     )
    #
    #     if errors:
    #         logger.warning("清理过程中发现错误:")
    #         for error in errors:
    #             logger.warning(f"  - {error}")
    #
    #     # 实际执行需要将dry_run设为False
    #     # success, failed, errors = cleaner.cleanup_by_factor_list(
    #     #     factor_names=all_factors,
    #     #     dry_run=False  # 实际删除
    #     # )
    #
    # except Exception as e:
    #     logger.error(f"读取CSV文件失败: {e}")
    #
    # logger.info("\n=== 清理工具演示完成 ===")
    # logger.info("💡 提示: 将 dry_run=False 以实际执行删除操作")