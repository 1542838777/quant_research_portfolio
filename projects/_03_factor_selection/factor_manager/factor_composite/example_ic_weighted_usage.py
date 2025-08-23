"""
IC加权因子合成使用示例
演示如何使用新的IC加权合成功能
"""

from pathlib import Path
import pandas as pd
from typing import List

# 导入必要的模块
from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager  
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer
from projects._03_factor_selection.utils.factor_processor import FactorProcessor
from projects._03_factor_selection.factor_manager.factor_composite.ic_weighted_synthesizer import (
    ICWeightedSynthesizer, 
    FactorWeightingConfig
)
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


def create_advanced_weighting_config() -> FactorWeightingConfig:
    """创建适合实盘的IC权重配置"""
    return FactorWeightingConfig(
        # 更严格的筛选标准，确保因子质量
        min_ic_mean=0.025,           # 提高IC均值要求
        min_ic_ir=0.35,              # 提高信息比率要求
        min_ic_win_rate=0.52,        # 略微提高胜率要求
        max_ic_p_value=0.05,         # 更严格的显著性要求
        
        # 实盘友好的权重设置
        max_single_weight=0.40,      # 避免单因子权重过大
        min_single_weight=0.08,      # 确保入选因子有意义权重
        max_factors_count=6,         # 控制因子数量，降低复杂度
        
        # 相关性控制
        correlation_threshold=0.65,   # 降低相关性阈值
        
        # IC评估周期
        lookback_periods=['5d', '21d']  # 短期+中期表现
    )


def demonstrate_ic_weighted_synthesis():
    """演示IC加权因子合成完整流程"""
    
    print("🚀 IC加权因子合成演示开始...")
    
    # 1. 初始化基础组件
    config_path = Path(__file__).parent.parent.parent / 'factory' / 'config.yaml'
    
    data_manager = DataManager(config_path)
    data_manager.prepare_basic_data()
    
    factor_manager = FactorManager(data_manager)
    factor_analyzer = FactorAnalyzer(factor_manager=factor_manager)
    factor_processor = FactorProcessor(data_manager.config)
    
    # 2. 创建IC加权合成器
    weighting_config = create_advanced_weighting_config()
    synthesizer = ICWeightedSynthesizer(
        factor_manager=factor_manager,
        factor_analyzer=factor_analyzer, 
        factor_processor=factor_processor,
        config=weighting_config
    )
    
    # 3. 定义候选因子（建议选择不同类别的因子）
    candidate_factors = [
        # 价值因子
        'bm_ratio', 'ep_ratio', 'sp_ratio',
        # 质量因子  
        'roe_ttm', 'gross_margin_ttm',
        # 成长因子
        'net_profit_growth_ttm', 'revenue_growth_ttm',
        # 动量因子
        'momentum_120d', 'momentum_20d',
        # 流动性因子
        'amihud_liquidity', 'turnover_rate_90d_mean',
        # 风险因子
        'volatility_120d', 'volatility_90d'
    ]
    
    stock_pool_name = 'institutional_stock_pool'  # 或你配置的其他股票池
    composite_factor_name = 'IC_Weighted_Alpha_V1'
    
    try:
        # 4. 执行IC加权合成
        logger.info(f"开始合成因子: {composite_factor_name}")
        
        composite_factor_df, synthesis_report = synthesizer.synthesize_ic_weighted_factor(
            composite_factor_name=composite_factor_name,
            stock_pool_index_name=stock_pool_name,
            candidate_factor_names=candidate_factors,
            force_recalculate_ic=False  # 使用缓存的IC数据
        )
        
        # 5. 显示合成报告
        synthesizer.print_synthesis_report(synthesis_report)
        
        # 6. 保存合成因子用于后续测试
        save_composite_factor(composite_factor_df, composite_factor_name, synthesis_report)
        
        # 7. 快速质量检查
        perform_quick_quality_check(composite_factor_df, composite_factor_name)
        
        logger.info("✅ IC加权因子合成演示完成！")
        
        return composite_factor_df, synthesis_report
        
    except Exception as e:
        logger.error(f"❌ 合成过程出现错误: {e}")
        raise


def save_composite_factor(factor_df: pd.DataFrame, factor_name: str, report: dict):
    """保存合成因子和报告"""
    try:
        # 创建结果目录
        results_dir = Path(__file__).parent.parent.parent / 'results' / 'composite_factors'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存因子数据
        factor_file = results_dir / f"{factor_name}_factor_data.parquet"
        factor_df.to_parquet(factor_file)
        
        # 保存合成报告
        import json
        report_copy = report.copy()
        
        # 处理时间戳序列化
        if 'synthesis_timestamp' in report_copy:
            report_copy['synthesis_timestamp'] = report_copy['synthesis_timestamp'].isoformat()
        
        report_file = results_dir / f"{factor_name}_synthesis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_copy, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 合成结果已保存:")
        logger.info(f"  因子数据: {factor_file}")
        logger.info(f"  合成报告: {report_file}")
        
    except Exception as e:
        logger.error(f"⚠️ 保存合成结果失败: {e}")


def perform_quick_quality_check(factor_df: pd.DataFrame, factor_name: str):
    """对合成因子进行快速质量检查"""
    logger.info(f"\n📊 {factor_name} 快速质量检查:")
    
    # 基本统计
    all_values = factor_df.stack().dropna()
    logger.info(f"  📈 数据覆盖: {len(all_values):,} 个有效观测值")
    logger.info(f"  📊 统计特征: 均值={all_values.mean():.4f}, 标准差={all_values.std():.4f}")
    logger.info(f"  📏 数据范围: [{all_values.min():.4f}, {all_values.max():.4f}]")
    
    # 每日有效股票数
    daily_counts = factor_df.notna().sum(axis=1)
    logger.info(f"  📅 每日有效股票数: 均值={daily_counts.mean():.1f}, 最小={daily_counts.min()}, 最大={daily_counts.max()}")
    
    # 稳定性检查
    monthly_means = factor_df.resample('M').apply(lambda x: x.stack().mean())
    monthly_stability = monthly_means.std()
    logger.info(f"  🔄 月度稳定性: {monthly_stability:.4f} (越小越稳定)")
    
    # 分位数检查
    q1, q5, q95, q99 = all_values.quantile([0.01, 0.05, 0.95, 0.99])
    logger.info(f"  📊 分位数检查: P1={q1:.3f}, P5={q5:.3f}, P95={q95:.3f}, P99={q99:.3f}")


def load_and_test_composite_factor(factor_name: str):
    """加载已保存的合成因子并进行测试"""
    try:
        results_dir = Path(__file__).parent.parent.parent / 'results' / 'composite_factors' 
        factor_file = results_dir / f"{factor_name}_factor_data.parquet"
        
        if not factor_file.exists():
            logger.error(f"❌ 未找到因子文件: {factor_file}")
            return None
        
        # 加载因子数据
        factor_df = pd.read_parquet(factor_file)
        logger.info(f"📥 成功加载合成因子: {factor_name}")
        
        # 这里可以调用你的因子测试流程
        # factor_analyzer.test_factor_entity_service(factor_name, factor_df, need_process_factor=False)
        
        return factor_df
        
    except Exception as e:
        logger.error(f"❌ 加载因子失败: {e}")
        return None


if __name__ == "__main__":
    # 演示完整流程
    try:
        composite_df, report = demonstrate_ic_weighted_synthesis()
        
        # 可以继续进行因子测试
        print("\n💡 接下来可以:")
        print("1. 对合成因子进行完整的单因子测试")
        print("2. 与原有的等权合成因子进行对比")
        print("3. 构建基于此合成因子的交易策略")
        
    except Exception as e:
        print(f"演示过程出现错误: {e}")
        import traceback
        traceback.print_exc()