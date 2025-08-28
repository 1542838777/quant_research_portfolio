"""
正交化功能使用示例

展示如何使用新实现的因子正交化功能：
1. 带正交化的专业因子合成
2. 正交化计划的执行
3. 完整的工作流程演示
"""

from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from projects._03_factor_selection.factor_manager.factor_composite.ic_weighted_synthesize_with_orthogonalization import (
    ICWeightedSynthesizer, FactorWeightingConfig
)
from projects._03_factor_selection.factor_manager.selector.rolling_ic_factor_selector import (
    RollingICSelectionConfig
)
from projects._03_factor_selection.config_manager.config_snapshot.config_snapshot_manager import (
    ConfigSnapshotManager
)
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)

def demo_orthogonalization_workflow():
    """演示完整的正交化工作流程"""
    
    logger.info("🚀 正交化功能演示开始")
    logger.info("=" * 80)
    
    # 1. 配置设置
    logger.info("📋 Step 1: 配置设置")
    
    # 因子权重配置 - 适合正交化的设置
    weighting_config = FactorWeightingConfig(
        min_ic_mean=0.012,           # 稍微降低IC要求，因为正交化后的因子IC可能会有所变化
        min_ic_ir=0.15,              # 调整IR阈值
        min_ic_win_rate=0.50,        # 保持胜率要求
        max_single_weight=0.4,       # 降低单个因子最大权重，促进多样化
        correlation_threshold=0.60   # 降低相关性阈值，允许更多因子参与
    )
    
    # 滚动IC选择配置 - 启用正交化
    selection_config = RollingICSelectionConfig(
        enable_orthogonalization=True,      # 关键：启用正交化功能
        high_corr_threshold=0.70,          # 高相关性阈值（红色区域：二选一）
        medium_corr_threshold=0.30,        # 中等相关性阈值（黄色区域：正交化）
        max_final_factors=8                # 最多选择8个因子
    )
    
    logger.info(f"  ✅ 权重配置: IC阈值={weighting_config.min_ic_mean:.3f}, 相关性阈值={weighting_config.correlation_threshold}")
    logger.info(f"  ✅ 选择配置: 正交化={'启用' if selection_config.enable_orthogonalization else '禁用'}")
    
    # 2. 候选因子列表 - 选择一些可能相关的因子
    logger.info("\n📊 Step 2: 定义候选因子列表")
    
    candidate_factors = [
        # 动量类因子（可能高度相关）
        'momentum_20d', 'momentum_60d', 'momentum_120d',
        
        # 波动率类因子（可能中度相关）
        'volatility_20d', 'volatility_60d', 'volatility_120d',
        
        # 基本面因子（相对独立）
        'ep_ratio', 'bm_ratio', 'roe_ttm',
        
        # 技术指标（可能有相关性）
        'rsi', 'macd', 'bollinger_position',
        
        # 流动性因子
        'amihud_liquidity', 'turnover_rate_20d_mean'
    ]
    
    logger.info(f"  📈 候选因子数量: {len(candidate_factors)}")
    for i, factor in enumerate(candidate_factors, 1):
        logger.info(f"    {i:2d}. {factor}")
    
    # 3. 创建合成器实例
    logger.info("\n🔧 Step 3: 初始化IC加权合成器")
    
    try:
        synthesizer = ICWeightedSynthesizer(
            factor_manager=None,  # 在实际使用中需要提供真实的manager
            factor_analyzer=None,
            factor_processor=None,
            config=weighting_config,
            selector_config=selection_config
        )
        logger.info("  ✅ 合成器初始化成功")
    except Exception as e:
        logger.error(f"  ❌ 合成器初始化失败: {e}")
        return
    
    # 4. 演示正交化计划结构
    logger.info("\n📋 Step 4: 正交化计划结构示例")
    
    # 模拟一个正交化计划（实际会由筛选器生成）
    mock_orthogonalization_plan = [
        {
            'original_factor': 'momentum_60d',
            'base_factor': 'momentum_120d',
            'orthogonal_name': 'momentum_60d_orth_vs_momentum_120d',
            'correlation': 0.72,
            'base_score': 87.5,
            'target_score': 73.2
        },
        {
            'original_factor': 'volatility_20d',
            'base_factor': 'volatility_60d',
            'orthogonal_name': 'volatility_20d_orth_vs_volatility_60d',
            'correlation': 0.58,
            'base_score': 65.8,
            'target_score': 69.1
        }
    ]
    
    logger.info(f"  📊 示例正交化计划: {len(mock_orthogonalization_plan)} 项")
    for i, plan in enumerate(mock_orthogonalization_plan, 1):
        logger.info(f"    {i}. {plan['original_factor']} (评分:{plan['target_score']:.1f})")
        logger.info(f"       ↓ 对 {plan['base_factor']} (评分:{plan['base_score']:.1f}) 正交化")
        logger.info(f"       → {plan['orthogonal_name']} (相关性:{plan['correlation']:.2f})")
    
    # 5. 正交化的核心价值
    logger.info("\n🎯 Step 5: 正交化的核心价值")
    
    advantages = [
        "🔹 降低因子间相关性，提高组合多样性",
        "🔹 保留因子独特的Alpha信号，消除重复信息",
        "🔹 通过残差提取获得'纯净'的因子暴露",
        "🔹 动态处理相关性，而不是简单地删除因子",
        "🔹 为多因子模型提供更稳健的输入"
    ]
    
    logger.info("  正交化技术优势:")
    for advantage in advantages:
        logger.info(f"    {advantage}")
    
    # 6. 实际使用建议
    logger.info("\n💡 Step 6: 实际使用建议")
    
    best_practices = [
        "🔸 在执行多因子合成前先进行因子筛选",
        "🔸 合理设置相关性阈值：高(>0.7)、中(0.3-0.7)、低(<0.3)",
        "🔸 保持基准因子的选择基于评分，而不是主观偏好",
        "🔸 定期检查正交化效果，确保残差确实降低了相关性",
        "🔸 考虑正交化对因子经济意义的影响"
    ]
    
    logger.info("  最佳实践:")
    for practice in best_practices:
        logger.info(f"    {practice}")
    
    # 7. 完整工作流程示例代码
    logger.info("\n💻 Step 7: 完整使用流程代码示例")
    
    example_code = '''
    # 1. 初始化合成器（带正交化配置）
    synthesizer = ICWeightedSynthesizer(
        factor_manager=your_factor_manager,
        factor_analyzer=your_analyzer,  
        factor_processor=your_processor,
        config=weighting_config,
        selector_config=selection_config  # enable_orthogonalization=True
    )
    
    # 2. 执行带正交化的专业合成
    composite_factor_df, synthesis_report = synthesizer.synthesize_with_orthogonalization(
        composite_factor_name="alpha_composite_v2_orthogonal",
        candidate_factor_names=candidate_factors,
        snap_config_id="your_config_snapshot_id",
        force_generate_ic=False
    )
    
    # 3. 检查正交化报告
    ortho_info = synthesis_report.get('orthogonalization', {})
    if ortho_info.get('orthogonalization_enabled'):
        print(f"正交化计划执行了 {ortho_info['orthogonalization_plan_count']} 项")
        for detail in ortho_info['orthogonalization_details']:
            print(f"  {detail['original_factor']} → {detail['orthogonal_name']}")
    
    # 4. 使用合成后的因子进行后续分析
    # composite_factor_df 现在包含了经过正交化处理的复合因子
    '''
    
    logger.info("  代码示例:")
    for line in example_code.strip().split('\n'):
        logger.info(f"    {line}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 正交化功能演示完成！")
    logger.info("🎯 您的因子工厂现在具备了智能的相关性处理能力")

if __name__ == "__main__":
    demo_orthogonalization_workflow()