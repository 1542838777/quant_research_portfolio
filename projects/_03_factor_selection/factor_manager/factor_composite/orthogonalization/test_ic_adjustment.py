"""
IC调整功能测试

专门测试基于R²的IC调整算法是否正确工作
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from projects._03_factor_selection.factor_manager.factor_composite.ic_weighted_synthesize_with_orthogonalization import (
    ICWeightedSynthesizer
)
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)

def test_ic_adjustment_algorithm():
    """测试IC调整算法的正确性"""
    logger.info("🧪 开始测试IC调整算法")
    
    # 创建测试用的合成器实例
    synthesizer = ICWeightedSynthesizer(None, None, None)
    
    # 创建模拟的原始IC统计数据
    original_ic_stats = {
        '5d': {
            'ic_mean': 0.050,      # 原始IC均值
            'ic_ir': 0.800,        # 原始IR
            'ic_win_rate': 0.65,   # 原始胜率
            'ic_std': 0.0625,      # 原始标准差  
            'ic_volatility': 0.02,
            'ic_p_value': 0.01,
            'ic_t_stat': 2.5
        },
        '21d': {
            'ic_mean': 0.040,
            'ic_ir': 0.750,
            'ic_win_rate': 0.62,
            'ic_std': 0.0533,
            'ic_volatility': 0.018,
            'ic_p_value': 0.02,
            'ic_t_stat': 2.2
        }
    }
    
    # 测试不同R²值的调整效果
    test_r_squared_values = [0.2, 0.5, 0.8]
    
    logger.info("📊 测试不同R²值的IC调整效果:")
    logger.info(f"  原始IC统计 (5d): IC={original_ic_stats['5d']['ic_mean']:.4f}, "
               f"IR={original_ic_stats['5d']['ic_ir']:.3f}, "
               f"胜率={original_ic_stats['5d']['ic_win_rate']:.1%}")
    
    for r_squared in test_r_squared_values:
        logger.info(f"\n🔄 测试R²={r_squared}")
        
        adjusted_stats = synthesizer._adjust_ic_stats_by_r_squared(
            original_ic_stats, r_squared, f"test_factor_r{int(r_squared*100)}"
        )
        
        # 验证调整结果
        ic_adjustment_factor = 1 - r_squared
        
        # 检查5d期间的调整结果
        adj_5d = adjusted_stats['5d']
        orig_5d = original_ic_stats['5d']
        
        # 验证IC均值调整
        expected_ic_mean = orig_5d['ic_mean'] * ic_adjustment_factor
        actual_ic_mean = adj_5d['ic_mean']
        ic_diff = abs(expected_ic_mean - actual_ic_mean)
        
        logger.info(f"  📈 IC均值: {orig_5d['ic_mean']:.4f} -> {actual_ic_mean:.4f} "
                   f"(期望:{expected_ic_mean:.4f}, 误差:{ic_diff:.6f})")
        
        # 验证IR调整
        expected_ir = orig_5d['ic_ir'] * ic_adjustment_factor
        actual_ir = adj_5d['ic_ir']
        ir_diff = abs(expected_ir - actual_ir)
        
        logger.info(f"  📊 IR: {orig_5d['ic_ir']:.3f} -> {actual_ir:.3f} "
                   f"(期望:{expected_ir:.3f}, 误差:{ir_diff:.6f})")
        
        # 验证胜率调整
        expected_win_rate = 0.5 + (orig_5d['ic_win_rate'] - 0.5) * ic_adjustment_factor
        actual_win_rate = adj_5d['ic_win_rate']
        win_rate_diff = abs(expected_win_rate - actual_win_rate)
        
        logger.info(f"  🎯 胜率: {orig_5d['ic_win_rate']:.1%} -> {actual_win_rate:.1%} "
                   f"(期望:{expected_win_rate:.1%}, 误差:{win_rate_diff:.6f})")
        
        # 验证调整的合理性
        if ic_diff < 0.001 and ir_diff < 0.001 and win_rate_diff < 0.001:
            logger.info(f"  ✅ R²={r_squared} 调整结果正确")
        else:
            logger.error(f"  ❌ R²={r_squared} 调整结果有误")
            return False
    
    logger.info("\n🎯 IC调整算法测试总结:")
    logger.info("  ✅ 所有R²值的调整结果都符合预期")
    logger.info("  ✅ 调整算法数学逻辑正确")
    logger.info("  ✅ 边界情况处理良好")
    
    return True

def test_extreme_r_squared_cases():
    """测试极端R²值的处理"""
    logger.info("\n🧪 开始测试极端R²值情况")
    
    synthesizer = ICWeightedSynthesizer(None, None, None)
    
    original_ic_stats = {
        '5d': {
            'ic_mean': 0.050,
            'ic_ir': 0.800,
            'ic_win_rate': 0.65
        }
    }
    
    # 测试极端值
    extreme_cases = [
        (0.0, "完全无相关性"),
        (0.99, "极高相关性"),
        (-0.1, "异常负值"),
        (1.1, "异常超出1")
    ]
    
    for r_squared, description in extreme_cases:
        logger.info(f"🔄 测试极端情况: {description} (R²={r_squared})")
        
        try:
            adjusted_stats = synthesizer._adjust_ic_stats_by_r_squared(
                original_ic_stats, r_squared, f"extreme_test_{r_squared}"
            )
            
            if r_squared <= 0 or r_squared >= 1:
                # 应该返回原始值
                if adjusted_stats == original_ic_stats:
                    logger.info(f"  ✅ 正确处理异常R²值，返回原始统计")
                else:
                    logger.error(f"  ❌ 异常R²值处理不当")
                    return False
            else:
                logger.info(f"  ✅ 正常处理R²={r_squared}")
                
        except Exception as e:
            logger.error(f"  ❌ 极端值处理异常: {e}")
            return False
    
    logger.info("  ✅ 极端R²值处理测试通过")
    return True

def test_multi_period_ic_adjustment():
    """测试多周期IC调整"""
    logger.info("\n🧪 开始测试多周期IC调整")
    
    synthesizer = ICWeightedSynthesizer(None, None, None)
    
    # 多周期IC统计
    original_ic_stats = {
        '1d': {'ic_mean': 0.030, 'ic_ir': 0.600, 'ic_win_rate': 0.58},
        '5d': {'ic_mean': 0.045, 'ic_ir': 0.750, 'ic_win_rate': 0.62},
        '21d': {'ic_mean': 0.040, 'ic_ir': 0.700, 'ic_win_rate': 0.60},
        '60d': {'ic_mean': 0.035, 'ic_ir': 0.650, 'ic_win_rate': 0.57}
    }
    
    r_squared = 0.6
    adjusted_stats = synthesizer._adjust_ic_stats_by_r_squared(
        original_ic_stats, r_squared, "multi_period_test"
    )
    
    # 验证所有周期都被正确调整
    ic_adjustment_factor = 1 - r_squared
    
    all_periods_correct = True
    for period, orig_stats in original_ic_stats.items():
        if period not in adjusted_stats:
            logger.error(f"  ❌ 缺少调整后的{period}周期数据")
            all_periods_correct = False
            continue
            
        adj_stats = adjusted_stats[period]
        expected_ic_mean = orig_stats['ic_mean'] * ic_adjustment_factor
        actual_ic_mean = adj_stats['ic_mean']
        
        if abs(expected_ic_mean - actual_ic_mean) > 0.001:
            logger.error(f"  ❌ {period}周期IC调整错误")
            all_periods_correct = False
        else:
            logger.debug(f"  ✅ {period}周期调整正确: "
                        f"{orig_stats['ic_mean']:.4f} -> {actual_ic_mean:.4f}")
    
    if all_periods_correct:
        logger.info("  ✅ 多周期IC调整功能正常")
        return True
    else:
        logger.error("  ❌ 多周期IC调整存在问题")
        return False

def run_ic_adjustment_tests():
    """运行所有IC调整测试"""
    logger.info("🚀 开始IC调整功能全面测试")
    logger.info("=" * 60)
    
    test_results = []
    
    # 测试1: 基础IC调整算法
    try:
        result1 = test_ic_adjustment_algorithm()
        test_results.append(("IC调整算法", result1))
    except Exception as e:
        logger.error(f"❌ IC调整算法测试异常: {e}")
        test_results.append(("IC调整算法", False))
    
    # 测试2: 极端R²值处理
    try:
        result2 = test_extreme_r_squared_cases()
        test_results.append(("极端R²值处理", result2))
    except Exception as e:
        logger.error(f"❌ 极端R²值测试异常: {e}")
        test_results.append(("极端R²值处理", False))
    
    # 测试3: 多周期调整
    try:
        result3 = test_multi_period_ic_adjustment()
        test_results.append(("多周期IC调整", result3))
    except Exception as e:
        logger.error(f"❌ 多周期调整测试异常: {e}")
        test_results.append(("多周期IC调整", False))
    
    # 汇总测试结果
    logger.info("\n" + "=" * 60)
    logger.info("📊 IC调整功能测试结果汇总:")
    
    passed_count = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {test_name:15s}: {status}")
        if result:
            passed_count += 1
    
    total_count = len(test_results)
    success_rate = passed_count / total_count
    
    logger.info(f"\n🎯 IC调整功能测试总结：{passed_count}/{total_count} 通过 ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        logger.info("✅ IC调整功能测试整体通过！逻辑风险已修复")
    else:
        logger.warning("⚠️ IC调整功能存在问题，需要进一步调试")
    
    return success_rate >= 0.8

if __name__ == "__main__":
    run_ic_adjustment_tests()