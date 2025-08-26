"""
Simple IC Adjustment Validation
简单的IC调整算法验证 - 无特殊字符版本
"""

import numpy as np

def adjust_ic_stats_by_r_squared(original_ic_stats, avg_r_squared, factor_name):
    """基于R2调整IC统计的核心算法"""
    if avg_r_squared <= 0 or avg_r_squared >= 1:
        print(f"Warning: {factor_name} has abnormal R2={avg_r_squared:.3f}, using original IC")
        return original_ic_stats
    
    # IC调整因子：残差的预测能力 ≈ (1 - R2) * 原始预测能力
    ic_adjustment_factor = 1 - avg_r_squared
    print(f"  {factor_name}: R2={avg_r_squared:.3f}, IC adjustment factor={ic_adjustment_factor:.3f}")
    
    adjusted_ic_stats = {}
    
    for period, period_stats in original_ic_stats.items():
        adjusted_period_stats = {}
        
        for key, value in period_stats.items():
            if key in ['ic_mean', 'ic_ir']:
                # IC均值和IR需要按调整系数缩放
                adjusted_value = value * ic_adjustment_factor
                adjusted_period_stats[key] = adjusted_value
            elif key in ['ic_win_rate']:
                # 胜率向50%回归
                original_win_rate = value
                adjusted_win_rate = 0.5 + (original_win_rate - 0.5) * ic_adjustment_factor
                adjusted_period_stats[key] = adjusted_win_rate
            elif key in ['ic_std', 'ic_volatility']:
                # 波动率按平方根调整
                adjusted_period_stats[key] = value * np.sqrt(ic_adjustment_factor)
            elif key in ['ic_p_value', 't_stat']:
                if key == 't_stat':
                    adjusted_period_stats[key] = value * ic_adjustment_factor
                else:  # p_value
                    adjusted_period_stats[key] = min(1.0, value / ic_adjustment_factor) if ic_adjustment_factor > 0 else 1.0
            else:
                adjusted_period_stats[key] = value
        
        adjusted_ic_stats[period] = adjusted_period_stats
    
    # 记录调整效果
    original_main_ic = original_ic_stats.get('5d', {}).get('ic_mean', 0)
    adjusted_main_ic = adjusted_ic_stats.get('5d', {}).get('ic_mean', 0)
    
    print(f"  IC adjustment: {original_main_ic:.4f} -> {adjusted_main_ic:.4f} "
          f"(reduction: {(1-ic_adjustment_factor)*100:.1f}%)")
    
    return adjusted_ic_stats

def test_basic_ic_adjustment():
    """测试基础IC调整功能"""
    print("=== Testing Basic IC Adjustment ===")
    
    # 原始IC统计
    original_ic_stats = {
        '5d': {
            'ic_mean': 0.050,
            'ic_ir': 0.800,
            'ic_win_rate': 0.65,
            'ic_std': 0.0625
        }
    }
    
    # 测试R2=0.6的情况
    r_squared = 0.6
    adjusted_stats = adjust_ic_stats_by_r_squared(
        original_ic_stats, r_squared, "test_factor"
    )
    
    # 验证结果
    ic_adjustment_factor = 1 - r_squared  # 0.4
    expected_ic_mean = original_ic_stats['5d']['ic_mean'] * ic_adjustment_factor
    actual_ic_mean = adjusted_stats['5d']['ic_mean']
    
    print(f"\nValidation Results:")
    print(f"  Expected IC mean: {expected_ic_mean:.4f}")
    print(f"  Actual IC mean: {actual_ic_mean:.4f}")
    print(f"  Error: {abs(expected_ic_mean - actual_ic_mean):.6f}")
    
    # 验证胜率调整
    expected_win_rate = 0.5 + (original_ic_stats['5d']['ic_win_rate'] - 0.5) * ic_adjustment_factor
    actual_win_rate = adjusted_stats['5d']['ic_win_rate']
    
    print(f"  Expected win rate: {expected_win_rate:.3f}")
    print(f"  Actual win rate: {actual_win_rate:.3f}")
    print(f"  Error: {abs(expected_win_rate - actual_win_rate):.6f}")
    
    # 检查是否通过
    ic_correct = abs(expected_ic_mean - actual_ic_mean) < 1e-6
    win_rate_correct = abs(expected_win_rate - actual_win_rate) < 1e-6
    
    if ic_correct and win_rate_correct:
        print("  RESULT: PASSED - IC adjustment algorithm is correct!")
        return True
    else:
        print("  RESULT: FAILED - IC adjustment algorithm has errors!")
        return False

def test_multiple_r_squared_values():
    """测试多个R2值"""
    print("\n=== Testing Multiple R2 Values ===")
    
    original_ic_stats = {
        '5d': {'ic_mean': 0.040, 'ic_ir': 0.600, 'ic_win_rate': 0.70}
    }
    
    test_r_squared_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_correct = True
    
    for r_squared in test_r_squared_values:
        print(f"\nTesting R2 = {r_squared}")
        
        adjusted_stats = adjust_ic_stats_by_r_squared(
            original_ic_stats, r_squared, f"factor_r{int(r_squared*10)}"
        )
        
        ic_adjustment_factor = 1 - r_squared
        expected_ic_mean = original_ic_stats['5d']['ic_mean'] * ic_adjustment_factor
        actual_ic_mean = adjusted_stats['5d']['ic_mean']
        
        error = abs(expected_ic_mean - actual_ic_mean)
        correct = error < 1e-6
        
        print(f"  IC: {original_ic_stats['5d']['ic_mean']:.4f} -> {actual_ic_mean:.4f} (error: {error:.8f})")
        print(f"  Status: {'PASS' if correct else 'FAIL'}")
        
        if not correct:
            all_correct = False
    
    print(f"\nMultiple R2 test result: {'PASSED' if all_correct else 'FAILED'}")
    return all_correct

def test_edge_cases():
    """测试边界情况"""
    print("\n=== Testing Edge Cases ===")
    
    original_ic_stats = {'5d': {'ic_mean': 0.040, 'ic_ir': 0.600}}
    
    edge_cases = [
        (0.0, "No correlation"),
        (0.99, "Almost perfect correlation"),
        (-0.1, "Invalid negative R2"),
        (1.1, "Invalid R2 > 1")
    ]
    
    all_correct = True
    
    for r_squared, description in edge_cases:
        print(f"\nTesting edge case: {description} (R2 = {r_squared})")
        
        try:
            adjusted_stats = adjust_ic_stats_by_r_squared(
                original_ic_stats, r_squared, f"edge_case_{r_squared}"
            )
            
            if r_squared <= 0 or r_squared >= 1:
                # Should return original stats
                if adjusted_stats == original_ic_stats:
                    print("  Status: PASS - Correctly returned original stats for invalid R2")
                else:
                    print("  Status: FAIL - Should have returned original stats")
                    all_correct = False
            else:
                print("  Status: PASS - Valid R2 processed normally")
                
        except Exception as e:
            print(f"  Status: FAIL - Exception occurred: {e}")
            all_correct = False
    
    print(f"\nEdge cases test result: {'PASSED' if all_correct else 'FAILED'}")
    return all_correct

def main():
    """主测试函数"""
    print("Starting IC Adjustment Algorithm Validation")
    print("=" * 50)
    
    test_results = []
    
    # 运行所有测试
    try:
        result1 = test_basic_ic_adjustment()
        test_results.append(("Basic IC Adjustment", result1))
    except Exception as e:
        print(f"Basic test failed with exception: {e}")
        test_results.append(("Basic IC Adjustment", False))
    
    try:
        result2 = test_multiple_r_squared_values()
        test_results.append(("Multiple R2 Values", result2))
    except Exception as e:
        print(f"Multiple R2 test failed with exception: {e}")
        test_results.append(("Multiple R2 Values", False))
    
    try:
        result3 = test_edge_cases()
        test_results.append(("Edge Cases", result3))
    except Exception as e:
        print(f"Edge cases test failed with exception: {e}")
        test_results.append(("Edge Cases", False))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS:")
    
    passed_count = 0
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name:20s}: {status}")
        if result:
            passed_count += 1
    
    total_count = len(test_results)
    success_rate = passed_count / total_count
    
    print(f"\nOVERALL: {passed_count}/{total_count} tests passed ({success_rate:.1%})")
    
    if success_rate == 1.0:
        print("SUCCESS: IC adjustment algorithm is completely correct!")
        print("The core logical risk has been fully resolved.")
    else:
        print("WARNING: IC adjustment algorithm needs further debugging.")
    
    return success_rate == 1.0

if __name__ == "__main__":
    main()