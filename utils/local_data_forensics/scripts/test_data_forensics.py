#!/usr/bin/env python3
"""
数据侦探工具测试脚本

用于测试和演示数据法证诊断器的功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from quant_lib.data_forensics import DataForensics


def main():
    """主测试函数"""
    print("🚀 启动数据侦探工具测试...")
    
    try:
        # 1. 初始化数据侦探
        forensics = DataForensics()
        
        # 2. 测试不同数据集和字段的NaN诊断
        test_cases = [
            {
                'field_name': 'close',
                'dataset_name': 'daily_hfq',
                'description': '后复权收盘价'
            },
            {
                'field_name': 'pe_ttm', 
                'dataset_name': 'daily_basic',
                'description': '市盈率TTM'
            },
            {
                'field_name': 'vol',
                'dataset_name': 'daily_hfq', 
                'description': '成交量'
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"🧪 测试案例 {i}: {test_case['description']} ({test_case['field_name']})")
            print(f"{'='*80}")
            
            try:
                forensics.diagnose_field_nan(
                    field_name=test_case['field_name'],
                    dataset_name=test_case['dataset_name'],
                    sample_stocks=5,
                    detailed_analysis=True
                )
            except Exception as e:
                print(f"❌ 测试案例 {i} 失败: {e}")
                continue
                
        print(f"\n{'='*80}")
        print("✅ 数据侦探工具测试完成！")
        print("💡 提示: 你可以根据诊断结果决定是否需要:")
        print("   - 检查数据下载的完整性")
        print("   - 更新股票基本信息")
        print("   - 处理停牌期间的数据填充")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return 1
        
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
