#!/usr/bin/env python3
"""
数据侦探工具完整使用示例

这个脚本展示了如何使用DataForensics类进行各种数据质量诊断
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from quant_lib.data_forensics import DataForensics


def single_field_diagnosis(forensics: DataForensics, field_name: str, dataset_name: str):
    """单个字段诊断示例"""
    print(f"\n🎯 单个字段诊断示例")
    print(f"字段: {field_name}, 数据集: {dataset_name}")
    
    forensics.diagnose_field_nan(
        field_name=field_name,
        dataset_name=dataset_name,
        sample_stocks=8,
        detailed_analysis=True
    )


def batch_diagnosis_example(forensics: DataForensics):
    """批量诊断示例"""
    print(f"\n🚀 批量诊断示例")
    
    # 定义要诊断的字段列表
    field_dataset_pairs = [
        ('close', 'daily_hfq'),      # 后复权收盘价
        ('vol', 'daily_hfq'),        # 成交量
        ('pe_ttm', 'daily_basic'),   # 市盈率TTM
        ('pb', 'daily_basic'),       # 市净率
        ('turnover_rate', 'daily_basic'),  # 换手率
    ]
    
    forensics.batch_diagnose(
        field_dataset_pairs=field_dataset_pairs,
        sample_stocks=3,
        detailed_analysis=False  # 批量诊断时通常不需要详细分析
    )


def quality_report_example(forensics: DataForensics):
    """数据质量报告示例"""
    print(f"\n📊 数据质量报告示例")
    
    # 生成报告文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data_quality_report_{timestamp}.json"
    
    # 生成报告
    report = forensics.generate_data_quality_report(output_path=report_path)
    
    # 显示报告摘要
    print(f"\n📋 数据质量报告摘要:")
    print(f"  -> 总体质量分数: {report['overall_quality_score']:.2%}")
    print(f"  -> 分析字段数量: {len(report['fields_analyzed'])}")
    print(f"  -> 报告生成时间: {report['generated_at']}")
    
    # 显示各字段质量分数
    print(f"\n📈 各字段质量分数:")
    for field_info in report['fields_analyzed']:
        field_name = field_info['field_name']
        dataset_name = field_info['dataset_name']
        quality_score = field_info['quality_score']
        nan_ratio = field_info['nan_ratio']
        
        status_emoji = "✅" if quality_score > 0.95 else "⚠️" if quality_score > 0.8 else "❌"
        print(f"  {status_emoji} {field_name}@{dataset_name}: {quality_score:.2%} (NaN率: {nan_ratio:.2%})")


def interactive_mode(forensics: DataForensics):
    """交互式诊断模式"""
    print(f"\n🎮 交互式诊断模式")
    print("输入 'help' 查看可用命令，输入 'quit' 退出")
    
    while True:
        try:
            user_input = input("\n🔍 请输入命令: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            elif user_input.lower() == 'help':
                print_help()
            elif user_input.startswith('diagnose '):
                # 解析诊断命令: diagnose field_name dataset_name
                parts = user_input.split()
                if len(parts) >= 3:
                    field_name = parts[1]
                    dataset_name = parts[2]
                    sample_stocks = int(parts[3]) if len(parts) > 3 else 5
                    
                    forensics.diagnose_field_nan(
                        field_name=field_name,
                        dataset_name=dataset_name,
                        sample_stocks=sample_stocks,
                        detailed_analysis=True
                    )
                else:
                    print("❌ 格式错误。正确格式: diagnose <field_name> <dataset_name> [sample_stocks]")
            elif user_input == 'batch':
                batch_diagnosis_example(forensics)
            elif user_input == 'report':
                quality_report_example(forensics)
            else:
                print("❌ 未知命令。输入 'help' 查看可用命令。")
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，退出程序。")
            break
        except Exception as e:
            print(f"❌ 执行命令时出错: {e}")


def print_help():
    """打印帮助信息"""
    help_text = """
🆘 可用命令:
  help                                    - 显示此帮助信息
  diagnose <field> <dataset> [samples]    - 诊断指定字段 (例: diagnose close daily_hfq 5)
  batch                                   - 执行批量诊断
  report                                  - 生成数据质量报告
  quit/exit/q                            - 退出程序

📝 示例:
  diagnose close daily_hfq 8             - 诊断daily_hfq中的close字段，抽样8只股票
  diagnose pe_ttm daily_basic 3          - 诊断daily_basic中的pe_ttm字段，抽样3只股票
"""
    print(help_text)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据侦探工具 - 诊断数据中的NaN值')
    parser.add_argument('--mode', choices=['single', 'batch', 'report', 'interactive'], 
                       default='interactive', help='运行模式')
    parser.add_argument('--field', help='字段名 (single模式使用)')
    parser.add_argument('--dataset', help='数据集名 (single模式使用)')
    parser.add_argument('--samples', type=int, default=5, help='抽样股票数量')
    
    args = parser.parse_args()
    
    print("🕵️ 数据侦探工具启动...")
    print("="*60)
    
    try:
        # 初始化数据侦探
        forensics = DataForensics()
        
        if args.mode == 'single':
            if not args.field or not args.dataset:
                print("❌ single模式需要指定 --field 和 --dataset 参数")
                return 1
            single_field_diagnosis(forensics, args.field, args.dataset)
            
        elif args.mode == 'batch':
            batch_diagnosis_example(forensics)
            
        elif args.mode == 'report':
            quality_report_example(forensics)
            
        elif args.mode == 'interactive':
            interactive_mode(forensics)
            
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
