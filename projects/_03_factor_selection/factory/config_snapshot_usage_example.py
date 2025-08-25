"""
配置快照管理器使用示例

本文件展示了如何使用配置快照管理系统：
1. 基本的快照创建和查询
2. 测试结果与配置关联
3. 配置回溯和对比分析
4. 历史测试查询

使用场景：
- 每次测试后自动保存配置
- 回溯分析历史测试的配置设置
- 对比不同测试版本的配置差异
- 团队协作中的配置版本管理
"""

from pathlib import Path
from datetime import datetime
import pandas as pd

from projects._03_factor_selection.factory.config_snapshot_manager import (
    ConfigSnapshotManager, 
    load_config_from_yaml
)
from projects._03_factor_selection.factory.enhanced_test_runner import EnhancedTestRunner


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("📋 配置快照管理器 - 基本使用示例")
    print("=" * 60)
    
    # 1. 初始化管理器
    workspace_root = r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\workspace"
    config_path = r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\factory\config.yaml"
    
    manager = ConfigSnapshotManager(workspace_root)
    
    # 2. 加载配置并创建快照
    config = load_config_from_yaml(config_path)
    
    snapshot_id = manager.create_snapshot(
        config=config,
        snapshot_name="动量因子测试配置_V1.0",
        test_context={
            'test_type': 'single_factor',
            'factors': ['momentum_120d', 'volatility_90d'],
            'stock_pools': ['institutional_stock_pool'],
            'researcher': '张三',
            'experiment_purpose': '验证动量因子在不同市场环境下的表现'
        }
    )
    
    print(f"✅ 创建配置快照: {snapshot_id}")
    
    # 3. 查看快照详情
    manager.print_snapshot_summary(snapshot_id)
    
    # 4. 模拟关联测试结果
    success = manager.link_test_result(
        snapshot_id=snapshot_id,
        factor_name="momentum_120d",
        stock_pool="000300",
        test_description="动量因子单因子测试 - 沪深300"
    )
    print(f"✅ 测试结果关联: {'成功' if success else '失败'}")
    
    return snapshot_id


def example_config_comparison():
    """配置对比示例"""
    print("\n" + "=" * 60)
    print("🔍 配置对比分析示例")  
    print("=" * 60)
    
    workspace_root = r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\workspace"
    config_path = r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\factory\config.yaml"
    
    manager = ConfigSnapshotManager(workspace_root)
    config = load_config_from_yaml(config_path)
    
    # 创建两个不同的配置快照
    # 快照1：原始配置
    snapshot_id1 = manager.create_snapshot(
        config=config,
        snapshot_name="基础配置_V1.0"
    )
    
    # 快照2：修改后的配置
    modified_config = config.copy()
    modified_config['evaluation']['n_groups'] = 10  # 修改分组数
    modified_config['preprocessing']['winsorization']['mad_threshold'] = 2.5  # 修改去极值阈值
    
    snapshot_id2 = manager.create_snapshot(
        config=modified_config,
        snapshot_name="优化配置_V2.0"
    )
    
    # 对比两个配置
    comparison = manager.compare_configs(snapshot_id1, snapshot_id2)
    
    print(f"📊 配置对比结果:")
    print(f"快照1: {snapshot_id1}")
    print(f"快照2: {snapshot_id2}")
    print(f"差异数量: {comparison['total_differences']}")
    
    for section, diff in comparison['differences'].items():
        print(f"\n📋 {section} 节差异:")
        print(f"  快照1: {diff['snapshot1']}")
        print(f"  快照2: {diff['snapshot2']}")


def example_historical_query():
    """历史查询示例"""
    print("\n" + "=" * 60)
    print("📚 历史配置查询示例")
    print("=" * 60)
    
    workspace_root = r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\workspace"
    manager = ConfigSnapshotManager(workspace_root)
    
    # 查询最近的配置快照
    recent_snapshots = manager.list_snapshots(limit=5)
    
    print(f"📋 最近的 {len(recent_snapshots)} 个配置快照:")
    for i, snapshot in enumerate(recent_snapshots, 1):
        print(f"  {i}. {snapshot['snapshot_id']}")
        print(f"     名称: {snapshot['snapshot_name']}")
        print(f"     时间: {snapshot['timestamp']}")
        print(f"     配置节: {', '.join(snapshot['config_sections'])}")
        print()


def example_enhanced_test_runner():
    """增强测试运行器使用示例"""
    print("\n" + "=" * 60)
    print("🚀 增强测试运行器示例")
    print("=" * 60)
    
    # 注意：这个示例需要实际的配置文件存在
    try:
        current_dir = Path(__file__).parent
        config_path = str(current_dir / 'config.yaml')
        experiments_config_path = str(current_dir / 'experiments.yaml')
        
        # 创建测试运行器
        test_runner = EnhancedTestRunner(config_path, experiments_config_path)
        
        # 模拟运行测试（实际情况会执行真实的因子测试）
        print("🔧 创建测试运行器...")
        print("📸 自动配置快照管理已集成")
        print("🧪 批量测试将自动:")
        print("   • 在测试前创建配置快照")
        print("   • 每个测试结果自动关联配置")
        print("   • 生成测试会话摘要")
        print("   • 提供完整的配置追踪链路")
        
        # 查看测试历史
        history = test_runner.get_test_history(limit=3)
        if history:
            print(f"\n📚 最近的测试历史:")
            for session in history:
                print(f"   会话: {session['session_id']}")
                print(f"   成功率: {session.get('success_rate', 0):.1%}")
                print(f"   快照: {session.get('snapshot_id', 'N/A')}")
        else:
            print("\n📚 暂无测试历史")
        
    except Exception as e:
        print(f"⚠️  示例需要实际配置文件: {e}")


def example_config_retrieval():
    """配置回溯示例"""
    print("\n" + "=" * 60)
    print("🔍 配置回溯查询示例")
    print("=" * 60)
    
    workspace_root = r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\workspace"
    manager = ConfigSnapshotManager(workspace_root)
    
    # 模拟查询特定测试的配置
    factor_name = "volatility_120d"
    stock_pool = "000300"
    
    print(f"🔍 查询因子 {factor_name} (股票池: {stock_pool}) 的测试配置...")
    
    # 获取测试对应的配置
    test_config = manager.get_test_config(
        factor_name=factor_name,
        stock_pool=stock_pool
    )
    
    if test_config:
        print("✅ 找到对应的配置快照")
        print(f"📊 配置节数量: {len(test_config)}")
        print(f"📋 包含的配置节: {list(test_config.keys())}")
        
        # 显示关键配置信息
        if 'evaluation' in test_config:
            eval_config = test_config['evaluation']
            print(f"\n📈 评价配置:")
            print(f"   分组数: {eval_config.get('n_groups', 'N/A')}")
            print(f"   前向周期: {eval_config.get('forward_periods', 'N/A')}")
        
        if 'preprocessing' in test_config:
            prep_config = test_config['preprocessing']
            print(f"\n🔧 预处理配置:")
            print(f"   去极值方法: {prep_config.get('winsorization', {}).get('method', 'N/A')}")
            print(f"   标准化方法: {prep_config.get('standardization', {}).get('method', 'N/A')}")
    else:
        print("⚠️  未找到对应的配置快照")


if __name__ == "__main__":
    """运行所有示例"""
    print("🎯 配置快照管理系统 - 完整示例")
    
    # 运行各个示例
    example_basic_usage()
    example_config_comparison()  
    example_historical_query()
    example_enhanced_test_runner()
    example_config_retrieval()
    
    print("\n" + "🎉" * 20)
    print("✅ 所有示例运行完成！")
    print("🎉" * 20)
    
    print("\n📖 使用总结:")
    print("1. 使用 EnhancedTestRunner 替代原有的测试流程")
    print("2. 每次测试自动创建配置快照并关联结果")
    print("3. 通过 ConfigSnapshotManager 查询历史配置")
    print("4. 支持配置对比和差异分析")
    print("5. 提供完整的测试配置追踪链路")