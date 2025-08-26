"""
测试残差化规则的脚本
"""
from projects._03_factor_selection.config_manager.function_load.load_config_file import _load_local_config_functional
from quant_lib import logger

if __name__ == "__main__":
    from projects._03_factor_selection.utils.data.residualization_rules import need_residualization_in_neutral_processing, print_residualization_summary
    
    config = _load_local_config_functional('D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\factory\\config.yaml')
    factor_definitions = config['factor_definition']
    test_s  =[x for x in factor_definitions if 'cal_require_base_fields_from_daily' in x]
    test_factors =[(item['name'],item['style_category']) for item in test_s ]
    # 基于你最新添加的因子进行测试

    print("🧪 完整因子库残差化规则测试")
    print("=" * 60)
    
    needs_residualization = []
    no_residualization = []
    
    for factor_name, style_cat in test_factors:
        logger.info(f"factor_name: {factor_name}, style_category: {style_cat}")
        need_resid = need_residualization_in_neutral_processing(factor_name, style_cat)
        
        if need_resid:
            needs_residualization.append((factor_name, style_cat))
        else:
            no_residualization.append((factor_name, style_cat))
    
    print(f"\n✅ 需要残差化的因子 ({len(needs_residualization)}个):")
    print("-" * 40)
    for factor_name, style_cat in needs_residualization:
        print(f"  📊 {factor_name:30s} ({style_cat})")
    
    print(f"\n❌ 不需要残差化的因子 ({len(no_residualization)}个):")  
    print("-" * 40)
    for factor_name, style_cat in no_residualization:
        print(f"  📈 {factor_name:30s} ({style_cat})")
    
    print(f"\n📊 统计汇总:")
    print(f"  - 总因子数: {len(test_factors)}")
    print(f"  - 需要残差化: {len(needs_residualization)} ({len(needs_residualization)/len(test_factors)*100:.1f}%)")
    print(f"  - 不需要残差化: {len(no_residualization)} ({len(no_residualization)/len(test_factors)*100:.1f}%)")
    
    print()
    print_residualization_summary()