"""
Fama-MacBeth回归示例

本示例展示如何使用Fama-MacBeth回归来检验因子的有效性。
Fama-MacBeth回归是学术界和顶尖量化机构检验因子有效性的"黄金标准"。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from quant_lib.data_loader import load_stock_data
from quant_lib.evaluation import (
    calculate_ic_vectorized,
    calculate_quantile_returns,
    run_fama_macbeth_regression
)


def simple_process_factor(factor_data: pd.DataFrame,
                         winsorize_quantiles: tuple = (0.01, 0.99),
                         standardize: bool = True) -> pd.DataFrame:
    """
    简单的因子处理函数

    Args:
        factor_data: 原始因子数据
        winsorize_quantiles: 去极值的分位数
        standardize: 是否标准化

    Returns:
        处理后的因子数据
    """
    result = factor_data.copy()

    # 去极值处理
    for date in result.index:
        values = result.loc[date].dropna()
        if len(values) > 10:  # 至少需要10个有效值
            lower_bound = values.quantile(winsorize_quantiles[0])
            upper_bound = values.quantile(winsorize_quantiles[1])
            result.loc[date] = values.clip(lower_bound, upper_bound)

    # 标准化处理
    if standardize:
        for date in result.index:
            values = result.loc[date].dropna()
            if len(values) > 1 and values.std() > 0:
                mean_val = values.mean()
                std_val = values.std()
                result.loc[date, values.index] = (values - mean_val) / std_val

    return result

def main():
    """主函数：演示完整的因子分析流程"""
    
    print("="*80)
    print("Fama-MacBeth回归因子检验示例")
    print("="*80)
    
    # 1. 加载数据
    print("\n1. 加载股票数据...")
    try:
        price_data, pe_data = load_stock_data()
        print(f"价格数据形状: {price_data.shape}")
        print(f"PE数据形状: {pe_data.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 2. 处理因子数据
    print("\n2. 处理PE因子数据...")
    # PE因子：低PE更好，所以取负值
    pe_factor_processed = simple_process_factor(
        -pe_data,  # 取负值，使得因子值越大越好
        winsorize_quantiles=(0.01, 0.99),
        standardize=True
    )
    print(f"处理后PE因子形状: {pe_factor_processed.shape}")
    
    # 3. IC分析
    print("\n3. 进行IC分析...")
    forward_returns_20d = price_data.shift(-20) / price_data - 1
    ic_series = calculate_ic_vectorized(
        pe_factor_processed, 
        forward_returns_20d, 
        method='pearson'
    )
    
    print(f"IC均值: {ic_series.mean():.4f}")
    print(f"IC标准差: {ic_series.std():.4f}")
    print(f"IR (信息比率): {ic_series.mean()/ic_series.std():.4f}")
    print(f"IC胜率: {(ic_series > 0).mean():.2%}")
    
    # 4. 分层回测
    print("\n4. 进行分层回测...")
    quantile_results = calculate_quantile_returns(
        pe_factor_processed,
        price_data,
        n_quantiles=5,
        forward_periods=[20]
    )
    
    # 分析分层回测结果
    returns_20d = quantile_results[20]
    mean_returns = returns_20d.mean()
    print("各分位数平均收益率:")
    for col in mean_returns.index:
        if col != 'TopMinusBottom':
            print(f"  {col}: {mean_returns[col]:.4f}")
    print(f"  多空组合: {mean_returns['TopMinusBottom']:.4f}")
    
    # 5. Fama-MacBeth回归检验
    print("\n5. 进行Fama-MacBeth回归检验...")
    fm_results = run_fama_macbeth_regression(
        factor_df=pe_factor_processed,
        price_df=price_data,
        forward_returns_period=20
    )
    
    # 6. 综合分析结果
    print("\n" + "="*80)
    print("综合分析结果")
    print("="*80)
    
    print(f"IC分析:")
    print(f"  - IC均值: {ic_series.mean():.4f}")
    print(f"  - IR: {ic_series.mean()/ic_series.std():.4f}")
    print(f"  - IC胜率: {(ic_series > 0).mean():.2%}")
    
    print(f"\n分层回测:")
    print(f"  - 多空组合收益: {mean_returns['TopMinusBottom']:.4f}")
    print(f"  - 单调性: {'✓' if mean_returns['Q5'] > mean_returns['Q1'] else '✗'}")
    
    print(f"\nFama-MacBeth回归:")
    print(f"  - 因子收益率: {fm_results['mean_factor_return']:.6f}")
    print(f"  - t统计量: {fm_results['t_statistic']:.4f}")
    print(f"  - p值: {fm_results['p_value']:.4f}")
    print(f"  - 显著性: {'✓ 显著' if fm_results['is_significant'] else '✗ 不显著'}")
    
    # 7. 结论
    print(f"\n结论:")
    ic_good = abs(ic_series.mean()) > 0.02
    ir_good = abs(ic_series.mean()/ic_series.std()) > 0.5
    monotonic = mean_returns['Q5'] > mean_returns['Q1']
    significant = fm_results['is_significant']
    
    score = sum([ic_good, ir_good, monotonic, significant])
    
    if score >= 3:
        print("✓ PE因子表现优秀，通过多项检验，具有较强的预测能力")
    elif score >= 2:
        print("△ PE因子表现一般，部分指标达标，可考虑进一步优化")
    else:
        print("✗ PE因子表现较差，建议重新设计或放弃使用")
    
    print(f"评分: {score}/4")
    print("="*80)


def compare_multiple_factors():
    """比较多个因子的Fama-MacBeth回归结果"""
    
    print("\n" + "="*80)
    print("多因子Fama-MacBeth回归比较")
    print("="*80)
    
    # 加载数据
    price_data, pe_data = load_stock_data()
    
    # 构造多个因子进行比较
    factors = {
        'PE_Factor': simple_process_factor(-pe_data),  # PE因子
        'Momentum_20d': simple_process_factor(price_data.pct_change(20)),  # 20日动量
        'Reversal_5d': simple_process_factor(-price_data.pct_change(5)),   # 5日反转
    }
    
    results_summary = []
    
    for factor_name, factor_data in factors.items():
        print(f"\n分析因子: {factor_name}")
        
        try:
            fm_results = run_fama_macbeth_regression(
                factor_df=factor_data,
                price_df=price_data,
                forward_returns_period=20
            )
            
            results_summary.append({
                'Factor': factor_name,
                'Mean_Return': fm_results['mean_factor_return'],
                'T_Stat': fm_results['t_statistic'],
                'P_Value': fm_results['p_value'],
                'Significant': fm_results['is_significant'],
                'Num_Periods': fm_results['num_periods']
            })
            
        except Exception as e:
            print(f"因子 {factor_name} 分析失败: {e}")
    
    # 汇总结果
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        print("\n" + "="*80)
        print("多因子比较结果汇总")
        print("="*80)
        print(summary_df.to_string(index=False, float_format='%.6f'))
        
        # 按t统计量排序
        summary_df_sorted = summary_df.sort_values('T_Stat', key=abs, ascending=False)
        print(f"\n最佳因子 (按|t值|排序):")
        for _, row in summary_df_sorted.iterrows():
            significance = "***" if abs(row['T_Stat']) > 2.58 else "**" if abs(row['T_Stat']) > 1.96 else "*" if abs(row['T_Stat']) > 1.64 else ""
            print(f"  {row['Factor']}: t={row['T_Stat']:.4f} {significance}")


if __name__ == "__main__":
    # 运行基本示例
    main()
    
    # 运行多因子比较
    try:
        compare_multiple_factors()
    except Exception as e:
        print(f"多因子比较失败: {e}")
