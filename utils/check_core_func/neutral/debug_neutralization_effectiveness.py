#!/usr/bin/env python3
"""
中性化效果验证脚本
验证因子中性化是否生效的多维度检查
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def verify_neutralization_effectiveness(original_factor_df, neutralized_factor_df, neutral_dfs, 
                                      test_dates=None, verbose=True):
    """
    全面验证中性化效果
    
    Args:
        original_factor_df: 原始因子数据 (index=date, columns=stocks)
        neutralized_factor_df: 中性化后因子数据
        neutral_dfs: 中性化用的风格因子字典
        test_dates: 测试日期列表，None表示测试所有日期
    """
    if test_dates is None:
        test_dates = original_factor_df.index[:10]  # 测试前10天
    
    results = {}
    
    print("=== 🔬 中性化效果验证报告 ===\n")
    
    # 1. 基础统计检查
    print("1️⃣ 基础统计检查")
    orig_stats = get_factor_stats(original_factor_df)
    neut_stats = get_factor_stats(neutralized_factor_df)
    
    print(f"   原始因子: 均值={orig_stats['mean']:.6f}, 标准差={orig_stats['std']:.6f}")
    print(f"   中性化后: 均值={neut_stats['mean']:.6f}, 标准差={neut_stats['std']:.6f}")
    print(f"   数据覆盖: 原始{orig_stats['coverage']:.2%} → 中性化后{neut_stats['coverage']:.2%}")
    
    # 2. 与风格因子相关性检查
    print("\n2️⃣ 与风格因子相关性检查")
    for style_name, style_df in neutral_dfs.items():
        if style_name.startswith('industry_'):
            continue  # 跳过行业哑变量
            
        orig_corr = calculate_average_correlation(original_factor_df, style_df, test_dates)
        neut_corr = calculate_average_correlation(neutralized_factor_df, style_df, test_dates)
        
        print(f"   vs {style_name:12s}: {orig_corr:7.4f} → {neut_corr:7.4f} "
              f"(降低了 {abs(orig_corr - neut_corr):.4f})")
        
        results[f'corr_reduction_{style_name}'] = abs(orig_corr - neut_corr)
    
    # 3. 行业中性化检查
    print("\n3️⃣ 行业中性化检查")
    industry_results = check_industry_neutralization(
        original_factor_df, neutralized_factor_df, neutral_dfs, test_dates
    )
    
    for industry, reduction in industry_results.items():
        print(f"   {industry}: 行业效应降低 {reduction:.4f}")
    
    # 4. 截面相关性保持检查
    print("\n4️⃣ 截面相关性保持检查")
    cross_corr = check_cross_sectional_correlation(original_factor_df, neutralized_factor_df, test_dates)
    print(f"   截面相关性保持度: {cross_corr:.4f} (>0.5为良好)")
    
    # 5. 具体日期验证
    if verbose:
        print("\n5️⃣ 具体日期验证")
        for date in test_dates[:3]:  # 展示前3天的详细情况
            daily_check(original_factor_df, neutralized_factor_df, neutral_dfs, date)
    
    return results

def get_factor_stats(factor_df):
    """获取因子基础统计"""
    values = factor_df.stack().dropna()
    total_possible = factor_df.shape[0] * factor_df.shape[1]
    coverage = len(values) / total_possible
    
    return {
        'mean': values.mean(),
        'std': values.std(),
        'coverage': coverage,
        'count': len(values)
    }

def calculate_average_correlation(factor_df, style_df, test_dates):
    """计算因子与风格因子的平均相关性"""
    correlations = []
    
    for date in test_dates:
        if date not in factor_df.index or date not in style_df.index:
            continue
            
        factor_values = factor_df.loc[date].dropna()
        style_values = style_df.loc[date].dropna()
        
        # 找到共同股票
        common_stocks = factor_values.index.intersection(style_values.index)
        if len(common_stocks) < 10:  # 至少10只股票才计算相关性
            continue
            
        factor_common = factor_values.loc[common_stocks]
        style_common = style_values.loc[common_stocks]
        
        # 计算Spearman相关性（更稳健）
        corr, pval = spearmanr(factor_common, style_common)
        if not np.isnan(corr):
            correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0.0

def check_industry_neutralization(original_factor_df, neutralized_factor_df, neutral_dfs, test_dates):
    """检查行业中性化效果"""
    # 提取行业哑变量
    industry_dummies = {k: v for k, v in neutral_dfs.items() if k.startswith('industry_')}
    
    if not industry_dummies:
        return {'no_industry_data': 0.0}
    
    results = {}
    
    # 重构行业归属矩阵
    for date in test_dates[:3]:  # 只检查前3天，避免过多输出
        if date not in original_factor_df.index:
            continue
            
        # 从哑变量重构每只股票的行业
        stock_industries = {}
        for stock in original_factor_df.columns:
            for ind_name, ind_df in industry_dummies.items():
                if date in ind_df.index and stock in ind_df.columns:
                    if ind_df.loc[date, stock] == 1:
                        industry_code = ind_name.replace('industry_', '')
                        stock_industries[stock] = industry_code
                        break
        
        if len(stock_industries) < 5:  # 行业信息太少
            continue
            
        # 计算行业内因子均值的离散程度
        orig_industry_effect = calculate_industry_effect(
            original_factor_df.loc[date], stock_industries
        )
        neut_industry_effect = calculate_industry_effect(
            neutralized_factor_df.loc[date], stock_industries
        )
        
        reduction = orig_industry_effect - neut_industry_effect
        results[f'{date.date()}'] = reduction
    
    return results

def calculate_industry_effect(factor_series, stock_industries):
    """计算行业效应强度"""
    industry_means = {}
    
    for stock, industry in stock_industries.items():
        if stock in factor_series.index and not pd.isna(factor_series[stock]):
            if industry not in industry_means:
                industry_means[industry] = []
            industry_means[industry].append(factor_series[stock])
    
    # 计算每个行业的均值
    industry_avg = {ind: np.mean(values) for ind, values in industry_means.items() 
                    if len(values) >= 2}
    
    if len(industry_avg) < 2:
        return 0.0
        
    # 行业均值的标准差代表行业效应强度
    return np.std(list(industry_avg.values()))

def check_cross_sectional_correlation(original_factor_df, neutralized_factor_df, test_dates):
    """检查截面相关性是否保持"""
    correlations = []
    
    for date in test_dates:
        if date not in original_factor_df.index or date not in neutralized_factor_df.index:
            continue
            
        orig_values = original_factor_df.loc[date].dropna()
        neut_values = neutralized_factor_df.loc[date].dropna()
        
        # 找到共同股票
        common_stocks = orig_values.index.intersection(neut_values.index)
        if len(common_stocks) < 10:
            continue
            
        orig_common = orig_values.loc[common_stocks]
        neut_common = neut_values.loc[common_stocks]
        
        corr, pval = spearmanr(orig_common, neut_common)
        if not np.isnan(corr):
            correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0.0

def daily_check(original_factor_df, neutralized_factor_df, neutral_dfs, date):
    """单日详细检查"""
    print(f"\n   📅 {date.date()} 详细检查:")
    
    orig_values = original_factor_df.loc[date].dropna()
    neut_values = neutralized_factor_df.loc[date].dropna()
    
    print(f"      有效股票数: {len(orig_values)} → {len(neut_values)}")
    print(f"      因子均值: {orig_values.mean():.6f} → {neut_values.mean():.6f}")
    print(f"      因子标准差: {orig_values.std():.6f} → {neut_values.std():.6f}")
    
    # 检查极端值变化
    orig_extreme = (orig_values.abs() > orig_values.abs().quantile(0.95)).sum()
    neut_extreme = (neut_values.abs() > neut_values.abs().quantile(0.95)).sum()
    print(f"      极端值数量: {orig_extreme} → {neut_extreme}")

if __name__ == "__main__":
    print("中性化效果验证脚本")
    print("请在主程序中调用 verify_neutralization_effectiveness() 函数")
    
    # 示例用法:
    # results = verify_neutralization_effectiveness(
    #     original_factor_df=your_original_factor,
    #     neutralized_factor_df=your_neutralized_factor, 
    #     neutral_dfs=your_neutral_dfs,
    #     test_dates=your_test_dates
    # )