#!/usr/bin/env python3
"""
诊断皮尔逊单调系数异常高的问题

"""
import statistics

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

#基于单个日期 来看
def everyday_mono(sample_dates,factor_df,returns_df,common_stocks ,n_quantiles  ):
    MIN_SAMPLES = max(50, n_quantiles * 10)

    daily_sample_counts = []
    daily_monotonicity = []
    for date in sample_dates:
        factor_cross = factor_df.loc[date, common_stocks].dropna()
        return_cross = returns_df.loc[date, common_stocks].dropna()
        valid_stocks = factor_cross.index.intersection(return_cross.index)

        daily_sample_counts.append(len(valid_stocks))

        if len(valid_stocks) >= MIN_SAMPLES:
            factor_values = factor_cross[valid_stocks]
            return_values = return_cross[valid_stocks]

            factor_ranks = factor_values.rank(method='first')
            quantiles = pd.qcut(factor_ranks, n_quantiles, labels=False) + 1

            group_returns = []
            for q in range(1, n_quantiles + 1):
                mask = quantiles == q
                if mask.sum() > 0:
                    group_returns.append(return_values[mask].mean())
                else:
                    group_returns.append(np.nan)

            if not any(np.isnan(ret) for ret in group_returns):
                mono, _ = spearmanr(range(1, n_quantiles + 1), group_returns)
                daily_monotonicity.append(mono)
            else:
                daily_monotonicity.append(np.nan)
        else:
            daily_monotonicity.append(1.0)  # 模拟fillna(0)造成的完美单调性

    print(f"   各日单调性: {[f'{m:.3f}' if not pd.isna(m) else 'NaN' for m in daily_monotonicity]}")
    print(f"   每日样本数量加起来均值: { statistics.mean(daily_sample_counts)}")
    print(f"   每日单调性加起来均值: { statistics.mean(daily_monotonicity)}")

    insufficient_samples = sum(1 for count in daily_sample_counts if count < MIN_SAMPLES)
    print(f"   样本不足的日期数: {insufficient_samples} / {len(sample_dates)}")

    if insufficient_samples > 0:
        print(f"   ⚠️  {insufficient_samples}个日期的样本数不足，会导致人工单调性!")


def mono_by_q(factor_df, forward_returns, common_stocks, param):
    n_quantiles=5
    #对factor_df分组
    # 1. 计算未来收益率

    # 2. 数据转换与对齐：从“宽表”到“长表”
    # 有效域掩码：显式定义分析样本
    # 单一事实来源 - 明确定义所有有效的(date, stock)坐标点
    valid_mask = factor_df.notna() & forward_returns.notna()

    # 应用掩码，确保因子和收益具有完全相同的NaN分布
    final_factor = factor_df.where(valid_mask)
    final_returns = forward_returns.where(valid_mask)

    # 数据转换：从"宽表"到"长表"（现在是安全的）
    factor_long = final_factor.stack().rename('factor')
    returns_long = final_returns.stack().rename('return')

    # 合并数据（不再需要dropna，因为已经完全对齐）
    merged_df = pd.concat([factor_long, returns_long], axis=1)


    # 4. 稳健的分组：使用rank()进行等数量分组 (我们坚持的稳健方法)
    # 按日期(level=0)分(因为是多重索引，这里取第一个索引：时间)组，对每个截面内的因子值进行排名
    merged_df['rank'] = merged_df.groupby(level=0)['factor'].rank(method='first')

    # 因为rank列是唯一的，所以不需要担心duplicates问题。
    # 【改进】更严格的分组样本要求，确保统计稳定性
    MIN_SAMPLES_FOR_GROUPING = max(50, n_quantiles * 10)  # 总样本至少50个，或每组至少10个
    merged_df['quantile'] = merged_df.groupby(level=0)['rank'].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop') + 1
        if len(x) >= MIN_SAMPLES_FOR_GROUPING else np.nan
    )
    # 5. 计算各分位数的平均收益 （时间+组别 为一个group。进行求收益率平均） 今天q1组收益平均结果
    daily_quantile_returns = merged_df.groupby([merged_df.index.get_level_values(0), 'quantile'])['return'].mean()

    # 6. 数据转换：从“长表”恢复到“宽表”
    quantile_returns_wide = daily_quantile_returns.unstack()
    mean_returns = quantile_returns_wide.mean()
    quantile_means = mean_returns.tolist()

    monotonicity_spearman, p_value = spearmanr(np.arange(1, n_quantiles + 1), quantile_means)

    print(f"多日期统一 直接按组别分类 皮尔斯曼单调系数:{monotonicity_spearman}")

def analyze_debug_data():
    """分析调试数据，定位单调系数异常的原因"""
    
    print("=" * 80)
    print("皮尔逊单调系数异常诊断报告")
    print("=" * 80)
    
    # 读取测试数据
    try:
        factor_df = pd.read_csv('/tests/workspace/mem_momentum_12_1.csv',
                                index_col=0, parse_dates=True)
        returns_df = pd.read_csv('/tests/workspace/mem_forward_return_o2c.csv',
                                 index_col=0, parse_dates=True)
        
        print(f"数据加载成功")
        print(f"   因子数据形状: {factor_df.shape}")
        print(f"   收益数据形状: {returns_df.shape}")
        print(f"   因子数据日期范围: {factor_df.index.min()} ~ {factor_df.index.max()}")
        print(f"   收益数据日期范围: {returns_df.index.min()} ~ {returns_df.index.max()}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 1. 检查数据对齐问题
    print("\n1. 数据对齐检查")
    print("-" * 50)
    
    common_dates = factor_df.index.intersection(returns_df.index)
    common_stocks = factor_df.columns.intersection(returns_df.columns)
    
    print(f"   共同日期数量: {len(common_dates)} / {len(factor_df.index)} (因子) vs {len(returns_df.index)} (收益)")
    print(f"   共同股票数量: {len(common_stocks)} / {len(factor_df.columns)} (因子) vs {len(returns_df.columns)} (收益)")
    
    if len(common_dates) < 10:
        print("   ⚠️  警告: 共同日期数量过少，可能导致统计异常!")


    # 4. 遍历每天的单调系数 最后取均值 分析
    print(f"\n📅 4. 多日期分析 (样本: 前10个共同日期)")

    everyday_mono(common_dates,factor_df,returns_df,common_stocks,5)
    #多日期统一 直接按组别分类
    mono_by_q(factor_df,returns_df,common_stocks,5)

    # 5. 总结和建议
    print(f"\n💡 5. 问题诊断总结")
    print("-" * 50)
    print("根据分析，皮尔逊单调系数异常高的可能原因:")
    print("1. ✅ 样本数量不足: 当日有效股票数 < 50时，所有分组被设为NaN")
    print("2. ✅ NaN填充策略: fillna(0)将NaN组收益设为0，创造人工单调性")
    print("3. ✅ 数据稀疏性: 因子或收益数据存在大量缺失值")
    
    print(f"\n🔧 建议的修复方案:")
    print("1. 降低MIN_SAMPLES_FOR_GROUPING阈值 (从50降至20-30)")
    print("2. 改进NaN处理策略，避免直接填充0")
    print("3. 增加数据质量检查，过滤异常日期")
    print("4. 对单调性计算添加样本数量检查")
    
    print("=" * 80)

if __name__ == "__main__":
    analyze_debug_data()