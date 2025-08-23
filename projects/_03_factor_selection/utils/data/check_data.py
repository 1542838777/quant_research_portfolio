
import  pandas as pd
import numpy as np
def check_data_quality_detail(factor_df, factor_name=None):

    """
    检查因子数据质量

    参数:
    factor_df: 因子数据DataFrame
    factor_name: 因子名称（可选）

    返回:
    quality_report: 数据质量报告字典
    """
    # 初始化质量报告
    quality_report = {}
    #init
    quality_report['serious_data'] = False

    # 1. 基本统计信息
    # #print("1. 基本统计信息:")
    # #print(f"   数据形状: {factor_df.shape}")
    # #print(f"   日期范围: {factor_df.index.min()} 到 {factor_df.index.max()}")
    # #print(f"   股票数量: {len(factor_df.columns)}")
    # 连续多少天 每天的数值整行都是nan
    quality_report['max_consecutive_all_nan_days'] = get_max_nan_days(factor_df)

    # 2. 零值比例
    zero_ratio = (factor_df == 0).sum().sum() / factor_df.size
    quality_report['zero_ratio'] = zero_ratio
    #print(f"2. 零值比例: {zero_ratio:.4%}")

    # 3. NaN值比例
    nan_ratio = factor_df.isna().sum().sum() / factor_df.size
    quality_report['nan_ratio'] = nan_ratio
    #print(f"3. NaN值比例: {nan_ratio:.4%}")

    # 4. 无限值检查
    inf_ratio = np.isinf(factor_df).sum().sum() / factor_df.size
    quality_report['inf_ratio'] = inf_ratio
    #print(f"4. 无限值比例: {inf_ratio:.4%}")

    # 5. 描述性统计
    #print("5. 描述性统计:")
    desc_stats = factor_df.describe()
    #print(desc_stats)
    quality_report['desc_stats'] = desc_stats

    # 6. 极端值分析
    #print("6. 极端值分析:")

    # 计算每个分位点的值
    quantiles = factor_df.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    #print("   分位点统计:")
    #print(quantiles)
    quality_report['quantiles'] = quantiles

    # 计算异常值比例 (使用IQR方法)
    Q1 = factor_df.quantile(0.25)
    Q3 = factor_df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_ratio = ((factor_df < lower_bound) | (factor_df > upper_bound)).sum().sum() / factor_df.size
    quality_report['outliers_ratio'] = outliers_ratio
    #print(f"   异常值比例 (IQR方法): {outliers_ratio:.4%}")

    # 8. 时间序列完整性检查

    # 检查每个时间点是否有数据
    date_coverage = factor_df.notna().any(axis=1).mean()
    quality_report['date_coverage'] = date_coverage
    #print(f"   时间点覆盖率: {date_coverage:.4%}")

    # 检查每个股票的时间序列完整性
    stock_coverage = factor_df.notna().any(axis=0).mean()
    quality_report['stock_coverage'] = stock_coverage
    #print(f"   股票覆盖率: {stock_coverage:.4%}")

    # 9. 数据稳定性检查
    #print("9. 数据稳定性检查:")

    # 10. 数据质量评分
    #print("10. 数据质量评分:")

    # 基于多个指标计算综合质量评分 (0-100)
    quality_score = 100 * (
            (1 - zero_ratio) * 0.2 +
            (1 - nan_ratio) * 0.3 +
            (1 - inf_ratio) * 0.2 +
            (1 - outliers_ratio) * 0.3
    )
    if quality_report['max_consecutive_all_nan_days'] >= 5:
        quality_score  = quality_score - 100

    quality_report['quality_score'] = quality_score

    if quality_score <= 80:
        quality_report['serious_data'] = True

    return quality_report


def get_max_nan_days(factor_df):
    # 2. 连续多少天整行 NaN
    # 生成一个布尔序列：当天是否全为 NaN
    all_nan_days = factor_df.isna().all(axis=1)

    # 找到最长连续 NaN 区间
    max_streak, current = 0, 0
    for v in all_nan_days:
        if v:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    return max_streak
