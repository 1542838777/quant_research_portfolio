import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set_style("whitegrid")


def calculate_spearman_monotonicity(group_returns):
    """
    计算分组收益的Spearman单调系数

    参数:
    group_returns: 各组平均收益列表，从低分组到高分组

    返回:
    corr: Spearman相关系数
    """
    group_index = np.arange(1, len(group_returns) + 1)
    corr, _ = stats.spearmanr(group_index, group_returns)
    return corr


def analyze_factor_performance(factor_data, return_data, factor_name, return_horizon, n_quantiles=5):
    """
    分析因子表现

    参数:
    factor_data: 因子数据DataFrame
    return_data: 收益率数据DataFrame
    factor_name: 因子名称
    return_horizon: 收益率期限
    n_quantiles: 分组数量

    返回:
    group_returns: 各组平均收益率
    monotonicity: 单调系数
    """
    # 确保数据对齐
    common_dates = factor_data.index.intersection(return_data.index)
    common_stocks = factor_data.columns.intersection(return_data.columns)

    factor_subset = factor_data.loc[common_dates, common_stocks]
    return_subset = return_data.loc[common_dates, common_stocks]

    print(f"分析因子: {factor_name}, 收益率期限: {return_horizon}")
    print(f"共同日期数量: {len(common_dates)}, 共同股票数量: {len(common_stocks)}")

    # 初始化存储各组收益的列表
    all_group_returns = []

    # 对每个交易日进行分组
    for date in common_dates:
        # 获取当日的因子值和收益率
        day_factors = factor_subset.loc[date]
        day_returns = return_subset.loc[date]

        # 移除缺失值
        valid_mask = day_factors.notna() & day_returns.notna()#关键操作 step remind
        if valid_mask.sum() < n_quantiles * 10:  # 确保有足够样本
            continue

        day_factors = day_factors[valid_mask]
        day_returns = day_returns[valid_mask] #关键操作 step

        # 按因子值分组
        try:
            quantiles = pd.qcut(day_factors, n_quantiles, labels=False, duplicates='drop')
            if len(quantiles.unique()) < n_quantiles:  # 确保分组成功
                continue

            # 计算每组的平均收益率
            group_returns = day_returns.groupby(quantiles).mean()
            all_group_returns.append(group_returns)
        except Exception as e:
            print(f"日期 {date} 分组失败: {e}")
            continue

    if not all_group_returns:
        print("没有足够的数据进行分组分析")
        return None, None

    # 计算各组的平均收益率（时间序列平均）
    all_group_returns_df = pd.DataFrame(all_group_returns)
    mean_group_returns = all_group_returns_df.mean()

    # 计算单调系数
    monotonicity = calculate_spearman_monotonicity(mean_group_returns.values)

    print(f"各组平均收益率: {mean_group_returns.values}")
    print(f"Spearman单调系数: {monotonicity:.3f}")

    return mean_group_returns, monotonicity


def plot_factor_performance(group_returns, monotonicity, factor_name, return_horizon):
    """
    绘制因子表现图表

    参数:
    group_returns: 各组平均收益率
    monotonicity: 单调系数
    factor_name: 因子名称
    return_horizon: 收益率期限
    """
    plt.figure(figsize=(10, 6))

    # 绘制分组收益率柱状图
    plt.subplot(1, 2, 1)
    groups = [f'Q{i + 1}' for i in range(len(group_returns))]
    plt.bar(groups, group_returns.values)
    plt.title(f'{factor_name} - {return_horizon}分组收益率')
    plt.ylabel('平均收益率')

    # 绘制收益率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(group_returns) + 1), group_returns.values, 'o-')
    plt.title(f'单调性: {monotonicity:.3f}')
    plt.xlabel('分组')
    plt.ylabel('平均收益率')
    plt.xticks(range(1, len(group_returns) + 1))

    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_data_quality(factor_df, factor_name=None):
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

    if factor_name:
        print(f"\n=== 检查因子 '{factor_name}' 的数据质量 ===")

    # 1. 基本统计信息
    print("1. 基本统计信息:")
    print(f"   数据形状: {factor_df.shape}")
    print(f"   日期范围: {factor_df.index.min()} 到 {factor_df.index.max()}")
    print(f"   股票数量: {len(factor_df.columns)}")

    # 2. 零值比例
    zero_ratio = (factor_df == 0).sum().sum() / factor_df.size
    quality_report['zero_ratio'] = zero_ratio
    print(f"2. 零值比例: {zero_ratio:.4%}")

    # 3. NaN值比例
    nan_ratio = factor_df.isna().sum().sum() / factor_df.size
    quality_report['nan_ratio'] = nan_ratio
    print(f"3. NaN值比例: {nan_ratio:.4%}")

    # 4. 无限值检查
    inf_ratio = np.isinf(factor_df).sum().sum() / factor_df.size
    quality_report['inf_ratio'] = inf_ratio
    print(f"4. 无限值比例: {inf_ratio:.4%}")

    # 5. 描述性统计
    print("5. 描述性统计:")
    desc_stats = factor_df.describe()
    print(desc_stats)
    quality_report['desc_stats'] = desc_stats

    # 6. 极端值分析
    print("6. 极端值分析:")

    # 计算每个分位点的值
    quantiles = factor_df.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print("   分位点统计:")
    print(quantiles)
    quality_report['quantiles'] = quantiles

    # 计算异常值比例 (使用IQR方法)
    Q1 = factor_df.quantile(0.25)
    Q3 = factor_df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_ratio = ((factor_df < lower_bound) | (factor_df > upper_bound)).sum().sum() / factor_df.size
    quality_report['outliers_ratio'] = outliers_ratio
    print(f"   异常值比例 (IQR方法): {outliers_ratio:.4%}")

    # 7. 数据分布可视化
    print("7. 生成数据分布可视化...")

    # 扁平化数据以便绘制直方图
    flat_data = factor_df.values.flatten()
    flat_data = flat_data[~np.isnan(flat_data)]  # 移除NaN值
    flat_data = flat_data[np.isfinite(flat_data)]  # 移除无限值

    if len(flat_data) > 0:
        plt.figure(figsize=(12, 5))

        # 直方图
        plt.subplot(1, 2, 1)
        plt.hist(flat_data, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{factor_name if factor_name else "因子"}值分布')
        plt.xlabel('因子值')
        plt.ylabel('频率')

        # 箱线图
        plt.subplot(1, 2, 2)
        plt.boxplot(flat_data)
        plt.title(f'{factor_name if factor_name else "因子"}值箱线图')
        plt.ylabel('因子值')

        plt.tight_layout()
        plt.show()
    else:
        print("   没有有效数据可用于可视化")

    # 8. 时间序列完整性检查
    print("8. 时间序列完整性检查:")

    # 检查每个时间点是否有数据
    date_coverage = factor_df.notna().any(axis=1).mean()
    quality_report['date_coverage'] = date_coverage
    print(f"   时间点覆盖率: {date_coverage:.4%}")

    # 检查每个股票的时间序列完整性
    stock_coverage = factor_df.notna().any(axis=0).mean()
    quality_report['stock_coverage'] = stock_coverage
    print(f"   股票覆盖率: {stock_coverage:.4%}")

    # 9. 数据稳定性检查
    print("9. 数据稳定性检查:")

    # 计算每个时间点的有效股票数量
    valid_stocks_per_date = factor_df.notna().sum(axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(valid_stocks_per_date.index, valid_stocks_per_date.values)
    plt.title('每个时间点的有效股票数量')
    plt.xlabel('日期')
    plt.ylabel('有效股票数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 计算有效股票数量的统计
    valid_stocks_stats = valid_stocks_per_date.describe()
    print("   有效股票数量统计:")
    print(valid_stocks_stats)
    quality_report['valid_stocks_stats'] = valid_stocks_stats

    # 10. 数据质量评分
    print("10. 数据质量评分:")

    # 基于多个指标计算综合质量评分 (0-100)
    quality_score = 100 * (
            (1 - zero_ratio) * 0.2 +
            (1 - nan_ratio) * 0.3 +
            (1 - inf_ratio) * 0.2 +
            (1 - outliers_ratio) * 0.3
    )

    quality_report['quality_score'] = quality_score
    print(f"   综合质量评分: {quality_score:.2f}/100")

    if quality_score >= 80:
        print("   数据质量: 优秀")
    elif quality_score >= 60:
        print("   数据质量: 良好")
    elif quality_score >= 40:
        print("   数据质量: 一般")
    else:
        print("   数据质量: 较差")

    return quality_report


# 使用示例
def run_check(factor_df):
        # 检查数据质量
        quality_report = check_data_quality(factor_df, "波动率因子")

        # 可以根据质量报告做出进一步决策
        if quality_report['quality_score'] < 60:
            print("\n警告: 数据质量较差，可能影响分析结果")
            print("建议进行数据清洗和处理")


def check_cross_sectional_duplicates(factor_df, threshold=0.1):
    """
    检查截面重复值问题

    参数:
    factor_df: 因子数据DataFrame
    threshold: 重复值比例阈值，超过此值认为问题严重

    返回:
    duplicate_report: 重复值检查报告
    """
    print("\n=== 截面重复值检查 ===")

    duplicate_report = {}

    # 1. 按日期检查重复值
    duplicate_ratios = []
    problematic_dates = []

    for date in factor_df.index:
        daily_values = factor_df.loc[date].dropna()

        if len(daily_values) == 0:
            continue

        # 计算重复值比例
        total_count = len(daily_values)
        unique_count = len(daily_values.unique())
        duplicate_ratio = 1 - (unique_count / total_count)
        duplicate_ratios.append(duplicate_ratio)

        if duplicate_ratio > threshold:
            problematic_dates.append((date, duplicate_ratio))

    # 整体重复值统计
    mean_duplicate_ratio = np.mean(duplicate_ratios) if duplicate_ratios else 0
    max_duplicate_ratio = np.max(duplicate_ratios) if duplicate_ratios else 0

    duplicate_report['mean_duplicate_ratio'] = mean_duplicate_ratio
    duplicate_report['max_duplicate_ratio'] = max_duplicate_ratio
    duplicate_report['problematic_dates'] = problematic_dates

    print(f"平均重复值比例: {mean_duplicate_ratio:.4%}")
    print(f"最大重复值比例: {max_duplicate_ratio:.4%}")

    # 2. 显示问题最严重的几个日期
    if problematic_dates:
        print(f"\n发现 {len(problematic_dates)} 个日期重复值比例超过 {threshold:.0%}:")
        problematic_dates_sorted = sorted(problematic_dates, key=lambda x: x[1], reverse=True)[:5]
        for date, ratio in problematic_dates_sorted:
            print(f"  {date.date()}: {ratio:.4%}")

    # 3. 检查特定日期的详细情况（示例）
    if problematic_dates:
        example_date = problematic_dates[0][0]
        print(f"\n示例日期 {example_date.date()} 的详细分析:")

        daily_values = factor_df.loc[example_date].dropna()
        value_counts = daily_values.value_counts()

        print(f"  总股票数: {len(daily_values)}")
        print(f"  唯一值数量: {len(value_counts)}")

        # 显示最常见的几个值及其出现次数
        print("  最常见因子值:")
        for value, count in value_counts.head(5).items():
            print(f"    {value}: {count}次 ({(count / len(daily_values)):.2%})")

    # 4. 可视化重复值比例的时间序列
    plt.figure(figsize=(12, 5))

    # 重复值比例时间序列
    plt.subplot(1, 2, 1)
    plt.plot(factor_df.index[:len(duplicate_ratios)], duplicate_ratios)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.0%})')
    plt.title('截面重复值比例时间序列')
    plt.xlabel('日期')
    plt.ylabel('重复值比例')
    plt.legend()

    # 重复值比例分布
    plt.subplot(1, 2, 2)
    plt.hist(duplicate_ratios, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.0%})')
    plt.title('重复值比例分布')
    plt.xlabel('重复值比例')
    plt.ylabel('频率')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 5. 评估严重程度
    if mean_duplicate_ratio > 0.3:
        severity = "严重"
        recommendation = "必须处理！分组结果完全不可信"
    elif mean_duplicate_ratio > 0.1:
        severity = "中等"
        recommendation = "需要处理，否则会影响分组质量"
    else:
        severity = "轻微"
        recommendation = "影响较小，但仍建议监控"

    print(f"\n严重程度评估: {severity}")
    print(f"处理建议: {recommendation}")

    return duplicate_report




# 使用示例
def check_cross_sectional_duplicates_run():
    # 读取测试数据
    try:
        factor_df = pd.read_csv(
            '/tests/workspace/mem_momentum_12_1.csv',
            index_col=0, parse_dates=True)


        # 检查截面重复值
        duplicate_report = check_cross_sectional_duplicates(factor_df)

        # 如果重复值严重，建议的处理方法
        if duplicate_report['mean_duplicate_ratio'] > 0.1:
            print("\n=== 处理建议 ===")
            print("1. 添加微小噪声打破重复值:")
            print("   factor_df = factor_df + np.random.normal(0, 1e-10, factor_df.shape)")
            print("2. 使用排名分组代替分位数分组")
            print("3. 增加因子精度，避免四舍五入导致的重复值")
            print("4. 检查因子计算过程，确保不会产生大量相同值")

    except Exception as e:
        print(f"数据处理出错: {e}")
        import traceback

        traceback.print_exc()

if __name__ == "__main__":

    check_cross_sectional_duplicates_run()
    # 读取测试数据
    try:
        factor_df = pd.read_csv(
            '/tests/workspace/mem_momentum_12_1.csv',
            index_col=0, parse_dates=True)
        returns_df = pd.read_csv(
            '/tests/workspace/mem_forward_return_o2c.csv',
            index_col=0, parse_dates=True)
        #检查因子数据质量
        run_check(factor_df )

        # 选择要分析的因子和收益率期限
        # 假设因子数据只有一列，或者您可以选择特定列
        factor_name = factor_df.columns[0] if len(factor_df.columns) == 1 else 'momentum_12_1'

        # 假设收益率数据有多列，每列对应不同期限
        # 这里选择第一列作为示例
        return_horizon = returns_df.columns[0] if len(returns_df.columns) > 0 else '1d'

        # 分析因子表现
        group_returns, monotonicity = analyze_factor_performance(
            factor_df, returns_df, factor_name, return_horizon
        )

        if group_returns is not None:
            # 绘制图表
            plot_factor_performance(group_returns, monotonicity, factor_name, return_horizon)

            # 输出详细分析
            print("\n详细分析:")
            for i, ret in enumerate(group_returns.values):
                print(f"第{i + 1}组平均收益率: {ret:.6f}")

            # 检查单调性是否异常高
            if abs(monotonicity) > 0.9:
                print(f"\n警告: 检测到异常高的单调系数 ({monotonicity:.3f})")
                print("可能的原因:")
                print("1. 因子与收益率确实有强关系")
                print("2. 数据中存在异常值或极端值")
                print("3. 分组数量太少或样本不足")
                print("4. 计算或数据处理错误")

    except Exception as e:
        print(f"数据处理出错: {e}")
        import traceback

        traceback.print_exc()