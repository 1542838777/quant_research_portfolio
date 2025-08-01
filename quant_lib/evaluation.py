import vectorbt as vbt
from pandas import Series

from quant_lib import logger
from quant_lib.utils.dataFrame_utils import align_dataframes

"""
评估模块

提供因子评价和策略评估功能，包括IC分析、分层回测、业绩归因等。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp

# 尝试导入statsmodels，如果没有安装则使用简化版本
try:
    import statsmodels.api as sm

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("警告: statsmodels未安装，将使用简化版本的回归分析")


def calculate_ic(factor_df: pd.DataFrame,
                 forward_returns: pd.DataFrame,
                 method: str = 'pearson') -> pd.Series:
    """
    计算因子IC值

    Args:
        factor_df: 因子值DataFrame，index为日期，columns为股票代码
        forward_returns: 未来收益率DataFrame，index为日期，columns为股票代码
        method: 相关系数计算方法，'pearson'或'spearman'

    Returns:
        IC值序列，index为日期
    """
    logger.info(f"计算{method}类型IC...")

    # 原始实现（循环方式）
    ic_series = pd.Series(index=factor_df.index)

    for date in factor_df.index:
        if date not in forward_returns.index:
            continue

        # 获取当天的因子值和未来收益率
        factor_values = factor_df.loc[date].dropna()
        returns = forward_returns.loc[date].dropna()

        # 找出共同的股票
        common_stocks = factor_values.index.intersection(returns.index)

        if len(common_stocks) < 10:  # 至少需要10只股票
            continue

        # 计算相关系数
        if method == 'pearson':
            ic, _ = stats.pearsonr(factor_values[common_stocks], returns[common_stocks])
        else:  # spearman
            ic, _ = stats.spearmanr(factor_values[common_stocks], returns[common_stocks])

        ic_series[date] = ic

    logger.info(f"IC计算完成: 均值={ic_series.mean():.4f}, IR={ic_series.mean() / ic_series.std():.4f}")
    return ic_series


# ok
def calculate_ic_vectorized(
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        forward_periods: List[int] = [1, 5, 20],
        method: str = 'spearman',
        min_stocks: int = 10
) -> Tuple[Dict[str, Series], Dict[str, pd.DataFrame]]:
    """
    【生产级版本】向量化计算因子IC值及相关统计指标。
    此版本逻辑严密，接口清晰，返回纯粹的IC序列和独立的统计数据。

    Args:
        factor_df: 因子值DataFrame
        forward_returns: 未来收益率DataFrame
        method: 相关系数计算方法, 'pearson'或'spearman'
        min_stocks: 每个日期至少需要的【有效配对】股票数量

    Returns:
        一个元组 (ic_series, stats_dict):
        - ic_series (pd.Series): IC时间序列，索引为满足条件的有效日期。
        - stats_dict (Dict): 包含IC均值、ICIR、t值、p值等核心统计指标的字典。
    """
    logger.info(f"\t向量化计算 {method.capitalize()} 类型IC (生产级版本)...")
    stats_periods_dict = {}
    ic_series_periods_dict = {}
    if factor_df.empty or price_df.empty:
        raise ValueError("输入的因子或价格数据为空，无法计算IC。")
    for period in forward_periods:
        forward_returns = price_df.shift(-period) / price_df - 1


        common_idx = factor_df.index.intersection(forward_returns.index)
        common_cols = factor_df.columns.intersection(forward_returns.columns)

        if len(common_idx) == 0 or len(common_cols) == 0:
            raise ValueError("因子数据和收益数据没有重叠的日期或股票，无法计算IC。")


        factor_aligned = factor_df.loc[common_idx, common_cols]
        returns_aligned = forward_returns.loc[common_idx, common_cols]

        # --- 2. 计算有效配对数并筛选日期 ---
        paired_valid_counts = (factor_aligned.notna() & returns_aligned.notna()).sum(axis=1)
        valid_dates = paired_valid_counts[paired_valid_counts >= min_stocks].index

        if valid_dates.empty:
            raise ValueError(f"没有任何日期满足最小股票数量({min_stocks})要求，无法计算IC。")


        # --- 3. 核心计算 ---
        ic_series = factor_aligned.loc[valid_dates].corrwith(
            returns_aligned.loc[valid_dates],
            axis=1,
            method=method.lower()
        ).rename("IC")  # 给Series命名是一个好习惯

        # --- 4. 计算统计指标 (只在有效IC序列上计算) ---
        ic_series_cleaned = ic_series.dropna()
        if len(ic_series_cleaned) < 2:  # t检验至少需要2个样本
            raise ValueError(f"有效IC值数量过少({len(ic_series_cleaned)})，无法计算统计指标。")


        # 修正胜率计算和添加更多统计指标
        ic_mean = ic_series_cleaned.mean()
        ic_std = ic_series_cleaned.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
        ic_t_stat, ic_p_value = stats.ttest_1samp(ic_series_cleaned, 0)

        # 胜率！。（表示正确出现的次数/总次数）
        ic_win_rate = ((ic_series_cleaned * ic_mean) > 0).mean()  # 这个就是计算胜率，简化版！ 计算的是IC值与IC均值同向的比例
        # 方向性检查
        if abs(ic_mean) > 1e-10 and np.sign(ic_t_stat) != np.sign(ic_mean):
            raise ValueError("严重错误：t统计量与IC均值方向不一致！")
        dayStr = f'{period}d'
        ic_series_periods_dict[dayStr] = ic_series
        stats_periods_dict[dayStr] = {
            # 'ic_series': ic_series,
            'ic_mean': ic_mean,  # >=0.02 及格 。超过0.04良好 超过0.06 超级好
            'ic_std': ic_std,  # 标准差，波动情况
            'ic_ir': ic_ir,  # 稳定性。>0.3才行 >0.5非常稳定优秀！
            'ic_win_rate': ic_win_rate,  # 胜率，在均值决定的方向上，正确出现的次数 >0.55才行
            'ic_abs_mean': ic_series_cleaned.abs().mean(),  # 不是很重要，这个值大的话，才有研究意义，能说明 在方向上有效果，而不是趋于0， 个人推荐>0.03
            'ic_t_stat': ic_t_stat,  # 大于2才有意义
            'ic_p_value': ic_p_value,  # <0.05 说明因子真的有效果
            'ic_significant': ic_p_value < 0.05,

            'ic_Valid Days': len(ic_series_cleaned),
            'ic_Total Days': len(common_idx),
            'ic_Coverage Rate': len(ic_series_cleaned) / len(common_idx)
        }
    return ic_series_periods_dict, stats_periods_dict


def calculate_ic_decay(factor_df: pd.DataFrame,
                       price_df: pd.DataFrame,
                       periods: List[int] = [1, 5, 10, 20, 60],
                       method: str = 'pearson',
                       use_vectorized: bool = True) -> pd.DataFrame:
    """
    计算因子IC衰减

    Args:
        factor_df: 因子值DataFrame
        price_df: 价格DataFrame
        periods: 未来时间周期列表
        method: 相关系数计算方法
        use_vectorized: 是否使用向量化计算方法

    Returns:
        不同时间周期的IC均值DataFrame
    """
    logger.info("计算IC衰减...")

    results = {
        'period': periods,
        'IC_Mean': [],
        'IC_IR': []
    }

    for period in periods:
        # 计算未来收益率
        forward_returns = price_df.shift(-period) / price_df - 1

        # 计算IC
        if use_vectorized:
            ic = calculate_ic_vectorized(factor_df, forward_returns, method)
        else:
            ic = calculate_ic(factor_df, forward_returns, method)

        # 存储结果
        results['IC_Mean'].append(ic.mean())
        results['IC_IR'].append(ic.mean() / ic.std() if ic.std() > 0 else 0)

    # 创建结果DataFrame
    result_df = pd.DataFrame(results).set_index('period')

    return result_df


# ok
def quantile_stats_result(results: Dict[int, pd.DataFrame], n_quantiles: int) -> Tuple[Dict[str, pd.DataFrame],Dict[str,  pd.DataFrame]]:
    """
    计算并汇总分层回测的关键性能指标。

    Args:
        results: 一个字典，键是向前看的周期（如1, 5），值是对应的分层收益DataFrame。
        n_quantiles: 分层回测中使用的分位数数量。

    Returns:
        一个字典，包含了每个周期的汇总统计指标。
    """
    quantile_stats_periods_dict = {}
    quantile_returns_periods_dict = {}

    # 修正了循环的写法，'result' 就是当前周期的DataFrame
    for period, result in results.items():
        if result.empty:
            continue

        mean_returns = result.mean()
        tmb_series = result['TopMinusBottom'].dropna()

        if tmb_series.empty:
            continue

        # --- 核心计算 ---
        tmb_mean_period_return = tmb_series.mean()
        tmb_std_period_return = tmb_series.std()

        # ✅ **新增**: 计算年化收益率
        # 假设数据的基础频率是每日
        tmb_annual_return = (tmb_mean_period_return / period) * 252 if period > 0 else 0

        # **优化**: 修正夏普比率以适应不同持有期
        # 夏普比率的年化因子是 sqrt(252 / period)
        tmb_sharpe = (tmb_mean_period_return / tmb_std_period_return) * np.sqrt(
            252 / period) if tmb_std_period_return > 0 and period > 0 else 0

        tmb_win_rate = (tmb_series > 0).mean()
        max_drawdown, mdd_start, mdd_end = calculate_max_drawdown_robust(tmb_series)

        # 单调性检验
        quantile_means = [mean_returns.get(f'Q{i + 1}', np.nan) for i in range(n_quantiles)]
        is_monotonic_by_group = False
        if not any(np.isnan(q) for q in quantile_means):
            is_monotonic_by_group = all(
                quantile_means[i] <= quantile_means[i + 1] for i in range(len(quantile_means) - 1))

        # --- 存储结果 ---
        # 'period' 变量用于创建描述性的键，如 '5d'
        quantile_returns_periods_dict[f'{period}d'] = result
        quantile_stats_periods_dict[f'{period}d'] = {
            # 'returns_data': result,
            'mean_returns': mean_returns,
            'tmb_return_period': tmb_mean_period_return,  # 特定周期的平均收益 (例如，5日平均收益)
            'tmb_annual_return': tmb_annual_return,  # **新增**: 年化后的多空组合收益率
            'tmb_sharpe': tmb_sharpe,  # **优化**: 周期调整后的夏普比率
            'tmb_win_rate': tmb_win_rate,
            'is_monotonic_by_group': is_monotonic_by_group,
            'max_drawdown': max_drawdown,              # ✅【新增】最大回撤值
            'mdd_start_date': mdd_start,            # ✅【新增】最大回撤开始日期
            'mdd_end_date': mdd_end,                # ✅【新增】最大回撤结束日期
            'quantile_means': quantile_means
        }

    return quantile_returns_periods_dict,quantile_stats_periods_dict
#ok
def calculate_quantile_returns(
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        n_quantiles: int = 5,
        forward_periods: List[int] = [1, 5, 20]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
   计算因子分位数的未来收益率。
    该版本采用向量化实现，并使用rank()进行稳健分组，

    Args:
        factor_df (pd.DataFrame): 因子值DataFrame (index=date, columns=stock)
        price_df: pd.DataFrame: 价格DataFrame (index=date, columns=stock)
        n_quantiles (int): 要划分的分位数数量，默认为5
        forward_periods (List[int]): 未来时间周期列表，如[1, 5, 20]

    Returns:
        Dict[int, pd.DataFrame]: 一个字典，键是未来时间周期(period)，
                                 值是对应的分位数收益DataFrame。
                                 每个DataFrame的index是日期，columns是Q1, Q2... TopMinusBottom。
    """
    results = {}

    for period in forward_periods:
        logger.info(f"  > 正在处理向前看 {period} 周期...")

        # 1. 计算未来收益率 (向量化)
        forward_returns = price_df.pct_change(periods=period).shift(-period)

        # 2. 数据转换与对齐：从“宽表”到“长表”
        factor_long = factor_df.stack().rename('factor')
        returns_long = forward_returns.stack().rename('return')

        # 3. 合并因子和收益，并丢弃任何一个为NaN的行
        merged_df = pd.concat([factor_long, returns_long], axis=1).dropna()

        if merged_df.empty:
            logger.warning(f"  > 在周期 {period}，因子和收益数据没有重叠，无法计算。")#  考虑 要不要直接报错 不能，因为forward_returns有nan很正常
            # 创建一个空的DataFrame以保持输出结构一致性
            empty_cols = [f'Q{i + 1}' for i in range(n_quantiles)] + ['TopMinusBottom']
            results[period] = pd.DataFrame(columns=empty_cols, dtype='float64')
            continue

        # 4. 稳健的分组：使用rank()进行等数量分组 (我们坚持的稳健方法)
        # 按日期(level=0)分(因为是多重索引，这里取第一个索引：时间)组，对每个截面内的因子值进行排名
        merged_df['rank'] = merged_df.groupby(level=0)['factor'].rank(method='first')


        # 因为rank列是唯一的，所以不需要担心duplicates问题。
        merged_df['quantile'] = merged_df.groupby(level=0)['rank'].transform(
            lambda x: pd.qcut(x, n_quantiles, labels=False) + 1
        )
        # 5. 计算各分位数的平均收益 （时间+组别 为一个group。进行求收益率平均）
        daily_quantile_returns = merged_df.groupby([merged_df.index.get_level_values(0), 'quantile'])['return'].mean()

        # 6. 数据转换：从“长表”恢复到“宽表”
        quantile_returns_wide = daily_quantile_returns.unstack()
        # 改个列名
        quantile_returns_wide.columns = [f'Q{int(col)}' for col in quantile_returns_wide.columns]

        # 7. 计算多空组合收益
        top_q_col = f'Q{n_quantiles}'
        bottom_q_col = 'Q1'

        if top_q_col in quantile_returns_wide.columns and bottom_q_col in quantile_returns_wide.columns:
            quantile_returns_wide['TopMinusBottom'] = quantile_returns_wide[top_q_col] - quantile_returns_wide[
                bottom_q_col]
        else:
            # 确保即使在极端情况下，列也存在，值为NaN，保持DataFrame结构一致
            quantile_returns_wide['TopMinusBottom'] = np.nan

        # 8. 存储结果
        results[period] = quantile_returns_wide.sort_index(axis=1)
    return  quantile_stats_result(results, n_quantiles)


def plot_ic_series(ic_series: pd.Series, title: str = 'IC时间序列', figsize: Tuple[int, int] = (12, 6)):
    """
    绘制IC时间序列图

    Args:
        ic_series: IC序列
        title: 图表标题
        figsize: 图表大小

    Returns:
        fig, ax: 图表对象，可用于进一步自定义或保存
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制IC序列
    ax.plot(ic_series, label='IC')

    # 绘制均值线
    ax.axhline(y=ic_series.mean(), color='r', linestyle='-', label=f'均值: {ic_series.mean():.4f}')

    # 绘制0线
    ax.axhline(y=0, color='k', linestyle='--')

    # 添加标题和标签
    ax.set_title(title)
    ax.set_xlabel('日期')
    ax.set_ylabel('IC值')
    ax.legend()
    ax.grid(True)

    return fig, ax


def plot_ic_decay(ic_decay_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)):
    """
    绘制IC衰减图

    Args:
        ic_decay_df: IC衰减DataFrame
        figsize: 图表大小

    Returns:
        fig, ax: 图表对象，可用于进一步自定义或保存
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制IC衰减曲线
    ax.plot(ic_decay_df.index, ic_decay_df['IC_Mean'], marker='o', linestyle='-')

    # 添加标题和标签
    ax.set_title('IC衰减曲线')
    ax.set_xlabel('持有期（天）')
    ax.set_ylabel('IC均值')
    ax.grid(True)

    return fig, ax


def plot_quantile_returns(quantile_returns: pd.DataFrame,
                          period: int,
                          figsize: Tuple[int, int] = (12, 6)):
    """
    绘制分位数收益图

    Args:
        quantile_returns: 分位数收益DataFrame
        period: 持有期
        figsize: 图表大小

    Returns:
        fig_returns, fig_tmb: 两个图表对象，分别为分位数收益图和多空组合收益图
    """
    # 计算累计收益
    cumulative_returns = (1 + quantile_returns).cumprod()

    # 绘制各分位数累计收益曲线
    fig_returns, ax_returns = plt.subplots(figsize=figsize)

    for col in cumulative_returns.columns:
        if col != 'TopMinusBottom':
            ax_returns.plot(cumulative_returns[col], label=col)

    # 添加标题和标签
    ax_returns.set_title(f'{period}日持有期分位数累计收益')
    ax_returns.set_xlabel('日期')
    ax_returns.set_ylabel('累计收益')
    ax_returns.legend()
    ax_returns.grid(True)

    # 绘制多空组合收益
    fig_tmb, ax_tmb = plt.subplots(figsize=figsize)
    ax_tmb.plot(cumulative_returns['TopMinusBottom'], color='r')
    ax_tmb.set_title(f'{period}日持有期多空组合累计收益')
    ax_tmb.set_xlabel('日期')
    ax_tmb.set_ylabel('累计收益')
    ax_tmb.grid(True)

    return fig_returns, fig_tmb


def calculate_turnover(positions_df: pd.DataFrame) -> pd.Series:
    """
    计算换手率

    Args:
        positions_df: 持仓DataFrame，index为日期，columns为股票代码

    Returns:
        换手率序列
    """
    logger.info("计算换手率...")

    turnover = pd.Series(index=positions_df.index[1:])

    for i in range(1, len(positions_df)):
        prev_pos = positions_df.iloc[i - 1]
        curr_pos = positions_df.iloc[i]

        # 计算持仓变化
        pos_change = abs(curr_pos - prev_pos).sum() / 2

        turnover.iloc[i - 1] = pos_change

    logger.info(f"换手率计算完成: 平均换手率={turnover.mean():.4f}")
    return turnover


def calculate_turnover_vectorized(positions_df: pd.DataFrame) -> pd.Series:
    """
    向量化计算换手率，效率更高

    Args:
        positions_df: 持仓DataFrame，index为日期，columns为股票代码

    Returns:
        换手率序列
    """
    logger.info("向量化计算换手率...")

    # 计算相邻日期之间的持仓变化
    pos_change = positions_df.diff(1).abs().sum(axis=1)

    # 根据公式，换手率是买入和卖出总额的一半
    turnover = pos_change / 2

    # 第一个日期没有换手率
    turnover = turnover.iloc[1:]

    logger.info(f"换手率计算完成: 平均换手率={turnover.mean():.4f}")
    return turnover


def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    计算夏普比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率

    Returns:
        夏普比率
    """
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0



def calculate_max_drawdown_robust(
        returns: pd.Series
) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    【健壮版】计算最大回撤及其开始和结束日期。
    此版本修复了在单调上涨行情下的边界Bug，并增强了对异常输入的处理。

    Args:
        returns: 收益率序列 (pd.Series)

    Returns:
        一个元组 (max_drawdown, start_date, end_date):
        - max_drawdown (float): 最大回撤值 (一个负数或0)。
        - start_date (pd.Timestamp or None): 最大回撤开始的日期（前期高点）。
        - end_date (pd.Timestamp or None): 最大回撤结束的日期（最低点）。
    """
    # --- 1. 输入验证与数据清洗 (提升健壮性) ---
    if returns is None or returns.empty:
        return 0.0, None, None

    # 填充NaN值，最常见的处理是视为当天无收益
    returns_cleaned = returns.fillna(0)

    # --- 2. 核心计算 (与原版逻辑相同) ---
    # 计算累计净值曲线 (通常从1开始)
    cumulative_returns = (1 + returns_cleaned).cumprod()

    # 计算历史最高点 (High-Water Mark)
    running_max = cumulative_returns.cummax()

    # 计算回撤序列 (当前值距离历史最高点的百分比)
    drawdown = (cumulative_returns / running_max) - 1

    # --- 3. 结果提取 (修复Bug) ---
    max_drawdown = drawdown.min()

    # 如果没有回撤 (策略一直盈利), 直接返回
    if max_drawdown == 0:
        return 0.0, None, None

    end_date = drawdown.idxmin()

    # 【核心Bug修复】
    # 在最大回撤结束点之前的序列中寻找高点
    peak_series = cumulative_returns.loc[:end_date]
    start_date = peak_series.idxmax()

    return max_drawdown, start_date, end_date


# ok 传入的必须是 shif之后的！！！！
def fama_macbeth_regression(
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        forward_returns_period: int = 20,
        weights_df: pd.DataFrame = None,
        neutral_factors: Dict[str, pd.DataFrame] = None
) -> Tuple[Series,Dict[str, Any]]:
    """
    【最终生产版】对单个因子进行Fama-MacBeth回归检验。
    此版本逻辑结构清晰，代码健壮，并使用Newey-West标准误修正t检验，符合学术界和业界的严格标准。
    """
    # 初始化logger
    from quant_lib.config.logger_config import setup_logger
    logger = setup_logger(__name__)

    logger.info(f"开始Fama-MacBeth回归分析 (前向收益期: {forward_returns_period}天)")

    # --- 0. 前置检查 ---
    if not HAS_STATSMODELS:
        logger.error("statsmodels未安装，无法执行Fama-MacBeth回归")
        return {'error': 'statsmodels未安装'}

    if factor_df.empty or price_df.empty:
        logger.error("输入数据为空")
        return {'error': '输入数据为空'}

    # 检查因子值是否有变化
    factor_std = factor_df.stack().std()
    if factor_std < 1e-6 or np.isnan(factor_std):
        logger.error("因子值在所有截面上几乎无变化或全为NaN，无法进行回归。")
        return {'error': '因子值无变化'}

    # --- 1. 数据准备 ---
    logger.info("\t步骤1: 准备和对齐数据...")
    try:
        # 修正：正确计算前向收益率
        forward_returns = price_df.shift(-forward_returns_period) / price_df - 1

        all_dfs_to_align  = {
            'factor': factor_df,
            'returns': forward_returns
        }
        if weights_df is not None:
            # 将T-1的权重信号加入待对齐字典
            all_dfs_to_align['weights'] = weights_df

        if neutral_factors is not None:
            # 将所有T-1的中性化信号加入待对齐字典
            all_dfs_to_align.update(neutral_factors)

        aligned_dfs = align_dataframes(all_dfs_to_align )

        if not all([not df.empty for df in aligned_dfs.values()]):
            raise ValueError("数据对齐后，一个或多个DataFrame为空。请检查输入数据的重叠部分。")

    except Exception as e:
        logger.error(f"数据准备或对齐失败: {e}")
        return {'error': f'数据准备失败: {e}'}

    aligned_factor = aligned_dfs['factor']
    aligned_returns = aligned_dfs['returns']

    # --- 2. 逐日截面回归 ---
    # logger.info("\t步骤2: 开始逐日截面回归...")
    factor_returns = []
    valid_dates = []
    total_dates_to_run = len(aligned_factor.index)

    for date in aligned_factor.index:
        # a) 准备当天数据
        y_series = aligned_returns.loc[date].rename('returns')
        x_df = pd.DataFrame({'factor': aligned_factor.loc[date]})

        if neutral_factors:
            for name in neutral_factors.keys():
                x_df[name] = aligned_dfs[name].loc[date]

        all_data_for_date = [y_series, x_df]
        if weights_df is not None:
            weights_series = np.sqrt(aligned_dfs['weights'].loc[date]).rename('weights')
            all_data_for_date.append(weights_series)

        # b) 清洗与验证
        combined_df = pd.concat(all_data_for_date, axis=1)
        regression_sample = combined_df.replace([np.inf, -np.inf], np.nan).dropna()

        min_samples_needed = x_df.shape[1] + 2
        if len(regression_sample) < min_samples_needed:
            continue

        # c) 执行模型
        try:
            y_final = regression_sample['returns']
            X_final = sm.add_constant(regression_sample[x_df.columns])

            if weights_df is not None:
                w_final = regression_sample['weights']
                if (w_final <= 0).any() or w_final.isna().any():
                    continue
                model = sm.WLS(y_final, X_final, weights=w_final).fit()
            else:
                model = sm.OLS(y_final, X_final).fit()

            if 'factor' not in model.params.index:
                continue

            factor_return = model.params['factor']
            if np.isnan(factor_return) or np.isinf(factor_return):
                continue

            factor_returns.append(factor_return)
            valid_dates.append(date)

        except (np.linalg.LinAlgError, ValueError, KeyError):
            continue
        except Exception as e:
            raise ValueError(f"日期 {date} 回归失败: {e}")


    # --- 3. 分析与报告 ---
    # logger.info("\t步骤3: 分析回归结果并生成报告...")
    num_success_dates = len(factor_returns)
    num_skipped_dates = total_dates_to_run - num_success_dates

    if num_success_dates < 20:
        raise ValueError(f"有效回归期数({num_success_dates})过少，无法进行可靠的统计检验。")

    factor_returns_series = pd.Series(factor_returns, index=pd.to_datetime(valid_dates), name='factor_returns')
    mean_factor_return = factor_returns_series.mean()

    # 修正：正确实现Newey-West t检验
    t_stat, p_value = np.nan, np.nan
    try:
        series_clean = factor_returns_series.dropna()
        n_obs = len(series_clean)

        # 构造回归：因子收益率 = 常数项 + 误差项
        # 检验常数项（即均值）是否显著不为0
        X_const = np.ones(n_obs).reshape(-1, 1)

        max_lags = min(int(n_obs ** 0.25), n_obs // 4)  # 防止lag过大

        nw_model = sm.OLS(series_clean, X_const).fit(
            cov_type='HAC',
            cov_kwds={'maxlags': max_lags, 'use_correction': True}
        )

        t_stat = nw_model.tvalues[0]
        p_value = nw_model.pvalues[0]
        # logger.info(f"已使用Newey-West(lags={max_lags})修正t检验。")

    except Exception as e:
        logger.warning(f"Newey-West t检验计算失败: {e}。回退到标准t检验。")
        try:
            series_clean = factor_returns_series.dropna()
            t_stat, p_value = ttest_1samp(series_clean, 0)
        except Exception as e2:
            raise ValueError(f"标准t检验也失败: {e2}")

    # 显著性判断
    is_significant = abs(t_stat) > 1.96 if not np.isnan(t_stat) else False

    significance_level = ''
    significance_desc = '无法计算'

    if not np.isnan(t_stat):
        # 添加显著性评级
        if abs(t_stat) > 2.58:
            significance_level = "⭐⭐⭐"
            significance_desc = "1%显著"
        elif abs(t_stat) > 1.96:
            significance_level = "⭐⭐"
            significance_desc = "5%显著"
        elif abs(t_stat) > 1.64:
            significance_level = "⭐"
            significance_desc = "10%显著"
        else:
            significance_level = ""
            significance_desc = "不显著"

    logger.info(
        f"\t\t回归完成。总天数: {total_dates_to_run}, 成功回归天数: {num_success_dates} \t 因子平均收益率: {mean_factor_return:.6f}, t统计量: {t_stat:.4f}, 显著性: {significance_desc}")

    if is_significant:
        logger.info("\t\t结论: ✓ 因子有效性得到验证！")
    else:
        logger.info("\t\t结论: ✗ 无法在统计上拒绝因子无效的原假设。")
    results_summary = {
        'mean_factor_return': mean_factor_return,
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'significance_level': significance_level,
        'significance_desc': significance_desc,
        'num_total_periods': total_dates_to_run,
        'num_valid_periods': num_success_dates,
        'success_rate': num_success_dates / total_dates_to_run if total_dates_to_run > 0 else 0,#有多大比例的交易日成功地完成了回归
        # 'factor_returns_series': factor_returns_series,
        'skipped_dates': num_skipped_dates,
    }

    return factor_returns_series,results_summary
def fama_macbeth(
                      factor_data: pd.DataFrame,
                      close_df: pd.DataFrame,
                      neutral_dfs: Dict[str, pd.DataFrame],
                      forward_periods,
                      circ_mv_df: pd.DataFrame,
                      factor_name: str) -> Tuple[Dict[str, pd.DataFrame],Dict[str, pd.DataFrame]]:
    """
    Fama-MacBeth回归法测试（黄金标准）

    Args:
        factor_data: 预处理后的因子数据
        factor_name: 因子名称

    Returns:
        Fama-MacBeth回归结果字典
    """
    fm_stat_results_periods_dict = {}
    factor_returns_series_periods_dict  = {}
    for period in forward_periods:
        # 运行Fama-MacBeth回归
        factor_returns_series,fm_result = fama_macbeth_regression(
            factor_df=factor_data,
            price_df=close_df,
            forward_returns_period=period,
            weights_df=circ_mv_df,  # <-- 传入 流通市值作为权重，执行WLS
            neutral_factors=neutral_dfs  # <-- 传入市值和行业作为控制变量
        )
        fm_stat_results_periods_dict[f'{period}d'] = fm_result
        factor_returns_series_periods_dict[f'{period}d'] = factor_returns_series
    return factor_returns_series_periods_dict,fm_stat_results_periods_dict