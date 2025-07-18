import vectorbt as vbt

from quant_lib.utils.dataFrame_utils import align_dataframes

"""
评估模块

提供因子评价和策略评估功能，包括IC分析、分层回测、业绩归因等。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
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

# 获取模块级别的logger
logger = logging.getLogger(__name__)


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


def calculate_ic_vectorized(factor_df: pd.DataFrame,
                            forward_returns: pd.DataFrame,
                            method: str = 'spearman',
                            min_stocks: int = 10) -> pd.Series:
    """
    向量化计算因子IC值，效率更高
    
    Args:
        factor_df: 因子值DataFrame，index为日期，columns为股票代码
        forward_returns: 未来收益率DataFrame，index为日期，columns为股票代码
        method: 相关系数计算方法，'pearson'或'spearman'
        min_stocks: 每个日期至少需要的股票数量
        
    Returns:
        IC值序列，index为日期
    """
    logger.info(f"向量化计算{method}类型IC...")

    # 确保两个DataFrame的索引对齐
    common_dates = factor_df.index.intersection(forward_returns.index)
    factor_df = factor_df.loc[common_dates]
    forward_returns = forward_returns.loc[common_dates]

    # 计算每个日期的有效股票数量
    valid_counts = pd.DataFrame({
        'factor': factor_df.notna().sum(axis=1),
        'returns': forward_returns.notna().sum(axis=1)
    })

    # 筛选出有足够股票的日期
    valid_dates = valid_counts[(valid_counts['factor'] >= min_stocks) &
                               (valid_counts['returns'] >= min_stocks)].index

    # 使用corrwith向量化计算IC
    ic_series = factor_df.loc[valid_dates].corrwith(
        forward_returns.loc[valid_dates],
        axis=1,
        method=method.lower()#这是一种确保 corrwith 函数能正确接收大小写不敏感的计算方法指令的健壮性设计。
    )

    logger.info(f"IC计算完成: 均值={ic_series.mean():.4f}, IR={ic_series.mean() / ic_series.std():.4f}")
    return ic_series


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


def calculate_quantile_returns(factor_df: pd.DataFrame,
                               price_df: pd.DataFrame,
                               n_quantiles: int = 5,
                               forward_periods: List[int] = [1, 5, 20]) -> Dict[int, pd.DataFrame]:
    """
    计算分位数收益
    
    Args:
        factor_df: 因子值DataFrame
        price_df: 价格DataFrame
        n_quantiles: 分位数数量
        forward_periods: 未来时间周期列表
        
    Returns:
        不同时间周期的分位数收益字典
    """
    logger.info(f"计算{n_quantiles}分位数收益...")

    results = {}

    for period in forward_periods:
        # 计算未来收益率
        forward_returns = price_df.shift(-period) / price_df - 1

        # 初始化结果DataFrame
        quantile_returns = pd.DataFrame(
            index=factor_df.index,
            columns=[f'Q{i + 1}' for i in range(n_quantiles)] + ['TopMinusBottom']
        )

        # 对每个日期进行分组计算
        for date in factor_df.index:
            if date not in forward_returns.index:
                continue

            # 获取当天的因子值和未来收益率
            factor_values = factor_df.loc[date].dropna()
            returns = forward_returns.loc[date].dropna()

            # 找出共同的股票
            common_stocks = factor_values.index.intersection(returns.index)

            if len(common_stocks) < n_quantiles * 5:  # 每组至少需要5只股票
                continue

            # 计算分位数
            factor_quantiles = pd.qcut(
                factor_values[common_stocks],
                n_quantiles,
                labels=[f'Q{i + 1}' for i in range(n_quantiles)],
                duplicates='drop'  # 处理极端情况下的重复边界
            )

            # 计算各分位数的平均收益
            for i in range(n_quantiles):
                quantile_label = f'Q{i + 1}'
                stocks_in_quantile = factor_quantiles[factor_quantiles == quantile_label].index
                if len(stocks_in_quantile) > 0:
                    quantile_returns.loc[date, quantile_label] = returns[stocks_in_quantile].mean()

            # 计算多空组合收益
            if pd.notna(quantile_returns.loc[date, f'Q{n_quantiles}']) and pd.notna(quantile_returns.loc[date, 'Q1']):
                quantile_returns.loc[date, 'TopMinusBottom'] = (
                        quantile_returns.loc[date, f'Q{n_quantiles}'] -
                        quantile_returns.loc[date, 'Q1']
                )

        # 存储结果
        results[period] = quantile_returns

    logger.info("分位数收益计算完成")
    return results


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


def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    计算最大回撤
    
    Args:
        returns: 收益率序列
        
    Returns:
        (最大回撤, 开始日期, 结束日期)
    """
    # 计算累计收益
    cumulative_returns = (1 + returns).cumprod()

    # 计算历史最高点
    running_max = cumulative_returns.cummax()

    # 计算回撤
    drawdown = (cumulative_returns / running_max - 1)

    # 找出最大回撤
    max_drawdown = drawdown.min()
    end_date = drawdown.idxmin()

    # 找出最大回撤的开始日期
    start_date = cumulative_returns[:end_date].idxmax()

    return max_drawdown, start_date, end_date


def run_fama_macbeth_regression(
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        forward_returns_period: int = 20,
        weights_df: pd.DataFrame = None,
        neutral_factors: dict = None
):
    """
    对单个因子进行Fama-MacBeth回归检验（支持WLS和中性化）。
    Fama-MacBeth回归是检验因子有效性的"黄金标准"，通过两步回归法：
    1. 第一步：对每个时间截面进行横截面回归，得到因子收益率序列。
    2. 第二步：对因子收益率序列进行时间序列分析，检验其显著性。

    Args:
        factor_df (pd.DataFrame): 预处理后的待测因子矩阵。
        price_df (pd.DataFrame): 复权收盘价矩阵。
        forward_returns_period (int): 预测未来收益的时间窗口。
        weights_df (pd.DataFrame, optional): 用于WLS回归的权重矩阵 (如流通市值)。默认为None，执行OLS。
        neutral_factors (dict, optional): 用于中性化的因子字典 (如{'mkt_cap': df, 'industry_dummies': df})。

    Returns:
        dict: 包含回归结果的字典。
    """
    print("\n" + "=" * 60)
    print(f"开始执行 Fama-MacBeth 回归分析 (预测周期: {forward_returns_period}天)")

    # 打印本次运行的模式
    if weights_df is not None:
        print("模式: 加权最小二乘 (WLS)")
    else:
        print("模式: 普通最小二乘 (OLS)")
    if neutral_factors:
        print(f"中性化控制变量: {list(neutral_factors.keys())}")
    print("-" * 60)

    # --- 1. 数据准备 ---
    # 计算未来收益率
    forward_returns = price_df.shift(-forward_returns_period) / price_df - 1
    # 【重要】修复前视偏差：使用T-1日的因子值
    factor_df_shifted = factor_df.shift(1)

    # 将所有需要用到的DataFrame放入一个字典，方便统一处理
    all_dfs_dict = {'factor': factor_df_shifted, 'returns': forward_returns}
    if weights_df is not None:
        all_dfs_dict['weights'] = weights_df.shift(1)  # 权重也需要用T-1日的数据
    if neutral_factors is not None:
        for name, df in neutral_factors.items():
            all_dfs_dict[name] = df.shift(1)  # 中性化因子也需要shift

    # 使用vectorbt的align功能可以高效地对齐所有DataFrame
    try:
        aligned_dfs = align_dataframes(all_dfs_dict)

    except Exception as e:
        print(f"警告：数据对齐失败: {e}。请确保vectorbt已安装。")
        # 此处可以添加手动的pandas对齐逻辑作为备用方案
        return {}

    # 提取出对齐后的核心数据
    aligned_factor = aligned_dfs['factor']
    aligned_returns = aligned_dfs['returns']

    # --- 2. 逐日进行截面回归 ---
    factor_returns = []
    valid_dates = []
    # 在函数开始处添加诊断信息
    print(f"=== 因子收益率回归诊断 ===")
    print(f"因子数据形状: {factor_df_shifted.shape}")
    print(f"收益率数据形状: {forward_returns.shape}")
    print(f"因子数据时间范围: {factor_df_shifted.index.min()} 至 {factor_df_shifted.index.max()}")
    print(f"收益率数据时间范围: {forward_returns.index.min()} 至 {forward_returns.index.max()}")
    print(f"因子数据非空比例: {factor_df_shifted.notna().sum().sum() / factor_df_shifted.size:.2%}")
    print(f"收益率数据非空比例: {forward_returns.notna().sum().sum() / forward_returns.size:.2%}")

    # 检查对齐后的数据
    print(f"对齐后因子数据形状: {aligned_factor.shape}")
    print(f"对齐后收益率数据形状: {aligned_returns.shape}")
    print(f"共同日期数量: {len(aligned_factor.index)}")
    # 在循环中添加计数器
    total_dates = 0
    skipped_dates = 0
    failed_dates = 0
    for date in aligned_factor.index:
        try:

            # a) 准备所有可能用到的数据Series
            y = aligned_returns.loc[date].rename('returns')
            X_df = pd.DataFrame({'factor': aligned_factor.loc[date]})
            if neutral_factors is not None:
                for name in neutral_factors.keys():
                    X_df[name] = neutral_factors[name].loc[date]

            # b) 【数据类型检查和转换】
            # 确保y是数值类型
            y = pd.to_numeric(y, errors='raise')

            # 确保X_df所有列都是数值类型
            for col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors='raise')

            # c) 将所有数据（包括权重）一次性放入一个列表
            all_data_list = [y, X_df]
            if weights_df is not None:
                weights = np.sqrt(aligned_dfs['weights'].loc[date]).rename('weights')
                # 确保权重也是数值类型
                weights = pd.to_numeric(weights, errors='coerce')
                weights = weights.where(weights > 0)
                all_data_list.append(weights)

            # d) 将所有数据横向合并
            combined_df = pd.concat(all_data_list, axis=1)

            # e) 进行dropna，并额外检查是否有无穷大值
            final_regression_sample = combined_df.dropna()

            # 【关键】检查并移除无穷大值
            final_regression_sample = final_regression_sample.replace([np.inf, -np.inf], np.nan).dropna()
            # 增加下面的诊断打印
            num_regressors = X_df.shape[1]
            min_samples_needed = num_regressors + 5
            print(f"--- 日期: {date.date()} ---")
            print(f"    最终回归样本数: {len(final_regression_sample)}")
            print(f"    需要的自变量数: {num_regressors}")
            print(f"    需要的最小样本数: {min_samples_needed}")
            # f) 数据质量检查
            if len(final_regression_sample) < X_df.shape[1] + 5:
                print("    >>> 样本数不足，跳过当天回归 <<<")
                continue

            # g) 从"最终样本"中分离出 y, X, w
            y_final = final_regression_sample['returns']
            x_columns = list(X_df.columns)
            X_final = final_regression_sample[x_columns]

            # 【关键】最后一次数据类型确认
            if not np.issubdtype(y_final.dtype, np.number):
                print(f"警告: 日期 {date} y_final数据类型异常: {y_final.dtype}")
                continue

            for col in X_final.columns:
                if not np.issubdtype(X_final[col].dtype, np.number):
                    print(f"警告: 日期 {date} X_final[{col}]数据类型异常: {X_final[col].dtype}")
                    continue

            # h) 执行回归
            if HAS_STATSMODELS:
                X_final = sm.add_constant(X_final)

                if weights_df is not None:
                    w_final = final_regression_sample['weights']
                    if not np.issubdtype(w_final.dtype, np.number):
                        print(f"警告: 日期 {date} 权重数据类型异常: {w_final.dtype}")
                        continue
                    model = sm.WLS(y_final, X_final, weights=w_final).fit()
                else:
                    model = sm.OLS(y_final, X_final).fit()
            else:
                continue

            # i) 提取结果
            factor_return_t = model.params['factor']
            factor_returns.append(factor_return_t)
            valid_dates.append(date)
            # 在各个continue点添加计数和原因
            if len(final_regression_sample) < X_df.shape[1] + 5:
                skipped_dates += 1
                if total_dates <= 5:  # 只打印前5个日期的详细信息
                    print(f"日期 {date}: 样本不足 ({len(final_regression_sample)} < {X_df.shape[1] + 5})")
                continue

        except Exception as e:
            print(f"警告: 日期 {date} 回归失败: {e}")
            # 【调试信息】打印更多详细信息
            try:
                print(f"  调试信息:")
                print(f"    y数据类型: {y.dtype if 'y' in locals() else 'N/A'}")
                if 'X_df' in locals():
                    print(f"    X_df数据类型: {dict(X_df.dtypes)}")
                if 'final_regression_sample' in locals():
                    print(f"    最终样本形状: {final_regression_sample.shape}")
                    print(f"    最终样本数据类型: {dict(final_regression_sample.dtypes)}")
            except:
                pass
            continue

    # --- 3. 分析与报告 ---
    if not factor_returns:
        print("错误：未能计算出任何有效的因子收益率。")
        return {'mean_factor_return': np.nan, 't_statistic': np.nan, 'p_value': np.nan, 'num_periods': 0}

    factor_returns_series = pd.Series(factor_returns, index=valid_dates, name='factor_returns')
    mean_factor_return = factor_returns_series.mean()
    t_stat, p_value = ttest_1samp(factor_returns_series.dropna(), 0)

    print(f"回归期数: {len(factor_returns_series)}")
    print(f"因子平均收益率 (Mean Lambda): {mean_factor_return:.6f}")
    print(f"因子收益率 t值 (t-statistic): {t_stat:.4f}")
    print(f"因子收益率 p值 (p-value)    : {p_value:.4f}")

    if abs(t_stat) > 2:
        print("结论: ✓ 因子有效性得到验证！")
    else:
        print("结论: ✗ 无法在统计上拒绝“因子无效”的原假设。")
    print("=" * 60)

    return {
        'mean_factor_return': mean_factor_return,
        't_statistic': t_stat,
        'p_value': p_value,
        'num_periods': len(factor_returns_series),
        'factor_returns_series': factor_returns_series
    }
