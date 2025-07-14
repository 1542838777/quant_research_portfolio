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
    
    logger.info(f"IC计算完成: 均值={ic_series.mean():.4f}, IR={ic_series.mean()/ic_series.std():.4f}")
    return ic_series


def calculate_ic_vectorized(factor_df: pd.DataFrame, 
                          forward_returns: pd.DataFrame, 
                          method: str = 'pearson',
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
        method=method.lower()
    )
    
    logger.info(f"IC计算完成: 均值={ic_series.mean():.4f}, IR={ic_series.mean()/ic_series.std():.4f}")
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
            columns=[f'Q{i+1}' for i in range(n_quantiles)] + ['TopMinusBottom']
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
                labels=[f'Q{i+1}' for i in range(n_quantiles)],
                duplicates='drop'  # 处理极端情况下的重复边界
            )
            
            # 计算各分位数的平均收益
            for i in range(n_quantiles):
                quantile_label = f'Q{i+1}'
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
        prev_pos = positions_df.iloc[i-1]
        curr_pos = positions_df.iloc[i]
        
        # 计算持仓变化
        pos_change = abs(curr_pos - prev_pos).sum() / 2
        
        turnover.iloc[i-1] = pos_change
    
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


def run_fama_macbeth_regression(factor_df: pd.DataFrame,
                               price_df: pd.DataFrame,
                               forward_returns_period: int = 20) -> Dict[str, float]:
    """
    对单个因子进行Fama-MacBeth回归检验。

    Fama-MacBeth回归是检验因子有效性的"黄金标准"，通过两步回归法：
    1. 第一步：对每个时间截面进行横截面回归，得到因子收益率序列
    2. 第二步：对因子收益率序列进行时间序列分析，检验其显著性

    Args:
        factor_df (pd.DataFrame): 经过预处理的因子值矩阵 (index=date, columns=stock)
        price_df (pd.DataFrame): 复权收盘价矩阵 (index=date, columns=stock)
        forward_returns_period (int): 预测未来收益的时间窗口，默认20天

    Returns:
        Dict[str, float]: 包含回归结果的字典，包括：
            - mean_factor_return: 因子平均收益率
            - t_statistic: t统计量
            - p_value: p值
            - num_periods: 有效回归期数
            - is_significant: 是否显著（|t| > 2）
    """
    logger.info(f"开始执行 Fama-MacBeth 回归分析 (预测周期: {forward_returns_period}天)")

    # --- 1. 数据准备 ---
    # 计算未来收益率
    forward_returns = price_df.shift(-forward_returns_period) / price_df - 1

    # 【重要】修复前视偏差：使用T-1日的因子值预测T+N日的收益
    factor_df_shifted = factor_df.shift(1)

    # 对齐数据，确保时间和股票维度都匹配
    aligned_factor, aligned_returns = factor_df_shifted.align(forward_returns, join='inner', axis=0)
    aligned_factor, aligned_returns = aligned_factor.align(aligned_returns, join='inner', axis=1)

    # --- 2. 逐日进行截面回归 ---
    factor_returns = []  # 用于存储每一期的因子收益率 (lambda_t)
    valid_dates = []     # 记录有效的回归日期

    # 遍历每一个交易日
    for date in aligned_factor.index:
        # 提取当天的截面数据
        y = aligned_returns.loc[date].dropna()  # 因变量：未来收益
        X = aligned_factor.loc[date].dropna()   # 自变量：因子值

        # 确保 y 和 X 的样本能对上
        common_stocks = y.index.intersection(X.index)
        y = y[common_stocks]
        X = X[common_stocks]

        # 数据质量检查
        if len(X) < 10:  # 至少需要10只股票才能进行有效回归
            continue

        if X.std() == 0:  # 因子值没有变化，无法回归
            continue

        try:
            if HAS_STATSMODELS:
                # 使用statsmodels进行OLS回归
                X_with_const = sm.add_constant(X)
                model = sm.OLS(y, X_with_const).fit()
                factor_return_t = model.params.iloc[1]
            else:
                # 使用numpy进行简单线性回归
                # 手动添加常数项
                X_matrix = np.column_stack([np.ones(len(X)), X.values])
                y_values = y.values

                # 使用最小二乘法: beta = (X'X)^(-1)X'y
                try:
                    beta = np.linalg.solve(X_matrix.T @ X_matrix, X_matrix.T @ y_values)
                    factor_return_t = beta[1]  # beta[0]是截距，beta[1]是因子系数
                except np.linalg.LinAlgError:
                    # 如果矩阵奇异，使用伪逆
                    beta = np.linalg.pinv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_values
                    factor_return_t = beta[1]

            factor_returns.append(factor_return_t)
            valid_dates.append(date)

        except Exception as e:
            logger.warning(f"日期 {date} 回归失败: {e}")
            continue

    if len(factor_returns) == 0:
        logger.error("未能计算出任何有效的因子收益率")
        return {
            'mean_factor_return': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'num_periods': 0,
            'is_significant': False
        }

    # --- 3. 分析因子收益率的时间序列 ---
    factor_returns_series = pd.Series(factor_returns, index=valid_dates)

    # 计算均值
    mean_factor_return = factor_returns_series.mean()

    # 对因子收益率序列进行t检验，判断其均值是否显著不为0
    # H0: E[lambda_t] = 0 (因子无效)
    # H1: E[lambda_t] != 0 (因子有效)
    t_stat, p_value = ttest_1samp(factor_returns_series.dropna(), 0)

    # 判断显著性（通常使用|t| > 2作为显著性标准）
    is_significant = abs(t_stat) > 2

    # --- 4. 输出报告 ---
    print("\n" + "="*60)
    print("Fama-MacBeth 回归分析结果")
    print("-"*60)
    print(f"回归期数: {len(factor_returns_series)}")
    print(f"因子平均收益率 (Mean Lambda): {mean_factor_return:.6f}")
    print(f"因子收益率标准差: {factor_returns_series.std():.6f}")
    print(f"因子收益率 t值 (t-statistic): {t_stat:.4f}")
    print(f"因子收益率 p值 (p-value): {p_value:.4f}")
    print(f"显著性水平: {'***' if abs(t_stat) > 2.58 else '**' if abs(t_stat) > 1.96 else '*' if abs(t_stat) > 1.64 else '不显著'}")

    if is_significant:
        print("结论: ✓ t值绝对值大于2，因子收益率在统计上显著不为0，因子有效性得到验证！")
    else:
        print("结论: ✗ t值绝对值小于2，无法在统计上拒绝'因子无效'的原假设。")

    print("="*60)

    # 返回结果字典
    results = {
        'mean_factor_return': mean_factor_return,
        't_statistic': t_stat,
        'p_value': p_value,
        'num_periods': len(factor_returns_series),
        'is_significant': is_significant,
        'factor_returns_series': factor_returns_series  # 可用于进一步分析
    }

    logger.info(f"Fama-MacBeth回归完成: 均值={mean_factor_return:.6f}, t值={t_stat:.4f}")

    return results