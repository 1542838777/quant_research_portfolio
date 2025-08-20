import vectorbt as vbt
from pandas import Series
from scipy.stats._mstats_basic import winsorize

from quant_lib import logger
from quant_lib.config.logger_config import log_warning
from quant_lib.utils.dataFrame_utils import align_dataframes

"""
评估模块

提供因子评价和策略评估功能，包括IC分析、分层回测、业绩归因等。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp, spearmanr

# 尝试导入statsmodels，如果没有安装则使用简化版本
try:
    import statsmodels.api as sm

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("警告: statsmodels未安装，将使用简化版本的回归分析")


# 我觉得这个更能说明因子的潜力，在运动过程中（真的过程（交易过程）中， 来看因子 跟此段收益率的协同关系
def calcu_forward_returns_open_close(period: int,
                                     close_df: pd.DataFrame,
                                     open_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算从T日开盘价到T+period-1日收盘价的未来收益率 (Open-to-Close)。
    这是一种更贴近实盘的、更严格的收益计算方式。

    Args:
        period (int): 持有周期。
        close_df (pd.DataFrame): 收盘价矩阵。
        open_df (pd.DataFrame): 开盘价矩阵。

    Returns:
        pd.DataFrame: O2C未来收益率矩阵。
    """
    # 1. 定义起点和终点价格
    # 起点是 T 日的开盘价，它本身不需要 shift
    close_df = close_df.copy(deep=True)
    open_df = open_df.copy(deep=True)
    start_price = open_df
    # 终点是 T+period-1 日的收盘价
    end_price = close_df.shift(-(period - 1))

    # 2. 创建“未来存续”掩码
    survived_mask = start_price.notna() & end_price.notna()

    # 3. 计算原始收益率，并应用掩码过滤
    forward_returns_raw = end_price / start_price - 1
    forward_returns = forward_returns_raw.where(survived_mask)
    winsorized_returns = forward_returns_raw.apply(
        lambda x: winsorize(x.dropna(), limits=[0.025, 0.025]),
        axis=1  # 沿行操作，即对每个时间截面
    )

    return winsorized_returns


# ok
##
#
#
# 你的策略是在 T-1日收盘后 做出决策。
#
# 真实的交易执行，最早只能在 T日开盘时 发生。
#
# 所以，一个基于 T-1 决策的收益，它的衡量起点也必须是 T日。
#
# 如果用这个代码，却匹配了一个从 T-1日 就已经开始计算的收益！ ，在T-1日决策的瞬间，就已经“偷看”到了T-1日到未来的收益，它没有模拟“持有”这个真实世界中的、需要时间流逝的过程。
#
# 所以，这个对齐方式虽然看起来 T-1 对上了 T-1，但它违反了真实世界“决策”与“执行”之间的时间差，是一个隐蔽的未来函数。#
##
# 太坑了，！！！！害我排查两天！这个经常搞出极度异常的单调性！（尤其是volatility相关因子！，而切换成o2c 骤降！瞬间恢复正常
# #
# def calcu_forward_returns_close_close(period, price_df):
#     # 1. 定义起点和终点价格 (严格遵循T-1原则)
#     start_price = price_df.shift(1)
#     end_price = price_df.shift(1 - period)
#
#     # 2. 创建“未来存续”掩码 (确保在持有期首尾股价都存在)
#     survived_mask = start_price.notna() & end_price.notna()
#
#     # 3. 计算原始收益率，并应用掩码过滤
#     forward_returns_raw = end_price / start_price - 1
#     forward_returns = forward_returns_raw.where(survived_mask)
#
#     # clip 操作应该在所有计算和过滤完成后进行
#     return forward_returns.clip(-0.15, 0.15)
# if __name__ == '__main__':
    # # 1. 构造一个简单的价格DataFrame
    # price_data = {'price': [100, 110, 121, 133.1, 146.41]}  # 每天上涨10%
    # dates = pd.to_datetime(['2025-08-01', '2025-08-02', '2025-08-03', '2025-08-04', '2025-08-05'])
    # price_df = pd.DataFrame(price_data, index=dates)
    #
    # # 2. 构造一个简单的因子DataFrame
    # factor_data = {'factor': [1, 2, 3, 4, 5]}  # 因子值每天递增
    # factor_df = pd.DataFrame(factor_data, index=dates)
    # # 计算2日收益率
    # buggy_returns = calcu_forward_returns_close_close(2,price_df)
    #
    # print("--- 你的函数计算出的收益率 ---")
    # print(buggy_returns)
    #
    # # 模拟你的主函数中的合并步骤
    # print("\n--- 模拟合并因子和收益 ---")
    # merged_df = pd.concat([factor_df, buggy_returns.rename(columns={'price': 'return'})], axis=1)
    # print(merged_df)
    # # 2. 构造一个简单的因子DataFrame
    # factor_data = {'factor': [1, 2, 3, 4, 5]}  # 因子值每天递增
    # factor_df = pd.DataFrame(factor_data, index=dates)

# ok
def calculate_ic(
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        forward_periods: List[int] = [1, 5, 20],
        method: str = 'spearman',
        returns_calculator: Callable[[int, pd.DataFrame], pd.DataFrame] = calcu_forward_returns_open_close,
        min_stocks: int = 20
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
        forward_returns = returns_calculator(period=period) #

        common_idx = factor_df.index.intersection(forward_returns.index)
        common_cols = factor_df.columns.intersection(forward_returns.columns)

        if len(common_idx) == 0 or len(common_cols) == 0:
            raise ValueError("因子数据和收益数据没有重叠的日期或股票，无法计算IC。")

        factor_aligned = factor_df.loc[common_idx, common_cols]
        returns_aligned = forward_returns.loc[common_idx, common_cols]

        # --- 2. 计算有效配对数并筛选日期 ---
        paired_valid_counts = (factor_aligned.notna() & returns_aligned.notna()).sum(axis=1)
        valid_dates = paired_valid_counts[paired_valid_counts >= min_stocks].index
        logger.info(f"calculate_ic 满足最小股票数量要求的日期数量:{len(valid_dates)}")

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
                       returns_calculator,
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
        forward_returns = returns_calculator(period = period)
        # 计算IC
        ic,_ = calculate_ic(factor_df, forward_returns, method)

        # 存储结果
        results['IC_Mean'].append(ic.mean())
        results['IC_IR'].append(ic.mean() / ic.std() if ic.std() > 0 else 0)

    # 创建结果DataFrame
    result_df = pd.DataFrame(results).set_index('period')

    return result_df


# ok
def quantile_stats_result(results: Dict[int, pd.DataFrame], n_quantiles: int) -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
   【V2 升级版】计算并汇总分层回测的关键性能指标。
   本版本使用 Spearman 秩相关系数来衡量单调性，提供一个连续的评分而不是二元的True/False。
   """

    quantile_stats_periods_dict = {}
    quantile_returns_periods_dict = {}

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

        #  计算年化收益率  假设数据的基础频率是每日
        tmb_annual_return = (tmb_mean_period_return / period) * 252 if period > 0 else 0

        # 修正夏普比率以适应不同持有期 夏普比率的年化因子是 sqrt(252 / period)
        tmb_sharpe = (tmb_mean_period_return / tmb_std_period_return) * np.sqrt(
            252 / period) if tmb_std_period_return > 0 and period > 0 else 0
        tmb_win_rate = (tmb_series > 0).mean()
        max_drawdown, mdd_start, mdd_end = calculate_max_drawdown_robust(tmb_series)

        # --- 【单调性检验 ---
        quantile_means = [mean_returns.get(f'Q{i + 1}', np.nan) for i in range(n_quantiles)]
        monotonicity_spearman = np.nan  # 默认为NaN

        # 只有在所有组的平均收益都有效时才计算
        if not any(np.isnan(q) for q in quantile_means):
            # 计算组别序号 [1, 2, 3, 4, 5] 与 组别平均收益 的等级相关性
            # spearmanr 返回 (相关系数, p值)，我们只需要相关系数
            monotonicity_spearman, p_value = spearmanr(np.arange(1, n_quantiles + 1), quantile_means)

            # 【新增】异常单调性检测
            if abs(monotonicity_spearman) >= 0.9:
                logger.warning(f"⚠️  检测到异常单调性: {monotonicity_spearman:.6f} (p={p_value:.6f})")
                logger.warning(f"   分位数收益: {[f'{q:.6f}' for q in quantile_means]}")

                # 检查是否所有收益都相同
                unique_returns = len(set(quantile_means))
                if unique_returns <= 2:
                    logger.error(f"❌ 严重问题：只有{unique_returns}个不同的分位数收益！")

                # 检查收益差异是否过大
                return_range = max(quantile_means) - min(quantile_means)
                if return_range > 0.05:  # 5%
                    logger.warning(f"   收益差异过大: {return_range:.4f}")

            elif abs(monotonicity_spearman) > 0.8:
                logger.info(f"✅ 强单调性: {monotonicity_spearman:.3f} - 这可能是高质量因子的表现")

        # --- 存储结果 ---
        # 'period' 变量用于创建描述性的键，如 '5d'
        quantile_returns_periods_dict[f'{period}d'] = result
        quantile_stats_periods_dict[f'{period}d'] = {
            # 'returns_data': result,
            'mean_returns': mean_returns,
            'tmb_mean_period_return': tmb_mean_period_return,  # 特定周期的平均收益 (例如，5日平均收益)
            'tmb_annual_return': tmb_annual_return,  # 年化后的多空组合收益率
            'tmb_sharpe': tmb_sharpe,  # * 周期调整后的夏普比率
            'tmb_win_rate': tmb_win_rate,
            'tmb_max_drawdown': max_drawdown,
            'mdd_start_date': mdd_start,  # 最大回撤开始日期
            'mdd_end_date': mdd_end,  # 最大回撤结束日期
            'quantile_means': quantile_means,
            # 【新增指标】: 用连续的Spearman相关系数代替二元的True/False
            'monotonicity_spearman': monotonicity_spearman
        }

    return quantile_returns_periods_dict, quantile_stats_periods_dict


def calculate_quantile_returns(
        factor_df: pd.DataFrame,
        returns_calculator: Callable,
        price_df: pd.DataFrame,
        n_quantiles: int = 5,
        forward_periods: List[int] = [1, 5, 20]
) -> Dict[int, pd.DataFrame]:
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
    #### todo 移除打点代码
    factor_df.to_csv('D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\tests\\workspace\\mem_volatility.csv')
    return_df= returns_calculator(period=3)
    return_df.to_csv('D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\tests\\workspace\\mem_forward_return_o2c.csv')
    ###
    # factor_df = pd.read_csv('D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\tests\\workspace\\local_volatility.csv', index_col=[0], parse_dates=True)

    results = {}
    for period in forward_periods:
        logger.info(f"  > 正在处理向前看 {period} 周期...")

        # 1. 计算未来收益率
        forward_returns = returns_calculator(period=period)

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

        if merged_df.empty:
            log_warning(
                f"  > 在周期 {period}，因子和收益数据没有重叠，无法计算。")  # 考虑 要不要直接报错 不能，因为forward_returns有nan很正常
            # 创建一个空的DataFrame以保持输出结构一致性
            empty_cols = [f'Q{i + 1}' for i in range(n_quantiles)] + ['TopMinusBottom']
            results[period] = pd.DataFrame(columns=empty_cols, dtype='float64')
            continue

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
        # 5. 计算各分位数的平均收益 （时间+组别 为一个group。进行求收益率平均）
        daily_quantile_returns = merged_df.groupby([merged_df.index.get_level_values(0), 'quantile'])['return'].mean()

        # 6. 数据转换：从“长表”恢复到“宽表”
        quantile_returns_wide = daily_quantile_returns.unstack()
        # 假设当天某个分组的所有股票都因为  未来不存续 而收益为NaN， （不存续：比如我们周期5，今天买的，第五天因为停牌卖不出去，导致无法拿到价格 导致returns 为nan） 你要不是不把这个收益率替换成0，后面再算累计收益的时候 会报错！
        # 那么该分组的平均收益也是NaN。我们假设这种情况下组合当天收益为0。
        quantile_returns_wide= quantile_returns_wide.fillna(0, inplace=False)
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
    return  results

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


# ok
def calculate_turnover(
        factor_df: pd.DataFrame,
        n_quantiles: int = 5,
        forward_periods: List[int] = [1, 5, 20]
) -> Dict[str, pd.Series]:
    """
    【新增】向量化计算因子在不同持有期下的换手率。

    Args:
        factor_df (pd.DataFrame): 因子值DataFrame (index=date, columns=stock)。
        n_quantiles (int): 分位数数量。
        forward_periods (List[int]): 调仓周期列表。

    Returns:
        Dict[str, pd.Series]: 字典，key为周期(如 '5d')，value为换手率的时间序列。
    """
    turnover_periods_dict = {}
    # 核心：计算每日的分位数归属
    # 使用rank(pct=True)比qcut更稳健，能直接得到百分位排名
    quantiles = factor_df.rank(axis=1, pct=True, method='first')
    ###假想满分1，
    # 因子rank值：0.83：表示我在0.83这个水位
    # 如果分5(n_quantiles)个桶（每个桶那就是1/n_quantiles = 0.2
    # 0.83是排在第几个桶。0.83/0.2 =4.01个 超过第4个桶，ceil 是5
    # np.ceil(quantiles * n_quantiles 这个就是等于 0.83/0.2->0.83/(1/5) ->0.83 * 5
    # #
    for period in forward_periods:
        # 将分位数矩阵向前移动`period`天
        quantiles_shifted = quantiles.shift(period)

        # 计算两个周期之间，股票的分位数变化了多少
        # 如果股票保持在同一个分位数内，变化为0，否则为1
        # 注意：这里我们比较的是分位数的“档位”，而不是具体的百分比值
        # (quantiles * n_quantiles).ceil() 会得到 1, 2, 3, 4, 5 的整数分位
        turnover_matrix = np.ceil(quantiles * n_quantiles) != np.ceil(quantiles_shifted * n_quantiles)

        # 每日换手率 = 发生变动的股票数 / 当天有效股票总数
        valid_counts = factor_df.notna().sum(axis=1)
        daily_turnover = turnover_matrix.sum(axis=1) / valid_counts.where(valid_counts > 0, np.nan)

        turnover_periods_dict[f'{period}d'] = daily_turnover.dropna().rename('turnover')

    return turnover_periods_dict


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


def fama_macbeth_regression(
        factor_df: pd.DataFrame, # <-- 接收原始 T 日因子
        returns_calculator: Callable,
        price_df: pd.DataFrame,
        forward_returns_period: int = 20,
        circ_mv_df_shifted: pd.DataFrame = None, # <-- 接收原始 T 日权重
        neutral_factors: Dict[str, pd.DataFrame] = None # 因为factor_df以及除杂过，现在不需要再次进行除杂了
) -> Tuple[Series,Series, Dict[str, Any]]:
    """
    【最终生产版】对单个因子进行Fama-MacBeth回归检验。
    此版本逻辑结构清晰，代码健壮，并使用Newey-West标准误修正t检验，符合学术界和业界的严格标准。
    return:Series 表示纯因子带来的收益，纯收益
    """
    # # 初始化logger
    # from quant_lib.config.logger_config import setup_logger
    # logger = setup_logger(__name__)
    logger.info(f"开始Fama-MacBeth回归分析 (前向收益期: {forward_returns_period}天)")

    # --- 0. 前置检查 ---
    if not HAS_STATSMODELS:
        raise ValueError("statsmodels未安装，无法执行Fama-MacBeth回归")
    if factor_df.empty:
        raise ValueError("输入的因子数据为空")
    factor_std = factor_df.stack().std()
    if factor_std < 1e-6 or np.isnan(factor_std):
        raise ValueError("因子值在所有截面上几乎无变化或全为NaN，无法进行回归。")
    # 【新增检查】如果传入了不应有的中性化因子，发出警告
    if neutral_factors:
        raise ValueError(
            "警告：已向本函数传入了预处理后的因子，但 neutral_factors 参数不为空。回归将继续，但请确认这是否是预期行为。")

    # --- 1. 数据准备 (已简化) ---
    logger.info("\t步骤1: 准备和对齐数据...")
    try:
        # 步骤A: 计算目标结果 (Y变量)
        forward_returns = returns_calculator(period=forward_returns_period)

        # 步骤B: 直接构建对齐字典。由于 neutral_factors 为空，流程大大简化。
        all_dfs_to_align = {
            'factor': factor_df,
            'returns': forward_returns
        }
        if circ_mv_df_shifted is not None:
            all_dfs_to_align['weights'] = circ_mv_df_shifted

        aligned_dfs = align_dataframes(all_dfs_to_align)

        # 步骤C: 从对齐结果中分离出 Y 和 X 的“原材料”
        aligned_returns = aligned_dfs['returns']
        aligned_factor = aligned_dfs['factor']
        aligned_weights = aligned_dfs.get('weights')  # 使用 .get() 安全获取


    except Exception as e:
        logger.error(f"数据准备或对齐失败: {e}")
        return pd.Series(dtype=float), {'error': f'数据准备失败: {e}'}

    # --- 2. 逐日截面回归 ---
    factor_returns = []
    factor_t_stats = [] # <--- 【新增】用于存储每日t值的列表
    valid_dates = []
    total_dates_to_run = len(aligned_factor.index)

    for date in aligned_factor.index:
        # a) 准备当天数据
        y_series = aligned_returns.loc[date].rename('returns')

        # 【简化】X变量的构建变得非常简单，只包含目标因子
        x_df = pd.DataFrame({'factor': aligned_factor.loc[date]})

        all_data_for_date = [y_series, x_df]
        if aligned_weights is not None:
            weights_series = np.sqrt(aligned_weights.loc[date]).rename('weights')
            all_data_for_date.append(weights_series)

        # b) 有效域掩码：显式定义当日有效样本
        # 先合并所有数据
        combined_df = pd.concat(all_data_for_date, axis=1, join='outer')
        # 显式定义有效掩码：所有变量都不为NaN
        valid_mask = combined_df.notna().all(axis=1)
        # 应用掩码
        combined_df = combined_df[valid_mask]

        # 样本量检查现在更简单
        if len(combined_df) < 10:  # 对于单变量回归，可以设置一个较小的绝对值门槛
            continue

        # c) 执行模型
        try:
            y_final = combined_df['returns']
            X_final = sm.add_constant(combined_df[['factor']])  # 只对因子列回归

            if aligned_weights is not None:
                w_final = combined_df['weights']
                if (w_final <= 0).any() or w_final.isna().any():
                    continue
                model = sm.WLS(y_final, X_final, weights=w_final).fit()
            else:
                model = sm.OLS(y_final, X_final).fit()

            if 'factor' not in model.params.index:
                continue

            factor_return = model.params['factor']
            # 【新增】提取因子对应的t值
            t_stat_daily = model.tvalues['factor']
            if np.isnan(factor_return) or np.isinf(factor_return):
                continue

            factor_returns.append(factor_return)
            factor_t_stats.append(t_stat_daily) # <--- 【新增】存入每日t值
            valid_dates.append(date)
        except (np.linalg.LinAlgError, ValueError):
            raise ValueError("失败")

    # --- 3. 分析与报告 ---
    # logger.info("\t步骤3: 分析回归结果并生成报告...")
    num_success_dates = len(factor_returns)
    num_skipped_dates = total_dates_to_run - num_success_dates

    if num_success_dates < 20:
        raise ValueError(f"有效回归期数({num_success_dates})过少，无法进行可靠的统计检验。")

    # --- 计算“t值绝对值均值” ---
    fm_t_stats_series = pd.Series(factor_t_stats, index=pd.to_datetime(valid_dates),
                                   name='factor_t_stats')

    mean_abs_t_stat = fm_t_stats_series.abs().mean()
    fm_returns_series = pd.Series(factor_returns, index=pd.to_datetime(valid_dates), name='factor_returns')
    mean_factor_return = fm_returns_series.mean()

    # 修正：正确实现Newey-West t检验
    t_stat, p_value = np.nan, np.nan
    try:
        series_clean = fm_returns_series.dropna()
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
        log_warning(f"Newey-West t检验计算失败: {e}。回退到标准t检验。")
        try:
            series_clean = fm_returns_series.dropna()
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
    fm_summary = {
        'mean_factor_return': mean_factor_return,
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_abs_t_stat': mean_abs_t_stat,  # <--- 【新增】存入我们新计算的指标
        'is_significant': is_significant,
        'significance_level': significance_level,
        'significance_desc': significance_desc,
        'num_total_periods': total_dates_to_run,
        'num_valid_periods': num_success_dates,
        'success_rate': num_success_dates / total_dates_to_run if total_dates_to_run > 0 else 0,  # 有多大比例的交易日成功地完成了回归
        # 'fm_returns_series': fm_returns_series,
        'skipped_dates': num_skipped_dates,
    }

    return fm_returns_series,fm_t_stats_series, fm_summary


def fama_macbeth(
        factor_data: pd.DataFrame,#以经shift处理过
        returns_calculator,
        close_df: pd.DataFrame,
        neutral_dfs: Dict[str, pd.DataFrame], #以经shift处理过
        forward_periods,
        circ_mv_df_shifted: pd.DataFrame,
        factor_name: str) -> Tuple[Dict[str, pd.DataFrame],Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Fama-MacBeth回归法测试（黄金标准）

    Args:
        factor_data: 预处理后的因子数据
        factor_name: 因子名称

    Returns:
        Fama-MacBeth回归结果字典
    """
    fm_summary_dict = {}
    fm_t_stats_series_dict = {}
    fm_returns_series_dict = {}
    for period in forward_periods:
        # 运行Fama-MacBeth回归
        fm_returns_series, fm_t_stats_series, fm_summary= fama_macbeth_regression(
            factor_df=factor_data,
            returns_calculator=returns_calculator,
            price_df=close_df,
            forward_returns_period=period,
            circ_mv_df_shifted=circ_mv_df_shifted,  # <-- 传入 流通市值作为权重，执行WLS
            neutral_factors=neutral_dfs  # <-- 传入市值和行业作为控制变量
        )
        fm_returns_series_dict[f'{period}d'] = fm_returns_series
        fm_t_stats_series_dict[f'{period}d'] = fm_t_stats_series
        fm_summary_dict[f'{period}d'] = fm_summary
    return fm_returns_series_dict,fm_t_stats_series_dict, fm_summary_dict


import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

# 假设 logger 已经配置好
import logging

logger = logging.getLogger(__name__)


# 只是用于绘图
def calculate_quantile_daily_returns(
        factor_df: pd.DataFrame,
        returns_calculator,  # 具体化Callable
        n_quantiles
) ->   pd.DataFrame:
    """
    【V3 最终版】计算因子分层组合的每日收益率。
    1. 修正了函数签名，防止因参数位置错误导致的TypeError。
    2. 增加了对因子数据类型的强制转换，确保qcut函数安全运行。

    Args:
        factor_df (pd.DataFrame): 因子值DataFrame (index=date, columns=stock)。
                                  这是T-1日的信息。
        price_df (pd.DataFrame): 每日收盘价矩阵 (index=date, columns=stock)。
        n_quantiles (int): (关键字参数) 要划分的分位数数量。

    Returns:
        Dict[str, pd.DataFrame]: 只有一个key的字典，值是分层组合的每日收益DataFrame。
    """
    logger.info("  > 正在计算分层组合的【每日】收益率 (用于绘图)...")
    forward_returns_1d = returns_calculator(period=1)
    # 2. 有效域掩码：显式定义分析样本
    # 单一事实来源 - 明确定义所有有效的(date, stock)坐标点
    valid_mask = factor_df.notna() & forward_returns_1d.notna()#好的合集

    # 应用掩码，确保因子和收益具有完全相同的NaN分布
    final_factor = factor_df.where(valid_mask)#坏的合集都为nan
    final_returns_1d = forward_returns_1d.where(valid_mask)#坏的合集都为nan stock进行操作，丢的nan都是一样的，就可有无脑concat了

    # 数据转换：从"宽表"到"长表"（现在是安全的）
    factor_long = final_factor.stack().rename('factor')
    returns_1d_long = final_returns_1d.stack().rename('return_1d')

    # 合并数据（不再需要dropna，因为已经完全对齐）
    merged_df = pd.concat([factor_long, returns_1d_long], axis=1)

    if merged_df.empty:
        raise ValueError("  > 因子和单日收益数据没有重叠，无法计算分层收益。")

    # 3. 确保因子值为数值类型
    merged_df['factor'] = pd.to_numeric(merged_df['factor'], errors='coerce')
    merged_df=merged_df.dropna(subset=['factor'], inplace=False)

    # 4. 稳健的分组
    merged_df['quantile'] = merged_df.groupby(level=0)['factor'].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop') + 1
    )

    # 5. 计算各分位数组合的每日平均收益
    daily_quantile_returns = merged_df.groupby([merged_df.index.get_level_values(0), 'quantile'])[
        'return_1d'].mean()

    # 6. 数据转换回“宽表”
    quantile_returns_wide = daily_quantile_returns.unstack()
    quantile_returns_wide.columns = [f'Q{int(col)}' for col in quantile_returns_wide.columns]

    # 7. 计算多空组合的每日收益（价差）
    top_q_col = f'Q{n_quantiles}'
    bottom_q_col = 'Q1'

    if top_q_col in quantile_returns_wide.columns and bottom_q_col in quantile_returns_wide.columns:
        quantile_returns_wide['TopMinusBottom'] = quantile_returns_wide[top_q_col] - quantile_returns_wide[
            bottom_q_col]
    else:
        quantile_returns_wide['TopMinusBottom'] = np.nan

    # 8. 返回结果
    # 我们用一个固定的key，比如 '21d'，让绘图函数能找到它
    return  quantile_returns_wide.sort_index(axis=1)
