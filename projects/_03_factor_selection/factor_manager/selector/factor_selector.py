import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable

from quant_lib.config.logger_config import log_warning


class FactorSelector:
    def __init__(self, leaderboard_df: pd.DataFrame, factor_returns_matrix: pd.DataFrame):
        """
        【优化版】初始化因子筛选器。

        Args:
            leaderboard_df (pd.DataFrame): 包含所有因子测试指标的总排行榜。
            factor_returns_matrix (pd.DataFrame): 包含所有因子收益率序列的总矩阵。
                                                 索引为日期，列为 'factor_name_period'。
        """
        if not isinstance(leaderboard_df, pd.DataFrame) or leaderboard_df.empty:
            raise ValueError("leaderboard_df 必须是一个非空的DataFrame。")
        if not isinstance(factor_returns_matrix, pd.DataFrame) or factor_returns_matrix.empty:
            raise ValueError("factor_returns_matrix 必须是一个非空的DataFrame。")

        self.leaderboard_df = leaderboard_df
        self.factor_returns_matrix = factor_returns_matrix

 ##

    # 全局排序 对所有因子进行综合排序，选出一个比如Top 50的大名单。 ，保证所有因子的个体质量都是顶尖的。
    #
    # 在“Top 50的大名单”   中进行分类和相关性分析:
    #
    # --对这Top 50的因子进行分类（价值、动量等）。
    #
    # --计算这50个因子之间的相关性矩阵。
    #
    # --从这50个最优秀的因子中，挑出比如10个，要求这10个因子彼此不相关，并且尽可能覆盖不同的风格类别。
    #
    # 例如，发现在动量类里，排名前5的因子相关性都高达0.8，那么你只保留其中综合排名最高的那一个。然后你再去价值类、质量类里做同样的操作。
    #
    # 这个混合策略，保证没有错过任何一个在全市场范围内表现优异的因子（质量），又通过后续的步骤保证了最终入选因子的多样性。稳健多因子模型。#
import pandas as pd
import numpy as np


def calculate_factor_score(purify_summary: Union[pd.Series, dict]) -> float:
    """
    【V3版】根据“专业级因子评分体系”为单个因子计算总分。
    此版本能自动判断因子方向（正向或反向），并应用相应的评分逻辑。

    输入: 一个 Series，包含了单个因子的所有指标。
    输出: 该因子的总得分。
    """
    score = 0

    # --- 指标提取 ---
    ic_mean = purify_summary.get('ic_mean', 0)
    ic_ir = purify_summary.get('ic_ir', 0)
    fm_t_stat = purify_summary.get('fm_t_statistic', 0)
    tmb_sharpe = purify_summary.get('tmb_sharpe', 0)
    tmb_max_drawdown = purify_summary.get('tmb_max_drawdown', 0)
    is_monotonic = purify_summary.get('is_monotonic_by_group', False)

    # --- 关键升级：自动判断因子方向 ---
    # 优先使用IC均值的符号判断。如果IC均值接近0，则使用t统计量的符号辅助判断。
    factor_direction = 1
    if ic_mean < -1e-4:  # 使用一个小的负数阈值避免噪音
        factor_direction = -1
    elif abs(ic_mean) <= 1e-4 and fm_t_stat < 0:
        factor_direction = -1

    # --- 评分开始 ---
    # 1. 预测能力分 (满分20) - 根据因子方向调整
    adj_ic_mean = ic_mean * factor_direction
    if adj_ic_mean > 0.05:
        score += 20
    elif 0.03 < adj_ic_mean <= 0.05:
        score += 15
    elif 0.01 < adj_ic_mean <= 0.03:
        score += 10
    elif 0 < adj_ic_mean <= 0.01:
        score += 5

    # 2. 稳定性分 (满分20) - ICIR通常看绝对值或调整后方向，这里以调整后为正评分
    adj_ic_ir = ic_ir * factor_direction
    if adj_ic_ir > 0.5:
        score += 20
    elif 0.3 < adj_ic_ir <= 0.5:
        score += 15
    elif 0.1 < adj_ic_ir <= 0.3:
        score += 10
    elif 0 < adj_ic_ir <= 0.1:
        score += 5

    # 3. 统计显著性分 (满分30) - t值看绝对值，不受方向影响
    t_abs = abs(fm_t_stat)
    if t_abs > 3.0:
        score += 30
    elif 2.0 < t_abs <= 3.0:
        score += 25
    elif 1.5 < t_abs <= 2.0:
        score += 15

    # 4. 策略表现分 (满分20) - 根据因子方向调整
    adj_tmb_sharpe = tmb_sharpe * factor_direction
    perf_score = 0
    if adj_tmb_sharpe > 1.0:
        perf_score = 20
    elif 0.5 < adj_tmb_sharpe <= 1.0:
        perf_score = 15
    elif 0.2 < adj_tmb_sharpe <= 0.5:
        perf_score = 10
    elif 0 < adj_tmb_sharpe <= 0.2:
        perf_score = 5

    # 风控扣分项 - 回撤总是负的，直接比较
    if tmb_max_drawdown < -0.5:
        perf_score -= 5

    score += max(0, perf_score)

    # 5. 单调性分 (满分10)
    if is_monotonic:
        score += 10

    return score

##
# 第一阶段：质量打分 - 使用我提供的“专业级因子评分体系”对每个因子进行绝对打分。
#
# 第二阶段：降维去重 - 将第一阶段中得分超过某个阈值（比如40分）的因子作为“高质量候选池”，然后用你代码里的相关性贪心算法，从这个池子里选出最终的因子组合。#
def get_top_factors(
        leaderboard_df: pd.DataFrame,
        factor_returns_matrix: pd.DataFrame,
        period_to_test: str = '20d',
        quality_score_threshold: float = 40.0,
        top_n_final: int = 10,
        correlation_threshold: float = 0.7
) -> pd.DataFrame:
    """
    【V3版】采用专业两阶段策略，筛选最终的、多样化的顶级因子。
    此版本使用V3评分函数，能自动处理正、反向因子。
    """
    print(f"--- V3: 开始筛选周期为 {period_to_test} 的顶级因子 ---")

    # --- 阶段一：质量打分与筛选 ---
    df_period = leaderboard_df[leaderboard_df['period'] == period_to_test].copy()
    if df_period.empty:
        raise ValueError(f"排行榜中没有周期为 {period_to_test} 的数据。")

    # 【升级点】调用V3评分函数
    df_period['score'] = df_period.apply(calculate_factor_score, axis=1)

    df_period.sort_values(by='score', ascending=False, inplace=True)

    candidate_df = df_period[df_period['score'] >= quality_score_threshold]

    candidate_factors_list = candidate_df['factor_name'].tolist()
    if not candidate_factors_list:
        print(f"警告：在周期 {period_to_test} 下，没有因子的综合得分超过 {quality_score_threshold} 的门槛。")
        return pd.DataFrame()

    print(f"通过专业打分（阈值>{quality_score_threshold}），筛选出 {len(candidate_factors_list)} 个高质量候选因子。")

    # --- 阶段二：多样化筛选（去相关性） ---
    # (此阶段逻辑与V2版本相同，因为已经是最佳实践)
    required_columns = [f"{factor}_{period_to_test}" for factor in candidate_factors_list]
    available_columns = [col for col in required_columns if col in factor_returns_matrix.columns]

    if len(available_columns) < 2:
        print("警告：有效的高质量因子收益序列少于2个，无法进行相关性分析。将直接返回得分最高的因子。")
        return candidate_df.head(top_n_final)

    returns_df = factor_returns_matrix[available_columns].dropna()
    correlation_matrix = returns_df.corr()

    final_selected_factors = []
    for candidate in candidate_factors_list:
        if len(final_selected_factors) >= top_n_final:
            break

        candidate_col = f"{candidate}_{period_to_test}"
        if candidate_col not in correlation_matrix.columns:
            continue

        if not final_selected_factors:
            final_selected_factors.append(candidate)
            continue

        selected_cols = [f"{f}_{period_to_test}" for f in final_selected_factors]
        correlations_with_selected = correlation_matrix.loc[candidate_col, selected_cols].abs()

        if correlations_with_selected.max() < correlation_threshold:
            final_selected_factors.append(candidate)

    print(f"--- 筛选完成 ---")
    print(f"最终选出 {len(final_selected_factors)} 个多样化顶级因子：")
    print(final_selected_factors)

    return df_period[df_period['factor_name'].isin(final_selected_factors)]
