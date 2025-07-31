
from pathlib import Path
import pandas as pd

from projects._03_factor_selection.factor_selector.factor_selector import FactorSelector


def choose_top_n_factors():

    base_dir = 'workspace/factor_results'  # 你的排行榜文件所在目录
    try:
        purify_summary_all = pd.read_parquet('../workspace/factor_results/all_single_factor_test_purify_summary.parquet')
        fm_returns_matrix_all = pd.read_parquet('../workspace/factor_results/all_single_factor_fm_returns_fm_return_series.parquet')
    except FileNotFoundError as e:
        raise ValueError(f"错误：缺少必要的数据文件: {e}")

    # 2. 实例化因子筛选器
    selector = FactorSelector(leaderboard_df=purify_summary_all, factor_returns_matrix=fm_returns_matrix_all)

    # 3. 执行筛选，得到“梦之队”
    # 筛选20天周期的因子
    top_factors_20d = selector.get_top_factors(
        period_to_test='20d',
        top_n_initial=50,
        top_n_final=10,
        correlation_threshold=0.7
    )

    print("\n--- 20天周期最终入选因子详情 ---")
    print(top_factors_20d)

    # 筛选5天周期的因子
    top_factors_5d = selector.get_top_factors(
        period_to_test='5d',
        top_n_initial=30,
        top_n_final=8,
        correlation_threshold=0.6
    )

    print("\n--- 5天周期最终入选因子详情 ---")
    print(top_factors_5d)
if __name__ == '__main__':
    choose_top_n_factors()