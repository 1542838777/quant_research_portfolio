import pandas as pd
import numpy as np


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

    def get_top_factors(
            self,
            period_to_test: str = '20d',
            top_n_initial: int = 50,
            top_n_final: int = 10,
            correlation_threshold: float = 0.7
    ) -> pd.DataFrame:
        """
        【最终版】采用混合策略，从排行榜中筛选出最终的、多样化的顶级因子。
        """
        print(f"--- 开始筛选周期为 {period_to_test} 的顶级因子 ---")

        # --- 1. 全局排序 ---
        df_period = self.leaderboard_df[self.leaderboard_df['period'] == period_to_test].copy()
        if df_period.empty:
            raise ValueError(f"排行榜中没有周期为 {period_to_test} 的数据。")
        #最基础的条件要满足
        # df_period = df_period[
        #     (df_period['fm_t_statistic'].abs() > 2.0) &
        #     (df_period['ic_ir'] > 0.5) &
        #     (df_period['sharpe_ratio'] > 0.8)
        #     ]

        df_period['sharpe_rank'] = df_period['tmb_sharpe'].rank(ascending=False, method='first')
        df_period['ic_ir_rank'] = df_period['ic_ir'].rank(ascending=False, method='first')
        df_period['fm_t_rank'] = df_period['fm_t_statistic'].abs().rank(ascending=False, method='first')
        df_period['composite_rank'] = df_period[['sharpe_rank', 'ic_ir_rank', 'fm_t_rank']].sum(axis=1)

        # 【Bug修正】按正确的列名 'composite_rank' 排序
        df_period.sort_values(by='composite_rank', ascending=True, inplace=True)

        candidate_df = df_period.head(top_n_initial)
        candidate_factors_list = candidate_df['factor_name'].tolist()
        print(f"通过最低门槛和综合排名，初步筛选出 {len(candidate_factors_list)} 个候选因子。")

        # --- 2. 准备相关性分析所需数据 ---
        # 【性能优化】使用向量化操作，一次性提取所有需要的收益序列
        required_columns = [f"{factor}_{period_to_test}" for factor in candidate_factors_list]

        # 找出在总矩阵中实际存在的列
        available_columns = [col for col in required_columns if col in self.factor_returns_matrix.columns]

        if len(available_columns) < 2:
            print("警告：有效的候选因子收益序列少于2个，无法进行相关性分析。将直接返回排名靠前的因子。")
            return candidate_df.head(top_n_final)

        returns_df = self.factor_returns_matrix[available_columns]
        # 为了提高相关性计算的稳健性，可以只使用共同的非空时段
        returns_df.dropna(inplace=True)

        correlation_matrix = returns_df.corr()

        # --- 3. 贪心算法筛选多样化因子 ---
        final_selected_factors = []
        # 将因子名设为索引，便于后续按排名顺序迭代
        candidate_df.set_index('factor_name', inplace=True)
        sorted_candidates = candidate_df.index.tolist()

        for candidate in sorted_candidates:
            if len(final_selected_factors) >= top_n_final:
                break

            # 构造列名
            candidate_col = f"{candidate}_{period_to_test}"
            if candidate_col not in correlation_matrix.columns:
                continue  # 如果某个因子的收益序列有问题（如全为NaN），则跳过

            if not final_selected_factors:
                final_selected_factors.append(candidate)
                continue

            # 构造已选因子的列名
            selected_cols = [f"{f}_{period_to_test}" for f in final_selected_factors]

            correlations_with_selected = correlation_matrix.loc[candidate_col, selected_cols].abs()

            if correlations_with_selected.max() < correlation_threshold:
                final_selected_factors.append(candidate)

        print(f"--- 筛选完成 ---")
        print(f"最终选出 {len(final_selected_factors)} 个多样化顶级因子：")
        print(final_selected_factors)

        return candidate_df.loc[final_selected_factors].reset_index()
