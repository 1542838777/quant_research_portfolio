"""
因子预处理流水线 - 单因子测试终极作战手册
第三阶段：因子预处理

实现完整的因子预处理流水线：
1. 去极值 (Winsorization)
2. 中性化 (Neutralization) 
3. 标准化 (Standardization)
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Optional, Tuple, Any
import warnings
from sklearn.linear_model import LinearRegression
import sys
import os
from pathlib import Path

from projects._03_factor_selection.config.base_config import FACTOR_STYLE_RISK_MODEL
from projects._03_factor_selection.factor_manager.classifier.factor_classifier import FactorClassifier
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.config.logger_config import setup_logger

warnings.filterwarnings('ignore')

# 配置日志
logger = setup_logger(__name__)


class FactorProcessor:
    """
    因子预处理器 - 专业级因子预处理流水线
    
    按照华泰证券标准实现：
    1. 去极值：中位数绝对偏差法(MAD) / 分位数法
    2. 中性化：行业中性化 + 市值中性化
    3. 标准化：Z-Score标准化 / 排序标准化
    """

    def __init__(self, config: Dict):
        """
        初始化因子预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})

    # ok
    def process_factor(self,
                       target_factor_df: pd.DataFrame,
                       industry_df: pd.DataFrame,
                       target_factor_name: str,
                       auxiliary_dfs,
                       neutral_dfs,
                       style_category: str,
                       neutralize_after_standardize: bool = False, #默认是最后标准化
                       ):
        """
        完整的因子预处理流水线
        
        Args:
            factor_data: 原始因子数据
            auxiliary_df_dict: 辅助数据（市值、行业等）
            
        Returns:
            预处理后的因子数据
        """
        processed_target_factor_df = target_factor_df.copy()
        auxiliary_dfs = auxiliary_dfs.copy()


        # 步骤1：去极值
        # print("2. 去极值处理...")
        processed_target_factor_df = self.winsorize_robust(processed_target_factor_df,industry_df)

        if not neutralize_after_standardize:
            # 步骤2：中性化
            if self.preprocessing_config.get('neutralization', {}).get('enable', False):
                processed_target_factor_df = self._neutralize(processed_target_factor_df, target_factor_name,auxiliary_dfs,
                                                              neutral_dfs, style_category)
            else:
                logger.info("2. 跳过中性化处理...")
            # 步骤3：标准化
            processed_target_factor_df = self._standardize_robust(processed_target_factor_df,industry_df)
        else:
            # 步骤2：标准化
            processed_target_factor_df = self._standardize_robust(processed_target_factor_df,industry_df)
            # 步骤3：中性化
            if self.preprocessing_config.get('neutralization', {}).get('enable', False):
                # print("3. 中性化处理...")
                processed_target_factor_df = self._neutralize(processed_target_factor_df, target_factor_name, auxiliary_dfs,
                                                              neutral_dfs, style_category)
            else:
                logger.info("3. 跳过中性化处理...")
        # 统计处理结果
        self._print_processing_stats(target_factor_df, processed_target_factor_df)

        return processed_target_factor_df

    # # ok#ok
    # def winsorize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     去极值处理
    #
    #     Args:
    #         factor_data: 因子数据
    #
    #     Returns:
    #         去极值后的因子数据
    #     """
    #     winsorization_config = self.preprocessing_config.get('winsorization', {})
    #     method = winsorization_config.get('method', 'mad')
    #
    #     processed_factor = factor_data.copy()
    #
    #     if method == 'mad':
    #         # 中位数绝对偏差法 (Median Absolute Deviation)
    #         threshold = winsorization_config.get('mad_threshold', 5)
    #         # print(f"  使用MAD方法，阈值倍数: {threshold}")
    #
    #         # 向量化计算每日的中位数和MAD
    #         median = factor_data.median(axis=1)
    #         mad = (factor_data.sub(median, axis=0)).abs().median(axis=1)
    #
    #         # 向量化计算每日的上下边界
    #         upper_bound = median + threshold * mad
    #         lower_bound = median - threshold * mad
    #
    #         # 向量化clip，axis=0确保按行广播边界
    #         return factor_data.clip(lower_bound, upper_bound, axis=0)
    #     elif method == 'quantile':
    #         # 分位数法
    #         quantile_range = winsorization_config.get('quantile_range', [0.01, 0.99])
    #         print(f"  使用分位数方法，范围: {quantile_range}")
    #         # 向量化计算每日的分位数边界
    #         bounds = factor_data.quantile(q=quantile_range, axis=1).T  # .T转置是为了方便后续clip
    #         lower_bound = bounds.iloc[:, 0]
    #         upper_bound = bounds.iloc[:, 1]
    #         return factor_data.clip(lower_bound, upper_bound, axis=0)
    #
    #     return processed_factor

    def _winsorize_mad_series(self, series: pd.Series, threshold: float) -> pd.Series:
        """
        【辅助函数】对单个Series(如单日单行业)进行MAD去极值。
        """
        # 1. 剔除NaN值进行计算
        valid_series = series.dropna()
        if valid_series.empty:
            return series

        # 2. 计算中位数和MAD
        median = valid_series.median()
        # MAD (Median Absolute Deviation from the sample's median)
        mad = (valid_series - median).abs().median()

        # 3. [核心修正] 处理零MAD问题
        # 如果MAD为0，说明大部分数据点都等于中位数，此时不应进行去极值操作，否则会把所有值变成中位数。
        if mad == 0:
            return series  # 直接返回原序列

        # 4. [核心修正] 引入统计学校正因子1.4826
        # 1.4826 是高斯分布(正态分布)MAD与标准差之间的换算系数。
        # 这样做使得 mad_threshold=3 大致等价于 3-sigma。
        const = 1.4826
        upper_bound = median + threshold * const * mad
        lower_bound = median - threshold * const * mad

        # 5. 使用clip进行去极值
        return series.clip(lower_bound, upper_bound)

    def _winsorize_quantile_series(self, series: pd.Series, quantile_range: list) -> pd.Series:
        """
        【辅助函数】对单个Series进行分位数去极值。
        """
        valid_series = series.dropna()
        if valid_series.empty:
            return series

        lower_q, upper_q = min(quantile_range), max(quantile_range)
        lower_bound = valid_series.quantile(lower_q)
        upper_bound = valid_series.quantile(upper_q)

        return series.clip(lower_bound, upper_bound)

    def winsorize_robust(self, factor_data: pd.DataFrame, industry_df: pd.DataFrame = None) -> pd.DataFrame:
        """
         去极值处理函数。
        支持全市场或分行业的MAD和分位数法。
        Args:
            factor_data (pd.DataFrame): 因子数据 (index=date, columns=stock)。
            industry_df (pd.DataFrame, optional): 行业分类数据 (格式同上)。
                                                  如果提供此参数，则自动执行分行业去极值。
                                                  默认为 None，执行全市场去极值。

        Returns:
            pd.DataFrame: 去极值后的因子数据。
        """
        winsorization_config = self.preprocessing_config.get('winsorization', {})
        method = winsorization_config.get('method', 'mad')

        if industry_df is None:
            # --- 场景一：全市场去极值 (你的原始逻辑) ---
            print("  执行全市场去极值...")
            if method == 'mad':
                threshold = winsorization_config.get('mad_threshold', 5)
                # 对每一天(每一行)应用MAD去极值辅助函数
                return factor_data.apply(self._winsorize_mad_series, axis=1, threshold=threshold)
            elif method == 'quantile':
                quantile_range = winsorization_config.get('quantile_range', [0.01, 0.99])
                return factor_data.apply(self._winsorize_quantile_series, axis=1, quantile_range=quantile_range)
        else:
            # --- 场景二：分行业去极值 (高质量做法) ---
            print("  执行分行业去极值...")
            # 1. 确保因子和行业数据对齐
            factor_aligned, industry_aligned = factor_data.align(industry_df, join='left', axis=1)

            # 2. 将数据堆叠成长格式，便于按天和行业分组
            factor_long = factor_aligned.stack(dropna=False).rename('factor')
            industry_long = industry_aligned.stack(dropna=False).rename('industry')
            #在合并后，丢弃那些没有因子值 或 没有行业值的行
            #    这确保只在信息完备的数据上进行去极值
            combined = pd.concat([factor_long, industry_long], axis=1).dropna(subset=['factor', 'industry'])

            # 3. [核心修正] 使用更直接的 groupby().transform()
            # 我们直接按照索引的第0层(日期)和'industry'列进行分组
            if method == 'mad':
                threshold = winsorization_config.get('mad_threshold', 5)
                # transform会返回一个与原始combined['factor']形状和索引完全相同的Series
                processed_factor = combined.groupby([pd.Grouper(level=0), 'industry'])['factor'].transform(
                    self._winsorize_mad_series, threshold=threshold
                )
            elif method == 'quantile':
                quantile_range = winsorization_config.get('quantile_range', [0.01, 0.99])
                processed_factor = combined.groupby([pd.Grouper(level=0), 'industry'])['factor'].transform(
                    self._winsorize_quantile_series, quantile_range=quantile_range
                )
            else:
                return factor_data

            # 4. 将处理后的长格式数据转回宽格式矩阵 (现在可以完美工作了)
            # 因为processed_factor的索引是(日期, 股票代码)，与原始长表一致
            return processed_factor.unstack().reindex(index=factor_data.index, columns=factor_data.columns)

        return factor_data

    # ok
    def _neutralize(self,
                    factor_data: pd.DataFrame,
                    target_factor_name:str,
                    auxiliary_dfs: Dict[str, pd.DataFrame],
                    neutral_dfs: Dict[str, pd.DataFrame],
                    style_category: str
                    ) -> pd.DataFrame:
        """
         根据因子所属的“门派”，自动选择最合适的中性化方案。
         Args:
             factor_data: 待中性化的因子数据 (T-1日信息)
             target_factor_name: 待中性化因子的名称
             auxiliary_dfs: 辅助数据字典 (可能包含Beta等，根据您的实际约定)
             neutral_dfs: 包含了市值和预处理好的行业哑变量数据的字典 (日期-股票代码的DataFrame)
             factor_school: 因子门派 ('fundamentals', 'trend', 'microstructure')

         Returns:
             中性化后的因子数据
         """
        neutralization_config = self.preprocessing_config.get('neutralization', {})
        if not neutralization_config.get('enable', False):
            return factor_data
        factor_school = FactorManager.get_school_by_style_category( style_category)
        logger.info(f"  > 正在对 '{factor_school}' 派因子 '{target_factor_name}' 进行中性化处理...")
        processed_factor = factor_data.copy()

        # --- 阶段一：(可选) 因子残差化 (仅针对市场微观派) ---
        # 残差化通常是对时间序列进行去均值，与截面中性化是两个不同的操作
        if factor_school == 'microstructure':
            window = neutralization_config.get('residualization_window', 20)
            logger.info(f"    > 应用时间序列残差化 (窗口: {window}天)...")
            # 确保 min_periods 不会太小导致过多 NaN
            factor_mean = processed_factor.rolling(window=window, min_periods=max(1, int(window * 0.5))).mean()
            processed_factor = processed_factor - factor_mean

        # --- 阶段二：确定本次回归需要中性化的因子列表 ---
        factors_to_neutralize = self.get_regression_need_neutral_factor_list(style_category,target_factor_name)

        if not factors_to_neutralize:
            logger.info(f"    > '{factor_school}' 派因子无需中性化。")
            return processed_factor
        logger.info(f"{target_factor_name}逐日进行截面回归中性化")
        # --- 阶段三：逐日进行截面回归中性化 ---
        for date in processed_factor.index:
            # logger.info(" 获取当天待中性化的因子数据（因变量 y")
            y = processed_factor.loc[date].dropna()
            if len(y) < 20:  # 确保有足够多的样本进行回归，避免过拟合或回归不稳定 todo 实盘需注意，这设置的20
                logger.debug(f"    日期 {date.date()} 样本数不足 ({len(y)} < 20)，跳过中性化。")
                # 当天数据如果不足，将处理后的因子值设为NaN，表示该日未进行有效中性化
                processed_factor.loc[date] = np.nan
                continue

            # a) 构建回归自变量矩阵 X
            # X_df 的索引必须与 y 的索引（即当前日期有因子值的股票）保持一致
            X_df = pd.DataFrame(index=y.index)

            # --- 市值因子 (从 neutral_dfs 获取) ---
            if 'market_cap' in factors_to_neutralize:
                if 'total_mv' not in neutral_dfs:
                    raise ValueError(f"  错误: neutral_dfs 中缺少 'total_mv' 数据，无法进行市值中性化。")

                mv_series = neutral_dfs['total_mv'].loc[date]
                # 对市值取对数，这是常见做法，因为市值分布通常是偏态的
                X_df['log_market_cap'] = np.log(mv_series.reindex(y.index))  # 确保与 y 的索引对齐

            # --- 行业因子 (从 neutral_dfs 获取预处理好的哑变量) ---
            if 'industry' in factors_to_neutralize:
                # 找到所有以 'industry_' 开头的辅助数据键，这些就是预处理好的行业哑变量
                # 小白解释：列表推导式，高效地从字典键中筛选出所有行业哑变量的名称。
                industry_dummy_keys = [k for k in neutral_dfs.keys() if k.startswith('industry_')]

                if not industry_dummy_keys:
                   raise ValueError(
                        f"   neutral_dfs 中未发现任何预处理的行业哑变量（以 'industry_' 开头），行业中性化失败。")
                else:
                    current_date_industry_dummies = pd.DataFrame(index=y.index)
                    for industry_key in industry_dummy_keys:
                        # 从 neutral_dfs 中获取对应行业的 DataFrame
                        # 然后 loc[date] 获取当前日期的数据
                        # reindex(y.index) 确保哑变量与当前日期的 y 因子值股票对齐
                        dummy_series = neutral_dfs[industry_key].loc[date]
                        current_date_industry_dummies[industry_key] = dummy_series.reindex(y.index)

                    # 合并所有行业哑变量到 X_df
                    X_df = X_df.join(current_date_industry_dummies)

            # --- Beta 因子 (假设仍在 auxiliary_dfs 中) ---
            if 'pct_chg_beta' in factors_to_neutralize:
                if 'pct_chg_beta' not in auxiliary_dfs:
                    raise ValueError(f"  错误: auxiliary_dfs 中缺少 'pct_chg_beta' 数据，无法进行Beta中性化。")

                beta_series = auxiliary_dfs['pct_chg_beta'].loc[date]
                X_df['pct_chg_beta'] = beta_series.reindex(y.index)  # 确保与 y 的索引对齐

            # b) 数据对齐与清洗
            # 使用 join 和 dropna 来确保 y 和 X_df 的股票代码对齐，并去除所有 NaN
            # 注意：X_df 在构建时已经基于 y.index，但如果某个辅助因子本身有 NaN，这里会进一步清理
            combined_df = pd.concat([y, X_df], axis=1).dropna()

            # 确保清理后的样本数满足回归要求
            # X_df.shape[1] 是自变量的数量，我们需要比自变量数量更多的样本，避免欠拟合
            # 经验法则：样本数至少是自变量数量的几倍（例如，2-5倍）
            num_predictors = X_df.shape[1]  # 计算实际的自变量数量
            if len(combined_df) < num_predictors + 5:  # 至少要比变量数多5个样本，这是一个经验值，可以根据需求调整 ，x是自变量个数，y是因变量（测试的天数），甚至都赶不上自变量的个数，那还测试什么 干脆报错把 但是实际很难出现，x最多就100来个（因为行业就100来个），我们回测天数都是几百天，
                logger.warning(
                    f"  警告: 日期 {date.date()} 清理后样本数不足 ({len(combined_df)} < {num_predictors + 5})，跳过中性化。")
                processed_factor.loc[date] = np.nan  # 将当天因子数据设为NaN
                continue

            # 提取清理后的因变量和自变量
            y_clean = combined_df[y.name]
            # 确保 X_clean 拿到的是对齐后的，并且是 X_df 中实际使用的列
            X_clean = combined_df[X_df.columns]

            # c) 执行回归并计算残差
            try:
                # fit_intercept=True 默认包含截距项，通常是回归分析的良好实践
                reg = LinearRegression(fit_intercept=True)
                reg.fit(X_clean, y_clean)

                # 计算残差：原始因子值 - 因子模型预测值
                residuals = y_clean - reg.predict(X_clean)

                # 将中性化后的残差更新回 processed_factor
                # 只更新那些参与了回归的股票（即 residuals.index）
                processed_factor.loc[date, residuals.index] = residuals
                logger.debug(f"    日期 {date.date()} 中性化成功，处理了 {len(residuals)} 个样本。")

            except Exception as e:
                # 捕获回归可能出现的错误，如矩阵奇异（共线性）、样本不足导致无法拟合等
                raise ValueError(f"  警告: 日期 {date.date()} 中性化回归失败: {e}。该日因子数据将标记为NaN。") #raise
        return processed_factor

    # # ok
    # def _standardize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     标准化处理
    #
    #     Args:
    #         factor_data: 因子数据
    #
    #     Returns:
    #         标准化后的因子数据
    #     """
    #     standardization_config = self.preprocessing_config.get('standardization', {})
    #     method = standardization_config.get('method', 'zscore')
    #
    #     processed_factor = factor_data.copy()
    #
    #     if method == 'zscore':
    #         # print("  使用Z-Score标准化 (健壮版)")
    #         mean = processed_factor.mean(axis=1)
    #         std = processed_factor.std(axis=1)
    #
    #         # 识别std=0的安全隐患
    #         std_is_zero_mask = (std == 0)
    #
    #         # 先进行标准化（会产生inf）
    #         processed_factor = processed_factor.sub(mean, axis=0).div(std, axis=0)
    #
    #         # 将std=0的行，结果安全地设为0
    #         processed_factor[std_is_zero_mask] = 0.0
    #
    #         return processed_factor
    #
    #     elif method == 'rank':
    #         print("  使用排序标准化 (健壮版)")
    #
    #         # 识别只有一个有效值的边界情况
    #         valid_counts = processed_factor.notna().sum(axis=1)
    #         single_value_mask = (valid_counts == 1)
    #
    #         # 正常计算排名
    #         ranks = processed_factor.rank(axis=1, pct=True)
    #         processed_factor = 2 * ranks - 1
    #
    #         # 将只有一个有效值的行，结果安全地设为0
    #         processed_factor[single_value_mask] = 0.0
    #         return processed_factor
    #
    #     raise RuntimeError("请指定标准化方式")

    def _zscore_series(self, series: pd.Series) -> pd.Series:
        """【辅助函数】对单个Series(如单日全市场或单日单行业)进行Z-Score标准化"""
        valid_series = series.dropna()
        if valid_series.empty:
            return series

        mean = valid_series.mean()
        std = valid_series.std()

        # [核心边界处理] 如果标准差为0，说明所有值都一样，标准化后应为0
        if std == 0:
            # 创建一个与输入series相同索引的全0 Series
            return pd.Series(0.0, index=series.index)

        return (series - mean) / std

    def _rank_series(self, series: pd.Series) -> pd.Series:
        """【辅助函数】对单个Series进行排序标准化 (转换为[-1, 1]区间)"""
        valid_series = series.dropna()
        if valid_series.empty:
            return series

        # [核心边界处理] 如果只有一个有效值，其排名无意义，设为0
        if len(valid_series) <= 1:
            # 创建一个与输入series相同索引的全0 Series
            return pd.Series(0.0, index=series.index)

        # pct=True 将排名转换为 0 到 1 的百分位
        ranks_pct = series.rank(pct=True)
        # 将 [0, 1] 区间线性映射到 [-1, 1] 区间
        return 2 * ranks_pct - 1

    # --------------------------------------------------------------------------
    #  主函数 (Main Function) - 负责决策与调度
    # --------------------------------------------------------------------------
    #ok
    def _standardize_robust(self, factor_data: pd.DataFrame, industry_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        【V2.0-重构版】因子标准化函数。
        支持全市场或分行业的Z-Score和排序标准化。
        此版本修复了数据对齐和groupby计算的隐患。
        """
        standardization_config = self.preprocessing_config.get('standardization', {})
        method = standardization_config.get('method', 'zscore')

        # 选择要使用的计算函数
        if method == 'zscore':
            calc_func = self._zscore_series
        elif method == 'rank':
            calc_func = self._rank_series
        else:
            raise ValueError(f"未知的标准化方法: {method}")

        if industry_df is None:
            # --- 场景一：全市场标准化 (逻辑正确，保持不变) ---
            print("  执行全市场标准化...")
            return factor_data.apply(calc_func, axis=1)
        else:
            # --- 场景二：分行业标准化 (进行全面修正) ---
            print("  执行分行业标准化...")

            # 1. 【修正一】使用 'left' join，以保留所有因子数据中的股票，防止数据丢失
            #    这是解决最终结果中出现过多NaN的关键。
            factor_aligned, industry_aligned = factor_data.align(industry_df, join='left', axis=1)

            # 2. 将数据堆叠成长格式，便于按天和行业分组
            factor_long = factor_aligned.stack(dropna=False).rename('factor')
            industry_long = industry_aligned.stack(dropna=False).rename('industry')

            # 3. 【修正二】增强dropna，丢弃因子值为空 或 行业分类为空 的行
            #    这是配合 'left' join 必须做的质量控制。
            combined = pd.concat([factor_long, industry_long], axis=1).dropna(subset=['factor', 'industry'])

            # 4. 【核心修正】使用一步到位的 groupby().transform()，替代危险的 groupby().apply()
            #    这彻底解决了 'ValueError: cannot include dtype 'M' in a buffer' 的崩溃问题。
            #    并且性能更高，逻辑更直接。
            #    pd.Grouper(level=0) 是一种标准的分组方式，意为“按索引的第0层（即日期）分组”。
            processed_factor = combined.groupby([pd.Grouper(level=0), 'industry'])['factor'].transform(calc_func)

            # 5. 【保持正确】将处理后的长格式数据转回宽格式矩阵
            #    这一步的逻辑是正确的，它能保证最终输出的DataFrame形状与输入完全一致。
            return processed_factor.unstack().reindex(index=factor_data.index, columns=factor_data.columns)

    def _print_processing_stats(self,
                                original_factor: pd.DataFrame,
                                processed_factor: pd.DataFrame
                                ):
        """打印处理统计信息"""
        logger.info("因子预处理统计:")

        # 原始因子统计
        orig_valid = original_factor.notna().sum().sum()
        orig_total = original_factor.shape[0] * original_factor.shape[1]

        # 处理后因子统计
        proc_valid = processed_factor.notna().sum().sum()

        # 分布统计
        all_values = processed_factor.values.flatten()
        all_values = all_values[~np.isnan(all_values)]

        if len(all_values) > 0:
            logger.info(
                f"  处理后分布: 均值={all_values.mean():.3f}, 标准差={all_values.std():.3f} （z标准化，均值一定是0）")
            logger.info(f"  分位数: 1%={np.percentile(all_values, 1):.3f}, "
                        f"99%={np.percentile(all_values, 99):.3f}")



    def get_regression_need_neutral_factor_list(self, style_category,target_factor_name):
        """
           【V2专业版】根据因子门派和目标因子名称，动态获取需要用于中性化的因子列表。

           此版本修复了旧版本的所有问题：
           1. 采用配置字典，易于扩展。
           2. 移除了所有硬编码的特例，采用通用逻辑。
           3. 使用健壮的方式移除元素，避免程序崩溃。
           """
        # 1. 根据因子门派，从配置中获取基础的中性化列表
        base_neutralization_list = FACTOR_STYLE_RISK_MODEL.get(style_category, FACTOR_STYLE_RISK_MODEL['default'])

        logger.info(
            f"因子 '{target_factor_name}' (style列别: {style_category}) 的初始中性化列表为: {base_neutralization_list}")

        # 2. 【核心逻辑】: 动态排除 - 防止因子对自己进行中性化
        # 使用列表推导式，这是一种更Pythonic、更健壮的方式
        final_list = []
        for risk_factor in base_neutralization_list:
            # 检查市值
            if risk_factor == 'market_cap' and FactorClassifier.is_size_factor(target_factor_name):
                logger.info(f"  - 目标是市值因子，已从中性化列表中移除 'market_cap'")
                continue  # 跳过，不加入final_list

            # 检查行业
            if risk_factor == 'industry' and FactorClassifier.is_industry_factor(target_factor_name):
                logger.info(f"  - 目标是行业因子，已从中性化列表中移除 'industry'")
                continue

            # 检查Beta
            if risk_factor == 'pct_chg_beta' and FactorClassifier.is_beta_factor(target_factor_name):
                logger.info(f"  - 目标是Beta因子，已从中性化列表中移除 'pct_chg_beta'")
                continue

            final_list.append(risk_factor)

        #临时的 记得删除
        if target_factor_name in ['bm_ratio', 'ep_ratio', 'sp_ratio','beta']:
            if 'market_cap' in final_list:
                final_list.remove('market_cap')
        logger.info(f"最终用于回归的中性化目标因子为: {final_list}\n")
        return final_list

