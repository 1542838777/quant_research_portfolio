"""
因子预处理流水线 - 单因子测试终极作战手册
第三阶段：因子预处理

实现完整的因子预处理流水线：
1. 去极值 (Winsorization)
2. 中性化 (Neutralization) 
3. 标准化 (Standardization)
"""
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    from sklearn.linear_model import LinearRegression
    HAS_STATSMODELS = False
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Optional, Tuple, Any
import warnings
from sklearn.linear_model import LinearRegression
import sys
import os
from pathlib import Path

from data.local_data_load import get_industry_record_df_processed
from projects._03_factor_selection.config.base_config import FACTOR_STYLE_RISK_MODEL
from projects._03_factor_selection.factor_manager.classifier.factor_classifier import FactorClassifier
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
from quant_lib.config.constant_config import permanent__day

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.config.logger_config import setup_logger, log_warning

warnings.filterwarnings('ignore')

# 配置日志
logger = setup_logger(__name__)


class PointInTimeIndustryMap:
    """
    一个高效的、支持即时查询(Point-in-Time)的行业地图管理器。
    """

    def __init__(self,raw_industry_df=None):
        """
        通过原始的、包含in_date和out_date的成员关系DataFrame进行初始化。
        这个过程会进行一次性预处理，构建高效的查询结构。
        """
        print("正在预处理历史行业数据，构建Point-in-Time地图...")
        if raw_industry_df is None:
            self._raw_data = get_industry_record_df_processed()
        else:
            self._raw_data = raw_industry_df

        # 1. 获取所有行业变动的“事件日”
        event_dates = pd.unique(np.concatenate([
            self._raw_data['in_date'],
            self._raw_data['out_date'] + pd.Timedelta(days=1)  # out_date当天失效，第二天变更 （理解：我们要的是状态变更生效的哪一天！
        ])).astype('datetime64[ns]')
        #... 快照A ... [2023-11-15] ... 快照B ... [2023-12-29] ... 快照C ... [2024-02-10] ... 快照D ... [2024-03-16] ... 快照E ...
        # 核心思想就是 快照B的start end 都来自于某天某只股票的事件（生效or剔除） 在整个快照period，可以理解为 这整个时期 所有行业都是稳定未变化的！
        #下面 遍历每个事件行动日！
        ###比如遍历到快照B的start日，20231115
        #### 注意期间的不需要遍历啊，这就是此设计的唯一的性能亮点 （为什么可以做到不需要遍历：见上面说的核心思想
        ##然后遍历快照B的end日， 20231229
        self._event_dates = sorted([d for d in event_dates if d < pd.Timestamp(permanent__day)])

        # 2. 为每个事件日生成一个行业地图快照
        self._maps_on_event_dates = {}
        for date in self._event_dates:
            # 筛选出在 `date` 当天有效的成员关系
            current_map_df = self._raw_data[
                (self._raw_data['in_date'] <= date) &
                (self._raw_data['out_date'] >= date)
                ]
            # 只保留需要的列，并设置索引
            self._maps_on_event_dates[date] = current_map_df[['ts_code', 'l1_code', 'l2_code']].set_index('ts_code')

        print(f"预处理完成！共生成 {len(self._event_dates)} 个历史快照。")
    #ok
    def get_map_for_date(self, query_date: pd.Timestamp) -> pd.DataFrame:
        """
        高效获取指定日期的行业地图。
        :param query_date: 需要查询的日期
        :return: 一个以ts_code为索引的DataFrame，包含l1_code和l2_code
        """
        # 使用二分查找找到正确的事件日索引
        # bisect_right 会找到 query_date 应该插入的位置
        # 它之前的那个事件日，就是我们需要的快照日期
        idx = bisect_right(self._event_dates, query_date) #event_Dates 间隔就是静态的日期，现在 需要查询query_date 位于哪段时间，返回query_date 左侧最接近的event_Date 就是我们这段时期的start。直接取用整个静态的结果！（但是这个函数返回的是目标query_date的索引，所以我们需要-1 才是左侧最接近event_date的start

        if idx == 0:
            # 如果查询日期比最早的事件日还早
            raise ValueError("查询日期比最早的事件日还早 肯定有问题！") #return  pd.DataFrame(columns=['l1_code', 'l2_code'])

        # 获取对应的历史快照
        target_event_date = self._event_dates[idx - 1]
        return self._maps_on_event_dates[target_event_date]
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
                       target_factor_name: str,
                       auxiliary_dfs,
                       neutral_dfs,
                       style_category: str,
                       neutralize_after_standardize: bool = False, #默认是最后标准化
                       pit_map:PointInTimeIndustryMap = None
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

        if pit_map is None:
            pit_map = PointInTimeIndustryMap()
        # 步骤1：去极值
        # print("2. 去极值处理...")
        processed_target_factor_df = self.winsorize_robust(processed_target_factor_df,pit_map)

        if not neutralize_after_standardize:
            # 步骤2：中性化
            if self.preprocessing_config.get('neutralization', {}).get('enable', False):
                processed_target_factor_df = self._neutralize(processed_target_factor_df, target_factor_name,
                                                              neutral_dfs, style_category)
            else:
                logger.info("2. 跳过中性化处理...")
            # 步骤3：标准化
            processed_target_factor_df = self._standardize_robust(processed_target_factor_df,pit_map)
        else:
            # 步骤2：标准化
            processed_target_factor_df = self._standardize_robust(processed_target_factor_df,pit_map)
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

    def _winsorize_mad_series(self, series: pd.Series, threshold: float,min_samples: int = 10) -> pd.Series:
        """
           MAD去极值
           - 新增 min_samples 参数，用于处理小样本组
           - 假设输入的 series 是已经 dropna() 过的
           """
        # 1. 检查有效样本数是否达到阈值
        if series.size < min_samples:
            return series  # 样本太少，不处理，直接返回原序列

        # 2. 计算中位数和MAD (此时series已不含NaN)
        median = series.median()
        mad = (series - median).abs().median()

        # 3. 处理零MAD问题
        if mad == 0:
            return series

        # 4. 计算边界并clip
        const = 1.4826
        upper_bound = median + threshold * const * mad
        lower_bound = median - threshold * const * mad

        return series.clip(lower_bound, upper_bound)

    def _winsorize_quantile_series(self, series: pd.Series, quantile_range: list,min_samples: int = 10) -> pd.Series:
        """
        【辅助函数】对单个Series进行分位数去极值。
        """
        # 1. 检查有效样本数是否达到阈值
        if series.size < min_samples:
            return series

        # 2. 计算分位数 (此时series已不含NaN)
        lower_q, upper_q = min(quantile_range), max(quantile_range)
        lower_bound = series.quantile(lower_q)
        upper_bound = series.quantile(upper_q)

        return series.clip(lower_bound, upper_bound)
        # =========================================================================
        # 【核心修改】新的辅助函数，处理单个截面日的回溯逻辑
        # =========================================================================
    #ok
    def _winsorize_cross_section_fallback(
            self,
            daily_factor_series: pd.Series,
            daily_industry_map: pd.DataFrame,
            config: dict
    ) -> pd.Series:
        """
        对单个截面日的因子数据执行“向上回溯”去极值。
        这是之前我们独立设计的 winsorize_by_industry_fallback 函数的类方法版本。
        """
        primary_col = config['primary_level']  # e.g., 'l2_code'
        fallback_col = config['fallback_level']  # e.g., 'l1_code'
        min_samples = config['min_samples']

        # 1. 数据整合
        df = daily_factor_series.to_frame(name='factor')
        merged_df = df.join(daily_industry_map, how='left')
        #   merge 之前，先将索引ts_code重置为一列，以防在merge(merged_df.merge(primary_stats, on=primary_col, how='left'))中丢失
        merged_df.reset_index(inplace=True)

        # 删除没有因子值或行业分类的数据
        merged_df.dropna(subset=['factor', primary_col, fallback_col], inplace=True)
        if merged_df.empty:
            return pd.Series(index=daily_factor_series.index, dtype=float)

        # 2. 计算各级别行业的统计数据
        def mad_func(s: pd.Series) -> float:
            return (s - s.median()).abs().median()

        primary_stats = merged_df.groupby(primary_col)['factor'].agg(['median', 'count', mad_func])
        primary_stats.rename(columns={'median': 'primary_median', 'count': 'primary_count', 'mad_func': 'primary_mad'},
                             inplace=True)

        fallback_stats = merged_df.groupby(fallback_col)['factor'].agg(['median', mad_func])
        fallback_stats.rename(columns={'median': 'fallback_median', 'mad_func': 'fallback_mad'}, inplace=True)

        # 3. 将统计数据映射回每只股票
        merged_df = merged_df.merge(primary_stats, on=primary_col, how='left')
        merged_df = merged_df.merge(fallback_stats, on=fallback_col, how='left')

        # 4. 核心回溯逻辑 不满足必须样本数目，就用一级行业的mad
        use_fallback = merged_df['primary_count'] < min_samples

        merged_df['final_median'] = np.where(use_fallback, merged_df['fallback_median'], merged_df['primary_median'])
        merged_df['final_mad'] = np.where(use_fallback, merged_df['fallback_mad'], merged_df['primary_mad'])

        merged_df['final_mad'].replace(0, 1e-9, inplace=True)#秒啊，如果是0的话 下面upper lower是一个值！ 导致最后所因子都是一个值！大忌！
        merged_df.set_index('ts_code', inplace=True)

        # 5. 执行去极值
        method = config.get('method', 'mad')
        if method == 'mad':
            threshold = config.get('mad_threshold', 3)
            const = 1.4826
            upper = merged_df['final_median'] + threshold * const * merged_df['final_mad']
            lower = merged_df['final_median'] - threshold * const * merged_df['final_mad']
        elif method == 'quantile':
            # 分位数法也可以应用回溯逻辑，但较为罕见。这里我们以MAD为主，分位数保持组内处理。
            # 如需分位数回溯，逻辑会更复杂，此处为简化。
            return merged_df['factor']  # 暂不处理quantile的回溯
        else:
            return merged_df['factor']

        winsorized_factor = merged_df['factor'].clip(lower=lower, upper=upper)

        # 返回一个与输入Series对齐的Series
        return winsorized_factor.reindex(daily_factor_series.index)

        # =========================================================================
        # 【核心修改】重构后的 winsorize_robust 函数
        # =========================================================================

    def winsorize_robust(self, factor_data: pd.DataFrame,pit_industry_map: PointInTimeIndustryMap = None) -> pd.DataFrame:
        """
        去极值处理函数。
        支持全市场或分行业（带向上回溯功能）的MAD和分位数法。

        Args:
            factor_data (pd.DataFrame): 因子数据 (index=date, columns=stock)。
            industry_map (pd.DataFrame, optional): 行业分类数据 (index=stock, columns=['l1_code', 'l2_code',...])
                                                   如果提供此参数，则执行分行业去极值。
        Returns:
            pd.DataFrame: 去极值后的因子数据。
        """
        winsorization_config = self.preprocessing_config.get('winsorization', {})
        industry_config = winsorization_config.get('by_industry')

        # --- 路径一：全市场去极值 (逻辑基本不变) ---
        if pit_industry_map is None or industry_config is None:
            print("  执行全市场去极值...")
            method = winsorization_config.get('method', 'mad')
            if method == 'mad':
                params = {'threshold': winsorization_config.get('mad_threshold', 5),
                          'min_samples': 1}  # 全市场不需min_samples
                return factor_data.apply(self._winsorize_mad_series, axis=1, **params)
            elif method == 'quantile':
                params = {'quantile_range': winsorization_config.get('quantile_range', [0.01, 0.99]), 'min_samples': 1}
                return factor_data.apply(self._winsorize_quantile_series, axis=1, **params)
            return factor_data

        # --- 路径二：分行业去极值 (采用回溯逻辑) ---
        else:
            print(
                f"  执行分行业去极值 (主行业: {industry_config['primary_level']}, 回溯至: {industry_config['fallback_level']})...")

            # 按天循环，在截面日上执行矢量化操作
            processed_data = {}
            for date in factor_data.index:
                # 获取当天的因子和行业数据
                daily_factor_series = factor_data.loc[date].dropna()

                # 如果当天没有有效因子值，则跳过
                if daily_factor_series.empty:
                    processed_data[date] = pd.Series(dtype=float)
                    log_warning(f"去极值过程中，发现当天{date}所有股票因子值都为空")
                    continue
                # 在循环内部，为每一天获取正确的历史地图
                daily_industry_map = pit_industry_map.get_map_for_date(date)

                processed_data[date] = self._winsorize_cross_section_fallback(
                    daily_factor_series=daily_factor_series,
                    daily_industry_map=daily_industry_map,
                    config=industry_config
                )

            # 将处理后的数据合并回DataFrame
            result_df = pd.DataFrame.from_dict(processed_data, orient='index')
            # 保持原始的索引和列顺序
            return result_df.reindex(index=factor_data.index, columns=factor_data.columns)

    # ok
    #考虑 传入的行业如果是二级行业那么行业变量多达130个！，我又不做全A，中证800才800，平均一个行业才5只股票 来进行中性化，有点不具参照！，必须用一级行业

    def _neutralize(self,
                    factor_data: pd.DataFrame,
                    target_factor_name: str,
                    # auxiliary_dfs: Dict[str, pd.DataFrame], # 在新架构下，beta也应在neutral_dfs中
                    neutral_dfs: Dict[str, pd.DataFrame],
                    style_category: str
                    ) -> pd.DataFrame:
        """
        【V2.0-重构版】根据因子所属的“门派”，自动选择最合适的中性化方案。
        此版本修复了潜在bug，并优化了数据构建流程，提升了运行效率和代码清晰度。
        """
        neutralization_config = self.preprocessing_config.get('neutralization', {})
        if not neutralization_config.get('enable', False):
            return factor_data

        factor_school = FactorManager.get_school_by_style_category(style_category)
        logger.info(f"  > 正在对 '{factor_school}' 派因子 '{target_factor_name}' 进行中性化处理...")
        processed_factor = factor_data.copy()

        # --- 阶段一：(可选) 因子残差化 ---
        if factor_school == 'microstructure':
            window = neutralization_config.get('residualization_window', 20)
            logger.info(f"    > 应用时间序列残差化 (窗口: {window}天)...")
            factor_mean = processed_factor.rolling(window=window, min_periods=max(1, int(window * 0.5))).mean()
            processed_factor = processed_factor - factor_mean

        # --- 阶段二：确定中性化因子列表 ---
        factors_to_neutralize = self.get_regression_need_neutral_factor_list(style_category, target_factor_name)
        if not factors_to_neutralize:
            logger.info(f"    > '{factor_school}' 派因子无需中性化。")
            return processed_factor

        logger.info(f"    > {target_factor_name} 将对以下风格进行中性化: {factors_to_neutralize}")

        skipped_days_count = 0
        total_days = len(processed_factor.index)

        # --- 阶段三：逐日截面回归中性化 ---
        for date in processed_factor.index:
            y_series = processed_factor.loc[date].dropna()
            if y_series.empty:
                skipped_days_count += 1
                continue

            # --- a) 【效率优化】构建回归自变量矩阵 X ---
            X_df_parts = []  # 使用一个列表来收集所有自变量 Series

            # --- 市值因子 ---
            if 'market_cap' in factors_to_neutralize:
                # 【命名统一】从 neutral_dfs 中寻找规模因子，名字可以是 'small_cap', 'log_circ_mv' 等
                # 我们假设传入的已经是log处理过的
                market_cap_key = 'small_cap'  # 与你 neutral_dfs 中定义的key保持一致
                if market_cap_key not in neutral_dfs:
                    raise ValueError(f"neutral_dfs 中缺少市值因子 '{market_cap_key}'。")
                mv_series = neutral_dfs[market_cap_key].loc[date].rename('log_market_cap')
                X_df_parts.append(mv_series)

            # --- 行业因子 ---
            if 'industry' in factors_to_neutralize:
                industry_dummy_keys = [k for k in neutral_dfs.keys() if k.startswith('industry_')]
                if not industry_dummy_keys:
                    raise ValueError("neutral_dfs 中未发现行业哑变量。")

                # 【效率优化】一次性从 neutral_dfs 中提取当天的所有行业哑变量
                daily_dummies_df = pd.concat(
                    [neutral_dfs[key].loc[date].rename(key) for key in industry_dummy_keys],
                    axis=1
                )
                X_df_parts.append(daily_dummies_df)

            # --- Beta 因子 ---
            if 'pct_chg_beta' in factors_to_neutralize:
                if 'pct_chg_beta' not in neutral_dfs:  # 建议将beta也统一放入neutral_dfs
                    raise ValueError("neutral_dfs 中缺少 'pct_chg_beta' 数据。")

                beta_series = neutral_dfs['pct_chg_beta'].loc[date].rename('pct_chg_beta')
                X_df_parts.append(beta_series)

            if not X_df_parts:
                continue

            # --- b) 【流程优化】将所有部分一次性合并，然后与 y 对齐 ---
            X_df = pd.concat(X_df_parts, axis=1)
            # 使用 join='inner' 可以一步到位地完成对齐和筛选
            combined_df = pd.concat([y_series.rename('factor'), X_df], axis=1, join='inner').dropna()

            # --- c) 样本量检查 (逻辑不变，但更健壮) ---
            num_predictors = X_df.shape[1]
            if len(combined_df) < num_predictors + 5:
                logger.warning(
                    f"  警告: 日期 {date.date()} 清理后样本数不足 ({len(combined_df)} < {num_predictors + 5})，跳过中性化。")
                # 注意：这里我们只跳过当天的中性化，而不将当天的所有因子值设为NaN，除非你确实希望如此
                # processed_factor.loc[date] = np.nan
                skipped_days_count += 1
                continue

            # --- d) 执行回归并计算残差 ---
            y_clean = combined_df['factor']
            # 使用 sm.add_constant 添加截距项，是 statsmodels 的标准做法
            X_clean = sm.add_constant(combined_df.drop(columns=['factor']))

            try:
                model = sm.OLS(y_clean, X_clean).fit()
                residuals = model.resid

                # 将中性化后的残差更新回 processed_factor
                processed_factor.loc[date, residuals.index] = residuals

            except Exception as e:
                logger.error(f"  错误: 日期 {date.date()} 中性化回归失败: {e}。该日因子数据将标记为NaN。")
                processed_factor.loc[date] = np.nan
                skipped_days_count += 1
        # 循环结束后，执行“熔断检查” ===
        # 从配置中获取最大跳过比例，如果未配置，则默认为10%
        max_skip_ratio = neutralization_config.get('max_skip_ratio', 0.10)

        actual_skip_ratio = skipped_days_count / total_days

        if actual_skip_ratio > max_skip_ratio:
            # 当实际跳过比例超过阈值时，直接抛出异常，中断程序
            raise ValueError(
                f"因子 '{target_factor_name}' 中性化失败：处理的 {total_days} 天中，"
                f"有 {skipped_days_count} 天 ({actual_skip_ratio:.2%}) 因样本不足被跳过，"
                f"超过了 {max_skip_ratio:.0%} 的容忍上限。"
                f"请检查上游因子数据质量或股票池设置。"
            )

        logger.info(f"  > 中性化完成。在 {total_days} 天中，共跳过了 {skipped_days_count} 天。")
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

    # 你的辅助函数稍作调整，专注于计算本身
    def _zscore_series(self, s: pd.Series) -> pd.Series:
        """【辅助函数】对单个Series进行Z-Score标准化"""
        if s.count() < 2: return pd.Series(0, index=s.index)
        std_val = s.std()
        if std_val == 0: return pd.Series(0, index=s.index)
        mean_val = s.mean()
        return (s - mean_val) / std_val

    def _rank_series(self, s: pd.Series) -> pd.Series:
        """【辅助函数】对单个Series进行排序标准化 (转换为[-1, 1]区间)"""
        return s.rank(pct=True, na_option='keep') * 2 - 1

        # =========================================================================
        # 【新增核心】处理截面标准化回溯的辅助函数
        # =========================================================================

    def _standardize_cross_section_fallback(
            self,
            daily_factor_series: pd.Series,
            daily_industry_map: pd.DataFrame,
            config: dict
    ) -> pd.Series:
        """对单个截面日的因子数据执行“向上回溯”Z-Score标准化。"""
        primary_col = config['primary_level']
        fallback_col = config['fallback_level']
        min_samples = config.get('min_samples', 3)  # 标准化至少需要2个点，设为3更稳健

        # 1. 数据整合
        df = daily_factor_series.to_frame(name='factor')
        merged_df = df.join(daily_industry_map, how='left')

        # 赶紧 先将索引ts_code重置为一列，以防在merge(merged_df.merge(primary_stats, on=primary_col, how='left'))中丢失
        merged_df.reset_index(inplace=True)

        merged_df.dropna(subset=['factor', primary_col, fallback_col], inplace=True)
        if merged_df.empty:
            return pd.Series(index=daily_factor_series.index, dtype=float)

        # 2. 计算各级别行业的统计数据 (mean, std, count)
        primary_stats = merged_df.groupby(primary_col)['factor'].agg(['mean', 'std', 'count'])
        primary_stats.rename(columns={'mean': 'primary_mean', 'std': 'primary_std', 'count': 'primary_count'},
                             inplace=True)

        fallback_stats = merged_df.groupby(fallback_col)['factor'].agg(['mean', 'std'])
        fallback_stats.rename(columns={'mean': 'fallback_mean', 'std': 'fallback_std'}, inplace=True)

        # 3. 将统计数据映射回每只股票
        merged_df = merged_df.merge(primary_stats, on=primary_col, how='left')
        merged_df = merged_df.merge(fallback_stats, on=fallback_col, how='left')

        # 4. 核心回溯逻辑
        use_fallback = merged_df['primary_count'] < min_samples
        merged_df['final_mean'] = np.where(use_fallback, merged_df['fallback_mean'], merged_df['primary_mean'])
        merged_df['final_std'] = np.where(use_fallback, merged_df['fallback_std'], merged_df['primary_std'])

        # 稳健性处理：如果最终选择的标准差还是0或NaN，则不进行标准化（返回中性值0）
        merged_df['final_std'].fillna(0, inplace=True)
        merged_df.loc[merged_df['final_std'] < 1e-9, 'final_std'] = 1.0  # 用1替换，避免除零，相当于 (factor - mean)
        #强制换回索引
        merged_df.set_index('ts_code', inplace=True)

        # 5. 执行Z-Score标准化
        standardized_factor = (merged_df['factor'] - merged_df['final_mean']) / merged_df['final_std']

        # 对于标准差为0导致std被设为1的组，其(factor-mean)可能不为0，需要手动设为0
        standardized_factor.loc[merged_df['final_std'] == 1.0] = 0

        return standardized_factor.reindex(daily_factor_series.index)

        # =========================================================================
        # 【核心升级】重构后的 standardiize_robust 函数
        # =========================================================================

    def _standardize_robust(self, factor_data: pd.DataFrame,
                           pit_industry_map: PointInTimeIndustryMap = None) -> pd.DataFrame:
        """
        【V3.0-PIT版】因子标准化函数。
        支持全市场或分行业（带向上回溯功能）的Z-Score和排序标准化。
        """
        config = self.preprocessing_config.get('standardization', {})
        method = config.get('method', 'zscore')
        industry_config = config.get('by_industry')

        # --- 路径一：全市场标准化 ---
        if pit_industry_map is None or industry_config is None:
            print("  执行全市场标准化...")
            if method == 'zscore':
                return factor_data.apply(self._zscore_series, axis=1)
            elif method == 'rank':
                return factor_data.apply(self._rank_series, axis=1)
            return factor_data

        # --- 路径二：分行业标准化 ---
        else:
            print(
                f"  执行分行业标准化 (主行业: {industry_config['primary_level']}, 回溯至: {industry_config['fallback_level']})...")

            # Rank法通常在全市场进行才有意义，分行业Rank后不同行业的序无法直接比较。
            # 这里我们约定，分行业标准化主要针对Z-Score。
            if method == 'rank':
                print("    警告：分行业Rank标准化逻辑复杂且不常用，将执行全市场Rank标准化。")
                return factor_data.apply(self._rank_series, axis=1)

            processed_data = {}
            for date in factor_data.index:
                daily_factor_series = factor_data.loc[date].dropna()
                if daily_factor_series.empty:
                    processed_data[date] = pd.Series(dtype=float)
                    log_warning(f"标准化过程中，发现当天{date}所有股票因子值都为空")
                    continue

                # 在循环内部，为每一天获取正确的历史地图
                daily_industry_map = pit_industry_map.get_map_for_date(date)
                processed_data[date] = self._standardize_cross_section_fallback(
                    daily_factor_series=daily_factor_series,
                    daily_industry_map=daily_industry_map,
                    config=industry_config
                )

            result_df = pd.DataFrame.from_dict(processed_data, orient='index')
            return result_df.reindex(index=factor_data.index, columns=factor_data.columns)

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

        # #临时的 记得删除
        # if target_factor_name in ['bm_ratio', 'ep_ratio', 'sp_ratio','beta']:
        #     if 'market_cap' in final_list:
        #         final_list.remove('market_cap')
        logger.info(f"最终用于回归的中性化目标因子为: {final_list}\n")
        return final_list
# 模拟一个更真实的、包含历史变更的行业隶属关系数据
def mock_full_historical_industry_data():
    """
    模拟 index_member_all 的全量历史返回
    - S3 在 2023-02-01 从 L2_A1 变更到 L2_A2
    - S6 早期存在，但在 2023-01-15 被剔除
    """
    data = [
        # S1, S2, S4, S5 保持不变
        ['L1_A', 'L2_A1', 'S1', '20200101', None],
        ['L1_A', 'L2_A1', 'S2', '20200101', None],
        ['L1_B', 'L2_B1', 'S4', '20200101', None],
        ['L1_B', 'L2_B1', 'S5', '20200101', None],
        # S3 的变更历史
        ['L1_A', 'L2_A1', 'S3', '20200101', '20230131'], # 旧的隶属关系，在31日结束
        ['L1_A', 'L2_A2', 'S3', '20230201', None],       # 新的隶属关系，从2月1日开始
        # S6 的历史
        ['L1_C', 'L2_C1', 'S6', '20200101', '20230115'],
    ]
    columns = ['l1_code', 'l2_code', 'ts_code', 'in_date', 'out_date']
    df = pd.DataFrame(data, columns=columns)
    df['in_date'] = pd.to_datetime(df['in_date'])
    # out_date 为 None 的表示至今有效，为了便于比较，我们用一个未来的日期代替
    df['out_date'] = pd.to_datetime(df['out_date']).fillna(pd.Timestamp(permanent__day))
    return df

from bisect import bisect_right



if __name__ == '__main__':
    # 1. 获取全量历史行业数据
    raw_industry_df = mock_full_historical_industry_data()

    # 2. 一次性构建PIT查询引擎
    pit_map_engine = PointInTimeIndustryMap(raw_industry_df)

    # 3. 准备因子数据，日期跨越S3的行业变更日
    factor_data_df = pd.DataFrame({
        'S1': [0.01, 0.02, 0.03],
        'S2': [0.03, 0.04, 0.05],
        'S3': [5.00, 0.01, 6.00],  # S3在 2023-01-31 和 2023-02-01 的极端值
        'S4': [-4.0, 0.05, 0.06],
        'S5': [0.06, 0.07, 0.08]
    }, index=pd.to_datetime(['2023-01-31', '2023-02-01', '2023-02-02']))

    # 4. 准备配置和QuantDeveloper实例
    app_config = {
        'preprocessing': {'winsorization': {
            'method': 'mad', 'mad_threshold': 3.0,
            'by_industry': {'primary_level': 'l2_code', 'fallback_level': 'l1_code', 'min_samples': 2}
        }}}
    developer = FactorProcessor(config=app_config)

    # 5. 执行去极值
    winsorized_df = developer.winsorize_robust(factor_data = factor_data_df, pit_industry_map=pit_map_engine)

    print("\n--- 原始因子数据 ---\n", factor_data_df)
    print("\n--- 去极值后因子数据 ---\n", winsorized_df)

    # 6. 验证 S3 在不同日期的处理逻辑
    print("\n--- 验证S3的行业归属和处理逻辑 ---")
    s3_val_before = winsorized_df.loc['2023-01-31', 'S3']
    s3_val_after = winsorized_df.loc['2023-02-01', 'S3']

    # 在2023-01-31，S3属于L2_A1，该组有S1,S2,S3三只股票，样本足够，使用组内数据
    print(f"2023-01-31, S3(5.0) 属于 L2_A1, 组员[S1,S2,S3], 因子[0.01,0.03,5.0], 处理后值为: {s3_val_before:.4f}")

    # 在2023-02-01，S3变更到L2_A2，该组只有它自己，样本不足，回溯到L1_A
    # L1_A 当天有 S1,S2,S3，因子值为 [0.02, 0.04, 0.01]，用这组的统计量来处理S3
    print(
        f"2023-02-01, S3(0.01) 属于 L2_A2(小样本), 回溯至 L1_A, 组员[S1,S2,S3], 因子[0.02,0.04,0.01], 处理后值为: {s3_val_after:.4f}")
