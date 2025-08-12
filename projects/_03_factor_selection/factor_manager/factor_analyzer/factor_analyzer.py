"""
单因子测试框架 - 专业版

实现华泰证券标准的三种单因子测试方法：
1. IC值分析法
2. 分层回测法  
3. Fama-MacBeth回归法（黄金标准）

支持批量测试、结果可视化和报告生成
"""
import os
import sys
import warnings
from datetime import datetime
from functools import partial
from typing import Callable, Any, Optional
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from scipy import stats

from projects._03_factor_selection.config.config_file.load_config_file import is_debug
from projects._03_factor_selection.factor_manager.factor_manager import FactorResultsManager
from projects._03_factor_selection.utils.factor_processor import FactorProcessor, PointInTimeIndustryMap
from projects._03_factor_selection.visualization_manager import VisualizationManager
from quant_lib import logger
from quant_lib.config.logger_config import log_flow_start

n_metrics_pass_rate_key = 'n_metrics_pass_rate'

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from quant_lib.evaluation import (
    calculate_ic,
    calculate_quantile_returns, fama_macbeth, calculate_turnover,
    calcu_forward_returns_close_close, calcu_forward_returns_open_close, quantile_stats_result,
    calculate_quantile_daily_returns

)

# # 导入新的可视化管理器
# try:
#     from visualization_manager import VisualizationManager
# except ImportError:
#     VisualizationManager = None

warnings.filterwarnings('ignore')


# 使用时 注意shift
def prepare_industry_dummies(
        pit_map: PointInTimeIndustryMap,
        trade_dates: pd.DatetimeIndex,
        stock_pool: list,
        level: str = 'l1_code',  # 接收来自配置的行业级别
        drop_first: bool = True  # <--- 新增一个参数，默认为True

) -> Dict[str, pd.DataFrame]:
    """
    根据指定的行业级别，从PointInTimeIndustryMap生成行业哑变量DataFrame字典。

    Args:
        pit_map: 预处理好的PointInTimeIndustryMap实例。
        trade_dates: 整个回测区间的交易日索引。
        stock_pool: 整个回测区间的股票池列表。
        level: 'l1_code' 或 'l2_code'，指定行业级别。

    Returns:
        一个字典，键为 'industry_行业代码'，值为对应的哑变量DataFrame (index=date, columns=stock)。
    """
    print(f"  正在基于 {level} 生成行业哑变量...")

    # 1. 构建一个包含所有日期和股票的“长格式”行业分类表
    all_daily_maps = []
    for date in trade_dates:
        daily_map = pit_map.get_map_for_date(date)
        if not daily_map.empty:
            daily_map = daily_map.reset_index()
            daily_map['date'] = date
            all_daily_maps.append(daily_map)

    if not all_daily_maps:
        return {}

    long_format_df = pd.concat(all_daily_maps)

    # 2. 使用 pd.get_dummies 高效生成哑变量
    # prefix='industry' 会自动给新生成的列加上 'industry_' 前缀
    dummies = pd.get_dummies(
        long_format_df[level],
        prefix='industry',
        dtype=float,
        drop_first=drop_first  # <--- 应用这个参数
    )
    dummy_df = pd.concat([long_format_df[['date', 'ts_code']], dummies], axis=1)

    # ======================= 侦探工具 #1 开始 =======================
    # 检查在 dummy_df 中是否存在 (date, ts_code) 的重复
    duplicates_mask = dummy_df.duplicated(subset=['date', 'ts_code'], keep=False)

    if duplicates_mask.any():
        print("‼️  找到了导致 pivot 失败的重复记录！详情如下：")

        # 筛选出所有重复的记录
        problematic_entries = dummy_df[duplicates_mask]

        # 为了看得更清楚，我们把原始的行业代码也加回来
        problematic_entries_with_industry = problematic_entries.merge(
            long_format_df[['date', 'ts_code', level]],
            on=['date', 'ts_code']
        )

        # 按照股票和日期排序，方便观察
        print(problematic_entries_with_industry.sort_values(by=['ts_code', 'date']))
    # 3. 将长格式的哑变量表转换为我们需要的“字典 of 宽格式DataFrame”
    # 这是性能关键点，避免在循环中重复透视
    dummy_dfs = {}

    # 获取所有哑变量的列名，例如 ['industry_801010.SI', 'industry_801020.SI', ...]
    industry_cols = [col for col in dummy_df.columns if col.startswith('industry_')]

    for col in industry_cols:
        # 使用 pivot 操作将每个行业哑变量列转换为 Date x Stock 的矩阵
        # fill_value=0 确保没有该公司没有该行业分类时，值为0
        pivoted_df = dummy_df.pivot(index='date', columns='ts_code', values=col).fillna(0)

        # 确保返回的DataFrame的索引和列与因子数据完全一致
        dummy_dfs[col] = pivoted_df.reindex(index=trade_dates, columns=stock_pool).fillna(0)

    print(f"  成功生成 {len(dummy_dfs)} 个 {level} 级别的行业哑变量。")
    return dummy_dfs


class FactorAnalyzer:
    """
    单因子(质检中心 IC分析、分层回测、F-M回归、绘图等）

    按照华泰证券标准实现三种测试方法的完整流程
    """

    def __init__(self,
                 factor_manager,
                 target_factors_dict: Dict[str, pd.DataFrame] = None,
                 target_factors_category_dict: Dict[str, Any] = None,
                 target_factor_school_type_dict: Dict[str, Any] = None

                 ):
        """
        初始化单因子测试器 -
        """
        # 必要检查
        if not factor_manager:
            raise RuntimeError("config 没有传递过来！")
        self.factor_manager = factor_manager
        data_manager = factor_manager.data_manager
        if self.factor_manager.data_manager is None or 'close' not in self.factor_manager.data_manager.raw_dfs:
            raise ValueError('close的df是必须的，请写入！')

        config = data_manager.config
        self.config = data_manager.config
        self.test_common_periods = self.config['evaluation'].get('forward_periods', [1, 5, 10, 20])
        self.n_quantiles = self.config.get('quantiles', 5)
        # 初始化因子预处理器
        self.factor_processor = FactorProcessor(self.config)
        self.factorResultsManager = FactorResultsManager()
        self.stock_pools_dict = data_manager.stock_pools_dict

        # 初始化数据

        self.target_factors_dict = target_factors_dict
        self.target_factors_style_category_dict = target_factors_category_dict
        self.target_school_type_dict = target_factor_school_type_dict

        self.backtest_start_date = config['backtest']['start_date']
        self.backtest_end_date = config['backtest']['end_date']
        self.backtest_period = f"{pd.to_datetime(self.backtest_start_date).strftime('%Y%m%d')} ~ {pd.to_datetime(self.backtest_end_date).strftime('%Y%m%d')}"
        self.visualizationManager = VisualizationManager(
            output_dir='D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\workspace\\visualizations'
        )

        # 决定延迟加载
        # self.master_beta_df = self.prepare_master_pct_chg_beta_dataframe()

        # 基于不同股票池！！！
        # self.close_df_diff_stock_pools_dict = self.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='close',need_shift=False)  # 只需要对齐股票就行 dict
        # self.circ_mv__shift_diff_stock_pools_dict = self.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='circ_mv',need_shift=True)
        # self.pct_chg_beta_shift_diff_stock_pools_dict = self.build_df_dict_base_on_diff_pool_can_set_shift( base_dict=self.get_pct_chg_beta_dict(), factor_name='pct_chg', need_shift=True)

        # 准备辅助【市值、行业】数据(用于中性值 计算！)
        # self.auxiliary_dfs_shift_diff_stock_polls_dict = self.build_auxiliary_dfs_shift_diff_stock_pools_dict()
        # self.prepare_for_neutral_dfs_shift_diff_stock_pools_dict = self.prepare_for_neutral_data_dict_shift_diff_stock_pools()
        # self.check_shape()

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

    def test_ic_analysis(self,
                         factor_data: pd.DataFrame,
                         returns_calculator: Callable[[int, pd.DataFrame, pd.DataFrame], pd.DataFrame],  # 具体化Callable
                         close_df: pd.DataFrame,
                         factor_name: str) -> Tuple[Dict[str, Series], Dict[str, pd.DataFrame]]:
        """
        IC值分析法测试

        Args:
            factor_data: 预处理后的因子数据
            factor_name: 因子名称


        Returns:
            IC分析结果字典
        """

        ic_series_periods_dict, stats_periods_dict = calculate_ic(factor_data, close_df,
                                                                  forward_periods=self.test_common_periods,
                                                                  method='spearman',
                                                                  returns_calculator=returns_calculator, min_stocks=10)

        return ic_series_periods_dict, stats_periods_dict

    def test_quantile_backtest(self,
                               factor_data: pd.DataFrame,
                               returns_calculator: Callable[[int, pd.DataFrame, pd.DataFrame], pd.DataFrame],
                               close_df: pd.DataFrame,
                               factor_name: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        分层回测法测试

        Args:
            factor_data: 预处理后的因子数据
            factor_name: 因子名称

        Returns:
            分层回测结果字典
        """
        # a) 获取分层【周期】收益率的时间序列
        quantile_returns_periods_dict = calculate_quantile_returns(
            factor_data,
            returns_calculator,
            close_df,
            n_quantiles=self.n_quantiles,
            forward_periods=self.test_common_periods
        )

        quantile_returns_periods_dict, quantile_stats_periods_dict = quantile_stats_result(
            quantile_returns_periods_dict, self.n_quantiles)

        return quantile_returns_periods_dict, quantile_stats_periods_dict

    def test_turnover_result(self, factor_data):
        logger.info("    > 正在计算因子换手率...")
        turnover_series_periods_dict = calculate_turnover(
            factor_df=factor_data,
            n_quantiles=self.n_quantiles,
            forward_periods=self.test_common_periods
        )
        turnover_stats_periods_dict = {}
        for period, turnover_series in turnover_series_periods_dict.items():
            turnover_stats_periods_dict[period] = {
                'turnover_mean': turnover_series.mean(),  # 周期平均换手率
                'turnover_annual': turnover_series.mean() * (252 / int(period[:-1]))  # 年化换手率
                ##
                # 假设平均每天10%的股票变动分位组数。
                # 周期10天
                # 年化算出来25.2
                # 资金来回滚动25次 有点费税费！#
            }
        return turnover_stats_periods_dict

    # def test_fama_macbeth(self,
    #                       factor_data: pd.DataFrame,
    #                       close_df: pd.DataFrame,
    #                       neutral_dfs: dict[str, pd.DataFrame],
    #                       circ_mv_df: pd.DataFrame,
    #                       factor_name: str) -> Tuple[Dict[str, pd.DataFrame],Dict[str, pd.DataFrame]]:
    #     return test_fama_macbeth(factor_data=factor_data, close_df=close_df, neutral_dfs=neutral_dfs,
    #                       circ_mv_df=circ_mv_df,
    #                       factor_name=factor_name)
    # 纯测试结果
    def comprehensive_test(self,
                           target_factor_name: str = None,
                           factor_df: pd.DataFrame = None,
                           preprocess_method: str = "standard",
                           returns_calculator: Callable[[int, pd.DataFrame, pd.DataFrame], pd.DataFrame] = None,
                           start_date: str = None, end_date: str = None,
                           need_process_factor: bool = True,
                           do_ic_test: bool = True, do_turnover_test: bool = True, do_quantile_test: bool = True,
                           do_fama_test: bool = True, do_style_correlation_test: bool = True,

                           ) -> Tuple[
        pd.DataFrame,
        Optional[Dict[str, pd.Series]],
        Optional[Dict[str, pd.DataFrame]],
        Optional[Dict[str, pd.DataFrame]],
        Optional[Dict[str, pd.DataFrame]],
        Optional[ pd.DataFrame],
        Optional[Dict[str, pd.DataFrame]],
        Optional[Dict[str, pd.DataFrame]],
        Optional[Dict[str, pd.DataFrame]],
        Optional[Dict[str, pd.DataFrame]],
        Optional[Dict[str, float]]
    ]:
        """
        综合测试 - 执行所有三种测试方法

        Args:
        Returns:
            综合测试结果字典
        """
        # if (not need_process_factor) & (not is_debug):
        #     raise ValueError("圣餐环境 不能跳过中xinghua")
        logger.info(f"开始测试因子: {target_factor_name}")
        # target_school = self.factor_manager.get_school_code_by_factor_name(target_factor_name)

        (final_neutral_dfs, style_category, pit_map
         ) = self.prepare_date_for_process_factor(target_factor_name, factor_df)
        if need_process_factor:
            # 1. 因子预处理
            factor_df = self.factor_processor.process_factor(
                target_factor_df=factor_df,
                target_factor_name=target_factor_name,
                neutral_dfs=final_neutral_dfs,  # <--- 传入权威的中性化数据篮子
                style_category=style_category,
                pit_map=pit_map,
                need_standardize = False
            )

        # 数据准备
        close_df, open_df, circ_mv_df, style_factor_dfs = self.prepare_date_for_core_test(target_factor_name)
        status_text = "需要处理" if need_process_factor else "不需要处理"
        log_flow_start(
            f"因子 {target_factor_name}（{status_text}）经过预处理之后，进入 core_three_test 测试"
        )

        ic_s, ic_st, q_r,q_daily_returns_df, q_st, turnover, fm_returns_series_dict, fm_t_stats_series_dict, fm_summary_dict, style_correlation_dict \
            = self.core_three_test(
            factor_df, target_factor_name, open_df, returns_calculator, close_df,
            final_neutral_dfs, circ_mv_df, style_factor_dfs, do_ic_test,
            do_turnover_test,
            do_quantile_test, do_fama_test, do_style_correlation_test)

        return factor_df, ic_s, ic_st, q_r,q_daily_returns_df, q_st, turnover, fm_returns_series_dict, fm_t_stats_series_dict, fm_summary_dict, style_correlation_dict

    # ok
    def _prepare_dfs_dict_by_diff_stock_pool(self, factor_names) -> Dict[str, pd.DataFrame]:
        """准备辅助数据（市值、行业等）"""
        ret_dict = {}
        for stock_poll_name, in self.stock_pools_dict.items():
            ret_dict[stock_poll_name] = {}

            for factor_name in factor_names:
                ret_dict[stock_poll_name].update(self.factor_manager.data_manager.raw_dfs[factor_name])

        return ret_dict

    # ok

    def _prepare_for_neutral_data(self, total_mv_df: pd.DataFrame, industry_df: pd.DataFrame) -> Dict[
        str, pd.DataFrame]:
        """
        准备辅助数据（市值、行业等）
         1. 通过 stack(dropna=False) 修复了因 shift(1) 导致首日索引丢失的问题。
         2. 修复了对包含NaN的序列进行排序时产生的TypeError。
         """
        # 假设 industry_df 传入时已经是 shift(1) 之后的结果
        industry_df = industry_df.replace([None, 'NONE', 'None'], np.nan)
        neutral_dict = {}
        logger.info("开始准备行业哑变量df，用于后续中性化使用...")

        # --- 1. 市值数据 ---
        neutral_dict['small_cap'] = total_mv_df

        # --- 2. 行业数据处理 ---

        # 使用 stack(dropna=False) 保留第一天的 NaN 数据行，确保索引的完整性。
        industry_stacked_series = industry_df.stack(dropna=False)

        # 【核心修正】: 从包含了NaN的Series中，安全地获取所有唯一的、非NaN的行业名称
        # 1. 先获取唯一值，结果可能包含 np.nan
        unique_values_with_nan = industry_stacked_series.unique()
        # 2. 过滤掉 np.nan。注意要用 pd.notna() 来判断，因为 np.nan != np.nan
        all_industries = sorted([str(ind) for ind in unique_values_with_nan if pd.notna(ind)])

        if not all_industries:
            print("警告：行业数据中未发现任何有效行业名称。")
            return neutral_dict

        industry_categorical_series = industry_stacked_series.astype(
            pd.CategoricalDtype(categories=all_industries)
        )

        # get_dummies 对于 NaN 输入，会生成全为0的哑变量行，这是我们期望的行为
        industry_dummies_df = pd.get_dummies(industry_categorical_series,
                                             prefix='industry',
                                             sparse=True)

        final_industry_dummies = {}

        # --- 3. 逐个行业哑变量进行 unstack 和存储 ---
        for col_name in industry_dummies_df.columns:
            # col_name 是 'industry_TMT', 'industry_医药' 等
            # 从 industry_dummies_df 中取出对应列，它是一个 Series，其索引仍然是 (日期, 股票代码) MultiIndex
            one_industry_series = industry_dummies_df[col_name]
            dummy_df_for_one_industry = one_industry_series.unstack(level=-1)

            # 使用原始 industry_df 的列（所有股票）来 reindex
            # 这会把在 stack() 过程中丢失的股票列加回来
            # fill_value=0 的意思是，这些被加回来的股票，它们在这个行业哑变量中的值是0
            dummy_df_for_one_industry = dummy_df_for_one_industry.reindex(
                columns=industry_df.columns,
                fill_value=0
            )

            final_industry_dummies[col_name] = dummy_df_for_one_industry

        # --- 4. 删除基准行业列 ---
        if len(all_industries) > 1:
            base_industry = all_industries[0]
            base_industry_col_name = f'industry_{base_industry}'
            if base_industry_col_name in final_industry_dummies:
                del final_industry_dummies[base_industry_col_name]

        # --- 5. 处理 NaN：将原始行业为 NaN 的位置对应的哑变量设为 NaN ---
        # 这一步非常重要，它确保了因为 shift(1) 产生的首日NaN，在哑变量矩阵中也被正确地标记为NaN
        nan_mask_df = industry_df.isna()

        for industry_col_key, dummy_df in final_industry_dummies.items():
            # dummy_df.mask(nan_mask_df) 会将 nan_mask_df 中为 True 的位置，在 dummy_df 中也设为 NaN
            final_industry_dummies[industry_col_key] = dummy_df.mask(nan_mask_df)
            final_industry_dummies[industry_col_key] = final_industry_dummies[industry_col_key].astype(float)

        neutral_dict.update(final_industry_dummies)

        # 此时，neutral_dict 中所有DataFrame的索引都与原始的 industry_df 完全一致，包含了第一天
        return neutral_dict

    def evaluation_score_dict(self,
                              ic_stats_periods_dict,
                              quantile_stats_periods_dict,
                              fm_stat_results_periods_dict
                              ) -> Dict[str, Any]:

        ret = {}
        for period in self.test_common_periods:
            ret[f'{period}d'] = self._evaluate_factor_score(f'{period}d', ic_stats_periods_dict,
                                                            quantile_stats_periods_dict,
                                                            fm_stat_results_periods_dict)
        return ret

    def _evaluate_factor_score(self,
                               main_period: str,
                               ic_results: Dict,
                               quantile_results: Dict,
                               fm_results: Dict) -> Dict[str, Any]:
        """
        综合评价因子表现
        """

        # IC评价
        ic_main = ic_results.get(main_period, {})
        cal_score_ic = self.cal_score_ic(ic_main.get('ic_mean'),
                                         ic_main.get('ic_ir'),
                                         ic_main.get('ic_win_rate'),
                                         ic_main.get('ic_p_value')
                                         )

        # 分层回测评价
        quantile_main = quantile_results.get(main_period, {})

        cal_score_quantile = self.cal_score_quantile_performance(quantile_main)

        # Fama-MacBeth评价
        fm_main = fm_results.get(main_period, {})
        cal_score_fama_macbeth = self.cal_score_fama_macbeth(fm_main)

        # 综合评分
        cal_score_factor_holistically = self.cal_score_factor_holistically(cal_score_ic, cal_score_quantile,
                                                                           cal_score_fama_macbeth)
        return cal_score_factor_holistically

    # 这是最原始的评测，很不准！ 不要参考score，只看看底层的基本数据就行！ 最后的calculate_factor_score是最权威的
    def overrall_summary(self, results: Dict[str, Any]):

        ic_analysis_dict = results['ic_analysis']
        quantile_backtest_dict = results['quantile_backtest']
        fama_macbeth_dict = results['fama_macbeth']
        evaluation_dict = results['evaluate_factor_score']
        rows = []
        total_score = []
        flatten_metrics_dict = {}

        for day, evaluation in evaluation_dict.items():
            cur_total_score = evaluation['final_score']
            total_score.append(cur_total_score)
            # 扁平化的核心指标字段
            flatten_metrics_dict[f'{day}_综合评分'] = cur_total_score
            sub = evaluation['sub']

            row = {

                '持有期': day,
                f'{day}_综合评分': cur_total_score,
                '总等级': evaluation['final_grade'],
                '结论': evaluation['conclusion'],
                #
                f'IC_{day}内部多指标通过率': sub['IC'][n_metrics_pass_rate_key],
                f'Quantile_{day}内部多指标通过率': sub['Quantile'][n_metrics_pass_rate_key],
                f'FM_{day}内部多指标通过率': sub['Fama-MacBeth'][n_metrics_pass_rate_key],

                f'IC_{day}内部多指标评级': sub['IC']['grade'],
                f'Quantile_{day}内部多指标评级': sub['Quantile']['grade'],
                f'FM_{day}内部多指标评级': sub['Fama-MacBeth']['grade'],

                'IC分析摘要': ic_analysis_dict[day],
                'Quantile分析摘要': quantile_backtest_dict[day],
                'FM分析摘要': fama_macbeth_dict[day]
            }
            merged_row = {**row}
            rows.append(merged_row)
        backtest_period = f'{self.backtest_start_date}~{self.backtest_end_date}'
        return {results['factor_name']:
                    {'测试日期': results['test_date'],
                     '回测周期': backtest_period,
                     'best_score': max(total_score), **flatten_metrics_dict,
                     'diff_day_perform': rows}}

    def cal_score_ic(self,
                     ic_mean: float,
                     ic_ir: float,
                     ic_win_rate: float,
                     ic_p_value: float) -> Dict:
        """
        【专业版】对IC进行多维度、分级、加权评分
        """

        # 1. 科学性检验 (准入门槛)
        is_significant = ic_p_value is not None and ic_p_value < 0.05

        # 2. 分级评分
        icir_score = 0
        if abs(ic_ir) > 0.5:
            icir_score = 2
        elif abs(ic_ir) > 0.3:
            icir_score = 1

        mean_score = 1 if abs(ic_mean) > 0.025 else 0
        win_rate_score = 1 if ic_win_rate > 0.55 else 0

        # 3. 计算总分 (满分4分)
        total_score = icir_score + mean_score + win_rate_score

        # 4. 生成评级和结论
        if not is_significant:
            grade = "D (不显著)"
            conclusion = "因子未通过显著性检验，结果可能由运气导致，不予采纳。"
        elif total_score == 4:
            grade = "A+ (优秀（100%指标达到）)"
            conclusion = "所有指标均表现优异，是顶级的Alpha因子。"
        elif total_score == 3:
            grade = "A (良好（75%指标达到）)"
            conclusion = "核心指标表现良好，具备很强的实战价值。"
        elif total_score == 2:
            grade = "B (及格（50%指标达到）)"
            conclusion = "部分指标达标，因子具备一定有效性，可作为备选。"
        else:
            grade = "C (较差)"
            conclusion = "核心指标表现不佳，建议优化或放弃。"

        return {
            'n_metrics_pass_rate': total_score / 4,
            'grade': grade,
            'is_significant': is_significant,
            'details': {
                'ICIR': f"{ic_ir:.2f} (得分:{icir_score})/(共计两分。一分就也很不错了)",
                'IC Mean': f"{ic_mean:.3f} (得分:{mean_score})",
                'Win Rate': f"{ic_win_rate:.2%} (得分:{win_rate_score})"
            },
            'conclusion': conclusion
        }

    def cal_score_quantile_performance(self, quantile_main: Dict) -> Dict[str, Any]:
        """
        【专业版】对分层回测结果进行多维度、分级、加权评分。
        """
        # --- 1. 提取核心指标 ---
        quantile_means = quantile_main.get('quantile_means', [])
        tmb_sharpe = quantile_main.get('tmb_sharpe', 0)
        tmb_annual_return = quantile_main.get('tmb_annual_return', 0)
        # 最大回撤通常是负数，我们取绝对值
        max_drawdown = abs(quantile_main.get('max_drawdown', 1.0))

        # --- 2. 分级评分 ---

        # a) TMB夏普评分 (核心指标, 满分2分)
        sharpe_score = 0
        if tmb_sharpe > 1.0:
            sharpe_score = 2
        elif tmb_sharpe > 0.5:
            sharpe_score = 1

        # b) 单调性评分 (结构指标, 满分1分)
        monotonicity_score = 0
        monotonicity_corr = np.nan
        if quantile_means and len(quantile_means) > 1:
            # 使用spearman秩相关系数计算单调程度
            monotonicity_corr, _ = stats.spearmanr(quantile_means, range(len(quantile_means)))
            if monotonicity_corr > 0.8:
                monotonicity_score = 1

        # c) 收益/风控评分 (实战指标, 满分2分)
        # 计算卡玛比率 (Calmar Ratio)
        calmar_ratio = tmb_annual_return / max_drawdown if max_drawdown > 0 else 0

        risk_return_score = 0
        if calmar_ratio > 0.5 and tmb_annual_return > 0.05:
            risk_return_score = 2
        elif calmar_ratio > 0.2 and tmb_annual_return > 0.03:
            risk_return_score = 1

        # --- 3. 汇总与评级 (总分5分) ---
        total_score = sharpe_score + monotonicity_score + risk_return_score

        if total_score == 5:
            grade = "A+ (强烈推荐（100%指标达到）)"
        elif total_score == 4:
            grade = "A (优秀（80%指标达到）)"
        elif total_score == 3:
            grade = "B (良好-（60%指标达到）)"
        elif total_score == 2:
            grade = "C (一般（40%指标达到）)"
        else:
            grade = "D (一般（0%~40%）)"

        return {
            'n_metrics_pass_rate': total_score / 5,
            'grade': grade,
            'details': {
                'TMB Sharpe': f"{tmb_sharpe:.2f} (得分:{sharpe_score})",
                'Monotonicity Corr': f"{monotonicity_corr:.2f} (得分:{monotonicity_score})",
                'Calmar Ratio': f"{calmar_ratio:.2f} (得分:{risk_return_score})",
                'TMB Annual Return': f"{tmb_annual_return:.2%}",
                'Max Drawdown': f"{max_drawdown:.2%}"
            },
            'conclusion': grade

        }

    def cal_score_fama_macbeth(self, fm_main: Dict) -> Dict[str, Any]:
        """
        【专业版】对Fama-MacBeth回归进行多维度、分离式评分
        """
        # --- 1. 提取核心指标 ---
        n_metrics_pass_rate = 0
        t_stat = fm_main.get('t_statistic', 0)
        mean_return = fm_main.get('mean_factor_return', 0)  # 这是周期平均收益
        num_periods = fm_main.get('num_valid_periods', 0)
        success_rate = fm_main.get('success_rate', 0)
        factor_returns_series = fm_main.get('factor_returns_series', pd.Series(dtype=float))

        # --- 2. 分离式评分 ---

        # a) 测试可信度评分 (满分3分)
        confidence_score = 0
        # 完整度评分 (0-1分)
        if success_rate >= 0.8: confidence_score += 1
        # 周期长度评分 (0-2分)
        if num_periods >= 252 * 3:
            confidence_score += 2
        elif num_periods >= 252:
            confidence_score += 1

        # b) 因子有效性评分 (满分5分)
        performance_score = 0
        # 显著性评分 (0-3分)
        if not np.isnan(t_stat):
            if abs(t_stat) > 2.58:
                performance_score += 3
            elif abs(t_stat) > 1.96:
                performance_score += 2
            elif abs(t_stat) > 1.64:
                performance_score += 1

        # 收益稳定性评分 (Lambda胜率, 0-1分)
        lambda_win_rate = 0
        if not factor_returns_series.empty:
            if factor_returns_series.mean() >= 0:
                lambda_win_rate = (factor_returns_series > 0).mean()
            else:
                lambda_win_rate = (factor_returns_series < 0).mean()
            if lambda_win_rate > 0.55:
                performance_score += 1

        # 经济意义评分 (年化收益, 0-1分)
        # 假设 daily period_len = 1, weekly = 5, etc.
        # 这里我们简单假设一个周期是252/num_periods年
        annualized_return = mean_return * (252 / self.test_common_periods[0])  # 简化处理，假设周期固定
        if abs(annualized_return) > 0.03:  # 年化因子收益超过3%
            performance_score += 1

        final_grade = "D"
        conclusion = "因子表现不佳。"
        # 综合评级 - --
        # a) 设立“一票否决”红线
        if confidence_score == 0 or success_rate <= 0.75:
            final_grade = "F (测试不可信)"
            conclusion = "测试质量完全不达标 (整理数据后 参与回归率过低：maybe：周期过短且数据不完整)。"
        else:
            # b) 根据表现分，确定基础评级
            base_grade = "D"
            if performance_score >= 4:
                base_grade = "A"
                conclusion = "因子表现优秀，具备很强的Alpha。"
            elif performance_score >= 3:
                base_grade = "B"
                conclusion = "因子表现良好，具备一定Alpha。"
            elif performance_score >= 2:
                base_grade = "C"
                conclusion = "因子表现一般，需进一步观察。"

            # c) 根据可信度分，进行调整并加注
            if confidence_score == 3:
                final_grade = base_grade + "+" if base_grade in ["A", "B"] else base_grade
                conclusion += " 测试结果具有极高可信度。"
            elif confidence_score == 2:
                final_grade = base_grade
                conclusion += " 测试结果可信度良好。"
            elif confidence_score == 1:
                # 可信度较低，下调评级
                if base_grade == "A":
                    final_grade = "B+"
                elif base_grade == "B":
                    final_grade = "C+"
                else:
                    final_grade = base_grade  # C和D不再下调
                conclusion += " [警告] 测试可信度较低，可能因周期较短，结论需谨慎对待。"
        grade_to_score_map = {
            'A+': 1.00,
            'A': 0.95,
            'B+': 0.90,
            'B': 0.80,
            'C+': 0.70,
            'C': 0.50,
            'D': 0.30,
            'F (测试不可信)': 0.00  # 明确处理F评级
        }
        n_metrics_pass_rate = grade_to_score_map.get(final_grade, 0.0)

        return {
            'n_metrics_pass_rate': n_metrics_pass_rate,
            'confidence_score': f"{confidence_score}/3",
            'performance_score': f"{performance_score}/5",
            'grade': final_grade,
            'conclusion': conclusion,
            'details': {
                't-statistic': f"{t_stat:.2f}",
                'Annualized Factor Return': f"{annualized_return:.2%}",
                'Lambda Win Rate': f"{lambda_win_rate:.2%}",
                'Valid Periods': num_periods,
                'Regression Success Rate': f"{success_rate:.2%}"
            }
        }

    def cal_score_factor_holistically(self,
                                      score_ic_eval: Dict,
                                      score_quantile_eval: Dict,
                                      score_fm_eval: Dict) -> Dict[str, Any]:
        """
        【最终投决会】综合IC、分层回测、Fama-MacBeth三大检验结果，对因子进行最终评级。
        """

        # --- 1. 提取各模块的核心评价结果 ---
        ic_n_metrics_pass_rate = score_ic_eval.get(n_metrics_pass_rate_key, 0)
        quantile_n_metrics_pass_rate = score_quantile_eval.get(n_metrics_pass_rate_key, 0)
        fm_n_metrics_pass_rate = score_fm_eval.get(n_metrics_pass_rate_key, 0)
        ic_is_significant = score_ic_eval.get('is_significant', False)

        quantile_grade = score_quantile_eval.get('grade', 'D')
        tmb_sharpe = score_quantile_eval.get('details', {}).get('TMB Sharpe', '0 (得分:0)').split(' ')[0]

        fm_performance_score_str = score_fm_eval.get('performance_score', '0/5')
        fm_confidence_score_str = score_fm_eval.get('confidence_score', '0/3')
        fm_grade = score_fm_eval.get('grade', '')

        # --- 2. 解析数值分数 ---
        try:
            fm_performance_score = int(fm_performance_score_str.split('/')[0])
            fm_confidence_score = int(fm_confidence_score_str.split('/')[0])
            tmb_sharpe = float(tmb_sharpe)
        except (ValueError, IndexError):
            # 如果解析失败，说明子报告有问题，直接返回错误
            return {'final_grade': 'F (错误)', 'conclusion': '一个或多个子评估报告格式错误，无法进行综合评价。'}

        # --- 3. 执行“一票否决”规则 ---
        deal_breaker_reason = []
        if not ic_is_significant:
            deal_breaker_reason.append("IC检验不显著，因子有效性无统计学支持。")
        elif fm_grade == 'F':  # 最拉跨的分！
            deal_breaker_reason.append(f"Fama-MacBeth检验可信度低(得分:{fm_confidence_score}/3)，结果不可靠。")
        elif tmb_sharpe < 0:
            deal_breaker_reason.append(f"分层回测多空夏普比率为负({tmb_sharpe:.2f})，因子在策略层面无效。")

        if deal_breaker_reason:
            return {
                'final_grade': 'F (否决)',
                'final_score': '0/100',
                'conclusion': f"因子存在致命缺陷: {deal_breaker_reason}",
                'sub': {
                    'IC': {"grade": score_ic_eval.get('grade'), n_metrics_pass_rate_key: ic_n_metrics_pass_rate},
                    'Quantile': {"grade": quantile_grade, n_metrics_pass_rate_key: quantile_n_metrics_pass_rate},
                    'Fama-MacBeth': {"grade": fm_grade, n_metrics_pass_rate_key: fm_n_metrics_pass_rate}
                }
            }

        # --- 4. 进行加权评分 (总分100分) ---
        # 权重分配: 分层回测(40%), F-M(35%), IC(25%)
        ic_max_score = 4
        quantile_max_score = 5
        fm_max_score = 5

        weighted_score = (
                ic_n_metrics_pass_rate * 25 +
                quantile_n_metrics_pass_rate * 40 +
                fm_n_metrics_pass_rate * 35
        )

        # --- 5. 给出最终评级和结论 ---
        final_grade = ""
        conclusion = ""
        if weighted_score >= 85:
            final_grade = "S (旗舰级)"
            conclusion = "因子在所有维度均表现卓越，是极其罕见的顶级Alpha，应作为策略核心。 "
        elif weighted_score >= 70:
            final_grade = "A (核心备选)"
            conclusion = "因子表现非常优秀且稳健，具备极强的实战价值，可纳入核心多因子模型。"
        elif weighted_score >= 50:
            final_grade = "B (值得关注(50%-70%得分率))"
            conclusion = "因子表现良好，通过了关键考验，具备一定Alpha能力，可纳入备选池持续跟踪。"
        elif weighted_score >= 35:
            final_grade = "C (35%的得分率，实在走投无路了，只有看看这类垃圾了)"
            conclusion = "35%的得分率，实在走投无路了，只有看看这类垃圾了-只有35%~50%的得分率"
        elif weighted_score >= 20:
            final_grade = "D (很差很差-只有20%~35%的得分率)"
            conclusion = "很差很差只有20%~35%的得分率"
        else:
            final_grade = "E (建议优化)"
            conclusion = "因子表现实在平庸，建议优化或仅作为分散化补充。（20%的得分率都没有）"

        return {
            'final_grade': final_grade,
            'final_score': f"{weighted_score:.1f}/100",
            'conclusion': conclusion,

            'sub': {
                'IC': {"grade": score_ic_eval.get('grade'), n_metrics_pass_rate_key: ic_n_metrics_pass_rate},
                'Quantile': {"grade": quantile_grade, n_metrics_pass_rate_key: quantile_n_metrics_pass_rate},
                'Fama-MacBeth': {"grade": fm_grade, n_metrics_pass_rate_key: fm_n_metrics_pass_rate}
            }
        }

    # ok 因为需要滚动计算，所以不依赖股票池的index（trade） 只要对齐股票列就好
    def get_pct_chg_beta_dict(self):
        dict = {}
        for pool_name, _ in self.stock_pools_dict.items():
            beta_df = self.get_pct_chg_beta_data_for_pool(pool_name)
            dict[pool_name] = beta_df
        return dict

    # ok ok 注意 用的时候别忘了shift（1）

    def check_shape(self):
        pool_names = self.stock_pools_dict.keys()
        pool_shape_config = {}
        for pool_name in pool_names:
            pool_shape_config[pool_name] = self.stock_pools_dict[pool_name].shape

        for pool_name, shape in pool_shape_config.items():
            if shape != self.factor_manager.close_df_diff_stock_pools_dict[pool_name].shape:
                raise ValueError("形状不一致 ，请必须检查")
            if shape != self.factor_manager.circ_mv__shift_diff_stock_pools_dict[pool_name].shape:
                raise ValueError("形状不一致 ，请必须检查")
            if shape != self.factor_manager.pct_chg_beta_shift_diff_stock_pools_dict[pool_name].shape:
                raise ValueError("形状不一致 ，请必须检查")

            if shape != self.factor_manager.auxiliary_dfs_shift_diff_stock_polls_dict[pool_name]['pct_chg_beta'].shape:
                raise ValueError("形状不一致 ，请必须检查")
            if shape != self.factor_manager.auxiliary_dfs_shift_diff_stock_polls_dict[pool_name]['industry'].shape:
                raise ValueError("形状不一致 ，请必须检查")
            if shape != self.factor_manager.auxiliary_dfs_shift_diff_stock_polls_dict[pool_name]['total_mv'].shape:
                raise ValueError("形状不一致 ，请必须检查")
            #  因为_prepare_for_neutral_data  入参的df 第一行是NAN（shift导致）经过industry_stacked_series = industry_df.stack().dropna() ，最后会少一行，很正常！所以暂且不判断这个长度
            # if shape != self.prepare_for_neutral_dfs_shift_diff_stock_pools_dict[pool_name]['industry_农业综合'].shape:
            #     raise ValueError("形状不一致 ，请必须检查")

            if shape != self.factor_manager.prepare_for_neutral_dfs_shift_diff_stock_pools_dict[pool_name][
                'total_mv'].shape:
                raise ValueError("形状不一致 ，请必须检查")

    def test_factor_entity_service(self,
                                   factor_name: str,
                                   factor_df: pd.DataFrame,
                                   preprocess_method: str = "standard",
                                   need_process_factor: bool = True,
                                   is_composite_factor: bool = True
                                   ) -> Dict[str, Any]:
        """
        测试单个因子
        综合评分
        保存结果
        画图保存
        """
        start_date, end_date, target_school, stock_index, stock_pool_name, close_df, open_df, style_category, test_configurations = self.prepare_date_for_entity_service(
            factor_name)
        all_configs_results = {}
        if is_composite_factor:
           return  self.test_factor_entity_service_for_composite_factor(factor_name,factor_df,test_configurations,start_date,end_date,stock_index)
        for calculator_name, func in test_configurations.items():
            # 执行测试
            log_flow_start(f"因子{factor_name}原始状态 进入comprehensive_test测试 ")
            raw_factor_df, ic_s_raw, ic_st_raw, q_r_raw, q_daily_returns_df_raw,q_st_raw, _, _, _, _, _ = self.comprehensive_test(
                target_factor_name=factor_name,
                factor_df=factor_df,
                returns_calculator=func,
                preprocess_method="standard",
                start_date=start_date,
                end_date=end_date,
                need_process_factor=False,
                do_ic_test=True, do_quantile_test=True, do_turnover_test=False, do_fama_test=False,
                do_style_correlation_test=False
            )
            log_flow_start(f"因子{factor_name}原始状态 进入comprehensive_test测试 ")
            proceessed_df, ic_s, ic_st, q_r, q_daily_returns_df_proc, q_st, turnover, fm_returns_series_dict, fm_t_stats_series_dict, fm_summary_dict, style_correlation_dict \
                = self.comprehensive_test(
                target_factor_name=factor_name,
                factor_df=factor_df,
                returns_calculator=func,
                preprocess_method="standard",
                start_date=start_date,
                end_date=end_date,
                need_process_factor=True,
                do_ic_test=True, do_turnover_test=True, do_quantile_test=True, do_fama_test=True,
                do_style_correlation_test=True
            )
            single_config_results = {
                "raw_factor_df": raw_factor_df,
                "processed_factor_df": proceessed_df,
                "ic_series_periods_dict_raw": ic_s_raw,
                "ic_stats_periods_dict_raw": ic_st_raw,
                "ic_series_periods_dict_processed": ic_s,
                "ic_stats_periods_dict_processed": ic_st,

                "quantile_returns_series_periods_dict_raw": q_r_raw,
                "quantile_stats_periods_dict_raw": q_st_raw,
                "q_daily_returns_df_raw": q_daily_returns_df_raw,

                "quantile_returns_series_periods_dict_processed": q_r,
                "quantile_stats_periods_dict_processed": q_st,
                "q_daily_returns_df_processed": q_daily_returns_df_proc,

                "fm_returns_series_periods_dict": fm_returns_series_dict,
                "fm_stat_results_periods_dict": fm_summary_dict,
                "turnover_stats_periods_dict": turnover,
                "style_correlation_dict": style_correlation_dict
            }
            # b) 将本次配置的所有结果打包
            self.factorResultsManager._save_factor_results(  # 假设保存函数在FactorManager中
                factor_name=factor_name,
                stock_index=stock_index,
                start_date=start_date,
                end_date=end_date,
                returns_calculator_func_name=calculator_name,
                results=single_config_results
            )
            all_configs_results[calculator_name] = single_config_results
        # overrall_summary_stats = self.landing_for_core_three_analyzer_result(target_factor_df, target_factor_name,
        #                                                                      style_category, "standard",
        #                                                                      ic_s, ic_st, q_r, q_st, fm_r, fm_st, turnover_st, style_corr
        #                                                                      )

        return all_configs_results

    def purify_summary_rows_contain_periods(self, comprehensive_results):
        factor_category = comprehensive_results.get('factor_category', 'Unknown')  # 使用.get增加健壮性
        factor_name = comprehensive_results['factor_name']

        ic_stats_periods_dict = comprehensive_results['ic_analysis']
        quantile_stats_periods_dict = comprehensive_results['quantile_backtest']
        fm_stat_results_periods_dict = comprehensive_results['fama_macbeth']

        # 以 ic_stats 的 keys 为准，确保所有字典都有这些周期
        periods = ic_stats_periods_dict.keys()
        purify_summary_rows = []

        for period in periods:
            # 在循环内部进行防御性检查，确保所有结果字典都包含当前周期
            if not all(period in d for d in [quantile_stats_periods_dict, fm_stat_results_periods_dict]):
                print(f"警告：因子 {factor_name} 在周期 {period} 的结果不完整，已跳过。")
                continue

            summary_row = {
                'factor_name': factor_name,
                'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'factor_category': factor_category,
                'backtest_period': self.backtest_period,
                'backtest_base_on_index': comprehensive_results['backtest_base_on_index'],

                'period': period,  # 【BUG已修正】这里应该是单个周期
                # 收益维度
                'tmb_sharpe': quantile_stats_periods_dict[period]['tmb_sharpe'],
                'tmb_annual_return': quantile_stats_periods_dict[period]['tmb_annual_return'],
                # 风险与稳定性维度 (Risk & Stability Dimension) - 过程有多颠簸
                'tmb_max_drawdown': quantile_stats_periods_dict[period]['max_drawdown'],
                'ic_ir': ic_stats_periods_dict[period]['ic_ir'],
                # 纯净度与独特性维度 (Purity & Uniqueness Dimension) - “是真Alpha还是只是风险暴露
                'fm_t_statistic': fm_stat_results_periods_dict[period]['t_statistic'],
                'monotonicity_spearman': quantile_stats_periods_dict[period]['monotonicity_spearman'],
                'ic_mean': ic_stats_periods_dict[period]['ic_mean']
            }
            # score = calculate_factor_score(summary_row)
            # summary_row['score'] = score
            purify_summary_rows.append(summary_row)
        return purify_summary_rows

    def build_fm_return_series_dict(self, fm_factor_returns_series_periods_dict, target_factor_name):
        fm_factor_returns = {}

        for period, return_series in fm_factor_returns_series_periods_dict.items():
            colum_name = f'{target_factor_name}_{period}'
            fm_factor_returns[colum_name] = return_series

        return fm_factor_returns

    def batch_test_factors(self,
                           target_factors_dict: Dict[str, pd.DataFrame],
                           **test_kwargs) :
        """
        批量测试因子
        """

        # 批量测试
        results = []
        for factor_name, factor_data in target_factors_dict.items():
            try:
                # 执行测试
                results.append( {factor_name:(self.test_factor_entity_service(
                    factor_name=factor_name,
                    factor_df=factor_data,
                    need_process_factor=True,
                    is_composite_factor=False,
                ))})
            except Exception as e:
                raise ValueError(f"✗ 因子{factor_name}测试失败: {e}") from e

        return results

    def core_three_test(self, factor_df, target_factor_name, open_df,
                        returns_calculator: Callable[[int, pd.DataFrame, pd.DataFrame], pd.DataFrame],
                        close_df,
                        prepare_for_neutral_shift_base_own_stock_pools_dfs, circ_mv_shift_df, style_factors_dict,
                        do_ic_test, do_turnover_test, do_quantile_test, do_fama_test, do_style_correlation_test
                        ) -> tuple[
        dict[str, Series] | None, dict[str, DataFrame] | None, dict[str, DataFrame] | None, dict[str, DataFrame] | None,pd.DataFrame | None,
        dict[Any, Any] | None, dict[str, DataFrame] | None, dict[str, DataFrame] | None, dict[str, DataFrame] | None,
        dict[str, float] | None]:

        # 1. IC值分析
        logger.info("\t2. 正式测试 之 IC值分析...")
        ic_s = ic_st = q_r = q_st = turnover = fm_returns_series_dict = fm_t_stats_series_dict = fm_summary_dict = style_correlation_dict = None
        if do_ic_test:
            ic_s, ic_st = self.test_ic_analysis(factor_df,
                                                returns_calculator, close_df,
                                                target_factor_name)
        # 2. 分层回测
        logger.info("\t3.  正式测试 之 分层回测...")
        if do_quantile_test:
            # 这是中性化之后的分组收益，也就是纯净的单纯因子自己带来的收益。至于在真实的市场上，禁不禁得起考验，这个无法看出。需要在原始因子（未除杂/中性化），然后分组查看收益才行！
            q_r, q_st = self.test_quantile_backtest(
                factor_df, returns_calculator, close_df, target_factor_name)

        if do_turnover_test:
            turnover = self.test_turnover_result(factor_df)

        q_daily_returns_df = calculate_quantile_daily_returns(factor_df,returns_calculator,  5
                                                                                )
        # 3. Fama-MacBeth回归
        if do_fama_test:
            logger.info("\t4.  正式测试 之 Fama-MacBeth回归...")
            fm_returns_series_dict, fm_t_stats_series_dict, fm_summary_dict = fama_macbeth(
                factor_data=factor_df, returns_calculator=returns_calculator, close_df=close_df,
                forward_periods=self.test_common_periods,
                neutral_dfs={}, circ_mv_df=circ_mv_shift_df,
                factor_name=target_factor_name)

        # 【新增】4. 风格相关性分析
        logger.info("\t5.  正式测试 之 风格相关性分析...")
        if do_style_correlation_test:
            style_correlation_dict = self.test_style_correlation(
                factor_df,
                style_factors_dict
            )
        return ic_s, ic_st, q_r,q_daily_returns_df, q_st, turnover, fm_returns_series_dict, fm_t_stats_series_dict, fm_summary_dict, style_correlation_dict

    def landing_for_core_three_analyzer_result(self, target_factor_df, target_factor_name, category, preprocess_method,
                                               ic_series_periods_dict, ic_stats_periods_dict,
                                               quantile_daily_returns_for_plot_dict, quantile_stats_periods_dict,
                                               factor_returns_series_periods_dict, fm_stat_results_periods_dict,
                                               turnover_stats_periods_dict, style_correlation_dict):
        #  综合评价
        evaluation_score_dict = self.evaluation_score_dict(ic_stats_periods_dict,
                                                           quantile_stats_periods_dict,
                                                           fm_stat_results_periods_dict)
        # 整合结果
        comprehensive_results = {
            'factor_name': target_factor_name,
            'factor_category': category,
            'backtest_base_on_index': self.factor_manager.get_stock_pool_index_by_factor_name(target_factor_name),
            'backtest_period': self.backtest_period,
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'preprocess_method': preprocess_method,
            'ic_analysis': ic_stats_periods_dict,
            'quantile_backtest': quantile_stats_periods_dict,
            'fama_macbeth': fm_stat_results_periods_dict,
            'evaluate_factor_score': evaluation_score_dict
        }
        overrall_summary_stats = self.overrall_summary(comprehensive_results)
        purify_summary_rows_contain_periods = self.purify_summary_rows_contain_periods(comprehensive_results)
        fm_return_series_dict = self.build_fm_return_series_dict(factor_returns_series_periods_dict, target_factor_name)

        self.factor_manager._save_results(overrall_summary_stats, file_name_prefix='overrall_summary')
        self.factor_manager.update_and_save_factor_purify_summary(purify_summary_rows_contain_periods,
                                                                  file_name_prefix='purify_summary')
        self.factor_manager.update_and_save_fm_factor_return_matrix(fm_return_series_dict,
                                                                    file_name_prefix='fm_return_series')
        # 画图保存
        all_periods = ic_stats_periods_dict.keys()
        # 在所有计算结束后，只调用一次统一报告函数
        self.visualizationManager.plot_unified_factor_report(
            backtest_base_on_index=comprehensive_results['backtest_base_on_index'],
            factor_name=target_factor_name,
            ic_series_periods_dict=ic_series_periods_dict,
            ic_stats_periods_dict=ic_stats_periods_dict,
            quantile_returns_series_periods_dict=quantile_daily_returns_for_plot_dict,
            quantile_stats_periods_dict=quantile_stats_periods_dict,
            factor_returns_series_periods_dict=factor_returns_series_periods_dict,
            fm_stat_results_periods_dict=fm_stat_results_periods_dict,
            turnover_stats_periods_dict=turnover_stats_periods_dict,
            style_correlation_dict=style_correlation_dict,
            factor_df=target_factor_df  # 传入未经shift的T日因子
        )

        return overrall_summary_stats

    def get_style_factors(self, stock_pool_name: str) -> Dict[str, pd.DataFrame]:
        """获取常见的风格因子, 并与股票池对齐"""
        style_factors = {}

        # for factor_name in ['total_mv', 'pb', 'ps_ttm', 'roe_ttm', 'momentum_21d']:#写死？ 还有别的吗 todo
        ##
        # 风格因子 (Style Factor) = 市场上公认的、能长期解释股票收益差异的几类因子。最著名的如：
        #
        # 规模 (Size): 市值大小。通常用总市值或流通市值的对数表示。
        #
        # 价值 (Value): 估值高低。如市盈率PE、市净率PB。
        #
        # 动量 (Momentum): 近期涨跌趋势。如过去N天的收益率。
        #
        # 质量 (Quality): 公司质地。如净资产收益率ROE。
        #
        # 波动率 (Volatility): 股价波动性。如过去N天的年化波动率。
        #
        # 真实数据案例： 假设你发明了一个“分析师上调评级次数”因子，回测发现效果很好。但如果你计算它和规模因子的相关性，发现高达0.6。这说明分析师更倾向于覆盖和评级大市值的公司。那么你的因子收益，很大一部分其实只是搭了“大盘股效应”的便车，并非真正独特的Alpha。当市场风格从大盘切换到小盘时，你的因子可能会突然失效。#
        style_factor_list = self.factor_manager.data_manager.config['evaluation']['style_factor_list']
        # style_factor_list = [
        #     # 规模因子 (必须对数化)
        #     'small_cap',
        #     # 价值因子 (建议用倒数)
        #     'bm_ratio', 'sp_ratio', 'ep_ratio',
        #     # 成长因子
        #     'net_profit_growth_ttm',
        #     'revenue_growth_ttm',
        #     # 质量因子
        #     'roe_ttm',
        #     'gross_margin_ttm',
        #     # 风险/波动因子
        #     'volatility_90d',
        #     'beta',
        #     # 动量/反转因子
        #     'reversal_21d',  # A股常用短期反转
        #     # 流动性因子
        #     'ln_turnover_value_90d'
        # ]
        for factor_name in style_factor_list:
            #   build_df_dict... 函数可以获取因子数据并应用T-1原则
            df = self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(
                factor_name=factor_name,
                need_shift=False
            )[stock_pool_name][factor_name]

            style_factors[factor_name] = df
        return style_factors

    def test_style_correlation(self,
                               factor_data: pd.DataFrame,
                               style_factors_dict: Dict[str, pd.DataFrame]
                               ) -> Dict[str, float]:
        """
        【新增】测试目标因子与一组风格因子的截面相关性。
        """
        logger.info("    > 正在计算与常见风格因子的相关性...")
        correlation_results = {}

        for style_name, style_df in style_factors_dict.items():
            # 对齐数据
            factor_aligned, style_aligned = factor_data.align(style_df, join='inner', axis=None)

            if factor_aligned.empty:
                correlation_results[style_name] = np.nan
                continue

            # 逐日计算截面相关性
            daily_corr = factor_aligned.corrwith(style_aligned, axis=1, method='spearman')

            # 存储平均相关性
            correlation_results[f'corr_with_{style_name}'] = daily_corr.mean()

        return correlation_results

    def prepare_date_for_process_factor(self, target_factor_name, target_factor_df):
        # 目标因子基础信息准备
        target_school = self.factor_manager.get_school_code_by_factor_name(target_factor_name)
        style_category = self.factor_manager.get_style_category(target_factor_name)
        stock_pool_name = self.factor_manager.get_stock_pool_name_by_factor_school(target_school)
        #
        # # 必要操作。确实要 每天真实的能交易的股票当中。所以需要跟动态股票池进行where.!
        # close_df = self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='close',
        #                                                                              need_shift=False)[
        #     stock_pool_name]  # 传入ic 、分组、回归的 close 必须是原始的  用于t日评测结果的
        # open_df = self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='open',
        #                                                                             need_shift=False)[
        #     stock_pool_name]  # 传入ic 、分组、回归的 close 必须是原始的  用于t日评测结果的


        # circ_mv_df = self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(
        #     factor_name='circ_mv',
        #     need_shift=False)[stock_pool_name]
        # ==============================================================================
        # 【核心改造】在此处统一准备权威的“中性化数据篮子 (neutral_dfs)”
        # ==============================================================================

        # 1. 从配置中读取所需的行业级别
        neutralization_config = self.factor_processor.preprocessing_config.get('neutralization', {})
        industry_level = neutralization_config.get('by_industry', {}).get('industry_level', 'l1_code')  # 默认为一级行业

        # 2. 初始化PIT地图
        pit_map = PointInTimeIndustryMap()  # 它能自动加载数据

        # 3. 动态生成所需的行业哑变量
        industry_dummies_dict = prepare_industry_dummies(
            pit_map=pit_map,
            trade_dates=target_factor_df.index,
            stock_pool=target_factor_df.columns,
            level=industry_level
        )
        # ：last ：方案全体整改。shift操作放在最后的测试阶段进行，逻辑更加明了！。后续也再不担心漏掉shift了
        # # 对字典中的每一个哑变量DataFrame进行shift 一视同仁，人家所有都是shift1 这也需要
        # industry_dummies_dict = {
        #     key: df.shift(1, fill_value=0) for key, df in industry_dummies_dict.items()
        # }
        final_neutral_dfs = {
            # 市值因子是必须的，
            'small_cap': self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='small_cap')[
                stock_pool_name]['small_cap'],
            'pct_chg_beta': self.factor_manager.get_prepare_aligned_factor_for_analysis('beta',True),  # 去beta中性化需要用到
            # 使用字典解包，将动态生成的行业哑变量添加进来
            **industry_dummies_dict
        }
        # style_factor_dfs = self.get_style_factors(stock_pool_name)

        return  final_neutral_dfs, style_category, pit_map

    def prepare_date_for_core_test(self, target_factor_name):
        target_school = self.factor_manager.get_school_code_by_factor_name(target_factor_name)
        stock_pool_name = self.factor_manager.get_stock_pool_name_by_factor_school(target_school)
        close_df = self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='close',
                                                                                     need_shift=False)[
            stock_pool_name]['close']  # 传入ic 、分组、回归的 close 必须是原始的  用于t日评测结果的
        open_df = self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='open',
                                                                                    need_shift=False)[
            stock_pool_name]['open']
        circ_mv_df = self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(
            factor_name='circ_mv',
            need_shift=False)[stock_pool_name]['circ_mv']
        style_factor_dfs = self.get_style_factors(stock_pool_name)
        return close_df, open_df, circ_mv_df, style_factor_dfs

    def prepare_date_for_entity_service(self, factor_name):
        start_date = self.factor_manager.data_manager.config['backtest']['start_date']
        end_date = self.factor_manager.data_manager.config['backtest']['end_date']
        target_school = self.factor_manager.get_school_code_by_factor_name(factor_name)
        stock_index = self.factor_manager.get_stock_pool_index_by_factor_name(factor_name)
        stock_pool_name = self.factor_manager.get_stock_pool_name_by_factor_school(target_school)
        style_category = \
            self.factor_manager.data_manager.get_which_field_of_factor_definition_by_factor_name(factor_name,
                                                                                                 'style_category').iloc[
                0]
        close_df = self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='close',
                                                                                     need_shift=False)[
            stock_pool_name]['close']  # 传入ic 、分组、回归的 close 必须是原始的  用于t日评测结果的
        open_df = self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='open',
                                                                                    need_shift=False)[
            stock_pool_name]['open']

        # 准备收益率计算器
        c2c_calculator = partial(calcu_forward_returns_close_close, price_df=close_df)
        o2c_calculator = partial(calcu_forward_returns_open_close, close_df=close_df, open_df=open_df)
        # 定义测试配置
        test_configurations = {
            'c2c': c2c_calculator,
            'o2c': o2c_calculator
        }
        returns_calculator_config = self.factor_manager.data_manager.config['evaluation']['returns_calculator']
        returns_calculator_result = {name: test_configurations[name] for name in returns_calculator_config}
        return start_date, end_date, target_school, stock_index, stock_pool_name, close_df, open_df, style_category, returns_calculator_result

    def test_factor_entity_service_for_composite_factor(self, factor_name, factor_df,test_configurations,start_date,end_date,stock_index):
        all_configs_results = {}
        for calculator_name, func in test_configurations.items():
            proceessed_df, ic_s, ic_st, q_r,q_daily_returns_df_proc, q_st, turnover, fm_returns_series_dict, fm_t_stats_series_dict, fm_summary_dict, style_correlation_dict = self.comprehensive_test(
                target_factor_name=factor_name,
                factor_df=factor_df,
                returns_calculator=func,
                preprocess_method="standard",
                start_date=start_date,
                end_date=end_date,
                need_process_factor=False,
                do_ic_test=True, do_turnover_test=True, do_quantile_test=True, do_fama_test=True,
                do_style_correlation_test=True
            )
            single_config_results = {
                "processed_factor_df": proceessed_df,
                "ic_series_periods_dict_processed": ic_s,
                "ic_stats_periods_dict_processed": ic_st,
                "quantile_returns_series_periods_dict_processed": q_r,
                "q_daily_returns_df_processed": q_daily_returns_df_proc,
                "quantile_stats_periods_dict_processed": q_st,
                "fm_returns_series_periods_dict": fm_returns_series_dict,
                "fm_stat_results_periods_dict": fm_summary_dict,
                "turnover_stats_periods_dict": turnover,
                "style_correlation_dict": style_correlation_dict
            }
            # b) 将本次配置的所有结果打包
            self.factorResultsManager._save_factor_results(  # 假设保存函数在FactorManager中
                factor_name=factor_name,
                stock_index=stock_index,
                start_date=start_date,
                end_date=end_date,
                returns_calculator_func_name=calculator_name,
                results=single_config_results
            )
            all_configs_results[calculator_name] = single_config_results
        return all_configs_results
