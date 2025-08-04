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
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series
from scipy import stats

from projects._03_factor_selection.factor_manager.factor_technical_cal.factor_technical_cal import \
    calculate_rolling_beta
from projects._03_factor_selection.factor_manager.selector.factor_selector import calculate_factor_score
from projects._03_factor_selection.utils.factor_processor import FactorProcessor
from projects._03_factor_selection.visualization_manager import VisualizationManager
from quant_lib import logger

n_metrics_pass_rate_key = 'n_metrics_pass_rate'

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from quant_lib.evaluation import (
    calculate_ic_vectorized,
    calculate_quantile_returns, fama_macbeth, calculate_quantile_daily_returns

)

# # 导入新的可视化管理器
# try:
#     from visualization_manager import VisualizationManager
# except ImportError:
#     VisualizationManager = None

warnings.filterwarnings('ignore')


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
        if data_manager is None or 'close' not in data_manager.raw_dfs:
            raise ValueError('close的df是必须的，请写入！')

        config = data_manager.config
        self.config = data_manager.config
        self.test_common_periods = self.config['evaluation'].get('forward_periods', [1, 5, 10, 20])
        self.n_quantiles = self.config.get('quantiles', 5)
        # 初始化因子预处理器
        self.factor_processor = FactorProcessor(self.config)
        self.stock_pools_dict = data_manager.stock_pools_dict

        # 初始化数据

        self.target_factors_dict = target_factors_dict
        self.target_factors_category_dict = target_factors_category_dict
        self.target_school_type_dict = target_factor_school_type_dict

        self.backtest_start_date = config['backtest']['start_date']
        self.backtest_end_date = config['backtest']['end_date']
        self.backtest_period = f"{pd.to_datetime(self.backtest_start_date).strftime('%Y%m%d')} ~ {pd.to_datetime(self.backtest_end_date).strftime('%Y%m%d')}"
        self.raw_dfs = data_manager.raw_dfs  # 纯原生 只有dfs间的对齐 除此之外 没有任何过滤，也没有shift（1）
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
        ic_series_periods_dict, stats_periods_dict = calculate_ic_vectorized(factor_data, close_df,
                                                                             forward_periods=self.test_common_periods,
                                                                             method='spearman')
        return ic_series_periods_dict, stats_periods_dict

    def test_quantile_backtest(self,
                               factor_data: pd.DataFrame,
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
        quantile_returns_periods_dict, quantile_stats_periods_dict = calculate_quantile_returns(
            factor_data,
            close_df,
            n_quantiles=self.n_quantiles,
            forward_periods=self.test_common_periods
        )

        return quantile_returns_periods_dict, quantile_stats_periods_dict

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
                           target_factor_name: str,
                           preprocess_method: str = "standard",
                           ) -> Tuple[
        Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        综合测试 - 执行所有三种测试方法

        Args:
        Returns:
            综合测试结果字典
        """
        logger.info(f"开始测试因子: {target_factor_name}")
        target_school = self.target_school_type_dict[target_factor_name]
        stock_pool_name = self.factor_manager.get_stock_pool_name_by_factor_school(target_school)
        target_factor_shift_df = self.target_factors_dict[target_factor_name]
        # 必要操作。确实要 每天真实的能交易的股票当中。所以需要跟动态股票池进行where.!
        close_df = self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(factor_name='close',
                                                                      need_shift=False)[
            stock_pool_name]  # 传入ic 、分组、回归的 close 必须是原始的  用于t日评测结果的
        auxiliary_shift_dfs_base_own_stock_pools = self.factor_manager.build_auxiliary_dfs_shift_diff_stock_pools_dict()[
            stock_pool_name]
        prepare_for_neutral_shift_base_own_stock_pools_dfs = \
        self.prepare_for_neutral_data_dict_shift_diff_stock_pools()[
            stock_pool_name]
        circ_mv_shift_df = self.factor_manager.build_df_dict_base_on_diff_pool_can_set_shift(
            factor_name='circ_mv',
            need_shift=True)[stock_pool_name]

        # 1. 因子预处理
        target_factor_processed = self.factor_processor.process_factor(
            target_factor_df=target_factor_shift_df,
            target_factor_name=target_factor_name,
            auxiliary_dfs=auxiliary_shift_dfs_base_own_stock_pools,
            neutral_dfs=prepare_for_neutral_shift_base_own_stock_pools_dfs,
            factor_school=target_school
        )
        return self.core_three_test(target_factor_processed,target_factor_name, close_df,prepare_for_neutral_shift_base_own_stock_pools_dfs,circ_mv_shift_df)

    # ok
    def _prepare_dfs_dict_by_diff_stock_pool(self, factor_names) -> Dict[str, pd.DataFrame]:
        """准备辅助数据（市值、行业等）"""
        ret_dict = {}
        for stock_poll_name, in self.stock_pools_dict.items():
            ret_dict[stock_poll_name] = {}

            for factor_name in factor_names:
                ret_dict[stock_poll_name].update(self.raw_dfs[factor_name])

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
        neutral_dict['total_mv'] = total_mv_df

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

    def prepare_for_neutral_data_dict_shift_diff_stock_pools(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        dict = {}
        for stock_pool_name, df_dict in self.factor_manager.build_auxiliary_dfs_shift_diff_stock_pools_dict().items():
            cur = self._prepare_for_neutral_data(df_dict['total_mv'], df_dict['industry'])
            dict[stock_pool_name] = cur
        return dict

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

            if shape != self.factor_manager.prepare_for_neutral_dfs_shift_diff_stock_pools_dict[pool_name]['total_mv'].shape:
                raise ValueError("形状不一致 ，请必须检查")

    def test_single_factor_entity_service(self,
                                          target_factor_name: str,
                                          preprocess_method: str = "standard",
                                          **test_kwargs) -> Tuple[
        Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        测试单个因子
        综合评分
        保存结果
        画图保存
        Args:

            factor_name: 因子名称

            **test_kwargs: 测试参数
        """

        # # 自动分类因子
        # 先注释下面的，问题：自动识别因子类型 函数有待补充！ 暂且不用 不是很要紧
        # if category is None and auto_register:
        #
        #
        #     # returns_data = self.single_factor_tester.get_returns_data()#todo！！！
        #     # category = self.factor_manager.classify_factor(factor_data, returns_data)
        #
        # # 自动注册因子
        # if auto_register:
        #     self.register_factor(
        #         name=factor_name,
        #         category=category or FactorCategory.CUSTOM,
        #         description=f"自动注册的{category.value if isinstance(category, FactorCategory) else category or '自定义'}因子",
        #         data_requirements=["price", "returns"]
        #     )

        # 执行测试
        ic_series_periods_dict, ic_stats_periods_dict, quantile_daily_returns_for_plot_dict, quantile_stats_periods_dict, factor_returns_series_periods_dict, fm_stat_results_periods_dict = \
            (
                self.comprehensive_test(
                    target_factor_name=target_factor_name,
                    preprocess_method="standard"
                ))
        overrall_summary_stats = self.landing_for_core_three_analyzer_result(target_factor_name, self.target_factors_category_dict[target_factor_name],"standard",ic_series_periods_dict, ic_stats_periods_dict, quantile_daily_returns_for_plot_dict, quantile_stats_periods_dict, factor_returns_series_periods_dict, fm_stat_results_periods_dict)

        return ic_series_periods_dict, quantile_daily_returns_for_plot_dict, factor_returns_series_periods_dict, overrall_summary_stats

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
                'is_monotonic_by_group': quantile_stats_periods_dict[period]['is_monotonic_by_group'],
                'ic_mean': ic_stats_periods_dict[period]['ic_mean']
            }
            score = calculate_factor_score(summary_row)
            summary_row['score'] = score
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
                           target_factors_category_dict: Dict[str, str],
                           target_factor_school_type_dict: Dict[str, str],
                           **test_kwargs) -> Dict[str, Any]:
        """
        批量测试因子
        """

        # 批量测试
        results = {}
        for factor_name, factor_data in target_factors_dict.items():
            try:
                # 执行测试
                ic_series_periods_dict, quantile_returns_series_periods_dict, factor_returns_series_periods_dict, summary_stats = (
                    self.test_single_factor_entity_service(
                        target_factor_name=factor_name,
                    ))
                results[factor_name] = summary_stats
            except Exception as e:
                # traceback.print_exc()
                raise ValueError(f"✗ 因子{factor_name}测试失败: {e}") from e

        return results

    def core_three_test(self, target_factor_processed, target_factor_name, close_df,prepare_for_neutral_shift_base_own_stock_pools_dfs,circ_mv_shift_df):
        # 1. IC值分析
        logger.info("\t2. 正式测试 之 IC值分析...")
        ic_series_periods_dict, ic_stats_periods_dict = self.test_ic_analysis(target_factor_processed, close_df,
                                                                              target_factor_name)

        # 2. 分层回测
        logger.info("\t3.  正式测试 之 分层回测...")
        quantile_returns_series_periods_dict, quantile_stats_periods_dict = self.test_quantile_backtest(
            target_factor_processed, close_df, target_factor_name)

        primary_period_key = list(quantile_returns_series_periods_dict.keys())[-1]
        #这是中性化之后的分组收益，也就是纯净的单纯因子自己带来的收益。至于在真实的市场上，禁不禁得起考验，这个无法看出。需要在原始因子（未除杂/中性化），然后分组查看收益才行！
        quantile_daily_returns_for_plot_dict = calculate_quantile_daily_returns(target_factor_processed, close_df, 5,
                                                                                primary_period_key)

        # 3. Fama-MacBeth回归
        logger.info("\t4.  正式测试 之 Fama-MacBeth回归...")
        factor_returns_series_periods_dict, fm_stat_results_periods_dict = fama_macbeth(
            factor_data=target_factor_processed, close_df=close_df, forward_periods=self.test_common_periods,
            neutral_dfs=prepare_for_neutral_shift_base_own_stock_pools_dfs, circ_mv_df=circ_mv_shift_df,
            factor_name=target_factor_name)

        return (ic_series_periods_dict, ic_stats_periods_dict,
                quantile_daily_returns_for_plot_dict, quantile_stats_periods_dict,
                factor_returns_series_periods_dict, fm_stat_results_periods_dict)

    def landing_for_core_three_analyzer_result(self,target_factor_name,category,preprocess_method, ic_series_periods_dict, ic_stats_periods_dict,
                                               quantile_daily_returns_for_plot_dict, quantile_stats_periods_dict,
                                               factor_returns_series_periods_dict, fm_stat_results_periods_dict):
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
        self.visualizationManager.plot_single_factor_results(
            comprehensive_results['backtest_base_on_index'],
            target_factor_name,
            ic_series_periods_dict,
            ic_stats_periods_dict,
            quantile_daily_returns_for_plot_dict,
            quantile_stats_periods_dict,
            factor_returns_series_periods_dict,
            fm_stat_results_periods_dict)
        return overrall_summary_stats
