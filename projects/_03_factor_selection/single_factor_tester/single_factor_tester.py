"""
单因子测试框架 - 专业版

实现华泰证券标准的三种单因子测试方法：
1. IC值分析法
2. 分层回测法  
3. Fama-MacBeth回归法（黄金标准）

支持批量测试、结果可视化和报告生成
"""
from pandas import Series

from data.local_data_load import load_index_daily, load_daily_hfq
from projects._03_factor_selection.data_manager.data_manager import DataManager

from quant_lib import logger
from ..factor_manager.factor_manager import FactorManager

from ..utils.factor_processor import FactorProcessor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
import os
import json
from datetime import datetime
import sys

from scipy import stats

n_metrics_pass_rate_key = 'n_metrics_pass_rate'

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from quant_lib.evaluation import (
    calculate_ic_vectorized,
    calculate_quantile_returns, fama_macbeth

)

# 导入新的可视化管理器
try:
    from visualization_manager import VisualizationManager
except ImportError:
    VisualizationManager = None

warnings.filterwarnings('ignore')


class SingleFactorTester:
    """
    单因子测试器 - 专业版

    按照华泰证券标准实现三种测试方法的完整流程
    """

    def __init__(self,
                 raw_dfs: Dict[str, pd.DataFrame] = None,
                 processed_raw_data: Dict[str, pd.DataFrame] = None,
                 stock_pools_dict: Dict[str, pd.DataFrame] = None,
                 config: Dict[str, Any] = None,
                 target_factors_dict: Dict[str, pd.DataFrame] = None,
                 target_factors_category_dict: Dict[str, Any] = None,
                 target_factor_school_type_dict: Dict[str, Any] = None

                 ):
        """
        初始化单因子测试器 - 兼容新架构

        Args:
            raw_dfs: 数据字典（新架构）
            price_data: 价格数据（向后兼容）
            config: 配置字典
        """
        # 处理配置
        if not config:
            raise RuntimeError("config 没有传递过来！")
        if raw_dfs is None or 'close' not in raw_dfs:
            raise ValueError('close的df是必须的，请写入！')

        self.config = config
        self.test_common_periods = config.get('forward_periods', [1, 5, 10, 20])
        self.n_quantiles = config.get('quantiles', 5)
        # 初始化因子预处理器
        self.factor_processor = FactorProcessor(self.config)

        # 初始化数据
        self.backtest_start_date = config['backtest']['start_date']
        self.backtest_end_date = config['backtest']['end_date']
        self.backtest_period =  f"{pd.to_datetime(self.backtest_start_date).strftime('%Y%m%d')} ~ {pd.to_datetime(self.backtest_end_date).strftime('%Y%m%d')}"
        self.raw_dfs = raw_dfs
        self.processed_raw_data = processed_raw_data  # 似乎没用
        self.stock_pools_dict = stock_pools_dict

        self.target_factors_dict = target_factors_dict
        self.target_factors_category_dict = target_factors_category_dict
        self.target_school_type_dict = target_factor_school_type_dict
        # 基于不同股票池！！！
        self.close_df_dict = self._prepare_dfs_dict_by_diff_stock_pool(['close'])
        self.circ_mv_dict = self._prepare_dfs_dict_by_diff_stock_pool(['circ_mv'])
        self.pct_chg_dict = self._prepare_dfs_dict_by_diff_stock_pool(['pct_chg'])
        self.prepare_master_pct_chg_beta_dataframe()
        self.pct_chg_bate_dict = self.get_pct_chg_beta_dict()

        # 准备辅助【市值、行业】数据(用于中性值 计算！)
        self.auxiliary_dfs_dict = self.build_auxiliary_dfs_dict()
        self.neutral_dfs_data_dict = self._prepare_neutral_data_dict()
        self.check_shape()

        # 创建输出目录
        # os.makedirs(output_dir, exist_ok=True)

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
            factor_name: 因子名称
            preprocess_method: 预处理方法
            save_results: 是否保存结果

        Returns:
            综合测试结果字典
        """
        logger.info(f"开始测试因子: {target_factor_name}")
        target_factor_data = self.target_factors_dict[target_factor_name]
        target_school = self.target_school_type_dict[target_factor_name]
        stock_pool_name = self.get_stock_pool_name_by_factor_school(target_school)
        # 1. 因子预处理
        target_factor_processed = self.factor_processor.process_factor(
            target_factor_df=target_factor_data,
            target_factor_name=target_factor_name,
            auxiliary_dfs=self.auxiliary_dfs_dict[stock_pool_name],
            neutral_dfs=self.neutral_dfs_data_dict[stock_pool_name],
            factor_school=target_school
        )
        # 必要操作。确实要 每天真实的能交易的股票当中。所以需要跟动态股票池进行where.!
        close_df = self.close_df_dict[stock_pool_name]['close']
        circ_mv_df = self.circ_mv_dict[stock_pool_name]['circ_mv']
        neutral_dfs = self.neutral_dfs_data_dict[stock_pool_name]

        # 2. IC值分析
        logger.info("\t2. 正式测试 之 IC值分析...")
        ic_series_periods_dict, ic_stats_periods_dict = self.test_ic_analysis(target_factor_processed, close_df,
                                                                              target_factor_name)

        # 3. 分层回测
        logger.info("\t3.  正式测试 之 分层回测...")
        quantile_returns_series_periods_dict, quantile_stats_periods_dict = self.test_quantile_backtest(
            target_factor_processed, close_df, target_factor_name)

        # 4. Fama-MacBeth回归
        logger.info("\t4.  正式测试 之 Fama-MacBeth回归...")
        factor_returns_series_periods_dict, fm_stat_results_periods_dict = fama_macbeth(
            factor_data=target_factor_processed, close_df=close_df, forward_periods=self.test_common_periods,
            neutral_dfs=neutral_dfs, circ_mv_df=circ_mv_df,
            factor_name=target_factor_name)

        return (ic_series_periods_dict, ic_stats_periods_dict,
                quantile_returns_series_periods_dict, quantile_stats_periods_dict,
                factor_returns_series_periods_dict, fm_stat_results_periods_dict)

    # ok
    def _prepare_dfs_dict_by_diff_stock_pool(self, factor_names) -> Dict[str, pd.DataFrame]:
        """准备辅助数据（市值、行业等）"""
        ret_dict = {}
        # 基于institutional_stock_pool
        dataManager_temp = DataManager(
            "factory/config.yaml",
            need_data_deal=False
        )
        for stock_poll_name, stock_pool in self.stock_pools_dict.items():
            ret_dict[stock_poll_name] = {}

            for factor_name in factor_names:
                df = dataManager_temp._align_many_raw_dfs_by_stock_pool_and_fill(
                    {factor_name: self.raw_dfs[factor_name]}, stock_pool)
                ret_dict[stock_poll_name].update(df)

        return ret_dict

    # ok
    def _prepare_neutral_data(self, total_mv_df, industry_df) -> Dict[str, pd.DataFrame]:
        """
         准备辅助数据（市值、行业等）- 最终修正版
         解决 MultiIndex 丢失的问题
         """
        industry_df = industry_df.replace([None, 'NONE', 'None'], np.nan)
        neutral_dict = {}
        logger.info("开始准备行业哑变量df。用于后续中性化使用")
        # --- 1. 市值数据 ---
        neutral_dict['total_mv'] = total_mv_df

        # --- 2. 行业数据处理 ---

        # 扁平化原始行业数据，
        # industry_stacked_series 确实带有 (日期, 股票代码) MultiIndex
        industry_stacked_series = industry_df.stack().dropna()

        # 获取所有唯一非NaN行业名称
        all_industries = sorted(industry_stacked_series.unique())

        if not all_industries:
            print("警告：行业数据中未发现任何有效行业名称。")
            return neutral_dict

        # 关键修正：
        # 将 industry_stacked_series 的数据类型直接设置为 CategoricalDtype
        # 这样，当 get_dummies 操作这个 Series 时，它会保留原始 Series 的索引
        # 并且知道所有的可能类别
        # 小白解释：
        # CategoricalDtype = 分类数据类型定义：它不仅仅是数据本身，更像是一个“模板”或“规则”。
        # 当你把一个 Series 转换为这种类型时，Series 仍然保留其索引，但其内部数据遵循 Categorical 的规则。
        industry_categorical_series = industry_stacked_series.astype(
            pd.CategoricalDtype(categories=all_industries)
        )

        # 现在，pd.get_dummies 直接操作这个带有 MultiIndex 的 Series
        # get_dummies 会自动继承 industry_categorical_series 的 MultiIndex 作为行索引
        # 并且根据 CategoricalDtype 的类别信息生成正确的哑变量列
        industry_dummies_df = pd.get_dummies(industry_categorical_series, prefix='industry', dtype=float)

        # 此时 industry_dummies_df 的结构是：
        #                      industry_TMT  industry_医药  industry_消费
        # Date        StockCode
        # 2023-01-01  000001.SZ           1.0          0.0          0.0
        #             000002.SZ           0.0          1.0          0.0
        # ...
        # 它的行索引是 MultiIndex，列是哑变量名称。这是正确的中间状态。

        final_industry_dummies = {}

        # --- 3. 逐个行业哑变量进行 unstack 和存储 ---
        for col_name in industry_dummies_df.columns:
            # col_name 是 'industry_TMT', 'industry_医药' 等
            # 从 industry_dummies_df 中取出对应列，它是一个 Series，其索引仍然是 (日期, 股票代码) MultiIndex
            one_industry_series = industry_dummies_df[col_name]

            # 对这个 Series 进行 unstack 操作
            # level=-1 将最内层索引（股票代码）从行索引移动到列索引
            dummy_df_for_one_industry = one_industry_series.unstack(level=-1)
            # ========================= 关键修正第二步 =========================
            # 使用原始 industry_df 的列（所有股票）来 reindex
            # 这会把在 stack() 过程中丢失的股票列加回来
            # fill_value=0 的意思是，这些被加回来的股票，它们在这个行业哑变量中的值是0
            dummy_df_for_one_industry = dummy_df_for_one_industry.reindex(
                columns=industry_df.columns,
                fill_value=0
            )
            # ================================================================

            final_industry_dummies[col_name] = dummy_df_for_one_industry

        # --- 4. 删除基准行业列 ---
        if len(all_industries) > 1:
            base_industry = all_industries[0]
            base_industry_col_name = f'industry_{base_industry}'
            if base_industry_col_name in final_industry_dummies:
                del final_industry_dummies[base_industry_col_name]

        # --- 5. 处理 NaN：将原始行业为 NaN 的位置对应的哑变量设为 NaN ---
        nan_mask_df = industry_df.isna()

        for industry_col_key, dummy_df in final_industry_dummies.items():
            # 这里需要注意，mask 操作要求 dummy_df 和 nan_mask_df 的索引和列对齐
            # 幸好，我们在循环内部生成的 dummy_df_for_one_industry 的索引和列是和原始 industry_df 对齐的
            final_industry_dummies[industry_col_key] = dummy_df.mask(nan_mask_df)  # 对应位置替换为nan
            final_industry_dummies[industry_col_key] = final_industry_dummies[industry_col_key].astype(float)

        neutral_dict.update(final_industry_dummies)

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

    def batch_test(self,
                   factors_dict: Dict[str, pd.DataFrame],
                   preprocess_method: str = "standard") -> Dict[str, Dict]:
        """
        批量测试多个因子

        Args:
            factors_dict: 因子字典，{因子名称: 因子数据}
            preprocess_method: 预处理方法

        Returns:
            批量测试结果字典
        """
        logger.info(f"开始批量测试 {len(factors_dict)} 个因子")

        batch_results = {}

        for i, (factor_name, factor_data) in enumerate(factors_dict.items(), 1):
            print(f"\n进度: {i}/{len(factors_dict)}")

            try:
                ic_series_periods_dict, quantile_returns_series_periods_dict, factor_returns_series_periods_dict, summary_stats = self.comprehensive_test(
                    factor_data,
                    factor_name,
                    preprocess_method=preprocess_method,
                    save_results=True
                )

                batch_results[factor_name] = summary_stats

            except Exception as e:
                raise ValueError(f"因子 {factor_name} 测试失败: {e}")
                batch_results[factor_name] = {'error': str(e)}
        return batch_results

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

    def _prepare_neutral_data_dict(self):
        dict = {}
        pct_chg_beta_dict = self.pct_chg_bate_dict
        for stock_pool_name, df_dict in self.auxiliary_dfs_dict.items():
            cur_bate_df = pct_chg_beta_dict[stock_pool_name]
            cur = self._prepare_neutral_data(df_dict['total_mv'], df_dict['industry'])
            dict[stock_pool_name] = cur
        return dict

    def get_stock_pool_name_by_factor_school(self, target_school):
        dataManager_temp = DataManager(
            "factory/config.yaml",
            need_data_deal=False
        )
        return dataManager_temp.get_stock_pool_name_by_factor_school(target_school)

    def get_stock_pool_name_by_factor_name(self, factor_name):
        school = self.target_school_type_dict[factor_name]
        pool_name = self.get_stock_pool_name_by_factor_school(school[0])
        return pool_name

    # ok 因为需要滚动计算，所以不依赖股票池的index（trade） 只要对齐股票列就好
    def get_pct_chg_beta_dict(self):
        dict = {}
        for pool_name, _ in self.stock_pools_dict.items():
            beta_df = self.get_pct_chg_beta_data_for_pool(pool_name)
            dict[pool_name] = beta_df
        return dict

    def prepare_master_pct_chg_beta_dataframe(self):
        """
        一个在系统初始化时调用的方法，用于生成一份统一的、覆盖所有股票的Beta矩阵。
        """
        logger.info("开始准备主Beta矩阵...")

        # 1. 整合所有股票池的股票代码，形成一个总的股票列表
        all_unique_stocks = set()
        for stock_pool in self.stock_pools_dict.values():
            all_unique_stocks.update(stock_pool.columns)

        master_stock_list = sorted(list(all_unique_stocks))

        # 2. 只调用一次 calculate_rolling_beta，计算所有股票的Beta
        logger.info(f"开始为总计 {len(master_stock_list)} 只股票计算统一的Beta...")
        self.master_beta_df = self.calculate_rolling_beta(
            self.config['backtest']['start_date'],
            self.config['backtest']['end_date'],
            master_stock_list
        )

    def get_pct_chg_beta_data_for_pool(self, pool_name):
        pool_stocks = self.stock_pools_dict[pool_name].columns

        # 直接从主Beta矩阵中按需选取，无需重新计算
        beta_for_this_pool = self.master_beta_df[pool_stocks]

        return beta_for_this_pool

    # ok ok
    def calculate_rolling_beta(
            self,
            start_date: str,
            end_date: str,
            cur_stock_codes: list,
            window: int = 60,
            min_periods: int = 20
    ) -> pd.DataFrame:
        """
        【最终健壮版】计算A股市场上每只股票相对于市场指数的滚动Beta值。
        此版本修复了数据对齐的隐患。

        Args:
            start_date (str): 回测开始日期, 格式 'YYYYMMDD'
            end_date (str): 回测结束日期, 格式 'YYYYMMDD'
            stock_returns (pd.DataFrame): 股票收益率宽表, index为datetime, values已处理为小数。
            window (int): 滚动窗口大小（天数）。
            min_periods (int): 窗口内计算所需的最小观测数。

        Returns:
            pd.DataFrame: 滚动Beta矩阵 (index=date, columns=stock)。
        """
        logger.info(f"开始计算滚动Beta (窗口: {window}天)...")

        # --- 1. 数据获取与准备 ---
        #  指数提前。但是入参传入的股票是死的，建议重新手动加载。但是考虑是否与股票池对应！ 答案：还是别跟动态股票池进行where了，疑问
        # 解释：
        # 为了计算滚动值，我们需要往前多取一些数据作为“缓冲”
        ##
        # 滚动历史因子 (Rolling History Factor)
        # 例子: pct_chg_beta, 动量因子 (Momentum), 滚动波动率 (Volatility)。
        #
        # 关键特征: 计算今天的值，需要过去N天连续、干净的历史数据。它的计算过程本身就是一个“时间序列”操作。
        #
        # 为什么预处理是“致命的”: 如果在计算之前，就用每日动态股票池把历史数据弄得“千疮百孔”（充满NaN），那么滚动窗口在回看历史时就找不到足够的数据，导致计算结果本身就是错误的（大量的NaN）。预处理污染了计算的“原材料”。#
        buffer_days = int(window * 1.7) + 5
        buffer_start_date = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y%m%d')
        # 1. Load the long-form DataFrame
        stock_data_long = load_daily_hfq(buffer_start_date, end_date, cur_stock_codes)

        # 2. It's better to modify the column before pivoting
        stock_data_long['pct_chg'] = stock_data_long['pct_chg'] / 100

        # 3. Correctly pivot the DataFrame to wide format
        # The 'columns' argument should be the name of the column containing the stock codes.
        stock_returns = pd.pivot_table(
            stock_data_long,
            index='trade_date',
            columns='ts_code',  # Use the column name 'ts_code'
            values='pct_chg'
        )

        # a) 获取市场指数的每日收益率 是否是自动过滤了 非交易日 yes
        market_returns_long = load_index_daily(buffer_start_date, end_date).assign(
            pct_chg=lambda x: x['pct_chg'] / 100)  # pct_chg = ...: 这指定了要创建或修改的列的名称 x：当前DataFrame
        market_returns = market_returns_long.set_index('trade_date')['pct_chg']
        market_returns.index = pd.to_datetime(market_returns.index)
        market_returns.name = 'market_return'  # chong'ming

        # --- 2. 【核心修正】显式数据对齐 ---
        # logger.info("  > 正在进行数据显式对齐...")
        # 使用 'left' join，以 stock_returns 的日期为基准
        # 这会创建一个统一的时间轴，并将市场收益精确地匹配到每个交易日
        combined_df = stock_returns.join(market_returns, how='left')

        # 更新 market_returns 为对齐后的版本，确保万无一失
        market_returns_aligned = combined_df.pop('market_return')  # 剔除这列！

        # --- 3. 滚动计算Beta ---
        # logger.info("  > 正在进行滚动计算...")
        # Beta = Cov(R_stock, R_market) / Var(R_market)

        # a) 现在，stock_returns 和 market_returns_aligned 的索引是100%对齐的
        rolling_cov = combined_df.rolling(window=window, min_periods=min_periods).cov(
            market_returns_aligned)  # 协方差关心的是两组数据之间的关系（描述两个变量之间的关系方向）（是不是都是一起）

        # b) 计算指数收益率的滚动方差
        rolling_var = market_returns_aligned.rolling(window=window, min_periods=min_periods).var()

        # c) 计算滚动Beta
        beta_df = rolling_cov.div(rolling_var, axis=0)

        # d) 截取我们需要的最终日期范围
        final_beta_df = beta_df.loc[start_date:end_date]

        logger.info(f"滚动Beta计算完成，最终矩阵形状: {final_beta_df.shape}")

        return final_beta_df

    # ok
    def build_auxiliary_dfs_dict(self):
        dict = self._prepare_dfs_dict_by_diff_stock_pool(['total_mv', 'industry'])
        pct_chg_beta_dict = self.pct_chg_bate_dict

        for stock_poll_name, df in pct_chg_beta_dict.items():
            # 补充beta
            dict[stock_poll_name].update({'pct_chg_beta': df})
        return dict

    def check_shape(self):
        pool_names = self.stock_pools_dict.keys()
        pool_shape_config = {}
        for pool_name in pool_names:
            pool_shape_config[pool_name] = self.stock_pools_dict[pool_name].shape

        for pool_name, shape in pool_shape_config.items():
            if shape != self.close_df_dict[pool_name]['close'].shape:
                raise ValueError("形状不一致 ，请必须检查")
            if shape != self.circ_mv_dict[pool_name]['circ_mv'].shape:
                raise ValueError("形状不一致 ，请必须检查")
            if shape != self.pct_chg_bate_dict[pool_name].shape:
                raise ValueError("形状不一致 ，请必须检查")

            if shape != self.auxiliary_dfs_dict[pool_name]['pct_chg_beta'].shape:
                raise ValueError("形状不一致 ，请必须检查")
            if shape != self.auxiliary_dfs_dict[pool_name]['industry'].shape:
                raise ValueError("形状不一致 ，请必须检查")
            if shape != self.auxiliary_dfs_dict[pool_name]['total_mv'].shape:
                raise ValueError("形状不一致 ，请必须检查")

            if shape != self.neutral_dfs_data_dict[pool_name]['industry_农业综合'].shape:
                raise ValueError("形状不一致 ，请必须检查")

            if shape != self.neutral_dfs_data_dict[pool_name]['total_mv'].shape:
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
        ic_series_periods_dict, ic_stats_periods_dict, quantile_returns_series_periods_dict, quantile_stats_periods_dict, factor_returns_series_periods_dict, fm_stat_results_periods_dict = \
            (
                self.comprehensive_test(
                    target_factor_name=target_factor_name,
                    preprocess_method="standard"
                ))
        # 5. 综合评价

        evaluation_score_dict = self.evaluation_score_dict(ic_stats_periods_dict,
                                                           quantile_returns_series_periods_dict,
                                                           fm_stat_results_periods_dict)
        # 整合结果
        comprehensive_results = {
            'factor_name': target_factor_name,
            'factor_category': self.target_factors_category_dict[target_factor_name],
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
        fm_return_series_dict = self.build_fm_return_series_dict(factor_returns_series_periods_dict,target_factor_name)


        test_kwargs.get('factor_manager')._save_results(overrall_summary_stats, file_name_prefix = 'overrall_summary')
        test_kwargs.get('factor_manager').update_and_save_factor_purify_summary(purify_summary_rows_contain_periods, file_name_prefix = 'purify_summary')
        test_kwargs.get('factor_manager').update_and_save_fm_factor_return_matrix(fm_return_series_dict, file_name_prefix = 'fm_return_series')
        # 画图保存
        test_kwargs.get('visualization_manager').plot_single_factor_results(
            target_factor_name,
            ic_series_periods_dict,
            ic_stats_periods_dict,
            quantile_returns_series_periods_dict,
            quantile_stats_periods_dict,
            factor_returns_series_periods_dict,
            fm_stat_results_periods_dict)

        return ic_series_periods_dict, quantile_returns_series_periods_dict, factor_returns_series_periods_dict, overrall_summary_stats

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
                'factor_category': factor_category,
                'backtest_period': self.backtest_period,

                'period': period,  # 【BUG已修正】这里应该是单个周期
                #收益维度
                'tmb_sharpe': quantile_stats_periods_dict[period]['tmb_sharpe'],
                'tmb_annual_return': quantile_stats_periods_dict[period]['tmb_annual_return'],
                # 风险与稳定性维度 (Risk & Stability Dimension) - 过程有多颠簸
                'tmb_max_drawdown': quantile_stats_periods_dict[period]['max_drawdown'],
                'ic_ir': ic_stats_periods_dict[period]['ic_ir'],
                #纯净度与独特性维度 (Purity & Uniqueness Dimension) - “是真Alpha还是只是风险暴露
                'fm_t_statistic': fm_stat_results_periods_dict[period]['t_statistic'],
                'is_monotonic_by_group': quantile_stats_periods_dict[period]['is_monotonic_by_group'],
                'ic_mean': ic_stats_periods_dict[period]['ic_mean']
            }
            purify_summary_rows.append(summary_row)
        return purify_summary_rows

    def build_fm_return_series_dict(self, fm_factor_returns_series_periods_dict, target_factor_name):
        periods = fm_factor_returns_series_periods_dict.keys()
        fm_factor_returns = {}

        for period ,return_series in fm_factor_returns_series_periods_dict.items():
            colum_name  = f'{target_factor_name}_{period}'
            fm_factor_returns[colum_name] = return_series

        return fm_factor_returns
