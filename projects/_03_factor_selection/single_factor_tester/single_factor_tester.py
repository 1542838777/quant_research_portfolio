"""
单因子测试框架 - 专业版

实现华泰证券标准的三种单因子测试方法：
1. IC值分析法
2. 分层回测法  
3. Fama-MacBeth回归法（黄金标准）

支持批量测试、结果可视化和报告生成
"""

from quant_lib import logger
from ..factor_manager.factor_manager import FactorManager
from ..factor_manager.registry.factor_registry import FactorCategory

from ..factor_manager.storage.single_storage import add_single_factor_test_result
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
    calculate_quantile_returns,
    run_fama_macbeth_regression
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
                 data_dict: Dict[str, pd.DataFrame] = None,
                 price_data: pd.DataFrame = None,
                 config: Dict[str, Any] = None,
                 output_dir: str = "results/single_factor_tests"):
        """
        初始化单因子测试器 - 兼容新架构

        Args:
            data_dict: 数据字典（新架构）
            price_data: 价格数据（向后兼容）
            config: 配置字典
            output_dir: 结果输出目录
        """
        # 处理配置
        if not config  :
            raise RuntimeError("config 没有传递过来！")
        if data_dict is None or 'close' not in data_dict:
            raise ValueError('close的df是必须的，请写入！')

        self.config = config
        self.test_common_periods = config.get('forward_periods', [1,5, 10, 20])
        self.n_quantiles = config.get('quantiles', 5)
        self.output_dir = output_dir
        # 初始化因子预处理器
        self.factor_processor = FactorProcessor(self.config)

        # 初始化数据
        self.backtest_start_date = config['backtest']['start_date']
        self.backtest_end_date = config['backtest']['end_date']
        self.data_dict = data_dict
        self.price_data = data_dict.get('close', data_dict.get('price'))
        # 准备辅助【市值、行业】数据(用于中性值 计算！)
        self.auxiliary_data = self._prepare_auxiliary_data()
        self.circ_mv_data = data_dict['circ_mv']
        self.neutral_dict_data = self._prepare_neutral_data()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

    def test_ic_analysis(self,
                         factor_data: pd.DataFrame,
                         factor_name: str) -> Dict[str, Any]:
        """
        IC值分析法测试

        Args:
            factor_data: 预处理后的因子数据
            factor_name: 因子名称


        Returns:
            IC分析结果字典
        """
        _, stats_dict = calculate_ic_vectorized(factor_data, self.price_data, forward_periods=self.test_common_periods,
                                                method='spearman')
        return stats_dict

    def test_quantile_backtest(self,
                               factor_data: pd.DataFrame,
                               factor_name: str) -> Dict[str, Any]:
        """
        分层回测法测试

        Args:
            factor_data: 预处理后的因子数据
            factor_name: 因子名称

        Returns:
            分层回测结果字典
        """
        backtest_results, stats = calculate_quantile_returns(
            factor_data,
            self.price_data,
            n_quantiles=self.n_quantiles,
            forward_periods=self.test_common_periods
        )

        return stats

    def test_fama_macbeth(self,
                          factor_data: pd.DataFrame,
                          factor_name: str) -> Dict[str, Any]:
        """
        Fama-MacBeth回归法测试（黄金标准）

        Args:
            factor_data: 预处理后的因子数据
            factor_name: 因子名称

        Returns:
            Fama-MacBeth回归结果字典
        """
        fm_results = {}
        for period in self.test_common_periods:
            # 运行Fama-MacBeth回归
            fm_result = run_fama_macbeth_regression(
                factor_df=factor_data,
                price_df=self.price_data,
                forward_returns_period=period,
                weights_df=self.circ_mv_data,  # <-- 传入 流通市值作为权重，执行WLS
                neutral_factors=self.neutral_dict_data  # <-- 传入市值和行业作为控制变量
            )
            fm_results[f'{period}d'] = fm_result
        return fm_results

    def comprehensive_test(self,
                           factor_data: pd.DataFrame,
                           factor_name: str,
                           preprocess_method: str = "standard",
                           save_results: bool = True) -> Dict[str, Any]:
        """
        综合测试 - 执行所有三种测试方法

        Args:
            factor_data: 原始因子数据
            factor_name: 因子名称
            preprocess_method: 预处理方法
            save_results: 是否保存结果

        Returns:
            综合测试结果字典
        """
        logger.info(f"开始测试因子: {factor_name}")

        # 1. 因子预处理
        factor_processed = self.factor_processor.process_factor(
            factor_data=factor_data,
            auxiliary_df_dict=self.auxiliary_data
        )
        # 2. IC值分析
        logger.info("\t2. 正式测试 之 IC值分析...")
        ic_results = self.test_ic_analysis(factor_processed, factor_name)

        # 3. 分层回测
        logger.info("\t3.  正式测试 之 分层回测...")
        quantile_results = self.test_quantile_backtest(factor_processed, factor_name)

        # 4. Fama-MacBeth回归
        logger.info("\t4.  正式测试 之 Fama-MacBeth回归...")
        fm_results = self.test_fama_macbeth(factor_processed, factor_name)

        # 5. 综合评价
        logger.info("5. 综合评价...")
        evaluation_score_dict = self.evaluation_score_dict(ic_results, quantile_results, fm_results)

        # 整合结果
        comprehensive_results = {
            'factor_name': factor_name,
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'preprocess_method': preprocess_method,
            'ic_analysis': ic_results,
            'quantile_backtest': quantile_results,
            'fama_macbeth': fm_results,
            'evaluate_factor_score': evaluation_score_dict
        }
        ret = self.summary(comprehensive_results, factor_name)

        # self._create_visualizations(ret, factor_name)
        return ret

    def _prepare_auxiliary_data(self) -> Dict[str, pd.DataFrame]:
        """准备辅助数据（市值、行业等）"""
        auxiliary_data = {}

        # 市值数据
        if 'total_mv' in self.data_dict:
            auxiliary_data['total_mv'] = self.data_dict['total_mv']
        # 行业数据
        if 'industry' in self.data_dict:
            auxiliary_data['industry'] = self.data_dict['industry']
        return auxiliary_data

    def _prepare_neutral_data(self) -> Dict[str, pd.DataFrame]:
        """准备辅助数据（市值、行业等）"""
        neutral_dict = {}

        # 市值数据
        if 'total_mv' in self.data_dict:
            neutral_dict['total_mv'] = self.data_dict['total_mv']
        # 行业数据
        # 行业数据 - 向量化版本
        if 'industry' in self.data_dict:
            # print("  转换行业数据为哑变量...")
            industry_df = self.data_dict['industry']

            # 获取所有唯一行业
            all_industries = sorted(industry_df.stack().dropna().unique())
            # print(f"    发现 {len(all_industries)} 个行业: {all_industries}")

            # 删除基准行业
            if len(all_industries) > 1:
                base_industry = all_industries[0]
                industries_to_create = all_industries[1:]
                # print(f"    删除基准行业: {base_industry}")
            else:
                industries_to_create = all_industries

            # 向量化创建哑变量
            for industry_name in industries_to_create:
                # 使用向量化操作创建哑变量
                industry_dummy = (industry_df == industry_name).astype(float)
                # NaN位置保持NaN
                industry_dummy = industry_dummy.where(industry_df.notna())

                neutral_dict[f'industry_{industry_name}'] = industry_dummy

            # print(f"    生成 {len(industries_to_create)} 个行业哑变量")

        return neutral_dict

    def evaluation_score_dict(self,
                              ic_results: Dict,
                              quantile_results: Dict,
                              fm_results: Dict) -> Dict[str, Any]:

        ret = {}
        for period in self.test_common_periods:
            ret[f'{period}d'] = self._evaluate_factor_score(f'{period}d', ic_results, quantile_results, fm_results)
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

    def _create_visualizations(self, results: Dict[str, Any], factor_name: str):
        """创建可视化图表"""
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'因子测试报告: {factor_name}', fontsize=16, fontweight='bold')

        # 1. IC时间序列图
        ax1 = axes[0, 0]
        for period in self.test_common_periods:
            period_key = f'{period}d'
            ic_series = results['ic_analysis'].get(period_key, {}).get('ic_series', pd.Series())
            if not ic_series.empty:
                ax1.plot(ic_series.index, ic_series.values, label=f'{period}日', alpha=0.7)
        ax1.set_title('IC时间序列')
        ax1.set_ylabel('IC值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. IC统计指标柱状图
        ax2 = axes[0, 1]
        periods = [f'{p}d' for p in self.test_common_periods]
        ic_means = [results['ic_analysis'].get(p, {}).get('ic_mean', 0) for p in periods]
        ic_irs = [results['ic_analysis'].get(p, {}).get('ic_ir', 0) for p in periods]

        x = np.arange(len(periods))
        width = 0.35
        ax2.bar(x - width / 2, ic_means, width, label='IC均值', alpha=0.8)
        ax2.bar(x + width / 2, ic_irs, width, label='IC_IR', alpha=0.8)
        ax2.set_title('IC统计指标')
        ax2.set_xlabel('预测周期')
        ax2.set_xticks(x)
        ax2.set_xticklabels(periods)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 分层收益率图
        ax3 = axes[0, 2]
        main_period = '20d'
        quantile_main = results['quantile_backtest'].get(main_period, {})
        if quantile_main:
            quantile_means = quantile_main.get('quantile_means', [])
            if quantile_means:
                quantiles = [f'Q{i}' for i in range(1, len(quantile_means) + 1)]
                ax3.bar(quantiles, quantile_means, alpha=0.8, color='skyblue')
                ax3.set_title(f'分层收益率 ({main_period})')
                ax3.set_ylabel('平均收益率')
                ax3.grid(True, alpha=0.3)

        # 4. 多空组合净值曲线
        ax4 = axes[1, 0]
        for period in self.test_common_periods:
            period_key = f'{period}d'
            quantile_data = results['quantile_backtest'].get(period_key, {})
            if quantile_data:
                returns_data = quantile_data.get('returns_data')
                if returns_data is not None and 'TopMinusBottom' in returns_data.columns:
                    tmb_series = returns_data['TopMinusBottom']
                    cumulative_returns = (1 + tmb_series).cumprod()
                    ax4.plot(cumulative_returns.index, cumulative_returns.values,
                             label=f'{period}日', alpha=0.8)
        ax4.set_title('多空组合净值曲线')
        ax4.set_ylabel('累计收益')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Fama-MacBeth t值图
        ax5 = axes[1, 1]
        periods = [f'{p}d' for p in self.test_common_periods]
        t_stats = [results['fama_macbeth'].get(p, {}).get('fama_macbeth_t_stat', 0) for p in periods]
        colors = ['red' if abs(t) > 2 else 'orange' if abs(t) > 1.64 else 'gray' for t in t_stats]

        ax5.bar(periods, t_stats, color=colors, alpha=0.8)
        ax5.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='显著水平(2)')
        ax5.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
        ax5.set_title('Fama-MacBeth t统计量')
        ax5.set_ylabel('t值')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. 综合评价雷达图
        ax6 = axes[1, 2]
        evaluation = results['evaluate_factor_score']

        # 雷达图数据
        categories = ['IC_IR', '单调性', 'FM显著性', '多空夏普']
        values = [
            min(abs(evaluation['ic_ir']) / 0.5, 1),  # 标准化到0-1
            1 if evaluation['is_monotonic_by_group'] else 0,
            1 if evaluation['fm_significant'] else 0,
            min(abs(evaluation['tmb_sharpe']) / 2, 1)  # 标准化到0-1
        ]

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values += values[:1]  # 闭合
        angles = np.concatenate((angles, [angles[0]]))

        ax6.plot(angles, values, 'o-', linewidth=2, alpha=0.8)
        ax6.fill(angles, values, alpha=0.25)
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 1)
        ax6.set_title('综合评价雷达图')
        ax6.grid(True)

        plt.tight_layout()

        # 保存图表
        chart_path = os.path.join(self.output_dir, f'{factor_name}_analysis_charts.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"可视化图表已保存: {chart_path}")




    def summary(self, results: Dict[str, Any], excel_path: str):

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
                results = self.comprehensive_test(
                    factor_data,
                    factor_name,
                    preprocess_method=preprocess_method,
                    save_results=True
                )
                # self.factor_manager._save_results(results, factor_name)

                batch_results[factor_name] = results

            except Exception as e:
                raise ValueError(f"因子 {factor_name} 测试失败: {e}")
                batch_results[factor_name] = {'error': str(e)}
        return batch_results


    def run_comprehensive_test_with_viz(self,
                                        factor_data: pd.DataFrame,
                                        factor_name: str,
                                        preprocess_method: str = "standard",
                                        generate_plots: bool = True) -> Dict[str, Any]:
        """
        运行综合测试并生成可视化（新增方法）

        Args:
            factor_data: 因子数据
            factor_name: 因子名称
            preprocess_method: 预处理方法
            generate_plots: 是否生成图表

        Returns:
            包含测试结果和图表路径的字典
        """
        # 运行标准测试
        test_results = self.run_comprehensive_test(
            factor_data=factor_data,
            factor_name=factor_name,
            preprocess_method=preprocess_method
        )

        # 生成可视化
        plot_paths = {}
        if generate_plots and self.viz_manager is not None:
            try:
                plot_paths = self.viz_manager.plot_single_factor_results(
                    test_results=test_results,
                    factor_name=factor_name,
                    save_plots=True
                )
                print(f"✓ 图表已生成并保存到: {self.viz_manager.output_dir}")
            except Exception as e:
                print(f"⚠ 图表生成失败: {e}")

        # 合并结果
        enhanced_results = test_results.copy()
        enhanced_results['plot_paths'] = plot_paths

        return enhanced_results

    def cal_score_ic(self,
                     ic_mean: float,
                     ic_ir: float,
                     ic_win_rate: float,
                     ic_p_value: float) -> Dict:
        """
        【专业版】对IC进行多维度、分级、加权评分
        """

        # 1. 科学性检验 (准入门槛)
        is_significant = ic_p_value < 0.05 if not np.isnan(ic_p_value) else False

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
                'final_score':'0/100',
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
