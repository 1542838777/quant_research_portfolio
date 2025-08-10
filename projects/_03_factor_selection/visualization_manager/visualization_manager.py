"""
可视化管理器 - 统一管理所有图表和报告生成

提供标准化的可视化接口，支持：
1. 单因子测试结果可视化
2. 多因子优化结果可视化
3. 性能对比图表
4. 交互式仪表板
5. 专业报告生成

Author: Quantitative Research Team
Date: 2024-12-19
"""
import matplotlib.dates as mdates # 导入日期格式化模块

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import gridspec
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from datetime import datetime
import warnings
import sys


# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.config.logger_config import setup_logger, log_success

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from quant_lib.utils.json_utils import load_json_with_numpy

# 配置日志
logger = setup_logger(__name__)


class VisualizationManager:
    """
    可视化管理器 - 统一管理所有图表生成
    
    功能：
    1. 单因子测试结果可视化
    2. 多因子优化结果可视化
    3. 性能对比和分析图表
    4. 交互式图表和仪表板
    """

    def __init__(self, output_dir: str = "visualizations", style: str = "default"):
        """
        初始化可视化管理器

        Args:
            output_dir: 图表输出目录
            style: 图表样式
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置图表样式
        self._setup_style(style)
        sns.set_palette("husl")

        # 颜色配置
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
        
        # logger.info(f"可视化管理器初始化完成，输出目录: {output_dir}")

    def _setup_style(self, style: str):
        """
        设置matplotlib样式

        Args:
            style: 样式名称
        """
        try:
            # 尝试使用指定的样式
            if style == "seaborn":
                # 如果是seaborn样式，使用seaborn-v0_8样式或默认样式
                available_styles = plt.style.available
                if 'seaborn-v0_8' in available_styles:
                    plt.style.use('seaborn-v0_8')
                elif 'seaborn-whitegrid' in available_styles:
                    plt.style.use('seaborn-whitegrid')
                else:
                    # 手动设置类似seaborn的样式
                    plt.rcParams.update({
                        'figure.facecolor': 'white',
                        'axes.facecolor': 'white',
                        'axes.edgecolor': 'black',
                        'axes.linewidth': 0.8,
                        'axes.grid': True,
                        'grid.color': 'gray',
                        'grid.alpha': 0.3,
                        'grid.linewidth': 0.5,
                        'font.size': 10,
                        'axes.labelsize': 10,
                        'axes.titlesize': 12,
                        'xtick.labelsize': 9,
                        'ytick.labelsize': 9,
                        'legend.fontsize': 9
                    })
            else:
                plt.style.use(style)
        except OSError:
            # 如果样式不存在，使用默认样式
            logger.warning(f"样式 '{style}' 不可用，使用默认样式")
            plt.style.use('default')

    def plot_single_factor_results(self,
                                   backtest_base_on_index,
                                   factor_name: str,
                                   target_period,
                                   ic_series_periods_dict: Dict[str, pd.Series],
                                   ic_stats_periods_dict: Dict[str, Dict[str, Any]],
                                   quantile_returns_series_periods_dict: Dict[str, pd.DataFrame],
                                   quantile_stats_periods_dict: Dict[str, Dict[str, Any]],
                                   factor_returns_series_periods_dict: Dict[str, pd.Series],
                                   fm_stat_results_periods_dict: Dict[str, Dict[str, Any]],
                                   save_plots: bool = True,
                                   ):

        """
        【V3版】最终版因子分析报告。
        右上角图表已升级为分层累计净值曲线，左下角为F-M纯净Alpha收益。
        """
        logger.info(f"开始为周期 {target_period} 绘图 for:{factor_name}")
        # 1. 创建 2x2 图表布局
        fig = plt.figure(figsize=(24, 20))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        fig.suptitle(f'单因子{factor_name}分析报告In_{backtest_base_on_index}', fontsize=28, y=0.97)

        # 2. 【左上】IC序列(柱状) + 累计IC(折线) - (维持V2版)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1_twin = ax1.twinx()
        primary_ic = ic_series_periods_dict.get(target_period)
        if primary_ic is not None:
            ax1.bar(primary_ic.index, primary_ic.values, width=1.0, color='royalblue', alpha=0.6, label=f'IC序列 ({target_period})')
        ax1.axhline(0, color='black', linestyle='--', lw=1)
        ax1.set_ylabel('IC值 (单期)', color='royalblue', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='royalblue')
        for period, ic_series in ic_series_periods_dict.items():
            ic_series.cumsum().plot(ax=ax1_twin, label=f'累计IC ({period})', lw=2.5)
        ax1_twin.set_ylabel('累计IC', fontsize=14)
        ax1.set_title('A. 因子IC序列与累计IC (有效性)', fontsize=18)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.tick_params(axis='x', rotation=45)
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        ax1.grid(False)

        # 3. 【右上，核心升级】分层累计净值曲线
        ax2 = fig.add_subplot(gs[0, 1])
        primary_q_returns = quantile_returns_series_periods_dict.get(target_period)
        if primary_q_returns is not None:
            # a) 绘制多空组合累计收益（灰色区域）
            tmb_cum_returns = (1 + primary_q_returns['TopMinusBottom']).cumprod()
            ax2.fill_between(tmb_cum_returns.index, 1, tmb_cum_returns, color='grey', alpha=0.3, label=f'多空组合 ({target_period})')

            # b) 绘制每个分层的累计净值曲线
            quantile_cols = [q for q in ['Q1','Q2','Q3','Q4','Q5'] if q in primary_q_returns.columns]
            for quantile in quantile_cols:
                (1 + primary_q_returns[quantile]).cumprod().plot(ax=ax2, label=f'{quantile} ({target_period})', lw=2)

        ax2.set_title(f'B. 分层累计净值 ({target_period})', fontsize=18)
        ax2.set_ylabel('累计净值')
        ax2.axhline(1, color='black', linestyle='--', lw=1)
        ax2.legend()
        ax2.grid(True)
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6)) # 每年一个主刻度
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.tick_params(axis='x', rotation=45)

        # 4. 【左下，核心升级】F-M纯净Alpha收益
        ax3 = fig.add_subplot(gs[1, 0])
        for period, fm_returns in factor_returns_series_periods_dict.items():
            (1 + fm_returns).cumprod().plot(ax=ax3, label=f'F-M纯净收益 ({period})', lw=2.5, linestyle='--')
        ax3.set_title('C. 因子纯净Alpha收益 (Fama-MacBeth)', fontsize=18)
        ax3.set_ylabel('累计净值')
        ax3.axhline(1, color='black', linestyle='--', lw=1)
        ax3.legend()
        ax3.grid(True)
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.tick_params(axis='x', rotation=45)

        # 5. 【右下】核心指标汇总 - (维持V2版)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        summary_data = []
        periods = sorted(ic_stats_periods_dict.keys(), key=lambda x: int(x[:-1]) if x[:-1].isdigit() else 99) # 兼容'20d_monthly'
        for period in periods:
            ic_ir = ic_stats_periods_dict.get(period, {}).get('ic_ir', np.nan)
            tmb_sharpe = quantile_stats_periods_dict.get(period, {}).get('tmb_sharpe', np.nan)
            fm_t_stat = fm_stat_results_periods_dict.get(period, {}).get('t_statistic', np.nan)
            mean_abs_t = fm_stat_results_periods_dict.get(period, {}).get('mean_abs_t_stat', np.nan)
            summary_data.append(
                [f'{period}', f'{ic_ir:.2f}', f'{tmb_sharpe:.2f}', f'{fm_t_stat:.2f}', f'{mean_abs_t:.2f}'])

        columns = ['周期', 'ICIR', '分层Sharpe', 'F-M t值', 't值绝对值均值']
        table = ax4.table(cellText=summary_data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(16)
        table.scale(1.0, 2.8)
        ax4.set_title('D. 核心指标汇总', fontsize=18, y=0.85)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path = Path(self.output_dir / factor_name / f"{factor_name}_in_{backtest_base_on_index}_{target_period}_evaluation.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(path)

    def plot_diagnostics_report(self,
                                backtest_base_on_index,
                                factor_name: str,
                                ic_series_periods_dict: Dict,
                                turnover_stats_periods_dict: Dict,
                                style_correlation_dict: Dict,
                                factor_df: pd.DataFrame,  # 需要原始因子值来画自相关图
                                period,
                                save_plots: bool = True):
        """
        【V4版-诊断报告】生成因子的诊断图表。
        """
        logger.info(f"开始绘制诊断图表 for:{factor_name}")
        fig = plt.figure(figsize=(24, 20))
        gs = gridspec.GridSpec(2, 2)
        fig.suptitle(f'单因子 {factor_name} 诊断报告 In_{backtest_base_on_index}', fontsize=28, y=0.97)

        # 1. 【左上】滚动IC图
        ax1 = fig.add_subplot(gs[0, 0])
        for period, ic_series in ic_series_periods_dict.items():
            ic_series.rolling(window=120).mean().plot(ax=ax1, label=f'滚动IC ({period}, W=120d)')
        ax1.set_title('A. 因子滚动IC (稳定性)')
        ax1.axhline(0, color='black', linestyle='--', lw=1)
        ax1.legend()

        # 2. 【右上】因子自相关图
        ax2 = fig.add_subplot(gs[0, 1])
        # 截面平均因子的自相关性
        mean_factor = factor_df.mean(axis=1).dropna()
        pd.plotting.autocorrelation_plot(mean_factor, ax=ax2)
        ax2.set_title('B. 因子自相关性 (持续性)')
        ##
        #
        # 相关系数，范围在-1到1之间。
        #
        # 解读： 它衡量的是，在所有股票上，今天的因子值排序与**Lag天前的因子值排序**的相似程度。
        #
        # +1：今天的排名和Lag天前的排名一模一样。
        #
        # 0：今天的排名和Lag天前的排名毫无关系，是完全独立的。
        #
        # -1：今天的排名和Lag天前的排名正好完全相反。#

        # 3. 【左下】因子换手率图
        ax3 = fig.add_subplot(gs[1, 0])
        turnover_data = {p: d['turnover_annual'] for p, d in turnover_stats_periods_dict.items()}
        pd.Series(turnover_data).plot(kind='bar', ax=ax3)
        ax3.set_title('C. 年化换手率 (交易成本)')
        ax3.set_ylabel('年化换手率')

        # 4. 【右下】风格相关性图
        ax4 = fig.add_subplot(gs[1, 1])
        pd.Series(style_correlation_dict).plot(kind='barh', ax=ax4)
        ax4.set_title('D. 与常见风格因子相关性 (独特性)')
        ax4.axvline(0, color='black', linestyle='--', lw=1)
        ##
        # ##
        #         #  检验新因子是否独特<br>（与Size, Value等基础风格因子比） | > 0.7 | 丢弃 |
        #         # | | 0.5 ~ 0.7 | 中性化后再测试，若仍有效则保留 |
        #         # | | < 0.5 | 保留 |
        #         # | 从因子库中挑选组合<br>（因子与因子之间比） | > 0.5 | “二选一”，或只保留更好的那个 |
        #         # | | < 0.5 | 可以考虑同时使用 |##

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 假设 self.output_dir 存在
        path = Path(
            self.output_dir / factor_name / f"{factor_name}_in_{backtest_base_on_index}_{period}_diagnostics_report.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(path)

    def plot_unified_factor_report(self,
                                   backtest_base_on_index: str,
                                   factor_name: str,
                                   results_path: str,
                                   default_config: str = 'o2c'
                                   ) -> str:
        """
        【V5.0 终极版】生成单因子综合评估报告 (3x2布局)。
        从硬盘加载指定配置 (默认为更严格的o2c) 的结果进行绘制。
        """
        logger.info(f"开始为因子 {factor_name} (配置: {default_config}) 生成统一评估报告...")

        # --- 1. 定位并加载核心数据 ---
        base_path = Path(results_path) / backtest_base_on_index / factor_name
        config_path = base_path / default_config

        summary_stats_file = config_path / 'summary_stats.json'
        if not summary_stats_file.exists():
            logger.error(f"未找到摘要文件: {summary_stats_file}，无法生成报告。")
            return ""

        stats = load_json_with_numpy(summary_stats_file)
        ic_stats_periods_dict = stats.get('ic_analysis', {})
        quantile_stats_periods_dict = stats.get('quantile_backtest', {})
        fm_stat_results_periods_dict = stats.get('fama_macbeth', {})
        turnover_stats_periods_dict = stats.get('turnover', {})
        style_correlation_dict = stats.get('style_correlation', {})

        # 找到最佳周期 (基于最高ICIR)
        try:
            best_period = max(ic_stats_periods_dict, key=lambda p: ic_stats_periods_dict[p].get('ic_ir', -np.inf))
        except ValueError:
            logger.warning("结果字典为空，无法确定最佳周期。")
            return ""

        logger.info(f"  > 自动识别最佳周期为: {best_period}")

        # --- 2. 创建 3x2 图表布局 ---
        fig = plt.figure(figsize=(24, 30))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.25)
        fig.suptitle(f'单因子 "{factor_name}" 综合评估报告 (基准: {backtest_base_on_index} | 配置: {default_config})',
                     fontsize=32, y=0.97)

        # --- A. 核心指标 vs. 持有周期 ---
        ax_a = fig.add_subplot(gs[0, 0])
        periods_numeric = sorted([int(p[:-1]) for p in ic_stats_periods_dict.keys()])
        periods_str = [f'{p}d' for p in periods_numeric]
        icir_values = [ic_stats_periods_dict.get(p, {}).get('ic_ir', np.nan) for p in periods_str]
        sharpe_values = [quantile_stats_periods_dict.get(p, {}).get('tmb_sharpe', np.nan) for p in periods_str]
        ax_a.plot(periods_numeric, icir_values, marker='o', linestyle='-', lw=2.5, label='ICIR')
        ax_a_twin = ax_a.twinx()
        ax_a_twin.plot(periods_numeric, sharpe_values, marker='s', linestyle='--', color='C1', label='分层Sharpe')
        ax_a.set_title('A. 核心指标 vs. 持有周期 (寻找最佳周期)', fontsize=18)
        # ... (此处省略了与之前版本相同的详细格式化代码)
        ax_a.grid(True, linestyle='--', alpha=0.6);
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.92))

        # --- B. 最佳周期分层净值曲线 ---
        ax_b = fig.add_subplot(gs[0, 1])
        q_returns_df = pd.read_parquet(config_path / f"quantile_returns_{best_period}.parquet")
        tmb_cum = (1 + q_returns_df['TopMinusBottom']).cumprod()
        ax_b.fill_between(tmb_cum.index, 1, tmb_cum, color='grey', alpha=0.3, label=f'多空组合 ({best_period})')
        for quantile in [q for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'] if q in q_returns_df.columns]:
            (1 + q_returns_df[quantile]).cumprod().plot(ax=ax_b, label=f'{quantile}', lw=2)
        ax_b.set_title(f'B. 最佳周期 ({best_period}) 分层累计净值', fontsize=18)
        # ... (省略格式化代码) ...
        ax_b.legend();
        ax_b.grid(True, linestyle='--', alpha=0.6)

        # --- C. 最佳周期累计IC vs. F-M纯净Alpha收益 ---
        ax_c = fig.add_subplot(gs[1, 0])
        ax_c_twin = ax_c.twinx()
        ic_series = pd.read_parquet(config_path / f"ic_series_{best_period}.parquet")
        fm_returns = pd.read_parquet(config_path / f"fm_returns_series_{best_period}.parquet")
        ic_series.cumsum().plot(ax=ax_c, label=f'累计IC ({best_period})', lw=2.5, color='C0')
        (1 + fm_returns).cumprod().plot(ax=ax_c_twin, label=f'F-M纯净收益 ({best_period})', lw=2.5, color='C1',
                                        linestyle='--')
        ax_c.set_title(f'C. 最佳周期 ({best_period}) IC vs. F-M Alpha', fontsize=18)
        # ... (省略格式化代码) ...
        ax_c.grid(True, linestyle='--', alpha=0.6);
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.61))

        # --- D. 因子自身特性 (自相关性 & 换手率) ---
        ax_d = fig.add_subplot(gs[1, 1])
        processed_factor_df = pd.read_parquet(config_path / "processed_factor.parquet")
        mean_factor = processed_factor_df.mean(axis=1).dropna()
        if len(mean_factor) > 1:
            pd.plotting.autocorrelation_plot(mean_factor, ax=ax_d, color='C2', lw=2.5)
        ax_d_twin = ax_d.twinx()
        turnover_data = {int(p[:-1]): d['turnover_annual'] for p, d in turnover_stats_periods_dict.items()}
        pd.Series(turnover_data).sort_index().plot(kind='bar', ax=ax_d_twin, alpha=0.6, color='C3')
        ax_d.set_title('D. 因子特性：自相关性 vs. 换手率', fontsize=18)
        # ... (省略格式化代码) ...

        # --- E. 风格暴露分析 (因子“DNA”鉴定) ---
        ax_e = fig.add_subplot(gs[2, 0])
        pd.Series(style_correlation_dict).sort_values(ascending=True).plot(kind='barh', ax=ax_e)
        ax_e.set_title('E. 风格暴露分析 (独特性)', fontsize=18)
        ax_e.axvline(0, color='black', linestyle='--', lw=1)
        ax_e.grid(True, axis='x', linestyle='--', alpha=0.6)

        # --- F. 核心指标汇总表 ---
        ax_f = fig.add_subplot(gs[2, 1])
        ax_f.axis('off')
        summary_data = []
        for period in periods_str:
            ic_ir = ic_stats_periods_dict.get(period, {}).get('ic_ir', np.nan)
            tmb_sharpe = quantile_stats_periods_dict.get(period, {}).get('tmb_sharpe', np.nan)
            fm_t_stat = fm_stat_results_periods_dict.get(period, {}).get('t_statistic', np.nan)
            mean_abs_t = fm_stat_results_periods_dict.get(period, {}).get('mean_abs_t_stat', np.nan)
            turnover = turnover_stats_periods_dict.get(period, {}).get('turnover_annual', np.nan)
            summary_data.append(
                [f'{period}', f'{ic_ir:.2f}', f'{tmb_sharpe:.2f}', f'{fm_t_stat:.2f}', f'{mean_abs_t:.2f}',
                 f'{turnover:.2f}'])
        columns = ['周期', 'ICIR', '分层Sharpe', 'F-M t值', 't值绝对值均值', '年化换手率']
        table = ax_f.table(cellText=summary_data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False);
        table.set_fontsize(16);
        table.scale(1.0, 2.5)
        ax_f.set_title('F. 核心指标汇总', fontsize=18, y=0.85)

        # --- 统一调整X轴日期格式 ---
        for ax in [ax_b, ax_c]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # --- 最终布局与保存 ---
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = f'{results_path}/{backtest_base_on_index}/{factor_name}/{factor_name}_unified_report_{default_config}.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"✓ 统一评估报告已保存至: {path}")
        return str(path)

    def plot_robustness_report(self,
                               backtest_base_on_index: str,
                               factor_name: str,
                               results_path: str
                               ) -> str:
        """
        【V1.0】生成 C2C vs. O2C 稳健性对比报告 (2x2布局)。
        从硬盘加载 C2C 和 O2C 两份结果进行对比。
        """
        logger.info(f"开始为因子 {factor_name} 生成稳健性对比报告...")

        # --- 1. 定位并加载 C2C 和 O2C 两份数据 ---
        base_path = Path(results_path) / backtest_base_on_index / factor_name
        c2c_path = base_path / 'c2c'
        o2c_path = base_path / 'o2c'

        if not c2c_path.exists() or not o2c_path.exists():
            logger.warning(f"因子 {factor_name} 的结果不完整 (缺少C2C或O2C)，无法生成稳健性报告。")
            return ""

        stats_c2c = load_json_with_numpy(c2c_path / 'summary_stats.json')
        stats_o2c = load_json_with_numpy(o2c_path / 'summary_stats.json')

        # 确定最佳周期 (基于更严格的O2C)
        best_period = max(stats_o2c['ic_analysis'], key=lambda p: stats_o2c['ic_analysis'][p].get('ic_ir', -np.inf))

        # --- 2. 创建 2x2 图表布局 ---
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        fig.suptitle(f'因子 "{factor_name}" Alpha稳健性分析 (C2C vs. O2C)', fontsize=32, y=0.97)
        axes = axes.flatten()

        # --- A. 核心指标对比 (ICIR分组柱状图) ---
        periods = sorted(stats_c2c['ic_analysis'].keys(), key=lambda p: int(p[:-1]))
        icir_c2c = [stats_c2c['ic_analysis'][p].get('ic_ir', 0) for p in periods]
        icir_o2c = [stats_o2c['ic_analysis'][p].get('ic_ir', 0) for p in periods]
        x = np.arange(len(periods))
        width = 0.35
        axes[0].bar(x - width / 2, icir_c2c, width, label='ICIR (C2C)')
        axes[0].bar(x + width / 2, icir_o2c, width, label='ICIR (O2C)')
        axes[0].set_title('A. ICIR 对比', fontsize=18)
        axes[0].set_xticks(x);
        axes[0].set_xticklabels(periods);
        axes[0].legend()
        axes[0].grid(True, axis='y', linestyle='--', alpha=0.6)

        # --- B. 最佳周期多空组合净值对比 ---
        q_returns_c2c = pd.read_parquet(c2c_path / f"quantile_returns_{best_period}.parquet")
        q_returns_o2c = pd.read_parquet(o2c_path / f"quantile_returns_{best_period}.parquet")
        (1 + q_returns_c2c['TopMinusBottom']).cumprod().plot(ax=axes[1], label=f'多空组合 (C2C, {best_period})',
                                                             linestyle='--')
        (1 + q_returns_o2c['TopMinusBottom']).cumprod().plot(ax=axes[1], label=f'多空组合 (O2C, {best_period})')
        axes[1].set_title(f'B. 最佳周期 ({best_period}) 多空组合净值对比', fontsize=18)
        axes[1].legend();
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # --- C. 最佳周期F-M纯净Alpha对比 ---
        fm_returns_c2c = pd.read_parquet(c2c_path / f"fm_returns_series_{best_period}.parquet")
        fm_returns_o2c = pd.read_parquet(o2c_path / f"fm_returns_series_{best_period}.parquet")
        (1 + fm_returns_c2c).cumprod().plot(ax=axes[2], label=f'F-M Alpha (C2C, {best_period})', linestyle='--')
        (1 + fm_returns_o2c).cumprod().plot(ax=axes[2], label=f'F-M Alpha (O2C, {best_period})')
        axes[2].set_title(f'C. 最佳周期 ({best_period}) F-M Alpha 对比', fontsize=18)
        axes[2].legend();
        axes[2].grid(True, linestyle='--', alpha=0.6)

        # --- D. Alpha衰减率量化总览 ---
        axes[3].axis('off')
        summary_data = []
        for period in periods:
            s_c2c = stats_c2c['quantile_backtest'][period].get('tmb_sharpe', 0)
            s_o2c = stats_o2c['quantile_backtest'][period].get('tmb_sharpe', 0)
            decay = (s_o2c - s_c2c) / abs(s_c2c) if abs(s_c2c) > 1e-6 else 0
            summary_data.append([f'{period}', f'{s_c2c:.2f}', f'{s_o2c:.2f}', f'{decay:.1%}'])
        columns = ['周期', 'Sharpe (C2C)', 'Sharpe (O2C)', '衰减率']
        table = axes[3].table(cellText=summary_data, colLabels=columns, loc='center')
        table.auto_set_font_size(False);
        table.set_fontsize(16);
        table.scale(1.0, 2.5)
        axes[3].set_title('D. Sharpe 稳健性分析', fontsize=18, y=0.85)

        # --- 最终布局与保存 ---
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = self.output_dir / backtest_base_on_index / factor_name / f"{factor_name}_robustness_report.png"
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"✓ 稳健性对比报告已保存至: {path}")
        return str(path)