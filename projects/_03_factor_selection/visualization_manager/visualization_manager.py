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

from statsmodels.tsa.stattools import acf
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
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

##fname 为你下载的字体库路径，注意 SourceHanSansSC-Bold.otf 字体的路径，这里放到工程本地目录下。
cn_font = FontProperties(fname=r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\quant_lib\font\SourceHanSansSC-Regular.otf", size=12)
x1 = np.array([1, 2, 3, 4])
y2 = np.array([6, 2, 13, 10])

plt.plot(x1, y2)
plt.xlabel("X轴", fontproperties=cn_font)
plt.ylabel("Y轴", fontproperties=cn_font)
plt.title("测试", fontproperties=cn_font)

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
                                   default_config: str = 'o2c',
                                   run_version: str = 'latest') -> str:
        """
              【V6.0 智能归因版】生成单因子综合评估报告 (3x2布局)。
              能自动检测并对比“原始”与“纯净”因子的表现。
              """
        logger.info(f"为因子 {factor_name} (配置: {default_config}, 版本: {run_version}) 生成统一评估报告...")

        # --- 1. 定位并加载核心数据 ---
        base_path = Path(results_path) / backtest_base_on_index / factor_name
        config_path = base_path / default_config

        # ... (版本定位逻辑不变) ...
        # [为简洁，省略版本定位的 _find_target_version_path 辅助函数和调用代码]
        target_version_path = _find_target_version_path(config_path,run_version) # ... a call to _find_target_version_path ...
        if not target_version_path: return ""

        summary_stats_file = target_version_path / 'summary_stats.json'
        if not summary_stats_file.exists(): return ""

        stats = load_json_with_numpy(summary_stats_file)

        # --- 【核心改造】同时加载 processed 和 raw 的统计数据 ---
        ic_stats_proc = stats.get('ic_analysis_processed', {})
        q_stats_proc = stats.get('quantile_backtest_processed', {})
        ic_stats_raw = stats.get('ic_analysis_raw', {})  # 如果不存在，会是空字典
        q_stats_raw = stats.get('quantile_backtest_raw', {})  # 如果不存在，会是空字典

        fm_stats = stats.get('fama_macbeth', {})
        turnover_stats = stats.get('turnover', {})
        style_corr = stats.get('style_correlation', {})

        best_period = self._find_best_period_by_rank(ic_stats_proc, q_stats_proc, fm_stats)

        # --- 2. 创建图表布局 ---
        fig = plt.figure(figsize=(24, 30))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.25)
        run_version_str = target_version_path.name
        fig.suptitle(
            f'单因子 "{factor_name}" 综合评估报告\n(基准: {backtest_base_on_index} | 配置: {default_config} | 版本: {run_version_str})',
            fontproperties=cn_font, fontsize=32, y=0.98)

        # --- A. 核心指标对比 (Raw vs. Processed) ---
        ax_a = fig.add_subplot(gs[0, 0])
        periods_numeric = sorted([int(p[:-1]) for p in ic_stats_proc.keys()])
        periods_str = [f'{p}d' for p in periods_numeric]

        # 绘制纯净因子的表现
        icir_proc = [ic_stats_proc.get(p, {}).get('ic_ir', np.nan) for p in periods_str]
        sharpe_proc = [q_stats_proc.get(p, {}).get('tmb_sharpe', np.nan) for p in periods_str]
        ax_a.plot(periods_numeric, icir_proc, marker='o', lw=2.5, label='ICIR (纯净)')
        ax_a_twin = ax_a.twinx()
        ax_a_twin.plot(periods_numeric, sharpe_proc, marker='s', linestyle='-', color='C1', label='分层Sharpe (纯净)')

        # 【智能绘图】如果存在原始因子的数据，则用虚线叠加
        if ic_stats_raw and q_stats_raw:
            icir_raw = [ic_stats_raw.get(p, {}).get('ic_ir', np.nan) for p in periods_str]
            sharpe_raw = [q_stats_raw.get(p, {}).get('tmb_sharpe', np.nan) for p in periods_str]
            ax_a.plot(periods_numeric, icir_raw, marker='o', lw=1.5, linestyle='--', color='C0', alpha=0.7,
                      label='ICIR (原始)')
            ax_a_twin.plot(periods_numeric, sharpe_raw, marker='s', lw=1.5, linestyle=':', color='C1', alpha=0.7,
                           label='分层Sharpe (原始)')

        ax_a.set_title('A. 核心指标对比 (纯净 vs. 原始)', fontproperties=cn_font, fontsize=18)
        ax_a.set_xlabel('持有周期 (天)', fontproperties=cn_font, fontsize=14)
        ax_a.set_ylabel('ICIR', fontproperties=cn_font, fontsize=14)
        ax_a_twin.set_ylabel('分层Sharpe', fontproperties=cn_font, fontsize=14)
        ax_a.legend(loc='upper left', bbox_to_anchor=(0.1, 0.93), prop=cn_font)
        ax_a.grid(True, linestyle='--', alpha=0.6)

        # --- B. 最佳周期分层净值曲线 (Raw vs. Processed) ---
        ax_b = fig.add_subplot(gs[0, 1])
        try:
            # 始终绘制纯净因子的分层
            q_returns_proc_df = pd.read_parquet(
                target_version_path / f"quantile_returns_processed_{best_period}.parquet")
            tmb_cum_proc = (1 + q_returns_proc_df['TopMinusBottom']).cumprod()
            ax_b.fill_between(tmb_cum_proc.index, 1, tmb_cum_proc, color='grey', alpha=0.3,
                              label=f'多空组合 (纯净, {best_period})')
            for quantile in [q for q in ['Q1', 'Q5'] if q in q_returns_proc_df.columns]:  # 只画Q1,Q5简化
                (1 + q_returns_proc_df[quantile]).cumprod().plot(ax=ax_b, label=f'{quantile} (纯净)', lw=2.5)

            # 【智能绘图】如果存在原始因子的分层数据，则用虚线叠加多空组合
            q_returns_raw_path = target_version_path / f"quantile_returns_raw_{best_period}.parquet"
            if q_returns_raw_path.exists():
                q_returns_raw_df = pd.read_parquet(q_returns_raw_path)
                (1 + q_returns_raw_df['TopMinusBottom']).cumprod().plot(ax=ax_b,
                                                                        label=f'多空组合 (原始, {best_period})',
                                                                        linestyle='--', lw=2.0)

        except FileNotFoundError:
            ax_b.text(0.5, 0.5, f"未能加载分层收益数据", ha='center', va='center', fontproperties=cn_font)
        ax_b.set_title(f'B. 最佳周期 ({best_period}) 分层累计净值', fontproperties=cn_font, fontsize=18)
        ax_b.set_ylabel('累计净值', fontproperties=cn_font, fontsize=14)
        ax_b.legend(prop=cn_font);
        ax_b.grid(True)

        # --- C. 最佳周期累计IC vs. F-M纯净Alpha收益 ---
        ax_c = fig.add_subplot(gs[1, 0])
        ax_c_twin = ax_c.twinx()
        try:
            ic_series = pd.read_parquet(target_version_path / f"ic_series_processed_{best_period}.parquet")
            fm_returns = pd.read_parquet(target_version_path / f"fm_returns_series_{best_period}.parquet")
            ic_series.cumsum().plot(ax=ax_c, label=f'累计IC ({best_period})', lw=2.5, color='C0')
            (1 + fm_returns).cumprod().plot(ax=ax_c_twin, label=f'F-M纯净收益 ({best_period})', lw=2.5, color='C1',
                                            linestyle='--')
        except FileNotFoundError:
            ax_c.text(0.5, 0.5, "未能加载IC或F-M序列数据", ha='center', va='center', fontproperties=cn_font)
        ax_c.set_title(f'C. 最佳周期 ({best_period}) IC vs. F-M Alpha', fontproperties=cn_font, fontsize=18)
        ax_c.set_ylabel('累计IC', fontproperties=cn_font, fontsize=14);
        ax_c_twin.set_ylabel('F-M纯净收益', fontproperties=cn_font, fontsize=14)
        ax_c.grid(True)
        # ▼▼▼▼▼ 【核心修正】 ▼▼▼▼▼
        # 1. 从两个坐标轴分别获取图例元素
        lines_c, labels_c = ax_c.get_legend_handles_labels()
        lines_twin_c, labels_twin_c = ax_c_twin.get_legend_handles_labels()

        ax_c.legend(lines_c + lines_twin_c, labels_c + labels_twin_c, loc='upper left', prop=cn_font)

        # --- D. 因子自身特性 (自相关性 & 换手率) ---
        ax_d = fig.add_subplot(gs[1, 1])
        try:
            processed_factor_df = pd.read_parquet(target_version_path / "processed_factor.parquet")
            mean_factor = processed_factor_df.mean(axis=1).dropna()
            if len(mean_factor) > 20:
                acf_values = acf(mean_factor, nlags=252, fft=True)
                ax_d.plot(acf_values, marker='.', linestyle='-', color='#2E8B57', label='自相关系数')
                n_obs = len(mean_factor);
                confidence_interval = 1.96 / np.sqrt(n_obs)
                ax_d.axhline(y=confidence_interval, color='grey', linestyle='--', lw=1.5)
                ax_d.axhline(y=-confidence_interval, color='grey', linestyle='--', lw=1.5)
            else:
                ax_d.text(0.5, 0.5, "有效数据点过少\n无法计算自相关性", ha='center', va='center',
                          fontproperties=cn_font)
        except FileNotFoundError:
            ax_d.text(0.5, 0.5, "未能加载processed_factor.parquet", ha='center', va='center',
                      fontproperties=cn_font)
        ax_d_twin = ax_d.twinx()
        turnover_data = {p: d['turnover_annual'] for p, d in turnover_stats.items()}
        turnover_series = pd.Series(turnover_data)
        turnover_series.index = pd.Categorical(turnover_series.index,
                                               categories=sorted(turnover_series.index, key=lambda x: int(x[:-1])),
                                               ordered=True)
        turnover_series = turnover_series.sort_index()
        color_cycle = plt.get_cmap('Paired')
        for i, (period, turnover_value) in enumerate(turnover_series.items()):
            ax_d_twin.bar(int(period[:-1]), turnover_value, width=5, alpha=0.7,
                          color=color_cycle(i / len(turnover_series)), label=f'年化换手率 ({period})')
        ax_d.set_title('D. 因子特性：自相关性 vs. 换手率', fontproperties=cn_font, fontsize=18)
        ax_d.set_xlabel('滞后期 / 持有周期 (天)', fontproperties=cn_font, fontsize=14)
        ax_d.set_ylabel('自相关系数', fontproperties=cn_font, fontsize=14, color='#2E8B57')
        ax_d_twin.set_ylabel('年化换手率', fontproperties=cn_font, fontsize=14, color='C0')
        ax_d.set_xlim(left=-5, right=260);
        ax_d.set_xticks([0, 50, 100, 150, 200, 250])
        lines_d, labels_d = ax_d.get_legend_handles_labels();
        lines_twin, labels_twin = ax_d_twin.get_legend_handles_labels()
        ax_d.legend(lines_d + lines_twin, labels_d + labels_twin, loc='upper right', prop=cn_font)
        ax_d.grid(True, linestyle='--', alpha=0.6)

        # --- E. 风格暴露分析 ---
        ax_e = fig.add_subplot(gs[2, 0])
        pd.Series(style_corr).sort_values(ascending=True).plot(kind='barh', ax=ax_e)
        ax_e.set_title('E. 风格暴露分析 (独特性)', fontproperties=cn_font, fontsize=18)
        ax_e.axvline(0, color='black', linestyle='--', lw=1);
        ax_e.grid(True, axis='x')

        # --- F. 核心指标汇总表 ---
        ax_f = fig.add_subplot(gs[2, 1])
        ax_f.axis('off')
        summary_data = []
        for period in periods_str:
            summary_data.append([
                f'{period}',
                f"{ic_stats_proc.get(period, {}).get('ic_ir', np.nan):.2f}",
                f"{q_stats_proc.get(period, {}).get('tmb_sharpe', np.nan):.2f}",
                f"{fm_stats.get(period, {}).get('t_statistic', np.nan):.2f}",
                f"{fm_stats.get(period, {}).get('mean_abs_t_stat', np.nan):.2f}",
                f"{turnover_stats.get(period, {}).get('turnover_annual', np.nan):.2f}"
            ])
        columns = ['周期', 'ICIR', '分层Sharpe', 'F-M t值', 't值绝对值均值', '年化换手率']
        table = ax_f.table(cellText=summary_data, colLabels=columns, loc='center')
        ax_f.set_title('F. 核心指标汇总', fontproperties=cn_font, fontsize=18, y=0.85)
        table.auto_set_font_size(False);
        table.set_fontsize(14)
        for cell in table.get_celld().values():
            cell.set_text_props(fontproperties=cn_font)

        for ax in [ax_b, ax_c]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # --- 最终布局与保存 ---
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        report_dir = base_path / 'reports'
        report_dir.mkdir(parents=True, exist_ok=True)
        save_path = report_dir / f"{factor_name}_unified_report_{default_config}_{run_version_str}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"✓ 统一评估报告已保存至: {save_path}")
        return str(save_path)


    def plot_robustness_report(self,
                               backtest_base_on_index: str,
                               factor_name: str,
                               results_path: str,
                               run_version: str = 'latest'
                               ) -> str:
        """
        【V2.1 数据结构适配版】生成 C2C vs. O2C 稳健性对比报告 (2x2布局)。
        能够正确地从包含 raw/processed 的新数据结构中，提取【processed】结果进行对比。
        """
        logger.info(f"为因子 {factor_name} (版本: {run_version}) 生成稳健性对比报告...")

        # --- 1. 定位并加载 C2C 和 O2C 两份【指定版本】的数据 ---
        base_path = Path(results_path) / backtest_base_on_index / factor_name

        def _find_target_version_path(config_path, version):
            if not config_path.is_dir(): return None
            version_dirs = [d for d in config_path.iterdir() if d.is_dir()]
            if not version_dirs: return None
            if version == 'latest':
                return sorted(version_dirs)[-1]
            else:
                path_to_find = config_path / version
                return path_to_find if path_to_find in version_dirs else None

        c2c_version_path = _find_target_version_path(base_path / 'c2c', run_version)
        o2c_version_path = _find_target_version_path(base_path / 'o2c', run_version)

        if not c2c_version_path or not o2c_version_path:
            logger.warning(f"因子 {factor_name} 的结果不完整 (缺少C2C或O2C的 '{run_version}' 版本)，无法生成稳健性报告。")
            return ""

        try:
            stats_c2c = load_json_with_numpy(c2c_version_path / 'summary_stats.json')
            stats_o2c = load_json_with_numpy(o2c_version_path / 'summary_stats.json')
        except FileNotFoundError:
            logger.warning(f"因子 {factor_name} 的 summary_stats.json 文件缺失，无法生成稳健性报告。")
            return ""

        # --- 【核心修正】从新的数据结构中，提取【processed】部分的统计结果 ---
        ic_stats_c2c = stats_c2c.get('ic_analysis_processed', {})
        ic_stats_o2c = stats_o2c.get('ic_analysis_processed', {})
        q_stats_c2c = stats_c2c.get('quantile_backtest_processed', {})
        q_stats_o2c = stats_o2c.get('quantile_backtest_processed', {})
        fm_stats_o2c = stats_o2c.get('fama_macbeth', {})  # F-M通常只在processed上跑

        # 确定最佳周期 (基于更严格的O2C的processed结果)
        try:
            best_period = self._find_best_period_by_rank(ic_stats_o2c, q_stats_o2c, fm_stats_o2c)
        except Exception as e:
            logger.warning(f"因子 {factor_name} 的O2C结果无法确定最佳周期: {e}")
            return ""

        # --- 2. 创建 2x2 图表布局 ---
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        run_version_str = c2c_version_path.name
        fig.suptitle(f'因子 "{factor_name}" Alpha稳健性分析 (C2C vs. O2C)\n(版本: {run_version_str})',
                     fontproperties=cn_font, fontsize=32, y=0.98)
        axes = axes.flatten()

        # --- A. 核心指标对比 (ICIR分组柱状图) ---
        periods = sorted(ic_stats_c2c.keys(), key=lambda p: int(p[:-1]))
        icir_c2c = [ic_stats_c2c.get(p, {}).get('ic_ir', 0) for p in periods]
        icir_o2c = [ic_stats_o2c.get(p, {}).get('ic_ir', 0) for p in periods]
        x = np.arange(len(periods))
        width = 0.35
        axes[0].bar(x - width / 2, icir_c2c, width, label='ICIR (C2C)')
        axes[0].bar(x + width / 2, icir_o2c, width, label='ICIR (O2C)')
        axes[0].set_title('A. ICIR 对比 (纯净因子)', fontproperties=cn_font, fontsize=18)
        # ... (坐标轴、图例等格式化代码不变) ...
        axes[0].set_ylabel('ICIR值', fontproperties=cn_font, fontsize=14)
        axes[0].set_xlabel('持有周期 (天)', fontproperties=cn_font, fontsize=14)
        axes[0].set_xticks(x);
        axes[0].set_xticklabels(periods, fontproperties=cn_font);
        axes[0].legend(prop=cn_font)

        # --- B. 最佳周期多空组合净值对比 ---
        try:
            # 【核心修正】加载带有 `_processed` 后缀的正确文件
            q_returns_c2c = pd.read_parquet(c2c_version_path / f"quantile_returns_processed_{best_period}.parquet")
            q_returns_o2c = pd.read_parquet(o2c_version_path / f"quantile_returns_processed_{best_period}.parquet")
            (1 + q_returns_c2c['TopMinusBottom']).cumprod().plot(ax=axes[1], label=f'多空组合 (C2C, {best_period})',
                                                                 linestyle='--')
            (1 + q_returns_o2c['TopMinusBottom']).cumprod().plot(ax=axes[1], label=f'多空组合 (O2C, {best_period})')
        except FileNotFoundError:
            axes[1].text(0.5, 0.5, "未能加载分层收益数据", ha='center', va='center', fontproperties=cn_font)
        axes[1].set_title(f'B. 最佳周期 ({best_period}) 多空组合净值对比', fontproperties=cn_font, fontsize=18)
        # ... (坐标轴、图例等格式化代码不变) ...
        axes[1].set_ylabel('累计净值', fontproperties=cn_font, fontsize=14);
        axes[1].legend(prop=cn_font);

        # --- C. 最佳周期F-M纯净Alpha对比 ---
        try:
            # F-M 收益序列通常不带后缀，因为只对 processed 因子计算
            fm_returns_c2c = pd.read_parquet(c2c_version_path / f"fm_returns_series_{best_period}.parquet")
            fm_returns_o2c = pd.read_parquet(o2c_version_path / f"fm_returns_series_{best_period}.parquet")
            (1 + fm_returns_c2c).cumprod().plot(ax=axes[2], label=f'F-M Alpha (C2C, {best_period})', linestyle='--')
            (1 + fm_returns_o2c).cumprod().plot(ax=axes[2], label=f'F-M Alpha (O2C, {best_period})')
        except FileNotFoundError:
            axes[2].text(0.5, 0.5, "未能加载F-M收益序列数据", ha='center', va='center', fontproperties=cn_font)
        axes[2].set_title(f'C. 最佳周期 ({best_period}) F-M Alpha 对比', fontproperties=cn_font, fontsize=18)
        # ... (坐标轴、图例等格式化代码不变) ...
        axes[2].set_ylabel('累计净值', fontproperties=cn_font, fontsize=14);
        axes[2].legend(prop=cn_font);

        # --- D. Alpha衰减率量化总览 ---
        axes[3].axis('off')
        summary_data = []
        for period in periods:
            s_c2c = q_stats_c2c.get(period, {}).get('tmb_sharpe', 0)
            s_o2c = q_stats_o2c.get(period, {}).get('tmb_sharpe', 0)
            decay = (s_o2c - s_c2c) / abs(s_c2c) if abs(s_c2c) > 1e-6 else 0
            summary_data.append([f'{period}d', f'{s_c2c:.2f}', f'{s_o2c:.2f}', f'{decay:.1%}'])
        columns = ['周期', 'Sharpe (C2C)', 'Sharpe (O2C)', '衰减率']
        table = axes[3].table(cellText=summary_data, colLabels=columns, loc='center')
        axes[3].set_title('D. Sharpe 稳健性分析 (纯净因子)', fontproperties=cn_font, fontsize=18, y=0.85)
        # ... (表格格式化代码不变) ...
        for cell in table.get_celld().values():
            cell.set_text_props(fontproperties=cn_font)

        # ... (统一调整X轴日期格式和最终保存的逻辑不变) ...
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        report_dir = base_path / 'reports'
        report_dir.mkdir(parents=True, exist_ok=True)

        path = report_dir / f"{factor_name}_robustness_report_{run_version_str}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"✓ 稳健性对比报告已保存至: {path}")
        return str(path)

    # 把它放在你的 VisualizationManager 类或者一个工具模块中
    def _find_best_period_by_rank(self,
                                  ic_stats: Dict,
                                  quantile_stats: Dict,
                                  fm_stats: Dict) -> str:
        """
        通过对多个核心指标进行综合排名，来选择最佳周期。
        """
        if not ic_stats: return "21d"  # 默认值

        periods = list(ic_stats.keys())
        if not periods: return "21d"

        # 1. 提取各个指标的Series
        icir_series = pd.Series({p: ic_stats.get(p, {}).get('ic_ir', -np.inf) for p in periods})
        sharpe_series = pd.Series({p: quantile_stats.get(p, {}).get('tmb_sharpe', -np.inf) for p in periods})
        fmt_series = pd.Series({p: abs(fm_stats.get(p, {}).get('t_statistic', 0)) for p in periods})

        # 2. 对每个指标进行排名 (分数越高，排名越靠前)
        rank_icir = icir_series.rank(ascending=False, method='first')
        rank_sharpe = sharpe_series.rank(ascending=False, method='first')
        rank_fmt = fmt_series.rank(ascending=False, method='first')

        # 3. 计算综合排名得分 (总排名数字越小越好)
        combined_rank_score = rank_icir * 0.4 + rank_sharpe * 0.4 + rank_fmt * 0.2

        # 4. 选出综合排名得分最低（即排名最靠前）的周期
        best_period = combined_rank_score.idxmin()

        return best_period

def _find_target_version_path(config_path, version):
        if not config_path.is_dir(): return None
        version_dirs = [d for d in config_path.iterdir() if d.is_dir()]
        if not version_dirs: return None
        if version == 'latest':
            return sorted(version_dirs)[-1]
        else:
            path_to_find = config_path / version
            return path_to_find if path_to_find in version_dirs else None