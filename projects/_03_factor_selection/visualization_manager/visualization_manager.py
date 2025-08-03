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
        
        logger.info(f"可视化管理器初始化完成，输出目录: {output_dir}")

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
                                   factor_name: str,
                                   ic_series_periods_dict: Dict[str, pd.Series],
                                   ic_stats_periods_dict: Dict[str, Dict[str, Any]],
                                   quantile_returns_series_periods_dict: Dict[str, pd.DataFrame],
                                   quantile_stats_periods_dict: Dict[str, Dict[str, Any]],
                                   factor_returns_series_periods_dict: Dict[str, pd.Series],
                                   fm_stat_results_periods_dict: Dict[str, Dict[str, Any]],
                                   save_plots: bool = True,
                                   ):

        """
        【V2版】一张图全面展示因子有效性、稳定性、区分度和纯净Alpha。
        借鉴了研报的视觉设计，同时保留了多周期对比的核心优势。
        """
        logger.info(f"开始绘图for:{factor_name}")
        primary_period = list(ic_series_periods_dict.keys())[-1]
        # 1. 创建 2x2 图表布局
        fig = plt.figure(figsize=(24, 20))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
        fig.suptitle(f'单因子分析报告 (Factor Tear Sheet): {factor_name}', fontsize=28, y=0.97)

        # 2. 【左上，升级版】IC序列(柱状) + 累计IC(折线)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1_twin = ax1.twinx()  # 创建次坐标轴

        # a) 绘制核心周期的IC序列柱状图
        primary_ic = ic_series_periods_dict.get(primary_period)
        if primary_ic is not None:
            # 【关键修改】使用 ax1.bar 替代 plot(kind='bar') 以保留datetime轴
            ax1.bar(primary_ic.index, primary_ic.values, width=1.0, color='royalblue', alpha=0.6,
                    label=f'IC序列 ({primary_period})')

        ax1.axhline(0, color='black', linestyle='--', lw=1)
        ax1.set_ylabel('IC值 (单期)', color='royalblue', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='royalblue')

        # b) 在次坐标轴上绘制所有周期的累计IC折线图
        for period, ic_series in ic_series_periods_dict.items():
            ic_series.cumsum().plot(ax=ax1_twin, label=f'累计IC ({period})', lw=2.5)

        ax1_twin.set_ylabel('累计IC', fontsize=14)
        ax1.set_title('A. 因子IC序列与累计IC (有效性)', fontsize=18)

        # 【关键升级】应用与研报一致的日期格式
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 每6个月一个主刻度
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 格式化为 年-月
        ax1.tick_params(axis='x', rotation=45)

        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        ax1.grid(False)

        # 3. 【右上，维持原版】分层年化收益率 (柱状图)
        ax2 = fig.add_subplot(gs[0, 1])
        quantile_annual_returns = pd.DataFrame({
            period: stats['mean_returns'].drop('TopMinusBottom') * (252 / int(period[:-1]))
            for period, stats in quantile_stats_periods_dict.items()
        })
        quantile_annual_returns.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('B. 分层年化收益率 (单调性)', fontsize=18)
        ax2.set_ylabel('年化收益率')
        ax2.axhline(0, color='black', linestyle='--', lw=1)
        ax2.tick_params(axis='x', rotation=0)
        ax2.legend(title='周期')
        ax2.grid(axis='y')

        # 4. 【左下，升级版】分层累计净值曲线
        ax3 = fig.add_subplot(gs[1, 0])
        primary_q_returns = quantile_returns_series_periods_dict.get(primary_period)
        if primary_q_returns is not None:
            # a) 绘制多空组合累计收益（灰色区域）
            tmb_cum_returns = (1 + primary_q_returns['TopMinusBottom']).cumprod()
            ax3.fill_between(tmb_cum_returns.index, 1, tmb_cum_returns, color='grey', alpha=0.3,
                             label=f'多空组合 ({primary_period})')

            # b) 绘制每个分层的累计净值曲线
            quantile_cols = [q for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'] if q in primary_q_returns.columns]
            for quantile in quantile_cols:
                (1 + primary_q_returns[quantile]).cumprod().plot(ax=ax3, label=f'{quantile} ({primary_period})',
                                                                 lw=2)

        ax3.set_title(f'C. 分层累计净值 ({primary_period})', fontsize=18)
        ax3.set_ylabel('累计净值')
        ax3.axhline(1, color='black', linestyle='--', lw=1)
        ax3.legend()
        ax3.grid(True)

        # 【关键升级】应用与研报一致的日期格式
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 每6个月一个主刻度
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 格式化为 年-月
        ax3.tick_params(axis='x', rotation=45)

        # 5. 【右下，维持原版】核心指标汇总 (表格)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        summary_data = []
        periods = sorted(ic_stats_periods_dict.keys(), key=lambda x: int(x[:-1]))
        for period in periods:
            ic_ir = ic_stats_periods_dict.get(period, {}).get('ic_ir', np.nan)
            tmb_sharpe = quantile_stats_periods_dict.get(period, {}).get('tmb_sharpe', np.nan)
            fm_t_stat = fm_stat_results_periods_dict.get(period, {}).get('t_statistic', np.nan)
            summary_data.append([f'{period}', f'{ic_ir:.2f}', f'{tmb_sharpe:.2f}', f'{fm_t_stat:.2f}'])

        columns = ['周期', 'ICIR', '分层Sharpe', 'F-M t值']
        table = ax4.table(cellText=summary_data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(16)
        table.scale(1.0, 2.8)
        ax4.set_title('D. 核心指标汇总', fontsize=18, y=0.85)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_plots:
            # 假设 self.output_dir 存在
            plot_path = self.output_dir / f"{factor_name}_evaluation_v2.png"  # 简化版
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            log_success(f"保存{factor_name}测评图片")
            return str(plot_path)
        else:
            plt.show()
            return ""
