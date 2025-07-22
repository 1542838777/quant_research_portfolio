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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
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

from quant_lib.config.logger_config import setup_logger

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
                                  test_results: Dict[str, Any],
                                  factor_name: str,
                                  save_plots: bool = True) -> Dict[str, str]:
        """
        绘制单因子测试结果
        
        Args:
            test_results: 测试结果字典
            factor_name: 因子名称
            save_plots: 是否保存图表
            
        Returns:
            生成的图表文件路径字典
        """
        logger.info(f"开始绘制因子 {factor_name} 的测试结果...")
        
        plot_paths = {}
        
        # 1. IC分析图
        if 'ic_analysis' in test_results:
            ic_path = self._plot_ic_analysis(
                test_results['ic_analysis'], factor_name, save_plots
            )
            plot_paths['ic_analysis'] = ic_path
        
        # 2. 分层回测图
        if 'quantile_backtest' in test_results:
            quantile_path = self._plot_quantile_backtest(
                test_results['quantile_backtest'], factor_name, save_plots
            )
            plot_paths['quantile_backtest'] = quantile_path
        
        # 3. Fama-MacBeth回归图
        if 'fama_macbeth' in test_results:
            fm_path = self._plot_fama_macbeth(
                test_results['fama_macbeth'], factor_name, save_plots
            )
            plot_paths['fama_macbeth'] = fm_path
        
        # 4. 综合评价图
        if 'evaluate_factor_score' in test_results:
            eval_path = self._plot_evaluation_summary(
                test_results['evaluate_factor_score'], factor_name, save_plots
            )
            plot_paths['evaluate_factor_score'] = eval_path
        
        logger.info(f"因子 {factor_name} 图表生成完成")
        return plot_paths
    
    def _plot_ic_analysis(self, 
                         ic_results: Dict[str, Any], 
                         factor_name: str,
                         save_plots: bool) -> str:
        """绘制IC分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{factor_name} - IC分析', fontsize=16, fontweight='bold')
        
        # IC时间序列
        if 'IC_Series' in ic_results:
            ic_series = ic_results['IC_Series']
            axes[0, 0].plot(ic_series.index, ic_series.values, 
                           color=self.colors['primary'], alpha=0.7)
            axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('IC时间序列')
            axes[0, 0].set_ylabel('IC值')
            axes[0, 0].grid(True, alpha=0.3)
        
        # IC分布直方图
        if 'IC_Series' in ic_results:
            axes[0, 1].hist(ic_series.values, bins=30, 
                           color=self.colors['secondary'], alpha=0.7)
            axes[0, 1].axvline(x=ic_series.mean(), color='red', 
                              linestyle='--', label=f'均值: {ic_series.mean():.4f}')
            axes[0, 1].set_title('IC分布')
            axes[0, 1].set_xlabel('IC值')
            axes[0, 1].set_ylabel('频数')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # IC累积图
        if 'IC_Series' in ic_results:
            ic_cumsum = ic_series.cumsum()
            axes[1, 0].plot(ic_cumsum.index, ic_cumsum.values, 
                           color=self.colors['success'])
            axes[1, 0].set_title('IC累积图')
            axes[1, 0].set_ylabel('累积IC')
            axes[1, 0].grid(True, alpha=0.3)
        
        # IC统计指标
        if 'IC_Mean' in ic_results:
            metrics = {
                'IC均值': ic_results.get('IC_Mean', 0),
                'IC标准差': ic_results.get('IC_Std', 0),
                'IC_IR': ic_results.get('IC_IR', 0),
                'IC胜率': ic_results.get('IC_WinRate', 0)
            }
            
            y_pos = np.arange(len(metrics))
            values = list(metrics.values())
            
            axes[1, 1].barh(y_pos, values, color=self.colors['info'])
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(list(metrics.keys()))
            axes[1, 1].set_title('IC统计指标')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(values):
                axes[1, 1].text(v + 0.001, i, f'{v:.4f}', 
                               va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f"{factor_name}_ic_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def _plot_quantile_backtest(self, 
                               quantile_results: Dict[str, Any],
                               factor_name: str,
                               save_plots: bool) -> str:
        """绘制分层回测图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{factor_name} - 分层回测分析', fontsize=16, fontweight='bold')
        
        # 分层净值曲线
        if 'quantile_returns' in quantile_results:
            quantile_returns = quantile_results['quantile_returns']
            cumulative_returns = (1 + quantile_returns).cumprod()
            
            for i, col in enumerate(cumulative_returns.columns):
                axes[0, 0].plot(cumulative_returns.index, cumulative_returns[col], 
                               label=f'Q{i+1}', linewidth=2)
            
            axes[0, 0].set_title('分层净值曲线')
            axes[0, 0].set_ylabel('累积收益')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 多空组合净值
        if 'long_short_returns' in quantile_results:
            ls_returns = quantile_results['long_short_returns']
            ls_cumulative = (1 + ls_returns).cumprod()
            
            axes[0, 1].plot(ls_cumulative.index, ls_cumulative.values, 
                           color=self.colors['danger'], linewidth=2)
            axes[0, 1].set_title('多空组合净值')
            axes[0, 1].set_ylabel('累积收益')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 分层收益率分布
        if 'quantile_returns' in quantile_results:
            quantile_returns.boxplot(ax=axes[1, 0])
            axes[1, 0].set_title('分层收益率分布')
            axes[1, 0].set_ylabel('收益率')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 分层统计指标
        if 'quantile_stats' in quantile_results:
            stats = quantile_results['quantile_stats']
            
            # 绘制年化收益率
            annual_returns = stats.get('annual_return', {})
            if annual_returns:
                x_pos = np.arange(len(annual_returns))
                values = list(annual_returns.values())
                
                bars = axes[1, 1].bar(x_pos, values, color=self.colors['primary'])
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels(list(annual_returns.keys()))
                axes[1, 1].set_title('分层年化收益率')
                axes[1, 1].set_ylabel('年化收益率')
                axes[1, 1].grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f"{factor_name}_quantile_backtest.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def _plot_fama_macbeth(self, 
                          fm_results: Dict[str, Any],
                          factor_name: str,
                          save_plots: bool) -> str:
        """绘制Fama-MacBeth回归图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{factor_name} - Fama-MacBeth回归分析', fontsize=16, fontweight='bold')
        
        # 因子收益率时间序列
        if 'factor_returns' in fm_results:
            factor_returns = fm_results['factor_returns']
            axes[0].plot(factor_returns.index, factor_returns.values, 
                        color=self.colors['primary'], alpha=0.7)
            axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0].set_title('因子收益率时间序列')
            axes[0].set_ylabel('因子收益率')
            axes[0].grid(True, alpha=0.3)
        
        # 统计检验结果
        stats_text = []
        if 't_statistic' in fm_results:
            stats_text.append(f"t统计量: {fm_results['t_statistic']:.4f}")
        if 'p_value' in fm_results:
            stats_text.append(f"p值: {fm_results['p_value']:.4f}")
        if 'factor_return_mean' in fm_results:
            stats_text.append(f"平均因子收益: {fm_results['factor_return_mean']:.4f}")
        if 'factor_return_std' in fm_results:
            stats_text.append(f"因子收益标准差: {fm_results['factor_return_std']:.4f}")
        
        axes[1].text(0.1, 0.5, '\n'.join(stats_text), 
                    transform=axes[1].transAxes, fontsize=12,
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[1].set_title('统计检验结果')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f"{factor_name}_fama_macbeth.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return ""
    
    def _plot_evaluation_summary(self, 
                                evaluation: Dict[str, Any],
                                factor_name: str,
                                save_plots: bool) -> str:
        """绘制综合评价图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{factor_name} - 综合评价', fontsize=16, fontweight='bold')
        
        # 评分雷达图
        if 'scores' in evaluation:
            scores = evaluation['scores']
            categories = list(scores.keys())
            values = list(scores.values())
            
            # 创建雷达图
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            values += values[:1]  # 闭合图形
            angles = np.concatenate((angles, [angles[0]]))
            
            axes[0].plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
            axes[0].fill(angles, values, alpha=0.25, color=self.colors['primary'])
            axes[0].set_xticks(angles[:-1])
            axes[0].set_xticklabels(categories)
            axes[0].set_ylim(0, 1)
            axes[0].set_title('评分雷达图')
            axes[0].grid(True)
        
        # 总评信息
        summary_text = []
        if 'total_score' in evaluation:
            summary_text.append(f"总分: {evaluation['total_score']:.2f}")
        if 'grade' in evaluation:
            summary_text.append(f"评级: {evaluation['grade']}")
        if 'recommendation' in evaluation:
            summary_text.append(f"建议: {evaluation['recommendation']}")
        
        axes[1].text(0.1, 0.5, '\n'.join(summary_text),
                    transform=axes[1].transAxes, fontsize=14,
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        axes[1].set_title('综合评价')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f"{factor_name}_evaluation.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return ""

    def plot_factor_comparison(self,
                              factor_results: Dict[str, Dict[str, Any]],
                              metrics: List[str] = None,
                              save_plots: bool = True) -> str:
        """
        绘制多因子对比图

        Args:
            factor_results: 多个因子的测试结果
            metrics: 对比指标列表
            save_plots: 是否保存图表
        """
        if metrics is None:
            metrics = ['ic_mean', 'ic_ir', 'fm_t_stat', 'overall_score']

        logger.info(f"开始绘制 {len(factor_results)} 个因子的对比图...")

        # 准备数据
        comparison_data = []
        for factor_name, results in factor_results.items():
            row = {'factor_name': factor_name}

            # 提取各项指标
            ic_results = results.get('ic_analysis', {})
            fm_results = results.get('fama_macbeth', {})
            evaluation = results.get('evaluate_factor_score', {})

            row['ic_mean'] = ic_results.get('IC_Mean', 0)
            row['ic_ir'] = ic_results.get('IC_IR', 0)
            row['fm_t_stat'] = fm_results.get('t_statistic', 0)
            row['overall_score'] = evaluation.get('total_score', 0)

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # 创建子图
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('因子性能对比', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for i, metric in enumerate(metrics[:4]):  # 最多显示4个指标
            if metric in df.columns:
                # 排序
                df_sorted = df.sort_values(metric, ascending=False)

                # 绘制柱状图
                bars = axes[i].bar(range(len(df_sorted)), df_sorted[metric],
                                  color=self.colors['primary'])
                axes[i].set_xticks(range(len(df_sorted)))
                axes[i].set_xticklabels(df_sorted['factor_name'], rotation=45, ha='right')
                axes[i].set_title(f'{metric.upper()}对比')
                axes[i].grid(True, alpha=0.3)

                # 添加数值标签
                for bar, value in zip(bars, df_sorted[metric]):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save_plots:
            plot_path = self.output_dir / "factor_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        else:
            plt.show()
            return ""
