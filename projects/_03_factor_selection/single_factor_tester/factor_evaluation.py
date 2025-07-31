"""
因子评价模块

提供对单个因子进行有效性测试的功能，包括IC分析和分层回测。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Union, Optional
import logging
from pathlib import Path
import datetime
import yaml

# 添加项目根目录到路径，以便导入自定义模块
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.data_loader import DataLoader
from quant_lib.evaluation import (
    calculate_ic_vectorized  # 添加向量化函数
)
from quant_lib.utils.file_utils import (
    ensure_dir_exists,
    save_to_csv,
    save_to_pickle,
    load_from_yaml
)
from quant_lib.config.constant_config import (
    LOCAL_PARQUET_DATA_DIR,
    RESULT_DIR
)
from quant_lib.config.logger_config import setup_logger

# 配置日志
logger = setup_logger(__name__)


class FactorEvaluator:
    """因子评价器类"""
    
    def __init__(self, 
                factor_df: pd.DataFrame, 
                price_df: pd.DataFrame,
                start_date: str,
                end_date: str,
                n_groups: int = 5,
                forward_periods: List[int] = [1, 5, 20],
                result_dir: Optional[Path] = None):
        """
        初始化因子评价器
        
        Args:
            factor_df: 因子DataFrame，index为日期，columns为股票代码
            price_df: 价格DataFrame，index为日期，columns为股票代码
            start_date: 开始日期
            end_date: 结束日期
            n_groups: 分层回测的分组数
            forward_periods: 未来收益率计算的时间周期列表，单位为交易日
            result_dir: 结果保存目录，如果为None则不保存结果
        """
        self.factor_df = factor_df
        self.price_df = price_df
        self.start_date = start_date
        self.end_date = end_date
        self.n_groups = n_groups
        self.forward_periods = forward_periods
        self.result_dir = result_dir
        
        # 筛选时间范围
        self._filter_time_range()
        
        # 初始化结果字典
        self.results = {
            'ic': {},
            'layered_returns': {},
            'metrics': {}
        }
        
        # 初始化未来收益率字典
        self.forward_returns_dict = {}
        
        logger.info(f"因子评价器初始化完成，时间范围: {start_date} 至 {end_date}")
    
    def _filter_time_range(self):
        """筛选指定时间范围的数据"""
        # 转换日期格式
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        
        # 筛选因子数据
        self.factor_df = self.factor_df.loc[
            (self.factor_df.index >= start_date) & 
            (self.factor_df.index <= end_date)
        ]
        
        # 筛选价格数据（需要包含额外的未来数据用于计算收益率）
        max_period = max(self.forward_periods)
        future_dates = self.price_df.index[self.price_df.index > end_date]
        if len(future_dates) >= max_period:
            extended_end_date = future_dates[max_period - 1]
        else:
            extended_end_date = self.price_df.index[-1]
            logger.warning(f"未来数据不足 {max_period} 个交易日，使用最后可用日期: {extended_end_date}")
        
        self.price_df = self.price_df.loc[
            (self.price_df.index >= start_date) & 
            (self.price_df.index <= extended_end_date)
        ]
    
    def comprehensive_evaluation(self):
        """
        运行因子评价
        
        执行IC分析和分层回测，计算相关指标
        """
        logger.info("开始运行因子评价...")
        
        # --- 优化点：预先计算所有周期的未来收益率 ---
        logger.info("预计算所有周期的未来收益率...")
        for period in self.forward_periods:
            self.forward_returns_dict[period] = self.price_df.shift(-period) / self.price_df - 1
            logger.debug(f"已计算 {period} 日未来收益率")
        
        # 计算IC
        self._calculate_ic()
        
        # 执行分层回测
        self._run_layered_backtest()
        
        # 计算评价指标
        self._calculate_metrics()
        
        # 保存结果
        if self.result_dir:
            self._save_results()
        
        logger.info("因子评价完成")
        return self.results
    
    def _calculate_ic(self):
        """计算信息系数(IC)"""
        logger.info("计算信息系数(IC)...")
        
        # 计算不同期限的IC
        for period in self.forward_periods:
            # 使用预计算的未来收益率
            forward_returns = self.forward_returns_dict[period]
            
            # 使用向量化计算IC（更高效）
            ic_series = calculate_ic_vectorized(self.factor_df, forward_returns, method='spearman')
            
            # 存储结果
            self.results['ic'][period] = ic_series
            
            # 计算IC统计指标
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std if ic_std != 0 else 0
            #胜率！。（表示正确出现的次数/总次数） 何为正确出现：均值为负，表示负相关，我们只考虑ic里面为负的才是正确预测
            ic_win_rate = ((ic_series * ic_mean) > 0).mean() #这个就是计算胜率，简化版！
            ic_t_stat, ic_p_value = stats.ttest_1samp(ic_series, 0)
            
            # 存储IC统计指标
            self.results['metrics'][f'ic_{period}'] = {
                'mean': ic_mean,
                'std': ic_std,
                'ir': ic_ir,
                'positive_ratio': ic_win_rate,
                't_stat': ic_t_stat,
                'p_value': ic_p_value,
                'significant': ic_p_value < 0.05
            }
            
            logger.info(f"{period}日IC - 均值: {ic_mean:.4f}, IR: {ic_ir:.4f}, "
                       f"显著性: {ic_p_value < 0.05} (p值: {ic_p_value:.4f})")
    
    def _run_layered_backtest(self):
        """执行分层回测"""
        logger.info("执行分层回测...")
        
        # 对每个预测期执行分层回测
        for period in self.forward_periods:
            # 使用预计算的未来收益率
            forward_returns = self.forward_returns_dict[period]
            
            # 初始化分组收益率DataFrame
            group_returns = pd.DataFrame(
                index=self.factor_df.index,
                columns=[f'G{i+1}' for i in range(self.n_groups)] + ['TopMinusBottom']
            )
            
            # 对每个日期进行分组回测
            for date in self.factor_df.index:
                if date not in forward_returns.index:
                    continue
                
                # 获取当天的因子值和未来收益率
                factor_values = self.factor_df.loc[date].dropna()
                returns = forward_returns.loc[date].dropna()
                
                # 找出共同的股票
                common_stocks = factor_values.index.intersection(returns.index)
                
                if len(common_stocks) < self.n_groups * 5:  # 每组至少需要5只股票
                    continue
                
                # 对因子值进行排序并分组
                factor_rank = pd.qcut(
                    factor_values[common_stocks],
                    self.n_groups,
                    labels=[f'G{i+1}' for i in range(self.n_groups)],
                    duplicates='drop'  # 处理极端情况下的重复边界
                )
                
                # 计算各分组的平均收益
                for i in range(self.n_groups):
                    group_label = f'G{i+1}'
                    stocks_in_group = factor_rank[factor_rank == group_label].index
                    if len(stocks_in_group) > 0:
                        group_returns.loc[date, group_label] = returns[stocks_in_group].mean()
                
                # 计算多空组合收益
                top_group = f'G{self.n_groups}'
                bottom_group = 'G1'
                
                # 更健壮的多空组合收益计算
                if pd.notna(group_returns.loc[date, top_group]) and pd.notna(group_returns.loc[date, bottom_group]):
                    group_returns.loc[date, 'TopMinusBottom'] = (
                        group_returns.loc[date, top_group] - 
                        group_returns.loc[date, bottom_group]
                    )
            
            # 计算累计收益
            cumulative_returns = (1 + group_returns).cumprod()
            
            # 存储结果
            self.results['layered_returns'][period] = {
                'returns': group_returns,
                'cumulative_returns': cumulative_returns
            }
            
            # 计算各组合的年化收益率和夏普比率
            for group in group_returns.columns:
                returns_series = group_returns[group].dropna()
                if len(returns_series) > 0:
                    annual_return = ((1 + returns_series).prod()) ** (252 / len(returns_series)) - 1
                    annual_volatility = returns_series.std() * np.sqrt(252)
                    sharpe = annual_return / annual_volatility if annual_volatility != 0 else 0
                    
                    # 存储指标
                    self.results['metrics'][f'group_{period}_{group}'] = {
                        'annual_return': annual_return,
                        'annual_volatility': annual_volatility,
                        'sharpe': sharpe
                    }
            
            # 输出多空组合的表现
            tmb_metrics = self.results['metrics'][f'group_{period}_TopMinusBottom']
            logger.info(f"{period}日多空组合 - 年化收益率: {tmb_metrics['annual_return']:.2%}, "
                       f"夏普比率: {tmb_metrics['sharpe']:.2f}")
    
    def _calculate_metrics(self):
        """计算综合评价指标"""
        logger.info("计算综合评价指标...")
        
        # 计算因子整体评分
        # 这里使用一个简单的评分方法：IC IR的平均值 + 多空组合夏普比率的平均值
        ic_ir_avg = np.mean([
            self.results['metrics'][f'ic_{period}']['ir']
            for period in self.forward_periods
        ])
        
        tmb_sharpe_avg = np.mean([
            self.results['metrics'][f'group_{period}_TopMinusBottom']['sharpe']
            for period in self.forward_periods
        ])
        
        # 计算整体得分
        overall_score = (ic_ir_avg + tmb_sharpe_avg) / 2
        
        # 存储整体评分
        self.results['metrics']['overall'] = {
            'ic_ir_avg': ic_ir_avg,
            'tmb_sharpe_avg': tmb_sharpe_avg,
            'overall_score': overall_score,
            'is_effective': overall_score > 0.3  # 使用0.3作为有效性阈值
        }
        
        logger.info(f"因子整体评分: {overall_score:.4f}, "
                   f"是否有效: {overall_score > 0.3}")

    def plot_ic_analysis(self, figsize: Tuple[int, int] = (15, 10)):
        """
        绘制IC分析图
        
        Args:
            figsize: 图表大小
        """
        # 创建子图
        fig, axes = plt.subplots(len(self.forward_periods), 1, figsize=figsize)
        if len(self.forward_periods) == 1:
            axes = [axes]
        
        # 绘制每个周期的IC时间序列
        for i, period in enumerate(self.forward_periods):
            ax = axes[i]
            ic_series = self.results['ic'][period]
            
            # 绘制IC序列
            ax.plot(ic_series, label='IC', color='blue', alpha=0.7)
            
            # 绘制均值线
            ax.axhline(y=ic_series.mean(), color='red', linestyle='-', 
                      label=f'均值: {ic_series.mean():.4f}')
            
            # 绘制0线
            ax.axhline(y=0, color='black', linestyle='--')
            
            # 添加标题和标签
            ax.set_title(f'{period}日IC时间序列 (IR: {self.results["metrics"][f"ic_{period}"]["ir"]:.4f})')
            ax.set_xlabel('日期')
            ax.set_ylabel('IC值')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        if self.result_dir:
            plt.savefig(self.result_dir / 'ic_analysis.png')
        
        return fig
    
    def plot_layered_returns(self, figsize: Tuple[int, int] = (15, 12)):
        """
        绘制分层回测图
        
        Args:
            figsize: 图表大小
        """
        # 创建子图
        fig, axes = plt.subplots(len(self.forward_periods), 1, figsize=figsize)
        if len(self.forward_periods) == 1:
            axes = [axes]
        
        # 绘制每个周期的分层回测结果
        for i, period in enumerate(self.forward_periods):
            ax = axes[i]
            cumulative_returns = self.results['layered_returns'][period]['cumulative_returns']
            
            # 使用不同颜色绘制各分组的累计收益曲线
            # 更新为新版matplotlib的颜色映射获取方式
            cmap = plt.colormaps.get('RdYlGn')
            colors = [cmap(j/(self.n_groups-1)) for j in range(self.n_groups)]
            
            for j in range(self.n_groups):
                group = f'G{j+1}'
                ax.plot(cumulative_returns[group], 
                       label=group, 
                       color=colors[j],
                       linewidth=2)
            
            # 绘制多空组合
            ax.plot(cumulative_returns['TopMinusBottom'], 
                   label='多空组合', 
                   color='black', 
                   linestyle='--',
                   linewidth=2)
            
            # 添加标题和标签
            tmb_metrics = self.results['metrics'][f'group_{period}_TopMinusBottom']
            ax.set_title(f'{period}日持有期分层回测 '
                        f'(多空组合年化收益: {tmb_metrics["annual_return"]:.2%}, '
                        f'夏普比率: {tmb_metrics["sharpe"]:.2f})')
            ax.set_xlabel('日期')
            ax.set_ylabel('累计收益')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        if self.result_dir:
            plt.savefig(self.result_dir / 'layered_returns.png')
        
        return fig
    
    def plot_monotonicity(self, figsize: Tuple[int, int] = (15, 10)):
        """
        绘制单调性分析图
        
        Args:
            figsize: 图表大小
        """
        # 创建子图
        fig, axes = plt.subplots(len(self.forward_periods), 1, figsize=figsize)
        if len(self.forward_periods) == 1:
            axes = [axes]
        
        # 绘制每个周期的单调性分析
        for i, period in enumerate(self.forward_periods):
            ax = axes[i]
            
            # 计算各分组的平均年化收益率
            annual_returns = []
            group_labels = []
            
            for j in range(self.n_groups):
                group = f'G{j+1}'
                metrics = self.results['metrics'][f'group_{period}_{group}']
                annual_returns.append(metrics['annual_return'])
                group_labels.append(group)
            
            # 获取颜色映射
            cmap = plt.colormaps.get('RdYlGn')
            colors = [cmap(j/(self.n_groups-1)) for j in range(self.n_groups)]
            
            # 绘制条形图
            bars = ax.bar(group_labels, annual_returns, color=colors)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            # 添加标题和标签
            ax.set_title(f'{period}日持有期各分组年化收益率')
            ax.set_xlabel('分组')
            ax.set_ylabel('年化收益率')
            ax.grid(True, axis='y')
            
            # 计算单调性指标
            monotonicity = np.corrcoef(np.arange(1, self.n_groups + 1), annual_returns)[0, 1]
            ax.text(0.02, 0.95, f'单调性系数: {monotonicity:.4f}', 
                   transform=ax.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        if self.result_dir:
            plt.savefig(self.result_dir / 'monotonicity.png')
        
        return fig
    
    def plot_all(self):
        """绘制所有分析图表"""
        self.plot_ic_analysis()
        self.plot_layered_returns()
        self.plot_monotonicity()
        # 不再在此处调用plt.show()，由调用者决定何时显示图表
    
    def get_summary(self) -> pd.DataFrame:
        """
        获取因子评价摘要
        
        Returns:
            摘要DataFrame
        """
        # 提取关键指标
        summary_data = {
            'factor_name': ['Factor'],
            'overall_score': [self.results['metrics']['overall']['overall_score']],
            'is_effective': [self.results['metrics']['overall']['is_effective']]
        }
        
        # 添加各期IC指标
        for period in self.forward_periods:
            ic_metrics = self.results['metrics'][f'ic_{period}']
            summary_data[f'ic_{period}_mean'] = [ic_metrics['mean']]
            summary_data[f'ic_{period}_ir'] = [ic_metrics['ir']]
            summary_data[f'ic_{period}_significant'] = [ic_metrics['significant']]
        
        # 添加各期多空组合指标
        for period in self.forward_periods:
            tmb_metrics = self.results['metrics'][f'group_{period}_TopMinusBottom']
            summary_data[f'tmb_{period}_return'] = [tmb_metrics['annual_return']]
            summary_data[f'tmb_{period}_sharpe'] = [tmb_metrics['sharpe']]
        
        return pd.DataFrame(summary_data)




