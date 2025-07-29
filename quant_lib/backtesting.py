"""
回测引擎模块

提供股票策略回测功能，支持多种回测方式和评估指标。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# 获取模块级别的logger
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置类"""
    start_date: str
    end_date: str
    stack_pool: List[str]
    rebalance_freq: str = 'day'  # 'day', 'week', 'month', 'quarter'
    n_stocks: int = 10
    fee_rate: float = 0.0003 * 2 + 0.001
    slippage: float = 0.0012
    benchmark: str = '000300.SH'
    capital: float = 1000000.0,
    sl_stop: float = 0.1
    tp_stop: float = 0.2
    sl_trail: float = 0.08


class BacktestResult:
    """回测结果类"""

    def __init__(self,
                 portfolio_returns: pd.Series,
                 benchmark_returns: pd.Series,
                 positions: pd.DataFrame,
                 turnover: pd.Series,
                 config: BacktestConfig):
        """
        初始化回测结果
        
        Args:
            portfolio_returns: 组合每日收益率序列
            benchmark_returns: 基准每日收益率序列
            positions: 持仓矩阵，index为日期，columns为股票代码
            turnover: 换手率序列
            config: 回测配置
        """
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.positions = positions
        self.turnover = turnover
        self.config = config

        # 计算累计收益
        self.portfolio_cumulative = (1 + portfolio_returns).cumprod() - 1
        self.benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1

        # 计算回撤
        self.portfolio_drawdown = self._calculate_drawdown(self.portfolio_cumulative)
        self.benchmark_drawdown = self._calculate_drawdown(self.benchmark_cumulative)

        # 计算性能指标
        self._calculate_metrics()

    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """计算回撤序列"""
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / (1 + rolling_max)
        return drawdown

    def _calculate_metrics(self):
        """计算性能指标"""
        # 年化收益率
        days = len(self.portfolio_returns)
        self.annual_return = ((1 + self.portfolio_returns).prod()) ** (252 / days) - 1
        self.benchmark_annual_return = ((1 + self.benchmark_returns).prod()) ** (252 / days) - 1

        # 年化波动率
        self.annual_volatility = self.portfolio_returns.std() * np.sqrt(252)
        self.benchmark_annual_volatility = self.benchmark_returns.std() * np.sqrt(252)

        # 夏普比率
        self.sharpe_ratio = self.annual_return / self.annual_volatility if self.annual_volatility != 0 else 0
        self.benchmark_sharpe_ratio = self.benchmark_annual_return / self.benchmark_annual_volatility if self.benchmark_annual_volatility != 0 else 0

        # 最大回撤
        self.max_drawdown = self.portfolio_drawdown.min()
        self.benchmark_max_drawdown = self.benchmark_drawdown.min()

        # 胜率
        excess_returns = self.portfolio_returns - self.benchmark_returns
        self.win_rate = (excess_returns > 0).mean()

        # 信息比率
        self.information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(
            252) if excess_returns.std() != 0 else 0

        # 年化超额收益
        self.alpha = self.annual_return - self.benchmark_annual_return

        # 换手率
        self.avg_turnover = self.turnover.mean()

    def summary(self) -> pd.DataFrame:
        """返回回测结果摘要"""
        metrics = {
            '指标': [
                '年化收益率', '年化波动率', '夏普比率', '最大回撤',
                '信息比率', 'Alpha', '胜率', '平均换手率'
            ],
            '策略': [
                f'{self.annual_return:.2%}',
                f'{self.annual_volatility:.2%}',
                f'{self.sharpe_ratio:.2f}',
                f'{self.max_drawdown:.2%}',
                f'{self.information_ratio:.2f}',
                f'{self.alpha:.2%}',
                f'{self.win_rate:.2%}',
                f'{self.avg_turnover:.2%}'
            ],
            '基准': [
                f'{self.benchmark_annual_return:.2%}',
                f'{self.benchmark_annual_volatility:.2%}',
                f'{self.benchmark_sharpe_ratio:.2f}',
                f'{self.benchmark_max_drawdown:.2%}',
                'N/A',
                'N/A',
                'N/A',
                'N/A'
            ]
        }
        return pd.DataFrame(metrics)

    def plot_returns(self, figsize: Tuple[int, int] = (12, 6)):
        """绘制收益率曲线"""
        plt.figure(figsize=figsize)
        plt.plot(self.portfolio_cumulative, label='策略')
        plt.plot(self.benchmark_cumulative, label='基准')
        plt.title('策略与基准累计收益对比')
        plt.xlabel('日期')
        plt.ylabel('累计收益')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_drawdown(self, figsize: Tuple[int, int] = (12, 6)):
        """绘制回撤曲线"""
        plt.figure(figsize=figsize)
        plt.plot(self.portfolio_drawdown, label='策略')
        plt.plot(self.benchmark_drawdown, label='基准')
        plt.title('策略与基准回撤对比')
        plt.xlabel('日期')
        plt.ylabel('回撤')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_monthly_returns(self, figsize: Tuple[int, int] = (12, 6)):
        """绘制月度收益热力图"""
        monthly_returns = self.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns.to_frame('returns')
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month

        pivot_table = monthly_returns.pivot_table(
            index='year',
            columns='month',
            values='returns'
        )

        plt.figure(figsize=figsize)
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.1%',
            cmap='RdYlGn',
            center=0,
            linewidths=1
        )
        plt.title('月度收益热力图')
        plt.show()


class BacktestEngine:
    """回测引擎类"""

    def __init__(self, config: BacktestConfig):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
        """
        self.config = config
        logger.info(f"初始化回测引擎: {config.start_date} 至 {config.end_date}")

    def run(self,
            price_df: pd.DataFrame,
            factor_df: pd.DataFrame,
            benchmark_returns: pd.Series) -> BacktestResult:
        """
        运行回测
        
        Args:
            price_df: 价格DataFrame，index为日期，columns为股票代码
            factor_df: 因子DataFrame，index为日期，columns为股票代码
            benchmark_returns: 基准收益率序列
            
        Returns:
            回测结果对象
        """
        logger.info("开始回测...")

        # 初始化结果数据结构
        dates = price_df.index
        portfolio_returns = pd.Series(0.0, index=dates)
        positions = pd.DataFrame(0.0, index=dates, columns=price_df.columns)
        turnover = pd.Series(0.0, index=dates)

        # 获取调仓日期
        rebalance_dates = self._get_rebalance_dates(dates)
        logger.info(f"调仓日期: {len(rebalance_dates)} 个")

        # 当前持仓
        current_positions = pd.Series(0.0, index=price_df.columns)

        # 遍历每个交易日
        for i, date in enumerate(dates):
            # 如果是调仓日，更新持仓
            if date in rebalance_dates:
                # 获取当前因子值
                if date in factor_df.index:
                    factor_values = factor_df.loc[date].dropna()

                    # 选择股票
                    selected_stocks = self._select_stocks(factor_values)

                    # 计算目标持仓
                    target_positions = self._calculate_target_positions(selected_stocks, price_df.loc[date])

                    # 计算换手率
                    turnover.loc[date] = self._calculate_turnover(current_positions, target_positions)

                    # 更新当前持仓
                    current_positions = target_positions

            # 记录当日持仓
            positions.loc[date] = current_positions

            # 如果不是最后一天，计算收益率
            if i < len(dates) - 1:
                next_date = dates[i + 1]
                daily_returns = price_df.loc[next_date] / price_df.loc[date] - 1
                portfolio_return = (current_positions * daily_returns).sum()

                # 扣除费用
                if date in rebalance_dates:
                    portfolio_return -= turnover.loc[date] * (self.config.fee_rate + self.config.slippage)

                portfolio_returns.loc[next_date] = portfolio_return

        # 创建回测结果对象
        result = BacktestResult(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            positions=positions,
            turnover=turnover,
            config=self.config
        )

        logger.info("回测完成")
        return result

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """获取调仓日期"""
        if self.config.rebalance_freq == 'day':
            return dates
        elif self.config.rebalance_freq == 'week':
            return dates[dates.weekday == 4]  # 每周五
        elif self.config.rebalance_freq == 'month':
            return dates[dates.is_month_end]
        elif self.config.rebalance_freq == 'quarter':
            return dates[dates.is_quarter_end]
        else:
            raise ValueError(f"不支持的调仓频率: {self.config.rebalance_freq}")

    def _select_stocks(self, factor_values: pd.Series) -> pd.Series:
        """选择股票"""
        # 按因子值排序，选择前N只股票
        sorted_stocks = factor_values.sort_values(ascending=False)
        selected_stocks = sorted_stocks.iloc[:self.config.n_stocks]
        return selected_stocks

    def _calculate_target_positions(self,
                                    selected_stocks: pd.Series,
                                    prices: pd.Series) -> pd.Series:
        """计算目标持仓"""
        # 等权重分配资金
        weights = pd.Series(1.0 / len(selected_stocks), index=selected_stocks.index)

        # 计算每只股票的目标持仓金额
        target_values = weights * self.config.capital

        # 计算股票数量
        stock_prices = prices.reindex(selected_stocks.index)
        stock_quantities = target_values / stock_prices

        # 创建完整的持仓Series
        target_positions = pd.Series(0.0, index=prices.index)
        target_positions.loc[stock_quantities.index] = stock_quantities / self.config.capital

        return target_positions

    def _calculate_turnover(self,
                            old_positions: pd.Series,
                            new_positions: pd.Series) -> float:
        """计算换手率"""
        return abs(old_positions - new_positions).sum() / 2


# 工厂函数，方便创建BacktestEngine实例
def create_backtest_engine(
        start_date: str,
        end_date: str,
        stack_pool: List[str],
        rebalance_freq: str = 'month',
        n_stocks: int = 50,
        fee_rate: float = 0.0003*2 + 0.001,
        slippage: float = 0.0012,
        benchmark: str = '000300.SH',
        capital: float = 1000000.0
) -> BacktestEngine:
    """
    创建BacktestEngine实例
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        stack_pool: 股票池
        rebalance_freq: 调仓频率，'day', 'week', 'month', 'quarter'
        n_stocks: 持仓股票数量
        fee_rate: 交易费率
        slippage: 滑点
        benchmark: 基准指数代码
        capital: 初始资金
        
    Returns:
        BacktestEngine实例
    """
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        stack_pool=stack_pool,
        rebalance_freq=rebalance_freq,
        n_stocks=n_stocks,
        fee_rate=fee_rate,
        slippage=slippage,
        benchmark=benchmark,
        capital=capital
    )
    return BacktestEngine(config)
