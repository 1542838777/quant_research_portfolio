"""
量化回测器 - 专业级因子策略回测工具

核心功能：
1. 多因子策略回测对比
2. 真实交易成本建模
3. 完整的风险调整收益分析
4. 可视化和报告生成
5. 实盘级数据对齐和验证

设计理念：
- 面向对象，可扩展
- 数据安全，严格对齐验证
- 交易成本真实建模
- 结果可复现，可解释
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import warnings
from datetime import datetime

from vectorbt.portfolio import CallSeqType

from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class BacktestConfig:
    """回测配置参数"""
    # 策略参数
    top_quantile: float = 0.2  # 做多分位数阈值（前20%）
    rebalancing_freq: str = 'W'  # 调仓频率 ('M'=月末, 'W'=周末, 'Q'=季末)
    
    # 交易成本参数
    commission_rate: float = 0.0003  # 佣金费率（万3）
    slippage_rate: float = 0.0010  # 滑点率（千1）
    stamp_duty: float = 0.0010  # 印花税（单边，卖出收取）
    min_commission: float = 5.0  # 最小佣金（元）
    
    # 回测参数
    initial_cash: float = 1000000.0  # 初始资金（100万）
    max_positions: int = 50  # 最大持仓数量
    
    # 风控参数
    max_weight_per_stock: float = 0.10  # 单股最大权重（10%）
    min_weight_threshold: float = 0.01  # 最小权重阈值（1%）
    
    # 数据验证参数
    min_data_coverage: float = 0.8  # 最小数据覆盖率
    max_missing_consecutive_days: int = 5  # 最大连续缺失天数


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_dataframes(price_df: pd.DataFrame, *factor_dfs: pd.DataFrame) -> Dict[str, any]:
        """
        验证价格数据和因子数据的一致性
        
        Args:
            price_df: 价格数据
            factor_dfs: 因子数据列表
            
        Returns:
            Dict: 验证结果统计
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # 检查索引类型
            if not isinstance(price_df.index, pd.DatetimeIndex):
                validation_results['errors'].append("价格数据索引必须是DatetimeIndex")
                validation_results['is_valid'] = False
            
            # 检查数据对齐
            reference_index = price_df.index
            reference_columns = price_df.columns
            
            for i, factor_df in enumerate(factor_dfs):
                factor_name = f"因子{i+1}"
                
                # 索引对齐检查
                if not factor_df.index.equals(reference_index):
                    missing_dates = reference_index.difference(factor_df.index)
                    if len(missing_dates) > 0:
                        validation_results['warnings'].append(
                            f"{factor_name}缺失{len(missing_dates)}个交易日"
                        )
                
                # 列对齐检查
                if not factor_df.columns.equals(reference_columns):
                    missing_stocks = reference_columns.difference(factor_df.columns)
                    if len(missing_stocks) > 0:
                        validation_results['warnings'].append(
                            f"{factor_name}缺失{len(missing_stocks)}只股票"
                        )
            
            # 统计信息
            validation_results['stats'] = {
                'date_range': (price_df.index.min(), price_df.index.max()),
                'trading_days': len(price_df.index),
                'stock_count': len(price_df.columns),
                'data_coverage': (1 - price_df.isnull().sum().sum() / price_df.size) * 100
            }
            
            logger.info(f"数据验证完成: {validation_results['stats']}")
            
        except Exception as e:
            validation_results['errors'].append(f"验证过程异常: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    @staticmethod
    def align_dataframes(price_df: pd.DataFrame, *factor_dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        对齐价格数据和因子数据
        
        Args:
            price_df: 价格数据
            factor_dfs: 因子数据列表
            
        Returns:
            Tuple: 对齐后的数据框列表
        """
        # 使用vectorbt的对齐功能
        aligned_data = vbt.base.reshape_fns.broadcast(price_df, *factor_dfs, 
                                                     keep_pd=True, 
                                                     align_index=True, 
                                                     align_columns=True)
        
        logger.info(f"数据对齐完成，最终维度: {aligned_data[0].shape}")
        return aligned_data


class StrategySignalGenerator:
    """策略信号生成器"""
    
    @staticmethod
    def generate_long_signals(
        factor_df: pd.DataFrame,
        config: BacktestConfig
    ) -> pd.DataFrame:
        """
        生成做多信号

        Args:
            factor_df: 因子数据
            config: 回测配置

        Returns:
            pd.DataFrame: 持仓信号（True=持有，False=不持有）
        """
        # 1. 计算每日排名百分位（越大排名越靠前）
        ranks = factor_df.rank(axis=1, pct=True, method='average', na_option='keep')

        # 2. 确定做多信号（排名在前top_quantile）
        long_signals_raw = ranks >= (1 - config.top_quantile)

        # 3. 按调仓频率进行重采样
        # .resample().last() 获取每个周期最后一天的信号
        # .reindex().ffill() 将信号前填充到每个交易日
        rebalance_signals = long_signals_raw.resample(config.rebalancing_freq).last()
        final_signals = rebalance_signals.reindex(factor_df.index, method='ffill')

        # 4. 控制最大持仓数量
        if config.max_positions > 0:
            final_signals = StrategySignalGenerator._limit_positions(
                final_signals, ranks, config.max_positions
            )

        logger.info(f"信号生成完成，平均持仓数: {final_signals.sum(axis=1).mean():.1f}")
        return final_signals.fillna(False)
    
    @staticmethod
    def _limit_positions(signals: pd.DataFrame, ranks: pd.DataFrame, max_positions: int) -> pd.DataFrame:
        """
        限制最大持仓数量，优先选择排名最高的股票
        
        Args:
            signals: 原始信号
            ranks: 排名数据
            max_positions: 最大持仓数
            
        Returns:
            pd.DataFrame: 限制后的信号
        """
        limited_signals = pd.DataFrame(False, index=signals.index, columns=signals.columns)
        
        for date in signals.index:
            date_signals = signals.loc[date]
            date_ranks = ranks.loc[date]
            
            if date_signals.sum() > max_positions:
                # 选择排名最高的max_positions只股票
                valid_stocks = date_signals[date_signals].index
                top_stocks = date_ranks[valid_stocks].nlargest(max_positions).index
                limited_signals.loc[date, top_stocks] = True
            else:
                limited_signals.loc[date] = date_signals
                
        return limited_signals

        # 建议放在你的 StrategySignalGenerator 类中

    # 建议放在你的 StrategySignalGenerator 类中
    # 放在你的 StrategySignalGenerator 类中

    @staticmethod
    def generate_long_holding_signals(factor_df: pd.DataFrame,price_df, config: BacktestConfig) -> pd.DataFrame:
        """
        【V5】生成每日目标“持仓”布尔矩阵 (True/False)。
        """
        ranks = factor_df.rank(axis=1, pct=True, method='average', na_option='keep')
        snapshot_ranks = ranks.resample(config.rebalancing_freq).last().dropna(how='all')
        if snapshot_ranks.empty:
            raise ValueError("没有有效的调仓日快照，请检查调仓频率或数据日期范围")

        final_positions_snapshot = pd.DataFrame(False, index=snapshot_ranks.index, columns=snapshot_ranks.columns)
        for dt in snapshot_ranks.index:
            # 1. 动态确定当日的可投资池 (关键步骤!)
            daily_valid_ranks = snapshot_ranks.loc[dt].dropna()

            if daily_valid_ranks.empty:
                continue  # 如果当天所有股票都为NaN，则跳过

            # 2. 在可投资池的基础上，计算目标持仓数
            # 使用 np.ceil 确保至少选择一只股票（如果比例很小），并处理边界情况
            num_to_select = int(np.ceil(len(daily_valid_ranks) * config.top_quantile))

            # 兼容 max_positions 设置
            if config.max_positions:
                num_to_select = min(num_to_select, config.max_positions)

            # 3. 使用 nlargest 直接、精确地选出Top N的股票
            chosen_stocks = daily_valid_ranks.nlargest(num_to_select).index

            # 4. 更新持仓快照
            final_positions_snapshot.loc[dt, chosen_stocks] = True
        # --- 开始三步调试法 ---
        print("\n" + "=" * 20 + " 法医式调试开始 " + "=" * 20)


        daily_holding_signals = final_positions_snapshot.reindex(factor_df.index, method='ffill').fillna(False)

        # daily_holding_signals = daily_holding_signals.where(factor_df.notna(), other=False)
        #  price_df 就是对齐后的价格
        is_tradable = price_df.notna()  # 当天有价格数据，就认为是可交易的

        # 将理论持仓信号与可交易信号做“与”运算
        # 只有当“我想持有”且“它能交易”时，我才真正持有它
        daily_holding_signals = daily_holding_signals & is_tradable # daily_holding_signals.sum
        # 步骤 1: 验证调仓决策的变化
        # .diff() 会计算当前行与上一行的差异, .abs()取绝对值, .sum()计算总变化
        turnover_counts = daily_holding_signals.astype(int).diff().abs().sum(axis=1)

        print("\n[步骤 1] 每个调仓日的持仓变动股票数:")
        print(turnover_counts)

        # 统计有多少个调仓日是完全没有换手的
        zero_turnover_days = (turnover_counts == 0).sum()
        total_rebalancing_days = len(turnover_counts)
        print(f"\n分析: 在 {total_rebalancing_days} 个调仓日中，有 {zero_turnover_days} 天的持仓是完全没有变化的。")
        print(f"换手率为零的调仓日占比: {zero_turnover_days / total_rebalancing_days:.2%}")
        print("=" * 60)
        return daily_holding_signals

    @staticmethod
    def generate_rebalancing_signals(holding_signals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        【V5.1 - 稳健版】将“持仓”矩阵转换为精确的“买入”和“卖出”信号矩阵。
        - 增加了强制类型转换，彻底解决 'invert' ufunc TypeError 问题。
        """
        logger.info("  -> V5.1: 将持仓信号转换为买卖信号 (已增加类型保护)...")

        # 【核心修正】在使用 ~ 操作符之前，进行严格的类型和空值处理

        # 1. 确保当前持仓信号是布尔型
        current_holdings = holding_signals.astype(bool)

        # 2. 对前一天的持仓信号，先填充移位产生的NaN，再强制转为布尔型
        prev_holdings = holding_signals.vbt.fshift(1).fillna(False).astype(bool)

        # 3. 现在所有数据都是干净的布尔型，逻辑运算可以安全执行
        entries = current_holdings & ~prev_holdings
        exits = ~current_holdings & prev_holdings

        return entries, exits

class TradingCostCalculator:
    """交易成本计算器 - 重构版本"""
    
    @staticmethod
    def get_single_side_costs(config: BacktestConfig) -> Dict[str, float]:
        """
        获取单边交易成本
        
        Args:
            config: 回测配置
            
        Returns:
            Dict: 单边成本字典
        """
        # 基础单边成本 (佣金)
        base_commission = config.commission_rate
        
        # 单边滑点
        single_slippage = config.slippage_rate
        
        # 印花税只在卖出时收取，我们将其分摊到买卖两边
        # 或者包含在一个稍高的综合费率中
        adjusted_commission = base_commission + (config.stamp_duty / 2)  # 分摊印花税
        
        costs = {
            'commission': adjusted_commission,
            'slippage': single_slippage,
            'combined_fee': adjusted_commission  # vectorbt的fees参数
        }
        
        logger.info(f"单边交易成本: 佣金{adjusted_commission:.4f}, 滑点{single_slippage:.4f}")
        return costs


class QuantBacktester:
    """量化回测器主类"""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回测器
        
        Args:
            config: 回测配置，None时使用默认配置
        """
        self.config = config or BacktestConfig()
        self.validator = DataValidator()
        self.signal_generator = StrategySignalGenerator()
        self.cost_calculator = TradingCostCalculator()
        
        # 存储回测结果
        self.portfolios: Dict[str, any] = {}
        self.validation_results: Dict = {}
        
        logger.info("QuantBacktester初始化完成")
        logger.info(f"配置参数: {self.config}")
    
    def prepare_data(
        self, 
        price_df: pd.DataFrame,
        factor_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        准备和验证回测数据
        
        Args:
            price_df: 价格数据
            factor_dict: 因子数据字典 {"因子名": DataFrame}
            
        Returns:
            Tuple: (对齐后的价格数据, 对齐后的因子数据字典)
        """
        logger.info(f"开始准备数据，价格数据维度: {price_df.shape}")
        logger.info(f"因子数量: {len(factor_dict)}")
        
        # 验证数据
        factor_dfs = list(factor_dict.values())
        self.validation_results = self.validator.validate_dataframes(price_df, *factor_dfs)
        
        if not self.validation_results['is_valid']:
            raise ValueError(f"数据验证失败: {self.validation_results['errors']}")
        
        if self.validation_results['warnings']:
            for warning in self.validation_results['warnings']:
                logger.warning(warning)
        
        # 对齐数据
        aligned_data = self.validator.align_dataframes(price_df, *factor_dfs)
        aligned_price = aligned_data[0]
        aligned_factors = {
            name: aligned_data[i+1] 
            for i, name in enumerate(factor_dict.keys())
        }
        
        logger.info(f"数据准备完成，最终维度: {aligned_price.shape}")
        return aligned_price, aligned_factors

    def run_backtest(
            self,
            price_df: pd.DataFrame,
            factor_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, any]:
        """
        运行回测 (V5 - 使用 from_signals 的最终生产版)
        - 使用手动生成的精确调仓信号，兼容所有vectorbt版本
        """
        logger.info("=" * 60)
        logger.info("🚀 开始执行回测 (V5 from_signals 最终版)")
        logger.info("=" * 60)

        # 1. 数据准备
        all_dfs = [price_df] + list(factor_dict.values())
        aligned_dfs = vbt.base.reshape_fns.broadcast(*all_dfs, keep_pd=True, align_index=True, align_columns=True)

        aligned_price = aligned_dfs[0]
        aligned_factors = {name: df for name, df in zip(factor_dict.keys(), aligned_dfs[1:])}

        logger.info(f"数据对齐完成，最终维度: {aligned_price.shape}")

        # 2. 逐个因子回测
        for factor_name, factor_data in aligned_factors.items():
            logger.info(f"🚀 开始回测因子: {factor_name}")

            # 3. 信号生成流水线
            # 首先，生成每日的目标持仓状态 全是true false 表示当日rank情况的true flase
            holding_signals = self.signal_generator.generate_long_holding_signals(factor_data, aligned_price,self.config)

            # 1. 【新增】创建目标权重矩阵
            # 1.1 计算每日应持有的股票总数
            num_positions = holding_signals.sum(axis=1)

            # 1.2 计算每日的等权权重 (例如，持有10只股票，每只权重为 1/10 = 0.1)
            #     为避免除以零 (在没有持仓的日子)，使用 .replace(np.inf, 0)
            target_weights = (1 / num_positions).replace([np.inf, -np.inf], 0)

            # 1.3 将每日权重值广播到当天的持仓股票上，形成一个与 aholding_signals 形状相同的权重矩阵
            #     不持有(False)的股票，权重自然为 0
            weights_df = holding_signals.mul(target_weights, axis=0)

            # 然后，将持仓状态转换为实际的买入/卖出交易信号
            entry_signals, exit_signals = self.signal_generator.generate_rebalancing_signals(holding_signals)

            # 4. 【核心】使用 from_signals 执行回测
            # 它会根据 entry_signals 在每个交易日自动等权重买入
            portfolio = vbt.Portfolio.from_signals(
                close=aligned_price,
                entries=entry_signals,
                exits=exit_signals,
                # call_seq='auto',  # first sell then buy 实测!
                # size_type="percent",#实测！
                # size= pd.Series(0.75, index=aligned_price.index),  # 动态仓位大小
                init_cash=self.config.initial_cash,
                fees=self.config.commission_rate,
                slippage=self.config.slippage_rate,
                freq='D'  # Portfolio的运作频率应与价格频率一致
            )

            self.portfolios[factor_name] = portfolio
            print(portfolio.stats())
            # portfolio.exit_trades.records：小数5位后更精确
            # portfolio.entry_trades.records -
            ##
            #
            #
            #
            #

            #
            # status: 交易状态。（实测
            #
            # 你看到的 stats 应该是 status 的笔误。这是一个枚举值：
            #
            # 0 代表 TradeStatus.Open: 交易已开仓，但尚未平仓。
            #
            # 1 代表 TradeStatus.Closed: 交易已经平仓，是一个完整的来回。#
            portfolio.entry_trades.records - portfolio.exit_trades.records
            # 5. 结果分析 (与之前版本相同)
            stats = portfolio.stats()
            # 在 portfolio = vbt.Portfolio.from_signals(...) 之后
            logger.info(f"【诊断信息】因子: {factor_name}")
            logger.info(f"  -> 总共产生的买入信号点: {entry_signals.sum().sum()}")
            logger.info(f"  -> 总共产生的卖出信号点: {exit_signals.sum().sum()}")

            # 检查期末持仓
            open_records=portfolio.positions.open.records


            # 获取所有交易记录
            trades = portfolio.trades.records

            # 过滤卖出方向 (direction == -1 表示卖出)
            sell_volume = (trades["size"] * trades["exit_price"]).sum()

            stamp_duty_cost = sell_volume * self.config.stamp_duty
            final_return_adj = stats['Total Return [%]'] - (stamp_duty_cost / self.config.initial_cash) * 100

            logger.info(f"✅ {factor_name} 回测完成")
            logger.info(f"   总收益: {stats['Total Return [%]']:.2f}%")
            logger.info(f"   夏普比率: {stats['Sharpe Ratio']:.3f}")
            logger.info(f"   最大回撤: {stats['Max Drawdown [%]']:.2f}%")
            logger.info(f"   年化换手率: {stats['Turnover']:.2%}")
            logger.info(f"   (事后调整印花税后) 总收益: {final_return_adj:.2f}%")

        logger.info("🎉 所有因子回测完成")
        return self.portfolios

    def get_comparison_table(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        生成因子对比表
        
        Args:
            metrics: 要对比的指标列表
            
        Returns:
            pd.DataFrame: 对比结果表
        """
        if not self.portfolios:
            raise ValueError("请先运行回测")
        
        if metrics is None:
            metrics = [
                'Total Return [%]',
                'Sharpe Ratio',
                'Calmar Ratio',
                'Max Drawdown [%]',
                'Win Rate [%]',
                'Profit Factor'
            ]
        
        comparison_data = {}
        for factor_name, portfolio in self.portfolios.items():
            stats = portfolio.stats()
            comparison_data[factor_name] = stats[metrics]
        
        comparison_df = pd.DataFrame(comparison_data).T
        logger.info("因子对比表生成完成")
        return comparison_df
    
    def plot_cumulative_returns(self, 
                              figsize: Tuple[int, int] = (15, 8),
                              save_path: Optional[str] = None) -> None:
        """
        绘制累积收益率曲线
        
        Args:
            figsize: 图片大小
            save_path: 保存路径
        """
        if not self.portfolios:
            raise ValueError("请先运行回测")
        
        plt.figure(figsize=figsize)
        
        for factor_name, portfolio in self.portfolios.items():
            returns = portfolio.returns()
            cumulative_returns = (1 + returns).cumprod()
            plt.plot(cumulative_returns.index, cumulative_returns.values, 
                    label=factor_name, linewidth=2)
        
        plt.title('因子策略累积收益率对比', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('累积收益率', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")
        
        plt.show()
    
    def plot_drawdown_analysis(self,
                             figsize: Tuple[int, int] = (15, 10),
                             save_path: Optional[str] = None) -> None:
        """
        绘制回撤分析图
        
        Args:
            figsize: 图片大小
            save_path: 保存路径
        """
        if not self.portfolios:
            raise ValueError("请先运行回测")
        
        n_factors = len(self.portfolios)
        fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=True)
        if n_factors == 1:
            axes = [axes]
        
        for i, (factor_name, portfolio) in enumerate(self.portfolios.items()):
            drawdown = portfolio.drawdown()
            axes[i].fill_between(drawdown.index, drawdown.values, 0, 
                               color='red', alpha=0.3)
            axes[i].set_title(f'{factor_name} - 回撤分析', fontsize=14)
            axes[i].set_ylabel('回撤 (%)', fontsize=12)
            axes[i].grid(True, alpha=0.3)
        
        plt.xlabel('日期', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"回撤图已保存: {save_path}")
        
        plt.show()
    
    def generate_full_report(self, 
                           report_dir: str = "backtest_reports") -> str:
        """
        生成完整的回测报告
        
        Args:
            report_dir: 报告保存目录
            
        Returns:
            str: 报告文件路径
        """
        if not self.portfolios:
            raise ValueError("请先运行回测")
        
        # 创建报告目录
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成对比表
        comparison_df = self.get_comparison_table()
        comparison_file = report_path / f"factor_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, encoding='utf-8-sig')
        
        # 生成图表
        returns_chart = report_path / f"cumulative_returns_{timestamp}.png"
        self.plot_cumulative_returns(save_path=str(returns_chart))
        
        drawdown_chart = report_path / f"drawdown_analysis_{timestamp}.png" 
        self.plot_drawdown_analysis(save_path=str(drawdown_chart))
        
        # 生成详细统计报告
        report_file = report_path / f"detailed_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("量化策略回测详细报告\n") 
            f.write("=" * 80 + "\n\n")
            
            f.write("回测配置:\n")
            f.write(f"  调仓频率: {self.config.rebalancing_freq}\n")
            f.write(f"  做多分位: {self.config.top_quantile:.1%}\n")
            f.write(f"  初始资金: {self.config.initial_cash:,.0f}\n")
            f.write(f"  交易费率: {TradingCostCalculator.calculate_total_fees(self.config):.4f}\n\n")
            
            f.write("数据验证结果:\n")
            for key, value in self.validation_results['stats'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("因子对比结果:\n")
            f.write(comparison_df.to_string())
            f.write("\n\n")
        
        logger.info(f"完整报告已生成: {report_path}")
        return str(report_path)


# 便捷函数
def quick_factor_backtest(
    price_df: pd.DataFrame,
    factor_dict: Dict[str, pd.DataFrame],
    config: Optional[BacktestConfig] = None
) -> Tuple[Dict[str, any], pd.DataFrame]:
    """
    快速因子回测函数
    
    Args:
        price_df: 价格数据
        factor_dict: 因子数据字典
        config: 回测配置
        
    Returns:
        Tuple: (回测结果字典, 对比表)
    """
    backtester = QuantBacktester(config)
    portfolios = backtester.run_backtest(price_df, factor_dict)
    comparison_table = backtester.get_comparison_table()
    
    return portfolios, comparison_table


if __name__ == "__main__":
    # 示例用法
    logger.info("QuantBacktester 示例运行")
    
    # 这里需要你提供真实的数据
    # price_df = load_price_data()
    # factor_dict = {
    #     'volatility_40d': load_factor_data('volatility_40d'),
    #     'composite_factor': load_factor_data('composite_factor')
    # }
    
    # # 配置回测参数
    # config = BacktestConfig(
    #     top_quantile=0.2,
    #     rebalancing_freq='M',
    #     commission_rate=0.0003,
    #     slippage_rate=0.001
    # )
    
    # # 运行回测
    # backtester = QuantBacktester(config)
    # portfolios = backtester.run_backtest(price_df, factor_dict)
    
    # # 生成对比和报告
    # comparison_table = backtester.get_comparison_table()
    # print(comparison_table)
    
    # backtester.plot_cumulative_returns()
    # report_path = backtester.generate_full_report()
    
    logger.info("QuantBacktester 示例完成")