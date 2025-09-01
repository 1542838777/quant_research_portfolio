"""
Backtrader策略实现 - 替代vectorBT的完整解决方案

核心改进：
1. 解决vectorBT的权重分配问题 - 使用order_target_percent自动处理现金管理
2. 简化复杂的信号生成逻辑 - 事件驱动模型天然处理状态管理
3. 优雅处理停牌和交易失败 - Backtrader内置重试和错误处理
4. 完整保留原有策略逻辑 - 因子排名、调仓频率、持仓控制等
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from quant_lib.config.logger_config import setup_logger
from quant_lib.rebalance_utils import generate_rebalance_dates

logger = setup_logger(__name__)


class FactorBasedStrategy(bt.Strategy):
    """
    因子策略 - 完整迁移原有vectorBT逻辑
    
    核心特性：
    - 事前计算因子排名和持仓信号
    - 自动处理现金管理和仓位分配
    - 内置停牌和交易失败处理
    - 完整的状态追踪和调试信息
    """
    
    params = (
        # 策略参数
        ('factor_data', None),           # 因子数据DataFrame
        ('holding_signals', None),       # 预计算的持仓信号
        ('rebalance_dates', []),         # 调仓日期列表
        ('max_positions', 10),           # 最大持仓数量
        ('max_holding_days', 60),        # 最大持仓天数
        ('retry_buy_days', 3),           # 买入重试天数
        
        # 调试参数
        ('debug_mode', True),            # 调试模式
        ('log_detailed', False),         # 详细日志
    )
    
    def __init__(self):
        """策略初始化"""
        logger.info("初始化FactorBasedStrategy...")
        
        # 转换调仓日期为set，提高查询效率
        self.rebalance_dates_set = set(pd.to_datetime(self.p.rebalance_dates).date)
        
        # 状态追踪变量（替代vectorBT中的复杂状态管理）
        self.actual_positions = {}       # 实际持仓：{stock_name: data_obj}
        self.holding_days = {}           # 持仓天数：{stock_name: days}
        self.pending_buys = {}           # 待买清单：{stock_name: (retry_count, target_date)}
        self.failed_trades_log = []      # 交易失败记录
        
        # 性能统计
        self.rebalance_count = 0
        self.total_orders = 0
        self.failed_orders = 0
        
        logger.info(f"策略参数: 最大持仓{self.p.max_positions}, 调仓日期{len(self.rebalance_dates_set)}个")
    
    def next(self):
        """
        策略主循环 - 替代vectorBT中的复杂for循环
        
        Backtrader自动处理：
        - 时间推进和日期管理
        - 现金和持仓状态
        - 订单执行和失败处理
        """
        current_date = self.datetime.date(0)
        
        # 更新持仓天数
        self._update_holding_days()
        
        # # 处理待买清单（重试逻辑）
        # self._process_pending_buys()

        # # 强制卖出超期持仓
        # self._force_exit_old_positions()
        
        # 检查是否为调仓日
        if current_date in self.rebalance_dates_set:
            self._rebalance_portfolio(current_date)
    
    def _update_holding_days(self):
        """更新所有持仓的天数"""
        for stock_name in list(self.holding_days.keys()):
            data_obj = self.getdatabyname(stock_name)
            if self.getposition(data_obj).size > 0:
                self.holding_days[stock_name] += 1
            else:
                # 如果已经没有持仓，清除记录
                if stock_name in self.holding_days:
                    del self.holding_days[stock_name]
    
    def _process_pending_buys(self):
        """
        处理待买清单 - 替代vectorBT中的pending_buys_tracker逻辑
        """
        current_date = self.datetime.date(0)
        
        # 处理待买清单中的股票
        for stock_name in list(self.pending_buys.keys()):
            retry_count, target_date = self.pending_buys[stock_name]
            
            # 检查是否超过重试期限
            days_since_target = (current_date - target_date).days
            if days_since_target > self.p.retry_buy_days:
                if self.p.debug_mode:
                    logger.info(f"买入重试超期: {stock_name}, 放弃购买")
                del self.pending_buys[stock_name]
                continue
            
            # 尝试买入
            data_obj = self.getdatabyname(stock_name)
            if self._is_tradable(data_obj):
                success = self._buy_with_equal_weight(stock_name, data_obj)
                if success:
                    del self.pending_buys[stock_name]
                    if self.p.debug_mode:
                        logger.info(f"延迟买入成功: {stock_name}")
    
    def _force_exit_old_positions(self):
        """
        强制卖出超期持仓 - 替代vectorBT中的force_exit_intent逻辑
        """
        if self.p.max_holding_days is None:
            return
            
        for stock_name, days in self.holding_days.items():
            if days >= self.p.max_holding_days:
                data_obj = self.getdatabyname(stock_name)
                position = self.getposition(data_obj)
                
                if position.size > 0 and self._is_tradable(data_obj):
                    self.order_target_percent(data=data_obj, target=0.0)
                    self.total_orders += 1
                    
                    if self.p.debug_mode:
                        logger.info(f"强制卖出超期持仓: {stock_name}, 持有{days}天")
    
    def _rebalance_portfolio(self, current_date):
        """
        调仓逻辑 - 替代vectorBT中的复杂权重计算和分配
        
        Args:
            current_date: 当前日期
        """
        if self.p.debug_mode:
            logger.info(f"--- 调仓日: {current_date} ---")
        
        self.rebalance_count += 1
        
        # 获取今日的目标持仓信号
        try:
            target_holdings = self.p.holding_signals.loc[current_date]
            target_stocks = target_holdings[target_holdings].index.tolist()
        except KeyError:
            if self.p.debug_mode:
                logger.warning(f"未找到日期{current_date}的持仓信号")
            return
        
        if self.p.debug_mode:
            logger.info(f"目标持仓股票: {len(target_stocks)}只")
        
        # 第一阶段：卖出不再需要的股票
        self._sell_unwanted_positions(target_stocks)
        
        # 第二阶段：买入新股票或调整权重
        self._buy_target_positions(target_stocks)
    
    def _sell_unwanted_positions(self, target_stocks: List[str]):
        """
        卖出不再需要的持仓 - 替代vectorBT中的exits逻辑
        
        Args:
            target_stocks: 目标持仓股票列表
        """
        positions_to_close = []
        
        # 检查当前所有持仓
        for data_obj in self.datas:
            stock_name = data_obj._name
            position = self.getposition(data_obj)
            
            if position.size > 0 and stock_name not in target_stocks:
                positions_to_close.append((stock_name, data_obj))
        
        # 执行卖出
        for stock_name, data_obj in positions_to_close:
            if self._is_tradable(data_obj):
                self.order_target_percent(data=data_obj, target=0.0)
                self.total_orders += 1
                
                if self.p.debug_mode:
                    logger.info(f"卖出: {stock_name}")
            else:
                if self.p.debug_mode:
                    logger.warning(f"卖出失败(停牌): {stock_name}")
                self.failed_orders += 1
    
    def _buy_target_positions(self, target_stocks: List[str]):
        """
        买入目标股票 - 使用等权重分配，自动解决vectorBT的现金管理问题
        
        Args:
            target_stocks: 目标股票列表
        """
        if not target_stocks:
            return
        
        # 计算目标权重（等权重）
        target_weight = 1.0 / len(target_stocks)
        
        successful_buys = 0
        failed_buys = 0
        
        for stock_name in target_stocks:
            data_obj = self.getdatabyname(stock_name)
            
            if self._is_tradable(data_obj):
                success = self._buy_with_equal_weight(stock_name, data_obj, target_weight)
                if success:
                    successful_buys += 1
                else:
                    failed_buys += 1
            else:
                # 停牌股票加入待买清单
                self.pending_buys[stock_name] = (0, self.datetime.date(0))
                failed_buys += 1
                
                if self.p.debug_mode:
                    logger.warning(f"买入失败(停牌): {stock_name}, 加入待买清单")
        
        if self.p.debug_mode:
            logger.info(f"买入执行: 成功{successful_buys}只, 失败{failed_buys}只")
    
    def _buy_with_equal_weight(self, stock_name: str, data_obj, target_weight: float = None) -> bool:
        """
        等权重买入股票 - 核心解决vectorBT现金分配问题的函数
        
        Args:
            stock_name: 股票名称
            data_obj: Backtrader数据对象
            target_weight: 目标权重，如果None则根据待买清单计算
            
        Returns:
            bool: 是否买入成功
        """
        try:
            if target_weight is None:
                # 从待买清单买入时，根据当前现金情况重新计算权重
                current_positions = len([d for d in self.datas if self.getposition(d).size > 0])
                pending_buys_count = len(self.pending_buys)
                target_weight = self.broker.get_cash() / (self.broker.get_value() * (current_positions + pending_buys_count + 1))
            
            # 使用Backtrader的自动权重分配
            order = self.order_target_percent(data=data_obj, target=target_weight)
            
            if order:
                self.total_orders += 1
                # 初始化持仓天数
                self.holding_days[stock_name] = 0
                return True
            else:
                self.failed_orders += 1
                return False
                
        except Exception as e:
            if self.p.debug_mode:
                logger.error(f"买入{stock_name}时发生错误: {e}")
            self.failed_orders += 1
            return False
    
    def _is_tradable(self, data_obj) -> bool:
        """
        检查股票是否可交易 - 替代vectorBT中的is_tradable_today逻辑
        
        Args:
            data_obj: Backtrader数据对象
            
        Returns:
            bool: 是否可交易
        """
        # 检查是否有价格数据（非停牌）
        return not np.isnan(data_obj.close[0]) and data_obj.close[0] > 0
    
    def notify_order(self, order):
        """
        订单状态通知 - 处理交易失败和重试逻辑
        """
        if order.status in [order.Completed]:
            action = "买入" if order.isbuy() else "卖出"
            if self.p.log_detailed:
                logger.info(f"{action}成功: {order.data._name}, "
                           f"数量: {order.executed.size:.0f}, "
                           f"价格: {order.executed.price:.2f}")
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            action = "买入" if order.isbuy() else "卖出"
            stock_name = order.data._name
            
            # 记录失败
            self.failed_trades_log.append({
                'date': self.datetime.date(0),
                'stock': stock_name,
                'action': action,
                'status': order.getstatusname(),
                'size': order.created.size
            })
            
            # 如果是买入失败，加入待买清单
            if order.isbuy() and stock_name not in self.pending_buys:
                self.pending_buys[stock_name] = (1, self.datetime.date(0))
                
            if self.p.debug_mode:
                logger.warning(f"{action}失败: {stock_name}, 原因: {order.getstatusname()}")
    
    def notify_trade(self, trade):
        """交易完成通知"""
        if trade.isclosed:
            pnl_pct = trade.pnlcomm / trade.value * 100
            holding_period = (trade.dtclose - trade.dtopen).days
            
            if self.p.log_detailed:
                logger.info(f"交易完成: {trade.data._name}, "
                           f"收益: {pnl_pct:.2f}%, "
                           f"持仓: {holding_period}天")
    
    def stop(self):
        """策略结束处理"""
        logger.info("=" * 60)
        logger.info("策略执行完成 - 统计汇总")
        logger.info("=" * 60)
        
        logger.info(f"调仓次数: {self.rebalance_count}")
        logger.info(f"总订单数: {self.total_orders}")
        logger.info(f"失败订单: {self.failed_orders}")
        logger.info(f"订单成功率: {(1 - self.failed_orders/max(self.total_orders, 1))*100:.1f}%")
        
        if self.failed_trades_log:
            logger.info(f"交易失败记录: {len(self.failed_trades_log)}次")
            # 显示前几次失败
            for i, record in enumerate(self.failed_trades_log[:5]):
                logger.info(f"  {i+1}. {record['date']} {record['stock']} {record['action']} - {record['status']}")
        
        final_value = self.broker.getvalue()
        total_return = (final_value / self.broker.startingcash - 1) * 100
        logger.info(f"最终资产: {final_value:,.2f}")
        logger.info(f"总收益率: {total_return:.2f}%")


class BacktraderFactorEngine:
    """
    Backtrader因子引擎 - 完整替代QuantBacktester
    
    核心优势：
    1. 解决vectorBT权重分配问题
    2. 简化复杂的状态管理
    3. 更好的调试和监控能力
    4. 真实的事件驱动交易模拟
    """
    
    def __init__(self, config_dict: Dict = None):
        """
        初始化引擎
        
        Args:
            config_dict: 配置参数字典，兼容原有BacktestConfig
        """
        self.config = config_dict or self._default_config()
        self.cerebro = None
        self.results = {}
        
        logger.info("BacktraderFactorEngine初始化完成")
        logger.info(f"配置: {self.config}")
    
    def _default_config(self) -> Dict:
        """默认配置 - 兼容原有BacktestConfig"""
        return {
            'top_quantile': 0.2,
            'rebalancing_freq': 'M',
            'commission_rate': 0.0003,
            'slippage_rate': 0.001,
            'stamp_duty': 0.001,
            'initial_cash': 1000000.0,
            'max_positions': 10,
            'max_holding_days': 60,
            'retry_buy_days': 3
        }
    
    def prepare_data_feeds(self, price_df: pd.DataFrame) -> List:
        """
        准备Backtrader数据源
        
        Args:
            price_df: 价格数据，列为股票代码，索引为日期
            
        Returns:
            List: Backtrader数据源列表
        """
        data_feeds = []
        
        for stock_code in price_df.columns:
            # 为每只股票创建标准OHLCV数据
            stock_data = pd.DataFrame(index=price_df.index)
            stock_data['close'] = price_df[stock_code]
            
            # 创建简化的OHLCV（使用收盘价模拟）
            stock_data['open'] = stock_data['close']
            stock_data['high'] = stock_data['close'] * 1.01  # 模拟1%高点
            stock_data['low'] = stock_data['close'] * 0.99   # 模拟1%低点
            stock_data['volume'] = 1000000  # 固定成交量
            
            # 移除NaN行（停牌日）
            stock_data = stock_data.dropna()
            
            if len(stock_data) > 0:
                # 创建Backtrader数据源
                data_feed = bt.feeds.PandasData(
                    dataname=stock_data,
                    name=stock_code,
                    fromdate=stock_data.index[0],
                    todate=stock_data.index[-1]
                )
                data_feeds.append(data_feed)
        
        logger.info(f"数据源准备完成: {len(data_feeds)}只股票")
        return data_feeds
    
    def generate_holding_signals(self, factor_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成持仓信号 - 完整替代原有的generate_long_holding_signals
        
        Args:
            factor_df: 因子数据
            price_df: 价格数据
            
        Returns:
            pd.DataFrame: 每日持仓信号（True/False矩阵）
        """
        logger.info("生成持仓信号...")
        
        # 计算每日排名百分位
        ranks = factor_df.rank(axis=1, pct=True, method='average', na_option='keep')
        
        # 生成调仓日期
        rebalance_dates = generate_rebalance_dates(factor_df.index, self.config['rebalancing_freq'])
        
        # 初始化持仓信号矩阵
        holding_signals = pd.DataFrame(False, index=factor_df.index, columns=factor_df.columns)
        
        # 当前持仓组合（在调仓间隔期间保持不变）
        current_positions = None
        
        for date in factor_df.index:
            is_rebalance_day = date in rebalance_dates
            
            if is_rebalance_day:
                # 调仓日：重新选择股票
                daily_valid_ranks = ranks.loc[date].dropna()
                
                if len(daily_valid_ranks) > 0:
                    # 计算目标持仓数
                    num_to_select = int(np.ceil(len(daily_valid_ranks) * self.config['top_quantile']))
                    if self.config['max_positions']:
                        num_to_select = min(num_to_select, self.config['max_positions'])
                    
                    # 选择排名最高的股票
                    chosen_stocks = daily_valid_ranks.nlargest(num_to_select).index
                    current_positions = chosen_stocks
            
            # 保持当前持仓（前向填充逻辑）
            if current_positions is not None:
                holding_signals.loc[date, current_positions] = True
        
        # 统计信息
        avg_positions = holding_signals.sum(axis=1).mean()
        zero_position_days = (holding_signals.sum(axis=1) == 0).sum()
        
        logger.info(f"持仓信号生成完成:")
        logger.info(f"  平均每日持仓: {avg_positions:.1f}只")
        logger.info(f"  零持仓天数: {zero_position_days}/{len(holding_signals)}")
        
        return holding_signals
    
    def run_backtest(self, price_df: pd.DataFrame, factor_dict: Dict[str, pd.DataFrame]) -> Dict[str, bt.Strategy]:
        """
        运行因子回测 - 完整替代原有的run_backtest函数
        
        Args:
            price_df: 价格数据
            factor_dict: 因子数据字典
            
        Returns:
            Dict: 回测结果字典 {因子名: 策略结果}
        """
        logger.info(f"开始Backtrader回测，因子数量: {len(factor_dict)}")
        
        results = {}
        
        for factor_name, factor_data in factor_dict.items():
            logger.info(f"回测因子: {factor_name}")
            
            # 生成持仓信号
            holding_signals = self.generate_holding_signals(factor_data, price_df)
            
            # 生成调仓日期
            rebalance_dates = generate_rebalance_dates(
                factor_data.index, 
                self.config['rebalancing_freq']
            )
            
            # 创建新的Cerebro实例
            cerebro = bt.Cerebro()
            
            # 添加数据源
            data_feeds = self.prepare_data_feeds(price_df)
            for data_feed in data_feeds:
                cerebro.adddata(data_feed)
            
            # 添加策略
            cerebro.addstrategy(
                FactorBasedStrategy,
                factor_data=factor_data,
                holding_signals=holding_signals,
                rebalance_dates=rebalance_dates,
                max_positions=self.config['max_positions'],
                max_holding_days=self.config['max_holding_days'],
                retry_buy_days=self.config['retry_buy_days'],
                debug_mode=True
            )
            
            # 设置初始资金和交易费用
            cerebro.broker.setcash(self.config['initial_cash'])
            
            # 计算综合费率 - 与原有逻辑保持一致
            comprehensive_fee = (
                self.config['commission_rate'] +
                self.config['slippage_rate'] +
                self.config['stamp_duty'] / 2
            )
            cerebro.broker.setcommission(commission=comprehensive_fee)
            
            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            # 运行回测
            try:
                strategy_results = cerebro.run()
                results[factor_name] = {
                    'strategy': strategy_results[0],
                    'final_value': cerebro.broker.getvalue(),
                    'analyzers': strategy_results[0].analyzers
                }
                
                logger.info(f"{factor_name} 回测完成，最终价值: {cerebro.broker.getvalue():,.2f}")
                
            except Exception as e:
                logger.error(f"{factor_name} 回测失败: {e}")
                results[factor_name] = None
        
        self.results = results
        logger.info("所有因子回测完成")
        return results
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        生成因子对比表 - 兼容原有接口
        
        Returns:
            pd.DataFrame: 对比结果表
        """
        if not self.results:
            raise ValueError("请先运行回测")
        
        comparison_data = {}
        
        for factor_name, result in self.results.items():
            if result is None:
                continue
                
            strategy = result['strategy']
            analyzers = result['analyzers']
            
            # 计算关键指标
            try:
                total_return = (result['final_value'] / self.config['initial_cash'] - 1) * 100
                sharpe_ratio = analyzers.sharpe.get_analysis().get('sharperatio', 0)
                max_drawdown = analyzers.drawdown.get_analysis()['max']['drawdown']
                annual_return = analyzers.returns.get_analysis().get('rnorm100', 0)
                
                # 交易统计
                trade_analysis = analyzers.trades.get_analysis()
                total_trades = trade_analysis.get('total', {}).get('total', 0)
                won_trades = trade_analysis.get('won', {}).get('total', 0)
                win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
                
                comparison_data[factor_name] = {
                    'Total Return [%]': total_return,
                    'Annual Return [%]': annual_return,
                    'Sharpe Ratio': sharpe_ratio,
                    'Max Drawdown [%]': abs(max_drawdown),
                    'Win Rate [%]': win_rate,
                    'Total Trades': total_trades
                }
                
            except Exception as e:
                logger.error(f"计算{factor_name}指标时出错: {e}")
                comparison_data[factor_name] = {
                    'Total Return [%]': 0,
                    'Annual Return [%]': 0,
                    'Sharpe Ratio': 0,
                    'Max Drawdown [%]': 0,
                    'Win Rate [%]': 0,
                    'Total Trades': 0
                }
        
        comparison_df = pd.DataFrame(comparison_data).T
        logger.info("因子对比表生成完成")
        return comparison_df
    
    def plot_results(self, factor_name: str = None):
        """
        绘制回测结果
        
        Args:
            factor_name: 指定因子名称，None时绘制所有因子
        """
        if not self.results:
            raise ValueError("请先运行回测")
        
        if factor_name:
            factors_to_plot = [factor_name] if factor_name in self.results else []
        else:
            factors_to_plot = list(self.results.keys())
        
        for fname in factors_to_plot:
            result = self.results[fname]
            if result is None:
                continue
                
            logger.info(f"绘制{fname}的结果图表...")
            # 这里可以添加自定义绘图逻辑
            # 或者使用Backtrader内置的plot功能


def migrate_from_vectorbt_config(vectorbt_config) -> Dict:
    """
    从vectorBT配置转换为Backtrader配置
    
    Args:
        vectorbt_config: 原有的BacktestConfig对象
        
    Returns:
        Dict: Backtrader配置字典
    """
    return {
        'top_quantile': getattr(vectorbt_config, 'top_quantile', 0.2),
        'rebalancing_freq': getattr(vectorbt_config, 'rebalancing_freq', 'M'),
        'commission_rate': getattr(vectorbt_config, 'commission_rate', 0.0003),
        'slippage_rate': getattr(vectorbt_config, 'slippage_rate', 0.001),
        'stamp_duty': getattr(vectorbt_config, 'stamp_duty', 0.001),
        'initial_cash': getattr(vectorbt_config, 'initial_cash', 1000000.0),
        'max_positions': getattr(vectorbt_config, 'max_positions', 10),
        'max_holding_days': getattr(vectorbt_config, 'max_holding_days', 60),
        'retry_buy_days': 3
    }


# 便捷迁移函数
def quick_migration_test(price_df: pd.DataFrame, factor_dict: Dict[str, pd.DataFrame], 
                        original_config) -> Tuple[Dict, pd.DataFrame]:
    """
    快速迁移测试 - 一键从vectorBT切换到Backtrader
    
    Args:
        price_df: 价格数据
        factor_dict: 因子数据字典
        original_config: 原有的BacktestConfig对象
        
    Returns:
        Tuple: (回测结果, 对比表)
    """
    logger.info("开始快速迁移测试...")
    
    # 转换配置
    bt_config = migrate_from_vectorbt_config(original_config)
    
    # 创建引擎并运行
    engine = BacktraderFactorEngine(bt_config)
    results = engine.run_backtest(price_df, factor_dict)
    comparison_table = engine.get_comparison_table()
    
    logger.info("迁移测试完成！")
    logger.info("结果对比（Backtrader vs vectorBT）:")
    print(comparison_table)
    
    return results, comparison_table


if __name__ == "__main__":
    logger.info("Backtrader策略测试")
    
    # 这里可以添加测试代码
    # 使用方式：
    # results, comparison = quick_migration_test(price_df, factor_dict, original_config)