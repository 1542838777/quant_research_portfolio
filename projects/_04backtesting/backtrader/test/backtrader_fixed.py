"""
修复后的Backtrader策略 - 解决调仓日期判断问题

核心修复：
1. 修复调仓日期的时区和格式匹配问题
2. 简化持仓信号生成逻辑
3. 确保策略能正常执行交易
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


class FixedFactorStrategy(bt.Strategy):
    """修复后的因子策略"""
    
    params = (
        ('factor_data', None),
        ('top_quantile', 0.2),
        ('max_positions', 10),
        ('rebalance_freq', 'M'),
        ('max_holding_days', 60),
        ('debug_mode', True),
    )
    
    def __init__(self):
        logger.info("初始化FixedFactorStrategy...")
        
        # 生成调仓日期 - 直接使用索引，避免时区问题
        self.factor_dates = list(self.p.factor_data.index)
        self.rebalance_dates = self._generate_rebalance_dates()
        
        # 状态变量
        self.current_positions = {}
        self.holding_days = {}
        self.rebalance_count = 0
        
        logger.info(f"策略初始化完成: 调仓日期{len(self.rebalance_dates)}个")
        logger.info(f"前几个调仓日期: {self.rebalance_dates[:3]}")
    
    def _generate_rebalance_dates(self):
        """生成调仓日期列表"""
        if self.p.rebalance_freq == 'M':
            # 月末调仓
            rebalance_dates = []
            current_month = None
            
            for date in self.factor_dates:
                month_key = (date.year, date.month)
                if current_month != month_key:
                    if current_month is not None:  # 不包括第一个月的第一天
                        rebalance_dates.append(date)
                    current_month = month_key
            
            return rebalance_dates
        else:
            # 其他频率可以后续添加
            return []
    
    def next(self):
        """策略主循环"""
        current_date = self.datetime.date(0)
        
        # 检查是否为调仓日
        is_rebalance_day = any(
            abs((current_date - rd.date()).days) <= 1 
            for rd in self.rebalance_dates
        )
        
        if is_rebalance_day:
            self._rebalance(current_date)
        
        # 更新持仓天数
        self._update_holding_days()
        
        # 强制卖出超期持仓
        self._force_exit_old_positions()
    
    def _rebalance(self, current_date):
        """执行调仓"""
        logger.info(f"--- 调仓日: {current_date} ---")
        self.rebalance_count += 1
        
        # 找到最近的因子数据日期
        factor_date = self._find_nearest_factor_date(current_date)
        if factor_date is None:
            logger.warning(f"未找到{current_date}的因子数据")
            return
        
        # 获取因子排名
        factor_values = self.p.factor_data.loc[factor_date].dropna()
        if factor_values.empty:
            return
        
        # 选择前N%的股票
        num_to_select = min(
            int(len(factor_values) * self.p.top_quantile),
            self.p.max_positions
        )
        
        target_stocks = factor_values.nlargest(num_to_select).index.tolist()
        
        logger.info(f"目标股票: {len(target_stocks)}只")
        
        # 卖出不需要的持仓
        self._sell_unwanted_positions(target_stocks)
        
        # 买入新股票
        self._buy_target_positions(target_stocks)
    
    def _find_nearest_factor_date(self, current_date):
        """找到最近的因子数据日期"""
        current_datetime = pd.Timestamp(current_date)
        
        for factor_date in reversed(self.factor_dates):
            if factor_date <= current_datetime:
                return factor_date
        
        return None
    
    def _sell_unwanted_positions(self, target_stocks):
        """卖出不需要的股票"""
        for data_obj in self.datas:
            stock_name = data_obj._name
            position = self.getposition(data_obj)
            
            if position.size > 0 and stock_name not in target_stocks:
                if self._is_tradable(data_obj):
                    self.order_target_percent(data=data_obj, target=0.0)
                    if self.p.debug_mode:
                        logger.info(f"卖出: {stock_name}")
    
    def _buy_target_positions(self, target_stocks):
        """买入目标股票"""
        if not target_stocks:
            return
        
        target_weight = 0.95 / len(target_stocks)  # 留5%现金
        
        for stock_name in target_stocks:
            try:
                data_obj = self.getdatabyname(stock_name)
                current_position = self.getposition(data_obj).size
                
                if current_position == 0 and self._is_tradable(data_obj):
                    self.order_target_percent(data=data_obj, target=target_weight)
                    self.holding_days[stock_name] = 0
                    if self.p.debug_mode:
                        logger.info(f"买入: {stock_name}, 权重: {target_weight:.2%}")
                        
            except Exception as e:
                if self.p.debug_mode:
                    logger.warning(f"买入{stock_name}失败: {e}")
    
    def _update_holding_days(self):
        """更新持仓天数"""
        for stock_name in list(self.holding_days.keys()):
            try:
                data_obj = self.getdatabyname(stock_name)
                if self.getposition(data_obj).size > 0:
                    self.holding_days[stock_name] += 1
                else:
                    del self.holding_days[stock_name]
            except:
                continue
    
    def _force_exit_old_positions(self):
        """强制卖出超期持仓"""
        if self.p.max_holding_days is None:
            return
        
        for stock_name, days in list(self.holding_days.items()):
            if days >= self.p.max_holding_days:
                try:
                    data_obj = self.getdatabyname(stock_name)
                    if self.getposition(data_obj).size > 0 and self._is_tradable(data_obj):
                        self.order_target_percent(data=data_obj, target=0.0)
                        if self.p.debug_mode:
                            logger.info(f"强制卖出: {stock_name}, 持有{days}天")
                except:
                    continue
    
    def _is_tradable(self, data_obj):
        """检查是否可交易"""
        try:
            return not np.isnan(data_obj.close[0]) and data_obj.close[0] > 0
        except:
            return False
    
    def stop(self):
        """策略结束"""
        final_value = self.broker.getvalue()
        total_return = (final_value / self.broker.startingcash - 1) * 100
        
        logger.info("=" * 60)
        logger.info("策略执行完成")
        logger.info(f"调仓次数: {self.rebalance_count}")
        logger.info(f"最终价值: {final_value:,.2f}")
        logger.info(f"总收益率: {total_return:.2f}%")


def quick_backtrader_test(price_df: pd.DataFrame, factor_dict: Dict[str, pd.DataFrame], 
                         config) -> Dict:
    """快速Backtrader测试"""
    logger.info("开始快速Backtrader测试...")
    
    results = {}
    
    for factor_name, factor_data in factor_dict.items():
        logger.info(f"测试因子: {factor_name}")
        
        # 数据对齐
        common_dates = price_df.index.intersection(factor_data.index)
        common_stocks = price_df.columns.intersection(factor_data.columns)
        
        aligned_price = price_df.loc[common_dates, common_stocks]
        aligned_factor = factor_data.loc[common_dates, common_stocks]
        
        logger.info(f"数据对齐: {aligned_price.shape}")
        
        # 创建Cerebro
        cerebro = bt.Cerebro()
        
        # 添加数据（限制数量以提高测试速度）
        max_stocks = 50
        selected_stocks = common_stocks[:max_stocks]
        
        for stock in selected_stocks:
            stock_data = pd.DataFrame(index=aligned_price.index)
            stock_data['close'] = aligned_price[stock].fillna(method='ffill')
            stock_data['open'] = stock_data['close']
            stock_data['high'] = stock_data['close'] * 1.01
            stock_data['low'] = stock_data['close'] * 0.99
            stock_data['volume'] = 1000000
            
            # 移除NaN行
            stock_data = stock_data.dropna()
            
            if len(stock_data) > 100:  # 至少要有100天数据
                data_feed = bt.feeds.PandasData(
                    dataname=stock_data,
                    name=stock
                )
                cerebro.adddata(data_feed)
        
        logger.info(f"添加了{len(cerebro.datas)}只股票数据")
        
        # 添加策略
        cerebro.addstrategy(
            FixedFactorStrategy,
            factor_data=aligned_factor[selected_stocks],
            top_quantile=getattr(config, 'top_quantile', 0.3),
            max_positions=getattr(config, 'max_positions', 10),
            rebalance_freq=getattr(config, 'rebalancing_freq', 'M'),
            max_holding_days=getattr(config, 'max_holding_days', 60),
            debug_mode=True
        )
        
        # 设置交易参数
        cerebro.broker.setcash(getattr(config, 'initial_cash', 300000))
        cerebro.broker.setcommission(commission=0.002)  # 简化费率
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        # 运行回测
        try:
            logger.info(f"开始运行{factor_name}回测...")
            start_time = datetime.now()
            
            strategy_results = cerebro.run()
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            strategy = strategy_results[0]
            final_value = cerebro.broker.getvalue()
            
            results[factor_name] = {
                'strategy': strategy,
                'final_value': final_value,
                'execution_time': execution_time,
                'analyzers': strategy.analyzers
            }
            
            logger.info(f"{factor_name} 回测完成: {final_value:,.2f}, 耗时{execution_time:.1f}秒")
            
        except Exception as e:
            logger.error(f"{factor_name} 回测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[factor_name] = None
    
    return results


def generate_comparison_table(results: Dict) -> pd.DataFrame:
    """生成对比表"""
    comparison_data = {}
    
    for factor_name, result in results.items():
        if result is None:
            continue
            
        try:
            final_value = result['final_value']
            initial_cash = 300000  # 默认初始资金
            total_return = (final_value / initial_cash - 1) * 100
            
            analyzers = result['analyzers']
            sharpe_analysis = analyzers.sharpe.get_analysis()
            drawdown_analysis = analyzers.drawdown.get_analysis()
            
            sharpe_ratio = sharpe_analysis.get('sharperatio', 0) or 0
            max_drawdown = abs(drawdown_analysis.get('max', {}).get('drawdown', 0))
            
            comparison_data[factor_name] = {
                'Total Return [%]': total_return,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown [%]': max_drawdown,
                'Final Value': final_value
            }
            
        except Exception as e:
            logger.error(f"计算{factor_name}指标时出错: {e}")
            comparison_data[factor_name] = {
                'Total Return [%]': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown [%]': 0,
                'Final Value': initial_cash
            }
    
    if comparison_data:
        return pd.DataFrame(comparison_data).T
    else:
        return pd.DataFrame()


# 修复后的一键测试函数
def fixed_backtrader_test(price_df, factor_dict, config):
    """修复后的Backtrader测试"""
    logger.info("🔧 使用修复后的Backtrader版本")
    
    # 运行修复后的回测
    results = quick_backtrader_test(price_df, factor_dict, config)
    
    # 生成结果对比表
    comparison_table = generate_comparison_table(results)
    
    return results, comparison_table


if __name__ == "__main__":
    logger.info("修复版Backtrader测试")
    
    # 这里可以添加测试代码
    pass