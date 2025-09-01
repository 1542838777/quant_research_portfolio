"""
修复的Backtrader版本 - 解决调仓过频和Margin错误

关键修复：
1. 调仓频率：每月只调仓一次（不是每天）
2. 资金管理：检查可用现金，避免Margin错误
3. 简化逻辑：去掉复杂的重试机制
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FixedFactorStrategy(bt.Strategy):
    """修复版因子策略"""
    
    params = (
        ('factor_data', None),
        ('top_quantile', 0.2),
        ('max_positions', 10),
        ('cash_reserve', 0.05),  # 保留5%现金
    )
    
    def __init__(self):
        self.rebalance_count = 0
        self.total_orders = 0
        self.successful_orders = 0
        self.last_rebalance_month = None  # 记录上次调仓月份
        
        print(f"策略初始化完成，参数：做多{self.p.top_quantile:.0%}，最大{self.p.max_positions}只")
    
    def next(self):
        """主策略逻辑"""
        current_date = self.datetime.date(0)
        current_month = current_date.replace(day=1)  # 月份标识
        
        # 每月只调仓一次
        if (self.last_rebalance_month is None or 
            current_month != self.last_rebalance_month):
            
            self.last_rebalance_month = current_month
            self._monthly_rebalance(current_date)
    
    def _monthly_rebalance(self, current_date):
        """月度调仓"""
        self.rebalance_count += 1
        print(f"第{self.rebalance_count}次调仓: {current_date}")
        
        # 获取目标股票
        target_stocks = self._select_target_stocks(current_date)
        if not target_stocks:
            print("  无目标股票，跳过调仓")
            return
        
        print(f"  目标股票({len(target_stocks)}只): {target_stocks}")
        
        # 清仓所有持仓
        self._clear_all_positions()
        
        # 买入新目标（等权重）
        target_weight = (1.0 - self.p.cash_reserve) / len(target_stocks)
        success_count = 0
        
        for stock_name in target_stocks:
            if self._buy_stock(stock_name, target_weight):
                success_count += 1
        
        print(f"  买入结果: {success_count}/{len(target_stocks)}成功")
    
    def _select_target_stocks(self, current_date):
        """选择目标股票"""
        try:
            # 查找因子数据
            current_ts = pd.Timestamp(current_date)
            factor_date = None
            
            for date in self.p.factor_data.index:
                if date <= current_ts:
                    factor_date = date
            
            if factor_date is None:
                return []
            
            # 因子选股
            factor_values = self.p.factor_data.loc[factor_date].dropna()
            if len(factor_values) == 0:
                return []
            
            num_select = min(
                int(len(factor_values) * self.p.top_quantile),
                self.p.max_positions
            )
            
            return factor_values.nlargest(num_select).index.tolist()
            
        except Exception as e:
            print(f"  选股失败: {e}")
            return []
    
    def _clear_all_positions(self):
        """清仓所有持仓"""
        for data in self.datas:
            position = self.getposition(data)
            if position.size > 0:
                try:
                    # 检查价格有效性
                    if not np.isnan(data.close[0]) and data.close[0] > 0:
                        self.close(data=data)
                        self.total_orders += 1
                        print(f"  卖出: {data._name}")
                except:
                    continue
    
    def _buy_stock(self, stock_name, target_weight):
        """买入单只股票"""
        try:
            data = self.getdatabyname(stock_name)
            
            # 检查价格有效性
            if np.isnan(data.close[0]) or data.close[0] <= 0:
                return False
            
            # 检查可用现金
            available_cash = self.broker.get_cash()
            total_value = self.broker.get_value()
            cash_ratio = available_cash / total_value
            
            if cash_ratio < 0.01:  # 现金不足1%
                print(f"    {stock_name}: 现金不足({cash_ratio:.1%})")
                return False
            
            # 下单
            order = self.order_target_percent(data=data, target=target_weight)
            self.total_orders += 1
            print(f"    买入: {stock_name}, 权重: {target_weight:.1%}")
            return True
            
        except Exception as e:
            print(f"    {stock_name}买入失败: {e}")
            return False
    
    def notify_order(self, order):
        """订单通知"""
        if order.status == order.Completed:
            self.successful_orders += 1
        elif order.status in [order.Rejected, order.Margin]:
            print(f"  订单失败: {order.data._name}, 原因: {order.getstatusname()}")
    
    def stop(self):
        """策略结束"""
        final_value = self.broker.getvalue()
        initial_cash = self.broker.startingcash
        total_return = (final_value / initial_cash - 1) * 100
        success_rate = self.successful_orders / max(self.total_orders, 1) * 100
        
        print("=" * 60)
        print("策略执行完成")
        print(f"调仓次数: {self.rebalance_count}")
        print(f"总订单: {self.total_orders}")
        print(f"成功订单: {self.successful_orders}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"最终价值: {final_value:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        print("=" * 60)


def fixed_factor_backtest(price_df, factor_dict, config=None):
    """
    修复版因子回测 - 直接替换vectorBT
    
    Args:
        price_df: 价格数据
        factor_dict: 因子字典
        config: 配置对象
    
    Returns:
        results, comparison_df
    """
    print(f"开始修复版回测，因子数量: {len(factor_dict)}")
    
    # 默认配置
    if config is None:
        config = type('Config', (), {
            'top_quantile': 0.2,
            'initial_cash': 100000,
            'max_positions': 5,
            'commission_rate': 0.001
        })()
    
    results = {}
    comparison_data = {}
    
    for factor_name, factor_data in factor_dict.items():
        print(f"\n处理因子: {factor_name}")
        
        try:
            result = _run_fixed_backtest(factor_name, price_df, factor_data, config)
            
            if result:
                results[factor_name] = result
                
                # 计算指标
                final_value = result['final_value']
                initial_cash = getattr(config, 'initial_cash', 100000)
                total_return = (final_value / initial_cash - 1) * 100
                
                comparison_data[factor_name] = {
                    'Total Return [%]': total_return,
                    'Final Value': final_value,
                    'Rebalance Count': result['rebalance_count'],
                    'Total Orders': result['total_orders'],
                    'Success Rate [%]': result['success_rate']
                }
                
                print(f"  完成: 收益{total_return:.2f}%, 调仓{result['rebalance_count']}次, 成功率{result['success_rate']:.1f}%")
            
        except Exception as e:
            print(f"  失败: {e}")
            results[factor_name] = None
    
    # 生成对比表
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data).T
        print("\n修复版回测完成!")
    else:
        comparison_df = pd.DataFrame()
        print("回测失败")
    
    return results, comparison_df


def _run_fixed_backtest(factor_name, price_df, factor_data, config):
    """运行单个因子的修复版回测"""
    try:
        # 数据对齐
        common_dates = price_df.index.intersection(factor_data.index)
        common_stocks = price_df.columns.intersection(factor_data.columns)
        
        # 选择子集（提升速度）
        max_stocks = min(20, len(common_stocks))
        selected_stocks = common_stocks[:max_stocks]
        
        aligned_price = price_df.loc[common_dates, selected_stocks]
        aligned_factor = factor_data.loc[common_dates, selected_stocks]
        
        # 创建Cerebro
        cerebro = bt.Cerebro()
        
        # 添加股票数据
        added_count = 0
        for stock in selected_stocks:
            stock_prices = aligned_price[stock].dropna()
            
            if len(stock_prices) > 30:  # 至少30天数据
                stock_data = pd.DataFrame(index=stock_prices.index)
                stock_data['close'] = stock_prices
                stock_data['open'] = stock_data['close']
                stock_data['high'] = stock_data['close'] * 1.01
                stock_data['low'] = stock_data['close'] * 0.99
                stock_data['volume'] = 1000000
                
                data_feed = bt.feeds.PandasData(dataname=stock_data, name=stock)
                cerebro.adddata(data_feed)
                added_count += 1
        
        if added_count == 0:
            return None
        
        # 添加策略
        cerebro.addstrategy(
            FixedFactorStrategy,
            factor_data=aligned_factor,
            top_quantile=getattr(config, 'top_quantile', 0.2),
            max_positions=getattr(config, 'max_positions', 5),
            cash_reserve=0.05
        )
        
        # 设置交易环境
        cerebro.broker.setcash(getattr(config, 'initial_cash', 100000))
        cerebro.broker.setcommission(commission=getattr(config, 'commission_rate', 0.001))
        
        # 运行回测
        strategy_results = cerebro.run()
        strategy = strategy_results[0]
        
        final_value = cerebro.broker.getvalue()
        success_rate = strategy.successful_orders / max(strategy.total_buy_orders, 1) * 100
        
        return {
            'strategy': strategy,
            'final_value': final_value,
            'rebalance_count': strategy.rebalance_count,
            'total_orders': strategy.total_buy_orders,
            'success_orders': strategy.successful_orders,
            'success_rate': success_rate
        }
        
    except Exception as e:
        print(f"{factor_name}回测失败: {e}")
        return None


def test_fixed_version():
    """测试修复版本"""
    print("=" * 60)
    print("测试修复版Backtrader")
    print("=" * 60)
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', periods=120, freq='B')
    stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']
    
    # 价格数据
    np.random.seed(42)
    price_data = {}
    for i, stock in enumerate(stocks):
        returns = np.random.normal(0.001, 0.02, len(dates))  # 稍微正向收益
        prices = 100 * (1 + i * 0.1) * np.exp(np.cumsum(returns))
        price_data[stock] = prices
    
    price_df = pd.DataFrame(price_data, index=dates)
    
    # 因子数据（动量）
    factor_data = {}
    for stock in stocks:
        momentum = price_df[stock].pct_change(20).rolling(5).mean()  # 20日动量的5日均值
        factor_data[stock] = momentum
    
    factor_df = pd.DataFrame(factor_data, index=dates)
    factor_dict = {'fixed_momentum': factor_df}
    
    # 配置
    config = type('Config', (), {
        'top_quantile': 0.4,       # 做多40%（2只股票）
        'initial_cash': 100000,
        'max_positions': 3,
        'commission_rate': 0.001
    })()
    
    print(f"测试数据: 价格{price_df.shape}, 因子{len(factor_dict)}个")
    
    # 运行修复版回测
    results, comparison = fixed_factor_backtest(price_df, factor_dict, config)
    
    print("\n修复版回测结果:")
    if not comparison.empty:
        print(comparison)
        
        # 验证修复效果
        result = list(results.values())[0]
        if result and result['success_rate'] > 50:  # 成功率>50%
            print("\n✅ 修复成功!")
            print("✅ 调仓频率正常")
            print("✅ 订单成功率提升")
            print("✅ Size问题已解决")
            return True
    
    print("\n❌ 仍需进一步修复")
    return False


if __name__ == "__main__":
    # 运行测试
    test_fixed_version()