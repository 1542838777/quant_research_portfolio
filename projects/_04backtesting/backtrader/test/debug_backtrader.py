"""
调试Backtrader问题 - 找出为什么调仓次数为0
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import backtrader as bt

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from quant_lib.rebalance_utils import generate_rebalance_dates


def debug_rebalance_dates():
    """调试调仓日期生成"""
    print("🔍 调试调仓日期生成...")
    
    # 加载真实数据
    try:
        result_manager = ResultLoadManager(
            calcu_return_type='c2c', 
            version='20190328_20231231',
            is_raw_factor=False
        )
        
        factor_data = result_manager.get_factor_data(
            'lqs_orthogonal_v1', '000906', '2020-01-01', '2020-12-31'
        )
        
        if factor_data is None:
            print("❌ 因子数据加载失败")
            return
        
        print(f"因子数据: {factor_data.shape}")
        print(f"因子日期范围: {factor_data.index.min()} ~ {factor_data.index.max()}")
        print(f"前几个日期: {factor_data.index[:5].tolist()}")
        
        # 生成调仓日期
        rebalance_dates = generate_rebalance_dates(factor_data.index, 'M')
        print(f"调仓日期数量: {len(rebalance_dates)}")
        print(f"前几个调仓日期: {rebalance_dates[:5]}")
        
        # 检查日期类型
        print(f"因子日期类型: {type(factor_data.index[0])}")
        print(f"调仓日期类型: {type(rebalance_dates[0])}")
        
        return factor_data, rebalance_dates
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


class DebugStrategy(bt.Strategy):
    """调试策略 - 专门查看调仓逻辑"""
    
    params = (
        ('factor_data', None),
        ('rebalance_dates', []),
        ('debug_mode', True),
    )
    
    def __init__(self):
        print("🔧 初始化调试策略...")
        
        self.rebalance_dates_list = list(self.p.rebalance_dates)
        self.rebalance_count = 0
        self.day_count = 0
        
        print(f"调仓日期列表: {len(self.rebalance_dates_list)}个")
        if self.rebalance_dates_list:
            print(f"前5个调仓日期: {self.rebalance_dates_list[:5]}")
    
    def next(self):
        """主循环 - 调试版本"""
        current_date = self.datetime.date(0)
        current_datetime = self.datetime.datetime(0)
        
        self.day_count += 1
        
        # 详细的调仓日期匹配调试
        is_rebalance = False
        
        # 方法1：精确匹配
        current_timestamp = pd.Timestamp(current_date)
        if current_timestamp in self.rebalance_dates_list:
            is_rebalance = True
            print(f"✅ 精确匹配调仓日期: {current_date}")
        
        # 方法2：模糊匹配（允许1天误差）
        if not is_rebalance:
            for rebalance_date in self.rebalance_dates_list:
                if abs((current_timestamp - rebalance_date).days) <= 1:
                    is_rebalance = True
                    print(f"✅ 模糊匹配调仓日期: {current_date} ≈ {rebalance_date.date()}")
                    break
        
        # 方法3：每月第一个交易日
        if not is_rebalance and current_date.day <= 3:
            is_rebalance = True
            print(f"✅ 月初调仓: {current_date}")
        
        if is_rebalance:
            self.rebalance_count += 1
            print(f">>> 第{self.rebalance_count}次调仓: {current_date}")
            
            # 简单买入逻辑测试
            if len(self.datas) > 0:
                data = self.datas[0]  # 买入第一只股票
                position = self.getposition(data)
                
                if position.size == 0:
                    # 买入10%权重
                    order = self.order_target_percent(data=data, target=0.1)
                    if order:
                        print(f"  买入订单已下达: {data._name}")
                    else:
                        print(f"  买入订单失败: {data._name}")
        
        # 每10天输出一次调试信息
        if self.day_count % 10 == 0:
            print(f"📅 第{self.day_count}天: {current_date}, 已调仓{self.rebalance_count}次")
    
    def notify_order(self, order):
        """订单通知"""
        if order.status == order.Completed:
            action = "买入" if order.isbuy() else "卖出"
            print(f"  ✅ {action}成功: {order.data._name}, 数量: {order.executed.size:.0f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            action = "买入" if order.isbuy() else "卖出"
            print(f"  ❌ {action}失败: {order.data._name}, 原因: {order.getstatusname()}")
    
    def stop(self):
        """策略结束"""
        final_value = self.broker.getvalue()
        total_return = (final_value / self.broker.startingcash - 1) * 100
        
        print("=" * 50)
        print("🔍 调试结果:")
        print(f"  总交易日: {self.day_count}")
        print(f"  调仓次数: {self.rebalance_count}")
        print(f"  最终价值: {final_value:,.2f}")
        print(f"  总收益率: {total_return:.2f}%")
        print("=" * 50)


def run_debug_test():
    """运行调试测试"""
    print("🔍 开始调试测试...")
    
    # 1. 调试调仓日期
    factor_data, rebalance_dates = debug_rebalance_dates()
    if factor_data is None:
        print("❌ 无法加载数据，停止调试")
        return
    
    # 2. 创建简单的价格数据用于测试
    price_data = pd.DataFrame(index=factor_data.index)
    
    # 选择几只股票创建价格数据
    selected_stocks = factor_data.columns[:5]
    for i, stock in enumerate(selected_stocks):
        # 创建简单的价格走势
        base_price = 100 + i * 10
        returns = np.random.normal(0.001, 0.02, len(factor_data))
        np.random.seed(42 + i)  # 确保可重现
        prices = base_price * np.exp(np.cumsum(returns))
        price_data[stock] = prices
    
    print(f"创建价格数据: {price_data.shape}")
    
    # 3. 创建Cerebro进行调试
    cerebro = bt.Cerebro()
    
    # 添加几只股票的数据
    for stock in selected_stocks:
        stock_prices = price_data[stock].dropna()
        
        stock_data = pd.DataFrame(index=stock_prices.index)
        stock_data['close'] = stock_prices
        stock_data['open'] = stock_data['close']
        stock_data['high'] = stock_data['close'] * 1.01
        stock_data['low'] = stock_data['close'] * 0.99
        stock_data['volume'] = 1000000
        
        data_feed = bt.feeds.PandasData(dataname=stock_data, name=stock)
        cerebro.adddata(data_feed)
    
    print(f"添加了{len(cerebro.datas)}只股票数据")
    
    # 4. 添加调试策略
    cerebro.addstrategy(
        DebugStrategy,
        factor_data=factor_data[selected_stocks],
        rebalance_dates=rebalance_dates,
        debug_mode=True
    )
    
    # 5. 设置参数
    cerebro.broker.setcash(300000)
    cerebro.broker.setcommission(commission=0.002)
    
    # 6. 运行调试
    print("开始调试运行...")
    try:
        results = cerebro.run()
        strategy = results[0]
        print("✅ 调试运行成功")
        
    except Exception as e:
        print(f"❌ 调试运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_debug_test()