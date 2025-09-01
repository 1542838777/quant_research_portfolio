"""
简化调试 - 找出调仓次数为0的根本原因
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime

class MinimalStrategy(bt.Strategy):
    """最简单的调试策略"""
    
    def __init__(self):
        print("策略初始化...")
        self.day_count = 0
        self.rebalance_count = 0
    
    def next(self):
        """每日执行"""
        current_date = self.datetime.date(0)
        self.day_count += 1
        
        # 每月第一个工作日调仓
        if current_date.day <= 3:
            self.rebalance_count += 1
            print(f"第{self.rebalance_count}次调仓: {current_date}")
            
            # 简单买入第一只股票
            if len(self.datas) > 0:
                data = self.datas[0]
                position = self.getposition(data)
                
                if position.size == 0:
                    order = self.order_target_percent(data=data, target=0.2)
                    print(f"  下达买入订单: {data._name}")
                else:
                    print(f"  已持有: {data._name}")
        
        # 每10天报告一次
        if self.day_count % 20 == 0:
            print(f"第{self.day_count}天: {current_date}, 已调仓{self.rebalance_count}次")
    
    def notify_order(self, order):
        """订单通知"""
        if order.status == order.Completed:
            action = "买入" if order.isbuy() else "卖出"
            print(f"  订单成功: {action} {order.data._name}, 数量: {order.executed.size:.0f}")
    
    def stop(self):
        """结束"""
        final_value = self.broker.getvalue()
        print(f"调试结果: 调仓{self.rebalance_count}次, 最终价值{final_value:,.2f}")


def quick_debug():
    """快速调试"""
    print("开始快速调试...")
    
    # 1. 创建简单数据
    dates = pd.date_range('2020-01-01', periods=100, freq='B')
    
    # 单只股票数据
    stock_data = pd.DataFrame(index=dates)
    stock_data['close'] = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    stock_data['open'] = stock_data['close']
    stock_data['high'] = stock_data['close'] * 1.01
    stock_data['low'] = stock_data['close'] * 0.99
    stock_data['volume'] = 1000000
    
    print(f"数据创建完成: {stock_data.shape}")
    
    # 2. 创建Cerebro
    cerebro = bt.Cerebro()
    
    # 添加数据
    data_feed = bt.feeds.PandasData(dataname=stock_data, name='TEST_STOCK')
    cerebro.adddata(data_feed)
    
    # 添加策略
    cerebro.addstrategy(MinimalStrategy)
    
    # 设置参数
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)
    
    print("开始运行调试...")
    
    # 3. 运行
    try:
        results = cerebro.run()
        strategy = results[0]
        
        final_value = cerebro.broker.getvalue()
        total_return = (final_value / 100000 - 1) * 100
        
        print("调试完成!")
        print(f"最终结果: {final_value:,.2f} ({total_return:.2f}%)")
        
        if strategy.rebalance_count > 0:
            print("✅ 调仓逻辑正常工作")
        else:
            print("❌ 调仓逻辑有问题")
        
        return True
        
    except Exception as e:
        print(f"调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("简化调试程序")
    print("=" * 50)
    
    # 运行简单调试
    quick_debug()
    
    print("\n" + "=" * 50)
    print("调试调仓日期")
    print("=" * 50)
    
    # 调试调仓日期
    debug_rebalance_dates()