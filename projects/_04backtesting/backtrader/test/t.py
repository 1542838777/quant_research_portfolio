import unittest
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import date, datetime

from projects._04backtesting.backtrader.backtrader_enhanced_strategy import EnhancedFactorStrategy


# 假设你的 EnhancedFactorStrategy 策略类定义在 strategy.py 文件中

# 同样，假设你用于生成调仓日历的辅助函数也在这里

class TestEnhancedFactorStrategy(unittest.TestCase):

    def setUp(self):
        """
        每个测试用例运行前都会调用的“准备”函数。
        我们在这里准备一个干净的Cerebro引擎。
        """
        self.cerebro = bt.Cerebro(cheat_on_open=True)
        self.cerebro.broker.set_coc(True)
        self.cerebro.broker.setcash(100000.0)  # 固定的初始资金，便于断言

    def run_scenario(self, price_df, holding_signals_df, strategy_params):
        """
        一个辅助函数，用于运行一个指定场景的回测。
        """
        # 1. 加载数据
        for stock in price_df.columns:
            df = pd.DataFrame(index=price_df.index)
            df['open'] = price_df[stock]
            df['high'] = price_df[stock]
            df['low'] = price_df[stock]
            df['close'] = price_df[stock]
            df['volume'] = 0
            df['openinterest'] = 0

            data_feed = bt.feeds.PandasData(datename=df, name=stock)
            self.cerebro.adddata(data_feed)

        # 2. 添加策略
        self.cerebro.addstrategy(EnhancedFactorStrategy, **strategy_params)

        # 3. 运行回测
        results = self.cerebro.run()
        return results[0]  # 返回策略实例，方便我们检查其内部状态

    # --- 我们将在这里编写所有的测试用例 ---

    def test_buy_retry_after_suspension(self):

        print("\n--- 正在测试【场景一：买入重试机制】---")

        # 1. 场景设计：构造数据
        dates = pd.to_datetime(pd.date_range(start='2024-01-01', periods=10))
        price_df = pd.DataFrame(100.0, index=dates, columns=['STOCK_A', 'STOCK_B'])

        # 让 STOCK_A 在 01-02 和 01-03 停牌
        price_df.loc['2024-01-02', 'STOCK_A'] = np.nan
        price_df.loc['2024-01-03', 'STOCK_A'] = np.nan

        # 理想持仓计划：在 01-02 买入 A，并一直持有
        holding_signals = pd.DataFrame(False, index=dates, columns=['STOCK_A', 'STOCK_B'])
        holding_signals.loc['2024-01-02':, 'STOCK_A'] = True

        # 策略参数
        params = {
            'holding_signals': holding_signals.shift(-1),  # 预处理，让T-1日能看到T日计划
            'rebalance_dates': [date(2024, 1, 2)],
            'max_positions': 1,
            'retry_buy_days': 3  # 设置3天重试期
        }

        # 2. 运行场景
        strategy_instance = self.run_scenario(price_df, holding_signals, params)

        # 3. 结果断言 (Assert)
        # a. 检查最终是否只持有一只股票
        final_positions = {d._name for d in strategy_instance.datas if strategy_instance.getposition(d).size > 0}
        self.assertEqual(len(final_positions), 1)
        self.assertIn('STOCK_A', final_positions)

        # b. 检查“待买清单”在最后是否为空
        self.assertEqual(len(strategy_instance.pending_buys), 0)

        # c. 检查交易记录，确认买入发生在复牌日（01-04）
        trades = strategy_instance.analyzers.trades.get_analysis()
        buy_trade = next(iter(trades.values()))[0]  # 获取第一笔交易
        buy_date = bt.num2date(buy_trade.dt).date()
        self.assertEqual(buy_date, date(2024, 1, 4))

        print("✅ 测试通过！")

    def test_pending_sell_on_intra_rebalance_suspension(self):


        print("\n--- 正在测试【场景二：持仓期间停牌】---")

        # 1. 场景设计
        dates = pd.to_datetime(pd.date_range(start='2024-01-01', periods=10))
        price_df = pd.DataFrame(100.0, index=dates, columns=['STOCK_A'])
        price_df.loc['2024-01-03', 'STOCK_A'] = np.nan  # T+2日停牌

        holding_signals = pd.DataFrame(True, index=dates, columns=['STOCK_A'])

        params = {
            'holding_signals': holding_signals.shift(-1),
            'rebalance_dates': [date(2024, 1, 1)],  # 只在第一天调仓买入
            'max_positions': 1
        }

        # 2. 运行
        strategy_instance = self.run_scenario(price_df, holding_signals, params)

        # 3. 断言
        # a. 检查“待卖清单”中是否有 STOCK_A
        self.assertIn('STOCK_A', strategy_instance.pending_sells)
        # b. 检查原因是否正确
        reason = strategy_instance.pending_sells['STOCK_A'][2]
        self.assertIn("停牌", reason)  # 模糊匹配原因字符串

        print("✅ 测试通过！")

    def test_buy_retry_timeout(self):
        print("\n--- 正在测试【场景三：买入任务超期】---")

        # 1. 场景设计
        dates = pd.to_datetime(pd.date_range(start='2024-01-01', periods=10))
        price_df = pd.DataFrame(100.0, index=dates, columns=['STOCK_A'])
        # 连续停牌4天，超过3天的重试期
        price_df.loc['2024-01-02':'2024-01-05', 'STOCK_A'] = np.nan

        holding_signals = pd.DataFrame(True, index=dates, columns=['STOCK_A'])

        params = {
            'holding_signals': holding_signals.shift(-1),
            'rebalance_dates': [date(2024, 1, 2)],
            'max_positions': 1,
            'retry_buy_days': 3  # 重试期为3天
        }

        # 2. 运行
        strategy_instance = self.run_scenario(price_df, holding_signals, params)

        # 3. 断言
        # a. 检查最终持仓应为空，因为买入任务已超期放弃
        final_positions = {d._name for d in strategy_instance.datas if strategy_instance.getposition(d).size > 0}
        self.assertEqual(len(final_positions), 0)

        # b. 检查“待买清单”在最后也应为空
        self.assertEqual(len(strategy_instance.pending_buys), 0)

        print("✅ 测试通过！")

    def test_max_holding_days_with_suspension(self):
        print("\n--- 正在测试【场景四：时间止损遇上停牌】---")

        # 1. 场景设计
        dates = pd.to_datetime(pd.date_range(start='2024-01-01', periods=10))
        price_df = pd.DataFrame(100.0, index=dates, columns=['STOCK_A'])
        # 在第4天（持仓的第3天）停牌
        price_df.loc['2024-01-04', 'STOCK_A'] = np.nan

        holding_signals = pd.DataFrame(True, index=dates, columns=['STOCK_A'])

        params = {
            'holding_signals': holding_signals.shift(-1),
            'rebalance_dates': [date(2024, 1, 1)],
            'max_positions': 1,
            'max_holding_days': 3  # 持仓上限3天
        }

        # 2. 运行
        strategy_instance = self.run_scenario(price_df, holding_signals, params)

        # 3. 断言
        # a. 检查“待卖清单”中是否有 STOCK_A
        self.assertIn('STOCK_A', strategy_instance.pending_sells)
        # b. 检查原因是否是“强制超期”
        reason = strategy_instance.pending_sells['STOCK_A'][2]
        self.assertEqual(reason, "强制超期")

        print("✅ 测试通过！")
            

if __name__ == '__main__':
    unittest.main()