#!/usr/bin/env python3
"""
测试修复后的回测器 - 验证指标计算正确性
"""

import pandas as pd
import numpy as np
from projects._04backtesting.quant_backtester import QuantBacktester, BacktestConfig
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)

def create_test_data():
    """创建测试数据"""
    # 创建100个交易日的测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E', 'STOCK_F']
    
    # 创建价格数据 - 模拟不同表现的股票
    np.random.seed(42)
    price_data = {}
    
    for i, stock in enumerate(stocks):
        # 不同股票有不同的趋势和波动率
        trend = 0.0005 * (i - 2.5)  # 有些上涨，有些下跌
        volatility = 0.015 + 0.005 * i  # 不同的波动率
        
        prices = [100.0]
        if (i==1):
            prices=[500]
        for _ in range(len(dates)-1):
            change = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))
        
        price_data[stock] = prices
    
    price_df = pd.DataFrame(price_data, index=dates)
    
    # 创建因子数据 - 基于价格动量和反转效应
    factor_data = {}
    for stock in stocks:
        # 基于过去收益率的因子
        returns = price_df[stock].pct_change().fillna(0)
        momentum_factor = returns.rolling(10).mean().fillna(0)
        reversal_factor = -returns.rolling(5).mean().fillna(0)
        composite_factor = momentum_factor + 0.5 * reversal_factor
        factor_data[stock] = composite_factor.values
    
    factor_df = pd.DataFrame(factor_data, index=dates)
    
    logger.info(f"测试数据创建完成: {price_df.shape}")
    logger.info(f"价格范围: {price_df.min().min():.2f}-{price_df.max().max():.2f}")
    logger.info(f"因子范围: {factor_df.min().min():.4f}-{factor_df.max().max():.4f}")
    
    return price_df, factor_df

def t_flow_backtester():
    """测试修复后的回测器"""
    logger.info("=" * 60)
    logger.info("🚀 测试修复后的回测器")
    logger.info("=" * 60)
    
    # 创建测试数据
    price_df, factor_df = create_test_data()
    
    # 配置回测参数
    config = BacktestConfig(
        top_quantile=0.3,  # 选择前30%股票
        rebalancing_freq='W',  # 周度调仓
        commission_rate=0.0003,
        slippage_rate=0.001,
        initial_cash=1000000.0,
        max_positions=3  # 最多持仓3只股票
    )
    
    # 创建回测器
    backtester = QuantBacktester(config)
    
    # 运行回测
    factor_dict = {'test_factor': factor_df}
    portfolios = backtester.run_backtest(price_df, factor_dict)
    
    # 验证结果
    portfolio = portfolios['test_factor']
    stats = backtester._calculate_corrected_stats(portfolio)
    
    logger.info("=" * 60)
    logger.info("📊 修复验证结果")
    logger.info("=" * 60)
    
    # 检查之前有问题的指标
    problematic_metrics = {
        'Total Trades': '总交易数',
        'Total Closed Trades': '已关闭交易数',
        'Total Open Trades': '未关闭交易数',
        'Win Rate [%]': '胜率(%)',
        'Best Trade [%]': '最佳交易(%)',
        'Worst Trade [%]': '最差交易(%)',
        'Avg Winning Trade [%]': '平均盈利交易(%)',
        'Avg Losing Trade [%]': '平均亏损交易(%)',
        'Profit Factor': '盈利因子',
        'Expectancy': '期望值',
        'Sharpe Ratio': '夏普比率',
        'Calmar Ratio': 'Calmar比率',
        'Omega Ratio': 'Omega比率',
        'Sortino Ratio': 'Sortino比率'
    }
    
    fixed_count = 0
    total_count = 0
    
    for metric, chinese_name in problematic_metrics.items():
        total_count += 1
        if metric in stats.index:
            value = stats[metric]
            if pd.isna(value):
                logger.error(f"❌ {chinese_name}: NaN (未修复)")
            elif np.isinf(value) and metric not in ['Sharpe Ratio', 'Calmar Ratio', 'Omega Ratio', 'Sortino Ratio']:
                logger.error(f"❌ {chinese_name}: {value} (inf值异常)")
            else:
                logger.info(f"✅ {chinese_name}: {value:.4f}")
                fixed_count += 1
        else:
            logger.warning(f"⚠️ {chinese_name}: 指标不存在")
    
    # 检查交易记录
    trades = portfolio.trades.records_readable
    logger.info(f"\n交易记录验证:")
    logger.info(f"  实际交易记录数: {len(trades)}")
    
    if len(trades) > 0:
        open_trades = trades[trades['Status'] == 0] if 'Status' in trades.columns else pd.DataFrame()
        closed_trades = trades[trades['Status'] == 1] if 'Status' in trades.columns else trades
        
        logger.info(f"  开仓交易数: {len(open_trades)}")
        logger.info(f"  已关闭交易数: {len(closed_trades)}")
        
        if len(closed_trades) > 0:
            profitable_trades = closed_trades[closed_trades['PnL'] > 0]
            logger.info(f"  盈利交易数: {len(profitable_trades)}")
            logger.info(f"  实际胜率: {len(profitable_trades)/len(closed_trades)*100:.1f}%")
            
            logger.info(f"\n前5笔交易示例:")
            print(closed_trades.head().to_string())
        
    logger.info("=" * 60)
    logger.info(f"✅ 修复验证完成: {fixed_count}/{total_count} 个指标正常")
    logger.info("=" * 60)
    
    return fixed_count >= total_count * 0.8  # 80%以上指标正常即为成功

if __name__ == "__main__":
    #最好的测试vectorBT demo.
     t_flow_backtester()
