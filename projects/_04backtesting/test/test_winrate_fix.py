#!/usr/bin/env python3
"""
测试Win Rate修复效果 - 验证不会错误覆盖正确的统计指标
"""

import pandas as pd
import numpy as np
from projects._04backtesting.quant_backtester import QuantBacktester, BacktestConfig
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)

def create_realistic_test_data():
    """创建有正常盈亏的测试数据"""
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    
    np.random.seed(42)
    price_data = {}
    
    for stock in stocks:
        # 创建有明显趋势和波动的价格序列
        prices = [100.0]
        trend = np.random.choice([-0.001, 0.0005, 0.002])  # 不同的趋势
        
        for _ in range(len(dates)-1):
            change = np.random.normal(trend, 0.025)  # 正常波动率
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))
            
        price_data[stock] = prices
    
    price_df = pd.DataFrame(price_data, index=dates)
    
    # 基于价格动量的因子
    factor_data = {}
    for stock in stocks:
        returns = price_df[stock].pct_change().fillna(0)
        momentum = returns.rolling(20).mean()
        factor_data[stock] = momentum.fillna(0)
    
    factor_df = pd.DataFrame(factor_data, index=dates)
    
    logger.info(f"测试数据创建:")
    logger.info(f"  价格数据: {price_df.shape}")
    logger.info(f"  价格范围: {price_df.min().min():.2f} - {price_df.max().max():.2f}")
    logger.info(f"  因子范围: {factor_df.min().min():.4f} - {factor_df.max().max():.4f}")
    
    return price_df, factor_df

def test_winrate_preservation():
    """测试Win Rate等指标的正确保留"""
    logger.info("=" * 60)
    logger.info("🧪 测试Win Rate等指标的正确保留")
    logger.info("=" * 60)
    
    # 创建测试数据
    price_df, factor_df = create_realistic_test_data()
    
    # 配置回测
    config = BacktestConfig(
        top_quantile=0.3,
        rebalancing_freq='W',
        initial_cash=200000.0
    )
    
    # 运行回测
    backtester = QuantBacktester(config)
    factor_dict = {'test_factor': factor_df}
    portfolios = backtester.run_backtest(price_df, factor_dict)
    
    portfolio = portfolios['test_factor']
    
    # 获取原始和修正后的统计
    original_stats = portfolio.stats()
    corrected_stats = backtester._calculate_corrected_stats(portfolio)
    
    logger.info("=" * 60)
    logger.info("📊 原始 vs 修正统计对比")
    logger.info("=" * 60)
    
    # 检查关键交易指标
    key_metrics = [
        'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
        'Avg Winning Trade [%]', 'Avg Losing Trade [%]', 
        'Profit Factor', 'Expectancy'
    ]
    
    all_correct = True
    
    for metric in key_metrics:
        if metric in original_stats.index:
            original_value = original_stats[metric]
            corrected_value = corrected_stats.get(metric, 'Missing')
            
            # 判断原始值是否正常
            is_original_abnormal = np.isinf(original_value) or np.isnan(original_value)
            
            if is_original_abnormal:
                logger.info(f"✅ {metric}:")
                logger.info(f"    原始: {original_value} (异常) -> 修正: {corrected_value}")
            else:
                # 原始值正常，应该保持不变
                if abs(float(corrected_value) - float(original_value)) < 1e-6:
                    logger.info(f"✅ {metric}: {original_value:.4f} (保持不变)")
                else:
                    logger.error(f"❌ {metric}: {original_value:.4f} (原始) -> {corrected_value} (错误修改)")
                    all_correct = False
        else:
            logger.warning(f"⚠️ {metric}: 原始统计中不存在")
    
    # 检查比率指标
    logger.info("\n比率指标检查:")
    ratio_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Omega Ratio', 'Calmar Ratio']
    
    for metric in ratio_metrics:
        if metric in original_stats.index:
            original_value = original_stats[metric]
            corrected_value = corrected_stats.get(metric, 'Missing')
            
            if np.isinf(original_value):
                logger.info(f"✅ {metric}: {original_value} -> {corrected_value} (已修复inf)")
            else:
                logger.info(f"✅ {metric}: {original_value:.4f} (正常)")
    
    # 显示交易详情验证
    trades = portfolio.trades.records_readable
    if len(trades) > 0:
        winning_trades = len(trades[trades['PnL'] > 0])
        total_trades = len(trades)
        manual_win_rate = winning_trades / total_trades * 100
        
        logger.info(f"\n手动验证:")
        logger.info(f"  总交易数: {total_trades}")
        logger.info(f"  盈利交易数: {winning_trades}")
        logger.info(f"  手动计算胜率: {manual_win_rate:.4f}%")
        logger.info(f"  原始Win Rate: {original_stats.get('Win Rate [%]', 'N/A')}")
        logger.info(f"  修正后Win Rate: {corrected_stats.get('Win Rate [%]', 'N/A')}")
        
        if abs(manual_win_rate - float(original_stats.get('Win Rate [%]', 0))) < 0.01:
            logger.info("✅ 原始Win Rate计算正确")
        else:
            logger.error("❌ 原始Win Rate计算有误")
            all_correct = False
    
    return all_correct

def main():
    """主函数"""
    logger.info("🚀 开始测试Win Rate修复")
    
    success = test_winrate_preservation()
    
    logger.info("=" * 60)
    logger.info("📋 测试结果")
    logger.info("=" * 60)
    
    if success:
        logger.info("🎉 测试通过！Win Rate等指标正确保留")
    else:
        logger.error("❌ 测试失败，存在错误覆盖正确指标的问题")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)