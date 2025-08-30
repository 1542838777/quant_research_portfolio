#!/usr/bin/env python3
"""
测试inf比率修复效果 - 验证极端情况下的稳健性
"""

import pandas as pd
import numpy as np
from projects._04backtesting.quant_backtester import QuantBacktester, BacktestConfig
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)

def create_extreme_low_volatility_data():
    """创建极低波动率测试数据 - 模拟用户的真实场景"""
    
    # 模拟4年多数据 (类似用户的1158天)
    dates = pd.date_range('2019-03-28', '2023-12-29', freq='D')
    stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D']
    
    logger.info(f"创建极端测试数据: {len(dates)}天, {len(stocks)}只股票")
    
    np.random.seed(42)
    price_data = {}
    
    for i, stock in enumerate(stocks):
        # 极低波动率的价格序列，模拟几乎没有交易的情况
        prices = [100.0]
        
        for day_idx in range(len(dates)-1):
            # 99%的时间价格不变，1%的时间有微小变化
            if np.random.random() < 0.01:  # 极低的变化频率
                change = np.random.normal(0, 0.001)  # 极小的波动
            else:
                change = 0.0  # 大部分时间价格不变
            
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))
        
        price_data[stock] = prices
    
    price_df = pd.DataFrame(price_data, index=dates)
    
    # 创建几乎无效的因子数据（大部分为0或极小值）
    factor_data = {}
    for stock in stocks:
        # 极少变化的因子值
        factor_values = np.zeros(len(dates))
        change_points = np.random.choice(len(dates), size=5, replace=False)
        for cp in change_points:
            factor_values[cp] = np.random.normal(0, 0.1)
        
        factor_data[stock] = factor_values
    
    factor_df = pd.DataFrame(factor_data, index=dates)
    
    # 输出统计信息
    logger.info(f"价格数据统计:")
    logger.info(f"  价格变化范围: {price_df.min().min():.4f} - {price_df.max().max():.4f}")
    logger.info(f"  平均日收益率: {price_df.pct_change().mean().mean():.8f}")
    logger.info(f"  收益率标准差: {price_df.pct_change().std().mean():.8f}")
    
    logger.info(f"因子数据统计:")
    logger.info(f"  因子值范围: {factor_df.min().min():.6f} - {factor_df.max().max():.6f}")
    logger.info(f"  非零因子值数量: {(factor_df != 0).sum().sum()}")
    
    return price_df, factor_df

def test_extreme_scenario():
    """测试极端场景下的指标修复"""
    logger.info("=" * 60)
    logger.info("🧪 测试极端场景下的inf修复效果")
    logger.info("=" * 60)
    
    # 创建极端测试数据
    price_df, factor_df = create_extreme_low_volatility_data()
    
    # 配置回测参数 - 模拟用户场景
    config = BacktestConfig(
        top_quantile=0.2,  # 选择前20%
        rebalancing_freq='M',  # 月度调仓
        initial_cash=300000.0,  # 30万初始资金
        commission_rate=0.0003,
        slippage_rate=0.001,
        stamp_duty=0.001
    )
    
    # 运行回测
    backtester = QuantBacktester(config)
    factor_dict = {'extreme_factor': factor_df}
    
    try:
        portfolios = backtester.run_backtest(price_df, factor_dict)
        portfolio = portfolios['extreme_factor']
        
        # 获取修正后的统计指标
        corrected_stats = backtester._calculate_corrected_stats(portfolio)
        
        logger.info("=" * 60)
        logger.info("✅ 极端场景测试结果")
        logger.info("=" * 60)
        
        # 检查之前有inf问题的指标
        problematic_metrics = [
            'Sharpe Ratio', 'Sortino Ratio', 'Omega Ratio', 'Profit Factor'
        ]
        
        all_fixed = True
        for metric in problematic_metrics:
            if metric in corrected_stats.index:
                value = corrected_stats[metric]
                if np.isinf(value):
                    logger.error(f"❌ {metric}: 仍为inf - {value}")
                    all_fixed = False
                elif np.isnan(value):
                    logger.error(f"❌ {metric}: 为NaN - {value}")
                    all_fixed = False
                else:
                    logger.info(f"✅ {metric}: {value:.4f}")
            else:
                logger.warning(f"⚠️ {metric}: 指标不存在")
                all_fixed = False
        
        # 显示其他关键指标
        logger.info(f"\n其他重要指标:")
        other_metrics = ['Total Return [%]', 'Max Drawdown [%]', 'Total Trades', 'Total Closed Trades']
        for metric in other_metrics:
            if metric in corrected_stats.index:
                logger.info(f"  {metric}: {corrected_stats[metric]:.4f}")
        
        # 检查原始vectorbt stats中的inf情况
        logger.info(f"\n原始vectorbt统计对比:")
        original_stats = portfolio.stats()
        for metric in problematic_metrics:
            if metric in original_stats.index:
                original_value = original_stats[metric]
                corrected_value = corrected_stats[metric] if metric in corrected_stats.index else 'N/A'
                
                if np.isinf(original_value):
                    logger.info(f"  {metric}: {original_value} (原始) -> {corrected_value} (修正)")
                else:
                    logger.info(f"  {metric}: {original_value:.4f} (原始) = {corrected_value} (修正)")
        
        return all_fixed
        
    except Exception as e:
        logger.error(f"❌ 回测执行失败: {e}")
        return False

def test_normal_scenario():
    """测试正常场景确保没有破坏原有功能"""
    logger.info("=" * 60)
    logger.info("🧪 测试正常场景确保功能完整")
    logger.info("=" * 60)
    
    # 创建正常的测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    
    np.random.seed(123)
    price_data = {}
    for stock in stocks:
        prices = [100.0]
        for _ in range(len(dates)-1):
            change = np.random.normal(0.001, 0.02)  # 正常的波动率
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))
        price_data[stock] = prices
    
    price_df = pd.DataFrame(price_data, index=dates)
    
    # 正常的因子数据
    factor_data = {}
    for stock in stocks:
        factor_data[stock] = np.random.normal(0, 1, len(dates))
    factor_df = pd.DataFrame(factor_data, index=dates)
    
    # 配置回测
    config = BacktestConfig(
        top_quantile=0.4,
        initial_cash=100000.0
    )
    
    backtester = QuantBacktester(config)
    factor_dict = {'normal_factor': factor_df}
    
    try:
        portfolios = backtester.run_backtest(price_df, factor_dict)
        portfolio = portfolios['normal_factor']
        corrected_stats = backtester._calculate_corrected_stats(portfolio)
        
        logger.info("正常场景关键指标:")
        key_metrics = ['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Win Rate [%]']
        for metric in key_metrics:
            if metric in corrected_stats.index:
                value = corrected_stats[metric]
                if np.isfinite(value):
                    logger.info(f"  ✅ {metric}: {value:.4f}")
                else:
                    logger.error(f"  ❌ {metric}: {value} (异常)")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 正常场景测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("🚀 开始测试inf修复效果")
    
    # 测试极端场景
    extreme_success = test_extreme_scenario()
    
    # 测试正常场景
    normal_success = test_normal_scenario()
    
    logger.info("=" * 60)
    logger.info("📋 测试总结")
    logger.info("=" * 60)
    
    if extreme_success and normal_success:
        logger.info("🎉 所有测试通过！inf问题已完全修复")
        return True
    else:
        logger.error("❌ 部分测试失败，需要进一步修复")
        logger.info(f"  极端场景: {'✅ 通过' if extreme_success else '❌ 失败'}")
        logger.info(f"  正常场景: {'✅ 通过' if normal_success else '❌ 失败'}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)