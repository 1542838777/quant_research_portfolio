"""
多因子模型回测脚本

执行多因子模型的回测流程，包括数据加载、因子计算、回测和评估。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
import datetime
from pathlib import Path
import argparse

# 添加项目根目录到路径，以便导入自定义模块
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.data_loader import DataLoader
from quant_lib.factor_factory import (
    create_factor,
    create_factor_combiner
)
from quant_lib.backtesting import create_backtest_engine
from quant_lib.evaluation import (
    calculate_ic,
    calculate_ic_decay,
    calculate_quantile_returns,
    plot_ic_series,
    plot_ic_decay,
    plot_quantile_returns
)
from quant_lib.utils.file_utils import (
    ensure_dir_exists,
    save_to_csv,
    save_to_pickle,
    load_from_yaml
)
from quant_lib.config.constant_config import (
    LOCAL_PARQUET_DATA_DIR,
    RESULT_DIR
)
from quant_lib.config.logger_config import setup_logger

# 配置日志
logger = setup_logger(__name__)


def load_config(config_path: str) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    logger.info(f"加载配置文件: {config_path}")
    return load_from_yaml(config_path)


def load_data(config: dict) -> dict:
    """
    加载数据
    
    Args:
        config: 配置字典
        
    Returns:
        数据字典
    """
    logger.info("开始加载数据...")
    
    # 创建数据加载器
    data_loader = DataLoader(data_path=LOCAL_PARQUET_DATA_DIR)
    
    # 确定需要加载的字段
    fields = ['close']
    for factor_config in config['factors']:
        if factor_config['name'] == 'value':
            fields.extend(factor_config['params']['fields'])
        elif factor_config['name'] == 'quality':
            fields.extend(factor_config['params']['fields'])
        elif factor_config['name'] == 'growth':
            fields.extend(factor_config['params']['fields'])
    
    # 加载数据
    data_dict = data_loader.get_raw_dfs_by_require_fields(fields=fields, start_date=config['start_date'],
                                                          end_date=config['end_date'])
    
    logger.info(f"数据加载完成，共加载 {len(data_dict)} 个字段")
    return data_dict


def calculate_factors(config: dict, data_dict: dict) -> pd.DataFrame:
    """
    计算因子
    
    Args:
        config: 配置字典
        data_dict: 数据字典
        
    Returns:
        组合因子DataFrame
    """
    logger.info("开始计算因子...")
    
    # 创建因子列表和权重列表
    factor_types = []
    weights = []
    
    for factor_config in config['factors']:
        factor_types.append(factor_config['name'])
        weights.append(factor_config['weight'])
    
    # 创建因子组合器
    factor_combiner = create_factor_combiner(factor_types, weights)
    
    # 计算组合因子
    composite_factor = factor_combiner.compute(data_dict)
    
    logger.info("因子计算完成")
    return composite_factor


def run_backtest(config: dict, data_dict: dict, factor_df: pd.DataFrame) -> dict:
    """
    执行回测
    
    Args:
        config: 配置字典
        data_dict: 数据字典
        factor_df: 因子DataFrame
        
    Returns:
        回测结果字典
    """
    logger.info("开始执行回测...")
    
    # 获取回测参数
    backtest_config = config['backtest']
    
    # 创建回测引擎
    engine = create_backtest_engine(
        start_date=config['start_date'],
        end_date=config['end_date'],
        universe=list(data_dict['close'].columns),
        rebalance_freq=backtest_config['rebalance_freq'],
        n_stocks=backtest_config['n_stocks'],
        fee_rate=backtest_config['fee_rate'],
        slippage=backtest_config['slippage'],
        benchmark=backtest_config['benchmark'],
        capital=backtest_config['capital']
    )
    
    # 准备基准收益率数据
    # 实际项目中应该从数据源加载基准指数数据
    # 这里简化处理，使用等权组合作为基准
    benchmark_returns = data_dict['close'].pct_change().mean(axis=1)
    
    # 执行回测
    result = engine.run(
        price_df=data_dict['close'],
        factor_df=factor_df,
        benchmark_returns=benchmark_returns
    )
    
    logger.info("回测完成")
    return result


def evaluate_factors(config: dict, factor_df: pd.DataFrame, data_dict: dict, result_dir: Path) -> None:
    """
    评估因子
    
    Args:
        config: 配置字典
        factor_df: 因子DataFrame
        data_dict: 数据字典
        result_dir: 结果保存目录
    """
    logger.info("开始评估因子...")
    
    # 计算未来收益率
    forward_returns = data_dict['close'].shift(-1) / data_dict['close'] - 1
    
    # 计算IC
    ic_series = calculate_ic(factor_df, forward_returns)
    
    # 计算IC衰减
    ic_decay = calculate_ic_decay(
        factor_df, 
        data_dict['close'], 
        periods=config['evaluate_factor_score']['ic_decay_periods']
    )
    
    # 计算分位数收益
    quantile_returns = calculate_quantile_returns(
        factor_df, 
        data_dict['close'], 
        n_quantiles=config['evaluate_factor_score']['quantile_count']
    )
    
    # 保存结果
    save_to_csv(ic_series.to_frame('IC'), result_dir / 'ic_series.csv')
    save_to_csv(ic_decay, result_dir / 'ic_decay.csv')
    
    for period, returns in quantile_returns.items():
        save_to_csv(returns, result_dir / f'quantile_returns_{period}.csv')
    
    # 绘制图表
    plot_ic_series(ic_series)
    plt.savefig(result_dir / 'ic_series.png')
    
    plot_ic_decay(ic_decay)
    plt.savefig(result_dir / 'ic_decay.png')
    
    for period, returns in quantile_returns.items():
        plot_quantile_returns(returns, period)
        plt.savefig(result_dir / f'quantile_returns_{period}.png')
    
    logger.info("因子评估完成")


def save_results(result, result_dir: Path) -> None:
    """
    保存回测结果
    
    Args:
        result: 回测结果对象
        result_dir: 结果保存目录
    """
    logger.info("保存回测结果...")
    
    # 保存性能指标
    metrics_df = result.summary()
    save_to_csv(metrics_df, result_dir / 'performance_metrics.csv')
    
    # 保存收益率序列
    returns_df = pd.DataFrame({
        'portfolio': result.portfolio_returns,
        'benchmark': result.benchmark_returns
    })
    save_to_csv(returns_df, result_dir / 'returns.csv')
    
    # 保存持仓信息
    save_to_csv(result.positions, result_dir / 'positions.csv')
    
    # 保存换手率
    save_to_csv(result.turnover.to_frame('turnover'), result_dir / 'turnover.csv')
    
    # 绘制收益曲线
    result.plot_returns()
    plt.savefig(result_dir / 'cumulative_returns.png')
    
    # 绘制回撤曲线
    result.plot_drawdown()
    plt.savefig(result_dir / 'drawdown.png')
    
    # 绘制月度收益热力图
    result.plot_monthly_returns()
    plt.savefig(result_dir / 'monthly_returns.png')
    
    logger.info(f"回测结果已保存至: {result_dir}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多因子模型回测')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 加载配置
    config_path = current_dir / args.config
    config = load_config(config_path)
    
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = RESULT_DIR / 'multi_factor_model' / timestamp
    ensure_dir_exists(result_dir)
    
    # 保存配置副本
    with open(result_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 加载数据
    data_dict = load_data(config)
    
    # 计算因子
    factor_df = calculate_factors(config, data_dict)
    
    # 执行回测
    result = run_backtest(config, data_dict, factor_df)
    
    # 评估因子
    evaluate_factors(config, factor_df, data_dict, result_dir)
    
    # 保存结果
    save_results(result, result_dir)
    
    # 打印性能摘要
    print("\n=== 回测性能摘要 ===")
    print(result.summary())


if __name__ == '__main__':
    main()