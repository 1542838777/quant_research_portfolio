"""
量化研究框架核心库

这个库提供了量化投资研究所需的核心功能，包括数据加载、因子计算、
回测引擎、评估指标计算等模块。
"""

__version__ = '0.1.0'

# 导出常用模块，方便用户导入
from quant_lib.data_loader import DataLoader
from quant_lib.backtesting import BacktestEngine
from quant_lib.evaluation import (
    calculate_ic, 
    calculate_quantile_returns,
    calculate_turnover,
    calculate_sharpe
)

# 设置日志
from quant_lib.config.logger_config import setup_logger
logger = setup_logger('quant_lib')

# 版本信息
__all__ = [
    'DataLoader',
    'BacktestEngine',
    'calculate_ic',
    'calculate_quantile_returns',
    'calculate_turnover',
    'calculate_sharpe',
]
