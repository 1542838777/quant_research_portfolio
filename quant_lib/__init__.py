"""
量化研究框架核心库

这个库提供了量化投资研究所需的核心功能，包括数据加载、因子计算、
回测引擎、评估指标计算等模块。
"""

__version__ = '0.1.0'



# 设置日志
from quant_lib.config.logger_config import setup_logger
logger = setup_logger('quant_lib')
