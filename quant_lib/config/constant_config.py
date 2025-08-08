"""
配置常量模块

存储项目中使用的各种常量配置。
"""

import os
from pathlib import Path

# 路径配置
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = ROOT_DIR / 'data'
MODEL_DIR = ROOT_DIR / 'models'
RESULT_DIR = ROOT_DIR / 'results'
LOG_DIR = ROOT_DIR / 'logs'
CONFIG_DIR = ROOT_DIR / 'config'

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 数据路径
LOCAL_PARQUET_DATA_DIR = Path('D:\\lqs\\quantity\\parquet_data')

# 回测配置
DEFAULT_BENCHMARK = '000300.SH'
DEFAULT_COMMISSION_RATE = 0.0003
DEFAULT_SLIPPAGE_RATE = 0.0002
DEFAULT_CAPITAL = 10000000.0

# 因子配置
DEFAULT_FACTOR_WEIGHTS = {
    'value': 0.3,
    'momentum': 0.3,
    'quality': 0.2,
    'growth': 0.1,
    'volatility': 0.1
}

# 时间配置
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_WEEK = 5

# 股票池配置
INDEX_COMPONENTS = {
    '000300.SH': '沪深300',
    '000905.SH': '中证500',
    '000016.SH': '上证50',
    '399006.SZ': '创业板指'
}

# 行业分类配置
INDUSTRY_CLASSIFICATION = {
    'sw_l1': '申万一级行业',
    'sw_l2': '申万二级行业',
    'sw_l3': '申万三级行业',
    'csi_l1': '中证一级行业',
    'csi_l2': '中证二级行业'
}

# 日志配置
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# API配置
TUSHARE_TOKEN_PATH = ROOT_DIR / 'quant_lib' / 'tushare' / 'tushare_token_manager' / 'token.txt'

# 绘图配置
PLOT_STYLE = 'seaborn'
PLOT_FIGSIZE = (12, 6)
PLOT_DPI = 100
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

parquet_file_names = ['adj_factor', 'daily', 'daily_basic', 'daily_hfq', 'fina_indicator.parquet', 'index_weights',
                     'margin_detail', 'stk_limit']
every_day_parquet_file_names = ['adj_factor', 'daily', 'daily_basic', 'daily_hfq', 'index_weights',
                     'margin_detail', 'stk_limit']
#'index_weights'  fina_indicator
need_fix = ['adj_factor', 'daily', 'daily_basic', 'daily_hfq',
                     'margin_detail', 'stk_limit']

permanent__day = '22000101'