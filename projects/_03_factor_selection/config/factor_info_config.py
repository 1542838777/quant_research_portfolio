
# =================================================================
# 步骤1：将填充规则定义为配置字典，清晰且易于扩展
# =================================================================
# 定义填充策略常量
FILL_STRATEGY_FFILL = 'ffill'       # 前向填充
FILL_STRATEGY_FFILL_LIMIT2 = 'ffill(limit=2)'       # 前向填充
FILL_STRATEGY_ZERO = 'zero_fill'    # 填充为0
FILL_STRATEGY_NONE = 'no_fill'      # 不进行任何填充
# 定义每个数据字段对应的策略
# 当未来有新因子时，你只需要在这里加一行，而不用修改函数代码
FACTOR_FILL_CONFIG = {
    # 价格类
    'close': FILL_STRATEGY_FFILL,
    'open': FILL_STRATEGY_FFILL,
    'high': FILL_STRATEGY_FFILL,
    'low': FILL_STRATEGY_FFILL,
    'pre_close': FILL_STRATEGY_FFILL,
    # 缓变类
    'pe_ttm': FILL_STRATEGY_FFILL_LIMIT2,
    'pb': FILL_STRATEGY_FFILL_LIMIT2,
    'total_mv': FILL_STRATEGY_FFILL_LIMIT2,
    'circ_mv': FILL_STRATEGY_FFILL_LIMIT2,
    # 高频交易类
    'turnover': FILL_STRATEGY_NONE,
    'volume': FILL_STRATEGY_NONE,
    'turnover_rate': FILL_STRATEGY_NONE,
    'pct_chg': FILL_STRATEGY_NONE,
    # 静态类
    'industry': FILL_STRATEGY_FFILL,
    'list_date': FILL_STRATEGY_FFILL,
    # 计算技术类
    'pe_ttm_inv': FILL_STRATEGY_NONE,
    'bm_ratio': FILL_STRATEGY_NONE,
    'turnover_rate_abnormal_20d': FILL_STRATEGY_NONE,
    'market_cap_log': FILL_STRATEGY_NONE,
    'momentum_20d': FILL_STRATEGY_NONE,
    # ... 你可以继续添加更多因子
}
