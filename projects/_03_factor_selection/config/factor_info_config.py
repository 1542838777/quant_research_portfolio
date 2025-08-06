
# =================================================================
# 步骤1：将填充规则定义为配置字典，清晰且易于扩展
# =================================================================
# 定义填充策略常量
FILL_STRATEGY_FFILL = 'ffill'       # 前向填充
FILL_STRATEGY_FFILL_LIMIT2 = 'ffill(limit=2)'       # 前向填充
FILL_STRATEGY_ZERO = 'zero_fill'    # 填充为0
FILL_STRATEGY_NONE = 'no_fill'      # 不进行任何填充
FILL_STRATEGY_CONDITIONAL_ZERO = 'conditional_zero' # 条件填充0 #todo 有空在实现，不要紧，用于判断pct_chg 在当且仅当确认停牌的状态下，才可以fillna(0) .turnover_rate逻辑同理

# 定义每个数据字段对应的策略
# 当未来有新因子时，你只需要在这里加一行，而不用修改函数代码
FACTOR_FILL_CONFIG = {
    # --------------------------------------------------------------------------
    #  一、基础数据层 (Base Data)
    # --------------------------------------------------------------------------

    # === 价格与估值类 (状态信息) ===
    # 在发现阶段，我们假设状态信息是长期有效的，用无限制ffill来保证数据连续性。
    # 这能避免因短期停牌导致因子值在统计中时有时无。
    'close': FILL_STRATEGY_FFILL,
    'open': FILL_STRATEGY_FFILL,
    'high': FILL_STRATEGY_FFILL,
    'low': FILL_STRATEGY_FFILL,
    'pre_close': FILL_STRATEGY_FFILL,
    'total_mv': FILL_STRATEGY_FFILL,
    'circ_mv': FILL_STRATEGY_FFILL,
    'pe_ttm': FILL_STRATEGY_FFILL,
    'pb': FILL_STRATEGY_FFILL,
    'ps_ttm': FILL_STRATEGY_FFILL,

    # === 交易行为类 (事件信息) ===
    # 在统计检验中，将缺失的事件(停牌)填充为0是常见做法。
    # 0代表“无交易”、“无换手”、“无涨跌”，这是一个中性的、合理的假设。
    'turnover_rate': FILL_STRATEGY_ZERO,
    'pct_chg': FILL_STRATEGY_ZERO,

    # === 静态信息类 (长效状态) ===
    'industry': FILL_STRATEGY_FFILL,
    'list_date': FILL_STRATEGY_FFILL,

    # --------------------------------------------------------------------------
    #  二、衍生因子层 (Derived Factors)
    # --------------------------------------------------------------------------
    # 在发现阶段，我们同样希望衍生因子的时间序列是连续的。
    # 对计算结果进行ffill，可以平滑掉因上游数据短期缺失导致的计算噪点。

    # === 规模 (Size) & 价值 (Value) ===
    # 这些是缓变量，使用ffill来确保其稳定性。
    'market_cap_log': FILL_STRATEGY_FFILL,
    'bm_ratio': FILL_STRATEGY_FFILL,
    'ep_ratio': FILL_STRATEGY_FFILL,
    'sp_ratio': FILL_STRATEGY_FFILL,
    'cfp_ratio': FILL_STRATEGY_FFILL,

    # === 质量 (Quality) & 成长 (Growth) ===
    # 财报类因子，季度更新，必须用ffill填充整个报告期。
    'roe_ttm': FILL_STRATEGY_FFILL,
    'gross_margin_ttm': FILL_STRATEGY_FFILL,
    'debt_to_assets': FILL_STRATEGY_FFILL,
    'net_profit_growth_yoy': FILL_STRATEGY_FFILL,
    'revenue_growth_yoy': FILL_STRATEGY_FFILL,

    # === 动量 (Momentum), 风险 (Risk), 流动性 (Liquidity) ===
    # 这些滚动计算的因子，其值在相邻两天变化不大，用ffill填充小的计算窗口断点是合理的。
    # 这样做可以大大提高因子的覆盖率，让IC序列和分组收益更稳定。
    'momentum_12_1': FILL_STRATEGY_FFILL,
    'momentum_20d': FILL_STRATEGY_FFILL,
    'beta': FILL_STRATEGY_FFILL,
    'volatility_120d': FILL_STRATEGY_FFILL,
    'turnover_rate_monthly_mean': FILL_STRATEGY_FFILL,
    'liquidity_amihud': FILL_STRATEGY_FFILL,

    # === 合成类因子 ===
    # 合成因子的填充策略应与它的成分因子保持一致。
    'value_composite': FILL_STRATEGY_FFILL,
}
