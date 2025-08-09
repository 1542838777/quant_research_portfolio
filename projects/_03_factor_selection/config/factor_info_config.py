##
#
#
#
# 原则一：基础数据层保持“纯净”
#
    # 所有最基础的数据，如close, open, pct_chg, turnover_rate，在被用于计算前，绝对不能进行ffill或zero_fill。它们必须保持最原始的状态，用NaN诚实地反映停牌、未上市、数据缺失等情况。
    #
    # 唯一的例外是基于交易状态对pct_chg和turnover_rate进行条件填充为0，但close价格本身绝不能动。
    #
# 原则二：计算时“尊重”缺失值
#
    # 您的_calculate_...函数，应该工作在这些“纯净”的基础数据之上。
    #
    # pandas的shift, rolling等函数在设计上就能很好地处理NaN。如果输入是NaN，它们的输出也自然是NaN。这是正确且期望的行为。
#
# 原则三：填充是最后一步，且服务于特定模型
    #
    # 因子计算完成后，您会得到一个含有“合法NaN”的纯净因子矩阵。
    #
    # 在get_backtest_ready_factor中，执行完.shift(1)后，再根据这个因子本身的特性，决定最后的填充策略。
    #
    # 对于动量这类变化较快的因子，通常不建议填充，或最多使用一个非常有限的ffill（如limit=1或2），以填补极个别的缺失点。
    #
    # 对于ROE这类季度更新的缓变因子，使用ffill才是合理的。#
# =================================================================
# 步骤1：将填充规则定义为配置字典，清晰且易于扩展
# =================================================================



#注：只能在最后一个阶段！！！！！再填充，原始数据 误去破坏
# 定义每个数据字段对应的策略
# 当未来有新因子时，你只需要在这里加一行，而不用修改函数代码
# 定义填充策略常量


# 定义填充策略常量
FILL_STRATEGY_NONE = 'no_fill'  # 不填充 (默认最安全)
FILL_STRATEGY_FFILL_UNLIMITED = 'ffill_unlimited'  # 无限制前向填充 (仅用于行业等静态数据)
FILL_STRATEGY_FFILL_LIMIT_65 = 'ffill_limit_65'  # 有限前向填充(约1个季度)，用于财报/估值类
FILL_STRATEGY_FFILL_LIMIT_5 = 'ffill_limit_5'  # 有限前向填充(约1周)，用于某些周度更新或短效因子
FILL_STRATEGY_CONDITIONAL_ZERO = 'conditional_zero'  # 条件填充0 (处理停牌的核心)

# =================================================================
#  最终填充配置字典 (用于 get_backtest_ready_factor)
# =================================================================
FACTOR_FILL_CONFIG = {
    # --------------------------------------------------------------------------
    #  一、基础数据层 (Base Data)
    # --------------------------------------------------------------------------
    # 保持绝对纯净，不在最终阶段填充
    'close': FILL_STRATEGY_NONE,
    'open': FILL_STRATEGY_NONE,
    'high': FILL_STRATEGY_NONE,
    'low': FILL_STRATEGY_NONE,
    'pre_close': FILL_STRATEGY_NONE,
    'pe_ttm': FILL_STRATEGY_NONE,  # pe_ttm等基础估值指标，其填充应在衍生因子层定义
    'pb': FILL_STRATEGY_NONE,
    'ps_ttm': FILL_STRATEGY_NONE,
    'circ_mv': FILL_STRATEGY_NONE,
    # 'total_mv': FILL_STRATEGY_NONE,

    # 交易行为类，依赖一个外部的交易状态flag来实现
    'turnover_rate': FILL_STRATEGY_CONDITIONAL_ZERO,
    'pct_chg': FILL_STRATEGY_CONDITIONAL_ZERO,

    # 静态信息类，信息永不或极少变化，使用无限制ffill是安全的
    'industry': FILL_STRATEGY_FFILL_UNLIMITED,
    'list_date': FILL_STRATEGY_FFILL_UNLIMITED,

    # --------------------------------------------------------------------------
    #  二、衍生因子层 (Derived Factors)
    # --------------------------------------------------------------------------
    # === 规模 (Size) & 价值 (Value) & 质量 (Quality) & 成长 (Growth) ===
    # 【核心修正】这些因子都强依赖于季度财报。其信息有效期为一个季度。
    # 我们设定一个比季度略长的limit（约65个交易日），作为风险控制的硬上限。
    # 这能有效防止您担心的“停牌两年”问题。
    'market_cap_log': FILL_STRATEGY_FFILL_LIMIT_65,
    'bm_ratio': FILL_STRATEGY_FFILL_LIMIT_65,
    'ep_ratio': FILL_STRATEGY_FFILL_LIMIT_65,
    'sp_ratio': FILL_STRATEGY_FFILL_LIMIT_65,
    'cfp_ratio': FILL_STRATEGY_FFILL_LIMIT_65,
    'roe_ttm': FILL_STRATEGY_FFILL_LIMIT_65,
    'small_cap': FILL_STRATEGY_FFILL_LIMIT_65,
    'gross_margin_ttm': FILL_STRATEGY_FFILL_LIMIT_65,
    'debt_to_assets': FILL_STRATEGY_FFILL_LIMIT_65,
    'net_profit_growth_yoy': FILL_STRATEGY_FFILL_LIMIT_65,
    'total_revenue_growth_yoy': FILL_STRATEGY_FFILL_LIMIT_65,
    'pct_chg_beta': FILL_STRATEGY_NONE,#todo

    # === 动量(Momentum), 风险(Risk), 流动性(Liquidity) ===
    # 【保持正确】维持我们之前的结论，这类短半衰期因子，不应填充。
    # 任何填充都会引入错误的陈旧信息。
    'momentum_12_1': FILL_STRATEGY_NONE,
    'momentum_20d': FILL_STRATEGY_NONE,
    'beta': FILL_STRATEGY_NONE,
    'volatility_120d': FILL_STRATEGY_NONE,
    'turnover_rate_monthly_mean': FILL_STRATEGY_NONE,
    'liquidity_amihud': FILL_STRATEGY_NONE,

    # === 合成类因子 ===
    # 【保持正确】合成因子不应有自己的填充逻辑。
    'value_composite': FILL_STRATEGY_NONE,
}