# 常用指数代码配置
INDEX_CODES = {
    "HS300": "000300.SH",     # 沪深300
    "ZZ500": "000905.SH",     # 中证500
    "ZZ800": "000906.SH",     # 中证800
    "ZZ1000": "000852.SH",    # 中证1000
    "ALL_A": "ALL_A"          # 所有A股（自定义标识）
}
# 定义了每个因子派系默认应该被哪些风险因子中性化
# 这是你的“风险模型”的核心
FACTOR_STYLE_RISK_MODEL = {
    'value': ['market_cap', 'industry'],
    'quality': ['market_cap', 'industry'],
    'growth': ['market_cap', 'industry'],
    'momentum': ['market_cap', 'industry', 'pct_chg_beta'],
    'reversal': ['market_cap', 'industry', 'pct_chg_beta'],
    'risk': ['market_cap', 'industry'],  # 测试风险因子时，默认只对其他基础风险因子中性化
    'sentiment': ['market_cap', 'industry', 'pct_chg_beta'],
    'technical': ['market_cap', 'industry', 'pct_chg_beta'],
    # 默认配置，适用于未明确分类的因子
    'default': ['market_cap', 'industry']
}