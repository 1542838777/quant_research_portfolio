import pandas as pd

from data.local_data_load import load_daily_hfq, load_index_daily, get_trading_dates
from quant_lib import logger


def calculate_rolling_beta(
        start_date: str,
        end_date: str,
        cur_stock_codes: list,
        window: int = 60,
        min_periods: int = 20
) -> pd.DataFrame:
    """
    【最终健壮版】计算A股市场上每只股票相对于市场指数的滚动Beta值。
    此版本修复了数据对齐的隐患。

    Args:
        start_date (str): 回测开始日期, 格式 'YYYYMMDD'
        end_date (str): 回测结束日期, 格式 'YYYYMMDD'
        stock_returns (pd.DataFrame): 股票收益率宽表, index为datetime, values已处理为小数。
        window (int): 滚动窗口大小（天数）。
        min_periods (int): 窗口内计算所需的最小观测数。

    Returns:
        pd.DataFrame: 滚动Beta矩阵 (index=date, columns=stock)。
    """
    logger.info(f"开始计算滚动Beta (窗口: {window}天)...")

    # --- 1. 数据获取与准备 ---
    #  指数提前。但是入参传入的股票是死的，建议重新手动加载。但是考虑是否与股票池对应！ 答案：还是别跟动态股票池进行where了，疑问
    # 解释：
    # 为了计算滚动值，我们需要往前多取一些数据作为“缓冲”
    ##
    # 滚动历史因子 (Rolling History Factor)
    # 例子: pct_chg_beta, 动量因子 (Momentum), 滚动波动率 (Volatility)。
    #
    # 关键特征: 计算今天的值，需要过去N天连续、干净的历史数据。(所以给他提前buffer) 它的计算过程本身就是一个“时间序列”操作。
    buffer_days = int(window * 1.7) + 5
    buffer_start_date = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y%m%d')
    # 1. Load the long-form DataFrame
    stock_data_long = load_daily_hfq(buffer_start_date, end_date, cur_stock_codes)

    # 2. It's better to modify the column before pivoting
    stock_data_long['pct_chg'] = stock_data_long['pct_chg'] / 100 #  check一下是否需要/100 需要

    # 3. Correctly pivot the DataFrame to wide format
    # The 'columns' argument should be the name of the column containing the stock codes.
    stock_returns = pd.pivot_table(
        stock_data_long,
        index='trade_date',
        columns='ts_code',  # Use the column name 'ts_code'
        values='pct_chg'
    )

    # a) 获取市场指数的每日收益率 是否是自动过滤了 非交易日 yes
    market_returns_long = load_index_daily(buffer_start_date, end_date).assign(
        pct_chg=lambda x: x['pct_chg'] / 100)  # pct_chg = ...: 这指定了要创建或修改的列的名称 x：当前DataFrame   check一下是否需要/100 需要
    market_returns = market_returns_long.set_index('trade_date')['pct_chg']
    market_returns.index = pd.to_datetime(market_returns.index)
    market_returns.name = 'market_return'  # chong'ming

    # --- 2. 【核心修正】显式数据对齐 ---
    # logger.info("  > 正在进行数据显式对齐...")
    # 使用 'left' join，以 stock_returns 的日期为基准
    # 这会创建一个统一的时间轴，并将市场收益精确地匹配到每个交易日
    combined_df = stock_returns.join(market_returns, how='left')

    # 更新 market_returns 为对齐后的版本，确保万无一失
    market_returns_aligned = combined_df.pop('market_return')  # 剔除这列！

    # --- 3. 滚动计算Beta ---
    # logger.info("  > 正在进行滚动计算...")
    # Beta = Cov(R_stock, R_market) / Var(R_market)

    # a) 现在，stock_returns 和 market_returns_aligned 的索引是100%对齐的
    rolling_cov = combined_df.rolling(window=window, min_periods=min_periods).cov(
        market_returns_aligned)  # 协方差关心的是两组数据之间的关系（描述两个变量之间的关系方向）（是不是都是一起）

    # b) 计算指数收益率的滚动方差
    rolling_var = market_returns_aligned.rolling(window=window, min_periods=min_periods).var()

    # c) 计算滚动Beta
    beta_df = rolling_cov.div(rolling_var, axis=0)

    # d) 截取我们需要的最终日期范围
    beta_df_in_range = beta_df.loc[start_date:end_date]

    # --- 4. 【核心修正】使用reindex确保最终索引是完整的交易日历 ---
    # a) 获取目标日期范围内的标准交易日历
    trading_index = pd.to_datetime( get_trading_dates(start_date, end_date))  # 确保是DatetimeIndex

    # b) 使用 reindex 将 beta 矩阵对齐到标准交易日历上
    # 缺失的日期（如初始窗口期）会自动用 NaN 填充
    final_beta_df = beta_df_in_range.reindex(trading_index)
    logger.info(f"滚动Beta计算完成，最终矩阵形状: {final_beta_df.shape}")

    return final_beta_df