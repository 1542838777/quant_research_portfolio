import pandas as pd
import numpy as np
from typing import Dict, Any

from data.local_data_load import load_index_daily, get_trading_dates, load_daily_hfq
from quant_lib import logger


class FactorCalculator:
    """
    【新增】因子计算器 (Factor Calculator)
    这是一个专门负责具体因子计算逻辑的类。
    它将所有的计算细节从 FactorManager 中分离出来，使得代码更清晰、更易于扩展。
    只做纯粹的计算，shift 以及对齐where股票池，下游自己处理！！！ remind
    """

    def __init__(self, factor_manager):
        """
        初始化因子计算器。

        Args:
            factor_manager: FactorManager 的实例。计算器需要通过它来获取依赖的因子。
        """
        # 注意：这里持有 FactorManager 的引用，以便在计算衍生因子时，
        # 可以通过 factor_manager.get_factor() 来获取基础因子，并利用其缓存机制。
        self.factor_manager = factor_manager
        print("FactorCalculator (因子计算器) 已准备就绪。")

    def _calculate_cashflow_ttm(self) -> pd.DataFrame:
        """
        【生产级】计算滚动12个月的经营活动现金流净额 (TTM)。
        """
        # --- 步骤〇：从DataManager获取原材料 ---
        try:
            # 通过 factor_manager 间接访问 data_manager
            cashflow_df = self.factor_manager.data_manager.get_raw_data('cashflow')
        except KeyError:
            raise ValueError("错误: DataManager中缺少 'cashflow' 原始数据。")

        print("    > 正在计算 cashflow_ttm...")

        # --- 步骤一：获取并处理单季度现金流 ---
        cashflow_df = cashflow_df.sort_values(by=['ts_code', 'end_date'])
        cashflow_df['n_cashflow_act_single_q'] = cashflow_df.groupby('ts_code')['n_cashflow_act'].diff()
        is_q1 = pd.to_datetime(cashflow_df['end_date']).dt.month == 3
        cashflow_df.loc[is_q1, 'n_cashflow_act_single_q'] = cashflow_df.loc[is_q1, 'n_cashflow_act']

        # --- 步骤二：计算滚动TTM值 ---
        cashflow_df['cashflow_ttm'] = cashflow_df.groupby('ts_code')['n_cashflow_act_single_q'].rolling(
            window=4, min_periods=4
        ).sum().reset_index(level=0, drop=True)

        # --- 步骤三：构建以公告日为索引的TTM宽表 ---
        ttm_df = cashflow_df[['ts_code', 'ann_date', 'cashflow_ttm']].dropna()
        ttm_df['ann_date'] = pd.to_datetime(ttm_df['ann_date'])

        cashflow_ttm_wide = ttm_df.pivot(index='ann_date', columns='ts_code', values='cashflow_ttm')

        # --- 步骤四：将TTM数据填充到每个交易日 ---
        all_trading_dates = self.factor_manager.data_manager.get_trading_dates()
        cashflow_ttm_daily = cashflow_ttm_wide.reindex(all_trading_dates).ffill()

        return cashflow_ttm_daily

    def _calculate_cfp_ratio(self) -> pd.DataFrame:
        """
        计算现金流市值比 (cfp_ratio = cashflow_ttm / total_mv)
        """
        print("    > 正在计算 cfp_ratio...")
        # --- 步骤一：获取依赖的因子 ---
        cashflow_ttm_df = self.factor_manager.get_factor('cashflow_ttm')
        total_mv_df = self.factor_manager.get_factor('total_mv')

        # --- 步骤二：对齐并计算 ---
        common_stocks = total_mv_df.columns.intersection(cashflow_ttm_df.columns)
        common_dates = total_mv_df.index.intersection(cashflow_ttm_df.index)

        cfp_ratio_df = cashflow_ttm_df.loc[common_dates, common_stocks] / total_mv_df.loc[common_dates, common_stocks]

        return cfp_ratio_df

    def _calculate_bm_ratio(self) -> pd.DataFrame:
        """
        【新增示例】计算市净率倒数 (bm_ratio = 1 / pb)
        """
        print("    > 正在计算 bm_ratio...")
        pb_df = self.factor_manager.get_factor('pb')  # pb是原始因子
        pb_df_positive = pb_df.where(pb_df > 0)  # 过滤掉负值和0
        bm_ratio_df = 1 / pb_df_positive
        return bm_ratio_df

    def _calculate_beta(self) -> pd.DataFrame:
        beta_df = calculate_rolling_beta(
            self.factor_manager.data_manager.config['backtest']['start_date'],
            self.factor_manager.data_manager.config['backtest']['end_date'],
            self.factor_manager.get_pool_of_factor_name_of_stock_codes('beta')
        )
        return beta_df * -1

    def _calculate_pe_ttm_inv(self) -> pd.DataFrame:
        pe_ttm_raw_df = self.factor_manager.get_factor('pe_ttm').copy()
        # PE为负或0时，其倒数无意义，设为NaN
        pe_ttm_raw_df = pe_ttm_raw_df.where(pe_ttm_raw_df > 0)
        return 1 / pe_ttm_raw_df

    def _calculate_market_cap_log_by_circ_mv(self) -> pd.DataFrame:
        circ_mv_df = self.factor_manager.get_factor('circ_mv').copy()
        # 保证为正数，避免log报错
        circ_mv_df = circ_mv_df.where(circ_mv_df > 0)
        # 使用 pandas 自带 log 函数，保持类型一致
        factor_df = circ_mv_df.apply(np.log)
        # 反向处理因子（仅为了视觉更好看）
        return factor_df * -1


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
    # 关键特征: 计算今天的值，需要过去N天连续、干净的历史数据。(所以给他提前buffer)它的计算过程本身就是一个“时间序列”操作。
    buffer_days = int(window * 1.7) + 5
    buffer_start_date = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y%m%d')
    # 1. Load the long-form DataFrame
    stock_data_long = load_daily_hfq(buffer_start_date, end_date, cur_stock_codes)

    # 2. It's better to modify the column before pivoting
    stock_data_long['pct_chg'] = stock_data_long['pct_chg'] / 100

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
        pct_chg=lambda x: x['pct_chg'] / 100)  # pct_chg = ...: 这指定了要创建或修改的列的名称 x：当前DataFrame
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
    trading_index = pd.to_datetime(get_trading_dates(start_date, end_date))  # 确保是DatetimeIndex

    # b) 使用 reindex 将 beta 矩阵对齐到标准交易日历上
    # 缺失的日期（如初始窗口期）会自动用 NaN 填充
    final_beta_df = beta_df_in_range.reindex(trading_index)
    logger.info(f"滚动Beta计算完成，最终矩阵形状: {final_beta_df.shape}")

    return final_beta_df
# --- 如何在你的主流程中使用 (用法完全不变！) ---
# from data_manager import DataManager

# # 1. 初始化数据仓库
# dm = DataManager(...)
# dm.prepare_all_data()

# # 2. 初始化因子引擎 (内部已自动创建计算器)
# fm = FactorManager(data_manager=dm)

# # 3. 获取你需要的因子
# # 第一次获取 bm_ratio 时，FactorManager 会委托 Calculator 去计算
# bm_factor = fm.get_factor('bm_ratio')

# # 第二次获取时，它会直接从 FactorManager 的缓存加载，速度极快
# bm_factor_again = fm.get_factor('bm_ratio')

# print("\n最终得到的 bm_ratio 因子:")
# print(bm_factor.head())
