import pandas as pd
import numpy as np
from typing import Dict, Any

from data.local_data_load import load_index_daily, get_trading_dates, load_daily_hfq, load_cashflow_df
from quant_lib import logger


class FactorCalculator:
    """
    【新增】因子计算器 (Factor Calculator)
    这是一个专门负责具体因子计算逻辑的类。
    它将所有的计算细节从 FactorManager 中分离出来，使得代码更清晰、更易于扩展。
    只做纯粹的计算，shift 以及对齐where股票池，下游自己处理！！！ remind
    """

    ##
    #
    # caclulate_函数 ，无需关心对齐，反正下游会align where 动态股票池！
    #  注意：涉及到rolling shift 操作，需要关注数据连续性。要考虑填充！
    #   ---ex：close可能是空的，因为停牌日就是空的，是nan，我们可以适当ffill
    #
    # #

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

    def _calculate_market_cap_log_by_circ_mv(self) -> pd.DataFrame:
        circ_mv_df = self.factor_manager.get_factor('circ_mv').copy()
        # 保证为正数，避免log报错
        circ_mv_df = circ_mv_df.where(circ_mv_df > 0)
        # 使用 pandas 自带 log 函数，保持类型一致
        factor_df = circ_mv_df.apply(np.log)
        # 反向处理因子（仅为了视觉更好看）
        return factor_df * -1

    # === 规模 (Size) ===
    def _calculate_market_cap_log(self) -> pd.DataFrame:
        """
        计算对数总市值。

        金融逻辑:
        市值的原始分布是严重右偏的（少数巨头公司市值极大），直接使用会受到极端值的影响。
        取对数可以使数据分布更接近正态分布，降低极端值的影响，是处理规模因子的标准做法。
        """
        print("    > 正在计算因子: market_cap_log...")
        # 1. 从FactorManager获取原始市值因子
        total_mv_df = self.factor_manager.get_factor('total_mv')
        # 2. 对所有数值应用自然对数。np.log会自动处理整个DataFrame。
        #    对于任何非正数，np.log会返回-inf或nan，后续处理中会被当做无效值。
        log_mv_df = np.log(total_mv_df)
        return log_mv_df

    # === 价值 (Value) ===
    def _calculate_bm_ratio(self) -> pd.DataFrame:
        """
        计算账面市值比 (Book-to-Market Ratio)，即市净率(PB)的倒数。

        金融逻辑:
        这是Fama-French三因子模型中的核心价值衡量指标。高BM比率意味着公司的账面价值
        相对于其市场价格更高，可能被市场低估。
        """
        print("    > 正在计算因子: bm_ratio...")
        pb_df = self.factor_manager.get_factor('pb')
        pb_df_positive = pb_df.where(pb_df > 0)
        bm_ratio_df = 1 / pb_df_positive
        return bm_ratio_df

    def _calculate_ep_ratio(self) -> pd.DataFrame:
        """
        计算盈利收益率 (Earnings Yield)，即市盈率(PE_TTM)的倒数。

        金融逻辑:
        衡量投资者每投入一元市值，可以获得多少公司盈利。它比PE更能直观地
        与债券收益率等其他资产回报率进行比较。
        """
        print("    > 正在计算因子: ep_ratio...")
        pe_ttm_df = self.factor_manager.get_factor('pe_ttm')
        pe_df_positive = pe_ttm_df.where(pe_ttm_df > 0)
        ep_ratio_df = 1 / pe_df_positive
        return ep_ratio_df

    def _calculate_sp_ratio(self) -> pd.DataFrame:
        """
        计算销售收益率 (Sales Yield)，即市销率(PS_TTM)的倒数。

        金融逻辑:
        衡量投资者每投入一元市值，可以获得多少销售收入。这个指标对于那些
        处于快速扩张期但尚未盈利的成长型公司（PE为负）尤其有价值。
        """
        print("    > 正在计算因子: sp_ratio...")
        ps_ttm_df = self.factor_manager.get_factor('ps_ttm')
        ps_df_positive = ps_ttm_df.where(ps_ttm_df > 0)
        sp_ratio_df = 1 / ps_df_positive
        return sp_ratio_df

        # 主要字段：n_cashflow_act：经营活动产生的现金流量净额 进行滚动平均
        # 代码大篇幅主要处理脏数据！多来自于ipo，因为股票未上市前，用的不准确的数据！

    def _calculate_cashflow_ttm(self) -> pd.DataFrame:
        """
           【】计算滚动12个月的经营活动现金流净额 (TTM)。

           输入:
           - cashflow_df: 原始现金流量表数据，包含['ann_date', 'ts_code', 'end_date', 'n_cashflow_act']
           - all_trading_dates: 一个包含所有交易日日期的pd.DatetimeIndex，用于构建最终的日度因子矩阵。

           输出:
           - 一个以交易日为索引(index)，股票代码为列(columns)的日度TTM因子矩阵。
           """
        print("--- 开始执行生产级TTM因子计算 ---")
        cashflow_df = load_cashflow_df()

        # === 步骤一：创建完美的季度时间标尺 ===
        print("2. 创建时间标尺以处理数据断点...")
        scaffold_min_max_end_date_df = cashflow_df.groupby('ts_code')['end_date'].agg(['min', 'max'])

        full_date_dfs = []
        for ts_code, row in scaffold_min_max_end_date_df.iterrows():
            date_range = pd.date_range(start=row['min'], end=row['max'], freq='Q-DEC')  # 所有的报告期日（季度最后一日
            full_date_dfs.append(pd.DataFrame({'ts_code': ts_code, 'end_date': date_range}))

        full_dates_df = pd.concat(full_date_dfs)

        # === 步骤二：数据对齐 ===
        print("3. 将原始数据合并到时间标尺上...")
        merged_df = pd.merge(full_dates_df, cashflow_df, on=['ts_code', 'end_date'], how='left')

        # === 步骤三：安全地计算单季度值 ===
        print("4. 安全计算单季度现金流...")
        # 使用ffill填充缺失的累计值，这对于处理IPO前只有年报的情况至关重要
        merged_df['n_cashflow_act_filled'] = merged_df.groupby('ts_code')['n_cashflow_act'].ffill()
        merged_df['n_cashflow_act_single_q'] = merged_df.groupby('ts_code')['n_cashflow_act_filled'].diff()

        is_q1 = merged_df['end_date'].dt.month == 3
        # Q1的值必须用原始值覆盖，且只在原始值存在(notna)时操作
        merged_df.loc[is_q1 & merged_df['n_cashflow_act'].notna(), 'n_cashflow_act_single_q'] = merged_df.loc[
            is_q1, 'n_cashflow_act']

        # === 步骤四：安全地计算TTM ===
        print("5. 安全计算滚动TTM值...")
        merged_df['cashflow_ttm'] = merged_df.groupby('ts_code')['n_cashflow_act_single_q'].rolling(
            window=4, min_periods=4
        ).sum().reset_index(level=0, drop=True)

        # ================================================================= #
        # --- 以下是您之前省略，但对于构建因子库至关重要的部分 ---
        # ================================================================= #

        # === 步骤五：构建以公告日为索引的TTM长表 ===
        print("6. 整理并过滤有效TTM值...")
        # 关键：只有当'ann_date'和'cashflow_ttm'同时存在时，这个数据点才是一个有效的、可用于交易的“事件”
        ttm_long_df = merged_df[['ts_code', 'ann_date', 'end_date', 'cashflow_ttm']].dropna()
        if ttm_long_df.empty:
            raise ValueError("警告: _calculate_cashflow_ttm 计算后没有产生任何有效的TTM数据点。")

        # === 步骤六：透视 (Pivot) ===
        print("7. 执行透视操作(Pivot)，将长表转换为宽表...")
        # 我们需要保留end_date用于后续的排序判断，确保pivot_table的'last'能取到正确的值
        ttm_long_df = ttm_long_df.sort_values(by=['ts_code', 'end_date'])
        # 目标：将“事件驱动”的数据(一行代表一次财报公布)转换为“时间序列”的宽表
        # 索引(index)是事件发生的日期(ann_date)，列(columns)是股票代码
        cashflow_ttm_wide = ttm_long_df.pivot_table(
            index='ann_date',
            columns='ts_code',
            values='cashflow_ttm',
            aggfunc='last'  # 使用'last'，因为数据已按end_date排序，'last'会选取最新的财报
        )

        # === 步骤七：重索引 (Reindex) & 前向填充 (Forward Fill) ===
        print("8. 对齐到交易日历并进行前向填充(ffill)...")
        # Reindex: 将稀疏的公告日数据，扩展到全部交易日上。非公告日的TTM值此时为NaN
        # ffill: 用最近一次已知的TTM值，填充未来的交易日。
        #      这完美模拟了真实情况：一个财报的效力会持续，直到下一个新财报出来为止。
        cashflow_ttm_daily = cashflow_ttm_wide.reindex(self.factor_manager.data_manager.trading_dates).ffill()

        print("--- 因子计算完成 ---")
        return cashflow_ttm_daily


    def _calculate_cfp_ratio(self) -> pd.DataFrame:
        """
            计算现金流市值比 (cfp_ratio = cashflow_ttm / total_mv)
            包含风险控制和健壮性处理。
            """
        print("    > 正在计算 cfp_ratio...")
        # --- 步骤一：获取依赖的因子 ---
        cashflow_ttm_df = self.factor_manager.get_factor('cashflow_ttm')
        total_mv_df = self.factor_manager.get_factor('total_mv')

        # --- 步骤二：对齐数据 (使用 .align) ---
        mv_aligned, ttm_aligned = total_mv_df.align(cashflow_ttm_df, join='inner', axis=None)

        # --- 步骤三：风险控制与预处理 (核心) ---
        # 1. 过滤小市值公司：这是不可逾越的纪律 （市值小，表示分母小，更容易被操控，比如现金突然多了10w，但是市值为1，那不是猛增10w倍率
        #    在实盘中，这个阈值甚至可能是20亿(2e9)或30亿(3e9)
        mv_aligned[mv_aligned < 1e8] = np.nan

        # 2. 过滤掉退市或长期停牌等市值为0或负的异常情况 ！
        mv_aligned[mv_aligned <= 0] = np.nan

        # --- 步骤四：计算因子 ---
        cfp_ratio_df = ttm_aligned / mv_aligned

        # --- 步骤五：后处理，清除计算过程中产生的inf ---
        # 在除法运算后，对无穷大值进行处理，统一替换为NaN
        cfp_ratio_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        print("    > cfp_ratio 计算完成，已包含风险控制。")
        return cfp_ratio_df

    # === 质量 (Quality) ===
    def _calculate_roe_ttm(self) -> pd.DataFrame:
        """
        计算滚动12个月的净资产收益率 (ROE_TTM)。

        金融逻辑:
        ROE是衡量公司为股东创造价值效率的核心指标。高ROE意味着公司能用更少的
        股东资本创造出更多的利润，是“好生意”的标志。

        注意: 这是一个依赖财报数据的复杂因子，其计算逻辑与 cashflow_ttm 类似。
              你需要确保你的 DataManager 能够提供包含 'net_profit' 和 'total_equity'
              的季度财务报表数据。
        """
        print("    > 正在计算因子: roe_ttm...")
        # 此处为占位符逻辑，你需要替换为与 _calculate_cashflow_ttm 类似的
        # 从原始财报计算单季 -> 滚动求和TTM -> 按公告日对齐 的完整流程。
        # 依赖的原始字段: net_profit (净利润), total_equity (股东权益)
        print("      > [警告] _calculate_roe_ttm 使用的是占位符实现！")
        total_mv_df = self.factor_manager.get_factor('total_mv')
        return pd.DataFrame(np.random.randn(*total_mv_df.shape), index=total_mv_df.index, columns=total_mv_df.columns)

    def _calculate_gross_margin_ttm(self) -> pd.DataFrame:
        """
        计算滚动12个月的销售毛利率。

        金融逻辑:
        毛利率反映了公司产品的定价能力和成本控制能力，是公司“护城河”的体现。
        持续的高毛利率通常意味着强大的品牌或技术优势。
        """
        print("    > 正在计算因子: gross_margin_ttm...")
        # 同样，这是一个需要从财报数据计算的复杂因子。
        # 依赖的原始字段: revenue (营业收入), op_cost (营业成本)
        print("      > [警告] _calculate_gross_margin_ttm 使用的是占位符实现！")
        total_mv_df = self.factor_manager.get_factor('total_mv')
        return pd.DataFrame(np.random.randn(*total_mv_df.shape), index=total_mv_df.index, columns=total_mv_df.columns)

    def _calculate_debt_to_assets(self) -> pd.DataFrame:
        """
        计算资产负债率。

        金融逻辑:
        衡量公司的财务杠杆水平。适度的杠杆可以提高股东回报，但过高的杠杆
        则意味着巨大的财务风险。这是一个衡量公司稳健性的重要指标。
        """
        print("    > 正在计算因子: debt_to_assets...")
        # 这是一个可以直接从最新财报获取的“时点”指标，不需要计算TTM。
        # 但同样需要处理财报发布延迟。
        # 依赖的原始字段: total_debt (总负债), total_assets (总资产)
        print("      > [警告] _calculate_debt_to_assets 使用的是占位符实现！")
        total_mv_df = self.factor_manager.get_factor('total_mv')
        return pd.DataFrame(np.random.randn(*total_mv_df.shape), index=total_mv_df.index, columns=total_mv_df.columns)

    # === 成长 (Growth) ===
    def _calculate_net_profit_growth_yoy(self) -> pd.DataFrame:
        """
        计算净利润同比增长率 (Year-over-Year)。

        金融逻辑:
        衡量公司盈利能力的增长速度，是成长性的核心体现。
        """
        print("    > 正在计算因子: net_profit_growth_yoy...")
        # 需要获取当季的单季净利润，和去年同期的单季净利润进行比较。
        # 这是一个非常复杂的计算，涉及到财报数据的滞后和对齐。
        # 依赖的原始字段: net_profit_single_q
        print("      > [警告] _calculate_net_profit_growth_yoy 使用的是占位符实现！")
        total_mv_df = self.factor_manager.get_factor('total_mv')
        return pd.DataFrame(np.random.randn(*total_mv_df.shape), index=total_mv_df.index, columns=total_mv_df.columns)

    def _calculate_revenue_growth_yoy(self) -> pd.DataFrame:
        """
        计算营业收入同比增长率 (Year-over-Year)。

        金融逻辑:
        衡量公司市场规模和业务扩张的速度。营收增长通常是利润增长的先行指标。
        """
        print("    > 正在计算因子: revenue_growth_yoy...")
        # 逻辑与净利润同比增长类似。
        # 依赖的原始字段: revenue_single_q
        print("      > [警告] _calculate_revenue_growth_yoy 使用的是占位符实现！")
        total_mv_df = self.factor_manager.get_factor('total_mv')
        return pd.DataFrame(np.random.randn(*total_mv_df.shape), index=total_mv_df.index, columns=total_mv_df.columns)

    # === 动量 (Momentum) ===
    def _calculate_momentum_12_1(self) -> pd.DataFrame:
        """
        计算过去12个月剔除最近1个月的累计收益率 (Momentum 12-1)。

        金融逻辑:
        这是最经典的动量因子，由Jegadeesh和Titman提出。它剔除了最近一个月的
        短期反转效应，旨在捕捉更稳健的中期价格惯性。
        """
        print("    > 正在计算因子: momentum_12_1...")
        # 1. 获取收盘价
        close_df = self.factor_manager.get_factor('close').copy(deep=True)
        # close_df.ffill(axis=0, inplace=True) #反驳：如果人家停牌一年，你非fill前一年的数据，那误差太大了 不行！
        # 2. 计算 T-21 (约1个月前) 的价格 与 T-252 (约1年前) 的价格之间的收益率
        #    shift(21) 获取的是约1个月前的价格
        #    shift(252) 获取的是约12个月前的价格
        momentum_df = close_df.shift(21) / close_df.shift(252) - 1
        return momentum_df

    def _calculate_momentum_20d(self) -> pd.DataFrame:
        """
        计算20日动量/收益率。

        金融逻辑:
        捕捉短期（约一个月）的价格惯性，即所谓的“强者恒强”。
        """
        print("    > 正在计算因子: momentum_20d...")
        close_df = self.factor_manager.get_factor('close').copy(deep=True)
        # close_df.reset_index(trading_index = self.factor_manager.data_manager.trading_dates() 不需要，raw_dfs生成的时候 就已经是trading_index了
        # close_df.ffill(axis=0, inplace=True)
        momentum_df = close_df.pct_change(periods=20)
        return momentum_df

    # === 风险 (Risk) ===
    def _calculate_beta(self) -> pd.DataFrame:
        beta_df = calculate_rolling_beta(
            self.factor_manager.data_manager.config['backtest']['start_date'],
            self.factor_manager.data_manager.config['backtest']['end_date'],
            self.factor_manager.get_pool_of_factor_name_of_stock_codes('beta')
        )
        return beta_df * -1

    def _calculate_volatility_120d(self) -> pd.DataFrame:
        """
        计算120日年化波动率。

        金融逻辑:
        衡量个股在过去约半年内的价格波动风险。经典的“低波动异象”认为，
        低波动率的股票长期来看反而有更高的风险调整后收益。
        """
        print("    > 正在计算因子: volatility_120d...")
        pct_chg_df = self.factor_manager.get_factor('pct_chg').copy(deep=True)
        # pct_chg_df.fillna(0, inplace=True)

        rolling_std_df = pct_chg_df.rolling(window=120, min_periods=60).std()
        annualized_vol_df = rolling_std_df * np.sqrt(252)
        return annualized_vol_df

    # === 流动性 (Liquidity) ===
    def _calculate_turnover_rate_monthly_mean(self) -> pd.DataFrame:
        """
        计算月平均换手率（21日滚动平均）。

        金融逻辑:
        衡量股票在近一个月的平均交易活跃度。过高或过低的换手率都可能包含特定信息。
        """
        print("    > 正在计算因子: turnover_rate_monthly_mean...")
        turnover_df = self.factor_manager.get_factor('turnover_rate').copy(deep=True)
        # turnover_df.fillna(0, inplace=True)

        # 使用21个交易日近似一个月
        monthly_mean_turnover_df = turnover_df.rolling(window=21, min_periods=15).mean()
        return monthly_mean_turnover_df

    def _calculate_liquidity_amihud(self) -> pd.DataFrame:
        """
        计算Amihud非流动性指标。

        金融逻辑:
        衡量单位成交额能引起多大的价格波动，公式为 abs(收益率) / 成交额。
        该值越大，说明股票的流动性越差，交易的冲击成本越高。
        """
        print("    > 正在计算因子: liquidity_amihud...")
        # 1. 获取依赖数据
        pct_chg_df = self.factor_manager.get_factor('pct_chg')
        # 假设你的DataManager可以提供以“元”为单位的日成交额'amount'
        amount_df = self.factor_manager.get_factor('amount')

        # 2. 【核心风险控制】: 将成交额为0的替换为一个极小值，防止除以0
        amount_df_safe = amount_df.where(amount_df > 0, 1e-9)

        # 3. 计算Amihud指标
        amihud_df = pct_chg_df.abs() / amount_df_safe
        return amihud_df


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
