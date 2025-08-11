from typing import Callable  # 引入Callable来指定函数类型的参数

import numpy as np
import pandas as pd

from data.local_data_load import load_index_daily, get_trading_dates, load_daily_hfq, load_cashflow_df, load_income_df, \
    load_balancesheet_df
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



    # === 规模 (Size) ===
    def _calculate_small_cap(self) -> pd.DataFrame:
        circ_mv_df = self.factor_manager.get_factor('circ_mv').copy()
        # 保证为正数，避免log报错
        circ_mv_df = circ_mv_df.where(circ_mv_df > 0)
        # 使用 pandas 自带 log 函数，保持类型一致
        factor_df = circ_mv_df.apply(np.log)
        return factor_df

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

    def _calculate_cfp_ratio(self) -> pd.DataFrame:
        """
            计算现金流市值比 (cfp_ratio = cashflow_ttm / total_mv)
            包含风险控制和健壮性处理。
            """
        print("    > 正在计算 cfp_ratio...")
        # --- 步骤一：获取依赖的因子 ---
        cashflow_ttm_df = self.factor_manager.get_factor('cashflow_ttm')
        total_mv_df = self.factor_manager.get_factor('small_cap')

        # --- 步骤二：对齐数据 (使用 .align) ---
        mv_aligned, ttm_aligned = total_mv_df.align(cashflow_ttm_df, join='inner', axis=None)

        # --- 步骤三：风险控制与预处理 (核心) ---
        # 1. 过滤小市值公司：这是不可逾越的纪律 （市值小，表示分母小，更容易被操控，比如现金突然多了10w，但是市值为1，那不是猛增10w倍率
        #    在实盘中，这个阈值甚至可能是20亿(2e9)或30亿(3e9) total_mv单位 万
        mv_aligned[mv_aligned < 1e4] = np.nan

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

        # --- 步骤一：获取分子和分母 ---
        # 调用我们刚刚实现的两个生产级函数
        net_profit_ttm_df = self._calculate_net_profit_ttm()
        total_equity_df = self._calculate_total_equity()

        # --- 步骤二：对齐数据 ---
        # align确保两个DataFrame的索引和列完全一致，避免错位计算
        # join='inner'会取两个因子都存在的股票和日期，是最安全的方式
        profit_aligned, equity_aligned = net_profit_ttm_df.align(total_equity_df, join='inner', axis=None)

        # --- 步骤三：风险控制与计算 ---
        # 核心风控：股东权益可能为负（公司处于资不抵债状态）。
        # 在这种情况下，ROE的计算没有经济意义，且会导致计算错误。
        # 我们将分母小于等于0的地方替换为NaN，这样除法结果也会是NaN。
        # 例如，2021年-2023年，一些陷入困境的地产公司净资产可能为负，其ROE必须被视为无效值。
        equity_aligned_safe = equity_aligned.where(equity_aligned > 0, np.nan)

        print("1. 计算 ROE TTM，并对分母进行风险控制(>0)...")
        roe_ttm_df = profit_aligned / equity_aligned_safe

        # --- 步骤四：后处理 ---
        # 尽管我们处理了分母为0的情况，但仍可能因浮点数问题产生无穷大值。
        # 统一替换为NaN，确保因子数据的干净。
        roe_ttm_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        print("--- 最终因子: roe_ttm 计算完成 ---")
        return roe_ttm_df


    def _calculate_gross_margin_ttm(self) -> pd.DataFrame:
        """
        【生产级】计算滚动12个月的销售毛利率 (Gross Margin TTM)。
        公式: (Revenue TTM - Operating Cost TTM) / Revenue TTM
        """
        print("--- 开始计算最终因子: gross_margin_ttm ---")

        # --- 步骤一：获取分子和分母的组成部分 ---
        revenue_ttm_df = self._calculate_total_revenue_ttm()
        op_cost_ttm_df = self._calculate_op_cost_ttm()

        # --- 步骤二：对齐数据 ---
        # 确保revenue和op_cost的索引和列完全一致，避免错位计算
        revenue_aligned, op_cost_aligned = revenue_ttm_df.align(op_cost_ttm_df, join='inner', axis=None)

        # --- 步骤三：风险控制与计算 ---
        # 核心风控：分母(营业收入)可能为0或负数(在极端或错误数据情况下)。
        # 我们将分母小于等于0的地方替换为NaN，这样除法结果也会是NaN，避免产生无穷大值。
        revenue_aligned_safe = revenue_aligned.where(revenue_aligned > 0, np.nan)

        print("1. 计算 Gross Margin TTM，并对分母进行风险控制(>0)...")
        gross_margin_ttm_df = (revenue_aligned - op_cost_aligned) / revenue_aligned_safe

        # --- 步骤四：后处理 (可选但推荐) ---
        # 理论上，毛利率不应超过100%或低于-100%太多，但极端情况可能出现。
        # 这里可以根据需要进行clip或winsorize，但暂时保持原样以观察原始分布。
        # 再次确保没有无穷大值。
        gross_margin_ttm_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        print("--- 最终因子: gross_margin_ttm 计算完成 ---")
        return gross_margin_ttm_df

    def _calculate_debt_to_assets(self) -> pd.DataFrame:
        """
        【生产级】计算每日可用的最新资产负债率。
        公式: Total Debt / Total Assets
        """
        print("--- 开始计算最终因子: debt_to_assets ---")

        # --- 步骤一：获取分子和分母 ---
        total_debt_df = self._calculate_total_debt()
        total_assets_df = self._calculate_total_assets()

        # --- 步骤二：对齐数据 ---
        debt_aligned, assets_aligned = total_debt_df.align(total_assets_df, join='inner', axis=None)

        # --- 步骤三：风险控制与计算 ---
        # 核心风控：分母(总资产)可能为0或负数。
        assets_aligned_safe = assets_aligned.where(assets_aligned > 0, np.nan)

        print("1. 计算 Debt to Assets，并对分母进行风险控制(>0)...")
        debt_to_assets_df = debt_aligned / assets_aligned_safe

        # --- 步骤四：后处理 ---
        debt_to_assets_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        print("--- 最终因子: debt_to_assets 计算完成 ---")
        return debt_to_assets_df
    # === 成长 (Growth) ===
    def _calculate_net_profit_growth_ttm(self) -> pd.DataFrame:
        """
        计算滚动12个月归母净利润的同比增长率(TTM YoY Growth)。
        """
        logger.info("    > 正在计算因子: net_profit_growth_ttm...")

        # 直接调用通用的TTM增长率计算引擎
        return self._calculate_financial_ttm_growth_factor(
            factor_name='net_profit_growth_ttm',
            ttm_factor_name='net_profit_ttm'  # 指定依赖的TTM因子名
        )

    def _calculate_revenue_growth_ttm(self) -> pd.DataFrame:
        """
        计算滚动12个月营业总收入的同比增长率(TTM YoY Growth)。
        """
        logger.info("    > 正在计算因子: revenue_growth_ttm...")

        # 同样调用通用的TTM增长率计算引擎
        return self._calculate_financial_ttm_growth_factor(
            factor_name='revenue_growth_ttm',
            ttm_factor_name='total_revenue_ttm'  # 指定依赖的TTM因子名
        )

    #ok
    def _calculate_net_profit_growth_yoy(self) -> pd.DataFrame:
        """
        【生产级】计算单季度归母净利润的同比增长率 (YoY)。
        """
        print("--- 开始计算最终因子: net_profit_growth_yoy ---")

        # --- 步骤一：获取基础数据 net_profit_single_q 的长表 ---
        net_profit_single_q_long = self._calculate_net_profit_single_q_long()

        # 确保数据按股票和报告期排序
        net_profit_single_q_long.sort_values(by=['ts_code', 'end_date'], inplace=True)

        # --- 步骤二：计算同比增长率 (YoY) ---
        # shift(4) 回溯去年同期
        net_profit_last_year_q = net_profit_single_q_long.groupby('ts_code')['net_profit_single_q'].shift(4)

        # 核心风控：去年同期净利润可能为0或负数，此时增长率无意义或失真。
        # 必须要求去年同期利润为正，才能计算有意义的增长率。
        net_profit_last_year_q_safe = net_profit_last_year_q.where(net_profit_last_year_q > 0, np.nan)

        # 计算同比增长率
        net_profit_single_q_long['net_profit_growth_yoy'] = \
            net_profit_single_q_long['net_profit_single_q'] / net_profit_last_year_q_safe - 1

        # --- 步骤三：将计算出的因子对齐到每日交易日历 ---
        # 1. 整理并过滤
        yoy_long_df = net_profit_single_q_long[['ts_code', 'ann_date', 'end_date', 'net_profit_growth_yoy']].copy()
        yoy_long_df.dropna(inplace=True)
        if yoy_long_df.empty:
            raise ValueError("警告: 计算 net_profit_growth_yoy 后没有产生任何有效的增长率数据点。")

        # 2. 透视 (Pivot)
        yoy_long_df.sort_values(by=['ts_code', 'end_date'], inplace=True)
        yoy_wide = yoy_long_df.pivot_table(
            index='ann_date',
            columns='ts_code',
            values='net_profit_growth_yoy',
            aggfunc='last'
        )

        trading_dates = self.factor_manager.data_manager.trading_dates
        yoy_daily = _broadcast_ann_date_to_daily(yoy_wide, trading_dates)

        print("--- 最终因子: net_profit_growth_yoy 计算完成 ---")
        return yoy_daily
    #ok
    def _calculate_total_revenue_growth_yoy(self) -> pd.DataFrame:
        """
        【生产级】计算单季度营业收入的同比增长率 (YoY)。
        """
        print("--- 开始计算最终因子: total_revenue_growth_yoy ---")

        # --- 步骤一：获取基础数据 total_revenue_single_q 的长表 ---
        # 调用我们的新引擎来获取每个公司每个季度的单季收入
        total_revenue_single_q_long = self._calculate_financial_single_q_factor(
            factor_name='total_revenue_single_q',
            data_loader_func=load_income_df,
            source_column='total_revenue'  # 确认使用总收入
        )

        # 确保数据按股票和报告期排序，这是 shift 操作准确无误的前提
        total_revenue_single_q_long.sort_values(by=['ts_code', 'end_date'], inplace=True)

        # --- 步骤二：计算同比增长率 (YoY) ---
        # shift(4) 在季度数据上，就是回溯4个季度，即去年同期
        revenue_last_year_q = total_revenue_single_q_long.groupby('ts_code')['total_revenue_single_q'].shift(4)

        # 核心风控：去年同期收入可能为0或负数，此时增长率无意义
        revenue_last_year_q_safe = revenue_last_year_q.where(revenue_last_year_q > 0, np.nan)

        # 计算同比增长率
        total_revenue_single_q_long['total_revenue_growth_yoy'] = \
            total_revenue_single_q_long['total_revenue_single_q'] / revenue_last_year_q_safe - 1

        # --- 步骤三：将计算出的因子对齐到每日交易日历 ---
        # 这里的逻辑和我们之前的引擎完全一样

        # 1. 整理并过滤有效的YoY值
        yoy_long_df = total_revenue_single_q_long[['ts_code', 'ann_date', 'end_date', 'total_revenue_growth_yoy']].copy()
        yoy_long_df.dropna(inplace=True)
        if yoy_long_df.empty:
            raise ValueError("警告: 计算 total_revenue_growth_yoy 后没有产生任何有效的增长率数据点。")

        # 2. 透视 (Pivot)
        yoy_long_df.sort_values(by=['ts_code', 'end_date'], inplace=True)
        yoy_wide = yoy_long_df.pivot_table(
            index='ann_date',
            columns='ts_code',
            values='total_revenue_growth_yoy',
            aggfunc='last'
        )

        # ---步骤四：调用通用广播引擎 ---
        trading_dates = self.factor_manager.data_manager.trading_dates
        yoy_daily = _broadcast_ann_date_to_daily(yoy_wide, trading_dates)

        print("--- 最终因子: total_revenue_growth_yoy 计算完成 ---")
        return yoy_daily
 
    # === 动量与反转 (Momentum & Reversal) ===
    def _calculate_momentum_120d(self) -> pd.DataFrame:
        """
        计算120日（约半年）动量/累计收益率。

        金融逻辑:
        捕捉市场中期的价格惯性，即所谓的“强者恒强，弱者恒弱”的趋势。
        这是构建趋势跟踪策略的基础。
        """
        logger.info("    > 正在计算因子: momentum_120d...")
        # 1. 获取基础数据：后复权收盘价
        close_df = self.factor_manager.get_factor('close').copy()

        # 2. 计算120个交易日前的价格到今天的收益率
        #    使用 .pct_change() 是最直接且能处理NaN的pandas原生方法
        momentum_df = close_df.pct_change(periods=120)

        logger.info("    > momentum_120d 计算完成。")
        return momentum_df

    def _calculate_reversal_21d(self) -> pd.DataFrame:
        """
        计算21日（约1个月）反转因子。

        金融逻辑:
        A股市场存在显著的短期均值回归现象。即过去一个月涨幅过高的股票，
        在未来倾向于下跌；反之亦然。因此，我们将短期收益率取负，
        得到的分数越高，代表其反转（上涨）的可能性越大。
        """
        logger.info("    > 正在计算因子: reversal_21d...")
        # 1. 获取基础数据：后复权收盘价
        close_df = self.factor_manager.get_factor('close').copy()

        # 2. 计算21日收益率
        return_21d = close_df.pct_change(periods=21)

        # 3. 将收益率取负，即为反转因子
        reversal_df = -return_21d

        logger.info("    > reversal_21d 计算完成。")
        return reversal_df
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
    def _calculate_volatility_90d(self) -> pd.DataFrame:
        """
        计算90日年化波动率。

        金融逻辑:
        衡量个股自身在过去约一个季度内的价格波动风险，也称为特质风险。
        经典的“低波动异象”(Low Volatility Anomaly)指出，长期来看，
        低波动率的股票组合能提供优于高波动率股票的风险调整后收益。
        """
        logger.info("    > 正在计算因子: volatility_90d...")
        # 1. 获取日收益率数据
        pct_chg_df = self.factor_manager.get_factor('pct_chg').copy()

        # 2. 计算90日滚动标准差
        #    min_periods=60 表示在计算初期，即使窗口不满90天，只要有60天数据也开始计算
        #    这是一个在保证数据质量和及时性之间的常见权衡。
        rolling_std_df = pct_chg_df.rolling(window=90, min_periods=60).std()

        # 3. 年化处理：标准差是时间的平方根的函数
        #    一年约有252个交易日
        annualized_vol_df = rolling_std_df * np.sqrt(252)

        logger.info("    > volatility_90d 计算完成。")
        return annualized_vol_df
    def _calculate_beta(self) -> pd.DataFrame:
        beta_df = calculate_rolling_beta(
            self.factor_manager.data_manager.config['backtest']['start_date'],
            self.factor_manager.data_manager.config['backtest']['end_date'],
            self.factor_manager.get_pool_of_factor_name_of_stock_codes('beta')
        )
        return beta_df

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
    def _calculate_turnover_rate_90d_mean(self) -> pd.DataFrame:
        """
        计算90日（约一季度）的平均换手率。

        金融逻辑:
        平滑后的换手率指标，比单日换手率更能反映一支股票在一段时间内的交易活跃度和
        市场关注度。过高或过低的换手率都可能预示着不同的投资机会或风险。
        """
        logger.info("    > 正在计算因子: turnover_rate_90d_mean...")
        # 1. 获取日换手率数据
        turnover_df = self.factor_manager.get_factor('turnover_rate').copy()

        # 2. 计算90日滚动平均
        mean_turnover_df = turnover_df.rolling(window=90, min_periods=60).mean()

        logger.info("    > turnover_rate_90d_mean 计算完成。")
        return mean_turnover_df

    def _calculate_ln_turnover_value_90d(self) -> pd.DataFrame:
        """
        计算90日日均成交额的对数。

        金融逻辑:
        日均成交额直接反映了资产的流动性容量。成交额越大的股票，能容纳的资金规模越大，
        交易时的冲击成本也越低。取对数是为了使数据分布更接近正态，便于进行回归分析。
        """
        logger.info("    > 正在计算因子: ln_turnover_value_90d...")
        # 1. 获取日成交额数据 (单位：元)
        amount_df = self.factor_manager.get_factor('amount').copy()

        # 2. 计算90日滚动平均成交额
        mean_amount_df = amount_df.rolling(window=90, min_periods=60).mean()

        # 3. 【核心风控】在取对数前，必须确保数值为正。
        #    使用 .where() 方法，将所有小于等于0的值替换为NaN，避免log函数报错。
        mean_amount_positive = mean_amount_df.where(mean_amount_df > 0)

        # 4. 计算对数
        ln_turnover_value_df = np.log(mean_amount_positive)

        logger.info("    > ln_turnover_value_90d 计算完成。")
        return ln_turnover_value_df

    # 你提供的代码中已有名为 _calculate_liquidity_amihud 的函数，其实现非常健壮，
    # 我在此处提供一个与YAML文件名完全匹配的版本，逻辑与你的版本一致，以确保可调用。
    # 如果你的FactorManager调用逻辑是严格按'name'匹配的，则需要这个函数。
    def _calculate_amihud_liquidity(self) -> pd.DataFrame:
        """
        计算Amihud非流动性指标。

        金融逻辑:
        衡量单位成交额能引起多大的价格波动，公式为 abs(日收益率) / 日成交额。
        该值越大，说明股票的流动性越差，即用少量资金交易就可能引发剧烈的价格变动，
        这通常意味着更高的交易冲击成本。
        """
        logger.info("    > 正在计算因子: amihud_liquidity...")
        # 1. 获取依赖数据
        pct_chg_df = self.factor_manager.get_factor('pct_chg').copy()
        amount_df = self.factor_manager.get_factor('amount').copy()

        # 2. 【核心风险控制】: 防止除以零。将成交额为0的替换为一个极小正数。
        #    这种处理方式可以保留数据点（结果为一个很大的数），而不是直接丢弃(NaN)。
        #    在某些场景下，这比直接替换为NaN能提供更多信息。
        amount_df_safe = amount_df.where(amount_df > 0, 1e-9)

        # 3. 计算Amihud指标的日度值
        #    注意：Amihud通常在月度或年度上进行平均，这里我们先计算日度值，
        #    可以在后续的因子处理中进行滚动平均以获得更稳定的信号。
        amihud_df = pct_chg_df.abs() / amount_df_safe

        logger.info("    > amihud_liquidity 计算完成。")
        return amihud_df
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

    ##财务basic数据

    def _calculate_cashflow_ttm(self) -> pd.DataFrame:
        """
           【】计算滚动12个月的经营活动现金流净额 (TTM)。
           输入:
           - cashflow_df: 原始现金流量表数据，包含['ann_date', 'ts_code', 'end_date', 'n_cashflow_act']
           - all_trading_dates: 一个包含所有交易日日期的pd.DatetimeIndex，用于构建最终的日度因子矩阵。
           输出:
           - 一个以交易日为索引(index)，股票代码为列(columns)的日度TTM因子矩阵。
           """
        return  self._calculate_financial_ttm_factor('cashflow_ttm',load_cashflow_df,'n_cashflow_act')

    def _calculate_net_profit_ttm(self) -> pd.DataFrame:
        """
        计算滚动12个月的归母净利润 (Net Profit TTM)。
        该函数逻辑与 _calculate_cashflow_ttm 完全一致，仅替换数据源和字段。
        """
        return  self._calculate_financial_ttm_factor('net_profit_ttm',load_income_df,'n_income_attr_p')

    def _calculate_total_equity(self) -> pd.DataFrame:
        """
        【生产级】获取每日可用的最新归母所有者权益。
        这是一个时点数据，无需计算TTM，但需要执行公告日对齐流程。
        """
        ret  = self._calculate_financial_snapshot_factor('total_equity',load_balancesheet_df,'total_hldr_eqy_exc_min_int')
        return ret

    def _calculate_total_revenue_ttm(self) -> pd.DataFrame:
        """
        【生产级】计算滚动12个月的营业总收入 (TTM)。
        利用通用TTM引擎计算得出。
        """
        # print("--- 调用通用引擎计算: revenue_ttm ---")
        return self._calculate_financial_ttm_factor(
            factor_name='total_revenue_ttm',
            data_loader_func=load_income_df,
            source_column='total_revenue'  # Tushare利润表中的“营业总收入”字段
        )

    def _calculate_op_cost_ttm(self) -> pd.DataFrame:
        """
        【生产级】计算滚动12个月的营业总成本 (TTM)。
        利用通用TTM引擎计算得出。
        """
        # print("--- 调用通用引擎计算: op_cost_ttm ---")
        return self._calculate_financial_ttm_factor(
            factor_name='op_cost_ttm',
            data_loader_func=load_income_df,
            source_column='oper_cost'  # Tushare利润表中的“减:营业成本”字段
        )

    def _calculate_total_debt(self) -> pd.DataFrame:
        """【生产级】获取每日可用的最新总负债。"""
        print("--- 调用Snapshot引擎计算: total_debt ---")
        return self._calculate_financial_snapshot_factor(
            factor_name='total_debt',
            data_loader_func=load_balancesheet_df,
            source_column='total_liab'  # Tushare资产负债表中的“负债合计”字段
        )

    def _calculate_total_assets(self) -> pd.DataFrame:
        """
        【生产级】获取每日可用的最新总资产。
        这是一个“时点”或“存量”指标，直接从最新的资产负债表中获取。

        此方法将作为计算其他财务比率（如资产负债率）的分母。
        """
        logger.info("--- 调用Snapshot引擎计算: total_assets ---")

        # 我们将直接调用通用的“时点”因子计算引擎，
        # 只需要告诉它要加载哪个数据表、并使用其中的哪一列即可。
        return self._calculate_financial_snapshot_factor(
            factor_name='total_assets',
            data_loader_func=load_balancesheet_df,  # 指定加载资产负债表
            source_column='total_assets'              # 指定使用资产负债表中的“资产总计”字段
        )

    def _calculate_net_profit_single_q_long(self) -> pd.DataFrame:
        """
        【内部函数】计算单季度归母净利润的长表。
        这是计算同比增长率的基础。
        """
        # print("--- 调用通用引擎计算: net_profit_single_q ---")
        return self._calculate_financial_single_q_factor(
            factor_name='net_profit_single_q',
            data_loader_func=load_income_df,
            source_column='n_income_attr_p'  # 确认使用归母净利润
        )
######################
    ##以下是模板
    # --- 私有的、可复用的计算引擎 ---
    def _calculate_financial_ttm_growth_factor(self,
                                               factor_name: str,
                                               ttm_factor_name: str,
                                               lookback_days: int = 252) -> pd.DataFrame:
        """
        【通用TTM增长率计算引擎】
        根据指定的TTM因子，计算其同比增长率。
        公式: (Current TTM / Last Year's TTM) - 1

        Args:
            factor_name (str): 最终生成的因子名称（用于日志记录）。
            ttm_factor_name (str): 依赖的TTM因子的名称。
            lookback_days (int): 回溯周期，默认为252个交易日（约一年）。

        Returns:
            pd.DataFrame: 计算出的TTM同比增长率因子矩阵。
        """
        logger.info(f"      > [引擎] 正在为 {ttm_factor_name} 计算TTM同比增长率...")

        # --- 步骤一：获取当期的TTM因子数据 ---
        ttm_df = self.factor_manager.get_factor(ttm_factor_name)

        # --- 步骤二：获取一年前（回溯期）的TTM因子数据 ---
        ttm_last_year = ttm_df.shift(lookback_days)

        # --- 步骤三：【核心风险控制】---
        # 金融逻辑：增长率的计算只有在分子(当期)和分母(去年同期)都为正时才有意义。
        # 这确保我们只比较“从盈利到盈利”的情况，避免了由盈转亏等情况带来的噪音。
        logger.info("        > 正在进行风险控制，确保分子和分母均为正数...")
        ttm_df_safe = ttm_df.where(ttm_df > 0)
        ttm_last_year_safe = ttm_last_year.where(ttm_last_year > 0)

        # --- 步骤四：计算同比增长率 ---
        growth_df = (ttm_df_safe / ttm_last_year_safe) - 1

        # --- 步骤五：后处理 ---
        # 防御性编程：清除计算过程中可能意外产生的无穷大值。
        # 采用重新赋值，避免使用 inplace=True。
        growth_df = growth_df.replace([np.inf, -np.inf], np.nan)

        logger.info(f"--- [引擎] 因子: {factor_name} 计算完成 ---")
        return growth_df

        ###A股市场早期，或一些公司在特定时期，只会披露年报和半年报，而缺少一季报和三季报的累计值。这会导致在我们的完美季度时间标尺上出现NaN。
        ### 所以这就是解决方案：实现了填充 跳跃的季度区间，新增填充的列：filled_col ，计算就在filled_col上面做diff。然后在平滑diff上做rolling。done
        ## 季度性数据ttm通用计算， 模板计算函数 ok
    def _calculate_financial_ttm_factor(self,
                                        factor_name: str,
                                        data_loader_func: Callable[[], pd.DataFrame],
                                        source_column: str) -> pd.DataFrame:
        """
        【通用生TTM因子计算引擎】(已重构)
        计算滚动12个月(TTM)的因子值。
        """
        print(f"--- [引擎] 开始计算TTM因子: {factor_name} ---")

        # --- 步骤一： 获取单季度数据 ---
        single_q_col_name = f"{source_column}_single_q"
        single_q_long_df = self._get_single_q_long_df(
            data_loader_func=data_loader_func,
            source_column=source_column,
            single_q_col_name=single_q_col_name
        )

        # --- 步骤二：在单季度数据的基础上，计算TTM ---
        single_q_long_df[factor_name] = single_q_long_df.groupby('ts_code')[single_q_col_name].rolling(
            window=4, min_periods=4
        ).sum().reset_index(level=0, drop=True)

        # --- 步骤三：格式化为日度因子矩阵 (Pivot -> Reindex -> ffill) ---
        ttm_long_df = single_q_long_df[['ts_code', 'ann_date', 'end_date', factor_name]].dropna()#factor_name是rooling 计算出来的，因为min_periods 所以有三行nan ，在这里会被移除行
        if ttm_long_df.empty:
            raise ValueError(f"警告: 计算因子 {factor_name} 后没有产生任何有效的TTM数据点。")

        ttm_long_df = ttm_long_df.sort_values(by=['ts_code', 'end_date'])
        ttm_wide = ttm_long_df.pivot_table(
            index='ann_date', #以ann_date 作为索引，这是无规则的index。假设100只股票，可能同一天有发布报告的股票只有一只
            columns='ts_code',
            values=factor_name,
            aggfunc='last'
        )#执行完之后的ttm_Wide 可能到处都是nan，原因：（以ann_date 作为索引，这是无规则的index。假设100只股票，可能同一天有发布报告的股票只有一只）
        # ttm_daily = (ttm_wide.reindex(self.factor_manager.data_manager.trading_dates) #注意 满目苍翼的ttmwide然后还被对齐索引（截断，）从trading开始日开始截，万一刚好这一天 股票值为nan，那么后面ffill也是nan，直到下一个有效ann_date
        #              .ffill())
        # filled_wide = ttm_wide.ffill() #基于上面的注意，这里单独做处理！
        ##
        # 很隐蔽的bug
        # 基于ann_date作为index
        # 也就意味着，比如整个7月没有任何股票进行发报告，即 ann_date不会出现在整个8月
        # 尽管有填充：filled_wide = ttm_wide.ffill() #基于上面的注意，这里单独做处理！ 可能是下一个月ann_date才有值 比如0905是下一个ann_date，上一个ann_date是0730
        # 如果我们传入的ttm_daily = filled_wide.reindex(self.factor_manager.data_manager.trading_dates).ffill() 交易日，是0804
        # 开始的，那么无法找到filled_wide的index是这一天的 那么默认就算是nan，然后经过fiil，到下一个ann_date(0905） 全是nan！！！#
        # ttm_daily = filled_wide.reindex(self.factor_manager.data_manager.trading_dates).ffill() 解决：避免阶段，提前全局弄
        # 1. 获取交易日历
        trading_dates = self.factor_manager.data_manager.trading_dates
        ret = _broadcast_ann_date_to_daily(ttm_wide, trading_dates)

        print(f"--- [引擎] 因子: {factor_name} 计算完成 ---")

        return ret

    # 加载财报中的“时点”数据，并将其正确地映射到每日的时间序列上。 ok
    def _calculate_financial_snapshot_factor(self,
                                             factor_name: str,
                                             data_loader_func: Callable[[], pd.DataFrame],
                                             source_column: str) -> pd.DataFrame:
        """
        【通用生产级“时点”因子计算引擎】
        根据指定的财务报表数据和字段，获取最新的“时点”因子值。
        适用于资产、负债、股东权益等“存量”指标。

        参数:
        - factor_name (str): 你想生成的最终因子名称，如 'total_assets'。
        - data_loader_func (Callable): 一个无参数的函数，用于加载原始财务数据DataFrame。
                                      例如: self.data_manager.load_balancesheet_df
        - source_column (str): 原始财务数据中的时点字段名。
                               例如: 'total_assets'

        返回:
        - 一个以交易日为索引(index)，股票代码为列(columns)的日度时点因子矩阵。
        """
        print(f"--- [通用引擎] 开始计算Snapshot因子: {factor_name} ---")

        # 使用传入的函数加载数据
        financial_df = data_loader_func()

        # 步骤一：选择数据并确保有效性
        snapshot_long_df = financial_df[['ts_code', 'ann_date', 'end_date', source_column]].copy()
        snapshot_long_df.dropna(inplace=True)
        if snapshot_long_df.empty:
            raise ValueError(f"警告: 计算因子 {factor_name} 时，从 {source_column} 字段未获取到有效数据。")

        # 步骤二：透视
        snapshot_long_df.sort_values(by=['ts_code', 'end_date'], inplace=True)
        snapshot_wide = snapshot_long_df.pivot_table(
            index='ann_date',
            columns='ts_code',
            values=source_column,
            aggfunc='last'
        )
        # --- 步骤三：【核心修正】使用合并索引的方法，进行稳健的重索引和填充 ---
        # 1. 获取交易日历
        trading_dates = self.factor_manager.data_manager.trading_dates

        # 2. 将稀疏的“公告日”索引与密集的“交易日”索引合并，并排序
        combined_index = snapshot_wide.index.union(trading_dates)

        # 3. 将 snapshot_wide 扩展到这个超级索引上，然后进行前向填充
        snapshot_filled_on_super_index = snapshot_wide.reindex(combined_index).ffill()

        # 4. 最后，从这个填充好的、完整的DataFrame中，只选取我们需要的交易日
        snapshot_daily = snapshot_filled_on_super_index.loc[trading_dates]

        print(f"--- [通用引擎] 因子: {factor_name} 计算完成 ---")
        return snapshot_daily

    def _calculate_financial_single_q_factor(self,
                                             factor_name: str,
                                             data_loader_func: Callable[[], pd.DataFrame],
                                             source_column: str) -> pd.DataFrame:
        """
        【通用生产级“单季度”因子计算引擎】(已重构)
        获取单季度的因子值的长表DataFrame。
        """
        print(f"--- [引擎] 开始准备单季度长表: {factor_name} ---")

        # 直接调用底层零件函数，获取单季度长表
        single_q_long_df = self._get_single_q_long_df(
            data_loader_func=data_loader_func,
            source_column=source_column,
            single_q_col_name=factor_name  # 输出列名就是我们想要的因子名
        )

        return single_q_long_df

    def _get_single_q_long_df(self,
                              data_loader_func: Callable[[], pd.DataFrame],
                              source_column: str,
                              single_q_col_name: str) -> pd.DataFrame:
        """
        【底层零件】从累计值财报数据中，计算出单季度值的长表DataFrame。
        这是所有TTM和YoY计算的共同基础。
        """
        print(f"    >  正在从 {source_column} 计算 {single_q_col_name}...")

        financial_df = data_loader_func()

        # 核心计算逻辑 (Scaffold -> Merge -> Diff)
        scaffold_df = financial_df.groupby('ts_code')['end_date'].agg(['min', 'max']) #记录一股票 两个时间点
        full_date_dfs = []
        for ts_code, row in scaffold_df.iterrows():
            date_range = pd.date_range(start=row['min'], end=row['max'], freq='Q-DEC')##记录一股票 两个时间点 期间所有报告期日(0331 0630 0930 1231
            full_date_dfs.append(pd.DataFrame({'ts_code': ts_code, 'end_date': date_range}))
        full_dates_df = pd.concat(full_date_dfs)

        merged_df = pd.merge(full_dates_df, financial_df, on=['ts_code', 'end_date'], how='left')

        filled_col = f"{source_column}_filled"
        merged_df[filled_col] = merged_df.groupby('ts_code')[source_column].ffill()
        merged_df[single_q_col_name] = merged_df.groupby('ts_code')[filled_col].diff()
        #第一个季度就是自己的值！（前面做了diff，现在需要更正季度为q1de！
        is_q1 = merged_df['end_date'].dt.month == 3
        merged_df.loc[is_q1 & merged_df[source_column].notna(), single_q_col_name] = merged_df.loc[is_q1, source_column]

        # 整理并返回包含单季度值的长表
        single_q_long_df = merged_df[['ts_code', 'ann_date', 'end_date', single_q_col_name]].copy()
        single_q_long_df.dropna(subset=[single_q_col_name, 'ann_date'], inplace=True)  # 确保公告日和计算值都存在

        return single_q_long_df



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

def _broadcast_ann_date_to_daily(
                                 sparse_wide_df: pd.DataFrame,
                                 trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    【核心通用工具】将一个基于稀疏公告日(ann_date)的宽表，
    安全地广播并填充到一个密集的交易日历上。

    这是解决所有财报类因子“期初NaN”问题的最终解决方案。

    Args:
        sparse_wide_df (pd.DataFrame): 以ann_date为索引的稀疏宽表。
        trading_dates (pd.DatetimeIndex): 目标交易日历。

    Returns:
        pd.DataFrame: 以交易日为索引的、被正确填充的稠密宽表。
    """
    # 1. 将稀疏的“公告日”索引与密集的“交易日”索引合并
    combined_index = sparse_wide_df.index.union(trading_dates)

    # 2. 扩展到“超级索引”上，然后进行决定性的前向填充
    filled_df = sparse_wide_df.reindex(combined_index).ffill()

    # 3. 最后，只裁剪出我们需要的交易日，并返回
    daily_df = filled_df.loc[trading_dates]

    return daily_df
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
