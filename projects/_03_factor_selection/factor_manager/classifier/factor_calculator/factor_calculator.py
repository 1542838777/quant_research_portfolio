from typing import Callable  # 引入Callable来指定函数类型的参数

import numpy as np
import pandas as pd
import pandas_ta as ta

from data.local_data_load import load_index_daily, load_cashflow_df, load_income_df, \
    load_balancesheet_df, load_dividend_events_long
from quant_lib import logger


## 数据统一 tushare 有时候给元 千元 万元!  现在需要达成:统一算元!
#remind:  turnover_rate 具体处理 都要/100
# total_mv, circ_mv 具体处理 *10000
# amount 具体处理  *1000
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
        # 可以通过 factor_manager.get_raw_factor() 来获取基础因子，并利用其缓存机制。
        self.factor_manager = factor_manager
        print("FactorCalculator (因子计算器) 已准备就绪。")



    # === 规模 (Size) ===
    def _calculate_log_circ_mv(self) -> pd.DataFrame:
        circ_mv_df = self.factor_manager.get_raw_factor('circ_mv').copy()
        # 保证为正数，避免log报错
        circ_mv_df = circ_mv_df.where(circ_mv_df > 0)
        factor_df = circ_mv_df.apply(np.log)
        return factor_df
    def _calculate_log_total_mv(self) -> pd.DataFrame:
        circ_mv_df = self.factor_manager.get_raw_factor('total_mv').copy()
        # 保证为正数，避免log报错
        circ_mv_df = circ_mv_df.where(circ_mv_df > 0)
        # 使用 pandas 自带 log 函数，保持类型一致
        factor_df = circ_mv_df.apply(np.log)
        return factor_df

    # === 价值 (Value) - 【V2.0 - 第一性原理版】 ===

    def _calculate_bm_ratio(self) -> pd.DataFrame:
        """
        【V2.0 - 第一性原理版】计算账面市值比 (Book-to-Market Ratio)。
        B/M = total_equity / total_mv
        """
        logger.info("  > 正在基于第一性原理，计算【权威 bm_ratio】...")

        # 1. 获取分子: 随时点的股东权益 (Book Value)
        #    我们的财报引擎，确保了这里使用的是 ann_date，无未来数据
        book_value_df = self.factor_manager.get_raw_factor('total_equity').copy(deep=True)

        # 2. 获取分母: 随时点的总市值 (Market Value)
        #    我们信任 daily_basic 中的 total_mv 是随时点正确的
        market_value_df = self.factor_manager.get_raw_factor('total_mv').copy(deep=True)

        # 3. 对齐数据
        book_aligned, market_aligned = book_value_df.align(market_value_df, join='right', axis=None)

        # 4. 对齐后，对低频的财报数据进行前向填充
        book_aligned_filled = book_aligned.ffill()

        # 5. 计算因子并进行风控
        market_positive = market_aligned.where(market_aligned > 0)
        book_positive = book_aligned_filled.where(book_aligned_filled > 0)

        bm_ratio_df = book_positive / market_positive
        return bm_ratio_df.replace([np.inf, -np.inf], np.nan)

    def _calculate_ep_ratio(self) -> pd.DataFrame:
        """
        【V2.0 - 第一性原理版】计算盈利收益率 (Earnings Yield)。
        E/P = net_profit_ttm / total_mv
        """
        logger.info("  > 正在基于第一性原理，计算【权威 ep_ratio】...")

        # 1. 获取分子: TTM归母净利润
        earnings_ttm_df = self.factor_manager.get_raw_factor('net_profit_ttm').copy(deep=True)

        # 2. 获取分母: 总市值
        market_value_df = self.factor_manager.get_raw_factor('total_mv').copy(deep=True)

        # 3. 对齐与填充
        earnings_aligned, market_aligned = earnings_ttm_df.align(market_value_df, join='right', axis=None)
        earnings_aligned_filled = earnings_aligned.ffill()

        # 4. 计算因子
        market_positive = market_aligned.where(market_aligned > 0)
        # 对于盈利，分子可以是负数，所以不做 book_positive 类似的筛选
        ep_ratio_df = earnings_aligned_filled / market_positive
        return ep_ratio_df.replace([np.inf, -np.inf], np.nan)

    def _calculate_sp_ratio(self) -> pd.DataFrame:
        """
        【V2.0 - 第一性原理版】计算销售收益率 (Sales Yield)。
        S/P = total_revenue_ttm / total_mv
        """
        logger.info("  > 正在基于第一性原理，计算【权威 sp_ratio】...")

        # 1. 获取分子: TTM营业总收入
        sales_ttm_df = self.factor_manager.get_raw_factor('total_revenue_ttm').copy(deep=True)

        # 2. 获取分母: 总市值
        market_value_df = self.factor_manager.get_raw_factor('total_mv').copy(deep=True)

        # 3. 对齐与填充
        sales_aligned, market_aligned = sales_ttm_df.align(market_value_df, join='right', axis=None)
        sales_aligned_filled = sales_aligned.ffill()

        # 4. 计算因子
        market_positive = market_aligned.where(market_aligned > 0)
        sales_positive = sales_aligned_filled.where(sales_aligned_filled > 0)  # 收入通常为正

        sp_ratio_df = sales_positive / market_positive
        return sp_ratio_df.replace([np.inf, -np.inf], np.nan)

    def _calculate_cfp_ratio(self) -> pd.DataFrame:
        """
        【V2.0 - 第一性原理版】计算现金流市值比 (Cash Flow Yield)。
        CF/P = cashflow_ttm / total_mv
        """
        logger.info("  > 正在基于第一性原理，计算【权威 cfp_ratio】...")

        # 1. 获取分子: TTM经营活动现金流
        cashflow_ttm_df = self.factor_manager.get_raw_factor('cashflow_ttm').copy(deep=True)

        # 2. 获取分母: 总市值
        market_value_df = self.factor_manager.get_raw_factor('total_mv').copy(deep=True)

        # 3. 对齐与填充
        cashflow_aligned, market_aligned = cashflow_ttm_df.align(market_value_df, join='right', axis=None)
        cashflow_aligned_filled = cashflow_aligned.ffill()

        # 4. 计算因子
        market_positive = market_aligned.where(market_aligned > 0)

        cfp_ratio_df = cashflow_aligned_filled / market_positive
        return cfp_ratio_df.replace([np.inf, -np.inf], np.nan)

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
        roe_ttm_df=roe_ttm_df.replace([np.inf, -np.inf], np.nan, inplace=False)

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
        return  gross_margin_ttm_df.replace([np.inf, -np.inf], np.nan, inplace=False)

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
        return debt_to_assets_df.replace([np.inf, -np.inf], np.nan, inplace=False)
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
        net_profit_single_q_long = net_profit_single_q_long.sort_values(by=['ts_code', 'end_date'], inplace=False)

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
        yoy_long_df= yoy_long_df.dropna(inplace=False)
        if yoy_long_df.empty:
            raise ValueError("警告: 计算 net_profit_growth_yoy 后没有产生任何有效的增长率数据点。")

        # 2. 透视 (Pivot)
        yoy_long_df = yoy_long_df.sort_values(by=['ts_code', 'end_date'], inplace=False)
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
        total_revenue_single_q_long= total_revenue_single_q_long.sort_values(by=['ts_code', 'end_date'], inplace=False)

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
        yoy_long_df=yoy_long_df.dropna(inplace=False)
        if yoy_long_df.empty:
            raise ValueError("警告: 计算 total_revenue_growth_yoy 后没有产生任何有效的增长率数据点。")

        # 2. 透视 (Pivot)
        yoy_long_df = yoy_long_df.sort_values(by=['ts_code', 'end_date'], inplace=False)
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
    ###材料 是否需要填充 介绍:
    ##
    # ：动量/反转类因子
    # 因子示例: _calculate_momentum_120d, _calculate_reversal_21d, _calculate_momentum_12_1, _calculate_momentum_20d
    #
    # 计算特性: 它们的核心逻辑是 price(t) / price(t-N) - 1。这类计算对价格的绝对时间间隔非常敏感。
    #
    # 推荐使用: self.factor_manager.get_raw_factor('close_hfq') (未经填充的版本)
    #
    # 理由:
    #
    # 想象一下momentum_12_1的计算：close_hfq.shift(21) / close_hfq.shift(252) - 1。
    #
    # 如果一只股票在t-252之后停牌了半年，然后复牌。如果你使用了close_hfq_filled，那么close_hfq.shift(252)取到的就是一个非常“陈腐”的、半年前的价格。用这个陈腐价格计算出的动量值，其经济学意义是存疑的。
    #
    # 更稳健的做法是使用未经填充的close_hfq。如果在t-21或t-252的任一时间点，股票是停牌的（值为NaN），那么最终的动量因子值也应该是NaN。我们宁愿在没有可靠数据时得到一个NaN，也不要一个基于陈腐数据计算出的错误值。#
    def _calculate_momentum_120d(self) -> pd.DataFrame:
        """
        计算120日（约半年）动量/累计收益率。

        金融逻辑:
        捕捉市场中期的价格惯性，即所谓的“强者恒强，弱者恒弱”的趋势。
        这是构建趋势跟踪策略的基础。
        """
        logger.info("    > 正在计算因子: momentum_120d...")
        # 1. 获取基础数据：后复权收盘价
        close_df = self.factor_manager.get_raw_factor('close_hfq').copy()

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
        close_df = self.factor_manager.get_raw_factor('close_hfq').copy()

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
        close_df = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)
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
        close_df = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)
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
        pct_chg_df = self.factor_manager.get_raw_factor('pct_chg').copy()

        # 2. 计算90日滚动标准差
        #    min_periods=60 表示在计算初期，即使窗口不满90天，只要有60天数据也开始计算
        #    这是一个在保证数据质量和及时性之间的常见权衡。
        rolling_std_df = pct_chg_df.rolling(window=90, min_periods=60).std()

        # 3. 年化处理：标准差是时间的平方根的函数
        #    一年约有252个交易日
        annualized_vol_df = rolling_std_df * np.sqrt(252)

        logger.info("    > volatility_90d 计算完成。")
        return annualized_vol_df

    def _calculate_beta(self, benchmark_index, window: int = 60,
                        min_periods: int = 20) -> pd.DataFrame:
        """
        1. 从FactorManager获取个股和指定的市场收益率。
        2. 准备数据缓冲期。
        3. 调用纯函数 `calculate_rolling_beta_pure` 进行计算。
        """
        logger.info(f"调度Beta计算任务 (基准: {benchmark_index}, 窗口: {window}天)...")

        # --- 1. 获取原材料 ---
        stock_returns = self.factor_manager.get_raw_factor('pct_chg')
        market_returns = self._calculate_market_pct_chg(index_code=benchmark_index)

        # --- 2. 准备 ---
        config = self.factor_manager.data_manager.config['backtest']
        start_date, end_date = config['start_date'], config['end_date']

        buffer_days = int(window * 1.7) + 5
        buffer_start_date = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')

        stock_returns_buffered = stock_returns.loc[buffer_start_date:end_date] #必须多点前缀数据,后面好进行rolling
        market_returns_buffered = market_returns.loc[buffer_start_date:end_date]

        # --- 3. 执行计算 ---
        beta_df_full = calculate_rolling_beta_pure(
            stock_returns=stock_returns_buffered,
            market_returns=market_returns_buffered,
            window=window,
            min_periods=min_periods
        )
        return beta_df_full

    def _calculate_volatility_120d(self) -> pd.DataFrame:
        """
        计算120日年化波动率。

        金融逻辑:
        衡量个股在过去约半年内的价格波动风险。经典的“低波动异象”认为，
        低波动率的股票长期来看反而有更高的风险调整后收益。

        数据处理逻辑:
        - 停牌期间的收益率为NaN，这是正确的，不应该填充为0
        - rolling.std()会自动忽略NaN值进行计算
        - min_periods=60确保至少有60个有效交易日才计算波动率
        """
        print("    > 正在计算因子: volatility_120d...")
        # pct_chg_df = self.factor_manager.get_raw_factor('pct_chg').copy(deep=True)
        close_hfq = self.factor_manager.get_raw_factor('close_hfq').copy()

        # 2. 计算120个交易日前的价格到今天的收益率
        #    使用 .pct_change() 是最直接且能处理NaN的pandas原生方法
        annualized_vol_df = close_hfq.pct_change().rolling(window=120, min_periods=60).std() * np.sqrt(252)
        # 【修复】不填充NaN，让rolling函数自然处理停牌期间的缺失值
        # 这样计算出的波动率更准确，只基于实际交易日的收益率

        return annualized_vol_df
    def _calculate_volatility_40d(self) -> pd.DataFrame:
        """
        计算120日年化波动率。

        金融逻辑:
        衡量个股在过去约半年内的价格波动风险。经典的“低波动异象”认为，
        低波动率的股票长期来看反而有更高的风险调整后收益。

        数据处理逻辑:
        - 停牌期间的收益率为NaN，这是正确的，不应该填充为0
        - rolling.std()会自动忽略NaN值进行计算
        - min_periods=60确保至少有60个有效交易日才计算波动率
        """
        # print("    > 正在计算因子: volatility_120d...")
        pct_chg_df = self.factor_manager.get_raw_factor('pct_chg').copy(deep=True)

        # 【修复】不填充NaN，让rolling函数自然处理停牌期间的缺失值
        # 这样计算出的波动率更准确，只基于实际交易日的收益率

        rolling_std_df = pct_chg_df.rolling(window=40, min_periods=20).std()
        annualized_vol_df = rolling_std_df * np.sqrt(252)

        # 【新增】数据质量检查
        if annualized_vol_df.isna().all().all():
            raise ValueError("警告：波动率计算结果全为NaN，请检查输入数据")

        return annualized_vol_df

    # === 流动性 (Liquidity) ===
    def _calculate_rolling_mean_turnover_rate(self, window: int, min_periods: int) -> pd.DataFrame:
        """【私有引擎】计算滚动平均换手率（以小数形式）。"""

        # 1. 获取原始换手率数据（其中停牌日为NaN） ---现在直接ffill（0） 符合金融要求
        turnover_df_filled = self.factor_manager.get_raw_factor('turnover_rate_fill_zero')

        # 2. 在填充后的、代表了真实交易活动的数据上进行滚动计算
        mean_turnover_df = turnover_df_filled.rolling(window=window, min_periods=min_periods).mean()

        return mean_turnover_df

    # --- 现在，原来的两个函数可以简化为下面这样 ---

    def _calculate_turnover_rate_90d_mean(self) -> pd.DataFrame:
        """计算90日滚动平均换手率。"""
        logger.info("    > 正在计算因子: turnover_rate_90d_mean...")
        # 直接调用通用引擎
        return self._calculate_rolling_mean_turnover_rate(window=90, min_periods=60)

    def _calculate_turnover_rate_monthly_mean(self) -> pd.DataFrame:
        """计算月度（21日）滚动平均换手率。"""
        logger.info("    > 正在计算因子: turnover_rate_monthly_mean...")
        # 直接调用通用引擎
        return self._calculate_rolling_mean_turnover_rate(window=21, min_periods=15)


    def _calculate_ln_turnover_value_90d(self) -> pd.DataFrame:
        """
        计算90日日均成交额的对数。

        金融逻辑:
        日均成交额直接反映了资产的流动性容量。成交额越大的股票，能容纳的资金规模越大，
        交易时的冲击成本也越低。取对数是为了使数据分布更接近正态，便于进行回归分析。
        """
        logger.info("    > 正在计算因子: ln_turnover_value_90d...")
        # 1. 获取日成交额数据 (单位：元)
        amount_df = self.factor_manager.get_raw_factor('amount_fill_zero').copy()

        # 2. 计算90日滚动平均成交额
        mean_amount_df = amount_df.rolling(window=90, min_periods=60).mean()

        # 3. 【核心风控】在取对数前，必须确保数值为正。
        #    使用 .where() 方法，将所有小于等于0的值替换为NaN，避免log函数报错。
        mean_amount_positive = mean_amount_df.where(mean_amount_df > 0)

        # 4. 计算对数
        ln_turnover_value_df = np.log(mean_amount_positive)

        logger.info("    > ln_turnover_value_90d 计算完成。")
        return ln_turnover_value_df



    def _calculate_amihud_liquidity(self) -> pd.DataFrame:
        """
       计算Amihud非流动性指标 - 最终生产版。

       处理流程:
       1. 计算原始日度Amihud指标。
       2. 对数变换(log1p)处理分布形状。
       3. 滚动平均以平滑信号。
       4. 截面标准化(Z-Score)处理数据尺度，使其对回归模型友好。
       """

        logger.info("    > 正在计算因子: amihud_liquidity (非流动性) -...")

        # 步骤 1: 计算原始日度Amihud
        pct_chg_df = self.factor_manager.get_raw_factor('pct_chg').copy()
        amount_df = self.factor_manager.get_raw_factor('amount').copy()

        amount_in_yuan = amount_df
        amount_in_yuan_safe = amount_in_yuan.where(amount_in_yuan > 0)
        daily_amihud_df = pct_chg_df.abs() / amount_in_yuan_safe

        # 2. 对数变换
        # 使用 np.log1p()，它计算的是 log(1 + x)，可以完美处理x接近0的情况。
        # 这是处理这类因子的标准做法。
        log_amihud_df = np.log1p(daily_amihud_df)

        # 3. 【】: 滚动平滑
        # 使用过去一个月（约20个交易日）的平均值来代表当天的流动性水平
        # 这会使因子信号更稳定，减少日常噪声。
        smoothed_log_amihud_df = log_amihud_df.rolling(window=20, min_periods=12).mean()

        # 【修正】移除因子计算阶段的标准化，保持原始经济含义
        # 标准化应该在预处理阶段统一进行，而不是在因子计算阶段
        final_amihud_df = smoothed_log_amihud_df

        logger.info("    > amihud_liquidity (最终版) 计算完成。")

        # 丢弃所有值都为NaN的行，这些通常是回测初期或数据不足的行
        return final_amihud_df.dropna(axis=0, how='all', inplace=False)

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

        #  三、新增进阶因子 (Advanced Factors)
        # =========================================================================

        # === 质量类深化 (Advanced Quality) ===


    def _calculate_operating_accruals(self) -> pd.DataFrame:
        """
        计算经营性应计利润 (Operating Accruals)。
        公式: (净利润TTM - 经营活动现金流TTM) / 总资产
        这是一个反向指标，值越高，利润质量越差，未来反转（下跌）风险越高。
        """
        logger.info("    > 正在计算因子: operating_accruals (经营性应计利润)...")

        # 1. 获取所需的基础因子
        net_profit_ttm = self.factor_manager.get_raw_factor('net_profit_ttm')
        cashflow_ttm = self.factor_manager.get_raw_factor('cashflow_ttm')
        total_assets = self.factor_manager.get_raw_factor('total_assets')

        # 2. 对齐数据 (核心修正部分)
        # 使用 reindex 的方式对齐多个DataFrame，这是更稳健和清晰的做法

        # 2.1 找到所有数据源索引的交集，这等价于 'inner' join
        common_index = net_profit_ttm.index.intersection(cashflow_ttm.index)
        common_index = common_index.intersection(total_assets.index)

        # 2.2 使用共同索引来对齐所有DataFrame
        profit_aligned = net_profit_ttm.reindex(common_index)
        cash_aligned = cashflow_ttm.reindex(common_index)
        assets_aligned = total_assets.reindex(common_index)

        # 3. 风险控制：总资产必须为正，防止除以0或负数
        # 使用 .where 方法，不满足条件的项会被设置为NaN，这很安全
        assets_aligned_safe = assets_aligned.where(assets_aligned > 0)

        # 4. 计算应计利润
        # 对齐后，可以直接进行元素级(element-wise)计算
        accruals = (profit_aligned - cash_aligned) / assets_aligned_safe

        # 5. 清理计算过程中可能产生的无穷大值
        accruals = accruals.replace([np.inf, -np.inf], np.nan)

        return accruals

    def _calculate_earnings_stability(self) -> pd.DataFrame:
        """
        计算盈利稳定性 (Earnings Stability) - 修正版。
        使用变异系数的倒数，衡量盈利的相对稳定性，剔除规模效应。
        公式: abs(滚动平均净利润) / 滚动净利润标准差
        这是一个正向指标，值越高，盈利相对越稳定。
        """
        logger.info("    > 正在计算因子: earnings_stability (盈利稳定性) - 修正版...")

        # 1. 获取单季度净利润长表数据 (此部分不变)
        net_profit_q_long = self._get_single_q_long_df(
            data_loader_func=load_income_df,
            source_column='n_income_attr_p',  # 归母净利润
            single_q_col_name='net_profit_single_q'
        )

        # 2. 计算滚动的平均值和标准差 (核心改动)
        # window=20 -> 5年 * 4季度/年
        # 使用 groupby + rolling 的标准模式
        grouped = net_profit_q_long.groupby('ts_code')['net_profit_single_q']
        rolling_stats = grouped.rolling(window=20, min_periods=12)

        mean_col = 'earnings_mean'
        std_col = 'earnings_std'
        net_profit_q_long[mean_col] = rolling_stats.mean().reset_index(level=0, drop=True)
        net_profit_q_long[std_col] = rolling_stats.std().reset_index(level=0, drop=True)

        # 3. 计算稳定性（信噪比），并进行风险控制
        stability_col = 'earnings_stability'

        # 风控1: 标准差极小（接近0），意味着盈利极其稳定。给一个封顶的大值，防止无穷大。
        # 风控2: 平均利润的绝对值也极小，此时信噪比无意义，设为0或NaN。
        # 这里我们创建一个条件，当标准差大于1e-6且平均利润绝对值也大于一个小数（如1000元）时才计算

        # 盈利标准差，处理极小值，避免除零
        std_safe = net_profit_q_long[std_col].clip(lower=1e-6)  # 将小于1e-6的值替换为1e-6

        # 盈利均值，处理绝对值过小的情况
        mean_safe = net_profit_q_long[mean_col].where(abs(net_profit_q_long[mean_col]) > 1000, 0)

        net_profit_q_long[stability_col] = abs(mean_safe) / std_safe

        # 4. 将结果广播到每日的宽表 (此部分不变)
        stability_long_df = net_profit_q_long[['ts_code', 'ann_date', 'end_date', stability_col]].dropna()
        stability_wide = stability_long_df.pivot_table(
            index='ann_date', columns='ts_code', values=stability_col, aggfunc='last'
        )
        trading_dates = self.factor_manager.data_manager.trading_dates
        stability_daily_df = _broadcast_ann_date_to_daily(stability_wide, trading_dates)

        return stability_daily_df

        # === 新增情绪类因子 (Sentiment) ===
    ##
    # 滚动技术指标类 (价格材料 必须喂给它是连续的
    # 因子示例: _calculate_rsi, _calculate_cci
    #
    # 计算特性: 它们的算法（尤其是在pandas_ta这样的库中）通常假定输入的时间序列是连续的。数据的中断（NaN）会导致指标计算中断，产生非常稀疏的因子值。
    #
    # 推荐使用: self.factor_manager.get_raw_factor(('close_hfq_filled', 10)) (经过填充的版本)
    #
    # 理由:
    #
    # 这是一个实用性和纯粹性之间的权衡。
    #
    # 为了得到一个更连续、在策略中更“可用”的因子信号，我们主动选择接受一个假设：“短期停牌（如10天内），股票的状态可以被认为是其停牌前的延续”。
    #
    # 我们用一个带limit的ffill来填充短期的NaN，以确保技术指标能够连续计算。这是一个主动的、有意识的建模选择。#
    def _calculate_rsi(self, window: int = 14) -> pd.DataFrame:
        """
        计算RSI (相对强弱指数)。
        衡量股价的超买超卖状态，是经典的反转信号。
        """
        logger.info(f"    > 正在计算因子: RSI (window={window})...")
        close_df = self.factor_manager.get_raw_factor(('close_hfq_filled', 10))

        # 使用 pandas_ta 库，通过 .apply 在每一列（每只股票）上独立计算
        rsi_df = close_df.apply(lambda x: ta.rsi(x, length=window), axis=0)

        return rsi_df

    def _calculate_cci(self, window: int = 20) -> pd.DataFrame:
        """
        计算CCI (顺势指标)。
        衡量股价是否超出其正常波动范围，可用于捕捉趋势的开启或反转。
        """
        logger.info(f"    > 正在计算因子: CCI (window={window})...")
        high_df = self.factor_manager.get_raw_factor(('high_hfq_filled',10))
        low_df = self.factor_manager.get_raw_factor(('low_hfq_filled',10))
        close_df = self.factor_manager.get_raw_factor(('close_hfq_filled',10))
        # CCI需要三列数据，我们按股票逐一计算
        cci_results = {}
        for stock_code in close_df.columns:
            # 确保该股票在所有价格数据中都存在
            if stock_code in high_df.columns and stock_code in low_df.columns:
                cci_series = ta.cci(
                    high=high_df[stock_code],
                    low=low_df[stock_code],
                    close=close_df[stock_code],
                    length=window
                )
                cci_results[stock_code] = cci_series

        cci_df = pd.DataFrame(cci_results)
        return cci_df

    ###惊喜
    # (确保在文件顶部导入):
    # from pandas.tseries.offsets import BDay # BDay代表Business Day
    ##
    # 核心逻辑：SUE衡量的是盈利的“惊喜”程度。
    #
    # 市场如何反映“惊喜”：一家公司发布了超预期的财报，市场最直接的反应是什么？股价会跳空高开，并在接下来几天持续上涨。这种现象被称为“盈余公告后漂移”(Post-Earnings Announcement Drift, PEAD)，是金融学里最著名、最稳健的异象之一。
    #
    # 计算财报发布日后几天的累计收益率(财报后漂移。
    # pead Post-Earnings Announcement Drift) #
    def _calculate_pead(self) -> pd.DataFrame:
        """
        计算修正后的PEAD因子，基于“盈利意外”而非未来收益。
        """
        logger.info(f"    > 正在计算【修正版】PEAD因子...")

        # 1. 加载包含“净利润”和“公告日”的财报数据
        income_df_long = load_income_df()  # 假设包含 'net_profit' 字段

        # 2. 计算某种形式的“盈利意外” (Earnings Surprise)
        #    这里使用一个简化版：(当季净利 - 去年同期净利) / |去年同期净利|
        #    一个更严谨的版本需要TTM数据或分析师预期数据。
        income_df_long['surprise'] = income_df_long.groupby('ts_code')['net_profit_ttm'].pct_change(periods=4)

        ann_surprise_long = income_df_long[['ts_code', 'ann_date', 'surprise']].dropna()
        ann_surprise_long['ann_date'] = pd.to_datetime(ann_surprise_long['ann_date'])

        # 3. 将稀疏的“盈利意外”事件，广播成每日因子值
        pead_series = ann_surprise_long.set_index(['ann_date', 'ts_code'])['surprise']
        pead_wide_sparse = pead_series.unstack()

        trading_dates = self.factor_manager.data_manager.trading_dates
        pead_daily_df = _broadcast_ann_date_to_daily(pead_wide_sparse, trading_dates)

        return pead_daily_df
    ##
    # 核心逻辑：分析师评级调整反映的是“聪明钱”对公司基本面预期的持续改善。
    #
    # 市场如何反映“持续改善”：一家基本面持续向好的公司，它的股价走势通常不是暴涨暴跌，而是稳步、持续地上涨。这种上涨通常伴随着较低的波动。这被称为“高质量的动量”。
    #
    # 风险调整后动量 /计算风险调整后的动量。#
    def _calculate_quality_momentum(self) -> pd.DataFrame:
        """
        计算风险调整后的动量因子 (Quality Momentum)。
        逻辑: 120日动量 / 90日波动率。
        作为“分析师评级上调”的代理，寻找那些稳步上涨的股票。
        """
        logger.info("    > 正在计算代理因子: Quality Momentum...")

        # 1. 获取动量和波动率因子
        momentum_120d = self.factor_manager.get_raw_factor('momentum_120d')
        volatility_90d = self.factor_manager.get_raw_factor('volatility_90d')

        # 2. 对齐数据
        mom_aligned, vol_aligned = momentum_120d.align(volatility_90d, join='inner', axis=None)

        # 3. 风险控制：波动率必须为正
        vol_aligned_safe = vol_aligned.where(vol_aligned > 0)

        # 4. 计算风险调整后的动量
        quality_momentum_df = mom_aligned / vol_aligned_safe

        quality_momentum_df = quality_momentum_df.replace([np.inf, -np.inf], np.nan)
        return quality_momentum_df
    ######################

    ##辅助函数
    #ok
    def _calculate_market_pct_chg(self, index_code) -> pd.Series:
        """【新增】根据指定的指数代码，计算其总回报收益率。"""
        """
           【V2.0 - 权威版】
           根据指数的不复权点位和分红数据，计算真实的总回报收益率。
           确保与个股pct_chg的计算逻辑完全统一。
           """
        logger.info(f"  > 正在基于第一性原理，计算市场基准 [{index_code}] 的权威pct_chg...")

        # --- 1. 获取最基础的原材料 ---
        # a) 获取指数的不复权日线数据 (需要你的DataManager支持)

        # b) 获取指数的分红事件 (需要你的DataManager支持)
        #    对于宽基指数，Tushare通常在 index_daily 接口中直接提供总回报的pct_chg
        #    但最严谨的做法，是获取其对应的ETF的分红数据，或使用总回报指数
        #    这里我们做一个简化，直接使用Tushare index_daily 中那个质量较高的pct_chg
        #    这是一种在严谨性和工程便利性上的权衡。

        index_daily_total_return = load_index_daily(index_code)
        market_pct_chg = index_daily_total_return['pct_chg'] / 100.0

        # 确保返回的Series有名字，便于后续join
        market_pct_chg.name = index_code

        return market_pct_chg
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
        ttm_df = self.factor_manager.get_raw_factor(ttm_factor_name)

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
        snapshot_long_df = financial_df[['ts_code', 'ann_date', 'end_date', source_column]].copy(deep=True)
        snapshot_long_df=snapshot_long_df.dropna(inplace=False)
        if snapshot_long_df.empty:
            raise ValueError(f"警告: 计算因子 {factor_name} 时，从 {source_column} 字段未获取到有效数据。")

        # 步骤二：透视
        snapshot_long_df=snapshot_long_df.sort_values(by=['ts_code', 'end_date'], inplace=False).copy(deep=True)
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

        financial_df = data_loader_func().copy(deep=True)

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
        single_q_long_df=single_q_long_df.dropna(subset=[single_q_col_name, 'ann_date'], inplace=False)  # 确保公告日和计算值都存在

        return single_q_long_df
    #ok 能对上 聚宽数据
    def _calculate_pct_chg(self) -> pd.DataFrame:
        close_hfq = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)
        ret  = close_hfq.pct_change()
        return  ret

    #daily_hfq亲测 后复权的close是可用的，因为涨跌幅跟聚宽一模一样！ 我们直接用，不需要下面这样复杂的计算！
    # #ok 对的上daily的pct_chg字段（ pct_chg, float, 涨跌幅【基于除权后的昨收计算的涨跌幅：（今收-除权昨收）/除权昨收
    # #也能和 t_bao_pct_chg 计算出来的数据对上！
    # def _calculate_pct_chg(self) -> pd.DataFrame:
    #     """
    #        根据“总回报恒等式”，直接从不复权价和分红送股事件计算真实总回报率。
    #        """
    #     logger.info("  > 正在基于第一性原理，计算【最终版】权威 pct_chg...")
    # 
    #     close_raw = self.factor_manager.get_raw_factor('close_raw')
    #     pre_close_raw = close_raw.shift(1)
    #     dividend_events = load_dividend_events_long()
    # 
    #     # 【调试输出】
    #     logger.info(f"  > close_raw形状: {close_raw.shape}")
    # 
    #     # 构建分红矩阵（未对齐）
    #     cash_div_matrix_raw = dividend_events.pivot_table(index='ex_date', columns='ts_code',
    #                                                       values='cash_div_tax').reindex(close_raw.index).fillna(0)
    #     stk_div_matrix_raw = dividend_events.pivot_table(index='ex_date', columns='ts_code', values='stk_div').reindex(
    #         close_raw.index).fillna(0)
    # 
    #     logger.info(f"  > 分红矩阵原始形状: cash_div={cash_div_matrix_raw.shape}, stk_div={stk_div_matrix_raw.shape}")
    # 
    #     # 【关键修复】强制对齐到close_raw的列，避免形状不匹配
    #     target_stocks = close_raw.columns
    #     cash_div_matrix = cash_div_matrix_raw.reindex(columns=target_stocks, fill_value=0)
    #     stk_div_matrix = stk_div_matrix_raw.reindex(columns=target_stocks, fill_value=0)
    # 
    #     logger.info(f"  > 对齐后形状: cash_div={cash_div_matrix.shape}, stk_div={stk_div_matrix.shape}")
    # 
    #     # 验证形状一致性
    #     assert close_raw.shape == cash_div_matrix.shape == stk_div_matrix.shape, \
    #         f"形状不匹配: close_raw={close_raw.shape}, cash_div={cash_div_matrix.shape}, stk_div={stk_div_matrix.shape}"
    # 
    #     # 核心公式: (今日收盘价 * (1 + 送股比例) + 每股派息) / 昨日收盘价 - 1
    #     numerator = close_raw * (1 + stk_div_matrix) + cash_div_matrix
    #     true_pct_chg = numerator / pre_close_raw - 1
    # 
    #     logger.info(f"  > 最终结果形状: {true_pct_chg.shape}")
    # 
    #     final_pct_chg = true_pct_chg.where(close_raw.notna())
    #     return final_pct_chg

    # # 涨跌幅能对的上
    # def _calculate_close_hfq(self) -> pd.DataFrame:
    #     """
    #     【return 后复权 close】
    #     使用真实的“总回报率”和“不复权收盘价”来计算后复权价格序列。
    #     """
    #     # 1. 获取最关键的两个输入数据
    #     true_pct_chg = self.factor_manager.get_raw_factor('pct_chg')  # 我们之前计算的真实总回报率 (涨跌幅)
    #     close_raw = self.factor_manager.get_raw_factor('close_raw')  # 当天真实价格 (不复权)
    #
    #     # 2. 处理边界情况：如果输入为空，则返回空DataFrame
    #     if close_raw.empty:
    #         raise  ValueError('价格data为空')
    #
    #     # 3. 计算每日的增长因子 (1 + 收益率)
    #     # 第一天的pct_chg是NaN，因为没有前一日的数据
    #     growth_factor = 1 + true_pct_chg
    #
    #     # 4. 使用.cumprod()计算自第一天以来的累积收益因子
    #     # cumprod() 会自动忽略开头的NaN值，从第一个有效数字开始累乘
    #     cumulative_growth_factor = growth_factor.cumprod()
    #
    #     # 5. 获取计算的基准价格 (即第一天的真实收盘价)
    #     base_price = close_raw.iloc[0]
    #
    #     # 6. 后复权价 = 基准价格 * 累积收益因子
    #     close_hfq = base_price * cumulative_growth_factor
    #
    #     # 7. 【关键修正】第一天的累积收益因子是NaN，导致第一天的后复权价也是NaN。
    #     # 我们必须将其修正为基准价格本身。
    #     close_hfq.iloc[0] = base_price
    #
    #     return close_hfq

    # def _calculate_close_hfq(self) -> pd.DataFrame:
    #     """
    #     【return 后复权 close
    #     """
    #     true_pct_chg = self.factor_manager.get_raw_factor('pct_chg')#我们刚才讨论的总回报率 涨跌幅
    #     close_raw = self.factor_manager.get_raw_factor('close_raw')#当天真实价格

    def _calculate_hfq_adj_factor(self) -> pd.DataFrame:
        close_raw  = self.factor_manager.get_raw_factor('close_raw').copy(deep=True)
        close_hfq  = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)
        ret = close_hfq/close_raw
        return ret
    def _calculate_vol_hfq(self) -> pd.DataFrame:
        ##
        # 复权成交量 = 原始成交量 / 复权因子
        #
        # # 其中复权因子 = 复权价格 / 原始价格
        # 复权因子 = close_adj / close_raw
        # vol_adj = vol_raw / 复权因子#
        """【V3.0 - 统一版】根据通用复权乘数计算【反向】复权成交量"""
        vol_raw = self.factor_manager.get_raw_factor('vol_raw')
        hfq_adj_factor = self.factor_manager.get_raw_factor('hfq_adj_factor')
        # 价格乘数是 < 1 的“折扣”，所以成交量要【除以】它，实现反向调整
        ##
        # 为什么 vol (成交量) 是要反向的？（for：确保跨时间历史数据的可比性
        # 一句话概括：因为vol（成交量）是一个实物数量（Physical Quantity）指标，它会因为送股、拆股等股本变化事件而发生剧烈跳变，导致历史数据和当前数据完全没有可比性，因此必须调整。#
        ##
        #
        # 比如10元一股 1000量 成交额1w
        #  次日因为分红，把现金给出去。股价：1元 ，但是每天成交额都是1w左右 （基于这个假设。 所以今天量：10000的量，显然无法跟昨天相比，跨倍数太多！。所以需要除以复权因子
        #
        # #
        ret =  vol_raw/hfq_adj_factor
        return ret

    #ok
    # def _calculatesss_adj_factor(self) -> pd.DataFrame:
    #     """1
    #     【V2.0 - 最终生产版】
    #     根据不复权价格和分红送股事件，从头开始计算一个严格“随时点”的
    #     累积复权因子。此版本已修复停牌对前收盘价获取的bug。
    #     """
    #     logger.info("  > 正在基于第一性原理，计算权威的 adj_factor (V2.0)...")
    #
    #     # --- 1. 获取原材料 ---
    #     # 最后一个有效价格填补停牌期间的NaN ，明确需要 必须无脑ffill，那怕是十年前的收盘价格！
    #
    #     close_raw_filled = self.factor_manager.get_raw_factor('close_raw_ffill')
    #     # 然后，在这个“无空洞”的价格序列上，获取前一天的价格
    #     pre_close_raw_robust = close_raw_filled.shift(1)
    #     dividend_events = load_dividend_events_long()
    #
    #     # --- 2. 计算【每日】的调整比例 ---
    #     daily_adj_ratio = pd.DataFrame(1.0, index=close_raw_filled.index, columns=close_raw_filled.columns)
    #
    #     for _, event in dividend_events.iterrows():
    #         event_date, evet_stock_code = event['ex_date'], event['ts_code']
    #         if event_date in daily_adj_ratio.index and evet_stock_code in daily_adj_ratio.columns:
    #             cash_div = event.get('cash_div_tax', 0)
    #             stk_div = event.get('stk_div', 0)
    #
    #             # 【修正】使用我们新计算的、更稳健的前收盘价
    #             prev_close = pre_close_raw_robust.at[event_date, evet_stock_code]
    #
    #             if pd.isna(prev_close) or prev_close <= 0 or (cash_div == 0 and stk_div == 0):
    #                 continue
    #
    #             numerator = prev_close - cash_div
    #             denominator = prev_close * (1 + stk_div)
    #
    #             if denominator > 0:
    #                 daily_adj_ratio.at[event_date, evet_stock_code] = numerator / denominator
    #
    #     # --- 3. 计算【累积】复权因子 ---
    #     daily_adj_ratio.replace(0, np.nan, inplace=True) # 因为 numerator = prev_close - cash_div极端情况会==0.这里对修复
    #     daily_adj_ratio.fillna(1.0, inplace=True) #对上面那行产生的nan 填充成1 ，1是干净的，对后续计算没有影响的 （因为我们下面是是累计乘法！
    #     adj_factor_df = daily_adj_ratio.cumprod(axis=0)
    #
    #     logger.info("  > ✓ 权威的 adj_factor (V2.0) 计算完成。")
    #     return adj_factor_df


    #填充好 ，供于重复使用！ （目前场景 计算cci 要求必须是连续的价格数据！且是后复权


    def _calculate_close_hfq_filled(self,limit: int) -> pd.DataFrame:
        open_hfq_unfilled = self.factor_manager.get_raw_factor('close_hfq').copy(deep=True)
        return open_hfq_unfilled.ffill(limit=limit)

    def _calculate_open_hfq_filled(self,limit: int ) -> pd.DataFrame:
        open_hfq_unfilled = self.factor_manager.get_raw_factor('open_hfq').copy(deep=True)
        return open_hfq_unfilled.ffill(limit=limit)

    def _calculate_high_hfq_filled(self,limit: int ) -> pd.DataFrame:
        open_hfq_unfilled = self.factor_manager.get_raw_factor('high_hfq').copy(deep=True)
        return open_hfq_unfilled.ffill(limit=limit)

    def _calculate_low_hfq_filled(self,limit: int ) -> pd.DataFrame:
        open_hfq_unfilled = self.factor_manager.get_raw_factor('low_hfq').copy(deep=True)
        return open_hfq_unfilled.ffill(limit=limit)

    ##基础换算！
    def _calculate_circ_mv(self):
        circ_mv = self.factor_manager.data_manager.raw_dfs['circ_mv'].copy(deep=True)#这里会递归啊，所以一定要开缓存，这样下此调用会走缓存！
        circ_mv = circ_mv * 10000
        return circ_mv
    def _calculate_total_mv(self):
        total_mv = self.factor_manager.data_manager.raw_dfs['total_mv'].copy(deep=True)#这里会递归啊，所以一定要开缓存，这样下此调用会走缓存！
        total_mv = total_mv * 10000
        return total_mv
    def _calculate_amount(self):
        amount = self.factor_manager.data_manager.raw_dfs['amount'].copy(deep=True)#这里会递归啊，所以一定要开缓存，这样下此调用会走缓存！
        amount = amount * 1000
        return amount
    def _calculate_turnover_rate(self):
        turnover_rate = self.factor_manager.data_manager.raw_dfs['turnover_rate'].copy(deep=True)#这里会递归啊，所以一定要开缓存，这样下此调用会走缓存！
        turnover_rate = turnover_rate / 100
        return turnover_rate
    ###标准内部件


    ##
    #  目前用于 计算adj_factor 必须是ffill#
    def _calculate_close_raw_ffill(self):
        ret = self.factor_manager.get_raw_factor('close_raw').copy(deep=True)
        return ret.ffill()

    ##
    # 估值与市值类 (每日更新)
    # 字段: ps_ttm, total_mv, circ_mv, pb, pe_ttm
    #
    # 金融含义: NaN代表停牌，或指标本身无效（如PE为负）。这些是**“状态类”**数据。
    #
    # 下游需求: 因子计算和中性化时，我们希望尽可能有多的有效数据点。
    #
    # 填充策略: FILL_STRATEGY_FFILL_LIMIT_65 (有限前向填充)。
    #
    # 理由:
    #
    # 经济学假设：一个公司在停牌期间，其估值水平和市值，最合理的估计就是它停牌前的状态。ffill完美地符合这个假设。
    #
    # 风险控制：我们不希望这个假设无限期地延续。如果一只股票停牌超过一个季度（约65个交易日），我们就认为它停牌前的信息已经“陈腐”，不再具有代表性。limit=65正是为了控制这个风险。#
    # def _calculate_ps_ttm_fill_limit65(self):
    #     return self.factor_manager.get_raw_factor('ps_ttm').ffill(limit=65)
    def _calculate_total_mv_fill_limit65(self):
        return self.factor_manager.get_raw_factor('total_mv').copy(deep=True).ffill(limit=65)

    def _calculate_circ_mv_fill_limit65(self):
        return self.factor_manager.get_raw_factor('circ_mv').copy(deep=True).ffill(limit=65)

    # def _calculate_pb_fill_limit65(self):
    #     return self.factor_manager.get_raw_factor('pb').ffill(limit=65)
    # def _calculate_pe_ttm_fill_limit65(self):
    #     return self.factor_manager.get_raw_factor('pe_ttm').ffill(limit=65)

    ##
    # 交易行为类
    # 字段: turnover_rate, amount
    #
    # 金融含义: NaN代表停牌，当天没有发生任何交易行为。
    #
    # 下游需求: 滚动计算流动性因子时，需要处理这些NaN。
    #
    # 填充策略: FILL_STRATEGY_ZERO (填充为0)。
    #
    # 理由:
    #
    # 经济学假设：这是最符合事实的假设。停牌 = 0成交量 = 0成交额 = 0换手率。
    #
    # ffill的危害：如果你对turnover_rate进行ffill，就等于错误地假设“停牌日的热度=停牌前一天”，这与事实完全相反。#
    ##        # 金融逻辑：停牌日的真实换手率就是0
    def _calculate_turnover_rate_fill_zero(self):
        """【标准件】生产一个将停牌日NaN处理为0的换手率序列"""
        turnover_df = self.factor_manager.get_raw_factor('turnover_rate')
        return turnover_df .fillna(0)
    def _calculate_amount_fill_zero(self):
        return self.factor_manager.get_raw_factor('amount').fillna(0)


    ##
    #  静态信息类
    # 字段: list_date, delist_date
    #
    # 金融含义: NaN代表股票还未上市或尚未退市。
    #
    # 下游需求: 确定股票的生命周期。
    #
    # 填充策略: FILL_STRATEGY_FFILL (无限制前向填充)。
    #
    # 理由: 一只股票的上市日期是一个永恒不变的事实。一旦这个信息出现，它就对该股票的整个生命周期有效。因此，使用无限制的ffill将这个事实广播到所有后续的日期，是完全正确的。#
    def _calculate_delist_date_raw_ffill(self):
        return self.factor_manager.get_raw_factor('delist_date').copy(deep=True).ffill()
    def _calculate_list_date_ffill(self):
        return self.factor_manager.get_raw_factor('list_date').copy(deep=True).ffill()


    # ###标准件
    #
    # def _calculate_price_adj_multiplier(self) -> pd.DataFrame:
    #     """
    #     【新增核心组件】
    #     根据权威的 close_hfq 和 close_raw，计算每日通用的复权乘数。
    #     这是所有其他复权价和复权量的基础。
    #     """
    #     logger.info("  > 正在生产核心标准件: price_adj_multiplier...")
    #
    #     # 依赖于我们之前已经定义好的、最权威的两个“标准件”
    #     close_hfq = self.factor_manager.get_raw_factor('close_hfq')
    #     close_raw = self.factor_manager.get_raw_factor('close_raw')
    #
    #     # 为防止除零错误，在close_raw为0的地方返回NaN
    #     price_adj_multiplier = close_hfq.div(close_raw).where(close_raw > 0)
    #
    #     return price_adj_multiplier
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
    filled_df = sparse_wide_df.reindex(combined_index).copy(deep=True).ffill()

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
# bm_factor = fm.get_raw_factor('bm_ratio')

# # 第二次获取时，它会直接从 FactorManager 的缓存加载，速度极快
# bm_factor_again = fm.get_raw_factor('bm_ratio')

# print("\n最终得到的 bm_ratio 因子:")
# print(bm_factor.head())

# 【已删除】standardize_cross_sectionally 函数
# 原因：因子计算阶段不应该进行标准化，应该在预处理阶段统一处理
# 如果需要截面标准化，请使用 FactorProcessor._standardize_robust 方法


def calculate_rolling_beta_pure(
        stock_returns: pd.DataFrame,
        market_returns: pd.Series,
        window: int = 60,
        min_periods: int = 20
) -> pd.DataFrame:
    """
    【】根据输入的个股和市场收益率，计算滚动Beta。
    这是一个独立的计算引擎，不涉及任何数据加载或预处理。

    Args:
        stock_returns (pd.DataFrame): 个股收益率矩阵 (index=date, columns=stock)。
        market_returns (pd.Series): 市场收益率序列 (index=date)。
        window (int): 滚动窗口大小（天数）。
        min_periods (int): 窗口内计算所需的最小观测数。

    Returns:
        pd.DataFrame: 滚动Beta矩阵 (index=date, columns=stock)，未做任何日期截取或移位。
    """
    logger.info(f"  > 开始执行滚动Beta计算 (窗口: {window}天)...")

    # --- 1. 数据对齐 ---
    # 使用 'left' join 确保所有股票的日期和市场收益率对齐
    # 这是计算逻辑的核心部分，必须保留
    combined_df = stock_returns.join(market_returns.rename('market_return'), how='left')
    market_returns_aligned = combined_df.pop('market_return')

    # --- 2. 滚动计算 ---
    # Beta = Cov(R_stock, R_market) / Var(R_market)

    # a) 计算滚动协方差
    # 在对齐后，combined_df 就是我们要的 stock_returns
    rolling_cov = combined_df.rolling(window=window, min_periods=min_periods).cov(market_returns_aligned)

    # b) 计算市场收益率的滚动方差
    rolling_var = market_returns_aligned.rolling(window=window, min_periods=min_periods).var()

    # c) 计算滚动Beta
    beta_df = rolling_cov.div(rolling_var, axis=0)

    return beta_df