"""
量化回测器 - 专业级因子策略回测工具

核心功能：
1. 多因子策略回测对比
2. 真实交易成本建模
3. 完整的风险调整收益分析
4. 可视化和报告生成
5. 实盘级数据对齐和验证

设计理念：
- 面向对象，可扩展
- 数据安全，严格对齐验证
- 交易成本真实建模
- 结果可复现，可解释
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import warnings
from datetime import datetime

from vectorbt.portfolio import CallSeqType, SizeType

from quant_lib.config.logger_config import setup_logger
from quant_lib.rebalance_utils import generate_rebalance_dates
from utils.math.math_utils import convert_to_sequential_percents

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class BacktestConfig:
    """回测配置参数"""
    # 策略参数
    top_quantile: float = 0.2  # 做多分位数阈值（前20%）
    rebalancing_freq: str = 'W'  # 调仓频率 ('M'=月末, 'W'=周末, 'Q'=季末)

    # 交易成本参数
    commission_rate: float = 0.0003  # 佣金费率（万3）
    slippage_rate: float = 0.0010  # 滑点率（千1）
    stamp_duty: float = 0.0010  # 印花税（单边，卖出收取）
    min_commission: float = 5.0  # 最小佣金（元）

    # 回测参数
    initial_cash: float = 1000000.0  # 初始资金（100万）
    max_positions: int = 10  # 最大持仓数量

    # 风控参数
    max_weight_per_stock: float = 0.10  # 单股最大权重（10%）
    min_weight_threshold: float = 0.01  # 最小权重阈值（1%）

    # 数据验证参数
    min_data_coverage: float = 0.8  # 最小数据覆盖率
    max_missing_consecutive_days: int = 5  # 最大连续缺失天数

    #股票持有信息：
    max_holding_days: int= 45


class DataValidator:
    """数据验证器"""

    @staticmethod
    def validate_dataframes(price_df: pd.DataFrame, *factor_dfs: pd.DataFrame) -> Dict[str, any]:
        """
        验证价格数据和因子数据的一致性
        
        Args:
            price_df: 价格数据
            factor_dfs: 因子数据列表
            
        Returns:
            Dict: 验证结果统计
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        try:
            # 检查索引类型
            if not isinstance(price_df.index, pd.DatetimeIndex):
                validation_results['errors'].append("价格数据索引必须是DatetimeIndex")
                validation_results['is_valid'] = False

            # 检查数据对齐
            reference_index = price_df.index
            reference_columns = price_df.columns

            for i, factor_df in enumerate(factor_dfs):
                factor_name = f"因子{i + 1}"

                # 索引对齐检查
                if not factor_df.index.equals(reference_index):
                    missing_dates = reference_index.difference(factor_df.index)
                    if len(missing_dates) > 0:
                        validation_results['warnings'].append(
                            f"{factor_name}缺失{len(missing_dates)}个交易日"
                        )

                # 列对齐检查
                if not factor_df.columns.equals(reference_columns):
                    missing_stocks = reference_columns.difference(factor_df.columns)
                    if len(missing_stocks) > 0:
                        validation_results['warnings'].append(
                            f"{factor_name}缺失{len(missing_stocks)}只股票"
                        )

            # 统计信息
            validation_results['stats'] = {
                'date_range': (price_df.index.min(), price_df.index.max()),
                'trading_days': len(price_df.index),
                'stock_count': len(price_df.columns),
                'data_coverage': (1 - price_df.isnull().sum().sum() / price_df.size) * 100
            }

            logger.info(f"数据验证完成: {validation_results['stats']}")

        except Exception as e:
            validation_results['errors'].append(f"验证过程异常: {str(e)}")
            validation_results['is_valid'] = False

        return validation_results

    @staticmethod
    def align_dataframes(price_df: pd.DataFrame, *factor_dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        对齐价格数据和因子数据
        
        Args:
            price_df: 价格数据
            factor_dfs: 因子数据列表
            
        Returns:
            Tuple: 对齐后的数据框列表
        """
        # 使用vectorbt的对齐功能
        aligned_data = vbt.base.reshape_fns.broadcast(price_df, *factor_dfs,
                                                      keep_pd=True,
                                                      align_index=True,
                                                      align_columns=True)

        logger.info(f"数据对齐完成，最终维度: {aligned_data[0].shape}")
        return aligned_data


class StrategySignalGenerator:
    """策略信号生成器"""

    @staticmethod
    def generate_long_signals(
            factor_df: pd.DataFrame,
            config: BacktestConfig
    ) -> pd.DataFrame:
        """
        生成做多信号

        Args:
            factor_df: 因子数据
            config: 回测配置

        Returns:
            pd.DataFrame: 持仓信号（True=持有，False=不持有）
        """
        # 1. 计算每日排名百分位（越大排名越靠前）
        ranks = factor_df.rank(axis=1, pct=True, method='average', na_option='keep')

        # 2. 确定做多信号（排名在前top_quantile）
        long_signals_raw = ranks >= (1 - config.top_quantile)

        # 3. 按调仓频率进行重采样
        # .resample().last() 获取每个周期最后一天的信号
        # .reindex().ffill() 将信号前填充到每个交易日
        rebalance_signals = long_signals_raw.resample(config.rebalancing_freq).last()
        final_signals = rebalance_signals.reindex(factor_df.index, method='ffill')

        # 4. 控制最大持仓数量
        if config.max_positions > 0:
            final_signals = StrategySignalGenerator._limit_positions(
                final_signals, ranks, config.max_positions
            )

        logger.info(f"信号生成完成，平均持仓数: {final_signals.sum(axis=1).mean():.1f}")
        return final_signals.fillna(False)

    @staticmethod
    def _limit_positions(signals: pd.DataFrame, ranks: pd.DataFrame, max_positions: int) -> pd.DataFrame:
        """
        限制最大持仓数量，优先选择排名最高的股票
        
        Args:
            signals: 原始信号
            ranks: 排名数据
            max_positions: 最大持仓数
            
        Returns:
            pd.DataFrame: 限制后的信号
        """
        limited_signals = pd.DataFrame(False, index=signals.index, columns=signals.columns)

        for date in signals.index:
            date_signals = signals.loc[date]
            date_ranks = ranks.loc[date]

            if date_signals.sum() > max_positions:
                # 选择排名最高的max_positions只股票
                valid_stocks = date_signals[date_signals].index
                top_stocks = date_ranks[valid_stocks].nlargest(max_positions).index
                limited_signals.loc[date, top_stocks] = True
            else:
                limited_signals.loc[date] = date_signals

        return limited_signals



    @staticmethod
    def generate_long_holding_signals(factor_df: pd.DataFrame, price_df, config: BacktestConfig) -> pd.DataFrame:
        """
        生成每日目标"持仓"布尔矩阵，确保满仓运作
        """
        # 计算每日排名
        ranks = factor_df.rank(axis=1, pct=True, method='average', na_option='keep')

        # 生成每日持仓信号，而不是只在调仓日
        daily_holding_signals = pd.DataFrame(False, index=factor_df.index, columns=factor_df.columns)
        # 获取调仓日期
        rebalance_dates = ranks.copy().reindex(generate_rebalance_dates(ranks.index,config.rebalancing_freq)).dropna(how='all').index

        # 当前持仓组合（在调仓间隔期间保持不变）
        current_positions = None

        for date in factor_df.index:
            # 检查是否为调仓日
            is_rebalance_day = date in rebalance_dates

            if is_rebalance_day:
                # 调仓日：重新选择股票
                daily_valid_ranks = ranks.loc[date].dropna()

                if len(daily_valid_ranks) > 0:
                    # 计算目标持仓数
                    num_to_select = int(np.ceil(len(daily_valid_ranks) * config.top_quantile))
                    if config.max_positions:
                        num_to_select = min(num_to_select, config.max_positions)

                    # 选择排名最高的股票
                    chosen_stocks = daily_valid_ranks.nlargest(num_to_select).index
                    current_positions = chosen_stocks
                    # logger.info(f"调仓日{date.strftime('%Y-%m-%d')}: 选择{len(chosen_stocks)}只股票")

            if current_positions is not None: #其实就是变相的ffill ，保持这次调仓及后面n天同状态 ，直到下一次调仓！
                #最新注释，交给下游 去判断
                # # 检查股票 是否可交易==>（有价格数据）
                # current_with_price_positions = price_df.loc[date, current_positions].notna()
                # tradable_positions = current_positions[current_with_price_positions]
                daily_holding_signals.loc[date, current_positions] = True

        # 验证持仓信号质量
        daily_positions = daily_holding_signals.sum(axis=1)
        avg_positions = daily_positions.mean()
        zero_position_days = (daily_positions == 0).sum()

        logger.info(f"  平均每日持仓数: {avg_positions:.1f}")
        logger.info(f"  零持仓天数: {zero_position_days}/{len(daily_positions)}")
        logger.info(f"  持仓覆盖率: {(1 - zero_position_days / len(daily_positions)):.1%}")

        if zero_position_days > len(daily_positions) * 0.1:  # 超过10%的日子没有持仓
            logger.warning(f"⚠️ 持仓信号质量差：{zero_position_days}天无持仓")
        else:
            logger.info(f"✅ 持仓信号质量良好")

        return daily_holding_signals

    @staticmethod
    def generate_rebalancing_signals(holding_signals: pd.DataFrame, force_exit_limit: int = None) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        将"持仓"矩阵转换为精确的"买入"和"卖出"信号矩阵。
        - 增加了强制类型转换，彻底解决 'invert' ufunc TypeError 问题。
        - 新增60天强制卖出逻辑 慎用！ 保持函数单一
        """

        # 【核心修正】在使用 ~ 操作符之前，进行严格的类型和空值处理

        # 1. 确保当前持仓信号是布尔型
        current_holdings = holding_signals.astype(bool)

        # 2. 对前一天的持仓信号，先填充移位产生的NaN，再强制转为布尔型
        prev_holdings = holding_signals.vbt.fshift(1).fillna(False).astype(bool)

        # 3. 基础买卖信号
        entries = current_holdings & ~prev_holdings
        exits = ~current_holdings & prev_holdings

        # 强制卖出逻辑 - 用于调试交易执行问题
        forced_exits=None
        # logger.info("  -> 正在添加60天强制卖出逻辑...")
        if force_exit_limit:
            # 创建持仓天数计数器
            holding_days = pd.DataFrame(0, index=holding_signals.index, columns=holding_signals.columns)
            forced_exits = pd.DataFrame(False, index=holding_signals.index, columns=holding_signals.columns)
            # for循环 填充天数！ 以及判断天数超过limit 给出强制卖出信号
            for i in range(1, len(holding_signals)):
                # 对于持续持有的股票，天数+1
                continuing_holds = current_holdings.iloc[i] & prev_holdings.iloc[i]
                holding_days.iloc[i] = np.where(continuing_holds,
                                                holding_days.iloc[i - 1] + 1,
                                                0)

                # 对于新买入的股票，天数重置为1
                new_entries = entries.iloc[i]  # 这个为true，那么昨天一定是false，没毛病
                holding_days.iloc[i] = np.where(new_entries, 1, holding_days.iloc[i])

                # 强制卖出持有超过180天的股票 (180天)
                force_exit_mask = holding_days.iloc[i] >= force_exit_limit
                forced_exits.iloc[i] = force_exit_mask & current_holdings.iloc[i]
                logger.info(f"  -> 强制超过{force_exit_limit}天卖出触发次数: {forced_exits.sum().sum()}")

        final_exits = exits
        # 5. 合并原有卖出信号和强制卖出信号
        if force_exit_limit:
            final_exits = exits | forced_exits
        return entries, final_exits


class TradingCostCalculator:
    """交易成本计算器 - 重构版本"""

    @staticmethod
    def get_single_side_costs(config: BacktestConfig) -> Dict[str, float]:
        """
        获取单边交易成本
        
        Args:
            config: 回测配置
            
        Returns:
            Dict: 单边成本字典
        """
        # 基础单边成本 (佣金)
        base_commission = config.commission_rate

        # 单边滑点
        single_slippage = config.slippage_rate

        # 印花税只在卖出时收取，我们将其分摊到买卖两边
        # 或者包含在一个稍高的综合费率中
        adjusted_commission = base_commission + (config.stamp_duty / 2)  # 分摊印花税

        costs = {
            'commission': adjusted_commission,
            'slippage': single_slippage,
            'combined_fee': adjusted_commission  # vectorbt的fees参数
        }

        logger.info(f"单边交易成本: 佣金{adjusted_commission:.4f}, 滑点{single_slippage:.4f}")
        return costs


class QuantBacktester:
    """量化回测器主类"""

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回测器
        
        Args:
            config: 回测配置，None时使用默认配置
        """
        self.config = config or BacktestConfig()
        self.validator = DataValidator()
        self.signal_generator = StrategySignalGenerator()
        self.cost_calculator = TradingCostCalculator()

        # 存储回测结果
        self.portfolios: Dict[str, any] = {}
        self.validation_results: Dict = {}

        logger.info("QuantBacktester初始化完成")
        logger.info(f"配置参数: {self.config}")

    def prepare_data(
            self,
            price_df: pd.DataFrame,
            factor_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        准备和验证回测数据
        
        Args:
            price_df: 价格数据
            factor_dict: 因子数据字典 {"因子名": DataFrame}
            
        Returns:
            Tuple: (对齐后的价格数据, 对齐后的因子数据字典)
        """
        logger.info(f"开始准备数据，价格数据维度: {price_df.shape}")
        logger.info(f"因子数量: {len(factor_dict)}")

        # 验证数据
        factor_dfs = list(factor_dict.values())
        self.validation_results = self.validator.validate_dataframes(price_df, *factor_dfs)

        if not self.validation_results['is_valid']:
            raise ValueError(f"数据验证失败: {self.validation_results['errors']}")

        if self.validation_results['warnings']:
            for warning in self.validation_results['warnings']:
                logger.warning(warning)

        # 对齐数据
        aligned_data = self.validator.align_dataframes(price_df, *factor_dfs)
        aligned_price = aligned_data[0]
        aligned_factors = {
            name: aligned_data[i + 1]
            for i, name in enumerate(factor_dict.keys())
        }

        logger.info(f"数据准备完成，最终维度: {aligned_price.shape}")
        return aligned_price, aligned_factors

    def _generate_improved_signals(
            self,
            holding_signals: pd.DataFrame,
            price_df: pd.DataFrame,
            max_holding_days: int = None,
            retry_buy_limit: int = 3  # 买入重试的有效期（天数）
    ):
        """
               -   a_holdings 状态变量，精确追踪每日真实持仓。

               - 彻底解决“负数持仓”、“僵尸信号”、“回声信号”等所有状态管理Bug。
               - 这是在 for 循环框架内最稳健的实现。
               buy_num_should_base_on_sell_num: 卖出去多少只，决定能买入多少只（场景：比如一天10股票都停牌，你哪里还有钱买入！ 今日可买入数量=前一天空余可购入股票数量+今天已经卖出数量！ 思考明天接着买吗？不接着买：那这只股票可能就错过了！接着买：有可能极端情况20天后才买入！，但是已经很晚了！也是有问题！ 你有什么好的方案吗
               #最新实测结论：buy_num_should_base_on_sell_num 我想多了，vector 已经帮我想到了！，自己做了判断，买不进去 不会强行买的！！！！
               #问题：但是vector默认是不会再次购入的，扔了多可惜 解决办法：
               - 新增“待买清单”逻辑，处理因停牌或资金不足而失败的买入信号。
                - 信号具有“有效期”，过期作废。 （最多只让重试n天
               """
        logger.info(f"【V7 最终版】开始生成信号，最大持仓: {max_holding_days}, 买入重试期: {retry_buy_limit}天")

        # --- 初始化 ---
        entries = pd.DataFrame(False, index=holding_signals.index, columns=holding_signals.columns)
        exits = pd.DataFrame(False, index=holding_signals.index, columns=holding_signals.columns)

        # --- 状态变量 ---
        # 【核心】“实际持仓”状态，作为循环内部的地面实况
        actual_holdings = pd.Series(False, index=holding_signals.columns)

        # 持仓天数计数器
        holding_days = pd.Series(0, index=holding_signals.columns)

        # “待卖清单”，追踪需要卖出但可能被延迟的股票
        pending_exits_tracker = pd.Series(False, index=holding_signals.columns)

        # “待买清单”和任务有效期计数器
        pending_buys_tracker = pd.Series(False, index=holding_signals.columns)
        pending_buys_age = pd.Series(0, index=holding_signals.columns)

        # --- 逐日循环生成信号 ---
        for i in range(len(holding_signals)):
            curr_holdings_plan = holding_signals.iloc[i]

            # --- 1. 状态更新（每日开始时）---
            # a. 对“待买清单”上的任务进行“老化”，并作废“过期”任务
            if pending_buys_tracker.any():
                pending_buys_age[pending_buys_tracker] += 1
                expired_buys = pending_buys_age > retry_buy_limit
                pending_buys_tracker[expired_buys] = False #不用再买入了
                pending_buys_age[expired_buys] = 0# 年龄也跟着置为0 反正不买了

            # b. 更新持仓天数: 只为实际持有的股票累加天数
            holding_days[actual_holdings] += 1   #有点跳跃。疑问：万一今天是卖出信号，你这里加一没有影响吗？答：确实没有，因为最后：holding_days[actual_holdings=0

            # --- 2. 【收集意图】计算所有可能的“卖出意图” ---
            # a. 正常调仓卖出: 计划不再持有，但我们实际还持有
            normal_exits_intent = ~curr_holdings_plan & actual_holdings

            # b. 强制持有期满卖出: 达到最大天数且仍在实际持有
            force_exit_intent = pd.Series(False, index=holding_signals.columns)
            if max_holding_days is not None:
                force_exit_intent = (holding_days >= max_holding_days) & actual_holdings

            # c. 合并所有卖出意图（包括昨日未完成的）
            total_intent_to_sell = normal_exits_intent | force_exit_intent | pending_exits_tracker

            # --- 3. 【收集意图】计算所有可能的“买入意图” ---
            # a. 今天新产生的买入意图: 计划要持有，但我们实际没有持有
            new_buy_intent = curr_holdings_plan & ~actual_holdings

            # b. 合并“待买清单”中的任务，形成“今日总买入意图”
            total_intent_to_buy = new_buy_intent | pending_buys_tracker

            # --- 4. 【处理执行】结合市场现实（停牌），决定今天实际的交易 ---
            is_tradable_today = price_df.iloc[i].notna()

            executable_exits = total_intent_to_sell & is_tradable_today
            executable_entries = total_intent_to_buy & is_tradable_today

            # 记录最终信号
            exits.iloc[i] = executable_exits
            entries.iloc[i] = executable_entries

            # --- 5.【更新状态】为“明天”准备好所有状态变量 ---
            # a. 更新“实际持仓”
            actual_holdings = (actual_holdings | executable_entries) & ~executable_exits

            # b. 更新“待卖清单”
            pending_exits_tracker = total_intent_to_sell & ~is_tradable_today

            # c. 更新“待买清单”和“年龄”
            pending_buys_tracker = total_intent_to_buy & ~executable_entries
            pending_buys_age[~pending_buys_tracker] = 0  # 不在清单上的，年龄归零 （非待买的，==0
            # 新加入清单的，年龄从1开始（因为今天已经算一天了）
            newly_pending_buys_mask = pending_buys_tracker & (pending_buys_age == 0)
            pending_buys_age[newly_pending_buys_mask] = 1

            # d. 更新“持仓天数”
            # 对于今天新买入的，天数设为1；对于今天卖出的，天数归零
            holding_days[executable_entries] = 1
            holding_days[executable_exits] = 0

        # --- 期末处理 ---
        # 在最后一个交易日，强制清仓所有仍在持有的、或仍在待卖清单上的股票
        final_holdings = actual_holdings | pending_exits_tracker
        is_tradable_last_day = price_df.iloc[-1].notna()
        exits.iloc[-1] = exits.iloc[-1] | (final_holdings & is_tradable_last_day)

        logger.info(f"信号生成完成: 买入信号({entries.sum().sum()})个, 总卖出信号({exits.sum().sum()})个")
        return entries, exits


    def run_backtest_old(
            self,
            price_df: pd.DataFrame,
            factor_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, any]:
        # 1. 数据准备
        all_dfs = [price_df] + list(factor_dict.values())
        aligned_dfs = vbt.base.reshape_fns.broadcast(*all_dfs, keep_pd=True, align_index=True, align_columns=True)

        aligned_price = aligned_dfs[0]
        aligned_factors = {name: df for name, df in zip(factor_dict.keys(), aligned_dfs[1:])}
        logger.info(f"数据对齐完成，最终维度: {aligned_price.shape}")
        # 2. 逐个因子回测
        for factor_name, factor_data in aligned_factors.items():
            logger.info(f"🚀 开始回测因子: {factor_name}")
            # 3. 信号生成流水线
            # 首先，生成每日的目标持仓状态 全是true false 表示当日rank情况的true flase
            holding_signals = self.signal_generator.generate_long_holding_signals(factor_data, aligned_price,
                                                                                  self.config)

            origin_weights_df = self.get_position_weights_by_per_weight(holding_signals)
            self.myself_debug_data(origin_weights_df)
            #照顾vector 专门为他算术！
            weights_df = convert_to_sequential_percents(origin_weights_df)
            # 计算合理的综合交易费用
            # 买入成本: 佣金(万3) + 滑点(千1) = 0.0003 + 0.001 = 0.0013
            # 卖出成本: 佣金(万3) + 印花税(千1) + 滑点(千1) = 0.0003 + 0.001 + 0.001 = 0.0023
            # 平均双边成本: (0.0013 + 0.0023) / 2 = 0.0018
            comprehensive_fee_rate = (
                    self.config.commission_rate +  # 佣金 0.0003
                    self.config.slippage_rate +  # 滑点 0.001
                    self.config.stamp_duty / 2  # 印花税分摊 0.0005
            )
            # 改进退出信号生成 - 确保在时间窗口结束时强制退出 (这样做，只是为了简单直观看出我的策略效果！
            improved_entries, improved_exits = self._generate_improved_signals(
                holding_signals, aligned_price, max_holding_days=self.config.max_holding_days
            )
            # 【新增调试】检查信号的详细情况
            self.debug_signal_generation(holding_signals, self.config, improved_entries, improved_exits, origin_weights_df,0,len(holding_signals)-1)

            # 1. 检查实际的交易记录
            portfolio = vbt.Portfolio.from_signals(
                call_seq='auto',  # first sell then buy 实测! 必须配置！
                group_by=True,  # 必须配置
                cash_sharing=True,  # 必须配置

                size_type="percent",  # 实测！ 持仓金额为百分比
                size=weights_df,

                #自定义df 价格数据
                close=aligned_price,
                #信号
                entries=improved_entries,
                exits=improved_exits,

                #交易过程中指标
                init_cash=self.config.initial_cash,
                fees=comprehensive_fee_rate,
                freq='D' #形容价格的
            )
            # 3. 检查持仓记录
            trades = portfolio.positions.records_readable
            expected_trades = improved_entries.sum().sum()
            logger.info(f"  期望交易数: {expected_trades}")
            logger.info(f"  实际交易数: {len(trades)}")
            print(portfolio.stats())

            self.create_final_report(portfolio, holding_signals, 0, len(holding_signals)-1)
            self.plot_cumulative_returns_curve(portfolio)
            records = portfolio.trades.records_readable
            records
            self.portfolios[factor_name] = portfolio

        logger.info(f"🎉 {factor_dict.keys()}因子回测完成")

        return self.portfolios

    def _generate_signals_final_version(
            self,
            holding_signals: pd.DataFrame,
            price_df: pd.DataFrame,
            max_holding_days: int = None,
            retry_buy_limit: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        【V7 - 状态机循环最终版 - 完整代码】
        - 引入 actual_holdings 状态变量，精确追踪每日真实持仓。
        - 新增“待买清单”逻辑，处理因停牌或资金不足而失败的买入信号，并设有“有效期”。
        - 彻底解决“负数持仓”、“僵尸信号”、“回声信号”等所有状态管理Bug。
        - 这是在 for 循环框架内最稳健的实现。

        Args:
            holding_signals: “理想持仓计划”矩阵 (来自上游函数)。
            price_df: 对齐后的价格数据。
            max_holding_days: 最大持仓天数，用于时间止损。
            retry_buy_limit: 买入信号失败后，重试的最大天数。

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (最终的买入信号, 最终的卖出信号)
        """
        logger.info(f"【V7 最终版】开始生成信号，最大持仓: {max_holding_days}, 买入重试期: {retry_buy_limit}天")

        # --- 初始化 ---
        entries = pd.DataFrame(False, index=holding_signals.index, columns=holding_signals.columns)
        exits = pd.DataFrame(False, index=holding_signals.index, columns=holding_signals.columns)

        # --- 状态变量 ---
        # 【核心】“实际持仓”状态，作为循环内部的地面实况
        actual_holdings = pd.Series(False, index=holding_signals.columns)

        # 持仓天数计数器
        holding_days = pd.Series(0, index=holding_signals.columns)

        # “待卖清单”，追踪需要卖出但可能被延迟的股票
        pending_exits_tracker = pd.Series(False, index=holding_signals.columns)

        # “待买清单”和任务有效期计数器
        pending_buys_tracker = pd.Series(False, index=holding_signals.columns)
        pending_buys_age = pd.Series(0, index=holding_signals.columns)

        # --- 逐日循环生成信号 ---
        for i in range(len(holding_signals)):
            curr_date = holding_signals.index[i]
            curr_holdings_plan = holding_signals.iloc[i]

            # --- 1. 状态更新（每日开始时）---
            # a. 对“待买清单”上的任务进行“老化”，并作废“过期”任务
            if pending_buys_tracker.any():
                pending_buys_age[pending_buys_tracker] += 1
                expired_buys = pending_buys_age > retry_buy_limit
                pending_buys_tracker[expired_buys] = False
                pending_buys_age[expired_buys] = 0

            # b. 更新持仓天数: 只为实际持有的股票累加天数
            holding_days[actual_holdings] += 1

            # --- 2. 【收集意图】计算所有可能的“卖出意图” ---
            # a. 正常调仓卖出: 计划不再持有，但我们实际还持有
            normal_exits_intent = ~curr_holdings_plan & actual_holdings

            # b. 强制持有期满卖出: 达到最大天数且仍在实际持有
            force_exit_intent = pd.Series(False, index=holding_signals.columns)
            if max_holding_days is not None:
                force_exit_intent = (holding_days >= max_holding_days) & actual_holdings

            # c. 合并所有卖出意图（包括昨日未完成的）
            total_intent_to_sell = normal_exits_intent | force_exit_intent | pending_exits_tracker

            # --- 3. 【收集意图】计算所有可能的“买入意图” ---
            # a. 今天新产生的买入意图: 计划要持有，但我们实际没有持有
            new_buy_intent = curr_holdings_plan & ~actual_holdings

            # b. 合并“待买清单”中的任务，形成“今日总买入意图”
            total_intent_to_buy = new_buy_intent | pending_buys_tracker

            # --- 4.【处理执行】结合市场现实（停牌），决定今天实际的交易 ---
            is_tradable_today = price_df.iloc[i].notna()

            executable_exits = total_intent_to_sell & is_tradable_today
            executable_entries = total_intent_to_buy & is_tradable_today

            # 记录最终信号
            exits.iloc[i] = executable_exits
            entries.iloc[i] = executable_entries

            # --- 5.【更新状态】为“明天”准备好所有状态变量 ---
            # a. 更新“实际持仓”
            actual_holdings = (actual_holdings | executable_entries) & ~executable_exits

            # b. 更新“待卖清单”
            pending_exits_tracker = total_intent_to_sell & ~is_tradable_today

            # c. 更新“待买清单”和“年龄”
            pending_buys_tracker = total_intent_to_buy & ~executable_entries
            pending_buys_age[~pending_buys_tracker] = 0  # 不在清单上的，年龄归零
            # 新加入清单的，年龄从1开始（因为今天已经算一天了）
            newly_pending_buys_mask = pending_buys_tracker & (pending_buys_age == 0)
            pending_buys_age[newly_pending_buys_mask] = 1

            # d. 更新“持仓天数”
            # 对于今天新买入的，天数设为1；对于今天卖出的，天数归零
            holding_days[executable_entries] = 1
            holding_days[executable_exits] = 0

        # --- 期末处理 ---
        # 在最后一个交易日，强制清仓所有仍在持有的、或仍在待卖清单上的股票
        final_holdings = actual_holdings | pending_exits_tracker
        is_tradable_last_day = price_df.iloc[-1].notna()
        exits.iloc[-1] = exits.iloc[-1] | (final_holdings & is_tradable_last_day)

        logger.info(f"信号生成完成: 买入信号({entries.sum().sum()})个, 总卖出信号({exits.sum().sum()})个")
        return entries, exits

    def generate_wide_format_orders(
            self,
            holding_signals: pd.DataFrame,
            price_df: pd.DataFrame,
            init_cash: float,
            fees: float,
            max_holding_days: int = None,
            retry_buy_limit: int = 3
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        【V9 - 宽格式订单生成器】
        - 在循环内完整模拟现金、持仓股数等状态。
        - 直接生成 from_orders 所需的 size, price, direction 三个宽格式DataFrame。
        - 这是在最底层接口上工作的终极解决方案。
        """
        logger.info(f"【V9 宽格式订单生成器版】开始生成精确订单...")

        # --- 初始化 ---
        # 为 size, price, direction 创建空的、与价格表对齐的DataFrame
        # 用 np.nan 填充，因为0可能是一个有效值（比如size=0）
        size_df = pd.DataFrame(np.nan, index=holding_signals.index, columns=holding_signals.columns)
        price_df_orders = pd.DataFrame(np.nan, index=holding_signals.index, columns=holding_signals.columns)
        direction_df = pd.DataFrame(np.nan, index=holding_signals.index, columns=holding_signals.columns)

        # --- 状态变量 (与V8相同) ---
        cash = init_cash
        actual_positions = pd.Series(0, index=holding_signals.columns)
        # ... (其他状态变量：holding_days, pending_exits, pending_buys, pending_buys_age)
        holding_days = pd.Series(0, index=holding_signals.columns)
        pending_exits_tracker = pd.Series(False, index=holding_signals.columns)
        pending_buys_tracker = pd.Series(False, index=holding_signals.columns)
        pending_buys_age = pd.Series(0, index=holding_signals.columns)

        # --- 逐日循环，模拟交易 ---
        for i in range(len(holding_signals)):
            curr_date = holding_signals.index[i]
            # ... (所有V8版本的状态更新和意图收集逻辑完全不变) ...
            # --- Start of copy from V8.1 logic ---
            if pending_buys_tracker.any():
                pending_buys_age[pending_buys_tracker] += 1
                expired_buys = pending_buys_age > retry_buy_limit
                pending_buys_tracker[expired_buys] = False
                pending_buys_age[expired_buys] = 0
            currently_held_mask = actual_positions > 0
            holding_days[currently_held_mask] += 1
            curr_holdings_plan = holding_signals.iloc[i]
            normal_exits_intent = ~curr_holdings_plan & currently_held_mask
            force_exit_intent = pd.Series(False, index=holding_signals.columns)
            if max_holding_days is not None:
                force_exit_intent = (holding_days >= max_holding_days) & currently_held_mask
            total_intent_to_sell = normal_exits_intent | force_exit_intent | pending_exits_tracker
            new_buy_intent = curr_holdings_plan & ~currently_held_mask
            total_intent_to_buy = new_buy_intent | pending_buys_tracker
            is_tradable_today = price_df.iloc[i].notna()
            executable_exits_mask = total_intent_to_sell & is_tradable_today
            executable_entries_mask = total_intent_to_buy & is_tradable_today
            # --- End of copy from V8.1 logic ---

            # --- 【核心改造】直接填充宽格式DataFrame，而不是append到list ---

            # a. 处理卖出
            stocks_to_sell = executable_exits_mask[executable_exits_mask].index
            for stock in stocks_to_sell:
                price = price_df.loc[curr_date, stock]
                size_to_sell = actual_positions[stock]
                if size_to_sell > 0:
                    size_df.loc[curr_date, stock] = size_to_sell
                    price_df_orders.loc[curr_date, stock] = price
                    direction_df.loc[curr_date, stock] = 1  # 1 for Sell
                    cash += size_to_sell * price * (1 - fees)
                    actual_positions[stock] = 0

            # b. 处理买入
            stocks_to_buy = executable_entries_mask[executable_entries_mask].index
            num_buys = len(stocks_to_buy)
            if num_buys > 0:
                cash_per_stock = cash / num_buys
                for stock in stocks_to_buy:
                    price = price_df.loc[curr_date, stock]
                    if price > 0:
                        size_to_buy = np.floor((cash_per_stock / (price * (1 + fees))) / 100) * 100
                        if size_to_buy > 0:
                            size_df.loc[curr_date, stock] = size_to_buy
                            price_df_orders.loc[curr_date, stock] = price
                            direction_df.loc[curr_date, stock] = -1  # 0 for Buy
                            cash -= size_to_buy * price * (1 + fees)
                            actual_positions[stock] += size_to_buy

            # ... (所有V8版本的状态更新逻辑完全不变) ...
            # --- Start of copy from V8.1 logic for state update ---
            executed_entries_today = pd.Series(False, index=holding_signals.columns)
            for stock in stocks_to_buy:  # More robust check for executed entries
                if size_df.loc[curr_date, stock] > 0:
                    executed_entries_today[stock] = True
            pending_exits_tracker = total_intent_to_sell & ~is_tradable_today
            pending_buys_tracker = total_intent_to_buy & ~executed_entries_today
            pending_buys_age[~pending_buys_tracker] = 0
            newly_pending_buys_mask = pending_buys_tracker & (pending_buys_age == 0)
            pending_buys_age[newly_pending_buys_mask] = 1
            holding_days[executed_entries_today] = 1
            holding_days[executable_exits_mask] = 0
            # --- End of copy from V8.1 logic for state update ---

        return size_df, price_df_orders, direction_df
    #from_order
    def run_backtest(
            self,
            price_df: pd.DataFrame,
            factor_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, any]:
        # 1. 数据准备
        all_dfs = [price_df] + list(factor_dict.values())
        aligned_dfs = vbt.base.reshape_fns.broadcast(*all_dfs, keep_pd=True, align_index=True, align_columns=True)

        aligned_price = aligned_dfs[0]
        aligned_factors = {name: df for name, df in zip(factor_dict.keys(), aligned_dfs[1:])}
        logger.info(f"数据对齐完成，最终维度: {aligned_price.shape}")
        # 2. 逐个因子回测
        for factor_name, factor_data in aligned_factors.items():
            logger.info(f"🚀 开始回测因子: {factor_name}")
            # 3. 信号生成流水线
            # 首先，生成每日的目标持仓状态 全是true false 表示当日rank情况的true flase
            holding_signals = self.signal_generator.generate_long_holding_signals(factor_data, aligned_price,
                                                                                  self.config)

            origin_weights_df = self.get_position_weights_by_per_weight(holding_signals)
            self.myself_debug_data(origin_weights_df)
            # 照顾vector 专门为他算术！
            weights_df = convert_to_sequential_percents(origin_weights_df)
            # 计算合理的综合交易费用
            # 买入成本: 佣金(万3) + 滑点(千1) = 0.0003 + 0.001 = 0.0013
            # 卖出成本: 佣金(万3) + 印花税(千1) + 滑点(千1) = 0.0003 + 0.001 + 0.001 = 0.0023
            # 平均双边成本: (0.0013 + 0.0023) / 2 = 0.0018
            comprehensive_fee_rate = (
                    self.config.commission_rate +  # 佣金 0.0003
                    self.config.slippage_rate +  # 滑点 0.001
                    self.config.stamp_duty / 2  # 印花税分摊 0.0005
            )
            # 改进退出信号生成 - 确保在时间窗口结束时强制退出 (这样做，只是为了简单直观看出我的策略效果！
            improved_entries, improved_exits = self._generate_improved_signals(
                holding_signals, aligned_price, max_holding_days=self.config.max_holding_days
            )
            # 【新增调试】检查信号的详细情况
            self.debug_signal_generation(holding_signals, self.config, improved_entries, improved_exits,
                                         origin_weights_df, 0, len(holding_signals) - 1)

            # 1. 检查实际的交易记录

            # 2. 【核心】调用V9订单生成器，获取三份“配料表”
            size_wide_df, price_wide_df, direction_wide_df = self.generate_wide_format_orders(
                holding_signals=holding_signals,
                price_df=aligned_price,
                init_cash=self.config.initial_cash,
                fees=comprehensive_fee_rate,
                max_holding_days=60
                # ... 其他参数 ...
            )

            # 3. 将这三份精确的“配料表”交给 vectorbt 的底层厨师
            portfolio = vbt.Portfolio.from_orders(
                close=aligned_price,
                size=size_wide_df,
                price=price_wide_df,
                direction=direction_wide_df,

                # 我们不再需要任何 sizing 或 signal 参数

                init_cash=self.config.initial_cash,
                fees=0,  # 因为我们已在循环中手动计算了费用
                freq='D'
            )


            # 3. 检查持仓记录
            trades = portfolio.positions.records_readable
            expected_trades = improved_entries.sum().sum()
            logger.info(f"  期望交易数: {expected_trades}")
            logger.info(f"  实际交易数: {len(trades)}")
            print(portfolio.stats())

            self.create_final_report(portfolio, holding_signals, 0, len(holding_signals) - 1)
            self.plot_cumulative_returns_curve(portfolio)
            records = portfolio.trades.records_readable
            records
            self.portfolios[factor_name] = portfolio

        logger.info(f"🎉 {factor_dict.keys()}因子回测完成")

        return self.portfolios

    def generate_orders_final_version(
            self,
            holding_signals: pd.DataFrame,
            price_df: pd.DataFrame,
            init_cash: float,
            fees: float,
            max_holding_days: int = None,
            retry_buy_limit: int = 3
    ) -> pd.DataFrame:
        """
        【V8.1 - 订单生成器最终版 - 完整代码】
        - 在循环内完整模拟现金、持仓股数等所有状态。
        - 生成精确的买卖订单列表，供 Portfolio.from_orders 使用。
        - 补全了所有状态更新逻辑，确保行为的精确性。
        - 这是在无法使用 target-based sizing 模式下的终极解决方案。
        """
        logger.info(f"【V8.1 订单生成器版】开始生成精确订单...")

        # --- 初始化 ---
        order_records = []

        # --- 状态变量 ---
        cash = init_cash
        actual_positions = pd.Series(0, index=holding_signals.columns)  # 追踪持仓股数
        holding_days = pd.Series(0, index=holding_signals.columns)
        pending_exits_tracker = pd.Series(False, index=holding_signals.columns)
        pending_buys_tracker = pd.Series(False, index=holding_signals.columns)
        pending_buys_age = pd.Series(0, index=holding_signals.columns)

        # --- 逐日循环，模拟交易 ---
        for i in range(len(holding_signals)):
            curr_date = holding_signals.index[i]
            curr_holdings_plan = holding_signals.iloc[i]

            # --- 1. 状态更新（每日开始时） ---
            if pending_buys_tracker.any():
                pending_buys_age[pending_buys_tracker] += 1
                expired_buys = pending_buys_age > retry_buy_limit
                pending_buys_tracker[expired_buys] = False
                pending_buys_age[expired_buys] = 0

            currently_held_mask = actual_positions > 0
            holding_days[currently_held_mask] += 1

            # --- 2. 【收集意图】 ---
            normal_exits_intent = ~curr_holdings_plan & currently_held_mask
            force_exit_intent = pd.Series(False, index=holding_signals.columns)
            if max_holding_days is not None:
                force_exit_intent = (holding_days >= max_holding_days) & currently_held_mask

            total_intent_to_sell = normal_exits_intent | force_exit_intent | pending_exits_tracker
            new_buy_intent = curr_holdings_plan & ~currently_held_mask
            total_intent_to_buy = new_buy_intent | pending_buys_tracker

            # --- 3.【处理执行】---
            is_tradable_today = price_df.iloc[i].notna()
            executable_exits_mask = total_intent_to_sell & is_tradable_today
            executable_entries_mask = total_intent_to_buy & is_tradable_today

            # --- 4.【生成精确订单】---
            executed_entries_today = pd.Series(False, index=holding_signals.columns)  # 精确记录当日成功买入

            # a. 处理卖出订单 (Sell Phase)
            stocks_to_sell = executable_exits_mask[executable_exits_mask].index
            for stock in stocks_to_sell:
                price = price_df.loc[curr_date, stock]
                size_to_sell = actual_positions[stock]
                if size_to_sell > 0:
                    order_records.append({
                        'Timestamp': curr_date, 'Symbol': stock, 'Size': size_to_sell,
                        'Side': 1, 'Price': price
                    })
                    cash += size_to_sell * price * (1 - fees)
                    actual_positions[stock] = 0

            # b. 处理买入订单 (Buy Phase)
            stocks_to_buy = executable_entries_mask[executable_entries_mask].index
            num_buys = len(stocks_to_buy)
            if num_buys > 0:
                cash_per_stock = cash / num_buys

                for stock in stocks_to_buy:
                    price = price_df.loc[curr_date, stock]
                    if price > 0:
                        size_to_buy = (cash_per_stock / (price * (1 + fees)))
                        size_to_buy = np.floor(size_to_buy / 100) * 100

                        if size_to_buy > 0:
                            order_records.append({
                                'Timestamp': curr_date, 'Symbol': stock, 'Size': size_to_buy,
                                'Side': 0, 'Price': price
                            })
                            cash -= size_to_buy * price * (1 + fees)
                            actual_positions[stock] += size_to_buy
                            executed_entries_today[stock] = True  # 精确标记成功买入

            # --- 5.【更新状态】为“明天”准备 ---
            # a. 待卖清单: 卖出意图存在，但今天无法交易
            pending_exits_tracker = total_intent_to_sell & ~is_tradable_today

            # b. 待买清单: 买入意图存在，但今天未成功执行
            pending_buys_tracker = total_intent_to_buy & ~executed_entries_today

            # c. 待买清单年龄
            pending_buys_age[~pending_buys_tracker] = 0
            newly_pending_buys_mask = pending_buys_tracker & (pending_buys_age == 0)
            pending_buys_age[newly_pending_buys_mask] = 1

            # d. 持仓天数
            holding_days[executed_entries_today] = 1  # 新买入的，天数设为1
            holding_days[executable_exits_mask] = 0  # 成功卖出的，天数归零

        # --- 期末处理与返回 ---
        if not order_records:
            return pd.DataFrame(columns=['Timestamp', 'Symbol', 'Size', 'Side', 'Price'])

        # 将订单列表转换为 vectorbt 需要的格式
        order_df = pd.DataFrame(order_records)
        order_df = order_df.rename(
            columns={'Timestamp': 'Order Timestamp', 'Symbol': 'Symbol', 'Size': 'Size', 'Side': 'Side', 'Price': 'Price'})
        order_df = order_df.set_index('Order Timestamp')

        return order_df

    def run_backtest_from_order(
            self,
            price_df: pd.DataFrame,
            factor_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, any]:
        # 1. 数据准备
        all_dfs = [price_df] + list(factor_dict.values())
        aligned_dfs = vbt.base.reshape_fns.broadcast(*all_dfs, keep_pd=True, align_index=True, align_columns=True)

        aligned_price = aligned_dfs[0]
        aligned_factors = {name: df for name, df in zip(factor_dict.keys(), aligned_dfs[1:])}
        logger.info(f"数据对齐完成，最终维度: {aligned_price.shape}")
        # 2. 逐个因子回测
        for factor_name, factor_data in aligned_factors.items():
            logger.info(f"🚀 开始回测因子: {factor_name}")
            # 3. 信号生成流水线
            # 首先，生成每日的目标持仓状态 全是true false 表示当日rank情况的true flase
            holding_signals = self.signal_generator.generate_long_holding_signals(factor_data, aligned_price,
                                                                                  self.config)

            origin_weights_df = self.get_position_weights_by_per_weight(holding_signals)
            self.myself_debug_data(origin_weights_df)
            #照顾vector 专门为他算术！
            weights_df = convert_to_sequential_percents(origin_weights_df)
            # 计算合理的综合交易费用
            # 买入成本: 佣金(万3) + 滑点(千1) = 0.0003 + 0.001 = 0.0013
            # 卖出成本: 佣金(万3) + 印花税(千1) + 滑点(千1) = 0.0003 + 0.001 + 0.001 = 0.0023
            # 平均双边成本: (0.0013 + 0.0023) / 2 = 0.0018
            comprehensive_fee_rate = (
                    self.config.commission_rate +  # 佣金 0.0003
                    self.config.slippage_rate +  # 滑点 0.001
                    self.config.stamp_duty / 2  # 印花税分摊 0.0005
            )
            # 改进退出信号生成 - 确保在时间窗口结束时强制退出 (这样做，只是为了简单直观看出我的策略效果！
            improved_entries, improved_exits = self._generate_improved_signals(
                holding_signals, aligned_price, max_holding_days=self.config.max_holding_days
            )
            # 【新增调试】检查信号的详细情况
            self.debug_signal_generation(holding_signals, self.config, improved_entries, improved_exits, origin_weights_df,0,len(holding_signals)-1)



    def get_daily_holdings_count_from_trades(self,portfolio, full_date_index: pd.DatetimeIndex) -> pd.Series:
        """
        【通用版】从最底层的交易记录中，精确重构每日的实际持仓股票数量。
        不依赖任何高版本 vectorbt 的特定属性。

        Args:
            portfolio: vectorbt 回测完成后返回的 Portfolio 对象。
            full_date_index: 完整的、包含所有回测日期的索引。

        Returns:
            pd.Series: 索引为日期，值为当天实际持仓股票数的序列。
        """
        trades = portfolio.trades.records_readable

        if trades.empty:
            # 如果没有任何交易，则始终持仓为0
            return pd.Series(0, index=full_date_index)

        # 1. 获取所有“入场”事件，按天统计
        entry_events = trades.groupby(trades['Entry Timestamp'].dt.date).size()
        entry_events.name = 'entries'

        # 2. 获取所有“出场”事件，按天统计
        exit_events = trades.groupby(trades['Exit Timestamp'].dt.date).size()
        exit_events.name = 'exits'

        # 3. 将事件合并，计算每日的“净持仓变化”
        #   注意 .dt.date 会导致索引变为 object，需要转回 datetime
        entry_events.index = pd.to_datetime(entry_events.index)
        exit_events.index = pd.to_datetime(exit_events.index)

        daily_net_change = entry_events.sub(exit_events, fill_value=0)

        # 4. 将“净变化”扩展到整个回测周期
        #   在没有交易的日子里，净变化为0
        daily_net_change_full = daily_net_change.reindex(full_date_index, fill_value=0)

        # 5. 【核心】对每日的净变化进行累积求和，得到每日的最终持仓数
        actual_positions_count = daily_net_change_full.cumsum().astype(int)

        return actual_positions_count
    def create_final_report(
            self,
            portfolio,
            holding_signals: pd.DataFrame,
            sidx: int,
            eidx: int
    ):
        """
        【法务审计级 V3.1 - 兼容版】
        - 使用从交易记录中重构的每日持仓数，兼容旧版 vectorbt。
        """
        logger.info("🔍【法务审计级 V3.1 兼容版】日志分析开始...")

        # --- 1. 使用我们新的、兼容性强的函数来获取“地面实况” ---
        actual_holdings_count = self.get_daily_holdings_count_from_trades(
            portfolio=portfolio,
            full_date_index=holding_signals.index
        )

        # --- 后续逻辑与之前完全相同 ---
        trades = portfolio.trades.records_readable
        daily_entries_count = trades.groupby(trades['Entry Timestamp'].dt.date).size()
        daily_exits_count = trades.groupby(trades['Exit Timestamp'].dt.date).size()

        log_df = pd.DataFrame(index=holding_signals.index)
        log_df['intended_holdings'] = holding_signals.sum(axis=1)
        log_df['actual_holdings'] = actual_holdings_count

        daily_entries_count.index = pd.to_datetime(daily_entries_count.index)
        daily_exits_count.index = pd.to_datetime(daily_exits_count.index)

        log_df['actual_entries'] = daily_entries_count
        log_df['actual_exits'] = daily_exits_count
        log_df = log_df.fillna(0).astype(int)

        # --- 3. 打印日志 ---
        sample_dates = log_df.index[sidx:eidx]
        for date in sample_dates:
            row = log_df.loc[date]
            log_msg = (
                f"{date.strftime('%Y-%m-%d')}: "
                f"实际持仓({row['actual_holdings']}), "
                f"实际卖出({row['actual_exits']}), "
                f"实际买入({row['actual_entries']})"
            )
            logger.info(log_msg)

        return log_df
    def _recalculate_trade_metric(self, corrected_stats, trades, metric):
        """重新计算特定的交易指标"""
        # 【修复】正确过滤已关闭交易 - Status可能是字符串'Closed'或整数1
        if 'Status' in trades.columns:
            status_values = trades['Status'].unique()
            if 'Closed' in status_values:
                closed_trades = trades[trades['Status'] == 'Closed']
            elif 1 in status_values:
                closed_trades = trades[trades['Status'] == 1]
            else:
                # 如果Status值未知，假设所有交易都已关闭
                closed_trades = trades
        else:
            closed_trades = trades

        if len(closed_trades) > 0:
            winning_trades = closed_trades[closed_trades['PnL'] > 0]
            losing_trades = closed_trades[closed_trades['PnL'] < 0]

            if metric == 'Win Rate [%]':
                win_rate = len(winning_trades) / len(closed_trades) * 100
                corrected_stats[metric] = win_rate
            elif metric == 'Best Trade [%]':
                corrected_stats[metric] = closed_trades['Return'].max() * 100
            elif metric == 'Worst Trade [%]':
                corrected_stats[metric] = closed_trades['Return'].min() * 100
            elif metric == 'Avg Winning Trade [%]':
                if len(winning_trades) > 0:
                    corrected_stats[metric] = winning_trades['Return'].mean() * 100
                else:
                    corrected_stats[metric] = 0
            elif metric == 'Avg Losing Trade [%]':
                if len(losing_trades) > 0:
                    corrected_stats[metric] = losing_trades['Return'].mean() * 100
                else:
                    corrected_stats[metric] = 0
            elif metric == 'Expectancy':
                corrected_stats[metric] = closed_trades['PnL'].mean()
        else:
            corrected_stats[metric] = 0


    def get_comparison_table(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        生成因子对比表
        
        Args:
            metrics: 要对比的指标列表
            
        Returns:
            pd.DataFrame: 对比结果表
        """
        if not self.portfolios:
            raise ValueError("请先运行回测")

        if metrics is None:
            metrics = [
                'Total Return [%]',
                'Sharpe Ratio',
                'Calmar Ratio',
                'Max Drawdown [%]',
                'Win Rate [%]',
                'Profit Factor'
            ]

        comparison_data = {}
        for factor_name, portfolio in self.portfolios.items():
            stats = portfolio.stats()
            comparison_data[factor_name] = stats[metrics]

        comparison_df = pd.DataFrame(comparison_data).T
        logger.info("因子对比表生成完成")
        return comparison_df

    def plot_cumulative_returns(self,
                                figsize: Tuple[int, int] = (15, 8),
                                save_path: Optional[str] = None) -> None:
        """
        绘制累积收益率曲线
        
        Args:
            figsize: 图片大小
            save_path: 保存路径
        """
        if not self.portfolios:
            raise ValueError("请先运行回测")

        plt.figure(figsize=figsize)

        for factor_name, portfolio in self.portfolios.items():
            returns = portfolio.returns()
            cumulative_returns = (1 + returns).cumprod()
            plt.plot(cumulative_returns.index, cumulative_returns.values,
                     label=factor_name, linewidth=2)

        plt.title('因子策略累积收益率对比', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('累积收益率', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")

        plt.show()

    def plot_drawdown_analysis(self,
                               figsize: Tuple[int, int] = (15, 10),
                               save_path: Optional[str] = None) -> None:
        """
        绘制回撤分析图
        
        Args:
            figsize: 图片大小
            save_path: 保存路径
        """
        if not self.portfolios:
            raise ValueError("请先运行回测")

        n_factors = len(self.portfolios)
        fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=True)
        if n_factors == 1:
            axes = [axes]

        for i, (factor_name, portfolio) in enumerate(self.portfolios.items()):
            drawdown = portfolio.drawdown()
            axes[i].fill_between(drawdown.index, drawdown.values, 0,
                                 color='red', alpha=0.3)
            axes[i].set_title(f'{factor_name} - 回撤分析', fontsize=14)
            axes[i].set_ylabel('回撤 (%)', fontsize=12)
            axes[i].grid(True, alpha=0.3)

        plt.xlabel('日期', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"回撤图已保存: {save_path}")

        plt.show()

    def generate_full_report(self,
                             report_dir: str = "backtest_reports") -> str:
        """
        生成完整的回测报告
        
        Args:
            report_dir: 报告保存目录
            
        Returns:
            str: 报告文件路径
        """
        if not self.portfolios:
            raise ValueError("请先运行回测")

        # 创建报告目录
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 生成对比表
        comparison_df = self.get_comparison_table()
        comparison_file = report_path / f"factor_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, encoding='utf-8-sig')

        # 生成图表
        returns_chart = report_path / f"cumulative_returns_{timestamp}.png"
        self.plot_cumulative_returns(save_path=str(returns_chart))

        drawdown_chart = report_path / f"drawdown_analysis_{timestamp}.png"
        self.plot_drawdown_analysis(save_path=str(drawdown_chart))

        # 生成详细统计报告
        report_file = report_path / f"detailed_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("量化策略回测详细报告\n")
            f.write("=" * 80 + "\n\n")

            f.write("回测配置:\n")
            f.write(f"  调仓频率: {self.config.rebalancing_freq}\n")
            f.write(f"  做多分位: {self.config.top_quantile:.1%}\n")
            f.write(f"  初始资金: {self.config.initial_cash:,.0f}\n")
            f.write(f"  交易费率: {TradingCostCalculator.calculate_total_fees(self.config):.4f}\n\n")

            f.write("数据验证结果:\n")
            for key, value in self.validation_results['stats'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write("因子对比结果:\n")
            f.write(comparison_df.to_string())
            f.write("\n\n")

        logger.info(f"完整报告已生成: {report_path}")
        return str(report_path)

    def get_position_weights_by_per_weight(self, holding_signals):

        # 1. 【新增】创建目标权重矩阵
        # 1.1 计算每日应持有的股票总数
        num_positions = holding_signals.sum(axis=1)

        # 1.2 计算每日的等权权重 (例如，持有10只股票，每只权重为 1/10 = 0.1)
        #     为避免除以零 (在没有持仓的日子)，使用 .replace(np.inf, 0)
        target_weights = (1 / num_positions).replace([np.inf, -np.inf], 0)

        # 1.3 将每日权重值广播到当天的持仓股票上，形成一个与 aholding_signals 形状相同的权重矩阵
        #     不持有(False)的股票，权重自然为 0
        weights_df = holding_signals.mul(target_weights, axis=0)
        return weights_df

    def debug_signal_generation(self, holding_signals, config,entry_signals, exit_signals,weights_df,sidx,eidx):
        logger.info("🔍 信号调试分析开始")
        # 检查前几天的信号情况
        sample_dates = generate_rebalance_dates(holding_signals.index,config.rebalancing_freq)

        # 检查是否所有信号都是False
        total_entries = entry_signals.sum().sum()
        total_exits = exit_signals.sum().sum()

        if total_entries == 0 or total_exits==0:
            raise ValueError("❌ 严重问题：没有生成任何买入信号！ 或者 没有任何卖出信号")
        logger.info(f"✅ 信号生成正常: 买入{total_entries}个, 卖出{total_exits}个")



        logger.info(f"  平均每天持仓股票数量: {holding_signals.sum(axis=1).mean()}")
        ##检查持仓权重 是否等于1
        logger.info(f"  每天平均持仓比例: {weights_df.sum(axis=1).mean()}")
        self._debug_holding_days(holding_signals, entry_signals, exit_signals)

    def about_cash(self, portfolio):
        logger.info(f"现金变化情况:")
        cash_flows  = portfolio.cash()
        initial_cash = float(cash_flows.iloc[0])
        final_cash = float(cash_flows.iloc[-1])
        pass

    def plot_cumulative_returns_curve(self,portfolio):
        cumulative_returns_builtin = (1 + portfolio.returns()).cumprod() - 1

        # 使用内置函数还有一个巨大的好处：可以直接调用 vbt 的绘图功能
        print("\n正在绘制权益曲线...")
        cumulative_returns_builtin.vbt.plot(title='Equity Curve').show()

    def myself_debug_data(self, origin_weights_df):
        #按列 整列至少有一个值不为0！
        origin_weights_df = origin_weights_df.loc[:, origin_weights_df.any(axis=0)]
        pass

    def _debug_holding_days(self, holding_signals, entry_signals, exit_signals):
        """
        分析持仓天数分布，识别"老妖股"（长期持有的股票）
        
        Args:
            holding_signals: 持仓信号矩阵
            entry_signals: 买入信号矩阵  
            exit_signals: 卖出信号矩阵
        """
        logger.info("🕵️ 开始分析持仓天数分布...")
        
        # 创建持仓天数统计字典
        stock_holding_stats = {}
        all_holding_periods = []
        
        # 遍历每只股票
        for stock in holding_signals.columns:
            stock_entries = entry_signals[stock]
            stock_exits = exit_signals[stock]
            stock_holdings = holding_signals[stock]
            
            # 找到所有买入时点
            entry_dates = stock_entries[stock_entries].index.tolist()
            exit_dates = stock_exits[stock_exits].index.tolist()
            
            if len(entry_dates) == 0:
                continue
                
            # 计算每次持仓周期
            holding_periods = []
            
            for entry_date in entry_dates:
                # 找到对应的卖出日期
                matching_exits = [exit_date for exit_date in exit_dates if exit_date > entry_date]
                
                if matching_exits:
                    exit_date = min(matching_exits)  # 最近的卖出日期
                    # 计算持仓天数
                    holding_days = (exit_date - entry_date).days
                    holding_periods.append(holding_days)
                    all_holding_periods.append(holding_days)
                else:
                    # 如果没有找到卖出信号，计算到最后一天的持仓天数
                    last_date = holding_signals.index[-1]
                    holding_days = (last_date - entry_date).days
                    holding_periods.append(holding_days)
                    all_holding_periods.append(holding_days)
            
            if holding_periods:
                stock_holding_stats[stock] = {
                    'total_trades': len(holding_periods),
                    'min_holding_days': min(holding_periods),
                    'max_holding_days': max(holding_periods),
                    'avg_holding_days': np.mean(holding_periods),
                    'holding_periods': holding_periods
                }
        
        if not all_holding_periods:
            logger.warning("⚠️ 没有找到任何持仓记录")
            return
            
        # 整体统计
        total_trades = len(all_holding_periods)
        avg_holding = np.mean(all_holding_periods)
        median_holding = np.median(all_holding_periods)
        max_holding = max(all_holding_periods)
        min_holding = min(all_holding_periods)
        
        logger.info(f"📊 持仓天数总体统计:")
        logger.info(f"  总交易次数: {total_trades}")
        logger.info(f"  平均持仓天数: {avg_holding:.1f}天")
        logger.info(f"  中位数持仓天数: {median_holding:.1f}天")
        logger.info(f"  最短持仓: {min_holding}天")
        logger.info(f"  最长持仓: {max_holding}天")
        
        # 持仓天数分布
        bins = [0, 7, 30, 60, 120, 240, float('inf')]
        bin_labels = ['<7天', '7-30天', '30-60天', '60-120天', '120-240天', '>240天']
        
        for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            if bin_end == float('inf'):
                count = sum(1 for days in all_holding_periods if days >= bin_start)
            else:
                count = sum(1 for days in all_holding_periods if bin_start <= days < bin_end)
            
            percentage = count / total_trades * 100
            logger.info(f"  {bin_labels[i]}: {count}次 ({percentage:.1f}%)")
        
        # 找出"老妖股" - 持仓超过120天的股票
        long_holding_threshold = 120
        old_monster_stocks = []
        
        for stock, stats in stock_holding_stats.items():
            max_days = stats['max_holding_days']
            if max_days >= long_holding_threshold:
                old_monster_stocks.append({
                    'stock': stock,
                    'max_holding_days': max_days,
                    'avg_holding_days': stats['avg_holding_days'],
                    'total_trades': stats['total_trades']
                })
        
        # 按最长持仓天数排序
        old_monster_stocks.sort(key=lambda x: x['max_holding_days'], reverse=True)
        
        if old_monster_stocks:
            logger.info(f"🐉 发现{len(old_monster_stocks)}只老妖股 (持仓>{long_holding_threshold}天):")
            
            # 显示前10只最"妖"的股票
            top_monsters = old_monster_stocks[:10]
            for i, stock_info in enumerate(top_monsters, 1):
                logger.info(f"  {i:2d}. {stock_info['stock']}: 最长{stock_info['max_holding_days']}天, "
                           f"平均{stock_info['avg_holding_days']:.1f}天, 共{stock_info['total_trades']}次交易")
                
            if len(old_monster_stocks) > 10:
                logger.info(f"  ... 还有{len(old_monster_stocks) - 10}只老妖股")
                
            # 超级妖股 - 持仓超过240天
            super_monsters = [s for s in old_monster_stocks if s['max_holding_days'] >= 240]
            if super_monsters:
                logger.info(f"👹 其中{len(super_monsters)}只超级妖股 (持仓>240天):")
                for stock_info in super_monsters:
                    logger.info(f"     {stock_info['stock']}: {stock_info['max_holding_days']}天")
        else:
            logger.info(f"✅ 没有发现老妖股 (所有股票持仓都<{long_holding_threshold}天)")
        
        logger.info("🕵️ 持仓天数分析完成")

    def today_need_exit(self, prev_holdings, curr_holdings, not_finishied_exit):
        today_exit_signal =  ~curr_holdings & prev_holdings
        if not_finishied_exit is not None: #昨天没卖出去，今天赶紧卖！
            today_exit_signal = today_exit_signal | not_finishied_exit

        return today_exit_signal


# 便捷函数
def quick_factor_backtest(
        price_df: pd.DataFrame,
        factor_dict: Dict[str, pd.DataFrame],
        config: Optional[BacktestConfig] = None
) -> Tuple[Dict[str, any], pd.DataFrame]:
    """
    快速因子回测函数
    
    Args:
        price_df: 价格数据
        factor_dict: 因子数据字典
        config: 回测配置
        
    Returns:
        Tuple: (回测结果字典, 对比表)
    """
    backtester = QuantBacktester(config)
    portfolios = backtester.run_backtest(price_df, factor_dict)
    comparison_table = backtester.get_comparison_table()

    return portfolios, comparison_table


if __name__ == "__main__":
    # 示例用法
    logger.info("QuantBacktester 示例运行")

    # 这里需要你提供真实的数据
    # price_df = load_price_data()
    # factor_dict = {
    #     'volatility_40d': load_factor_data('volatility_40d'),
    #     'composite_factor': load_factor_data('composite_factor')
    # }

    # # 配置回测参数
    # config = BacktestConfig(
    #     top_quantile=0.2,
    #     rebalancing_freq='M',
    #     commission_rate=0.0003,
    #     slippage_rate=0.001
    # )

    # # 运行回测
    # backtester = QuantBacktester(config)
    # portfolios = backtester.run_backtest(price_df, factor_dict)

    # # 生成对比和报告
    # comparison_table = backtester.get_comparison_table()
    # print(comparison_table)

    # backtester.plot_cumulative_returns()
    # report_path = backtester.generate_full_report()

    logger.info("QuantBacktester 示例完成")
