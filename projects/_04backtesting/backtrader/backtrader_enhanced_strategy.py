"""
Backtrader增强策略 - 完整迁移vectorBT复杂逻辑

关键改进：
1. 完整迁移_generate_improved_signals的复杂状态管理
2. 自动处理停牌、重试、超期等所有边缘情况
3. 使用Backtrader事件驱动模型替代复杂for循环
4. 保持原有策略的所有核心逻辑和参数
"""
import logging

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import warnings

from data.local_data_load import load_trading_lists, get_last_b_day

warnings.filterwarnings('ignore')

from quant_lib.config.logger_config import setup_logger
from quant_lib.rebalance_utils import generate_rebalance_dates

logger = setup_logger(__name__)


class OrderState(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


class EnhancedFactorStrategy(bt.Strategy):
    """
    增强因子策略 - 完整替代vectorBT复杂逻辑
    
    完整迁移原有的：
    - actual_holdings状态追踪
    - pending_exits_tracker（待卖清单）
    - pending_buys_tracker（待买清单）
    - 重试逻辑和有效期管理
    - 强制卖出超期持仓
    """

    params = (
        # 策略核心参数
        ('factor_data', None),  # 因子数据DataFrame
        ('holding_signals', None),  # 预计算的持仓信号矩阵
        ('rebalance_dates', []),  # 调仓日期列表
        ('max_positions', 10),  # 最大持仓数量
        ('max_holding_days', 60),  # 最大持仓天数（强制卖出）
        ('retry_buy_days', 3),  # 买入重试天数

        # 风控参数
        ('max_weight_per_stock', 0.15),  # 单股最大权重
        ('min_weight_threshold', 0.01),  # 最小权重阈值
        ('emergency_exit_threshold', 0.9),  # 紧急止损阈值

        # 调试参数
        ('debug_mode', True),  # 调试模式
        ('log_detailed', True),  # 详细日志
        ('enable_retry', True),  # 启用重试机制
        ('trading_days', None),  # 交易日列表
    )

    def __init__(self):
        """策略初始化 - 替代vectorBT的状态变量初始化"""
        logger.info("初始化EnhancedFactorStrategy...")

        # === 核心状态变量 - 完整替代vectorBT中的状态追踪 ===

        # 1. 调仓日期处理
        self.rebalance_dates_set = set(pd.to_datetime(self.p.rebalance_dates).date)

        # 2. 持仓状态追踪（替代actual_holdings）
        self.actual_positions = {}  # {stock_name: data_obj}
        self.holding_start_dates = {}  # {stock_name: entry_date}
        self.holding_days_counter = {}  # {stock_name: days}

        # 3. 待处理队列（替代pending_exits_tracker和pending_buys_tracker）
        self.pending_sells = {}  # {stock_name: (retry_count, target_date, reason)}
        self.pending_buys = {}  # {stock_name: (retry_count, target_date, target_weight)}

        # 4. 交易重试管理（完全替代vectorBT中的复杂重试逻辑）
        self.buy_retry_log = {}  # {stock_name: [失败日期列表]}
        self.sell_retry_log = {}  # {stock_name: [失败日期列表]}

        # 5. 性能统计和调试
        self.daily_stats = []  # 每日统计信息
        self.rebalance_count = 0 #调仓次数！
        self.submit_buy_orders = 0 #
        self.submit_sell_orders = 0
        self.success_buy_orders = 0
        self.success_sell_orders = 0
        self.failed_orders = 0 #（别担心，反正有重试

        # 6. 风险控制
        self.emergency_exits = 0  # 紧急止损次数
        self.forced_exits = 0  # 强制超期卖出次数

        #辅助信息



        logger.info(f"策略初始化完成:")
        logger.info(f"  调仓日期: {len(self.rebalance_dates_set)}个")
        logger.info(f"  最大持仓: {self.p.max_positions}只")
        logger.info(f"  最大持有期: {self.p.max_holding_days}天")
        logger.info(f"  重试期限: {self.p.retry_buy_days}天")

    def next(self):
        """
        策略主循环 - 完整替代vectorBT中的复杂for循环
        执行顺序（严格按照原有逻辑）：
        1. 状态更新（持仓天数、重试计数等）
        2. 处理强制卖出（超期持仓）
        3. 处理待卖清单
        4. 处理待买清单
        5. 调仓日执行（如果是调仓日）
        6. 记录统计和调试信息
        """
        current_date = self.datetime.date(0)

        # === 第1步：日常状态更新 ===
        self._daily_state_update()

        # === 第2步：处理强制卖出（替代force_exit_intent逻辑）===
        self._process_forced_exits()

        # === 第3步：处理待卖清单（替代pending_exits_tracker）===
        self._process_pending_sells()

        # === 第5步：调仓日执行（如果是调仓日）=== #新菜 逻辑提前！
        if current_date in self.rebalance_dates_set:
            self._execute_rebalancing(current_date)

        # === 第4步：处理待买清单（替代pending_buys_tracker）=== #剩菜，有余力再买
        self._process_pending_buys() #bug todo  万一是当天调仓日买入失败的票呢？  这次调仓日就不买了才对啊， 这里要调整下

        # === 第6步：记录统计信息 ===
        if self.p.log_detailed:
            self._log_daily_status(current_date)
    #step1
    def _daily_state_update(self):
        """
        每日状态更新 - 替代vectorBT中每日循环开始的状态更新
        """
        # 更新持仓天数
        for stock_name in list(self.holding_days_counter.keys()):
            data_obj = self.getdatabyname(stock_name)
            if self.getposition(data_obj).size > 0:
                self.holding_days_counter[stock_name] += 1
            else:
                # 清理已平仓的记录
                self._cleanup_position_records(stock_name)

        # 更新待买清单的"年龄"（替代pending_buys_age逻辑）
        current_date = self.datetime.date(0)
        for stock_name in list(self.pending_buys.keys()):
            retry_count, target_date, target_weight = self.pending_buys[stock_name]
            days_elapsed = (current_date - target_date).days

            if days_elapsed > self.p.retry_buy_days:
                # 超期，放弃买入
                del self.pending_buys[stock_name] #todo 测试！
                if self.p.debug_mode:
                    logger.info(f"买入任务超期放弃: {stock_name}")

    def _process_forced_exits(self):
        """
        处理强制卖出 - 完整替代vectorBT中的force_exit_intent逻辑
        """
        if self.p.max_holding_days is None:
            return

        for stock_name, days in self.holding_days_counter.items():
            if days >= self.p.max_holding_days:
                data_obj = self.getdatabyname(stock_name)
                position = self.getposition(data_obj)

                if position.size > 0:
                    if self._is_tradable(data_obj):
                        # 立即强制卖出
                        order = self.order_target_percent(data=data_obj, target=0.0)
                        self.forced_exits += 1

                        if self.p.debug_mode:
                            logger.info(f"强制卖出超期持仓: {stock_name}, 持有{days}天")
                    else:
                        # 无法交易，加入待卖清单
                        self.pending_sells[stock_name] = (0, self.datetime.date(0), "强制超期")

    def _process_pending_sells(self):
        """
        处理待卖清单 - 替代vectorBT中的pending_exits_tracker逻辑
        """
        current_date = self.datetime.date(0)

        for stock_name in list(self.pending_sells.keys()):
            retry_count, target_date, reason = self.pending_sells[stock_name]
            data_obj = self.getdatabyname(stock_name)

            if self.getposition(data_obj).size > 0 and self._is_tradable(data_obj):
                # 尝试卖出
                order = self._submit_order_with_pending(stock_name=stock_name, data_obj=data_obj, target_weight=0,
                                                        action='sell')
                # if order:
                #     del self.pending_sells[stock_name]
                #     if self.p.debug_mode:
                #         logger.info(f"延迟卖出成功: {stock_name}, 原因: {reason}")

            # 清理已无持仓的记录
            elif self.getposition(data_obj).size == 0:
                del self.pending_sells[stock_name]

    def _process_pending_buys(self):
        """
        处理待买清单 - 替代vectorBT中的pending_buys_tracker逻辑
        """
        for stock_name in list(self.pending_buys.keys()):
            retry_count, target_date, target_weight = self.pending_buys[stock_name]
            data_obj = self.getdatabyname(stock_name)

            # 检查是否已经持有（可能通过其他方式买入了）
            if self.getposition(data_obj).size > 0:
                del self.pending_buys[stock_name]
                continue

            # 尝试买入
            if self._is_tradable(data_obj):
                self._submit_order_with_pending(stock_name, data_obj, target_weight, 'buy')

    def _execute_rebalancing(self, current_date):
        """
        执行调仓 - 替代vectorBT中复杂的调仓逻辑
        Args:
            current_date: 调仓日期
        """
        if self.p.debug_mode:
            logger.info(f"--- 调仓日: {current_date} ---")

        self.rebalance_count += 1

        # 获取今日的目标持仓信号
        try:
            target_holdings_signal = self.p.holding_signals.loc[pd.to_datetime(current_date)]
            today_want_hold_stocks = target_holdings_signal[target_holdings_signal].index.tolist()
        except KeyError:
            if self.p.debug_mode:
                logger.warning(f"\t\t未找到日期{current_date}的持仓信号")
            return

        if self.p.debug_mode:
            logger.info(f"\t\t目标持仓: {len(today_want_hold_stocks)}只股票")

        # === 阶段1：处理卖出（normal_exits_intent + pending_exits） ===
        self._execute_sell_phase(today_want_hold_stocks)

        # === 阶段2：处理买入（new_buy_intent + pending_buys） ===
        self._execute_buy_phase(today_want_hold_stocks)

    def _execute_sell_phase(self, today_want_hold_stocks: List[str]):
        """
        执行卖出阶段 - 替代vectorBT中的normal_exits_intent逻辑
        
        Args:
            target_stocks: 今日目标持仓股票列表
        """
        sells_attempted = 0
        sells_successful = 0

        # 遍历当前所有持仓
        for data_obj in self.datas:
            stock_name = data_obj._name
            position = self.getposition(data_obj)
            # 这个股票都不是持仓状态
            if position.size <= 0:
                continue

            # 不在今天目标持仓 应该卖掉！
            should_sell_due_to_rebalance = stock_name not in today_want_hold_stocks
            # 遍历所有持仓，发现某只停牌！ 应该也卖掉！
            is_untradable_today = not self._is_tradable(data_obj)
            reason = "发现持仓期间的股票停牌" if is_untradable_today else "调仓不再持有这股票"

            # 只要满足以上任一理由，就必须处理这只股票
            if should_sell_due_to_rebalance or is_untradable_today:
                if self._is_tradable(data_obj):
                    self._submit_order_with_pending(stock_name=stock_name, data_obj=data_obj, target_weight=0.0,
                                                    action='sell')
                else:
                    # 停牌，无法卖出，加入待卖清单
                    if stock_name in self.pending_sells:
                        sells_attempted = self.pending_sells[stock_name][0]
                    self.pending_sells[stock_name] = (sells_attempted+1, self.datetime.date(0), f"{reason}-但停牌导致卖出失败的") #log todo


    def _execute_buy_phase(self, target_stocks: List[str]):
        """
        执行买入阶段 - 替代vectorBT中的new_buy_intent逻辑
        
        Args:
            target_stocks: 目标股票列表
        """
        if not target_stocks:
            return

        # 计算等权重目标权重（考虑当前持仓数量）
        current_holdings_count = len([d for d in self.datas if self.getposition(d).size > 0])
        can_add_positions = min(len(target_stocks), self.p.max_positions-current_holdings_count)
        
        # 防止除零错误
        if can_add_positions <= 0:
            if self.p.debug_mode:
                logger.warning(f"无法添加新持仓: 当前持仓{current_holdings_count}, 最大持仓{self.p.max_positions}")
            return
        
        # 调整权重：给新买入留出空间
        if current_holdings_count > 0:
            target_weight = 0.8 / can_add_positions  # 80%仓位，避免现金不足
        else:
            target_weight = 0.9 / can_add_positions  # 首次建仓可以90%
            

        buys_attempted = 0
        buys_successful = 0

        for stock_name in target_stocks:
            data_obj = self.getdatabyname(stock_name)
            current_position = self.getposition(data_obj).size
            if current_position > 0:
                #持仓状态下！暂不支持加仓！ 先跳过
                continue

            # 只对未持有的股票执行买入
            buys_attempted += 1

            if self._is_tradable(data_obj):
                self._submit_order_with_pending(stock_name, data_obj, target_weight, 'buy')
            else:
                # 停牌，加入待买清单
                self.pending_buys[stock_name] = (0, self.datetime.date(0), target_weight)
                if self.p.debug_mode:
                    logger.warning(
                        f"\t\t\t{self.datetime.date(0)}买入失败(停牌): {stock_name}, 加入待买清单")  # todo 回测 待测试

    #调用函数之前，必须提前判断价格是否存在！
    def _submit_order_with_pending(self, stock_name: str, data_obj, target_weight: float, action: str) -> bool:
        ret = self._submit_order(stock_name, data_obj, target_weight, action)
        if ret:
            return True
        if action=='sell':
             # 如果强制卖出也失败，继续正常流程 (往往是因为当天买入的，无法卖出！）。。。。但是今天卖不掉 就不卖了嘛 比不可能！ 放明天卖
            self.pending_sells[stock_name] = self.push_to_pending_sells(stock_name, "提交强制卖出订单失败")
        if action=='buy':
            self.pending_buys[stock_name] = self.push_to_pending_buys(stock_name, "提交买入订单失败")
        return False
    def _submit_order(self, stock_name: str, data_obj, target_weight: float, action: str) -> bool:
        """
        提交订单 -
        Args:
            stock_name: 股票名称 data_obj: 数据对象 target_weight: 目标权重
        Returns:
            bool: 是否创单成功
        """
        if action == 'buy':
            self.submit_buy_orders += 1
        else:
            self.submit_sell_orders += 1
        try:
            current_position, current_cash, current_price = self.debug_data_for_submit(stock_name,action,target_weight)
            # 对于卖出订单 （无脑强制卖）（为了照顾：如果是COC当日买卖限制  （第一个c是昨日收盘价！我们人为理解是昨天的买入！。但是此框架今天买入！。导致今天无法卖出！，所以出此下策！强制卖！
            if action == 'sell':
                order = self.order_target_size(data=data_obj, target=0)
                if order:
                    logger.info(
                        f"\t\t\t\t{self.datetime.date(0)}-{action}订单提交(强制): {stock_name}")
                    return True



            #买入
            if action == 'buy':
                order = self.order_target_percent(data=data_obj, target=target_weight)
                if order:
                    logger.info(
                        f"\t\t\t\t{self.datetime.date(0)}-{action}订单提交: {stock_name}, 目标权重: {target_weight}")
                    return True

            #都是失败
            # 详细的失败原因分析
            failure_reason = "未知原因"
            if action == 'sell':
                if current_position <= 0:
                    failure_reason = "无持仓可卖"
                elif np.isnan(current_price) or current_price <= 0:
                    failure_reason = "停牌无法卖出"
                else:
                    # 检查是否是COC导致的当日买卖限制
                    if stock_name in self.holding_start_dates:
                        holding_start = self.holding_start_dates[stock_name]
                        if holding_start == self.datetime.date(0):
                            failure_reason = "COC当日买卖限制"
                        else:
                            failure_reason = "卖出订单被拒绝"
                    else:
                        failure_reason = "卖出订单被拒绝"
            elif action == 'buy':
                if current_cash < current_price * 100:  # 至少能买100股
                    failure_reason = "现金不足"
                elif np.isnan(current_price) or current_price <= 0:
                    failure_reason = "价格异常/停牌"
                elif target_weight < 0.001:
                    failure_reason = "目标权重过小"
                else:
                    failure_reason = "买入订单被拒绝"
                    
            logger.warning(f"{self.datetime.date(0)}-{action}订单提交失败: {stock_name} (原因: {failure_reason})")
            logger.warning(f"  现金: {current_cash:.2f}, 持仓: {current_position}, 价格: {current_price:.2f}, 目标权重: {target_weight}")
            return False
                
        except Exception as e:
            logger.error(f"{self.datetime.date(0)}-Error executing {action} order for {stock_name}: {e}")
            logger.error(f"  异常详情: 现金={self.broker.get_cash():.2f}, 价格={data_obj.close[0]:.2f}")
            raise ValueError(e)

    def _is_tradable(self, data_obj) -> bool:
        """
        检查股票是否可交易 - 完整替代vectorBT中的is_tradable_today逻辑
        Args:  data_obj: 数据对象
        Returns:bool: 是否可交易
        """
        try:
            # 检查是否有有效价格数据
            current_price = data_obj.close[0]
            return not (np.isnan(current_price) or current_price <= 0)
        except:
            return False

    def _cleanup_position_records(self, stock_name: str):
        """
        清理已平仓股票的所有记录
        Args:
            stock_name: 股票名称
        """
        records_to_clean = [
            self.holding_start_dates,
            self.holding_days_counter,
            self.actual_positions
        ]

        for record_dict in records_to_clean:
            if stock_name in record_dict:
                del record_dict[stock_name]
    def debug_data_for_submit(self,stock_name,action,target_weight):
        data_obj = self.getdatabyname(stock_name)
        current_position = self.getposition(data_obj).size
        # check
        if action == 'sell':
            if target_weight != 0:
                raise   ValueError("卖出订单的目标权重必须为0")
            if current_position <= 0:
                raise  ValueError("卖出订单 但是居然没有发现持仓（大概率是之前买入失败！严重错误 或者是这卖出信号不准！")

        # 增加调试信息：检查订单提交前的状态
        current_cash = self.broker.get_cash()

        current_price = data_obj.close[0]

        current_value,_ = self.get_current_value_approximate()

        if self.p.debug_mode:
            logger.debug(f"\t\t\t订单前状态 - 现金: {current_cash:.2f}, 总价值: {current_value:.2f}, "
                         f"{stock_name}价格: {current_price:.2f}, 当前持仓: {current_position}")
        return current_position, current_cash ,current_price
    def get_current_value_approximate(self):
        """
       解决Backtrader在某些情况下get_value返回NaN的问题
        Returns:
            float: 估算的总资产价值
        """
        current_value = self.broker.get_value()
        if not np.isnan(current_value):
            return current_value,False

        current_cash = self.broker.get_cash()

        # 估算总价值
        sum_value=0
        for d, pos in self.positions.items():
            if pos.size != 0:
                valid_price = self.find_last_notNa_price(d)
                if not np.isnan(valid_price):
                    sum_value += pos.size * valid_price

        return current_cash + sum_value, True
    def find_last_notNa_price(self,data):
        """
           从当前bar往前找，返回最近一个非NaN的收盘价
           Args:
               data: Backtrader数据feed
           Returns:
               float: 最近的有效价格（找不到则返回 np.nan）
           """
        # 当前索引位置
        i = 0
        # 从当前开始往前查找
        while True:
            try:
                price = data.close[-i]
            except IndexError:
                # 超出历史范围，返回NaN
                return np.nan
            if not np.isnan(price):
                return price
            i += 1
    def refresh_for_success_buy(self, stock_name: str, pending_buys_snap):
        """
        成功买入后的记录刷新
        Args:
            stock_name: 股票名称
        """
        self.success_buy_orders += 1
        # 初始化持仓记录
        self.holding_start_dates[stock_name] = self.datetime.date(0)
        self.holding_days_counter[stock_name] = 1
        self.actual_positions[stock_name] = self.getdatabyname(stock_name)
        # 移除，反正我今天是买到了
        if stock_name in pending_buys_snap:
            del self.pending_buys[stock_name]

    def refresh_for_success_sell(self, stock_name: str, pending_sells_snap):
        """
        成功卖出后的记录刷新
        Args:
            stock_name: 股票名称
        """
        self.success_sell_orders += 1
        # 清理已平仓的记录
        self._cleanup_position_records(stock_name)
        # 移除，反正我今天是卖出去了
        if stock_name in pending_sells_snap:
            del self.pending_sells[stock_name]

    def push_to_pending_sells(self,stock_name,descrip):
        old_retrys = 0
        if stock_name in self.pending_sells:
            old_retrys = self.pending_sells[stock_name]
        self.pending_sells[stock_name] = (old_retrys+1, self.datetime.date(0), descrip)
    def push_to_pending_buys(self,stock_name,descrip):
        old_retrys = 0
        if stock_name in self.pending_buys:
            old_retrys = self.pending_buys[stock_name]
        self.pending_sells[stock_name] = (old_retrys+1, self.datetime.date(0), descrip)
    def notify_order(self, order):
        """
        订单状态通知 - 增强的交易状态处理
        """
        stock_name = order.data._name
        current_date = self.datetime.date(0)
        pending_buys_snap = self.pending_buys
        pending_sells_snap = self.pending_sells
        # 订单成功执行
        if order.status == order.Completed:
            action = "买入" if order.isbuy() else "卖出"
            actionTimeType = "延迟日级别重试" if (
                        (stock_name in pending_sells_snap) or (stock_name in pending_sells_snap)) else "调仓"

            if order.isbuy():
                # 初始化持仓记录
                self.refresh_for_success_buy(stock_name, pending_buys_snap)
            if order.issell():
                # 卖出成功，清理记录
                self.refresh_for_success_sell(stock_name, pending_sells_snap)

            if self.p.log_detailed:
                current = get_last_b_day(self.p.trading_days,pd.Timestamp(self.datetime.date(0)))
                logger.info(f"\t\t\t{current}--{actionTimeType}-{action}-成功: {stock_name}, "
                            f"股数: {order.executed.size:.0f}, "
                            f"价格: {order.executed.price:.2f},"
                            f"乘积: {order.executed.price * order.executed.size}")

        # 订单失败处理
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.failed_orders += 1
            action = "买入" if order.isbuy() else "卖出"

            # 记录失败原因
            failure_record = {
                'date': current_date,
                'stock': stock_name,
                'action': action,
                'status': order.getstatusname(),
                'price': order.data.close[0],
                'cash': self.broker.get_cash(),
                'value': self.broker.get_value()
            }

            # 根据失败类型决定重试策略
            if order.isbuy() and self.p.enable_retry:
                # 买入失败，加入待买清单（如果还没在清单中）
                if stock_name not in self.pending_buys:
                    # 使用固定权重，避免复杂计算
                    fallback_weight = 0.1  # 失败重试时使用较小权重 todo
                    self.pending_buys[stock_name] = (1, current_date, fallback_weight)

                    if self.p.debug_mode:
                        logger.info(f"买入失败，加入重试: {stock_name}, 原因: {failure_record}")

            elif order.issell():
                # 卖出失败，加入待卖清单
                if stock_name not in self.pending_sells:
                    self.pending_sells[stock_name] = (1, current_date, "卖出重试")

                    if self.p.debug_mode:
                        logger.warning(f"卖出失败，加入重试: {stock_name}, 原因: {order.getstatusname()}")

    # 注意场景！
    def _calculate_dynamic_weight(self, need_buy_count, ) -> float:  # todo 需要测试 回测
        """
        动态计算目标权重 - 根据当前现金和持仓情况
        Returns:
            float: 动态计算的目标权重
        """
        # 计算当前实际持仓数量
        current_positions = len([d for d in self.datas if self.getposition(d).size > 0])

        # 计算待买数量
        pending_count = len(self.pending_buys)

        # 总目标持仓数
        total_target = min(self.p.max_positions, current_positions + pending_count + 1)

        # 动态权重分配
        if total_target > 0:
            return 1.0 / total_target
        else:
            return 1.0 / self.p.max_positions

    def _log_daily_status(self, current_date):
        """
        记录每日状态 - 用于调试和监控
        
        Args:
            current_date: 当前日期
        """
        # 统计当前状态
        current_holdings_count = len([d for d in self.datas if self.getposition(d).size > 0])
        pending_sells_count = len(self.pending_sells)
        pending_buys_count = len(self.pending_buys)
        total_value = self.broker.get_value()
        cash_ratio = self.broker.get_cash() / total_value

        daily_stat = {
            'date': current_date,
            'holdings': current_holdings_count,
            'pending_sells': pending_sells_count,
            'pending_buys': pending_buys_count,
            'cash_ratio': cash_ratio,
            'total_value': total_value
        }

        self.daily_stats.append(daily_stat)

        if self.p.debug_mode:
            logger.info(f"\t\t{current_date}: 持仓{current_holdings_count}只, "
                        f"待卖{pending_sells_count}只, 待买{pending_buys_count}只, "
                        f"现金比例{cash_ratio:.1%}")

    def stop(self):
        """策略结束处理 - 详细统计和分析"""
        logger.info("=" * 80)
        logger.info("策略执行完成 - 详细统计报告")
        logger.info("=" * 80)

        # 基本统计
        final_value = self.broker.getvalue()
        total_return = (final_value / self.broker.startingcash - 1) * 100

        logger.info(f"资金统计:")
        logger.info(f"  初始资金: {self.broker.startingcash:,.2f}")
        logger.info(f"  最终资金: {final_value:,.2f}")
        logger.info(f"  总收益率: {total_return:.2f}%")

        # 交易统计
        success_rate = self.success_buy_orders / max(self. submit_buy_orders, 1) * 100
        #统计基准！！
        logger.info(f"交易统计:")
        logger.info(f"  总提交订单数（提交买入订单数）: {self. submit_buy_orders}")
        logger.info(f"  失败订单（别担心，反正有重试: {self.failed_orders}")
        logger.info(f"  买入成功率: {success_rate:.1f}%")

        # 调仓统计
        logger.info(f"调仓统计:")
        logger.info(f"  调仓次数: {self.rebalance_count}")
        logger.info(f"  强制卖出: {self.forced_exits}次")
        logger.info(f"  紧急止损: {self.emergency_exits}次")

        # 待处理队列统计
        if self.pending_buys or self.pending_sells:
            logger.info(f"未完成任务:")
            logger.info(f"  待买清单: {len(self.pending_buys)}只")
            logger.info(f"  待卖清单: {len(self.pending_sells)}只")

            if self.pending_buys:
                logger.info("  待买股票:", list(self.pending_buys.keys()))
            if self.pending_sells:
                logger.info("  待卖股票:", list(self.pending_sells.keys()))

        # 持仓分析
        self._analyze_holding_patterns()

    def _analyze_holding_patterns(self):
        """
        分析持仓模式 - 替代vectorBT中的_debug_holding_days逻辑
        """
        if not self.daily_stats:
            return

        logger.info("持仓模式分析:")

        # 转换为DataFrame进行分析
        stats_df = pd.DataFrame(self.daily_stats)

        avg_holdings = stats_df['holdings'].mean()
        max_holdings = stats_df['holdings'].max()
        min_holdings = stats_df['holdings'].min()

        avg_cash_ratio = stats_df['cash_ratio'].mean()

        logger.info(f"  平均持仓: {avg_holdings:.1f}只")
        logger.info(f"  最大持仓: {max_holdings}只")
        logger.info(f"  最小持仓: {min_holdings}只")
        logger.info(f"  平均现金比例: {avg_cash_ratio:.1%}")

        # 分析待处理队列的变化
        avg_pending_buys = stats_df['pending_buys'].mean()
        avg_pending_sells = stats_df['pending_sells'].mean()

        if avg_pending_buys > 0.5:
            logger.warning(f"⚠️ 买入执行困难，平均待买: {avg_pending_buys:.1f}只")
        if avg_pending_sells > 0.5:
            logger.warning(f"⚠️ 卖出执行困难，平均待卖: {avg_pending_sells:.1f}只")


class BacktraderMigrationEngine:
    """
    Backtrader迁移引擎 - 一键式从vectorBT迁移的完整解决方案
    """

    def __init__(self, original_config=None):
        """
        初始化迁移引擎
        
        Args:
            original_config: 原有的vectorBT配置对象
        """
        self.original_config = original_config
        self.bt_config = self._convert_config(original_config)
        self.results = {}

        logger.info("BacktraderMigrationEngine初始化完成")

    def _convert_config(self, vectorbt_config) -> Dict:
        """
        配置转换 - 从vectorBT配置转换为Backtrader配置
        
        Args:
            vectorbt_config: 原有配置对象
            
        Returns:
            Dict: Backtrader配置字典
        """
        if vectorbt_config is None:
            return self._default_config()

        return {
            'top_quantile': getattr(vectorbt_config, 'top_quantile', 0.2),
            'rebalancing_freq': getattr(vectorbt_config, 'rebalancing_freq', 'M'),
            'commission_rate': getattr(vectorbt_config, 'commission_rate', 0.0003),
            'slippage_rate': getattr(vectorbt_config, 'slippage_rate', 0.001),
            'stamp_duty': getattr(vectorbt_config, 'stamp_duty', 0.001),
            'initial_cash': getattr(vectorbt_config, 'initial_cash', 1000000.0),
            'max_positions': getattr(vectorbt_config, 'max_positions', 10),
            'max_holding_days': getattr(vectorbt_config, 'max_holding_days', 60),
            'retry_buy_days': 3,
            'max_weight_per_stock': getattr(vectorbt_config, 'max_weight_per_stock', 0.15),
            'min_weight_threshold': getattr(vectorbt_config, 'min_weight_threshold', 0.01)
        }

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'top_quantile': 0.2,
            'rebalancing_freq': 'M',
            'commission_rate': 0.0003,
            'slippage_rate': 0.001,
            'stamp_duty': 0.001,
            'initial_cash': 1000000.0,
            'max_positions': 10,
            'max_holding_days': 60,
            'retry_buy_days': 3,
            'max_weight_per_stock': 0.15,
            'min_weight_threshold': 0.01
        }

    def migrate_and_run(self, price_df: pd.DataFrame, factor_dict: Dict[str, pd.DataFrame],
                        comparison_with_vectorbt: bool = True) -> Dict:
        """
        一键迁移并运行 - 完整替代原有的run_backtest函数
        
        Args:
            price_df: 价格数据
            factor_dict: 因子数据字典
            comparison_with_vectorbt: 是否与vectorBT结果对比
            
        Returns:
            Dict: 迁移结果
        """
        migration_results = {}

        for factor_name, factor_data in factor_dict.items():

            try:
                # === 1. 数据对齐（兼容原有逻辑）===
                aligned_price, aligned_factor = self._align_data(price_df, factor_data)

                # === 2. 生成持仓信号（完整替代generate_long_holding_signals）===
                holding_signals = self._generate_holding_signals(aligned_factor, aligned_price)


                cerebro = bt.Cerebro()
                cerebro.broker.set_coc(True)  # cheat-on-close: 当天收盘价成交

                # 添加数据源
                self.add_wide_df_to_cerebro(cerebro, aligned_price, aligned_factor)

                # 生成调仓日期
                rebalance_dates = generate_rebalance_dates(
                    aligned_factor.index,
                    self.bt_config['rebalancing_freq']
                )

                # 添加策略
                cerebro.addstrategy(
                    EnhancedFactorStrategy,
                    factor_data=aligned_factor,
                    holding_signals=holding_signals,
                    rebalance_dates=rebalance_dates,
                    max_positions=self.bt_config['max_positions'],
                    max_holding_days=self.bt_config['max_holding_days'],
                    retry_buy_days=self.bt_config['retry_buy_days'],
                    debug_mode=True,
                    trading_days=load_trading_lists(aligned_factor.index[0],aligned_price.index[-1]),
                    log_detailed=True
                )

                # === 4. 配置交易环境 ===
                cerebro.broker.setcash(self.bt_config['initial_cash'])
                # 综合费率计算 #todo 有空再改为 买卖分别计算税率 影响不大
                comprehensive_fee = (
                        self.bt_config['commission_rate'] +
                        self.bt_config['slippage_rate'] +
                        self.bt_config['stamp_duty'] / 2
                )
                cerebro.broker.setcommission(commission=comprehensive_fee)

                # === 5. 添加分析器 ===
                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
                cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

                # === 6. 执行回测 ===
                logger.info(f"开始执行{factor_name}回测...")
                start_time = datetime.now()

                strategy_results = cerebro.run()

                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()

                # === 7. 提取结果 ===
                strategy = strategy_results[0]
                final_value = cerebro.broker.getvalue()

                migration_results[factor_name] = {
                    'strategy': strategy,
                    'final_value': final_value,
                    'execution_time': execution_time,
                    'analyzers': strategy.analyzers,
                    'config_used': self.bt_config.copy()
                }

                logger.info(f"{factor_name} 迁移完成: 最终价值 {final_value:,.2f}, "
                            f"耗时 {execution_time:.2f}秒")

            except Exception as e:
                raise ValueError("失败") from e

        self.results = migration_results

        return migration_results

    import backtrader as bt
    import pandas as pd
    from typing import List

    def add_wide_df_to_cerebro(self, cerebro: bt.Cerebro, wide_price_df: pd.DataFrame,
                               factor_wide_df: pd.DataFrame) -> None:
        """
        读取一个宽格式的DataFrame，并为每一列（每只股票）创建和添加一个
        独立的Backtrader数据源。
        Args:
            cerebro: backtrader.Cerebro 引擎实例。
            wide_price_df: 宽格式的价格DataFrame (index=date, columns=symbols, values=close)。
        """
        wide_price_df, factor_wide_df = self._align_data(wide_price_df, factor_wide_df)
        # 获取 startTime end time
        startTime = wide_price_df.index[0]
        endTime = wide_price_df.index[-1]
        # --- 遍历宽格式DataFrame的每一列 ---
        for stock_symbol in wide_price_df.columns:
            # 1. 为单只股票准备符合OHLCV格式的数据
            #    注意：Backtrader 需要 open, high, low, close, volume, openinterest 这几个标准列名
            df_single_stock = pd.DataFrame(index=wide_price_df.index)

            # 【核心】将宽表中的'close'价格，赋给符合backtrader格式的DataFrame
            df_single_stock['close'] = wide_price_df[stock_symbol]

            # 简化处理：如果你的宽表没有OHLV数据，可以用close填充
            # 在真实的回测中，你应该传入包含真实OHLCV的宽表
            df_single_stock['open'] = df_single_stock['close']
            df_single_stock['high'] = df_single_stock['close']
            df_single_stock['low'] = df_single_stock['close']
            df_single_stock['close'] = df_single_stock['close']
            df_single_stock['volume'] = 0  # 如果没有成交量数据，用0填充
            df_single_stock['openinterest'] = 0  # 股票没有这个，必须用0填充
            # 因子数据 todo
            # df_single_stock['r_20d'] =

            # 2. 为这只股票创建一个独立的 PandasData Feed
            #    `name=stock_symbol` 至关重要，用于后续在策略中通过名字识别它
            data_feed = bt.feeds.PandasData(
                dataname=df_single_stock,
                fromdate=startTime, todate=endTime
            )
            # 3. 将这个独立的数据源添加到 Cerebro
            cerebro.adddata(data_feed, name=stock_symbol)

    def _align_data(self, price_df: pd.DataFrame, factor_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        数据对齐 - 兼容原有的vectorBT对齐逻辑
        
        Args:
            price_df: 价格数据
            factor_df: 因子数据
        Returns:
            Tuple: 对齐后的(价格数据, 因子数据)
        """
        # 时间对齐
        common_dates = price_df.index.intersection(factor_df.index)

        # 股票对齐  
        common_stocks = price_df.columns.intersection(factor_df.columns)

        aligned_price = price_df.loc[common_dates, common_stocks]
        aligned_factor = factor_df.loc[common_dates, common_stocks]

        logger.info(f"数据对齐完成: {aligned_price.shape}, 共同日期{len(common_dates)}, 共同股票{len(common_stocks)}")

        return aligned_price, aligned_factor

    def _generate_holding_signals(self, factor_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成持仓信号 - 完整替代原有逻辑
        
        Args:
            factor_df: 对齐后的因子数据
            price_df: 对齐后的价格数据
            
        Returns:
            pd.DataFrame: 持仓信号矩阵
        """
        # 计算每日排名百分位
        ranks = factor_df.rank(axis=1, pct=True, method='average', na_option='keep')

        # 生成调仓日期
        rebalance_dates = generate_rebalance_dates(factor_df.index, self.bt_config['rebalancing_freq'])

        # 初始化持仓信号矩阵
        holding_signals = pd.DataFrame(False, index=factor_df.index, columns=factor_df.columns)

        # 当前持仓组合（调仓间隔期间保持不变）
        current_positions = None

        for date in factor_df.index:
            is_rebalance_day = date in rebalance_dates

            if is_rebalance_day:
                # 调仓日：重新选择股票
                daily_valid_ranks = ranks.loc[date].dropna()

                if len(daily_valid_ranks) > 0:
                    # 计算目标持仓数
                    num_to_select = int(np.ceil(len(daily_valid_ranks) * self.bt_config['top_quantile']))
                    if self.bt_config['max_positions']:
                        num_to_select = min(num_to_select, self.bt_config['max_positions'])

                    # 选择排名最高的股票
                    chosen_stocks = daily_valid_ranks.nlargest(num_to_select).index
                    current_positions = chosen_stocks

            # 重复前一天
            if current_positions is not None:
                holding_signals.loc[date, current_positions] = True

        #兜底保证，停牌日的持仓信号为False
        holding_signals[price_df.isna() | (price_df <= 0)] = False
        return holding_signals

    def get_comparison_with_vectorbt(self, vectorbt_results: Dict = None) -> pd.DataFrame:
        """
        与vectorBT结果对比
        
        Args:
            vectorbt_results: vectorBT回测结果
            
        Returns:
            pd.DataFrame: 对比结果表
        """
        if not self.results:
            raise ValueError("请先运行Backtrader回测")

        # 提取Backtrader结果
        bt_comparison_data = {}

        for factor_name, result in self.results.items():
            if result is None:
                continue

            try:
                analyzers = result['analyzers']
                total_return = (result['final_value'] / self.bt_config['initial_cash'] - 1) * 100
                sharpe_ratio = analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
                max_drawdown = abs(analyzers.drawdown.get_analysis()['max']['drawdown'])

                bt_comparison_data[factor_name] = {
                    'Total Return [%]': total_return,
                    'Sharpe Ratio': sharpe_ratio,
                    'Max Drawdown [%]': max_drawdown,
                    'Framework': 'Backtrader'
                }

            except Exception as e:
                logger.error(f"提取{factor_name}结果时出错: {e}")

        bt_df = pd.DataFrame(bt_comparison_data).T

        # 如果提供了vectorBT结果，进行对比
        if vectorbt_results:
            # 这里可以添加详细的对比逻辑
            logger.info("Backtrader vs vectorBT 结果对比:")
            print(bt_df)

        return bt_df


# === 便捷迁移函数 ===

def one_click_migration(price_df: pd.DataFrame, factor_dict: Dict[str, pd.DataFrame],
                        original_vectorbt_config=None) -> Tuple[Dict, pd.DataFrame]:
    """
    Args:
        price_df: 价格数据
        factor_dict: 因子数据字典  
        original_vectorbt_config: 原有的vectorBT配置
    Returns:
        Tuple: (Backtrader回测结果, 对比表)
    """
    # 创建迁移引擎
    migration_engine = BacktraderMigrationEngine(original_vectorbt_config)

    # 执行迁移和回测
    results = migration_engine.migrate_and_run(price_df, factor_dict)

    # 生成对比表
    comparison_table = migration_engine.get_comparison_with_vectorbt()

    return results, comparison_table


if __name__ == "__main__":
    logger.info("Backtrader增强策略测试")

    # 测试示例：
    # 假设你有原有的数据和配置
    # results, comparison = one_click_migration(price_df, factor_dict, original_config)
