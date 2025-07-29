"""
数据管理器 - 单因子测试终极作战手册
第二阶段：数据加载与股票池构建

实现配置驱动的数据加载和动态股票池构建功能
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
import sys
import os

from pandas import DatetimeIndex

from data.local_data_load import load_index_daily
from data.namechange_date_manager import fill_end_date_field
from quant_lib.data_loader import DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR
from quant_lib.config.logger_config import setup_logger

warnings.filterwarnings('ignore')

# 配置日志
logger = setup_logger(__name__)


def check_field_level_completeness(raw_df: Dict[str, pd.DataFrame]):
    dfs = raw_df.copy()
    for item_name, df in dfs.items():
        logger.info("原始字段缺失率体检报告:")
        # missing_rate_daily = df.isna().mean(axis=1)

        # logger.info(f"{item_name}因子缺失率最高的10天 between {first_date} and {end_date}")
        # logger.info(f"{missing_rate_daily.sort_values(ascending=False).head(10)}")  # 其实也不需要太看重，只能说是辅助日志，如果总缺失率高 可以看看整个辅助排查而已！

        # 计算每只股票（每一列）的缺失率(相当于看这股票 在这一段时间的完整率！---》推导：最后一天才上市！，那么缺失率可能高达99.99% 所以不需要看重这个！)  注释掉
        missing_rate_per_stock = df.isna().mean(axis=0)
        #
        # logger.info(f"{item_name}（不是很重要）因子缺失率最高的10只股票 between {first_date} and {end_date}")
        # logger.info(f"{missing_rate_per_stock.sort_values(ascending=False).head(10)}")

        # 计算整个DataFrame的缺失率
        total_cells = df.size
        df_all_cells = df.isna().sum().sum()
        global_na_ratio = df_all_cells / total_cells
        logger.info(_get_nan_comment(item_name, global_na_ratio))


def _get_nan_comment(field: str, rate: float) -> str:
    logger.info(f"field：{field}在原始raw_df 确实占比为：{rate}")
    if rate >= 0.5:
        raise f"field:{field}缺失率超过50% 必须检查"
    """根据字段名称和缺失率，提供专家诊断意见"""
    if field in ['pe_ttm', 'pe', 'pb',
                 'pb_ttm'] and rate <= 0.4:  # 亲测 很正常，有的垃圾股票 price earning 为负。那么tushare给我的数据就算nan，合理！
        return " (正常现象: 主要代表公司亏损)"

    if field in ['dv_ttm', 'dv_ratio']:
        return " (正常现象: 主要代表公司不分红, 后续应填充为0)"

    if field in ['industry']:  # 亲测 industry 可以直接放行，不需要care 多少缺失率！因为也就300个，而且全是退市的，
        return "正常现象：不需要care 多少缺失率"
    if field in ['circ_mv', 'close', 'total_mv',
                 'turnover_rate', 'open', 'high', 'low',
                 'pre_close'] and rate < 0.2:  # 亲测 一大段时间，可能有的股票最后一个月才上市，导致前面空缺，有缺失 那很正常！
        return "正常现象：不需要care 多少缺失率"
    if field in ['list_date'] and rate <= 0.01:
        return "正常现象：不需要care 多少缺失率"
    if field in ['pct_chg'] and rate <= 0.10:
        return  "正常"
    raise ValueError(f"(🚨 警告: 此字段{field}缺失ratio:{rate}!) 请自行配置通过ratio 或则是缺失率太高！")


class DataManager:
    """
    数据管理器 - 负责数据加载和股票池构建
    
    按照配置文件的要求，实现：
    1. 原始数据加载
    2. 动态股票池构建
    3. 数据质量检查
    4. 数据对齐和预处理
    """

    def __init__(self, config_path: str, need_data_deal: bool = True):
        """
        初始化数据管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.st_matrix = None
        self.config = self._load_config(config_path)
        if need_data_deal:
            self.data_loader = DataLoader(data_path=LOCAL_PARQUET_DATA_DIR)
            self.raw_dfs = {}
            self.stock_pools_dict = None

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def processed_raw_data_dict_by_stock_pool_(self) -> Dict[str, pd.DataFrame]:
        """
        优化的两阶段数据处理流水线（只加载一次数据）

        Returns:
            处理后的数据字典
        """

        # 获取时间范围
        start_date = self.config['backtest']['start_date']
        end_date = self.config['backtest']['end_date']

        # 确定所有需要的字段（一次性确定）
        all_required_fields = self._get_required_fields()

        # === 一次性加载所有raw数据(互相对齐) ===

        self.raw_dfs = self.data_loader.get_raw_dfs_by_require_fields(fields=all_required_fields,
                                                                      start_date=start_date, end_date=end_date)

        check_field_level_completeness(self.raw_dfs)
        logger.info(f"raw_dfs加载完成，共加载 {len(self.raw_dfs)} 个字段")

        # === 第一阶段：基于已加载数据构建权威股票池 ===
        logger.info("第一阶段：构建两个权威股票池（各种过滤！）")
        self._build_stock_pools_from_loaded_data(start_date, end_date)

        # === 第二阶段：基于股票池对齐和清洗所有数据 ===
        logger.info("第二阶段：(根据因子门派类别)对齐和填充所有因子数据")

        # 使用权威股票池对齐和填充数据
        self.processed_raw_data = self._align_many_raw_dfs_by_stock_pool_and_fill(self.raw_dfs)
        # 强行检查一下数据！完整率！ 不应该在这里检查！，太晚了， 已经被stock_pool_df 动了手脚了（低市值的会被置为nan，

        return self.processed_raw_data

    # ok
    def _build_stock_pools_from_loaded_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        第一阶段：基于已加载的数据构建权威股票池

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            权威股票池DataFrame
        """
        # print("1. 验证股票池构建所需数据...")

        # 验证必需字段是否已加载
        required_fields_for_universe = ['close', 'total_mv', 'turnover_rate', 'industry', 'list_date']
        missing_fields = [field for field in required_fields_for_universe if field not in self.raw_dfs]

        if missing_fields:
            raise ValueError(f"构建股票池缺少必需字段: {missing_fields}")

        # 获取所有股票和交易日期
        ts_codes = list(set(self.get_price_data().columns))
        trading_dates = self.data_loader.get_trading_dates(start_date=start_date, end_date=end_date)

        # 构建ST矩阵
        self.build_st_period_from_namechange(ts_codes, self.get_namechange_data(), trading_dates)
        self.build_diff_stock_pools()

    def build_diff_stock_pools(self) -> pd.DataFrame:
        stock_pool_df_dict = {}
        stock_pool_profiles = self.config['stock_pool_profiles']
        for universe_profile in stock_pool_profiles:
            pool_name = next(iter(universe_profile))
            product_universe = self.product_stock_pool(universe_profile, pool_name)
            stock_pool_df_dict[pool_name] = product_universe
        self.stock_pools_dict = stock_pool_df_dict

    # institutional_profile   = stock_pool_profiles['institutional_profile']#为“基本面派”和“趋势派”因子，提供一个高市值、高流动性的环境
    # microstructure_profile = stock_pool_profiles['microstructure_profile']#用于 微观（量价/情绪）因子
    # product_universe =self.product_universe (microstructure_profile,trading_dates)

    # 对于 是先 fill 还是先where 的考量 ：还是别先ffill了：极端例子：停牌了99天的，100。 若先ffill那么 这100天都是借来的数据！  如果先where。那么直接统统nan了。在ffill也是nan，更具真实
    # ok
    def _align_many_raw_dfs_by_stock_pool_and_fill(self, raw_dfs: Dict[str, pd.DataFrame],
                                                   stock_pool_param: pd.DataFrame = None,
                                                   ) -> Dict[str, pd.DataFrame]:
        """
        第二阶段：使用权威股票池对齐和清洗所有数据

        Args:
            raw_dfs: 原始数据字典
            stock_pool_df: 权威股票池DataFrame
        Returns:
            对齐和清洗后的数据字典
        """
        aligned_data = {}
        for factor_name, raw_df in raw_dfs.items():
            # 1. 确定当前因子需要哪个股票池！
            aligned_df = self.__align_one_raw_dfs_by_stock_pool_and_fill(factor_name, raw_df, stock_pool_param)
            aligned_data[factor_name] = aligned_df
        return aligned_data

    def _get_required_fields(self) -> List[str]:
        """获取所有需要的字段"""
        required_fields = set()

        # 基础字段
        required_fields.update([
            'pct_chg',  # 股票收益与指数收益的联动beta (用于中性化 进一步净化因子 它能为动量因子“降噪”，额外剔除市场系统性风险（Beta）的影响。

            'close',
            'pb',  # 为了计算价值类因子
            'total_mv', 'turnover_rate',  # 为了过滤 很差劲的股票 仅此而已，不会作其他计算 、'total_mv'还可 用于计算中性化
            'industry',  # 用于计算中性化
            'circ_mv',  # 流通市值 用于WOS，加权最小二方跟  ，回归法会用到
            'list_date',  # 上市日期,

            'open', 'high', 'low', 'pre_close'  # 为了计算次日是否一字马涨停
        ])

        # 目标因子所需基础字段
        target_factors_for_evaluation = self.config['target_factors_for_evaluation']
        required_fields.update(self.get_cal_base_factors(target_factors_for_evaluation['fields']))

        # 中性化需要的字段
        neutralization = self.config['preprocessing']['neutralization']
        if neutralization['enable']:
            if 'industry' in neutralization['factors']:
                required_fields.add('industry')
            if 'market_cap' in neutralization['factors']:
                required_fields.add('total_mv')
        return list(required_fields)

    def _check_data_quality(self):
        """检查数据质量"""
        print("  检查数据完整性和质量...")

        for field_name, df in self.raw_dfs.items():
            # 检查数据形状
            print(f"  {field_name}: {df.shape}")

            # 检查缺失值比例
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            print(f"    缺失值比例: {missing_ratio:.2%}")

            # 检查异常值
            if field_name in ['close', 'total_mv', 'pb', 'pe_ttm']:
                negative_ratio = (df <= 0).sum().sum() / df.notna().sum().sum()
                print(f"  极值(>99%分位) 占比: {((df > df.quantile(0.99)).sum().sum()) / (df.shape[0] * df.shape[1])}")

                if negative_ratio > 0:
                    print(f"    警告: {field_name} 存在 {negative_ratio:.2%} 的非正值")

    def _build_universe(self) -> pd.DataFrame:
        """
        构建动态股票池
        Returns:
            股票池DataFrame，True表示该股票在该日期可用
        """
        print("  构建基础股票池...")

        # 第一步：基础股票池 - 有价格数据的股票
        if 'close' not in self.raw_dfs:
            raise ValueError("缺少价格数据，无法构建股票池")

        base_stock_pool_df = self.raw_dfs['close'].notna()
        final_stock_pool_df = base_stock_pool_df
        self.show_stock_nums_for_per_day('根据收盘价notna生成的', base_stock_pool_df)
        # 第二步：各种过滤！
        # --基础过滤 指数成分股过滤（如果启用）
        index_config = self.config['stack_pool'].get('index_filter', {})
        if index_config.get('enable', False):
            # print(f"    应用指数过滤: {index_config['index_code']}")
            final_stock_pool_df = self._build_dynamic_index_universe(base_stock_pool_df, index_config['index_code'])
            # ✅ 在这里进行列修剪是合理的！ 因为中证800成分股是基于外部规则，不是基于未来数据表现
            valid_stocks = final_stock_pool_df.columns[final_stock_pool_df.any(axis=0)]
            final_stock_pool_df = final_stock_pool_df[valid_stocks]
        # --普适性 过滤 （通用过滤）
        final_stock_pool_df = self._filter_new_stocks(final_stock_pool_df, 6)  # 新股票数据少，不具参考
        final_stock_pool_df = self._filter_st_stocks(final_stock_pool_df)  # 剔除ST股票

        # 其他各种指标过滤条件
        universe_filters = self.config['stack_pool']['filters']

        # 2. 流动性过滤
        if 'min_liquidity_percentile' in universe_filters:
            print("    应用流动性过滤...")
            final_stock_pool_df = self._filter_by_liquidity(
                final_stock_pool_df,
                universe_filters['min_liquidity_percentile']
            )

        # 3. 市值过滤
        if 'min_market_cap_percentile' in universe_filters:
            # print("    应用市值过滤...")
            final_stock_pool_df = self._filter_by_market_cap(
                final_stock_pool_df,
                universe_filters['min_market_cap_percentile']
            )

        # 剔除次日停牌股票
        final_stock_pool_df = self._filter_next_day_suspended(final_stock_pool_df)
        # 剔除涨停股票
        final_stock_pool_df = self._filter_next_day_limit_up(final_stock_pool_df)
        return final_stock_pool_df

    # ok
    def build_st_period_from_namechange(
            self,
            ts_codes: list,
            namechange_df: pd.DataFrame,
            trading_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
         【最终无懈可击版】根据namechange历史，重建每日“已知风险”状态矩阵。
         此版本通过searchsorted隐式处理初始状态，逻辑最简且结果正确。
         """
        logger.info("正在根据名称变更历史，重建每日‘已知风险’状态st矩阵...")

        # --- 1. 准备工作 ---
        if not trading_dates._is_monotonic_increasing:
            trading_dates = trading_dates.sort_values(ascending=True)

        # 【关键】必须按“生效日”排序，以确保状态的正确延续和覆盖
        namechange_df['start_date'] = pd.to_datetime(namechange_df['start_date'])
        namechange_df.sort_values(by=['ts_code', 'start_date'], inplace=True)

        # 【关键】必须用 np.nan 初始化，作为“未知状态”
        st_matrix = pd.DataFrame(np.nan, index=trading_dates, columns=ts_codes)

        # --- 2. “打点”：一个循环处理所有历史事件 ---
        for ts_code, group in namechange_df.groupby('ts_code'):
            group_sorted = group.sort_values(by='start_date')
            for _, row in group_sorted.iterrows():
                start_date = row['start_date']

                # 发生在回测期前的日期，会被自动映射到位置 0  or 发生在回测期内的日期，会被映射到它对应的正确位置
                start_date_loc = trading_dates.searchsorted(start_date,
                                                            side='left')  # 遍历trading_dates找到首个>=start_date的下标！ 如果是rigths ：则首个>的下标

                # 只处理那些能影响到我们回测周期的事件
                if start_date_loc < len(trading_dates):
                    name_upper = row['name'].upper()
                    is_risk_event = 'ST' in name_upper or name_upper.startswith('S')
                    # 使用.iloc进行赋值
                    start_trade_date = pd.DatetimeIndex(trading_dates)[start_date_loc]
                    st_matrix.loc[start_trade_date, ts_code] = is_risk_event

        # --- 3. “传播”与“收尾” ---
        st_matrix.ffill(inplace=True)
        st_matrix.fillna(False, inplace=True)

        logger.info("每日‘已知风险’状态矩阵重建完毕。")
        self.st_matrix = st_matrix.astype(bool)
        return self.st_matrix

    # ok 为什么不需要shift1 因为企业上市信息，很很早的信息，不属于后面信息
    def _filter_new_stocks(self, stock_pool_df: pd.DataFrame, months: int = 6) -> pd.DataFrame:
        """
        剔除上市时间小于指定月数的股票。
        """

        if 'list_date' not in self.raw_dfs:
            raise ValueError("缺少上市日期数据(list_date)，跳过新股过滤。")

        list_dates_df = self.raw_dfs['list_date']
        if list_dates_df.empty:
            return stock_pool_df

        # --- 1. 对齐数据 ---
        aligned_universe, aligned_list_dates = stock_pool_df.align(list_dates_df, join='left')

        # --- 2. 【核心修正】强制转换数据类型 ---
        # 在提取 .values 之前，确保整个DataFrame是np.datetime64类型
        # errors='coerce' 会将任何无法转换的值（比如空值或错误字符串）变成 NaT (Not a Time)
        try:
            list_dates_converted = aligned_list_dates.apply(pd.to_datetime, errors='raise')
        except Exception as e:
            raise ValueError(f"上市日期数据无法转换为日期格式，请检查数据源: {e}")
            # return stock_pool_df  # Or handle error appropriately

        # --- 3. 向量化计算 ---
        dates_arr = aligned_universe.index.values[:, np.newaxis]

        # 现在 list_dates_arr 的 dtype 将是 <M8[ns]
        list_dates_arr = list_dates_converted.values

        # 由于 NaT - NaT = NaT, 我们需要处理 NaT。广播计算本身不会报错。
        time_since_listing = dates_arr - list_dates_arr

        # --- 4. 创建并应用掩码 ---
        threshold = pd.Timedelta(days=months * 30.5)
        # NaT < threshold 会是 False, 所以 NaT 值不会被错误地当作新股
        is_new_mask = time_since_listing < threshold

        aligned_universe.values[is_new_mask] = False
        self.show_stock_nums_for_per_day("6个月内上市的过滤！", aligned_universe)
        return aligned_universe

    # ok
    def _filter_st_stocks(self, stock_pool_df: pd.DataFrame) -> pd.DataFrame:
        if self.st_matrix is None:
            raise ValueError("    警告: 未能构建ST状态矩阵，无法过滤ST股票。")
        # 【核心】将“历史真相”矩阵整体向前（未来）移动一天。 (因为st_matrix 是以据生效start_Day日计算的。t下单，只能用t-1的数据跑，t单日的st无法感知！
        # 这确保了我们在T日做决策时，看到的是T-1日的真实状态 。
        st_mask_shifted = self.st_matrix.shift(1, fill_value=False)
        # 对齐两个DataFrame的索引和列，确保万无一失
        # join='left' 表示以stock_pool_df的形状为准
        aligned_universe, aligned_st_status = stock_pool_df.align(st_mask_shifted, join='left',
                                                                  fill_value=False)  # 至少做 行列 保持一致的对齐。 下面才做赋值！ #fill_value=False ：st_Df只能对应一部分的股票池_Df.股票池_Df剩余的行列 用false填充！

        # 将ST的股票从universe中剔除
        # aligned_st_status为True的地方，在universe中就应该为False
        aligned_universe[aligned_st_status] = False

        # 统计过滤效果
        original_count = stock_pool_df.sum(axis=1).mean()
        filtered_count = aligned_universe.sum(axis=1).mean()
        st_filtered_count = original_count - filtered_count
        print(f"      ST股票过滤: 平均每日剔除 {st_filtered_count:.0f} 只ST股票")
        self.show_stock_nums_for_per_day(f'by_ST状态(判定来自于name的变化历史)_filter', aligned_universe)

        return aligned_universe

    # ok
    def _filter_by_liquidity(self, stock_pool_df: pd.DataFrame, min_percentile: float) -> pd.DataFrame:
        """按流动性过滤 """
        if 'turnover_rate' not in self.raw_dfs:
            raise RuntimeError("缺少换手率数据，无法进行流动性过滤")

        turnover_df = self.raw_dfs['turnover_rate']
        turnover_df = turnover_df.shift(1)  # 取用的t日数据，必须前移

        # 1. 【确定样本】只保留 stock_pool_df 中为 True 的换手率数据
        # “只对当前股票池计算”
        valid_turnover = turnover_df.where(stock_pool_df)

        # 2. 【计算标准】沿行（axis=1）一次性计算出每日的分位数阈值
        thresholds = valid_turnover.quantile(min_percentile, axis=1)

        # 3. 【应用标准】将原始换手率与每日阈值进行比较，生成过滤掩码
        low_liquidity_mask = turnover_df.lt(thresholds, axis=0)

        # 4. 将需要剔除的股票在 stock_pool_df 中设为 False
        stock_pool_df[low_liquidity_mask] = False
        self.show_stock_nums_for_per_day(f'by_剔除流动性低的_filter', stock_pool_df)

        return stock_pool_df

    # ok
    def _filter_by_market_cap(self,
                              stock_pool_df: pd.DataFrame,
                              min_percentile: float) -> pd.DataFrame:
        """
        按市值过滤 -

        Args:
            stock_pool_df: 动态股票池
            min_percentile: 市值最低百分位阈值

        Returns:
            过滤后的动态股票池
        """
        if 'total_mv' not in self.raw_dfs:
            raise RuntimeError("缺少市值数据，无法进行市值过滤")

        mv_df = self.raw_dfs['total_mv']
        mv_df = mv_df.shift(1)

        # 1. 【屏蔽】只保留在当前股票池(stock_pool_df)中的股票市值，其余设为NaN
        valid_mv = mv_df.where(stock_pool_df)

        # 2. 【计算标准】向量化计算每日的市值分位数阈值
        # axis=1 确保了我们是按行（每日）计算分位数
        thresholds = valid_mv.quantile(min_percentile, axis=1)

        # 3. 【生成掩码】将原始市值与每日阈值进行比较
        # .lt() 是“小于”操作，axis=0 确保了 thresholds 这个Series能按行正确地广播
        small_cap_mask = mv_df.lt(thresholds, axis=0)

        # 4. 【应用过滤】将所有市值小于当日阈值的股票，在股票池中标记为False
        # 这是一个跨越整个DataFrame的布尔运算，极其高效
        stock_pool_df[small_cap_mask] = False
        self.show_stock_nums_for_per_day(f'by_剔除市值低的_filter', stock_pool_df)

        return stock_pool_df

    # ok
    def _filter_next_day_limit_up(self, stock_pool_df: pd.DataFrame) -> pd.DataFrame:
        """
         剔除在T日开盘即一字涨停的股票。
        这是为了模拟真实交易约束，因为这类股票在开盘时无法买入。
        Args:
            stock_pool_df: 动态股票池DataFrame (T-1日决策，用于T日)
        Returns:
            过滤后的动态股票池DataFrame
        """
        logger.info("    应用次日涨停股票过滤...")

        # --- 1. 数据准备与验证 ---
        required_data = ['open', 'high', 'low', 'pre_close']
        for data_key in required_data:
            if data_key not in self.raw_dfs:
                raise RuntimeError(f"缺少行情数据 '{data_key}'，无法过滤次日涨停股票")

        open_df = self.raw_dfs['open']
        high_df = self.raw_dfs['high']
        low_df = self.raw_dfs['low']
        pre_close_df = self.raw_dfs['pre_close']  # T日的pre_close就是T-1日的close

        # --- 2. 向量化计算每日涨停价 ---
        # a) 创建一个与pre_close_df形状相同的、默认值为1.1的涨跌幅限制矩阵
        limit_rate = pd.DataFrame(1.1, index=pre_close_df.index, columns=pre_close_df.columns)

        # b) 识别科创板(688开头)和创业板(300开头)的股票，将其涨跌幅限制设为1.2
        star_market_stocks = [col for col in limit_rate.columns if str(col).startswith('688')]
        chinext_stocks = [col for col in limit_rate.columns if str(col).startswith('300')]
        limit_rate[star_market_stocks] = 1.2
        limit_rate[chinext_stocks] = 1.2

        # c) 计算理论涨停价 (这里不需要shift，因为pre_close已经是T-1日的信息)
        limit_up_price = (pre_close_df * limit_rate).round(2)

        # --- 3. 生成“开盘即涨停”的布尔掩码 (Mask) ---
        # 条件1: T日的开盘价、最高价、最低价三者相等 (一字板的特征)
        is_one_word_board = (open_df == high_df) & (open_df == low_df)

        # 条件2: T日的开盘价大于或等于理论涨停价
        is_at_limit_price = open_df >= limit_up_price

        # 最终的掩码：两个条件同时满足
        limit_up_mask = is_one_word_board & is_at_limit_price

        # --- 4. 应用过滤 ---
        # 将在T日开盘即涨停的股票，在T日的universe中剔除
        # 这个操作是“未来”的，但它是良性的，因为它模拟的是“无法交易”的现实
        # 它不需要.shift(1)，因为我们是拿T日的状态，来过滤T日的池子
        stock_pool_df[limit_up_mask] = False

        self.show_stock_nums_for_per_day('过滤次日涨停股后--final', stock_pool_df)
        return stock_pool_df

    # ok
    def _filter_next_day_suspended(self, stock_pool_df: pd.DataFrame) -> pd.DataFrame:
        """
          剔除次日停牌股票 -

          Args:
              stock_pool_df: 动态股票池DataFrame

          Returns:
              过滤后的动态股票池DataFrame
          """
        if 'close' not in self.raw_dfs:
            raise RuntimeError(" 缺少价格数据，无法过滤次日停牌股票")

        close_df = self.raw_dfs['close']

        # 1. 创建一个代表“当日有价格”的布尔矩阵
        today_has_price = close_df.notna()

        # 2. 创建一个代表“次日有价格”的布尔矩阵
        #    shift(-1) 将 T+1 日的数据，移动到 T 日的行。这就在一瞬间完成了所有“next_date”的查找
        #    fill_value=True 优雅地处理了最后一天，我们假设最后一天之后不会停牌
        tomorrow_has_price = close_df.notna().shift(-1, fill_value=True)

        # 3. 计算出所有“次日停牌”的掩码 (Mask) （为什么要剔除！质疑自己：明天的事情我为什么要管？ 答：你不怕明天停牌卖不出去？ 还有个原因：ic 计算收益率，会把明天的收益0 一样进行计算！。那怎么得了！）
        #    次日停牌 = 今日有价 & 明日无价
        next_day_suspended_mask = today_has_price & (~tomorrow_has_price)

        # 4. 一次性从股票池中剔除所有被标记的股票
        #    这个布尔运算会自动按索引对齐，应用到整个DataFrame
        stock_pool_df[next_day_suspended_mask] = False

        return stock_pool_df

    def _load_dynamic_index_components(self, index_code: str,
                                       start_date: str, end_date: str) -> pd.DataFrame:
        """加载动态指数成分股数据"""
        # print(f"    加载 {index_code} 动态成分股数据...")

        index_file_name = index_code.replace('.', '_')
        index_data_path = LOCAL_PARQUET_DATA_DIR / 'index_weights' / index_file_name

        if not index_data_path.exists():
            raise ValueError(f"未找到指数 {index_code} 的成分股数据，请先运行downloader下载")

        # 直接读取分区数据，pandas会自动合并所有year=*分区
        components_df = pd.read_parquet(index_data_path)
        components_df['trade_date'] = pd.to_datetime(components_df['trade_date'])

        # 时间范围过滤
        # 大坑啊 ，start_date必须提前6个月！！！ 因为最场6个月才有新的数据！ （新老数据间隔最长可达6个月！）。后面逐日填充成分股信息：原理就是取上次数据进行填充的！
        extended_start_date = pd.Timestamp(start_date) - pd.DateOffset(months=6)
        mask = (components_df['trade_date'] >= extended_start_date) & \
               (components_df['trade_date'] <= pd.Timestamp(end_date))
        components_df = components_df[mask]

        # print(f"    成功加载符合当前回测时间段： {len(components_df)} 条成分股记录")
        return components_df

    def _build_dynamic_index_universe(self, stock_pool_df, index_code: str) -> pd.DataFrame:
        """构建动态指数股票池"""
        start_date = self.config['backtest']['start_date']
        end_date = self.config['backtest']['end_date']

        # 加载动态成分股数据
        components_df = self._load_dynamic_index_components(index_code, start_date, end_date)

        # 获取交易日序列
        trading_dates = self.data_loader.get_trading_dates(start_date, end_date)

        # 🔧 修复：创建新的DataFrame，而不是修改原有的
        index_stock_pool_df = stock_pool_df.copy()

        # 逐日填充成分股信息
        for date in trading_dates:
            if date not in index_stock_pool_df.index:
                continue

            # 获取当日成分股
            daily_components = components_df[
                components_df['trade_date'] == date
                ]['con_code'].tolist()

            if daily_components:
                # 🔧 修复：在基础股票池的基础上，进一步筛选指数成分股
                valid_stocks = index_stock_pool_df.columns.intersection(daily_components)

                # 只保留既在基础股票池中，又是指数成分股的股票
                current_universe = index_stock_pool_df.loc[date].copy()  # 当前基础股票池
                index_stock_pool_df.loc[date, :] = False  # 先清零

                # 同时满足两个条件：1)在基础股票池中 2)是指数成分股
                final_valid_stocks = []
                for stock in valid_stocks:
                    if current_universe[stock]:  # 在基础股票池中
                        final_valid_stocks.append(stock)

                index_stock_pool_df.loc[
                    date, final_valid_stocks] = True  # 以上 强行保证了 一定是有close（即current_universe[stock]为true） 还保证一定是目标成分股

            else:
                # 当日无成分股数据，使用最近一次的成分股
                recent_components = components_df[
                    components_df['trade_date'] <= date
                    ]
                if not recent_components.empty:
                    latest_date = recent_components['trade_date'].max()
                    latest_components = recent_components[
                        recent_components['trade_date'] == latest_date
                        ]['con_code'].tolist()

                    valid_stocks = index_stock_pool_df.columns.intersection(latest_components)
                    current_universe = index_stock_pool_df.loc[date].copy()

                    index_stock_pool_df.loc[date, :] = False
                    final_valid_stocks = [stock for stock in valid_stocks if current_universe[stock]]
                    index_stock_pool_df.loc[date, final_valid_stocks] = True
        self.show_stock_nums_for_per_day(f'by_成分股指数_filter', index_stock_pool_df)

        return index_stock_pool_df

    def get_factor_data(self) -> pd.DataFrame:
        """
        计算目标因子数据

        Returns:
            因子数据DataFrame
        """
        target_factors_for_evaluation = self.config['target_factors_for_evaluation']
        factor_name = target_factors_for_evaluation['name']
        fields = target_factors_for_evaluation['fields']

        print(f"\n计算目标因子: {factor_name}")

        # 使用处理后的数据
        data_source = getattr(self, 'processed_data', self.raw_dfs)

        # 简单的因子计算逻辑
        if factor_name == 'pe_inv' and 'pe_ttm' in fields:
            # PE倒数因子
            pe_data = data_source['pe_ttm']
            factor_data = 1 / pe_data
            factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
        else:
            # 默认使用第一个字段
            factor_data = data_source[fields[0]]

        return factor_data

    def get_universe(self) -> pd.DataFrame:
        """获取股票池"""
        return self.stock_pool_df

    def get_price_data(self) -> pd.DataFrame:
        """获取价格数据"""
        return self.raw_dfs['close']

    def get_namechange_data(self) -> pd.DataFrame:
        """获取name改变的数据"""
        namechange_path = LOCAL_PARQUET_DATA_DIR / 'namechange.parquet'

        return pd.read_parquet(namechange_path)

    def save_data_summary(self, output_dir: str):
        """保存数据摘要"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存股票池统计
        universe_stats = {
            'daily_count': self.stock_pool_df.sum(axis=1),
            'stock_coverage': self.stock_pool_df.sum(axis=0)
        }

        summary_path = os.path.join(output_dir, 'data_summary.xlsx')
        with pd.ExcelWriter(summary_path) as writer:
            # 每日股票数统计
            universe_stats['daily_count'].to_frame('stock_count').to_excel(
                writer, sheet_name='daily_stock_count'
            )

            # 股票覆盖统计
            universe_stats['stock_coverage'].to_frame('coverage_days').to_excel(
                writer, sheet_name='stock_coverage'
            )

            # 数据质量报告
            quality_report = []
            for field_name, df in self.raw_dfs.items():
                quality_report.append({
                    'field': field_name,
                    'shape': f"{df.shape[0]}x{df.shape[1]}",
                    'missing_ratio': f"{df.isnull().sum().sum() / (df.shape[0] * df.shape[1]):.2%}",
                    'valid_ratio': f"{df.notna().sum().sum() / (df.shape[0] * df.shape[1]):.2%}"
                })

            pd.DataFrame(quality_report).to_excel(
                writer, sheet_name='data_quality', index=False
            )

        print(f"数据摘要已保存到: {summary_path}")

    def show_stock_nums_for_per_day(self, describe_text, index_stock_pool_df):
        daily_count = index_stock_pool_df.sum(axis=1)
        logger.info(f"    {describe_text}动态股票池:")
        logger.info(f"      平均每日股票数: {daily_count.mean():.0f}")
        logger.info(f"      最少每日股票数: {daily_count.min():.0f}")
        logger.info(f"      最多每日股票数: {daily_count.max():.0f}")

    # 输入学术因子，返回计算所必须的base 因子
    def get_cal_base_factors(self, target_factors: list[str]) -> set:
        factor_df = pd.DataFrame(self.config['factor_definition'])  # 将 list[dict] 转为 DataFrame
        result = set()

        for target_factors_for_evaluation in target_factors:
            matched = factor_df[factor_df['name'] == target_factors_for_evaluation]
            if not matched.empty:
                base_fields = matched.iloc[0]['cal_require_base_fields']
                result.update(base_fields)  # 用 update 合并列表到 set

        return result
    #ok
    def product_stock_pool(self, stock_pool_config_profile, pool_name):
        """
                构建动态股票池
                Returns:
                    股票池DataFrame，True表示该股票在该日期可用
                """
        logger.info(f"  构建{pool_name}动态股票池...")
        # 第一步：基础股票池 - 有价格数据的股票
        if 'close' not in self.raw_dfs:
            raise ValueError("缺少价格数据，无法构建股票池")

        final_stock_pool_df = self.raw_dfs['close'].notna()
        self.show_stock_nums_for_per_day('根据收盘价notna生成的', final_stock_pool_df)
        # 第二步：各种过滤！
        # --基础过滤 指数成分股过滤（如果启用）
        index_config = stock_pool_config_profile[pool_name].get('index_filter', {})
        if index_config.get('enable', False):
            # print(f"    应用指数过滤: {index_config['index_code']}")
            final_stock_pool_df = self._build_dynamic_index_universe(final_stock_pool_df, index_config['index_code'])
            # ✅ 在这里进行列修剪是合理的！ 因为中证800成分股是基于外部规则，不是基于未来数据表现
            valid_stocks = final_stock_pool_df.columns[final_stock_pool_df.any(axis=0)]
            final_stock_pool_df = final_stock_pool_df[valid_stocks]
        # --普适性 过滤 （通用过滤）
        final_stock_pool_df = self._filter_new_stocks(final_stock_pool_df, 6)  # 新股票数据少，不具参考
        final_stock_pool_df = self._filter_st_stocks(final_stock_pool_df)  # 剔除ST股票

        # 其他各种指标过滤条件
        universe_filters = stock_pool_config_profile[pool_name]['filters']

        # 2. 流动性过滤
        if 'min_liquidity_percentile' in universe_filters:
            print("    应用流动性过滤...")
            final_stock_pool_df = self._filter_by_liquidity(
                final_stock_pool_df,
                universe_filters['min_liquidity_percentile']
            )

        # 3. 市值过滤
        if 'min_market_cap_percentile' in universe_filters:
            # print("    应用市值过滤...")
            final_stock_pool_df = self._filter_by_market_cap(
                final_stock_pool_df,
                universe_filters['min_market_cap_percentile']
            )
        # 剔除次日停牌股票
        if universe_filters['remove_next_day_suspended']:
            final_stock_pool_df = self._filter_next_day_suspended(final_stock_pool_df)
        # 剔除涨停股票
        if universe_filters['remove_next_day_limit_up']:
            final_stock_pool_df = self._filter_next_day_limit_up(final_stock_pool_df)
        return final_stock_pool_df

    def get_school_code_by_factor_name(self, factor_name):
        factor_dict = {item['name']: item for item in self.config['factor_definition']}
        return factor_dict[factor_name]['school']

    def get_stock_pool_by_factor_name(self, factor_name):
        school_code = self.get_school_code_by_factor_name(factor_name)
        if school_code in ['fundamentals', 'trend']:
            return self.stock_pools_dict['institutional_stock_pool']
        if school_code in ['microstructure']:
            return self.stock_pools_dict['microstructure_stock_pool']
        raise ValueError(f'没有定义因子:{factor_name}属于哪一门派')

    def get_stock_pool_name_by_factor_school(self, factor_school):
        if factor_school in ['fundamentals', 'trend']:
            return 'institutional_stock_pool'
        if factor_school in ['microstructure']:
            return 'microstructure_stock_pool'
        raise ValueError('没有定义因子属于哪一门派')

    def __align_one_raw_dfs_by_stock_pool_and_fill(self, factor_name, raw_df_param,
                                                   stock_pool_param: pd.DataFrame = None):
        # 定义不同类型数据的填充策略
        HIGH_FREQ_FIELDS = ['turnover', 'volume', 'returns', 'turnover_rate','pct_chg']  #
        SLOW_MOVING_FIELDS = ['pe_ttm', 'pb', 'total_mv', 'circ_mv']  # 缓变数据，限制前向填充
        STATIC_FIELDS = ['industry', 'list_date']  # 静态数据，无限前向填充
        PRICE_FIELDS = ['close', 'open', 'high', 'low', 'pre_close']  # 价格数据，特殊处理
        raw_df = raw_df_param.copy(deep=True)
        if stock_pool_param is not None:
            stock_pool_df = stock_pool_param
        else:
            stock_pool_df = self.get_stock_pool_by_factor_name(factor_name)

        # 步骤1: 对齐到修剪后的股票池 对齐到主模板（stock_pool_df的形状）
        aligned_df = raw_df.reindex(index=stock_pool_df.index, columns=stock_pool_df.columns)
        aligned_df = aligned_df.sort_index()
        aligned_df = aligned_df.where(stock_pool_df)

        # 步骤2: 根据数据类型应用不同的填充策略
        if factor_name in HIGH_FREQ_FIELDS:
            # 高频数据 暂时不ffill，因为在停牌日，交易相关的活动活动（（成交量、换手率 确实是空的），你去ffill之气的那不就大错了；至于fill（0）还是保持nan，让下游自己考虑，这里不提前一棍子打死
            # aligned_df = aligned_df.where(stock_pool_df).fillna(0)
            aligned_df = aligned_df

        elif factor_name in SLOW_MOVING_FIELDS:
            # 缓变数据：先限制前向填充，再应用股票池过滤
            aligned_df = aligned_df.ffill(limit=2)  # 最多前向填充2天

        elif factor_name in STATIC_FIELDS:
            # 静态数据：无限前向填充，再应用股票池过滤
            aligned_df = aligned_df.ffill()  # 任由他填充又何妨，反正我前期做了自动宽化填充

        elif factor_name in PRICE_FIELDS:
            # 价格数据：只保留股票池内的数据  单因子测试需要计算收益率，价格数据不能中断 值得深入思考。
            #赞成fill理由：。根据标准的基金会计准则，在停牌期间，一只股票的价值并没有消失或变成未知。为了计算每日的投资组合净值，它的价值必须被定义为**“最后一个可获得的公允价值”**，也就是它停牌前的最后一个价格/市值。
            # 我还是觉得 污染了正确性！ 后期有空再解决 todo
            # 停牌股票仍需定价来计算组合净值和收益
            aligned_df = aligned_df.ffill()

        else:
            raise RuntimeError(f"此因子{factor_name}没有指明频率，无法进行填充")
        return aligned_df

def create_data_manager(config_path: str) -> DataManager:
    """
    创建数据管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        DataManager实例
    """
    return DataManager(config_path)
