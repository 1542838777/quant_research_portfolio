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

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.data_loader import DataLoader, logger
from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR

warnings.filterwarnings('ignore')





class DataManager:
    """
    数据管理器 - 负责数据加载和股票池构建
    
    按照配置文件的要求，实现：
    1. 原始数据加载
    2. 动态股票池构建
    3. 数据质量检查
    4. 数据对齐和预处理
    """

    def __init__(self, config_path: str):
        """
        初始化数据管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.st_matrix = None
        self.config = self._load_config(config_path)
        self.data_loader = DataLoader(data_path=LOCAL_PARQUET_DATA_DIR)
        self.raw_data = {}
        self.universe_df = None

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
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
        print(f"需要加载的所有字段: {all_required_fields}")

        # === 一次性加载所有数据 ===
        print("\n" + "=" * 50)
        print("一次性加载所有要求的字段数据")
        print("=" * 50)

        self.raw_data = self.data_loader.get_raw_dfs_by_require_fields(fields=all_required_fields,
                                                                       start_date=start_date, end_date=end_date)
        print(f"数据加载完成，共加载 {len(self.raw_data)} 个字段")

        # === 第一阶段：基于已加载数据构建权威股票池 ===
        print("\n" + "=" * 50)
        print("第一阶段：构建权威股票池（各种过滤！）")
        print("=" * 50)

        self.universe_df = self._build_universe_from_loaded_data(start_date, end_date)

        # === 第二阶段：基于股票池对齐和清洗所有数据 ===
        print("\n" + "=" * 50)
        print("第二阶段：对齐和填充所有因子数据")
        print("=" * 50)

        # 使用权威股票池对齐和清洗数据
        self.processed_data = self._align_and_clean_all_data(self.raw_data, self.universe_df)

        return self.processed_data

    def _build_universe_from_loaded_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        第一阶段：基于已加载的数据构建权威股票池

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            权威股票池DataFrame
        """
        print("1. 验证股票池构建所需数据...")

        # 验证必需字段是否已加载
        required_fields_for_universe = ['close', 'total_mv', 'turnover_rate', 'industry', 'list_date']
        missing_fields = [field for field in required_fields_for_universe if field not in self.raw_data]

        if missing_fields:
            raise ValueError(f"构建股票池缺少必需字段: {missing_fields}")

        print(f"   ✅ 股票池构建所需字段已全部加载")

        print("\n2. 构建ST股票矩阵...")
        # 获取所有股票和交易日期
        ts_codes = list(set(self.get_price_data().columns))
        namechange = self.get_namechange_data()
        trading_dates = self.data_loader.get_trading_dates(start_date=start_date, end_date=end_date)

        # 构建ST矩阵
        self.build_st_period_from_namechange(ts_codes, namechange, trading_dates)

        print("\n3. 执行股票池构建...")
        universe_df = self._build_universe()

        print(f"\n权威股票池构建完成！")
        print(f"   平均每日股票数: {universe_df.sum(axis=1).mean():.0f}")
        print(f"   股票池形状: {universe_df.shape}")

        return universe_df

    def _align_and_clean_all_data(self, raw_dfs: Dict[str, pd.DataFrame],
                                  universe_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        第二阶段：使用权威股票池对齐和清洗所有数据

        Args:
            raw_dfs: 原始数据字典
            universe_df: 权威股票池DataFrame

        Returns:
            对齐和清洗后的数据字典
        """
        print("1. 定义数据分类和填充策略...")

        # 定义不同类型数据的填充策略
        HIGH_FREQ_FIELDS = ['turnover', 'volume', 'returns', 'turnover_rate']  # 高频数据，用0填充
        SLOW_MOVING_FIELDS = ['pe_ttm', 'pb', 'total_mv', 'circ_mv']  # 缓变数据，限制前向填充
        STATIC_FIELDS = ['industry', 'list_date']  # 静态数据，无限前向填充
        PRICE_FIELDS = ['close', 'open', 'high', 'low']  # 价格数据，特殊处理

        print(f"\n2. 开始对齐和清洗 {len(raw_dfs)} 个数据字段...")

        aligned_data = {}
        for name, df in raw_dfs.items():
            print(f"   处理字段: {name}")

            # 步骤1: 对齐到主模板（universe_df的形状）
            aligned_df = df.reindex(index=universe_df.index, columns=universe_df.columns)
            aligned_df = aligned_df.sort_index()

            # 步骤2: 根据数据类型应用不同的填充策略
            if name in HIGH_FREQ_FIELDS:
                # 高频数据：只在股票池内填充0，股票池外保持NaN
                aligned_df = aligned_df.where(universe_df).fillna(0)
                print(f"     -> 高频数据，股票池内用0填充")

            elif name in SLOW_MOVING_FIELDS:
                # 缓变数据：先限制前向填充，再应用股票池过滤
                aligned_df = aligned_df.ffill(limit=2)  # 最多前向填充2天
                aligned_df = aligned_df.where(universe_df)
                print(f"     -> 缓变数据，限制前向填充2天")

            elif name in STATIC_FIELDS:
                # 静态数据：无限前向填充，再应用股票池过滤
                aligned_df = aligned_df.ffill()
                aligned_df = aligned_df.where(universe_df)
                print(f"     -> 静态数据，无限前向填充")

            elif name in PRICE_FIELDS:
                # 价格数据：只保留股票池内的数据，不填充
                aligned_df = aligned_df.where(universe_df)
                print(f"     -> 价格数据，仅保留股票池内数据")

            else:
                raise RuntimeError(f"此因子{name}没有指明频率，无法进行填充")

            aligned_data[name] = aligned_df

        print(f"\n数据对齐和清洗完成！")
        print(f"   处理后数据形状统一为: {universe_df.shape}")

        return aligned_data

    def _get_required_fields(self) -> List[str]:
        """获取所有需要的字段"""
        required_fields = set()

        # 基础字段
        required_fields.update(['close',
                                'pb',  # 为了计算价值类因子
                                'total_mv', 'turnover_rate',  # 为了过滤 很差劲的股票 仅此而已，不会作其他计算 、'total_mv'还可 用于计算中性化
                                'industry',  # 用于计算中性化
                                'circ_mv',  # 流通市值 用于WOS，加权最小二方跟  ，回归法会用到
                                'list_date'  # 上市日期
                                ])

        # 目标因子字段
        target_factor = self.config['target_factor']
        required_fields.update(target_factor['fields'])

        # 中性化需要的字段
        neutralization = self.config['preprocessing']['neutralization']
        if neutralization['enable']:
            if 'industry' in neutralization['factors']:
                required_fields.add('industry')
            if 'market_cap' in neutralization['factors']:
                required_fields.add('total_mv')

        # # 股票池过滤需要的字段
        # universe_filters = self.config['universe']['filters']
        # if universe_filters.get('remove_st', False):
        #     print()
        #     # required_fields.add('name')  # 用于识别ST股票 改成 使用时 当场添加，现在过早加入，前期跟着别的字段一起经历那么多 没必要

        return list(required_fields)

    def _check_data_quality(self):
        """检查数据质量"""
        print("  检查数据完整性和质量...")

        for field_name, df in self.raw_data.items():
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
        if 'close' not in self.raw_data:
            raise ValueError("缺少价格数据，无法构建股票池")

        close_df = self.raw_data['close']
        universe_df = close_df.notna()
        self.show_stock_nums_for_per_day('根据收盘价notna生成的最原始的股票池', universe_df)
        # 第二步：指数成分股过滤（如果启用）
        index_config = self.config['universe'].get('index_filter', {})
        if index_config.get('enable', False):
            print(f"    应用指数过滤: {index_config['index_code']}")
            universe_df = self._build_dynamic_index_universe(universe_df, index_config['index_code'])

        # 应用各种过滤条件
        universe_filters = self.config['universe']['filters']

        # 1. 剔除ST股票
        if universe_filters.get('remove_st', False):
            print("    应用ST股票过滤...")
            universe_df = self._filter_st_stocks(universe_df)

        # 2. 流动性过滤
        if 'min_liquidity_percentile' in universe_filters:
            print("    应用流动性过滤...")
            universe_df = self._filter_by_liquidity(
                universe_df,
                universe_filters['min_liquidity_percentile']
            )

        # 3. 市值过滤
        if 'min_market_cap_percentile' in universe_filters:
            print("    应用市值过滤...")
            universe_df = self._filter_by_market_cap(
                universe_df,
                universe_filters['min_market_cap_percentile']
            )

        # 4. 剔除次日停牌股票
        if universe_filters.get('remove_next_day_suspended', False):
            print("    应用次日停牌股票过滤...")
            universe_df = self._filter_next_day_suspended(universe_df)

        # 统计股票池信息
        daily_count = universe_df.sum(axis=1)
        print(f"    过滤后（市值、换手率...)股票池统计:")
        print(f"      平均每日股票数: {daily_count.mean():.0f}")
        print(f"      最少每日股票数: {daily_count.min():.0f}")
        print(f"      最多每日股票数: {daily_count.max():.0f}")

        return universe_df

    def build_st_period_from_namechange(
            self,
            ts_codes: list,
            namechange_df: pd.DataFrame,
            trading_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
          【专业重构版】根据namechange历史数据，重建每日ST状态的布尔矩阵。
          此版本能正确处理数据不完整和初始状态问题。
          """
        print("正在根据名称变更历史，重建每日风险警示状态矩阵...")

        # 1. 创建一个“未知状态”的画布，用 np.nan 初始化
        st_matrix = pd.DataFrame(np.nan, index=trading_dates, columns=ts_codes)

        namechange_df['start_date'] = pd.to_datetime(namechange_df['start_date'])
        # 关键：我们只需要按股票分组，并在组内按时间排序
        namechange_df.sort_values(by=['ts_code', 'start_date'], inplace=True)

        # 2. 【核心】只在状态【发生改变】的当天打点标记
        # 我们使用 groupby().apply() 来避免外层循环，更高效
        def mark_events(group):
            for _, row in group.iterrows():
                date = row['start_date']
                if date in st_matrix.index:
                    name_upper = row['name'].upper()
                    is_risk_stock = 'ST' in name_upper or name_upper.startswith('S')
                    st_matrix.loc[date, row['ts_code']] = is_risk_stock
            return None  # apply不需要返回值

        namechange_df.groupby('ts_code').apply(mark_events)

        # 3. 【魔法】使用ffill()，用每个时点的已知状态，填充后续所有的“未知状态”
        # 这是整个逻辑的核心，它正确地假设了“状态会一直持续，直到下一次变更”
        st_matrix.ffill(inplace=True)

        # 4. 【收尾】将所有剩余的“未知状态”填充为False
        # 这包括：从未有过名称变更的股票，以及在第一次名称变更前的所有日期
        st_matrix.fillna(False, inplace=True)

        print("每日风险警示状态矩阵重建完毕。")
        self.st_matrix = st_matrix

    def _filter_st_stocks(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        if self.st_matrix is None:
            print("    警告: 未能构建ST状态矩阵，无法过滤ST股票。")
            return universe_df

        # 对齐两个DataFrame的索引和列，确保万无一失
        # join='left' 表示以universe_df的形状为准
        aligned_universe, aligned_st_status = universe_df.align(self.st_matrix, join='left',
                                                                fill_value=False)  # 至少做 行列 保持一致的对齐。 下面才做赋值！ #fill_value=False ：st_Df只能对应一部分的股票池_Df.股票池_Df剩余的行列 用false填充！

        # 将ST的股票从universe中剔除
        # aligned_st_status为True的地方，在universe中就应该为False
        aligned_universe[aligned_st_status] = False

        # 统计过滤效果
        original_count = universe_df.sum(axis=1).mean()
        filtered_count = aligned_universe.sum(axis=1).mean()
        st_filtered_count = original_count - filtered_count
        print(f"      ST股票过滤: 平均每日剔除 {st_filtered_count:.0f} 只ST股票")
        self.show_stock_nums_for_per_day(f'by_{'ST状态(判定来自于name的变化历史)'}_filter', aligned_universe)

        return aligned_universe

    def _filter_by_liquidity(self, universe_df: pd.DataFrame, min_percentile: float) -> pd.DataFrame:
        """按流动性过滤 """
        if 'turnover_rate' not in self.raw_data:
            raise RuntimeError("缺少换手率数据，无法进行流动性过滤")

        turnover_df = self.raw_data['turnover_rate']

        # 1. 【确定样本】只保留 universe_df 中为 True 的换手率数据
        # “只对当前股票池计算”
        valid_turnover = turnover_df.where(universe_df)

        # 2. 【计算标准】沿行（axis=1）一次性计算出每日的分位数阈值
        thresholds = valid_turnover.quantile(min_percentile, axis=1)

        # 3. 【应用标准】将原始换手率与每日阈值进行比较，生成过滤掩码
        low_liquidity_mask = turnover_df.lt(thresholds, axis=0)

        # 4. 将需要剔除的股票在 universe_df 中设为 False
        universe_df[low_liquidity_mask] = False
        self.show_stock_nums_for_per_day(f'by_{'剔除流动性低的'}_filter', universe_df)

        return universe_df

    def _filter_by_market_cap(self,
                              universe_df: pd.DataFrame,
                              min_percentile: float) -> pd.DataFrame:
        """
        按市值过滤 -

        Args:
            universe_df: 动态股票池
            min_percentile: 市值最低百分位阈值

        Returns:
            过滤后的动态股票池
        """
        if 'total_mv' not in self.raw_data:
            raise RuntimeError("缺少市值数据，无法进行市值过滤")

        mv_df = self.raw_data['total_mv']

        # 1. 【屏蔽】只保留在当前股票池(universe_df)中的股票市值，其余设为NaN
        valid_mv = mv_df.where(universe_df)

        # 2. 【计算标准】向量化计算每日的市值分位数阈值
        # axis=1 确保了我们是按行（每日）计算分位数
        thresholds = valid_mv.quantile(min_percentile, axis=1)

        # 3. 【生成掩码】将原始市值与每日阈值进行比较
        # .lt() 是“小于”操作，axis=0 确保了 thresholds 这个Series能按行正确地广播
        small_cap_mask = mv_df.lt(thresholds, axis=0)

        # 4. 【应用过滤】将所有市值小于当日阈值的股票，在股票池中标记为False
        # 这是一个跨越整个DataFrame的布尔运算，极其高效
        universe_df[small_cap_mask] = False
        self.show_stock_nums_for_per_day(f'by_{'剔除市值低的'}_filter', universe_df)

        return universe_df

    def _filter_next_day_suspended(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        """
          剔除次日停牌股票 -

          Args:
              universe_df: 动态股票池DataFrame

          Returns:
              过滤后的动态股票池DataFrame
          """
        if 'close' not in self.raw_data:
            raise RuntimeError(" 缺少价格数据，无法过滤次日停牌股票")

        close_df = self.raw_data['close']

        # 1. 创建一个代表“当日有价格”的布尔矩阵
        today_has_price = close_df.notna()

        # 2. 创建一个代表“次日有价格”的布尔矩阵
        #    shift(-1) 将 T+1 日的数据，移动到 T 日的行。这就在一瞬间完成了所有“next_date”的查找
        #    fill_value=True 优雅地处理了最后一天，我们假设最后一天之后不会停牌
        tomorrow_has_price = close_df.notna().shift(-1, fill_value=True)

        # 3. 计算出所有“次日停牌”的掩码 (Mask)
        #    次日停牌 = 今日有价 & 明日无价
        next_day_suspended_mask = today_has_price & (~tomorrow_has_price)

        # 4. 一次性从股票池中剔除所有被标记的股票
        #    这个布尔运算会自动按索引对齐，应用到整个DataFrame
        universe_df[next_day_suspended_mask] = False

        return universe_df

    def _load_dynamic_index_components(self, index_code: str,
                                       start_date: str, end_date: str) -> pd.DataFrame:
        """加载动态指数成分股数据"""
        print(f"    加载 {index_code} 动态成分股数据...")

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

        print(f"    成功加载符合当前回测时间段： {len(components_df)} 条成分股记录")
        return components_df

    def _build_dynamic_index_universe(self, universe_df, index_code: str) -> pd.DataFrame:
        """构建动态指数股票池"""
        start_date = self.config['backtest']['start_date']
        end_date = self.config['backtest']['end_date']

        # 加载动态成分股数据
        components_df = self._load_dynamic_index_components(index_code, start_date, end_date)

        # 获取交易日序列
        trading_dates = self.data_loader.get_trading_dates(start_date, end_date)

        # 🔧 修复：创建新的DataFrame，而不是修改原有的
        index_universe_df = universe_df.copy()

        # 逐日填充成分股信息
        for date in trading_dates:
            if date not in index_universe_df.index:
                continue

            # 获取当日成分股
            daily_components = components_df[
                components_df['trade_date'] == date
                ]['con_code'].tolist()

            if daily_components:
                # 🔧 修复：在基础股票池的基础上，进一步筛选指数成分股
                valid_stocks = index_universe_df.columns.intersection(daily_components)

                # 只保留既在基础股票池中，又是指数成分股的股票
                current_universe = index_universe_df.loc[date].copy()  # 当前基础股票池
                index_universe_df.loc[date, :] = False  # 先清零

                # 同时满足两个条件：1)在基础股票池中 2)是指数成分股
                final_valid_stocks = []
                for stock in valid_stocks:
                    if current_universe[stock]:  # 在基础股票池中
                        final_valid_stocks.append(stock)

                index_universe_df.loc[
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

                    valid_stocks = index_universe_df.columns.intersection(latest_components)
                    current_universe = index_universe_df.loc[date].copy()

                    index_universe_df.loc[date, :] = False
                    final_valid_stocks = [stock for stock in valid_stocks if current_universe[stock]]
                    index_universe_df.loc[date, final_valid_stocks] = True
        self.show_stock_nums_for_per_day(f'by_{'成分股指数'}_filter', index_universe_df)

        return index_universe_df

    def get_factor_data(self) -> pd.DataFrame:
        """
        计算目标因子数据

        Returns:
            因子数据DataFrame
        """
        target_factor = self.config['target_factor']
        factor_name = target_factor['name']
        fields = target_factor['fields']

        print(f"\n计算目标因子: {factor_name}")

        # 使用处理后的数据
        data_source = getattr(self, 'processed_data', self.raw_data)

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
        return self.universe_df

    def get_price_data(self) -> pd.DataFrame:
        """获取价格数据"""
        # 使用处理后的数据，如果不存在则使用原始数据
        data_source = getattr(self, 'processed_data', self.raw_data)
        return data_source['close']

    def get_namechange_data(self) -> pd.DataFrame:
        """获取name改变的数据"""
        namechange_path = LOCAL_PARQUET_DATA_DIR / 'namechange.parquet'

        return pd.read_parquet(namechange_path)

    def save_data_summary(self, output_dir: str):
        """保存数据摘要"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存股票池统计
        universe_stats = {
            'daily_count': self.universe_df.sum(axis=1),
            'stock_coverage': self.universe_df.sum(axis=0)
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
            for field_name, df in self.raw_data.items():
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

    def show_stock_nums_for_per_day(self, describe_text, index_universe_df):
        daily_count = index_universe_df.sum(axis=1)
        print(f"    {describe_text}动态股票池构建完成:")
        print(f"      平均每日股票数: {daily_count.mean():.0f}")
        print(f"      最少每日股票数: {daily_count.min():.0f}")
        print(f"      最多每日股票数: {daily_count.max():.0f}")

def create_data_manager(config_path: str) -> DataManager:
    """
    创建数据管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        DataManager实例
    """
    return DataManager(config_path)
