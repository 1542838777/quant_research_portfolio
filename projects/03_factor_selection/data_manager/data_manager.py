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
        根据配置文件加载所有需要的数据
        
        Returns:
            数据字典
        """
        print("=" * 80)
        print("第二阶段：数据加载与股票池构建")
        print("=" * 80)

        # 1. 确定需要加载的字段
        required_fields = self._get_required_fields()
        print(f"\n1. 确定需要加载的字段: {required_fields}")

        # 2. 加载原始数据
        print("\n2. 加载原始数据...")
        start_date = self.config['backtest']['start_date']
        end_date = self.config['backtest']['end_date']

        self.raw_data = self.data_loader.load_data(
            fields=required_fields,
            start_date=start_date,
            end_date=end_date
        )

        print(f"数据加载完成，共加载 {len(self.raw_data)} 个字段")

        # 3. 数据质量检查
        print("\n3. 数据质量检查...")
        self._check_data_quality()

        # 4. 构建动态股票池
        print("\n4. 构建动态股票池...")
        # 获取所有股票
        ts_codes = list(set(self.get_price_data().columns))
        namechange = self.get_namechange_data()
        trading_dates = self.data_loader.get_trading_dates(start_date=start_date, end_date=end_date)

        self.build_st_period_from_namechange(ts_codes, namechange, trading_dates)
        self.universe_df = self._build_universe()

        # 5. 应用股票池过滤
        print("\n5. 应用股票池过滤...")
        self._apply_universe_filter()

        return self.raw_data

    def _get_required_fields(self) -> List[str]:
        """获取所有需要的字段"""
        required_fields = set()

        # 基础字段
        required_fields.update(['close',
                                'pb',#为了计算价值类因子
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

        # 股票池过滤需要的字段
        universe_filters = self.config['universe']['filters']
        if universe_filters.get('remove_st', False):
            print()
            # required_fields.add('name')  # 用于识别ST股票 改成 使用时 当场添加，现在过早加入，前期跟着别的字段一起经历那么多 没必要

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
            if field_name in ['close', 'total_mv','pb', 'pe_ttm']:
                negative_ratio = (df <= 0).sum().sum() / df.notna().sum().sum()
                print(f"  极值(>99%分位) 占比: {((df > df.quantile(0.99)).sum().sum())/(df.shape[0] * df.shape[1])}")

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
        print(f"    基础（未过滤）股票池：{universe_df.sum(axis=1).mean():.0f} 只股票/日")
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
        【专业版】根据namechange历史数据，重建每日ST状态的布尔矩阵。
        此版本已增强，可同时识别ST, *ST, S, SST等多种风险警示状态。

        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            ts_codes: 所有股票代码的列表
            namechange_df: 从Tushare下载的namechange历史数据
            trade_cal: 交易日历DataFrame

        Returns:
            pd.DataFrame: 一个布尔矩阵，True代表当天该股票处于风险警示状态。
        """
        print("正在根据名称变更历史，重建每日风险警示状态矩阵...")
        # 获取真实的交易日

        # 创建一个空的“画布”
        st_matrix = pd.DataFrame(False, index=trading_dates, columns=ts_codes)

        # 格式化日期并排序，确保事件按时间发生
        namechange_df['start_date'] = pd.to_datetime(namechange_df['start_date'])
        namechange_df.sort_values(by='start_date', inplace=True)

        # 按股票代码分组，处理每只股票的ST历史
        for stock, group in namechange_df.groupby('ts_code'):
            if stock not in st_matrix.columns:
                continue
            group = group.sort_values(by='start_date')  # 保证组内有序

            for _, row in group.iterrows():
                # --- 【核心逻辑修正】 ---
                # 一个更健壮的检查，同时处理ST, *ST, S, SST等情况
                name_upper = row['name'].upper()
                is_risk_stock = 'ST' in name_upper or name_upper.startswith('S')

                start = row['start_date']
                # 结束日期是下一次名称变更的开始日期，或者是无穷远
                end = row['end_date'] if pd.notna(row['end_date']) else pd.to_datetime('2200-01-01')

                # 根据is_risk_stock的值，来标记整个区间的状态 为true表示st股票
                st_matrix.loc[start:end, stock] = is_risk_stock

        # 【重要】向前填充，因为namechange只记录变更时点，我们需要填充期间的状态
        st_matrix.ffill(inplace=True)

        print("每日风险警示状态矩阵重建完毕。")
        self.st_matrix = st_matrix

    def _filter_st_stocks(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        print("    应用ST股票过滤...")
        if self.st_matrix is None:
            print("    警告: 未能构建ST状态矩阵，无法过滤ST股票。")
            return universe_df

        # 对齐两个DataFrame的索引和列，确保万无一失
        # join='left' 表示以universe_df的形状为准
        aligned_universe, aligned_st_status = universe_df.align(self.st_matrix, join='left', fill_value=False)#至少做 行列 保持一致的对齐。 下面才做赋值！ #fill_value=False ：st_Df只能对应一部分的股票池_Df.股票池_Df剩余的行列 用false填充！

        # 将ST的股票从universe中剔除
        # aligned_st_status为True的地方，在universe中就应该为False
        aligned_universe[aligned_st_status] = False

        # 统计过滤效果
        original_count = universe_df.sum(axis=1).mean()
        filtered_count = aligned_universe.sum(axis=1).mean()
        st_filtered_count = original_count - filtered_count
        print(f"      ST股票过滤: 平均每日剔除 {st_filtered_count:.0f} 只ST股票")

        return aligned_universe

    def _filter_by_liquidity(self, universe_df: pd.DataFrame,
                             min_percentile: float) -> pd.DataFrame:
        """按流动性过滤"""
        if 'turnover_rate' not in self.raw_data:
            print("    警告: 缺少换手率数据，无法进行流动性过滤")
            return universe_df

        turnover_df = self.raw_data['turnover_rate']

        for date in universe_df.index:
            if date in turnover_df.index:
                turnover_values = turnover_df.loc[date]
                valid_turnover = turnover_values[universe_df.loc[date]]

                if len(valid_turnover) > 10:
                    threshold = valid_turnover.quantile(min_percentile)  # 从小到大val 排在20%位置的阈值
                    low_liquidity_mask = turnover_values < threshold  # 比阈值还小！ 果然不要！
                    universe_df.loc[date, low_liquidity_mask] = False

        return universe_df

    def _filter_by_market_cap(self, universe_df: pd.DataFrame,
                              min_percentile: float) -> pd.DataFrame:
        """按市值过滤"""
        if 'total_mv' not in self.raw_data:
            print("    警告: 缺少市值数据，无法进行市值过滤")
            return universe_df

        mv_df = self.raw_data['total_mv']

        for date in universe_df.index:
            if date in mv_df.index:
                mv_values = mv_df.loc[date]
                valid_mv = mv_values[universe_df.loc[date]]

                if len(valid_mv) > 10:
                    threshold = valid_mv.quantile(min_percentile)
                    small_cap_mask = mv_values < threshold
                    universe_df.loc[date, small_cap_mask] = False

        return universe_df

    def _filter_next_day_suspended(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        """剔除次日停牌股票"""
        if 'close' not in self.raw_data:
            print("    警告: 缺少价格数据，无法过滤次日停牌股票")
            return universe_df

        close_df = self.raw_data['close']

        # 获取所有交易日
        trading_dates = universe_df.index.tolist()

        for i, date in enumerate(trading_dates[:-1]):  # 排除最后一天
            next_date = trading_dates[i + 1]

            # 今日有价格但明日无价格的股票，认为明日停牌
            today_has_price = close_df.loc[date].notna()
            tomorrow_has_price = close_df.loc[next_date].notna()

            # 明日停牌的股票：今日有价格但明日无价格
            next_day_suspended = today_has_price & (~tomorrow_has_price)

            # 从今日股票池中剔除明日停牌的股票
            universe_df.loc[date, next_day_suspended] = False

        return universe_df

    def _filter_by_index_components(self, universe_df: pd.DataFrame,
                                    index_code: str) -> pd.DataFrame:
        """根据指数成分股过滤"""
        try:
            # 加载指数成分股数据
            index_components = self._load_index_components(index_code)

            # 只保留成分股
            valid_stocks = universe_df.columns.intersection(index_components)
            filtered_universe = universe_df[valid_stocks].copy()

            print(f"    指数 {index_code} 成分股过滤完成，保留 {len(valid_stocks)} 只股票")
            return filtered_universe

        except Exception as e:
            print(f"    指数成分股过滤失败: {e}，使用原始股票池")
            return universe_df

    ## return ts_code list
    def _load_index_components(self, index_code: str) -> list:
        """加载指数成分股列表"""
        # 方案1：从本地文件加载
        components_file = LOCAL_PARQUET_DATA_DIR / 'index_weights' / f"{index_code.replace('.', '_')}"
        if components_file.exists():
            df = pd.read_parquet(components_file)
            return df['con_code'].unique().tolist()

        raise ValueError(f"未找到指数 {index_code} 的成分股数据")

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
        #大坑啊 ，start_date必须提前6个月！！！ 因为最场6个月才有新的数据！ （新老数据间隔最长可达6个月！）。后面逐日填充成分股信息：原理就是取上次数据进行填充的！
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

        # 获取所有可能的股票代码
        all_stocks = self.raw_data['close'].columns

        # 逐日填充成分股信息
        for date in trading_dates:
            # 获取当日成分股
            daily_components = components_df[
                components_df['trade_date'] == date
                ]['con_code'].tolist()

            if daily_components:
                # 当日有成分股数据
                valid_stocks = universe_df.columns.intersection(daily_components)
                # 正确做法：先清零，再设置
                universe_df.loc[date, :] = False  # 先把当日所有股票设为False
                universe_df.loc[date, valid_stocks] = True  # 再把成分股设为True
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

                    valid_stocks = universe_df.columns.intersection(latest_components)
                    universe_df.loc[date, :] = False  # 先把当日所有股票设为False
                    universe_df.loc[date, valid_stocks] = True  # 再把成分股设为True

        daily_count = universe_df.sum(axis=1)
        print(f"    动态指数股票池构建完成:")
        print(f"      平均每日股票数: {daily_count.mean():.0f}")
        print(f"      最少每日股票数: {daily_count.min():.0f}")
        print(f"      最多每日股票数: {daily_count.max():.0f}")

        return universe_df

    def _apply_universe_filter(self):
        """将股票池过滤应用到所有数据"""
        print("  将股票池过滤应用到所有数据...")

        for field_name, df in self.raw_data.items():
            # 将不在股票池中的数据设为NaN
            self.raw_data[field_name] = df.where(self.universe_df)

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

        # 简单的因子计算逻辑
        if factor_name == 'pe_inv' and 'pe_ttm' in fields:
            # PE倒数因子
            pe_data = self.raw_data['pe_ttm']
            factor_data = 1 / pe_data
            factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
        else:
            # 默认使用第一个字段
            factor_data = self.raw_data[fields[0]]

        return factor_data

    def get_universe(self) -> pd.DataFrame:
        """获取股票池"""
        return self.universe_df

    def get_price_data(self) -> pd.DataFrame:
        """获取价格数据"""
        return self.raw_data['close']

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


def create_data_manager(config_path: str) -> DataManager:
    """
    创建数据管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        DataManager实例
    """
    return DataManager(config_path)


