# 文件名: data_validator.py
# 作用：一个生产级的、健壮的数据完整性检验工具。
#      用于审计 downloader.py 下载的数据是否完整、准确、自洽。
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import warnings
import json
from functools import reduce

# 假设您的项目结构，这是您常量配置文件所在的位置
# 请根据您的实际情况修改
from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR, parquet_file_names, every_day_parquet_file_names
from quant_lib.utils import is_trading_day

warnings.filterwarnings('ignore')


class DataIntegrityChecker:
    """数据完整性检验器 - 专业定版"""

    def __init__(self, data_path: Optional[Path] = None, start_year: int = 2018):
        """
        初始化检验器
        Args:
            data_path: 数据存储路径，如果为None则使用默认路径
            start_year: 检查的起始年份
        """
        self.data_path = data_path or LOCAL_PARQUET_DATA_DIR

        self.start_day = pd.to_datetime('20180101')
        self.end_day = pd.Timestamp.today().normalize()
        self.end_day = pd.to_datetime('20250712')
        self._cache = {}  # 优化建议：增加类内缓存，避免重复读取文件

        if not self.data_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {self.data_path}")

        print(f"数据完整性检验器初始化完成")
        print(f"数据路径: {self.data_path}")
        print(f"检验时间范围: {self.start_day} - {self.end_day}")

    def _load_data(self, file_name: str, partitioned: bool = False) -> Optional[pd.DataFrame]:
        """
        【优化】一个带缓存的、统一的数据加载方法
        Args:
            file_name: 相对于数据根目录的文件或文件夹名
            partitioned: 是否是按年份分区的目录
        """
        if file_name in self._cache:
            return self._cache[file_name]

        file_path = self.data_path / file_name
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                self._cache[file_name] = df
                return df.copy()  # 返回副本以防意外修改
            except Exception as e:
                print(f"  - 读取失败: {file_name}, {e}")
                return None
        print(f"  - 文件或目录不存在: {file_name}")
        return None

    def check_basic_files(self) -> None:
        """检查基础文件是否存在"""
        print("\n" + "=" * 50)
        print("1. 基础文件存在性检查")
        print("=" * 50)

        basic_files = {
            'trade_cal.parquet': '交易日历',
            'stock_basic.parquet': '股票基本信息',
            'namechange.parquet': '股票名称变更历史'
        }

        for file_name, description in basic_files.items():
            df = self._load_data(file_name)
            exists = df is not None
            status = "✓ 存在" if exists else "✗ 缺失"
            print(f"{description:15} {status}")

            if exists:
                print(f"  - 记录数: {len(df):,}")
                if file_name == 'trade_cal.parquet':
                    df['cal_date'] = pd.to_datetime(df['cal_date'])

                    mask = (df['cal_date'] >= self.start_day) & (df['cal_date'] <= self.end_day) & (df['is_open'] == 1)
                    df = df[mask]

                    print(f"  - 交易日数量: {df.shape[0]:,}")
                    print(f"  - 时间范围: {df['cal_date'].min()} ~ {df['cal_date'].max()}")
                elif file_name == 'stock_basic.parquet':
                    print(f"  - 股票数量: {df['ts_code'].nunique():,}")

    def check_trading_days_completeness(self) -> None:
        """检查交易日完整性"""
        print("\n" + "=" * 50)
        print("2. 交易日完整性检查")
        print("=" * 50)

        trade_cal = self._load_data('trade_cal.parquet')
        if trade_cal is None:
            raise ValueError("✗ 无法加载交易日历，跳过此项检查。")

        trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'])

        mask = (trade_cal['is_open'] == 1) & (trade_cal['cal_date'] >= self.start_day) & (
                trade_cal['cal_date'] <= self.end_day)
        expected_dates = set(trade_cal[mask]['cal_date'])

        for dataset in every_day_parquet_file_names:
            print(f"\n检查 {dataset} 的交易日完整性:")
            df = self._load_data(dataset)
            if df is None:
                continue

            actual_dates = set(pd.to_datetime(df['trade_date'].unique()))
            missing_days = expected_dates - actual_dates

            if not missing_days:
                print(f"  ✓ [通过] 所有 {len(expected_dates)} 个交易日的数据均存在。")
            else:
                print(f"  🚨 [失败] 缺失了 {len(missing_days)} 个交易日的数据！")
                print(f"     例如: {sorted([d.strftime('%Y-%m-%d') for d in list(missing_days)[:5]])}"
                      f"{'...' if len(missing_days) > 5 else ''}")

    def check_stock_coverage_robust(self, sample_size: int = 30) -> None:
        """【关键修正 & 重构版】使用每日抽样检查股票覆盖度的完整性"""
        print("\n" + "=" * 50)
        print("3. 股票覆盖度检查 (每日随机抽样)")
        print("=" * 50)

        # --- 1. 一次性加载所有需要的数据 ---
        stock_basic = self._load_data('stock_basic.parquet')
        price_df = self._load_data('daily_hfq')
        trade_cal = self._load_data('trade_cal.parquet')

        if stock_basic is None or price_df is None or trade_cal is None:
            print("✗ 缺少stock_basic, daily_hfq或trade_cal数据，跳过此项检查。")
            return

        # --- 2. 一次性进行数据类型转换和准备 ---
        stock_basic['list_date'] = pd.to_datetime(stock_basic['list_date'])
        stock_basic['delist_date'] = pd.to_datetime(stock_basic['delist_date'])

        # 【Bug修复】强制将日期列转为字符串再转为datetime，确保能正确解析整数日期
        price_df['trade_date'] = pd.to_datetime(price_df['trade_date'].astype(str))
        trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'].astype(str))

        # --- 3. 准备交易日历和抽样 ---
        trading_dates_series = trade_cal[trade_cal['is_open'] == 1]['cal_date'].sort_values()

        # 创建一个 日期 -> 上一交易日 的映射，避免在循环中重复查找
        prev_trade_date_map = trading_dates_series.shift(1).set_axis(trading_dates_series)

        all_trading_dates = price_df['trade_date'].unique()
        sample_dates = np.random.choice(all_trading_dates, min(sample_size, len(all_trading_dates)), replace=False)

        print(f"  将在 {len(sample_dates)} 个随机交易日上进行抽样检查...")

        all_coverage = []
        # --- 4. 清爽、高效的循环体 ---
        missing_stocks_ret = []
        for date in sorted(sample_dates):
            # 获取上一交易日
            prev_date = prev_trade_date_map.get(date)
            if pd.isna(prev_date):
                continue
            # 计算当日的期望股票池
            expected_mask_today = (stock_basic['list_date'] <= date) & \
                                  ((stock_basic['delist_date'].isna()) | (stock_basic['delist_date'] > date)) \
                                  & (~stock_basic['ts_code'].str.endswith('.BJ'))
            expected_stocks_today = set(stock_basic[expected_mask_today]['ts_code'])

            # 计算当日的实际股票池
            actual_stocks_today = set(price_df[price_df['trade_date'] == date]['ts_code'])

            # 计算覆盖率
            coverage = len(actual_stocks_today) / len(expected_stocks_today) if expected_stocks_today else 1.0
            all_coverage.append(coverage)

            # 打印诊断信息
            if coverage < 1:
                print(
                    f"  {date}: ⚠ 覆盖率较低: {coverage:.2%}，缺失 {len(expected_stocks_today) - len(actual_stocks_today)} 只股票")

                # --- 更清晰的诊断逻辑 ---
                missing_stocks = expected_stocks_today - actual_stocks_today
                missing_stocks_ret.append(missing_stocks)
                if missing_stocks:
                    print(f"    -> 缺失股票示例: {list(missing_stocks)[:3]}")

                    # 检查这些缺失的股票，昨天是否存在
                    prev_day_actual_stocks = set(price_df[price_df['trade_date'] == prev_date]['ts_code'])
                    existed_yesterday = set(missing_stocks).intersection(prev_day_actual_stocks)
                    print(f'今天缺失数量{len(missing_stocks)},昨天存在的数量{existed_yesterday}')
                    if len(existed_yesterday) == len(missing_stocks):
                        print('今天没有的，昨天全有')

        coommen_miss_ts_codes = reduce(lambda a, b: set(a) & set(b), missing_stocks_ret)
        print("抽样10天都缺失的股票:", coommen_miss_ts_codes)

        # --- 5. 最终报告 ---
        avg_coverage = np.mean(all_coverage) if all_coverage else 0
        if avg_coverage < 0.98:
            print(f"\n  🚨 [警告] 随机抽样日的平均覆盖率为 {avg_coverage:.2%}, 可能存在股票缺失问题(主要为停牌)。")
        else:
            print(f"\n  ✓ [通过] 随机抽样日的平均覆盖率为 {avg_coverage:.2%}, 覆盖度较高。")

    def check_metric_consistency(self) -> None:
        """【新增】检验关键指标的交叉一致性"""
        print("\n" + "=" * 50)
        print("4. 关键指标交叉验证 (市值)")
        print("=" * 50)

        daily_basic = self._load_data('daily_basic')
        if daily_basic is None:
            print("✗ 缺少daily_basic数据，跳过此项检查。")
            return

        df = daily_basic.copy()
        df = df[df['total_mv'] > 0]

        # 注意单位: total_share(万股), total_mv(万元), close(元)
        df['calculated_mv'] = df['close'] * df['total_share']
        df['mv_diff_ratio'] = (df['total_mv'] - df['calculated_mv']).abs() / df['total_mv']

        high_diff_records_ratio = (df['mv_diff_ratio'] > 0.01).mean()

        print(f"  “总市值”与“收盘价×总股本”差异大于1%的记录占比: {high_diff_records_ratio:.2%}")

        if high_diff_records_ratio > 0.01:
            print(f"  🚨 [警告] 市值交叉验证：total_mv与'close*total_share'存在显著差异！")
            print(f"     这通常是由于未对`close`价格进行复权处理导致的。")
        else:
            print(f"  ✓ [通过] 市值交叉验证通过，数据内部一致性较高。")

    def run_all_checks(self, save_path: Optional[str] = 'data_integrity_report.txt') -> None:
        """执行所有检验并生成报告"""
        print("\n" + "=" * 60)
        print(f"数据完整性检查报告 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print("=" * 60)

        self.check_basic_files()
        self.check_trading_days_completeness()
        self.check_stock_coverage_robust()
        self.check_metric_consistency()

        print("\n" + "=" * 60)
        print("所有检查完成！请查看上述输出了解数据完整性情况。")
        print("=" * 60)


if __name__ == '__main__':
    # 创建检验器实例
    checker = DataIntegrityChecker()

    # 执行所有检查
    checker.run_all_checks()
