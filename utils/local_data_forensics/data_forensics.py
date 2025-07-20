# 文件名: data_forensics.py
# 作用：一个高效的数据法证工具，用于诊断宽表中字段的NaN值，
#      并区分其原因是"合理缺失"（上市前/退市后/停牌）还是"可疑缺失"。
#      使用向量化操作提高效率，支持分区数据结构。

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR

warnings.filterwarnings('ignore')

# 检查是否有pyarrow支持
try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("⚠️ 警告: 未安装pyarrow，将尝试使用fastparquet或其他引擎")


class DataForensics:
    """数据法证诊断器 - 高效向量化版本"""

    def __init__(self, data_path: Path = None):
        """
        初始化诊断器，预加载基础参照数据。
        """
        self.data_path = data_path or LOCAL_PARQUET_DATA_DIR
        print(f"数据法证诊断器初始化...")
        print(f"使用数据路径: {self.data_path}")

        # 1. 预加载股票基本信息，作为我们的"户籍系统"
        try:
            self.stock_basic = self._safe_read_parquet(self.data_path / 'stock_basic.parquet')
            self.stock_basic['list_date'] = pd.to_datetime(self.stock_basic['list_date'])
            self.stock_basic['delist_date'] = pd.to_datetime(self.stock_basic['delist_date'])

            # 创建便于查询的Series
            self.list_dates = self.stock_basic.set_index('ts_code')['list_date']
            self.delist_dates = self.stock_basic.set_index('ts_code')['delist_date']
            print(f"✓ 股票基本信息加载成功，共 {len(self.stock_basic)} 只股票。")
        except Exception as e:
            raise FileNotFoundError(f"无法加载 stock_basic.parquet，诊断无法进行: {e}")

        # 2. 预加载交易日历
        try:
            self.trade_cal = self._safe_read_parquet(self.data_path / 'trade_cal.parquet')
            self.trade_cal['cal_date'] = pd.to_datetime(self.trade_cal['cal_date'])
            self.trading_dates = self.trade_cal[self.trade_cal['is_open'] == 1]['cal_date'].sort_values()
            print(f"✓ 交易日历加载成功，共 {len(self.trading_dates)} 个交易日。")
        except Exception as e:
            print(f"⚠️ 警告: 无法加载交易日历: {e}")
            self.trading_dates = None

    def _safe_read_parquet(self, file_path: Path) -> pd.DataFrame:
        """
        安全读取parquet文件，尝试不同的引擎
        """
        engines = ['pyarrow', 'fastparquet'] if HAS_PYARROW else ['fastparquet']

        for engine in engines:
            try:
                return pd.read_parquet(file_path, engine=engine)
            except ImportError:
                continue
            except Exception as e:
                if engine == engines[-1]:  # 最后一个引擎也失败了
                    raise e
                continue

        # 如果所有引擎都失败，尝试不指定引擎
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            raise ImportError(f"无法读取parquet文件 {file_path}。请安装 pyarrow 或 fastparquet: pip install pyarrow") from e

    def _load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        智能加载数据集，支持分区和单文件两种格式

        Args:
            dataset_name: 数据集名称，如 'daily_hfq' 或 'stock_basic.parquet'

        Returns:
            加载的DataFrame
        """
        dataset_path = self.data_path / dataset_name

        if dataset_path.is_dir():
            # 分区数据，直接读取整个目录
            print(f"  -> 检测到分区数据，加载整个目录: {dataset_name}")
            df = self._safe_read_parquet(dataset_path)
        elif dataset_path.with_suffix('.parquet').exists():
            # 单文件数据
            print(f"  -> 检测到单文件数据: {dataset_name}")
            df = self._safe_read_parquet(dataset_path.with_suffix('.parquet'))
        else:
            raise FileNotFoundError(f"数据集不存在: {dataset_name}")

        return df

    def diagnose_field_nan(self, field_name: str, dataset_name: str, 
                          sample_stocks: int = 10, detailed_analysis: bool = True):
        """
        对指定字段的NaN值进行高效诊断。

        Args:
            field_name (str): 要诊断的字段名 (如 'close')
            dataset_name (str): 该字段所在的数据集名 (如 'daily_hfq')
            sample_stocks (int): 随机抽取多少只股票进行详细诊断
            detailed_analysis (bool): 是否进行详细的个股分析
        """
        print("\n" + "="*70)
        print(f"🔍 开始对数据集 <{dataset_name}> 中的字段 <{field_name}> 进行NaN诊断")
        print("="*70)

        # 1. 加载目标数据
        try:
            df = self._load_dataset(dataset_name)
            
            # 检查字段是否存在
            if field_name not in df.columns:
                print(f"🚨 错误: 字段 '{field_name}' 不存在于数据集 '{dataset_name}' 中")
                print(f"可用字段: {list(df.columns)}")
                return
                
            # 将长表转换为宽表
            print(f"  -> 正在转换为宽表格式...")
            wide_df = df.pivot_table(index='trade_date', columns='ts_code', values=field_name)
            wide_df.index = pd.to_datetime(wide_df.index)
            wide_df = wide_df.sort_index()
            
            print(f"✓ 成功加载并转换宽表，形状: {wide_df.shape}")
            print(f"  -> 时间范围: {wide_df.index.min().date()} 至 {wide_df.index.max().date()}")
            print(f"  -> 股票数量: {wide_df.shape[1]}")
            
        except Exception as e:
            print(f"🚨 错误: 加载或转换 {dataset_name} 失败: {e}")
            return

        # 2. 全局NaN统计
        nan_mask = wide_df.isna()
        total_nans = nan_mask.sum().sum()
        total_cells = wide_df.size
        nan_ratio = total_nans / total_cells
        
        print(f"\n📊 全局NaN统计:")
        print(f"  -> 总NaN数量: {total_nans:,}")
        print(f"  -> 总单元格数: {total_cells:,}")
        print(f"  -> NaN比例: {nan_ratio:.2%}")

        if total_nans == 0:
            print("✅ [优秀] 该字段没有任何NaN值！")
            return

        # 3. 向量化归因分析
        print(f"\n🕵️ 开始向量化归因分析...")
        attribution_results = self._vectorized_attribution_analysis(wide_df, nan_mask)
        
        # 4. 输出归因统计
        self._print_attribution_summary(attribution_results, total_nans)
        
        # 5. 详细个股分析（可选）
        if detailed_analysis and sample_stocks > 0:
            self._detailed_stock_analysis(wide_df, nan_mask, sample_stocks)

    def _vectorized_attribution_analysis(self, wide_df: pd.DataFrame, 
                                       nan_mask: pd.DataFrame) -> Dict[str, int]:
        """
        使用向量化操作进行高效的NaN归因分析
        
        Returns:
            归因结果字典
        """
        print("  -> 执行向量化归因计算...")
        
        # 获取所有有NaN的股票
        stocks_with_nan = nan_mask.columns[nan_mask.any()]
        
        # 初始化计数器
        attribution = {
            'before_listing': 0,
            'after_delisting': 0, 
            'during_trading': 0,
            'unknown_stock': 0
        }
        
        # 批量处理股票
        for stock in stocks_with_nan:
            stock_nan_dates = wide_df.index[nan_mask[stock]]
            
            # 获取该股票的上市和退市日期
            list_date = self.list_dates.get(stock)
            delist_date = self.delist_dates.get(stock)
            
            if pd.isna(list_date):
                attribution['unknown_stock'] += len(stock_nan_dates)
                continue
                
            # 向量化比较
            before_listing_mask = stock_nan_dates < list_date
            attribution['before_listing'] += before_listing_mask.sum()
            
            if pd.notna(delist_date):
                after_delisting_mask = stock_nan_dates > delist_date
                attribution['after_delisting'] += after_delisting_mask.sum()
                
                # 剩余的就是交易期间的NaN
                during_trading_count = len(stock_nan_dates) - before_listing_mask.sum() - after_delisting_mask.sum()
            else:
                during_trading_count = len(stock_nan_dates) - before_listing_mask.sum()
                
            attribution['during_trading'] += during_trading_count
            
        return attribution

    def _print_attribution_summary(self, attribution: Dict[str, int], total_nans: int):
        """打印归因分析摘要"""
        print(f"\n📋 NaN归因分析结果:")
        print(f"  ✅ 上市前缺失: {attribution['before_listing']:,} ({attribution['before_listing']/total_nans:.1%}) - 合理")
        print(f"  ✅ 退市后缺失: {attribution['after_delisting']:,} ({attribution['after_delisting']/total_nans:.1%}) - 合理") 
        print(f"  ℹ️  交易期间缺失: {attribution['during_trading']:,} ({attribution['during_trading']/total_nans:.1%}) - 大概率停牌")
        print(f"  ❓ 未知股票缺失: {attribution['unknown_stock']:,} ({attribution['unknown_stock']/total_nans:.1%}) - 需要检查")
        
        # 计算可疑程度
        suspicious_ratio = attribution['unknown_stock'] / total_nans
        if suspicious_ratio > 0.05:  # 超过5%认为可疑
            print(f"\n⚠️  警告: 未知股票缺失比例较高 ({suspicious_ratio:.1%})，建议检查数据完整性！")
        else:
            print(f"\n✅ 数据质量良好，大部分NaN都有合理解释。")

    def _detailed_stock_analysis(self, wide_df: pd.DataFrame, nan_mask: pd.DataFrame,
                                sample_stocks: int):
        """详细的个股分析"""
        print(f"\n🔬 详细个股分析 (抽样 {sample_stocks} 只股票):")
        print("-" * 50)

        # 找到NaN最多的股票进行抽样分析
        nan_counts_per_stock = nan_mask.sum().sort_values(ascending=False)
        stocks_to_check = nan_counts_per_stock.head(sample_stocks).index

        for i, stock in enumerate(stocks_to_check, 1):
            print(f"\n[{i}] 股票: {stock} (NaN数量: {nan_counts_per_stock[stock]})")

            stock_series = wide_df[stock]
            nan_dates = stock_series[stock_series.isna()].index

            list_date = self.list_dates.get(stock)
            delist_date = self.delist_dates.get(stock)

            if pd.isna(list_date):
                print("    ❓ 警告: 在stock_basic中未找到该股票信息")
                continue

            # 归因分析
            before_listing = nan_dates[nan_dates < list_date]
            after_delisting = nan_dates[nan_dates > delist_date] if pd.notna(delist_date) else pd.DatetimeIndex([])
            during_trading = nan_dates.drop(before_listing).drop(after_delisting, errors='ignore')

            print(f"    📅 上市: {list_date.date()}, 退市: {delist_date.date() if pd.notna(delist_date) else '未退市'}")

            if not before_listing.empty:
                print(f"    ✅ 上市前NaN: {len(before_listing)}个")
            if not after_delisting.empty:
                print(f"    ✅ 退市后NaN: {len(after_delisting)}个")
            if not during_trading.empty:
                print(f"    ℹ️  交易期间NaN: {len(during_trading)}个")

                # 分析连续性
                if len(during_trading) > 1:
                    gaps = (during_trading.to_series().diff() > pd.Timedelta('1 day')).sum()
                    print(f"       -> 形成 {gaps + 1} 个连续缺失区间")

                    # 显示最长的缺失区间
                    if len(during_trading) > 5:
                        print(f"       -> 最近缺失日期: {during_trading[-3:].strftime('%Y-%m-%d').tolist()}")

    def batch_diagnose(self, field_dataset_pairs: List[Tuple[str, str]],
                      sample_stocks: int = 5, detailed_analysis: bool = False):
        """
        批量诊断多个字段

        Args:
            field_dataset_pairs: [(field_name, dataset_name), ...] 的列表
            sample_stocks: 每个字段抽样分析的股票数量
            detailed_analysis: 是否进行详细分析
        """
        print(f"\n🚀 开始批量诊断 {len(field_dataset_pairs)} 个字段...")

        results_summary = []

        for i, (field_name, dataset_name) in enumerate(field_dataset_pairs, 1):
            print(f"\n{'='*80}")
            print(f"📊 批量诊断进度: {i}/{len(field_dataset_pairs)}")
            print(f"{'='*80}")

            try:
                # 执行单个字段诊断
                self.diagnose_field_nan(
                    field_name=field_name,
                    dataset_name=dataset_name,
                    sample_stocks=sample_stocks,
                    detailed_analysis=detailed_analysis
                )
                results_summary.append((field_name, dataset_name, "✅ 成功"))

            except Exception as e:
                print(f"❌ 诊断失败: {e}")
                results_summary.append((field_name, dataset_name, f"❌ 失败: {str(e)[:50]}"))

        # 输出批量诊断摘要
        print(f"\n{'='*80}")
        print("📋 批量诊断摘要:")
        print(f"{'='*80}")
        for field_name, dataset_name, status in results_summary:
            print(f"  {status} | {field_name} @ {dataset_name}")

    def generate_data_quality_report(self, output_path: Optional[str] = None) -> Dict:
        """
        生成数据质量报告

        Args:
            output_path: 报告保存路径，如果为None则只返回结果不保存

        Returns:
            数据质量报告字典
        """
        print(f"\n📊 生成数据质量报告...")

        # 定义要检查的核心字段
        core_fields = [
            ('close', 'daily_hfq'),
            ('vol', 'daily_hfq'),
            ('pe_ttm', 'daily_basic'),
            ('pb', 'daily_basic'),
            ('turnover_rate', 'daily_basic')
        ]

        report = {
            'generated_at': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'fields_analyzed': [],
            'overall_quality_score': 0.0
        }

        quality_scores = []

        for field_name, dataset_name in core_fields:
            print(f"\n  -> 分析 {field_name} @ {dataset_name}")

            try:
                # 加载数据并计算NaN统计
                df = self._load_dataset(dataset_name)
                if field_name not in df.columns:
                    continue

                wide_df = df.pivot_table(index='trade_date', columns='ts_code', values=field_name)
                nan_mask = wide_df.isna()

                total_nans = nan_mask.sum().sum()
                total_cells = wide_df.size
                nan_ratio = total_nans / total_cells if total_cells > 0 else 0

                # 计算质量分数 (1 - nan_ratio)
                quality_score = max(0, 1 - nan_ratio)
                quality_scores.append(quality_score)

                # 向量化归因分析
                attribution = self._vectorized_attribution_analysis(wide_df, nan_mask)

                field_report = {
                    'field_name': field_name,
                    'dataset_name': dataset_name,
                    'total_nans': total_nans,
                    'total_cells': total_cells,
                    'nan_ratio': nan_ratio,
                    'quality_score': quality_score,
                    'attribution': attribution
                }

                report['fields_analyzed'].append(field_report)

            except Exception as e:
                print(f"    ❌ 分析失败: {e}")
                continue

        # 计算总体质量分数
        if quality_scores:
            report['overall_quality_score'] = sum(quality_scores) / len(quality_scores)

        # 保存报告
        if output_path:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"✅ 数据质量报告已保存至: {output_path}")

        return report


# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 实例化诊断器
    forensics = DataForensics()

    # 2. 单个字段诊断
    # forensics.diagnose_field_nan(
    #     field_name='close',
    #     dataset_name='daily_hfq',
    #     sample_stocks=8,
    #     detailed_analysis=True
    # )
    batch_fields = [
        # ('industry','stock_basic.parquet'),
        ('pe_ttm', 'daily_basic'),

        ('close', 'daily_hfq'),
        ('turnover_rate', 'daily_basic'),

        ('pb','daily_basic'),
        ('circ_mv','daily_basic'),
        # ('list_date','stock_basic.parquet'),
        ('total_mv','daily_basic')
    ]

    # 3. 批量诊断示例
    forensics.batch_diagnose(batch_fields, sample_stocks=3)

    # 4. 生成数据质量报告
    report = forensics.generate_data_quality_report('data_quality_report.json')

    print("\n" + "="*70)
    print("🎯 诊断完成！你可以继续诊断其他字段:")

