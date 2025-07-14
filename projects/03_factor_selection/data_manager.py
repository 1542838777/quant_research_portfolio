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

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.data_loader import DataLoader
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
        print("="*80)
        print("第二阶段：数据加载与股票池构建")
        print("="*80)
        
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
                                'total_mv', 'turnover_rate',#为了过滤 很差劲的股票 仅此而已，不会作其他计算 、'total_mv'还可 用于计算中性化
                                'industry',#用于计算中性化
                                'circ_mv'#流通市值 用于WOS，加权最小二方跟  ，回归法会用到
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
            required_fields.add('name')  # 用于识别ST股票
        
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
            if field_name in ['close', 'total_mv']:
                negative_ratio = (df <= 0).sum().sum() / df.notna().sum().sum()
                if negative_ratio > 0:
                    print(f"    警告: {field_name} 存在 {negative_ratio:.2%} 的非正值")
    
    def _build_universe(self) -> pd.DataFrame:
        """
        构建动态股票池
        
        Returns:
            股票池DataFrame，True表示该股票在该日期可用
        """
        print("  构建基础股票池...")
        
        # 获取价格数据作为基础
        if 'close' not in self.raw_data:
            raise ValueError("缺少价格数据，无法构建股票池")
        
        close_df = self.raw_data['close']
        universe_df = pd.DataFrame(
            index=close_df.index, 
            columns=close_df.columns, 
            data=False
        )
        
        # 基础过滤：有价格数据的股票
        universe_df = close_df.notna()
        
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
        print(f"    股票池统计:")
        print(f"      平均每日股票数: {daily_count.mean():.0f}")
        print(f"      最少每日股票数: {daily_count.min():.0f}")
        print(f"      最多每日股票数: {daily_count.max():.0f}")
        
        return universe_df
    
    def _filter_st_stocks(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        """剔除ST股票"""
        if 'name' not in self.raw_data:
            print("    警告: 缺少股票名称数据，无法过滤ST股票")
            return universe_df
        
        name_df = self.raw_data['name']
        
        # 识别ST股票（名称包含ST、*ST等）
        st_patterns = ['ST', '*ST', 'S*ST', 'SST']
        
        for date in universe_df.index:
            if date in name_df.index:
                names = name_df.loc[date]
                for pattern in st_patterns:
                    st_mask = names.str.contains(pattern, na=False)
                    universe_df.loc[date, st_mask] = False
        
        return universe_df
    
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
                    threshold = valid_turnover.quantile(min_percentile)
                    low_liquidity_mask = turnover_values < threshold
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


# 使用示例
if __name__ == "__main__":
    # 创建数据管理器
    config_path = "config.yml"
    data_manager = create_data_manager(config_path)
    
    # 加载数据
    data_dict = data_manager.load_all_data()
    
    # 获取因子数据
    factor_data = data_manager.get_factor_data()
    print(f"\n因子数据形状: {factor_data.shape}")
    
    # 保存数据摘要
    data_manager.save_data_summary("results/data_summary")
