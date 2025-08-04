from dataclasses import dataclass, field
from typing import List, Dict, Any
import copy
from dataclasses import dataclass, field, asdict

from projects._03_factor_selection.config.base_config import INDEX_CODES


# 使用 @dataclass 装饰器，Python会自动为我们生成__init__, __repr__等方法
@dataclass
class IndexFilterConfig:
    """指数过滤配置"""
    enable: bool
    index_code: str
    def to_dict(self): return asdict(self)


@dataclass
class PoolFiltersConfig:
    """普适性的过滤器配置"""
    remove_st: bool = True
    remove_new_stocks: bool = True
    adapt_tradeable_matrix_by_suspend_resume: bool = True
    min_liquidity_percentile: float = 0.0
    min_market_cap_percentile: float = 0.0
    def to_dict(self): return asdict(self)


@dataclass
class StockPoolProfile:
    """单个股票池的完整配置"""
    index_filter: IndexFilterConfig
    filters: PoolFiltersConfig
    def to_dict(self): return asdict(self)


@dataclass
class BacktestConfig:
    """回测时间配置"""
    start_date: str
    end_date: str
    def to_dict(self): return asdict(self)


# 这是最顶层的完整配置对象
@dataclass
class FullQuantConfig:
    """最终生成的完整配置对象"""
    backtest: BacktestConfig
    stock_pool_profiles: Dict[str, StockPoolProfile]
    target_factors_for_evaluation: Dict[str, List[str]]  # {"fields": target_factors}

    # 提供一个方法，方便地将自身转换为字典，以便系统其他部分使用
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backtest": self.backtest.to_dict(),
            "stock_pool_profiles": {
                name: profile.to_dict() for name, profile in self.stock_pool_profiles.items()
            },
            "target_factors_for_evaluation": self.target_factors_for_evaluation
        }


# ==============================================================================
# 预设股票池模板 (Stock Pool Presets)
# 你可以在这里定义所有常用的股票池配置
# ==============================================================================

# 模板1: 机构标准池 (基于沪深300)

def make_pool_profile(pool_name, Index_filter, index_code,remove_st,remove_new_stocks,adapt_tradeable_matrix_by_suspend_resume, min_liquidity_percentile, min_market_cap_percentile):
    profile = StockPoolProfile(
        index_filter=IndexFilterConfig(enable=Index_filter, index_code=index_code),
        filters=PoolFiltersConfig(
            remove_st = remove_st,
            remove_new_stocks = remove_new_stocks,
            adapt_tradeable_matrix_by_suspend_resume = adapt_tradeable_matrix_by_suspend_resume,
            min_liquidity_percentile=min_liquidity_percentile,
            min_market_cap_percentile=min_market_cap_percentile
        )
    )
    return {
        pool_name:  # 没办法，写死吧，这个设计回旋镖打了自己
        profile
    }


CSI300_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000300.SH',True,True,True, 0, 0)
CSI300_more_filter_profile = make_pool_profile('institutional_stock_pool', True, '000300.SH',True,True,True, 0.1, 0.05)
CSI1000_more_filter_profile = make_pool_profile('institutional_stock_pool', True, INDEX_CODES['ZZ1000'],True,True,True, 0.1, 0.05)
CSI300_none_TFF_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000300.SH',True,False,False, 0, 0)
CSI300_none_FTF_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000300.SH',False,True,False, 0, 0)
CSI300_none_FFT_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000300.SH',False,False,True, 0, 0)
CSI300_none_FFF_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000300.SH',False,False,False, 0, 0)
CSI500_most_basic_profile = make_pool_profile('institutional_stock_pool', True, '000905.SH', True,True,True,0, 0)
# 用于我需要在最真实的环境，交易，需要必须要过滤流动差劲的
pool_for_massive_test_CSI800_profile = make_pool_profile('institutional_stock_pool', True, '000905.SH', True,True,True,0.1, 0.05)
pool_for_massive_test_MICROSTRUCTURE_profile = make_pool_profile('microstructure_stock_pool', True, '000905.SH', True,True,True,0.2, 0.2)


def generate_dynamic_config(
        start_date: str,
        end_date: str,
        target_factors: List[str],
        pool_profiles
) -> Dict[str, Any]:
    """
    【最终版】动态生成量化回测配置字典。

    Args:
        start_date (str): 回测开始日期, 'YYYY-MM-DD'
        end_date (str): 回测结束日期, 'YYYY-MM-DD'
        target_factors (List[str]): 要测试的因子名称列表, e.g., ['market_cap_log', 'beta']
        pool_custom_name (str): 生成的配置中，这个股票池的名字

    Returns:
        Dict[str, Any]: 一个完全合规的、可直接用于系统的配置字典。
    """
    print(f"🚀 正在动态生成配置...")
    print(f"   - 时间范围: {start_date} -> {end_date}")
    print(f"   - 目标因子: {target_factors}")
    print(f"   - 股票池模板: {pool_profiles.keys()}")

    # 1. 检查预设是否存在

    # 2. 构建回测时间配置
    backtest_conf = BacktestConfig(start_date=start_date, end_date=end_date)

    # 3. 构建因子配置
    factors_conf = {"fields": target_factors}

    # 4. 构建股票池配置 (使用深拷贝以防修改原始模板)
    #    这里只生成一个股票池，因为动态配置通常是针对单次实验的

    # 5. 组装成最终的完整配置对象
    full_config = FullQuantConfig(
        backtest=backtest_conf,
        stock_pool_profiles=pool_profiles,
        target_factors_for_evaluation=factors_conf
    )

    # 6. 返回字典格式
    return full_config.to_dict()
