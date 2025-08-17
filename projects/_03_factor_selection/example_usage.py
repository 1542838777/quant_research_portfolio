"""
重构后的策略工厂使用示例

本脚本演示如何使用重构后的策略工厂进行完整的因子研究流程：
1. 单因子测试
2. 类别内优化
3. 类别间优化
4. 可视化和报告生成

"""
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict

from data.local_data_load import load_suspend_d_df, load_dividend_events_long
from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
from projects._03_factor_selection.factor_manager.registry.factor_registry import FactorCategory

# 添加项目根目录到路径
# project_root = Path(__file__).parent.parent.parent.parent
# sys.path.append(str(project_root))
#
# # 添加当前项目目录到路径，以支持相对导入
# current_project = Path(__file__).parent
# sys.path.insert(0, str(current_project))

# 导入重构后的模块 - 使用绝对导入
from projects._03_factor_selection.factory.strategy_factory import StrategyFactory
from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR
from quant_lib.config.logger_config import setup_logger, log_success
from quant_lib.tushare.api_wrapper import call_pro_tushare_api
from quant_lib.tushare.tushare_client import TushareClient
from quant_lib.utils.test import check_step

# 配置日志
logger = setup_logger(__name__)


def verify_pct_chg(factor_manager):
    pct_chg=factor_manager.get_raw_factor('pct_chg')
    pct_chg = pct_chg['000001.SZ']

    daily=pd.read_parquet(LOCAL_PARQUET_DATA_DIR/'daily')
    daily.index = pd.to_datetime(daily['trade_date'])
    daily['trade_date'] = daily.index
    daily = daily[daily['trade_date']>=pd.to_datetime('20230920')]
    daily.sort_index(inplace=True)
    daily = daily[daily['ts_code']=='000001.SZ']

    print(1)


def verify_volatility_calculation(factor_manager):
    """验证波动率计算逻辑"""
    pct_chg = factor_manager.get_raw_factor('pct_chg')
    volatility_120d = factor_manager.get_raw_factor('volatility_120d')

    # 手工计算某只股票的120日年化波动率
    stock_code = '000001.SZ'
    stock_returns = pct_chg[stock_code].dropna()

    # 手工计算120日滚动标准差，然后年化
    manual_vol = stock_returns.rolling(window=120, min_periods=60).std() * np.sqrt(252)
    engine_vol = volatility_120d[stock_code]

    # 对比结果
    comparison = pd.DataFrame({
        'manual': manual_vol,
        'engine': engine_vol,
        'diff': manual_vol - engine_vol
    }).dropna()

    print("年化波动率计算对比:")
    print(comparison.tail(10))
    print(f"最大差异: {comparison['diff'].abs().max()}")

    # 检查是否一致
    if comparison['diff'].abs().max() < 1e-10:
        print("✓ 波动率计算逻辑正确")
    else:
        print("✗ 波动率计算存在差异")


def check_look_ahead_bias(factor_manager):
    """检查是否存在前瞻偏差"""
    vol_factor = factor_manager.get_raw_factor('volatility_120d')
    pct_chg = factor_manager.get_raw_factor('pct_chg')

    # 检查因子计算是否用了未来数据
    test_date = '2024-12-19'

    print(f"检查 {test_date} 的波动率计算:")
    print(f"因子值: {vol_factor.loc[test_date, '000001.SZ']}")

    # 手动验证：只用历史数据计算
    historical_returns = pct_chg.loc[:test_date, '000001.SZ'].iloc[:-1]  # 不包含当天
    manual_vol = historical_returns.tail(120).std() * np.sqrt(252)

    print(f"手动计算(仅用历史): {manual_vol}")
    print(f"差异: {abs(vol_factor.loc[test_date, '000001.SZ'] - manual_vol)}")


def debug_ic_calculation_detailed(factor_manager):
    """详细调试IC计算过程"""
    vol_factor = factor_manager.get_raw_factor('volatility_120d')
    pct_chg = factor_manager.get_raw_factor('pct_chg')

    # 选择最近的几个交易日
    recent_dates = vol_factor.index[-5:]

    print("详细IC计算检查:")
    print("=" * 50)

    for i, date in enumerate(recent_dates[:-1]):  # 最后一天没有下期收益
        next_date = recent_dates[i + 1]

        # 获取截面数据
        factor_cross = vol_factor.loc[date].dropna()
        return_cross = pct_chg.loc[next_date].dropna()

        # 找共同股票
        common_stocks = factor_cross.index.intersection(return_cross.index)

        if len(common_stocks) >= 10:
            factor_vals = factor_cross[common_stocks]
            return_vals = return_cross[common_stocks]

            # 计算相关系数
            pearson_ic = factor_vals.corr(return_vals)
            spearman_ic = factor_vals.corr(return_vals, method='spearman')

            print(f"{date} -> {next_date}:")
            print(f"  样本数: {len(common_stocks)}")
            print(f"  Pearson IC: {pearson_ic:.4f}")
            print(f"  Spearman IC: {spearman_ic:.4f}")

            # 检查极端情况
            if abs(spearman_ic) > 0.5:
                print(f"  ⚠️  IC过高，检查数据:")
                print(f"    因子值范围: {factor_vals.min():.4f} - {factor_vals.max():.4f}")
                print(f"    收益范围: {return_vals.min():.4f} - {return_vals.max():.4f}")

                # 检查是否有异常股票
                extreme_returns = return_vals[abs(return_vals) > 0.1]
                if len(extreme_returns) > 0:
                    print(f"    极端收益股票: {len(extreme_returns)}只")
                    print(f"    极端收益值: {extreme_returns.values}")



#快速测试ic (未经过中性化
def analyze_why_better_performance(factor_manager):
    """分析为什么沪深300表现更好"""

    # 对比沪深300 vs 全市场
    start_date, end_date = '20241215', '20250624'

    # 沪深300股票池
    hs300_pool =factor_manager.data_manager.stock_pools_dict['fast']

    # 获取因子数据
    vol_factor = factor_manager.get_raw_factor('volatility_120d')
    pct_chg = factor_manager.get_raw_factor('pct_chg')

    # 选择测试日期
    test_date = vol_factor.index[-10]
    next_date = vol_factor.index[-9]

    print("沪深300 vs 全市场对比:")
    print("=" * 50)

    # 全市场数据
    factor_all = vol_factor.loc[test_date].dropna()
    return_all = pct_chg.loc[next_date].dropna()
    common_all = factor_all.index.intersection(return_all.index)

    # 沪深300数据
    hs300_stocks = list(hs300_pool.columns)
    factor_hs300 = factor_all[factor_all.index.intersection(hs300_stocks)]
    return_hs300 = return_all[return_all.index.intersection(hs300_stocks)]
    common_hs300 = factor_hs300.index.intersection(return_hs300.index)

    print(f"全市场股票数: {len(common_all)}")
    print(f"沪深300股票数: {len(common_hs300)}")

    # 计算IC对比
    if len(common_all) > 100:
        ic_all = factor_all[common_all].corr(return_all[common_all], method='spearman')
        print(f"全市场Spearman IC: {ic_all:.4f}")

    if len(common_hs300) > 50:
        ic_hs300 = factor_hs300[common_hs300].corr(return_hs300[common_hs300], method='spearman')
        print(f"沪深300 Spearman IC: {ic_hs300:.4f}")

    # 分析波动率分布差异
    print(f"\n波动率分布对比:")
    print(f"全市场波动率: 均值={factor_all[common_all].mean():.4f}, 标准差={factor_all[common_all].std():.4f}")
    print(f"沪深300波动率: 均值={factor_hs300[common_hs300].mean():.4f}, 标准差={factor_hs300[common_hs300].std():.4f}")

    # 分析收益率分布差异
    print(f"\n收益率分布对比:")
    print(f"全市场收益: 均值={return_all[common_all].mean():.6f}, 标准差={return_all[common_all].std():.4f}")
    print(f"沪深300收益: 均值={return_hs300[common_hs300].mean():.6f}, 标准差={return_hs300[common_hs300].std():.4f}")



def verify_data(factor_manager):
    stock_codes=factor_manager.data_manager.stock_pools_dict['fast'].columns
    _120d = factor_manager.get_raw_factor('bm_ratio')
    volatility_120d = factor_manager.get_raw_factor('volatility_120d')
    earnings_stability = factor_manager.get_raw_factor('earnings_stability')
    operating_accruals = factor_manager.get_raw_factor('operating_accruals')
    print(1)
    pass


def main():
    # 2. 初始化数据仓库
    logger.info("1. 加载底层原始因子raw_dict数据...")

    # 【修复】使用绝对路径避免工作目录问题
    current_dir = Path(__file__).parent
    config_path = current_dir / 'factory' / 'config.yaml'
    experiments_path = current_dir / 'factory' / 'experiments.yaml'

    # 验证配置文件存在
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not experiments_path.exists():
        raise FileNotFoundError(f"实验配置文件不存在: {experiments_path}")

    # 【修复】添加异常处理
    try:
        data_manager = DataManager(config_path=str(config_path), experiments_config_path=str(experiments_path))
        data_manager.prepare_basic_data()
        factor_manager = FactorManager(data_manager)
    except FileNotFoundError as e:
        logger.error(f"数据文件缺失: {e}")
        raise
    except Exception as e:
        logger.error(f"数据管理器初始化失败: {e}")
        raise

    # 【修复】使用正式的缓存清理方法
    logger.info("!!! 正在执行“硬重启”：强制清理因子缓存...")
    factor_manager.clear_cache()
    # analyze_why_better_performance(factor_manager)
    # verify_data(factor_manager)

    # 测试时间聚合效果
    # mono_aggregated = simulate_your_aggregation_method(factor_manager)
    # comprehensive_factor_test(factor_manager)
    # verify_pct_chg(factor_manager) #通过

    # check_look_ahead_bias(factor_manager)#通过
    # debug_ic_calculation_detailed(factor_manager)#表现正常 有正有负!
    # verify_volatility_calculation(factor_manager)通过
    # verify_adj_factor_timing(factor_manager)通过
    # verify_adj_factor(factor_manager, ts_code_to_check='600519.SH', ex_date_to_check='2024-06-19')通过

    # 3. 创建示例因子
    logger.info("3. 创建目标学术因子...")
    factor_analyzer = FactorAnalyzer(factor_manager=factor_manager )

    # 6. 批量测试因子
    logger.info("5. 批量测试因子...")
    #读取 实验文件，获取需要做的实验
    experiments_df = data_manager.get_experiments_df()
    logger.info(f"发现 {len(experiments_df)} 个实验配置")

    # 批量测试
    results = []
    for index, config in experiments_df.iterrows():
        try:
            factor_name = config['factor_name']
            stock_pool_name = config['stock_pool_name']

            logger.info(f"开始测试因子: {factor_name} (股票池: {stock_pool_name})")

            # 执行测试
            test_result = factor_analyzer.test_factor_entity_service_route(
                factor_name=factor_name,
                stock_pool_index_name=stock_pool_name,
            )

            results.append({
                'factor_name': factor_name,
                'stock_pool_name': stock_pool_name,
                'result': test_result
            })

        except Exception as e:
            raise ValueError(f"✗ 因子 {factor_name} 测试失败: {e}") from e

    log_success(f"✓ 批量测试完成，成功测试 {len(results)} 个因子")
    return results
    # batch_results = factor_analyzer.batch_test_factors(
    #     target_factors_dict=target_factors_dict
    # )

    #
    # # 11. 多因子优化
    # print("\n10. 多因子优化...")
    # try:
    #     optimized_factor = factory.optimize_factors(
    #         factor_data_dict=target_factors_dict,
    #         intra_method='ic_weighted',
    #         cross_method='max_diversification'
    #     )
    #     print(f"✓ 多因子优化完成，生成最终因子: {optimized_factor.shape}")
    # except Exception as e:
    #     print(f"✗ 多因子优化失败: {e}")
    #
    # # 12. 导出结果
    # print("\n11. 导出结果...")
    # try:
    #     exported_files = factory.export_results()
    #     print("✓ 结果导出完成:")
    #     for file_type, file_path in exported_files.items():
    #         print(f"  {file_type}: {file_path}")
    # except Exception as e:
    #     print(f"✗ 结果导出失败: {e}")

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)


def verify_adj_factor(factor_manager, ts_code_to_check: str, ex_date_to_check: str):
    """
    对 adj_factor 的计算结果进行“总回报恒等式”检验。
    """
    logger.info(f"--- 开始对 {ts_code_to_check} 在 {ex_date_to_check} 的复权因子进行检验 ---")

    ex_date = pd.to_datetime(ex_date_to_check)
    prev_date = ex_date - pd.Timedelta(days=1)  # 简化处理，真实情况应取前一交易日

    # --- 1. 获取所有需要的数据 ---
    close_raw = factor_manager.get_raw_factor('close_raw')
    dividend_events = load_dividend_events_long()

    # --- 2. 方法A: 通过你的新引擎计算出的pct_chg ---
    # 假设你的pct_chg计算函数已经更新
    call_pro_tushare_api('daily',ts_code='00001.SZ,00002.SZ',start_date='20240122',end_date='20240329')
    pct_chg_from_engine = factor_manager.get_raw_factor('pct_chg')
    engine_return = pct_chg_from_engine.at[ex_date, ts_code_to_check]

    # --- 3. 方法B: 根据第一性原理手动计算 ---
    # a) 获取价格
    price_t = close_raw.at[ex_date, ts_code_to_check]
    price_t_minus_1 = close_raw.at[prev_date, ts_code_to_check]

    # b) 获取当天的分红事件
    event = dividend_events[
        (dividend_events['ts_code'] == ts_code_to_check) &
        (pd.to_datetime(dividend_events['ex_date']) == ex_date)
        ].iloc[0]
    cash_div = event.get('cash_div_tax', 0)
    stk_div = event.get('stk_div', 0)

    # c) 计算理论总回报率
    # 注意：这里的公式需要同时处理分红和送股
    theoretical_return = (price_t * (1 + stk_div) + cash_div) / price_t_minus_1 - 1

    # --- 4. 对比结果 ---
    print(f"方法A (引擎计算) 的收益率: {engine_return:.6f}")
    print(f"方法B (理论计算) 的收益率: {theoretical_return:.6f}")

    if np.isclose(engine_return, theoretical_return):
        print("✓【检验通过】: 引擎计算的总回报与理论值一致！")
    else:
        print("✗【检验失败】: 引擎计算的总回报与理论值不一致！请检查 adj_factor 计算逻辑。")


def verify_adj_factor_timing(factor_manager):
    """
    【第三重检验】检验复权因子的调整日期是否精确且完整。
    """
    logger.info("\n" + "=" * 20 + " 【第三重检验】开始：事件日精确性与完整性 " + "=" * 20)

    # --- 1. 获取原材料 ---
    adj_factor_df = factor_manager.get_raw_factor('adj_factor')
    dividend_events = load_dividend_events_long()

    # --- 2. 找出所有“因子实际发生变化”的事件点 ---
    logger.info("  > 正在扫描因子实际发生变化的所有 (日期, 股票) 点...")

    # a) 计算因子值的日度变化率，不为0或NaN的地方就是发生了变化
    factor_changes = adj_factor_df.pct_change()

    # b) 使用 .stack() 将宽表转换为长序列，并移除没有变化的点
    #    stack() 会自动移除NaN，非常方便
    factor_changes_series = factor_changes.stack()
    actual_change_points = factor_changes_series[factor_changes_series != 0].index

    # 将结果转换为一个 (日期, 股票代码) 的集合，便于比较
    actual_change_set = set(actual_change_points.to_list())
    logger.info(f"  > 发现 {len(actual_change_set)} 个实际发生调整的事件点。")

    # --- 3. 找出所有“理论上应该发生变化”的事件点 ---
    logger.info("  > 正在从分红送股数据中，构建理论事件点集合...")

    # a) 筛选出有效的事件
    #    送股和派息都为0的预案，不是有效的除权事件
    valid_events = dividend_events[
        (dividend_events['stk_div'] > 0) | (dividend_events['cash_div_tax'] > 0)
        ].copy()
    valid_events['ex_date'] = pd.to_datetime(valid_events['ex_date'])

    # b) 构造 (日期, 股票代码) 的元组列表
    theoretical_change_tuples = [
        (row['ex_date'], row['ts_code']) for _, row in valid_events.iterrows()
    ]

    # c) 转换为集合
    theoretical_change_set = set(theoretical_change_tuples)
    logger.info(f"  > 发现 {len(theoretical_change_set)} 个理论上应发生调整的事件点。")

    # --- 4. 对比两个集合，给出最终诊断 ---
    logger.info("\n--- 最终检验结果 ---")

    # a) 检查是否有“理论上应该发生，但实际未发生”的事件 (漏报)
    missed_events = theoretical_change_set - actual_change_set
    if not missed_events:
        logger.info("✓【检验通过】完整性：所有分红送股事件，都在复权因子中得到了正确体现。")
    else:
        logger.error(f"✗【检验失败】完整性：发现了 {len(missed_events)} 个被遗漏的事件！")
        logger.error(f"  > 遗漏样本: {list(missed_events)[:5]}")

    # b) 检查是否有“实际发生了，但理论上不该发生”的事件 (误报)
    spurious_events = actual_change_set - theoretical_change_set
    if not spurious_events:
        logger.info("✓【检验通过】精确性：复权因子只在正确的除权除息日发生了变化。")
    else:
        logger.error(f"✗【检验失败】精确性：发现了 {len(spurious_events)} 个错误的调整事件！")
        logger.error(f"  > 错误样本: {list(spurious_events)[:5]}")

    if not missed_events and not spurious_events:
        print("\n" + "✓" * 20 + " 【最终结论】复权因子引擎通过了全部时点检验，堪称完美！ " + "✓" * 20)
    else:
        print("\n" + "✗" * 20 + " 【最终结论】复权因子引擎存在时点偏差，请根据上面的提示进行排查。 " + "✗" * 20)


# --- 使用示例 ---
# factor_manager = FactorManager(...)

if __name__ == "__main__":
    main()
