"""
重构后的策略工厂使用示例

本脚本演示如何使用重构后的策略工厂进行完整的因子研究流程：
1. 单因子测试
2. 类别内优化
3. 类别间优化
4. 可视化和报告生成

"""
import traceback

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

def main():
    # 2. 初始化数据仓库
    logger.info("1. 加载底层原始因子raw_dict数据...")
    data_manager  = DataManager(config_path='factory/config.yaml',experiments_config_path='factory/experiments.yaml')
    data_manager.prepare_basic_data()

    factor_manager  = FactorManager(data_manager)
    # verify_adj_factor_timing(factor_manager)
    # verify_adj_factor(factor_manager, ts_code_to_check='600519.SH', ex_date_to_check='2024-06-19')

    # 3. 创建示例因子
    logger.info("3. 创建目标学术因子...")
    factor_analyzer = FactorAnalyzer(factor_manager=factor_manager )

    # 6. 批量测试因子
    logger.info("5. 批量测试因子...")
    #读取 实验文件，获取需要做的实验
    experiments = data_manager.get_experiments_df()
    # 批量测试
    results = []
    for index, config in experiments.iterrows():
        try:
            # 执行测试
            results.append({index: (factor_analyzer.test_factor_entity_service_route(
                factor_name=config[0],
                stock_pool_index_name=config[1],
            ))})
        except Exception as e:
            raise ValueError(f"✗ 因子{index}测试失败: {e}") from e

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
