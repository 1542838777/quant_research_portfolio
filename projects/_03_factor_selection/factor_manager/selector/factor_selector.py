import json
from pathlib import Path
from typing import Union, Dict, Any

import pandas as pd
from plotly.data import stocks

from projects._03_factor_selection.visualization_manager import VisualizationManager
from quant_lib import logger
from quant_lib.config.symbols_constants import WARNING
from quant_lib.utils.json_utils import load_json_with_numpy


class FactorSelector:
    def __init__(self):
        self.visualizationManager = VisualizationManager(
        )

    # ==============================================================================
    #                      主分析流程
    # ==============================================================================
    def run_factor_analysis(self, TARGET_STOCK_POOL: str = '000300.SH', TARGET_PERIOD: str = '21d'):
        # --- 0. 初始化 ---
        # (这里省略了你的初始化代码，假设 factor_analyzer 已经准备好)
        # --- 1. 定义分析目标 ---
        RESULTS_PATH = 'D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\workspace\\result'

        # --- 2. 构建特定周期和股票池的排行榜 ---
        leaderboard_df = self.build_leaderboard(
            results_path=RESULTS_PATH,
            stock_pool=TARGET_STOCK_POOL,
            target_period=TARGET_PERIOD
        )
        print("\n--- 因子排行榜 (部分结果) ---")
        print(leaderboard_df.head())

        # --- 3. 从排行榜中，筛选出最终的优选因子 ---
        top_factors_df = self.get_top_factors(
            leaderboard_df=leaderboard_df,
            results_path=RESULTS_PATH,
            quality_score_threshold=0.0,
            top_n_final=5,
            correlation_threshold=0.5
        )

        print("\n--- 最终入选的顶级因子详情 ---")
        print(top_factors_df)

        # --- 4. 只为最终筛选出的优选因子，生成精美的报告 ---
        logger.info("\n--- 开始为顶级因子生成详细报告 ---")
        for factor_name in top_factors_df['factor_name']:
            print(f"正在为因子 '{factor_name}' 生成报告...")
            # 4.1 生成主报告 (3x2 统一评估报告)
            # 绘图函数现在需要从硬盘加载数据，我们只需告知关键信息
            self.visualizationManager.plot_unified_factor_report(
                backtest_base_on_index=TARGET_STOCK_POOL,
                factor_name=factor_name,
                results_path=RESULTS_PATH,  # <--- 传入成果库的根路径
                # 你可以决定主报告默认使用C2C还是O2C的结果
                default_config='o2c'
            )

            # 4.2 生成稳健性对比报告 (2x2 C2C vs O2C)
            self.visualizationManager.plot_robustness_report(
                backtest_base_on_index=TARGET_STOCK_POOL,
                factor_name=factor_name,
                results_path=RESULTS_PATH
            )
            #
            # # 4.1 调用新的IC报告函数
            # self.visualizationManager.plot_ic_report(
            #     backtest_base_on_index=TARGET_STOCK_POOL,
            #     factor_name=factor_name,
            #     results_path=RESULTS_PATH,
            #     default_config='c2c',  # 或 'c2c'
            # )
            #
            # # 4.2 调用新的分层净值报告函数
            # self.visualizationManager.plot_quantile_report(
            #     backtest_base_on_index=TARGET_STOCK_POOL,
            #     factor_name=factor_name,
            #     results_path=RESULTS_PATH,
            #     default_config='c2c',  # 或 'c2c'
            #     target_period='21d')  # 指定要展示哪个周期的分层结果
            # 4.2 调用新的分层净值报告函数
            self.visualizationManager.plot_ic_quantile_panel(
                backtest_base_on_index=TARGET_STOCK_POOL,
                factor_name=factor_name,
                results_path=RESULTS_PATH,
                default_config='c2c'   # 或 'c2c'
            )
            # 调用新的归因分析面板函数
            self.visualizationManager.plot_attribution_panel(
                backtest_base_on_index=TARGET_STOCK_POOL,
                factor_name=factor_name,
                results_path=RESULTS_PATH,
                default_config='o2c'
            )

    def build_leaderboard(self,
                          results_path: str,
                          stock_pool: str,
                          target_period: str,
                          run_version: str = 'latest'
                          ) -> pd.DataFrame:
        """
        【V2.0-版本化】从硬盘扫描所有因子的测试结果，并构建对比排行榜。
        能够自动定位最新的回测版本，或加载指定版本。
        """
        logger.info(f"正在为股票池 [{stock_pool}] 周期 [{target_period}] 构建排行榜 (版本: {run_version})...")
        leaderboard_data = []
        base_path = Path(results_path) / stock_pool

        # --- 内部辅助函数，用于安全地查找和加载指定版本的stats文件 ---
        def _find_and_load_stats(factor_dir: Path, config_name: str, version: str) -> Dict[str, Any] | None:
            config_path = factor_dir / config_name
            if not config_path.is_dir():
                return None

            version_dirs = [d for d in config_path.iterdir() if d.is_dir()]
            if not version_dirs:
                return None

            target_version_path = None
            if version == 'latest':
                target_version_path = sorted(version_dirs)[-1]  # 按字母顺序排序，最新的日期版本会在最后
            else:
                path_to_find = config_path / version
                if path_to_find in version_dirs:
                    target_version_path = path_to_find

            if not target_version_path:
                return None

            summary_file = target_version_path / 'summary_stats.json'
            if summary_file.exists():
                return load_json_with_numpy(summary_file)
            return None

        # --- 辅助函数结束 ---

        # 1. 扫描所有因子目录
        for factor_dir in base_path.iterdir():
            if not factor_dir.is_dir():
                continue
            factor_name = factor_dir.name

            # 2. 【核心改造】使用辅助函数，查找并加载C2C和O2C的统计结果
            stats_c2c = _find_and_load_stats(factor_dir, 'c2c', run_version)
            stats_o2c = _find_and_load_stats(factor_dir, 'o2c', run_version)

            # 3. 如果任何一个配置的结果缺失，则跳过这个因子，保证数据的完整性
            if not stats_c2c or not stats_o2c:
                log_warning(f"因子 {factor_name} 的结果不完整 (缺少C2C或O2C的 '{run_version}' 版本)，已跳过。")
                continue

            # 4. 从加载的字典中，提取指定周期的核心指标
            # 使用 .get(key, {}) 来安全地处理 period 可能不存在的情况
            c2c_q = stats_c2c.get('quantile_backtest_processed', {}).get(target_period, {})
            o2c_q = stats_o2c.get('quantile_backtest_processed', {}).get(target_period, {})

            c2c_ic = stats_c2c.get('ic_analysis_processed', {}).get(target_period, {})
            o2c_ic = stats_o2c.get('ic_analysis_processed', {}).get(target_period, {})

            c2c_fm = stats_c2c.get('fama_macbeth', {}).get(target_period, {})
            o2c_fm = stats_o2c.get('fama_macbeth', {}).get(target_period, {})
            if (len(o2c_fm) ==0) and (len(c2c_ic) ==0) or(len(c2c_q) ==0) :
                continue
            # 5. 合并为一行宽表数据
            row = {
                'factor_name': factor_name,
                'period': target_period,
                'stock_pool': stock_pool,
                # 从字典中安全地 .get() 每个指标
                'tmb_sharpe_c2c': c2c_q.get('tmb_sharpe'),
                'tmb_sharpe_o2c': o2c_q.get('tmb_sharpe'),
                'tmb_max_drawdown_o2c': o2c_q.get('tmb_max_drawdown'),
                'monotonicity_spearman_o2c': o2c_q.get('monotonicity_spearman'),

                'ic_mean_c2c': c2c_ic.get('ic_mean'),
                'ic_mean_o2c': o2c_ic.get('ic_mean'),
                'ic_ir_c2c': c2c_ic.get('ic_ir'),
                'ic_ir_o2c': o2c_ic.get('ic_ir'),

                'fm_t_statistic_c2c': c2c_fm.get('t_statistic'),
                'fm_t_statistic_o2c': o2c_fm.get('t_statistic'),
            }
            leaderboard_data.append(row)

        if not leaderboard_data:
            raise ValueError(
                f"在路径 {base_path} 下，针对周期 {target_period} 和版本 '{run_version}'，没有找到任何完整的因子测试结果。")

        leaderboard_df = pd.DataFrame(leaderboard_data).set_index('factor_name', drop=False)

        # 6. 应用V5评分函数
        leaderboard_df['score'] = leaderboard_df.apply(calculate_factor_score_ultimate, axis=1)

        return leaderboard_df.sort_values(by='score', ascending=False)

    ##
    # 第一阶段：质量打分 - 使用我提供的“专业级因子评分体系”对每个因子进行绝对打分。
    #
    # 第二阶段：降维去重 - 将第一阶段中得分超过某个阈值（比如40分）的因子作为“高质量候选池”，然后用你代码里的相关性贪心算法，从这个池子里选出最终的因子组合。#
    def get_top_factors(self,
                        leaderboard_df: pd.DataFrame,
                        results_path: str,
                        run_version: str = 'latest',  # <-- 增加 run_version 参数
                        quality_score_threshold: float = 0.0,#todo 测试模式
                        top_n_final: int = 10,
                        correlation_threshold: float = 1) -> pd.DataFrame: #todo 测试模式
        """
        【重构版】从排行榜中，筛选出最终的、多样化的顶级因子。
        """
        period = leaderboard_df['period'].iloc[0]
        stock_pool = leaderboard_df['stock_pool'].iloc[0]
        logger.info(f"--- 开始筛选周期为 {period} 的顶级因子 ---")

        # --- 1. 质量筛选 ---
        candidate_df = leaderboard_df[leaderboard_df['score'] >= quality_score_threshold]
        candidate_factors_list = candidate_df['factor_name'].tolist()
        if not candidate_factors_list:
            log_warning(f"没有因子的综合得分超过 {quality_score_threshold}。")
            return pd.DataFrame()
        logger.info(f"通过专业打分，筛选出 {len(candidate_factors_list)} 个高质量候选因子。")

        # --- 2. 多样化筛选 (去相关性) ---
        # 我们使用F-M因子收益率序列来计算相关性，因为它更纯净
        factor_returns_matrix =load_fm_returns_matrix(leaderboard_df=candidate_df, results_path=results_path,stock_pool=stock_pool, period = period, config = 'o2c', run_version = run_version)
        correlation_matrix = factor_returns_matrix.corr()

        final_selected_factors = []
        # 贪心算法：从得分最高的因子开始
        for candidate in candidate_factors_list:
            if len(final_selected_factors) >= top_n_final:
                break

            if not final_selected_factors:  # 第一个直接入选
                final_selected_factors.append(candidate)
                continue

            # 计算与已选因子的最大相关性
            correlations_with_selected = correlation_matrix.loc[candidate, final_selected_factors].abs()

            if correlations_with_selected.max() < correlation_threshold:
                final_selected_factors.append(candidate)

        logger.info(f"--- 筛选完成 ---")
        logger.info(f"最终选出 {len(final_selected_factors)} 个多样化顶级因子：{final_selected_factors}")

        return leaderboard_df[leaderboard_df['factor_name'].isin(final_selected_factors)]


##

# 全局排序 对所有因子进行综合排序，选出一个比如Top 50的大名单。 ，保证所有因子的个体质量都是顶尖的。
#
# 在“Top 50的大名单”   中进行分类和相关性分析:
#
# --对这Top 50的因子进行分类（价值、动量等）。
#
# --计算这50个因子之间的相关性矩阵。
#
# --从这50个最优秀的因子中，挑出比如10个，要求这10个因子彼此不相关，并且尽可能覆盖不同的风格类别。
#
# 例如，发现在动量类里，排名前5的因子相关性都高达0.8，那么你只保留其中综合排名最高的那一个。然后你再去价值类、质量类里做同样的操作。
#
# 这个混合策略，保证没有错过任何一个在全市场范围内表现优异的因子（质量），又通过后续的步骤保证了最终入选因子的多样性。稳健多因子模型。#


#
# def calculate_factor_score(purify_summary: Union[pd.Series, dict]) -> float:
#     """
#     【V3版】根据“专业级因子评分体系”为单个因子计算总分。
#     此版本能自动判断因子方向（正向或反向），并应用相应的评分逻辑。
#
#     输入: 一个 Series，包含了单个因子的所有指标。
#     输出: 该因子的总得分。
#     """
#     score = 0
#
#     # --- 指标提取 ---
#     ic_mean = purify_summary.get('ic_mean', 0)
#     ic_ir = purify_summary.get('ic_ir', 0)
#     fm_t_stat = purify_summary.get('fm_t_statistic', 0)
#     tmb_sharpe = purify_summary.get('tmb_sharpe', 0)
#     tmb_max_drawdown = purify_summary.get('tmb_max_drawdown', 0)
#     monotonicity_spearman = purify_summary.get('monotonicity_spearman')
#
#     # --- 关键升级：自动判断因子方向 ---
#     # 优先使用IC均值的符号判断。如果IC均值接近0，则使用t统计量的符号辅助判断。
#     factor_direction = 1
#     if ic_mean < -1e-4:  # 使用一个小的负数阈值避免噪音
#         factor_direction = -1
#     elif abs(ic_mean) <= 1e-4 and fm_t_stat < 0:
#         factor_direction = -1
#
#     # --- 评分开始 ---
#     # 1. 预测能力分 (满分20) - 根据因子方向调整
#     adj_ic_mean = ic_mean * factor_direction
#     if adj_ic_mean > 0.05:
#         score += 20
#     elif 0.03 < adj_ic_mean <= 0.05:
#         score += 15
#     elif 0.01 < adj_ic_mean <= 0.03:
#         score += 10
#     elif 0 < adj_ic_mean <= 0.01:
#         score += 5
#
#     # 2. 稳定性分 (满分20) - ICIR通常看绝对值或调整后方向，这里以调整后为正评分
#     adj_ic_ir = ic_ir * factor_direction
#     if adj_ic_ir > 0.5:
#         score += 20
#     elif 0.3 < adj_ic_ir <= 0.5:
#         score += 15
#     elif 0.1 < adj_ic_ir <= 0.3:
#         score += 10
#     elif 0 < adj_ic_ir <= 0.1:
#         score += 5
#
#     # 3. 统计显著性分 (满分30) - t值看绝对值，不受方向影响
#     t_abs = abs(fm_t_stat)
#     if t_abs > 3.0:
#         score += 30
#     elif 2.0 < t_abs <= 3.0:
#         score += 25
#     elif 1.5 < t_abs <= 2.0:
#         score += 15
#
#     # 4. 策略表现分 (满分20) - 根据因子方向调整
#     adj_tmb_sharpe = tmb_sharpe * factor_direction
#     perf_score = 0
#     if adj_tmb_sharpe > 1.0:
#         perf_score = 20
#     elif 0.5 < adj_tmb_sharpe <= 1.0:
#         perf_score = 15
#     elif 0.2 < adj_tmb_sharpe <= 0.5:
#         perf_score = 10
#     elif 0 < adj_tmb_sharpe <= 0.2:
#         perf_score = 5
#
#     # 风控扣分项 - 回撤总是负的，直接比较
#     if tmb_max_drawdown < -0.5:
#         perf_score -= 5
#
#     score += max(0, perf_score)
#
#     # 5. 单调性分 (满分10) - 根据Spearman相关系数进行分级评分
#     if monotonicity_spearman is not None and pd.notna(monotonicity_spearman):
#         # 我们关心的是单调性的强度，所以取绝对值
#         mono_abs = abs(monotonicity_spearman)
#
#         # 只有当单调性达到中等强度(0.5)以上时，才开始加分
#         if mono_abs >= 0.5:
#             # 直接将Spearman相关系数乘以总分10，得到一个连续的分数
#             # 这样 mono_abs=0.5 得5分, mono_abs=1.0 得10分
#             # 这是一个简单、直观且精细的评分方式
#             score += mono_abs * 10
#         # 低于0.5，我们认为单调性不显著，不额外加分
#
#     return score

def calculate_factor_score_ultimate(summary_row: Union[pd.Series, dict]) -> float:
    """
    【基础分 (Base Score): 完全基于Processed + O2C这两个最严格、最纯净的维度进行计算。它衡量的是一个因子在“理想状态”下的绝对实力。

稳健性惩罚 (Robustness Penalty): 检验这个“理想实力”能否经受住真实交易成本的考验。如果一个因子对隔夜跳空收益过于敏感，它会在这里被扣分。

Alpha纯度惩罚 (Purity Penalty): 检验这个“理想实力”的来源。如果一个因子的亮眼表现，仅仅是因为它搭了市值、行业风格的便车，它会在这里被“打回原形”，受到重罚。
    """
    # --- 1. 指标提取 (安全地获取所有可能用到的指标) ---
    # 纯净因子(Processed)的表现
    ic_ir_proc_c2c = summary_row.get('ic_ir_processed_c2c', 0.0)
    ic_ir_proc_o2c = summary_row.get('ic_ir_processed_o2c', 0.0)
    ic_mean_proc_o2c = summary_row.get('ic_mean_processed_o2c', 0.0)
    tmb_sharpe_proc_c2c = summary_row.get('tmb_sharpe_processed_c2c', 0.0)
    tmb_sharpe_proc_o2c = summary_row.get('tmb_sharpe_processed_o2c', 0.0)
    fm_t_stat_proc_o2c = summary_row.get('fm_t_statistic_processed_o2c', 0.0)
    tmb_max_drawdown_proc_o2c = summary_row.get('tmb_max_drawdown_processed_o2c', 0.0)
    monotonicity_spearman_proc_o2c = summary_row.get('monotonicity_spearman_processed_o2c')

    # 原始因子(Raw)的表现 (仅用于计算纯度惩罚)
    tmb_sharpe_raw_c2c = summary_row.get('tmb_sharpe_raw_c2c', 0.0)
    tmb_sharpe_raw_o2c = summary_row.get('tmb_sharpe_raw_o2c', 0.0)

    # --- 2. 自动判断因子方向 (基于最纯净、最稳健的 O2C + Processed 指标) ---
    factor_direction = 1
    if ic_mean_proc_o2c < -1e-4:
        factor_direction = -1
    elif abs(ic_mean_proc_o2c) <= 1e-4 and fm_t_stat_proc_o2c < 0:
        factor_direction = -1

    # --- 3. 计算基础分 (完全基于 Processed + O2C 指标) ---
    base_score = 0
    # (此处的评分标准完整继承自你最初的V3版，确保了评价的延续性)

    # 3.1 预测能力分 (满分20)
    adj_ic_mean = ic_mean_proc_o2c * factor_direction
    if adj_ic_mean > 0.05:
        base_score += 20
    elif adj_ic_mean > 0.03:
        base_score += 15
    elif adj_ic_mean > 0.01:
        base_score += 10

    # 3.2 稳定性分 (满分20)
    adj_ic_ir = ic_ir_proc_o2c * factor_direction
    if adj_ic_ir > 0.5:
        base_score += 20
    elif adj_ic_ir > 0.3:
        base_score += 15
    elif adj_ic_ir > 0.1:
        base_score += 10

    # 3.3 统计显著性分 (满分30)
    t_abs = abs(fm_t_stat_proc_o2c)
    if t_abs > 3.0:
        base_score += 30
    elif t_abs > 2.0:
        base_score += 25
    elif t_abs > 1.5:
        base_score += 15

    # 3.4 策略表现分 (满分20)
    adj_tmb_sharpe = tmb_sharpe_proc_o2c * factor_direction
    perf_score = 0
    if adj_tmb_sharpe > 1.0:
        perf_score = 20
    elif adj_tmb_sharpe > 0.5:
        perf_score = 15
    elif adj_tmb_sharpe > 0.2:
        perf_score = 10

    # 风控扣分项
    if tmb_max_drawdown_proc_o2c < -0.5: perf_score -= 5
    base_score += max(0, perf_score)

    # 3.5 单调性分 (满分10)
    if pd.notna(monotonicity_spearman_proc_o2c) and abs(monotonicity_spearman_proc_o2c) >= 0.5:
        base_score += abs(monotonicity_spearman_proc_o2c) * 10

    # --- 4. 计算惩罚分 ---
    # 4.1 【稳健性惩罚】C2C vs O2C (Alpha质量)
    robustness_penalty = 0
    if tmb_sharpe_proc_c2c * factor_direction > 0.3:
        denominator = max(abs(tmb_sharpe_proc_c2c), 1e-6)
        decay_ratio = (tmb_sharpe_proc_c2c - tmb_sharpe_proc_o2c) / denominator
        if decay_ratio > 0.3: robustness_penalty += 10
        if decay_ratio > 0.5: robustness_penalty += 15

    # 4.2 【Alpha纯度惩罚】Raw vs Processed (Alpha来源)
    purity_penalty = 0
    # 我们用更严格的O2C结果来对比Raw和Processed
    if tmb_sharpe_raw_o2c * factor_direction > 0.3:
        denominator = max(abs(tmb_sharpe_raw_o2c), 1e-6)
        decay_ratio = (tmb_sharpe_raw_o2c - tmb_sharpe_proc_o2c) / denominator
        if decay_ratio > 0.5:  # 如果纯净化的过程让夏普衰减了50%以上
            purity_penalty += 15  # 重罚，说明因子一半以上的收益来自风格暴露
        if decay_ratio > 0.8:  # 如果衰减超过80%
            purity_penalty += 25  # 极刑，这基本是个伪因子

    # --- 5. 计算最终得分 ---
    final_score = base_score - robustness_penalty - purity_penalty

    # logger.info(f"因子 {summary_row.get('factor_name', '')}: Base({base_score:.1f}) - Robustness({robustness_penalty:.1f}) - Purity({purity_penalty:.1f}) = Final({max(0, final_score):.1f})")

    return max(0, final_score)

def load_fm_returns_matrix(
        leaderboard_df: pd.DataFrame,
        results_path: str,
        stock_pool: str,
        period: str,
        config: str = 'o2c', #默认优用o2c
        run_version: str = 'latest'  # <-- 新增参数
) -> pd.DataFrame:
    """
    【V2.0-版本化】辅助函数：加载多个因子的F-M收益序列，用于计算相关性。
    能够自动定位最新的回测版本，或加载指定版本。
    """
    all_returns = {}

    # 遍历排行榜中的每一个候选因子
    for factor_name in leaderboard_df['factor_name']:

        # --- 【核心修正】在这里找到正确的“版本”文件夹 ---
        base_path = Path(results_path) / stock_pool / factor_name / config
        if not base_path.is_dir():
            raise ValueError(f"{factor_name}数据文件{base_path}缺失")

        version_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        if not version_dirs:
            raise ValueError(f"{factor_name}文件路径{base_path}下缺失文件")

        target_version_path = None
        if run_version == 'latest':
            target_version_path = sorted(version_dirs)[-1]
        else:
            path_to_find = base_path / run_version
            if path_to_find in version_dirs:
                target_version_path = path_to_find

        if not target_version_path:
            raise ValueError(f"因子 {factor_name} 未找到版本 '{run_version}'，跳过加载。")
        # --- 版本定位结束 ---

        # 使用定位到的正确路径加载数据
        file_path = target_version_path / f"fm_returns_series_{period}.parquet"

        if file_path.exists():
            # 读取 Series 并重命名，以因子名为列名
            return_series = pd.read_parquet(file_path).squeeze()
            all_returns[factor_name] = return_series
        else:
            raise ValueError(f"未找到文件: {file_path}")

    if not all_returns:
        logger.error(f"在版本 '{run_version}' 下，未能为任何因子加载F-M收益序列。")
        return pd.DataFrame()

    return pd.DataFrame(all_returns)


if __name__ == '__main__':
    fase = FactorSelector()
    fase.run_factor_analysis(TARGET_STOCK_POOL='000906.SH.SH', TARGET_PERIOD='5d')
    # fase.run_factor_analysis(TARGET_STOCK_POOL='000852.SH', TARGET_PERIOD='21d')
