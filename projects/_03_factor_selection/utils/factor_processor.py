"""
因子预处理流水线 - 单因子测试终极作战手册
第三阶段：因子预处理

实现完整的因子预处理流水线：
1. 去极值 (Winsorization)
2. 中性化 (Neutralization) 
3. 标准化 (Standardization)
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Optional, Tuple, Any
import warnings
from sklearn.linear_model import LinearRegression
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.config.logger_config import setup_logger

warnings.filterwarnings('ignore')

# 配置日志
logger = setup_logger(__name__)


class FactorProcessor:
    """
    因子预处理器 - 专业级因子预处理流水线
    
    按照华泰证券标准实现：
    1. 去极值：中位数绝对偏差法(MAD) / 分位数法
    2. 中性化：行业中性化 + 市值中性化
    3. 标准化：Z-Score标准化 / 排序标准化
    """

    def __init__(self, config: Dict):
        """
        初始化因子预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})

    # ok
    def process_factor(self,
                       target_factor_df: pd.DataFrame,
                       target_factor_name: str,
                       auxiliary_dfs,
                       neutral_dfs,
                       factor_school: str
                       ):
        """
        完整的因子预处理流水线
        
        Args:
            factor_data: 原始因子数据
            auxiliary_df_dict: 辅助数据（市值、行业等）
            
        Returns:
            预处理后的因子数据
        """
        processed_target_factor_df = target_factor_df.copy()
        auxiliary_dfs = auxiliary_dfs.copy()
        # 用昨天的数据！
        processed_target_factor_df = processed_target_factor_df.shift(1)  # 用昨天的数据！
        for name, df in auxiliary_dfs.items():
            auxiliary_dfs[name] = df.shift(1)

        # 步骤1：去极值
        # print("2. 去极值处理...")
        processed_target_factor_df = self._winsorize(processed_target_factor_df)

        # 步骤2：中性化
        if self.preprocessing_config.get('neutralization', {}).get('enable', False):
            # print("3. 中性化处理...")
            processed_target_factor_df = self._neutralize(processed_target_factor_df, target_factor_name,auxiliary_dfs,neutral_dfs, factor_school)
        else:
            print("3. 跳过中性化处理...")

        # 步骤3：标准化
        # print("4. 标准化处理...")
        processed_target_factor_df = self._standardize(processed_target_factor_df)

        # 统计处理结果
        self._print_processing_stats(target_factor_df, processed_target_factor_df)

        return processed_target_factor_df

    # ok#okdiff
    def _winsorize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        去极值处理
        
        Args:
            factor_data: 因子数据
            
        Returns:
            去极值后的因子数据
        """
        winsorization_config = self.preprocessing_config.get('winsorization', {})
        method = winsorization_config.get('method', 'mad')

        processed_factor = factor_data.copy()

        if method == 'mad':
            # 中位数绝对偏差法 (Median Absolute Deviation)
            threshold = winsorization_config.get('mad_threshold', 5)
            # print(f"  使用MAD方法，阈值倍数: {threshold}")

            # 向量化计算每日的中位数和MAD
            median = factor_data.median(axis=1)
            mad = (factor_data.sub(median, axis=0)).abs().median(axis=1)

            # 向量化计算每日的上下边界
            upper_bound = median + threshold * mad
            lower_bound = median - threshold * mad

            # 向量化clip，axis=0确保按行广播边界
            return factor_data.clip(lower_bound, upper_bound, axis=0)
        elif method == 'quantile':
            # 分位数法
            quantile_range = winsorization_config.get('quantile_range', [0.01, 0.99])
            print(f"  使用分位数方法，范围: {quantile_range}")
            # 向量化计算每日的分位数边界
            bounds = factor_data.quantile(q=quantile_range, axis=1).T  # .T转置是为了方便后续clip
            lower_bound = bounds.iloc[:, 0]
            upper_bound = bounds.iloc[:, 1]
            return factor_data.clip(lower_bound, upper_bound, axis=0)

        return processed_factor

    # ok
    def _neutralize(self,
                    factor_data: pd.DataFrame,
                    target_factor_name:str,
                    auxiliary_dfs: Dict[str, pd.DataFrame],
                    neutral_dfs: Dict[str, pd.DataFrame],
                    factor_school: str  # <-- 新增的关键参数：因子门派
                    ) -> pd.DataFrame:
        """
         根据因子所属的“门派”，自动选择最合适的中性化方案。
         Args:
             factor_data: 待中性化的因子数据 (T-1日信息)
             target_factor_name: 待中性化因子的名称
             auxiliary_dfs: 辅助数据字典 (可能包含Beta等，根据您的实际约定)
             neutral_dfs: 包含了市值和预处理好的行业哑变量数据的字典 (日期-股票代码的DataFrame)
             factor_school: 因子门派 ('fundamentals', 'trend', 'microstructure')

         Returns:
             中性化后的因子数据
         """
        neutralization_config = self.preprocessing_config.get('neutralization', {})
        if not neutralization_config.get('enable', False):
            return factor_data

        logger.info(f"  > 正在对 '{factor_school}' 派因子 '{target_factor_name}' 进行中性化处理...")
        processed_factor = factor_data.copy()

        # --- 阶段一：(可选) 因子残差化 (仅针对市场微观派) ---
        # 残差化通常是对时间序列进行去均值，与截面中性化是两个不同的操作
        if factor_school == 'microstructure':
            window = neutralization_config.get('residualization_window', 20)
            logger.info(f"    > 应用时间序列残差化 (窗口: {window}天)...")
            # 确保 min_periods 不会太小导致过多 NaN
            factor_mean = processed_factor.rolling(window=window, min_periods=max(1, int(window * 0.5))).mean()
            processed_factor = processed_factor - factor_mean

        # --- 阶段二：确定本次回归需要中性化的因子列表 ---
        factors_to_neutralize = []
        if factor_school == 'fundamentals':
            factors_to_neutralize = ['market_cap', 'industry']
        elif factor_school == 'momentum':
            factors_to_neutralize = ['market_cap', 'industry', 'pct_chg_beta']
        elif factor_school == 'microstructure':
            factors_to_neutralize = ['market_cap', 'industry']

        if not factors_to_neutralize:
            logger.info(f"    > '{factor_school}' 派因子无需中性化。")
            return processed_factor

        # --- 阶段三：逐日进行截面回归中性化 ---
        for date in processed_factor.index:
            # 获取当天待中性化的因子数据（因变量 y）
            y = processed_factor.loc[date].dropna()
            if len(y) < 20:  # 确保有足够多的样本进行回归，避免过拟合或回归不稳定 todo 实盘需注意，这设置的20
                logger.debug(f"    日期 {date.date()} 样本数不足 ({len(y)} < 20)，跳过中性化。")
                # 当天数据如果不足，将处理后的因子值设为NaN，表示该日未进行有效中性化
                processed_factor.loc[date] = np.nan
                continue

            # a) 构建回归自变量矩阵 X
            # X_df 的索引必须与 y 的索引（即当前日期有因子值的股票）保持一致
            X_df = pd.DataFrame(index=y.index)

            # --- 市值因子 (从 neutral_dfs 获取) ---
            if 'market_cap' in factors_to_neutralize:
                if 'total_mv' not in neutral_dfs:
                    raise ValueError(f"  错误: neutral_dfs 中缺少 'total_mv' 数据，无法进行市值中性化。")

                mv_series = neutral_dfs['total_mv'].loc[date]
                # 对市值取对数，这是常见做法，因为市值分布通常是偏态的
                X_df['log_market_cap'] = np.log(mv_series.reindex(y.index))  # 确保与 y 的索引对齐

            # --- 行业因子 (从 neutral_dfs 获取预处理好的哑变量) ---
            if 'industry' in factors_to_neutralize:
                # 找到所有以 'industry_' 开头的辅助数据键，这些就是预处理好的行业哑变量
                # 小白解释：列表推导式，高效地从字典键中筛选出所有行业哑变量的名称。
                industry_dummy_keys = [k for k in neutral_dfs.keys() if k.startswith('industry_')]

                if not industry_dummy_keys:
                   raise ValueError(
                        f"   neutral_dfs 中未发现任何预处理的行业哑变量（以 'industry_' 开头），行业中性化失败。")
                else:
                    current_date_industry_dummies = pd.DataFrame(index=y.index)
                    for industry_key in industry_dummy_keys:
                        # 从 neutral_dfs 中获取对应行业的 DataFrame
                        # 然后 loc[date] 获取当前日期的数据
                        # reindex(y.index) 确保哑变量与当前日期的 y 因子值股票对齐
                        dummy_series = neutral_dfs[industry_key].loc[date]
                        current_date_industry_dummies[industry_key] = dummy_series.reindex(y.index)

                    # 合并所有行业哑变量到 X_df
                    X_df = X_df.join(current_date_industry_dummies)

            # --- Beta 因子 (假设仍在 auxiliary_dfs 中) ---
            if 'pct_chg_beta' in factors_to_neutralize:
                if 'pct_chg_beta' not in auxiliary_dfs:
                    raise ValueError(f"  错误: auxiliary_dfs 中缺少 'pct_chg_beta' 数据，无法进行Beta中性化。")

                beta_series = auxiliary_dfs['pct_chg_beta'].loc[date]
                X_df['pct_chg_beta'] = beta_series.reindex(y.index)  # 确保与 y 的索引对齐

            # b) 数据对齐与清洗
            # 使用 join 和 dropna 来确保 y 和 X_df 的股票代码对齐，并去除所有 NaN
            # 注意：X_df 在构建时已经基于 y.index，但如果某个辅助因子本身有 NaN，这里会进一步清理
            combined_df = pd.concat([y, X_df], axis=1).dropna()

            # 确保清理后的样本数满足回归要求
            # X_df.shape[1] 是自变量的数量，我们需要比自变量数量更多的样本，避免欠拟合
            # 经验法则：样本数至少是自变量数量的几倍（例如，2-5倍）
            num_predictors = X_df.shape[1]  # 计算实际的自变量数量
            if len(combined_df) < num_predictors + 5:  # 至少要比变量数多5个样本，这是一个经验值，可以根据需求调整 ，x是自变量个数，y是因变量（测试的天数），甚至都赶不上自变量的个数，那还测试什么 干脆报错把 但是实际很难出现，x最多就100来个（因为行业就100来个），我们回测天数都是几百天，
                logger.warning(
                    f"  警告: 日期 {date.date()} 清理后样本数不足 ({len(combined_df)} < {num_predictors + 5})，跳过中性化。")
                processed_factor.loc[date] = np.nan  # 将当天因子数据设为NaN
                continue

            # 提取清理后的因变量和自变量
            y_clean = combined_df[y.name]
            # 确保 X_clean 拿到的是对齐后的，并且是 X_df 中实际使用的列
            X_clean = combined_df[X_df.columns]

            # c) 执行回归并计算残差
            try:
                # fit_intercept=True 默认包含截距项，通常是回归分析的良好实践
                reg = LinearRegression(fit_intercept=True)
                reg.fit(X_clean, y_clean)

                # 计算残差：原始因子值 - 因子模型预测值
                residuals = y_clean - reg.predict(X_clean)

                # 将中性化后的残差更新回 processed_factor
                # 只更新那些参与了回归的股票（即 residuals.index）
                processed_factor.loc[date, residuals.index] = residuals
                logger.debug(f"    日期 {date.date()} 中性化成功，处理了 {len(residuals)} 个样本。")

            except Exception as e:
                # 捕获回归可能出现的错误，如矩阵奇异（共线性）、样本不足导致无法拟合等
                raise ValueError(f"  警告: 日期 {date.date()} 中性化回归失败: {e}。该日因子数据将标记为NaN。") #raise
        return processed_factor

    # ok
    def _standardize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化处理
        
        Args:
            factor_data: 因子数据
            
        Returns:
            标准化后的因子数据
        """
        standardization_config = self.preprocessing_config.get('standardization', {})
        method = standardization_config.get('method', 'zscore')

        processed_factor = factor_data.copy()

        if method == 'zscore':
            # print("  使用Z-Score标准化 (健壮版)")
            mean = processed_factor.mean(axis=1)
            std = processed_factor.std(axis=1)

            # 识别std=0的安全隐患
            std_is_zero_mask = (std == 0)

            # 先进行标准化（会产生inf）
            processed_factor = processed_factor.sub(mean, axis=0).div(std, axis=0)

            # 将std=0的行，结果安全地设为0
            processed_factor[std_is_zero_mask] = 0.0

            return processed_factor

        elif method == 'rank':
            print("  使用排序标准化 (健壮版)")

            # 识别只有一个有效值的边界情况
            valid_counts = processed_factor.notna().sum(axis=1)
            single_value_mask = (valid_counts == 1)

            # 正常计算排名
            ranks = processed_factor.rank(axis=1, pct=True)
            processed_factor = 2 * ranks - 1

            # 将只有一个有效值的行，结果安全地设为0
            processed_factor[single_value_mask] = 0.0
            return processed_factor

        raise RuntimeError("请指定标准化方式")

    def _print_processing_stats(self,
                                original_factor: pd.DataFrame,
                                processed_factor: pd.DataFrame
                                ):
        """打印处理统计信息"""
        logger.info("因子预处理统计:")

        # 原始因子统计
        orig_valid = original_factor.notna().sum().sum()
        orig_total = original_factor.shape[0] * original_factor.shape[1]

        # 处理后因子统计
        proc_valid = processed_factor.notna().sum().sum()

        # 分布统计
        all_values = processed_factor.values.flatten()
        all_values = all_values[~np.isnan(all_values)]

        if len(all_values) > 0:
            logger.info(
                f"  处理后分布: 均值={all_values.mean():.3f}, 标准差={all_values.std():.3f} （z标准化，均值一定是0）")
            logger.info(f"  分位数: 1%={np.percentile(all_values, 1):.3f}, "
                        f"99%={np.percentile(all_values, 99):.3f}")


class FactorCalculator:
    """
    因子计算器 - 根据配置计算目标因子
    """

    def __init__(self, config: Dict):
        """
        初始化因子计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.target_factor_config = config.get('target_factors_for_evaluation', {})

    def calculate_factor(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算目标因子
        
        Args:
            data_dict: 数据字典
            
        Returns:
            因子数据
        """
        factor_name = self.target_factor_config.get('name', 'unknown_factor')
        fields = self.target_factor_config.get('fields', [])

        print(f"\n计算目标因子: {factor_name}")
        print(f"使用字段: {fields}")

        if factor_name == 'pe_inv' and 'pe_ttm' in fields:
            # PE倒数因子
            pe_data = data_dict['pe_ttm']
            factor_data = 1 / pe_data
            factor_data = factor_data.replace([np.inf, -np.inf], np.nan)

        elif factor_name == 'pb_inv' and 'pb' in fields:
            # PB倒数因子
            pb_data = data_dict['pb']
            factor_data = 1 / pb_data
            factor_data = factor_data.replace([np.inf, -np.inf], np.nan)

        elif factor_name == 'roe' and 'roe' in fields:
            # ROE因子
            factor_data = data_dict['roe']

        else:
            # 默认使用第一个字段
            if fields and fields[0] in data_dict:
                factor_data = data_dict[fields[0]]
            else:
                raise ValueError(f"无法计算因子 {factor_name}，缺少必要字段")

        print(f"因子计算完成，数据形状: {factor_data.shape}")
        return factor_data


def create_factor_processor(config: Dict) -> FactorProcessor:
    """
    创建因子预处理器
    
    Args:
        config: 配置字典
        
    Returns:
        FactorProcessor实例
    """
    return FactorProcessor(config)


def create_factor_calculator(config: Dict) -> FactorCalculator:
    """
    创建因子计算器
    
    Args:
        config: 配置字典
        
    Returns:
        FactorCalculator实例
    """
    return FactorCalculator(config)


# 使用示例
if __name__ == "__main__":
    # 加载配置
    with open("config.yml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建模拟数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    stocks = [f'stock_{i:03d}' for i in range(50)]

    # 模拟因子数据
    np.random.seed(42)
    factor_data = pd.DataFrame(
        index=dates,
        columns=stocks,
        data=np.random.randn(100, 50)
    )

    # 添加一些极值
    factor_data.iloc[10:15, 5:10] *= 10

    # 模拟股票池
    stock_pool_df = pd.DataFrame(
        index=dates,
        columns=stocks,
        data=True
    )

    # 模拟辅助数据
    auxiliary_df_dict = {
        'total_mv': pd.DataFrame(
            index=dates,
            columns=stocks,
            data=np.random.lognormal(10, 1, (100, 50))
        )
    }

    # 创建处理器并处理
    processor = create_factor_processor(config)
    processed_factor = processor.process_factor(
        factor_data,
        stock_pool_df,
        auxiliary_df_dict
    )

    print(f"\n处理完成！")
    print(f"原始因子形状: {factor_data.shape}")
    print(f"处理后形状: {processed_factor.shape}")
