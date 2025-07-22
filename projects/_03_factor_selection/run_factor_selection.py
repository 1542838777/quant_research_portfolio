"""
因子选择主脚本

演示完整的多因子筛选流程，包括单因子有效性检验、因子相关性分析和多因子合成。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import datetime
import yaml
import argparse

# 添加项目根目录到路径，以便导入自定义模块
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from factor_evaluation import (
    evaluate_factor,
    batch_evaluate_factors,
    FactorEvaluator
)
from quant_lib.data_loader import DataLoader
from data_manager import DataManager
from quant_lib.factor_factory import (
    create_factor,
    create_factor_combiner,
    BaseFactor
)
from quant_lib.backtesting import create_backtest_engine
from quant_lib.utils.file_utils import (
    ensure_dir_exists,
    save_to_csv,
    save_to_pickle,
    load_from_yaml
)
from quant_lib.config.constant_config import (
    LOCAL_PARQUET_DATA_DIR,
    RESULT_DIR
)
from quant_lib.evaluation import (
    calculate_ic_vectorized,
    calculate_turnover_vectorized
)
from quant_lib.config.logger_config import setup_logger

# 配置日志
logger = setup_logger(__name__)


def load_config(config_path: str) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    logger.info(f"加载配置文件: {config_path}")
    return load_from_yaml(config_path)


def load_data(config_path: str) -> tuple:
    """
    使用DataManager加载数据和构建股票池

    Args:
        config_path: 配置文件路径

    Returns:
        (data_dict, universe_df): 数据字典和股票池DataFrame
    """
    logger.info("开始加载数据...")

    # 创建数据管理器（读取本地config 以及 读取本地quarket文件 字段结构）
    data_manager = DataManager(config_path)

    # 加载所有数据（包括股票池构建）
    data_dict = data_manager.load_all_data()

    # 获取股票池
    # universe_df = data_manager.get_universe()

    logger.info(f"数据加载完成，共加载 {len(data_dict)} 个字段")
    # logger.info(f"股票池构建完成，平均每日股票数: {universe_df.sum(axis=1).mean():.0f}")

    return data_dict#, universe_df


def generate_factors(config: dict, data_dict: dict) -> Dict[str, pd.DataFrame]:
    """
    生成因子
    
    Args:
        config: 配置字典
        data_dict: 数据字典
        
    Returns:
        因子字典，键为因子名称，值为因子DataFrame
    """
    logger.info("开始生成因子...")
    
    # 因子字典
    factor_dict = {}
    
    # 生成各类因子
    for factor_config in config['factors']:
        factor_name = factor_config['name']
        factor_type = factor_config['type']
        
        logger.info(f"生成因子: {factor_name} (类型: {factor_type})")
        
        # 创建因子
        factor = create_factor(factor_type, name=factor_name)
        
        # 计算因子值
        factor_df = factor.compute(data_dict)
        
        # 存储因子
        factor_dict[factor_name] = factor_df
    
    logger.info(f"因子生成完成，共 {len(factor_dict)} 个因子")
    return factor_dict


def step1_single_factor_test(config: dict, factor_dict: Dict[str, pd.DataFrame], 
                           price_df: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    """
    第一步：单因子有效性检验
    
    Args:
        config: 配置字典
        factor_dict: 因子字典
        price_df: 价格DataFrame
        result_dir: 结果保存目录
        
    Returns:
        因子评价摘要DataFrame
    """
    logger.info("===== 第一步：单因子有效性检验 =====")
    
    # 创建结果目录
    step1_dir = result_dir / '01_single_factor_test'
    ensure_dir_exists(step1_dir)
    
    # 批量评价因子
    factor_summary = batch_evaluate_factors(
        factor_dict=factor_dict,
        price_df=price_df,
        start_date=config['backtest']['start_date'],
        end_date=config['backtest']['end_date'],
        n_groups=config['evaluate_factor_score']['n_groups'],
        forward_periods=config['evaluate_factor_score']['forward_periods'],
        result_dir=step1_dir,
        plot_results=True
    )
    
    # 显示图表
    plt.show()
    
    # 筛选有效因子
    effective_factors = factor_summary[factor_summary['is_effective'] == True]
    
    # 保存有效因子列表
    save_to_csv(effective_factors, step1_dir / 'effective_factors.csv')
    
    # 输出结果
    logger.info(f"单因子有效性检验完成，共 {len(factor_dict)} 个因子中有 {len(effective_factors)} 个有效")
    logger.info(f"有效因子: {effective_factors['factor_name'].tolist()}")
    
    return factor_summary


def step2_factor_correlation(config: dict, factor_dict: Dict[str, pd.DataFrame], 
                           factor_summary: pd.DataFrame, result_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    第二步：因子相关性分析
    
    Args:
        config: 配置字典
        factor_dict: 因子字典
        factor_summary: 因子评价摘要DataFrame
        result_dir: 结果保存目录
        
    Returns:
        筛选后的因子字典
    """
    logger.info("===== 第二步：因子相关性分析 =====")
    
    # 创建结果目录
    step2_dir = result_dir / '02_factor_correlation'
    ensure_dir_exists(step2_dir)
    
    # 筛选有效因子
    effective_factors = factor_summary[factor_summary['is_effective'] == True]
    effective_factor_names = effective_factors['factor_name'].tolist()
    
    if len(effective_factor_names) <= 1:
        logger.warning("有效因子数量不足，无法进行相关性分析")
        return {name: factor_dict[name] for name in effective_factor_names}
    
    # 提取有效因子数据
    effective_factor_dict = {name: factor_dict[name] for name in effective_factor_names}
    
    # 计算因子之间的相关性
    logger.info("计算因子相关性矩阵...")
    
    # 对齐因子数据
    aligned_factors = {}
    common_dates = None
    
    for name, df in effective_factor_dict.items():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)
    
    # 计算每个日期的因子相关性，然后取平均
    daily_corr_matrices = []
    
    for date in common_dates:
        factor_values = {}
        
        for name, df in effective_factor_dict.items():
            factor_values[name] = df.loc[date].dropna()
        
        # 找出所有因子共同的股票
        common_stocks = None
        for values in factor_values.values():
            if common_stocks is None:
                common_stocks = values.index
            else:
                common_stocks = common_stocks.intersection(values.index)
        
        if len(common_stocks) < 30:  # 至少需要30只股票
            continue
        
        # 创建因子值DataFrame
        factor_df = pd.DataFrame({
            name: values[common_stocks]
            for name, values in factor_values.items()
        })
        
        # 计算相关性矩阵
        corr_matrix = factor_df.corr(method='spearman')
        daily_corr_matrices.append(corr_matrix)
    
    # 计算平均相关性矩阵
    avg_corr_matrix = pd.concat(daily_corr_matrices).groupby(level=0).mean()
    
    # 保存相关性矩阵
    save_to_csv(avg_corr_matrix, step2_dir / 'factor_correlation.csv')
    
    # 绘制相关性热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(avg_corr_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, center=0)
    plt.title('因子相关性矩阵')
    plt.tight_layout()
    plt.savefig(step2_dir / 'factor_correlation_heatmap.png')
    plt.show()
    
    # 筛选低相关因子
    logger.info("筛选低相关因子...")
    
    # 获取因子IR值
    factor_ir = {}
    for _, row in effective_factors.iterrows():
        name = row['factor_name']
        # 使用平均IR作为因子质量指标
        ir_cols = [col for col in row.index if 'ic_' in col and '_ir' in col]
        factor_ir[name] = row[ir_cols].mean()
    
    # 按IR值排序因子
    sorted_factors = sorted(factor_ir.items(), key=lambda x: x[1], reverse=True)
    
    # 贪心算法选择因子
    selected_factors = []
    correlation_threshold = config['factor_selection']['correlation_threshold']
    
    for factor_name, ir in sorted_factors:
        # 如果是第一个因子，直接选择
        if not selected_factors:
            selected_factors.append(factor_name)
            continue
        
        # 检查与已选因子的相关性
        is_correlated = False
        for selected in selected_factors:
            if abs(avg_corr_matrix.loc[factor_name, selected]) > correlation_threshold:
                is_correlated = True
                break
        
        # 如果与已选因子相关性低，则选择
        if not is_correlated:
            selected_factors.append(factor_name)
    
    # 保存选择的因子
    selected_factors_df = effective_factors[effective_factors['factor_name'].isin(selected_factors)]
    save_to_csv(selected_factors_df, step2_dir / 'selected_factors.csv')
    
    # 筛选后的因子字典
    selected_factor_dict = {name: factor_dict[name] for name in selected_factors}
    
    # 输出结果
    logger.info(f"因子相关性分析完成，从 {len(effective_factor_names)} 个有效因子中筛选出 {len(selected_factors)} 个低相关因子")
    logger.info(f"选择的因子: {selected_factors}")
    
    return selected_factor_dict


def step3_factor_combination(config: dict, selected_factor_dict: Dict[str, pd.DataFrame], 
                           price_df: pd.DataFrame, result_dir: Path) -> Dict:
    """
    第三步：多因子合成与回测
    
    Args:
        config: 配置字典
        selected_factor_dict: 筛选后的因子字典
        price_df: 价格DataFrame
        result_dir: 结果保存目录
        
    Returns:
        回测结果字典
    """
    logger.info("===== 第三步：多因子合成与回测 =====")
    
    # 创建结果目录
    step3_dir = result_dir / '03_factor_combination'
    ensure_dir_exists(step3_dir)
    
    # 因子名称列表
    factor_names = list(selected_factor_dict.keys())
    
    if not factor_names:
        logger.warning("没有可用的因子，无法进行多因子合成")
        return {}
    
    # 对齐因子数据
    logger.info("对齐因子数据...")
    aligned_factors = {}
    common_dates = None
    common_stocks = None
    
    for name, df in selected_factor_dict.items():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)
    
    for date in common_dates:
        stocks_at_date = None
        
        for name, df in selected_factor_dict.items():
            values = df.loc[date].dropna()
            
            if stocks_at_date is None:
                stocks_at_date = values.index
            else:
                stocks_at_date = stocks_at_date.intersection(values.index)
        
        if common_stocks is None:
            common_stocks = stocks_at_date
        else:
            common_stocks = common_stocks.union(stocks_at_date)
    
    # 创建对齐后的因子DataFrame
    for name, df in selected_factor_dict.items():
        aligned_df = pd.DataFrame(index=common_dates, columns=common_stocks)
        
        for date in common_dates:
            values = df.loc[date].dropna()
            aligned_df.loc[date, values.index] = values
        
        aligned_factors[name] = aligned_df
    
    # 因子标准化
    logger.info("标准化因子...")
    standardized_factors = {}
    
    for name, df in aligned_factors.items():
        std_df = pd.DataFrame(index=df.index, columns=df.columns)
        
        for date in df.index:
            values = df.loc[date].dropna()
            
            if len(values) > 0:
                mean = values.mean()
                std = values.std()
                
                if std > 0:
                    std_df.loc[date, values.index] = (values - mean) / std
        
        standardized_factors[name] = std_df
    
    # 合成多因子
    logger.info("合成多因子...")
    
    # 确定因子权重
    if config['factor_combination']['weight_method'] == 'equal':
        # 等权重
        weights = {name: 1.0 / len(factor_names) for name in factor_names}
    else:
        # 使用IR作为权重
        ir_values = {}
        total_ir = 0
        
        for name in factor_names:
            # 这里简化处理，实际应该从评价结果中获取IR
            ir = 0.5  # 默认IR值
            ir_values[name] = ir
            total_ir += ir
        
        weights = {name: ir / total_ir for name, ir in ir_values.items()}
    
    # 合成因子
    composite_factor = pd.DataFrame(index=common_dates, columns=common_stocks)
    
    for date in common_dates:
        composite_values = pd.Series(0.0, index=common_stocks)
        valid_weights = {}
        total_weight = 0
        
        for name, df in standardized_factors.items():
            values = df.loc[date].dropna()
            
            if len(values) > 0:
                valid_weights[name] = weights[name]
                total_weight += weights[name]
                composite_values[values.index] += values * weights[name]
        
        # 权重归一化
        if total_weight > 0:
            composite_values = composite_values / total_weight
        
        composite_factor.loc[date] = composite_values
    
    # 保存合成因子
    save_to_csv(composite_factor, step3_dir / 'composite_factor.csv')
    
    # 评价合成因子
    logger.info("评价合成因子...")
    composite_result = evaluate_factor(
        factor_name='composite',
        factor_df=composite_factor,
        price_df=price_df,
        start_date=config['backtest']['start_date'],
        end_date=config['backtest']['end_date'],
        n_groups=config['evaluate_factor_score']['n_groups'],
        forward_periods=config['evaluate_factor_score']['forward_periods'],
        result_dir=step3_dir,
        plot_results=True
    )
    plt.show()
    
    # 样本内外测试
    logger.info("进行样本内外测试...")
    
    # 划分样本内外时间段
    all_dates = price_df.index
    mid_point = len(all_dates) // 2
    in_sample_end = all_dates[mid_point]
    out_sample_start = all_dates[mid_point + 1]
    
    # 样本内测试
    in_sample_result = evaluate_factor(
        factor_name='composite_in_sample',
        factor_df=composite_factor,
        price_df=price_df,
        start_date=config['backtest']['start_date'],
        end_date=in_sample_end.strftime('%Y-%m-%d'),
        n_groups=config['evaluate_factor_score']['n_groups'],
        forward_periods=config['evaluate_factor_score']['forward_periods'],
        result_dir=step3_dir / 'in_sample',
        plot_results=True
    )
    plt.show()
    
    # 样本外测试
    out_sample_result = evaluate_factor(
        factor_name='composite_out_sample',
        factor_df=composite_factor,
        price_df=price_df,
        start_date=out_sample_start.strftime('%Y-%m-%d'),
        end_date=config['backtest']['end_date'],
        n_groups=config['evaluate_factor_score']['n_groups'],
        forward_periods=config['evaluate_factor_score']['forward_periods'],
        result_dir=step3_dir / 'out_sample',
        plot_results=True
    )
    plt.show()
    
    # 比较样本内外结果
    in_sample_metrics = in_sample_result['summary']
    out_sample_metrics = out_sample_result['summary']
    
    # 合并样本内外结果
    comparison = pd.concat([
        in_sample_metrics.add_prefix('in_sample_'),
        out_sample_metrics.add_prefix('out_sample_')
    ], axis=1)
    
    save_to_csv(comparison, step3_dir / 'in_out_sample_comparison.csv')
    
    # 输出结果
    logger.info("多因子合成与回测完成")
    logger.info(f"合成因子使用的因子: {factor_names}")
    logger.info(f"合成因子的权重: {weights}")
    
    # 返回结果
    return {
        'composite_factor': composite_factor,
        'weights': weights,
        'evaluate_factor_score': composite_result,
        'in_sample': in_sample_result,
        'out_sample': out_sample_result,
        'comparison': comparison
    }


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='因子选择流程')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 加载配置
    config_path = current_dir / args.config
    config = load_config(config_path)
    
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = RESULT_DIR / 'factor_selection' / timestamp
    ensure_dir_exists(result_dir)
    
    # 保存配置副本
    # dev 不需要保存现场！
    # with open(result_dir / 'config.yaml', 'w') as f:
    #     yaml.dump(config, f, default_flow_style=False)
    
    # 加载数据
    data_dict = load_data(config_path)    # 必须只考虑在每个时间点上，universe_df中为True的股票。

    
    # 生成因子
    factor_dict = generate_factors(config, data_dict)
    
    # 第一步：单因子有效性检验
    factor_summary = step1_single_factor_test(
        config=config,
        factor_dict=factor_dict,
        price_df=data_dict['close'],
        result_dir=result_dir
    )
    
    # 第二步：因子相关性分析
    selected_factor_dict = step2_factor_correlation(
        config=config,
        factor_dict=factor_dict,
        factor_summary=factor_summary,
        result_dir=result_dir
    )
    
    # 第三步：多因子合成与回测
    combination_result = step3_factor_combination(
        config=config,
        selected_factor_dict=selected_factor_dict,
        price_df=data_dict['close'],
        result_dir=result_dir
    )
    
    # 输出最终结果
    logger.info("\n===== 因子选择流程完成 =====")
    logger.info(f"结果保存在: {result_dir}")


if __name__ == "__main__":
    main() 