"""
【V3.3 最终完美版】专为因子合成设计的评分系统
修复V3.2的逻辑问题，实现真正的"大道至简"
"""

import pandas as pd
import numpy as np
from typing import Union, Dict

# ==============================================================================
# V3.3 CONFIG: 真正的大道至简 - 清晰的维度归属
# ==============================================================================
FACTOR_EVALUATION_CONFIG_V33 = {
    'category_weights': {
        'prediction': 0.35,
        'strategy': 0.20, 
        'stability': 0.25,
        'purity': 0.10,
        'composability': 0.10,
    },
    'indicators': {
        # 每个指标明确定义：类型、分层、权重、归属维度
        'adj_ic_mean': {
            'category': 'prediction',
            'type': 'positive',
            'tiers': [(0.06, 100), (0.04, 85), (0.025, 70), (0.015, 55), (0.008, 40), (0.003, 20)],
            'weight': 0.45
        },
        'adj_ic_ir': {
            'category': 'prediction', 
            'type': 'positive',
            'tiers': [(0.6, 100), (0.4, 85), (0.25, 70), (0.15, 55), (0.08, 40), (0.03, 20)],
            'weight': 0.35
        },
        'fm_t_abs': {
            'category': 'prediction',
            'type': 'positive', 
            'tiers': [(3.0, 100), (2.3, 85), (1.8, 70), (1.3, 55), (0.8, 40)],
            'weight': 0.20
        },
        
        'adj_sharpe': {
            'category': 'strategy',
            'type': 'positive',
            'tiers': [(1.0, 100), (0.7, 85), (0.4, 70), (0.2, 55), (0.05, 40)],
            'weight': 0.6
        },
        'max_drawdown': {
            'category': 'strategy',
            'type': 'negative',
            'tiers': [(-0.15, 100), (-0.25, 85), (-0.4, 70), (-0.6, 55), (-0.8, 20)],
            'weight': 0.4
        },
        
        'monotonicity_abs': {
            'category': 'stability',
            'type': 'positive',
            'tiers': [(0.8, 100), (0.65, 85), (0.45, 70), (0.25, 55), (0.1, 40)],
            'weight': 0.4
        },
        'ic_decay': {
            'category': 'stability',
            'type': 'negative', 
            'tiers': [(0.1, 100), (0.25, 85), (0.4, 70), (0.6, 55), (0.8, 20)],
            'weight': 0.3
        },
        'ic_consistency': {
            'category': 'stability',
            'type': 'positive',
            'tiers': [(0.85, 100), (0.75, 85), (0.65, 70), (0.55, 55), (0.45, 40)],
            'weight': 0.3
        },
        
        'neutralization_decay': {
            'category': 'purity',
            'type': 'negative',
            'tiers': [(0.15, 100), (0.3, 85), (0.5, 70), (0.7, 55), (0.9, 20)],
            'weight': 1.0
        },
        
        'factor_uniqueness': {
            'category': 'composability',
            'type': 'positive',
            'tiers': [(0.8, 100), (0.65, 85), (0.45, 70), (0.25, 55), (0.1, 40)],
            'weight': 0.5
        },
        'regime_adaptability': {
            'category': 'composability',
            'type': 'positive', 
            'tiers': [(0.8, 100), (0.65, 85), (0.45, 70), (0.25, 55), (0.1, 40)],
            'weight': 0.5
        }
    },
    'composability_filters': {
        # 硬性过滤条件：不达标的因子直接降级
        'min_ic_mean_abs': 0.01,
        'min_ic_ir_abs': 0.1, 
        'min_monotonicity_abs': 0.2,
        'max_neutralization_decay': 0.85,
        'max_drawdown_limit': -0.9
    }
}

def get_metric(data: Union[pd.Series, dict], key: str, default=0.0):
    """安全获取指标值"""
    if isinstance(data, dict):
        val = data.get(key, default)
    else:
        val = getattr(data, key, default) if hasattr(data, key) else data.get(key, default)
    return default if pd.isna(val) else val

def _get_score_by_tier(value: float, indicator_config: Dict) -> float:
    """根据指标配置计算得分"""
    tiers = indicator_config['tiers']
    indicator_type = indicator_config['type']
    
    if indicator_type == 'positive':  # 值越大越好
        for threshold, score in tiers:
            if value >= threshold:
                return score
    elif indicator_type == 'negative':  # 值越小越好
        for threshold, score in sorted(tiers, key=lambda x: x[0]):
            if value <= threshold:
                return score
    return 0

def calculate_advanced_metrics(summary_row: Union[pd.Series, dict]) -> Dict:
    """
    计算高级指标 - 更专业的实现
    """
    
    # 1. 因子独特性：基于与市场的相关性
    market_corr = abs(get_metric(summary_row, 'market_correlation_o2o', 0.4))
    factor_uniqueness = max(0, 1 - market_corr)
    
    # 2. 市场环境适应性：基于IC稳定性
    ic_mean_abs = abs(get_metric(summary_row, 'ic_mean_processed_o2o', 0))
    ic_std = get_metric(summary_row, 'ic_std_processed_o2o', 0.1)
    
    if ic_std > 0:
        stability_ratio = ic_mean_abs / ic_std
        regime_adaptability = min(1.0, stability_ratio / 1.5)  # 归一化到0-1
    else:
        regime_adaptability = 0.5
    
    # 3. IC一致性：基于胜率的改进版本
    win_rate = get_metric(summary_row, 'tmb_win_rate_processed_o2o', 0.5)
    # IC一致性 = max(正向一致性, 负向一致性)
    ic_consistency = max(win_rate, 1 - win_rate)
    
    return {
        'factor_uniqueness': factor_uniqueness,
        'regime_adaptability': regime_adaptability, 
        'ic_consistency': ic_consistency
    }

def apply_composability_filters(adj_metrics: Dict, config: Dict) -> bool:
    """
    应用合成适配性过滤器
    返回True表示通过过滤，False表示不适合合成
    """
    filters = config['composability_filters']
    
    checks = [
        adj_metrics['adj_ic_mean'] >= filters['min_ic_mean_abs'],
        adj_metrics['adj_ic_ir'] >= filters['min_ic_ir_abs'],
        adj_metrics['monotonicity_abs'] >= filters['min_monotonicity_abs'],
        adj_metrics['neutralization_decay'] <= filters['max_neutralization_decay'],
        adj_metrics['max_drawdown'] >= filters['max_drawdown_limit']
    ]
    
    return all(checks)

def calculate_factor_score_v33(summary_row: Union[pd.Series, dict], 
                              config: Dict = None) -> pd.Series:
    """
    【V3.3 最终完美版】因子评分函数
    
    核心改进：
    1. 清晰的维度归属定义
    2. 统一的指标处理流程  
    3. 简化的得分计算逻辑
    4. 鲁棒的过滤机制
    """
    
    if config is None:
        config = FACTOR_EVALUATION_CONFIG_V33
    
    # === 1. 基础指标提取 ===
    base_metrics = {
        'ic_mean': get_metric(summary_row, 'ic_mean_processed_o2o'),
        'ic_ir': get_metric(summary_row, 'ic_ir_processed_o2o'), 
        'fm_t': get_metric(summary_row, 'fm_t_statistic_processed_o2o'),
        'sharpe': get_metric(summary_row, 'tmb_sharpe_processed_o2o'),
        'max_drawdown': get_metric(summary_row, 'tmb_max_drawdown_processed_o2o'),
        'monotonicity': get_metric(summary_row, 'monotonicity_spearman_processed_o2o'),
        'ic_decay': get_metric(summary_row, 'ic_decay_o2o', 0.5),
        'sharpe_raw': get_metric(summary_row, 'tmb_sharpe_raw_o2o')
    }
    
    # === 2. 高级指标计算 ===
    advanced_metrics = calculate_advanced_metrics(summary_row)
    
    # === 3. 因子方向判断 ===
    factor_direction = 1
    if base_metrics['ic_mean'] < -0.008:
        factor_direction = -1
    elif abs(base_metrics['ic_mean']) <= 0.008 and base_metrics['fm_t'] < -1.5:
        factor_direction = -1
    
    # === 4. 调整后指标计算 ===
    adj_metrics = {
        'adj_ic_mean': abs(base_metrics['ic_mean']),  # 关注预测强度
        'adj_ic_ir': abs(base_metrics['ic_ir']),
        'fm_t_abs': abs(base_metrics['fm_t']),
        'adj_sharpe': base_metrics['sharpe'] * factor_direction,
        'max_drawdown': base_metrics['max_drawdown'],
        'monotonicity_abs': abs(base_metrics['monotonicity']),
        'ic_decay': base_metrics['ic_decay'],
        'factor_uniqueness': advanced_metrics['factor_uniqueness'],
        'regime_adaptability': advanced_metrics['regime_adaptability'],
        'ic_consistency': advanced_metrics['ic_consistency']
    }
    
    # 计算中性化衰减
    if abs(base_metrics['sharpe_raw']) > 0.1:
        raw_sharpe_adj = base_metrics['sharpe_raw'] * factor_direction
        decay = abs(raw_sharpe_adj - adj_metrics['adj_sharpe']) / max(abs(base_metrics['sharpe_raw']), 1e-6)
        adj_metrics['neutralization_decay'] = decay
    else:
        adj_metrics['neutralization_decay'] = 0.3
    
    # === 5. 合成适配性检查 ===
    composability_passed = apply_composability_filters(adj_metrics, config)
    
    # === 6. 指标评分 ===
    indicator_scores = {}
    for indicator_name, indicator_config in config['indicators'].items():
        value = adj_metrics.get(indicator_name, 0)
        score = _get_score_by_tier(value, indicator_config)
        indicator_scores[indicator_name] = score
    
    # === 7. 维度得分计算 ===
    category_scores = {}
    for category in config['category_weights'].keys():
        category_score = 0
        total_weight = 0
        
        for indicator_name, indicator_config in config['indicators'].items():
            if indicator_config['category'] == category:
                weight = indicator_config['weight']
                score = indicator_scores[indicator_name]
                category_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            category_scores[f'{category.capitalize()}_Score'] = category_score / total_weight
        else:
            category_scores[f'{category.capitalize()}_Score'] = 0
    
    # === 8. 最终得分计算 ===
    final_score = sum(
        category_scores[f'{cat.capitalize()}_Score'] * weight
        for cat, weight in config['category_weights'].items()
    )
    
    # === 9. 等级判定 ===
    if not composability_passed:
        grade = "F (不适合合成)"
        final_score = min(final_score, 35)  # 不通过过滤的因子最高35分
    elif final_score >= 85:
        grade = "S+ (顶级合成因子)"
    elif final_score >= 75:
        grade = "S (优秀合成因子)"
    elif final_score >= 65:
        grade = "A (良好合成因子)"
    elif final_score >= 50:
        grade = "B (可用合成因子)"
    elif final_score >= 35:
        grade = "C (需要改进)"
    else:
        grade = "D (不推荐)"
    
    # === 10. 结果输出 ===
    result = pd.Series(category_scores)
    result['Final_Score'] = round(final_score, 2)
    result['Grade'] = grade
    result['Factor_Direction'] = factor_direction
    result['Composability_Passed'] = composability_passed
    
    return result

# === 批量评分和筛选函数 ===
def batch_score_factors_v33(factor_results_df: pd.DataFrame) -> pd.DataFrame:
    """批量为因子评分"""
    scores_list = []
    for idx, row in factor_results_df.iterrows():
        score_result = calculate_factor_score_v33(row)
        score_result.name = idx
        scores_list.append(score_result)
    
    scores_df = pd.concat(scores_list, axis=1).T
    result_df = pd.concat([factor_results_df, scores_df], axis=1)
    
    return result_df.sort_values('Final_Score', ascending=False)

def get_composable_factors(scored_df: pd.DataFrame, min_score: float = 65) -> pd.DataFrame:
    """获取适合合成的高质量因子"""
    composable = scored_df[
        (scored_df['Composability_Passed'] == True) & 
        (scored_df['Final_Score'] >= min_score)
    ]
    return composable.sort_values('Final_Score', ascending=False)

# === 测试代码 ===
if __name__ == "__main__":
    print("🚀 V3.3最终完美版因子评分系统测试")
    print("=" * 50)
    
    # 测试数据
    test_data = {
        'ic_mean_processed_o2o': 0.042,
        'ic_ir_processed_o2o': 0.48,
        'fm_t_statistic_processed_o2o': 2.6,
        'tmb_sharpe_processed_o2o': 0.75,
        'tmb_max_drawdown_processed_o2o': -0.28,
        'tmb_win_rate_processed_o2o': 0.58,
        'monotonicity_spearman_processed_o2o': 0.62,
        'ic_decay_o2o': 0.22,
        'tmb_sharpe_raw_o2o': 0.95,
        'market_correlation_o2o': 0.35,
        'ic_std_processed_o2o': 0.09
    }
    
    result = calculate_factor_score_v33(test_data)
    
    print("\n📊 评分结果:")
    print("-" * 30)
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n🎯 评分解读:")
    print(f"该因子获得 {result['Final_Score']:.1f} 分，等级为 {result['Grade']}")
    print(f"合成适配性: {'✅ 通过' if result['Composability_Passed'] else '❌ 未通过'}")
    
    if result['Final_Score'] >= 65:
        print("💡 建议: 该因子适合用于多因子合成")
    else:
        print("💡 建议: 该因子需要进一步优化或不适合合成")
