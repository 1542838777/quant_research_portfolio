"""
并发测试结果分析脚本

快速分析和可视化并发因子测试的结果
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


def load_concurrent_test_results(results_path: str) -> Dict:
    """加载并发测试结果"""
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✅ 成功加载测试结果: {results_path}")
        logger.info(f"📊 测试元数据: {data.get('metadata', {})}")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ 加载结果文件失败: {e}")
        raise


def extract_ic_performance_summary(results: List[Dict]) -> pd.DataFrame:
    """提取IC表现汇总"""
    
    summary_data = []
    
    for result in results:
        if result['status'] != 'success':
            continue
            
        factor_name = result['factor_name']
        test_results = result.get('test_results', {})
        ic_stats = test_results.get('ic_stats', {})
        
        # 提取不同周期的IC统计
        for period, stats in ic_stats.items():
            if isinstance(stats, dict):
                summary_data.append({
                    'factor_name': factor_name,
                    'period': period,
                    'ic_mean': stats.get('ic_mean', np.nan),
                    'ic_ir': stats.get('ic_ir', np.nan),
                    'ic_win_rate': stats.get('ic_win_rate', np.nan),
                    'ic_t_stat': stats.get('ic_t_stat', np.nan),
                    'ic_p_value': stats.get('ic_p_value', np.nan),
                    'ic_abs_mean': stats.get('ic_abs_mean', np.nan),
                })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        return df
    else:
        logger.warning("⚠️ 未找到有效的IC统计数据")
        return pd.DataFrame()


def create_factor_ranking_report(ic_summary: pd.DataFrame, period: str = '21d') -> pd.DataFrame:
    """创建因子排名报告"""
    
    if ic_summary.empty:
        return pd.DataFrame()
    
    # 筛选特定周期
    period_data = ic_summary[ic_summary['period'] == period].copy()
    
    if period_data.empty:
        logger.warning(f"⚠️ 未找到周期 {period} 的数据")
        return pd.DataFrame()
    
    # 计算综合得分 (IC绝对值 * IR)
    period_data['综合得分'] = period_data['ic_abs_mean'] * period_data['ic_ir']
    
    # 按综合得分排序
    ranking = period_data.sort_values('综合得分', ascending=False).reset_index(drop=True)
    ranking['排名'] = range(1, len(ranking) + 1)
    
    # 选择关键列
    key_columns = ['排名', 'factor_name', 'ic_mean', 'ic_ir', 'ic_win_rate', 
                   'ic_abs_mean', 'ic_p_value', '综合得分']
    
    ranking_report = ranking[key_columns].round(4)
    
    return ranking_report


def print_top_factors_summary(ranking_df: pd.DataFrame, top_n: int = 10):
    """打印顶级因子汇总"""
    
    if ranking_df.empty:
        logger.warning("⚠️ 排名数据为空，无法生成汇总")
        return
    
    print("\n" + "="*80)
    print(f"🏆 Top {top_n} 因子表现汇总")
    print("="*80)
    
    top_factors = ranking_df.head(top_n)
    
    print(f"{'排名':>4} {'因子名称':<25} {'IC均值':>8} {'IC_IR':>8} {'胜率':>6} {'综合得分':>8}")
    print("-" * 80)
    
    for _, row in top_factors.iterrows():
        print(f"{row['排名']:>4} {row['factor_name']:<25} "
              f"{row['ic_mean']:>8.4f} {row['ic_ir']:>8.4f} "
              f"{row['ic_win_rate']:>6.1%} {row['综合得分']:>8.4f}")
    
    # 统计汇总
    print("\n📊 表现统计:")
    print(f"  - IC均值 > 0.02: {len(top_factors[top_factors['ic_mean'].abs() > 0.02])} 个")
    print(f"  - IC_IR > 0.3:  {len(top_factors[top_factors['ic_ir'] > 0.3])} 个") 
    print(f"  - 胜率 > 55%:   {len(top_factors[top_factors['ic_win_rate'] > 0.55])} 个")
    print(f"  - p值 < 0.05:   {len(top_factors[top_factors['ic_p_value'] < 0.05])} 个")


def analyze_factor_categories(ranking_df: pd.DataFrame):
    """分析不同类别因子的表现"""
    
    if ranking_df.empty:
        return
    
    # 因子类别映射（根据你的因子命名规律）
    category_mapping = {
        # 价值因子
        'bm_ratio': '价值', 'ep_ratio': '价值', 'sp_ratio': '价值', 'cfp_ratio': '价值',
        
        # 质量因子  
        'roe_ttm': '质量', 'roa_ttm': '质量', 'roe_change_q': '质量',
        'gross_margin_ttm': '质量', 'debt_to_assets': '质量', 
        'operating_accruals': '质量', 'earnings_stability': '质量',
        
        # 成长因子
        'net_profit_growth_yoy': '成长', 'total_revenue_growth_yoy': '成长',
        
        # 动量因子
        'momentum_12_1': '动量', 'momentum_20d': '动量', 'momentum_60d': '动量',
        'momentum_120d': '动量', 'momentum_pct_60d': '动量', 'sharpe_momentum_60d': '动量',
        
        # 反转因子
        'reversal_5d': '反转', 'reversal_21d': '反转',
        
        # 风险因子
        'beta': '风险', 'volatility_40d': '风险', 'volatility_90d': '风险', 'volatility_120d': '风险',
        
        # 流动性因子
        'turnover_rate_90d_mean': '流动性', 'turnover_rate_monthly_mean': '流动性',
        'ln_turnover_value_90d': '流动性', 'turnover_t1_div_t20d_avg': '流动性', 
        'amihud_liquidity': '流动性',
        
        # 技术因子
        'rsi': '技术', 'cci': '技术',
        
        # 规模因子
        'log_total_mv': '规模', 'log_circ_mv': '规模',
        
        # 其他
        'pead': '事件', 'quality_momentum': '复合', 'vwap_deviation_20d': '微观结构',
        'large_trade_ratio_10d': '资金流'
    }
    
    # 添加类别列
    ranking_df['类别'] = ranking_df['factor_name'].map(category_mapping).fillna('其他')
    
    # 按类别统计
    category_stats = ranking_df.groupby('类别').agg({
        'ic_mean': ['count', 'mean'],
        'ic_ir': 'mean', 
        'ic_win_rate': 'mean',
        '综合得分': 'mean'
    }).round(4)
    
    category_stats.columns = ['因子数量', '平均IC', '平均IR', '平均胜率', '平均综合得分']
    category_stats = category_stats.sort_values('平均综合得分', ascending=False)
    
    print("\n" + "="*70)
    print("📈 因子类别表现分析")
    print("="*70)
    print(category_stats)
    

def plot_ic_performance_heatmap(ic_summary: pd.DataFrame, save_path: str = None):
    """绘制IC表现热力图"""
    
    if ic_summary.empty:
        return
    
    # 创建透视表
    pivot_data = ic_summary.pivot_table(
        index='factor_name', 
        columns='period', 
        values='ic_mean',
        aggfunc='first'
    )
    
    plt.figure(figsize=(12, 16))
    sns.heatmap(
        pivot_data, 
        annot=True, 
        fmt='.3f', 
        cmap='RdYlBu_r', 
        center=0,
        cbar_kws={'label': 'IC Mean'}
    )
    
    plt.title('因子IC表现热力图', fontsize=16, pad=20)
    plt.xlabel('预测周期', fontsize=12)
    plt.ylabel('因子名称', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 热力图已保存: {save_path}")
    
    plt.show()


def main():
    """主分析函数"""
    
    # 查找最新的结果文件
    current_dir = Path(__file__).parent
    result_files = list(current_dir.glob("concurrent_factor_test_results_*.json"))
    
    if not result_files:
        logger.error("❌ 未找到并发测试结果文件")
        logger.info("💡 请先运行 concurrent_factor_testing.py 生成结果")
        return
    
    # 使用最新的结果文件
    latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"📂 使用结果文件: {latest_result}")
    
    # 加载数据
    data = load_concurrent_test_results(str(latest_result))
    results = data.get('results', [])
    
    if not results:
        logger.error("❌ 结果数据为空")
        return
    
    # 提取IC性能汇总
    ic_summary = extract_ic_performance_summary(results)
    
    if ic_summary.empty:
        logger.error("❌ 无法提取IC性能数据")
        return
    
    # 分析21日周期的表现 (中期预测，最重要)
    ranking_21d = create_factor_ranking_report(ic_summary, period='21d')
    
    if not ranking_21d.empty:
        # 打印顶级因子
        print_top_factors_summary(ranking_21d, top_n=15)
        
        # 分析因子类别表现  
        analyze_factor_categories(ranking_21d)
        
        # 保存排名结果
        output_file = current_dir / f"factor_ranking_21d_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
        ranking_21d.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 排名结果已保存: {output_file}")
    
    # 绘制热力图 (可选)
    try:
        plot_ic_performance_heatmap(ic_summary, 
                                   save_path=current_dir / "ic_performance_heatmap.png")
    except Exception as e:
        logger.warning(f"⚠️ 绘制热力图失败: {e}")
    
    # 筛选有潜力的因子用于合成
    potential_factors = ranking_21d[
        (ranking_21d['ic_abs_mean'] > 0.02) & 
        (ranking_21d['ic_ir'] > 0.3) &
        (ranking_21d['ic_p_value'] < 0.1)
    ]
    
    if not potential_factors.empty:
        print(f"\n🎯 建议用于IC加权合成的因子 ({len(potential_factors)}个):")
        print("=" * 60)
        for _, row in potential_factors.head(10).iterrows():
            print(f"  - {row['factor_name']:<25} (IC: {row['ic_mean']:6.3f}, IR: {row['ic_ir']:6.3f})")
        
        # 生成因子合成建议
        factor_list = potential_factors['factor_name'].tolist()
        print(f"\n💡 代码示例:")
        print(f"candidate_factors = {factor_list}")
    else:
        print("\n⚠️ 当前没有因子满足建议的合成标准 (IC>0.02, IR>0.3, p<0.1)")
        print("建议:")
        print("1. 调整因子计算参数")  
        print("2. 尝试其他预处理方法")
        print("3. 考虑新的因子构造思路")


if __name__ == "__main__":
    main()