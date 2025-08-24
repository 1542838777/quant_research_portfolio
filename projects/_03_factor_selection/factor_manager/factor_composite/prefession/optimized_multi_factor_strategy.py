"""
基于你的v3测试结果的优化多因子策略
专门针对IC较低但相关性小的因子进行组合优化

根据你的测试结果，识别出的有价值因子组合：
- 正向因子：earnings_stability(IC=0.016), amihud_liquidity(IC=0.019), ep_ratio(IC=0.017)
- 负向因子：volatility_40d(IC=-0.029), turnover_rate_monthly_mean(IC=-0.024)  #todo 考虑加一下方向：然后重跑一下 这改过的
- 技术因子：rsi(IC=-0.015)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from pathlib import Path

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from projects._03_factor_selection.factory.strategy_factory import StrategyFactory
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedMultiFactorStrategy:
    """基于你测试结果的优化多因子策略"""
    
    def __init__(self):
        self.factory = StrategyFactory("../../../factory/config.yaml")
        self.analyzer = FactorAnalyzer()
        
        # 根据你的v3测试结果精选的因子组合
        self.factor_groups = {
            # 价值质量类（正向Alpha）
            'quality_value': {
                'earnings_stability': {'ic': 0.0164, 'weight': 0.3},
                'amihud_liquidity': {'ic': 0.0186, 'weight': 0.25},
                'ep_ratio': {'ic': 0.0166, 'weight': 0.2}
            },
            
            # 风险控制类（负向，用于风险控制）
            'risk_control': {
                'volatility_40d': {'ic': -0.0287, 'weight': 0.15},
                'turnover_rate_monthly_mean': {'ic': -0.0239, 'weight': 0.1}
            },
            
            # 技术类（短期信号）
            # 'technical': {
            #     'rsi': {'ic': -0.0152, 'weight': 0.05}
            # }
        }
        
    def create_targeted_combination(self) -> pd.DataFrame:
        """创建针对性的因子组合"""
        logger.info("开始创建基于测试结果的优化因子组合...")
        
        all_factor_data = {}
        
        # 加载每个因子的数据
        for group_name, factors in self.factor_groups.items():
            logger.info(f"处理因子组: {group_name}")
            
            for factor_name, factor_info in factors.items():
                try:
                    logger.info(f"  加载因子: {factor_name}")
                    
                    # 使用工厂类加载因子数据
                    result = self.factory.test_single_factor(
                        factor_name=factor_name,
                        stock_pool="ZZ800"
                    )
                    
                    if 'factor_data' in result and result['factor_data'] is not None:
                        factor_data = result['factor_data'].copy()
                        
                        # 处理因子方向（负IC因子需要反转）
                        if factor_info['ic'] < 0:
                            factor_data = -factor_data
                            logger.info(f"    {factor_name} 为负IC，已反转")
                        
                        # 标准化
                        factor_data = factor_data.fillna(0)
                        factor_mean = factor_data.mean()
                        factor_std = factor_data.std()
                        factor_data = (factor_data - factor_mean) / (factor_std + 1e-8)
                        
                        all_factor_data[factor_name] = {
                            'data': factor_data,
                            'weight': factor_info['weight'],
                            'ic': abs(factor_info['ic'])
                        }
                        
                        logger.info(f"    成功加载，权重: {factor_info['weight']:.2f}")
                    
                except Exception as e:
                    logger.warning(f"  跳过因子{factor_name}: {e}")
                    continue
        
        if not all_factor_data:
            logger.error("没有成功加载任何因子数据")
            return None
        
        # 计算组合因子
        combined_factor = None
        total_weight = sum(info['weight'] for info in all_factor_data.values())
        
        logger.info("开始因子加权组合:")
        for factor_name, info in all_factor_data.items():
            normalized_weight = info['weight'] / total_weight
            weighted_factor = info['data'] * normalized_weight
            
            if combined_factor is None:
                combined_factor = weighted_factor
            else:
                combined_factor = combined_factor.add(weighted_factor, fill_value=0)
            
            logger.info(f"  {factor_name}: 权重 {normalized_weight:.3f}, IC {info['ic']:.4f}")
        
        # 最终标准化
        combined_mean = combined_factor.mean()
        combined_std = combined_factor.std()
        combined_factor = (combined_factor - combined_mean) / (combined_std + 1e-8)
        
        logger.info("因子组合创建完成")
        return combined_factor
    
    def test_strategy_performance(self, combined_factor: pd.DataFrame) -> Dict:
        """测试策略表现"""
        logger.info("开始测试优化策略表现...")
        
        try:
            # 执行综合测试
            (
                processed_factor,
                ic_series_dict, ic_stats_dict,
                quantile_daily_returns_dict, quantile_stats_dict,
                factor_returns_dict, fm_results_dict,
                turnover_dict, style_corr_dict,
                factor_risk_dict, pct_chg_beta_dict
            ) = self.analyzer.comprehensive_test(
                target_factor_name="optimized_multi_factor",
                factor_data_shifted=combined_factor,
                stock_pool_index_name="ZZ800",
                preprocess_method="standard",
                returns_calculator=None,
                need_process_factor=True,
                do_ic_test=True,
                do_quantile_test=True,
                do_fama_test=True
            )
            
            # 提取关键指标
            result = {
                'ic_mean': ic_stats_dict.get('5d', {}).get('ic_mean', 0) if ic_stats_dict else 0,
                'icir': ic_stats_dict.get('5d', {}).get('ic_ir', 0) if ic_stats_dict else 0,
                'ic_positive_ratio': ic_stats_dict.get('5d', {}).get('ic_positive_ratio', 0) if ic_stats_dict else 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"策略测试失败: {e}")
            return {}
    
    def calculate_expected_performance(self) -> Dict:
        """基于理论计算预期表现"""
        total_ic_squared = 0
        total_weight = 0
        
        for group_name, factors in self.factor_groups.items():
            for factor_name, info in factors.items():
                ic = abs(info['ic'])
                weight = info['weight']
                total_ic_squared += (weight * ic) ** 2
                total_weight += weight
        
        # 假设因子间相关性约0.3（基于经验）
        avg_correlation = 0.3
        n_factors = sum(len(factors) for factors in self.factor_groups.values())
        
        # 组合IC理论估算
        expected_ic = np.sqrt(total_ic_squared + 2 * avg_correlation * total_ic_squared * (n_factors - 1) / n_factors)
        
        return {
            'expected_ic': expected_ic,
            'n_factors': n_factors,
            'total_weight': total_weight
        }
    
    def run_complete_test(self):
        """运行完整测试流程"""
        logger.info("=== 开始优化多因子策略测试 ===")
        
        # 1. 理论预期
        expected = self.calculate_expected_performance()
        logger.info(f"理论预期IC: {expected['expected_ic']:.4f}")
        logger.info(f"因子数量: {expected['n_factors']}")
        
        # 2. 创建组合因子
        combined_factor = self.create_targeted_combination()
        
        if combined_factor is None:
            logger.error("因子组合创建失败")
            return
        
        # 3. 测试表现
        actual_result = self.test_strategy_performance(combined_factor)
        
        # 4. 输出对比结果
        print("\n" + "="*60)
        print("优化多因子策略测试结果")
        print("="*60)
        
        print(f"\n理论预期:")
        print(f"预期IC:     {expected['expected_ic']:.4f}")
        
        if actual_result:
            print(f"\n实际表现:")
            print(f"实际IC:     {actual_result['ic_mean']:.4f}")
            print(f"ICIR:       {actual_result['icir']:.3f}")
            print(f"胜率:       {actual_result['ic_positive_ratio']:.1%}")
            
            # 与单因子对比
            single_factor_avg_ic = 0.019  # 根据你的数据估算
            improvement = (actual_result['ic_mean'] - single_factor_avg_ic) / single_factor_avg_ic if single_factor_avg_ic > 0 else 0
            
            print(f"\n改进效果:")
            print(f"单因子平均IC: {single_factor_avg_ic:.4f}")
            print(f"组合提升:     {improvement:.1%}")
            
            if actual_result['ic_mean'] > single_factor_avg_ic:
                print(f"\n✅ 组合成功！IC提升明显")
            else:
                print(f"\n⚠️  组合效果有待改进")
        
        # 5. 保存结果
        self.save_strategy_results(combined_factor, actual_result, expected)
        
        print("="*60)
    
    def save_strategy_results(self, combined_factor: pd.DataFrame, 
                            actual_result: Dict, expected: Dict):
        """保存策略结果"""
        workspace_dir = Path("../../../workspace")
        workspace_dir.mkdir(exist_ok=True)
        
        # 保存组合因子
        factor_file = workspace_dir / "optimized_multi_factor_strategy.parquet"
        combined_factor.to_parquet(factor_file)
        logger.info(f"优化组合因子已保存: {factor_file}")
        
        # 保存测试结果
        summary = {
            'strategy_type': 'optimized_multi_factor',
            'creation_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'expected_ic': expected.get('expected_ic', 0),
            'actual_ic': actual_result.get('ic_mean', 0) if actual_result else 0,
            'actual_icir': actual_result.get('icir', 0) if actual_result else 0,
            'n_factors': expected.get('n_factors', 0),
            'factor_composition': str(self.factor_groups)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_file = workspace_dir / "strategy_performance_summary.xlsx"
        summary_df.to_excel(summary_file, index=False)
        logger.info(f"策略总结已保存: {summary_file}")


def main():
    """主函数"""
    strategy = OptimizedMultiFactorStrategy()
    strategy.run_complete_test()


if __name__ == "__main__":
    main()