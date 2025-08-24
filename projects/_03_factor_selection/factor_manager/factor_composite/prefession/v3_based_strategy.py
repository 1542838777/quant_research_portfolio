"""
基于v3未经过残差化版本.csv的专门优化策略
直接使用你已有的测试结果，挑选最优的低相关性因子组合

从你的v3测试结果中筛选出的最佳因子：
- earnings_stability: IC=0.0164, ICIR=0.284 (A级质量因子)  
- amihud_liquidity: IC=0.0186, ICIR=0.240 (A级流动性因子)
- volatility_40d: IC=-0.0287, ICIR=-0.274 (A级风险因子，反向)
- ep_ratio: IC=0.0166, ICIR=0.242 (B级价值因子)
- rsi: IC=-0.0152, ICIR=-0.157 (B级技术因子，反向)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional

# 添加项目路径
import sys

from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_composite.prefession.create_multi_factor_strategy import \
    ab_config_path, ab_experiments_path

sys.path.append(str(Path(__file__).parent.parent.parent))

from projects._03_factor_selection.factory.strategy_factory import StrategyFactory
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class V3BasedStrategy:
    """基于v3测试结果的精选策略"""
    
    def __init__(self):
        config_path = "D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\factory\\config.yaml"
        self.factory = StrategyFactory(config_path)

        data_manager = DataManager(config_path=ab_config_path,
                                   experiments_config_path=ab_experiments_path)
        # data_manager.prepare_basic_data()
        f_manager = FactorManager(data_manager)

        self.analyzer = FactorAnalyzer(f_manager)
        
        # 从你的v3结果中精选的因子组合（已验证有效）
        self.selected_factors = {
            'earnings_stability': {
                'ic': 0.01643, 'icir': 0.284, 'direction': 1, 'weight': 0.25,
                'category': 'quality', 'grade': 'A'
            },
            'amihud_liquidity': {
                'ic': 0.01857, 'icir': 0.240, 'direction': 1, 'weight': 0.25,  
                'category': 'liquidity', 'grade': 'A'
            },
            'volatility_40d': {
                'ic': -0.02865, 'icir': -0.274, 'direction': -1, 'weight': 0.25,
                'category': 'risk', 'grade': 'A'
            },
            'ep_ratio': {
                'ic': 0.01656, 'icir': 0.242, 'direction': 1, 'weight': 0.15,
                'category': 'value', 'grade': 'B'
            },
            'rsi': {
                'ic': -0.01524, 'icir': -0.157, 'direction': -1, 'weight': 0.1,
                'category': 'technical', 'grade': 'B'  
            }
        }
        
        self.stock_pool = "ZZ800"  # 中证800
        
    def calculate_theoretical_performance(self) -> Dict:
        """计算理论预期表现"""
        logger.info("计算理论预期表现...")
        
        # 计算加权IC
        total_weight = sum(info['weight'] for info in self.selected_factors.values())
        weighted_ic_sum = sum(abs(info['ic']) * info['weight'] for info in self.selected_factors.values())
        avg_weighted_ic = weighted_ic_sum / total_weight
        
        # 计算加权ICIR  
        weighted_icir_sum = sum(abs(info['icir']) * info['weight'] for info in self.selected_factors.values())
        avg_weighted_icir = weighted_icir_sum / total_weight
        
        # 假设因子间相关性（基于不同类别）
        correlation_estimate = 0.25  # 低相关性假设
        n_factors = len(self.selected_factors)
        
        # 组合IC的理论估计（考虑相关性）
        variance_sum = sum((info['weight'] * abs(info['ic'])) ** 2 for info in self.selected_factors.values())
        
        # 简化的相关性影响
        correlation_effect = 2 * correlation_estimate * sum(
            self.selected_factors[f1]['weight'] * abs(self.selected_factors[f1]['ic']) * 
            self.selected_factors[f2]['weight'] * abs(self.selected_factors[f2]['ic'])
            for i, f1 in enumerate(self.selected_factors.keys())
            for f2 in list(self.selected_factors.keys())[i+1:]
        )
        
        expected_ic = np.sqrt(variance_sum + correlation_effect)
        
        return {
            'expected_ic': expected_ic,
            'avg_weighted_ic': avg_weighted_ic,
            'avg_weighted_icir': avg_weighted_icir,
            'n_factors': n_factors,
            'correlation_assumption': correlation_estimate
        }
        
    def create_factor_combination(self) -> Optional[pd.DataFrame]:
        """创建因子组合"""
        logger.info("开始创建V3因子组合...")
        
        factor_data_dict = {}
        
        # 加载每个因子的数据
        for factor_name, factor_info in self.selected_factors.items():
            try:
                logger.info(f"加载因子: {factor_name}")
                
                # 使用StrategyFactory加载因子数据
                result = self.factory.test_single_factor(
                    factor_name=factor_name,
                    stock_pool=self.stock_pool
                )
                
                if 'factor_data' in result and result['factor_data'] is not None:
                    factor_data = result['factor_data'].copy()
                    
                    # 处理方向
                    if factor_info['direction'] == -1:
                        factor_data = -factor_data
                        logger.info(f"  {factor_name} 已反转方向")
                    
                    # 标准化
                    factor_data = factor_data.fillna(0)
                    factor_mean = factor_data.mean()
                    factor_std = factor_data.std()
                    
                    if factor_std > 1e-8:
                        factor_data = (factor_data - factor_mean) / factor_std
                    
                    factor_data_dict[factor_name] = {
                        'data': factor_data,
                        'weight': factor_info['weight'],
                        'ic': abs(factor_info['ic']),
                        'category': factor_info['category']
                    }
                    
                    logger.info(f"  成功加载，权重: {factor_info['weight']:.2f}")
                    
                else:
                    logger.warning(f"  {factor_name}: 无法获取因子数据")
                    
            except Exception as e:
                logger.warning(f"跳过因子 {factor_name}: {e}")
                continue
                
        if not factor_data_dict:
            logger.error("没有成功加载任何因子数据")
            return None
            
        # 组合因子
        combined_factor = None
        total_weight = sum(info['weight'] for info in factor_data_dict.values())
        
        logger.info("因子权重分配:")
        for factor_name, info in factor_data_dict.items():
            normalized_weight = info['weight'] / total_weight
            weighted_factor = info['data'] * normalized_weight
            
            if combined_factor is None:
                combined_factor = weighted_factor
            else:
                combined_factor = combined_factor.add(weighted_factor, fill_value=0)
                
            logger.info(f"  {factor_name} ({info['category']}): {normalized_weight:.3f}")
            
        # 最终标准化  
        if combined_factor is not None:
            combined_mean = combined_factor.mean()
            combined_std = combined_factor.std()
            
            if combined_std > 1e-8:
                combined_factor = (combined_factor - combined_mean) / combined_std
                
        logger.info("V3因子组合创建完成")
        return combined_factor
        
    def test_strategy(self, combined_factor: pd.DataFrame) -> Dict:
        """测试策略表现"""
        logger.info("开始测试V3策略表现...")
        
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
                target_factor_name="v3_optimized_strategy",
                factor_data_shifted=combined_factor,
                stock_pool_index_name=self.stock_pool,
                preprocess_method="standard",
                returns_calculator=None,
                need_process_factor=True,
                do_ic_test=True,
                do_quantile_test=True,
                do_fama_test=True
            )
            
            # 提取结果
            result = {}
            if ic_stats_dict and '5d' in ic_stats_dict:
                stats_5d = ic_stats_dict['5d']
                result = {
                    'ic_mean': stats_5d.get('ic_mean', 0),
                    'icir': stats_5d.get('ic_ir', 0),
                    'ic_positive_ratio': stats_5d.get('ic_positive_ratio', 0),
                    'ic_std': stats_5d.get('ic_std', 0)
                }
                
            return result
            
        except Exception as e:
            logger.error(f"策略测试失败: {e}")
            return {}
            
    def run_complete_strategy(self):
        """运行完整的策略测试"""
        logger.info("=== V3优化策略测试开始 ===")
        
        # 1. 理论预期
        theoretical = self.calculate_theoretical_performance()
        logger.info(f"理论预期IC: {theoretical['expected_ic']:.4f}")
        
        # 2. 创建组合
        combined_factor = self.create_factor_combination()
        if combined_factor is None:
            return
            
        # 3. 测试表现
        actual_result = self.test_strategy(combined_factor)
        
        # 4. 输出详细结果
        print("\\n" + "="*70)
        print("V3因子优化策略测试结果")
        print("="*70)
        
        print(f"\\n选中的因子组合:")
        print("-"*50)
        for name, info in self.selected_factors.items():
            direction = "正向" if info['direction'] == 1 else "反向"  
            print(f"{name:20} | IC:{info['ic']:7.4f} | ICIR:{info['icir']:6.3f} | {direction} | {info['grade']}级")
            
        print(f"\\n理论预期:")
        print("-"*50)
        print(f"预期IC:        {theoretical['expected_ic']:.4f}")
        print(f"加权平均IC:    {theoretical['avg_weighted_ic']:.4f}")
        print(f"加权平均ICIR:  {theoretical['avg_weighted_icir']:.3f}")
        
        if actual_result:
            print(f"\\n实际表现:")
            print("-"*50)
            print(f"实际IC:        {actual_result['ic_mean']:.4f}")
            print(f"实际ICIR:      {actual_result['icir']:.3f}")
            print(f"IC胜率:        {actual_result['ic_positive_ratio']:.1%}")
            print(f"IC标准差:      {actual_result.get('ic_std', 0):.4f}")
            
            # 与理论对比
            ic_vs_theory = actual_result['ic_mean'] / theoretical['expected_ic'] if theoretical['expected_ic'] > 0 else 0
            
            print(f"\\n策略评估:")
            print("-"*50)
            print(f"实际/理论IC:   {ic_vs_theory:.1%}")
            
            # 与单因子平均对比
            avg_single_ic = np.mean([abs(info['ic']) for info in self.selected_factors.values()])
            improvement = (abs(actual_result['ic_mean']) - avg_single_ic) / avg_single_ic if avg_single_ic > 0 else 0
            
            print(f"单因子平均IC:  {avg_single_ic:.4f}")
            print(f"组合提升率:    {improvement:.1%}")
            
            if abs(actual_result['ic_mean']) > avg_single_ic:
                print(f"\\n✅ 策略成功！组合效果优于单因子平均水平")
            else:
                print(f"\\n⚠️  策略需要优化，考虑调整权重或因子选择")
                
        # 5. 保存结果
        self.save_strategy_results(combined_factor, actual_result, theoretical)
        
        print("="*70)
        
    def save_strategy_results(self, combined_factor: pd.DataFrame, 
                            actual_result: Dict, theoretical: Dict):
        """保存策略结果"""
        workspace_dir = Path("D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\workspace")
        workspace_dir.mkdir(exist_ok=True)
        
        # 保存组合因子
        factor_file = workspace_dir / "v3_optimized_strategy.parquet"
        combined_factor.to_parquet(factor_file)
        logger.info(f"V3策略因子已保存: {factor_file}")
        
        # 保存详细结果
        results_summary = {
            'strategy_name': 'V3_Optimized_Strategy',
            'creation_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'factor_count': len(self.selected_factors),
            'selected_factors': list(self.selected_factors.keys()),
            'factor_weights': {k: v['weight'] for k, v in self.selected_factors.items()},
            'theoretical_ic': theoretical.get('expected_ic', 0),
            'actual_ic': actual_result.get('ic_mean', 0) if actual_result else 0,
            'actual_icir': actual_result.get('icir', 0) if actual_result else 0,
            'ic_positive_ratio': actual_result.get('ic_positive_ratio', 0) if actual_result else 0
        }
        
        results_df = pd.DataFrame([results_summary])
        results_file = workspace_dir / "v3_strategy_results.xlsx"
        results_df.to_excel(results_file, index=False)
        logger.info(f"策略结果已保存: {results_file}")


def main():
    """主函数"""
    strategy = V3BasedStrategy()
    strategy.run_complete_strategy()


if __name__ == "__main__":
    main()