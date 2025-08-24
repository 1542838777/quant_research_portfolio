"""
快速多因子组合测试 - 验证"璞玉"因子组合的威力

根据你的实际测试结果，选择几个典型的微弱Alpha因子进行组合测试
证明：1+1 > 2 的组合效应

Usage:
    python quick_multi_factor_test.py
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Dict

from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
from projects._03_factor_selection.factory.strategy_factory import StrategyFactory
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuickMultiFactorTester:
    """快速多因子测试器"""
    
    def __init__(self):
        self.factory = StrategyFactory("D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\factory\\config.yaml")
        f_manager = FactorManager()
        self.analyzer = FactorAnalyzer(f_manager)
        
        # 根据你的测试结果，选择这些"璞玉"因子
        self.selected_factors = [
            'volatility_120d',      # 风险因子，IC约-0.02，稳定的负alpha
            'volatility_90d',       # 风险因子，与120d相关但时间窗口不同
            'turnover_rate_monthly_mean',  # 流动性因子
            'rsi',                  # 技术指标，短期反转
            'cci'                   # 技术指标，补充RSI
        ]
        
        self.stock_pool = "ZZ800"
    
    def test_individual_factors(self) -> Dict[str, Dict]:
        """测试各个单因子的表现"""
        logger.info("开始测试单个因子表现...")
        
        individual_results = {}
        
        for factor_name in self.selected_factors:
            logger.info(f"测试因子: {factor_name}")
            
            try:
                result = self.factory.test_single_factor(
                    factor_name=factor_name,
                    stock_pool=self.stock_pool
                )
                
                if 'evaluation_result' in result:
                    eval_result = result['evaluation_result']
                    individual_results[factor_name] = {
                        'ic_mean': eval_result.get('ic_mean', 0),
                        'icir': eval_result.get('icir', 0),
                        'ic_positive_ratio': eval_result.get('ic_positive_ratio', 0),
                        'factor_data': result.get('factor_data')
                    }
                    
                    logger.info(f"  IC: {eval_result.get('ic_mean', 0):.4f}, "
                              f"ICIR: {eval_result.get('icir', 0):.3f}")
                
            except Exception as e:
                logger.warning(f"测试{factor_name}失败: {e}")
                continue
        
        return individual_results
    
    def create_simple_combination(self, factor_results: Dict[str, Dict]) -> pd.DataFrame:
        """创建简单的因子组合"""
        logger.info("开始创建因子组合...")
        
        # 方法1: IC加权组合
        combined_factor = None
        total_abs_ic = sum(abs(info['ic_mean']) for info in factor_results.values())
        
        if total_abs_ic == 0:
            logger.warning("所有因子IC均为0，使用等权重")
            weight_per_factor = 1.0 / len(factor_results)
            weights = {name: weight_per_factor for name in factor_results.keys()}
        else:
            weights = {name: abs(info['ic_mean']) / total_abs_ic 
                      for name, info in factor_results.items()}
        
        logger.info("因子权重分配:")
        for name, weight in weights.items():
            ic = factor_results[name]['ic_mean']
            logger.info(f"  {name}: {weight:.3f} (IC: {ic:.4f})")
        
        # 组合因子数据
        for factor_name, info in factor_results.items():
            factor_data = info['factor_data']
            weight = weights[factor_name]
            
            if factor_data is not None:
                # 处理因子方向（负IC因子需要反转）
                if info['ic_mean'] < 0:
                    factor_data = -factor_data
                    logger.info(f"  {factor_name} IC为负，已反转")
                
                weighted_factor = factor_data * weight
                
                if combined_factor is None:
                    combined_factor = weighted_factor
                else:
                    combined_factor = combined_factor.add(weighted_factor, fill_value=0)
        
        logger.info("因子组合完成")
        return combined_factor
    
    def test_combined_factor(self, combined_factor: pd.DataFrame) -> Dict:
        """测试组合因子效果"""
        logger.info("开始测试组合因子...")
        
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
                target_factor_name="multi_factor_combination",
                factor_data_shifted=combined_factor,
                stock_pool_index_name=self.stock_pool,
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
            logger.error(f"组合因子测试失败: {e}")
            return {}
    
    def run_comparison_test(self):
        """运行对比测试"""
        logger.info("=== 开始快速多因子对比测试 ===")
        
        # 1. 测试单个因子
        individual_results = self.test_individual_factors()
        
        if not individual_results:
            logger.error("没有有效的单因子结果")
            return
        
        # 2. 创建组合
        combined_factor = self.create_simple_combination(individual_results)
        
        # 3. 测试组合效果
        combined_result = self.test_combined_factor(combined_factor)
        
        # 4. 输出对比结果
        print("\\n" + "="*60)
        print("多因子组合效果对比")
        print("="*60)
        
        print("\\n单因子表现:")
        print("-"*40)
        for name, result in individual_results.items():
            print(f"{name:25} | IC: {result['ic_mean']:6.4f} | ICIR: {result['icir']:6.3f}")
        
        # 计算单因子平均效果
        avg_ic = np.mean([r['ic_mean'] for r in individual_results.values()])
        avg_icir = np.mean([r['icir'] for r in individual_results.values()])
        
        print(f"\\n单因子平均水平:")
        print(f"平均IC:   {avg_ic:6.4f}")
        print(f"平均ICIR: {avg_icir:6.3f}")
        
        if combined_result:
            combined_ic = combined_result.get('ic_mean', 0)
            combined_icir = combined_result.get('icir', 0)
            
            print(f"\\n组合因子表现:")
            print("-"*40)
            print(f"组合IC:   {combined_ic:6.4f}")
            print(f"组合ICIR: {combined_icir:6.3f}")
            
            # 计算提升效果
            ic_improvement = (combined_ic - avg_ic) / abs(avg_ic) if avg_ic != 0 else 0
            icir_improvement = (combined_icir - avg_icir) / abs(avg_icir) if avg_icir != 0 else 0
            
            print(f"\\n组合效果:")
            print("-"*40)
            print(f"IC提升:   {ic_improvement:6.1%}")
            print(f"ICIR提升: {icir_improvement:6.1%}")
            
            if combined_ic > avg_ic:
                print(f"\\n✅ 组合成功！IC从 {avg_ic:.4f} 提升到 {combined_ic:.4f}")
            else:
                print(f"\\n⚠️  组合效果一般，可能需要调整因子选择或权重")
            
            # 保存结果
            self.save_results(individual_results, combined_result, combined_factor)
        
        print("="*60)
    
    def save_results(self, individual_results: Dict, combined_result: Dict, combined_factor: pd.DataFrame):
        """保存测试结果"""
        workspace_dir = Path("../../../workspace")
        workspace_dir.mkdir(exist_ok=True)
        
        # 保存组合因子数据
        factor_file = workspace_dir / "quick_multi_factor_combination.parquet"
        combined_factor.to_parquet(factor_file)
        logger.info(f"组合因子数据已保存: {factor_file}")
        
        # 保存对比结果
        comparison_data = []
        
        # 添加单因子结果
        for name, result in individual_results.items():
            comparison_data.append({
                'factor_name': name,
                'type': 'single',
                'ic_mean': result['ic_mean'],
                'icir': result['icir'],
                'ic_positive_ratio': result['ic_positive_ratio']
            })
        
        # 添加组合结果
        if combined_result:
            comparison_data.append({
                'factor_name': 'multi_factor_combination',
                'type': 'combined',
                'ic_mean': combined_result.get('ic_mean', 0),
                'icir': combined_result.get('icir', 0),
                'ic_positive_ratio': combined_result.get('ic_positive_ratio', 0)
            })
        
        # 保存为Excel
        comparison_df = pd.DataFrame(comparison_data)
        excel_file = workspace_dir / "multi_factor_comparison.xlsx"
        comparison_df.to_excel(excel_file, index=False)
        logger.info(f"对比结果已保存: {excel_file}")


def main():
    """主函数"""
    tester = QuickMultiFactorTester()
    tester.run_comparison_test()


if __name__ == "__main__":
    main()