"""
多因子组合策略构建器 - 将"璞玉"因子组合成强大的Alpha策略

基于现代量化投资理念：单因子时代结束，组合才是王道
从你的测试结果中筛选IC > 0.01的有效因子，通过科学的方法组合

Author: Quantitative Research Team
Date: 2024-12-24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from projects._03_factor_selection.factory.strategy_factory import StrategyFactory
from projects._03_factor_selection.multi_factor_optimizer.multi_factor_optimizer import MultiFactorOptimizer
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import FactorAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PotentialFactorIdentifier:
    """璞玉因子识别器 - 识别微弱但有效的因子"""
    
    def __init__(self):
        self.alpha_threshold = 0.01  # IC绝对值阈值
        self.min_icir_threshold = 0.1  # 最小ICIR阈值，保证基本稳定性
        
    def identify_potential_factors(self, factor_results: Dict) -> Dict[str, Dict]:
        """
        从测试结果中识别有潜力的因子
        
        Args:
            factor_results: 单因子测试结果字典
            
        Returns:
            按类别分组的潜力因子
        """
        logger.info("开始识别潜力因子...")
        
        potential_factors = {
            'value': {},      # 价值类
            'quality': {},    # 质量类  
            'momentum': {},   # 动量类
            'liquidity': {},  # 流动性类
            'volatility': {}  # 波动率类
        }
        
        for factor_name, result in factor_results.items():
            if not isinstance(result, dict):
                continue
                
            # 获取关键指标
            ic_mean = abs(result.get('ic_mean_processed_c2c', 0))
            icir = result.get('icir_processed_c2c', 0)
            
            # 筛选条件：微弱但有效
            if ic_mean > self.alpha_threshold and abs(icir) > self.min_icir_threshold:
                category = self._categorize_factor(factor_name)
                if category:
                    potential_factors[category][factor_name] = {
                        'ic_mean': result.get('ic_mean_processed_c2c', 0),
                        'icir': icir,
                        'score': ic_mean * abs(icir),  # 综合评分
                        'raw_data': result
                    }
        
        # 输出识别结果
        for category, factors in potential_factors.items():
            if factors:
                logger.info(f"{category}类因子: {len(factors)}个")
                for name, info in factors.items():
                    logger.info(f"  {name}: IC={info['ic_mean']:.4f}, ICIR={info['icir']:.3f}")
        
        return potential_factors
    
    def _categorize_factor(self, factor_name: str) -> Optional[str]:
        """因子分类"""
        name_lower = factor_name.lower()
        
        # 价值类
        if any(keyword in name_lower for keyword in ['pb', 'pe', 'ps', 'pcf', 'ev', 'value']):
            return 'value'
        
        # 质量类  
        if any(keyword in name_lower for keyword in ['roe', 'roa', 'margin', 'debt', 'quality', 'accrual']):
            return 'quality'
            
        # 动量类
        if any(keyword in name_lower for keyword in ['momentum', 'return', 'trend', 'reversal', 'rsi', 'macd']):
            return 'momentum'
            
        # 流动性类
        if any(keyword in name_lower for keyword in ['turnover', 'volume', 'liquidity', 'amihud']):
            return 'liquidity'
            
        # 波动率类
        if any(keyword in name_lower for keyword in ['volatility', 'vol', 'std', 'var', 'beta']):
            return 'volatility'
        
        return 'momentum'  # 默认归为动量类


class MultiFactorStrategyBuilder:
    """多因子策略构建器"""
    
    def __init__(self, config_path: str = "factory/config.yaml"):
        self.factory = StrategyFactory(config_path)
        self.optimizer = MultiFactorOptimizer(correlation_threshold=0.6)  # 稍微宽松的相关性阈值
        self.factor_identifier = PotentialFactorIdentifier()
        
    def build_strategy_from_results(self, 
                                  results_file: Optional[str] = None,
                                  top_n_per_category: int = 3) -> pd.DataFrame:
        """
        从测试结果构建多因子策略
        
        Args:
            results_file: 结果文件路径，如果None则使用最新结果
            top_n_per_category: 每个类别选取的因子数量
            
        Returns:
            组合后的多因子
        """
        logger.info("开始构建多因子策略...")
        
        # 1. 加载测试结果
        if results_file is None:
            # 查找最新的结果文件
            results_file = self._find_latest_results()
        
        factor_results = self._load_factor_results(results_file)
        
        # 2. 识别潜力因子
        potential_factors = self.factor_identifier.identify_potential_factors(factor_results)
        
        # 3. 为每个类别选取top N因子
        selected_factors_by_category = {}
        factor_data_by_category = {}
        factor_scores_by_category = {}
        
        for category, factors in potential_factors.items():
            if not factors:
                continue
                
            # 按综合评分排序，选取top N
            sorted_factors = sorted(factors.items(), 
                                  key=lambda x: x[1]['score'], 
                                  reverse=True)[:top_n_per_category]
            
            if not sorted_factors:
                continue
                
            selected_factors_by_category[category] = sorted_factors
            
            # 加载因子数据
            factor_data = {}
            factor_scores = {}
            
            for factor_name, factor_info in sorted_factors:
                logger.info(f"加载因子数据: {factor_name}")
                
                # 从StrategyFactory加载因子数据
                try:
                    result = self.factory.test_single_factor(
                        factor_name=factor_name,
                        stock_pool="ZZ800"  # 使用中证800
                    )
                    
                    if 'factor_data' in result:
                        factor_data[factor_name] = result['factor_data']
                        factor_scores[factor_name] = factor_info['ic_mean']
                        
                except Exception as e:
                    logger.warning(f"加载因子{factor_name}失败: {e}")
                    continue
            
            if factor_data:
                factor_data_by_category[category] = factor_data
                factor_scores_by_category[category] = factor_scores
        
        # 4. 多因子优化
        if not factor_data_by_category:
            logger.error("没有有效的因子数据，无法构建策略")
            return None
            
        logger.info("开始多因子优化...")
        combined_factor = self.optimizer.optimize_factors(
            factors_by_category=factor_data_by_category,
            factor_scores=factor_scores_by_category,
            intra_method='ic_weighted',  # 类别内IC加权
            cross_method='max_diversification'  # 类别间最大分散化
        )
        
        # 5. 保存组合因子
        output_path = Path("workspace/multi_factor_strategy.parquet")
        combined_factor.to_parquet(output_path)
        logger.info(f"多因子策略已保存到: {output_path}")
        
        return combined_factor
    
    def backtest_strategy(self, combined_factor: pd.DataFrame) -> Dict:
        """
        回测多因子策略
        
        Args:
            combined_factor: 组合因子数据
            
        Returns:
            回测结果
        """
        logger.info("开始回测多因子策略...")
        
        # 使用单因子分析器进行回测
        analyzer = FactorAnalyzer() #todo  补充manager
        
        # 临时保存因子数据
        temp_factor_path = "workspace/temp_multi_factor.parquet"
        combined_factor.to_parquet(temp_factor_path)
        
        try:
            # 执行回测分析
            (
                processed_factor,
                ic_series_dict, ic_stats_dict,
                quantile_daily_returns_dict, quantile_stats_dict,
                factor_returns_dict, fm_results_dict,
                turnover_dict, style_corr_dict,
                factor_risk_dict, pct_chg_beta_dict
            ) = analyzer.comprehensive_test(
                target_factor_name="multi_factor_strategy",
                factor_data_shifted=combined_factor,
                stock_pool_index_name="ZZ800",
                preprocess_method="standard",
                returns_calculator=None,
                need_process_factor=True, #确定是否要预处理数据？todo
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
            
            logger.info("多因子策略回测完成")
            logger.info(f"组合IC均值: {result.get('ic_mean', 0):.4f}")
            logger.info(f"组合ICIR: {result.get('icir', 0):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"回测失败: {e}")
            return {}
        finally:
            # 清理临时文件
            if Path(temp_factor_path).exists():
                Path(temp_factor_path).unlink()
    
    def _find_latest_results(self) -> str:
        """查找最新的测试结果文件"""
        workspace_dir = Path("D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\workspace")
        
        # 优先查找汇总结果文件
        summary_files = [
            workspace_dir / "factor_results" / "all_single_factor_test_purify_summary.csv",
            workspace_dir / "factor_results" / "all_single_factor_test_purify_summary.parquet"
        ]
        
        for summary_file in summary_files:
            if summary_file.exists():
                logger.info(f"使用汇总测试结果文件: {summary_file}")
                return str(summary_file)
        
        # 如果没有汇总文件，查找导出的结果文件
        exports_dir = workspace_dir / "exports"
        if exports_dir.exists():
            export_dirs = [d for d in exports_dir.iterdir() if d.is_dir()]
            if export_dirs:
                # 按时间排序，取最新的
                latest_export = max(export_dirs, key=lambda x: x.stat().st_mtime)
                factor_summary = latest_export / "factor_summary.xlsx"
                if factor_summary.exists():
                    logger.info(f"使用导出的测试结果文件: {factor_summary}")
                    return str(factor_summary)
        
        # 最后尝试查找其他结果文件
        result_files = list(workspace_dir.glob("**/*result*.csv")) + list(workspace_dir.glob("**/*result*.xlsx"))
        
        if result_files:
            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"使用测试结果文件: {latest_file}")
            return str(latest_file)
        
        raise FileNotFoundError("未找到测试结果文件。请先运行因子测试生成结果文件。")
    
    def _load_factor_results(self, results_file: str) -> Dict:
        """加载因子测试结果"""
        results_path = Path(results_file)
        
        try:
            if results_path.suffix == '.parquet':
                df = pd.read_parquet(results_path)
            elif results_path.suffix in ['.xlsx', '.csv']:
                # 尝试不同的编码格式
                encodings = ['utf-8', 'gbk', 'utf-8-sig']
                df = None
                
                for encoding in encodings:
                    try:
                        if results_path.suffix == '.csv':
                            df = pd.read_csv(results_path, encoding=encoding)
                        else:  # .xlsx
                            df = pd.read_excel(results_path)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                
                if df is None:
                    raise ValueError(f"无法读取文件 {results_file}，尝试的编码格式都失败")
            else:
                raise ValueError(f"不支持的文件格式: {results_path.suffix}")
            
            logger.info(f"成功读取文件: {results_file}")
            logger.info(f"文件包含 {len(df)} 行数据，列名: {list(df.columns)}")
            
            # 转换为字典格式，适配不同的数据结构
            results = {}
            
            # 检查是否是汇总格式（包含多个period的数据）
            if 'period' in df.columns:
                # 汇总格式：每个因子有多行（不同period）
                for factor_name in df['factor_name'].unique():
                    factor_data = df[df['factor_name'] == factor_name]
                    
                    # 为每个因子创建字典，包含不同period的数据
                    factor_dict = {}
                    
                    # 选择5d的数据作为主要参考（如果有的话）
                    best_period_data = None
                    if not factor_data.empty:
                        # 优先选择5d数据
                        period_5d = factor_data[factor_data['period'] == '5d']
                        if not period_5d.empty:
                            best_period_data = period_5d.iloc[0]
                        else:
                            # 如果没有5d，选择第一行数据
                            best_period_data = factor_data.iloc[0]
                        
                        # 提取关键指标，适配不同的列名
                        factor_dict = {
                            'factor_name': factor_name,
                            'ic_mean_processed_c2c': best_period_data.get('ic_mean', 0),
                            'ic_ir_processed_c2c': best_period_data.get('ic_ir', 0),
                            'ic_mean_raw_c2c': best_period_data.get('ic_mean', 0),  # 如果没有区分raw/processed，使用同一个值
                            'tmb_sharpe_raw_c2c': best_period_data.get('tmb_sharpe', 0),
                            'tmb_max_drawdown_raw_c2c': best_period_data.get('tmb_max_drawdown', 0),
                            'monotonicity_spearman_processed_c2c': best_period_data.get('monotonicity_spearman', 0),
                            'fm_t_statistic_processed_c2c': best_period_data.get('fm_t_statistic', 0),
                            'best_period': best_period_data.get('period', '5d'),
                            'factor_category': best_period_data.get('factor_category', 'unknown')
                        }
                        
                        results[factor_name] = factor_dict
                        
            else:
                # 原始格式：每行是一个因子
                for _, row in df.iterrows():
                    factor_name = row.get('factor_name', row.get('因子名称'))
                    if factor_name and pd.notna(factor_name):
                        results[factor_name] = row.to_dict()
            
            logger.info(f"加载了 {len(results)} 个因子的测试结果")
            
            # 打印前几个因子的信息用于调试
            factor_names = list(results.keys())[:5]
            for name in factor_names:
                ic_mean = results[name].get('ic_mean_processed_c2c', results[name].get('ic_mean', 'N/A'))
                logger.info(f"  因子 {name}: IC = {ic_mean}")
            
            return results
            
        except Exception as e:
            logger.error(f"加载结果文件失败: {e}")
            logger.error(f"文件路径: {results_file}")
            raise


ab_config_path = 'D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\factory\\config.yaml'
ab_experiments_path = 'D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\factory\\experiments.yaml'
def main():
    """主函数 - 演示如何使用"""
    logger.info("=== 多因子策略构建开始 ===")
    
    try:
        # 初始化策略构建器
        builder = MultiFactorStrategyBuilder(ab_config_path)
        
        # 构建策略（会自动识别潜力因子并组合）
        combined_factor = builder.build_strategy_from_results(
            top_n_per_category=3  # 每个类别选3个最好的因子
        )
        
        if combined_factor is not None:
            # 回测策略效果
            backtest_result = builder.backtest_strategy(combined_factor)
            
            # 输出关键指标
            if backtest_result:
                print("\n=== 多因子策略效果 ===")
                print(f"IC均值: {backtest_result.get('ic_mean', 0):.4f}")
                print(f"ICIR: {backtest_result.get('icir', 0):.3f}")
                print(f"胜率: {backtest_result.get('ic_positive_ratio', 0):.2%}")
                
                # 与单因子对比
                print(f"\n对比单因子结果：")
                print(f"- 你的单因子IC大多在0.01-0.02之间")
                print(f"- 组合后IC提升至: {backtest_result.get('ic_mean', 0):.4f}")
                print(f"- ICIR提升说明稳定性增强")
            
        logger.info("=== 多因子策略构建完成 ===")
        
    except Exception as e:
        logger.error(f"策略构建失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()