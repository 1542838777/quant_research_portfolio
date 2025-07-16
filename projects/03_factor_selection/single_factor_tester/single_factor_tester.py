"""
单因子测试框架 - 专业版

实现华泰证券标准的三种单因子测试方法：
1. IC值分析法
2. 分层回测法  
3. Fama-MacBeth回归法（黄金标准）

支持批量测试、结果可视化和报告生成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
import os
import json
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from quant_lib.evaluation import (
    calculate_ic_vectorized,
    calculate_quantile_returns,
    run_fama_macbeth_regression
)

# 导入新的可视化管理器
try:
    from visualization_manager import VisualizationManager
except ImportError:
    VisualizationManager = None

warnings.filterwarnings('ignore')

class SingleFactorTester:
    """
    单因子测试器 - 专业版
    
    按照华泰证券标准实现三种测试方法的完整流程
    """
    
    def __init__(self,
                 data_dict: Dict[str, pd.DataFrame] = None,
                 price_data: pd.DataFrame = None,
                 config: Dict[str, Any] = None,
                 output_dir: str = "results/single_factor_tests"):
        """
        初始化单因子测试器 - 兼容新架构

        Args:
            data_dict: 数据字典（新架构）
            price_data: 价格数据（向后兼容）
            config: 配置字典
            output_dir: 结果输出目录
        """
        # 处理配置
        if config is None:
            config = {}

        self.config = config
        self.test_periods = config.get('forward_periods', [5, 10, 20])
        self.n_quantiles = config.get('quantiles', 5)
        self.output_dir = output_dir

        # 处理数据
        if data_dict is not None:
            self.data_dict = data_dict
            self.price_data = data_dict.get('close_price', data_dict.get('price'))
        elif price_data is not None:
            self.price_data = price_data
            self.data_dict = {'price': price_data}
        else:
            raise ValueError("必须提供 data_dict 或 price_data")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化可视化管理器
        if VisualizationManager is not None:
            self.viz_manager = VisualizationManager(
                output_dir=os.path.join(output_dir, "visualizations")
            )
        else:
            self.viz_manager = None

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def preprocess_factor(self, 
                         factor_data: pd.DataFrame,
                         method: str = "standard") -> pd.DataFrame:
        """
        因子预处理 - 按华泰证券标准
        
        Args:
            factor_data: 原始因子数据
            method: 预处理方法，"standard"或"robust"
            
        Returns:
            预处理后的因子数据
        """
        processed_factor = factor_data.copy()
        
        if method == "standard":
            # 标准方法：中位数去极值 + 标准化
            for date in processed_factor.index:
                values = processed_factor.loc[date].dropna()
                if len(values) > 10:
                    # 中位数去极值
                    DM = values.median()
                    DM1 = (values - DM).abs().median()
                    upper_bound = DM + 5 * DM1
                    lower_bound = DM - 5 * DM1
                    processed_factor.loc[date] = values.clip(lower_bound, upper_bound)
                    
                    # 标准化
                    values_clipped = processed_factor.loc[date].dropna()
                    if values_clipped.std() > 0:
                        mean_val = values_clipped.mean()
                        std_val = values_clipped.std()
                        processed_factor.loc[date, values_clipped.index] = (values_clipped - mean_val) / std_val
                        
        elif method == "robust":
            # 稳健方法：分位数去极值 + 标准化
            for date in processed_factor.index:
                values = processed_factor.loc[date].dropna()
                if len(values) > 10:
                    # 1%-99%分位数去极值
                    lower_bound = values.quantile(0.01)
                    upper_bound = values.quantile(0.99)
                    processed_factor.loc[date] = values.clip(lower_bound, upper_bound)
                    
                    # 标准化
                    values_clipped = processed_factor.loc[date].dropna()
                    if values_clipped.std() > 0:
                        mean_val = values_clipped.mean()
                        std_val = values_clipped.std()
                        processed_factor.loc[date, values_clipped.index] = (values_clipped - mean_val) / std_val
        
        return processed_factor
    
    def test_ic_analysis(self, 
                        factor_data: pd.DataFrame,
                        factor_name: str) -> Dict[str, Any]:
        """
        IC值分析法测试
        
        Args:
            factor_data: 预处理后的因子数据
            factor_name: 因子名称
            
        Returns:
            IC分析结果字典
        """
        ic_results = {}
        
        for period in self.test_periods:
            # 计算未来收益
            forward_returns = self.price_data.shift(-period) / self.price_data - 1
            
            # 计算IC序列（使用Spearman相关系数）
            ic_series = calculate_ic_vectorized(factor_data, forward_returns, method='spearman')
            
            # 计算IC统计指标
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_ratio = (ic_series > 0).mean()
            
            ic_results[f'{period}d'] = {
                'ic_series': ic_series,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_ir': ic_ir,
                'ic_positive_ratio': ic_positive_ratio,
                'ic_abs_mean': ic_series.abs().mean()
            }
        
        return ic_results
    
    def test_quantile_backtest(self, 
                              factor_data: pd.DataFrame,
                              factor_name: str) -> Dict[str, Any]:
        """
        分层回测法测试
        
        Args:
            factor_data: 预处理后的因子数据
            factor_name: 因子名称
            
        Returns:
            分层回测结果字典
        """
        quantile_results = {}
        
        for period in self.test_periods:
            # 进行分层回测
            backtest_results = calculate_quantile_returns(
                factor_data,
                self.price_data,
                n_quantiles=self.n_quantiles,
                forward_periods=[period]
            )
            
            # 分析结果
            returns_data = backtest_results[period]
            mean_returns = returns_data.mean()
            
            # 计算统计指标
            tmb_series = returns_data['TopMinusBottom']
            tmb_sharpe = tmb_series.mean() / tmb_series.std() * np.sqrt(252) if tmb_series.std() > 0 else 0
            tmb_win_rate = (tmb_series > 0).mean()
            
            # 单调性检验
            quantile_means = [mean_returns[f'Q{i}'] for i in range(1, self.n_quantiles + 1)]
            is_monotonic = all(quantile_means[i] <= quantile_means[i+1] for i in range(len(quantile_means)-1))
            
            quantile_results[f'{period}d'] = {
                'returns_data': returns_data,
                'mean_returns': mean_returns,
                'tmb_return': mean_returns['TopMinusBottom'],
                'tmb_sharpe': tmb_sharpe,
                'tmb_win_rate': tmb_win_rate,
                'is_monotonic': is_monotonic,
                'quantile_means': quantile_means
            }
        
        return quantile_results
    
    def test_fama_macbeth(self, 
                         factor_data: pd.DataFrame,
                         factor_name: str) -> Dict[str, Any]:
        """
        Fama-MacBeth回归法测试（黄金标准）
        
        Args:
            factor_data: 预处理后的因子数据
            factor_name: 因子名称
            
        Returns:
            Fama-MacBeth回归结果字典
        """
        fm_results = {}
        
        for period in self.test_periods:
            # 运行Fama-MacBeth回归
            fm_result = run_fama_macbeth_regression(
                factor_df=factor_data,
                price_df=self.price_data,
                forward_returns_period=period
                # weights_df=weights_df,  # <-- 传入流通市值作为权重，执行WLS
                # neutral_factors=neutral_factors  # <-- 传入市值和行业作为控制变量
            )
            
            # 添加显著性评级
            t_stat = fm_result['t_statistic']
            if abs(t_stat) > 2.58:
                significance_level = "***"
                significance_desc = "1%显著"
            elif abs(t_stat) > 1.96:
                significance_level = "**"
                significance_desc = "5%显著"
            elif abs(t_stat) > 1.64:
                significance_level = "*"
                significance_desc = "10%显著"
            else:
                significance_level = ""
                significance_desc = "不显著"
            
            fm_result['significance_level'] = significance_level
            fm_result['significance_desc'] = significance_desc
            
            fm_results[f'{period}d'] = fm_result
        
        return fm_results

    def comprehensive_test(self,
                          factor_data: pd.DataFrame,
                          factor_name: str,
                          preprocess_method: str = "standard",
                          save_results: bool = True) -> Dict[str, Any]:
        """
        综合测试 - 执行所有三种测试方法

        Args:
            factor_data: 原始因子数据
            factor_name: 因子名称
            preprocess_method: 预处理方法
            save_results: 是否保存结果

        Returns:
            综合测试结果字典
        """
        print(f"\n{'='*80}")
        print(f"开始测试因子: {factor_name}")
        print(f"{'='*80}")

        # 1. 因子预处理
        print("1. 因子预处理...")
        factor_processed = self.preprocess_factor(factor_data, method=preprocess_method)

        # 2. IC值分析
        print("2. IC值分析...")
        ic_results = self.test_ic_analysis(factor_processed, factor_name)

        # 3. 分层回测
        print("3. 分层回测...")
        quantile_results = self.test_quantile_backtest(factor_processed, factor_name)

        # 4. Fama-MacBeth回归
        print("4. Fama-MacBeth回归...")
        fm_results = self.test_fama_macbeth(factor_processed, factor_name)

        # 5. 综合评价
        print("5. 综合评价...")
        evaluation = self._evaluate_factor(ic_results, quantile_results, fm_results)

        # 整合结果
        comprehensive_results = {
            'factor_name': factor_name,
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'preprocess_method': preprocess_method,
            'ic_analysis': ic_results,
            'quantile_backtest': quantile_results,
            'fama_macbeth': fm_results,
            'evaluation': evaluation
        }

        # 6. 生成报告和可视化
        if save_results:
            self._save_results(comprehensive_results, factor_name)
            self._create_visualizations(comprehensive_results, factor_name)
            self._generate_report(comprehensive_results, factor_name)

        # 7. 打印摘要
        self._print_summary(comprehensive_results)

        return comprehensive_results

    def _evaluate_factor(self,
                        ic_results: Dict,
                        quantile_results: Dict,
                        fm_results: Dict) -> Dict[str, Any]:
        """
        综合评价因子表现
        """
        # 以20日为主要评价周期
        main_period = '20d'

        # IC评价
        ic_main = ic_results.get(main_period, {})
        ic_ir = ic_main.get('ic_ir', 0)
        ic_good = abs(ic_ir) > 0.3

        # 分层回测评价
        quantile_main = quantile_results.get(main_period, {})
        is_monotonic = quantile_main.get('is_monotonic', False)
        tmb_sharpe = quantile_main.get('tmb_sharpe', 0)

        # Fama-MacBeth评价
        fm_main = fm_results.get(main_period, {})
        fm_significant = fm_main.get('is_significant', False)
        t_stat = fm_main.get('t_statistic', 0)

        # 综合评分
        score = sum([ic_good, is_monotonic, fm_significant])

        # 评价等级
        if score >= 3:
            grade = "A"
            conclusion = "优秀 - 通过所有检验，具有显著预测能力"
        elif score >= 2:
            grade = "B"
            conclusion = "良好 - 通过部分检验，可考虑使用"
        elif score >= 1:
            grade = "C"
            conclusion = "一般 - 部分指标达标，需要优化"
        else:
            grade = "D"
            conclusion = "较差 - 建议重新设计或放弃"

        return {
            'main_period': main_period,
            'ic_ir': ic_ir,
            'ic_good': ic_good,
            'is_monotonic': is_monotonic,
            'tmb_sharpe': tmb_sharpe,
            'fm_significant': fm_significant,
            't_statistic': t_stat,
            'score': score,
            'grade': grade,
            'conclusion': conclusion
        }

    def _print_summary(self, results: Dict[str, Any]):
        """打印测试摘要"""
        factor_name = results['factor_name']
        evaluation = results['evaluation']
        main_period = evaluation['main_period']

        print(f"\n{'='*80}")
        print(f"因子测试摘要: {factor_name}")
        print(f"{'='*80}")

        print(f"主要评价周期: {main_period}")
        print(f"综合评分: {evaluation['score']}/3")
        print(f"评价等级: {evaluation['grade']}")
        print(f"结论: {evaluation['conclusion']}")

        print(f"\n详细指标:")
        print(f"  IC_IR: {evaluation['ic_ir']:.4f} ({'✓' if evaluation['ic_good'] else '✗'})")
        print(f"  分层单调性: {'✓' if evaluation['is_monotonic'] else '✗'}")
        print(f"  多空夏普: {evaluation['tmb_sharpe']:.4f}")
        print(f"  FM t值: {evaluation['t_statistic']:.4f} ({'✓' if evaluation['fm_significant'] else '✗'})")

        print(f"\n各周期IC_IR:")
        for period in self.test_periods:
            period_key = f'{period}d'
            ic_ir = results['ic_analysis'].get(period_key, {}).get('ic_ir', 0)
            print(f"  {period}日: {ic_ir:.4f}")

        print("="*80)

    def _create_visualizations(self, results: Dict[str, Any], factor_name: str):
        """创建可视化图表"""
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'因子测试报告: {factor_name}', fontsize=16, fontweight='bold')

        # 1. IC时间序列图
        ax1 = axes[0, 0]
        for period in self.test_periods:
            period_key = f'{period}d'
            ic_series = results['ic_analysis'].get(period_key, {}).get('ic_series', pd.Series())
            if not ic_series.empty:
                ax1.plot(ic_series.index, ic_series.values, label=f'{period}日', alpha=0.7)
        ax1.set_title('IC时间序列')
        ax1.set_ylabel('IC值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. IC统计指标柱状图
        ax2 = axes[0, 1]
        periods = [f'{p}d' for p in self.test_periods]
        ic_means = [results['ic_analysis'].get(p, {}).get('ic_mean', 0) for p in periods]
        ic_irs = [results['ic_analysis'].get(p, {}).get('ic_ir', 0) for p in periods]

        x = np.arange(len(periods))
        width = 0.35
        ax2.bar(x - width/2, ic_means, width, label='IC均值', alpha=0.8)
        ax2.bar(x + width/2, ic_irs, width, label='IC_IR', alpha=0.8)
        ax2.set_title('IC统计指标')
        ax2.set_xlabel('预测周期')
        ax2.set_xticks(x)
        ax2.set_xticklabels(periods)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 分层收益率图
        ax3 = axes[0, 2]
        main_period = '20d'
        quantile_main = results['quantile_backtest'].get(main_period, {})
        if quantile_main:
            quantile_means = quantile_main.get('quantile_means', [])
            if quantile_means:
                quantiles = [f'Q{i}' for i in range(1, len(quantile_means) + 1)]
                ax3.bar(quantiles, quantile_means, alpha=0.8, color='skyblue')
                ax3.set_title(f'分层收益率 ({main_period})')
                ax3.set_ylabel('平均收益率')
                ax3.grid(True, alpha=0.3)

        # 4. 多空组合净值曲线
        ax4 = axes[1, 0]
        for period in self.test_periods:
            period_key = f'{period}d'
            quantile_data = results['quantile_backtest'].get(period_key, {})
            if quantile_data:
                returns_data = quantile_data.get('returns_data')
                if returns_data is not None and 'TopMinusBottom' in returns_data.columns:
                    tmb_series = returns_data['TopMinusBottom']
                    cumulative_returns = (1 + tmb_series).cumprod()
                    ax4.plot(cumulative_returns.index, cumulative_returns.values,
                            label=f'{period}日', alpha=0.8)
        ax4.set_title('多空组合净值曲线')
        ax4.set_ylabel('累计收益')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Fama-MacBeth t值图
        ax5 = axes[1, 1]
        periods = [f'{p}d' for p in self.test_periods]
        t_stats = [results['fama_macbeth'].get(p, {}).get('t_statistic', 0) for p in periods]
        colors = ['red' if abs(t) > 2 else 'orange' if abs(t) > 1.64 else 'gray' for t in t_stats]

        ax5.bar(periods, t_stats, color=colors, alpha=0.8)
        ax5.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='显著水平(2)')
        ax5.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
        ax5.set_title('Fama-MacBeth t统计量')
        ax5.set_ylabel('t值')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. 综合评价雷达图
        ax6 = axes[1, 2]
        evaluation = results['evaluation']

        # 雷达图数据
        categories = ['IC_IR', '单调性', 'FM显著性', '多空夏普']
        values = [
            min(abs(evaluation['ic_ir']) / 0.5, 1),  # 标准化到0-1
            1 if evaluation['is_monotonic'] else 0,
            1 if evaluation['fm_significant'] else 0,
            min(abs(evaluation['tmb_sharpe']) / 2, 1)  # 标准化到0-1
        ]

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values += values[:1]  # 闭合
        angles = np.concatenate((angles, [angles[0]]))

        ax6.plot(angles, values, 'o-', linewidth=2, alpha=0.8)
        ax6.fill(angles, values, alpha=0.25)
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 1)
        ax6.set_title('综合评价雷达图')
        ax6.grid(True)

        plt.tight_layout()

        # 保存图表
        chart_path = os.path.join(self.output_dir, f'{factor_name}_analysis_charts.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"可视化图表已保存: {chart_path}")

    def _save_results(self, results: Dict[str, Any], factor_name: str):
        """保存测试结果"""
        # 准备可序列化的结果
        serializable_results = self._make_serializable(results)

        # 保存JSON格式
        json_path = os.path.join(self.output_dir, f'{factor_name}_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        # 保存Excel格式的摘要
        excel_path = os.path.join(self.output_dir, f'{factor_name}_summary.xlsx')
        self._save_excel_summary(results, excel_path)

        print(f"测试结果已保存:")
        print(f"  JSON: {json_path}")
        print(f"  Excel: {excel_path}")

    def _make_serializable(self, obj):
        """将结果转换为可序列化格式"""
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, pd.Series):
            # 将索引转换为字符串
            series_dict = {}
            for k, v in obj.items():
                key = str(k) if hasattr(k, '__str__') else k
                series_dict[key] = self._make_serializable(v)
            return series_dict
        elif isinstance(obj, pd.DataFrame):
            # 将索引和列名都转换为字符串
            df_dict = {}
            for idx, row in obj.iterrows():
                row_dict = {}
                for col, val in row.items():
                    col_key = str(col) if hasattr(col, '__str__') else col
                    row_dict[col_key] = self._make_serializable(val)
                idx_key = str(idx) if hasattr(idx, '__str__') else idx
                df_dict[idx_key] = row_dict
            return df_dict
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif pd.isna(obj):
            return None
        else:
            try:
                # 尝试转换为基本Python类型
                if hasattr(obj, 'item'):  # numpy标量
                    return obj.item()
                return obj
            except:
                return str(obj)

    def _save_excel_summary(self, results: Dict[str, Any], excel_path: str):
        """保存Excel格式的摘要报告"""
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 1. 综合评价表
            evaluation = results['evaluation']
            eval_df = pd.DataFrame([{
                '因子名称': results['factor_name'],
                '测试日期': results['test_date'],
                '评价等级': evaluation['grade'],
                '综合评分': f"{evaluation['score']}/3",
                'IC_IR(20d)': f"{evaluation['ic_ir']:.4f}",
                '分层单调性': '是' if evaluation['is_monotonic'] else '否',
                '多空夏普(20d)': f"{evaluation['tmb_sharpe']:.4f}",
                'FM t值(20d)': f"{evaluation['t_statistic']:.4f}",
                'FM显著性': '是' if evaluation['fm_significant'] else '否',
                '结论': evaluation['conclusion']
            }])
            eval_df.to_excel(writer, sheet_name='综合评价', index=False)

            # 2. IC分析结果表
            ic_data = []
            for period in self.test_periods:
                period_key = f'{period}d'
                ic_result = results['ic_analysis'].get(period_key, {})
                ic_data.append({
                    '预测周期': f'{period}日',
                    'IC均值': f"{ic_result.get('ic_mean', 0):.4f}",
                    'IC标准差': f"{ic_result.get('ic_std', 0):.4f}",
                    'IC_IR': f"{ic_result.get('ic_ir', 0):.4f}",
                    'IC胜率': f"{ic_result.get('ic_positive_ratio', 0):.2%}",
                    'IC绝对值均值': f"{ic_result.get('ic_abs_mean', 0):.4f}"
                })
            ic_df = pd.DataFrame(ic_data)
            ic_df.to_excel(writer, sheet_name='IC分析', index=False)

            # 3. 分层回测结果表
            quantile_data = []
            for period in self.test_periods:
                period_key = f'{period}d'
                quantile_result = results['quantile_backtest'].get(period_key, {})
                quantile_data.append({
                    '预测周期': f'{period}日',
                    '多空收益': f"{quantile_result.get('tmb_return', 0):.4f}",
                    '多空夏普': f"{quantile_result.get('tmb_sharpe', 0):.4f}",
                    '多空胜率': f"{quantile_result.get('tmb_win_rate', 0):.2%}",
                    '单调性': '是' if quantile_result.get('is_monotonic', False) else '否'
                })
            quantile_df = pd.DataFrame(quantile_data)
            quantile_df.to_excel(writer, sheet_name='分层回测', index=False)

            # 4. Fama-MacBeth回归结果表
            fm_data = []
            for period in self.test_periods:
                period_key = f'{period}d'
                fm_result = results['fama_macbeth'].get(period_key, {})
                fm_data.append({
                    '预测周期': f'{period}日',
                    '因子收益率': f"{fm_result.get('mean_factor_return', 0):.6f}",
                    't统计量': f"{fm_result.get('t_statistic', 0):.4f}",
                    'p值': f"{fm_result.get('p_value', 1):.4f}",
                    '回归期数': fm_result.get('num_periods', 0),
                    '显著性': fm_result.get('significance_desc', '不显著')
                })
            fm_df = pd.DataFrame(fm_data)
            fm_df.to_excel(writer, sheet_name='FM回归', index=False)

    def _generate_report(self, results: Dict[str, Any], factor_name: str):
        """生成文字报告"""
        report_path = os.path.join(self.output_dir, f'{factor_name}_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"单因子测试报告\n")
            f.write(f"{'='*60}\n\n")

            f.write(f"因子名称: {results['factor_name']}\n")
            f.write(f"测试日期: {results['test_date']}\n")
            f.write(f"预处理方法: {results['preprocess_method']}\n")
            f.write(f"测试周期: {', '.join([f'{p}日' for p in self.test_periods])}\n\n")

            # 综合评价
            evaluation = results['evaluation']
            f.write(f"综合评价\n")
            f.write(f"{'-'*30}\n")
            f.write(f"评价等级: {evaluation['grade']}\n")
            f.write(f"综合评分: {evaluation['score']}/3\n")
            f.write(f"结论: {evaluation['conclusion']}\n\n")

            # 详细结果
            f.write(f"详细测试结果\n")
            f.write(f"{'-'*30}\n")

            # IC分析
            f.write(f"1. IC值分析法\n")
            for period in self.test_periods:
                period_key = f'{period}d'
                ic_result = results['ic_analysis'].get(period_key, {})
                f.write(f"   {period}日: IC_IR={ic_result.get('ic_ir', 0):.4f}, ")
                f.write(f"IC胜率={ic_result.get('ic_positive_ratio', 0):.2%}\n")

            # 分层回测
            f.write(f"\n2. 分层回测法\n")
            for period in self.test_periods:
                period_key = f'{period}d'
                quantile_result = results['quantile_backtest'].get(period_key, {})
                f.write(f"   {period}日: 多空收益={quantile_result.get('tmb_return', 0):.4f}, ")
                f.write(f"夏普比率={quantile_result.get('tmb_sharpe', 0):.4f}, ")
                f.write(f"单调性={'是' if quantile_result.get('is_monotonic', False) else '否'}\n")

            # Fama-MacBeth回归
            f.write(f"\n3. Fama-MacBeth回归法\n")
            for period in self.test_periods:
                period_key = f'{period}d'
                fm_result = results['fama_macbeth'].get(period_key, {})
                f.write(f"   {period}日: t值={fm_result.get('t_statistic', 0):.4f}, ")
                f.write(f"显著性={fm_result.get('significance_desc', '不显著')}\n")

            f.write(f"\n测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"测试报告已保存: {report_path}")

    def batch_test(self,
                   factors_dict: Dict[str, pd.DataFrame],
                   preprocess_method: str = "standard") -> Dict[str, Dict]:
        """
        批量测试多个因子

        Args:
            factors_dict: 因子字典，{因子名称: 因子数据}
            preprocess_method: 预处理方法

        Returns:
            批量测试结果字典
        """
        print(f"\n{'='*80}")
        print(f"开始批量测试 {len(factors_dict)} 个因子")
        print(f"{'='*80}")

        batch_results = {}

        for i, (factor_name, factor_data) in enumerate(factors_dict.items(), 1):
            print(f"\n进度: {i}/{len(factors_dict)}")

            try:
                results = self.comprehensive_test(
                    factor_data,
                    factor_name,
                    preprocess_method=preprocess_method,
                    save_results=True
                )
                batch_results[factor_name] = results

            except Exception as e:
                print(f"因子 {factor_name} 测试失败: {e}")
                batch_results[factor_name] = {'error': str(e)}

        # 生成批量测试摘要
        self._generate_batch_summary(batch_results)

        return batch_results

    def _generate_batch_summary(self, batch_results: Dict[str, Dict]):
        """生成批量测试摘要"""
        summary_data = []

        for factor_name, results in batch_results.items():
            if 'error' in results:
                continue

            evaluation = results.get('evaluation', {})
            summary_data.append({
                '因子名称': factor_name,
                '评价等级': evaluation.get('grade', 'N/A'),
                '综合评分': evaluation.get('score', 0),
                'IC_IR(20d)': evaluation.get('ic_ir', 0),
                '分层单调性': evaluation.get('is_monotonic', False),
                '多空夏普(20d)': evaluation.get('tmb_sharpe', 0),
                'FM t值(20d)': evaluation.get('t_statistic', 0),
                'FM显著性': evaluation.get('fm_significant', False)
            })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # 按评分排序
            summary_df = summary_df.sort_values(['综合评分', 'IC_IR(20d)'], ascending=[False, False])

            # 保存摘要
            summary_path = os.path.join(self.output_dir, 'batch_test_summary.xlsx')
            summary_df.to_excel(summary_path, index=False)

            print(f"\n批量测试摘要已保存: {summary_path}")

            # 打印Top 5因子
            print(f"\nTop 5 因子:")
            print("-" * 40)
            for i, row in summary_df.head().iterrows():
                print(f"{row['因子名称']}: {row['评价等级']}级 (评分: {row['综合评分']}/3)")

        print(f"\n批量测试完成!")
        print(f"结果保存在: {self.output_dir}")
        print("="*80)

    def run_comprehensive_test_with_viz(self,
                                       factor_data: pd.DataFrame,
                                       factor_name: str,
                                       preprocess_method: str = "standard",
                                       generate_plots: bool = True) -> Dict[str, Any]:
        """
        运行综合测试并生成可视化（新增方法）

        Args:
            factor_data: 因子数据
            factor_name: 因子名称
            preprocess_method: 预处理方法
            generate_plots: 是否生成图表

        Returns:
            包含测试结果和图表路径的字典
        """
        # 运行标准测试
        test_results = self.run_comprehensive_test(
            factor_data=factor_data,
            factor_name=factor_name,
            preprocess_method=preprocess_method
        )

        # 生成可视化
        plot_paths = {}
        if generate_plots and self.viz_manager is not None:
            try:
                plot_paths = self.viz_manager.plot_single_factor_results(
                    test_results=test_results,
                    factor_name=factor_name,
                    save_plots=True
                )
                print(f"✓ 图表已生成并保存到: {self.viz_manager.output_dir}")
            except Exception as e:
                print(f"⚠ 图表生成失败: {e}")

        # 合并结果
        enhanced_results = test_results.copy()
        enhanced_results['plot_paths'] = plot_paths

        return enhanced_results
