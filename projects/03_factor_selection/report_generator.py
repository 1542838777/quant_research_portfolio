"""
报告生成器 - 单因子测试终极作战手册
第五阶段：专业分析报告生成

自动生成专业级的因子分析报告，包括：
1. Jupyter Notebook报告模板
2. Excel详细数据报告
3. Word格式分析报告
4. 可视化图表集合
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ReportGenerator:
    """
    专业报告生成器
    
    生成多种格式的因子分析报告：
    - Jupyter Notebook交互式报告
    - Excel数据报告
    - 可视化图表集合
    - JSON格式的结构化数据
    """
    
    def __init__(self, config: Dict):
        """
        初始化报告生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.output_config = config.get('output', {})
        
    def generate_comprehensive_report(self, 
                                    results: Dict[str, Any],
                                    factor_data: pd.DataFrame,
                                    output_dir: str):
        """
        生成综合报告
        
        Args:
            results: 评价结果
            factor_data: 因子数据
            output_dir: 输出目录
        """
        print("\n" + "="*60)
        print("第五阶段：生成专业分析报告")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        factor_name = results['factor_name']
        
        # 1. 生成Jupyter Notebook报告
        print("1. 生成Jupyter Notebook报告...")
        self._generate_notebook_report(results, factor_data, output_dir, factor_name)
        
        # 2. 生成Excel数据报告
        print("2. 生成Excel数据报告...")
        self._generate_excel_report(results, factor_data, output_dir, factor_name)
        
        # 3. 保存JSON结构化数据
        print("3. 保存JSON结构化数据...")
        self._save_json_data(results, output_dir, factor_name)
        
        # 4. 生成可视化图表
        print("4. 生成可视化图表...")
        self._generate_visualization_suite(results, output_dir, factor_name)
        
        # 5. 生成摘要报告
        print("5. 生成摘要报告...")
        self._generate_summary_report(results, output_dir, factor_name)
        
        print(f"\n报告生成完成！输出目录: {output_dir}")
    
    def _generate_notebook_report(self, results: Dict, factor_data: pd.DataFrame, 
                                output_dir: str, factor_name: str):
        """生成Jupyter Notebook报告"""
        notebook_content = self._create_notebook_template(results, factor_name)
        
        notebook_path = os.path.join(output_dir, f'{factor_name}_research_report.ipynb')
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_content, f, ensure_ascii=False, indent=2)
        
        print(f"  Notebook报告: {notebook_path}")
    
    def _create_notebook_template(self, results: Dict, factor_name: str) -> Dict:
        """创建Notebook模板"""
        summary = results.get('summary', {})
        
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# {factor_name} 因子有效性分析报告\n",
                        "\n",
                        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                        f"**评价等级**: {summary.get('grade', 'N/A')}\n",
                        f"**综合得分**: {summary.get('score', 0)}/3\n",
                        f"**结论**: {summary.get('conclusion', '无')}\n",
                        "\n",
                        "---\n"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 1. 研究背景与目标\n",
                        "\n",
                        "本报告对**{factor_name}**因子进行完整的有效性检验，采用华泰证券标准的三位一体检验方法：\n",
                        "\n",
                        "1. **IC分析** - 相关性检验\n",
                        "2. **Fama-MacBeth回归** - 学术黄金标准\n",
                        "3. **分层回测** - 实际投资效果验证\n",
                        "\n",
                        "### 1.1 因子定义\n",
                        "\n",
                        f"- **因子名称**: {factor_name}\n",
                        f"- **因子描述**: {self.config.get('target_factor', {}).get('description', '无描述')}\n",
                        f"- **数据来源**: {', '.join(self.config.get('target_factor', {}).get('fields', []))}\n",
                        f"- **测试周期**: {self.config.get('backtest', {}).get('start_date', '')} 至 {self.config.get('backtest', {}).get('end_date', '')}\n"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 2. 数据预处理\n",
                        "\n",
                        "### 2.1 股票池构建\n",
                        "\n",
                        f"- **基础股票池**: {self.config.get('universe', {}).get('base', '')}\n",
                        "- **过滤条件**:\n",
                        f"  - 剔除ST股票: {self.config.get('universe', {}).get('filters', {}).get('remove_st', False)}\n",
                        f"  - 最小上市天数: {self.config.get('universe', {}).get('filters', {}).get('min_list_days', 0)}天\n",
                        f"  - 流动性过滤: 剔除后{self.config.get('universe', {}).get('filters', {}).get('min_liquidity_percentile', 0)*100:.0f}%\n",
                        "\n",
                        "### 2.2 因子预处理\n",
                        "\n",
                        f"- **去极值方法**: {self.config.get('preprocessing', {}).get('winsorization', {}).get('method', '')}\n",
                        f"- **中性化**: {self.config.get('preprocessing', {}).get('neutralization', {}).get('enable', False)}\n",
                        f"- **标准化方法**: {self.config.get('preprocessing', {}).get('standardization', {}).get('method', '')}\n"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 3. IC分析结果\n",
                        "\n",
                        "### 3.1 IC统计指标\n",
                        "\n",
                        self._format_ic_results_for_notebook(results.get('ic_analysis', {}))
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 4. Fama-MacBeth回归结果\n",
                        "\n",
                        "### 4.1 回归统计\n",
                        "\n",
                        self._format_fm_results_for_notebook(results.get('fama_macbeth', {}))
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 5. 分层回测结果\n",
                        "\n",
                        "### 5.1 分层收益率\n",
                        "\n",
                        self._format_quantile_results_for_notebook(results.get('quantile_backtest', {}))
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 6. 综合评价与结论\n",
                        "\n",
                        "### 6.1 评价体系\n",
                        "\n",
                        "本研究采用三维评价体系，每个维度1分，总分3分：\n",
                        "\n",
                        f"- **IC有效性** ({1 if summary.get('ic_good', False) else 0}/1): IC_IR > 0.3\n",
                        f"- **FM显著性** ({1 if summary.get('fm_significant', False) else 0}/1): t统计量显著\n",
                        f"- **分层单调性** ({1 if summary.get('qt_monotonic', False) else 0}/1): 收益率单调\n",
                        "\n",
                        f"**综合得分**: {summary.get('score', 0)}/3\n",
                        f"**评价等级**: {summary.get('grade', 'N/A')}\n",
                        "\n",
                        "### 6.2 最终结论\n",
                        "\n",
                        f"{summary.get('conclusion', '无结论')}\n",
                        "\n",
                        "### 6.3 投资建议\n",
                        "\n",
                        self._generate_investment_advice(summary)
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        return notebook
    
    def _format_ic_results_for_notebook(self, ic_results: Dict) -> List[str]:
        """格式化IC结果为Notebook格式"""
        lines = ["| 预测周期 | IC均值 | IC_IR | IC胜率 | 显著性 |\n",
                "| --- | --- | --- | --- | --- |\n"]
        
        for period_key, result in ic_results.items():
            if 'error' not in result:
                ic_mean = result.get('ic_mean', 0)
                ic_ir = result.get('ic_ir', 0)
                ic_win_rate = result.get('ic_positive_ratio', 0)
                is_significant = result.get('is_significant', False)
                
                lines.append(
                    f"| {period_key} | {ic_mean:.4f} | {ic_ir:.4f} | "
                    f"{ic_win_rate:.2%} | {'是' if is_significant else '否'} |\n"
                )
        
        return lines
    
    def _format_fm_results_for_notebook(self, fm_results: Dict) -> List[str]:
        """格式化FM结果为Notebook格式"""
        lines = ["| 预测周期 | 因子收益率 | t统计量 | p值 | 显著性 |\n",
                "| --- | --- | --- | --- | --- |\n"]
        
        for period_key, result in fm_results.items():
            if 'error' not in result:
                factor_return = result.get('factor_return', 0)
                t_stat = result.get('t_statistic', 0)
                p_value = result.get('p_value', 1)
                significance = result.get('significance_desc', '不显著')
                
                lines.append(
                    f"| {period_key} | {factor_return:.4f} | {t_stat:.4f} | "
                    f"{p_value:.4f} | {significance} |\n"
                )
        
        return lines
    
    def _format_quantile_results_for_notebook(self, qt_results: Dict) -> List[str]:
        """格式化分层回测结果为Notebook格式"""
        lines = ["| 预测周期 | 多空收益率 | 多空夏普 | 单调性 |\n",
                "| --- | --- | --- | --- |\n"]
        
        for period_key, result in qt_results.items():
            if 'error' not in result:
                long_short_return = result.get('long_short_return', 0)
                long_short_sharpe = result.get('long_short_sharpe', 0)
                is_monotonic = result.get('is_monotonic', False)
                
                lines.append(
                    f"| {period_key} | {long_short_return:.4f} | {long_short_sharpe:.4f} | "
                    f"{'是' if is_monotonic else '否'} |\n"
                )
        
        return lines
    
    def _generate_investment_advice(self, summary: Dict) -> str:
        """生成投资建议"""
        grade = summary.get('grade', 'D')
        
        if grade == 'A':
            return ("**强烈推荐**: 该因子通过了所有检验，具有优秀的预测能力和投资价值。"
                   "建议作为核心因子纳入多因子模型，权重可设置为较高水平。")
        elif grade == 'B':
            return ("**推荐使用**: 该因子表现良好，具有一定的投资价值。"
                   "建议作为辅助因子使用，或与其他因子组合使用以提高稳定性。")
        elif grade == 'C':
            return ("**谨慎使用**: 该因子表现一般，需要进一步优化。"
                   "建议结合其他强因子使用，或考虑改进因子构造方法。")
        else:
            return ("**不建议使用**: 该因子未通过有效性检验，不具备投资价值。"
                   "建议放弃该因子，或重新设计因子构造逻辑。")
    
    def _generate_excel_report(self, results: Dict, factor_data: pd.DataFrame,
                             output_dir: str, factor_name: str):
        """生成Excel数据报告"""
        excel_path = os.path.join(output_dir, f'{factor_name}_detailed_report.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 1. 综合摘要
            self._write_summary_sheet(results, writer)
            
            # 2. IC分析详情
            self._write_ic_analysis_sheet(results.get('ic_analysis', {}), writer)
            
            # 3. Fama-MacBeth详情
            self._write_fm_analysis_sheet(results.get('fama_macbeth', {}), writer)
            
            # 4. 分层回测详情
            self._write_quantile_analysis_sheet(results.get('quantile_backtest', {}), writer)
            
            # 5. 因子数据样本
            factor_sample = factor_data.head(100)  # 只保存前100行作为样本
            factor_sample.to_excel(writer, sheet_name='因子数据样本')
        
        print(f"  Excel报告: {excel_path}")
    
    def _write_summary_sheet(self, results: Dict, writer):
        """写入摘要表"""
        summary = results.get('summary', {})
        
        summary_data = [
            ['因子名称', results.get('factor_name', '')],
            ['评价等级', summary.get('grade', '')],
            ['综合得分', f"{summary.get('score', 0)}/3"],
            ['IC有效性', '通过' if summary.get('ic_good', False) else '未通过'],
            ['FM显著性', '通过' if summary.get('fm_significant', False) else '未通过'],
            ['分层单调性', '通过' if summary.get('qt_monotonic', False) else '未通过'],
            ['结论', summary.get('conclusion', '')],
            ['', ''],
            ['详细指标', ''],
            ['IC均值', f"{summary.get('detailed_metrics', {}).get('ic_mean', 0):.4f}"],
            ['IC胜率', f"{summary.get('detailed_metrics', {}).get('ic_win_rate', 0):.2%}"],
            ['FM t统计量', f"{summary.get('detailed_metrics', {}).get('fm_t_stat', 0):.4f}"],
            ['FM p值', f"{summary.get('detailed_metrics', {}).get('fm_p_value', 1):.4f}"],
            ['多空收益率', f"{summary.get('detailed_metrics', {}).get('qt_long_short_return', 0):.4f}"]
        ]
        
        summary_df = pd.DataFrame(summary_data, columns=['指标', '数值'])
        summary_df.to_excel(writer, sheet_name='综合摘要', index=False)
    
    def _write_ic_analysis_sheet(self, ic_results: Dict, writer):
        """写入IC分析表"""
        ic_data = []
        
        for period_key, result in ic_results.items():
            if 'error' not in result:
                ic_data.append({
                    '预测周期': period_key,
                    'IC均值': result.get('ic_mean', 0),
                    'IC标准差': result.get('ic_std', 0),
                    'IC_IR': result.get('ic_ir', 0),
                    'IC胜率': result.get('ic_positive_ratio', 0),
                    'IC绝对值均值': result.get('ic_abs_mean', 0),
                    't统计量': result.get('t_statistic', 0),
                    'p值': result.get('p_value', 1),
                    '显著性': '是' if result.get('is_significant', False) else '否'
                })
        
        if ic_data:
            ic_df = pd.DataFrame(ic_data)
            ic_df.to_excel(writer, sheet_name='IC分析', index=False)
    
    def _write_fm_analysis_sheet(self, fm_results: Dict, writer):
        """写入Fama-MacBeth分析表"""
        fm_data = []
        
        for period_key, result in fm_results.items():
            if 'error' not in result:
                fm_data.append({
                    '预测周期': period_key,
                    '因子收益率': result.get('factor_return', 0),
                    '标准误': result.get('std_error', 0),
                    't统计量': result.get('t_statistic', 0),
                    'p值': result.get('p_value', 1),
                    '显著性水平': result.get('significance_level', ''),
                    '显著性描述': result.get('significance_desc', ''),
                    '有效期数': result.get('valid_periods', 0)
                })
        
        if fm_data:
            fm_df = pd.DataFrame(fm_data)
            fm_df.to_excel(writer, sheet_name='Fama-MacBeth回归', index=False)
    
    def _write_quantile_analysis_sheet(self, qt_results: Dict, writer):
        """写入分层回测分析表"""
        qt_data = []
        
        for period_key, result in qt_results.items():
            if 'error' not in result:
                qt_data.append({
                    '预测周期': period_key,
                    '多空收益率': result.get('long_short_return', 0),
                    '多空夏普比率': result.get('long_short_sharpe', 0),
                    '单调性': '是' if result.get('is_monotonic', False) else '否'
                })
        
        if qt_data:
            qt_df = pd.DataFrame(qt_data)
            qt_df.to_excel(writer, sheet_name='分层回测', index=False)
    
    def _save_json_data(self, results: Dict, output_dir: str, factor_name: str):
        """保存JSON格式的结构化数据"""
        # 处理不能序列化的对象
        json_results = self._prepare_for_json(results)
        
        json_path = os.path.join(output_dir, f'{factor_name}_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"  JSON数据: {json_path}")
    
    def _prepare_for_json(self, obj):
        """准备对象用于JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _generate_visualization_suite(self, results: Dict, output_dir: str, factor_name: str):
        """生成可视化图表套件"""
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 这里可以调用factor_evaluator的绘图功能
        print(f"  可视化图表: {viz_dir}")
    
    def _generate_summary_report(self, results: Dict, output_dir: str, factor_name: str):
        """生成文字摘要报告"""
        summary = results.get('summary', {})
        
        report_lines = [
            f"# {factor_name} 因子分析摘要报告",
            f"",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**评价等级**: {summary.get('grade', 'N/A')}",
            f"**综合得分**: {summary.get('score', 0)}/3",
            f"",
            f"## 核心结论",
            f"",
            f"{summary.get('conclusion', '无结论')}",
            f"",
            f"## 详细指标",
            f"",
            f"- IC均值: {summary.get('detailed_metrics', {}).get('ic_mean', 0):.4f}",
            f"- IC胜率: {summary.get('detailed_metrics', {}).get('ic_win_rate', 0):.2%}",
            f"- FM t统计量: {summary.get('detailed_metrics', {}).get('fm_t_stat', 0):.4f}",
            f"- 多空收益率: {summary.get('detailed_metrics', {}).get('qt_long_short_return', 0):.4f}",
            f"",
            f"## 投资建议",
            f"",
            f"{self._generate_investment_advice(summary)}"
        ]
        
        report_path = os.path.join(output_dir, f'{factor_name}_summary.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  摘要报告: {report_path}")


def create_report_generator(config: Dict) -> ReportGenerator:
    """
    创建报告生成器
    
    Args:
        config: 配置字典
        
    Returns:
        ReportGenerator实例
    """
    return ReportGenerator(config)
