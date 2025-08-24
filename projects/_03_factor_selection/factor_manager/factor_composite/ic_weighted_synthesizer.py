"""
IC加权因子合成器 - 专业级因子合成引擎

核心功能：
1. 基于历史IC表现的智能权重分配
2. 多维度因子筛选机制
3. 动态权重调整
4. 风险控制和稳健性检验

设计理念：
- 以实盘盈利为终极目标
- 平衡因子预测能力与稳定性
- 防止过拟合，注重泛化能力
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

from projects._03_factor_selection.factor_manager.factor_composite.factor_synthesizer import FactorSynthesizer
from projects._03_factor_selection.factor_manager.storage.result_load_manager import ResultLoadManager
from projects._03_factor_selection.factor_manager.storage.rolling_ic_manager import RollingICManager, ICCalculationConfig, ICSnapshot
from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class FactorWeightingConfig:
    """因子权重配置"""
    # IC筛选标准
    min_ic_mean: float = 0.02          # 最小IC均值阈值
    min_ic_ir: float = 0.3             # 最小IC信息比率阈值
    min_ic_win_rate: float = 0.50      # 最小IC胜率阈值
    max_ic_p_value: float = 0.10       # 最大IC显著性p值
    
    # 权重计算参数
    ic_decay_halflife: int = 60        # IC权重衰减半衰期(天)
    max_single_weight: float = 0.50    # 单个因子最大权重
    min_single_weight: float = 0.05    # 单个因子最小权重
    
    # 风险控制
    max_factors_count: int = 8         # 最大因子数量
    correlation_threshold: float = 0.70 # 因子间相关性阈值
    
    # 回看期设置
    lookback_periods: List[str] = None  # IC计算周期
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = ['5d', '21d']


@dataclass
class FactorQualityReport:
    """因子质量报告"""
    factor_name: str
    ic_stats: Dict[str, float]
    weight: float
    selection_reason: str
    risk_flags: List[str]


class ICWeightCalculator:
    """IC权重计算器 - 核心算法引擎"""
    
    def __init__(self, config: FactorWeightingConfig):
        self.config = config
        
    def calculate_ic_based_weights(
        self, 
        factor_ic_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        基于IC统计的智能权重分配
        
        Args:
            factor_ic_stats: {factor_name: {period: ic_stats_dict}}
            
        Returns:
            Dict[factor_name, weight]: 标准化后的权重分配
        """
        logger.info("🧮 开始计算IC加权权重...")
        
        # 1. 计算综合IC得分
        factor_scores = {}
        for factor_name, periods_stats in factor_ic_stats.items():
            score = self._calculate_composite_ic_score(periods_stats)
            factor_scores[factor_name] = score
            logger.debug(f"  {factor_name}: 综合IC得分 = {score:.4f}")
        
        # 2. 基于得分计算原始权重
        raw_weights = self._convert_scores_to_weights(factor_scores)
        
        # 3. 应用约束和标准化
        final_weights = self._apply_weight_constraints(raw_weights)
        
        logger.info(f"✅ IC权重计算完成，共{len(final_weights)}个因子被分配权重")
        return final_weights
    
    def _calculate_composite_ic_score(self, periods_stats: Dict[str, Dict]) -> float:
        """计算因子的综合IC得分"""
        period_scores = []
        
        for period, stats in periods_stats.items():
            if not stats or 'ic_mean' not in stats:
                continue
                
            # 提取关键指标
            ic_mean = abs(stats.get('ic_mean', 0))           # 使用绝对值
            ic_ir = stats.get('ic_ir', 0)
            ic_win_rate = stats.get('ic_win_rate', 0.5)
            ic_t_stat = abs(stats.get('ic_t_stat', 0))
            
            # 多维度评分模型
            # 1. IC均值得分 (40%权重)
            ic_mean_score = min(ic_mean / 0.05, 1.0) * 0.4
            
            # 2. IC稳定性得分 (30%权重) 
            ic_stability_score = min(ic_ir / 0.5, 1.0) * 0.3
            
            # 3. IC胜率得分 (20%权重)
            ic_win_score = max(0, (ic_win_rate - 0.5) * 2) * 0.2
            
            # 4. IC显著性得分 (10%权重)
            ic_sig_score = min(ic_t_stat / 2.0, 1.0) * 0.1
            
            period_score = ic_mean_score + ic_stability_score + ic_win_score + ic_sig_score
            period_scores.append(period_score)
        
        # 多周期平均，给短期稍高权重
        if not period_scores:
            return 0.0
            
        if len(period_scores) == 1:
            return period_scores[0]
        else:
            # --- 改进的加权方案 ---
            # 使用指数衰减权重，给短期更高权重，但依然考虑所有周期
            # decay_rate 越小，权重衰减越慢
            decay_rate = 0.75
            weights = np.array([decay_rate ** i for i in range(len(period_scores))])
            weights /= weights.sum()  # 权重归一化

            logger.debug(f"  多周期权重 (从1d到120d): {[f'{w:.2f}' for w in weights]}")
            return np.average(period_scores, weights=weights)
    
    def _convert_scores_to_weights(self, factor_scores: Dict[str, float]) -> Dict[str, float]:
        """将得分转换为权重"""
        if not factor_scores:
            return {}
        
        # 使用 softmax 函数将得分转换为权重，增强区分度
        scores = np.array(list(factor_scores.values()))
        
        # 过滤掉得分过低的因子
        valid_mask = scores > 0.1
        if not valid_mask.any():
            logger.warning("⚠️ 所有因子得分都过低，使用等权重")
            return {name: 1.0/len(factor_scores) for name in factor_scores.keys()}
        
        # 对有效因子应用 softmax
        valid_scores = scores[valid_mask] 
        valid_names = [name for i, name in enumerate(factor_scores.keys()) if valid_mask[i]]
        
        # 温度参数控制权重集中度，温度越高越平均
        temperature = 2.0
        exp_scores = np.exp(valid_scores / temperature)
        softmax_weights = exp_scores / exp_scores.sum()
        
        return dict(zip(valid_names, softmax_weights))
    
    def _apply_weight_constraints(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """应用权重约束"""
        if not raw_weights:
            return {}
        
        weights = raw_weights.copy()
        
        # 1. 应用单因子权重上下限
        for name in weights:
            weights[name] = np.clip(weights[name], 
                                  self.config.min_single_weight, 
                                  self.config.max_single_weight)
        
        # 2. 重新标准化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: w/total_weight for name, w in weights.items()}
        
        return weights


class FactorQualityFilter:
    """因子质量筛选器"""
    
    def __init__(self, config: FactorWeightingConfig):
        self.config = config
    
    def filter_factors_by_quality(
        self, 
        factor_ic_stats: Dict[str, Dict[str, Dict]]
    ) -> Tuple[Dict[str, Dict[str, Dict]], List[FactorQualityReport]]:
        """
        基于多维度质量指标筛选因子
        
        Returns:
            (filtered_factor_stats, quality_reports)
        """
        logger.info("🔍 开始因子质量筛选...")
        
        filtered_stats = {}
        quality_reports = []
        
        for factor_name, periods_stats in factor_ic_stats.items():
            report = self._evaluate_single_factor(factor_name, periods_stats)
            quality_reports.append(report)
            
            if self._passes_quality_filter(report):
                filtered_stats[factor_name] = periods_stats
                logger.info(f"✅ {factor_name}: 通过筛选 (权重={report.weight:.3f}) - {report.selection_reason}")
            else:
                logger.info(f"❌ {factor_name}: 未通过筛选 - {report.selection_reason}")
        
        logger.info(f"📊 筛选结果: {len(filtered_stats)}/{len(factor_ic_stats)} 个因子通过质量检验")
        return filtered_stats, quality_reports
    
    def _evaluate_single_factor(self, factor_name: str, periods_stats: Dict) -> FactorQualityReport:
        """评估单个因子质量"""
        # 计算关键指标的综合表现
        ic_means = []
        ic_irs = []
        ic_win_rates = []
        ic_p_values = []
        
        for period, stats in periods_stats.items():
            if stats and 'ic_mean' in stats:
                ic_means.append(abs(stats.get('ic_mean', 0)))
                ic_irs.append(stats.get('ic_ir', 0))
                ic_win_rates.append(stats.get('ic_win_rate', 0.5))
                ic_p_values.append(stats.get('ic_p_value', 1.0))
        
        if not ic_means:
            return FactorQualityReport(
                factor_name=factor_name,
                ic_stats={},
                weight=0.0,
                selection_reason="缺少有效的IC统计数据",
                risk_flags=["数据不足"]
            )
        
        # 综合统计
        avg_ic_mean = np.mean(ic_means)
        avg_ic_ir = np.mean(ic_irs)
        avg_win_rate = np.mean(ic_win_rates)
        min_p_value = min(ic_p_values)
        
        ic_stats = {
            'avg_ic_mean': avg_ic_mean,
            'avg_ic_ir': avg_ic_ir, 
            'avg_win_rate': avg_win_rate,
            'min_p_value': min_p_value
        }
        
        # 质量评估
        risk_flags = []
        selection_reason = ""
        if avg_ic_mean < self.config.min_ic_mean:
            risk_flags.append(f"IC均值过低({avg_ic_mean:.3f})")
        if avg_ic_ir < self.config.min_ic_ir:
            risk_flags.append(f"IC信息比率过低({avg_ic_ir:.3f})")
        if avg_win_rate < self.config.min_ic_win_rate:
            risk_flags.append(f"IC胜率过低({avg_win_rate:.3f})")
        if min_p_value > self.config.max_ic_p_value:
            risk_flags.append(f"IC显著性不足(p={min_p_value:.3f})")
        
        if not risk_flags:
            selection_reason = f"高质量因子 (IC={avg_ic_mean:.3f}, IR={avg_ic_ir:.2f}, 胜率={avg_win_rate:.1%})"
            weight = self._calculate_factor_weight(ic_stats)
        else:
            selection_reason = "; ".join(risk_flags)
            weight = 0.0
            
        return FactorQualityReport(
            factor_name=factor_name,
            ic_stats=ic_stats,
            weight=weight,
            selection_reason=selection_reason,
            risk_flags=risk_flags
        )
    
    def _passes_quality_filter(self, report: FactorQualityReport) -> bool:
        """判断因子是否通过质量筛选"""
        return len(report.risk_flags) == 0 and report.weight > 0
    
    def _calculate_factor_weight(self, ic_stats: Dict[str, float]) -> float:
        """基于IC统计计算因子初步权重"""
        # 这里可以实现更复杂的权重逻辑
        # 目前使用简单的综合得分
        ic_mean = ic_stats.get('avg_ic_mean', 0)
        ic_ir = ic_stats.get('avg_ic_ir', 0)
        win_rate = ic_stats.get('avg_win_rate', 0.5)
        
        # 综合得分
        score = ic_mean * 0.5 + ic_ir * 0.3 + max(0, win_rate - 0.5) * 0.2
        return min(score, 1.0)


class ICWeightedSynthesizer(FactorSynthesizer):
    """IC加权因子合成器 - 继承并扩展现有功能"""
    
    def __init__(self, factor_manager, factor_analyzer, factor_processor, 
                 config: Optional[FactorWeightingConfig] = None):
        super().__init__(factor_manager, factor_analyzer, factor_processor)
        
        self.config = config or FactorWeightingConfig()
        self.weight_calculator = ICWeightCalculator(self.config)
        self.quality_filter = FactorQualityFilter(self.config)
        
        # 滚动IC管理器 - 核心改进
        rolling_ic_config = ICCalculationConfig(
            lookback_months=12,
            forward_periods=self.config.lookback_periods,
            calculation_frequency='M'
        )
        
        storage_root = r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\workspace\rolling_ic"
        self.rolling_ic_manager = RollingICManager(storage_root, rolling_ic_config)
        
        # 缓存IC统计数据，避免重复计算
        self._ic_stats_cache = {}
    
    def synthesize_ic_weighted_factor(
        self,
        composite_factor_name: str,
        stock_pool_index_name: str, 
        candidate_factor_names: List[str],
        force_recalculate_ic: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        IC加权因子合成主流程
        
        Args:
            composite_factor_name: 复合因子名称
            stock_pool_index_name: 股票池名称
            candidate_factor_names: 候选因子列表
            force_recalculate_ic: 是否强制重新计算IC
            
        Returns:
            (composite_factor_df, synthesis_report)
        """
        logger.info(f"\n🚀 开始IC加权因子合成: {composite_factor_name}")
        logger.info(f"📊 候选因子数量: {len(candidate_factor_names)}")
        logger.info(f"📈 目标股票池: {stock_pool_index_name}")
        
        # 第一步：收集候选因子的IC统计数据
        factor_ic_stats = self._collect_factor_ic_stats(
            candidate_factor_names, 
            stock_pool_index_name,
            force_recalculate_ic
        )
        
        # 第二步：质量筛选
        qualified_factor_stats, quality_reports = self.quality_filter.filter_factors_by_quality(
            factor_ic_stats
        )
        
        if not qualified_factor_stats:
            raise ValueError("❌ 没有因子通过质量筛选，无法进行合成")
        
        # 第三步：计算权重
        factor_weights = self.weight_calculator.calculate_ic_based_weights(
            qualified_factor_stats
        )
        
        # 第四步：执行加权合成
        composite_factor_df = self._execute_weighted_synthesis(
            composite_factor_name,
            stock_pool_index_name,
            factor_weights
        )
        
        # 生成合成报告
        synthesis_report = self._generate_synthesis_report(
            composite_factor_name, 
            candidate_factor_names,
            factor_weights,
            quality_reports
        )
        
        logger.info(f"✅ IC加权因子合成完成: {composite_factor_name}")
        return composite_factor_df, synthesis_report
    
    def calculate_rolling_weights(
        self,
        candidate_factor_names: List[str],
        stock_pool_index_name: str,
        calculation_date: str,
        resultLoadManager: ResultLoadManager
    ) -> Dict[str, float]:
        """
        滚动权重计算 - 核心改进：完全避免前视偏差
        
        Args:
            candidate_factor_names: 候选因子列表
            stock_pool_index_name: 股票池名称
            calculation_date: 权重计算时点（严格不使用此时点之后的数据）
            factor_data_source: 因子数据源
            return_data_source: 收益数据源
            
        Returns:
            Dict[factor_name, weight]: 基于历史IC的权重分配
        """
        logger.info(f"🔄 开始滚动权重计算 @ {calculation_date}")
        logger.info(f"📊 候选因子: {len(candidate_factor_names)} 个")
        
        # 第一步：获取截止到calculation_date的历史IC数据
        historical_ic_stats = {}
        
        for factor_name in candidate_factor_names:
            try:
                # 从滚动IC管理器获取历史IC快照
                latest_snapshot = self.rolling_ic_manager.get_ic_at_timepoint(
                    factor_name, stock_pool_index_name, calculation_date
                )
                
                if latest_snapshot and latest_snapshot.ic_stats:
                    historical_ic_stats[factor_name] = latest_snapshot.ic_stats
                    logger.debug(f"  ✅ {factor_name}: 获取历史IC @ {calculation_date}")
                else:
                    snapshot = self.rolling_ic_manager._calculate_ic_snapshot(
                        factor_name, stock_pool_index_name, calculation_date,
                        resultLoadManager
                    )
                    if snapshot and snapshot.ic_stats:
                        historical_ic_stats[factor_name] = snapshot.ic_stats
                        # 保存快照以供后续使用
                        self.rolling_ic_manager._save_snapshot(snapshot)
                        logger.debug(f"  🔄 {factor_name}: 实时计算IC @ {calculation_date}")
                    else:
                        logger.warning(f"  ❌ {factor_name}: 无法计算历史IC--正常：因为不满足120个观测点！")
            except Exception as e:
                logger.error(f"  ❌ {factor_name}: IC获取失败 - {e}")
                continue
        
        if not historical_ic_stats:
            logger.error("❌ 无任何因子的历史IC数据，无法计算权重")
            return {}
        
        logger.info(f"📊 成功获取 {len(historical_ic_stats)} 个因子的历史IC数据")
        
        # 第二步：基于历史IC进行质量筛选
        qualified_factor_stats, quality_reports = self.quality_filter.filter_factors_by_quality(
            historical_ic_stats
        )
        
        if not qualified_factor_stats:
            logger.warning("⚠️ 无因子通过质量筛选，返回等权重")
            equal_weight = 1.0 / len(candidate_factor_names)
            return {name: equal_weight for name in candidate_factor_names}
        
        # 第三步：计算权重（仅基于历史IC表现）
        factor_weights = self.weight_calculator.calculate_ic_based_weights(
            qualified_factor_stats
        )
        
        # 第四步：为未通过筛选的因子分配0权重
        final_weights = {}
        for factor_name in candidate_factor_names:
            final_weights[factor_name] = factor_weights.get(factor_name, 0.0)
        
        # 日志记录
        selected_factors = [name for name, weight in final_weights.items() if weight > 0]
        logger.info(f"✅ 滚动权重计算完成 @ {calculation_date}")
        logger.info(f"📊 选中因子: {len(selected_factors)}/{len(candidate_factor_names)}")
        
        for factor_name, weight in sorted(final_weights.items(), key=lambda x: x[1], reverse=True):
            if weight > 0:
                logger.info(f"  🎯 {factor_name}: {weight:.1%}")
        
        return final_weights
    
    def _collect_factor_ic_stats(
        self, 
        factor_names: List[str],
        stock_pool_index_name: str,
        force_recalculate: bool = False
    ) -> Dict[str, Dict[str, Dict]]:
        """收集因子IC统计数据"""
        logger.info("📊 正在收集因子IC统计数据...")
        
        factor_ic_stats = {}
        
        for factor_name in factor_names:
            cache_key = f"{factor_name}_{stock_pool_index_name}"
            
            if not force_recalculate and cache_key in self._ic_stats_cache:
                factor_ic_stats[factor_name] = self._ic_stats_cache[cache_key]
                logger.debug(f"  📥 {factor_name}: 使用缓存数据")
                continue
            
            try:
                # 从已保存的测试结果中读取IC统计
                ic_stats = self._load_factor_ic_stats(factor_name, stock_pool_index_name)
                
                if ic_stats:
                    factor_ic_stats[factor_name] = ic_stats
                    self._ic_stats_cache[cache_key] = ic_stats
                    logger.debug(f"  ✅ {factor_name}: IC数据加载成功")
                else:
                    logger.warning(f"  ⚠️ {factor_name}: 未找到IC统计数据，跳过")
                    
            except Exception as e:
                logger.error(f"  ❌ {factor_name}: 加载IC数据失败 - {e}")
                continue
        
        logger.info(f"📊 IC数据收集完成: {len(factor_ic_stats)}/{len(factor_names)} 个因子")
        return factor_ic_stats
    
    def _load_factor_ic_stats(self, factor_name: str, stock_pool_name: str) -> Optional[Dict]:
        """从保存的测试结果中加载IC统计数据"""
        try:
            # 构建结果文件路径 (基于你现有的保存逻辑)
            from projects._03_factor_selection.factor_manager.storage import ResultStorage
            stats= ResultLoadManager.get_ic_stats_from_local( stock_pool_name,factor_name)

            if stats is None:
                raise ValueError("未找到IC统计数据")
        except Exception as e:
            raise ValueError(f"加载{factor_name}的IC数据失败: {e}")

    def _execute_weighted_synthesis(
        self,
        composite_factor_name: str,
        stock_pool_index_name: str,
        factor_weights: Dict[str, float]
    ) -> pd.DataFrame:
        """执行加权因子合成"""
        logger.info(f"⚖️ 开始执行加权合成，使用{len(factor_weights)}个因子")
        
        processed_factors = []
        weights_list = []
        
        for factor_name, weight in factor_weights.items():
            logger.info(f"  🔄 处理因子: {factor_name} (权重: {weight:.3f})")
            
            # 处理单个因子
            processed_df = self.get_pre_processed_sub_factor_df(factor_name, stock_pool_index_name)
            
            processed_factors.append(processed_df)
            weights_list.append(weight)
        
        if not processed_factors:
            raise ValueError("没有任何因子被成功处理")
        
        # 加权合成
        composite_factor_df = self._weighted_combine_factors(processed_factors, weights_list)
        
        # 最终标准化
        composite_factor_df = self.processor._standardize_robust(composite_factor_df)
        
        logger.info(f"✅ 加权合成完成: {composite_factor_name}")
        return composite_factor_df
    
    def _weighted_combine_factors(
        self, 
        factor_dfs: List[pd.DataFrame], 
        weights: List[float]
    ) -> pd.DataFrame:
        """加权合并因子数据框"""
        if len(factor_dfs) != len(weights):
            raise ValueError("因子数量与权重数量不匹配")
        
        # 确保权重归一化
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 加权合并
        result_df = None
        for i, (df, weight) in enumerate(zip(factor_dfs, weights)):
            weighted_df = df * weight
            
            if result_df is None:
                result_df = weighted_df
            else:
                result_df = result_df.add(weighted_df, fill_value=0)
        
        return result_df
    
    def _generate_synthesis_report(
        self,
        composite_factor_name: str,
        candidate_factors: List[str],
        final_weights: Dict[str, float],
        quality_reports: List[FactorQualityReport]
    ) -> Dict:
        """生成因子合成报告"""
        
        # 按权重排序
        sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
        
        report = {
            'composite_factor_name': composite_factor_name,
            'synthesis_timestamp': pd.Timestamp.now(),
            'candidate_factors_count': len(candidate_factors),
            'qualified_factors_count': len(final_weights),
            'final_weights': dict(sorted_weights),
            'top_3_factors': sorted_weights[:3],
            'config_used': {
                'min_ic_mean': self.config.min_ic_mean,
                'min_ic_ir': self.config.min_ic_ir,
                'min_ic_win_rate': self.config.min_ic_win_rate,
                'max_single_weight': self.config.max_single_weight
            },
            'quality_summary': {
                'passed': len([r for r in quality_reports if not r.risk_flags]),
                'failed': len([r for r in quality_reports if r.risk_flags]),
                'main_failure_reasons': self._summarize_failure_reasons(quality_reports)
            }
        }
        
        return report
    
    def _summarize_failure_reasons(self, quality_reports: List[FactorQualityReport]) -> Dict[str, int]:
        """汇总失败原因统计"""
        failure_counts = {}
        
        for report in quality_reports:
            for flag in report.risk_flags:
                reason = flag.split('(')[0]  # 提取原因主体
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
        
        return failure_counts
    
    def print_synthesis_report(self, report: Dict):
        """打印合成报告"""
        print(f"\n{'='*60}")
        print(f"📊 IC加权因子合成报告")
        print(f"{'='*60}")
        print(f"🎯 合成因子名称: {report['composite_factor_name']}")
        print(f"⏰ 合成时间: {report['synthesis_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📈 候选因子数量: {report['candidate_factors_count']}")
        print(f"✅ 通过筛选数量: {report['qualified_factors_count']}")
        
        print(f"\n🏆 最终权重分配:")
        for factor_name, weight in report['final_weights'].items():
            print(f"  {factor_name:20s}: {weight:6.1%}")
        
        print(f"\n🥇 权重前三名:")
        for i, (factor_name, weight) in enumerate(report['top_3_factors'], 1):
            print(f"  {i}. {factor_name}: {weight:.1%}")
        
        quality_summary = report['quality_summary']
        print(f"\n📋 质量筛选汇总:")
        print(f"  ✅ 通过: {quality_summary['passed']} 个")
        print(f"  ❌ 失败: {quality_summary['failed']} 个")
        
        if quality_summary['main_failure_reasons']:
            print(f"  主要失败原因:")
            for reason, count in quality_summary['main_failure_reasons'].items():
                print(f"    - {reason}: {count} 个因子")
        
        print(f"{'='*60}")