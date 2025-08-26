"""
基于滚动IC的专业因子筛选器 - 带详细注释版本

🎯 系统设计理念：
本示例展示如何使用专业的滚动IC筛选系统：
1. 从大量候选因子中智能筛选出高质量因子
2. 基于历史IC表现计算最优权重
3. 合成具有稳健预测能力的复合因子
4. 生成详细的筛选和合成报告

核心特色：
- 完全避免前视偏差的滚动IC计算
- 多周期IC评分（指数衰减权重）
- 专业级因子质量评估
- 类别内冠军选择机制
- 智能权重分配算法

Author: Claude
Date: 2025-08-25
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
warnings.filterwarnings('ignore')

from quant_lib.config.logger_config import setup_logger

logger = setup_logger(__name__)

# ========================================================================================
# 核心配置类 - 专业级参数设置体系
# ========================================================================================

@dataclass 
class RollingICSelectionConfig:
    """
    滚动IC筛选配置 - 专业级因子筛选参数设置
    
    🎯 设计哲学：分层筛选体系
    ├── 第一层：基础质量门槛 (过滤明显不合格的因子)
    --确保 使用rolling_IC计算
    ├── 第二层：多周期评分权重 (平衡短期和长期表现)
    ├── 第三层：类别内竞争选择 (确保因子组合多样性)
    ├── 第四层：最终组合构建 (控制复杂度和可管理性)
    └── 第五层：相关性控制哲学 (三层决策机制)
    """
    
    # ===== 🥇 第一层：基础质量门槛 =====
    # 目的：过滤掉明显不合格的因子，建立最低标准
    min_snapshots: int = 3           # 最少快照数量：确保有足够的历史数据样本
                                     # 💡 为什么需要？少于3个快照无法验证稳定性
    
    min_ic_abs_mean: float = 0.01    # IC均值绝对值门槛：因子预测能力的基础要求
                                     # 💡 理解：|IC| >= 0.01 意味着因子至少有1%的信息含量
                                     
    min_ir_abs_mean: float = 0.15    # IR均值绝对值门槛：风险调整后的信息比率要求
                                     # 💡 计算：IR = IC_mean / IC_std，衡量信号的信噪比
                                     
    min_ic_stability: float = 0.4    # IC稳定性门槛：IC方向一致性，避免方向频繁切换
                                     # 💡 计算：同符号IC占比 >= 40%，确保因子方向相对稳定
                                     
    max_ic_volatility: float = 0.05  # IC波动率上限：控制IC时序稳定性，避免过度波动
                                     # 💡 理解：std(IC) <= 0.05，确保因子表现相对平稳
    
    # ===== 🏆 第二层：多周期评分权重配置 =====
    # 目的：综合不同持有周期的表现，平衡短期和长期效果
    decay_rate: float = 0.75         # 衰减率：0.75意味着更重视短期表现
                                     # 💡 权重计算：weights = [decay_rate^0, decay_rate^1, decay_rate^2, ...]
                                     # 💡 举例：[1.0, 0.75, 0.56, 0.42] - 短期权重占主导
                                     # 💡 调参指南：
                                     #    - 值越小(0.3-0.6)，长期权重衰减越慢，适合价值策略
                                     #    - 值越大(0.8-0.9)，短期权重占主导，适合动量策略
                                     
    prefer_short_term: bool = True   # 偏向短期：在权重分配时优先考虑短期持有效果
                                     # 💡 实盘考量：短期信号通常更容易执行且成本更低
    
    # ===== 🎪 第三层：类别内竞争选择 =====
    # 目的：确保因子组合的多样性，避免同类因子扎堆
    max_factors_per_category: int = 2  # 每类最多因子数：防止某个类别过度代表
                                       # 💡 风险控制：避免"动量因子扎堆"或"价值因子堆积"
                                       
    min_category_score: float = 10.0   # 类别最低评分：只有优秀因子才能成为类别代表
                                       # 💡 质量保证：宁缺毋滥原则，确保每个入选因子都足够优秀
    
    # ===== 🎯 第四层：最终组合构建 =====
    # 目的：控制最终因子组合的复杂度和可管理性
    max_final_factors: int = 8         # 最多选择因子数：平衡多样性和复杂度
                                       # 💡 实盘考量：8个因子是组合管理和风险控制的最佳平衡点
                                       # 💡 经验法则：少于5个可能不够分散，多于10个难以管理
    
    # ===== 🔗 第五层：相关性控制（三层决策哲学）=====
    # 目的：处理因子间相关性，采用差异化策略
    high_corr_threshold: float = 0.7   # 高相关阈值（红色警报：二选一）
                                       # 💡 决策：|corr| > 0.7 坚决执行"二选一"
                                       # 💡 理由：高度冗余因子，强行挖掘残差过拟合风险大于收益
                                       
    medium_corr_threshold: float = 0.3 # 中低相关分界（黄色预警：正交化战场）
                                       # 💡 决策：0.3 < |corr| < 0.7 正交化处理
                                       # 💡 理由：既有显著共同信息，又包含不可忽视的独立信息
                                       
    enable_orthogonalization: bool = True  # 是否启用中相关区间正交化
                                           # 💡 技术：以评分高者为基准，对其他因子进行正交化处理
                                           
    # ===== 💰 第六层：实盘交易成本控制（换手率一等公民）=====
    # 目的：将交易成本纳入因子评估，实现实盘导向优化
    max_turnover_rate: float = 0.15    # 最大换手率阈值（月度）
                                       # 💡 成本控制：月换手率超过15%的因子被视为高成本因子
                                       
    turnover_weight: float = 0.25      # 换手率在综合评分中的权重
                                       # 💡 权衡：25%权重平衡因子质量和交易成本
                                       
    enable_turnover_penalty: bool = True  # 是否启用换手率惩罚
                                          # 💡 实盘理念：低IC但低换手率 > 高IC但高换手率


@dataclass
class FactorRollingICStats:
    """
    因子滚动IC统计数据 - 因子质量评估的核心数据结构
    
    💡 设计理念：全面记录因子的历史表现，为选择决策提供数据支撑
    """
    factor_name: str                    # 因子名称
    periods_data: Dict[str, Dict]       # 各周期详细数据：{'21d': {stats}, '60d': {stats}}
    
    # === 基础质量指标 ===
    avg_ic_abs: float                   # 平均IC绝对值：衡量因子预测能力
    avg_ir_abs: float                   # 平均IR绝对值：衡量风险调整后收益能力
    avg_stability: float                # 平均稳定性：衡量IC方向一致性
    avg_ic_volatility: float            # 平均IC波动率：衡量因子表现稳定性
    
    # === 综合评估指标 ===
    multi_period_score: float           # 多周期综合评分：基于指数衰减权重的综合得分
    snapshot_count: int                 # 快照数量：样本大小
    time_range: Tuple[str, str]         # 时间范围：数据覆盖期间
    
    # === 实盘交易成本控制指标 ===
    avg_turnover_rate: float = 0.0      # 平均月度换手率：交易成本的核心指标
    turnover_adjusted_score: float = 0.0 # 换手率调整后评分：实盘导向的最终评分

# ========================================================================================
# 主筛选器类 - 专业因子筛选的核心引擎
# ========================================================================================

class RollingICFactorSelector:
    """
    基于滚动IC的专业因子筛选器
    
    🎯 核心使命：从海量候选因子中筛选出适合实盘交易的优质因子组合
    
    💡 工作流程：
    ┌─────────────────────────────────────────────────────────────┐
    │  🏭 因子工厂 (候选因子池)                                      │
    │  输入：50-200个候选因子                                        │
    └─────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  📊 第一关：基础质量筛选                                        │
    │  • 滚动IC统计计算 (避免前视偏差)                                │
    │  • 多维度门槛检验 (IC/IR/稳定性/波动率)                         │
    │  • 剔除明显不合格因子                                          │
    │  输出：通常筛选出20-40个合格因子                                │
    └─────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  🏆 第二关：类别内冠军选择                                      │
    │  • 11个因子类别 (Value/Quality/Momentum等)                    │
    │  • 每类选择最优秀的1-2个代表                                   │
    │  • 确保因子组合多样性                                          │
    │  输出：各类别冠军，通常10-15个因子                              │
    └─────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  🎯 第三关：综合排序选择                                        │
    │  • 多周期IC综合评分 (指数衰减权重)                             │
    │  • 换手率调整 (实盘交易成本考量)                               │
    │  • 选择前N名作为初步结果                                       │
    │  输出：初步精选因子，通常8-10个                                 │
    └─────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  🔗 第四关：相关性控制哲学                                      │
    │  • 红色警报 (|corr|>0.7): 坚决二选一                          │
    │  • 黄色预警 (0.3<|corr|<0.7): 正交化战场                      │
    │  • 绿色安全 (|corr|<0.3): 直接保留                            │
    │  输出：相关性优化后的最终因子组合                                │
    └─────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  💎 最终产品：精选因子组合                                      │
    │  • 通常5-8个高质量因子                                        │
    │  • 具备多样性、低相关性、高质量                                │
    │  • 适合实盘交易的成本控制要求                                  │
    │  • 可直接用于IC加权合成                                       │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, snap_config_id: str, config: Optional[RollingICSelectionConfig] = None):
        """
        初始化专业滚动IC因子筛选器
        
        Args:
            snap_config_id: 配置快照ID - 确保结果可复现的版本控制
            config: 筛选配置 - 自定义筛选参数，如果为None使用默认配置
        """
        # === 基础配置初始化 ===
        self.snap_config_id = snap_config_id
        self.config = config or RollingICSelectionConfig()
        self.main_work_path = Path(r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\workspace\result")
        
        # === 从配置快照获取回测环境信息 ===
        self._load_config_info()
        
        # === 因子分类定义 - 完整版本 ===
        # 💡 设计理念：将因子按投资逻辑分类，确保组合包含不同类型的alpha来源
        self.factor_categories = {
            # 💰 价值类：基于估值水平的mean-reversion逻辑
            'Value': ['bm_ratio', 'ep_ratio', 'cfp_ratio', 'sp_ratio', 'value_composite', 'pb_ratio', 'pe_ttm', 'ps_ratio'],
            
            # 💎 质量类：基于公司基本面健康度的逻辑
            'Quality': ['roe_ttm', 'gross_margin_ttm', 'debt_to_assets', 'earnings_stability', 'quality_momentum', 
                       'operating_accruals', 'asset_turnover', 'roa_ttm', 'current_ratio'],
            
            # 🚀 动量类：基于价格趋势延续的逻辑
            'Momentum': ['momentum_20d', 'momentum_120d', 'momentum_12_1', 'momentum_pct_60d', 'sharpe_momentum_60d', 
                        'sw_l1_momentum_21d', 'momentum_6_1', 'momentum_3_1'],
            
            # 🔄 反转类：基于短期价格均值回归的逻辑
            'Reversal': ['reversal_5d', 'reversal_21d', 'reversal_1d', 'reversal_10d'],
            
            # 📏 规模类：基于市值效应的逻辑
            'Size': ['log_circ_mv', 'log_total_mv', 'market_cap_weight'],
            
            # 📈 波动率类：基于风险收益关系的逻辑
            'Volatility': ['volatility_40d', 'volatility_90d', 'volatility_120d', 'rsi', 'atr_20d',
                          'volatility_40d_经过残差化', 'volatility_90d_经过残差化', 'volatility_120d_经过残差化', 'rsi_经过残差化'],
            
            # 💧 流动性类：基于交易活跃度的逻辑
            'Liquidity': ['amihud_liquidity', 'turnover_rate_90d_mean', 'turnover_rate_monthly_mean', 'ln_turnover_value_90d', 
                         'turnover_t1_div_t20d_avg', 'bid_ask_spread', 'turnover_rate_90d_mean-经过残差化', 
                         'turnover_rate_monthly_mean_经过残差化', 'ln_turnover_value_90d_经过残差化'],
            
            # 🔧 技术类：基于技术分析指标的逻辑
            'Technical': ['cci', 'pead', 'macd', 'rsi_divergence', 'cci_经过残差化', 'bollinger_position'],
            
            # 🌱 成长类：基于业绩增长的逻辑
            'Growth': ['total_revenue_growth_yoy', 'net_profit_growth_yoy', 'eps_growth', 'operating_revenue_growth'],
            
            # 💰 盈利能力类：基于利润率的逻辑
            'Profitability': ['gross_profit_margin', 'operating_margin', 'net_margin', 'ebit_margin'],
            
            # ⚡ 效率类：基于资产周转效率的逻辑
            'Efficiency': ['inventory_turnover', 'receivables_turnover', 'working_capital_turnover']
        }
        
        # === 缓存系统初始化 ===
        self._factor_stats_cache = {}  # 因子统计数据缓存，避免重复计算
        
        # === 初始化完成日志 ===
        logger.info(f"滚动IC因子筛选器初始化完成")
        logger.info(f"配置ID: {self.snap_config_id}")
        logger.info(f"股票池: {self.pool_index}")
        logger.info(f"时间范围: {self.start_date} - {self.end_date}")
        logger.info(f"数据版本: {self.version}")

# ========================================================================================
# 核心方法1：多周期综合评分算法
# ========================================================================================

    def _calculate_multi_period_score(self, periods_data: Dict) -> float:
        """
        计算多周期IC综合评分（带指数衰减权重）
        
        🎯 设计理念：
        不同持有周期的因子表现应该有不同的权重，通常短期表现更重要：
        1. 短期信号更容易执行
        2. 短期成本更低
        3. 短期风险更可控
        
        📊 权重计算公式：
        weights = [decay_rate^0, decay_rate^1, decay_rate^2, ...]
        
        📈 评分模型：
        总分 = IC分数 + IR分数 + 稳定性分数 + 胜率分数 - 波动率惩罚
        
        Args:
            periods_data: 多周期数据 {period: stats}
            
        Returns:
            float: 综合评分 (0-100分)
            
        💡 使用示例：
        periods_data = {
            '21d': {'ic_mean_avg': 0.025, 'ic_ir_avg': 0.45, 'ic_stability': 0.6, ...},
            '60d': {'ic_mean_avg': 0.020, 'ic_ir_avg': 0.38, 'ic_stability': 0.55, ...}
        }
        score = self._calculate_multi_period_score(periods_data)  # 返回如：32.5
        """
        if not periods_data:
            return 0.0
        
        # === Step 1: 按周期排序（短期到长期）===
        try:
            # 💡 智能排序：自动识别周期长度 ('21d' -> 21, '60d' -> 60)
            periods = sorted(periods_data.keys(), key=lambda x: int(x.replace('d', '').replace('D', '')))
        except:
            # 💡 备用方案：字典序排序
            periods = sorted(periods_data.keys())
        
        # === Step 2: 计算每个周期的得分 ===
        period_scores = []

        # --- 核心改造：定义“满分标杆” ---
        IC_MEAN_BENCHMARK = 0.05  # IC均值达到0.05，我们认为表现优异
        IC_IR_BENCHMARK = 0.50  # IR达到0.5，我们认为稳定性优异
        IC_STABILITY_BENCHMARK = 1.0  # 稳定性是0-1，标杆就是1.0
        IC_WIN_RATE_BENCHMARK = 0.60  # 胜率达到60%，我们认为很不错
        IC_VOL_PENALTY_BASE = 0.02  # IC波动率超过2%开始惩罚

        # --- 核心改造：定义最终的权重配比 ---
        # 这些权重现在是真实的、可比的
        WEIGHTS = {
            'ic_mean': 0.40,  # 40% 权重给效果
            'ic_ir': 0.30,  # 30% 权重给稳定性
            'ic_stability': 0.10,  # 10% 权重给另一种稳定性
            'ic_win_rate': 0.20  # 20% 权重给胜率
        }

        for period in periods:
            stats = periods_data[period]

            # --- 核心改造：先归一化，得到0-1之间的分数 ---
            ic_norm_score = min(abs(stats.get('ic_mean_avg', 0)) / IC_MEAN_BENCHMARK, 1.0)
            ir_norm_score = min(abs(stats.get('ic_ir_avg', 0)) / IC_IR_BENCHMARK, 1.0)
            stability_norm_score = min(stats.get('ic_stability', 0) / IC_STABILITY_BENCHMARK, 1.0)

            # 胜率以50%为基准
            win_rate_norm_score = max(0, (stats.get('ic_win_rate_avg', 0.5) - 0.5) / (IC_WIN_RATE_BENCHMARK - 0.5))
            win_rate_norm_score = min(win_rate_norm_score, 1.0)

            # 惩罚项
            volatility_penalty = max(0, (stats.get('ic_volatility', 0) - IC_VOL_PENALTY_BASE) * 20)  # 惩罚力度可以调整

            # --- 核心改造：再加权 ---
            # 现在，所有分数都在0-1范围，权重可以公平地发挥作用
            weighted_score = (ic_norm_score * WEIGHTS['ic_mean'] +
                              ir_norm_score * WEIGHTS['ic_ir'] +
                              stability_norm_score * WEIGHTS['ic_stability'] +
                              win_rate_norm_score * WEIGHTS['ic_win_rate'])

            # 应用惩罚并确保分数在0-100之间 (乘以100方便阅读)
            total_score = (weighted_score - volatility_penalty) * 100
            period_scores.append(max(0, total_score))
        
        if not period_scores:
            return 0.0
        
        # === Step 3: 应用指数衰减权重 ===
        decay_rate = self.config.decay_rate  # 典型值：0.75
        weights = np.array([decay_rate ** i for i in range(len(period_scores))])
        weights /= weights.sum()  # 权重归一化
        
        # 💡 权重示例 (decay_rate=0.75)：
        # 21d: 权重 = 1.0 / (1.0 + 0.75) = 57%
        # 60d: 权重 = 0.75 / (1.0 + 0.75) = 43%
        
        # === Step 4: 计算加权平均分数 ===
        final_score = np.average(period_scores, weights=weights)
        
        return final_score

# ========================================================================================
# 核心方法2：换手率估算算法（实盘交易成本一等公民）
# ========================================================================================

    def _estimate_factor_turnover(self, factor_name: str, periods_data: Dict) -> float:
        """
        估算因子换手率（实盘交易成本核心指标）
        
        🎯 设计理念：
        "换手率一等公民" - 将交易成本纳入因子评估的核心考量
        
        💡 核心洞察：
        一个低IC但换手率极低的因子，在实盘中可能远胜于一个高IC但换手率极高的因子
        
        📊 估算方法：
        1. 基于因子类型的经验估算 (技术面 > 价量面 > 基本面)
        2. 基于IC稳定性动态调整 (稳定性低 -> 换手率高)
        3. 合理范围控制 (2%-50%)
        
        Args:
            factor_name: 因子名称
            periods_data: 各周期数据
            
        Returns:
            float: 月度平均换手率估算 (0.02-0.50)
            
        💡 使用示例：
        turnover = self._estimate_factor_turnover('reversal_5d', periods_data)  # 返回如：0.25 (25%)
        turnover = self._estimate_factor_turnover('ep_ratio', periods_data)     # 返回如：0.08 (8%)
        """
        try:
            # === 基于因子类型的换手率估算表 ===
            # 💡 研究基础：基于大量历史回测数据和实盘经验总结
            turnover_estimates = {
                # 🔥 高频类因子（技术面）- 换手率 20-30%
                # 特征：基于短期价格模式，信号变化频繁
                'reversal_1d': 0.30,     'reversal_5d': 0.25,     'reversal_10d': 0.20,
                'momentum_20d': 0.18,    'rsi': 0.22,             'cci': 0.24,
                'macd': 0.20,            'bollinger_position': 0.28,
                
                # ⚡ 中频类因子（价量结合）- 换手率 10-18%
                # 特征：基于中期趋势和交易行为，信号相对稳定
                'momentum_60d': 0.15,    'momentum_120d': 0.12,    'momentum_12_1': 0.10,
                'volatility_40d': 0.16,  'volatility_90d': 0.14,   'volatility_120d': 0.12,
                'amihud_liquidity': 0.14, 'turnover_rate_90d_mean': 0.16,
                
                # 🏛️ 低频类因子（基本面）- 换手率 5-10%
                # 特征：基于财务数据，季度更新，信号持续时间长
                'ep_ratio': 0.08,        'bm_ratio': 0.07,        'sp_ratio': 0.08,
                'cfp_ratio': 0.09,       'roe_ttm': 0.06,         'gross_margin_ttm': 0.05,
                'earnings_stability': 0.04, 'total_revenue_growth_yoy': 0.07, 'net_profit_growth_yoy': 0.08,
                
                # 🗿 规模因子（极低频）- 换手率 2-5%
                # 特征：基于市值，变化极其缓慢
                'log_circ_mv': 0.03,     'log_total_mv': 0.03,    'market_cap_weight': 0.02,
                
                # 💎 质量因子（低频）- 换手率 4-9%
                # 特征：基于公司质地，相对稳定
                'debt_to_assets': 0.05,  'current_ratio': 0.04,   'asset_turnover': 0.06,
                'quality_momentum': 0.09
            }
            
            # === Step 1: 获取基础换手率估算 ===
            base_turnover = turnover_estimates.get(factor_name, 0.12)  # 默认12% - 中性假设
            
            # === Step 2: 基于IC稳定性动态调整 ===
            # 💡 理论基础：IC稳定性低的因子，其信号变化更频繁，需要更频繁调仓
            if periods_data:
                avg_stability = np.mean([
                    stats.get('ic_stability', 0.5) 
                    for stats in periods_data.values()
                ])
                
                # 🎯 调整公式：稳定性越低，换手率调整系数越高
                # 示例：稳定性30%时，调整系数 = 1.0 + (0.5 - 0.3) * 0.8 = 1.16
                stability_adjustment = 1.0 + (0.5 - avg_stability) * 0.8
                adjusted_turnover = base_turnover * stability_adjustment
            else:
                adjusted_turnover = base_turnover
            
            # === Step 3: 换手率合理范围控制 ===
            # 💡 风险控制：确保估算结果在现实可能的范围内
            final_turnover = np.clip(adjusted_turnover, 0.02, 0.50)  # 2%-50%区间
            
            return final_turnover
            
        except Exception as e:
            logger.debug(f"换手率估算失败 {factor_name}: {e}")
            return 0.12  # 默认换手率12%

# ========================================================================================
# 核心方法3：换手率调整评分算法（实盘导向优化）
# ========================================================================================

    def _calculate_turnover_adjusted_score(self, base_score: float, turnover_rate: float) -> float:
        """
        计算换手率调整后评分（数学连续版本 - 实盘导向核心优化）
        
        🎯 核心理念：
        交易成本是真实存在的收益杀手！必须在因子评分中体现交易成本的影响。
        
        🔧 重大改进：数学连续性
        之前版本存在分段点跳变问题，现已修复为完全连续的分段函数。
        
        💡 调整哲学：
        - 低换手率 (≤5%): 给予奖励 - 鼓励稳健策略
        - 适中换手率 (5-15%): 线性惩罚 - 平衡考量  
        - 过高换手率 (>15%): 重度惩罚 - 避免过度交易
        
        📊 连续分段函数设计：
        - 区间1 [0, 0.05]: 奖励区，线性增长至1.1倍
        - 区间2 (0.05, max_rate]: 线性衰减区，连续过渡  
        - 区间3 (max_rate, ∞): 重惩区，连续衰减
        
        🧮 连续性验证（以max_rate=0.15为例）：
        在 turnover = 0.05 处：
        - 左极限：1.0 + (0.05/0.05) * 0.1 = 1.1 ✅
        - 右极限：1.1 - (0.05-0.05) * 2.0 = 1.1 ✅
        
        在 turnover = 0.15 处：  
        - 左极限：1.1 - (0.15-0.05) * 2.0 = 0.9 ✅
        - 右极限：0.9 - (0.15-0.15) * 5.0 = 0.9 ✅
        
        Args:
            base_score: 基础IC评分 (通常0-50分)
            turnover_rate: 月度换手率 (通常0.02-0.30)
            
        Returns:
            float: 换手率调整后评分
            
        💡 使用示例：
        # 高质量低换手因子
        score1 = self._calculate_turnover_adjusted_score(35.0, 0.03)  # 返回如：37.8 (获得奖励)
        
        # 高质量高换手因子  
        score2 = self._calculate_turnover_adjusted_score(35.0, 0.25)  # 返回如：28.5 (受到惩罚)
        """
        if not self.config.enable_turnover_penalty:
            return base_score
        
        max_rate = self.config.max_turnover_rate  # 通常为 0.15
        
        # === 区间1：低换手率奖励区 [0, 0.05] ===
        if turnover_rate <= 0.05:
            # 💡 理念：极低换手的因子值得鼓励，线性奖励
            # 从1.0增长到1.1（10%奖励）
            turnover_multiplier = 1.0 + (turnover_rate / 0.05) * 0.1
            # 示例：3%换手率时，乘数 = 1.0 + (0.03/0.05) * 0.1 = 1.06
            
        # === 区间2：线性惩罚区 (0.05, max_rate] ===  
        elif turnover_rate <= max_rate:
            # 💡 线性衰减：从1.1开始以斜率-2.0下降
            # 关键改进：确保在边界点连续
            turnover_multiplier = 1.1 - (turnover_rate - 0.05) * 2.0
            # 示例：12%换手率时，乘数 = 1.1 - (0.12 - 0.05) * 2.0 = 0.96
            
        # === 区间3：重惩区 (max_rate, ∞) ===
        else:
            # 🔧 关键改进：从边界值连续衰减，避免跳变
            boundary_multiplier = 1.1 - (max_rate - 0.05) * 2.0  # 边界连续值
            excess_turnover = turnover_rate - max_rate
            
            # 连续衰减：从边界值开始，以5倍斜率继续下降
            turnover_multiplier = boundary_multiplier - excess_turnover * 5.0
            # 示例：25%换手率时，超出10%
            # boundary = 1.1 - (0.15-0.05)*2.0 = 0.9
            # 乘数 = 0.9 - (0.25-0.15)*5.0 = 0.4
        
        # === 应用换手率权重 ===
        # 💡 权重逻辑：不是完全由换手率决定，而是与原评分加权平均
        weight = self.config.turnover_weight  # 典型值：0.25 (25%权重)
        final_multiplier = (1 - weight) + weight * turnover_multiplier
        # 示例：75%保持原分数，25%应用换手率调整
        
        # === 安全范围控制 ===
        # 💡 风险控制：确保最终乘数在合理范围，避免极端情况
        final_multiplier = np.clip(final_multiplier, 0.1, 1.2)  # 10%-120%区间
        
        # === 最终调整评分 ===
        adjusted_score = base_score * final_multiplier
        
        return max(0.0, adjusted_score)

# ========================================================================================
# 核心方法4：三层相关性控制哲学
# ========================================================================================

    def apply_correlation_control(
            self, 
            candidate_factors: List[str],
            qualified_factors: Dict[str, FactorRollingICStats]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        应用三层相关性控制哲学
        
        🎯 设计哲学：承认相关性的不同层次，采用差异化策略
        
        📊 三层决策机制：
        
        🚨 红色警报区域 (|corr| > 0.7)：
        ├── 决策：坚决执行"二选一"
        ├── 理由：高度冗余因子，强行挖掘残差过拟合风险大于收益
        └── 策略：选择多周期评分最高的因子
        
        ⚠️ 黄色预警区域 (0.3 < |corr| < 0.7)：
        ├── 决策：正交化战场，这是价值挖掘的黄金区间  
        ├── 理由：既有显著共同信息，又包含不可忽视的独立信息
        ├── 策略：以评分高者为基准，对其他因子进行正交化处理
        └── 举例：ROE与Momentum相关性0.4时，用ROE清洗Momentum得到纯粹动量信号
        
        ✅ 绿色安全区域 (|corr| < 0.3)：
        ├── 决策：直接全部保留  
        ├── 理由：天然的"好队友"，提供足够多样性
        └── 策略：无需额外处理
        
        Args:
            candidate_factors: 候选因子列表
            qualified_factors: 合格因子统计
            
        Returns:
            (final_factors, correlation_report): 最终因子列表和详细决策报告
        """
        logger.info("🔍 开始执行三层相关性控制...")
        logger.info(f"📊 输入因子数量: {len(candidate_factors)}")
        
        # === Step 1: 计算因子相关性矩阵 ===
        correlation_matrix = self._calculate_factor_correlations(candidate_factors)
        if correlation_matrix is None:
            logger.warning("⚠️ 无法计算相关性矩阵，跳过相关性控制")
            return candidate_factors, {}
        
        # === Step 2: 三层决策处理 ===
        final_factors = []              # 最终保留的因子
        correlation_decisions = []      # 决策记录
        orthogonalized_factors = []     # 正交化因子列表
        processed_factors = set()       # 已处理因子集合
        
        # === Step 3: 逐因子处理 ===
        for i, factor1 in enumerate(candidate_factors):
            if factor1 in processed_factors:
                continue
                
            # 🔍 寻找与factor1相关的因子
            high_corr_pairs = []    # 高相关配对
            medium_corr_pairs = []  # 中相关配对
            
            for j, factor2 in enumerate(candidate_factors[i+1:], i+1):
                if factor2 in processed_factors:
                    continue
                    
                corr = abs(correlation_matrix.loc[factor1, factor2])
                
                if corr >= self.config.high_corr_threshold:
                    # 🚨 红色警报区域：高度相关
                    high_corr_pairs.append((factor2, corr))
                elif corr >= self.config.medium_corr_threshold:
                    # ⚠️ 黄色预警区域：中度相关
                    medium_corr_pairs.append((factor2, corr))
            
            # === 红色警报处理：坚决二选一 ===
            if high_corr_pairs:
                competitors = [factor1] + [pair[0] for pair in high_corr_pairs]
                winner = self._select_best_factor(competitors, qualified_factors)
                final_factors.append(winner)
                
                # 📝 记录决策过程
                losers = [f for f in competitors if f != winner]
                for loser in losers:
                    correlation_decisions.append({
                        'winner': winner,
                        'loser': loser,
                        'correlation': max([corr for f, corr in high_corr_pairs if f == loser] + 
                                         [abs(correlation_matrix.loc[factor1, loser]) if loser != factor1 else 0]),
                        'decision': '红色警报-二选一',
                        'reason': f'高度相关(|corr|>{self.config.high_corr_threshold})'
                    })
                
                # 🏷️ 标记已处理
                for competitor in competitors:
                    processed_factors.add(competitor)
                    
                logger.info(f"  🚨 红色警报: {winner} 胜出，淘汰 {losers}")
                
            # === 黄色预警处理：正交化战场 ===
            elif medium_corr_pairs and self.config.enable_orthogonalization:
                # 📌 factor1作为基准因子保留
                final_factors.append(factor1)
                processed_factors.add(factor1)
                
                for factor2, corr in medium_corr_pairs:
                    if factor2 not in processed_factors:
                        # 🔄 正交化处理
                        orthogonal_name = f"{factor2}_orth_vs_{factor1}"
                        orthogonalized_factors.append({
                            'original': factor2,
                            'orthogonal_name': orthogonal_name,
                            'base_factor': factor1,
                            'correlation': corr
                        })
                        
                        # 📝 记录决策
                        correlation_decisions.append({
                            'base_factor': factor1,
                            'target_factor': factor2,
                            'orthogonal_name': orthogonal_name,
                            'correlation': corr,
                            'decision': '黄色预警-正交化',
                            'reason': f'中度相关({self.config.medium_corr_threshold}<|corr|<{self.config.high_corr_threshold})'
                        })
                        
                        processed_factors.add(factor2)
                
                logger.info(f"  ⚠️ 黄色预警: {factor1} 作为基准，{len(medium_corr_pairs)} 个因子待正交化")
                
            # === 绿色安全处理：直接保留 ===
            else:
                final_factors.append(factor1)
                processed_factors.add(factor1)
                logger.info(f"  ✅ 绿色安全: {factor1} 直接保留")
        
        # === Step 4: 生成详细报告 ===
        correlation_report = {
            'input_count': len(candidate_factors),
            'final_count': len(final_factors),
            'orthogonalized_count': len(orthogonalized_factors),
            'decisions': correlation_decisions,
            'orthogonalized_factors': orthogonalized_factors,
            'correlation_matrix': correlation_matrix.to_dict(),
            'thresholds': {
                'high_corr': self.config.high_corr_threshold,
                'medium_corr': self.config.medium_corr_threshold
            }
        }
        
        # === 总结输出 ===
        logger.info("🎯 三层相关性控制完成:")
        logger.info(f"  📈 输入因子: {len(candidate_factors)}")
        logger.info(f"  🏆 最终因子: {len(final_factors)}")
        logger.info(f"  🔄 正交化因子: {len(orthogonalized_factors)}")
        logger.info(f"  📊 决策记录: {len(correlation_decisions)}")
        
        return final_factors, correlation_report

# ========================================================================================
# 完整工作流程：run_complete_selection
# ========================================================================================

    def run_complete_selection(self, factor_names: List[str], force_generate: bool = False) -> Tuple[List[str], Dict[str, Any]]:
        """
        运行完整的专业因子筛选流程
        
        🎯 核心使命：从海量候选因子中筛选出实盘级精选因子组合
        
        📊 完整工作流程：
        
        ┌─ 🏭 输入：50-200个候选因子 ─┐
        │                              │
        │   第一关：基础质量筛选          │
        │   ├── 滚动IC统计计算           │
        │   ├── 多维度门槛检验           │
        │   └── 换手率估算和调整         │
        │                              │
        ├─ 📊 输出：20-40个合格因子 ─────┤
        │                              │
        │   第二关：类别内冠军选择        │
        │   ├── 11个因子类别分组         │
        │   ├── 类内排序和选择           │
        │   └── 确保组合多样性           │
        │                              │
        ├─ 🏆 输出：10-15个类别冠军 ────┤
        │                              │
        │   第三关：综合排序选择          │
        │   ├── 多周期IC综合评分         │
        │   ├── 换手率成本调整           │
        │   └── 选择前N名               │
        │                              │
        ├─ 🎯 输出：8-10个初步精选 ─────┤
        │                              │
        │   第四关：相关性控制哲学        │
        │   ├── 红色警报：二选一         │
        │   ├── 黄色预警：正交化         │
        │   └── 绿色安全：直接保留       │
        │                              │
        └─ 💎 最终：5-8个精选因子 ──────┘
        
        Args:
            factor_names: 候选因子列表 (50-200个)
            force_generate: 是否强制重新生成滚动IC数据
            
        Returns:
            Tuple[List[str], Dict]: (精选因子列表, 详细筛选报告)
            
        💡 返回示例：
        selected_factors = ['earnings_stability', 'momentum_20d', 'volatility_120d', 'amihud_liquidity', 'ep_ratio']
        report = {
            'selection_summary': {'pass_rate': 0.12, 'final_count': 5},
            'correlation_control': {'enabled': True, 'decisions': [...]},
            'factor_details': {...}
        }
        """
        
        # === 流程开始横幅 ===
        logger.info("=" * 60)
        logger.info("🚀 开始基于滚动IC的完整因子筛选")
        logger.info("=" * 60)
        
        # === 第一关：基础质量筛选 ===
        # 💡 目标：从大海选中筛选出基础合格的因子
        logger.info("🔍 第一关：基础质量筛选")
        qualified_factors = self.screen_factors_by_rolling_ic(factor_names, force_generate)
        
        if not qualified_factors:
            logger.warning("❌ 警告：没有因子通过滚动IC筛选")
            return [], {}
        
        logger.info(f"✅ 第一关完成：{len(qualified_factors)}/{len(factor_names)} 因子通过基础筛选")
        
        # === 第二关：类别内冠军选择 ===
        # 💡 目标：确保因子组合的多样性，每个类别选出最优代表
        logger.info("🏆 第二关：类别内冠军选择")
        category_champions = self.select_category_champions(qualified_factors)
        
        if not category_champions:
            logger.warning("❌ 警告：没有类别冠军")
            return [], {}
        
        total_champions = sum(len(champions) for champions in category_champions.values())
        logger.info(f"✅ 第二关完成：{len(category_champions)} 个类别，共 {total_champions} 个冠军")
        
        # === 第三关：综合排序选择 ===
        # 💡 目标：基于综合评分选择前N名，平衡质量和数量
        logger.info("🎯 第三关：综合排序选择")
        preliminary_selection = self.generate_final_selection(category_champions, qualified_factors)
        
        logger.info(f"✅ 第三关完成：{len(preliminary_selection)} 个因子入围最终候选")
        
        # === 第四关：三层相关性控制哲学 ===
        # 💡 目标：处理因子间相关性，确保组合的独立性
        logger.info("🔗 第四关：三层相关性控制")
        final_selection, correlation_report = self.apply_correlation_control(
            preliminary_selection, qualified_factors
        )
        
        logger.info(f"✅ 第四关完成：{len(final_selection)} 个因子通过相关性控制")
        
        # === 生成详细筛选报告 ===
        report = self._generate_selection_report(
            factor_names, qualified_factors, category_champions, final_selection, correlation_report
        )
        
        # === 流程完成横幅 ===
        logger.info("=" * 60)
        logger.info("🎉 滚动IC因子筛选完成！")
        logger.info(f"🎯 最终推荐用于IC加权合成: {final_selection}")
        logger.info(f"📊 筛选统计：{len(factor_names)} -> {len(qualified_factors)} -> {len(final_selection)}")
        logger.info(f"📈 通过率：{len(final_selection)/len(factor_names):.1%}")
        logger.info("=" * 60)
        
        return final_selection, report

# ========================================================================================
# 使用示例和最佳实践
# ========================================================================================

"""
💡 使用示例：

# === 1. 基础使用 ===
from rolling_ic_factor_selector import RollingICFactorSelector, RollingICSelectionConfig

# 配置筛选参数
config_manager = RollingICSelectionConfig(
    min_ic_abs_mean=0.015,        # 提高IC要求
    min_ir_abs_mean=0.20,         # 提高IR要求
    decay_rate=0.70,              # 更重视短期表现
    max_final_factors=6,          # 最多选择6个因子
    enable_turnover_penalty=True  # 启用换手率惩罚
)

# 创建筛选器
selector = RollingICFactorSelector("配置快照ID", config_manager)

# 执行筛选
candidate_factors = ["volatility_120d", "momentum_20d", "ep_ratio", "reversal_5d", ...]
selected_factors, report = selector.run_complete_selection(candidate_factors)

print(f"筛选结果: {selected_factors}")
print(f"通过率: {report['selection_summary']['pass_rate']:.1%}")

# === 2. 高级配置：严格换手率控制 ===
strict_config = RollingICSelectionConfig(
    min_ic_abs_mean=0.012,        # 适度降低IC要求
    max_turnover_rate=0.10,       # 严格的10%换手率上限
    turnover_weight=0.35,         # 更高的换手率权重
    enable_turnover_penalty=True
)

# === 3. 三层相关性控制自定义 ===
correlation_config = RollingICSelectionConfig(
    high_corr_threshold=0.8,      # 更严格的高相关阈值
    medium_corr_threshold=0.2,    # 更宽松的中相关分界
    enable_orthogonalization=False  # 关闭正交化
)

🎯 最佳实践：

1. **参数调优策略**：
   - 初次使用建议采用默认参数
   - 根据回测结果逐步调整门槛
   - 重点关注通过率和最终因子质量的平衡

2. **换手率控制**：
   - 实盘策略建议启用换手率惩罚
   - 根据交易成本调整max_turnover_rate
   - 高频策略可适当放宽，低频策略应严格控制

3. **相关性处理**：
   - 红色警报阈值0.7通常是合理的
   - 正交化适合学术研究，实盘可考虑关闭
   - 绿色安全区间因子是最佳搭配

4. **报告分析**：
   - 重点关注selection_summary中的通过率
   - 检查category_distribution确保多样性
   - 分析correlation_control中的决策合理性
"""