# 专业滚动IC因子筛选+IC加权合成系统

## 系统概述

这是一个专业级的量化因子筛选和合成系统，专为实盘盈利策略设计。系统基于滚动IC（Information Coefficient）完全避免前视偏差，通过多周期评估和指数衰减权重实现智能因子筛选和最优权重分配。

## 核心特色

- **滚动IC计算**: 完全避免前视偏差，确保实盘可用性
- **多周期IC评分**: 指数衰减权重 `weights = np.array([decay_rate ** i for i in range(len(period_scores))])`
- **专业因子质量评估**: 多维度评估体系（IC均值、IR、稳定性、波动率）
- **类别内冠军选择**: 11个因子类别（Value、Quality、Momentum等）智能选择
- **智能权重分配**: IC加权算法自动优化合成权重
- **端到端流程**: 从海选到最终合成的完整自动化

## 系统架构

### 核心类

1. **`RollingICFactorSelector`**: 专业滚动IC筛选器
   - 位置: `factor_manager/selector/rolling_ic_factor_selector.py`
   - 功能: 执行完整的因子筛选流程

2. **`ICWeightedSynthesizer`**: IC加权因子合成器
   - 位置: `factor_manager/factor_composite/ic_weighted_synthesizer.py`
   - 功能: 集成筛选器，执行加权合成

3. **`RollingICManager`**: 滚动IC管理器
   - 位置: `factor_manager/storage/rolling_ic_manager.py`
   - 功能: 无前视偏差的IC计算和存储

### 配置类

- **`RollingICSelectionConfig`**: 筛选配置参数
- **`FactorWeightingConfig`**: 权重分配配置

## 快速开始

### 1. 基本筛选

```python
from projects._03_factor_selection.factor_manager.selector.rolling_ic_factor_selector import (
    RollingICFactorSelector, RollingICSelectionConfig
)

# 配置筛选参数
config = RollingICSelectionConfig(
    min_ic_abs_mean=0.015,        # IC均值绝对值门槛
    min_ir_abs_mean=0.20,         # IR均值绝对值门槛
    min_ic_stability=0.45,        # IC稳定性门槛
    decay_rate=0.70,              # 衰减率（越小越重视长期）
    max_final_factors=8,          # 最终选择因子数
    # 三层相关性控制哲学
    high_corr_threshold=0.7,      # 红色警报：坚决二选一
    medium_corr_threshold=0.3,    # 黄色预警：正交化战场  
    enable_orthogonalization=True # 启用中相关区间正交化
)

# 创建筛选器
selector = RollingICFactorSelector("配置快照ID", config)

# 执行筛选
candidate_factors = ["volatility_120d", "momentum_20d", "ep_ratio", "reversal_5d"]
selected_factors, report = selector.run_complete_selection(candidate_factors)

print(f"筛选结果: {selected_factors}")
```

### 2. 集成筛选+合成

```python
from projects._03_factor_selection.factor_manager.factor_composite.ic_weighted_synthesize_with_orthogonalization import (
    ICWeightedSynthesizer, FactorWeightingConfig
)

# 配置合成参数
synthesizer_config = FactorWeightingConfig(
    min_ic_mean=0.010,
    min_ic_ir=0.15,
    max_single_weight=0.4
)

# 创建合成器（需要factor_manager等依赖）
synthesizer = ICWeightedSynthesizer(
    factor_manager, factor_analyzer, factor_processor,
    config=synthesizer_config,
    selector_config=config
)

# 执行专业筛选+IC加权合成
composite_factor, synthesis_report = synthesizer.synthesize_with_professional_selection(
    composite_factor_name="smart_composite",
    candidate_factor_names=candidate_factors,
    snap_config_id="配置快照ID"
)

# 打印报告
synthesizer.print_synthesis_report(synthesis_report)
```

## 筛选流程

### 第一步: 基础质量筛选
- IC均值绝对值 >= min_ic_abs_mean
- IR均值绝对值 >= min_ir_abs_mean  
- IC稳定性 >= min_ic_stability
- IC波动率 <= max_ic_volatility
- 快照数量 >= min_snapshots

### 第二步: 类别内冠军选择
支持的因子类别:
- **Value**: 价值因子（EP、BP、SP等）
- **Quality**: 质量因子（ROE、毛利率等）
- **Momentum**: 动量因子（价格动量、收益动量）
- **Reversal**: 反转因子（短期反转）
- **Size**: 规模因子（市值相关）
- **Volatility**: 波动率因子
- **Liquidity**: 流动性因子
- **Technical**: 技术指标因子
- **Growth**: 成长因子
- **Profitability**: 盈利能力因子
- **Efficiency**: 运营效率因子

### 第三步: 初步最终选择
- 按多周期综合评分排序
- 选择前N名（可配置）

### 第四步: 三层相关性控制哲学 ⭐️ 
**核心理念**: 承认相关性的不同层次，采用差异化策略

**🚨 红色警报区域** (`|corr| > 0.7`)
- **决策**: 坚决执行"二选一"
- **理由**: 高度冗余因子，强行挖掘残差过拟合风险大于收益
- **策略**: 选择多周期评分最高的因子

**⚠️ 黄色预警区域** (`0.3 < |corr| < 0.7`) 
- **决策**: 正交化战场，这是价值挖掘的黄金区间
- **理由**: 既有显著共同信息，又包含不可忽视的独立信息
- **策略**: 以评分高者为基准，对其他因子进行正交化处理
- **举例**: ROE与Momentum相关性0.4时，用ROE清洗Momentum得到纯粹动量信号

**✅ 绿色安全区域** (`|corr| < 0.3`)
- **决策**: 直接全部保留  
- **理由**: 天然的"好队友"，提供足够多样性
- **策略**: 无需额外处理

### 第五步: 生成详细筛选报告
- 包含相关性控制决策记录
- 正交化因子详细信息

## 多周期评分算法

系统使用指数衰减权重对不同周期的IC表现进行综合评分：

```python
# 衰减权重计算
weights = np.array([decay_rate ** i for i in range(len(period_scores))])
weights /= weights.sum()  # 权重归一化

# 加权平均分数
final_score = np.average(period_scores, weights=weights)
```

**参数说明:**
- `decay_rate`: 衰减率，越小权重衰减越慢
- `period_scores`: 各周期评分数组（从短期到长期）
- 权重分配：短期权重更高，但所有周期都有贡献

## 配置参数说明

### 筛选配置 (RollingICSelectionConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_snapshots` | 3 | 最少快照数量 |
| `min_ic_abs_mean` | 0.01 | IC均值绝对值门槛 |
| `min_ir_abs_mean` | 0.15 | IR均值绝对值门槛 |
| `min_ic_stability` | 0.4 | IC稳定性门槛 |
| `max_ic_volatility` | 0.05 | IC波动率上限 |
| `decay_rate` | 0.75 | 衰减率 |
| `max_factors_per_category` | 2 | 每类最多因子数 |
| `max_final_factors` | 8 | 最多最终因子数 |
| `high_corr_threshold` | 0.7 | 高相关阈值（红色警报） |  
| `medium_corr_threshold` | 0.3 | 中低相关分界（黄色预警） |
| `enable_orthogonalization` | True | 是否启用正交化处理 |

### 权重配置 (FactorWeightingConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_ic_mean` | 0.015 | 最小IC均值阈值 |
| `min_ic_ir` | 0.183 | 最小IR阈值 |
| `min_ic_win_rate` | 0.52 | 最小胜率阈值 |
| `max_single_weight` | 0.5 | 单因子最大权重 |
| `min_single_weight` | 0.05 | 单因子最小权重 |

## 测试验证

运行系统架构测试：

```bash
cd projects/_03_factor_selection
python test_rolling_ic_selector.py
```

测试项目包括：
- ✅ 筛选器初始化
- ✅ 质量筛选架构
- ✅ 类别选择架构  
- ✅ 完整流程架构

## 输出报告

系统生成详细的筛选和合成报告，包含：

1. **筛选统计**: 候选数量、合格数量、通过率
2. **评分分布**: 平均评分、最高评分、最低评分
3. **类别分布**: 各类别因子数量分布
4. **最终权重**: 每个因子的IC权重分配
5. **时间信息**: 筛选时间范围和数据版本

## 注意事项

1. **数据依赖**: 需要预先计算的因子数据和滚动IC数据
2. **时间配置**: 确保配置的时间范围有对应的历史数据
3. **参数调优**: 根据实际数据特征调整筛选阈值
4. **版本控制**: 使用配置快照ID确保结果可复现

## 技术特点

- **无前视偏差**: 严格的滚动计算确保实盘可用性
- **三层相关性控制**: 创新的相关性处理哲学，兼顾效率与信息保留
- **智能权重**: 基于历史IC表现的自动权重优化  
- **正交化处理**: 中相关区间的价值挖掘，提取纯粹信号
- **模块化设计**: 筛选和合成功能独立，便于扩展
- **丰富报告**: 详细的筛选过程和相关性决策记录
- **参数可调**: 灵活的配置系统适应不同需求

---

**Author**: Claude  
**Date**: 2025-08-25  
**System**: Professional Rolling IC Factor Selection & IC-Weighted Synthesis