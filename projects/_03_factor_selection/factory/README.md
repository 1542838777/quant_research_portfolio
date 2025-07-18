# 策略工厂使用指南

## 🎯 概述

策略工厂是一个完整的量化研究解决方案，提供从单因子测试到多因子策略构建的全流程支持。它整合了因子管理、测试评估、优化组合、可视化等功能，为团队协作和成果复用提供标准化接口。

## 🏗️ 架构设计

```
策略工厂 (StrategyFactory)
├── 因子管理器 (FactorManager)
│   ├── 因子注册表 (FactorRegistry)
│   ├── 因子分类器 (FactorClassifier)
│   └── 测试结果存储
├── 单因子测试器 (SingleFactorTester)
│   ├── IC分析
│   ├── 分层回测
│   └── Fama-MacBeth回归
├── 多因子优化器 (MultiFactorOptimizer)
│   ├── 类别内优化 (IntraCategoryOptimizer)
│   └── 类别间优化 (CrossCategoryOptimizer)
└── 可视化管理器 (VisualizationManager)
    ├── 单因子图表
    ├── 对比分析图
    └── 交互式仪表板
```

## 🚀 快速开始

### 1. 基础使用

```python
from strategy_factory import StrategyFactory
from factor_manager import FactorCategory

# 初始化策略工厂
factory = StrategyFactory(
   config_path="config.yaml",
   workspace_dir="my_research"
)

# 加载数据
data_dict = factory.get_raw_dfs_by_require_fields(,,

   # 测试单个因子
factor_data = data_dict['pb'].apply(lambda x: 1 / x)  # PB倒数
result = factory.test_single_factor(
   factor_data=factor_data,
   factor_name="PB_factor",
   category=FactorCategory.VALUE
)

print(f"因子评分: {result['evaluation']['total_score']:.2f}")
```

### 2. 批量测试

```python
# 准备多个因子
factors = {
    'PB_factor': 1 / data_dict['pb'],
    'PE_factor': 1 / data_dict['pe'],
    'momentum_20d': data_dict['close_price'].pct_change(20),
    'ROE_factor': data_dict['roe']
}

# 定义类别映射
category_mapping = {
    'PB_factor': FactorCategory.VALUE,
    'PE_factor': FactorCategory.VALUE,
    'momentum_20d': FactorCategory.MOMENTUM,
    'ROE_factor': FactorCategory.QUALITY
}

# 批量测试
results = factory.batch_test_factors(
    factor_data_dict=factors,
    category_mapping=category_mapping
)

# 查看性能汇总
summary = factory.get_factor_performance_summary()
print(summary.head())
```

### 3. 多因子优化

```python
from multi_factor_optimizer import MultiFactorOptimizer

# 按类别分组因子
factors_by_category = {
    'value': {'PB_factor': pb_factor, 'PE_factor': pe_factor},
    'momentum': {'momentum_20d': mom_factor},
    'quality': {'ROE_factor': roe_factor}
}

# 获取因子评分
factor_scores = {
    'value': {'PB_factor': 0.05, 'PE_factor': 0.03},
    'momentum': {'momentum_20d': 0.08},
    'quality': {'ROE_factor': 0.06}
}

# 执行优化
optimizer = MultiFactorOptimizer()
optimized_factor = optimizer.optimize_factors(
    factors_by_category=factors_by_category,
    factor_scores=factor_scores,
    intra_method='ic_weighted',
    cross_method='max_diversification'
)

# 测试优化后的因子
final_result = factory.test_single_factor(
    factor_data=optimized_factor,
    factor_name="optimized_multi_factor"
)
```

## 📊 可视化功能

### 1. 单因子图表

```python
from visualization_manager import VisualizationManager

viz_manager = VisualizationManager(output_dir="charts")

# 生成单因子测试图表
plot_paths = viz_manager.plot_single_factor_results(
    test_results=result,
    factor_name="PB_factor",
    save_plots=True
)
```

### 2. 因子对比

```python
# 生成多因子对比图
comparison_path = viz_manager.plot_factor_comparison(
    factor_results=results,
    metrics=['ic_mean', 'ic_ir', 'overall_score']
)
```

### 3. 交互式仪表板

```python
# 创建交互式仪表板
dashboard_path = viz_manager.create_interactive_dashboard(
    factor_results=results,
    save_html=True
)
```

## 🔧 配置文件

创建 `config.yaml` 配置文件：

```yaml
data:
  start_date: '2020-01-01'
  end_date: '2024-12-31'
  universe: 'hs300'
  benchmark: '000300.SH'

factor_test:
  forward_periods: [1, 5, 20]
  quantiles: 5
  preprocessing:
    winsorization:
      enable: true
      method: 'mad'
      mad_threshold: 5
    neutralization:
      enable: true
      factors: ['market_cap', 'industry']
    standardization:
      enable: true
      method: 'zscore'

optimization:
  method: 'equal_weight'
  max_factors_per_category: 3
  correlation_threshold: 0.7

output:
  save_plots: true
  generate_report: true
  export_excel: true
```

## 📈 因子评价体系

### 评分标准

**综合评分 (0-3分)**
- **IC有效性** (1分): IC_IR > 0.3
- **FM显著性** (1分): |t统计量| > 1.96  
- **分层单调性** (1分): 收益率呈单调性

### 评价等级

- **A级(3分)**: 优秀 - 通过所有检验
- **B级(2分)**: 良好 - 通过部分检验  
- **C级(1分)**: 一般 - 需要优化
- **D级(0分)**: 较差 - 建议放弃

## 🔄 完整工作流程

### 第一阶段：单因子测试

```python
# 1. 注册因子
factory.register_factor(
    name="my_factor",
    category=FactorCategory.VALUE,
    description="自定义价值因子",
    data_requirements=["pb", "pe"]
)

# 2. 测试因子
result = factory.test_single_factor(
    factor_data=factor_data,
    factor_name="my_factor"
)

# 3. 查看结果
evaluation = result['evaluation']
print(f"评级: {evaluation['grade']}")
print(f"评分: {evaluation['total_score']}")
```

### 第二阶段：类别内优化

```python
from multi_factor_optimizer import IntraCategoryOptimizer

intra_optimizer = IntraCategoryOptimizer()

# 相关性去重
selected_factors = intra_optimizer.remove_correlated_factors(
    factor_data_dict=value_factors,
    factor_scores=value_scores
)

# IC加权组合
combined_factor = intra_optimizer.ic_weighted_combination(
    factor_data_dict=selected_factor_data,
    ic_scores=ic_scores
)
```

### 第三阶段：类别间优化

```python
from multi_factor_optimizer import CrossCategoryOptimizer

cross_optimizer = CrossCategoryOptimizer()

# 最大分散化权重
category_weights = cross_optimizer.max_diversification_weights(
    category_factors=category_factors,
    category_scores=category_scores
)

# 风险平价权重
risk_parity_weights = cross_optimizer.risk_parity_weights(
    category_factors=category_factors
)
```

### 第四阶段：结果导出

```python
# 导出所有结果
exported_files = factory.export_results()

# 生成综合报告
report_paths = viz_manager.generate_comprehensive_report(
    factor_results=all_results,
    category_summary=category_summary
)
```

## 🛠️ 高级功能

### 1. 因子流水线

```python
# 定义因子配置
factor_configs = [
    {
        'name': 'momentum_5d',
        'data': momentum_5d_data,
        'test_params': {'preprocess_method': 'standard'}
    },
    {
        'name': 'momentum_20d',
        'data': momentum_20d_data,
        'test_params': {'preprocess_method': 'robust'}
    }
]

# 创建并运行流水线
pipeline = factory.create_factor_pipeline(factor_configs)
results = pipeline.run()
```

### 2. 自定义因子类别

```python
from factor_manager import FactorCategory

# 扩展因子类别
class CustomFactorCategory(FactorCategory):
    SENTIMENT = "sentiment"
    ALTERNATIVE = "alternative"

# 注册自定义类别因子
factory.register_factor(
    name="sentiment_factor",
    category=CustomFactorCategory.SENTIMENT,
    description="情绪因子"
)
```

### 3. 批量结果分析

```python
# 获取各类别表现最好的因子
for category in FactorCategory:
    top_factors = factory.get_top_factors(
        category=category,
        top_n=3,
        min_score=2.0
    )
    print(f"{category.value}: {top_factors}")

# 生成性能报告
report = factory.factor_manager.generate_performance_report()
print(f"总因子数: {report['summary']['total_factors']}")
print(f"A级因子数: {report['summary']['grade_distribution']['A']}")
```

## 📝 最佳实践

### 1. 因子命名规范

- 使用描述性名称：`momentum_20d` 而不是 `factor1`
- 包含时间窗口：`volatility_60d`
- 标明因子类型：`value_pb_factor`

### 2. 数据质量检查

```python
# 检查数据完整性
def check_data_quality(factor_data):
    missing_ratio = factor_data.isnull().sum().sum() / factor_data.size
    if missing_ratio > 0.3:
        print(f"警告: 缺失数据比例过高 ({missing_ratio:.2%})")
    
    return missing_ratio < 0.5

# 使用前检查
if check_data_quality(factor_data):
    result = factory.test_single_factor(factor_data, "my_factor")
```

### 3. 结果解读

```python
def interpret_results(result):
    evaluation = result['evaluation']
    
    print(f"因子评级: {evaluation['grade']}")
    print(f"综合评分: {evaluation['total_score']}/3")
    
    if evaluation['grade'] == 'A':
        print("✓ 优秀因子，建议使用")
    elif evaluation['grade'] == 'B':
        print("○ 良好因子，可考虑使用")
    else:
        print("✗ 因子表现不佳，建议优化或放弃")
```

## 🤝 团队协作

### 1. 因子共享

```python
# 导出因子注册表
factory.factor_manager.registry.save_registry()

# 在其他项目中加载
new_factory = StrategyFactory()
shared_factors = new_factory.factor_manager.registry.factors
```

### 2. 结果复现

```python
# 保存测试配置
test_config = {
    'preprocess_method': 'standard',
    'forward_periods': [5, 20],
    'quantiles': 5
}

# 使用相同配置测试
result = factory.test_single_factor(
    factor_data=factor_data,
    factor_name="reproducible_factor",
    **test_config
)
```

## 🔍 故障排除

### 常见问题

1. **数据加载失败**
   - 检查数据路径和格式
   - 确认数据字段名称匹配

2. **因子测试报错**
   - 检查因子数据的时间对齐
   - 确认数据中没有全为NaN的行/列

3. **可视化失败**
   - 检查是否安装了plotly
   - 确认输出目录有写入权限

### 调试技巧

```python
# 开启详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查中间结果
print(f"因子数据形状: {factor_data.shape}")
print(f"缺失值比例: {factor_data.isnull().mean().mean():.2%}")
```

---

**更多示例和详细文档请参考：**
- `enhanced_run_factor_selection.py` - 完整使用示例
- `strategy_factory.py` - 核心API文档
- `factor_manager.py` - 因子管理功能
- `multi_factor_optimizer.py` - 优化算法详解
