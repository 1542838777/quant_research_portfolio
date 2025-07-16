# 因子选择模块 - 重构版

本目录包含了重构后的因子选择模块，提供了一套完整的量化研究工具链，从单因子测试到多因子策略构建。

## 目录结构

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

## 主要组件

### 策略工厂 (StrategyFactory)

策略工厂是整个模块的核心，整合了所有子组件，提供了统一的接口。它负责管理数据、因子、测试、优化等组件，并提供标准化的研究流程。

主要功能：
- 数据加载和管理
- 因子注册和分类
- 单因子测试和批量测试
- 多因子优化
- 结果导出和可视化

### 因子管理器 (FactorManager)

因子管理器负责管理因子的注册、分类和测试结果存储。它包含以下子组件：

- **因子注册表 (FactorRegistry)**: 管理因子的元数据和注册信息
- **因子分类器 (FactorClassifier)**: 自动分类因子并进行聚类分析
- **测试结果存储**: 保存和检索因子测试结果

### 单因子测试器 (SingleFactorTester)

单因子测试器负责执行因子的有效性测试，包括：

- **IC分析**: 计算信息系数(IC)和相关指标
- **分层回测**: 执行分层回测，计算收益率和风险指标
- **Fama-MacBeth回归**: 执行Fama-MacBeth回归检验

### 多因子优化器 (MultiFactorOptimizer)

多因子优化器负责组合多个因子，包括：

- **类别内优化 (IntraCategoryOptimizer)**: 在同一类别内选择和组合因子
- **类别间优化 (CrossCategoryOptimizer)**: 在不同类别间进行因子配置

### 可视化管理器 (VisualizationManager)

可视化管理器负责生成各种图表和报告，包括：

- **单因子图表**: 生成单个因子的分析图表
- **对比分析图**: 生成多个因子的对比图表
- **交互式仪表板**: 生成交互式的分析仪表板

## 使用示例

```python
# 初始化策略工厂
factory = StrategyFactory(
    config_path="factory/config.yaml",
    workspace_dir="workspace"
)

# 加载数据
data_dict = factory.load_data()

# 测试单个因子
result = factory.test_single_factor(
    factor_data=factor_data,
    factor_name="PE_factor",
    category=FactorCategory.VALUE
)

# 批量测试因子
batch_results = factory.batch_test_factors(
    factor_data_dict=factor_data_dict,
    auto_register=True,
    category_mapping=factor_category_mapping
)

# 获取因子性能汇总
performance_summary = factory.get_factor_performance_summary()

# 多因子优化
optimized_factor = factory.optimize_factors(
    factor_data_dict=factor_data_dict,
    intra_method='ic_weighted',
    cross_method='max_diversification'
)

# 导出结果
exported_files = factory.export_results()
```

更详细的使用示例请参考 `example_usage.py` 文件。

## 重构改进

相比原始版本，重构后的代码有以下改进：

1. **模块化设计**: 将功能拆分为多个独立模块，每个模块负责特定功能
2. **层次化结构**: 清晰的层次结构，便于理解和维护
3. **接口统一**: 统一的接口设计，便于使用和扩展
4. **功能增强**: 
   - 增加了因子自动分类功能
   - 增加了因子聚类和相关性分析
   - 增加了多种优化方法
5. **文档完善**: 每个模块都有详细的文档和注释

## 依赖项

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- statsmodels (可选，用于高级回归分析)
- vectorbt (可选，用于高效的向量化计算)

## 后续计划

1. 增加更多的因子构建方法
2. 增加机器学习模型集成
3. 增加因子归因分析
4. 增加更多的可视化组件
5. 增加回测系统集成