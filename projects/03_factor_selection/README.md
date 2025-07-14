# 因子选择项目

本项目实现了一个完整的多因子筛选流程，包括单因子有效性检验、因子相关性分析和多因子合成。

## 项目结构

```
03_factor_selection/
├── config.yaml             # 配置文件
├── factor_evaluation.py    # 因子评价模块
├── single_factor_tester.py # 专业单因子测试框架
├── data_manager.py         # 数据管理器
├── factor_processor.py     # 因子预处理器
├── report_generator.py     # 报告生成器
├── run_factor_selection.py # 主脚本
├── results/                # 结果输出目录
└── README.md               # 项目说明
```

## 多因子筛选流程

整个因子筛选流程分为三个主要步骤：

### 第一步：单因子有效性检验 (The Factor Litmus Test)

对每一个备选因子进行独立的检验，判断它本身是否具备预测股票未来收益的能力。使用两种互补的方法：

1. **信息系数(IC)分析**：
   - 计算截面Rank IC：因子值排名与未来收益率排名的斯皮尔曼相关系数
   - 分析IC均值、IC标准差和信息比率(IR)
   - 有效因子标准：IC均值 > 2%，IR > 0.3

2. **分层回测**：
   - 按因子值将股票分为5组，构建等权投资组合
   - 观察各组净值曲线的单调性
   - 分析多空组合(做多第一组，做空第五组)的收益

### 第二步：因子相关性分析 (Building a Diversified Team)

确保最终入选的因子之间相关性较低，使每个因子都能从不同维度贡献Alpha。

1. **计算相关系数矩阵**：
   - 计算有效因子之间两两的相关系数
   - 使用热力图可视化相关性矩阵

2. **筛选低相关因子**：
   - 找出高度相关(相关系数绝对值 > 0.5)的因子对
   - 对于高相关因子对，保留IR值更高的因子

### 第三步：多因子合成与回测 (The Final Assembly)

将筛选出的低相关因子合成为一个综合因子，并进行回测评估。

1. **因子合成**：
   - 对因子进行标准化处理
   - 使用等权重或IR加权方法合成综合因子

2. **样本内外测试**：
   - 在样本内(In-Sample)和样本外(Out-of-Sample)分别进行回测
   - 对比样本内外的表现，评估策略的稳健性

3. **性能评估**：
   - 分析夏普比率、最大回撤、信息比率等指标
   - 对比单因子和多因子策略的表现

## 使用方法

1. 配置`config.yaml`文件，定义因子和评价参数
2. 运行主脚本：

```bash
python run_factor_selection.py --config config.yaml
```

3. 查看结果：程序会在`results/factor_selection/`目录下生成带时间戳的结果文件夹

## 结果解读

- `01_single_factor_test/`：单因子有效性检验结果
- `02_factor_correlation/`：因子相关性分析结果
- `03_factor_combination/`：多因子合成与回测结果

每个目录下都包含详细的CSV数据文件和可视化图表，帮助理解因子的表现和选择过程。

## 新增：专业单因子测试框架

### 🎯 SingleFactorTester - 华泰证券标准

本项目新增了专业级的单因子测试框架 `SingleFactorTester`，实现华泰证券标准的三种测试方法：

1. **IC值分析法** - 相关性检验
2. **分层回测法** - 实际投资效果验证
3. **Fama-MacBeth回归法** - 学术"黄金标准"

### 核心文件

- `single_factor_tester.py` - 专业测试框架
- `example_usage.py` - 使用示例和演示

### 快速使用

```python
from single_factor_tester import SingleFactorTester

# 创建测试器
tester = SingleFactorTester(
    price_data=price_data,
    test_periods=[5, 10, 20],
    output_dir="results/single_factor_tests"
)

# 综合测试单个因子
results = tester.comprehensive_test(
    factor_data=your_factor,
    factor_name='My_Factor',
    save_results=True
)

# 批量测试多个因子
batch_results = tester.batch_test(factors_dict)
```

### 自动生成结果

- **可视化图表** - 6合1分析图表
- **Excel报告** - 详细统计结果
- **JSON数据** - 完整测试数据
- **文字报告** - 专业分析报告

### 评价体系

**综合评分 (0-3分)**：
- IC_IR > 0.3 → +1分
- 分层单调性 → +1分
- FM回归显著 → +1分

**评价等级**：
- A级(3分): 优秀 - 通过所有检验
- B级(2分): 良好 - 通过部分检验
- C级(1分): 一般 - 需要优化
- D级(0分): 较差 - 建议放弃

### 运行示例

```bash
# 运行使用示例
python example_usage.py
```

## 专业级单因子测试案例研究

### 案例概述

在量化投资领域，我们通常会有大量的候选因子（"因子动物园"），但并非所有因子都适合纳入最终的多因子模型。本项目实现了一个科学、严谨的三步筛选流程，帮助我们构建一个高质量的多因子模型。

### 单因子测试终极作战手册 📚

#### 🎯 目标
对一个因子（以PE为例）进行完整的、专业级别的有效性检验，并生成一份可用于求职的分析报告。

#### 🏗️ 框架架构

**核心模块**
```
📦 单因子测试框架
├── 📄 config.yaml                  # 配置文件（作战计划）
├── 🔧 data_manager.py              # 数据管理器
├── ⚙️ factor_processor.py          # 因子预处理器
├── 📊 factor_evaluation.py         # 因子评价器
├── 📋 report_generator.py          # 报告生成器
├── 🚀 single_factor_tester.py      # 专业单因子测试框架
├── 🚀 run_factor_selection.py      # 主运行脚本
└── 📖 README.md                    # 使用指南
```

#### 🚀 快速开始

**1. 环境准备**
```bash
# 确保在项目根目录
cd quant_research_portfolio/projects/03_factor_selection

# 检查依赖
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl pyyaml
```

**2. 配置设置**
编辑 `config.yaml` 文件，设置你的测试参数

**3. 运行测试**
```bash
# 运行完整的多因子筛选流程
python run_factor_selection.py --config config.yaml

# 使用专业单因子测试框架
from single_factor_tester import SingleFactorTester
```

#### 📋 七个步骤详解

**第一阶段：准备工作 🛠️**
- 步骤一：配置你的"作战计划" (config.yaml)
- 步骤二：数据加载与股票池构建

**第二阶段：因子预处理 ⚙️**
- 步骤三：因子预处理流水线（去极值、中性化、标准化）

**第三阶段：三位一体检验 🔍**
- 步骤四：IC分析 - 评估因子预测能力
- 步骤五：Fama-MacBeth回归 - 学术"黄金标准"检验
- 步骤六：分层回测 - 实际投资效果验证

**第四阶段：报告生成 📋**
- 步骤七：专业报告生成（Jupyter Notebook、Excel报告、可视化图表）

#### 📊 评价体系

**综合评分 (0-3分)**
- **IC有效性** (1分): IC_IR > 0.3
- **FM显著性** (1分): |t统计量| > 1.96
- **分层单调性** (1分): 收益率呈单调性

**评价等级**
- **A级(3分)**: 优秀 - 通过所有检验
- **B级(2分)**: 良好 - 通过部分检验
- **C级(1分)**: 一般 - 需要优化
- **D级(0分)**: 较差 - 建议放弃

## 扩展方向

1. 添加更多类型的因子
2. 实现更复杂的因子合成方法(如PCA, 机器学习方法)
3. 增加风险控制和行业中性化处理
4. 添加交易成本和滑点模型
5. 集成更多专业级测试方法