# 量化研究框架

这是一个用于量化投资研究的工业级框架，提供了从数据获取、因子计算到回测评估的完整流程。

## 项目结构

```
quant_research_portfolio/
├── data/                   # 数据处理模块
├── projects/               # 研究项目
│   ├── 01_multi_factor_model/  # 多因子模型项目
│   └── 02_ml_stock_selection/  # 机器学习选股项目
├── quant_lib/              # 核心库
│   ├── backtesting.py      # 回测引擎
│   ├── config/             # 配置模块
│   ├── data_loader.py      # 数据加载器
│   ├── evaluation.py       # 评估模块
│   ├── factor_factory.py   # 因子工厂
│   ├── ml_utils.py         # 机器学习工具
│   ├── tushare/            # Tushare API封装
│   └── utils/              # 工具函数
└── setup.py                # 安装脚本
```

## 核心功能

### 数据管理

- 智能数据加载器，自动识别字段与数据源的映射
- 高效的数据对齐和预处理
- 支持本地Parquet文件和API数据源

### 因子计算

- 模块化的因子设计，支持自定义因子
- 内置多种常用因子：价值、动量、质量、成长、波动率等
- 因子处理：去极值、标准化、中性化

### 回测引擎

- 灵活的回测框架，支持多种回测策略
- 详细的回测结果和性能指标
- 可视化分析工具

### 评估系统

- 因子评价：IC分析、分层回测
- 策略评估：收益率、风险指标、换手率
- 结果可视化

### 机器学习支持

- 特征工程工具
- 模型训练和评估
- 超参数优化

## 安装

```bash
# 克隆仓库
git clone https://github.com/username/quant_research_portfolio.git
cd quant_research_portfolio

# 安装依赖
pip install -e .
```

## 使用示例

### 运行多因子模型回测

```bash
cd projects/01_multi_factor_model
python run_backtest.py --config_manager config_manager.yaml
```

### 自定义因子

```python
from quant_lib.factor_factory import BaseFactor

 

### 加载数据

```python
from quant_lib.data_loader import DataLoader

# 创建数据加载器
loader = DataLoader()

# 加载数据
data_dict = loader.get_raw_dfs_by_require_fields(fields=['close', 'pe_ttm', 'pb'], start_date='2020-01-01',
                                                 end_date='2023-12-31')
```

## 配置

配置文件使用YAML格式，可以方便地定义回测参数、因子设置等。示例：

```yaml
# 回测时间范围
start_date: '2020-01-01'
end_date: '2023-12-31'

# 因子设置
factors:
  - name: 'value'
    weight: 0.3
  - name: 'momentum'
    weight: 0.3
```

## 扩展

框架设计遵循开闭原则，易于扩展：

1. 添加新因子：继承`BaseFactor`类
2. 添加新数据源：扩展`DataLoader`类
3. 自定义回测策略：基于`BacktestEngine`类

## 依赖

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn
- tushare
- pyarrow
- pyyaml

## 许可证

MIT 