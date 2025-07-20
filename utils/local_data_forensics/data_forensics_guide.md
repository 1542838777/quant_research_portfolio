# 数据侦探工具使用指南

## 🕵️ 工具简介

数据侦探工具（DataForensics）是一个专业的数据质量诊断工具，用于分析量化数据中的NaN值，并智能判断这些缺失值是"合理的"还是"可疑的"。

### 核心功能

1. **智能NaN归因分析** - 区分上市前、退市后、停牌期间和异常缺失
2. **向量化高效计算** - 使用pandas向量化操作，处理大规模数据
3. **批量诊断** - 支持同时诊断多个字段和数据集
4. **数据质量报告** - 生成详细的数据质量评估报告
5. **交互式模式** - 提供友好的命令行交互界面

## 🚀 快速开始

### 基本使用

```python
from quant_lib.data_forensics import DataForensics

# 1. 初始化数据侦探
forensics = DataForensics()

# 2. 诊断单个字段
forensics.diagnose_field_nan(
    field_name='close',           # 字段名
    dataset_name='daily_hfq',     # 数据集名
    sample_stocks=8,              # 抽样分析的股票数量
    detailed_analysis=True        # 是否进行详细分析
)
```

### 命令行使用

```bash
# 交互式模式（推荐）
python scripts/run_data_forensics.py --mode interactive

# 单字段诊断
python scripts/run_data_forensics.py --mode single --field close --dataset daily_hfq --samples 8

# 批量诊断
python scripts/run_data_forensics.py --mode batch

# 生成数据质量报告
python scripts/run_data_forensics.py --mode report
```

## 📊 诊断逻辑

数据侦探通过三个关键问题来判断NaN值的合理性：

### 1. "当时你出生了吗？" 
- 检查NaN是否发生在股票上市日期之前
- **结论**: 上市前的NaN是合理的 ✅

### 2. "当时你已经退隐江湖了吗？"
- 检查NaN是否发生在股票退市日期之后  
- **结论**: 退市后的NaN是合理的 ✅

### 3. "你当天是否请假了？"
- 检查NaN是否发生在正常交易期间
- **结论**: 交易期间的NaN大概率是停牌 ℹ️

### 4. 可疑缺失
- 排除以上所有情况后剩余的NaN
- **结论**: 可能存在数据质量问题 ⚠️

## 🎯 使用场景

### 场景1: 新数据源验证
```python
# 验证新下载的数据是否完整
forensics.diagnose_field_nan('close', 'daily_hfq')
```

### 场景2: 定期数据质量检查
```python
# 批量检查核心字段
core_fields = [
    ('close', 'daily_hfq'),
    ('pe_ttm', 'daily_basic'),
    ('vol', 'daily_hfq')
]
forensics.batch_diagnose(core_fields)
```

### 场景3: 生成质量报告
```python
# 生成详细的数据质量报告
report = forensics.generate_data_quality_report('quality_report.json')
print(f"总体质量分数: {report['overall_quality_score']:.2%}")
```

## 📈 输出解读

### 归因分析结果
```
📋 NaN归因分析结果:
  ✅ 上市前缺失: 1,234,567 (45.2%) - 合理
  ✅ 退市后缺失: 234,567 (8.6%) - 合理  
  ℹ️  交易期间缺失: 1,123,456 (41.2%) - 大概率停牌
  ❓ 未知股票缺失: 156,789 (5.7%) - 需要检查
```

### 质量评级
- **✅ 优秀** (95%+): 数据质量很好，NaN都有合理解释
- **⚠️ 良好** (80-95%): 数据基本可用，少量异常
- **❌ 需要关注** (<80%): 存在较多数据质量问题

## 🔧 高级功能

### 自定义诊断
```python
# 自定义抽样数量和分析深度
forensics.diagnose_field_nan(
    field_name='turnover_rate',
    dataset_name='daily_basic',
    sample_stocks=15,           # 抽样15只股票
    detailed_analysis=True      # 显示详细的个股分析
)
```

### 交互式诊断
```bash
python scripts/run_data_forensics.py --mode interactive

# 在交互模式中使用命令:
🔍 请输入命令: diagnose close daily_hfq 10
🔍 请输入命令: batch
🔍 请输入命令: report
🔍 请输入命令: help
```

## 📝 支持的数据集

工具自动识别以下数据结构：

### 分区数据集
- `daily_hfq/` - 后复权日线数据
- `daily_basic/` - 每日基本面数据
- `daily/` - 原始日线数据
- 其他按年份分区的数据集

### 单文件数据集  
- `stock_basic.parquet` - 股票基本信息
- `trade_cal.parquet` - 交易日历
- `namechange.parquet` - 股票名称变更

## ⚡ 性能优化

### 向量化计算
- 使用pandas向量化操作替代循环
- 批量处理股票，避免逐个遍历
- 智能缓存，减少重复数据加载

### 内存管理
- 按需加载数据，避免全量加载
- 及时释放不需要的DataFrame
- 支持大规模数据集处理

## 🛠️ 故障排除

### 常见问题

**Q: 提示"无法加载 stock_basic.parquet"**
```
A: 检查数据路径配置，确保stock_basic.parquet文件存在
```

**Q: 字段不存在错误**
```
A: 使用 df.columns 检查数据集中的实际字段名
```

**Q: 内存不足**
```
A: 减少sample_stocks参数，或分批处理数据
```

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

forensics = DataForensics()
```

## 📊 扩展开发

### 添加新的归因逻辑
```python
class CustomDataForensics(DataForensics):
    def _custom_attribution_analysis(self, wide_df, nan_mask):
        # 添加自定义的NaN归因逻辑
        pass
```

### 集成到数据管道
```python
# 在数据更新后自动运行质量检查
def post_data_update_check():
    forensics = DataForensics()
    report = forensics.generate_data_quality_report()
    
    if report['overall_quality_score'] < 0.9:
        send_alert("数据质量告警", report)
```

## 🎯 最佳实践

1. **定期检查**: 建议每次数据更新后运行诊断
2. **重点关注**: 优先检查核心字段（价格、成交量等）
3. **阈值设置**: 根据业务需求设置质量分数阈值
4. **结果存档**: 保存历史质量报告，跟踪数据质量趋势
5. **自动化**: 集成到数据管道中，实现自动化质量监控

---

💡 **提示**: 数据侦探工具不仅能发现问题，更重要的是帮助你理解数据的特性和局限性，为后续的量化分析提供可靠的数据基础。
