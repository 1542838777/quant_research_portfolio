# Fama-MacBeth回归实现文档

## 概述

Fama-MacBeth回归是学术界和顶尖量化机构检验因子有效性的"黄金标准"。本文档介绍了在量化研究框架中实现的Fama-MacBeth回归功能。

## 理论背景

### 什么是Fama-MacBeth回归？

Fama-MacBeth回归是一种两步回归方法，用于检验因子的风险溢价是否显著：

1. **第一步（横截面回归）**：对每个时间点t，使用所有股票的截面数据进行回归
   ```
   R_i,t+N = α_t + β_t × Factor_i,t + ε_i,t+N
   ```
   其中：
   - R_i,t+N：股票i在t+N期的收益率
   - Factor_i,t：股票i在t期的因子值
   - β_t：t期的因子收益率（风险溢价）

2. **第二步（时间序列分析）**：对第一步得到的因子收益率序列{β_t}进行统计检验
   ```
   H0: E[β_t] = 0 (因子无风险溢价)
   H1: E[β_t] ≠ 0 (因子有风险溢价)
   ```

### 为什么是"黄金标准"？

1. **控制横截面相关性**：通过逐期回归避免了股票间的横截面相关性问题
2. **时间序列稳健性**：通过时间序列分析检验因子收益率的持续性
3. **学术认可度高**：被广泛应用于学术研究和实务中
4. **统计严谨性**：提供了严格的统计检验框架

## 实现特点

### 核心功能

我们的实现位于 `quant_lib/evaluation.py` 中的 `run_fama_macbeth_regression` 函数，具有以下特点：

1. **前视偏差修正**：使用T-1期的因子值预测T+N期的收益率
2. **数据质量控制**：自动处理缺失值和异常情况
3. **灵活的回归引擎**：支持statsmodels和numpy两种回归方法
4. **详细的统计输出**：提供完整的统计检验结果

### 函数签名

```python
def run_fama_macbeth_regression(
    factor_df: pd.DataFrame, 
    price_df: pd.DataFrame, 
    forward_returns_period: int = 20
) -> Dict[str, float]:
```

### 参数说明

- `factor_df`: 因子值矩阵，index为日期，columns为股票代码
- `price_df`: 复权收盘价矩阵，index为日期，columns为股票代码  
- `forward_returns_period`: 预测未来收益的时间窗口，默认20天

### 返回结果

返回字典包含以下键值：
- `mean_factor_return`: 因子平均收益率（风险溢价）
- `t_statistic`: t统计量
- `p_value`: p值
- `num_periods`: 有效回归期数
- `is_significant`: 是否显著（|t| > 2）
- `factor_returns_series`: 因子收益率时间序列

## 使用示例

### 基本用法

```python
from quant_lib.evaluation import run_fama_macbeth_regression
from quant_lib.data_loader import load_stock_data

# 1. 加载数据
price_data, pe_data = load_stock_data()

# 2. 处理因子（PE因子取负值，低PE更好）
pe_factor = -pe_data

# 3. 运行Fama-MacBeth回归
results = run_fama_macbeth_regression(
    factor_df=pe_factor,
    price_df=price_data,
    forward_returns_period=20
)

# 4. 查看结果
print(f"因子收益率: {results['mean_factor_return']:.6f}")
print(f"t统计量: {results['t_statistic']:.4f}")
print(f"是否显著: {results['is_significant']}")
```

### 完整的因子分析流程

```python
# 完整的因子分析包括：IC分析 + 分层回测 + Fama-MacBeth回归

from quant_lib.evaluation import (
    calculate_ic,
    calculate_quantile_returns,
    run_fama_macbeth_regression
)

# 1. IC分析
forward_returns = price_data.shift(-20) / price_data - 1
ic_series = calculate_ic(pe_factor, forward_returns)
print(f"IC均值: {ic_series.mean():.4f}")
print(f"IR: {ic_series.mean() / ic_series.std():.4f}")

# 2. 分层回测
quantile_results = calculate_quantile_returns(
    pe_factor, price_data, n_quantiles=5, forward_periods=[20]
)
tmb_return = quantile_results[20]['TopMinusBottom'].mean()
print(f"多空组合收益: {tmb_return:.4f}")

# 3. Fama-MacBeth回归
fm_results = run_fama_macbeth_regression(pe_factor, price_data, 20)
print(f"FM回归t值: {fm_results['t_statistic']:.4f}")

# 4. 综合评价
ic_good = abs(ic_series.mean()) > 0.02
monotonic = quantile_results[20].mean()['Q5'] > quantile_results[20].mean()['Q1']
significant = fm_results['is_significant']

if ic_good and monotonic and significant:
    print("✓ 因子通过所有检验，具有显著的预测能力")
else:
    print("✗ 因子未通过全部检验，需要进一步优化")
```

## 结果解读

### 统计显著性标准

- **|t| > 2.58**: 1%显著性水平 (***) 
- **|t| > 1.96**: 5%显著性水平 (**)
- **|t| > 1.64**: 10%显著性水平 (*)
- **|t| ≤ 1.64**: 不显著

### 实务判断标准

1. **学术标准**: |t| > 1.96 (5%显著性)
2. **实务标准**: |t| > 2.0 (更保守)
3. **严格标准**: |t| > 2.58 (1%显著性)

### 结果示例

```
Fama-MacBeth 回归分析结果
------------------------------------------------------------
回归期数: 245
因子平均收益率 (Mean Lambda): 0.002341
因子收益率标准差: 0.008765
因子收益率 t值 (t-statistic): 4.1823
因子收益率 p值 (p-value): 0.0000
显著性水平: ***
结论: ✓ t值绝对值大于2，因子收益率在统计上显著不为0，因子有效性得到验证！
```

## 技术实现细节

### 数据预处理

1. **前视偏差修正**: `factor_df.shift(1)` 确保使用历史因子值
2. **数据对齐**: 自动处理时间和股票维度的对齐
3. **缺失值处理**: 自动跳过缺失值过多的日期

### 回归实现

支持两种回归方法：

1. **statsmodels方法** (推荐)：
   ```python
   X_with_const = sm.add_constant(X)
   model = sm.OLS(y, X_with_const).fit()
   factor_return = model.params.iloc[1]
   ```

2. **numpy方法** (备选)：
   ```python
   X_matrix = np.column_stack([np.ones(len(X)), X.values])
   beta = np.linalg.solve(X_matrix.T @ X_matrix, X_matrix.T @ y_values)
   factor_return = beta[1]
   ```

### 质量控制

- 每日至少需要10只股票才进行回归
- 自动检测因子值是否有变化
- 异常回归自动跳过并记录日志

## 最佳实践

### 因子预处理建议

1. **去极值**: 使用1%-99%分位数截尾
2. **标准化**: 按日期进行Z-score标准化
3. **中性化**: 根据需要进行行业或市值中性化

### 参数选择建议

1. **预测周期**: 
   - 短期: 5-10天
   - 中期: 20天 (推荐)
   - 长期: 60天

2. **最小样本**: 每日至少30只股票，总期数至少100期

### 结果验证

建议结合多种方法验证因子有效性：
1. IC分析 (相关性)
2. 分层回测 (单调性)
3. Fama-MacBeth回归 (显著性)
4. 多因子模型 (增量贡献)

## 注意事项

1. **数据质量**: 确保价格和因子数据的质量和一致性
2. **生存偏差**: 注意处理退市股票的影响
3. **交易成本**: 实际应用中需要考虑交易成本
4. **样本外检验**: 避免过度拟合，进行样本外验证

## 扩展功能

未来可以考虑添加：
1. 多因子Fama-MacBeth回归
2. 滚动窗口回归
3. 异方差稳健标准误
4. Bootstrap置信区间
5. 因子收益率的时间序列分析
