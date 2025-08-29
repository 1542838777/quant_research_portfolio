使用方式

  🎯 快速对比（推荐）

  from quant_backtester import quick_factor_backtest
  from data_loader_adapter import BacktestDataLoader, PredefinedDatasets

  # 1. 加载数据                                                                                                                                                                        
  data_loader = BacktestDataLoader()
  price_df, factor_dict = PredefinedDatasets.get_champion_vs_composite(data_loader)

  # 2. 一行代码完成回测                                                                                                                                                                
  portfolios, comparison_table = quick_factor_backtest(price_df, factor_dict)

  # 3. 查看结果                                                                                                                                                                        
  print(comparison_table)

  🔬 完整分析

  # 创建回测器                                                                                                                                                                         
  backtester = QuantBacktester(config)

  # 运行回测                                                                                                                                                                           
  portfolios = backtester.run_backtest(price_df, factor_dict)

  # 生成报告                                                                                                                                                                           
  comparison = backtester.get_comparison_table()
  backtester.plot_cumulative_returns()
  backtester.plot_drawdown_analysis()
  report_path = backtester.generate_full_report()

  关键特性

  ✅ 实盘级精度

  - 真实交易成本：万3佣金 + 千1滑点 + 千1印花税
  - 数据严格对齐：避免前视偏差
  - 持仓数量控制：防止过度分散

  📊 专业分析

  - 风险调整收益：夏普比率、卡玛比率
  - 回撤分析：最大回撤、平均回撤持续期
  - 交易统计：胜率、盈亏比、换手率

  🎨 可视化

  - 净值曲线对比
  - 回撤时序图
  - 完整HTML报告

  示例文件

  - backtest_factor_comparison_example.py - 4个完整示例
  - data_loader_adapter.py - 数据加载适配器

  现在你可以进行"苹果vs苹果"的严格对比了！系统确保两个因子在完全相同的规则、成本、时间下回测，结果具有完全的可比性。
