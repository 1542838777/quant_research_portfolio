切换成前复权
--step1：对比tushare daily_qfq 数据跟聚宽 数据  ，---满足一致
--step2: if (col in ['close', 'open', 'high', 'low']) & (  # 实测 amount 和vol 在daily和 在daily_hfq数值一模一样！
                            logical_name == 'daily_hfq'):  ------>替换成daily_qfq, 
            --所以 我们需要确保 open 
                        --确保 high
                        --确保low 都要跟聚宽的qfq一致！

---step3：全部替换变量名(替换前必须注意当前没qfq字段) _hfq ->qfq   
 系统性名称替换

  需要全局替换：
  - close_hfq → close_qfq                                                                                                                                                              
  - open_hfq → open_qfq                                                                                                                                                                
  - high_hfq → high_qfq                                                                                                                                                                
  - low_hfq → low_qfq                                                                                                                                                                  
  - vol_hfq → vol_qfq                                                                                                                                                                  
  - hfq_adj_factor → qfq_adj_factor     
---step4 中文名 替换 后复权-》前复权

---其他计算逻辑不用动 （逻辑不因复权方式的改变而改变。