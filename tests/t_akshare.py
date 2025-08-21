import akshare as ak

# --- 示例1: 获取沪深300当前最新的成分股 ---
try:
    print("--- Akshare 获取沪深300当前最新成分股 (来自新浪) ---")
    index_stock_cons_sina_df = ak.index_stock_cons_sina(symbol="000300")
    print(index_stock_cons_sina_df.head())
except Exception as e:
    print(f"从新浪获取失败: {e}")

# --- 示例2: 获取中证1000当前最新的成分股 (来自中证指数官网) ---
# 注意：中证官网的接口可能因反爬策略而不稳定
try:
    print("\n--- Akshare 获取中-证1000当前最新成分股 (来自中证官网) ---")
    index_stock_cons_csindex_df = ak.index_stock_cons(symbol="000852")
    print(index_stock_cons_csindex_df.head())
except Exception as e:
    print(f"从中证官网获取失败: {e}")

# --- 示例3: 获取沪深300历史上的成分股调整信息 ---
try:
    print("\n--- Akshare 获取沪深300历史成分股调整记录 ---")
    stock_hist_df = ak.stock_zh_index_value_csindex(symbol="000300")
    asda = ak.stock_analyst_detail_em(analyst_id="11000200926", indicator="历史跟踪成分股")
    asda = ak.index_csi_cons_hist_csi(symbol="000300", date="20240628")
    asda = ak.index_cons_change_csindex(symbol="000300", date="20240628")
    print(stock_hist_df.head())
except Exception as e:
    print(f"获取历史调整记录失败: {e}")

