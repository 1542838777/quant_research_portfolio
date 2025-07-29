"""
测试动态股票池功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_manager import DataManager

def test_universe():
    """测试股票池构建"""
    print("测试动态股票池功能...")
    
    # 配置文件路径
    config_path = Path(__file__).parent / 'config.yaml'
    
    try:
        # 创建数据管理器
        data_manager = DataManager(str(config_path))
        
        # 加载数据（包括构建股票池）
        data_dict = data_manager.load_all_data()
        
        # 获取股票池
        stock_pool_df = data_manager.get_universe()
        
        # 显示股票池统计信息
        print("\n股票池统计信息:")
        print(f"时间范围: {stock_pool_df.index[0]} 到 {stock_pool_df.index[-1]}")
        print(f"股票数量: {len(stock_pool_df.columns)}")
        
        daily_count = stock_pool_df.sum(axis=1)
        print(f"平均每日股票数: {daily_count.mean():.0f}")
        print(f"最少每日股票数: {daily_count.min():.0f}")
        print(f"最多每日股票数: {daily_count.max():.0f}")
        
        # 显示前几天的股票池情况
        print("\n前5天股票池情况:")
        for i in range(min(5, len(stock_pool_df))):
            date = stock_pool_df.index[i]
            count = stock_pool_df.iloc[i].sum()
            print(f"{date.strftime('%Y-%m-%d')}: {count}只股票")
        
        print("\n测试成功！动态股票池已正常构建。")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_universe()
