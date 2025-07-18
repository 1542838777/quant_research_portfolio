"""
测试修复后的数据加载器
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.data_loader import DataLoader
from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR

def test_data_loader():
    """测试数据加载器的字段检查功能"""
    print("测试修复后的数据加载器...")
    
    try:
        # 创建数据加载器
        data_loader = DataLoader(data_path=LOCAL_PARQUET_DATA_DIR)
        
        # 测试加载一些基础字段
        test_fields = ['close', 'total_mv', 'pe_ttm']
        
        print(f"尝试加载字段: {test_fields}")
        
        # 加载数据
        data_dict = data_loader.load_data(
            fields=test_fields,
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        print(f"成功加载 {len(data_dict)} 个字段:")
        for field_name, df in data_dict.items():
            print(f"  {field_name}: {df.shape}")
        
        print("\n✅ 数据加载器测试成功！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loader()
