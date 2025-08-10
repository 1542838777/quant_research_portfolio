import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict

class NumpyEncoder(json.JSONEncoder):
    """
    一个自定义的JSON编码器，用于处理Numpy的数据类型。
    当json.dump遇到它不认识的Numpy类型时，会调用这个类的default方法。
    """
    def default(self, obj: Any) -> Any:
        # 如果对象是Numpy的整数类型，则转换为Python的标准int
        if isinstance(obj, np.integer):
            return int(obj)
        # 如果对象是Numpy的浮点数类型，则转换为Python的标准float
        elif isinstance(obj, np.floating):
            # 处理无穷大和NaN值，将其转换为None(在JSON中为null)
            if np.isinf(obj) or np.isnan(obj):
                return None
            return float(obj)
        # 如果对象是Numpy的布尔类型，则转换为Python的标准bool
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # 如果对象是Numpy的数组，则转换为Python的列表
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # 如果对象是pandas的Timestamp，转换为标准的ISO格式字符串
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # 对于其他所有它不认识的类型，调用父类的默认方法（这通常会引发一个TypeError）
        return super().default(obj)

def save_json_with_numpy(data: Dict, file_path: Path):
    """
    使用自定义的NumpyEncoder，安全地将包含Numpy数据类型的字典保存为JSON文件。

    Args:
        data (Dict): 需要保存的字典。
        file_path (Path): 目标文件路径。
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
        print(f"✓ 数据已成功保存至: {file_path}")
    except Exception as e:
        print(f"❌ 保存JSON文件时出错: {e}")
        raise

def load_json_with_numpy(file_path: Path) -> Dict:
    """
    从JSON文件中加载数据。
    这是一个对标准json.load的简单封装，主要用于保持接口的对称性。
    因为保存时已经处理了Numpy类型，加载时使用标准库即可。

    Args:
        file_path (Path): 源文件路径。

    Returns:
        Dict: 从文件中加载的字典。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # print(f"✓ 数据已成功从 {file_path} 加载。")
        return data
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"❌ 错误：文件 {file_path} 不是一个有效的JSON文件。")
        raise
    except Exception as e:
        print(f"❌ 加载JSON文件时发生未知错误: {e}")
        raise