import os
import json


# 场景： 单因子测试，追加新的因子测试结果 （会直接覆盖原始 original_data 中已有的同名 key 的值。也就是说：
def update_json_file(filepath, new_data):
    """
    将 new_data 追加写入到 JSON 文件的最外层（字典合并）。
    :param filepath: str, JSON 文件路径
    :param new_data: dict, 需要追加写入的键值对
    """
    # 创建文件夹（如果不存在）
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 如果文件已存在，先读取旧内容
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                original_data = json.load(f)
                if not isinstance(original_data, dict):
                    raise ValueError("原始 JSON 文件不是 dict 格式")
            except json.JSONDecodeError:
                original_data = {}
    else:
        original_data = {}

    # 合并字典
    original_data.update(new_data)

    # 写入回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, ensure_ascii=False, indent=2)

    print(f"[+] 已更新 JSON 文件：{filepath}")


def read_json(filepath):
    """读取 JSON 文件，返回 dict，若文件不存在或为空则返回空 dict"""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("JSON 文件不是 dict 格式")
            return data
        except json.JSONDecodeError:
            return {}
