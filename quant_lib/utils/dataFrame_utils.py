import pandas as pd
from typing import Dict

from pandas import DataFrame


#对齐传入的多个df （不限个数）
def align_dfs(**dfs: object) -> Dict[str, pd.DataFrame]:
    """
    对齐多个 DataFrame，返回一个 dict[str, DataFrame]，保留原来的 key。
    对齐索引和列，只有共同部分会被保留。
    """
    if not dfs:
        return {}

    # 只保留 value 是 DataFrame 的项
    dfs_items = [(k, v) for k, v in dfs.items() if isinstance(v, pd.DataFrame)]
    if not dfs_items:
        return {}

    # 找出所有DataFrame的共同索引和列
    common_index = None
    common_columns = None
    for _, df in dfs_items:
        if common_index is None:
            common_index = df.index
            common_columns = df.columns
        else:
            common_index = common_index.intersection(df.index)
            common_columns = common_columns.intersection(df.columns)

    if len(common_index) == 0 or len(common_columns) == 0:
        raise ValueError("警告：没有共同的索引或列，无法对齐")

    # 对齐所有DataFrame
    result_dict = {}
    for key, df in dfs_items:
        aligned_df = df.reindex(index=common_index, columns=common_columns)
        result_dict[key] = aligned_df

    return result_dict

def align_dataframes(all_dfs_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    对齐多个DataFrame的索引和列

    Args:
        all_dfs_dict: DataFrame字典

    Returns:
        对齐后的DataFrame字典
    """
    if not all_dfs_dict:
        return {}

    try:
        # 找出所有DataFrame的共同索引和列
        common_index = None
        common_columns = None

        for name, df in all_dfs_dict.items():
            # print(f"处理DataFrame: {name}, 形状: {df.shape}")

            if common_index is None:
                common_index = df.index
                common_columns = df.columns
            else:
                common_index = common_index.intersection(df.index)
                common_columns = common_columns.intersection(df.columns)

        # print(f"共同索引数量: {len(common_index)}, 共同列数量: {len(common_columns)}")

        if len(common_index) == 0 or len(common_columns) == 0:
            raise ValueError("警告：没有共同的索引或列，无法对齐")
            # return {}

        # 对齐所有DataFrame
        aligned_dfs_dict = {}
        for key, df in all_dfs_dict.items():
            aligned_df = df.reindex(index=common_index, columns=common_columns)
            aligned_dfs_dict[key] = aligned_df

        return aligned_dfs_dict

    except Exception as e:
        raise ValueError(f"数据对齐失败: {e}")
