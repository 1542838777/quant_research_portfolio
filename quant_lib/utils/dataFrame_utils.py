import pandas as pd
from typing import Dict

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
            print(f"处理DataFrame: {name}, 形状: {df.shape}")

            if common_index is None:
                common_index = df.index
                common_columns = df.columns
            else:
                common_index = common_index.intersection(df.index)
                common_columns = common_columns.intersection(df.columns)

        print(f"共同索引数量: {len(common_index)}, 共同列数量: {len(common_columns)}")

        if len(common_index) == 0 or len(common_columns) == 0:
            print("警告：没有共同的索引或列，无法对齐")
            return {}

        # 对齐所有DataFrame
        aligned_dfs_dict = {}
        for key, df in all_dfs_dict.items():
            aligned_df = df.reindex(index=common_index, columns=common_columns)
            aligned_dfs_dict[key] = aligned_df

        return aligned_dfs_dict

    except Exception as e:
        print(f"数据对齐失败: {e}")
        return {}