# =============================================================================
# quant_lib/data_loader.py
# 职责：提供一个智能的数据加载器，能够根据请求的字段，自动从正确的
#      本地Parquet文件中加载、处理和对齐数据。
# =============================================================================
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from collections import defaultdict

from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR


class DataLoader:
    """
    一个智能的数据加载器类。
    它在初始化时会自动扫描数据目录，建立一个“字段 -> 文件”的映射，
    从而能够高效、动态地加载所需数据。
    """

    def __init__(self, LOCAL_PARQUET_DATA_DIR: Path):
        """
        初始化DataLoader，并构建字段映射。

        Args:
            LOCAL_PARQUET_DATA_DIR (Path): 存储Parquet文件的根目录。
        """
        self.LOCAL_PARQUET_DATA_DIR = LOCAL_PARQUET_DATA_DIR
        if not self.LOCAL_PARQUET_DATA_DIR.exists():
            raise FileNotFoundError(f"数据路径不存在: {self.LOCAL_PARQUET_DATA_DIR}")

        print("DataLoader正在初始化，开始扫描数据文件以构建字段映射...")
        self._field_map = self._build_field_map()
        print("字段映射构建完毕，DataLoader准备就绪。")
        # 打印部分映射以供检查
        print("示例映射:", list(self._field_map.items())[:5])

    def _build_field_map(self) -> dict:
        """
        扫描数据目录，构建一个从字段名到其所在逻辑数据集名称的映射。
        例如：{'close': 'daily_hfq', 'pe': 'daily_basic', ...}
        """
        field_to_file_map = {}
        # 使用 .rglob('*.parquet') 可以递归查找所有子目录中的parquet文件
        for file_path in self.LOCAL_PARQUET_DATA_DIR.rglob('*.parquet'):
            try:
                # 只读取第一行来获取列名，极大提升速度，避免加载整个文件
                columns = pq.read_schema(file_path).names

                # --- 【核心修正】智能识别逻辑名称 ---
                if file_path.stem == 'data':
                    # 如果是分区数据 (e.g., .../daily_hfq/year=2022/data.parquet)
                    # 它的逻辑名称是其上上级目录的名称
                    logical_name = file_path.parent.parent.name
                else:
                    # 如果是单文件 (e.g., .../stock_basic.parquet)
                    # 它的逻辑名称就是文件名本身
                    logical_name = file_path.stem

                for col in columns:
                    if col not in field_to_file_map:
                        field_to_file_map[col] = logical_name
            except Exception as e:
                print(f"警告：读取文件 {file_path} 的元数据失败: {e}")
        return field_to_file_map

    def load_data(self, fields: list, start_date: str, end_date: str) -> dict:
        """
        根据给定的字段列表，加载、处理并返回对齐后的宽表数据字典。

        Args:
            fields (list): 需要加载的字段名称列表, e.g., ['close', 'pe', 'roe']
            start_date (str): 开始日期
            end_date (str): 结束日期

        Returns:
            dict: 键为字段名，值为处理好的宽表DataFrame。
        """
        print(f"\n开始加载字段: {fields}")

        # --- 1. 根据请求的字段，确定需要加载哪些文件和哪些列 ---
        file_to_fields = defaultdict(list)
        # 自动添加回测必须的基础列
        base_fields = ['ts_code', 'trade_date']

        for field in list(set(fields + base_fields)):  # 确保基础字段也被映射
            logical_name = self._field_map.get(field)
            if logical_name:
                file_to_fields[logical_name].append(field)

        print(f"需要加载的数据集及对应字段: {dict(file_to_fields)}")

        # --- 2. 动态加载和转换数据 ---
        raw_dfs = {}
        for logical_name, columns_to_load in file_to_fields.items():
            try:
                # Pandas可以智能地读取分区或单个文件
                # 我们只需要提供逻辑名称作为路径的一部分
                file_path = self.LOCAL_PARQUET_DATA_DIR / logical_name

                # 高效读取：只加载需要的列
                long_df = pd.read_parquet(file_path, columns=list(set(columns_to_load+base_fields)))

                # 时间范围筛选
                long_df['trade_date'] = pd.to_datetime(long_df['trade_date'])
                long_df = long_df[(long_df['trade_date'] >= start_date) & (long_df['trade_date'] <= end_date)]

                # 将长表转换为宽表
                for col in columns_to_load:
                    if col not in base_fields:
                        wide_df = long_df.pivot_table(index='trade_date', columns='ts_code', values=col)
                        raw_dfs[col] = wide_df
            except Exception as e:
                print(f"错误：处理数据集 {logical_name} 失败: {e}")

        # --- 3. 对齐所有加载的宽表 ---
        if not raw_dfs:
            print("警告：没有加载到任何数据。")
            return {}

        common_dates = None
        common_stocks = None
        for name, df in raw_dfs.items():
            if common_dates is None:
                common_dates = df.index
                common_stocks = df.columns
            else:
                common_dates = common_dates.intersection(df.index)
                common_stocks = common_stocks.intersection(df.columns)

        # --- 4. 最终裁剪和填充 ---
        aligned_data = {}
        for name, df in raw_dfs.items():
            # 重新索引以确保所有df具有完全相同的索引和列，然后再填充
            aligned_df = df.reindex(index=common_dates, columns=common_stocks)
            aligned_df = aligned_df.sort_index()
            # 使用前向填充处理缺失值
            aligned_df = aligned_df.ffill()
            aligned_data[name] = aligned_df

        print("数据加载和对齐完毕。")
        return aligned_data

# =============================================================================
# 使用示例
# =============================================================================
if __name__ == '__main__':
    # 1. 定义你的数据路径

    # 2. 创建DataLoader实例 (这一步会自动扫描并建立映射)
    loader = DataLoader(LOCAL_PARQUET_DATA_DIR=LOCAL_PARQUET_DATA_DIR)

    # 3. 定义你这次研究需要的字段列表
    required_fields = ['close', 'pe_ttm', 'pb', 'roe']

    # 4. 调用load_data方法，获取你需要的所有数据
    data_dict = loader.load_data(
        fields=required_fields,
        start_date='2022-01-01',
        end_date='2023-12-31'
    )

    # 5. 查看结果
    if data_dict:
        print("\n--- 加载数据预览 ---")
        for name, df in data_dict.items():
            print(f"\nDataFrame: '{name}'")
            print(df.head())
