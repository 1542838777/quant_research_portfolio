# -*- coding: utf-8 -*-

import tushare as ts
from token_manager import get_valid_token

# import pandas as pd


def initialize_tushare_with_token(token):
    """使用指定token初始化tushare"""
    ts.set_token(token)
    return ts.pro_api()


def main():
    print("=== tushare 基本使用示例 ===\n")

    # 1. 获取有效的 API Token
    print("1. 获取有效的 API Token...")
    token = get_valid_token()
    if not token:
        print("✗ 无法获取有效的token，程序退出")
        return
    print()

    # 2. 初始化 Pro API
    print("2. 初始化 Pro API...")
    pro = initialize_tushare_with_token(token)
    print("✓ Pro API 初始化成功\n")

    # 3. 获取指数日线数据
    print("3. 获取上证指数日线数据...")
    try:
        df_index = pro.index_daily(
            ts_code="000001.SH", start_date="20250620", end_date="20250628"  # 上证指数
        )
        print(f"✓ 获取到 {len(df_index)} 条指数数据")
        print("最新5条数据:")
        print(df_index.head())
        print()
        
    except Exception as e:
        print(f"✗ 获取指数数据失败: {e}")

    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()
