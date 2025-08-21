import pandas as pd
import akshare as ak
from datetime import datetime

from data.local_data_load import load_income_df, load_trading_lists


##注意 这份Excel数据中，如果一只股票的“剔除日期”是 2022-02-22，那么在 2022-02-22 这一天它已经【不算】在指数内了。
##换句话说，2022-02-22 这个日期代表的是剔除操作的“生效日”，即这一天是它被剔除后的第一天。它在指数内的最后一天是 2022-02-21。
# ==============================================================================
# --- 1. 用户配置区 (请根据您的实际情况修改) ---
# ==============================================================================

# 请将这里替换为您Excel文件的【绝对路径或相对路径】
# Windows示例: "C:/Users/YourUser/Documents/csi500_components.xlsx"
# Mac/Linux示例: "/home/user/data/csi500_components.xlsx"
EXCEL_FILE_PATH_500 = "D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\models\index_files\指数成分(中证500_000905).xlsx"
EXCEL_FILE_PATH_300 = "D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\models\index_files\指数成分(沪深300_000300).xlsx"

# 请设置您要检查的回测开始和结束日期
START_DATE = "2018-01-01"
END_DATE = "2025-07-20"

# 请设置指数的理论成分股数量 (中证500/沪深300等，用于对比)
EXPECTED_COUNT_500 = 500
EXPECTED_COUNT_300 = 300


# ==============================================================================
# --- 2. 核心代码区 (通常无需修改) ---
# ==============================================================================

def format_stock_code(row: pd.Series) -> str:
    """
    根据交易市场，将纯数字的股票代码转换为带后缀的格式。
    注意：您系统是sz结尾的，但上交所是ss，这里做了兼容处理。
    """
    code = str(row['成分券代码']).zfill(6)
    market = row['交易市场']
    if market == 'XSHE':  # 深圳证券交易所
        return f"{code}.SZ"
    elif market == 'XSHG':  # 上海证券交易所
        return f"{code}.SS"
    else:
        return code



def verify_index_constituents(file_path: str, start_date: str, end_date: str, EXPECTED_COUNT_500: int):
    """
    主验证函数：读取Excel，执行每日数量检查，并输出报告。
    """
    print("--- 开始验证指数成分股数据的完整性 ---")

    # --- 步骤 1: 加载并预处理数据 ---
    try:
        print(f"正在读取Excel文件: {file_path} ...")
        df = pd.read_excel(file_path)
        print("文件读取成功！数据样例如下：")
        print(df.head())
    except FileNotFoundError:
        print(f"【错误】文件未找到！请检查路径: {EXCEL_FILE_PATH_500}")
        return
    except Exception as e:
        print(f"【错误】读取Excel文件失败: {e}")
        return

    # 数据清洗和格式转换
    print("\n正在进行数据预处理...")
    # a) 股票代码格式转换
    df['stock_code'] = df.apply(format_stock_code, axis=1)

    # b) 日期格式转换 (errors='coerce' 会将空值或错误格式变为 NaT)
    df['in_date'] = pd.to_datetime(df['纳入日期'], errors='coerce')
    df['out_date'] = pd.to_datetime(df['剔除日期'], errors='coerce')

    # 丢弃日期格式不正确的行
    df.dropna(subset=['in_date'], inplace=True)
    print("股票代码和日期格式处理完毕。")

    # --- 步骤 2: 获取回测区间内的所有交易日 ---
    print(f"\n正在获取 {start_date} 到 {end_date} 的交易日历...")
    try:
        trade_dates_in_range = load_trading_lists(start_date,end_date)

        print(f"获取到 {len(trade_dates_in_range)} 个交易日。")
    except Exception as e:
        print(f"【错误】从Akshare获取交易日历失败，请检查网络连接: {e}")
        return

    # --- 步骤 3: 遍历交易日，统计每日成分股数量 ---
    print("\n开始逐日统计成分股数量，过程可能需要一些时间...")
    daily_counts = []
    for current_date in trade_dates_in_range:
        # 核心筛选逻辑：
        # 1. 纳入日期 <= 当前日期
        # 2. 并且 (剔除日期为空 OR 剔除日期 > 当前日期)
        mask = (df['in_date'] <= current_date) & \
               (df['out_date'].isnull() | (df['out_date'] > current_date))

        count = mask.sum()
        daily_counts.append({'date': current_date, 'count': count})

    counts_df = pd.DataFrame(daily_counts)
    print("每日数量统计完成！")

    # --- 步骤 4: 分析结果并生成报告 ---
    print("\n--- 验证结果分析报告 ---")
    if counts_df.empty:
        print("【警告】在指定日期范围内没有统计到任何数据。")
        return

    # a) 总体统计
    min_count = counts_df['count'].min()
    max_count = counts_df['count'].max()
    mean_count = counts_df['count'].mean()

    print(f"检查周期: {start_date} 至 {end_date}")
    print(f"理论成分股数量: {EXPECTED_COUNT_500}")
    print(f"实际最小数量: {min_count}")
    print(f"实际最大数量: {max_count}")
    print(f"实际平均数量: {mean_count:.2f}")

    # b) 查找有问题的日期
    # 定义一个容忍区间，例如理论数量的±2%
    tolerance = 0.02
    lower_bound = EXPECTED_COUNT_500 * (1 - tolerance)
    upper_bound = EXPECTED_COUNT_500 * (1 + tolerance)

    problematic_dates_df = counts_df[
        (counts_df['count'] < lower_bound) |
        (counts_df['count'] > upper_bound)
        ]

    print("-" * 25)
    if problematic_dates_df.empty:
        print("【结论】恭喜！数据质量非常高。")
        print("在整个回测期间，每日成分股数量都在理论值±2%的正常范围内，没有发现明显的遗漏。")
    else:
        print(f"【结论】警告！数据存在明显的遗漏或错误。")
        print(f"共发现 {len(problematic_dates_df)} 个交易日的成分股数量异常（超出理论值±2%）。")
        print("异常日期及数量如下（最多展示20条）：")
        print(problematic_dates_df.head(20))
        print("\n建议：请仔细核对上述日期的前后发生了什么调整事件，或检查您的原始数据源。")


if __name__ == "__main__":

    # verify_index_constituents(
    #     file_path=EXCEL_FILE_PATH_500,
    #     start_date=START_DATE,
    #     end_date=END_DATE,
    #     EXPECTED_COUNT_500=EXPECTED_COUNT_500
    # )
    verify_index_constituents(
        file_path=EXCEL_FILE_PATH_300,
        start_date=START_DATE,
        end_date=END_DATE,
        EXPECTED_COUNT_500=EXPECTED_COUNT_300
    )