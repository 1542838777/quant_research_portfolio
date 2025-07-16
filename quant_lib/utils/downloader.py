# 文件名: downloader.py
# 作用：一个健壮的数据下载器，用于从Tushare获取所有需要的A股核心数据，
#      并以高效的Parquet格式保存在本地，为后续的量化研究做准备。
#      本脚本已处理好API的频率和单次条数限制。
import time
from datetime import timedelta, datetime
from pathlib import Path

import pandas as pd
from yfinance import download

from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR
from quant_lib.tushare.api_wrapper import call_pro_tushare_api, call_ts_tushare_api
from quant_lib.tushare.tushare_client import TushareClient

# --- 2. 全局配置 ---
START_YEAR = 2018
END_YEAR = datetime.now().year
BATCH_SIZE = 20


# --- 3. 辅助函数 ---
def get_year_end(year):
    if year == END_YEAR:
        beijing_now = datetime.utcnow() + timedelta(hours=8)
        beijing_yesterday = beijing_now - timedelta(days=1)
        return beijing_yesterday.strftime('%Y%m%d')
    return f'{year}1231'


def download_index_weights():
    """下载指数成分股历史数据"""
    print("\n===== 开始下载指数成分股数据 =====")

    # 常用指数列表
    index_codes = [
        '000300.SH',  # 沪深300
        '000905.SH',  # 中证500
        '000906.SH',  # 中证800
        '000852.SH',  # 中证1000
    ]

    for index_code in index_codes:
        index_path = LOCAL_PARQUET_DATA_DIR / 'index_weights' / f"{index_code.replace('.', '_')}"

        for year in range(START_YEAR, END_YEAR + 1):
            year_path = index_path / f"year={year}"

            if not year_path.exists():
                print(f"--- 正在下载 {year} 年的 {index_code} 成分股数据 ---")

                year_start = f'{year}0101'
                year_end = get_year_end(year)

                # 获取该年度所有交易日
                trade_cal = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'trade_cal.parquet')
                trade_dates = trade_cal[
                    (trade_cal['cal_date'] >= year_start) &
                    (trade_cal['cal_date'] <= year_end) &
                    (trade_cal['is_open'] == 1)
                    ]['cal_date'].tolist()

                all_weights = []

                # 逐日获取成分股
                for i, trade_date in enumerate(trade_dates):
                    print(f"    处理交易日 {trade_date} ({i + 1}/{len(trade_dates)})...")

                    try:
                        df_weight = call_pro_tushare_api(
                            'index_weight',
                            index_code=index_code,
                            trade_date=trade_date
                            # fields='index_code,con_code,trade_date,weight'
                        )

                        if not df_weight.empty:
                            all_weights.append(df_weight)
                        else:
                            print(f"      {trade_date} 无成分股数据")

                    except Exception as e:
                        print(f"      {trade_date} 获取失败: {e}")
                        continue

                # 保存年度数据
                if all_weights:
                    final_df = pd.concat(all_weights, ignore_index=True)
                    final_df.drop_duplicates(inplace=True)

                    year_path.mkdir(parents=True, exist_ok=True)
                    final_df.to_parquet(year_path / 'data.parquet')
                    print(f"成功保存 {year} 年的 {index_code} 成分股数据")
                else:
                    print(f"未获取到 {year} 年的 {index_code} 成分股数据")
            else:
                print(f"{year} 年的 {index_code} 成分股数据已存在，跳过下载")

def download_stock_info(stock_basic_path):
    if not stock_basic_path.exists():
        print("--- 正在下载股票基本信息 ---")
        stock_basic =   TushareClient.get_pro().stock_basic( list_status = 'L,D,P', fields='ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type')
        if not stock_basic.empty:
            stock_basic.to_parquet(stock_basic_path)
    else:
        print("股票基本信息已存在，跳过下载。")


def download_stock_change_name_details():
    # 在 downloader.py 的“下载配套数据”部分，增加以下逻辑

    namechange_path = LOCAL_PARQUET_DATA_DIR / 'namechange.parquet'
    if not namechange_path.exists():
        print("--- 正在下载股票名称变更历史 ---")
        # Tushare的namechange接口可能需要循环获取，因为它有单次返回限制
        # 一个稳健的做法是获取所有股票列表，然后逐个调用
        stock_list = call_pro_tushare_api("stock_basic", list_status = 'L,D,P',fields='ts_code')['ts_code'].tolist()
        all_changes = []
        startIdx=0
        for stock in stock_list:
            print(f"开始处理第{startIdx+1}/{len(stock_list)}只股票")
            df = call_pro_tushare_api("namechange", ts_code=stock)
            df.drop_duplicates(inplace=True)
            all_changes.append(df)
            startIdx+=1

        namechange_df = pd.concat(all_changes)
        if not namechange_df.empty:
            namechange_df.to_parquet(namechange_path)
            print(f"股票名称变更历史已保存到 {namechange_path}")
    else:
        print("股票名称变更历史已存在，跳过下载。")


# --- 4. 主下载逻辑 ---
if __name__ == '__main__':
    LOCAL_PARQUET_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- 下载配套数据 (逻辑不变) ---
    print("\n===== 开始下载全局配套数据 =====")
    trade_cal_path = LOCAL_PARQUET_DATA_DIR / 'trade_cal.parquet'
    if not trade_cal_path.exists():
        print("--- 正在下载交易日历 ---")
        trade_cal = call_pro_tushare_api("trade_cal", start_date=f'{START_YEAR}0101', end_date=f'{END_YEAR}1231')
        if not trade_cal.empty:
            trade_cal.to_parquet(trade_cal_path)
            print(f"交易日历已保存到 {trade_cal_path}")
    else:
        print("交易日历已存在，跳过下载。")

    stock_basic_path = LOCAL_PARQUET_DATA_DIR / 'stock_basic.parquet'

    download_stock_info(stock_basic_path)

    # --- 按年份循环下载核心数据 ---
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n===== 开始处理年份: {year} =====")
        year_start = f'{year}0101'
        year_end = get_year_end(year)

        all_stocks_df = pd.read_parquet(stock_basic_path)
        symbols = all_stocks_df[~all_stocks_df['ts_code'].str.endswith('.BJ')]['ts_code'].unique().tolist()
        print(f"获取到 {year} 年全市场 {len(symbols)} 只股票作为处理对象。")

        data_to_download_tasks = {
            # 模式: by_stock (逐个股票循环，适用于pro_bar等)
            'daily_hfq': {'func': 'pro_bar', 'params': {'adj': 'hfq', 'asset': 'E'}, 'mode': 'by_stock',
                          'api_type': 'ts'},
            'margin_detail': {'func': 'margin_detail', 'params': {}, 'mode': 'by_stock'},  # <-- 修正下载模式为 'by_stock'
            'stk_limit': {'func': 'stk_limit', 'params': {}, 'mode': 'by_stock'},

            # 模式: batch_stock (按股票代码分批，适用于大多数pro接口)
            'daily': {'func': 'daily', 'params': {}, 'mode': 'batch_stock'},
            'daily_basic': {'func': 'daily_basic', 'params': {}, 'mode': 'batch_stock'},
            'adj_factor': {'func': 'adj_factor', 'params': {}, 'mode': 'batch_stock'},
            'fina_indicator_vip': {'func': 'fina_indicator_vip', 'params': {},
                                   'mode': 'batch_stock'},

        }

        for name, info in data_to_download_tasks.items():
            year_path = LOCAL_PARQUET_DATA_DIR / name / f"year={year}"
            if not year_path.exists():
                print(f"--- 正在下载 {year} 年的 {name} 数据 ---")
                all_data_list = []

                download_mode = info['mode']

                if download_mode == 'by_stock':
                    # 【新逻辑】按单个股票循环下载，保证数据完整性
                    print(f"  采用'逐个股票'模式下载...")
                    for i, symbol in enumerate(symbols):
                        print(f"    处理股票 {symbol} ({i + 1}/{len(symbols)})...")
                        api_params = {'ts_code': symbol, 'start_date': year_start, 'end_date': year_end,
                                      **info.get('params', {})}

                        api_type = info.get('api_type', 'pro')
                        if api_type == 'ts':
                            df_batch = call_ts_tushare_api(info['func'], **api_params)
                        else:
                            df_batch = call_pro_tushare_api(info['func'], **api_params)
                        all_data_list.append(df_batch)

                elif download_mode == 'batch_stock':
                    # 【旧逻辑】按股票分批下载
                    print(f"  采用'按股票分批'模式下载...")
                    for i in range(0, len(symbols), BATCH_SIZE):
                        batch_symbols = symbols[i: i + BATCH_SIZE]
                        symbols_str = ",".join(batch_symbols)
                        print(f"    处理批次 {i // BATCH_SIZE + 1} ({len(batch_symbols)}只股票)...")

                        api_params = {'ts_code': symbols_str, 'start_date': year_start, 'end_date': year_end,
                                      **info['params']}

                        df_batch = call_pro_tushare_api(info['func'], **api_params)
                        all_data_list.append(df_batch)

                # --- 通用的数据合并与保存逻辑 ---
                if all_data_list:
                    final_df = pd.concat(all_data_list, ignore_index=True)
                    if not final_df.empty:
                        if name == 'fina_indicator_vip':  # 只有他源数据有问题，会有重复的
                            final_df.sort_values(by='ann_date', ascending=True, inplace=True)
                            final_df.drop_duplicates(subset=['ts_code', 'end_date'], keep='last', inplace=True)
                        else:
                            final_df.drop_duplicates(inplace=True)

                        year_path.mkdir(parents=True, exist_ok=True)
                        final_df.to_parquet(year_path / 'data.parquet')
                        print(f"成功保存 {year} 年的 {name} 数据。")
            else:
                print(f"{year} 年的 {name} 数据已存在，跳过下载。")

    print("\n===== 所有数据下载任务完成！ =====")
#todo 注意 股票状态 需要每天刷新！
#todo trade_date 每天刷新！
#也就是每天重新下载download_stock_info ，更新股票状态1