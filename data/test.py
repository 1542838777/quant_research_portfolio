import pandas as pd

from data.data_loader import LOCAL_PARQUET_DATA_DIR
from quant_lib.tushare.api_wrapper import call_pro_tushare_api, call_ts_tushare_api
from quant_lib.tushare.tushare_client import TushareClient


def get_fields_map():
    result = []
    paths = ['adj_factor', 'daily_basic', 'daily_hfq', 'fina_indicator_vip', 'margin_detail', 'stock_basic.parquet',
             'trade_cal.parquet']
    for path in paths:
        df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / path)
        result.append({
            'name': path,
            'fields': df.columns
        })
    return result


if __name__ == '__main__':
    long_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / "stk_limit")
    long_df = long_df[long_df['ts_code']=='002550.SZ']


    data_to_download_tasks = {
        'stk_limit': {'func': 'stk_limit', 'params': {}, 'mode': 'by_stock'},
        # 模式: batch_stock (按股票代码分批，适用于大多数pro接口)
        'fina_indicator_vip': {'func': 'fina_indicator_vip', 'params': {},
                               'mode': 'batch_stock'}

    }
    symbols = ['600000.SH']
    year_start = '20220101'
    year_end = '20221231'

    for name, info in data_to_download_tasks.items():

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

        elif download_mode == 'batch_stock':
            # 【旧逻辑】按股票分批下载
            print(f"  采用'按股票分批'模式下载...")
            for i in range(0, len(symbols), 20):
                batch_symbols = symbols[i: i + 20]
                symbols_str = ",".join(batch_symbols)
                print(f"    处理批次 {i // 20 + 1} ({len(batch_symbols)}只股票)...")

                api_params = {'ts_code': symbols_str, 'start_date': year_start, 'end_date': year_end,
                              **info['params']}

                df_batch = call_pro_tushare_api(info['func'], **api_params)
