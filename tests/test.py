
import  pandas as pd

from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR

import akshare as ak

from quant_lib.tushare.api_wrapper import call_pro_tushare_api, call_ts_tushare_api
from quant_lib.tushare.tushare_client import TushareClient


def tesasdadst():
    ret =     call_ts_tushare_api("pro_bar",ts_code='000001.SZ', start_date='20250711',end_date='20250711' ,adj='hfq')
    daily_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR/'daily'/'year=2025'/'data.parquet')
    daily_basic_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR/'daily_basic'/'year=2025'/'data.parquet')
    hfq_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR/'daily_hfq'/'year=2025'/'data.parquet')

    print()
# def testAK():
#     df = ak.stock_zh_a_hist(symbol="sz000002",
#                             start_date="20181228",
#                             end_date="20181228",
#                             adjust="hfq")  # adjust="hfq" 表示后复权
#     print(df[['date', 'close']])

def read_():
    for name in ['daily_basic', 'daily_hfq', 'adj_factor', 'namechange.parquet', 'stock_basic.parquet']:

        df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR/name)
        print(f"{name}\n")
        print(f'{df.isna().mean()}')
        print("---------------------\n")

def api():
    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily_hfq')
    df = df[df['ts_code'] == '002251.SZ']

    df = df[(df['trade_date']=='20230111') & (df['ts_code'] == '002251.SZ')]
    print(df)
if __name__ == '__main__':
    api()