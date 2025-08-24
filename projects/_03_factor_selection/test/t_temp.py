from data.local_data_load import load_sw_daily, load_sw_l_n_codes
from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR
from quant_lib.tushare.data.downloader import download_sw_daily, download_sw_basic_info
import pandas as pd
if __name__ == '__main__':
    load_sw_daily(load_sw_l_n_codes('一级行业指数'))