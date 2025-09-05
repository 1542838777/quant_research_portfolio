from data.local_data_load import load_sw_l_n_codes
from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR
import pandas as pd

from quant_lib.tushare.data.downloader import get_all_stock_basic_from_api, download_qfq

if __name__ == '__main__':
    download_qfq()