import pandas as pd

from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR


def load_index_daily(start_date, end_date):

    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'index_daily.parquet')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
    df = df[mask]
    return df
