import pandas as pd

from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR
from quant_lib.tushare.data.downloader import download_index_weights


def get_fields_map():
    result = []
    paths = ['adj_factor', 'daily', 'daily_basic', 'daily_hfq', 'fina_indicator_vip', 'margin_detail', 'stk_limit',
             'stock_basic.parquet',
             'trade_cal.parquet']
    for path in paths:
        df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / path)
        result.append({
            'name': path,
            'fields': df.columns
        })
    return result


if __name__ == '__main__':
    download_index_weights()

    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR/'namechange.parquet')
    print(df.colun)

    print(f'总数{sum}')

    ##
    #
    #
    # D:\lqs\codeAbout\py\env\vector_bt_env\Scripts\python.exe D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\data\test.py
    # [{'name': 'adj_factor', 'fields': Index(['ts_code', 'trade_date', 'adj_factor', 'year'], dtype='object')}, {'name': 'daily', 'fields': Index(['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close',
    #        'change', 'pct_chg', 'vol', 'amount', 'year'],
    #       dtype='object')}, {'name': 'daily_basic', 'fields': Index(['ts_code', 'trade_date', 'close', 'turnover_rate', 'turnover_rate_f',
    #        'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
    #        'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv',
    #        'circ_mv', 'year'],
    #       dtype='object')}, {'name': 'daily_hfq', 'fields': Index(['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close',
    #        'change', 'pct_chg', 'vol', 'amount', 'year'],
    #       dtype='object')}, {'name': 'fina_indicator_vip', 'fields': Index(['ts_code', 'ann_date', 'end_date', 'eps', 'dt_eps', 'total_revenue_ps',
    #        'revenue_ps', 'capital_rese_ps', 'surplus_rese_ps', 'undist_profit_ps',
    #        ...
    #        'bps_yoy', 'assets_yoy', 'eqt_yoy', 'tr_yoy', 'or_yoy', 'q_sales_yoy',
    #        'q_op_qoq', 'equity_yoy', 'update_flag', 'year'],
    #       dtype='object', length=110)}, {'name': 'margin_detail', 'fields': Index(['trade_date', 'ts_code', 'rzye', 'rqye', 'rzmre', 'rqyl', 'rzche',
    #        'rqchl', 'rqmcl', 'rzrqye', 'year'],
    #       dtype='object')}, {'name': 'stk_limit', 'fields': Index(['trade_date', 'ts_code', 'up_limit', 'down_limit', 'year'], dtype='object')}, {'name': 'stock_basic.parquet', 'fields': Index(['ts_code', 'symbol', 'name', 'area', 'industry', 'cnspell', 'market',
    #        'list_date', 'act_name', 'act_ent_type'],
    #       dtype='object')}, {'name': 'trade_cal.parquet', 'fields': Index(['exchange', 'cal_date', 'is_open', 'pretrade_date'], dtype='object')}]
    # 总数187
    #
    # Process finished with exit code 0#