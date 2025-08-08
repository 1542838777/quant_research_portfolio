import pandas as pd

from data.local_data_load import get_industry_record_df_processed
from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR
from quant_lib.tushare.api_wrapper import call_pro_tushare_api, call_ts_tushare_api
from quant_lib.tushare.data.downloader import download_index_weights, download_index_daily_info, download_suspend_d, \
    download_cashflow, download_income, download_balancesheet, download_industry_record
from quant_lib.tushare.tushare_client import TushareClient

# daily_hfq 有问题
def get_fields_map():
    result = []
    paths = ['adj_factor', 'daily', 'daily_basic', 'daily_hfq',  'index_weights', 'margin_detail',
             'stk_limit',

             'cashflow.parquet',
             'income.parquet',
             'index_daily.parquet',
             'namechange.parquet',
             'stock_basic.parquet',
             'fina_indicator.parquet',
             'suspend_d.parquet',
             'trade_cal.parquet',
             'balancesheet.parquet',

             ]

    for path in paths:
        orin_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / path)
        print(f'logic_name:{path}')
        if 'trade_date' in orin_df.columns:
            orin_df['trade_date'] = pd.to_datetime(orin_df['trade_date'])
            df = orin_df.copy(deep=True)
            df = df[(df['trade_date'] >= pd.to_datetime('20250611')) & (df['ts_code'] == '000012.SZ')]
            df['trade_date_and_tscode_重复次数'] = df.groupby(['trade_date','ts_code'])['trade_date'].transform('count')
            dup_rows = df[df['trade_date_and_tscode_重复次数'] > 1].sort_values(by='ts_code')
            print("结果如下----")
            print(dup_rows)

        #
        # print(f'logic_name:{path}')
        # print(f'\t fields:{list(df.columns)}')
        result.append({
            'name': path,
            'fields': list(df.columns)
        })
    return result


# daily_hfq 有问题
def dup_check():
    result = []
    # call_pro_tushare_api('index_daily', ts_code='000300.SH', start_date='20100101', end_date='20250711')
    # call_ts_tushare_api('pro_bar',**{'adj': 'hfq', 'asset': 'E','ts_code':'000002.SZ'})
    # api_df  = call_ts_tushare_api("pro_bar", ts_code='000012.SZ', start_date='20180601', end_date='20250711')
    # call_ts_tushare_api('pro_bar', adj =  'hfq', asset='E',ts_code='000002.SZ')

    paths = [
        'cashflow.parquet',
        'income.parquet'
    ]
    for path in paths:
        orin_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / path)
        print(f'logic_name:{path}')
        if 'trade_date' in orin_df.columns:
            orin_df['trade_date'] = pd.to_datetime(orin_df['trade_date'])
            df = orin_df.copy(deep=True)
            df = df[(df['trade_date'] >= pd.to_datetime('20250611')) & (df['ts_code'] == '000012.SZ')]
            df['trade_date_and_tscode_重复次数'] = df.groupby(['trade_date','ts_code'])['trade_date'].transform('count')
            dup_rows = df[df['trade_date_and_tscode_重复次数'] > 1].sort_values(by='ts_code')
            print("结果如下----")
            print(dup_rows)

        #
        # print(f'logic_name:{path}')
        # print(f'\t fields:{list(df.columns)}')
        result.append({
            'name': path,
            'fields': list(df.columns)
        })
    return result



# daily_hfq 有问题
def dup_check_report_type():
    result = []
    # call_pro_tushare_api('index_daily', ts_code='000300.SH', start_date='20100101', end_date='20250711')
    # call_ts_tushare_api('pro_bar',**{'adj': 'hfq', 'asset': 'E','ts_code':'000002.SZ'})
    # api_df  = call_ts_tushare_api("pro_bar", ts_code='000012.SZ', start_date='20180601', end_date='20250711')
    # call_ts_tushare_api('pro_bar', adj =  'hfq', asset='E',ts_code='000002.SZ')

    paths = [
        'income.parquet',
        'index_daily.parquet',
    ]
    for path in paths:
        orin_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / path)
        print(f'logic_name:{path}')
        if 'end_date' in orin_df.columns:
            orin_df['end_date'] = pd.to_datetime(orin_df['end_date'])
            df = orin_df.copy(deep=True)
            df = df[df['report_type'] !='1' ]
            print("结果如下----")
            print(df)

        #
        # print(f'logic_name:{path}')
        # print(f'\t fields:{list(df.columns)}')
        result.append({
            'name': path,
            'fields': list(df.columns)
        })
    return result


def compare_df_rows(df, index1, index2):
    """
    比较 df 中 index1 和 index2 两行，返回它们不一致的列名列表
    """
    row1 = df.loc[index1]
    row2 = df.loc[index2]

    # 比较所有列，找出值不同的列
    diff_cols = [
        col for col in df.columns
        if not (pd.isna(row1[col]) and pd.isna(row2[col])) and row1[col] != row2[col]
    ]
    print(f"Index {index1} 与 Index {index2} 不同的列：")
    print(diff_cols)
    return diff_cols


if __name__ == '__main__':
    df = get_industry_record_df_processed()
    ori_df= pd.read_parquet(LOCAL_PARQUET_DATA_DIR/'industry_record.parquet')
    df.sort_values(by=['ts_code','in_date'], ascending=[True,True], inplace=True)

    result = df.groupby('ts_code').filter(lambda x: (x['out_date'].isna()).sum() >2)
    result.sort_values(by=['ts_code','in_date'], ascending=[True,True], inplace=True)
    print(1)







    # imcome_df['f_ann_date'] = pd.to_datetime(imcome_df['f_ann_date'])
    # imcome_df = imcome_df[imcome_df['f_ann_date'] != imcome_df['ann_date']]
    # dup_check_report_type()
    # df['end_date'] = pd.to_datetime(df['end_date'])
    # df = df[df['end_date'] >= pd.to_datetime('2025')]
    # print(list(df.columns))

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
    #       dtype='object')}, {'name': 'fina_indicator.parquet', 'fields': Index(['ts_code', 'ann_date', 'end_date', 'eps', 'dt_eps', 'total_revenue_ps',
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
