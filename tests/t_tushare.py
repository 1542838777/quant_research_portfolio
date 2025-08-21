from quant_lib.tushare.api_wrapper import call_pro_tushare_api, call_ts_tushare_api


def t_index_member():
    df = call_pro_tushare_api('index_member',index_code='000300.SH',date='20180718')
    print(1)
if __name__ == '__main__':
    t_index_member()