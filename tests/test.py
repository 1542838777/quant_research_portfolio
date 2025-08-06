import pandas as pd

from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR, parquet_file_names, every_day_parquet_file_names, \
    need_fix
from quant_lib.config.logger_config import setup_logger

import akshare as ak

from quant_lib.tushare.api_wrapper import call_pro_tushare_api, call_ts_tushare_api
from quant_lib.tushare.tushare_client import TushareClient

# 配置日志
logger = setup_logger(__name__)


def tesasdadst():
    ret = call_ts_tushare_api("pro_bar", ts_code='000001.SZ', start_date='20250711', end_date='20250711', adj='hfq')
    daily_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily' / 'year=2025' / 'data.parquet')
    daily_basic_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily_basic' / 'year=2025' / 'data.parquet')
    hfq_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily_hfq' / 'year=2025' / 'data.parquet')

    logger.info("测试完成")


# def testAK():
#     df = ak.stock_zh_a_hist(symbol="sz000002",
#                             start_date="20181228",
#                             end_date="20181228",
#                             adjust="hfq")  # adjust="hfq" 表示后复权
#     print(df[['date', 'close']])

def read_():
    for name in ['daily_basic', 'daily_hfq', 'adj_factor', 'namechange.parquet', 'stock_basic.parquet']:
        df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / name)
        print(f"{name}\n")
        print(f'{df.isna().mean()}')
        print("---------------------\n")


def api():
    df = call_ts_tushare_api('pro_bar', ts_code='688086.SH', start_date='20180101', end_date='20181231')
    all_stocks_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'stock_basic.parquet')
    symbols = all_stocks_df[~all_stocks_df['ts_code'].str.endswith('.BJ')]['ts_code'].unique().tolist()
    df = call_ts_tushare_api('pro_bar', ts_code='000006.SZ', adj='hfq', start_date='20180101',
                             end_date='20251011')  # 到20240305都是有数据的
    print(df)



def red_close():
    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily_basic')
    df = df[df['ts_code'] == '000005.SZ']
    print(df)


def read_namechange():
    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'namechange.parquet')
    df = df[df['ts_code'] == '000001.SZ']
    df = df.sort_values('start_date')
    print(df)


def compare_local_and_net():
    miss_ts_codes = ['000006.SZ', '000007.SZ', '000019.SZ', '000029.SZ', '000031.SZ', '000034.SZ', '000035.SZ', '000038.SZ', '000046.SZ', '000056.SZ', '000156.SZ', '000415.SZ', '000503.SZ', '000506.SZ', '000509.SZ', '000511.SZ', '000516.SZ', '000533.SZ', '000534.SZ', '000540.SZ', '000547.SZ', '000564.SZ', '000566.SZ', '000606.SZ', '000616.SZ', '000629.SZ', '000663.SZ', '000693.SZ', '000703.SZ', '000757.SZ', '000766.SZ', '000793.SZ', '000796.SZ', '000812.SZ', '000820.SZ', '000900.SZ', '000912.SZ', '000930.SZ', '000939.SZ', '000948.SZ', '000950.SZ', '000998.SZ', '001872.SZ', '002004.SZ', '002005.SZ', '002012.SZ', '002033.SZ', '002037.SZ', '002047.SZ', '002072.SZ', '002075.SZ', '002082.SZ', '002085.SZ', '002098.SZ', '002103.SZ', '002113.SZ', '002121.SZ', '002122.SZ', '002143.SZ', '002147.SZ', '002161.SZ', '002163.SZ', '002168.SZ', '002189.SZ', '002198.SZ', '002212.SZ', '002213.SZ', '002219.SZ', '002252.SZ', '002259.SZ', '002260.SZ', '002263.SZ', '002301.SZ', '002309.SZ', '002312.SZ', '002321.SZ', '002345.SZ', '002357.SZ', '002358.SZ', '002366.SZ', '002384.SZ', '002398.SZ', '002413.SZ', '002423.SZ', '002427.SZ', '002437.SZ', '002442.SZ', '002450.SZ', '002451.SZ', '002464.SZ', '002507.SZ', '002509.SZ', '002512.SZ', '002517.SZ', '002520.SZ', '002523.SZ', '002524.SZ', '002532.SZ', '002538.SZ', '002545.SZ', '002575.SZ', '002580.SZ', '002584.SZ', '002592.SZ', '002604.SZ', '002607.SZ', '002619.SZ', '002621.SZ', '002622.SZ', '002647.SZ', '002656.SZ', '002661.SZ', '002662.SZ', '002675.SZ', '002692.SZ', '002694.SZ', '002699.SZ', '002721.SZ', '002726.SZ', '002738.SZ', '002739.SZ', '002770.SZ', '002799.SZ', '002837.SZ', '002851.SZ', '002857.SZ', '002870.SZ', '300004.SZ', '300032.SZ', '300038.SZ', '300049.SZ', '300056.SZ', '300064.SZ', '300067.SZ', '300071.SZ', '300086.SZ', '300089.SZ', '300090.SZ', '300103.SZ', '300116.SZ', '300128.SZ', '300134.SZ', '300143.SZ', '300146.SZ', '300156.SZ', '300159.SZ', '300173.SZ', '300198.SZ', '300208.SZ', '300225.SZ', '300238.SZ', '300240.SZ', '300266.SZ', '300278.SZ', '300280.SZ', '300290.SZ', '300292.SZ', '300312.SZ', '300317.SZ', '300325.SZ', '300337.SZ', '300341.SZ', '300356.SZ', '300383.SZ', '300392.SZ', '300409.SZ', '300411.SZ', '300424.SZ', '300441.SZ', '300444.SZ', '300455.SZ', '300477.SZ', '300578.SZ', '300593.SZ', '300640.SZ', '300656.SZ', '300659.SZ', '300682.SZ', '600051.SH', '600069.SH', '600076.SH', '600083.SH', '600084.SH', '600086.SH', '600112.SH', '600117.SH', '600122.SH', '600145.SH', '600148.SH', '600150.SH', '600157.SH', '600165.SH', '600179.SH', '600187.SH', '600221.SH', '600226.SH', '600228.SH', '600241.SH', '600255.SH', '600257.SH', '600270.SH', '600273.SH', '600290.SH', '600306.SH', '600309.SH', '600365.SH', '600378.SH', '600393.SH', '600399.SH', '600401.SH', '600432.SH', '600485.SH', '600490.SH', '600512.SH', '600515.SH', '600539.SH', '600568.SH', '600614.SH', '600634.SH', '600666.SH', '600673.SH', '600682.SH', '600685.SH', '600711.SH', '600715.SH', '600733.SH', '600734.SH', '600735.SH', '600751.SH', '600753.SH', '600784.SH', '600794.SH', '600798.SH', '600804.SH', '600806.SH', '600807.SH', '600844.SH', '600856.SH', '600866.SH', '600868.SH', '600869.SH', '600870.SH', '600877.SH', '600960.SH', '601216.SH', '601369.SH', '601619.SH', '601700.SH', '603021.SH', '603032.SH', '603033.SH', '603066.SH', '603222.SH', '603309.SH', '603318.SH', '603333.SH', '603393.SH', '603398.SH', '603508.SH', '603520.SH', '603538.SH', '603598.SH', '603603.SH', '603616.SH', '603659.SH', '603667.SH', '603696.SH', '603718.SH', '603778.SH', '603822.SH', '603843.SH', '603887.SH', '603986.SH', '603988.SH', '603998.SH']
    daily_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily')
    daily_df = daily_df[daily_df['ts_code'] == '000806.SZ ']
    local_hfq_ret = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily_hfq')
    daily_basic = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily_basic')
    namechange = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'namechange.parquet')
    suspend_d = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'suspend_d.parquet')
    stock_basic = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'stock_basic.parquet')
    final_indicator_vip = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'final_indicator_vip')

    for miss_ts_code in miss_ts_codes:
        net_ret = call_ts_tushare_api("pro_bar", ts_code=miss_ts_code, start_date='20180101', end_date='20250711', adj='hfq')
        in_net  = miss_ts_code in net_ret['ts_code'].tolist()
        in_local_hfq = miss_ts_code in local_hfq_ret['ts_code'].tolist()
        # in_local_daily = miss_ts_code in local_daily['ts_code'].tolist()
        # in_local_daily_basic = miss_ts_code in local_daily_basic['ts_code'].tolist()
        # print(f" miss_ts_code{miss_ts_code},net:{in_net},local_hfq_ret:{in_local_hfq},local_daily:{in_local_daily},local_daily_basic:{in_local_daily_basic}")

if __name__ == '__main__':

    compare_local_and_net()
