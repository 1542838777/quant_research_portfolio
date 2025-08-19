import numpy as np
import pandas as pd
import os
import platform

import baostock as bs
import pandas as pd

from data.local_data_load import load_income_df, load_dividend_events_long
from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR, parquet_file_names, every_day_parquet_file_names, \
    need_fix
from quant_lib.config.logger_config import setup_logger

import akshare as ak

from quant_lib.tushare.api_wrapper import call_pro_tushare_api, call_ts_tushare_api
from quant_lib.tushare.data.downloader import download_fina_indicator
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
    for name in ['daily_basic', 'daily_hfq', 'namechange.parquet', 'stock_basic.parquet']:
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
    miss_ts_codes = ['000006.SZ', '000007.SZ', '000019.SZ', '000029.SZ', '000031.SZ', '000034.SZ', '000035.SZ',
                     '000038.SZ', '000046.SZ', '000056.SZ', '000156.SZ', '000415.SZ', '000503.SZ', '000506.SZ',
                     '000509.SZ', '000511.SZ', '000516.SZ', '000533.SZ', '000534.SZ', '000540.SZ', '000547.SZ',
                     '000564.SZ', '000566.SZ', '000606.SZ', '000616.SZ', '000629.SZ', '000663.SZ', '000693.SZ',
                     '000703.SZ', '000757.SZ', '000766.SZ', '000793.SZ', '000796.SZ', '000812.SZ', '000820.SZ',
                     '000900.SZ', '000912.SZ', '000930.SZ', '000939.SZ', '000948.SZ', '000950.SZ', '000998.SZ',
                     '001872.SZ', '002004.SZ', '002005.SZ', '002012.SZ', '002033.SZ', '002037.SZ', '002047.SZ',
                     '002072.SZ', '002075.SZ', '002082.SZ', '002085.SZ', '002098.SZ', '002103.SZ', '002113.SZ',
                     '002121.SZ', '002122.SZ', '002143.SZ', '002147.SZ', '002161.SZ', '002163.SZ', '002168.SZ',
                     '002189.SZ', '002198.SZ', '002212.SZ', '002213.SZ', '002219.SZ', '002252.SZ', '002259.SZ',
                     '002260.SZ', '002263.SZ', '002301.SZ', '002309.SZ', '002312.SZ', '002321.SZ', '002345.SZ',
                     '002357.SZ', '002358.SZ', '002366.SZ', '002384.SZ', '002398.SZ', '002413.SZ', '002423.SZ',
                     '002427.SZ', '002437.SZ', '002442.SZ', '002450.SZ', '002451.SZ', '002464.SZ', '002507.SZ',
                     '002509.SZ', '002512.SZ', '002517.SZ', '002520.SZ', '002523.SZ', '002524.SZ', '002532.SZ',
                     '002538.SZ', '002545.SZ', '002575.SZ', '002580.SZ', '002584.SZ', '002592.SZ', '002604.SZ',
                     '002607.SZ', '002619.SZ', '002621.SZ', '002622.SZ', '002647.SZ', '002656.SZ', '002661.SZ',
                     '002662.SZ', '002675.SZ', '002692.SZ', '002694.SZ', '002699.SZ', '002721.SZ', '002726.SZ',
                     '002738.SZ', '002739.SZ', '002770.SZ', '002799.SZ', '002837.SZ', '002851.SZ', '002857.SZ',
                     '002870.SZ', '300004.SZ', '300032.SZ', '300038.SZ', '300049.SZ', '300056.SZ', '300064.SZ',
                     '300067.SZ', '300071.SZ', '300086.SZ', '300089.SZ', '300090.SZ', '300103.SZ', '300116.SZ',
                     '300128.SZ', '300134.SZ', '300143.SZ', '300146.SZ', '300156.SZ', '300159.SZ', '300173.SZ',
                     '300198.SZ', '300208.SZ', '300225.SZ', '300238.SZ', '300240.SZ', '300266.SZ', '300278.SZ',
                     '300280.SZ', '300290.SZ', '300292.SZ', '300312.SZ', '300317.SZ', '300325.SZ', '300337.SZ',
                     '300341.SZ', '300356.SZ', '300383.SZ', '300392.SZ', '300409.SZ', '300411.SZ', '300424.SZ',
                     '300441.SZ', '300444.SZ', '300455.SZ', '300477.SZ', '300578.SZ', '300593.SZ', '300640.SZ',
                     '300656.SZ', '300659.SZ', '300682.SZ', '600051.SH', '600069.SH', '600076.SH', '600083.SH',
                     '600084.SH', '600086.SH', '600112.SH', '600117.SH', '600122.SH', '600145.SH', '600148.SH',
                     '600150.SH', '600157.SH', '600165.SH', '600179.SH', '600187.SH', '600221.SH', '600226.SH',
                     '600228.SH', '600241.SH', '600255.SH', '600257.SH', '600270.SH', '600273.SH', '600290.SH',
                     '600306.SH', '600309.SH', '600365.SH', '600378.SH', '600393.SH', '600399.SH', '600401.SH',
                     '600432.SH', '600485.SH', '600490.SH', '600512.SH', '600515.SH', '600539.SH', '600568.SH',
                     '600614.SH', '600634.SH', '600666.SH', '600673.SH', '600682.SH', '600685.SH', '600711.SH',
                     '600715.SH', '600733.SH', '600734.SH', '600735.SH', '600751.SH', '600753.SH', '600784.SH',
                     '600794.SH', '600798.SH', '600804.SH', '600806.SH', '600807.SH', '600844.SH', '600856.SH',
                     '600866.SH', '600868.SH', '600869.SH', '600870.SH', '600877.SH', '600960.SH', '601216.SH',
                     '601369.SH', '601619.SH', '601700.SH', '603021.SH', '603032.SH', '603033.SH', '603066.SH',
                     '603222.SH', '603309.SH', '603318.SH', '603333.SH', '603393.SH', '603398.SH', '603508.SH',
                     '603520.SH', '603538.SH', '603598.SH', '603603.SH', '603616.SH', '603659.SH', '603667.SH',
                     '603696.SH', '603718.SH', '603778.SH', '603822.SH', '603843.SH', '603887.SH', '603986.SH',
                     '603988.SH', '603998.SH']
    daily_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily')
    daily_df = daily_df[daily_df['ts_code'] == '000806.SZ ']
    local_hfq_ret = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily_hfq')
    daily_basic = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily_basic')
    namechange = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'namechange.parquet')
    suspend_d = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'suspend_d.parquet')
    stock_basic = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'stock_basic.parquet')
    final_indicator_vip = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'fina_indicator.parquet')
    df = call_pro_tushare_api("fina_indicator", end_date='20241231')

    for miss_ts_code in miss_ts_codes:
        net_ret = call_ts_tushare_api("pro_bar", ts_code=miss_ts_code, start_date='20180101', end_date='20250711',
                                      adj='hfq')
        in_net = miss_ts_code in net_ret['ts_code'].tolist()
        in_local_hfq = miss_ts_code in local_hfq_ret['ts_code'].tolist()
        # in_local_daily = miss_ts_code in local_daily['ts_code'].tolist()
        # in_local_daily_basic = miss_ts_code in local_daily_basic['ts_code'].tolist()
        # print(f" miss_ts_code{miss_ts_code},net:{in_net},local_hfq_ret:{in_local_hfq},local_daily:{in_local_daily},local_daily_basic:{in_local_daily_basic}")


def notify(title, message):
    """通用通知函数"""
    system_name = platform.system()
    if system_name == "Darwin":  # macOS
        os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
    elif system_name == "Linux":  # Linux
        # 需要安装 libnotify-bin: sudo apt-get install libnotify-bin
        os.system(f'notify-send "{title}" "{message}"')
    elif system_name == "Windows":  # Windows
        # 需要安装一个库: pip install win10toast-persist
        from win10toast_persist import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(title, message, duration=10)  # 持续10秒
from pathlib import Path

def verify_pb_lookahead_bias( DATE_TO_CHECK :str = '2024-10-08'):

    """
    一个独立的验证脚本，用于检验Tushare daily_basic接口中的pb字段
    是否存在基于财报公告日的未来数据。
    """
    # ▼▼▼▼▼ 【请修改】替换成你自己的数据文件路径 ▼▼▼▼▼
    DAILY_BASIC_PATH = Path( LOCAL_PARQUET_DATA_DIR/'daily_basic')
    BALANCESHEET_PATH = Path( LOCAL_PARQUET_DATA_DIR/'balancesheet.parquet')
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # --- 1. 设定我们的“观测”目标 ---
    STOCK_TO_CHECK = '600519.SH'  # 以贵州茅台为例
    DATE_TO_CHECK_TS = pd.to_datetime(DATE_TO_CHECK)

    print(f"--- 开始对质实验：检验 {STOCK_TO_CHECK} 在 {DATE_TO_CHECK} 的PB值 ---")

    # --- 2. 加载数据 ---
    try:
        daily_basic_df = pd.read_parquet(DAILY_BASIC_PATH)
        balancesheet_df = pd.read_parquet(BALANCESHEET_PATH)
        # 转换日期格式以便比较
        daily_basic_df['trade_date'] = pd.to_datetime(daily_basic_df['trade_date'])
        balancesheet_df['ann_date'] = pd.to_datetime(balancesheet_df['ann_date'])
        balancesheet_df['end_date'] = pd.to_datetime(balancesheet_df['end_date'])
    except FileNotFoundError as e:
        print(f"✗ 错误：数据文件未找到，请检查路径。 {e}")
        return

    # --- 3. 获取【方法A：Tushare预计算】的PB值 ---
    tushare_pb_series = daily_basic_df[
        (daily_basic_df['ts_code'] == STOCK_TO_CHECK) &
        (daily_basic_df['trade_date'] == DATE_TO_CHECK_TS)
        ]
    if tushare_pb_series.empty:
        print(f"✗ 错误：在daily_basic中未找到 {STOCK_TO_CHECK} 在 {DATE_TO_CHECK} 的数据。")
        return

    tushare_pb = tushare_pb_series.iloc[0]['pb']
    total_mv = tushare_pb_series.iloc[0]['total_mv']  # 单位：万元

    # --- 4. 计算【方法B：我们自己的“第一性原理”】PB值 ---

    # a) 找到在观测日当天，市场上已知的、最新的财报
    known_reports = balancesheet_df[
        (balancesheet_df['ts_code'] == STOCK_TO_CHECK) &
        (balancesheet_df['ann_date'] <= DATE_TO_CHECK_TS)  # 核心：公告日必须早于或等于观测日
        ].sort_values(by='end_date', ascending=False)  # 按报告期倒序，拿到最新的

    if known_reports.empty:
        print(f"✗ 错误：在balancesheet中未找到 {STOCK_TO_CHECK} 在 {DATE_TO_CHECK} 之前的任何已公布财报。")
        return

    latest_known_report = known_reports.iloc[0]
    book_value = latest_known_report['total_hldr_eqy_exc_min_int']  # 单位：元
    book_value_in_wanyuan = book_value / 10000.0  # 统一单位为万元

    # b) 手动计算我们自己的PB
    our_pb = total_mv / book_value_in_wanyuan if book_value_in_wanyuan > 0 else np.nan

    # --- 5. “对质”结果 ---
    print("\n" + "=" * 20 + " 【对质结果】 " + "=" * 20)
    print(f"观测日期: {DATE_TO_CHECK}")
    print(f"方法A (Tushare `daily_basic`): PB = {tushare_pb:.4f}")
    print(f"方法B (我们基于公告日计算): PB = {our_pb:.4f}")
    print(f" - 使用的总市值: {total_mv:,.2f} 万元")
    print(
        f" - 使用的净资产 (来自 {latest_known_report['end_date']} 财报, 公告于 {latest_known_report['ann_date']}): {book_value_in_wanyuan:,.2f} 万元")
    print("=" * 50)

    if not np.isclose(tushare_pb, our_pb):
        print("\n✓ 【结论】存在显著差异！Tushare的PB值很可能提前使用了尚未公布的财报数据。")
        print("   “幽灵”的藏身之处，已被确认！")
    else:
        print("\n✓ 【结论】两者一致。该数据点未发现未来数据。")

def find_dividend_and_bonus_stocks():
    dividend_events = load_dividend_events_long()
    """
    找出某天同时有送股 + 分红的股票，方便你人工对比验证
    dividend_events: 必须包含 ['ex_date', 'ts_code', 'cash_div_tax', 'stk_div']
    """
    # 过滤掉同时现金分红>0 且 送股>0 的情况
    df = dividend_events[
        (dividend_events['cash_div_tax'] > 0) &
        (dividend_events['stk_div'] > 0)
    ].copy()

    # 排序方便看
    df = df.sort_values(['ex_date', 'ts_code'])

    return df[['ex_date', 'ts_code', 'cash_div_tax', 'stk_div']]

def look_daily_pct_chg():
    daily_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily')
    daily_df['trade_date'] = pd.to_datetime(daily_df['trade_date'])
    pct_chg_wide = pd.pivot_table(
        daily_df,
        index='trade_date',  # 行索引：交易日
        columns='ts_code',  # 列索引：股票代码
        values='pct_chg'  # 值：收盘价
    )
    _0721 = pct_chg_wide['300721.SZ']
    print(1)

import akshare as ak

# 获取后复权数据
def get_from_akshare():
    # hfq_long_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily_hfq')
    # hfq_long_df['trade_date'] = pd.to_datetime(hfq_long_df['trade_date'])
    #
    # close_adj_df = pd.pivot_table(hfq_long_df,index='trade_date', columns='ts_code', values='close')
    # close_adj_df = close_adj_df['000001.SZ']
    hfq = ak.stock_zh_a_hist(symbol="000008",start_date='20230302', period="daily", adjust="hfq")
    qfq = ak.stock_zh_a_hist(symbol="000008",start_date='20230302', period="daily", adjust="qfq")
    print(1)

def check_dividend_and_bonus(stock_code, target_date: str):
    """
    在 target_date 找出同时有现金分红+送股的股票，
    并计算手工总回报率，用来和 debug 状态的 pct_chg 对比
    """
    # 确保时间是 datetime
    # 读取长表
    daily_long_df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR / 'daily')
    # 转换成长 -> 宽
    close_df = pd.pivot_table(
        daily_long_df,
        index='trade_date',  # 行索引：交易日
        columns='ts_code',  # 列索引：股票代码
        values='close'  # 值：收盘价
    )

    # 确保日期是 datetime 并排序
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index()

    dividend_events = load_dividend_events_long()

    # 取目标日事件
    dividend_event_ann_ex = dividend_events[dividend_events['ann_date'] > dividend_events['ex_date']]
    dividend_event_ann_eq_ex = dividend_events[dividend_events['ann_date'] == dividend_events['ex_date']]
    dividend_event_ann_less_ex = dividend_events[dividend_events['ann_date'] < dividend_events['ex_date']]
    df = dividend_events[
        (dividend_events['ex_date'] == pd.to_datetime(target_date)) &
        # (dividend_events['ts_code'] == stock_code) &
        (dividend_events['cash_div_tax'] > 0) &
        (dividend_events['stk_div'] > 0)
        ].copy()

    results = []
    for _, row in df.iterrows():
        ts_code = row['ts_code']
        cash_div = row['cash_div_tax']
        stk_div = row['stk_div']

        # 前一日价格 & 当日价格
        pre_close = close_df.loc[pd.to_datetime(target_date) - pd.Timedelta(days=1), ts_code]
        close_today = close_df.loc[pd.to_datetime(target_date), ts_code]

        # 公式: (今日收盘 * (1+送股比例) + 派息) / 昨日收盘 - 1
        pct_chg_manual = (close_today * (1 + stk_div) + cash_div) / pre_close - 1

        results.append({
            "date": target_date,
            "ts_code": ts_code,
            "pre_close": pre_close,
            "close_today": close_today,
            "cash_div": cash_div,
            "stk_div": stk_div,
            "pct_chg_manual": pct_chg_manual
        })

    return pd.DataFrame(results)

def t_bao_pct_chg():
    # 登录系统
    lg = bs.login()
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)

    # 获取贵州茅台历史K线数据
    # 字段说明：date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST
    rs = bs.query_history_k_data_plus("sz.003017",
                                      "date,code,close,preclose,pctChg,adjustflag",
                                      start_date='2023-10-09', end_date='2025-07-10',
                                      frequency="d", adjustflag="1")  #复权类型，默认不复权：3；    1：后复权；   2：前复权

    print('query_history_k_data_plus respond error_code:' + rs.error_code)
    print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

    # 打印结果
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    # 退出系统
    bs.logout()

    # Baostock返回的pctChg单位是%，需要转换为小数
    result['pctChg'] = pd.to_numeric(result['pctChg']) / 100
    print("\nBaostock数据:")
    print(result)

    # 重点关注2023-06-29这一行的pctChg

if __name__ == '__main__':
    t_bao_pct_chg()
    find_dividend_and_bonus_stocks()
    df_check = check_dividend_and_bonus( '300971.SZ','2023-03-22')
    print(df_check)

    df = pd.read_parquet(LOCAL_PARQUET_DATA_DIR/'daily_hfq')
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df[(df['trade_date'] >= pd.to_datetime('20250415')) and(df['trade_date'] <=pd.to_datetime('20250715') )]
    df = call_ts_tushare_api("pro_bar", ts_code="000008.SZ",start_date='20180101',adj = 'hfq', end_date='20180501'
                             )
    local_df = load_income_df()
    local_df = local_df[local_df['ts_code'] == '000806.SZ']
    print(1)
    # df = call_pro_tushare_api('income_vip', period='20240930')
    # cash_df = call_pro_tushare_api('cashflow_vip', period= '20240930')
    # compare_local_and_net()
    notify("PyCharm任务完成", "回测已成功结束！✅")
