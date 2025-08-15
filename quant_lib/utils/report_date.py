import pandas as pd


def get_reporting_period_day_list(start_date, end_date):
    """
    返回 start_date 到 end_date 之间所有标准财报期的日期（每季度末一天）
    格式为 YYYYMMDD 字符串列表

    Parameters:
        start_date: str, 格式 'YYYYMMDD' 或 'YYYY-MM-DD'
        end_date: str, 格式 'YYYYMMDD' 或 'YYYY-MM-DD'

    Returns:
        List[str], e.g. ['20170331', '20170630', ...]
    """
    # 转为 datetime 格式
    start_year = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # 所有季度末的月份-日
    quarter_end_days = ['03-31', '06-30', '09-30', '12-31']

    result = []
    current_year = start_year.year
    end_year = end_date.year

    while current_year <= end_year:
        for q_end in quarter_end_days:
            rpt_date = pd.to_datetime(f'{current_year}-{q_end}')
            if start_year <= rpt_date <= end_date:
                result.append(rpt_date.strftime('%Y%m%d'))
        current_year += 1

    return result
if __name__ == '__main__':
    print(get_reporting_period_day_list('20170101', '20191011'))
