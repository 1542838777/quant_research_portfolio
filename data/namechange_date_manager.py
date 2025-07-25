from datetime import timedelta

import pandas as pd


def fill_end_date_field(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # 避免修改原始 DataFrame
    df.sort_values(by=['ts_code', 'start_date'], inplace=True)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])

    for ts_code, group in df.groupby('ts_code'):
        group = group.sort_values(by=['ts_code', 'start_date'])
        idx = group.index.to_list()  # 获取原始索引用于回写

        for i in range(len(group) - 1):  # 排除最后一行
            end_date = group.iloc[i]['end_date']
            if not pd.isna(end_date):
                continue
            next_start = pd.to_datetime(group.iloc[i + 1]['start_date'])
            df.at[idx[i], 'end_date'] = next_start - timedelta(days=1)

        #单独对最后一个区间做处理！ 我认为直接end_Date 设置为2099 反正后面都没区间了，起一个连续作用
        if pd.isna(df.at[idx[-1], 'end_date']):
            df.at[idx[-1], 'end_date'] = pd.to_datetime('20990101')

    return df

