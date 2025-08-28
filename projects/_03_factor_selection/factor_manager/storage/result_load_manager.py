import json
from pathlib import Path

import  pandas as pd

from quant_lib.evaluation.evaluation import calculate_forward_returns_c2c


#ic_series_processed_60d.parquet
# def build_targer_name(core_eveluation_type,is_raw_factor, forward_period):
#
#     if core_eveluation_type == 'ic':
#         if is_raw_factor:
#             return f'ic_raw_{forward_period}d'
#         else:
#             return f'ic_processed_{forward_period}d'
#     elif core_eveluation_type == 'tmb':
#         if is_raw_factor:
#             return f'tmb_raw_{forward_period}d'
#         else:
#             return f'tmb_processed_{forward_period}d'
#     elif core_eveluation_type == 'monotonicity':
#     pass


def load_summary_stats(param):
    d1 = json.load(open(param, 'r', encoding='utf-8'))
    return d1
def load_ic_stats(json:json=None,is_raw_factor:bool=False):
    subfix='_processed'
    if is_raw_factor:
        subfix='_raw'
    ic_stas =json.get(f'ic_analysis{subfix}')
    return ic_stas
class ResultLoadManager:
    def __init__(self, calcu_return_type='c2c', version:str=None, is_raw_factor: bool=False):
        if version is None:
            raise ValueError('请指定版本')
        self.main_work_path = Path(r"D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\workspace\result")
        self.calcu_type = calcu_return_type
        self.version = version
        self.is_raw_factor = is_raw_factor
    #严禁使用！这是整体的ic。整个周期的（严重未来寒函数
    # def get_ic_stats_from_local(self, stock_pool_index, factor_name):
    #     path = self.get_factor_self_path(stock_pool_index, factor_name)
    #     ret = load_summary_stats(path / "summary_stats.json")
    #     ic_stas = load_ic_stats(ret, self.is_raw_factor)
    #     return ic_stas

    def get_factor_data(self, factor_name, stock_pool_index, start_date, end_date):
        if self.is_raw_factor:
            raise ValueError('暂不支持raw因子数据')
        factor_self_path = self.get_factor_self_path(stock_pool_index, factor_name)
        df = pd.read_parquet(factor_self_path / "processed_factor.parquet")
        df.index = pd.to_datetime(df.index)
        df = df.loc[start_date:end_date]
        return df

    def get_return_data(self, stock_pool_index,start_date, end_date, period_days):
        path = self.main_work_path / stock_pool_index /'close_hfq'/ self.version / 'close_hfq.parquet'
        df =pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        returns = calculate_forward_returns_c2c(period=period_days, close_df=df)
        #过滤时间"
        returns = returns.loc[start_date:end_date]

        return returns

    def get_factor_self_path(self, stock_pool_index, factor_name):
        return self.main_work_path / stock_pool_index / factor_name / self.calcu_type / self.version

if __name__ == '__main__':
    manager = ResultLoadManager()
    # manager.get_factor_data('volatility_90d','000906', '20190328', '20231231')
    manager.get_return_data( '000906','20230601', '20240710', 1)
